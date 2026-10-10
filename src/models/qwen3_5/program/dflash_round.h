#pragma once

// One shard-local masked-draft (DFlash2) round for the TP-2 route (tp2-plan.md 3.6, stage B4).
//
// The single-device route sequences the same three steps inside ProgramImpl: the target prefill taps
// its residual blocks into the draft's feature sink, the commit appends the captured window into the
// draft's local context, and the decode round runs the masked-block proposal against that context.
// The TP-2 core owns the draft whole on shard 0 and drives those steps itself
// (src/runtime/engine/tp2_generation_core.cpp), so this component owns the buffers they need and
// orders them:
//
//   1. the draft's persistent context - one prefill chunk of target features, its positions, the
//      pending staging buffer, and the local K/V ring - built exactly like the shard-local layout in
//      TP2GenerationCore::build_shard;
//   2. the exact-B DFlash2 decode frame, planned through the shared round planner (round_buffers.h)
//      rather than through the single-device sequence plan;
//   3. a proposal workspace that is deliberately separate from the shard workspace: the production
//      proposal begins by resetting the arena it is handed (execution/draft.cpp:284), and the shard
//      workspace holds resident buffers (the prefill hidden state) that must survive the round.
//
// The round publishes the proposal outputs the verify stage consumes: the K draft token ids, the
// [candidate_k, K, B] candidate ids and proposal q, and the [K+1, B] masked query tokens with their
// positions. It does not verify, accept or fold anything - that is the next stage.

#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/program/storage/draft_context.h"

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/layout.h"
#include "models/qwen3_5/state/decoder_state.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// The DFlash2 context and frame use the same device alignment the runtime core reserves for the
// shard-local draft context (tp2_generation_core.cpp:151 kDflashContextAlignment).
inline constexpr std::size_t kDFlash2RoundAlignment = 256;

// The plan shapes of one shard-local masked-draft round. feature_columns is the target prefill chunk
// the sink captures; draft_window is K. Everything else mirrors the draft's own config.
struct DFlash2RoundSpec {
    std::int32_t hidden                  = 0;
    std::int32_t output_rows             = 0;
    std::int32_t target_features         = 0;
    std::int32_t feature_columns         = 0;
    std::int32_t feature_lanes           = 0;
    std::uint32_t local_layers           = 0;
    std::uint32_t local_window           = 0;
    std::int32_t local_kv_heads          = 0;
    std::int32_t local_head_dim          = 0;
    std::uint32_t batch_capacity         = 1;
    std::uint32_t draft_window           = 0;
    std::size_t proposal_workspace_bytes = 0;
    std::vector<std::uint32_t> target_layer_ids;
};

// Derives the round's shapes from the draft/target configs. Throws when the draft is not a DFlash2
// masked draft, so a caller cannot accidentally plan a round the proposal seam cannot run.
[[nodiscard]] DFlash2RoundSpec plan_dflash2_round(const DraftConfig& draft, const TextConfig& target,
                                                  std::uint32_t feature_columns,
                                                  std::uint32_t draft_window,
                                                  std::size_t proposal_workspace_bytes,
                                                  std::uint32_t batch_capacity = 1);

// Peak device workspace one DFlash2 masked-block proposal forward consumes at this geometry, planned
// with the same capacity functions the single-device route sizes its arena with
// (planning/startup.cpp build_workspace_plan). width is K+1 and batch the resident rows; the
// attention envelope is bounded by capacity exactly as it is there.
[[nodiscard]] std::size_t
dflash2_proposal_workspace_bytes(const Parameters& parameters, const TextConfig& target,
                                 std::uint32_t capacity, ProposalHead proposal_head,
                                 std::int32_t width, std::int32_t batch);

// The proposal outputs one round hands to the next stage. Every view aliases the round's own frame,
// so it stays valid until the next proposal overwrites it.
struct DFlash2Proposal {
    Tensor drafts;        // [K, B] selected draft token ids
    Tensor candidate_ids; // [candidate_k, K, B] top-k candidates per draft step
    Tensor scores;        // [candidate_k, K, B] FP32 proposal q over those candidates
    Tensor query_ids;     // [K+1, B] masked-block query tokens
    Tensor positions;     // [K+1, B] draft-attention positions of the query block
};

class DFlash2Round {
public:
    DFlash2Round(DeviceContext& device, const DFlash2RoundSpec& spec);
    ~DFlash2Round();

    DFlash2Round(const DFlash2Round&)            = delete;
    DFlash2Round& operator=(const DFlash2Round&) = delete;

    // --- owned buffers ---
    [[nodiscard]] detail::DFlashPersistentState& state() noexcept { return *state_; }
    [[nodiscard]] const detail::DFlashPersistentState& state() const noexcept { return *state_; }
    [[nodiscard]] CyclicKVCache& ring() noexcept { return *ring_; }
    [[nodiscard]] const CyclicKVCache& ring() const noexcept { return *ring_; }
    [[nodiscard]] std::int32_t target_features() const noexcept { return spec_.target_features; }
    [[nodiscard]] std::size_t context_bytes() const noexcept { return context_bytes_; }
    [[nodiscard]] std::size_t ring_payload_bytes() const noexcept { return ring_payload_bytes_; }
    [[nodiscard]] std::size_t frame_bytes() const noexcept { return frame_bytes_; }

    // --- persistent context image (checkpoint / retention) ---
    // The draft's context is the projected target residual, so it is not a function of the target KV
    // or GDN state and nothing recomputes it without the target forward that produced it. A host
    // checkpoint, a device snapshot and a session slab therefore carry these bytes beside the target
    // state image they pair with, and restore them together. The payload is the one resident lane's
    // local K/V ring, layer-major K then V exactly like the single-device host layout
    // (state/state_image.cpp); the absolute token frontier it reaches is a scalar the caller keeps
    // beside the image, because the ring's coverage is [frontier - window, frontier) and no byte of
    // the payload encodes it.
    [[nodiscard]] std::size_t context_image_bytes() const noexcept { return ring_payload_bytes_; }
    // The bytes one resident lane's ring occupies inside that image. A multi-lane round keeps one
    // ring slot per lane, so an image carries every lane and a caller addresses its own through this
    // extent - the same shape the GDN state image uses (core/linear_attention_state.h).
    [[nodiscard]] std::size_t lane_context_image_bytes() const noexcept {
        return ring_payload_bytes_ / static_cast<std::size_t>(spec_.batch_capacity);
    }
    void copy_context_to_host(std::byte* destination, cudaStream_t stream,
                              std::int32_t lane = 0) const;
    void copy_context_from_host(const std::byte* source, cudaStream_t stream,
                                std::int32_t lane = 0);
    void copy_context_to_device(DeviceSpan destination, cudaStream_t stream,
                                std::int32_t lane = 0) const;
    void copy_context_from_device(DeviceSpan source, cudaStream_t stream,
                                  std::int32_t lane = 0);
    [[nodiscard]] std::size_t proposal_workspace_capacity() const noexcept;
    [[nodiscard]] std::size_t proposal_workspace_peak() const noexcept;
    void reset_proposal_workspace_peak() noexcept;
    // Zeroes the whole draft context (features, positions, pending staging and the K/V ring), which
    // is the state of a round that has consumed no prefill chunk yet.
    void zero_context();

    // --- prefill capture (target forward -> draft context) ---
    // The sink the target prefill taps its residual blocks into. Its consumer commits each captured
    // chunk through append() on the caller's resident workspace.
    [[nodiscard]] DFlashFeatureSink make_prefill_sink(ExecutionCore execution, std::int32_t lane = 0);
    // Commits one captured feature window [target_features, width, B] into the draft's local ring.
    // exact is the number of live columns the target forward produced; lane names the destination
    // lane (0 for the one resident session).
    void append(ExecutionCore execution, const Tensor& features, const Tensor& positions,
                std::uint32_t exact, std::int32_t lane = 0);

    // --- verify capture (target verify window -> draft context) ---
    // The sink one target verify window taps its residual blocks into. Unlike the prefill sink it
    // owns no consumer: the captured columns land in the draft's pending staging buffer and are
    // committed one round later by append_pending, exactly as the single-device route's
    // prepare_ragged_prefix hand-off does. width is the verify window (K+1) and batch the resident
    // rows; the destination lane is frame active_lanes. batch is the number of live rows this round
    // (frame rows beyond it are not captured), and it must equal the verify window's lane count.
    [[nodiscard]] DFlashFeatureSink make_verify_sink(std::int32_t batch = 1);
    // Commits the pending verify-window features [start, end) - the columns of the window that the
    // next round has committed - into the draft's local ring at those absolute positions. start is
    // the draft's context frontier, end the target's execution frontier; end - start must fit the
    // round's feature_lanes, which is the pending staging width. A zero-width call is a no-op.
    void append_pending(ExecutionCore execution, std::uint32_t start, std::uint32_t end);
    // The multi-lane form of append_pending: one (lane, state slot, start, end) row per live lane.
    // Every row's state slot is the draft ring slot that lane's proposal wrote, and end - start is
    // that lane's own committed width. Lanes with a zero-width window are skipped. One ragged-prefix
    // pass fans the staged features out to the per-lane ring slots.
    void append_pending_batch(ExecutionCore execution, const std::int32_t* slots,
                              const std::int32_t* starts, const std::int32_t* ends,
                              std::int32_t batch);
    [[nodiscard]] std::uint32_t draft_window() const noexcept { return spec_.draft_window; }
    [[nodiscard]] std::int32_t feature_lanes() const noexcept { return spec_.feature_lanes; }

    // --- proposal ---
    [[nodiscard]] qwen3_5::DFlashDecodeIngress& ingress() noexcept { return ingress_; }
    [[nodiscard]] const qwen3_5::DFlashDecodeIngress& ingress() const noexcept { return ingress_; }
    [[nodiscard]] qwen3_5::DFlashDecodeEgress& egress() noexcept { return egress_; }
    // Runs the production masked-block proposal (propose_dflash2_batch) on the round's own arena and
    // leaves its outputs in the frame. card registers the peer half of the column-split token
    // embedding on TP-2; it is null whenever the text stack is not split.
    void propose(ExecutionCore execution, const qwen3_5::PagedKVCache& text_cache, TextContext* card,
                 std::uint32_t k, DFlashEnvelopes envelopes, std::int32_t batch = 1);

    [[nodiscard]] qwen3_5::DFlashDecodeState& frame() noexcept { return *frame_; }
    [[nodiscard]] const qwen3_5::DFlashDecodeState& frame() const noexcept { return *frame_; }
    [[nodiscard]] const DFlash2Proposal& proposal() const noexcept { return proposal_; }

private:
    // The execution state the round's own buffers provide: the frame is the round's RoundState (its
    // dflash_prefill count storage serves an oversized append) and work is one of the round's arenas.
    // Everything else comes from the caller.
    [[nodiscard]] ExecutionCore round_execution(const ExecutionCore& source, DeviceArena& work) const;

    DeviceContext* device_ = nullptr;
    DFlash2RoundSpec spec_;

    // Draft context (persistent).
    std::unique_ptr<DeviceArena> context_arena_;
    std::unique_ptr<CyclicKVCache> ring_;
    std::unique_ptr<detail::DFlashPersistentState> state_;
    std::size_t context_bytes_      = 0;
    std::size_t ring_payload_bytes_ = 0;

    // Exact-B decode frame (persistent).
    std::unique_ptr<DeviceArena> frame_arena_;
    std::unique_ptr<qwen3_5::RoundState> io_;
    qwen3_5::DFlashDecodeState* frame_ = nullptr;
    std::size_t frame_bytes_           = 0;

    // Transient proposal workspace, deliberately not the caller's resident arena.
    std::unique_ptr<DeviceArena> proposal_workspace_;
    // Scratch that outlives the frame: the continuation hidden the verify stage stores.
    std::unique_ptr<DeviceArena> scratch_arena_;
    Tensor continuation_hidden_;
    // The verify sink's lane bindings, sliced to the round's live rows. They must outlive the sink
    // because scatter_bf16_batch requires lanes/valid_columns to have exactly one entry per source
    // column, while the frame rows are sized for batch_capacity.
    Tensor verify_lanes_;
    Tensor verify_valid_;

    qwen3_5::DFlashDecodeIngress ingress_{};
    qwen3_5::DFlashDecodeEgress egress_{};
    DFlash2Proposal proposal_;
};

} // namespace ninfer::models::qwen3_5::execution
