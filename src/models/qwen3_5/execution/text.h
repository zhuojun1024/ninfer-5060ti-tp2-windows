#pragma once
#include "models/qwen3_5/program/internal.h"


#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "core/tp/device_pair.h"
#include "core/weight.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/sparse_moe.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/program/vision_control.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using Phase = qwen3_5::TextPhase;

// Vision sidecar for one tensor-parallel prefill chunk. A multimodal prompt carries the 3-axis
// (temporal, height, width) RoPE table the Vision tower produced, for every chunk, and the chunks
// that overlap an encoded item additionally scatter that item's embeddings into the residual
// stream.
//
// `embeddings` is bound into the Vision shard's workspace (that shard owns the tower), so only the
// peer side of the forward reads it: the pair copies this chunk's contiguous column range into both
// shards' arenas with one in-place all-reduce over a zeroed text-shard buffer, which is exact (a
// sum with zero) and keeps the transfer bounded by the chunk rather than by the item.
struct Tp2VisionChunk {
    // Encoded item overlapping this chunk, or null for a chunk with no image tokens.
    const qwen3_5::VisionItemControl* control = nullptr;
    [[nodiscard]] bool has_item() const noexcept { return control != nullptr; }
    // [hidden, merged_count] BF16 on the Vision (peer) shard.
    const Tensor* embeddings = nullptr;
    // Prompt-wide [3, prompt_tokens] RoPE positions, row-major per axis.
    const std::int32_t* positions = nullptr;
    std::size_t prompt_tokens     = 0;
};

enum class GdnStateAction : std::uint8_t {
    UpdateInPlace,
    RecordForReplay,
};

struct NullTap {
    static constexpr bool enabled = false;
};

struct PrefillChunkResult {
    std::uint32_t processed_tokens = 0;
    bool finalized                 = false;
    runtime::ExecutionTiming timing;
};

struct DFlashFeatureSink {
    static constexpr bool enabled = true;
    using PrefillConsumer         = std::function<void(const Tensor&, const Tensor&, bool)>;

    Tensor* features                  = nullptr;
    Tensor* positions                 = nullptr;
    Tensor* batch_features            = nullptr;
    const Tensor* batch_lanes         = nullptr;
    const Tensor* batch_valid_columns = nullptr;
    std::int32_t batch_width          = 0;
    std::int32_t batch_size           = 0;
    std::span<const std::uint32_t> layers;
    PrefillConsumer consume_prefill;
    std::uint32_t captured_mask = 0;
    std::int32_t active_tokens  = 0;

    void begin(const Tensor& value);
    void capture_layer(int layer, const Tensor& value, cudaStream_t stream);
    void capture_positions(const Tensor& source, cudaStream_t stream);
    void consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);
};

class VisionPrefillSession;

class TextContext {
public:
    TextContext(DeviceContext& ctx, const execution::Parameters& weights, WorkspaceArena& work,
                qwen3_5::PagedKVCacheView kv, LinearAttentionStatePool& state,
                qwen3_5::RoundState& io, Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                std::uint32_t text_kv_base,
                qwen3_5::PagedKVCacheView mtp_kv           = qwen3_5::PagedKVCacheView(),
                const qwen3_5::PagedKVCache* batch_text_kv = nullptr,
                const qwen3_5::PagedKVCache* batch_mtp_kv  = nullptr,
                // TP-2 prefill all-reduce overlap: split a prefill chunk into equal sub-blocks of
                // this width and run each sub-block's collective on the pair's second stream while
                // the next sub-block computes. 0 disables it; only a chunk holding at least two
                // whole sub-blocks is split. Single-GPU contexts ignore it.
                std::uint32_t prefill_overlap = 0);
    ~TextContext();

    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    void set_proposal_head(const LinearParameters* weight, const std::int32_t* ids,
                           int count) noexcept {
        proposal_head_     = weight;
        proposal_head_ids_ = ids;
        proposal_head_n_   = count;
    }

    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling_config_ = config; }

    // Declared-name mask for the prefill's final-column sample. It stays empty on the
    // causal-scoring route, which never samples and must keep the unmasked distribution.
    void set_prefill_token_mask(const Tensor& mask) noexcept { prefill_token_mask_ = mask; }

    void set_prefill_split_frontier(std::int64_t position) noexcept {
        prefill_split_frontier_ = position;
    }

    void set_rewrite_checkpoint_hidden_output(Tensor* output) noexcept {
        rewrite_checkpoint_hidden_output_ = output;
    }

    void set_mtp_proposal_extent(std::uint32_t extent) noexcept { mtp_proposal_extent_ = extent; }

    // The MTP layer's KV execution row the prefill chunk appends into and attends over. The TP-2
    // route primes one lane at a time, so it re-points this view (and the round state's scalar KV
    // row) before each lane's priming (P2.1b).
    void set_mtp_prefill_view(qwen3_5::PagedKVCacheView view) noexcept { mtp_kv_ = view; }

    // Tensor-parallel surface: run one layer's mixer (attention or GDN) on this shard's device,
    // writing the row-parallel output-projection delta (without the residual add) into delta.
    // A tensor-parallel driver calls this on each shard context and all-reduces the delta before
    // adding it to the shared residual. The single-GPU path uses single_layer instead.
    // `activation` (optional) holds the mixer's output projection back: the mixer leaves this
    // shard's half of its pre-projection activation in that [full width, T] buffer, zeroes the rest
    // and returns without touching delta, and the driver merges the pair and calls
    // tp_mixer_activation_delta. A weight whose output projection gathers its input columns needs
    // that, because its stored columns address heads on both shards (see tp_mixer_activation_width).
    void tp_mixer_layer(const BlockParameters& block, Tensor& x, Tensor& delta, std::size_t layer,
                        Phase phase, Tensor* activation = nullptr);
    // Second stage of that split mixer: project the pair's full-width activation through this
    // shard's half of the output projection, leaving the partial delta the driver all-reduces.
    void tp_mixer_activation_delta(const BlockParameters& block, const Tensor& activation,
                                   Tensor& delta, std::size_t layer, Phase phase);
    void tp_mlp_layer(const BlockParameters& block, Tensor& x, std::size_t layer, Phase phase);
    // Tensor-parallel FFN tail: post-mixer norm + the sharded FFN, writing the row-parallel
    // down-projection delta (without the residual add) into `delta`. The driver all-reduces the
    // delta across shards and adds the sum to the shared residual once. The mixer is replicated,
    // so `x` is identical on every shard and no all-reduce is needed after the mixer.
    void tp_mlp_delta(const BlockParameters& block, Tensor& x, Tensor& delta, std::size_t layer,
                      Phase phase);
    // Tensor-parallel lockstep layer loop: runs each layer's mixer (replicated) on both shards,
    // then the sharded FFN delta on both shards, all-reduces the delta, and adds the summed delta
    // to the shared residual once. The mixer is replicated, so the residual is identical on both
    // shards at the FFN input and no all-reduce is needed after the mixer.
    template <class Tap>
    void run_layers_tp2(TextContext& peer, tp::DevicePair& pair, Tensor& x, Tensor& x_peer,
                        Phase ph, Tap& tap);
    // The bytes this shard's next op streams from HBM, which decode's collective windows prefetch.
    // Decode has no compute to overlap with a collective (the layer chain is strictly sequential),
    // but it does have an idle memory system for the whole rendezvous, and these bytes are
    // independent of the reduction. Null when the layer has nothing to hint.
    [[nodiscard]] const Weight* tp_mixer_stream_weight(std::size_t layer) const;
    [[nodiscard]] const Weight* tp_ffn_stream_weight(std::size_t layer) const;
    [[nodiscard]] tp::DevicePair::ArPrefetchWindow tp_prefetch_window(const Weight* mine,
                                                                      const Weight* peer) const;
    // Tensor-parallel prefill with the all-reduce overlapped against compute. The chunk is split into
    // `parts` 64-aligned sub-blocks and each sub-block's collective runs on the pair's second stream
    // while the next sub-block keeps the compute stream busy. The 64-token boundary keeps the result
    // bit-identical to run_layers_tp2's single block. Experimental; the caller env-gates it.
    void run_layers_tp2_overlap(TextContext& peer, tp::DevicePair& pair, Tensor& x, Tensor& x_peer,
                                Phase ph, int width, std::int32_t first_position);

    // Tensor-parallel single-token forward at `position`: embedding on both shards, the lockstep
    // layer loop, the final norm, and the lm_head. The head is either replicated in full on both
    // shards or vocabulary-split (see project_head_tp2); either way both shards leave the complete
    // full-vocabulary logits in their buffer. `position` is the absolute cache/RoPE position, so
    // the mixers attend to the cached prefix [0, position) and update the paged KV cache and GDN
    // state in place, enabling a stateful autoregressive decode. `logits` and `logits_peer` must
    // be full-vocabulary BF16 [V, 1] buffers.
    // mtp_input_hidden (optional, [hidden,1] BF16) receives the final-norm hidden (the lm_head
    // input), which is the hidden the multi-token-prediction layer consumes.
    void forward_tp2(TextContext& peer, tp::DevicePair& pair, std::int32_t token,
                     std::int32_t position, Tensor& logits, Tensor& logits_peer,
                     Tensor* mtp_input_hidden = nullptr);
    // The plain (non-speculative) decode step, shaped for CUDA Graph capture. It runs exactly the
    // sequence forward_tp2 runs - same bindings, same Phase::Verify - but the two per-round host
    // values (the decoded token and its absolute cache/RoPE position) are read from pinned host
    // memory instead of being baked into a set_i32_scalar kernel argument, so a captured graph
    // records a memcpy node that every replay re-reads. `envelope` is the capture parameter: only
    // its bounds matter, because the small-T route derives the active split count and the key
    // partition from the device-side positions. The peer's logits buffer is allocated internally
    // (the caller only samples from this shard's), so the captured layout is self-contained.
    // `lane` selects the paged-KV execution row and the linear-attention state slot this window
    // appends to. It is 0 for the single-lane route; the multi-lane route binds one lane per serial
    // walk (P1.4), and the batched window below takes the whole lane vector instead.
    void forward_tp2_decode_window(TextContext& peer, tp::DevicePair& pair,
                                   const std::int32_t* token, const std::int32_t* position,
                                   ops::CausalAttentionExecutionEnvelope envelope, Tensor& logits,
                                   std::int32_t lane = 0);
    // The plain decode step for a batch of resident sequences, one column (lane) per sequence. It
    // runs exactly the sequence forward_tp2_decode_window runs, with every per-round operand
    // widened to a [B] vector: tokens are the lane tokens, positions their absolute cache/RoPE
    // positions, kv_table_rows the paged-KV execution-table row each lane appends to, and
    // state_slots the linear-attention state slot each lane owns. All four are pinned host arrays
    // of batch int32 read by both shards (the same values drive each). envelope must bound the
    // whole batch inclusive key extent. The peer logits buffer is allocated internally, so the
    // caller samples from this shard [V, B] logits.
    void forward_tp2_decode_window_batch(TextContext& peer, tp::DevicePair& pair,
                                         const std::int32_t* tokens, const std::int32_t* positions,
                                         const std::int32_t* kv_table_rows,
                                         const std::int32_t* state_slots, std::int32_t batch,
                                         ops::CausalAttentionExecutionEnvelope envelope,
                                         Tensor& logits);
    // Tensor-parallel single-token forward at `position` returning the argmax token id on this
    // shard's device. Both shards hold the complete logits, so their argmax must agree; a
    // disagreement is a shard divergence.
    [[nodiscard]] std::int32_t forward_tp2_token(TextContext& peer, tp::DevicePair& pair,
                                                 std::int32_t token, std::int32_t position = 0);
    // Tensor-parallel batched prefill forward. Runs the lockstep layer loop at Phase::Prefill over
    // the T tokens of one chunk and then projects the columns the caller asked for through the
    // lm_head. `ids` are this chunk's prompt tokens and `first_position` is the absolute position
    // of its first token, so consecutive chunks continue the same sequence: the replicated mixers
    // append to the paged KV cache and update the GDN state in place, exactly like the
    // single-device prefill chunk.
    // Exactly one of the two logits forms is requested. The common prefill chunk needs only its
    // last column, which drives the first sample, so it passes `logits` / `logits_peer`
    // (full-vocabulary BF16 [V,1] buffers, one per shard). A speculative verify window needs every
    // column, because each column scores the draft that follows it, so it passes `logits_columns`
    // and leaves the [V,1] buffers null: that buffer's last column already is the last column's
    // logits, and projecting it again would repeat a whole lm_head read per round.
    // mtp_input_hidden (optional, [hidden,T] BF16) receives the whole chunk's final-norm hidden
    // (the lm_head input), which is the hidden the multi-token-prediction layer consumes.
    // logits_columns (optional, [V,T] BF16) receives every column's logits and hidden_columns
    // (optional, [hidden,T] BF16) every column's final-norm hidden.
    // `phase` selects the layer loop the window runs in. A speculative verify window must use
    // Phase::Verify: that is the phase the single-token decode path runs, so its per-column logits
    // agree with the token the decode path would commit instead of drifting on near-ties.
    // `vision` (optional) switches the chunk to the multimodal bindings: the mixer reads the prompt's
    // 3-axis RoPE table instead of its plain absolute positions, and a chunk overlapping an encoded
    // item scatters that item's embeddings into the residual stream before the layer loop.
    // `lane` binds the chunk to one lane's paged-KV execution row and linear-attention state slot;
    // the multi-lane route prefills its lanes one at a time against their own lane (P1.4).
    void forward_tp2_prefill(TextContext& peer, tp::DevicePair& pair, std::span<const int> ids,
                             std::int32_t first_position, Tensor* logits, Tensor* logits_peer,
                             Tensor* mtp_input_hidden = nullptr, Tensor* logits_columns = nullptr,
                             Tensor* hidden_columns = nullptr, Phase phase = Phase::Prefill,
                             const Tp2VisionChunk* vision = nullptr,
                             DFlashFeatureSink* sink = nullptr, std::int32_t lane = 0);

    // Tensor-parallel verify window with capture-safe inputs. It runs exactly forward_tp2_prefill's
    // prefill phase over a small speculative window, but every per-round input is supplied as pinned
    // host memory that the captured sequence copies with a memcpy node, and the attention envelope is
    // a capture parameter instead of a per-round value. That makes the whole sequence replayable from
    // a CUDA Graph: the caller rewrites ids/positions in place before each replay and launches the
    // graph. ids and positions are [T] int32 arrays of window tokens and their absolute cache/RoPE
    // positions; both must stay valid (and stable in address) for as long as the captured graph
    // exists. envelope must cover the window's inclusive key extent.
    // valid_columns is a [1] pinned host int32 naming how many leading window columns own their
    // cache slot. Columns beyond it are clamp duplicates (the same absolute position as the last
    // valid column) and are kept from appending KV and from publishing logits; nullptr leaves every
    // column appending, which is what a window of distinct positions (the MTP window) wants. Like
    // ids/positions it must stay address-stable, and a captured graph re-reads its value on every
    // replay (the clamp extent changes per round).
    // The trailing three operands are required, not defaulted: each of them changes the window's
    // semantics when absent (no hidden capture; no masked-draft feature sink, which silently drops
    // pending-feature columns and only costs acceptance; no clamp extent, which silently leaves every
    // column appending KV). Every caller states its intent so an omission cannot compile, and the
    // capture and eager routes must pass the same operands.
    void forward_tp2_window(TextContext& peer, tp::DevicePair& pair, const std::int32_t* ids,
                            const std::int32_t* positions,
                            ops::CausalAttentionExecutionEnvelope envelope, Tensor& logits_columns,
                            Tensor* hidden_columns, DFlashFeatureSink* sink,
                            const std::int32_t* valid_columns);
    // The same window for a batch of resident sequences: each lane owns `width` consecutive columns
    // and the aggregate is `width * batch`, laid out lane by lane (dim 0 fastest, so every
    // per-column operand is indexed `lane * width + column`). kv_table_rows and state_slots are
    // [batch] pinned host arrays naming each lane's paged-KV execution row and linear-attention state
    // slot; the per-column device bindings repeat them across that lane's window. logits_columns is
    // [V, width*batch] and hidden_columns, when requested, the same. A one-lane window keeps the
    // scalar bindings (extra lanes are what publish the per-column vectors), so the single-lane
    // verify route stays byte-identical. valid_columns here is [batch] pinned host memory: the
    // attention op takes one clamp extent per lane, and forward_tp2_window keeps its [1] contract.
    void forward_tp2_window_batch(TextContext& peer, tp::DevicePair& pair, const std::int32_t* ids,
                                  const std::int32_t* positions,
                                  const std::int32_t* kv_table_rows,
                                  const std::int32_t* state_slots, std::int32_t width,
                                  std::int32_t batch,
                                  ops::CausalAttentionExecutionEnvelope envelope,
                                  Tensor& logits_columns, Tensor* hidden_columns,
                                  DFlashFeatureSink* sink, const std::int32_t* valid_columns);

    // Registers the peer shard's context and the device pair. The tensor-parallel driver sets this
    // on both shards once, so operations that only run on one shard (the MTP stem) can still drive
    // the peer's half of a column-split weight.
    void set_tp_peer(TextContext* peer, tp::DevicePair* pair);

    // Every TP-2 handoff this context performs, in one place. The payload decides the transport:
    // DevicePair::allreduce combines its two operands with BF16 addition, so it carries only payloads
    // whose bits are BF16-safe, and only as a gather or broadcast where one operand is a zeroed or
    // disjoint buffer; DevicePair::sendrecv moves bytes verbatim and is the only transport for token
    // ids, FP32 scores, or anything that must arrive unchanged. C is drafts * batch.
    //
    //   merge_local_row_blocks      BF16 [rows, T]                 allreduce  disjoint halves: each
    //                                                                         shard zeroes the full
    //                                                                         buffer and writes only
    //                                                                         its own row block
    //   embedding_tp2               I32 ids, padded to 16B         allreduce  broadcast over a zeroed
    //                                                                         peer. The one integer
    //                                                                         payload on this path, so
    //                                                                         a signaling-NaN bit
    //                                                                         pattern is not preserved
    //                                                                         (tp2-plan.md 3.3)
    //   proposal_argmax             BF16 [hidden, T]               allreduce  broadcast over a zeroed
    //                                                                         peer
    //   proposal_topk_tp2  hidden    BF16 [hidden, C]               allreduce  broadcast over a zeroed
    //                                                                         peer
    //   proposal_topk_tp2  merge     I32 [top_k, C] plus            sendrecv  byte-exact; the peer's
    //                                FP32 [top_k, C]                          receive slot is scratch
    //                                                                         it never reads
    //   dflash_selector_tp2  draft   BF16 [hidden, C]               allreduce  broadcast over a zeroed
    //                                                                         peer
    //   dflash_selector_tp2  forward I32 ids, FP32 scores,          sendrecv  byte-exact, packed in
    //                                I32 anchors, I32 frontiers                 that order, align16;
    //                                                                          the peer unpacks in the
    //                                                                          same order
    //   dflash_selector_tp2  back    I32 drafts, FP32 q             sendrecv  byte-exact, packed in
    //                                                                          that order, align16
    //   forward_tp2_prefill  vision  BF16 [hidden, count]           allreduce  broadcast over a zeroed
    //                                                                          local buffer
    //
    // The allreduce payloads are exact because the receiving operand is zero (broadcast) or holds a
    // disjoint region (gather), never because a real sum is wanted.

    // Full-width token embedding for a component that runs on this shard alone but consumes the
    // complete hidden state. The masked draft's proposal block is that case on TP-2: its weights
    // are replicated whole on shard 0 while the shared text token embedding is column-split, so
    // the peer half must be gathered through the registered pair. A replicated table is the plain
    // local gather.
    void embedding_full_width(const Tensor& ids, Tensor& out);

    void set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot);
    void set_gdn_state_action(GdnStateAction action, const GdnReplayRecords* replay_records);

    // Tensor-parallel split: point this context at the per-shard TextConfig (head counts halved)
    // and record the shard index. The mixer and weight ops then run on the per-shard geometry,
    // while the components the spec keeps replicated (norms, GDN gating) use the full-model config.
    // With no shard config set the context runs the full model (single-GPU path).
    void set_shard_config(const TextConfig* config, int shard_index) noexcept {
        shard_config_ = config;
        shard_index_  = shard_index;
    }
    [[nodiscard]] const TextConfig& shard_config() const noexcept {
        return shard_config_ != nullptr ? *shard_config_ : config_;
    }

    [[nodiscard]] const LinearParameters* proposal_head() const noexcept { return proposal_head_; }

    [[nodiscard]] const std::int32_t* proposal_head_ids() const noexcept {
        return proposal_head_ids_;
    }

    [[nodiscard]] int proposal_head_n() const noexcept { return proposal_head_n_; }

    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    [[nodiscard]] PrefillChunkResult
    prefill_chunk(const qwen3_5::PreparedPromptData& input, std::uint32_t begin,
                  std::uint32_t nominal_length, VisionPrefillSession& vision, bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   VisionPrefillSession& vision,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    void ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                               const Tensor& rope_positions, const Tensor& kv_table_rows,
                               const Tensor& linear_state_source_slots,
                               const Tensor& linear_state_destination_slots,
                               ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                               Tensor& logits);
    // `logits_mask` is the declared-name mask for the flattened [vocab, width*batch] verify
    // logits. It is applied between the verify projection and its argmax, so a masked column can
    // never be licensed by the greedy or sparse accept kernels. Null means unconstrained.
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens, const Tensor* logits_mask);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens, DFlashFeatureSink& sink,
                             const Tensor* logits_mask);
    void mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                  const Tensor& cache_positions, const Tensor& rope_positions,
                                  const Tensor& valid_columns, const Tensor& kv_table_rows,
                                  ops::CausalAttentionExecutionEnvelope envelope,
                                  Tensor& mtp_hidden);
    void mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens);

    // TP-2 masked draft: stable top sixteen of the reduced proposal head when the table is
    // vocabulary-split. Each shard ranks its own row block of the reduced vocabulary after the pair
    // broadcasts the draft's hidden state, then the pair unions the thirty-two candidates and merges
    // them back to the exact top sixteen of the whole table - the same list the replicated table
    // produces. `candidate_ids`/`candidate_scores` are the I32/FP32 [16,U] results on this shard,
    // which is the one that runs the masked draft; the peer's copy is scratch. Requires a registered
    // peer whose proposal head holds the matching half of the rows.
    void proposal_topk_tp2(const Tensor& hidden, Tensor& candidate_ids, Tensor& candidate_scores);

    // TP-2 masked draft: the DFlash2 selector's codebooks live on the peer shard, so the draft state
    // travels there, the selector runs on that shard's stream and the drafts and proposal q come back
    // unchanged. `host_configs` are the batch's sampling configs from the host ingress, which the
    // peer uploads to its own device. Requires a registered peer whose parameters hold the selector.
    void dflash_selector_tp2(const Tensor& hidden, const Tensor& candidate_ids,
                             const Tensor& candidate_scores, const Tensor& anchors,
                             const Tensor& frontiers, const ops::SamplingConfig* host_configs,
                             Tensor& draft_tokens, Tensor& proposal_q);
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                           int logits_column, Tensor* logits, Tensor* draft_token,
                           const Tensor* explicit_rope_positions = nullptr,
                           const Tensor* input_embeddings        = nullptr);
    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, ops::CausalAttentionExecutionEnvelope envelope,
                             Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token);
    // MTP prefill chunk: appends the chunk's MTP K/V from the target residual stream and, on the
    // final chunk, emits the layer's hidden state, full-vocabulary logits and first draft token.
    // Public because the tensor-parallel route drives the MTP layer per chunk from the runtime.
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                           const Tensor& positions, const Tensor& rope_positions,
                           ops::CausalAttentionExecutionEnvelope envelope, bool final_chunk,
                           Tensor* final_hidden, Tensor* logits, Tensor* draft_token);
private:
    // Vocabulary-parallel output head (TP-2). Projects `hidden` / `hidden_peer` through each
    // shard's half of the output head and assembles the complete [V, T] logits into `logits` and
    // `logits_peer`, so sampling and the speculative accept/reject see the full vocabulary on
    // both shards. With a replicated head this is the two independent full projections that agree
    // exactly, and no collective runs.
    void project_head_tp2(TextContext& peer, tp::DevicePair& pair, const Tensor& hidden,
                          const Tensor& hidden_peer, Tensor& logits, Tensor& logits_peer);

    // True when the token embedding owns only half the hidden columns (RowParallel split).
    [[nodiscard]] bool embedding_is_split() const noexcept;

    // Token embedding on the tensor-parallel pair. Each shard gathers its own half of the hidden
    // columns into a compact [hidden/2, T] block; merge_local_row_blocks then assembles the full
    // [hidden, T] state on both shards. With a replicated table this is the two independent gathers
    // (identical rows, no collective). ids_peer already holding the same token ids on the peer skips
    // the broadcast; a null x_peer destination allocates a peer scratch window for the merge.
    void embedding_tp2(TextContext& peer, tp::DevicePair& pair, const Tensor& ids,
                       const Tensor* ids_peer, Tensor& x, Tensor* x_peer);

    // Places each shard's compact [local_rows, T] block at its half of the contiguous [rows, T]
    // destination, zeroes the other half and all-reduces the pair, so both destinations end with
    // the sum of two disjoint row blocks (their concatenation). The row blocks are strided windows
    // of a wider parent, so each side needs a compact source and an explicit placement copy.
    void merge_local_row_blocks(TextContext& peer, tp::DevicePair& pair, const Tensor& local,
                                const Tensor& local_peer, Tensor& destination,
                                Tensor& destination_peer, std::int32_t local_rows);

    [[nodiscard]] bool mtp_enabled() const noexcept {
        return mtp_kv_.valid() || batch_mtp_kv_ != nullptr;
    }

    void attn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase,
                  Tensor* delta = nullptr, Tensor* activation = nullptr);
    void gdn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase,
                 Tensor* delta = nullptr, Tensor* activation = nullptr);
    void mlp_tail(const BlockParameters& weights, Tensor& x, Phase phase,
                  const ops::SparseMoeHints& hints);
    // One transformer layer's mixer (attention or GDN) on this shard's device. When delta is
    // null the mixer's output projection is added to the residual in place (single-GPU path);
    // otherwise the row-parallel output projection is written to delta without the residual
    // add, so a tensor-parallel driver can all-reduce the delta across shards first.
    void mixer_layer(const BlockParameters& block, Tensor& x, std::size_t layer, Phase phase,
                     Tensor* delta = nullptr, Tensor* activation = nullptr);
    // Full width of the activation this layer's mixer output projection gathers, or 0 when the
    // projection multiplies the mixer's own rows and the mixer runs in one stage.
    [[nodiscard]] std::int32_t tp_mixer_activation_width(const BlockParameters& block) const;
    // One transformer layer's post-mixer FFN (post-attention norm + dense FFN) on this shard's
    // device, updating the residual in place. Its row-parallel down projection leaves a partial
    // residual, so the caller all-reduces before the next layer.
    void mlp_layer(const BlockParameters& block, Tensor& x, std::size_t layer, Phase phase);
    // Runs one full transformer layer (mixer + post-mixer FFN) on this shard's device. The
    // single-GPU path; a tensor-parallel driver calls mixer_layer / mlp_layer directly so it can
    // interleave all-reduce at the two per-layer boundaries.
    void single_layer(const BlockParameters& block, Tensor& x, std::size_t layer, Phase phase);
    [[nodiscard]] ops::SparseMoeHints next_projection_hints(int layer) const;
    void run_layers(Tensor& x, Phase phase);
    template <class Tap>
    void run_layers(Tensor& x, Phase phase, Tap& tap);
    template <class Tap>
    void target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                  const Tensor& rope_positions, const Tensor& valid_columns,
                                  const Tensor& kv_table_rows,
                                  const Tensor& linear_state_source_slots,
                                  ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                                  Tensor& logits, Tensor& target_tokens,
                                  const Tensor* logits_mask, Tap& tap);

    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                          Tensor& x, Tensor& ah);
    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                          const Tensor* input_embeddings);
    void proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens);

    struct MultimodalPrefill {
        std::span<const int> token_ids;
        std::span<const std::int32_t> positions;
        VisionPrefillSession* vision = nullptr;
        std::uint32_t begin          = 0;
        std::int32_t rope_delta      = 0;
    };

    struct TextPrefill {
        std::span<const int> token_ids;
        std::uint32_t begin = 0;
    };

    template <class Tap>
    [[nodiscard]] PrefillChunkResult
    prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                 const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end);
    DeviceContext& ctx_;
    const Parameters& parameters_;
    const TextConfig& config_;
    const TextConfig* shard_config_ = nullptr;
    std::int32_t shard_index_       = 0;
    // Peer shard registration for operations that run on one shard alone (see set_tp_peer).
    TextContext* peer_tp_    = nullptr;
    tp::DevicePair* pair_tp_ = nullptr;
    WorkspaceArena& work_;
    qwen3_5::PagedKVCacheView kv_;
    qwen3_5::PagedKVCacheView mtp_kv_;
    const qwen3_5::PagedKVCache* batch_text_kv_ = nullptr;
    const qwen3_5::PagedKVCache* batch_mtp_kv_  = nullptr;
    LinearAttentionStatePool& state_;
    qwen3_5::RoundState& io_;
    Tensor& prefill_hidden_;
    std::uint32_t prefill_chunk_;
    std::uint32_t prefill_overlap_;
    std::uint32_t text_kv_base_;
    const Tensor* active_cache_positions_                                          = nullptr;
    const Tensor* active_rope_positions_                                           = nullptr;
    const Tensor* active_kv_table_rows_                                            = nullptr;
    const Tensor* active_linear_state_source_slots_                                = nullptr;
    const Tensor* active_linear_state_destination_slots_                           = nullptr;
    const Tensor* active_valid_columns_                                            = nullptr;
    const Tensor* active_backend_kv_table_rows_                                    = nullptr;
    const ops::CausalAttentionExecutionEnvelope* active_causal_attention_envelope_ = nullptr;
    std::int32_t active_sequence_batch_                                            = 0;
    std::int32_t active_sequence_width_                                            = 0;
    std::int32_t rope_delta_                                                       = 0;
    std::int32_t linear_state_source_slot_                                         = 0;
    std::int32_t linear_state_destination_slot_                                    = 0;
    GdnStateAction gdn_state_action_          = GdnStateAction::UpdateInPlace;
    const GdnReplayRecords* replay_records_   = nullptr;
    std::int64_t prefill_split_frontier_      = -1;
    Tensor* rewrite_checkpoint_hidden_output_ = nullptr;
    std::uint32_t mtp_proposal_extent_        = 0;
    // Bytes each decode collective's rendezvous window may pull into L2, from this shard's own next
    // op (see tp_prefetch_window in text.cpp). 32 MiB is the measured optimum; see the constructor
    // for the sweep, and NINFER_TP2_DECODE_PREFETCH_MIB=0 keeps the window empty.
    std::size_t decode_prefetch_bytes_ = std::size_t{32} << 20;

    const Weight* embed_                        = nullptr;
    const Tensor* final_norm_                   = nullptr;
    const LinearParameters* lm_head_            = nullptr;
    const LinearParameters* proposal_head_      = nullptr;
    const std::int32_t* proposal_head_ids_      = nullptr;
    int proposal_head_n_                        = 0;
    const ops::SamplingConfig* sampling_config_ = nullptr;
    Tensor prefill_token_mask_;
    const MtpParameters* mtp_                   = nullptr;
    // DFlash2 selector weights when this shard materializes them (the one-device route, or the peer
    // shard under TP-2); the masked draft's shard drives the peer's copy instead.
    const SelectorParameters* selector_ = nullptr;
};

template <class Tap>
void TextContext::run_layers_tp2(TextContext& peer, tp::DevicePair& pair, Tensor& x, Tensor& x_peer,
                                 Phase ph, Tap& tap) {
    for (std::size_t layer = 0; layer < parameters_.text.layers.size(); ++layer) {
        const auto& block      = parameters_.text.layers[layer];
        const auto& block_peer = peer.parameters_.text.layers[layer];
        // Rotation across the two collectives of this layer: each one waits on the peer with the
        // memory system idle, so it pulls the bytes the next op is about to want. The mixer's window
        // fetches this layer's FFN, the FFN's window fetches the next layer's mixer. Prefill keeps
        // its own overlapped route and passes nothing here.
        tp::DevicePair::ArPrefetchWindow mixer_window;
        tp::DevicePair::ArPrefetchWindow ffn_window;
        if (ph != Phase::Prefill && decode_prefetch_bytes_ != 0) {
            mixer_window =
                tp_prefetch_window(tp_ffn_stream_weight(layer), peer.tp_ffn_stream_weight(layer));
            ffn_window = tp_prefetch_window(tp_mixer_stream_weight(layer + 1),
                                            peer.tp_mixer_stream_weight(layer + 1));
        }
        // One layer's activations are dead once the layer is done. Without this scope the two
        // deltas below would accumulate over all 64 layers, which only fits at T=1 - at a batched
        // prefill width each layer would leak 2 * hidden * T * 2 bytes (1.3 GiB at T=1024).
        auto layer_scope      = work_.scope();
        auto layer_scope_peer = peer.work_.scope();
        // Allocate both deltas before any mixer/FFN work: the mixer and FFN ops allocate in a
        // nested scope that frees on exit, so the deltas (allocated in this outer scope) survive
        // the ops and remain valid for the allreduce.
        Tensor mixer_delta      = work_.alloc(DType::BF16, {dimension(config_.hidden_size), x.ne[1]});
        Tensor mixer_delta_peer = peer.work_.alloc(DType::BF16, {dimension(config_.hidden_size), x.ne[1]});
        // A mixer whose output projection gathers its input columns stores those columns in an
        // order of its own that spans both shards' heads (a GGUF block checkpoint's tiled value
        // heads), so neither shard can project before the pair has assembled the full-width
        // activation the gather indexes. Each shard leaves its own heads in its half of the buffer,
        // the pair all-reduces the disjoint halves, and both then project the shared activation
        // into their partial delta.
        const std::int32_t activation_width = tp_mixer_activation_width(block);
        Tensor mixer_activation;
        Tensor mixer_activation_peer;
        Tensor* activation      = nullptr;
        Tensor* activation_peer = nullptr;
        if (activation_width > 0) {
            mixer_activation      = work_.alloc(DType::BF16, {activation_width, x.ne[1]});
            mixer_activation_peer = peer.work_.alloc(DType::BF16, {activation_width, x_peer.ne[1]});
            activation            = &mixer_activation;
            activation_peer       = &mixer_activation_peer;
        }
        ctx_.bind_to_current_thread();
        tp_mixer_layer(block, x, mixer_delta, layer, ph, activation);
        peer.ctx_.bind_to_current_thread();
        peer.tp_mixer_layer(block_peer, x_peer, mixer_delta_peer, layer, ph, activation_peer);
        if (activation != nullptr) {
            ctx_.bind_to_current_thread();
            pair.allreduce(activation->data, activation_peer->data, activation->bytes(),
                           ctx_.stream, peer.ctx_.stream);
            ctx_.bind_to_current_thread();
            tp_mixer_activation_delta(block, *activation, mixer_delta, layer, ph);
            peer.ctx_.bind_to_current_thread();
            peer.tp_mixer_activation_delta(block_peer, *activation_peer, mixer_delta_peer, layer, ph);
        }
        // The mixer output projections are row-parallel, so each shard holds a partial mixer
        // output; all-reduce the delta and add the sum to the shared residual once. The allreduce
        // stages both deltas on the compute streams that produced them (D2H after the mixer
        // kernels, H2D before residual_add), so no pre-sync is needed.
        ctx_.bind_to_current_thread();
        pair.allreduce(mixer_delta.data, mixer_delta_peer.data, mixer_delta.bytes(), ctx_.stream,
                       peer.ctx_.stream, mixer_window);
        ctx_.bind_to_current_thread();
        ops::residual_add(mixer_delta, x, ctx_.stream);
        peer.ctx_.bind_to_current_thread();
        ops::residual_add(mixer_delta_peer, x_peer, peer.ctx_.stream);
        Tensor delta      = work_.alloc(DType::BF16, {dimension(config_.hidden_size), x.ne[1]});
        Tensor delta_peer = peer.work_.alloc(DType::BF16, {dimension(config_.hidden_size), x.ne[1]});
        ctx_.bind_to_current_thread();
        tp_mlp_delta(block, x, delta, layer, ph);
        peer.ctx_.bind_to_current_thread();
        peer.tp_mlp_delta(block_peer, x_peer, delta_peer, layer, ph);
        ctx_.bind_to_current_thread();
        pair.allreduce(delta.data, delta_peer.data, delta.bytes(), ctx_.stream, peer.ctx_.stream,
                       ffn_window);
        ctx_.bind_to_current_thread();
        ops::residual_add(delta, x, ctx_.stream);
        peer.ctx_.bind_to_current_thread();
        ops::residual_add(delta_peer, x_peer, peer.ctx_.stream);
        if constexpr (Tap::enabled) {
            tap.capture_layer(static_cast<int>(layer), x, ctx_.stream);
        }
    }
}

} // namespace ninfer::models::qwen3_5::execution
