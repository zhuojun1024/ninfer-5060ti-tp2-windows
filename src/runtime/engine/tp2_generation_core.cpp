#include "runtime/engine/tp2_generation_core.h"

#include "runtime/engine/turn_replay.h"

#include "artifact/formats.h"
#include "artifact/reader.h"
#include "core/cyclic_kv_cache.h"
#include "core/layout.h"
#include "models/registry.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/frontend/tool_call_constraint.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/prefix_identity.h"
#include "models/qwen3_5/program/dflash_round.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/target_logprobs.h"
#include "ninfer/ops/token_mask.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::runtime {
namespace {

using namespace ninfer;
namespace qwen = ninfer::models::qwen3_5;
using Clock = std::chrono::steady_clock;

// Phase timing for the TP-2 prefill/decode walk, enabled with NINFER_TP2_TIMING=1. It exists to
// attribute a decode round to its phases while the round is being optimised (see the worklog, Round 11);
// with the flag off every call is one branch on a cached bool. The marks are recorded on shard A's
// stream and read back after that round's synchronization, so every event has completed and
// cudaEventElapsedTime cannot report a partial interval.
struct Tp2RoundTiming {
    bool enabled = false;
    bool events_ready = false;
    bool created = false;
    cudaEvent_t mark[7]{};
    double mtp_ms     = 0.0;
    double verify_ms  = 0.0;
    double accept_ms  = 0.0;
    double copy_ms    = 0.0;
    // The plain batched route records the same five marks for its own phase split - 0-1 the batched
    // decode step, 1-2 grammar masking, 2-3 sampling, 3-4 the logits readback - and keeps them in
    // their own accumulators so its report names each phase for what it is.
    double decode_ms   = 0.0;
    double grammar_ms  = 0.0;
    double sample_ms   = 0.0;
    double readback_ms = 0.0;
    double sync_ms    = 0.0;
    double round_ms   = 0.0;
    double prefill_ms = 0.0;
    std::uint64_t rounds    = 0;
    std::uint64_t committed = 0;
    // Plain route only: how many prefill chunks were stepped, and how many of them ran a decode
    // round for the lanes that already held a token. It is the direct evidence that a lane beside a
    // long prefill is served once per chunk instead of once per prompt.
    std::uint64_t prefill_chunks = 0;
    std::uint64_t pump_rounds    = 0;

    // The events are recorded on shard A's stream only, so they must be created with shard A's
    // device current: an event from another device makes cudaEventRecord fail with
    // cudaErrorInvalidResourceHandle, and because these paths do not check CUDA errors that latched
    // failure would be reported by the next unrelated CUDA_CHECK in the round.
    void init() {
        if (created) { return; }
        created = true;
        const char* env = std::getenv("NINFER_TP2_TIMING");
        enabled         = env != nullptr && env[0] == '1';
        if (!enabled) { return; }
        for (cudaEvent_t& event : mark) {
            if (cudaEventCreateWithFlags(&event, cudaEventDefault) != cudaSuccess) {
                (void)cudaGetLastError();
                enabled = false;
                return;
            }
        }
        events_ready = true;
    }

    // Drop a latched failure from a timing call itself: this facility must never change what the
    // engine reports, and every diagnostic call failure means "report nothing" for that interval.
    static void discard_error() { (void)cudaGetLastError(); }

    void reset() {
        mtp_ms = verify_ms = accept_ms = copy_ms = sync_ms = round_ms = prefill_ms = 0.0;
        decode_ms = grammar_ms = sample_ms = readback_ms = 0.0;
        prefill_chunks = 0;
        pump_rounds    = 0;
        rounds    = 0;
        committed = 0;
    }

    void record(int index, cudaStream_t stream) {
        if (!events_ready) { return; }
        if (cudaEventRecord(mark[index], stream) != cudaSuccess) {
            discard_error();
            events_ready = false;
        }
    }

    [[nodiscard]] double elapsed(int begin, int end) const {
        if (!events_ready) { return 0.0; }
        float ms = 0.0F;
        if (cudaEventElapsedTime(&ms, mark[begin], mark[end]) != cudaSuccess) {
            discard_error();
            return 0.0;
        }
        return ms;
    }

    // Called once per round after that round's synchronization: every mark up to the licenced-token
    // copy has completed, so the intervals are final. The fold runs after the synchronization, and
    // the speculative report states its cost as the round's remainder instead of a separate event
    // pair: elapsed() queried inside the round reads a mark pair that has not completed and always
    // reported 0.00 (see report()).
    void close_round(double sync_wait_ms) {
        if (!events_ready) { return; }
        mtp_ms += elapsed(0, 1);
        verify_ms += elapsed(1, 2);
        accept_ms += elapsed(2, 3);
        copy_ms += elapsed(3, 4);
        sync_ms += sync_wait_ms;
    }

    // The plain batched route's round split, in its own labels. It has no draft, verify or accept
    // phase, so it reports none of them; the prefill of the lane each round served is reported per
    // request by the serving layer instead.
    void close_round_plain(double sync_wait_ms) {
        if (!events_ready) { return; }
        decode_ms += elapsed(0, 1);
        grammar_ms += elapsed(1, 2);
        sample_ms += elapsed(2, 3);
        readback_ms += elapsed(3, 4);
        sync_ms += sync_wait_ms;
    }

    void report_plain(const char* label) const {
        if (!enabled || rounds == 0) { return; }
        const double n = static_cast<double>(rounds);
        std::fprintf(stderr,
                     "[tp2-time] %s rounds=%llu committed=%llu avg_round=%.2fms decode=%.2f "
                     "grammar=%.2f sample=%.2f readback=%.2f sync_wait=%.2f prefill_chunks=%llu "
                     "pump_rounds=%llu\n",
                     label, static_cast<unsigned long long>(rounds),
                     static_cast<unsigned long long>(committed), round_ms / n, decode_ms / n,
                     grammar_ms / n, sample_ms / n, readback_ms / n, sync_ms / n,
                     static_cast<unsigned long long>(prefill_chunks),
                     static_cast<unsigned long long>(pump_rounds));
    }

    void report(const char* label) const {
        if (!enabled || rounds == 0) { return; }
        const double n = static_cast<double>(rounds);
        std::fprintf(stderr,
                     "[tp2-time] %s rounds=%llu committed=%llu avg_round=%.2fms mtp=%.2f verify=%.2f "
                     "accept=%.2f copy=%.2f sync_wait=%.2f rest=%.2f prefill=%.1fms prefill_chunks=%llu pump_rounds=%llu\n",
                     label, static_cast<unsigned long long>(rounds),
                     static_cast<unsigned long long>(committed), round_ms / n, mtp_ms / n,
                     verify_ms / n, accept_ms / n, copy_ms / n, sync_ms / n,
                     (round_ms - mtp_ms - verify_ms - accept_ms - copy_ms) / n, prefill_ms,
                     static_cast<unsigned long long>(prefill_chunks),
                     static_cast<unsigned long long>(pump_rounds));
    }
};

Tp2RoundTiming& tp2_timing() {
    static Tp2RoundTiming timing;
    return timing;
}


// Workspace arena per shard. The single-token forward_tp2 peaks at ~600 KB (full-vocab logits
// plus a handful of [N,1] activations), but a batched prefill chunk holds every intermediate of the
// widest layer at once: at the maximum chunk width the fused FFN gate/up [17408, T] BF16 alone is
// ~36 MiB and the per-layer peak is ~140 MiB. 192 MiB covers the largest chunk with headroom while
// keeping the per-shard steady state (weights 10.7 GiB + KV + GDN state + snapshots + workspace)
// under the 16 GB card limit. The context ceiling is set by the KV pool, so this budget is spent
// exactly: at 262,144 tokens the 128 MiB a 384 MiB arena would hold is the difference between
// fitting the card and not, and the oversized arena overflow is a reported error, not corruption.
constexpr std::size_t kWorkspaceBytes = 192ULL << 20;
// Causal scoring adds buffers the generation route never holds at once, because the forward writes
// every position's logits instead of only the last one. Three of them scale with the tile, i.e. the
// forward's chunk width, and at the packed 248,320-token vocabulary they are large:
//   * the caller's per-column logits tile, [vocab, T] BF16                  = 2 * vocab * T
//   * the vocabulary-split head's own partial block, [vocab/2, T] BF16      = 1 * vocab * T
//   * that projection's activation workspace, an FP32 [vocab/2, T] scratch  = 2 * vocab * T
// The peer shard holds the same three (the forward allocates the peer's logits mirror itself), so
// both arenas pay 5 * vocab * T. At 1024 columns that is 1.2 GiB per shard -- far beyond the whole
// generation arena -- and 512 columns already needs 606 MiB. 256 columns needs 303 MiB, which fits
// beside the 262,144-token KV pool (~1 GiB per shard) with room to spare, and it keeps the walk at
// four forwards per 1024 tokens. The arena is sized from this same constant, so tile and arena move
// together.
constexpr std::uint32_t kScoreTile = 256;
// Arena headroom for the three buffers above, in units of one [vocab, kScoreTile] BF16 tile. They
// total 5 * vocab * T / (2 * vocab * T) = 2.5 such tiles; two whole tiles of extra extent leaves the
// four-way split head and its workspace slack, and the arena keeps kWorkspaceBytes underneath so a
// nested scope inside the forward can still reach the generation allowance.
constexpr std::size_t kScoreWorkspaceTiles = 2;
// Smallest Vision item ceiling the route will fall back to when the full envelope does not fit
// beside the KV pool: below this an ordinary photo would no longer fit in one item, so the route
// reports the failure instead of silently admitting only thumbnails.
constexpr std::uint32_t kVisionItemTokenFloor = 2048;

// TP-2 keeps one resident conversation plus this many host-resident ones unless
// --max-private-continuations overrides it. The host KV budget is the real bound on how many fit.
constexpr std::uint32_t kTp2DefaultSessions = 6;

// Upper bound on the batched TP-2 prefill chunk width. One chunk reads every weight once, so the
// per-token weight traffic that dominates the single-token walk is amortized over the whole chunk;
// the cross-device allreduce bytes per token and the tensor-core work per token do not shrink with
// the chunk, so widening the chunk trades only the weight term against the per-chunk activation
// peak. The engine option (--prefill-chunk) picks the actual width.
constexpr std::uint32_t kPrefillChunkMaximum = 1024;
// Finest spacing between prefix-reuse checkpoints in pinned host memory, in prompt tokens. It
// matches the checkpoint step llama.cpp uses by default. A ring of N slots widens it to
// ceil(max_context / N) so that N * stride always covers the whole context; a prompt that diverges
// inside the previous prompt then restarts at most one stride behind its shared prefix.
constexpr std::uint32_t kReuseCheckpointStride = 8192;
// The last few chunk ends of a walk are checkpointed on top of the stride grid. A chat client
// re-renders the assistant turn it is about to continue, so the divergence a turn boundary shows
// sits just behind the previous prompt's end - observed gaps run from ~150 tokens to tens of
// thousands. The grid alone only guarantees a checkpoint within one stride of it; this window
// makes the common small gap land within one chunk. llama.cpp calls the same idea a "near prompt
// end" checkpoint (its checkpoint-min-step does not apply there either).
constexpr std::uint32_t kReuseTailWindow = 8192;
// Host ring slots reserved for those tail anchors. Without the reservation a long session fills the
// ring with them (each prefill leaves up to one per chunk) and evicts the position grid, which is
// exactly the coverage a divergence deep inside the history needs.
constexpr std::uint32_t kReuseTailCheckpointCount = 8;
// Spacing of the anchors written while an answer is generated. The position grid only fires on
// absolute stride multiples, so an answer that never crosses one leaves the prompt end as the only
// boundary behind it: a client that re-renders that answer then re-prefills the whole thing (an
// observed 20,319-token answer crossed no grid point and fell back 22,989 tokens). One anchor per
// this many committed tokens, written into the tail ring the prefill already reserves, bounds that
// fallback to a few thousand tokens without spending a slot. The ring's tail count widens the
// coverage: seven slots cover 57K generated tokens before the oldest anchor rotates out.
constexpr std::uint32_t kReuseGeneratedTailStride = 8192;
// Host ring slots reserved for the divergence anchor: the position where the last prefill's prompt
// stopped matching the lineage it inherited. A new session - and a context compression - shares only
// the client's stable system prompt and tool definitions with the conversation before it, so its
// divergence lands well inside the previous prompt. Neither other group reaches there: the position
// grid's first checkpoint sits a whole stride in, and a small ring widens that stride to tens of
// thousands of tokens; the tail anchors sit at the opposite end. One slot is enough because a client
// renders its stable prefix once, and because the ring prunes on a single lineage: a checkpoint past
// the newest shared prefix is dead for good, so a second slot could never hold a rival prefix.
constexpr std::uint32_t kReuseDivergenceCheckpointCount = 1;
// Slots for the stable block's own anchor: the boundary the prompt itself names as the end of its
// leading instruction block. It needs a slot of its own because the observed divergence owns the one
// above and both boundaries matter in the same walk - the observed one is where this prompt stopped
// matching the conversation before it, the block one is where the next conversation of the agent will.
constexpr std::uint32_t kReuseBlockCheckpointCount = 1;
// How far before the observed divergence the anchor is placed. A shared prefix is only reusable
// while every later prompt agrees on every token before it, and the index where two prompts first
// differ is where the tokenizer stopped emitting the same ids - the token that straddles the stable
// block's end absorbs a different amount of what follows it in each pair. The next request that
// renders the same block can therefore report a shared prefix one or two tokens shorter than the one
// that froze the anchor, and the prune above would then discard it. The margin trades single-digit
// tokens of recompute for keeping the whole prefix.
constexpr std::uint32_t kReuseDivergenceMargin = 8;

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    return tokens == 0 ? 0U : 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

// WSL2 re-enumerates GPUs on every VM restart, so the indices passed via --devices can
// silently point at the wrong cards (e.g. the Tesla T10 instead of a second RTX 5060 Ti).
// The engine is sm_120a-only, so a mismatched card fails deep inside the first kernel launch
// with a cryptic cudaErrorSymbolNotFound. Fail fast at construction with an actionable message.
// '--devices' takes CUDA runtime indices. The default CUDA_DEVICE_ORDER is fastest-first, which
// need not match the PCI-bus order that 'nvidia-smi -L' prints, so the actionable form of the hint
// is the CUDA-visible device list itself.
std::string cuda_visible_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) { return {}; }
    std::string out;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) { continue; }
        if (!out.empty()) { out += ", "; }
        out += std::to_string(i) + "=" + prop.name + " (sm_" +
               std::to_string(prop.major * 10 + prop.minor) + ")";
    }
    return out;
}

void validate_tp2_devices(const DeviceContext& a, const DeviceContext& b) {
    const int cap_a = a.compute_capability();
    const int cap_b = b.compute_capability();
    if (cap_a != 120 || cap_b != 120) {
        throw std::invalid_argument(
            "TP-2 requires two sm_120a devices (compute capability 12.0); device " +
            std::to_string(a.device) + " (" + a.props.name + ") is sm_" + std::to_string(cap_a) +
            " and device " + std::to_string(b.device) + " (" + b.props.name + ") is sm_" +
            std::to_string(cap_b) +
            ". WSL2 re-enumerates GPUs across VM restarts - pass the two RTX 5060 Ti indices via "
            "--devices. CUDA-visible devices: " +
            cuda_visible_devices());
    }
    if (std::strcmp(a.props.name, b.props.name) != 0) {
        throw std::invalid_argument(
            "TP-2 requires identical devices; device " + std::to_string(a.device) + " is " +
            a.props.name + " and device " + std::to_string(b.device) + " is " + b.props.name +
            ". Pass two identical devices via --devices. CUDA-visible devices: " +
            cuda_visible_devices());
    }
}

ops::SamplingConfig make_sampling_config(const ResolvedSamplingParameters& source) {
    ops::SamplingConfig out;
    out.temperature       = source.temperature;
    out.top_k             = source.top_k;
    out.top_p             = source.top_p;
    out.min_p             = source.min_p;
    out.presence_penalty  = source.presence_penalty;
    out.frequency_penalty = source.frequency_penalty;
    out.seed              = source.seed;
    out.token_counts      = nullptr;
    return out;
}

// Moves a workspace arena to an exact offset. A verify graph bakes the addresses of the round's
// scratch, so a replay must reproduce them rather than hope two rounds happened to allocate the
// same amount. The arena never moves below `floor`, the watermark the current request keeps alive.
void position_arena(WorkspaceArena& arena, std::size_t floor, std::size_t target) {
    if (target < floor) {
        throw std::logic_error("TP-2 verify CUDA Graph was captured below this request's workspace");
    }
    const std::size_t used = arena.used();
    if (target < used) {
        arena.rewind(target);
    } else if (target > used) {
        (void)arena.alloc_bytes(target - used);
    }
}

// Closes the DevicePair's capture window on every exit. capture_group rolls the CUDA stream capture
// back when its body throws, but it cannot reset the pair's own capture state: a left-open window
// makes every later capture throw and makes later eager collectives take the captured id path.
struct ArCaptureGuard {
    tp::DevicePair& pair;
    ~ArCaptureGuard() { pair.end_capture(); }
};


// Prefix-reuse trace helper: a divergence is a byte-level disagreement, so the diagnostic has to
// show the bytes either side of it with the whitespace that separates two renderings made visible.
void print_escaped_bytes(std::string_view text, std::size_t from, std::size_t length) {
    constexpr std::size_t kMaximumBytes = 160;
    for (std::size_t index = from, printed = 0;
         index < text.size() && printed < length && printed < kMaximumBytes; ++index, ++printed) {
        const auto byte = static_cast<unsigned char>(text[index]);
        switch (byte) {
            case '\n': std::fputs("\\n", stderr); break;
            case '\r': std::fputs("\\r", stderr); break;
            case '\t': std::fputs("\\t", stderr); break;
            case '"': std::fputs("\\\"", stderr); break;
            case '\\': std::fputs("\\\\", stderr); break;
            default:
                if (byte < 0x20U || byte == 0x7fU) { std::fprintf(stderr, "\\x%02x", byte); }
                else { std::fputc(static_cast<char>(byte), stderr); }
                break;
        }
    }
}

void print_reuse_span(const char* label, const TP2GenerationCore& core,
                      std::span<const TokenId> tokens) {
    const std::string text = core.frontend().decode_tokens(tokens);
    std::fprintf(stderr, " %s[%zu]=\"", label, tokens.size());
    print_escaped_bytes(text, 0, text.size());
    std::fputs("\"", stderr);
}

} // namespace

TP2GenerationCore::TP2GenerationCore(const EngineOptions& options, int device_a, int device_b)
    : options_(options), pair_(device_a, device_b) {
    const Clock::time_point load_start = Clock::now();
    shard_a_.device = DeviceContext(device_a);
    shard_b_.device = DeviceContext(device_b);
    validate_tp2_devices(shard_a_.device, shard_b_.device);
    lanes_ = (options_.max_concurrency == 0 ? 1U : options_.max_concurrency);

    models::LoadOptions load;
    load.vision         = options.enable_vision;
    load.speculative    = options.speculative.backend;
    load.proposal_head  = options.speculative.proposal_head;
    load.embedding_host = options.embedding_host;

    mtp_enabled_ = options.speculative.backend == SpeculativeBackend::Mtp;
    if (mtp_enabled_) {
        mtp_drafts_ = std::clamp<std::uint32_t>(options.speculative.draft_tokens, 1U, 5U);
    }
    dflash2_enabled_ = options.speculative.backend == SpeculativeBackend::DFlash2;
    if (dflash2_enabled_) {
        dflash_drafts_ = std::clamp<std::uint32_t>(options.speculative.draft_tokens, 1U,
                                                   qwen::kDFlashDecodeMaximumDrafts);
    }
    // The plain, MTP and DFlash2 rounds all carry a batch (tp2-plan-concurrency.md P2.1c/P2.1b);
    // only `--spec dflash` has no TP-2 context layout, so a multi-lane run on that route would
    // silently feed every lane through row 0. Collapse it and report. The plain route keeps its
    // lanes.
    //
    // `normalize_engine_options` already applies `tp2_generation_concurrency`, so a service built
    // from normalized options never reaches this branch; it is the belt-and-braces guard for a core
    // constructed directly from un-normalized options, and it reports rather than corrupting.
    if (lanes_ > 1U && !mtp_enabled_ && !dflash2_enabled_ &&
        options_.speculative.backend != SpeculativeBackend::None) {
        std::fprintf(stderr,
                     "[tp2-lane] --max-concurrency %u collapses to 1 lane on the speculative "
                     "route (only the MTP and DFlash2 rounds are batched; see "
                     "tp2-plan-concurrency.md)\n",
                     lanes_);
        lanes_ = 1U;
    }

    artifact::Reader reader(options.artifact_path);
    auto plan = qwen::plan_load(reader, load);
    auto [model_a, model_b] =
        qwen::materialize_model_tp2(std::move(plan), shard_a_.device, shard_b_.device,
                                    &options.startup_observer);
    shard_a_.model = std::move(model_a);
    shard_b_.model = std::move(model_b);

    shard_a_.parameters = std::make_unique<qwen::execution::Parameters>(*shard_a_.model);
    shard_b_.parameters = std::make_unique<qwen::execution::Parameters>(*shard_b_.model);

    // Prefix-reuse checkpoints in pinned host memory: the same budget the single-device context
    // cache uses for its complete host state images. Each checkpoint is one state image per shard
    // (~73 MiB), so the ring is enough to cover the whole context at kReuseCheckpointStride or
    // coarser; a zero budget keeps the ring disabled, which is what --no-prefix-reuse selects.
    {
        const std::uint32_t host_slots = options_.context_cache.host_state_images;
        session_retention_floor_tokens_ = options_.context_cache.session_retention_floor_tokens;
        // tp2-plan-concurrency.md 12.6: a checkpoint is one compact single-lane state image, and
        // each lane owns its own slice of the ring, so the pinned budget is the lanes=1 budget
        // whatever the lane count. The batch executors recall into their own lane (Stage 1d), so
        // the ring is live at every lane count.
        if (host_slots != 0) {
            host_checkpoint_divergence_slots_ = kReuseDivergenceCheckpointCount;
            host_checkpoint_block_slots_      = kReuseBlockCheckpointCount;
            const std::uint32_t total = host_slots + host_checkpoint_divergence_slots_ +
                                        host_checkpoint_block_slots_;
            host_checkpoint_slots_per_lane_ = std::max(1U, total / lanes_);
            const std::uint32_t per_lane = host_checkpoint_slots_per_lane_;
            const std::uint32_t fixed =
                host_checkpoint_divergence_slots_ + host_checkpoint_block_slots_;
            const std::uint32_t flexible = per_lane > fixed ? per_lane - fixed : 0U;
            host_checkpoint_tail_slots_ =
                std::max(1U, std::min(kReuseTailCheckpointCount, flexible / 2U));
            const std::uint32_t used = fixed + host_checkpoint_tail_slots_;
            host_checkpoint_grid_slots_ = per_lane > used ? per_lane - used : 1U;
            const std::uint32_t per_slot =
                (options_.max_context + host_checkpoint_grid_slots_ - 1U) /
                host_checkpoint_grid_slots_;
            const std::uint32_t rounded  = (per_slot + 127U) / 128U * 128U;
            host_checkpoint_stride_      = std::max(kReuseCheckpointStride, rounded);
            // The decision layer. It owns the positions and their slots; the ring above owns the
            // images. The slot count is the ring's, because for now the ring allocates them.
            reuse_ledger_ = std::make_unique<ReuseLedger>(
                ReuseBudget{host_checkpoint_stride_, lanes_ * host_checkpoint_slots_per_lane_ +
                                                         lanes_});
        }
        // The masked-draft route carries its draft context through this same ring (stage B6): each
        // checkpoint holds the target state image and the draft ring at the same frontier, so a
        // reused prefix restores the context the skipped tokens produced.
    }

    // Cross-session retention budget. --host-kv-mib is the whole host KV budget, split evenly
    // between the two shards; the arenas that back it are built on the first eviction, so a workload
    // that never leaves one conversation never allocates it. The backing is pageable unless
    // --host-kv-pinned asks for it. A zero budget keeps the catalog empty, which is exactly the
    // pre-retention behaviour.
    {
        const std::size_t host_kv_bytes = options_.context_cache.host_kv_capacity_bytes;
        const std::uint32_t sessions =
            options_.context_cache.max_private_continuations.value_or(kTp2DefaultSessions);
        // One entry is the resident conversation; retention needs room for at least one more.
        // DFlash2 takes part: a session slab holds the draft ring beside the target KV and GDN
        // state, so a recall restores the whole conversation rather than only its target half.
        // A session image carries the same per-lane state, so the catalog splits its slabs by lane
        // (tp2-plan-concurrency.md 12.6).
        if (host_kv_bytes / 2 != 0 && sessions > 1) {
            host_kv_shard_bytes_ = host_kv_bytes / 2;
            session_capacity_    = sessions;
        }
        std::fprintf(stderr,
                     "[mem] host sessions capacity %zu | host KV %.1f MiB/shard | retention %s\n",
                     session_capacity_, static_cast<double>(host_kv_shard_bytes_) / 1048576.0,
                     session_capacity_ == 0 ? "disabled" : "enabled");
    }

    // The text KV pool is one physical allocation shared by every lane. A multi-lane route hands it
    // out dynamically: a request reserves exactly the pages its own prompt plus output budget need
    // when it is admitted and returns them when it retires, so a lane that runs alone may use the
    // whole pool instead of a fixed 1/lanes slice (tp2-plan-kv-sharing.md). The admission
    // ceiling is a policy value clamped to the pool minus the route's write margin; the clamp
    // guarantees that a request admitted at this ceiling always finds its pages, which is what keeps
    // the executors' requeue path from waiting forever.
    //
    // Single-lane keeps reading options_.max_context directly: the page rounding above must not
    // widen the ceiling it has always advertised.
    if (lanes_ > 1) {
        const std::uint32_t pool_tokens =
            pages_for_tokens(options_.max_context) * static_cast<std::uint32_t>(kPagedKVPageSize);
        const std::uint32_t margin = lane_kv_margin();
        if (pool_tokens <= margin) {
            throw std::logic_error("TP-2 KV pool is too small for the route's write margin");
        }
        lane_context_limit_ = pool_tokens - margin;
        // `--lane-context` is the operator's ceiling for one lane (llama.cpp's
        // --kv-unified-per-slot): a smaller value caps what one lane may reserve, so four large
        // requests can still run four-up instead of the first one taking the whole pool. A larger
        // value is clamped to the pool ceiling above, which is the only value that cannot
        // deadlock.
        if (options_.lane_context != 0U) {
            if (options_.lane_context < lane_context_limit_) {
                lane_context_limit_ = options_.lane_context;
            }
            std::fprintf(stderr,
                         "[mem] TP-2 lane admission ceiling %u tokens (--lane-context %u)\n",
                         lane_context_limit_, options_.lane_context);
        } else {
            // No ceiling was given, so the pool's own limit stands and a lane that arrives first may
            // reserve all of it: the requests queued behind it then cannot be admitted alongside it
            // even though the route has the slots, which is what the whole pool looks like from the
            // outside. Say so instead of silently applying a ceiling, because a hard default of
            // pool/lanes_ would make submit() reject a prompt longer than one lane's share even when
            // that lane is the only request in the system.
            std::fprintf(stderr,
                         "[mem] TP-2 lane admission ceiling %u tokens (the whole KV pool): one lane "
                         "may reserve all of it, so --max-concurrency %u requests cannot be admitted "
                         "together until it retires; set --lane-context to cap one lane\n",
                         lane_context_limit_, lanes_);
        }
        // The batched executors stage their per-lane operands here, one section per operand with
        // the lane index as the stride. A captured batch step reads these addresses through memcpy
        // nodes, so the buffer may not move between rounds (P2.2b).
        batch_lane_host_ = std::make_unique<PinnedHostBuffer>(
            5 * static_cast<std::size_t>(lanes_) * sizeof(std::int32_t), true);
    }

    if (lanes_ == 1U && options_.lane_context != 0U) {
        // The single lane owns the whole context by construction, so a ceiling here would only
        // shorten the one request the route can run. Say so instead of applying it silently.
        std::fprintf(stderr,
                     "[mem] --lane-context %u is ignored at --max-concurrency 1: the single "
                     "lane owns the whole context\n",
                     options_.lane_context);
    }

    build_shard(shard_a_, 0);
    build_shard(shard_b_, 1);
    if (session_capacity_ != 0) {
        // Report the budget in the unit an operator sizes against: full-context conversations per
        // shard. The MTP layer's own slab is not counted, so shard 0 holds slightly fewer.
        const HostKVPageLayout layout =
            plan_host_kv_page_layout(shard_a_.decoder->text_kv.page_pool().geometry());
        const std::size_t full_session =
            layout.page_stride * pages_for_tokens(options_.max_context);
        const std::size_t fits = full_session == 0 ? 0 : host_kv_shard_bytes_ / full_session;
        std::fprintf(stderr,
                     "[tp2-session] host budget %.1f MiB/shard holds %zu of %zu full-context "
                     "sessions (%.1f MiB each)\n",
                     static_cast<double>(host_kv_shard_bytes_) / 1048576.0, fits,
                     session_capacity_ - 1,
                     static_cast<double>(full_session) / 1048576.0);
    }
    // Column-split weights (the token embedding) are consumed by operations that run on one shard
    // alone, so both contexts learn their peer and the pair once both shards exist.
    shard_a_.context->set_tp_peer(shard_b_.context.get(), &pair_);
    shard_b_.context->set_tp_peer(shard_a_.context.get(), &pair_);

    if ((mtp_enabled_ || dflash2_enabled_) && pair_.in_kernel_allreduce()) {
        const char* env        = std::getenv("NINFER_TP2_VERIFY_GRAPH");
        // A captured window bakes one lane's state-slot and KV-row bindings and freezes the
        // workspace watermark it was captured at, so the batched route runs eager and the multi-lane
        // captures land with the rest of P2.2.
        verify_graph_enabled_  = (env == nullptr || env[0] != '0') && lanes_ == 1U;
    }
    {
        const char* env = std::getenv("NINFER_TP2_MTP_CHAIN_GRAPH");
        if (env != nullptr && std::strcmp(env, "0") == 0) {
            mtp_chain_mode_ = StepLaunchMode::EagerBucket;
        } else if (env != nullptr && std::strcmp(env, "exact") == 0) {
            mtp_chain_mode_ = StepLaunchMode::EagerExact;
        }
        if (!pair_.in_kernel_allreduce()) { mtp_chain_mode_ = StepLaunchMode::EagerExact; }
    }
    if (mtp_enabled_ || dflash2_enabled_) {
        // The verify window is assembled in this portable pinned buffer every round: the capture
        // path reads it through memcpy nodes, and the eager path reads it directly. MTP assembles
        // it from the host proposal chain; DFlash2 reads the verify ids/positions the proposal
        // published back from the frame. Both use the same [ids(W), positions(W), valid(1)] layout;
        // the trailing int carries the clamp extent, which varies per round and therefore reaches
        // the captured graph the way ids/positions do.
        const std::uint32_t width = (mtp_enabled_ ? mtp_drafts_ : dflash_drafts_) + 1U;
        verify_window_host_ = std::make_unique<PinnedHostBuffer>(
            static_cast<std::size_t>(2) * width * sizeof(std::int32_t) + sizeof(std::int32_t), true);
    }
    if (mtp_enabled_ || dflash2_enabled_) {
        // The envelope buckets are the single-GPU decode graph's split-policy boundaries, so a
        // window always covers the same attention route and launch geometry. The bucket is keyed by
        // the window's widest visible extent, which is what the envelope carries; both the capture
        // and the eager reference use them, so the two differ only in how they are launched. The
        // masked-draft route has its own profile set and target envelope: the single-GPU DFlash2
        // graph captures {1, frontier + width} per profile.
        const std::uint32_t window_width = (mtp_enabled_ ? mtp_drafts_ : dflash_drafts_) + 1U;
        const auto profiles =
            mtp_enabled_
                ? qwen::detail::mtp_graph_profiles(options_.max_context, mtp_drafts_)
                : qwen::detail::dflash_graph_profiles(SpeculativeBackend::DFlash2,
                                                      options_.max_context, dflash_drafts_, 1);
        for (const auto& profile : profiles) {
            WindowGraph graph;
            graph.visible_begin = mtp_enabled_ ? profile.min + 1U : 1U;
            graph.visible_end   = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                options_.max_context, static_cast<std::uint64_t>(profile.max) + window_width));
            verify_graphs_.push_back(std::move(graph));
        }
        // The batch form of the window (P2.2c). Every per-lane binding is read from this pinned
        // member through memcpy nodes, so one capture covers any assignment of a lane count; a
        // one-column batch still bakes the GDN slot its scalar path publishes and therefore needs
        // one graph per lane. The sections are laid out with the startup lane count, not the
        // round's live one, so a capture replays on the addresses it baked.
        batch_window_host_ = std::make_unique<PinnedHostBuffer>(
            (static_cast<std::size_t>(2) * window_width + 3U) * lanes_ * sizeof(std::int32_t), true);
        const char* batch_env = std::getenv("NINFER_TP2_VERIFY_BATCH_GRAPH");
        if (lanes_ > 1U && pair_.in_kernel_allreduce() &&
            !(batch_env != nullptr && std::strcmp(batch_env, "exact") == 0)) {
            verify_batch_mode_ = (batch_env != nullptr && std::strcmp(batch_env, "0") == 0)
                                     ? StepLaunchMode::EagerBucket
                                     : StepLaunchMode::Graph;
            for (const auto& profile : profiles) {
                const std::uint32_t visible_begin = mtp_enabled_ ? profile.min + 1U : 1U;
                const std::uint32_t visible_end = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                    options_.max_context, static_cast<std::uint64_t>(profile.max) + window_width));
                for (std::uint32_t batch = 1; batch <= lanes_; ++batch) {
                    if (batch > 1U) {
                        WindowGraph graph;
                        graph.visible_begin = visible_begin;
                        graph.visible_end   = visible_end;
                        graph.batch         = static_cast<std::int32_t>(batch);
                        verify_batch_graphs_.push_back(std::move(graph));
                        continue;
                    }
                    for (std::uint32_t slot = 0; slot < lanes_; ++slot) {
                        WindowGraph graph;
                        graph.visible_begin = visible_begin;
                        graph.visible_end   = visible_end;
                        graph.batch         = 1;
                        graph.lane          = static_cast<std::int32_t>(slot);
                        verify_batch_graphs_.push_back(std::move(graph));
                    }
                }
            }
        }
    }
    // The draft chain is replayed only by the single-lane serial walk (execute_walk ->
    // mtp_propose_window). The multi-lane route runs its MTP steps through the batched entry points,
    // so building these profiles and reserving their rendezvous channels there is dead weight
    // (P3.1). The same gate keeps the startup log from advertising a chain the run never uses.
    if (mtp_enabled_ && lanes_ == 1U) {
        // The draft chain is the same kind of object as the verify window: one captured sequence per
        // envelope bucket whose only per-round inputs are the anchor token and its position scalars.
        // They reach it through this pinned [anchor, position, position+1, drafts(K)] buffer, which
        // memcpy nodes re-read on every replay, and the chain publishes its drafts back into the
        // trailing slots. The buckets cover the chain's widest step (the last autoregressive one).
        mtp_chain_host_ = std::make_unique<PinnedHostBuffer>(
            static_cast<std::size_t>(3 + mtp_drafts_) * sizeof(std::int32_t), true);
        for (const auto& profile :
             qwen::detail::mtp_graph_profiles(options_.max_context, mtp_drafts_)) {
            WindowGraph graph;
            graph.visible_begin = profile.min + 1U;
            graph.visible_end   = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                options_.max_context, static_cast<std::uint64_t>(profile.max) + mtp_drafts_ + 1U));
            mtp_chain_graphs_.push_back(std::move(graph));
        }
    }

    // The plain (non-speculative) decode step is the same kind of object as the verify window: one
    // token, the same launch sequence per round, and its only per-round inputs are the decoded token,
    // its absolute position and the attention envelope. Its buckets are therefore the ordinary
    // (one-token visible window) split-policy boundaries, exactly the set the single-GPU ordinary
    // decode graph uses. The step contains the pair's allreduce, so it may only be captured when
    // that transport is the in-kernel one: the P2P and host-staging transports synchronize the
    // caller's streams, which a captured sequence cannot express.
    {
        const char* env = std::getenv("NINFER_TP2_DECODE_GRAPH");
        if (env != nullptr && std::strcmp(env, "0") == 0) {
            decode_step_mode_ = StepLaunchMode::EagerBucket;
        } else if (env != nullptr && std::strcmp(env, "exact") == 0) {
            decode_step_mode_ = StepLaunchMode::EagerExact;
        }
        if (!pair_.in_kernel_allreduce()) { decode_step_mode_ = StepLaunchMode::EagerExact; }
        // The single-lane step bakes its KV execution row and GDN slot into the captured sequence
        // (lane 0), so the graph cannot be replayed on behalf of another lane. The multi-lane route
        // runs the step eagerly, where the lane is an ordinary forward argument (P1.4).
        if (lanes_ > 1) { decode_step_mode_ = StepLaunchMode::EagerExact; }
        if (decode_step_mode_ != StepLaunchMode::EagerExact) {
            for (const auto& profile : qwen::detail::ordinary_graph_profiles(options_.max_context)) {
                WindowGraph graph;
                graph.visible_begin = profile.min + 1U;
                graph.visible_end   = std::min(options_.max_context, profile.max + 1U);
                decode_graphs_.push_back(std::move(graph));
            }
        }
        decode_window_host_ = std::make_unique<PinnedHostBuffer>(
            static_cast<std::size_t>(2) * sizeof(std::int32_t), true);
        // The batched plain step on the multi-lane route. Its lanes are bound from device memory as
        // soon as there are two of them, so one graph covers every assignment of a given lane count;
        // a one-column batch still runs the scalar GDN path, whose slot is a host value the capture
        // bakes and therefore needs one graph per lane. The single-lane vector above is unreachable
        // here: the multi-lane route forces the scalar step to eager.
        {
            const char* batch_env = std::getenv("NINFER_TP2_DECODE_BATCH_GRAPH");
            const bool capture_batch =
                lanes_ > 1 && pair_.in_kernel_allreduce() &&
                !(batch_env != nullptr && std::strcmp(batch_env, "exact") == 0);
            if (capture_batch) {
                decode_batch_mode_ = (batch_env != nullptr && std::strcmp(batch_env, "0") == 0)
                                         ? StepLaunchMode::EagerBucket
                                         : StepLaunchMode::Graph;
                for (const auto& profile :
                     qwen::detail::ordinary_graph_profiles(options_.max_context)) {
                    const std::uint32_t visible_begin = profile.min + 1U;
                    const std::uint32_t visible_end =
                        std::min(options_.max_context, profile.max + 1U);
                    for (std::uint32_t width = 1; width <= lanes_; ++width) {
                        if (width > 1) {
                            WindowGraph graph;
                            graph.visible_begin = visible_begin;
                            graph.visible_end   = visible_end;
                            graph.batch         = static_cast<std::int32_t>(width);
                            decode_batch_graphs_.push_back(std::move(graph));
                            continue;
                        }
                        for (std::uint32_t slot = 0; slot < lanes_; ++slot) {
                            WindowGraph graph;
                            graph.visible_begin = visible_begin;
                            graph.visible_end   = visible_end;
                            graph.batch         = 1;
                            graph.lane          = static_cast<std::int32_t>(slot);
                            decode_batch_graphs_.push_back(std::move(graph));
                        }
                    }
                }
            }
        }
        std::fprintf(stderr,
                     "[tp2-graph] plain decode step: %s | batched: %s | verify step: %s | "
                     "verify batch: %s | mtp chain: %s\n",
                     decode_step_mode_ == StepLaunchMode::Graph        ? "graph"
                     : decode_step_mode_ == StepLaunchMode::EagerBucket ? "eager(bucket)"
                                                                       : "eager(exact)",
                     decode_batch_mode_ == StepLaunchMode::Graph        ? "graph"
                     : decode_batch_mode_ == StepLaunchMode::EagerBucket ? "eager(bucket)"
                                                                        : "eager(exact)",
                     verify_graph_enabled_ ? "graph" : "eager",
                     verify_batch_mode_ == StepLaunchMode::Graph        ? "graph"
                     : verify_batch_mode_ == StepLaunchMode::EagerBucket ? "eager(bucket)"
                                                                        : "eager(exact)",
                     (!mtp_enabled_ || lanes_ > 1U)                   ? "n/a"
                     : mtp_chain_mode_ == StepLaunchMode::Graph        ? "graph"
                     : mtp_chain_mode_ == StepLaunchMode::EagerBucket ? "eager(bucket)"
                                                                     : "eager(exact)");
    }

    // The rendezvous id space is one channel per captured graph, and the bucket sets above are the
    // complete set of graphs that can ever be captured (single- and multi-lane verify windows, the
    // single-lane MTP draft chain, single- and multi-lane plain decode steps). Reserve them all: the
    // multi-lane
    // families alone are one graph per (bucket, lane count), far past the fixed 16 default, and a
    // bucket that cannot get a channel fails every later request that needs it.
    pair_.reserve_ar_channels(verify_graphs_.size() + verify_batch_graphs_.size() +
                              mtp_chain_graphs_.size() + decode_graphs_.size() +
                              decode_batch_graphs_.size());

    frontend_ = std::make_unique<qwen::Frontend>(
        qwen::make_frontend(shard_a_.model->resources(),
                            {.chat_template_path     = options.chat_template_path,
                             .architecture           = shard_a_.model->config().text.architecture,
                             .vision_enabled         = options.enable_vision,
                             .max_context            = options.max_context,
                             .media_cache_bytes      = options.media_cache_bytes,
                             .media_live_bytes       = options.media_live_bytes,
                             .media_preprocess_threads = options.media_preprocess_threads,
                             .vision_item_tokens       = options.vision_item_tokens}));
    load_seconds_ = std::chrono::duration<double>(Clock::now() - load_start).count();
    // P0.2: the multi-lane route runs its batches on a core-owned driver thread, never on the
    // submitter's. A submitter that drove its own batch would have to run it to the end before it
    // could return, which is exactly what withheld a finished response behind a slower peer.
    if (lanes_ > 1U) { lane_driver_ = std::thread([this] { drive_lane_queue(); }); }
}

TP2GenerationCore::~TP2GenerationCore() {
    stop_lane_driver();
    // The catalog's host KV slabs are suballocations of the host arenas, so the entries have to be
    // released before the arenas they were taken from.
    sessions_.clear();
}

void TP2GenerationCore::stop_lane_driver() {
    if (!lane_driver_.joinable()) { return; }
    {
        std::unique_lock<std::mutex> queue(lane_queue_mutex_);
        lane_driver_stop_ = true;
        // Whoever is still queued will never be picked up: this is the last moment anything can
        // publish a result for it, so fail it here instead of leaving its submitter waiting.
        for (auto& pending : lane_queue_) {
            pending->failure = std::make_exception_ptr(RequestError(
                RequestErrorKind::Unavailable, "TP-2 driver stopped before this request ran"));
            pending->complete = true;
        }
        lane_queue_.clear();
        lane_queue_cv_.notify_all();
    }
    lane_driver_.join();
}

void TP2GenerationCore::build_shard(Shard& shard, int shard_index) {
    const auto& config = shard.model->config().text;
    const std::uint32_t capacity = options_.max_context;
    // Multi-token prediction runs on shard 0 (see the class comment): only that shard plans and
    // materializes the MTP layer's own KV cache.
    const bool mtp_shard = mtp_enabled_ && shard_index == 0;
    // Resource provisioning width: how many concurrent decode lanes every shared resource is
    // planned for. `kTp2GenerationMaxConcurrency` (include/ninfer/tp2_capacity.h) pins this to
    // one today, so the layouts below stay byte-for-byte the single-request ones; raising that
    // gate additionally requires the batch execution path and per-lane KV/GDN publication, not
    // just this width. P1.2 of tp2-plan-concurrency.md.
    const std::int32_t lanes = static_cast<std::int32_t>(lanes_);

    // Per-shard config: the mixer head counts are halved (each shard owns half the attention/GDN
    // heads), so the KV cache and GDN state are sized for the per-shard geometry while the
    // replicated components (embeddings, norms, lm_head) keep the full-model config.
    auto shard_config = std::make_unique<models::qwen3_5::TextConfig>(config);
    if (shard_config->attention) {
        shard_config->attention->num_attention_heads /= 2;
        shard_config->attention->num_key_value_heads /= 2;
    }
    if (shard_config->gdn) {
        shard_config->gdn->linear_num_key_heads   /= 2;
        shard_config->gdn->linear_num_value_heads /= 2;
    }
    shard.shard_config = std::move(shard_config);
    const auto& scfg = *shard.shard_config;

    // Paged KV cache: full-attention layers only, one execution-table row per lane.
    // The MTP layer appends its own K/V up to `mtp_drafts_` positions past the committed frontier
    // (the speculative window), so its pool carries that many tokens of extra physical pages; its
    // execution row only maps the logical capacity.
    // Everything resident before the pool: the CUDA context and this shard's weight share.
    double resident_bytes = 0.0;
    const std::uint32_t mtp_physical_pages =
        mtp_shard
            ? pages_for_tokens(capacity) +
                  (mtp_drafts_ == 0 ? 0U : 1U + (mtp_drafts_ - 1U) / kPagedKVPageSize)
            : 0U;
    LayoutBuilder kv_builder;
    const qwen::DecoderStateLayout kv_layout = qwen::plan_decoder_state(
        kv_builder,
        qwen::DecoderStateSpec{
            .full_attention_layers = scfg.full_attention_layers,
            .mtp_layers            = 1,
            .capacity              = capacity,
            .kv_heads              = qwen::execution::dimension(scfg.attention->num_key_value_heads),
            // The MTP layer is replicated (not head-split), so its cache holds the full head count.
            .mtp_kv_heads =
                qwen::execution::dimension(config.attention->num_key_value_heads),
            .attention_head_dim    = qwen::execution::dimension(scfg.attention->head_dim),
            .kv_storage            = options_.kv_cache,
            .enable_mtp            = mtp_shard,
            .kv_table_rows         = lanes,
            .text_physical_page_groups = pages_for_tokens(capacity),
            .mtp_physical_page_groups  = mtp_physical_pages,
        });
    const std::size_t kv_bytes = kv_builder.finish(256);
    shard.device.bind_to_current_thread();
    {
        std::size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        resident_bytes = static_cast<double>(total_bytes - free_bytes) / 1048576.0;
    }
    shard.kv_arena = std::make_unique<DeviceArena>(kv_bytes);
    shard.decoder  = std::make_unique<qwen::DecoderState>(shard.kv_arena->alloc_bytes(kv_bytes, 256),
                                                          kv_layout);

    // GDN state pool: linear-attention layers, one slot per lane, zeroed. Sized for the per-shard
    // geometry (half the value heads and conv channels); each shard owns its local heads.
    const LinearAttentionStatePoolSpec gdn_spec{
        .layers         = scfg.linear_attention_layers,
        .conv_channels  = (scfg.gdn ? qwen::execution::dimension(scfg.gdn->conv_channels()) : 0),
        .conv_width     = (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_conv_kernel_dim - 1) : 0),
        .value_heads    = (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_num_value_heads) : 0),
        .value_head_dim = (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_value_head_dim) : 0),
        .key_head_dim   = (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_key_head_dim) : 0),
        .slot_count     = lanes,
        .conv_dtype     = DType::BF16,
    };
    LayoutBuilder state_builder;
    const LinearAttentionStatePoolLayout state_layout =
        plan_linear_attention_state_pool(state_builder, gdn_spec);
    const std::size_t state_bytes = state_builder.finish(256);
    shard.device.bind_to_current_thread();
    // Live linear-attention pool (two buffers), one prefix-reuse snapshot per slot, and the MTP
    // round scratch. `state_bytes` already covers `lanes` slots, so the per-lane cost is
    // `state_planes * (state_bytes / lanes)`. The host checkpoint ring serves the prompt end and every
    // rewind boundary when it exists, so those planes are not carved at all and slot 0 stays empty;
    // without the ring the store path needs the prompt-end plane and every plane is carved.
    //
    // The round scratch is read only by the speculative routes: execute_spec_batch restores the
    // pre-verify image out of it, and execute_walk's DFlash2 and MTP branches snapshot into it. The
    // plain batch executor never touches it, so a plain-route configuration does not carve it and
    // saves one whole-pool state image per lane (73.4 MiB x lanes per shard).
    const bool scratch_needed = mtp_enabled_ || dflash2_enabled_;
    const std::size_t first_state_plane = host_checkpoint_stride_ != 0 ? 1U : 0U;
    const std::size_t state_planes =
        (scratch_needed ? 1U : 0U) + shard.state_snapshots.size() - first_state_plane;
    shard.state_arena   = std::make_unique<DeviceArena>(state_planes * state_bytes);
    shard.state_backing = shard.state_arena->alloc_bytes(state_bytes, 256);
    for (std::size_t slot = first_state_plane; slot < shard.state_snapshots.size(); ++slot) {
        if (slot == kRoundScratchSlot && !scratch_needed) { continue; }
        shard.state_snapshots[slot] = shard.state_arena->alloc_bytes(state_bytes, 256);
    }
    CUDA_CHECK(cudaMemset(shard.state_backing.data, 0, state_bytes));
    shard.state = std::make_unique<LinearAttentionStatePool>(shard.state_backing, state_layout);
    shard.lane_state_geometry = make_lane_state_geometry(shard);
    if (host_checkpoint_stride_ != 0) {
        // Pageable, not pinned: the ring is the largest host-side allocation in the engine (tens of
        // one-lane state images per shard) and page-locked memory is a per-process budget the engine
        // also spends on the execute windows and the session slabs, while these images are read or
        // written only at the reuse boundaries a prompt crosses. The price is that a checkpoint
        // transfer stages through the driver and blocks its host thread instead of overlaying with
        // compute - a slower store and recall, not a correctness change: no CUDA Graph capture
        // region references the ring (see docs/serving.md). A slot holds one compact single-lane
        // image and belongs to the lane whose slice of the ring it is, so the host footprint is the
        // lanes=1 budget whatever the lane count (tp2-plan-concurrency.md 12.6).
        // One lane's slice per lane, plus the prompt-end slot appended after all of them: those images
        // are written once per completed prefill and never rotate, so they belong to no lane's slice
        // (see snapshot_host_checkpoint).
        const std::uint32_t slots = lanes_ * host_checkpoint_slots_per_lane_ + lanes_;

        shard.host_checkpoint_grid_slots = host_checkpoint_grid_slots_;
        shard.host_checkpoints.reserve(slots);
        for (std::uint32_t index = 0; index < slots; ++index) {
            Shard::HostCheckpoint checkpoint;
            checkpoint.buffer =
                std::make_unique<HostBuffer>(shard.lane_state_geometry.image_bytes,
                                             HostPinning::Pageable);
            shard.host_checkpoints.push_back(std::move(checkpoint));
        }
        std::fprintf(stderr,
                     "[mem] host checkpoint ring %u slots x %.1f MiB/lane/shard (%s)\n", slots,
                     static_cast<double>(shard.lane_state_geometry.image_bytes) / 1048576.0,
                     shard.host_checkpoints.front().buffer->pinned() ? "pinned" : "pageable");
    }

    std::size_t record_bytes        = 0;
    std::size_t round_bytes         = 0;
    std::size_t dflash_image_bytes  = 0;
    std::size_t dflash_planes       = 0;
    if (mtp_enabled_ || dflash2_enabled_) {
        // ReplaySSM records for one verify window wide, one physical row per lane, per-shard GDN
        // geometry.
        // Both shards verify the window, so both need their own records and fold plan.
        // The verify forward records here instead of advancing the live state, and the fold replays
        // the accepted prefix back into it (in place: with a single row the Op allows it, so the
        // state pool still needs one slot). MTP and DFlash2 use the same mechanism; only the window
        // width differs.
        LayoutBuilder record_builder;
        const GdnReplayRecordLayout record_layout = plan_gdn_replay_records(
            record_builder,
            GdnReplayRecordSpec{
                .layers          = qwen::execution::dimension(scfg.linear_attention_layers),
                .record_capacity = lanes,
                .width           = static_cast<std::int32_t>(mtp_enabled_ ? mtp_drafts_ : dflash_drafts_) + 1,
                .conv_channels =
                    (scfg.gdn ? qwen::execution::dimension(scfg.gdn->conv_channels()) : 0),
                .qk_heads =
                    (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_num_key_heads) : 0),
                .value_heads =
                    (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_num_value_heads) : 0),
                .key_dim =
                    (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_key_head_dim) : 0),
                .value_dim =
                    (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_value_head_dim) : 0),
            });
        record_bytes = record_builder.finish(256);
        shard.record_arena = std::make_unique<DeviceArena>(record_bytes);
        shard.device.bind_to_current_thread();
        const DeviceSpan record_backing = shard.record_arena->alloc_bytes(record_bytes, 256);
        // One bound record/fold pair per batch the route can form. The widest view sizes the backing;
        // every narrower one re-plans the same geometry with its own row count so that a layer's
        // record slice stays contiguous, which is what the record op requires.
        shard.replay_views.clear();
        shard.replay_views.reserve(static_cast<std::size_t>(lanes));
        for (std::int32_t batch = 1; batch <= lanes; ++batch) {
            LayoutBuilder batch_builder;
            GdnReplayRecordSpec batch_spec = record_layout.spec;
            batch_spec.record_capacity     = batch;
            const GdnReplayRecordLayout batch_layout =
                plan_gdn_replay_records(batch_builder, batch_spec);
            if (batch_builder.finish(256) > record_bytes) {
                throw std::logic_error(
                    "TP-2 replay record geometry does not scale down to a smaller batch");
            }
            Shard::ReplayViews views;
            views.records = GdnReplayRecords(record_backing, batch_layout);
            views.fold    = std::make_unique<ops::GdnReplayFoldPlan>(
                views.records, shard.state->all_layers_view());
            shard.replay_views.push_back(std::move(views));
        }
    }

    // Workspace arena: ample for the single-token forward (full-vocab logits plus a handful of
    // [N,1] activations) and the sampling workspace. Causal scoring keeps a [vocab, kScoreTile] BF16
    // per-column logits tile alive across the whole forward, on this shard and (allocated by the
    // forward) on the peer, so it takes that extra extent on top of the generation allowance.
    shard.device.bind_to_current_thread();
    workspace_bytes_ = kWorkspaceBytes;
    if (options_.purpose == EnginePurpose::CausalScoring) {
        workspace_bytes_ += kScoreWorkspaceTiles *
                            static_cast<std::size_t>(qwen::execution::dimension(config.vocab_size)) *
                            kScoreTile * sizeof(std::uint16_t);
    }
    shard.workspace = std::make_unique<DeviceArena>(workspace_bytes_);
    shard.prefill_hidden = shard.workspace->alloc(DType::BF16,
                                                  {2 * qwen::execution::dimension(config.hidden_size), lanes});
    if (mtp_shard) {
        // Survives every round scope: the first round's MTP bridge consumes the last prompt column's
        // final-norm hidden, captured during prefill priming.
        shard.mtp_anchor_hidden =
            shard.workspace->alloc(DType::BF16, {qwen::execution::dimension(config.hidden_size), lanes});
    }
    // Vision (--vision) runs on the shard that materialized the Vision component: the static split
    // that keeps the MTP layer on shard 0 puts the Vision tower and its encode/handoff workspace on
    // shard 1. The workspace is planned by the same planner the single-device route uses, with this
    // shard's text workspace capacity as the general extent so the item handoff region lands above
    // every text activation, and it is owned by a dedicated arena that outlives every request.
    std::size_t vision_bytes = 0;
    if (options_.enable_vision && shard_index == 1) {
        if (!shard.model->config().vision || !shard.parameters->vision) {
            throw std::invalid_argument("TP-2 vision shard has no Vision component parameters");
        }
        // The Vision workspace is the one resident block whose extent follows the largest image the
        // route admits, and it is what decides whether the tower fits beside the 262,144-token KV
        // pool. Its item ceiling is therefore selected by what this device actually accepts: the plan
        // is rebuilt at a halved ceiling until cudaMalloc succeeds. The driver's own free-memory
        // reading is not usable here (on this WSL2 host cudaMemGetInfo under-reports free bytes by
        // about a gigabyte), so a refused allocation is the only honest signal. The ceiling that
        // survives is reported below and bounds what a request may ask for.
        const auto& vision_config     = *shard.model->config().vision;
        const auto& vision_parameters = *shard.parameters->vision;
        std::uint32_t max_item        = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(capacity, options_.vision_item_tokens));
        for (;;) {
            vision_workspace_ = qwen::execution::VisionContext::plan_workspace(
                vision_config, vision_parameters, max_item, kWorkspaceBytes);
            shard.device.bind_to_current_thread();
            try {
                vision_arena_ = std::make_unique<DeviceArena>(vision_workspace_->capacity_bytes);
                break;
            } catch (const std::runtime_error&) {
                vision_arena_.reset();
                vision_workspace_.reset();
                // A refused cudaMalloc leaves cudaErrorMemoryAllocation latched in the runtime's
                // last-error slot, where the next kernel launch's check would report it as that
                // launch's own failure. Read it once here so the retry starts from a clean state.
                (void)cudaGetLastError();
                if (max_item <= kVisionItemTokenFloor) { throw; }
                max_item = std::max<std::uint32_t>(kVisionItemTokenFloor, max_item / 2);
            }
        }
        vision_bytes = vision_workspace_->capacity_bytes;
        std::fprintf(stderr,
                     "[mem] vision shard %d item ceiling %u | encode peak %.1f | handoff %.1f | "
                     "arena %.1f MiB\n",
                     shard_index, max_item,
                     static_cast<double>(vision_workspace_->encode_peak_bytes) / 1048576.0,
                     static_cast<double>(vision_workspace_->handoff_capacity_bytes) / 1048576.0,
                     static_cast<double>(vision_bytes) / 1048576.0);
    }
    // DFlash2 masked-draft round: the draft component is materialized whole on shard 0 (the split
    // spec keeps dflash2/* shard-local), so only that shard pays for the target-feature staging, the
    // draft's own local K/V and the round's proposal workspace. The round owns all of it
    // (program/dflash_round.h) so the production assembly is exactly the one the loader-only test
    // drives. One prefill chunk of features is enough because TP-2 clamps its chunk to at most
    // kPrefillChunkMaximum (1024) while the draft's local window is 2048, so no prefill ever stages
    // more than one window's worth of target residual.
    if (shard.parameters->draft.has_value()) {
        const auto& draft = *shard.model->config().draft;
        if (!draft.dflash2.has_value()) {
            // The single-device DFlash v1 draft materializes its context through the same state,
            // but through the per-layer path (and a full KV layer) that this core does not own.
            throw std::logic_error("TP-2 masked draft requires the DFlash2 context layout");
        }
        const std::uint32_t feature_columns = prefill_chunk_width(config);
        const std::uint32_t draft_window = options_.speculative.draft_tokens;
        // The proposal gets an arena of its own, sized by the planner the single-device route uses.
        // It is deliberately not carved out of the shard workspace: propose_dflash2_batch resets the
        // arena it is handed, and that workspace holds the resident prefill hidden state.
        const std::size_t proposal_bytes = qwen::execution::dflash2_proposal_workspace_bytes(
            *shard.parameters, config, capacity, ProposalHead::Full,
            static_cast<std::int32_t>(draft_window) + 1, static_cast<std::int32_t>(lanes_));
        const qwen::execution::DFlash2RoundSpec dflash_spec = qwen::execution::plan_dflash2_round(
            draft, config, feature_columns, draft_window, proposal_bytes, lanes_);
        shard.device.bind_to_current_thread();
        shard.dflash_round =
            std::make_unique<qwen::execution::DFlash2Round>(shard.device, dflash_spec);
        std::fprintf(stderr,
                     "[mem] shard %d DFlash2 round | targets %d x %d | ring %.1f | context %.1f | "
                     "frame %.1f | proposal %.1f MiB\n",
                     shard_index, dflash_spec.target_features, dflash_spec.feature_columns,
                     static_cast<double>(shard.dflash_round->ring_payload_bytes()) / 1048576.0,
                     static_cast<double>(shard.dflash_round->context_bytes()) / 1048576.0,
                     static_cast<double>(shard.dflash_round->frame_bytes()) / 1048576.0,
                     static_cast<double>(proposal_bytes) / 1048576.0);
        // The draft context at the reuse boundaries the GDN snapshots freeze. It is one ring per
        // boundary, allocated here because only the round knows the ring's image size; the state
        // arena itself holds no draft slot (see Shard::dflash_snapshots). Each boundary carries one
        // compact ring image per lane, so a snapshot is lanes_ images and a lane addresses its own
        // slice - the same shape as the target state image beside it. The host ring serves the same
        // boundaries as it does for the target state, so the planes it serves are not carved; at one
        // reuse boundary that leaves no draft plane at all.
        const std::size_t dflash_lane_image = shard.dflash_round->lane_context_image_bytes();
        const std::size_t dflash_image = static_cast<std::size_t>(lanes_) * dflash_lane_image;
        dflash_image_bytes             = dflash_lane_image;
        shard.device.bind_to_current_thread();
        const std::size_t first_dflash_plane = host_checkpoint_stride_ != 0 ? 1U : 0U;
        dflash_planes = shard.dflash_snapshots.size() - first_dflash_plane;
        if (dflash_planes != 0) {
            shard.dflash_snapshot_arena =
                std::make_unique<DeviceArena>(dflash_planes * dflash_image);
            for (std::size_t slot = first_dflash_plane; slot < shard.dflash_snapshots.size();
                 ++slot) {
                shard.dflash_snapshots[slot] =
                    shard.dflash_snapshot_arena->alloc_bytes(dflash_image, 256);
            }
            CUDA_CHECK(cudaMemset(shard.dflash_snapshot_arena->base(), 0,
                                  shard.dflash_snapshot_arena->capacity()));
        }
        // The checkpoint ring pairs each target state image with the draft ring at the same
        // frontier. A slot belongs to one lane, so its draft half is one compact image rather than
        // the whole multi-lane ring - the shape the target half already has. The ring is only
        // allocated once the round exists, which is why it is a second pass over a ring sized above.
        for (auto& checkpoint : shard.host_checkpoints) {
            checkpoint.dflash_buffer =
                std::make_unique<HostBuffer>(dflash_lane_image, HostPinning::Pageable);
        }
    }
    // One startup ledger line per shard: every resident block is allocated before the first
    // request, so this is the whole device budget at the requested context ceiling.
    {
        std::size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        // Weights the kernels read out of mapped host memory are not part of the device budget this
        // line enumerates, so they are reported beside it rather than folded into weights+ctx.
        const std::uint64_t host_mapped = shard.model->storage_stats().host_mapped_bytes;
        std::fprintf(stderr,
                     "[mem] shard %d capacity %u lanes %u | weights+ctx %.1f | host-mapped %.1f | "
                     "kv %.1f | "
                     "state %.1f | "
                     "record %.1f | draft-snap %.1f | round %.1f | workspace %u.0 | vision %.1f | "
                     "free %.1f of %.1f MiB\n",
                     shard_index, capacity, lanes_, resident_bytes,
                     static_cast<double>(host_mapped) / 1048576.0,
                     static_cast<double>(kv_bytes) / 1048576.0,
                     static_cast<double>(state_planes * state_bytes) / 1048576.0,
                     static_cast<double>(record_bytes) / 1048576.0,
                     static_cast<double>(dflash_planes * lanes_ * dflash_image_bytes) / 1048576.0,
                     static_cast<double>(round_bytes) / 1048576.0,
                     static_cast<unsigned>(workspace_bytes_ >> 20),
                     static_cast<double>(vision_bytes) / 1048576.0,
                     static_cast<double>(free_bytes) / 1048576.0,
                     static_cast<double>(total_bytes) / 1048576.0);
        if (!shard.host_checkpoints.empty()) {
            // The per-slot size is one compact single-lane target state image, not the whole pool:
            // each lane owns its own slice of the ring and a slot holds one lane's image. The masked
            // draft adds one compact ring image to each slot on the shard that owns it, which is
            // the second figure the line reports. The ring is pageable, so these bytes are host
            // commit rather than page-locked memory.
            std::fprintf(stderr,
                         "[mem] host-checkpoints shard %d slots %zu (grid %zu + tail %u + "
                         "divergence %u + block %u + prompt-end %u) x %.1f MiB | stride %u tok | "
                         "%s %.1f MiB\n",
                         shard_index, shard.host_checkpoints.size(),
                         shard.host_checkpoint_grid_slots, host_checkpoint_tail_slots_,
                         host_checkpoint_divergence_slots_, host_checkpoint_block_slots_, lanes_,
                         static_cast<double>(shard.lane_state_geometry.image_bytes) / 1048576.0,
                         host_checkpoint_stride_,
                         shard.host_checkpoints.front().buffer->pinned() ? "pinned" : "pageable",
                         static_cast<double>((shard.lane_state_geometry.image_bytes + dflash_image_bytes) *
                                             shard.host_checkpoints.size()) /
                             1048576.0);
        }
    }

    // The execution context is constructed at the end of this function: the MTP layer needs its KV
    // execution view, which only exists after the page list above is materialized.

    // Materialize the KV pages at startup. The pool is one fixed physical allocation; on the
    // single-lane route the whole of it is pinned here and reused in place across requests, so a
    // request only re-zeros the GDN state and the block-table mapping never changes.
    //
    // On a multi-lane route the pool is shared and handed out per request instead: this block only
    // acquires one execution row per lane, and reserve_lane_kv() republishes a lane's row with the
    // pages of whichever request currently holds that lane (S1 of tp2-plan-kv-sharing.md).
    // Every page is then owned by exactly one live request, so a lane can never reach another
    // request's pages, and the admitted window is bound to the pages that request reserved.
    {
        auto& pool   = shard.decoder->text_kv.page_pool();
        auto& tables = shard.decoder->text_kv.execution_tables();
        const std::uint32_t pages = pages_for_tokens(capacity);
        shard.device.bind_to_current_thread();
        shard.kv_lane_pages.resize(static_cast<std::size_t>(lanes));
        shard.kv_lane_handles.resize(static_cast<std::size_t>(lanes));
        shard.kv_rows.clear();
        shard.kv_rows.reserve(static_cast<std::size_t>(lanes));
        if (lanes_ == 1U) {
            auto& leases  = shard.kv_lane_pages[0];
            auto& handles = shard.kv_lane_handles[0];
            auto reserved = pool.reserve(pages);
            if (!reserved.has_value()) {
                throw std::logic_error("TP-2 KV page reservation failed");
            }
            DeviceKVPageReservation reservation = std::move(*reserved);
            leases.clear();
            leases.reserve(pages);
            pool.materialize(reservation, pages, leases);
            handles.clear();
            handles.reserve(leases.size());
            for (const auto& lease : leases) { handles.push_back(lease.handle()); }
            shard.kv_rows.push_back(tables.acquire(0));
            tables.publish(shard.kv_rows.back().handle(), 0, handles, shard.device.stream);
        } else {
            // Each row is published per request; the invariant the static split used to pin here (A3
            // of tp2-plan-concurrency-review-remediation.md) is checked in reserve_lane_kv()
            // against the pages that request actually reserved.
            for (std::int32_t lane = 0; lane < lanes; ++lane) {
                shard.kv_rows.push_back(tables.acquire(lane));
            }
            std::fprintf(stderr,
                         "[mem] shard %d KV pool %u pages (%u tokens) shared by %d lanes\n",
                         shard_index, pages,
                         pages * static_cast<std::uint32_t>(kPagedKVPageSize), lanes);
        }
    }

    // The MTP layer's own attention context. On the single-lane route it gets the same fixed-page
    // treatment as the text cache: one physical page list pinned for the process, one execution row,
    // published once (P2.1b). A multi-lane route shares it too and republishes the lane's row per
    // request exactly like the text cache (S1). Only the `logical` slice is ever mapped; the layer's
    // extra page groups are dead capacity exactly as on the single-GPU route.
    if (mtp_shard) {
        auto* cache = shard.decoder->mtp_cache();
        if (cache == nullptr) { throw std::logic_error("TP-2 MTP KV cache was not planned"); }
        auto& pool   = cache->page_pool();
        auto& tables = cache->execution_tables();
        const std::uint32_t logical_pages = pages_for_tokens(capacity);
        shard.device.bind_to_current_thread();
        shard.mtp_lane_pages.resize(static_cast<std::size_t>(lanes));
        shard.mtp_lane_handles.assign(lanes, {});
        shard.mtp_rows.clear();
        shard.mtp_rows.reserve(lanes);
        shard.mtp_views.clear();
        shard.mtp_views.reserve(lanes);
        if (lanes_ == 1U) {
            auto reserved = pool.reserve(mtp_physical_pages);
            if (!reserved.has_value()) {
                throw std::logic_error("TP-2 MTP KV page reservation failed");
            }
            DeviceKVPageReservation reservation = std::move(*reserved);
            shard.mtp_pages.reserve(mtp_physical_pages);
            pool.materialize(reservation, mtp_physical_pages, shard.mtp_pages);
            auto& handles = shard.mtp_lane_handles[0];
            handles.reserve(logical_pages);
            for (std::uint32_t i = 0; i < logical_pages; ++i) {
                handles.push_back(shard.mtp_pages[i].handle());
            }
            KVExecutionRowLease row = tables.acquire(0);
            tables.publish(row.handle(), 0, handles, shard.device.stream);
            shard.mtp_views.push_back(cache->execution_view(row));
            shard.mtp_rows.push_back(std::move(row));
        } else {
            for (std::int32_t lane = 0; lane < lanes; ++lane) {
                KVExecutionRowLease row = tables.acquire(lane);
                shard.mtp_views.push_back(cache->execution_view(row));
                shard.mtp_rows.push_back(std::move(row));
            }
            std::fprintf(stderr, "[mem] shard %d MTP KV pool %u pages shared by %u lanes\n",
                         shard_index, logical_pages, lanes);
        }
    }

    // MTP proposal scratch (shard 0): the round state carries the step token/position, the RoPE
    // delta, the MTP KV table row and the MTP prefill/autoregressive buffers for K drafts.
    qwen::PagedKVCacheView mtp_view;
    const qwen::PagedKVCache* batch_mtp = nullptr;
    if (mtp_shard) {
        LayoutBuilder round_builder;
        qwen::RoundStateLayout round_layout = qwen::begin_round_state_layout(
            round_builder,
            qwen::RoundStateSpec{
                .hidden         = qwen::execution::dimension(config.hidden_size),
                .output_rows    = qwen::execution::dimension(config.vocab_size),
                .batch_capacity = static_cast<std::uint32_t>(lanes),
                .draft_window   = mtp_drafts_,
                .backend        = SpeculativeBackend::Mtp,
            });
        qwen::complete_round_state_layout(round_builder, round_layout);
        round_bytes = round_builder.finish(256);
        shard.round_arena = std::make_unique<DeviceArena>(round_bytes);
        shard.device.bind_to_current_thread();
        DeviceSpan backing = shard.round_arena->alloc_bytes(round_bytes, 256);
        CUDA_CHECK(cudaMemset(backing.data, 0, round_bytes));
        shard.io = qwen::RoundState(backing, round_layout);
        ops::set_i32_scalar(shard.io.backend_kv_table_row, 0, shard.device.stream);
        // Lane 0's view is only the constructor's default; priming re-points it per lane (P2.1b).
        mtp_view  = shard.mtp_views.empty() ? qwen::PagedKVCacheView() : shard.mtp_views.front();
        batch_mtp = shard.decoder->mtp_cache();
    }

    {
        // The text KV is bound through the batch cache (the TP-2 path never uses a per-request
        // execution view); the MTP layer binds both its execution view (prefill chunk) and its
        // batch cache (proposal steps).
        qwen::PagedKVCacheView empty_kv;
        const qwen::PagedKVCache* batch_text = &shard.decoder->text_kv;
        shard.context = std::make_unique<qwen::execution::TextContext>(
            shard.device, *shard.parameters, *shard.workspace, empty_kv, *shard.state, shard.io,
            shard.prefill_hidden, options_.prefill_chunk, 0, mtp_view, batch_text, batch_mtp,
            options_.prefill_overlap);
        shard.context->set_shard_config(&scfg, shard_index);
        if (mtp_shard) { shard.context->set_mtp_proposal_extent(mtp_drafts_); }
    }
}

void TP2GenerationCore::mtp_prefill_priming(Shard& shard, const int* ids, std::uint32_t length,
                                            std::uint32_t first_position, Tensor& mtp_input,
                                            const Tensor* last_token, bool final_chunk,
                                            std::int32_t lane) {
    const auto& config = shard.model->config().text;
    const std::int32_t hidden = qwen::execution::dimension(config.hidden_size);
    const std::int32_t vocab  = qwen::execution::dimension(config.vocab_size);
    auto& ws = *shard.workspace;
    shard.device.bind_to_current_thread();
    cudaStream_t stream = shard.device.stream;
    // The MTP layer owns one execution row per lane (P2.1b). Priming is serial across lanes, so it
    // re-points both the prefill view's own K/V appends and the round state's scalar row - which the
    // non-batch attention reads - at this lane before the chunk runs.
    if (lane < 0 || static_cast<std::size_t>(lane) >= shard.mtp_views.size()) {
        throw std::logic_error("TP-2 MTP priming lane is outside the lane rows");
    }
    shard.context->set_mtp_prefill_view(shard.mtp_views[static_cast<std::size_t>(lane)]);
    ops::set_i32_scalar(shard.io.backend_kv_table_row, lane, stream);
    // The MTP layer's embedding column i is the token at position i+1: its input is shifted by one
    // relative to the hidden columns. Only the final chunk's last column is unknown here (the token
    // that follows the prompt), so it is overwritten on the device with the sampled token.
    std::vector<int> shifted(length, 0);
    const std::uint32_t known_columns = last_token != nullptr ? length - 1 : length;
    for (std::uint32_t i = 0; i < known_columns; ++i) { shifted[i] = ids[i + 1]; }
    Tensor ids_t = ws.alloc(DType::I32, {static_cast<std::int32_t>(length)});
    CUDA_CHECK(cudaMemcpyAsync(ids_t.data, shifted.data(), sizeof(int) * length,
                               cudaMemcpyHostToDevice, stream));
    if (last_token != nullptr) {
        // Element offset, not a byte offset: Tensor::data is void*, so plain pointer arithmetic
        // would land on the wrong column and corrupt two adjacent token ids.
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(ids_t.data) + (length - 1),
                                   last_token->data, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    Tensor positions = ws.alloc(DType::I32, {static_cast<std::int32_t>(length)});
    ops::fill_i32_positions(positions, static_cast<std::int32_t>(first_position), 1, stream);
    const std::uint32_t visible = first_position + length;
    const ops::CausalAttentionExecutionEnvelope envelope{visible, visible};
    // The MTP layer's own K/V appends at the same absolute positions as the text layers, so the
    // cache positions and the RoPE positions coincide (the TP-2 path runs without a RoPE delta).
    // No proposal here: the round loop's first bridge re-runs this last column and proposes the
    // window itself, so priming only has to leave the MTP K/V complete.
    shard.context->mtp_prefill_chunk(ids_t, mtp_input, nullptr, positions, positions, envelope,
                                     false, nullptr, nullptr, nullptr);
    if (final_chunk) {
        // This chunk's last column is the prompt's last position: its hidden feeds the first bridge.
        // One hidden row - this lane's own column of the [hidden, lanes] anchor tensor - not the whole
        // tensor, whose remaining columns belong to other lanes (P2.1b).
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<std::uint8_t*>(shard.mtp_anchor_hidden.data) +
                static_cast<std::size_t>(lane) * static_cast<std::size_t>(hidden) *
                    sizeof(std::uint16_t),
            mtp_input.slice(1, static_cast<std::int32_t>(length) - 1, 1).data,
            static_cast<std::size_t>(hidden) * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice,
            stream));
    }
}

TP2GenerationCore::WindowGraph* TP2GenerationCore::select_window_graph(
    std::vector<WindowGraph>& graphs, std::uint32_t visible_end, std::int32_t batch,
    std::int32_t lane) {
    for (WindowGraph& graph : graphs) {
        if (graph.batch != batch) { continue; }
        // A capture that binds every slot from device memory carries no lane identity, so it
        // matches any assignment of its width; a single-column capture baked one slot.
        if (graph.lane != -1 && graph.lane != lane) { continue; }
        if (graph.visible_begin <= visible_end && visible_end <= graph.visible_end) { return &graph; }
    }
    return nullptr;
}

TP2GenerationCore::WindowGraph* TP2GenerationCore::reusable_window_graph(
    std::vector<WindowGraph>& graphs, std::uint32_t visible_end, std::int32_t batch,
    std::int32_t lane) {
    WindowGraph* graph = select_window_graph(graphs, visible_end, batch, lane);
    if (graph == nullptr || !graph->captured || graph->round_base[0] < shard_a_.round_base ||
        graph->round_base[1] < shard_b_.round_base) {
        return nullptr;
    }
    return graph;
}

// Launches one captured window on both devices. Nothing here may synchronize the host: a replay is
// enqueued back to back with the round's other work, and the two devices order themselves through
// the in-kernel allreduce's arrival tokens.
void TP2GenerationCore::launch_window_graph(WindowGraph& graph) {
    // Publish this launch's rendezvous id block before the replay: the graph's memcpy node reads the
    // pinned cell, so every replay runs on ids that were never used before and no arrival slot left
    // over from an earlier round can satisfy a spin.
    pair_.arm_round(graph.ar_channel);
    shard_a_.device.bind_to_current_thread();
    graph.executable[0].launch(shard_a_.device.stream);
    shard_b_.device.bind_to_current_thread();
    graph.executable[1].launch(shard_b_.device.stream);
    shard_a_.device.bind_to_current_thread();
}

// Instantiates and uploads both replicas of a freshly captured window. See the header for why the
// upload cannot be left to the first launch.
void TP2GenerationCore::install_window_graph(WindowGraph& graph) {
    shard_a_.device.bind_to_current_thread();
    graph.executable[0].instantiate(graph.definition[0]);
    shard_b_.device.bind_to_current_thread();
    graph.executable[1].instantiate(graph.definition[1]);
    shard_a_.device.bind_to_current_thread();
    graph.executable[0].upload(shard_a_.device.stream);
    shard_b_.device.bind_to_current_thread();
    graph.executable[1].upload(shard_b_.device.stream);
    // The uploads are asynchronous on their own streams, so drain them here, while the devices are
    // still idle: no later launch may be the one that waits for an upload.
    shard_a_.device.bind_to_current_thread();
    shard_a_.device.synchronize();
    shard_b_.device.bind_to_current_thread();
    shard_b_.device.synchronize();
    shard_a_.device.bind_to_current_thread();
    graph.captured = true;
}

void TP2GenerationCore::capture_verify_graph(WindowGraph& graph, const std::int32_t* ids,
                                             const std::int32_t* positions, Tensor& logits_columns,
                                             Tensor& hidden_columns,
                                             qwen::execution::DFlashFeatureSink* sink,
                                             const std::int32_t* valid_columns) {
    Shard& shard_a = shard_a_;
    Shard& shard_b = shard_b_;
    const ops::CausalAttentionExecutionEnvelope envelope{graph.visible_begin, graph.visible_end};
    shard_a.device.bind_to_current_thread();
    graph.round_base[0]  = shard_a.round_base;
    graph.round_base[1]  = shard_b.round_base;
    graph.arena_begin[0] = shard_a.workspace->used();
    graph.arena_begin[1] = shard_b.workspace->used();
    // The verify's state transitions are recorded, not applied: the fold replays the committed
    // columns from the pre-round snapshot. The action is a launch-time choice, so the capture must
    // run with it set and a replay picks it up from the recorded kernels.
    shard_a.context->set_gdn_state_action(qwen::execution::GdnStateAction::RecordForReplay,
                                          &shard_a.records_for(1));
    shard_b.context->set_gdn_state_action(qwen::execution::GdnStateAction::RecordForReplay,
                                          &shard_b.records_for(1));
    if (graph.ar_channel == tp::DevicePair::kNoArChannel) {
        graph.ar_channel = pair_.create_ar_channel();
    }
    DecodeGraphDefinition* definitions[2] = {&graph.definition[0], &graph.definition[1]};
    cudaStream_t streams[2] = {shard_a.device.stream, shard_b.device.stream};
    pair_.begin_capture(graph.ar_channel);
    const ArCaptureGuard capture_guard{pair_};
    DecodeGraphDefinition::capture_group(definitions, streams, [&] {
        // The masked-draft sink and the clamp extent are part of the captured window: without the
        // forwarding here the capture silently takes the default nullptr/nullptr and every replay
        // runs the unmasked, sink-less sequence (the clamp KV-append race is back).
        shard_a.context->forward_tp2_window(*shard_b.context, pair_, ids, positions, envelope,
                                           logits_columns, &hidden_columns, sink, valid_columns);
    });
    pair_.end_capture();
    graph.arena_bytes[0] = shard_a.workspace->used() - graph.arena_begin[0];
    graph.arena_bytes[1] = shard_b.workspace->used() - graph.arena_begin[1];
    install_window_graph(graph);
}

void TP2GenerationCore::capture_verify_batch_graph(
    WindowGraph& graph, const std::int32_t* ids, const std::int32_t* positions,
    const std::int32_t* kv_table_rows, const std::int32_t* state_slots, std::int32_t width,
    std::int32_t batch, Tensor& logits_columns, Tensor& hidden_columns,
    qwen::execution::DFlashFeatureSink* sink, const std::int32_t* valid_columns) {
    Shard& shard_a = shard_a_;
    Shard& shard_b = shard_b_;
    const ops::CausalAttentionExecutionEnvelope envelope{graph.visible_begin, graph.visible_end};
    shard_a.device.bind_to_current_thread();
    graph.round_base[0]  = shard_a.round_base;
    graph.round_base[1]  = shard_b.round_base;
    graph.arena_begin[0] = shard_a.workspace->used();
    graph.arena_begin[1] = shard_b.workspace->used();
    // Same recorded-transition contract as the single-lane capture, with the record view of this
    // lane count: the fold replays the committed columns of every active lane from the pre-round
    // snapshot, and the action is a launch-time choice the capture bakes like any other argument.
    shard_a.context->set_gdn_state_action(qwen::execution::GdnStateAction::RecordForReplay,
                                          &shard_a.records_for(batch));
    shard_b.context->set_gdn_state_action(qwen::execution::GdnStateAction::RecordForReplay,
                                          &shard_b.records_for(batch));
    DecodeGraphDefinition* definitions[2] = {&graph.definition[0], &graph.definition[1]};
    cudaStream_t streams[2] = {shard_a.device.stream, shard_b.device.stream};
    if (graph.ar_channel == tp::DevicePair::kNoArChannel) {
        graph.ar_channel = pair_.create_ar_channel();
    }
    pair_.begin_capture(graph.ar_channel);
    const ArCaptureGuard capture_guard{pair_};
    DecodeGraphDefinition::capture_group(definitions, streams, [&] {
        // The per-lane bindings reach the captured sequence through memcpy nodes from the caller's
        // pinned sections, exactly as ids/positions do, so the window carries no lane identity of
        // its own beyond the one a one-column capture bakes.
        shard_a.context->forward_tp2_window_batch(*shard_b.context, pair_, ids, positions,
                                                  kv_table_rows, state_slots, width, batch, envelope,
                                                  logits_columns, &hidden_columns, sink,
                                                  valid_columns);
    });
    pair_.end_capture();
    graph.arena_bytes[0] = shard_a.workspace->used() - graph.arena_begin[0];
    graph.arena_bytes[1] = shard_b.workspace->used() - graph.arena_begin[1];
    install_window_graph(graph);
}

void TP2GenerationCore::run_verify_window(const std::int32_t* ids, std::int32_t first_position,
                                          Tensor& logits_columns, Tensor& hidden_columns,
                                          qwen::execution::DFlashFeatureSink* sink,
                                          std::int32_t valid_columns) {
    Shard& shard_a = shard_a_;
    Shard& shard_b = shard_b_;
    const auto width    = static_cast<std::size_t>(logits_columns.ne[1]);
    const auto* positions = ids + width;
    const std::uint32_t visible_end =
        static_cast<std::uint32_t>(first_position) + static_cast<std::uint32_t>(width);
    // The clamp extent varies per round, so it reaches the forward (and a captured graph) through
    // the pinned buffer's trailing int exactly like ids/positions: a set_i32_scalar would bake the
    // capture-time value into the kernel and every replay would clamp to it.
    auto* host_window             = static_cast<std::int32_t*>(verify_window_host_->data());
    const std::int32_t* valid_ptr = nullptr;
    if (valid_columns > 0) {
        host_window[2 * width] = valid_columns;
        valid_ptr              = host_window + 2 * width;
    }
    if (!verify_graph_enabled_) {
        // The eager route runs the same window and the same envelope; only the launch differs. That
        // keeps NINFER_TP2_VERIFY_GRAPH=0 a true A/B of the captured sequence rather than a
        // comparison of two different attention routes. It is also the batch entry with a single
        // lane, so the one-lane eager route and the batched one cannot drift apart.
        run_verify_window_batch(ids, positions, nullptr, nullptr, static_cast<std::int32_t>(width), 1,
                                first_position, logits_columns, hidden_columns, sink, valid_ptr);
    } else {
        WindowGraph* graph = select_window_graph(verify_graphs_, visible_end);
        if (graph == nullptr) {
            throw std::logic_error("TP-2 verify CUDA Graph coverage is incomplete");
        }
        if (reusable_window_graph(verify_graphs_, visible_end) == nullptr) {
            // Capturing records the sequence without executing it, so the capture round still has to
            // run the window it just captured (with the operands the capture itself allocated). The
            // masked-draft feature sink travels with the capture: its scatter addresses are baked
            // and its per-round lane/column tensors are re-read by the kernels on every replay.
            capture_verify_graph(*graph, ids, positions, logits_columns, hidden_columns, sink,
                                 valid_ptr);
            launch_window_graph(*graph);
        } else {
            if (shard_a.workspace->used() != graph->arena_begin[0] ||
                shard_b.workspace->used() != graph->arena_begin[1]) {
                throw std::logic_error(
                    "TP-2 verify CUDA Graph replay found a different workspace layout");
            }
            launch_window_graph(*graph);
            // The captured body allocated its operands during capture; a replay does not run that
            // host code, so advance both arenas by what the capture consumed. A side the body only
            // touched through scope-unwound scratch nets zero, and zero is a no-op here exactly as
            // in position_arena.
            if (graph->arena_bytes[0] > 0) { (void)shard_a.workspace->alloc_bytes(graph->arena_bytes[0]); }
            if (graph->arena_bytes[1] > 0) { (void)shard_b.workspace->alloc_bytes(graph->arena_bytes[1]); }
        }
    }
    shard_a.context->set_gdn_state_action(qwen::execution::GdnStateAction::UpdateInPlace, nullptr);
    shard_b.context->set_gdn_state_action(qwen::execution::GdnStateAction::UpdateInPlace, nullptr);
}

void TP2GenerationCore::run_verify_window_batch(const std::int32_t* ids,
                                                const std::int32_t* positions,
                                                const std::int32_t* kv_table_rows,
                                                const std::int32_t* state_slots, std::int32_t width,
                                                std::int32_t batch, std::int32_t first_position,
                                                Tensor& logits_columns, Tensor& hidden_columns,
                                                qwen::execution::DFlashFeatureSink* sink,
                                                const std::int32_t* valid_columns) {
    const bool per_lane = kv_table_rows != nullptr || state_slots != nullptr;
    if (width < 1 || batch < 1) {
        throw std::logic_error("TP-2 verify window needs a positive width and lane count");
    }
    if (batch > 1 && !per_lane) {
        throw std::logic_error("the batched TP-2 verify window needs per-lane bindings");
    }
    const std::uint32_t visible_end =
        static_cast<std::uint32_t>(first_position) + static_cast<std::uint32_t>(width);
    // The bucket envelope and the captured sequence come from the family this call owns: the batch
    // captures when the caller binds every lane from device memory on the multi-lane route, the
    // single-lane family everywhere else (which is also what the one-lane eager arm below reads its
    // bucket from). Both families are built from the same profiles, so the envelope is the one the
    // single-lane route always ran and the eager arms keep comparing the launch mechanism alone.
    const bool batch_family = per_lane && !verify_batch_graphs_.empty();
    std::vector<WindowGraph>& graphs = batch_family ? verify_batch_graphs_ : verify_graphs_;
    const StepLaunchMode mode = batch_family ? verify_batch_mode_ : StepLaunchMode::EagerExact;
    // The envelope comes from the single-lane family, which is built for every route that owns a
    // verify window and carries every bucket unsliced; the batch family below only keys the launch.
    const WindowGraph* bucket = select_window_graph(verify_graphs_, visible_end);
    if (bucket == nullptr) {
        throw std::logic_error("TP-2 verify window coverage is incomplete");
    }
    const ops::CausalAttentionExecutionEnvelope envelope{bucket->visible_begin, bucket->visible_end};
    // The record view follows the lane count, so a round driving batch lanes records into batch
    // physical rows and the fold reads the same geometry. A captured window carries the record
    // kernels and never re-runs this host code, so the action is cleared after every arm.
    const auto set_record = [&] {
        for (Shard* shard : {&shard_a_, &shard_b_}) {
            shard->context->set_gdn_state_action(qwen::execution::GdnStateAction::RecordForReplay,
                                                 &shard->records_for(batch));
        }
    };
    const auto reset_record = [&] {
        for (Shard* shard : {&shard_a_, &shard_b_}) {
            shard->context->set_gdn_state_action(qwen::execution::GdnStateAction::UpdateInPlace,
                                                 nullptr);
        }
    };
    if (mode == StepLaunchMode::EagerExact) {
        set_record();
        shard_a_.context->forward_tp2_window_batch(*shard_b_.context, pair_, ids, positions,
                                                   kv_table_rows, state_slots, width, batch, envelope,
                                                   logits_columns, &hidden_columns, sink,
                                                   valid_columns);
        reset_record();
        return;
    }
    // A one-column batch still runs the scalar GDN path, whose state slot is a host value the
    // capture bakes; a wider batch binds every slot from device memory and carries no lane identity.
    const std::int32_t lane = batch == 1 ? (state_slots != nullptr ? state_slots[0] : 0) : -1;
    WindowGraph* graph = select_window_graph(graphs, visible_end, batch, lane);
    if (graph == nullptr) {
        throw std::logic_error("TP-2 batched verify window coverage is incomplete");
    }
    const bool captured = mode == StepLaunchMode::Graph;
    WindowGraph* reusable =
        captured ? reusable_window_graph(graphs, visible_end, batch, lane) : nullptr;
    if (reusable == nullptr) {
        if (captured) {
            capture_verify_batch_graph(*graph, ids, positions, kv_table_rows, state_slots, width,
                                       batch, logits_columns, hidden_columns, sink, valid_columns);
            launch_window_graph(*graph);
        } else {
            // The eager arm runs the same bucket envelope, so the A/B compares the launch mechanism
            // alone.
            set_record();
            shard_a_.context->forward_tp2_window_batch(*shard_b_.context, pair_, ids, positions,
                                                       kv_table_rows, state_slots, width, batch,
                                                       envelope, logits_columns, &hidden_columns,
                                                       sink, valid_columns);
        }
        reset_record();
        return;
    }
    if (shard_a_.workspace->used() != graph->arena_begin[0] ||
        shard_b_.workspace->used() != graph->arena_begin[1]) {
        throw std::logic_error(
            "TP-2 batched verify CUDA Graph replay found a different workspace layout");
    }
    launch_window_graph(*graph);
    if (graph->arena_bytes[0] > 0) { (void)shard_a_.workspace->alloc_bytes(graph->arena_bytes[0]); }
    if (graph->arena_bytes[1] > 0) { (void)shard_b_.workspace->alloc_bytes(graph->arena_bytes[1]); }
    reset_record();
}

void TP2GenerationCore::fold_verify_window(const std::int32_t* state_slots,
                                           const std::int32_t* commit_columns, std::int32_t batch) {
    if (batch < 1 || state_slots == nullptr || commit_columns == nullptr) {
        throw std::logic_error("the TP-2 verify fold needs one row per active lane");
    }
    std::vector<ops::GdnReplayFoldRow> rows(static_cast<std::size_t>(batch));
    for (std::int32_t lane = 0; lane < batch; ++lane) {
        const std::int32_t slot = state_slots[lane];
        rows[static_cast<std::size_t>(lane)] = ops::GdnReplayFoldRow{
            .source_state_slot      = slot,
            .destination_state_slot = slot,
            .commit_columns         = commit_columns[lane]};
    }
    const std::span<const ops::GdnReplayFoldRow> fold_rows(rows.data(), rows.size());
    for (Shard* shard : {&shard_a_, &shard_b_}) {
        shard->device.bind_to_current_thread();
        shard->fold_for(batch).execute(fold_rows, shard->device.stream);
    }
}

void TP2GenerationCore::capture_decode_graph(WindowGraph& graph, const std::int32_t* token,
                                             const std::int32_t* position, Tensor& logits) {
    Shard& shard_a = shard_a_;
    Shard& shard_b = shard_b_;
    const ops::CausalAttentionExecutionEnvelope envelope{graph.visible_begin, graph.visible_end};
    shard_a.device.bind_to_current_thread();
    graph.round_base[0]  = shard_a.round_base;
    graph.round_base[1]  = shard_b.round_base;
    graph.arena_begin[0] = shard_a.workspace->used();
    graph.arena_begin[1] = shard_b.workspace->used();
    // The step's state transitions are already applied in place: a plain decode is not speculative,
    // so unlike the verify window it neither records nor replays anything.
    DecodeGraphDefinition* definitions[2] = {&graph.definition[0], &graph.definition[1]};
    cudaStream_t streams[2] = {shard_a.device.stream, shard_b.device.stream};
    if (graph.ar_channel == tp::DevicePair::kNoArChannel) {
        graph.ar_channel = pair_.create_ar_channel();
    }
    pair_.begin_capture(graph.ar_channel);
    const ArCaptureGuard capture_guard{pair_};
    DecodeGraphDefinition::capture_group(definitions, streams, [&] {
        shard_a.context->forward_tp2_decode_window(*shard_b.context, pair_, token, position,
                                                  envelope, logits, active_lane_);
    });
    pair_.end_capture();
    graph.arena_bytes[0] = shard_a.workspace->used() - graph.arena_begin[0];
    graph.arena_bytes[1] = shard_b.workspace->used() - graph.arena_begin[1];
    install_window_graph(graph);
}

Tensor TP2GenerationCore::run_plain_decode_step(std::int32_t token, std::uint32_t position) {
    Shard& shard_a = shard_a_;
    Shard& shard_b = shard_b_;
    const std::int32_t vocab = qwen::execution::dimension(shard_a.model->config().text.vocab_size);
    if (decode_step_mode_ == StepLaunchMode::EagerExact) {
        // The pre-graph route: the exact visible extent, with the token and the position baked into
        // the forward's own scalar kernels. Kept as the reference arm of the A/B.
        Tensor logits_a = shard_a.workspace->alloc(DType::BF16, {vocab, 1});
        Tensor logits_b = shard_b.workspace->alloc(DType::BF16, {vocab, 1});
        shard_a.context->forward_tp2(*shard_b.context, pair_, token,
                                     static_cast<std::int32_t>(position), logits_a, logits_b);
        return logits_a;
    }
    const std::uint32_t visible_end = position + 1U;
    WindowGraph* graph = select_window_graph(decode_graphs_, visible_end);
    if (graph == nullptr) { throw std::logic_error("TP-2 decode step coverage is incomplete"); }
    const bool captured = decode_step_mode_ == StepLaunchMode::Graph;
    WindowGraph* reusable =
        captured ? reusable_window_graph(decode_graphs_, visible_end) : nullptr;
    // A replay must reproduce the addresses the capture baked into its kernel arguments, so both
    // arenas first go back to the captured watermark: the capture ran its host-side allocations, a
    // replay does not.
    position_arena(*shard_a.workspace, shard_a.round_base,
                   reusable != nullptr ? reusable->round_base[0] : shard_a.round_base);
    position_arena(*shard_b.workspace, shard_b.round_base,
                   reusable != nullptr ? reusable->round_base[1] : shard_b.round_base);
    Tensor logits = shard_a.workspace->alloc(DType::BF16, {vocab, 1});
    auto* window  = static_cast<std::int32_t*>(decode_window_host_->data());
    window[0]     = token;
    window[1]     = static_cast<std::int32_t>(position);
    const ops::CausalAttentionExecutionEnvelope envelope{graph->visible_begin, graph->visible_end};
    if (reusable == nullptr) {
        if (captured) {
            // Capturing records the sequence without executing it, so this round still has to run the
            // step it just captured, with the operands the capture itself allocated.
            capture_decode_graph(*graph, window, window + 1, logits);
            launch_window_graph(*graph);
        } else {
            // The eager arm runs the same sequence with the same bucket envelope, so the A/B compares
            // the launch mechanism alone.
            shard_a.context->forward_tp2_decode_window(*shard_b.context, pair_, window, window + 1,
                                                       envelope, logits, active_lane_);
        }
        return logits;
    }
    if (shard_a.workspace->used() != graph->arena_begin[0] ||
        shard_b.workspace->used() != graph->arena_begin[1]) {
        throw std::logic_error("TP-2 decode CUDA Graph replay found a different workspace layout");
    }
    launch_window_graph(*graph);
    // The captured body allocated its operands during capture; a replay does not run that host code,
    // so advance both arenas by what the capture consumed.
    if (graph->arena_bytes[0] > 0) { (void)shard_a.workspace->alloc_bytes(graph->arena_bytes[0]); }
    if (graph->arena_bytes[1] > 0) { (void)shard_b.workspace->alloc_bytes(graph->arena_bytes[1]); }
    return logits;
}

void TP2GenerationCore::capture_decode_batch_graph(WindowGraph& graph, const std::int32_t* tokens,
                                                   const std::int32_t* positions,
                                                   const std::int32_t* kv_table_rows,
                                                   const std::int32_t* state_slots,
                                                   std::int32_t columns, Tensor& logits) {
    Shard& shard_a = shard_a_;
    Shard& shard_b = shard_b_;
    const ops::CausalAttentionExecutionEnvelope envelope{graph.visible_begin, graph.visible_end};
    shard_a.device.bind_to_current_thread();
    graph.round_base[0]  = shard_a.round_base;
    graph.round_base[1]  = shard_b.round_base;
    graph.arena_begin[0] = shard_a.workspace->used();
    graph.arena_begin[1] = shard_b.workspace->used();
    // A plain decode advances its state in place, exactly like the single-lane step: nothing here
    // records or replays a GDN transition.
    DecodeGraphDefinition* definitions[2] = {&graph.definition[0], &graph.definition[1]};
    cudaStream_t streams[2] = {shard_a.device.stream, shard_b.device.stream};
    if (graph.ar_channel == tp::DevicePair::kNoArChannel) {
        graph.ar_channel = pair_.create_ar_channel();
    }
    pair_.begin_capture(graph.ar_channel);
    const ArCaptureGuard capture_guard{pair_};
    DecodeGraphDefinition::capture_group(definitions, streams, [&] {
        shard_a.context->forward_tp2_decode_window_batch(*shard_b.context, pair_, tokens, positions,
                                                        kv_table_rows, state_slots, columns, envelope,
                                                        logits);
    });
    pair_.end_capture();
    graph.arena_bytes[0] = shard_a.workspace->used() - graph.arena_begin[0];
    graph.arena_bytes[1] = shard_b.workspace->used() - graph.arena_begin[1];
    install_window_graph(graph);
}

void TP2GenerationCore::run_plain_decode_step_batch(
    const std::int32_t* tokens, const std::int32_t* positions, const std::int32_t* kv_table_rows,
    const std::int32_t* state_slots, std::int32_t columns,
    const ops::CausalAttentionExecutionEnvelope& exact_envelope, Tensor& logits) {
    Shard& shard_a = shard_a_;
    Shard& shard_b = shard_b_;
    if (decode_batch_mode_ == StepLaunchMode::EagerExact) {
        // The pre-graph route: the exact visible extent, with the per-lane operands read straight
        // from the pinned arrays. Kept as the reference arm of the A/B.
        shard_a.context->forward_tp2_decode_window_batch(*shard_b.context, pair_, tokens, positions,
                                                        kv_table_rows, state_slots, columns,
                                                        exact_envelope, logits);
        return;
    }
    const std::uint32_t visible_end = exact_envelope.max_visible_keys;
    // A one-column batch still runs the scalar GDN path, which reads the slot published by
    // set_linear_state_slots, so its captured sequence bakes that slot and needs one graph per lane.
    const std::int32_t lane = columns == 1 ? state_slots[0] : -1;
    WindowGraph* graph =
        select_window_graph(decode_batch_graphs_, visible_end, columns, lane);
    if (graph == nullptr) {
        throw std::logic_error("TP-2 batched decode step coverage is incomplete");
    }
    const bool captured = decode_batch_mode_ == StepLaunchMode::Graph;
    WindowGraph* reusable =
        captured ? reusable_window_graph(decode_batch_graphs_, visible_end, columns, lane) : nullptr;
    const ops::CausalAttentionExecutionEnvelope envelope{graph->visible_begin, graph->visible_end};
    if (reusable == nullptr) {
        if (captured) {
            capture_decode_batch_graph(*graph, tokens, positions, kv_table_rows, state_slots, columns,
                                       logits);
            launch_window_graph(*graph);
        } else {
            // The eager arm runs the same sequence with the same bucket envelope, so the A/B
            // compares the launch mechanism alone.
            shard_a.context->forward_tp2_decode_window_batch(*shard_b.context, pair_, tokens,
                                                            positions, kv_table_rows, state_slots,
                                                            columns, envelope, logits);
        }
        return;
    }
    if (shard_a.workspace->used() != graph->arena_begin[0] ||
        shard_b.workspace->used() != graph->arena_begin[1]) {
        throw std::logic_error(
            "TP-2 batched decode CUDA Graph replay found a different workspace layout");
    }
    launch_window_graph(*graph);
    if (graph->arena_bytes[0] > 0) { (void)shard_a.workspace->alloc_bytes(graph->arena_bytes[0]); }
    if (graph->arena_bytes[1] > 0) { (void)shard_b.workspace->alloc_bytes(graph->arena_bytes[1]); }
}

void TP2GenerationCore::mtp_chain_body(Shard& shard, Tensor& mtp_input, const std::int32_t* pins,
                                       std::int32_t* host_drafts, const WindowGraph& bucket,
                                       DeviceArena& ws) {
    auto& ctx = *shard.context;
    const auto& config = shard.model->config().text;
    const std::int32_t hidden = qwen::execution::dimension(config.hidden_size);
    const std::int32_t vocab  = qwen::execution::dimension(config.vocab_size);
    shard.device.bind_to_current_thread();
    cudaStream_t stream = shard.device.stream;
    // The three per-round scalars reach the chain through graph-capturable copies from the pinned
    // buffer: a set_i32_scalar would bake the capture's values into its kernel, and every replay
    // would run at the capture's position. The autoregressive position advances on the device.
    Tensor anchor_dev  = ws.alloc(DType::I32, {1});
    Tensor positions   = ws.alloc(DType::I32, {1});
    Tensor ar_position = ws.alloc(DType::I32, {1});
    CUDA_CHECK(cudaMemcpyAsync(anchor_dev.data, pins, sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(positions.data, pins + 1, sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ar_position.data, pins + 2, sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    Tensor drafts    = ws.alloc(DType::I32, {static_cast<std::int32_t>(mtp_drafts_)});
    Tensor ar_hidden = ws.alloc(DType::BF16, {hidden, 1});
    Tensor logits    = ws.alloc(DType::BF16, {vocab, 1});
    // Graph and EagerBucket run one bucket-wide envelope for every step - the captured launch
    // geometry must cover every replayed position the bucket contains. The envelope only sets split
    // policy and chunking; the kernels derive their visible set from the device positions. EagerExact
    // reproduces the pre-graph per-step envelopes as the A/B partner.
    const bool exact = mtp_chain_mode_ == StepLaunchMode::EagerExact;
    const auto step_envelope = [&](std::uint32_t visible) {
        return ops::CausalAttentionExecutionEnvelope{exact ? visible : bucket.visible_begin,
                                                     exact ? visible : bucket.visible_end};
    };
    const std::uint32_t base = static_cast<std::uint32_t>(pins[1]);
    Tensor draft0 = drafts.slice(0, 0, 1);
    ctx.mtp_forward_batch(anchor_dev, mtp_input, positions, step_envelope(base + 1U), ar_hidden, 0,
                          &logits, &draft0);
    for (std::uint32_t i = 1; i < mtp_drafts_; ++i) {
        Tensor previous    = drafts.slice(0, static_cast<std::int32_t>(i) - 1, 1);
        Tensor next_draft  = drafts.slice(0, static_cast<std::int32_t>(i), 1);
        Tensor step_hidden = ws.alloc(DType::BF16, {hidden, 1});
        ctx.mtp_forward_ar_step(previous, ar_hidden, ar_position, step_envelope(base + i + 1U),
                                step_hidden, logits, next_draft);
        CUDA_CHECK(cudaMemcpyAsync(ar_hidden.data, step_hidden.data, ar_hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
        ops::increment_i32_scalar(ar_position, stream);
    }
    // The drafts land in the pinned buffer: the round reads them after one sync, and a replay
    // publishes a fresh set without running any host code.
    CUDA_CHECK(cudaMemcpyAsync(host_drafts, drafts.data, sizeof(TokenId) * mtp_drafts_,
                               cudaMemcpyDeviceToHost, stream));
}

void TP2GenerationCore::capture_mtp_chain_graph(WindowGraph& graph, Tensor& mtp_input,
                                                const std::int32_t* pins, std::int32_t* host_drafts,
                                                DeviceArena& ws) {
    Shard& shard_a = shard_a_;
    Shard& shard_b = shard_b_;
    shard_a.device.bind_to_current_thread();
    graph.round_base[0]  = shard_a.round_base;
    graph.round_base[1]  = shard_b.round_base;
    graph.arena_begin[0] = ws.used();
    graph.arena_begin[1] = shard_b.workspace->used();
    // The chain runs on shard 0, but the column-split token embedding (and a split proposal head)
    // enqueue pair work on both streams, so the capture records both like the verify window does.
    DecodeGraphDefinition* definitions[2] = {&graph.definition[0], &graph.definition[1]};
    cudaStream_t streams[2] = {shard_a.device.stream, shard_b.device.stream};
    if (graph.ar_channel == tp::DevicePair::kNoArChannel) {
        graph.ar_channel = pair_.create_ar_channel();
    }
    pair_.begin_capture(graph.ar_channel);
    const ArCaptureGuard capture_guard{pair_};
    DecodeGraphDefinition::capture_group(definitions, streams, [&] {
        mtp_chain_body(shard_a, mtp_input, pins, host_drafts, graph, ws);
    });
    pair_.end_capture();
    graph.arena_bytes[0] = ws.used() - graph.arena_begin[0];
    graph.arena_bytes[1] = shard_b.workspace->used() - graph.arena_begin[1];
    install_window_graph(graph);
}

std::vector<TokenId> TP2GenerationCore::mtp_propose_window(Shard& shard, Tensor& mtp_input,
                                                           std::int32_t anchor,
                                                           std::uint32_t position,
                                                           DeviceArena& ws) {
    // The chain exists only on the single-lane MTP route (P3.1); the multi-lane route never calls
    // this, and a future caller that does gets a diagnostic instead of a null dereference.
    if (mtp_chain_host_ == nullptr) {
        throw std::logic_error("TP-2 MTP chain is only built for the single-lane MTP route");
    }
    // Every per-round input reaches the chain through this pinned buffer: [anchor(1), position(1),
    // position+1(1), drafts(K)]. The capture re-reads it through memcpy nodes on every replay and
    // the chain publishes its drafts back into the trailing slots.
    auto* pins            = static_cast<std::int32_t*>(mtp_chain_host_->data());
    std::int32_t* host_drafts = pins + 3;
    pins[0]               = anchor;
    pins[1]               = static_cast<std::int32_t>(position);
    pins[2]               = static_cast<std::int32_t>(position) + 1;
    const std::uint32_t visible_end = position + mtp_drafts_ + 1U;
    WindowGraph* bucket = select_window_graph(mtp_chain_graphs_, visible_end);
    if (bucket == nullptr) {
        throw std::logic_error("TP-2 MTP chain coverage is incomplete");
    }
    if (mtp_chain_mode_ != StepLaunchMode::Graph) {
        // The eager routes run the same body: EagerBucket shares the captured sequence's bucket
        // envelope (an A/B of the launch mechanism alone) and EagerExact reproduces the pre-graph
        // per-step envelopes (an A/B of the envelope's reduction shape).
        mtp_chain_body(shard, mtp_input, pins, host_drafts, *bucket, ws);
    } else {
        WindowGraph* graph = bucket;
        if (reusable_window_graph(mtp_chain_graphs_, visible_end) == nullptr) {
            // Capturing records the sequence without executing it, so the capture round still has to
            // run the chain it just captured (through the graph, with the operands the capture
            // allocated).
            capture_mtp_chain_graph(*graph, mtp_input, pins, host_drafts, ws);
            launch_window_graph(*graph);
        } else {
            if (ws.used() != graph->arena_begin[0] ||
                shard_b_.workspace->used() != graph->arena_begin[1]) {
                throw std::logic_error(
                    "TP-2 MTP chain CUDA Graph replay found a different workspace layout");
            }
            launch_window_graph(*graph);
            // The captured body allocated its operands during capture; a replay does not run that
            // host code, so advance the arena by what the capture consumed. A side the body only
            // touched through scope-unwound scratch nets zero, and zero is a no-op here exactly as
            // in position_arena.
            if (graph->arena_bytes[0] > 0) { (void)ws.alloc_bytes(graph->arena_bytes[0]); }
            if (graph->arena_bytes[1] > 0) {
                (void)shard_b_.workspace->alloc_bytes(graph->arena_bytes[1]);
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(shard.device.stream));
    std::vector<TokenId> host(mtp_drafts_, 0);
    for (std::uint32_t i = 0; i < mtp_drafts_; ++i) {
        host[i] = static_cast<TokenId>(host_drafts[i]);
    }
    return host;
}

// Width one TP-2 prefill chunk runs with. The engine option sets it, the model's chunk maximum
// bounds it, and the cross-device all-reduce staging buffer bounds it again: a chunk's every layer
// issues a BF16 all-reduce - the mixer and FFN deltas at [hidden, T], a gather mixer's activation at
// its own full width, a Vision handoff at [hidden, count] - and the DevicePair carries one payload in
// one transport call. A payload past the in-kernel staging buffer would take the host-staging
// fallback, which synchronizes both compute streams on every collective, so a chunk that does not fit
// is narrowed here instead of silently changing transport. The widest payload per token is the
// largest of those widths; the draft's feature staging and the prefill sinks are sized from this
// same width.
std::uint32_t TP2GenerationCore::prefill_chunk_width(const qwen::TextConfig& config) const {
    const std::uint32_t width =
        std::min<std::uint32_t>(std::max<std::uint32_t>(options_.prefill_chunk, 64U),
                                kPrefillChunkMaximum);
    const std::size_t staging = pair_.in_kernel_allreduce_bytes();
    // Zero means the pair has no in-kernel transport: the host-staging fallback grows its pinned
    // buffer to any payload, so only the option and the workspace bound the chunk.
    if (staging == 0) { return width; }
    std::uint64_t widest = config.hidden_size;
    if (config.gdn) { widest = std::max(widest, config.gdn->value_width()); }
    if (config.attention) { widest = std::max(widest, config.attention->query_width()); }
    const std::size_t per_token = static_cast<std::size_t>(widest) * sizeof(std::uint16_t);
    const std::size_t fits      = per_token == 0 ? width : staging / per_token;
    // The narrowest chunk is 64 tokens. A model wide enough that even that overflows the buffer keeps
    // the 64-token chunk and is carried by the fallback's own capacity rather than refused here.
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>(width, std::max<std::size_t>(fits, 64U)));
}

std::optional<qwen::execution::DFlashFeatureSink>
TP2GenerationCore::make_dflash_prefill_sink(Shard& shard, std::int32_t lane) {
    if (shard.dflash_round == nullptr) { return std::nullopt; }
    // The chunk width the prefill forward will actually run with. The sink's buffer is sized for the
    // maximum and each chunk slices its own prefix out of it (capture_positions copies the matching
    // positions prefix, consume_prefill_chunk slices the feature prefix). Only the prefill fields are
    // set: the batch (verify-window) fields stay null until the masked draft has its verify wiring.
    // The single-lane route stages into the ring's lane 0; a batched lane stages into its own slot,
    // and the whole captured chunk is committed either way.
    const std::uint32_t chunk = prefill_chunk_width(shard.model->config().text);
    return shard.dflash_round->make_prefill_sink(qwen::execution::ExecutionCore{
        .device           = shard.device,
        .parameters       = *shard.parameters,
        .work             = *shard.workspace,
        .linear_attention = *shard.state,
        .replay_records   = nullptr,
        .io               = shard.io,
        .prefill_hidden   = shard.prefill_hidden,
        .prefill_chunk    = chunk,
        .proposal_head    = options_.speculative.proposal_head,
    }, lane);
}

void TP2GenerationCore::store_dflash_image(Shard& shard, void* image, std::size_t image_bytes,
                                           std::uint32_t lane) {
    if (shard.dflash_round == nullptr) { return; }
    const std::size_t lane_bytes = shard.dflash_round->lane_context_image_bytes();
    if (image_bytes < static_cast<std::size_t>(lane + 1) * lane_bytes) {
        throw std::logic_error("draft context host image is smaller than the lane's draft ring");
    }
    shard.device.bind_to_current_thread();
    shard.dflash_round->copy_context_to_host(
        static_cast<std::byte*>(image) + static_cast<std::size_t>(lane) * lane_bytes,
        shard.device.stream, static_cast<std::int32_t>(lane));
}

void TP2GenerationCore::load_dflash_image(Shard& shard, const void* image, std::size_t image_bytes,
                                          std::uint32_t lane) {
    if (shard.dflash_round == nullptr) { return; }
    const std::size_t lane_bytes = shard.dflash_round->lane_context_image_bytes();
    if (image_bytes < static_cast<std::size_t>(lane + 1) * lane_bytes) {
        throw std::logic_error("draft context host image is smaller than the lane's draft ring");
    }
    shard.device.bind_to_current_thread();
    shard.dflash_round->copy_context_from_host(
        static_cast<const std::byte*>(image) + static_cast<std::size_t>(lane) * lane_bytes,
        shard.device.stream, static_cast<std::int32_t>(lane));
}

void TP2GenerationCore::snapshot_dflash_state(Shard& shard, std::size_t slot,
                                              std::uint32_t lane) {
    if (shard.dflash_round == nullptr) { return; }
    if (slot >= shard.dflash_snapshots.size() || shard.dflash_snapshots[slot].data == nullptr) {
        // Slot 0's draft image is served by the host checkpoint ring when that ring exists (the
        // PromptEnd slot), so there is no device plane to fill and nothing to report.
        if (slot == 0 && host_checkpoint_stride_ != 0) { return; }
        throw std::logic_error("draft context snapshot slot is unavailable");
    }
    const std::size_t lane_bytes = shard.dflash_round->lane_context_image_bytes();
    const DeviceSpan snapshot     = shard.dflash_snapshots[slot];
    if (snapshot.bytes < static_cast<std::size_t>(lane + 1) * lane_bytes) {
        throw std::logic_error("draft context snapshot is smaller than the lane's draft ring");
    }
    shard.device.bind_to_current_thread();
    shard.dflash_round->copy_context_to_device(
        DeviceSpan{static_cast<std::byte*>(snapshot.data) +
                       static_cast<std::size_t>(lane) * lane_bytes,
                   lane_bytes},
        shard.device.stream, static_cast<std::int32_t>(lane));
}

TP2GenerationCore::Submission TP2GenerationCore::submit(
    qwen::PreparedPrompt prompt, PromptSummary summary, double prepare_seconds,
    ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
    GenerationObservationOptions, std::chrono::steady_clock::time_point pending_deadline) {
    auto output = frontend_->make_output_session(prompt, options.stop, options.output,
                                                 options.execution.thinking);
    // Multi-lane (P1.4, S1): the KV pool is shared and handed out per request, so the ceiling here
    // is the admission policy - `--lane-context` when configured, the whole pool otherwise - already
    // clamped by the route's write margin, which is what lets the executors' requeue path terminate.
    // Single-lane keeps the advertised options_.max_context.
    const std::uint32_t context_window = lane_admission_limit();
    // A prompt longer than the admission ceiling can never fit the pool, so reject it before the
    // subtraction below wraps it into a capacity that looks infinite.
    if (lanes_ > 1 && summary.prompt_tokens > context_window) {
        // Carry the arithmetic in the message: the caller has to know whether the per-lane policy
        // or the pool itself is the binding constraint before deciding what to change.
        const std::uint32_t pool_pages = pages_for_tokens(options_.max_context);
        const std::uint32_t pool_tokens =
            pool_pages * static_cast<std::uint32_t>(kPagedKVPageSize);
        const bool policy_bound =
            options_.lane_context != 0U && context_window == options_.lane_context;
        char message[512];
        if (policy_bound) {
            std::snprintf(message, sizeof(message),
                          "prompt exceeds this lane's context capacity: prompt %u tokens > lane "
                          "ceiling %u tokens (--lane-context %u); raise --lane-context to widen it",
                          summary.prompt_tokens, context_window, options_.lane_context);
        } else {
            std::snprintf(message, sizeof(message),
                          "prompt exceeds this lane's context capacity: prompt %u tokens > lane "
                          "ceiling %u tokens (KV pool %u pages x %u tokens = %u tokens minus a "
                          "%u-token write margin, --lane-context 0 = whole pool); raise "
                          "--max-context to widen it",
                          summary.prompt_tokens, context_window, pool_pages,
                          static_cast<unsigned>(kPagedKVPageSize), pool_tokens, lane_kv_margin());
        }
        throw RequestError(RequestErrorKind::ContextLengthExceeded, message);
    }
    const std::uint32_t capacity_output =
        context_window - summary.prompt_tokens + static_cast<std::uint32_t>(1);
    const std::uint32_t effective =
        std::min(options.execution.requested_output_tokens, capacity_output);
    const FinishReason limit_reason =
        options.execution.requested_output_tokens <= capacity_output ? FinishReason::OutputLimit
                                                                     : FinishReason::ContextCapacity;
    if (limit_reason == FinishReason::ContextCapacity) {
        // The ceiling, not the client, shortened this answer. Say so once per request so a
        // truncated completion is not read as a model or client failure.
        std::fprintf(stderr,
                     "[tp2-capacity] prompt %u + requested output %u exceeds the %u-token lane "
                     "context ceiling; the output budget is clamped to %u\n",
                     summary.prompt_tokens, options.execution.requested_output_tokens,
                     context_window, effective);
    }
    try {
        output.validate_generation_capacity(effective);
    } catch (const std::invalid_argument& error) {
        throw RequestError(RequestErrorKind::ThinkingBudgetCapacityInsufficient, error.what());
    }
    GenerationBudget budget(effective, limit_reason);
    auto request = std::make_unique<Request>(std::move(prompt), std::move(output), summary,
                                             prepare_seconds, std::move(budget),
                                             options.execution.sampling, consumer_mode);
    // The FIFO wait is the only part of the deadline this core owns: preparation and media already
    // ran against it in the service layer, and an admitted request is no longer pending.
    request->pending_deadline = pending_deadline;
    // Preparation is already behind us, so this instant is the same anchor the single-device route
    // uses: the queue wait a lane measures from here excludes preparation, and prepare_seconds is
    // added back exactly once when the result is assembled.
    request->submitted = Clock::now();
    return Submission(*this, std::move(request));
}

GenerationResult TP2GenerationCore::Submission::wait(OutputSink* sink,
                                                     const CancellationView& cancellation) {
    if (owner_ == nullptr || request_ == nullptr) {
        throw std::logic_error("concurrent submission is empty");
    }
    const bool streaming = request_->consumer_mode == OutputConsumerMode::Streaming;
    if (streaming != (sink != nullptr)) {
        throw std::invalid_argument(
            "GenerationHandle wait sink does not match its submitted consumer mode");
    }
    // The multi-lane route owns its own admission: the request joins the core's FIFO instead of
    // taking the device here, and the core's driver thread holds `execution_mutex_` while it runs
    // the batch that picks it up. The single-lane route keeps the direct hand-off.
    if (owner_->lanes_ > 1) {
        return owner_->wait_lanes(std::move(request_), sink, cancellation);
    }
    // Take the single execution slot. A request cancelled while queued still enters execute(),
    // whose first cancellation check is cheap and returns a Cancelled result with the proper
    // preview state.
    std::unique_lock<std::mutex> slot(owner_->execution_mutex_);
    return owner_->execute(*request_, sink, cancellation);
}

namespace {

// Env-gated lane trace: the same switch the driver's batch line reads, so one env var turns the
// whole per-lane story on. It lives here, above the driver, because drive_lane_queue prints the
// batch line this switch gates.
bool lane_trace_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("NINFER_TP2_LANE_TRACE");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Multi-lane admission (tp2-plan-concurrency.md P1.4)
// ---------------------------------------------------------------------------------------------

// A submission to the multi-lane route never touches the device itself. It appends itself to the
// FIFO and waits for the core's own driver thread to retire it. The driver forms batches of at most
// `lanes_` from the queue head, so admission is FIFO and no lane is ever handed a request younger
// than one still waiting.
//
// P0.2: the submitter is never the driver, and each member is retired the moment its own lane
// finishes (publish_lane). Together those rules stop a finished request from waiting behind a batch
// it does not depend on, which is what used to hold a 773 ms walk for four minutes until the client
// disconnected first (HTTP 499).
//
// While the member is still queued the submitter owns two checks the executors cannot make yet: the
// request's absolute deadline (`--pending-timeout-ms`, enforced here as the documented 503) and the
// client's cancellation. Neither can be signalled into `lane_queue_cv_` - the deadline is not a
// mutex-protected state change and the cancellation flag is an atomic set by the HTTP thread - so
// the wait is bounded and re-checks them. Admission hands both over: `admitted` is set by whoever
// takes the member out of the FIFO, and a member put back by `requeue_lane_front` clears it again.
constexpr std::chrono::milliseconds kQueuedCancelPollInterval{250};

// `--pending-timeout-ms` is measured from request acquisition, so a request can already be expired
// when it reaches this FIFO. A default-constructed point and `time_point::max()`
// (`DeadlinePolicy::UnboundedStartup`) both mean "no deadline".
bool queued_deadline_passed(std::chrono::steady_clock::time_point deadline) {
    return deadline != std::chrono::steady_clock::time_point{} &&
           deadline != std::chrono::steady_clock::time_point::max() &&
           std::chrono::steady_clock::now() >= deadline;
}

// A queued member whose client is gone never gets a lane: retire it here, the way the single-GPU
// route retires a cancelled pending request. The terminal preview and the Cancelled finish reason
// are the ones the executors' own first cancellation check produces, so the submitter sees the same
// shape whether the disconnect was noticed before or after admission.
void TP2GenerationCore::drop_cancelled_lane(PendingRequest& pending) {
    (void)pending.request->output.preview_terminal(FinishReason::Cancelled);
    pending.result.finish_reason = FinishReason::Cancelled;
    publish_lane(pending);
}

// An expired queued member is an error, not a cancelled completion: no preview is emitted and the
// stored failure is what the submitter rethrows.
void TP2GenerationCore::drop_expired_lane(PendingRequest& pending) {
    pending.failure = std::make_exception_ptr(RequestError(
        RequestErrorKind::QueueTimeout, "inference request expired while waiting for admission"));
    publish_lane(pending);
}

GenerationResult TP2GenerationCore::wait_lanes(std::unique_ptr<Request> request, OutputSink* sink,
                                               const CancellationView& cancellation) {
    // `--pending-timeout-ms` is absolute and starts before preparation, so it can already be in the
    // past by the time the request reaches this FIFO. A default-constructed point and
    // `time_point::max()` (`DeadlinePolicy::UnboundedStartup`) both mean "no deadline".
    const std::chrono::steady_clock::time_point deadline = request->pending_deadline;
    const bool bounded =
        deadline != std::chrono::steady_clock::time_point{} &&
        deadline != std::chrono::steady_clock::time_point::max();
    auto pending              = std::make_shared<PendingRequest>();
    pending->request          = std::move(request);
    pending->sink             = sink;
    pending->cancellation     = cancellation;
    const std::shared_ptr<PendingRequest> mine = pending;
    bool queued_timed_out                      = false;
    bool queued_cancelled                      = false;
    {
        std::unique_lock<std::mutex> queue(lane_queue_mutex_);
        lane_queue_.push_back(std::move(pending));
        // Wake the driver thread's batch-formation window: this enqueue is exactly the arrival it is
        // waiting for before it commits to a batch.
        lane_queue_cv_.notify_all();
        // P0.2: the submitter never drives. It waits for its own member only, so a request whose
        // lane finishes early returns immediately even while the rest of its batch keeps decoding.
        for (;;) {
            if (mine->complete) { break; }
            if (mine->admitted) {
                // The driver owns it now; a requeue clears `admitted` and wakes this thread, so this
                // is not necessarily the end of the wait.
                lane_queue_cv_.wait(queue, [&] { return mine->complete || !mine->admitted; });
                continue;
            }
            // The deadline outranks the cancellation, the order the single-GPU admission uses.
            if (queued_deadline_passed(deadline)) {
                queued_timed_out = true;
                break;
            }
            if (cancellation.requested()) {
                queued_cancelled = true;
                break;
            }
            const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            if (!bounded) {
                lane_queue_cv_.wait_for(queue, kQueuedCancelPollInterval);
            } else {
                lane_queue_cv_.wait_until(queue, std::min(deadline, now + kQueuedCancelPollInterval));
            }
        }
        if (queued_cancelled || queued_timed_out) {
            // Still in the FIFO: `admitted` is only ever set under this lock, together with the pop
            // that takes the member out of the queue, and a requeue puts it back before clearing it.
            lane_queue_.erase(std::remove_if(lane_queue_.begin(), lane_queue_.end(),
                                             [&](const std::shared_ptr<PendingRequest>& queued) {
                                                 return queued.get() == mine.get();
                                             }),
                              lane_queue_.end());
        }
    }
    // Both paths retire the member exactly as the driver's own pop does, so the submitter sees the
    // same shape whichever thread noticed: HTTP 503 request_queue_timeout for the deadline
    // (docs/serving.md, --pending-timeout-ms), 499 for a client that is already gone.
    if (queued_timed_out) { drop_expired_lane(*mine); }
    if (queued_cancelled) { drop_cancelled_lane(*mine); }
    if (mine->failure != nullptr) { std::rethrow_exception(mine->failure); }
    return std::move(mine->result);
}

// How long the driver thread waits for peers to arrive once it has found the queue non-empty. A
// submission only wakes the driver, so without this window a burst of requests that are already
// inside submit() would still run one at a time: each would drain its own batch before the next one
// landed. The window only delays a batch that is not yet full, and the batch it delays would have
// run for hundreds of milliseconds anyway.
constexpr std::chrono::microseconds kBatchFormationWindow{3000};

// Upper bound on the whole formation phase. The driver keeps the window open while the batch is
// still short of the route's lane capacity and members keep arriving, so a burst whose submissions
// are spread over more than one window lands in a single batch instead of splitting into a first
// batch and stragglers that then wait a whole round for mid-batch admission. A lone request never
// reaches this bound: the extension stops as soon as one window passes with an empty queue, so it
// only ever pays the initial window.
constexpr std::chrono::microseconds kBatchFormationMaximum{50000};

// Retire one member. The result and any streamed preview are already in place when this is called,
// so setting `complete` under the queue lock and waking the waiters is the whole hand-off.
void TP2GenerationCore::publish_lane(PendingRequest& pending) {
    // Read the count before `complete` is visible: the submitter owns the result from the moment it
    // wakes, and this thread must not touch it afterwards.
    const std::size_t tokens = pending.result.generated_token_ids.size();
    std::size_t queued       = 0;
    {
        std::unique_lock<std::mutex> queue(lane_queue_mutex_);
        pending.complete = true;
        queued           = lane_queue_.size();
        lane_queue_cv_.notify_all();
    }
    static const bool lane_trace = [] {
        const char* env = std::getenv("NINFER_TP2_LANE_TRACE");
        return env != nullptr && env[0] == '1';
    }();
    if (lane_trace) {
        std::fprintf(stderr, "[tp2-lane] publish tokens=%zu queued=%zu\n", tokens, queued);
    }
}

// The driver thread's loop. Each pass takes the oldest lanes_ requests, runs them on the devices,
// and immediately looks for work that arrived in the meantime. A member is retired by its own
// lane's finalize, so the end of the batch never gates a member's response.
void TP2GenerationCore::drive_lane_queue() {
    for (;;) {
        std::vector<std::shared_ptr<PendingRequest>> batch;
        std::vector<std::shared_ptr<PendingRequest>> cancelled_queued;
        std::vector<std::shared_ptr<PendingRequest>> expired_queued;
        {
            std::unique_lock<std::mutex> queue(lane_queue_mutex_);
            if (lane_queue_.empty()) {
                // Idle: sleep until a submission arrives rather than polling the queue.
                lane_queue_cv_.wait(queue,
                                    [&] { return lane_driver_stop_ || !lane_queue_.empty(); });
                if (lane_queue_.empty()) { return; } // the destructor asked the driver to stop
                // Give the rest of the burst that is already inside submit() the same window the
                // leader-driver used to give it, so it lands in this batch.
                lane_queue_cv_.wait_for(queue, kBatchFormationWindow);
            }
            batch.reserve(lanes_);
            // Drain what has already arrived. Taking the member out of the submitter's hands happens
            // first, whatever happens to it below: from here the driver owns its retirement.
            const auto take_from_queue = [&] {
                while (batch.size() < lanes_ && !lane_queue_.empty()) {
                    std::shared_ptr<PendingRequest> next = std::move(lane_queue_.front());
                    lane_queue_.pop_front();
                    next->admitted = true;
                    // Neither of these may take a lane, and the deadline outranks the cancellation,
                    // as it does in the single-GPU admission.
                    if (queued_deadline_passed(next->request->pending_deadline)) {
                        expired_queued.push_back(std::move(next));
                        continue;
                    }
                    if (next->cancellation.requested()) {
                        // The client disconnected while it waited. Do not spend a lane and a round on
                        // a request that would abandon both at its first check.
                        cancelled_queued.push_back(std::move(next));
                        continue;
                    }
                    batch.push_back(std::move(next));
                }
            };
            // One window for the rest of the burst, and then keep the window open while members keep
            // arriving and the batch is still short of lane capacity. Each iteration drains the queue
            // first, so a batch that fills stops immediately and the phase never waits longer than
            // kBatchFormationMaximum.
            const Clock::time_point formation_deadline = Clock::now() + kBatchFormationMaximum;
            for (;;) {
                take_from_queue();
                if (batch.size() >= lanes_ || Clock::now() >= formation_deadline) { break; }
                if (!lane_queue_cv_.wait_for(queue, kBatchFormationWindow,
                                             [&] { return lane_driver_stop_ || !lane_queue_.empty(); }) ||
                    lane_driver_stop_) {
                    break;
                }
            }
        }
        for (auto& pending : expired_queued) { drop_expired_lane(*pending); }
        for (auto& pending : cancelled_queued) { drop_cancelled_lane(*pending); }
        if (batch.empty()) {
            if (lane_driver_stop_) { return; }
            continue;
        }
        // Reachable batched shapes on this route: the plain route, MTP and DFlash2. --spec dflash
        // is rejected at construction, so there is no route refusal left to test: every batch that
        // reaches this loop is batchable.
        // A multimodal member no longer forces the serial lane walk (P2.4): the batch prefills one
        // lane at a time, so that lane's Vision session runs on the single startup arena exactly as
        // the serial walk's does.
        // P2.1: a tool-call grammar no longer refuses the batch either. Each executor keeps one
        // constraint per lane and masks that lane's own columns of the round's logits, so a request
        // carrying tools takes the same batched route as one that does not.
        // P0.1: batch formation is inside the try. Reading a member's prepared prompt can throw (an
        // empty prompt is an invalid_argument), and with the driver on its own thread that exception
        // would otherwise escape the loop and abort the process. Here it fails this batch whole,
        // which is what the D4 whole-batch failure rule already means.
        std::unique_lock<std::mutex> device(execution_mutex_);
        try {
            if (lane_trace_enabled()) {
                std::fprintf(stderr, "[tp2-lane] batch=%zu capacity=%u path=batched\n", batch.size(),
                             static_cast<unsigned>(lanes_));
            }
            // Writes every member's result in place. Both speculative routes drive their own batched
            // proposal/verify/accept round; the plain route decodes one token per lane.
            if (dflash2_enabled_ || mtp_enabled_) {
                execute_spec_batch(batch);
            } else {
                execute_plain_batch(batch);
            }
        } catch (...) {
            // D4: the batch commits or fails whole. A lane that already finished its own walk is
            // discarded with the rest. Its published retention state is deliberately not rolled
            // back: the device pools still hold the prompt that lane prefilled, so the catalog and
            // the checkpoint ring can keep naming it.
            //
            // The sink only surfaces a generic internal error, so report the reason here; without
            // this line a failing batch is undiagnosable from the client side.
            const std::exception_ptr failure = std::current_exception();
            try {
                std::rethrow_exception(failure);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[tp2-lane] batch of %zu failed: %s\n", batch.size(), error.what());
            } catch (...) {
                std::fprintf(stderr, "[tp2-lane] batch of %zu failed: unknown error\n", batch.size());
            }
            // A failed batch leaves every lane's device state where the catalog cannot name it. The
            // blast radius is therefore the whole lane set, not the lane that raised: a round is one
            // collective, so a mid-round failure (an allreduce stall above all) can desync a lane that
            // never raised, and the KV/GDN images of the others were written by a partially executed
            // round. Discarding every lane's recall claim and host checkpoints costs a re-prefill;
            // guessing which lanes are still sound would cost correctness (P3.4/B4, docs/serving.md).
            session_invalidate_all();
            // The throw may have landed between a launch and its wait - a stalled collective returns
            // from the kernel only on its own bounded expiry - so drain both streams before any
            // resource goes back: a page handed to another lane while a kernel is still writing it is
            // somebody else's tokens.
            for (Shard* shard : {&shard_a_, &shard_b_}) {
                shard->device.bind_to_current_thread();
                CUDA_CHECK(cudaStreamSynchronize(shard->device.stream));
            }
            shard_a_.device.bind_to_current_thread();
            // S1: a round that threw can leave a lane's pages reserved with no request owning them.
            // They go back to the shared pool here, so a failed batch does not cost the pool the
            // capacity of every lane it touched.
            for (std::int32_t lane = 0; lane < lanes_; ++lane) {
                release_lane_kv(static_cast<std::uint32_t>(lane));
            }
            // D4 narrowed by P0.2: a member whose lane already retired keeps the result it published
            // - that lane committed, and its submitter may already be gone - so only the members
            // still running take the failure. Writing `failure` under the queue lock before `complete`
            // is set is what lets a submitter observe both without a race.
            std::unique_lock<std::mutex> queue(lane_queue_mutex_);
            for (auto& member : batch) {
                if (!member->complete) { member->failure = failure; }
            }
        }
        // Safety net: an executor that abandoned a lane before its own finalize still has to wake
        // that member's submitter, or the request would wait for a result nobody will publish.
        for (auto& member : batch) {
            if (!member->complete) { publish_lane(*member); }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Batched-lane prefix retention (tp2-plan-concurrency.md P2.3)
// ---------------------------------------------------------------------------------------------
//
// A batched lane owns one KV row, one GDN state slot and one slice of every retention buffer, so
// it runs the same boundary scan the single-lane walk runs on the lane it is prefilling: recall the
// conversation the prompt belongs to, accept the deepest boundary this lane's own lineage offers,
// restore its GDN state (and, on a masked-draft route, its draft ring), then prefill only the
// suffix. The three executors share these steps rather than a copy each.
std::vector<TP2GenerationCore::MediaSpan> TP2GenerationCore::collect_media_spans(
    const qwen::PreparedPromptData& data, std::size_t limit) {
    std::vector<MediaSpan> spans;
    spans.reserve(data.vision_items.size());
    for (const qwen::VisionItem& item : data.vision_items) {
        if (item.token_spans.empty()) { continue; }
        std::uint32_t begin = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t end   = 0;
        for (const qwen::TokenSpan& span : item.token_spans) {
            begin = std::min(begin, static_cast<std::uint32_t>(span.begin));
            end   = std::max(end, static_cast<std::uint32_t>(span.begin + span.count));
        }
        // An item the prompt stops in the middle of is not one this prompt carries whole, so it
        // must not license a boundary: the KV a reuse would keep ends inside it.
        if (end > limit) { continue; }
        MediaSpan entry;
        entry.begin = begin;
        entry.end   = end;
        entry.item  = item;
        spans.push_back(std::move(entry));
    }
    return spans;
}

std::size_t TP2GenerationCore::media_prefix_cap(std::span<const MediaSpan> cached,
                                                std::span<const MediaSpan> incoming,
                                                std::size_t shared_prefix) {
    std::size_t left_index  = 0;
    std::size_t right_index = 0;
    while (left_index < cached.size() || right_index < incoming.size()) {
        const MediaSpan* left  = left_index < cached.size() ? &cached[left_index] : nullptr;
        const MediaSpan* right = right_index < incoming.size() ? &incoming[right_index] : nullptr;
        // A boundary at or before an item's begin does not reuse that item, so it cannot constrain
        // the boundary, and the lists are ordered, so nothing after it can either.
        if (left != nullptr && left->begin >= shared_prefix) { left = nullptr; }
        if (right != nullptr && right->begin >= shared_prefix) { right = nullptr; }
        if (left == nullptr && right == nullptr) { break; }
        if (left != nullptr && right != nullptr && left->begin == right->begin) {
            if (!qwen::detail::same_vision_item(left->item, right->item)) { return left->begin; }
            ++left_index;
            ++right_index;
            continue;
        }
        // One prompt shows an item where the other does not: the KV past that point belongs to
        // media the other prompt never showed.
        return left != nullptr && (right == nullptr || left->begin < right->begin) ? left->begin
                                                                                  : right->begin;
    }
    return shared_prefix;
}

TP2GenerationCore::LaneReuse TP2GenerationCore::scan_lane_reuse(std::uint32_t lane,
                                                               std::span<const TokenId> token_ids,
                                                               std::span<const MediaSpan> media,
                                                               std::uint32_t prompt_tokens,
                                                               std::size_t replay_split,
                                                               bool adopted, bool trace) {
    RetentionState& lane_state = retention(lane);
    LaneReuse result;
    ReuseReason reuse_reason  = ReuseReason::NoAnchorAtOrBefore;
    std::size_t shared_prefix = 0;
    lane_state.reuse_source   = ReuseSource::None;
    if (lane_state.cached_state_valid && !lane_state.cached_prompt_tokens.empty()) {
        const std::size_t common =
            std::min(lane_state.cached_prompt_tokens.size(), static_cast<std::size_t>(prompt_tokens));
        while (shared_prefix < common &&
               lane_state.cached_prompt_tokens[shared_prefix] == token_ids[shared_prefix]) {
            ++shared_prefix;
        }
        // The token ids cannot see media: every image merges to the same placeholder ids, so two
        // prompts showing different pictures compare equal here. The identity behind those ids is
        // what tells them apart, and a boundary that would keep KV from another picture is not this
        // prompt's to reuse.
        shared_prefix = media_prefix_cap(lane_state.cached_media, media, shared_prefix);
        // The position this prompt stops agreeing with what the lane held. The prefill freezes the
        // state there so the next request restarts from it instead of from the shared block boundary,
        // which is 350 tokens earlier on every turn.
        lane_state.reuse_boundary_target   = static_cast<std::uint32_t>(shared_prefix);
        lane_state.reuse_boundary_captured = false;
        // A checkpoint is only this prompt's while the lineage agrees on the tokens before it, so a
        // checkpoint past the shared prefix can never become usable again.
        for (Shard* shard : {&shard_a_, &shard_b_}) {
            for (auto& checkpoint : shard->host_checkpoints) {
                if (checkpoint.position[lane] > shared_prefix) { checkpoint.valid[lane] = false; }
            }
        }
        // Every image this lane can reach is registered in the ledger, which then makes the one
        // decision. Registration replaces the old "take the deepest" scan; the order below is the
        // priority that scan used, so an equal position still resolves to the same source.
        const std::uint64_t conversation_key =
            lane_state.active_session != kNoSession
                ? static_cast<std::uint64_t>(lane_state.active_session)
                : (0x8000'0000'0000'0000ULL | static_cast<std::uint64_t>(lane));
        ConversationLog& conversation = reuse_ledger_->conversation(conversation_key);
        conversation.anchors.clear();
        // Only the part this prompt agrees with is comparable, and the media identity already capped
        // it: registering the rest would let the decision answer with a position the prompt left.
        conversation.tokens.assign(lane_state.cached_prompt_tokens.begin(),
                                   lane_state.cached_prompt_tokens.begin() +
                                       static_cast<std::ptrdiff_t>(shared_prefix));
        conversation.device_extent = static_cast<std::uint32_t>(shared_prefix);
        conversation.host_extent   = static_cast<std::uint32_t>(shared_prefix);

        struct Reachable {
            std::uint32_t position;
            std::uint32_t slot;
            ReuseSource source;
        };
        std::vector<Reachable> reachable;
        const auto offer = [&](std::uint32_t position, std::uint32_t slot, ReuseSource source,
                               AnchorStorage storage) {
            if (position == 0 || position > shared_prefix || position >= prompt_tokens) { return; }
            reachable.push_back(Reachable{position, slot, source});
            ReuseLedger::AnchorRequest request;
            request.position   = position;
            request.provenance = source == ReuseSource::LiveState ? AnchorProvenance::PromptEnd
                                                                  : AnchorProvenance::Grid;
            request.policy     = source == ReuseSource::LiveState ? AnchorPolicy::Pinned
                                                                  : AnchorPolicy::Lru;
            request.storage    = storage;
            request.kv_source  = storage == AnchorStorage::Host ? AnchorKvSource::HostSlab
                                                                : AnchorKvSource::Device;
            request.lane       = static_cast<std::int32_t>(lane);
            request.slot       = storage == AnchorStorage::Host ? static_cast<std::int32_t>(slot)
                                                                : kNoSlot;
            reuse_ledger_->record(conversation, request);
        };
        if (lane_state.live_state_valid && lane_state.active_session != kNoSession) {
            offer(sessions_[lane_state.active_session].frontier, 0, ReuseSource::LiveState,
                  AnchorStorage::Device);
        }
        for (std::uint32_t slot = 0; slot < kReuseSnapshotCount; ++slot) {
            // A slot the host ring serves has no device plane; the same boundary is offered below
            // through its checkpoint instead.
            if (shard_a_.state_snapshots[slot].data == nullptr) { continue; }
            offer(lane_state.cached_boundaries[slot], slot, ReuseSource::DeviceSnapshot,
                  AnchorStorage::Device);
        }
        for (std::uint32_t slot = 0; slot < shard_a_.host_checkpoints.size(); ++slot) {
            const auto& checkpoint = shard_a_.host_checkpoints[slot];
            if (checkpoint.valid[lane]) {
                offer(checkpoint.position[lane], slot, ReuseSource::HostCheckpoint,
                      AnchorStorage::Host);
            }
        }
        const DecideResult decision = reuse_ledger_->decide(conversation, token_ids);
        reuse_reason                = decision.reason;
        result.reuse_reason         = decision.reason;
        if (decision.reason == ReuseReason::Used && decision.anchor != nullptr) {
            for (const Reachable& item : reachable) {
                if (item.position != decision.reuse) { continue; }
                result.tokens           = item.position;
                result.slot             = item.slot;
                lane_state.reuse_source = item.source;
                break;
            }
        }
    }
    if (trace) {
        std::size_t valid_checkpoints = 0;
        for (const auto& checkpoint : shard_a_.host_checkpoints) {
            valid_checkpoints += checkpoint.valid[lane] ? 1U : 0U;
        }
        const char* source = lane_state.reuse_source == ReuseSource::HostCheckpoint ? "host"
                             : lane_state.reuse_source == ReuseSource::LiveState    ? "live"
                             : lane_state.reuse_source == ReuseSource::DeviceSnapshot ? "device"
                                                                             : "none";
        std::fprintf(stderr,
                     "[tp2-reuse] lane=%u prompt=%u cached=%zu shared=%zu replay_split=%zu "
                     "adopted=%d prefill_end=%u host=%zu/%zu stride=%u -> reuse=%u slot=%u "
                     "src=%s reason=%s\n",
                     lane, prompt_tokens, lane_state.cached_prompt_tokens.size(), shared_prefix,
                     replay_split, adopted ? 1 : 0, lane_state.cached_boundaries[0],
                     valid_checkpoints, shard_a_.host_checkpoints.size(), host_checkpoint_stride_,
                     result.tokens, result.slot, source,
                     reuse_reason == ReuseReason::Used ? "used" : "no-anchor");
    }
    // Tag the checkpoints this prefill leaves behind and start the ring at the first stride multiple
    // past the reused boundary: a checkpoint at the boundary itself would only duplicate the device
    // snapshot the prefill starts from. The new checkpoints become usable when it completes.
    lane_state.host_checkpoint_live_id = host_checkpoint_next_id_++;
    if (host_checkpoint_stride_ != 0) {
        result.next_host_checkpoint = host_checkpoint_stride_;
        while (result.next_host_checkpoint <= result.tokens) {
            result.next_host_checkpoint += host_checkpoint_stride_;
        }
    }
    result.shared_prefix = static_cast<std::uint32_t>(shared_prefix);
    return result;
}

TP2GenerationCore::LaneAnchors TP2GenerationCore::plan_lane_anchors(
    std::uint32_t lane, std::span<const TokenId> token_ids, std::uint32_t prompt_tokens,
    std::uint32_t reuse, std::uint32_t shared_prefix, std::uint32_t prefill_chunk,
    std::uint32_t block_frontier) {
    RetentionState& lane_state = retention(lane);
    LaneAnchors anchors;
    // A checkpoint that already sits on a boundary stays untouched, which is what keeps a reused
    // stable prefix free of work: the state the anchor would freeze is one the ring already holds.
    const auto has_valid_host_checkpoint_at = [](const Shard& shard, std::uint32_t position,
                                                 std::uint32_t checkpoint_lane) {
        for (const auto& checkpoint : shard.host_checkpoints) {
            if (checkpoint.valid[checkpoint_lane] && checkpoint.position[checkpoint_lane] == position) {
                return true;
            }
        }
        return false;
    };
    // Divergence anchor. shared_prefix is where this prompt stopped matching the lineage it
    // inherited, and the next new session - or the next context compression - renders the same stable
    // block and diverges in the same place, so it is the one position a later request is already
    // known to want. Freezing the state there lets that request restart on the boundary instead of at
    // the grid point behind it. The anchor sits kReuseDivergenceMargin tokens *before* the observed
    // divergence rather than on it: the next pair can report a shared prefix a token or two shorter,
    // and an anchor on the exact index would be pruned before it could ever be used. Only a frontier
    // strictly inside this prefill is worth freezing: at or below reuse the prefill's own starting
    // state is already checkpointed, and at the prompt end the device snapshot holds it.
    //
    // The anchor does not always belong to the conversation the prefill was recalled from. A prompt
    // that only opens with a stable block is served by the shallowest image still reachable, while
    // the entry whose history actually carries that block can share far more with it - and it offers
    // no recall at all, because every boundary it holds sits past the shared prefix. The block's end
    // still has to be frozen on it: that entry is the one the next conversation of the family is
    // recalled from, and a ring slot does not survive the evictions a session slab does.
    std::uint32_t anchor_prefix = shared_prefix;
    for (std::size_t index = 0; index < sessions_.size(); ++index) {
        const SessionEntry& entry = sessions_[index];
        if (index == lane_state.anchor_session || entry.device_lane >= 0 || entry.tokens.empty()) {
            continue;
        }
        if (entry.host_shared_state[0] == nullptr || entry.host_shared_state[1] == nullptr) {
            continue;
        }
        const std::size_t common = std::min(entry.tokens.size(), token_ids.size());
        std::size_t shared       = 0;
        while (shared < common && entry.tokens[shared] == token_ids[shared]) { ++shared; }
        if (shared <= anchor_prefix || shared >= prompt_tokens) { continue; }
        const std::uint32_t position = shared > kReuseDivergenceMargin
                                           ? static_cast<std::uint32_t>(shared) - kReuseDivergenceMargin
                                           : static_cast<std::uint32_t>(shared);
        if (position <= reuse || position > entry.host_kv_end) { continue; }
        anchor_prefix             = static_cast<std::uint32_t>(shared);
        lane_state.anchor_session = index;
    }
    const std::uint32_t anchor_target = anchor_prefix > kReuseDivergenceMargin
                                            ? anchor_prefix - kReuseDivergenceMargin
                                            : anchor_prefix;
    // The anchor only ever lands on the prefill's own chunk grid, for the same reason the rewind
    // snapshots do: a chunk truncated to a target derived from engine history makes the same prompt
    // prefill with different chunk widths from one prefill to the next, and the last chunk's logits
    // then differ by far more than a reduction-order change - enough to flip the first sample against
    // a from-scratch oracle. Every clamp a prefill applies already sits on the grid fixed by `reuse`,
    // `prompt_tokens` and `prefill_chunk`, so rounding the anchor down costs only the part of a
    // chunk behind it and never a split.
    anchors.anchor_position = anchor_target;
    if (anchors.anchor_position > reuse) {
        anchors.anchor_position =
            reuse + ((anchors.anchor_position - reuse) / prefill_chunk) * prefill_chunk;
    }
    anchors.anchor_session    = lane_state.anchor_session;
    anchors.anchor_divergence = host_checkpoint_stride_ != 0 && anchors.anchor_position > reuse &&
                                anchors.anchor_position < prompt_tokens &&
                                !has_valid_host_checkpoint_at(shard_a_, anchors.anchor_position, lane) &&
                                !has_valid_host_checkpoint_at(shard_b_, anchors.anchor_position, lane);
    // Block anchor. The prompt names where its own leading instruction block ends, and the prefill
    // that walks it is the only one that can freeze the state there: the block is what every
    // conversation of the same agent re-renders, but the first conversation that shares it is the one
    // that has to walk it, so an observation always arrives too late for it. The boundary is kept for
    // this conversation's own entry (see session_store_active), which is what lets the next one start
    // on the block. The anchor sits on the prefill's own chunk grid, not a fixed distance behind the
    // boundary: a boundary that coincides with a chunk end is the state the prefill committed there
    // anyway, and a recall starting on it re-walks the chunks a from-scratch prefill of the same
    // prompt would, so the recalled answer is the oracle's and not merely a plausible one. The margin
    // keeps the anchor clear of the boundary the divergence may reach back to.
    anchors.block_frontier = block_frontier;
    const std::uint32_t block_target = block_frontier > kReuseDivergenceMargin
                                           ? block_frontier - kReuseDivergenceMargin
                                           : block_frontier;
    anchors.block_position = block_target;
    if (anchors.block_position > reuse) {
        anchors.block_position = reuse + ((anchors.block_position - reuse) / prefill_chunk) * prefill_chunk;
    }
    anchors.block_anchor = host_checkpoint_stride_ != 0 && anchors.block_position > reuse &&
                           anchors.block_position < prompt_tokens &&
                           !has_valid_host_checkpoint_at(shard_a_, anchors.block_position, lane) &&
                           !has_valid_host_checkpoint_at(shard_b_, anchors.block_position, lane);
    return anchors;
}

void TP2GenerationCore::capture_lane_anchor_from_device(std::uint32_t lane, std::uint32_t reuse,
                                                        std::uint32_t shared_prefix) {
    // The conversation this prompt was compared against stopped matching it exactly where the prefill
    // starts, so the device state sitting there right now is its state at that boundary too: freeze it
    // before the first chunk overwrites it. A later conversation opening with the same stable block
    // can then be recalled onto the owner's own slabs instead of prefilling the block again. The
    // device pools still hold it because the recall restored it, or because the eviction left the
    // device lineage standing on it, and nothing has run since. A draft ring the caller restored to
    // `reuse` is at the same boundary, so this captures the two halves together.
    RetentionState& lane_state = retention(lane);
    if (lane_state.anchor_session != kNoSession && reuse != 0 && reuse == shared_prefix) {
        session_capture_shared_state(lane_state.anchor_session, reuse, true, nullptr, nullptr, lane);
    }
}

void TP2GenerationCore::capture_lane_anchor_frozen(std::uint32_t lane, std::uint32_t anchor_position) {
    // The divergence anchor is the state of the block the two histories still agreed on: the entry
    // whose history the scan compared this prompt against agrees with it up to the same token, so the
    // anchor is its state there as well. Copying it into that entry's own image pairs the state with
    // the KV its slabs already hold, and unlike the ring slot it survives the next evictions. It lands
    // before session_publish, which may erase the entry and shift every later index.
    RetentionState& lane_state = retention(lane);
    if (lane_state.anchor_session == kNoSession) { return; }
    const void* frozen[2]    = {nullptr, nullptr};
    const void* frozen_draft = nullptr;
    bool complete            = true;
    for (std::size_t shard_index = 0; shard_index < 2 && complete; ++shard_index) {
        Shard& shard = shard_index == 0 ? shard_a_ : shard_b_;
        for (const auto& checkpoint : shard.host_checkpoints) {
            // The ring has not been published yet, so the search is by the id this prefill tagged its
            // own writes with rather than by `valid`.
            if (checkpoint.buffer != nullptr &&
                checkpoint.prefill_id[lane] == lane_state.host_checkpoint_live_id &&
                checkpoint.position[lane] == anchor_position) {
                frozen[shard_index] = checkpoint.buffer->data();
                // The draft half of the anchor lives in the same checkpoint slot, so a checkpoint
                // without it cannot carry the boundary.
                if (shard.dflash_round != nullptr && checkpoint.dflash_buffer != nullptr) {
                    frozen_draft = checkpoint.dflash_buffer->data();
                }
                break;
            }
        }
        complete = frozen[shard_index] != nullptr &&
                   (shard.dflash_round == nullptr || frozen_draft != nullptr);
    }
    if (!complete) { return; }
    for (Shard* shard : {&shard_a_, &shard_b_}) {
        shard->device.bind_to_current_thread();
        CUDA_CHECK(cudaStreamSynchronize(shard->device.stream));
    }
    session_capture_shared_state(lane_state.anchor_session, anchor_position, false, frozen,
                                 frozen_draft, lane);
}

void TP2GenerationCore::restore_lane_gdn(std::uint32_t lane, const LaneReuse& reuse) {
    const ReuseSource source = retention(lane).reuse_source;
    for (Shard* shard : {&shard_a_, &shard_b_}) {
        shard->device.bind_to_current_thread();
        switch (source) {
        case ReuseSource::None:
            // A lane only owns its own slot, so a reset must not touch its neighbours.
            zero_lane_state(shard->lane_state_geometry, shard->state_backing.data,
                            static_cast<std::int32_t>(lane), shard->device.stream);
            break;
        case ReuseSource::LiveState:
            // The device state already sits at this boundary: a restored session put it there, or
            // the conversation that just decoded left it exactly at its frontier.
            break;
        case ReuseSource::DeviceSnapshot:
            copy_lane_state(shard->lane_state_geometry, shard->state_backing.data,
                            static_cast<std::int32_t>(lane),
                            shard->state_snapshots[reuse.slot].data, cudaMemcpyDeviceToDevice,
                            shard->device.stream);
            break;
        case ReuseSource::HostCheckpoint:
            // The slot belongs to this lane, so its buffer *is* the lane's compact state image:
            // the lane is expressed by the slot index (begin + lane * slots_per_lane), never by
            // a byte offset into the slot.
            copy_lane_state(shard->lane_state_geometry, shard->state_backing.data,
                            static_cast<std::int32_t>(lane),
                            static_cast<std::byte*>(
                                shard->host_checkpoints[reuse.slot].buffer->data()),
                            cudaMemcpyHostToDevice, shard->device.stream);
            break;
        }
    }
}

void TP2GenerationCore::restore_lane_dflash(std::uint32_t lane, const LaneReuse& reuse,
                                            bool zero_when_none) {
    Shard& shard = shard_a_;
    if (shard.dflash_round == nullptr) { return; }
    shard.device.bind_to_current_thread();
    switch (retention(lane).reuse_source) {
    case ReuseSource::None:
        if (zero_when_none) { shard.dflash_round->zero_context(); }
        return;
    case ReuseSource::LiveState:
        // The live ring already reaches this boundary: a recall restored it, or the conversation
        // that just decoded left it flushed to its frontier at publish.
        return;
    case ReuseSource::DeviceSnapshot: {
        const std::size_t lane_bytes = shard.dflash_round->lane_context_image_bytes();
        shard.dflash_round->copy_context_from_device(
            DeviceSpan{static_cast<std::byte*>(shard.dflash_snapshots[reuse.slot].data) +
                           static_cast<std::size_t>(lane) * lane_bytes,
                       lane_bytes},
            shard.device.stream, static_cast<std::int32_t>(lane));
        return;
    }
    case ReuseSource::HostCheckpoint: {
        const Shard::HostCheckpoint& checkpoint = shard.host_checkpoints[reuse.slot];
        if (checkpoint.dflash_buffer == nullptr ||
            checkpoint.dflash_frontier[lane] != checkpoint.position[lane]) {
            throw std::logic_error(
                "TP-2 DFlash2 checkpoint does not carry the draft context at its frontier");
        }
        // The slot carries this lane's image and not a slice of a multi-lane image: the lane that
        // owns the slot is the one that wrote it, so the image starts at the buffer.
        shard.dflash_round->copy_context_from_host(
            static_cast<const std::byte*>(checkpoint.dflash_buffer->data()), shard.device.stream,
            static_cast<std::int32_t>(lane));
        return;
    }
    }
}

void TP2GenerationCore::publish_lane_prefill(
    std::uint32_t lane, std::uint32_t prompt_tokens, const std::vector<TokenId>& tokens,
    std::span<const MediaSpan> media,
    const models::qwen3_5::PreparedContextCache& cache_hints) {
    RetentionState& lane_state = retention(lane);
    // Publish this lane's lineage. The prompt-end boundary is this lane's own: it becomes a
    // checkpoint in the ring's prompt-end slot, and the catalog entry is this lane's own too.
    lane_state.cached_prompt_tokens.assign(tokens.begin(), tokens.end());
    lane_state.cached_media.assign(media.begin(), media.end());
    // The answer this prefill starts is anchored from here on: everything the walk just did is
    // covered by its grid and tail checkpoints, so the span these anchors fill is the generated one.
    lane_state.next_generated_checkpoint = prompt_tokens + kReuseGeneratedTailStride;
    lane_state.cached_boundaries.fill(0);
    lane_state.cached_state_valid = true;
    session_publish(tokens, prompt_tokens, cache_hints, media, lane);
    if (lane_state.active_session != kNoSession) {
        sessions_[lane_state.active_session].prompt_end = prompt_tokens;
    }
    // The prefill end is a boundary like any other, so it is published into the ring's prompt-end
    // slot: that is where a store reads the prompt-end image and where a returning turn stands
    // without a device plane. It is written before the sweep below, which is what makes it usable.
    snapshot_host_checkpoint(shard_a_, prompt_tokens, HostRing::PromptEnd, lane);
    snapshot_host_checkpoint(shard_b_, prompt_tokens, HostRing::PromptEnd, lane);
    // Only now are this prefill's checkpoints usable: their state is one the prefill reached and
    // their KV prefix is one it wrote.
    publish_lane_checkpoints(lane);
    // A configuration without the ring keeps the device plane that carries the same image; with the
    // ring it is not carved and the checkpoint above is the only copy. execute_walk writes the whole
    // plane at its own prefill end; a batched lane writes only its own slice, which is all a
    // per-lane boundary needs, and the slice stays valid until this lane prefills again.
    for (Shard* shard : {&shard_a_, &shard_b_}) {
        if (shard->state_snapshots[0].data == nullptr) { continue; }
        shard->device.bind_to_current_thread();
        // See snapshot_lane_state in execute_walk: the device-to-device branch copies into its
        // first device pointer, so the plane is the destination and the pool is the source.
        copy_lane_state(shard->lane_state_geometry, shard->state_snapshots[0].data,
                        static_cast<std::int32_t>(lane), shard->state_backing.data,
                        cudaMemcpyDeviceToDevice, shard->device.stream);
    }
    // The masked draft's ring is part of that same boundary, and it rides the checkpoint above when
    // the ring exists: a lane that later accepts this boundary has to get its draft context back
    // too, or the skipped prefix would leave a hole in the ring.
    if (shard_a_.dflash_round != nullptr) { snapshot_dflash_state(shard_a_, 0, lane); }
    lane_state.cached_boundaries[0] = prompt_tokens;
}

void TP2GenerationCore::publish_partial_prefill(std::uint32_t lane, std::uint32_t frontier,
                                               const std::vector<TokenId>& tokens,
                                               const models::qwen3_5::PreparedPromptData& data) {
    RetentionState& lane_state = retention(lane);
    // The chunks that finished wrote KV for tokens the prompt really has and left the GDN state at
    // the end of the last one, so the catalog can name where the prefill stopped and the retry of the
    // same prompt continues from there instead of prefilling its whole history again. This mirrors
    // the walk's own cancel path; a batched lane reaches the same boundary one chunk at a time.
    for (Shard* shard : {&shard_a_, &shard_b_}) {
        if (shard->state_snapshots[0].data == nullptr) { continue; }
        shard->device.bind_to_current_thread();
        // See snapshot_lane_state in execute_walk: the device-to-device branch copies into its first
        // device pointer, so the plane is the destination and the pool is the source.
        copy_lane_state(shard->lane_state_geometry, shard->state_snapshots[0].data,
                        static_cast<std::int32_t>(lane), shard->state_backing.data,
                        cudaMemcpyDeviceToDevice, shard->device.stream);
    }
    if (shard_a_.dflash_round != nullptr) { snapshot_dflash_state(shard_a_, 0, lane); }
    lane_state.cached_boundaries[0] = frontier;
    // A cancelled prefill reached no rewind boundary, so every rewind slot this lane still carried
    // is dropped: the slot above is where it stopped.
    for (std::size_t slot = 1; slot < kReuseSnapshotCount; ++slot) {
        lane_state.cached_boundaries[slot] = 0;
    }
    lane_state.cached_state_valid = true;
    // The truncated prompt end is a boundary like a completed one, so it goes into the ring's
    // prompt-end slot and the sweep makes it usable: that is what lets the retry of the same prompt
    // stand on it.
    snapshot_host_checkpoint(shard_a_, frontier, HostRing::PromptEnd, lane);
    snapshot_host_checkpoint(shard_b_, frontier, HostRing::PromptEnd, lane);
    publish_lane_checkpoints(lane);
    session_publish(
        std::vector<TokenId>(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(frontier)),
        frontier, data.context_cache, collect_media_spans(data, frontier), lane);
    if (lane_state.active_session != kNoSession) {
        sessions_[lane_state.active_session].prompt_end = frontier;
    }
}

void TP2GenerationCore::invalidate_lane_prefill(std::uint32_t lane) {
    RetentionState& lane_state = retention(lane);
    // A torn prefill leaves this lane's slot somewhere no boundary names, so retire the lineage and
    // the catalog claim with it: the next request must not stand on a state this one left half
    // written.
    lane_state.cached_state_valid = false;
    lane_state.cached_boundaries.fill(0);
    lane_state.live_state_valid = false;
    session_invalidate_active(lane);
}

namespace {

// P2.1: one lane's tool-call grammar inside a batched round. The serial walk keeps a single
// constraint for its single lane; a batch keeps one per admitted lane, each fed from its own lane's
// published text, so a round can mix grammar-constrained and unconstrained lanes.
struct LaneGrammar {
    std::shared_ptr<qwen::frontend::ToolCallConstraint> constraint;
    std::size_t fed = 0;
};

// Feeds a lane's newly published text to its constraint, the way the serial walk's
// constraint_advance() does.
void grammar_advance(LaneGrammar& grammar, const TP2GenerationCore::Request& request) {
    if (grammar.constraint == nullptr) {
        return;
    }
    const std::string_view raw = request.output.raw_content_text();
    if (raw.size() > grammar.fed) {
        grammar.constraint->feed(raw.substr(grammar.fed));
        grammar.fed = raw.size();
    }
}

// True while this lane's tokens must obey its declared-name grammar: the request carries a tool
// contract and the lane is not inside a reasoning block (the serial walk's constraint_live()).
bool grammar_live(const LaneGrammar& grammar, const TP2GenerationCore::Request& request) {
    return grammar.constraint != nullptr && !request.output.in_reasoning();
}

}  // namespace

// P2.2: one request that arrived while a batch was running. The caller holds `execution_mutex_`
// (the driver takes it around every executor), so the lock order execution_mutex_ ->
// lane_queue_mutex_ holds; the queue itself is only ever touched under its own lock.
std::shared_ptr<TP2GenerationCore::PendingRequest> TP2GenerationCore::try_pop_lane_queue() {
    for (;;) {
        std::shared_ptr<PendingRequest> pending;
        {
            std::unique_lock<std::mutex> queue(lane_queue_mutex_);
            if (lane_queue_.empty()) { return nullptr; }
            pending = std::move(lane_queue_.front());
            lane_queue_.pop_front();
            // Same hand-off as the driver's own batch formation: the member belongs to the driver
            // from here, so the submitter stops enforcing its deadline and its cancellation.
            pending->admitted = true;
            if (!queued_deadline_passed(pending->request->pending_deadline) &&
                !pending->cancellation.requested()) {
                return pending;
            }
        }
        // Neither an expired deadline nor a gone client may take a lane, and the deadline outranks
        // the cancellation, as it does in the single-GPU admission.
        if (queued_deadline_passed(pending->request->pending_deadline)) {
            drop_expired_lane(*pending);
        } else {
            drop_cancelled_lane(*pending);
        }
    }
}

// S1 (tp2-plan-kv-sharing.md): one request's share of the shared KV pool. The request gets
// exactly the pages its own prompt plus output budget need - rounded up to the page grid, plus the
// route's write margin - on both shards and, when the MTP layer is live, on shard 0's own cache.
// The reservation is all-or-nothing and happens before the request is admitted, so a pool that
// cannot cover it right now leaves nothing behind and the caller puts the request back at the head
// of the queue. The admission ceiling already excludes the margin, so a request that fits the policy
// always fits an empty pool: the wait is always for another lane to retire, never a deadlock.
namespace {

// Env-gated KV-pool trace: what each request reserves versus what it uses, the pool's own page
// counters, and the run count and wall time of every session store/recall. Off by default; it exists
// to quantify internal fragmentation and the cost of the S1 host-slab round trip.
bool kv_trace_enabled() {
    const char* env = std::getenv("NINFER_TP2_KV_TRACE");
    return env != nullptr && env[0] == '1';
}

}  // namespace

std::uint32_t TP2GenerationCore::lane_kv_pages(std::uint32_t need_tokens) const noexcept {
    return pages_for_tokens(need_tokens + lane_kv_margin());
}

bool TP2GenerationCore::reserve_lane_kv(std::uint32_t lane, std::uint32_t need_tokens) {
    if (lanes_ <= 1U) { return true; }
    const auto index = static_cast<std::size_t>(lane);
    // A round that threw can leave a lane's pages leased with no request owning them. Take them back
    // before asking for the new request's share, so a failed batch cannot starve the pool.
    shard_a_.kv_lane_pages[index].clear();
    shard_a_.kv_lane_handles[index].clear();
    shard_b_.kv_lane_pages[index].clear();
    shard_b_.kv_lane_handles[index].clear();
    if (mtp_enabled_) {
        shard_a_.mtp_lane_pages[index].clear();
        shard_a_.mtp_lane_handles[index].clear();
    }
    const std::uint32_t pages = lane_kv_pages(need_tokens);
    // The window this lane gets and the pages materialized below come from the same expression, so an
    // admitted request always fits: lane_kv_pages rounds the need up, lane_kv_window subtracts the
    // margin back out, and the published pages cover the whole window plus the margin. What is *not*
    // structural is a pool handing back exactly the pages it reserved, so each materialization below
    // checks its lease count instead of re-deriving that arithmetic.
    Shard* shards[2] = {&shard_a_, &shard_b_};
    // Reserve on every pool first: a pool that cannot cover the request releases what the others
    // already reserved through the reservation destructors, so nothing is half-admitted.
    std::optional<DeviceKVPageReservation> text[2];
    for (std::size_t s = 0; s < 2; ++s) {
        text[s] = shards[s]->decoder->text_kv.page_pool().reserve(pages);
        if (!text[s].has_value()) { return false; }
    }
    std::optional<DeviceKVPageReservation> mtp;
    if (mtp_enabled_) {
        mtp = shard_a_.decoder->mtp_cache()->page_pool().reserve(pages);
        if (!mtp.has_value()) { return false; }
    }
    for (std::size_t s = 0; s < 2; ++s) {
        Shard& shard   = *shards[s];
        auto&  pool    = shard.decoder->text_kv.page_pool();
        auto&  leases  = shard.kv_lane_pages[index];
        auto&  handles = shard.kv_lane_handles[index];
        leases.clear();
        handles.clear();
        try {
            leases.reserve(pages);
            pool.materialize(*text[s], pages, leases);
            if (leases.size() != pages) {
                throw std::logic_error("TP-2 KV pool materialized a lane with the wrong page count");
            }
            handles.reserve(leases.size());
            for (const DeviceKVPageLease& lease : leases) { handles.push_back(lease.handle()); }
        } catch (...) {
            // A half-materialized lane would keep leases that no request owns and that only the
            // request's own release can return, so take them back before the failure unwinds.
            leases.clear();
            handles.clear();
            throw;
        }
        // The lane's row is republished over this request's own pages. Nothing has to be cleared
        // first: the window above never reaches past them, and the device arena is not zeroed, so a
        // block-table entry the route cannot reach is never read.
        shard.decoder->text_kv.execution_tables().publish(shard.kv_rows[index].handle(), 0, handles,
                                                          shard.device.stream);
    }
    if (mtp.has_value()) {
        Shard& shard   = shard_a_;
        auto*  cache   = shard.decoder->mtp_cache();
        auto&  pool    = cache->page_pool();
        auto&  leases  = shard.mtp_lane_pages[index];
        auto&  handles = shard.mtp_lane_handles[index];
        leases.clear();
        handles.clear();
        try {
            leases.reserve(pages);
            pool.materialize(*mtp, pages, leases);
            if (leases.size() != pages) {
                throw std::logic_error("TP-2 MTP KV pool materialized a lane with the wrong page count");
            }
            handles.reserve(leases.size());
            for (const DeviceKVPageLease& lease : leases) { handles.push_back(lease.handle()); }
        } catch (...) {
            leases.clear();
            handles.clear();
            throw;
        }
        cache->execution_tables().publish(shard.mtp_rows[index].handle(), 0, handles,
                                          shard.device.stream);
    }
    if (kv_trace_enabled()) {
        const auto& pool_a = shard_a_.decoder->text_kv.page_pool();
        const auto& pool_b = shard_b_.decoder->text_kv.page_pool();
        std::fprintf(stderr,
                     "[tp2-kv] reserve lane=%u pages=%u need=%u margin=%u runs=%u/%u "
                     "capacity=%u/%u allocated=%u/%u reserved=%u/%u available=%u/%u\n",
                     lane, pages, need_tokens, lane_kv_margin(),
                     pool_a.contiguous_run_count(shard_a_.kv_lane_handles[index]),
                     pool_b.contiguous_run_count(shard_b_.kv_lane_handles[index]),
                     pool_a.capacity_pages(), pool_b.capacity_pages(), pool_a.allocated_pages(),
                     pool_b.allocated_pages(), pool_a.reserved_pages(), pool_b.reserved_pages(),
                     pool_a.available_pages(), pool_b.available_pages());
    }
    return true;
}

// The inverse of reserve_lane_kv: hand the lane's pages back to the shared pool and drop every claim
// that its device KV still describes a conversation. The pages may be handed to another lane the
// moment they are free, so retention state that still named them would make the next request
// admitted onto this lane skip its prefill and read somebody else's tokens.
void TP2GenerationCore::release_lane_kv(std::uint32_t lane) noexcept {
    if (lanes_ <= 1U) { return; }
    const auto index = static_cast<std::size_t>(lane);
    Shard* shards[2] = {&shard_a_, &shard_b_};
    for (Shard* shard : shards) {
        if (index < shard->kv_lane_pages.size()) { shard->kv_lane_pages[index].clear(); }
        if (index < shard->kv_lane_handles.size()) { shard->kv_lane_handles[index].clear(); }
    }
    if (index < shard_a_.mtp_lane_pages.size()) { shard_a_.mtp_lane_pages[index].clear(); }
    if (index < shard_a_.mtp_lane_handles.size()) { shard_a_.mtp_lane_handles[index].clear(); }
    if (kv_trace_enabled()) {
        const auto& pool_a = shard_a_.decoder->text_kv.page_pool();
        const auto& pool_b = shard_b_.decoder->text_kv.page_pool();
        std::fprintf(stderr,
                     "[tp2-kv] release lane=%u capacity=%u/%u allocated=%u/%u reserved=%u/%u "
                     "available=%u/%u\n",
                     lane, pool_a.capacity_pages(), pool_b.capacity_pages(),
                     pool_a.allocated_pages(), pool_b.allocated_pages(), pool_a.reserved_pages(),
                     pool_b.reserved_pages(), pool_a.available_pages(), pool_b.available_pages());
    }
    RetentionState& lane_state = retention(lane);
    lane_state.cached_prompt_tokens.clear();
    lane_state.cached_media.clear();
    lane_state.cached_boundaries.fill(0);
    lane_state.cached_state_valid      = false;
    lane_state.live_state_valid        = false;
    lane_state.reuse_source            = ReuseSource::None;
    lane_state.anchor_session          = kNoSession;
    lane_state.block_anchor_position   = 0;
    lane_state.block_anchor_prefill_id = 0;
    invalidate_host_checkpoints(lane);
    for (SessionEntry& entry : sessions_) {
        if (entry.device_lane == static_cast<std::int32_t>(lane)) { entry.device_lane = -1; }
    }
    // The entry itself stays in the catalog: retire_lane_session() has already copied it into its
    // host slabs when the budget allowed, and a host-resident entry is a recall candidate again.
    lane_state.active_session = kNoSession;
}

// A lane's pages are about to go back to the shared pool, so the conversation its device KV holds
// has to survive somewhere else or the next request admitted onto this lane would extend a prefix
// that is no longer there. Copy it into its host slabs while the pages are still mapped; a
// conversation the host budget cannot keep simply loses the newest part of its reuse, exactly as a
// recall that cannot store the outgoing session does.
void TP2GenerationCore::retire_lane_session(std::uint32_t lane) {
    if (lanes_ <= 1U || session_capacity_ == 0) { return; }
    if (retention(lane).active_session == kNoSession) { return; }
    if (session_store_active(lane)) { return; }
    while (session_evict_one()) {
        if (session_store_active(lane)) { return; }
    }
}

// The admission could not serve this request right now, so it keeps the place it had in the FIFO
// order instead of losing it to the requests that arrived after it. The driver holds
// `execution_mutex_` around every executor, which is the order execution_mutex_ -> lane_queue_mutex_.
void TP2GenerationCore::requeue_lane_front(std::shared_ptr<PendingRequest> pending) {
    std::unique_lock<std::mutex> queue(lane_queue_mutex_);
    // Back in the submitter's hands: it enforces the deadline and the cancellation again, and this
    // notify is what wakes it out of the admission wait it was parked in.
    pending->admitted = false;
    lane_queue_.push_front(std::move(pending));
    lane_queue_cv_.notify_all();
}

// The positions one request's own pages have to cover: its prompt plus the output budget it was
// admitted with (`budget.remaining()` is that budget at admission and only shrinks afterwards). The
// last sampled token is never forwarded, so its slot is never read: the reservation counts one
// position less, exactly as the single-GPU plan's `reserved_context_tokens` does. Counting the whole
// budget would ask for one page more than the pool holds whenever the budget was clamped to
// `ceiling - prompt + 1`, and no lane could ever be admitted for such a request.
std::uint32_t TP2GenerationCore::lane_need_tokens(const PendingRequest& pending) const noexcept {
    const std::uint32_t budget = pending.request->budget.remaining();
    return pending.request->summary.prompt_tokens + (budget == 0U ? 0U : budget - 1U);
}

// ---------------------------------------------------------------------------------------------
// Batched plain walk (tp2-plan-concurrency.md P1.4c)
// ---------------------------------------------------------------------------------------------
//
// One round of this walk runs a single batched decode window for every lane still live. That shared
// forward is where the throughput comes from: a single-lane step is weight-stream bound, so two
// lanes cost one weight pass instead of two.
//
// Reachability: the driver routes a batch here on the plain route only; the speculative routes run
// execute_spec_batch instead. Every admitted member is batched, including one that carries a tool
// grammar (P2.1 masks each lane's own columns) or media.
//
// Invariants this function owns:
//  - Lane "slot" is both the KV execution row and the GDN state slot that lane owns, exactly as
//    the serial walk binds them. A batched lane is structurally a single-lane plain request whose
//    prefix reuse is scanned on that lane alone: prefill from the accepted boundary, first-token
//    sample, then one token per shared round.
//  - Retention is per lane (P2.3). Each lane scans its own lineage, restores its own GDN slot and
//    publishes its own catalog entry, checkpoint ring slots and plane 0 slice. No lane ever reads
//    another lane's state, and the pool is never wiped as a whole.
//  - D4: any throw leaves the driver to mark the whole batch failed. A lane that had already
//    finished is deliberately discarded with the rest.
//  - The arena watermark below is fixed by the constructor's lane count rather than by this batch
//    (P2.2b), so the captured batch decode step finds the same layout on every replay.
void TP2GenerationCore::execute_plain_batch(
    std::vector<std::shared_ptr<PendingRequest>>& batch) {
    DeviceArena& ws_a = *shard_a_.workspace;
    DeviceArena& ws_b = *shard_b_.workspace;
    auto& ctx_a       = *shard_a_.context;
    auto& ctx_b       = *shard_b_.context;
    shard_a_.device.bind_to_current_thread();

    const std::int32_t vocab =
        qwen::execution::dimension(shard_a_.model->config().text.vocab_size);
    const std::int32_t public_tokens =
        static_cast<std::int32_t>(shard_a_.model->resources().public_token_count);
    const std::uint32_t prefill_chunk = prefill_chunk_width(shard_a_.model->config().text);
    // Prefix-reuse trace, the same switch the single-lane walk reads (NINFER_TP2_REUSE_TRACE).
    const bool reuse_trace = [] {
        const char* env = std::getenv("NINFER_TP2_REUSE_TRACE");
        return env != nullptr && env[0] == '1';
    }();
    // The tail anchors cover the last few chunk ends of a lane's prefill. Their count is bounded by
    // the tail sub-ring, so a narrow chunk cannot flood the ring with anchors that all sit within one
    // chunk of the prompt end.
    const std::uint32_t tail_span =
        std::min<std::uint32_t>(kReuseTailWindow,
                                prefill_chunk * std::max(1U, host_checkpoint_tail_slots_));

    Tp2RoundTiming& timing = tp2_timing();
    timing.init();
    timing.reset();

    // One entry per admitted lane. The slot is the batch position, which is simultaneously the KV
    // execution row and the GDN state slot handed to the forwards.
    struct LaneState {
        PendingRequest* pending     = nullptr;
        std::uint32_t slot          = 0;
        GenerationResult result;
        std::int32_t current        = 0;
        std::uint32_t position      = 0;
        std::uint32_t prompt_tokens = 0;
        bool have_first             = false;
        bool finished               = false;
        // Set once this lane's prefill published its lineage; only then does the device state at
        // `position` describe a conversation a later turn may extend.
        bool prefilled              = false;
        Clock::time_point begin;
        Clock::time_point first_token;
        // S1: the context window the pages this lane reserved give it. The executor sets it after
        // admit_lane returns - admit_lane resets the whole state struct on entry.
        std::uint32_t kv_window = 0;
    };
    // P2.2: the lane arrays cover the whole lane capacity rather than the batch that happened to
    // arrive, because a lane that retires frees its slot for a request still waiting in the queue.
    const std::size_t lane_capacity = lanes_;
    std::vector<LaneState> lanes(lane_capacity);
    for (std::size_t index = 0; index < batch.size(); ++index) {
        lanes[index].pending     = batch[index].get();
        lanes[index].slot        = static_cast<std::uint32_t>(index);
        lanes[index].begin       = Clock::now();
        lanes[index].first_token = lanes[index].begin;
    }

    // Batch-resident state lives below every round watermark, so a round scope can never move it.
    auto lifetime_a = ws_a.scope();
    auto lifetime_b = ws_b.scope();

    // Filled by admit_lane for every lane it admits, including the ones that arrive later.
    std::vector<ops::SamplingConfig> configs(lane_capacity);

    // P2.1: one declared-name grammar per lane, built from that lane's own prompt contract before any
    // round runs, plus the mask staging the rounds reuse. A lane without a tool contract keeps a null
    // constraint and is never masked. The logits domain is the packed embedding row count, which the
    // artifact contract allows to be wider than the tokenizer public domain the constraint table
    // covers; build_mask() excludes the rows in between.
    const std::size_t logits_domain = static_cast<std::size_t>(vocab);
    // Built by admit_lane, one lane at a time, so a lane admitted mid-batch gets its own constraint.
    std::vector<LaneGrammar> grammar(lane_capacity);
    bool any_grammar = false;
    std::vector<std::uint8_t> tool_mask_one(static_cast<std::size_t>(vocab), std::uint8_t{1});
    std::vector<std::uint8_t> tool_mask_host(static_cast<std::size_t>(vocab) * lane_capacity,
                                             std::uint8_t{1});

    // A request with a configured penalty accumulates its committed tokens in device memory, one
    // slice per lane so the shared sample reads per-lane counts. The array is sized for the widest
    // batch this route can form rather than for this batch: the round watermark, and with it every
    // captured batch graph, has to be the same for every request. A request with no configured
    // penalty simply never reads its slice, and only the live span is cleared.
    // admit_lane points each admitted lane's config at its own slice of this array.
    std::int32_t* counts_base = nullptr;
    {
        Tensor counts = ws_a.alloc(
            DType::I32, {static_cast<std::int32_t>(public_tokens * static_cast<std::int32_t>(lanes_))});
        shard_a_.device.bind_to_current_thread();
        CUDA_CHECK(cudaMemsetAsync(
            counts.data, 0,
            sizeof(std::int32_t) * static_cast<std::size_t>(public_tokens) * lane_capacity,
            shard_a_.device.stream));
        counts_base = static_cast<std::int32_t*>(counts.data);
    }

    // Device copy of the per-lane sampler configs. Staging this once lets the prefill samples read
    // their own lane's entry directly; the decode rounds still build a compacted copy per round,
    // because a lane that left the batch no longer occupies its original column. Like the counts
    // array, the reservation covers the whole lane count so the watermark does not move with the
    // batch, and only the live lanes are filled.
    DeviceSpan configs_dev = ws_a.alloc_bytes(sizeof(ops::SamplingConfig) * lanes_, 256);
    shard_a_.device.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(configs_dev.data, configs.data(),
                               sizeof(ops::SamplingConfig) * lane_capacity,
                               cudaMemcpyHostToDevice, shard_a_.device.stream));

    // Pinned host staging for the four per-lane vectors the batched window copies, plus the
    // per-round logical positions. This is the core member, not a round local: a captured batch step
    // reads the four through memcpy nodes, so their addresses are baked into the graph.
    std::int32_t* host_tokens    = batch_lane_host_section(0);
    std::int32_t* host_positions = batch_lane_host_section(1);
    std::int32_t* host_kv_rows   = batch_lane_host_section(2);
    std::int32_t* host_slots     = batch_lane_host_section(3);
    std::int32_t* host_logical   = batch_lane_host_section(4);

    // Per-lane preview publication, mirroring the single-lane lambda: a terminal call first names
    // the reason the walk stopped, then the committed preview is appended to this lane's result and
    // streamed to its own sink.
    auto publish_preview = [](LaneState& lane, bool terminal) {
        Request& request = *lane.pending->request;
        if (terminal) {
            (void)request.output.preview_terminal(request.budget.limit_reason());
        }
        auto published = request.output.commit_preview();
        for (const auto& delta : published) {
            if (delta.channel == OutputChannel::Reasoning) {
                lane.result.reasoning += delta.text;
            } else {
                lane.result.content += delta.text;
            }
            if (lane.pending->sink != nullptr) { lane.pending->sink->publish(delta); }
        }
    };

    // Fills a lane's result from everything its walk accumulated, in the same order the single-lane
    // walk does at its exit. A lane that never produced a first token reports an empty decode.
    // Publish a retired lane's lineage before its result takes the generated tokens away: the pools
    // hold prompt plus committed output and the live GDN state sits at the frontier the lane's last
    // committed round left - one short of the history, because the last sampled token is never
    // forwarded. execute_walk publishes the same pair at its exit.
    auto finalize = [this](LaneState& lane) {
        Request& request = *lane.pending->request;
        if (lane.prefilled) {
            auto& data = qwen::PreparedPromptAccess::mutable_view(request.prompt);
            std::vector<TokenId> history(data.token_ids.begin(), data.token_ids.end());
            history.insert(history.end(), request.generated.begin(), request.generated.end());
            session_publish(history, static_cast<std::uint32_t>(lane.position), data.context_cache,
                            collect_media_spans(data, data.token_ids.size()), lane.slot);
            // The anchors this answer wrote sit past a state the walk committed, so they become
            // usable on the same terms as the prefill's: once the walk stopped naming its frontier.
            publish_lane_checkpoints(lane.slot);
        }
        const Clock::time_point done = Clock::now();
        lane.result.generated_token_ids = std::move(request.generated);
        lane.result.tool_calls          = request.output.take_tool_calls();
        lane.result.tool_call_parse     = request.output.tool_call_parse_diagnostics();
        lane.result.reasoning_tokens    = request.output.reasoning_tokens();
        lane.result.matched_stop_string = request.output.matched_stop_string();
        lane.result.thinking            = request.output.thinking_stats();
        const double total = std::chrono::duration<double>(done - lane.begin).count();
        // Both reported spans start where the core accepted the request, matching the single-device
        // route: the serve layer adds prepare_seconds to the first-token span and subtracts it from
        // the total, so a span that had already excluded preparation read back as zero whenever the
        // generation was shorter than its own preparation. `prompt_wall_seconds` and
        // `prefill_seconds` deliberately stay admission-relative: they define the prefill rate.
        const double queue_wait = lane.result.engine_timing.queue_wait_seconds;
        if (lane.have_first) {
            const double decode =
                std::chrono::duration<double>(done - lane.first_token).count();
            lane.result.timings.decode_seconds          = decode;
            lane.result.timings.generation_wall_seconds = decode;
            lane.result.timings.first_token_seconds =
                lane.result.timings.prepare_seconds + lane.result.timings.prompt_wall_seconds +
                queue_wait;
            lane.result.timings.total_seconds =
                lane.result.timings.prepare_seconds + total + queue_wait;
        } else {
            lane.result.timings.decode_seconds          = 0.0;
            lane.result.timings.generation_wall_seconds = 0.0;
            lane.result.timings.prompt_wall_seconds     = total;
            lane.result.timings.first_token_seconds =
                lane.result.timings.prepare_seconds + total + queue_wait;
            lane.result.timings.total_seconds =
                lane.result.timings.prepare_seconds + total + queue_wait;
        }
        lane.pending->result = std::move(lane.result);
        // S1: the pages this request held go back to the shared pool as it retires, so the next
        // request admitted onto this lane can use them. The session is copied into its host slabs
        // first - that copy reads the lane's device pages - because once they are free another lane
        // may overwrite them, and retention state that still named them would make the next request
        // skip its prefill and read somebody else's tokens.
        retire_lane_session(lane.slot);
        release_lane_kv(lane.slot);
        // P0.2: retire the member here rather than at the end of the batch. This lane is done, and
        // its submitter must be free to return while the other lanes keep running.
        publish_lane(*lane.pending);
    };

    // Each lane's GDN slot is brought to its own starting state in the loop below: a lane that
    // reuses a boundary is copied from its image, and a lane that does not is zeroed. The pool as a
    // whole is never wiped, because a lane continuing a conversation holds the state its previous
    // round left, and that state is exactly what its reuse boundary names (P2.3 Stage 1).

    // Prefill is serial per lane: each lane owns a private KV row and GDN slot, so the only shared
    // resource is the workspace arena, and the chunk scopes below hand it back between lanes.
    std::vector<std::size_t> active;
    active.reserve(lane_capacity);

    // The decode round, callable from the admission path as well as from the loop below. A lane
    // that is already holding a token keeps publishing while another lane prefills: the prefill
    // chunk loop pumps one round between two of its own chunks, so a newcomer's prompt no longer
    // stalls every lane beside it for the prompt's whole duration. Exactly one lane ever prefills
    // (admit_lane runs them one at a time), so the serial prefill order, every lane's TTFT and the
    // single startup vision session all stay as they were; only the lanes' token stream changes.
    // The round never retires or admits a lane, so it can never re-enter admit_lane.
    auto pump_decode_round = [&]() {
        if (active.empty()) { return; }
        const std::int32_t columns = static_cast<std::int32_t>(active.size());
        std::uint32_t max_position = 0;
        for (std::size_t column = 0; column < active.size(); ++column) {
            const LaneState& lane  = lanes[active[column]];
            host_tokens[column]    = lane.current;
            host_positions[column] = static_cast<std::int32_t>(lane.position);
            host_kv_rows[column]   = static_cast<std::int32_t>(lane.slot);
            host_slots[column]     = static_cast<std::int32_t>(lane.slot);
            host_logical[column]   = static_cast<std::int32_t>(lane.position) + 1;
            max_position           = std::max(max_position, lane.position);
        }
        position_arena(ws_a, shard_a_.round_base, shard_a_.round_base);
        position_arena(ws_b, shard_b_.round_base, shard_b_.round_base);
        auto scope_a = ws_a.scope();
        auto scope_b = ws_b.scope();
        shard_a_.device.bind_to_current_thread();
        // The configs and the logical positions are restaged per round because a lane that left the
        // batch no longer occupies its original column.
        DeviceSpan round_configs =
            ws_a.alloc_bytes(sizeof(ops::SamplingConfig) * static_cast<std::size_t>(columns), 256);
        std::vector<ops::SamplingConfig> round_host(static_cast<std::size_t>(columns));
        for (std::size_t column = 0; column < active.size(); ++column) {
            round_host[column] = configs[lanes[active[column]].slot];
        }
        CUDA_CHECK(cudaMemcpyAsync(round_configs.data, round_host.data(),
                                   sizeof(ops::SamplingConfig) * static_cast<std::size_t>(columns),
                                   cudaMemcpyHostToDevice, shard_a_.device.stream));
        Tensor logical_positions = ws_a.alloc(DType::I32, {columns});
        CUDA_CHECK(cudaMemcpyAsync(logical_positions.data, host_logical,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(columns),
                                   cudaMemcpyHostToDevice, shard_a_.device.stream));
        Tensor logits = ws_a.alloc(DType::BF16, {vocab, columns});
        // The envelope is a launch/workspace promise over the batch maximum, not a mask: every
        // column's own position still bounds its visible keys inside the kernel.
        const ops::CausalAttentionExecutionEnvelope envelope{1, max_position + 1};
        // Phase marks, matching the walk and the spec batch: 0 round start, 1 after the decode
        // forward, 2 after the grammar mask, 3 after the sample, 4 after the token readback. The
        // plain route has no proposal chain and no fold, so mtp and fold stay zero. `rounds` counts
        // committed columns (as it does on the spec route), so every accumulated phase is the
        // per-lane share of a round and avg_round is directly comparable with verify.
        const Clock::time_point round_start = Clock::now();
        timing.record(0, shard_a_.device.stream);
        run_plain_decode_step_batch(host_tokens, host_positions, host_kv_rows, host_slots, columns,
                                    envelope, logits);
        timing.record(1, shard_a_.device.stream);
        // P2.1: mask each live lane's own column with that lane's own declared-name grammar, after
        // the decode forward that produced the logits and before the sample reads them. Column c of
        // the round's logits belongs to lane active[c], so block c is that lane's mask.
        if (any_grammar) {
            std::fill(tool_mask_host.begin(), tool_mask_host.end(), std::uint8_t{1});
            std::int32_t masked_columns = 0;
            for (std::int32_t column = 0; column < columns; ++column) {
                const std::size_t lane_index = active[static_cast<std::size_t>(column)];
                LaneGrammar& lane_grammar    = grammar[lane_index];
                const Request& lane_request  = *lanes[lane_index].pending->request;
                if (!grammar_live(lane_grammar, lane_request)) {
                    continue;
                }
                grammar_advance(lane_grammar, lane_request);
                if (lane_grammar.constraint->build_mask(logits_domain, tool_mask_one)) {
                    const std::size_t base =
                        static_cast<std::size_t>(column) * static_cast<std::size_t>(vocab);
                    std::copy(tool_mask_one.begin(), tool_mask_one.end(),
                              tool_mask_host.begin() + static_cast<std::ptrdiff_t>(base));
                    ++masked_columns;
                }
            }
            if (masked_columns > 0) {
                Tensor tool_mask_dev = ws_a.alloc(DType::U8, {vocab, columns});
                shard_a_.device.bind_to_current_thread();
                CUDA_CHECK(cudaMemcpyAsync(tool_mask_dev.data, tool_mask_host.data(),
                                           sizeof(std::uint8_t) * static_cast<std::size_t>(vocab) *
                                               static_cast<std::size_t>(columns),
                                           cudaMemcpyHostToDevice, shard_a_.device.stream));
                ops::apply_token_mask(logits, tool_mask_dev, shard_a_.device.stream);
                if (lane_trace_enabled()) {
                    std::fprintf(stderr, "[tp2-lane] grammar masked %d of %d columns\n",
                                 static_cast<int>(masked_columns), static_cast<int>(columns));
                }
            }
        }
        timing.record(2, shard_a_.device.stream);
        Tensor sampled = ws_a.alloc(DType::I32, {columns});
        shard_a_.device.bind_to_current_thread();
        ops::sample(logits, sampled, public_tokens,
                    static_cast<const ops::SamplingConfig*>(round_configs.data), logical_positions,
                    ops::kSamplePurposeDecode, ws_a, shard_a_.device.stream);
        timing.record(3, shard_a_.device.stream);
        std::vector<std::int32_t> next(static_cast<std::size_t>(columns), 0);
        CUDA_CHECK(cudaMemcpyAsync(next.data(), sampled.data,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(columns),
                                   cudaMemcpyDeviceToHost, shard_a_.device.stream));
        timing.record(4, shard_a_.device.stream);
        // The wait itself is timed on the host: the phase marks cover the GPU work, and this is the
        // interval a scheduler could hide behind the next round's host work.
        const Clock::time_point sync_start = Clock::now();
        CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
        timing.close_round_plain(
            std::chrono::duration<double, std::milli>(Clock::now() - sync_start).count());
        abort_if_ar_stalled();

        for (std::size_t column = 0; column < active.size(); ++column) {
            LaneState& lane  = lanes[active[column]];
            Request& request = *lane.pending->request;
            std::vector<TokenId> step{static_cast<TokenId>(next[column])};
            const std::uint32_t remaining = request.budget.remaining();
            if (step.size() > remaining) { step.resize(remaining); }
            const OutputDecision decision =
                request.output.preview_model(step, remaining, request.budget.limit_reason());
            if (decision.accepted_tokens == 0 || decision.accepted_tokens > step.size()) {
                throw std::logic_error("TP-2 output policy returned an invalid licensed prefix");
            }
            request.generated.insert(request.generated.end(), step.begin(),
                                     step.begin() + decision.accepted_tokens);
            request.budget.commit(decision.accepted_tokens);
            committed_decode_tokens_ += decision.accepted_tokens;
            ++decode_rounds_;
            publish_preview(lane, false);
            lane.current = static_cast<std::int32_t>(step.back());
            ++lane.position;
            if (decision.finished()) {
                lane.result.finish_reason = decision.finish_reason;
                lane.finished             = true;
                finalize(lane);
            }
            timing.committed += decision.accepted_tokens;
            ++timing.rounds;
        }
        timing.round_ms +=
            std::chrono::duration<double, std::milli>(Clock::now() - round_start).count();
        std::vector<std::size_t> still;
        still.reserve(active.size());
        for (const std::size_t index : active) {
            if (!lanes[index].finished) { still.push_back(index); }
        }
        active.swap(still);
    };

    // Round watermark. Prefill's chunk scopes have handed the arena back, so the decode rounds
    // allocate above this point only and rewind to it every round.
    shard_a_.round_base = ws_a.used();
    shard_b_.round_base = ws_b.used();

    // P2.2: one lane's whole setup as a callable unit, so a slot a retired lane leaves free can take
    // the next queued request at any round boundary. The caller appends the member to `batch` before
    // this runs, so a throw here still reaches the driver's failure propagation and its publish
    // safety net. Returns true when the lane is left decoding.
    auto admit_lane = [&](std::size_t index, PendingRequest& pending) -> bool {
        LaneState& lane = lanes[index];
        lane             = LaneState{};
        lane.pending     = &pending;
        lane.slot        = static_cast<std::uint32_t>(index);
        lane.begin       = Clock::now();
        lane.first_token = lane.begin;
        // The reported TTFT starts at the core's own acceptance, not at this admit: a request that
        // waited in the FIFO paid that time before any compute ran. `prompt_wall_seconds` stays
        // admission-relative because it defines the prefill rate this lane reports.
        lane.result.engine_timing.queue_wait_seconds =
            std::chrono::duration<double>(lane.begin - pending.request->submitted).count();
        // This lane's own sampler config and grammar, built here because a lane may be admitted long
        // after the batch was formed. The device copy of the configs is refreshed entry by entry.
        configs[index] = make_sampling_config(pending.request->sampling);
        if (configs[index].presence_penalty != 0.0F || configs[index].frequency_penalty != 0.0F) {
            configs[index].token_counts =
                counts_base +
                static_cast<std::size_t>(index) * static_cast<std::size_t>(public_tokens);
        }
        grammar[index] = LaneGrammar{};
        {
            const auto& lane_data = qwen::PreparedPromptAccess::view(pending.request->prompt);
            if (lane_data.tool_call_output != nullptr) {
                grammar[index].constraint =
                    frontend_->make_tool_call_constraint(lane_data.tool_call_output);
                if (grammar[index].constraint != nullptr) {
                    if (grammar[index].constraint->vocab_size() > logits_domain) {
                        throw std::logic_error(
                            "TP-2 tool-call constraint vocabulary " +
                            std::to_string(grammar[index].constraint->vocab_size()) +
                            " exceeds the logits domain " + std::to_string(logits_domain));
                    }
                    any_grammar = true;
                }
            }
        }
        shard_a_.device.bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<std::uint8_t*>(configs_dev.data) +
                static_cast<std::size_t>(index) * sizeof(ops::SamplingConfig),
            &configs[index], sizeof(ops::SamplingConfig), cudaMemcpyHostToDevice,
            shard_a_.device.stream));
        Request& request = *lane.pending->request;
        // The batch column is also the KV execution row and the GDN state slot this lane owns, which
        // is the lane index every session_* and retention helper takes (P2.3).
        const std::uint32_t lane_id = lane.slot;
        auto& data = qwen::PreparedPromptAccess::mutable_view(request.prompt);
        lane.prompt_tokens = static_cast<std::uint32_t>(data.token_ids.size());
        lane.result.prompt = request.summary;
        lane.result.timings.prepare_seconds = request.prepare_seconds;
        lane.result.speculative             = SpeculativeStats{};
        active_lane_ = static_cast<std::int32_t>(lane.slot);
        for (Shard* shard : {&shard_a_, &shard_b_}) {
            shard->device.bind_to_current_thread();
            shard->context->set_linear_state_slots(active_lane_, active_lane_);
        }

        // A client that re-rendered the answer this lane generated hands back a prompt whose last
        // turn is the tokens the lineage really produced; adopting it keeps the conversation
        // resident instead of retiring it over a re-tokenisation difference.
        const TurnAdoption adoption =
            adopt_generated_turn(data, lane.prompt_tokens, reuse_trace, lane_id);
        if (adoption.adopted) {
            lane.prompt_tokens               = adoption.prompt_tokens;
            request.summary.prompt_tokens    = lane.prompt_tokens;
            lane.result.prompt.prompt_tokens = lane.prompt_tokens;
        }
        const std::vector<TokenId>& tokens = data.token_ids;
        const std::span<const TokenId> token_ids(tokens.data(), tokens.size());
        lane.position = lane.prompt_tokens;
        // The media identity this prompt carries. Every lane builds its own spans from its own
        // prompt, so a batch mixes media and text lanes without either seeing the other's pictures.
        const std::vector<MediaSpan> prompt_media = collect_media_spans(data, data.token_ids.size());
        // Cross-session recall: a prompt belonging to a conversation another lane holds swaps that
        // conversation onto this lane before the boundary scan reads the lineage.
        session_recall(lane_id, token_ids, prompt_media);

        // Prompt-prefix reuse, the scan execute_walk runs for a single lane, on this lane's own
        // lineage: the host checkpoint ring, plane 0's slice of this lane (written at this lane's
        // own prefill end, below), and the catalog entry. Two lanes in one batch never read each
        // other's state, so the batch route reuses prefixes exactly as the serial walk does.
        const LaneReuse lane_reuse =
            scan_lane_reuse(lane_id, token_ids, prompt_media, lane.prompt_tokens, adoption.divergence,
                            adoption.adopted, reuse_trace);
        const std::uint32_t reuse          = lane_reuse.tokens;
        std::uint32_t next_host_checkpoint = lane_reuse.next_host_checkpoint;
        // The two anchors the serial walk plans, on this lane's own chunk grid: the boundary a later
        // conversation of this family is known to want, and the end of this prompt's own leading
        // instruction block. Without them the batched route offers only the grid and tail
        // checkpoints, which is why a new conversation sharing just the system block re-prefilled it.
        const LaneAnchors anchors = plan_lane_anchors(
            lane_id, token_ids, lane.prompt_tokens, reuse, lane_reuse.shared_prefix, prefill_chunk,
            data.context_cache.leading_instruction_frontier.value_or(0));
        const bool anchor_divergence        = anchors.anchor_divergence;
        const std::uint32_t anchor_position = anchors.anchor_position;
        const bool block_anchor             = anchors.block_anchor;
        const std::uint32_t block_position  = anchors.block_position;

        // Restore this lane's GDN state to the boundary the scan accepted. A reused prefix needs no KV
        // work at all: the pages already hold it.
        restore_lane_gdn(lane_id, lane_reuse);
        // The device state still stands on the boundary this lane inherited, so freeze it before the
        // first chunk overwrites it.
        capture_lane_anchor_from_device(lane_id, reuse, lane_reuse.shared_prefix);

        lane.result.reused_prompt_tokens = reuse;
        lane.result.prefix_reuse_path    = reuse_path(reuse, anchors.block_frontier, lane_id);
        lane.result.reuse_source         = reuse_source_kind(lane_retention_[lane_id].reuse_source);
        lane.result.reuse_reason         = lane_reuse.reuse_reason == ReuseReason::Used
                                               ? ReuseReasonKind::Used
                                               : ReuseReasonKind::NoAnchorAtOrBefore;
        if (lane.pending->sink != nullptr) {
            lane.pending->sink->start(
                GenerationStart{.prompt = request.summary, .reused_prompt_tokens = reuse});
        }

        // This lane's Vision session, on top of the startup plan. Prefill is serial per lane, so the
        // single startup arena is reused sequentially and only one session is ever alive; the plan
        // must outlive the session, which binds it by reference.
        const bool media = data.has_media();
        qwen::execution::VisionPrefillPlan vision_plan;
        std::unique_ptr<qwen::execution::VisionPrefillSession> vision_session =
            open_vision_session(data, reuse, vision_plan);
        bool cancelled                  = lane.pending->cancellation.requested();
        FinishReason first_token_finish = FinishReason::None;
        // Prefill starts at the accepted boundary: the KV pages before it are the restored ones and
        // the GDN state was just brought to exactly that position, so re-forwarding the prefix would
        // both duplicate pages and advance the state twice.
        // The chunk loop's own progress, hoisted out of its scope: the cancel path below publishes
        // this lane's session up to the last chunk that really finished.
        std::uint32_t prefilled = reuse;
        for (std::uint32_t t0 = reuse; !cancelled && t0 < lane.prompt_tokens;) {
            if (lane.pending->cancellation.requested()) {
                cancelled = true;
                break;
            }
            // Interleave one decode round for the lanes that already hold a token, so this lane's
            // prompt does not stall every lane beside it for the prompt's whole duration. The
            // previous chunk's scopes have handed the arena back, so the round starts at the same
            // watermark it does from the main loop.
            ++timing.prefill_chunks;
            if (!active.empty()) {
                ++timing.pump_rounds;
                pump_decode_round();
                // A one-column round republishes the scalar GDN slot pair; restore the binding this
                // lane's prefill set up so the chunk that follows still runs on its own slot.
                for (Shard* shard : {&shard_a_, &shard_b_}) {
                    shard->device.bind_to_current_thread();
                    shard->context->set_linear_state_slots(active_lane_, active_lane_);
                }
            }
            std::uint32_t length = std::min(prefill_chunk, lane.prompt_tokens - t0);
            // A multimodal chunk is capped at the boundary of the item it overlaps, so the encoder
            // hands out one item at a time and the scatter below stays one contiguous column range
            // of it; the next chunk re-enters the same item.
            qwen::execution::VisionChunk vision_chunk;
            if (vision_session) {
                vision_chunk = vision_session->prepare_chunk(t0, length);
                length       = static_cast<std::uint32_t>(vision_chunk.length);
            }
            // An anchor is a state this prefill must freeze exactly, so the chunk that would step
            // over one ends on it - the same clamp the serial walk applies, and for the same reason:
            // the anchor is only ever planned on this prefill's own chunk grid.
            if (anchor_divergence && anchor_position > t0 && anchor_position < t0 + length) {
                length = anchor_position - t0;
            }
            if (block_anchor && block_position > t0 && block_position < t0 + length) {
                length = block_position - t0;
            }
            // The boundary is where this prompt stopped agreeing with the lane's history, so the chunk
            // that would step over it ends on it: an image captured there is only loadable if it sits
            // exactly on that token. One extra short chunk per request is the price.
            if (retention(lane_id).reuse_boundary_target > t0 &&
                retention(lane_id).reuse_boundary_target < t0 + length) {
                length = retention(lane_id).reuse_boundary_target - t0;
            }
            qwen::execution::Tp2VisionChunk media_chunk;
            const qwen::execution::Tp2VisionChunk* media_ptr = nullptr;
            if (media) {
                media_chunk.control       = vision_chunk.control;
                media_chunk.embeddings    = &vision_chunk.embeddings;
                media_chunk.positions     = data.positions.data();
                media_chunk.prompt_tokens = data.token_ids.size();
                media_ptr                 = &media_chunk;
            }
            auto scope_a    = ws_a.scope();
            auto scope_b    = ws_b.scope();
            Tensor logits_a = ws_a.alloc(DType::BF16, {vocab, 1});
            Tensor logits_b = ws_b.alloc(DType::BF16, {vocab, 1});
            shard_a_.device.bind_to_current_thread();
            ctx_a.forward_tp2_prefill(ctx_b, pair_,
                                      std::span<const int>(token_ids.data() + t0, length),
                                      static_cast<std::int32_t>(t0), &logits_a, &logits_b, nullptr,
                                      nullptr, nullptr, qwen::TextPhase::Prefill, media_ptr, nullptr,
                                      active_lane_);
            if (host_checkpoint_stride_ != 0) {
                // One checkpoint per stride, tagged with the frontier this chunk actually reached, so
                // a chunk width that does not divide the stride cannot mislabel a state; plus the
                // dense tail window. The prompt end is skipped either way: the lane publishes it as
                // its frontier into the ring's own prompt-end slot.
                const std::uint32_t frontier = t0 + length;
                if (!retention(lane_id).reuse_boundary_captured &&
                    retention(lane_id).reuse_boundary_target != 0 &&
                    frontier >= retention(lane_id).reuse_boundary_target) {
                    capture_reuse_boundary(lane_id);
                }
                if (frontier >= next_host_checkpoint) {
                    snapshot_host_checkpoint(shard_a_, frontier, HostRing::Grid, lane_id);
                    snapshot_host_checkpoint(shard_b_, frontier, HostRing::Grid, lane_id);
                    next_host_checkpoint =
                        (frontier / host_checkpoint_stride_ + 1U) * host_checkpoint_stride_;
                } else if (frontier != lane.prompt_tokens &&
                           static_cast<std::uint64_t>(frontier) + tail_span > lane.prompt_tokens) {
                    snapshot_host_checkpoint(shard_a_, frontier, HostRing::Tail, lane_id);
                    snapshot_host_checkpoint(shard_b_, frontier, HostRing::Tail, lane_id);
                }
                if (anchor_divergence && frontier == anchor_position) {
                    snapshot_host_checkpoint(shard_a_, frontier, HostRing::Divergence, lane_id);
                    snapshot_host_checkpoint(shard_b_, frontier, HostRing::Divergence, lane_id);
                }
                if (block_anchor && frontier == block_position) {
                    snapshot_host_checkpoint(shard_a_, frontier, HostRing::Block, lane_id);
                    snapshot_host_checkpoint(shard_b_, frontier, HostRing::Block, lane_id);
                    // The id names the prefill that wrote it, so an eviction can tell this
                    // conversation's own block state from a later conversation's rewrite of the slot.
                    retention(lane_id).block_anchor_position   = block_position;
                    retention(lane_id).block_anchor_prefill_id =
                        retention(lane_id).host_checkpoint_live_id;
                }
            }
            if (t0 + length == lane.prompt_tokens) {
                Tensor logical_pos_lane = ws_a.alloc(DType::I32, {1});
                shard_a_.device.bind_to_current_thread();
                ops::set_i32_scalar(logical_pos_lane,
                                    static_cast<std::int32_t>(lane.prompt_tokens),
                                    shard_a_.device.stream);
                // P2.1: the serial walk masks the first token at the same site. The grammar starts in
                // its free-text position, so build_mask() normally reports the empty prefix as
                // unconstrained; keeping the site makes the batched route behave like the serial one
                // if a constrained first position is ever introduced.
                if (grammar_live(grammar[index], request)) {
                    grammar_advance(grammar[index], request);
                    if (grammar[index].constraint->build_mask(logits_domain, tool_mask_one)) {
                        Tensor tool_mask_first = ws_a.alloc(DType::U8, {vocab, 1});
                        shard_a_.device.bind_to_current_thread();
                        CUDA_CHECK(cudaMemcpyAsync(tool_mask_first.data, tool_mask_one.data(),
                                                   tool_mask_one.size(), cudaMemcpyHostToDevice,
                                                   shard_a_.device.stream));
                        ops::apply_token_mask(logits_a, tool_mask_first, shard_a_.device.stream);
                    }
                }
                Tensor sampled_a = ws_a.alloc(DType::I32, {1});
                ops::sample(logits_a, sampled_a, public_tokens,
                            static_cast<const ops::SamplingConfig*>(configs_dev.data) + lane.slot,
                            logical_pos_lane, ops::kSamplePurposePrefill, ws_a,
                            shard_a_.device.stream);
                // The exactness oracle: the prefill's own logits column, so a recall can be compared
                // against a recomputation instead of against the token it happened to sample. Greedy
                // output can hide a small state error in one run and expose it in another; the logit is
                // the quantity that says whether the restored state is the same state.
                if (const char* dump = std::getenv("NINFER_TP2_LOGIT_DUMP");
                    dump != nullptr && dump[0] == '1') {
                    std::vector<std::uint16_t> codes(static_cast<std::size_t>(vocab), 0U);
                    CUDA_CHECK(cudaMemcpyAsync(codes.data(), logits_a.data,
                                               codes.size() * sizeof(std::uint16_t),
                                               cudaMemcpyDeviceToHost, shard_a_.device.stream));
                    CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
                    std::array<float, 8> value{};
                    std::array<std::uint32_t, 8> id{};
                    value.fill(-3.0e38F);
                    for (std::size_t token = 0; token < codes.size(); ++token) {
                        const std::uint32_t wide = static_cast<std::uint32_t>(codes[token]) << 16;
                        float logit              = 0.0F;
                        std::memcpy(&logit, &wide, sizeof(logit));
                        if (logit <= value[7]) { continue; }
                        std::size_t place = 7;
                        while (place > 0 && value[place - 1] < logit) {
                            value[place] = value[place - 1];
                            id[place]    = id[place - 1];
                            --place;
                        }
                        value[place] = logit;
                        id[place]    = static_cast<std::uint32_t>(token);
                    }
                    std::fprintf(stderr, "[tp2-logits] pos=%u top=", t0 + length);
                    for (std::size_t place = 0; place < 8; ++place) {
                        std::fprintf(stderr, "%u:%.4f ", id[place],
                                     static_cast<double>(value[place]));
                    }
                    std::fprintf(stderr, "\n");
                }
                std::int32_t first = 0;
                CUDA_CHECK(cudaMemcpyAsync(&first, sampled_a.data, sizeof(std::int32_t),
                                           cudaMemcpyDeviceToHost, shard_a_.device.stream));
                CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
                abort_if_ar_stalled();
                const TokenId first_token        = static_cast<TokenId>(first);
                const std::uint32_t first_budget = request.budget.remaining();
                if (first_budget == 0) {
                    throw std::logic_error("prefill sampled a token with no output budget left");
                }
                const OutputDecision first_decision = request.output.preview_model(
                    std::span<const TokenId>(&first_token, 1), first_budget,
                    request.budget.limit_reason());
                if (first_decision.accepted_tokens != 1) {
                    throw std::logic_error("output policy rejected the prefill's first token");
                }
                request.generated.push_back(first_token);
                request.budget.commit(1);
                lane.have_first  = true;
                lane.first_token = Clock::now();
                lane.current     = static_cast<std::int32_t>(first_token);
                lane.result.timings.prompt_wall_seconds =
                    std::chrono::duration<double>(lane.first_token - lane.begin).count();
                // The Vision encode is reported separately from the text walk, matching the serial
                // walk and the single-device route; prompt wall time (the response's TTFT) is the
                // whole span either way.
                lane.result.timings.vision_seconds =
                    vision_session ? vision_session->elapsed_seconds() : 0.0;
                lane.result.timings.prefill_seconds =
                    std::max(0.0, lane.result.timings.prompt_wall_seconds -
                                      lane.result.timings.vision_seconds);
                publish_preview(lane, false);
                if (first_decision.finished()) {
                    first_token_finish        = first_decision.finish_reason;
                    lane.result.finish_reason = first_decision.finish_reason;
                }
            }
            t0 += length;
            prefilled = t0;
        }
        if (vision_session) {
            // Every item this lane's prefill overlapped is encoded and its embeddings are in the KV
            // now, so release the host patch payloads and the handoff binding: the decode loop never
            // revisits them. The arena itself is reused by the next lane's session.
            vision_session->release_encoded_media_payloads();
            vision_session->retire_handoff();
        }
        if (cancelled) {
            // A prefill that completed at least one chunk published what it reached, the way the walk
            // does: the retry of the same prompt then continues from there. Before the first chunk
            // nothing moved, and the recall's bookkeeping already describes the device pools, so that
            // case retires the lineage instead.
            if (prefilled > 0) {
                publish_partial_prefill(lane_id, prefilled, tokens, data);
            } else {
                invalidate_lane_prefill(lane_id);
            }
            (void)request.output.preview_terminal(FinishReason::Cancelled);
            lane.result.finish_reason = FinishReason::Cancelled;
            publish_preview(lane, false);
            finalize(lane);
            return false;
        }
        // The anchor goes into the owning entry's shared image before the publish below, which can
        // evict that entry and shift every later index.
        if (anchor_divergence) { capture_lane_anchor_frozen(lane_id, anchor_position); }
        computed_prefill_tokens_ += lane.prompt_tokens - reuse;
        record_reuse(lane.result);
        publish_lane_prefill(lane_id, lane.prompt_tokens, tokens, prompt_media, data.context_cache);
        lane.prefilled = true;
        lane.finished = first_token_finish != FinishReason::None;
        if (lane.finished) {
            finalize(lane);
            return false;
        }
        return true;
    };

    // S1: a member only takes its lane once the pages its own prompt plus output budget need are
    // available. The driver already collected the batch, so a member the shared pool cannot serve
    // right now goes back to the head of the queue and leaves the batch - the driver's safety net
    // publishes whatever is still in `batch`, and a requeued member must not be published empty.
    for (std::size_t index = 0; index < batch.size();) {
        const std::uint32_t need_tokens = lane_need_tokens(*batch[index]);
        if (!reserve_lane_kv(static_cast<std::uint32_t>(index), need_tokens)) {
            requeue_lane_front(std::move(batch[index]));
            batch.erase(batch.begin() + static_cast<std::ptrdiff_t>(index));
            continue;
        }
        if (admit_lane(index, *batch[index])) {
            lanes[index].kv_window = lane_kv_window(lane_kv_pages(need_tokens));
            active.push_back(index);
        }
        ++index;
    }

    for (;;) {
        // Retire the lanes that asked to stop before this round is formed. A cancelled or
        // budget-exhausted lane publishes its terminal preview and leaves; the remaining lanes keep
        // decoding, so one lane's limit does not stall the others.
        std::vector<std::size_t> live;
        live.reserve(active.size());
        for (const std::size_t index : active) {
            LaneState& lane  = lanes[index];
            Request& request = *lane.pending->request;
            if (lane.pending->cancellation.requested()) {
                (void)request.output.preview_terminal(FinishReason::Cancelled);
                lane.result.finish_reason = FinishReason::Cancelled;
                publish_preview(lane, false);
                finalize(lane);
                continue;
            }
            if (request.budget.remaining() == 0) {
                (void)request.output.preview_terminal(request.budget.limit_reason());
                lane.result.finish_reason = request.budget.limit_reason();
                publish_preview(lane, false);
                finalize(lane);
                continue;
            }
            live.push_back(index);
        }
        active.swap(live);

        // P2.2: every slot this round left free takes the next request that arrived while the batch
        // was running, so a short request no longer waits for the long one beside it to finish.
        while (active.size() < lane_capacity) {
            std::shared_ptr<PendingRequest> next = try_pop_lane_queue();
            if (next == nullptr) { break; }
            std::size_t index = lane_capacity;
            for (std::size_t candidate = 0; candidate < lane_capacity; ++candidate) {
                if (std::find(active.begin(), active.end(), candidate) == active.end()) {
                    index = candidate;
                    break;
                }
            }
            if (index == lane_capacity) { break; }
            // S1: the request's own pages have to be available before it may take the lane. If the
            // shared pool cannot cover it right now it goes back to the head of the queue rather
            // than losing its place to the requests that arrived after it. It is never dropped: the
            // admission ceiling guarantees a request that fits the policy fits an empty pool, so
            // this only ever waits for another lane to retire.
            const std::uint32_t need_tokens = lane_need_tokens(*next);
            if (!reserve_lane_kv(static_cast<std::uint32_t>(index), need_tokens)) {
                requeue_lane_front(std::move(next));
                break;
            }
            batch.push_back(next);
            if (lane_trace_enabled()) {
                std::fprintf(stderr, "[tp2-lane] admit slot=%zu live=%zu\n", index,
                             active.size() + 1);
            }
            if (admit_lane(index, *next)) {
                lanes[index].kv_window = lane_kv_window(lane_kv_pages(need_tokens));
                active.push_back(index);
            }
        }
        if (active.empty()) { break; }

        pump_decode_round();
    }
    timing.report_plain("plain-batch");
}

// ---------------------------------------------------------------------------------------------
// Lane-parallel masked-draft walk (P2.1c)
// ---------------------------------------------------------------------------------------------
//
// The batched sibling of the DFlash2 branch in execute_lane: one round drives every live lane
// through a single proposal, a single batched verify window and a single replay fold, then commits
// each lane's licensed prefix on its own.
//
// The batch is dense: active lane column occupies frame row column. A lane's own KV row, GDN state
// slot and draft-ring slot travel through the ingress slot vectors as values (lane.slot), never as
// the row index, so rows compact cleanly when a lane retires. Every frame tensor is sliced to
// columns along its outermost (batch) dimension, which stays contiguous.
//
// Grammar-constrained members reach here too: P2.1 masks each lane's own columns of the verify
// window before the argmax that reads it.
//
// A round costs every lane the slowest lane's proposal plus verify, so a retired lane frees a row
// but not time. Continuous batching is out of scope for P2.1c.
void TP2GenerationCore::execute_spec_batch(
    std::vector<std::shared_ptr<PendingRequest>>& batch) {
    DeviceArena& ws_a = *shard_a_.workspace;
    DeviceArena& ws_b = *shard_b_.workspace;
    auto& ctx_a       = *shard_a_.context;
    auto& ctx_b       = *shard_b_.context;
    shard_a_.device.bind_to_current_thread();

    const std::int32_t vocab = qwen::execution::dimension(shard_a_.model->config().text.vocab_size);
    const std::int32_t hidden =
        qwen::execution::dimension(shard_a_.model->config().text.hidden_size);
    const std::int32_t public_tokens =
        static_cast<std::int32_t>(shard_a_.model->resources().public_token_count);
    const std::uint32_t prefill_chunk = prefill_chunk_width(shard_a_.model->config().text);
    // Prefix-reuse trace, the same switch the single-lane walk and the plain batch read.
    const bool reuse_trace = [] {
        const char* env = std::getenv("NINFER_TP2_REUSE_TRACE");
        return env != nullptr && env[0] == '1';
    }();
    // The tail anchors cover the last few chunk ends of a lane's prefill. Their count is bounded by
    // the tail sub-ring, so a narrow chunk cannot flood the ring with anchors that all sit within one
    // chunk of the prompt end.
    const std::uint32_t tail_span =
        std::min<std::uint32_t>(kReuseTailWindow,
                                prefill_chunk * std::max(1U, host_checkpoint_tail_slots_));
    // DFlash2 and MTP each propose their own draft count, so one verify window is a column wider.
    // Both rounds are batched (P2.1b); only the live route's count is non-zero.
    const std::int32_t width =
        static_cast<std::int32_t>(mtp_enabled_ ? mtp_drafts_ : dflash_drafts_) + 1;
    Tp2RoundTiming& timing = tp2_timing();
    timing.init();
    timing.reset();

    struct LaneState {
        PendingRequest* pending = nullptr;
        std::uint32_t slot      = 0;
        GenerationResult result;
        std::int32_t current        = 0;
        std::uint32_t position      = 0;
        std::uint32_t prompt_tokens = 0;
        bool have_first             = false;
        bool finished               = false;
        // Set once this lane's prefill published its lineage, so finalize knows there is one to
        // publish at the frontier.
        bool prefilled = false;
        Clock::time_point begin;
        Clock::time_point first_token;
        // S1: the context window the pages this lane reserved give it. The executor sets it after
        // admit_lane returns - admit_lane resets the whole state struct on entry.
        std::uint32_t kv_window = 0;
    };

    // P2.2: the lane arrays cover the whole lane capacity rather than the batch that happened to
    // arrive, because a lane that retires frees its slot for a request still waiting in the queue.
    const std::size_t lane_capacity = lanes_;
    std::vector<LaneState> lanes(lane_capacity);
    for (std::size_t index = 0; index < batch.size(); ++index) {
        lanes[index].pending     = batch[index].get();
        lanes[index].slot        = static_cast<std::uint32_t>(index);
        lanes[index].begin       = Clock::now();
        lanes[index].first_token = lanes[index].begin;
    }
    auto lifetime_a = ws_a.scope();
    auto lifetime_b = ws_b.scope();

    // Filled by admit_lane for every lane it admits, including the ones that arrive later.
    std::vector<ops::SamplingConfig> configs(lane_capacity);

    // P2.1: the same per-lane grammar and mask staging the plain batch keeps, sized for a
    // speculative window (width columns per lane) instead of one column per lane.
    const std::size_t logits_domain = static_cast<std::size_t>(vocab);
    // Built by admit_lane, one lane at a time, so a lane admitted mid-batch gets its own constraint.
    std::vector<LaneGrammar> grammar(lane_capacity);
    bool any_grammar = false;
    std::vector<std::uint8_t> tool_mask_one(static_cast<std::size_t>(vocab), std::uint8_t{1});
    std::vector<std::uint8_t> tool_mask_host(
        static_cast<std::size_t>(vocab) * static_cast<std::size_t>(width) * lanes_,
        std::uint8_t{1});

    // Same reservation rule as the plain batch: the penalty counts cover the widest batch this
    // route can form, so the round watermark -- and with it every captured verify graph -- is the
    // same for every request. A request with no configured penalty never reads its slice, and only
    // the live span is cleared.
    // admit_lane points each admitted lane's config at its own slice of this array.
    std::int32_t* counts_base = nullptr;
    {
        Tensor counts = ws_a.alloc(
            DType::I32, {static_cast<std::int32_t>(public_tokens * static_cast<std::int32_t>(lanes_))});
        shard_a_.device.bind_to_current_thread();
        CUDA_CHECK(cudaMemsetAsync(
            counts.data, 0,
            sizeof(std::int32_t) * static_cast<std::size_t>(public_tokens) * lane_capacity,
            shard_a_.device.stream));
        counts_base = static_cast<std::int32_t*>(counts.data);
    }
    // The prefill-tail samples read their own lane's entry through this staging copy, so it carries
    // the same whole-lane-count reservation as the counts array.
    DeviceSpan configs_dev = ws_a.alloc_bytes(sizeof(ops::SamplingConfig) * lanes_, 256);
    shard_a_.device.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(configs_dev.data, configs.data(),
                               sizeof(ops::SamplingConfig) * lane_capacity, cudaMemcpyHostToDevice,
                               shard_a_.device.stream));

    // Pinned staging for one round: the verify window's per-lane ids/positions/valid columns, plus
    // the per-lane KV row and state slot vectors the window binds. This is the core member, not a
    // round local: a captured verify window reads the whole staging block through memcpy nodes, so
    // its address is baked into every graph and each section is strided by the startup lane count.
    const std::size_t window_span = static_cast<std::size_t>(width) * lanes_;
    const std::size_t id_span     = static_cast<std::size_t>(width) * lanes_;
    std::int32_t* spec_ids       = batch_window_base();
    std::int32_t* spec_positions = spec_ids + id_span;
    std::int32_t* spec_valid     = spec_positions + id_span;
    std::int32_t* spec_kv_rows   = spec_valid + lanes_;
    std::int32_t* spec_slots     = spec_kv_rows + lanes_;

    // Licensed prefixes come back through a second pinned allocation, together with the fold and
    // retirement vectors computed from them.
    PinnedHostBuffer accept_host((window_span + 6 * lanes_) * sizeof(std::int32_t), true);
    auto* accept_base             = static_cast<std::int32_t*>(accept_host.data());
    std::int32_t* licensed        = accept_base;
    std::int32_t* licensed_counts = licensed + window_span;
    std::int32_t* fold_slots      = licensed_counts + lanes_;
    std::int32_t* fold_columns    = fold_slots + lanes_;
    std::int32_t* tail_slots      = fold_columns + lanes_;
    std::int32_t* tail_starts     = tail_slots + lanes_;
    std::int32_t* tail_ends       = tail_starts + lanes_;

    // The frontier each lane's pending draft features were staged at is retention(slot)'s own
    // dflash_context_frontier (P2.3 Stage 2), the same member the single-lane walk keeps and the host
    // checkpoint ring reads, one lane at a time. The next round appends the columns this round
    // committed from there.

    // Per-lane preview publication, mirroring the single-lane lambda: a terminal call first names
    // the reason the walk stopped, then the committed preview is appended to this lane's result and
    // streamed to its own sink.
    auto publish_preview = [](LaneState& lane, bool terminal) {
        Request& request = *lane.pending->request;
        if (terminal) {
            (void)request.output.preview_terminal(request.budget.limit_reason());
        }
        auto published = request.output.commit_preview();
        for (const auto& delta : published) {
            if (delta.channel == OutputChannel::Reasoning) {
                lane.result.reasoning += delta.text;
            } else {
                lane.result.content += delta.text;
            }
            if (lane.pending->sink != nullptr) { lane.pending->sink->publish(delta); }
        }
    };

    // Fills a lane's result from everything its walk accumulated, in the same order the single-lane
    // walk does at its exit. A lane that never produced a first token reports an empty decode.
    // Publish a retired lane's lineage before its result takes the generated tokens away, exactly as
    // the plain batch and execute_walk do: the pools hold prompt plus committed output, and the live
    // GDN state sits at the frontier the lane's last committed round left.
    auto finalize = [this](LaneState& lane) {
        Request& request = *lane.pending->request;
        if (lane.prefilled) {
            auto& data = qwen::PreparedPromptAccess::mutable_view(request.prompt);
            std::vector<TokenId> history(data.token_ids.begin(), data.token_ids.end());
            history.insert(history.end(), request.generated.begin(), request.generated.end());
            session_publish(history, static_cast<std::uint32_t>(lane.position), data.context_cache,
                            collect_media_spans(data, data.token_ids.size()), lane.slot);
            // The anchors this answer wrote sit past a state the walk committed, so they become
            // usable on the same terms as the prefill's: once the walk stopped naming its frontier.
            publish_lane_checkpoints(lane.slot);
        }
        const Clock::time_point done = Clock::now();
        lane.result.generated_token_ids = std::move(request.generated);
        lane.result.tool_calls          = request.output.take_tool_calls();
        lane.result.tool_call_parse     = request.output.tool_call_parse_diagnostics();
        lane.result.reasoning_tokens    = request.output.reasoning_tokens();
        lane.result.matched_stop_string = request.output.matched_stop_string();
        lane.result.thinking            = request.output.thinking_stats();
        const double total = std::chrono::duration<double>(done - lane.begin).count();
        // Same anchor and same reasoning as the plain batch: the reported first-token and total spans
        // start where the core accepted the request, while `prompt_wall_seconds` stays
        // admission-relative because it defines the prefill rate this lane reports.
        const double queue_wait = lane.result.engine_timing.queue_wait_seconds;
        if (lane.have_first) {
            const double decode = std::chrono::duration<double>(done - lane.first_token).count();
            lane.result.timings.decode_seconds          = decode;
            lane.result.timings.generation_wall_seconds = decode;
            lane.result.timings.first_token_seconds =
                lane.result.timings.prepare_seconds + lane.result.timings.prompt_wall_seconds +
                queue_wait;
            lane.result.timings.total_seconds =
                lane.result.timings.prepare_seconds + total + queue_wait;
        } else {
            lane.result.timings.decode_seconds          = 0.0;
            lane.result.timings.generation_wall_seconds = 0.0;
            lane.result.timings.prompt_wall_seconds     = total;
            lane.result.timings.first_token_seconds =
                lane.result.timings.prepare_seconds + total + queue_wait;
            lane.result.timings.total_seconds =
                lane.result.timings.prepare_seconds + total + queue_wait;
        }
        lane.pending->result = std::move(lane.result);
        // S1: the pages this request held go back to the shared pool as it retires, so the next
        // request admitted onto this lane can use them. The session is copied into its host slabs
        // first - that copy reads the lane's device pages - because once they are free another lane
        // may overwrite them, and retention state that still named them would make the next request
        // skip its prefill and read somebody else's tokens.
        retire_lane_session(lane.slot);
        release_lane_kv(lane.slot);
        // P0.2: retire the member here rather than at the end of the batch. This lane is done, and
        // its submitter must be free to return while the other lanes keep running.
        publish_lane(*lane.pending);
    };

    // Each lane's GDN slot is brought to its own starting state in the loop below: a lane that
    // reuses a boundary is copied from its image, and a lane that does not is zeroed. The pool as a
    // whole is never wiped, because a lane continuing a conversation holds the state its previous
    // round left, and that state is exactly what its reuse boundary names (P2.3 Stage 2).

    // Prefill stays serial per lane: each lane stages its own masked-draft features into its own
    // draft-ring slot, so every lane's ring is primed before the first shared proposal.
    std::vector<std::size_t> active;
    active.reserve(lane_capacity);

    shard_a_.round_base = ws_a.used();
    shard_b_.round_base = ws_b.used();


    // Null on the MTP route: the artifact is loaded without a masked-draft component there, so
    // every use below is guarded by `dflash2_enabled_` (see the pending-feature tail).
    auto* round = shard_a_.dflash_round.get();
    qwen::execution::ExecutionCore dflash_execution{
        .device           = shard_a_.device,
        .parameters       = *shard_a_.parameters,
        .work             = *shard_a_.workspace,
        .linear_attention = *shard_a_.state,
        .replay_records   = nullptr,
        .io               = shard_a_.io,
        .prefill_hidden   = shard_a_.prefill_hidden,
        .prefill_chunk    = options_.prefill_chunk,
        .proposal_head    = options_.speculative.proposal_head,
    };

    std::vector<std::int32_t> lane_extent(lane_capacity, 0);
    std::vector<std::int32_t> append_slots(lane_capacity, 0);
    std::vector<std::int32_t> append_starts(lane_capacity, 0);
    std::vector<std::int32_t> append_ends(lane_capacity, 0);
    // MTP round staging (P2.1b): the packed per-frame-row vectors, the per-step cache positions and
    // the per-lane hidden selectors. The reservation is the whole lane count: a lane admitted
    // mid-batch must find its own entry even when the round it joins has fewer columns than that.
    std::vector<std::int32_t> mtp_pack(7 * lanes_, 0);
    std::vector<std::int32_t> mtp_step_host(lanes_, 0);
    std::vector<std::int32_t> mtp_selectors_h(lanes_, 0);

    // The speculative route pumps its own round: while one lane prefills, the lanes that already
    // hold a token run one full speculative round - draft, verify, accept - between two of that
    // lane's chunks, so they keep publishing instead of stalling for the prompt's whole duration.
    // Exactly one lane ever prefills, so the serial prefill order and every lane's TTFT stay as
    // they were; the round never retires, admits or re-enters admit_lane. The newcomer's own draft
    // ring is not part of the pumped round - it joins `active` only after its prefill returns - so
    // the shared proposal only ever covers lanes whose rings are primed.
    auto pump_spec_round = [&]() {
        if (active.empty()) { return; }
        const Clock::time_point round_start = Clock::now();
        const std::int32_t columns = static_cast<std::int32_t>(active.size());
        // The live speculative route owns the draft count: every lane reserved its own pages when it
        // was admitted, so the window those pages give it bounds the positions a round may write.
        const std::uint32_t draft_window = mtp_enabled_ ? mtp_drafts_ : dflash_drafts_;
        std::uint32_t max_position = 0;
        for (std::int32_t column = 0; column < columns; ++column) {
            const LaneState& lane = lanes[active[column]];
            const std::uint32_t budget_remaining = lane.pending->request->budget.remaining();
            const std::uint32_t max_by_budget = budget_remaining > 1U ? budget_remaining - 1U : 0U;
            const std::uint32_t capacity_left =
                lane.position + 1U < lane.kv_window ? lane.kv_window - lane.position - 1U : 0U;
            lane_extent[column] = static_cast<std::int32_t>(
                std::min({draft_window, max_by_budget, capacity_left}));
            max_position = std::max(max_position, lane.position);
        }

        position_arena(ws_a, shard_a_.round_base, shard_a_.round_base);
        position_arena(ws_b, shard_b_.round_base, shard_b_.round_base);
        auto scope_a = ws_a.scope();
        auto scope_b = ws_b.scope();
        shard_a_.device.bind_to_current_thread();

        // The accept kernel reads one sampling config per frame row, so the lanes' configs are
        // compacted into frame order the same way the plain batch does it.
        DeviceSpan round_configs =
            ws_a.alloc_bytes(sizeof(ops::SamplingConfig) * static_cast<std::size_t>(columns), 256);
        std::vector<ops::SamplingConfig> round_host(static_cast<std::size_t>(columns));
        for (std::int32_t column = 0; column < columns; ++column) {
            round_host[static_cast<std::size_t>(column)] = configs[lanes[active[column]].slot];
        }
        CUDA_CHECK(cudaMemcpyAsync(round_configs.data, round_host.data(),
                                   sizeof(ops::SamplingConfig) * static_cast<std::size_t>(columns),
                                   cudaMemcpyHostToDevice, shard_a_.device.stream));

        timing.record(0, shard_a_.device.stream);

        // ---- Proposal and verify ------------------------------------------------------------
        // DFlash2 proposes through its ring and selector; MTP runs its own autoregressive draft
        // chain against each lane's own MTP KV row (P2.1b). Both routes end with the same per-lane
        // verify inputs, so the batched verify window is shared, and both leave their licensed
        // prefixes in the pinned buffers the per-lane commit below reads.
        Tensor window_hidden3d;
        Tensor mtp_selectors;
        if (mtp_enabled_) {
            std::int32_t max_extent = 0;
            for (std::int32_t column = 0; column < columns; ++column) {
                max_extent = std::max(max_extent, lane_extent[column]);
            }
            const std::size_t mtp_stride = static_cast<std::size_t>(columns);
            const std::size_t hidden_bytes =
                static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);

            // One packed host image stages every per-frame-row vector the chain and the accept op
            // bind: anchors, base positions, extents, lengths, MTP KV rows, valid widths and the
            // next-round hidden selectors.
            // This engine's tensors are densest along dim 0, so a rank-1 slice of one flat image
            // stays contiguous while a row sliced out of a {7, columns} tensor and reshaped does not.
            Tensor mtp_vec = ws_a.alloc(DType::I32, {7 * columns});
            auto vec_row   = [&mtp_vec, columns](std::int32_t index) {
                return mtp_vec.slice(0, index * columns, columns);
            };
            Tensor mtp_anchors = vec_row(0);
            Tensor mtp_bases   = vec_row(1);
            Tensor mtp_extents = vec_row(2);
            Tensor mtp_lengths = vec_row(3);
            Tensor mtp_rows    = vec_row(4);
            Tensor mtp_valid   = vec_row(5);
            mtp_selectors      = vec_row(6);
            for (std::int32_t column = 0; column < columns; ++column) {
                const LaneState& lane       = lanes[active[column]];
                const std::size_t slot      = static_cast<std::size_t>(column);
                mtp_pack[0 * mtp_stride + slot] = lane.current;
                mtp_pack[1 * mtp_stride + slot] = static_cast<std::int32_t>(lane.position);
                mtp_pack[2 * mtp_stride + slot] = lane_extent[column];
                mtp_pack[3 * mtp_stride + slot] = static_cast<std::int32_t>(lane.position);
                mtp_pack[4 * mtp_stride + slot] = static_cast<std::int32_t>(lane.slot);
                mtp_pack[5 * mtp_stride + slot] = 1;
                mtp_pack[6 * mtp_stride + slot] = 0;
            }
            shard_a_.device.bind_to_current_thread();
            CUDA_CHECK(cudaMemcpyAsync(mtp_vec.data, mtp_pack.data(),
                                       7 * mtp_stride * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, shard_a_.device.stream));

            // The chain runs one MTP column per step for every lane: step s feeds the anchor (s == 0)
            // or the previous step's draft with the lane's own hidden, at cache position
            // `lane.position - 1 + s`, and proposes the next draft. The chain of a lane with a smaller
            // extent is computed but unused, exactly as the single-lane route proposes its whole
            // window before the accept decides.
            // A tensor's strides are densest along dim 0, so the {K, columns} view of a flat image
            // reads lane b's drafts at `flat[b * K + i]` - request-major, the layout the verify
            // and accept kernels index (`drafts[row * k + i]`). Every write below therefore lands
            // one lane's drafts at stride K, not one step's drafts contiguously.
            Tensor mtp_drafts_flat =
                ws_a.alloc(DType::I32, {static_cast<std::int32_t>(mtp_drafts_) * columns});
            Tensor mtp_drafts =
                mtp_drafts_flat.view({static_cast<std::int32_t>(mtp_drafts_), columns});
            const std::size_t draft_row_bytes =
                static_cast<std::size_t>(mtp_drafts_) * sizeof(std::int32_t);
            Tensor chain_logits = ws_a.alloc(DType::BF16, {vocab, columns});
            Tensor chain_anchor = ws_a.alloc(DType::BF16, {hidden, 1, columns});
            for (std::int32_t column = 0; column < columns; ++column) {
                const LaneState& lane = lanes[active[column]];
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<std::uint8_t*>(chain_anchor.data) +
                        static_cast<std::size_t>(column) * hidden_bytes,
                    static_cast<const std::uint8_t*>(shard_a_.mtp_anchor_hidden.data) +
                        static_cast<std::size_t>(lane.slot) * hidden_bytes,
                    hidden_bytes, cudaMemcpyDeviceToDevice, shard_a_.device.stream));
            }
            Tensor chain_a = ws_a.alloc(DType::BF16, {hidden, 1, columns});
            Tensor chain_b = ws_a.alloc(DType::BF16, {hidden, 1, columns});
            // The proposal writes a rank-1 request-major row, while the chain entry wants
            // {1, columns} column-fastest and the next step's ids are a strided read of the drafts
            // image. Both sides are staged through their own contiguous buffer.
            Tensor step_ids    = ws_a.alloc(DType::I32, {1, columns});
            Tensor step_drafts = ws_a.alloc(DType::I32, {columns});
            // The chain's cache positions are refilled every step. The buffer is allocated once,
            // outside the loop: the number of chain steps is this round's maximum lane extent, and
            // an allocation inside the loop would move the verify window's workspace watermark
            // with the draft acceptance, breaking the captured verify graph's replay layout.
            Tensor positions = ws_a.alloc(DType::I32, {1, columns});
            for (std::int32_t step = 0; step < max_extent; ++step) {
                Tensor& out = (step % 2 == 0) ? chain_a : chain_b;
                const Tensor& in =
                    step == 0 ? chain_anchor : (step % 2 == 0 ? chain_b : chain_a);
                Tensor ids;
                if (step == 0) {
                    ids = mtp_anchors.view({1, columns});
                } else {
                    ids = step_ids;
                    // One previous-step draft per lane, gathered out of the request-major image:
                    // four bytes every K int32.
                    CUDA_CHECK(cudaMemcpy2DAsync(
                        step_ids.data, sizeof(std::int32_t),
                        static_cast<const std::uint8_t*>(mtp_drafts_flat.data) +
                            static_cast<std::size_t>(step - 1) * sizeof(std::int32_t),
                        draft_row_bytes, sizeof(std::int32_t),
                        static_cast<std::size_t>(columns), cudaMemcpyDeviceToDevice,
                        shard_a_.device.stream));
                }
                for (std::int32_t column = 0; column < columns; ++column) {
                    mtp_step_host[static_cast<std::size_t>(column)] =
                        mtp_pack[1 * mtp_stride + static_cast<std::size_t>(column)] - 1 + step;
                }
                CUDA_CHECK(cudaMemcpyAsync(positions.data, mtp_step_host.data(),
                                           mtp_stride * sizeof(std::int32_t),
                                           cudaMemcpyHostToDevice, shard_a_.device.stream));
                // The envelope steers split policy only; the kernels take their visible set from
                // the per-lane positions, and the widest lane bounds every lane's window.
                const std::uint32_t visible = max_position + static_cast<std::uint32_t>(step);
                ctx_a.mtp_forward_decode_batch(
                    ids, in, positions, positions, mtp_valid, mtp_rows,
                    ops::CausalAttentionExecutionEnvelope{visible, visible}, out);
                shard_a_.device.bind_to_current_thread();
                ctx_a.mtp_propose_batch(out.view({hidden, columns}), chain_logits, step_drafts);
                // Scatter that step back into every lane's own draft row: four bytes every K int32.
                CUDA_CHECK(cudaMemcpy2DAsync(
                    static_cast<std::uint8_t*>(mtp_drafts_flat.data) +
                        static_cast<std::size_t>(step) * sizeof(std::int32_t),
                    draft_row_bytes, step_drafts.data, sizeof(std::int32_t),
                    sizeof(std::int32_t), static_cast<std::size_t>(columns),
                    cudaMemcpyDeviceToDevice, shard_a_.device.stream));
            }

            // The verify window is one column wider than the longest lane's extent, so a lane with
            // fewer valid drafts is still verified over its anchor column.
            Tensor mtp_verify_ids       = ws_a.alloc(DType::I32, {width, columns});
            Tensor mtp_verify_positions = ws_a.alloc(DType::I32, {width, columns});
            ops::speculative_prepare_verify_inputs(mtp_anchors, mtp_drafts, mtp_bases, mtp_extents,
                                                   mtp_verify_ids, mtp_verify_positions,
                                                   shard_a_.device.stream);
            Tensor window_logits2d = ws_a.alloc(DType::BF16, {vocab, width * columns});
            Tensor window_logits3d = window_logits2d.view({vocab, width, columns});
            Tensor window_hidden2d = ws_a.alloc(DType::BF16, {hidden, width * columns});
            window_hidden3d        = window_hidden2d.view({hidden, width, columns});
            const std::size_t window_live = static_cast<std::size_t>(width) * columns;
            CUDA_CHECK(cudaMemcpyAsync(spec_ids, mtp_verify_ids.data,
                                       window_live * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                       shard_a_.device.stream));
            CUDA_CHECK(cudaMemcpyAsync(spec_positions, mtp_verify_positions.data,
                                       window_live * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                       shard_a_.device.stream));
            for (std::int32_t column = 0; column < columns; ++column) {
                const LaneState& lane = lanes[active[column]];
                spec_valid[column]    = lane_extent[column] + 1;
                spec_kv_rows[column]  = static_cast<std::int32_t>(lane.slot);
                spec_slots[column]    = static_cast<std::int32_t>(lane.slot);
            }
            CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
            timing.record(1, shard_a_.device.stream);

            for (Shard* shard : {&shard_a_, &shard_b_}) {
                shard->device.bind_to_current_thread();
                CUDA_CHECK(cudaMemcpyAsync(shard->state_snapshots[kRoundScratchSlot].data,
                                           shard->state_backing.data, shard->state_backing.bytes,
                                           cudaMemcpyDeviceToDevice, shard->device.stream));
            }
            // The same ulp-drain the DFlash2 window needs: the two shards' bf16 kernels must agree
            // before the window reads either one.
            CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
            shard_b_.device.bind_to_current_thread();
            CUDA_CHECK(cudaStreamSynchronize(shard_b_.device.stream));
            CUDA_CHECK(cudaDeviceSynchronize());
            shard_a_.device.bind_to_current_thread();
            CUDA_CHECK(cudaDeviceSynchronize());
            run_verify_window_batch(spec_ids, spec_positions, spec_kv_rows, spec_slots, width,
                                    columns, static_cast<std::int32_t>(max_position),
                                    window_logits2d, window_hidden2d, nullptr, spec_valid);
            timing.record(2, shard_a_.device.stream);

            // P2.1: the declared-name mask has to follow the verify forward that produces the logits
            // and precede the argmax that reads them. Window column t of lane l (the round's column
            // index) owns the mask block at t + width * l, matching the window's own layout.
            if (any_grammar) {
                std::fill(tool_mask_host.begin(), tool_mask_host.end(), std::uint8_t{1});
                std::int32_t masked_columns = 0;
                for (std::int32_t column = 0; column < columns; ++column) {
                    const std::size_t lane_index = active[static_cast<std::size_t>(column)];
                    LaneGrammar& lane_grammar    = grammar[lane_index];
                    const Request& lane_request  = *lanes[lane_index].pending->request;
                    if (!grammar_live(lane_grammar, lane_request)) {
                        continue;
                    }
                    grammar_advance(lane_grammar, lane_request);
                    std::string drafted_prefix;
                    for (std::int32_t t = 0; t < width; ++t) {
                        if (lane_grammar.constraint->build_mask_after(drafted_prefix, logits_domain,
                                                                      tool_mask_one)) {
                            const std::size_t base =
                                (static_cast<std::size_t>(t) + static_cast<std::size_t>(width) *
                                                                  static_cast<std::size_t>(column)) *
                                static_cast<std::size_t>(vocab);
                            std::copy(tool_mask_one.begin(), tool_mask_one.end(),
                                      tool_mask_host.begin() + static_cast<std::ptrdiff_t>(base));
                            ++masked_columns;
                        }
                        if (t + 1 < width) {
                            const std::size_t next =
                                static_cast<std::size_t>(t + 1) + static_cast<std::size_t>(width) *
                                                                     static_cast<std::size_t>(column);
                            drafted_prefix.append(lane_grammar.constraint->piece(
                                static_cast<std::size_t>(spec_ids[next])));
                        }
                    }
                }
                if (masked_columns > 0) {
                    Tensor tool_mask_dev =
                        ws_a.alloc(DType::U8, {vocab, static_cast<std::int32_t>(window_live)});
                    shard_a_.device.bind_to_current_thread();
                    CUDA_CHECK(cudaMemcpyAsync(tool_mask_dev.data, tool_mask_host.data(),
                                               tool_mask_dev.bytes(), cudaMemcpyHostToDevice,
                                               shard_a_.device.stream));
                    ops::apply_token_mask(window_logits2d, tool_mask_dev, shard_a_.device.stream);
                    if (lane_trace_enabled()) {
                        std::fprintf(stderr, "[tp2-lane] grammar masked %d window columns of %d lanes\n",
                                     static_cast<int>(masked_columns), static_cast<int>(columns));
                    }
                }
            }

            Tensor mtp_target          = ws_a.alloc(DType::I32, {width, columns});
            Tensor mtp_target_flat     = mtp_target.view({width * columns});
            Tensor mtp_licensed        = ws_a.alloc(DType::I32, {width, columns});
            Tensor mtp_licensed_counts = ws_a.alloc(DType::I32, {columns});
            Tensor mtp_accepted        = ws_a.alloc(DType::I32, {columns});
            ops::argmax(window_logits2d, mtp_target_flat, public_tokens, shard_a_.device.stream);
            ops::speculative_accept_greedy_drafts(
                mtp_target, window_logits3d, mtp_drafts, mtp_extents, mtp_lengths, mtp_anchors,
                mtp_licensed, mtp_licensed_counts, mtp_accepted, public_tokens,
                reinterpret_cast<const ops::SamplingConfig*>(round_configs.data), ws_a,
                shard_a_.device.stream);
            timing.record(3, shard_a_.device.stream);
            CUDA_CHECK(cudaMemcpyAsync(licensed, mtp_licensed.data,
                                       window_live * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                       shard_a_.device.stream));
            CUDA_CHECK(cudaMemcpyAsync(licensed_counts, mtp_licensed_counts.data,
                                       static_cast<std::size_t>(columns) * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, shard_a_.device.stream));
            CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
            timing.record(4, shard_a_.device.stream);
        } else {
        // ---- Proposal -----------------------------------------------------------------------
        // Each lane stages the features the previous round's verify wrote into its own ring slot,
        // then one batched proposal fills every frame row of the ingress.
        for (std::int32_t column = 0; column < columns; ++column) {
            const LaneState& lane = lanes[active[column]];
            append_slots[column]  = static_cast<std::int32_t>(lane.slot);
            append_starts[column] =
                static_cast<std::int32_t>(retention(lane.slot).dflash_context_frontier);
            append_ends[column]   = static_cast<std::int32_t>(lane.position);
        }
        round->append_pending_batch(dflash_execution, append_slots.data(), append_starts.data(),
                                   append_ends.data(), columns);
        for (std::int32_t column = 0; column < columns; ++column) {
            retention(lanes[active[column]].slot).dflash_context_frontier =
                lanes[active[column]].position;
        }

        qwen::DFlashDecodeIngress& ingress = round->ingress();
        ingress                            = {};
        for (std::int32_t column = 0; column < columns; ++column) {
            const LaneState& lane = lanes[active[column]];
            ingress.anchors[column]                 = static_cast<TokenId>(lane.current);
            ingress.execution_frontiers[column]     = static_cast<std::int32_t>(lane.position);
            ingress.context_frontiers[column]       = static_cast<std::int32_t>(lane.position);
            ingress.proposal_extents[column]        = lane_extent[column];
            ingress.proposal_valid_columns[column]  = width;
            ingress.target_valid_columns[column]    = lane_extent[column] + 1;
            for (std::int32_t k = 0; k < width; ++k) {
                ingress.target_rope_positions[static_cast<std::size_t>(column) * width + k] =
                    static_cast<std::int32_t>(lane.position) + std::min(k, lane_extent[column]);
            }
            ingress.text_kv_table_rows[column]      = static_cast<std::int32_t>(lane.slot);
            ingress.dflash_kv_table_rows[column]    = static_cast<std::int32_t>(lane.slot);
            ingress.active_lanes[column]            = static_cast<std::int32_t>(lane.slot);
            ingress.state_source_slots[column]      = static_cast<std::int32_t>(lane.slot);
            ingress.state_destination_slots[column] = static_cast<std::int32_t>(lane.slot);
            ingress.sampling[column]                = configs[lane.slot];
        }
        // The envelopes are a launch/workspace promise over the batch maximum, not a mask.
        const qwen::execution::DFlashEnvelopes envelopes{
            .local  = {0U, max_position},
            .full   = {0U, max_position},
            .append = {0U, static_cast<std::uint32_t>(width)}};
        shard_a_.device.bind_to_current_thread();
        round->propose(dflash_execution, shard_a_.decoder->text_kv, shard_a_.context.get(),
                      dflash_drafts_, envelopes, columns);
        timing.record(1, shard_a_.device.stream);

        // ---- Batched verify window ----------------------------------------------------------
        qwen::DFlashDecodeState& frame = round->frame();
        // The accept op wants the [V, width, columns] view; the window wants it flattened.
        Tensor window_logits3d = frame.target_logits.slice(2, 0, columns);
        Tensor window_logits   = window_logits3d.view({vocab, width * columns});
        Tensor window_hidden =
            frame.target_hidden.slice(2, 0, columns).view({hidden, width * columns});

        Tensor frame_verify_ids       = frame.verify_ids.slice(1, 0, columns);
        Tensor frame_verify_positions = frame.verify_positions.slice(1, 0, columns);
        Tensor frame_drafts           = frame.draft_tokens.slice(1, 0, columns);
        Tensor frame_candidate_ids    = frame.candidate_ids.slice(2, 0, columns);
        Tensor frame_proposal_q       = frame.proposal_q.slice(2, 0, columns);
        Tensor frame_extents          = frame.proposal_extents.slice(0, 0, columns);
        Tensor frame_frontiers        = frame.execution_frontiers.slice(0, 0, columns);
        Tensor frame_anchors          = frame.anchors.slice(0, 0, columns);
        Tensor frame_target_argmax    = frame.target_argmax.slice(1, 0, columns);
        // `argmax` writes one flat entry per window column; accept reads the same memory as
        // [width, columns], so the two views must share the column-fastest flattening.
        Tensor target_argmax_flat     = frame_target_argmax.view({width * columns});
        Tensor frame_licensed         = frame.licensed_tokens.slice(1, 0, columns);
        Tensor frame_licensed_counts  = frame.licensed_counts.slice(0, 0, columns);
        Tensor frame_accepted         = frame.accepted_drafts.slice(0, 0, columns);

        shard_a_.device.bind_to_current_thread();
        ops::speculative_prepare_verify_inputs(frame_anchors, frame_drafts, frame_frontiers,
                                               frame_extents, frame_verify_ids,
                                               frame_verify_positions, shard_a_.device.stream);
        // The pinned window is sized for the full lane capacity, but a round that retired a lane only
        // fills `width * columns` of it, so both the source tensor and the copy use the live span.
        const std::size_t window_live = static_cast<std::size_t>(width) * columns;
        CUDA_CHECK(cudaMemcpyAsync(spec_ids, frame_verify_ids.data,
                                   window_live * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                   shard_a_.device.stream));
        CUDA_CHECK(cudaMemcpyAsync(spec_positions, frame_verify_positions.data,
                                   window_live * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                   shard_a_.device.stream));
        for (std::int32_t column = 0; column < columns; ++column) {
            const LaneState& lane = lanes[active[column]];
            spec_valid[column]    = lane_extent[column] + 1;
            spec_kv_rows[column]  = static_cast<std::int32_t>(lane.slot);
            spec_slots[column]    = static_cast<std::int32_t>(lane.slot);
        }
        CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));

        // The target state image is captured before the verify mutates it, so the fold can replay
        // each lane's committed columns from the same frontier the single-lane route uses.
        for (Shard* shard : {&shard_a_, &shard_b_}) {
            shard->device.bind_to_current_thread();
            CUDA_CHECK(cudaMemcpyAsync(shard->state_snapshots[kRoundScratchSlot].data,
                                       shard->state_backing.data, shard->state_backing.bytes,
                                       cudaMemcpyDeviceToDevice, shard->device.stream));
        }

        qwen::execution::DFlashFeatureSink verify_sink = round->make_verify_sink(columns);
        // The masked draft's bf16 rounds differ by an ulp between the two shards' kernels, so both
        // devices are drained before the window reads either one (DFlash2 requires this).
        CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
        shard_b_.device.bind_to_current_thread();
        CUDA_CHECK(cudaStreamSynchronize(shard_b_.device.stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        shard_a_.device.bind_to_current_thread();
        CUDA_CHECK(cudaDeviceSynchronize());
        run_verify_window_batch(spec_ids, spec_positions, spec_kv_rows, spec_slots, width, columns,
                                static_cast<std::int32_t>(max_position), window_logits,
                                window_hidden, &verify_sink, spec_valid);
        timing.record(2, shard_a_.device.stream);

        // P2.1: the same declared-name mask the serial DFlash2 window applies, per lane. It has to
        // follow the verify forward that produces the logits (the forward would overwrite it) and
        // precede the argmax that reads them. Window column t of lane l (the round's column index)
        // owns the mask block at t + width * l, matching the window's own layout.
        if (any_grammar) {
            std::fill(tool_mask_host.begin(), tool_mask_host.end(), std::uint8_t{1});
            std::int32_t masked_columns = 0;
            for (std::int32_t column = 0; column < columns; ++column) {
                const std::size_t lane_index = active[static_cast<std::size_t>(column)];
                LaneGrammar& lane_grammar    = grammar[lane_index];
                const Request& lane_request  = *lanes[lane_index].pending->request;
                if (!grammar_live(lane_grammar, lane_request)) {
                    continue;
                }
                grammar_advance(lane_grammar, lane_request);
                std::string drafted_prefix;
                for (std::int32_t t = 0; t < width; ++t) {
                    if (lane_grammar.constraint->build_mask_after(drafted_prefix, logits_domain,
                                                                  tool_mask_one)) {
                        const std::size_t base =
                            (static_cast<std::size_t>(t) + static_cast<std::size_t>(width) *
                                                              static_cast<std::size_t>(column)) *
                            static_cast<std::size_t>(vocab);
                        std::copy(tool_mask_one.begin(), tool_mask_one.end(),
                                  tool_mask_host.begin() + static_cast<std::ptrdiff_t>(base));
                        ++masked_columns;
                    }
                    if (t + 1 < width) {
                        const std::size_t next =
                            static_cast<std::size_t>(t + 1) + static_cast<std::size_t>(width) *
                                                                 static_cast<std::size_t>(column);
                        drafted_prefix.append(lane_grammar.constraint->piece(
                            static_cast<std::size_t>(spec_ids[next])));
                    }
                }
            }
            if (masked_columns > 0) {
                Tensor tool_mask_dev =
                    ws_a.alloc(DType::U8, {vocab, static_cast<std::int32_t>(window_live)});
                shard_a_.device.bind_to_current_thread();
                CUDA_CHECK(cudaMemcpyAsync(tool_mask_dev.data, tool_mask_host.data(),
                                           tool_mask_dev.bytes(), cudaMemcpyHostToDevice,
                                           shard_a_.device.stream));
                ops::apply_token_mask(window_logits, tool_mask_dev, shard_a_.device.stream);
                if (lane_trace_enabled()) {
                    std::fprintf(stderr, "[tp2-lane] grammar masked %d window columns of %d lanes\n",
                                 static_cast<int>(masked_columns), static_cast<int>(columns));
                }
            }
        }

        ops::argmax(window_logits, target_argmax_flat, public_tokens, shard_a_.device.stream);
        ops::speculative_accept_sparse_drafts(
            frame_target_argmax, window_logits3d, frame_drafts, frame_candidate_ids,
            frame_proposal_q, frame_extents, frame_frontiers, frame_anchors, frame_licensed,
            frame_licensed_counts, frame_accepted, public_tokens,
            reinterpret_cast<const ops::SamplingConfig*>(round_configs.data),
            ops::SpeculativeAcceptExecutionEnvelope{false}, ws_a, shard_a_.device.stream);
        timing.record(3, shard_a_.device.stream);
        CUDA_CHECK(cudaMemcpyAsync(licensed, frame_licensed.data,
                                   window_live * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                   shard_a_.device.stream));
        CUDA_CHECK(cudaMemcpyAsync(licensed_counts, frame_licensed_counts.data,
                                   static_cast<std::size_t>(columns) * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, shard_a_.device.stream));
        CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
        timing.record(4, shard_a_.device.stream);
        }

        const Clock::time_point sync_start = Clock::now();
        shard_b_.device.bind_to_current_thread();
        CUDA_CHECK(cudaStreamSynchronize(shard_b_.device.stream));
        shard_a_.device.bind_to_current_thread();
        timing.close_round(
            std::chrono::duration<double, std::milli>(Clock::now() - sync_start).count());
        abort_if_ar_stalled();

        // ---- Per-lane licensing -------------------------------------------------------------
        std::int32_t tail_count = 0;
        for (std::int32_t column = 0; column < columns; ++column) {
            LaneState& lane  = lanes[active[column]];
            Request& request = *lane.pending->request;
            const std::int32_t count = licensed_counts[column];
            if (count < 1 || count > width) {
                throw std::logic_error(
                    mtp_enabled_ ? "TP-2 MTP round produced an invalid licensed prefix"
                                 : "TP-2 DFlash2 round produced an invalid licensed prefix");
            }
            std::vector<TokenId> step(
                licensed + static_cast<std::ptrdiff_t>(column) * width,
                licensed + static_cast<std::ptrdiff_t>(column) * width + count);
            if (lane_extent[column] == 0) {
                lane.result.speculative.fallback_steps += 1;
            } else {
                lane.result.speculative.rounds += 1;
                lane.result.speculative.drafted_tokens +=
                    static_cast<std::uint64_t>(lane_extent[column]);
                lane.result.speculative.accepted_tokens += static_cast<std::uint64_t>(count - 1);
                for (std::int32_t position = 0; position < count - 1; ++position) {
                    lane.result.speculative
                        .accepted_per_position[static_cast<std::size_t>(position)] += 1;
                }
            }
            const std::uint32_t remaining = request.budget.remaining();
            if (step.size() > remaining) { step.resize(remaining); }
            const OutputDecision decision =
                request.output.preview_model(step, remaining, request.budget.limit_reason());
            if (decision.accepted_tokens == 0 || decision.accepted_tokens > step.size()) {
                throw std::logic_error("TP-2 output policy returned an invalid licensed prefix");
            }
            request.generated.insert(request.generated.end(), step.begin(),
                                     step.begin() + decision.accepted_tokens);
            request.budget.commit(decision.accepted_tokens);
            committed_decode_tokens_ += decision.accepted_tokens;
            ++decode_rounds_;
            publish_preview(lane, false);

            const std::uint32_t committed = decision.accepted_tokens;
            const std::uint32_t base      = lane.position;
            lane.position                 = base + committed;
            lane.current                  = static_cast<std::int32_t>(step[committed - 1]);
            fold_slots[column]            = static_cast<std::int32_t>(lane.slot);
            fold_columns[column]          = static_cast<std::int32_t>(committed);
            if (mtp_enabled_) {
                mtp_selectors_h[column] = static_cast<std::int32_t>(committed) - 1;
            }
            timing.committed += committed;
            ++timing.rounds;
            if (decision.finished()) {
                lane.result.finish_reason = decision.finish_reason;
                lane.finished             = true;
                retention(lane.slot).dflash_context_frontier = lane.position;
                if (lane.position > base) {
                    tail_slots[tail_count]  = static_cast<std::int32_t>(lane.slot);
                    tail_starts[tail_count] = static_cast<std::int32_t>(base);
                    tail_ends[tail_count]   = static_cast<std::int32_t>(lane.position);
                    ++tail_count;
                }
                finalize(lane);
            } else {
                // The next round appends exactly the columns this round committed.
                retention(lane.slot).dflash_context_frontier = base;
            }
            // Anchors inside the answer. The position grid only fires on absolute stride multiples,
            // so an answer that never crosses one leaves the prompt end as the only boundary behind
            // it and a client that re-renders that answer re-prefills all of it. The tail ring
            // already holds the prefill's last chunk ends, so these anchors cost no extra slot; the
            // draft half has to reach the frontier the anchor names, exactly as the grid write in
            // the serial walk makes it, or a restore refuses the checkpoint.
            if (host_checkpoint_tail_slots_ != 0 && !decision.finished()) {
                const std::uint32_t frontier = lane.position - 1U;
                std::uint32_t& next          = retention(lane.slot).next_generated_checkpoint;
                if (next != 0 && frontier >= next) {
                    const std::uint32_t draft_frontier =
                        retention(lane.slot).dflash_context_frontier;
                    if (dflash2_enabled_ && frontier > draft_frontier) {
                        // Hand the pending window to the draft ring for this lane alone, then keep
                        // the frontier it now reaches: the round-level append below only carries the
                        // lanes that finished, and it also resets every other lane to `base`.
                        const std::int32_t flush_slot  = static_cast<std::int32_t>(lane.slot);
                        const std::int32_t flush_start = static_cast<std::int32_t>(draft_frontier);
                        const std::int32_t flush_end   = static_cast<std::int32_t>(frontier);
                        round->append_pending_batch(dflash_execution, &flush_slot, &flush_start,
                                                    &flush_end, 1);
                        retention(lane.slot).dflash_context_frontier = frontier;
                    }
                    snapshot_host_checkpoint(shard_a_, frontier, HostRing::Tail, lane.slot);
                    snapshot_host_checkpoint(shard_b_, frontier, HostRing::Tail, lane.slot);
                    next = frontier + kReuseGeneratedTailStride;
                }
            }
        }

        // MTP carries its next anchor in its own hidden column: the verify window's hidden one column
        // before the new position is exactly what the next chain's first bridge consumes (P2.1b).
        if (mtp_enabled_) {
            shard_a_.device.bind_to_current_thread();
            const std::size_t hidden_bytes =
                static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
            CUDA_CHECK(cudaMemcpyAsync(mtp_selectors.data, mtp_selectors_h.data(),
                                       static_cast<std::size_t>(columns) * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, shard_a_.device.stream));
            Tensor selected = ws_a.alloc(DType::BF16, {hidden, columns});
            ops::speculative_select_accepted_hidden(window_hidden3d, mtp_selectors, selected,
                                                    shard_a_.device.stream);
            for (std::int32_t column = 0; column < columns; ++column) {
                const std::uint32_t slot = lanes[active[column]].slot;
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<std::uint8_t*>(shard_a_.mtp_anchor_hidden.data) +
                        static_cast<std::size_t>(slot) * hidden_bytes,
                    static_cast<const std::uint8_t*>(selected.data) +
                        static_cast<std::size_t>(column) * hidden_bytes,
                    hidden_bytes, cudaMemcpyDeviceToDevice, shard_a_.device.stream));
            }
        }

        // Both shards return to the pre-round image and one fold replays each lane's committed
        // columns out of its own record row into its own state slot.
        timing.record(5, shard_a_.device.stream);
        for (Shard* shard : {&shard_a_, &shard_b_}) {
            shard->device.bind_to_current_thread();
            CUDA_CHECK(cudaMemcpyAsync(shard->state_backing.data,
                                       shard->state_snapshots[kRoundScratchSlot].data,
                                       shard->state_backing.bytes, cudaMemcpyDeviceToDevice,
                                       shard->device.stream));
        }
        fold_verify_window(fold_slots, fold_columns, columns);
        // The pending-feature tail belongs to the masked-draft ring, which the MTP route does not
        // have at all (its draft round is null there).
        if (tail_count > 0 && dflash2_enabled_) {
            round->append_pending_batch(dflash_execution, tail_slots, tail_starts, tail_ends,
                                        tail_count);
        }
        timing.record(6, shard_a_.device.stream);

        std::vector<std::size_t> still;
        still.reserve(active.size());
        for (std::size_t index : active) {
            if (!lanes[index].finished) { still.push_back(index); }
        }
        active.swap(still);
        timing.round_ms +=
            std::chrono::duration<double, std::milli>(Clock::now() - round_start).count();
    };

    // P2.2: one lane's whole setup as a callable unit, so a slot a retired lane leaves free can take
    // the next queued request at any round boundary. The caller appends the member to `batch` before
    // this runs, so a throw here still reaches the driver's failure propagation and its publish
    // safety net. Returns true when the lane is left decoding.
    auto admit_lane = [&](std::size_t index, PendingRequest& pending) -> bool {
        LaneState& lane  = lanes[index];
        lane             = LaneState{};
        lane.pending     = &pending;
        lane.slot        = static_cast<std::uint32_t>(index);
        lane.begin       = Clock::now();
        lane.first_token = lane.begin;
        // The reported TTFT starts at the core's own acceptance, not at this admit: a request that
        // waited in the FIFO paid that time before any compute ran. `prompt_wall_seconds` stays
        // admission-relative because it defines the prefill rate this lane reports.
        lane.result.engine_timing.queue_wait_seconds =
            std::chrono::duration<double>(lane.begin - pending.request->submitted).count();
        // This lane's own sampler config and grammar, built here because a lane may be admitted long
        // after the batch was formed. The device copy of the configs is refreshed entry by entry.
        configs[index] = make_sampling_config(pending.request->sampling);
        if (configs[index].presence_penalty != 0.0F || configs[index].frequency_penalty != 0.0F) {
            configs[index].token_counts =
                counts_base +
                static_cast<std::size_t>(index) * static_cast<std::size_t>(public_tokens);
        }
        grammar[index] = LaneGrammar{};
        {
            const auto& lane_data = qwen::PreparedPromptAccess::view(pending.request->prompt);
            if (lane_data.tool_call_output != nullptr) {
                grammar[index].constraint =
                    frontend_->make_tool_call_constraint(lane_data.tool_call_output);
                if (grammar[index].constraint != nullptr) {
                    if (grammar[index].constraint->vocab_size() > logits_domain) {
                        throw std::logic_error(
                            "TP-2 tool-call constraint vocabulary " +
                            std::to_string(grammar[index].constraint->vocab_size()) +
                            " exceeds the logits domain " + std::to_string(logits_domain));
                    }
                    any_grammar = true;
                }
            }
        }
        shard_a_.device.bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<std::uint8_t*>(configs_dev.data) +
                static_cast<std::size_t>(index) * sizeof(ops::SamplingConfig),
            &configs[index], sizeof(ops::SamplingConfig), cudaMemcpyHostToDevice,
            shard_a_.device.stream));
        Request& request = *lane.pending->request;
        // The batch column is also the KV execution row and the GDN state slot this lane owns, which
        // is the lane index every session_* and retention helper takes (P2.3).
        const std::uint32_t lane_id = lane.slot;
        auto& data = qwen::PreparedPromptAccess::mutable_view(request.prompt);
        lane.prompt_tokens = static_cast<std::uint32_t>(data.token_ids.size());
        lane.position      = lane.prompt_tokens;
        lane.result.prompt = request.summary;
        lane.result.timings.prepare_seconds = request.prepare_seconds;
        const std::uint32_t draft_window = mtp_enabled_ ? mtp_drafts_ : dflash_drafts_;
        lane.result.speculative =
            SpeculativeStats{.backend               = mtp_enabled_ ? SpeculativeBackend::Mtp
                                                                   : SpeculativeBackend::DFlash2,
                             .enabled               = true,
                             .draft_window          = draft_window,
                             .accepted_per_position = std::vector<std::uint64_t>(draft_window, 0)};
        active_lane_ = static_cast<std::int32_t>(lane.slot);
        for (Shard* shard : {&shard_a_, &shard_b_}) {
            shard->device.bind_to_current_thread();
            shard->context->set_linear_state_slots(active_lane_, active_lane_);
        }

        // A client that re-rendered the answer this lane generated hands back a prompt whose last
        // turn is the tokens the lineage really produced; adopting it keeps the conversation
        // resident instead of retiring it over a re-tokenisation difference.
        const TurnAdoption adoption =
            adopt_generated_turn(data, lane.prompt_tokens, reuse_trace, lane_id);
        if (adoption.adopted) {
            lane.prompt_tokens               = adoption.prompt_tokens;
            lane.position                    = lane.prompt_tokens;
            request.summary.prompt_tokens    = lane.prompt_tokens;
            lane.result.prompt.prompt_tokens = lane.prompt_tokens;
        }
        const std::vector<TokenId>& tokens = data.token_ids;
        const std::span<const TokenId> token_ids(tokens.data(), tokens.size());
        const std::vector<MediaSpan> prompt_media = collect_media_spans(data, data.token_ids.size());

        // Cross-session recall and the boundary scan, both on this lane's own lineage, exactly as the
        // plain batch runs them (P2.3 Stage 2). Every lane owns its own GDN slot, draft-ring slot and
        // slice of the host ring, so two lanes never share a boundary.
        session_recall(lane_id, token_ids, prompt_media);
        const LaneReuse lane_reuse =
            scan_lane_reuse(lane_id, token_ids, prompt_media, lane.prompt_tokens, adoption.divergence,
                            adoption.adopted, reuse_trace);
        const std::uint32_t reuse          = lane_reuse.tokens;
        std::uint32_t next_host_checkpoint = lane_reuse.next_host_checkpoint;
        // The two anchors the serial walk plans, on this lane's own chunk grid: the boundary a later
        // conversation of this family is known to want, and the end of this prompt's own leading
        // instruction block. Without them the batched route offers only the grid and tail
        // checkpoints, which is why a new conversation sharing just the system block re-prefilled it.
        const LaneAnchors anchors = plan_lane_anchors(
            lane_id, token_ids, lane.prompt_tokens, reuse, lane_reuse.shared_prefix, prefill_chunk,
            data.context_cache.leading_instruction_frontier.value_or(0));
        const bool anchor_divergence        = anchors.anchor_divergence;
        const std::uint32_t anchor_position = anchors.anchor_position;
        const bool block_anchor             = anchors.block_anchor;
        const std::uint32_t block_position  = anchors.block_position;
        restore_lane_gdn(lane_id, lane_reuse);
        // The batched lane rebuilds its whole draft ring from its own prefill sink below, so there is
        // nothing to zero here: the ring is only read once this lane's prefill has primed it.
        restore_lane_dflash(lane_id, lane_reuse, false);
        retention(lane_id).dflash_context_frontier = reuse;
        // The device state still stands on the boundary this lane inherited, so freeze it before the
        // first chunk overwrites it.
        capture_lane_anchor_from_device(lane_id, reuse, lane_reuse.shared_prefix);
        lane.result.reused_prompt_tokens = reuse;
        lane.result.prefix_reuse_path    = reuse_path(reuse, anchors.block_frontier, lane_id);
        lane.result.reuse_source         = reuse_source_kind(lane_retention_[lane_id].reuse_source);
        lane.result.reuse_reason         = lane_reuse.reuse_reason == ReuseReason::Used
                                               ? ReuseReasonKind::Used
                                               : ReuseReasonKind::NoAnchorAtOrBefore;
        if (lane.pending->sink != nullptr) {
            lane.pending->sink->start(
                GenerationStart{.prompt = request.summary, .reused_prompt_tokens = reuse});
        }

        // This lane's Vision session, on top of the startup plan. Prefill is serial per lane, so the
        // single startup arena is reused sequentially and only one session is ever alive; the plan
        // must outlive the session, which binds it by reference.
        const bool media = data.has_media();
        qwen::execution::VisionPrefillPlan vision_plan;
        std::unique_ptr<qwen::execution::VisionPrefillSession> vision_session =
            open_vision_session(data, reuse, vision_plan);
        bool cancelled                  = lane.pending->cancellation.requested();
        FinishReason first_token_finish = FinishReason::None;
        // The chunk loop's own progress, hoisted out of its scope: the cancel path below publishes
        // this lane's session up to the last chunk that really finished.
        std::uint32_t prefilled = reuse;
        for (std::uint32_t t0 = reuse; !cancelled && t0 < lane.prompt_tokens;) {
            if (lane.pending->cancellation.requested()) {
                cancelled = true;
                break;
            }
            // Interleave one speculative round for the lanes that already hold a token, so this
            // lane's prompt does not stall every lane beside it for the prompt's whole duration.
            // The previous chunk's scopes have handed the arena back, so the round starts at the
            // same watermark it does from the main loop.
            ++timing.prefill_chunks;
            if (!active.empty()) {
                ++timing.pump_rounds;
                pump_spec_round();
                // A one-column round republishes the scalar GDN slot pair; restore the binding this
                // lane's prefill set up so the chunk that follows still runs on its own slot.
                for (Shard* shard : {&shard_a_, &shard_b_}) {
                    shard->device.bind_to_current_thread();
                    shard->context->set_linear_state_slots(active_lane_, active_lane_);
                }
            }
            std::uint32_t length = std::min(prefill_chunk, lane.prompt_tokens - t0);
            // A multimodal chunk is capped at the boundary of the item it overlaps, so the encoder
            // hands out one item at a time and the scatter below stays one contiguous column range
            // of it; the next chunk re-enters the same item. The MTP priming below sees the capped
            // length, exactly as the serial walk's does.
            qwen::execution::VisionChunk vision_chunk;
            if (vision_session) {
                vision_chunk = vision_session->prepare_chunk(t0, length);
                length       = static_cast<std::uint32_t>(vision_chunk.length);
            }
            // An anchor is a state this prefill must freeze exactly, so the chunk that would step
            // over one ends on it - the same clamp the serial walk applies, and for the same reason:
            // the anchor is only ever planned on this prefill's own chunk grid.
            if (anchor_divergence && anchor_position > t0 && anchor_position < t0 + length) {
                length = anchor_position - t0;
            }
            if (block_anchor && block_position > t0 && block_position < t0 + length) {
                length = block_position - t0;
            }
            // The boundary is where this prompt stopped agreeing with the lane's history, so the chunk
            // that would step over it ends on it: an image captured there is only loadable if it sits
            // exactly on that token. One extra short chunk per request is the price.
            if (retention(lane_id).reuse_boundary_target > t0 &&
                retention(lane_id).reuse_boundary_target < t0 + length) {
                length = retention(lane_id).reuse_boundary_target - t0;
            }
            qwen::execution::Tp2VisionChunk media_chunk;
            const qwen::execution::Tp2VisionChunk* media_ptr = nullptr;
            if (media) {
                media_chunk.control       = vision_chunk.control;
                media_chunk.embeddings    = &vision_chunk.embeddings;
                media_chunk.positions     = data.positions.data();
                media_chunk.prompt_tokens = data.token_ids.size();
                media_ptr                 = &media_chunk;
            }
            auto scope_a               = ws_a.scope();
            auto scope_b               = ws_b.scope();
            Tensor logits_a            = ws_a.alloc(DType::BF16, {vocab, 1});
            Tensor logits_b            = ws_b.alloc(DType::BF16, {vocab, 1});
            // MTP priming consumes the chunk's final-norm hidden, so the forward hands it back.
            Tensor mtp_input_a;
            if (mtp_enabled_) {
                mtp_input_a = ws_a.alloc(DType::BF16, {hidden, static_cast<std::int32_t>(length)});
            }
            // The masked draft taps this shard's prefill residual; the ring slot is this lane's own.
            auto dflash_sink = make_dflash_prefill_sink(shard_a_, active_lane_);
            shard_a_.device.bind_to_current_thread();
            ctx_a.forward_tp2_prefill(
                ctx_b, pair_, std::span<const int>(token_ids.data() + t0, length),
                static_cast<std::int32_t>(t0), &logits_a, &logits_b,
                mtp_enabled_ ? &mtp_input_a : nullptr, nullptr, nullptr, qwen::TextPhase::Prefill,
                media_ptr, dflash_sink ? &*dflash_sink : nullptr, active_lane_);
            if (dflash_sink) { retention(lane_id).dflash_context_frontier = t0 + length; }
            if (host_checkpoint_stride_ != 0) {
                // One checkpoint per stride, tagged with the frontier this chunk actually reached, so
                // a chunk width that does not divide the stride cannot mislabel a state; plus the
                // dense tail window. The prompt end is skipped either way: the lane publishes it as
                // its frontier into the ring's own prompt-end slot.
                const std::uint32_t frontier = t0 + length;
                if (!retention(lane_id).reuse_boundary_captured &&
                    retention(lane_id).reuse_boundary_target != 0 &&
                    frontier >= retention(lane_id).reuse_boundary_target) {
                    capture_reuse_boundary(lane_id);
                }
                if (frontier >= next_host_checkpoint) {
                    snapshot_host_checkpoint(shard_a_, frontier, HostRing::Grid, lane_id);
                    snapshot_host_checkpoint(shard_b_, frontier, HostRing::Grid, lane_id);
                    next_host_checkpoint =
                        (frontier / host_checkpoint_stride_ + 1U) * host_checkpoint_stride_;
                } else if (frontier != lane.prompt_tokens &&
                           static_cast<std::uint64_t>(frontier) + tail_span > lane.prompt_tokens) {
                    snapshot_host_checkpoint(shard_a_, frontier, HostRing::Tail, lane_id);
                    snapshot_host_checkpoint(shard_b_, frontier, HostRing::Tail, lane_id);
                }
                if (anchor_divergence && frontier == anchor_position) {
                    snapshot_host_checkpoint(shard_a_, frontier, HostRing::Divergence, lane_id);
                    snapshot_host_checkpoint(shard_b_, frontier, HostRing::Divergence, lane_id);
                }
                if (block_anchor && frontier == block_position) {
                    snapshot_host_checkpoint(shard_a_, frontier, HostRing::Block, lane_id);
                    snapshot_host_checkpoint(shard_b_, frontier, HostRing::Block, lane_id);
                    // The id names the prefill that wrote it, so an eviction can tell this
                    // conversation's own block state from a later conversation's rewrite of the slot.
                    retention(lane_id).block_anchor_position   = block_position;
                    retention(lane_id).block_anchor_prefill_id =
                        retention(lane_id).host_checkpoint_live_id;
                }
            }
            if (mtp_enabled_ && t0 + length != lane.prompt_tokens) {
                // The prompt's last column is primed after its first token exists, below.
                mtp_prefill_priming(shard_a_, token_ids.data() + t0, length, t0, mtp_input_a,
                                    nullptr, false, active_lane_);
            }
            if (t0 + length == lane.prompt_tokens) {
                Tensor logical_pos_lane = ws_a.alloc(DType::I32, {1});
                shard_a_.device.bind_to_current_thread();
                ops::set_i32_scalar(logical_pos_lane, static_cast<std::int32_t>(lane.prompt_tokens),
                                    shard_a_.device.stream);
                // P2.1: the same first-token mask the serial walk applies, on this lane's own
                // grammar.
                if (grammar_live(grammar[index], request)) {
                    grammar_advance(grammar[index], request);
                    if (grammar[index].constraint->build_mask(logits_domain, tool_mask_one)) {
                        Tensor tool_mask_first = ws_a.alloc(DType::U8, {vocab, 1});
                        shard_a_.device.bind_to_current_thread();
                        CUDA_CHECK(cudaMemcpyAsync(tool_mask_first.data, tool_mask_one.data(),
                                                   tool_mask_one.size(), cudaMemcpyHostToDevice,
                                                   shard_a_.device.stream));
                        ops::apply_token_mask(logits_a, tool_mask_first, shard_a_.device.stream);
                    }
                }
                Tensor sampled_a = ws_a.alloc(DType::I32, {1});
                ops::sample(logits_a, sampled_a, public_tokens,
                            static_cast<const ops::SamplingConfig*>(configs_dev.data) + lane.slot,
                            logical_pos_lane, ops::kSamplePurposePrefill, ws_a,
                            shard_a_.device.stream);
                // The exactness oracle: the prefill's own logits column, so a recall can be compared
                // against a recomputation instead of against the token it happened to sample. Greedy
                // output can hide a small state error in one run and expose it in another; the logit is
                // the quantity that says whether the restored state is the same state.
                if (const char* dump = std::getenv("NINFER_TP2_LOGIT_DUMP");
                    dump != nullptr && dump[0] == '1') {
                    std::vector<std::uint16_t> codes(static_cast<std::size_t>(vocab), 0U);
                    CUDA_CHECK(cudaMemcpyAsync(codes.data(), logits_a.data,
                                               codes.size() * sizeof(std::uint16_t),
                                               cudaMemcpyDeviceToHost, shard_a_.device.stream));
                    CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
                    std::array<float, 8> value{};
                    std::array<std::uint32_t, 8> id{};
                    value.fill(-3.0e38F);
                    for (std::size_t token = 0; token < codes.size(); ++token) {
                        const std::uint32_t wide = static_cast<std::uint32_t>(codes[token]) << 16;
                        float logit              = 0.0F;
                        std::memcpy(&logit, &wide, sizeof(logit));
                        if (logit <= value[7]) { continue; }
                        std::size_t place = 7;
                        while (place > 0 && value[place - 1] < logit) {
                            value[place] = value[place - 1];
                            id[place]    = id[place - 1];
                            --place;
                        }
                        value[place] = logit;
                        id[place]    = static_cast<std::uint32_t>(token);
                    }
                    std::fprintf(stderr, "[tp2-logits] pos=%u top=", t0 + length);
                    for (std::size_t place = 0; place < 8; ++place) {
                        std::fprintf(stderr, "%u:%.4f ", id[place],
                                     static_cast<double>(value[place]));
                    }
                    std::fprintf(stderr, "\n");
                }
                std::int32_t first = 0;
                CUDA_CHECK(cudaMemcpyAsync(&first, sampled_a.data, sizeof(std::int32_t),
                                           cudaMemcpyDeviceToHost, shard_a_.device.stream));
                CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
                abort_if_ar_stalled();
                const TokenId first_token       = static_cast<TokenId>(first);
                const std::uint32_t first_budget = request.budget.remaining();
                if (first_budget == 0) {
                    throw std::logic_error("prefill sampled a token with no output budget left");
                }
                const OutputDecision first_decision = request.output.preview_model(
                    std::span<const TokenId>(&first_token, 1), first_budget,
                    request.budget.limit_reason());
                if (first_decision.accepted_tokens != 1) {
                    throw std::logic_error("output policy rejected the prefill's first token");
                }
                request.generated.push_back(first_token);
                request.budget.commit(1);
                lane.have_first = true;
                lane.first_token = Clock::now();
                lane.current     = static_cast<std::int32_t>(first_token);
                lane.result.timings.prompt_wall_seconds =
                    std::chrono::duration<double>(lane.first_token - lane.begin).count();
                // The Vision encode is reported separately from the text walk, matching the serial
                // walk and the single-device route; prompt wall time (the response's TTFT) is the
                // whole span either way.
                lane.result.timings.vision_seconds =
                    vision_session ? vision_session->elapsed_seconds() : 0.0;
                lane.result.timings.prefill_seconds =
                    std::max(0.0, lane.result.timings.prompt_wall_seconds -
                                      lane.result.timings.vision_seconds);
                publish_preview(lane, false);
                if (mtp_enabled_) {
                    // The final MTP column embeds the token just sampled, so the MTP layer's own K/V
                    // for the prompt is appended only after the first token exists.
                    mtp_prefill_priming(shard_a_, token_ids.data() + t0, length, t0, mtp_input_a,
                                        &sampled_a, true, active_lane_);
                }
                if (first_decision.finished()) {
                    first_token_finish        = first_decision.finish_reason;
                    lane.result.finish_reason = first_decision.finish_reason;
                }
            }
            t0 += length;
            prefilled = t0;
        }
        if (vision_session) {
            // Every item this lane's prefill overlapped is encoded and its embeddings are in the KV
            // now, so release the host patch payloads and the handoff binding: the decode loop never
            // revisits them. The arena itself is reused by the next lane's session.
            vision_session->release_encoded_media_payloads();
            vision_session->retire_handoff();
        }
        if (cancelled) {
            // A prefill that completed at least one chunk published what it reached, the way the walk
            // does: the retry of the same prompt then continues from there. Before the first chunk
            // nothing moved, and the recall's bookkeeping already describes the device pools, so that
            // case retires the lineage instead.
            if (prefilled > 0) {
                publish_partial_prefill(lane_id, prefilled, tokens, data);
            } else {
                invalidate_lane_prefill(lane_id);
            }
            (void)request.output.preview_terminal(FinishReason::Cancelled);
            lane.result.finish_reason = FinishReason::Cancelled;
            publish_preview(lane, false);
            finalize(lane);
            return false;
        }
        // The anchor goes into the owning entry's shared image before the publish below, which can
        // evict that entry and shift every later index.
        if (anchor_divergence) { capture_lane_anchor_frozen(lane_id, anchor_position); }
        computed_prefill_tokens_ += lane.prompt_tokens - reuse;
        record_reuse(lane.result);
        publish_lane_prefill(lane_id, lane.prompt_tokens, tokens, prompt_media, data.context_cache);
        lane.prefilled = true;
        lane.finished   = first_token_finish != FinishReason::None;
        if (lane.finished) {
            finalize(lane);
            return false;
        }
        return true;
    };

    // S1: a member only takes its lane once the pages its own prompt plus output budget need are
    // available. The driver already collected the batch, so a member the shared pool cannot serve
    // right now goes back to the head of the queue and leaves the batch - the driver's safety net
    // publishes whatever is still in `batch`, and a requeued member must not be published empty.
    for (std::size_t index = 0; index < batch.size();) {
        const std::uint32_t need_tokens = lane_need_tokens(*batch[index]);
        if (!reserve_lane_kv(static_cast<std::uint32_t>(index), need_tokens)) {
            requeue_lane_front(std::move(batch[index]));
            batch.erase(batch.begin() + static_cast<std::ptrdiff_t>(index));
            continue;
        }
        if (admit_lane(index, *batch[index])) {
            lanes[index].kv_window = lane_kv_window(lane_kv_pages(need_tokens));
            active.push_back(index);
        }
        ++index;
    }


    for (;;) {
        // Retirement scan: a lane that ran out of budget or was cancelled leaves the batch before
        // the round is shaped, so every frame row in this round belongs to a live lane.
        {
            std::vector<std::size_t> live;
            live.reserve(active.size());
            for (std::size_t index : active) {
                LaneState& lane  = lanes[index];
                Request& request = *lane.pending->request;
                if (lane.pending->cancellation.requested()) {
                    (void)request.output.preview_terminal(FinishReason::Cancelled);
                    lane.result.finish_reason = FinishReason::Cancelled;
                    publish_preview(lane, false);
                    finalize(lane);
                    continue;
                }
                if (request.budget.remaining() == 0) {
                    (void)request.output.preview_terminal(request.budget.limit_reason());
                    lane.result.finish_reason = request.budget.limit_reason();
                    publish_preview(lane, false);
                    finalize(lane);
                    continue;
                }
                live.push_back(index);
            }
            active.swap(live);
        }

        // P2.2: every slot this round left free takes the next request that arrived while the batch
        // was running, so a short request no longer waits for the long one beside it to finish.
        while (active.size() < lane_capacity) {
            std::shared_ptr<PendingRequest> next = try_pop_lane_queue();
            if (next == nullptr) { break; }
            std::size_t index = lane_capacity;
            for (std::size_t candidate = 0; candidate < lane_capacity; ++candidate) {
                if (std::find(active.begin(), active.end(), candidate) == active.end()) {
                    index = candidate;
                    break;
                }
            }
            if (index == lane_capacity) { break; }
            // S1: the request's own pages have to be available before it may take the lane. If the
            // shared pool cannot cover it right now it goes back to the head of the queue rather
            // than losing its place to the requests that arrived after it. It is never dropped: the
            // admission ceiling guarantees a request that fits the policy fits an empty pool, so
            // this only ever waits for another lane to retire.
            const std::uint32_t need_tokens = lane_need_tokens(*next);
            if (!reserve_lane_kv(static_cast<std::uint32_t>(index), need_tokens)) {
                requeue_lane_front(std::move(next));
                break;
            }
            batch.push_back(next);
            if (lane_trace_enabled()) {
                std::fprintf(stderr, "[tp2-lane] admit slot=%zu live=%zu\n", index,
                             active.size() + 1);
            }
            if (admit_lane(index, *next)) {
                lanes[index].kv_window = lane_kv_window(lane_kv_pages(need_tokens));
                active.push_back(index);
            }
        }
        if (active.empty()) { break; }

        pump_spec_round();
    }
    timing.report("spec-batch");
}


// ---------------------------------------------------------------------------------------------
// Cross-session KV retention
//
// The device KV page pool is materialized once at startup and reused in place, so it holds exactly
// one session at a time. Turning a client to another conversation therefore copies the resident
// session's KV and GDN state into host KV slabs and copies the returning session back, which
// costs one PCIe round trip instead of a full prefill. The catalog below owns both the token
// history that describes what the device pools must reproduce and the host slabs that hold it while
// another conversation is resident.
// ---------------------------------------------------------------------------------------------

namespace {

bool session_trace_enabled() {
    const char* env = std::getenv("NINFER_TP2_SESSION_TRACE");
    return env != nullptr && env[0] == '1';
}

// A session key is the client's own text and can be long: the trace keeps a bounded label so a line
// stays readable and two lines for the same conversation compare equal. '-' is an unnamed entry.
std::string session_key_label(const models::qwen3_5::PreparedSessionKey* key) {
    if (key == nullptr) { return "-"; }
    constexpr std::size_t kTraceKeyBytes = 48;
    const std::string_view view          = key->view();
    return view.size() <= kTraceKeyBytes ? std::string(view)
                                         : std::string(view.substr(0, kTraceKeyBytes)) + "...";
}

// Length of the token prefix two histories share. A rewrite of one conversation - a folded context,
// a re-rendered tail - keeps a long prefix, while a one-off call beside it (a title, review or
// summary request carrying the session's own header) opens on instructions of its own and covers
// neither history.
std::size_t shared_history_prefix(const std::vector<TokenId>& first,
                                  const std::vector<TokenId>& second) {
    const std::size_t limit = std::min(first.size(), second.size());
    std::size_t shared      = 0;
    while (shared < limit && first[shared] == second[shared]) { ++shared; }
    return shared;
}

// How much of the shorter of two histories a shared prefix may leave out and still mean "this turn
// continues that entry". A folded context shortens the tail and a re-rendered generated reply
// changes it, both inside a bounded tail, while a one-off call beside the conversation diverges
// where its own instructions begin - which is not a tail.
constexpr std::size_t kSessionRewrittenTailTokens = 256;

} // namespace

void TP2GenerationCore::abort_if_ar_stalled() {
    if (!pair_.in_kernel_allreduce()) { return; }
    // Healthy path: one mapped host load, no stream touched. The trip flag, not the counter skew, is
    // the trigger: the decode paths drain only shard A before reading their sample, so the mirroring
    // shard is legitimately a call or two behind at that point and its skew would false-positive.
    if (!pair_.ar_stalled()) { return; }
    const std::uint64_t stalled_id = pair_.ar_last_id();
    for (Shard* shard : {&shard_a_, &shard_b_}) {
        shard->device.bind_to_current_thread();
        CUDA_CHECK(cudaStreamSynchronize(shard->device.stream));
    }
    shard_a_.device.bind_to_current_thread();
    pair_.clear_ar_stall();
    invalidate_host_checkpoints();
    session_invalidate_all();
    throw RequestError(
        RequestErrorKind::Unavailable,
        "TP-2 allreduce stalled at rendezvous id " + std::to_string(stalled_id) +
            ": the request was failed and the reusable context was discarded");
}

TP2GenerationCore::LaneStateGeometry TP2GenerationCore::make_lane_state_geometry(const Shard& shard) {
    LaneStateGeometry geometry;
    if (shard.state == nullptr) { return geometry; }
    const LinearAttentionStateSlotView view = shard.state->slot_view(0);
    const std::byte* const base = static_cast<const std::byte*>(shard.state_backing.data);
    geometry.conv_bytes      = view.conv_layer_bytes;
    geometry.recurrent_bytes = view.recurrent_layer_bytes;
    geometry.conv_base       = static_cast<const std::byte*>(view.conv_layer0.data) - base;
    geometry.recurrent_base  = static_cast<const std::byte*>(view.recurrent_layer0.data) - base;
    geometry.conv_pitch      = view.conv_layer_pitch_bytes;
    geometry.recurrent_pitch = view.recurrent_layer_pitch_bytes;
    geometry.layers          = view.layers;
    geometry.image_bytes     = static_cast<std::size_t>(geometry.layers) *
                               (geometry.conv_bytes + geometry.recurrent_bytes);
    // copy_lane_state derives a lane's device address arithmetically instead of asking the pool
    // for it, so the geometry it assumes has to be the geometry the pool has. NInfer tensors vary
    // dim 0 fastest, so the slot -- the last dimension of both state shapes -- varies slowest and a
    // lane is one contiguous block per layer, with the layers one pitch apart. Check that against
    // the pool once per shard: a pool that stopped satisfying it would otherwise let every lane
    // but the first read and write a neighbour's state, silently.
    if (geometry.image_bytes == 0) { return geometry; }
    for (std::int32_t lane = 0; lane < shard.state->slot_count(); ++lane) {
        const std::ptrdiff_t lane_conv =
            static_cast<std::ptrdiff_t>(lane) * static_cast<std::ptrdiff_t>(geometry.conv_bytes);
        const std::ptrdiff_t lane_recurrent = static_cast<std::ptrdiff_t>(lane) *
                                              static_cast<std::ptrdiff_t>(geometry.recurrent_bytes);
        for (std::uint32_t layer = 0; layer < geometry.layers; ++layer) {
            const Tensor conv      = shard.state->conv_slot(layer, lane);
            const Tensor recurrent = shard.state->recurrent_slot(layer, lane);
            const std::ptrdiff_t layer_conv =
                static_cast<std::ptrdiff_t>(layer) * geometry.conv_pitch;
            const std::ptrdiff_t layer_recurrent =
                static_cast<std::ptrdiff_t>(layer) * geometry.recurrent_pitch;
            if (static_cast<const std::byte*>(conv.data) !=
                    base + geometry.conv_base + lane_conv + layer_conv ||
                static_cast<const std::byte*>(recurrent.data) !=
                    base + geometry.recurrent_base + lane_recurrent + layer_recurrent ||
                conv.bytes() != geometry.conv_bytes ||
                recurrent.bytes() != geometry.recurrent_bytes) {
                throw std::logic_error(
                    "TP-2 lane state geometry disagrees with the linear-attention pool: lane " +
                    std::to_string(lane) + " layer " + std::to_string(layer) +
                    " is not one contiguous state image per layer at the pool's layer pitch");
            }
        }
    }
    return geometry;
}

// A lane's GDN image is one contiguous block per layer inside the pool, and those blocks are
// separated by the pool's layer pitch, but the compact pinned image is those blocks back to back,
// so each half moves with a strided 2-D copy rather than one flat memcpy. The device-to-device
// branch writes into its first device pointer and reads from its second.
void TP2GenerationCore::copy_lane_state(const LaneStateGeometry& geometry, const void* device_base,
                                        std::int32_t lane, void* other, cudaMemcpyKind kind,
                                        cudaStream_t stream) {
    if (geometry.image_bytes == 0) { return; }
    const std::size_t offset = static_cast<std::size_t>(lane);
    std::byte* const base = static_cast<std::byte*>(const_cast<void*>(device_base));
    std::byte* const device_conv      = base + geometry.conv_base + offset * geometry.conv_bytes;
    std::byte* const device_recurrent = base + geometry.recurrent_base + offset * geometry.recurrent_bytes;
    if (kind == cudaMemcpyDeviceToDevice) {
        // Both sides are whole pools with the same layout, so the lane's slice is strided here too.
        // This writes into `device_base` and reads from `other`: a restore passes the pool first,
        // a snapshot passes the plane first.
        const std::byte* const source = static_cast<const std::byte*>(other);
        CUDA_CHECK(cudaMemcpy2DAsync(
            device_conv, geometry.conv_pitch,
            source + geometry.conv_base + offset * geometry.conv_bytes, geometry.conv_pitch,
            geometry.conv_bytes, geometry.layers, kind, stream));
        CUDA_CHECK(cudaMemcpy2DAsync(
            device_recurrent, geometry.recurrent_pitch,
            source + geometry.recurrent_base + offset * geometry.recurrent_bytes,
            geometry.recurrent_pitch, geometry.recurrent_bytes, geometry.layers, kind, stream));
        return;
    }
    std::byte* const host = static_cast<std::byte*>(other);
    std::byte* const host_recurrent =
        host + static_cast<std::size_t>(geometry.layers) * geometry.conv_bytes;
    if (kind == cudaMemcpyDeviceToHost) {
        CUDA_CHECK(cudaMemcpy2DAsync(host, geometry.conv_bytes, device_conv, geometry.conv_pitch,
                                     geometry.conv_bytes, geometry.layers, kind, stream));
        CUDA_CHECK(cudaMemcpy2DAsync(host_recurrent, geometry.recurrent_bytes, device_recurrent,
                                     geometry.recurrent_pitch, geometry.recurrent_bytes,
                                     geometry.layers, kind, stream));
        return;
    }
    CUDA_CHECK(cudaMemcpy2DAsync(device_conv, geometry.conv_pitch, host, geometry.conv_bytes,
                                 geometry.conv_bytes, geometry.layers, kind, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(device_recurrent, geometry.recurrent_pitch, host_recurrent,
                                 geometry.recurrent_bytes, geometry.recurrent_bytes,
                                 geometry.layers, kind, stream));
}

void TP2GenerationCore::zero_lane_state(const LaneStateGeometry& geometry, void* device_base,
                                        std::int32_t lane, cudaStream_t stream) {
    if (geometry.image_bytes == 0) { return; }
    std::byte* const base = static_cast<std::byte*>(device_base);
    std::byte* const device_conv =
        base + geometry.conv_base + static_cast<std::size_t>(lane) * geometry.conv_bytes;
    std::byte* const device_recurrent =
        base + geometry.recurrent_base + static_cast<std::size_t>(lane) * geometry.recurrent_bytes;
    CUDA_CHECK(cudaMemset2DAsync(device_conv, geometry.conv_pitch, 0, geometry.conv_bytes,
                                 geometry.layers, stream));
    CUDA_CHECK(cudaMemset2DAsync(device_recurrent, geometry.recurrent_pitch, 0,
                                 geometry.recurrent_bytes, geometry.layers, stream));
}

void TP2GenerationCore::invalidate_host_checkpoints() {
    for (std::uint32_t lane = 0; lane < lanes_; ++lane) { invalidate_host_checkpoints(lane); }
}

void TP2GenerationCore::invalidate_host_checkpoints(std::uint32_t lane) {
    // The ring is sliced by lane, so one lane's stale lineage says nothing about another's.
    Shard* const shards[2] = {&shard_a_, &shard_b_};
    for (Shard* shard : shards) {
        for (auto& checkpoint : shard->host_checkpoints) { checkpoint.valid[lane] = false; }
    }
}

void TP2GenerationCore::session_invalidate_active(std::uint32_t lane) {
    RetentionState& lane_state = retention(lane);
    if (lane_state.active_session != kNoSession) { session_drop(lane_state.active_session); }
    lane_state.active_session     = kNoSession;
    lane_state.live_state_valid   = false;
    lane_state.cached_state_valid = false;
    lane_state.cached_prompt_tokens.clear();
    lane_state.cached_media.clear();
    invalidate_host_checkpoints(lane);
    lane_state.reuse_source = ReuseSource::None;
}

void TP2GenerationCore::session_invalidate_all() {
    for (std::uint32_t lane = 0; lane < lanes_; ++lane) { session_invalidate_active(lane); }
}

void TP2GenerationCore::session_drop(std::size_t index) {
    if (index >= sessions_.size()) { return; }
    sessions_.erase(sessions_.begin() + static_cast<std::ptrdiff_t>(index));
    // Every lane's claim on a catalog entry is an index into the same vector, so each one has to be
    // told the vector shifted. Both the resident claim and the comparison anchor are adjusted: the
    // only reader of the anchor in the current flow (capture_lane_anchor_frozen) lands before the
    // store that can drop, so a stale anchor is not reachable today - but leaving one behind is one
    // reordering away from freezing state into the wrong entry, and keeping the shift consistent
    // here costs nothing.
    for (RetentionState& lane_state : lane_retention_) {
        if (lane_state.active_session == index) {
            lane_state.active_session = kNoSession;
        } else if (lane_state.active_session != kNoSession && lane_state.active_session > index) {
            --lane_state.active_session;
        }
        if (lane_state.anchor_session == index) {
            lane_state.anchor_session = kNoSession;
        } else if (lane_state.anchor_session != kNoSession && lane_state.anchor_session > index) {
            --lane_state.anchor_session;
        }
    }
}

std::uint32_t TP2GenerationCore::private_retention_weight(RetentionClass retention) noexcept {
    // The single-GPU context cache's weights, so both routes shed the same conversation first.
    switch (retention) {
    case RetentionClass::SharedStable: return 0;
    case RetentionClass::Disposable: return 1;
    case RetentionClass::RecentPrivate: return 4;
    case RetentionClass::LiveSession: return 16;
    }
    return 0;
}

void TP2GenerationCore::demote_session_owners(
    const models::qwen3_5::PreparedSessionKey& key) {
    for (SessionEntry& entry : sessions_) {
        if (!entry.session || !(*entry.session == key)) { continue; }
        entry.session.reset();
        entry.retention_weight = private_retention_weight(RetentionClass::RecentPrivate);
    }
}

std::size_t TP2GenerationCore::session_owner_index(
    const models::qwen3_5::PreparedSessionKey& key) const {
    // Newest first: a conversation's owner is the entry the next turn extends, and the prune below
    // keeps that entry the only one carrying the name.
    for (std::size_t index = sessions_.size(); index-- > 0;) {
        const SessionEntry& entry = sessions_[index];
        if (entry.session && *entry.session == key) { return index; }
    }
    return kNoSession;
}

std::size_t TP2GenerationCore::session_prune_keyed(
    const models::qwen3_5::PreparedSessionKey& key, std::size_t keep) {
    for (std::size_t index = 0; index < sessions_.size();) {
        const SessionEntry& entry = sessions_[index];
        const bool replaced_owner = index != keep && entry.device_lane < 0 && entry.session &&
                                    *entry.session == key;
        if (!replaced_owner) {
            ++index;
            continue;
        }
        session_drop(index);
        // session_drop erases and shifts whatever lane state pointed past 'index', and this loop's
        // own keep-index has to follow it.
        if (keep != kNoSession && keep > index) { --keep; }
    }
    return keep;
}

void TP2GenerationCore::bind_entry_session(
    SessionEntry& entry, const models::qwen3_5::PreparedContextCache& cache_hints) {
    if (!cache_hints.session_key) {
        // An unnamed request makes the entry anonymous: the client never said it would return.
        entry.session.reset();
    } else {
        // Only the newest entry of a conversation is its session owner, because that is the entry
        // the next turn extends; the previous one becomes an ordinary continuation.
        if (cache_hints.update_session_index) { demote_session_owners(*cache_hints.session_key); }
        entry.session = *cache_hints.session_key;
    }
    entry.retention_weight = private_retention_weight(cache_hints.retention);
}

bool TP2GenerationCore::session_evict_one(std::uint32_t publisher_weight) {
    // Lowest retention weight first, least recently used within a weight. A conversation the client
    // named is retained against the anonymous ones around it, which is what keeps a long agent
    // session reusable while one-off writes keep taking the device pools.
    std::size_t victim = kNoSession;
    for (std::size_t index = 0; index < sessions_.size(); ++index) {
        if (sessions_[index].device_lane >= 0) { continue; }
        if (sessions_[index].retention_weight > publisher_weight) { continue; }
        if (victim == kNoSession ||
            sessions_[index].retention_weight < sessions_[victim].retention_weight ||
            (sessions_[index].retention_weight == sessions_[victim].retention_weight &&
             sessions_[index].lru_clock < sessions_[victim].lru_clock)) {
            victim = index;
        }
    }
    if (victim == kNoSession) { return false; }
    if (session_trace_enabled()) {
        std::fprintf(stderr, "[tp2-session] evict frontier=%u tokens=%zu clock=%llu weight=%u\n",
                     sessions_[victim].frontier, sessions_[victim].tokens.size(),
                     static_cast<unsigned long long>(sessions_[victim].lru_clock),
                     sessions_[victim].retention_weight);
    }
    session_drop(victim);
    ++session_evictions_;
    return true;
}

bool TP2GenerationCore::session_ensure_host_slabs(SessionEntry& entry, std::uint32_t pages) {
    if (pages == 0) { return false; }
    Shard* const shards[2] = {&shard_a_, &shard_b_};
    for (std::size_t index = 0; index < 2; ++index) {
        Shard& shard = *shards[index];
        // The arena pins its whole budget on construction, so a workload that never leaves one
        // conversation never pays for the capability.
        if (host_kv_arena_[index] == nullptr) {
            std::vector<HostKVPageLayout> layouts;
            layouts.push_back(plan_host_kv_page_layout(shard.decoder->text_kv.page_pool().geometry()));
            if (index == 0 && mtp_enabled_) {
                const qwen::PagedKVCache* mtp = shard.decoder->mtp_cache();
                if (mtp == nullptr) { return false; }
                layouts.push_back(plan_host_kv_page_layout(mtp->page_pool().geometry()));
            }
            shard.device.bind_to_current_thread();
            try {
                host_kv_arena_[index] = std::make_unique<HostKVArena>(
                    host_kv_shard_bytes_, layouts,
                    options_.context_cache.host_kv_pinned ? HostPinning::PreferPinned
                                                          : HostPinning::Pageable);
            } catch (const std::exception& error) {
                // A failed allocation can leave cudaErrorMemoryAllocation latched, where the next
                // unrelated CUDA_CHECK would report it as that call's own failure; report both halves
                // here, because a silent failure would degrade every later request to a full prefill
                // with no other trace. A refused pin does not reach this path: the backing store falls
                // back to pageable memory on its own.
                const cudaError_t latched = cudaGetLastError();
                std::fprintf(stderr,
                             "[tp2-session] host KV arena shard %zu refused: %.1f MiB (%s): "
                             "%s\n",
                             index, static_cast<double>(host_kv_shard_bytes_) / 1048576.0,
                             cudaGetErrorString(latched), error.what());
                host_kv_arena_[index].reset();
                return false;
            }
            std::fprintf(stderr,
                         "[tp2-session] host KV arena shard %zu: %.1f MiB %s, %zu layouts\n",
                         index, static_cast<double>(host_kv_shard_bytes_) / 1048576.0,
                         host_kv_arena_[index]->backing_pinned() ? "pinned" : "pageable",
                         layouts.size());
        }
        const HostKVPageLayout* layout =
            host_kv_arena_[index]->layout_for(shard.decoder->text_kv.page_pool().geometry());
        if (layout == nullptr) { return false; }
        // The GDN state image is independent of the KV extent: it is always one full state per
        // shard, and it survives a frontier that a later eviction grows.
        if (entry.host_state[index] == nullptr) {
            shard.device.bind_to_current_thread();
            try {
                entry.host_state[index] =
                    std::make_unique<PinnedHostBuffer>(shard.state_backing.bytes, true);
            } catch (const std::exception& error) {
                const cudaError_t latched = cudaGetLastError();
                std::fprintf(stderr,
                             "[tp2-session] host state image shard %zu refused: %.1f MiB pinned "
                             "(%s): %s\n",
                             index, static_cast<double>(shard.state_backing.bytes) / 1048576.0,
                             cudaGetErrorString(latched), error.what());
                return false;
            }
        }
        // The masked draft's context image travels with the target state image at every boundary the
        // entry can be recalled on. The frontier one is mandatory: a recall that restored the target
        // state without the draft ring would leave the draft describing tokens before the boundary,
        // and the skipped prefix produces no target residual to rebuild it, so the entry would not be
        // safe to recall at all. A shard that owns no draft (shard 1, and every other backend) skips
        // all three.
        if (shard.dflash_round != nullptr) {
            // One compact ring image per lane, so a lane restores its own slice of the slab exactly
            // as it does for the target state image above.
            const std::size_t dflash_bytes =
                static_cast<std::size_t>(lanes_) * shard.dflash_round->lane_context_image_bytes();
            shard.device.bind_to_current_thread();
            if (entry.host_dflash[index] == nullptr) {
                try {
                    entry.host_dflash[index] =
                        std::make_unique<PinnedHostBuffer>(dflash_bytes, true);
                } catch (const std::exception& error) {
                    const cudaError_t latched = cudaGetLastError();
                    std::fprintf(stderr,
                                 "[tp2-session] host draft context image shard %zu refused: %.1f MiB "
                                 "pinned (%s): %s\n",
                                 index, static_cast<double>(dflash_bytes) / 1048576.0,
                                 cudaGetErrorString(latched), error.what());
                    return false;
                }
            }
            if (entry.host_dflash_prompt[index] == nullptr) {
                try {
                    entry.host_dflash_prompt[index] =
                        std::make_unique<PinnedHostBuffer>(dflash_bytes, true);
                } catch (const std::exception& error) {
                    const cudaError_t latched = cudaGetLastError();
                    std::fprintf(stderr,
                                 "[tp2-session] warning: host prompt-end draft image shard %zu "
                                 "refused: %.1f MiB pinned (%s): %s; this session needs the generated "
                                 "tail reproduced to be recalled\n",
                                 index, static_cast<double>(dflash_bytes) / 1048576.0,
                                 cudaGetErrorString(latched), error.what());
                }
            }
            if (entry.host_dflash_shared[index] == nullptr) {
                try {
                    entry.host_dflash_shared[index] =
                        std::make_unique<PinnedHostBuffer>(dflash_bytes, true);
                } catch (const std::exception& error) {
                    const cudaError_t latched = cudaGetLastError();
                    std::fprintf(stderr,
                                 "[tp2-session] warning: host shared-prefix draft image shard %zu "
                                 "refused: %.1f MiB pinned (%s): %s; this session will not carry a "
                                 "stable prefix for the next conversation\n",
                                 index, static_cast<double>(dflash_bytes) / 1048576.0,
                                 cudaGetErrorString(latched), error.what());
                }
            }
        }
        // The two extra images are best effort: they make a returning conversation cheaper, but the
        // frontier image above is what makes it recallable at all. A shortage of pinned memory
        // therefore degrades the recall instead of dropping the conversation.
        if (entry.host_prompt_state[index] == nullptr) {
            shard.device.bind_to_current_thread();
            try {
                entry.host_prompt_state[index] =
                    std::make_unique<PinnedHostBuffer>(shard.state_backing.bytes, true);
            } catch (const std::exception& error) {
                const cudaError_t latched = cudaGetLastError();
                std::fprintf(stderr,
                             "[tp2-session] warning: host prompt-end state image shard %zu refused: "
                             "%.1f MiB pinned (%s): %s; this session needs the generated tail "
                             "reproduced to be recalled\n",
                             index, static_cast<double>(shard.state_backing.bytes) / 1048576.0,
                             cudaGetErrorString(latched), error.what());
            }
        }
        if (entry.host_shared_state[index] == nullptr) {
            shard.device.bind_to_current_thread();
            try {
                entry.host_shared_state[index] =
                    std::make_unique<PinnedHostBuffer>(shard.state_backing.bytes, true);
            } catch (const std::exception& error) {
                const cudaError_t latched = cudaGetLastError();
                std::fprintf(stderr,
                             "[tp2-session] warning: host shared-prefix state image shard %zu "
                             "refused: %.1f MiB pinned (%s): %s; this session will not carry a "
                             "stable prefix for the next conversation\n",
                             index, static_cast<double>(shard.state_backing.bytes) / 1048576.0,
                             cudaGetErrorString(latched), error.what());
            }
        }
        if (entry.host_kv[index] != nullptr && entry.host_kv[index]->page_count() >= pages) {
            // A slab that already covers this frontier is reused as it is; only the first `pages`
            // pages carry the session, and the rest is stale.
            continue;
        }
        entry.host_kv[index].reset();
        std::optional<HostKVAllocation> allocation = host_kv_arena_[index]->allocate(*layout, pages);
        if (!allocation.has_value()) { return false; }
        entry.host_kv[index] = std::make_unique<HostKVAllocation>(std::move(*allocation));
    }
    if (mtp_enabled_) {
        const qwen::PagedKVCache* mtp = shard_a_.decoder->mtp_cache();
        const HostKVPageLayout* layout =
            host_kv_arena_[0]->layout_for(mtp->page_pool().geometry());
        if (layout == nullptr) { return false; }
        if (entry.host_mtp_kv == nullptr || entry.host_mtp_kv->page_count() < pages) {
            entry.host_mtp_kv.reset();
            std::optional<HostKVAllocation> allocation =
                host_kv_arena_[0]->allocate(*layout, pages);
            if (!allocation.has_value()) { return false; }
            entry.host_mtp_kv = std::make_unique<HostKVAllocation>(std::move(*allocation));
        }
        entry.host_mtp_pages = pages;
    }
    return true;
}

// At most one entry describes the conversation a lane's device state holds: a recall restores one
// and a publish installs one, and every other entry's claim on that lane is gone the moment that
// happens. The eviction scan skips entries resident on any lane because their KV never reached the
// slabs, so a stale lane makes an entry unevictable - and lets the scans believe a device still
// holds a conversation it has already overwritten.
template <typename Sessions>
static void mark_device_lane(Sessions& sessions, std::size_t index, std::int32_t lane) {
    for (std::size_t other = 0; other < sessions.size(); ++other) {
        if (other != index && sessions[other].device_lane == lane) { sessions[other].device_lane = -1; }
    }
    sessions[index].device_lane = lane;
}

bool TP2GenerationCore::session_store_active(std::uint32_t lane) {
    RetentionState& lane_state = retention(lane);
    if (lane_state.active_session == kNoSession) { return true; }
    SessionEntry& entry = sessions_[lane_state.active_session];
    if (entry.device_lane != static_cast<std::int32_t>(lane)) { return true; }
    const std::uint32_t pages = pages_for_tokens(entry.frontier);
    if (pages == 0) {
        // Nothing to copy, and nothing a recall could stand on either: clear the boundaries so a
        // later request cannot restore a state whose KV the device pools do not hold.
        entry.device_lane       = -1;
        entry.host_prompt_end   = 0;
        entry.host_shared_end   = 0;
        entry.host_frontier_end = 0;
        entry.host_kv_end       = 0;
        return true;
    }
    if (!session_ensure_host_slabs(entry, pages)) { return false; }
    Shard* const shards[2] = {&shard_a_, &shard_b_};
    // The prompt-end image the store publishes, per shard: the ring's PromptEnd checkpoint when the
    // ring exists, otherwise the device plane. The two are mutually exclusive - a plane the ring
    // serves is not carved - and the copies differ in direction, so the source is picked here and
    // consumed after the synchronize below.
    const Clock::time_point store_start = kv_trace_enabled() ? Clock::now() : Clock::time_point{};
    const Shard::HostCheckpoint* prompt_end_checkpoints[2] = {nullptr, nullptr};
    bool prompt_end_captured                              = true;
    for (std::size_t index = 0; index < 2; ++index) {
        Shard& shard = *shards[index];
        shard.device.bind_to_current_thread();
        // This lane's own KV row and GDN slot carry the conversation being stored.
        const std::span<const DeviceKVPageHandle> source(shard.kv_lane_handles[lane].data(), pages);
        const HostKVAllocationView view =
            host_kv_arena_[index]->writable_view(*entry.host_kv[index]).subview(0, pages);
        shard.decoder->text_kv.page_pool().copy_to_host(source, view, shard.device.stream);
        copy_lane_state(shard.lane_state_geometry, shard.state_backing.data,
                        static_cast<std::int32_t>(lane),
                        static_cast<std::byte*>(entry.host_state[index]->data()) +
                            static_cast<std::size_t>(lane) * shard.lane_state_geometry.image_bytes,
                        cudaMemcpyDeviceToHost, shard.device.stream);
        // The state the last completed prefill froze at this conversation's own prompt end. A client
        // that re-renders the answer it was handed sends a next prompt that contains that whole
        // prompt and then diverges inside the generated tail, so this image is the one its return
        // trip can stand on. The prefill published it as a checkpoint at that frontier and tagged it
        // with this lane's live prefill id, which is what tells this conversation's own image from a
        // later rewrite of the same slot.
        for (const Shard::HostCheckpoint& checkpoint : shard.host_checkpoints) {
            if (checkpoint.valid[lane] && checkpoint.position[lane] == entry.prompt_end &&
                checkpoint.prefill_id[lane] == lane_state.host_checkpoint_live_id) {
                prompt_end_checkpoints[index] = &checkpoint;
                break;
            }
        }
        if (entry.host_prompt_state[index] != nullptr &&
            prompt_end_checkpoints[index] == nullptr) {
            if (shard.state_snapshots[0].data != nullptr) {
                copy_lane_state(shard.lane_state_geometry, shard.state_snapshots[0].data,
                                static_cast<std::int32_t>(lane),
                                static_cast<std::byte*>(entry.host_prompt_state[index]->data()) +
                                    static_cast<std::size_t>(lane) *
                                        shard.lane_state_geometry.image_bytes,
                                cudaMemcpyDeviceToHost, shard.device.stream);
            } else {
                prompt_end_captured = false;
            }
        }
        // The masked draft's context rides every target image without an extent of its own: the
        // evicted frontier is the live ring, and the prompt-end image is the checkpoint above (or the
        // device plane the GDN copy just read when there is no ring). Both are the flat ring image,
        // so either source is one copy.
        if (shard.dflash_round != nullptr) {
            const std::size_t draft_lane_bytes = shard.dflash_round->lane_context_image_bytes();
            store_dflash_image(shard, entry.host_dflash[index]->data(),
                               entry.host_dflash[index]->size(), lane);
            if (entry.host_dflash_prompt[index] != nullptr &&
                prompt_end_checkpoints[index] == nullptr) {
                if (shard.dflash_snapshots[0].data != nullptr) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        static_cast<std::byte*>(entry.host_dflash_prompt[index]->data()) +
                            static_cast<std::size_t>(lane) * draft_lane_bytes,
                        static_cast<const std::byte*>(shard.dflash_snapshots[0].data) +
                            static_cast<std::size_t>(lane) * draft_lane_bytes, draft_lane_bytes,
                        cudaMemcpyDeviceToHost, shard.device.stream));
                } else {
                    prompt_end_captured = false;
                }
            }
        }
    }
    if (mtp_enabled_) {
        shard_a_.device.bind_to_current_thread();
        // The MTP KV this lane owns, not the whole table: each lane holds its own slice of the MTP
        // page pool, so the session slab is one lane's extent.
        const std::span<const DeviceKVPageHandle> source(shard_a_.mtp_lane_handles[lane].data(),
                                                         entry.host_mtp_pages);
        const HostKVAllocationView view =
            host_kv_arena_[0]->writable_view(*entry.host_mtp_kv).subview(0, entry.host_mtp_pages);
        shard_a_.decoder->mtp_cache()->page_pool().copy_to_host(source, view, shard_a_.device.stream);
    }
    // A copy still in flight is not state: the entry only becomes host-resident once both devices
    // have finished their D2H, so an error here leaves the session on the device instead.
    for (std::size_t index = 0; index < 2; ++index) {
        shards[index]->device.bind_to_current_thread();
        CUDA_CHECK(cudaStreamSynchronize(shards[index]->device.stream));
    }
    if (kv_trace_enabled()) {
        std::fprintf(
            stderr, "[tp2-kv] store lane=%u pages=%u frontier=%u runs=%u/%u ms=%.1f\n", lane,
            pages, entry.frontier,
            shard_a_.decoder->text_kv.page_pool().contiguous_run_count(
                shard_a_.kv_lane_handles[lane]),
            shard_b_.decoder->text_kv.page_pool().contiguous_run_count(
                shard_b_.kv_lane_handles[lane]),
            std::chrono::duration<double, std::milli>(Clock::now() - store_start).count());
    }
    // The prompt-end image is a host checkpoint when the ring exists, and the D2H that filled it rode
    // the shard stream, so the synchronize above is what orders these host-to-host moves after it.
    // Without the ring the device plane already supplied the image.
    for (std::size_t index = 0; index < 2; ++index) {
        const Shard::HostCheckpoint* checkpoint = prompt_end_checkpoints[index];
        if (checkpoint == nullptr) { continue; }
        Shard& shard = *shards[index];
        const std::size_t image_bytes = shard.lane_state_geometry.image_bytes;
        if (entry.host_prompt_state[index] != nullptr) {
            std::memcpy(static_cast<std::byte*>(entry.host_prompt_state[index]->data()) +
                            static_cast<std::size_t>(lane) * image_bytes,
                        checkpoint->buffer->data(), image_bytes);
        }
        // The checkpoint's draft half is one lane's compact image while the session slab holds one
        // image per lane, so only the slab side is a straight offset.
        if (shard.dflash_round != nullptr && entry.host_dflash_prompt[index] != nullptr &&
            checkpoint->dflash_buffer != nullptr) {
            const std::size_t draft_lane_bytes = shard.dflash_round->lane_context_image_bytes();
            std::memcpy(static_cast<std::byte*>(entry.host_dflash_prompt[index]->data()) +
                            static_cast<std::size_t>(lane) * draft_lane_bytes,
                        checkpoint->dflash_buffer->data(), draft_lane_bytes);
        }
    }

    shard_a_.device.bind_to_current_thread();
    entry.device_lane = -1;
    // The slabs now carry the KV up to the frontier they were filled at, which is the extent every
    // later capture and recall has to stay inside. A shared-prefix image this shorter extent no
    // longer reaches - a partial recall whose walk stopped before it - goes with them.
    entry.host_kv_end       = entry.frontier;
    entry.host_frontier_end = entry.frontier;
    if (entry.host_shared_end > entry.host_kv_end) { entry.host_shared_end = 0; }
    // The prompt-end boundary is only offered when its target image *and* its draft image exist; a
    // recall that restored one without the other would run the draft against the wrong tokens.
    const bool prompt_end_images =
        entry.host_prompt_state[0] != nullptr && entry.host_prompt_state[1] != nullptr &&
        (shard_a_.dflash_round == nullptr || entry.host_dflash_prompt[0] != nullptr);
    entry.host_prompt_end = (prompt_end_images && prompt_end_captured) ? entry.prompt_end : 0;
    // The block boundary this conversation's own prefill froze, while the ring still holds the walk
    // that wrote it: the slabs just took the KV that sits before the boundary, so pairing the frozen
    // state with them gives the entry a boundary the next conversation opening with the same block is
    // recalled on. Everything else the entry carries describes its own history; this describes the
    // part of it that outlives the conversation.
    if (lane_state.block_anchor_position != 0 && lane_state.block_anchor_position <= entry.frontier) {
        const void* frozen[2]   = {nullptr, nullptr};
        const void* block_draft = nullptr;
        Shard* const shards[2] = {&shard_a_, &shard_b_};
        for (std::size_t index = 0; index < 2; ++index) {
            for (const Shard::HostCheckpoint& checkpoint : shards[index]->host_checkpoints) {
                if (!checkpoint.valid[lane] ||
                    checkpoint.position[lane] != lane_state.block_anchor_position ||
                    checkpoint.prefill_id[lane] != lane_state.block_anchor_prefill_id ||
                    checkpoint.buffer == nullptr) {
                    continue;
                }
                frozen[index] = checkpoint.buffer->data();
                if (index == 0 && checkpoint.dflash_buffer != nullptr) {
                    block_draft = checkpoint.dflash_buffer->data();
                }
                break;
            }
        }
        if (frozen[0] != nullptr && frozen[1] != nullptr) {
            session_capture_shared_state(lane_state.active_session, lane_state.block_anchor_position, false, frozen,
                                         block_draft, lane);
        }
    }
    entry.lru_clock       = ++session_lru_clock_;
    ++session_stores_;
    return true;
}

void TP2GenerationCore::session_restore(SessionEntry& entry, std::uint32_t boundary,
                                        RecallState state, std::uint32_t lane) {
    const std::uint32_t pages = pages_for_tokens(boundary);
    if (pages == 0) {
        entry.tokens.clear();
        entry.frontier         = 0;
        entry.host_frontier_end = 0;
        mark_device_lane(sessions_, static_cast<std::size_t>(&entry - sessions_.data()),
                         static_cast<std::int32_t>(lane));
        return;
    }
    Shard* const shards[2] = {&shard_a_, &shard_b_};
    const Clock::time_point recall_start = kv_trace_enabled() ? Clock::now() : Clock::time_point{};
    for (std::size_t index = 0; index < 2; ++index) {
        Shard& shard = *shards[index];
        shard.device.bind_to_current_thread();
        const std::span<const DeviceKVPageHandle> destination(shard.kv_lane_handles[lane].data(),
                                                              pages);
        const HostKVAllocationConstView view =
            host_kv_arena_[index]->view(*entry.host_kv[index]).subview(0, pages);
        shard.decoder->text_kv.page_pool().copy_from_host(view, destination, shard.device.stream);

                // Every boundary is one compact lane image at a fixed offset, so the choice of which one to
        // load is a pointer, not a type: the lane-local boundary lives on the shard rather than in a
        // session entry.
        const void* frozen_image = nullptr;
        switch (state) {
            case RecallState::Frontier: frozen_image = entry.host_state[index]->data(); break;
            case RecallState::PromptEnd:
                frozen_image = entry.host_prompt_state[index]->data();
                break;
            case RecallState::Shared: frozen_image = entry.host_shared_state[index]->data(); break;
            case RecallState::Boundary: frozen_image = shard.reuse_boundary_state->data(); break;
        }
        copy_lane_state(shard.lane_state_geometry, shard.state_backing.data,
                        static_cast<std::int32_t>(lane),
                        static_cast<std::byte*>(const_cast<void*>(frozen_image)) +
                            static_cast<std::size_t>(lane) * shard.lane_state_geometry.image_bytes,
                        cudaMemcpyHostToDevice, shard.device.stream);
        // The draft context comes back with the target state it belongs to: the skipped prefix
        // produces no target residual, so nothing else rebuilds the ring, and a recall that restored
        // only the target half would verify against a draft describing the wrong tokens. A boundary is
        // only offered when its draft pair exists, so a null here means it never was.
        if (shard.dflash_round != nullptr) {
            const void* frozen_dflash = nullptr;
            std::size_t frozen_bytes  = 0;
            switch (state) {
                case RecallState::Frontier:
                    frozen_dflash = entry.host_dflash[index]->data();
                    frozen_bytes  = entry.host_dflash[index]->size();
                    break;
                case RecallState::PromptEnd:
                    frozen_dflash = entry.host_dflash_prompt[index]->data();
                    frozen_bytes  = entry.host_dflash_prompt[index]->size();
                    break;
                case RecallState::Shared:
                    frozen_dflash = entry.host_dflash_shared[index]->data();
                    frozen_bytes  = entry.host_dflash_shared[index]->size();
                    break;
                case RecallState::Boundary:
                    frozen_dflash = shard.reuse_boundary_dflash->data();
                    frozen_bytes  = shard.reuse_boundary_dflash->size();
                    break;
            }
            load_dflash_image(shard, frozen_dflash, frozen_bytes, lane);
        }
    }
    if (mtp_enabled_) {
        shard_a_.device.bind_to_current_thread();
        const std::span<const DeviceKVPageHandle> destination(shard_a_.mtp_lane_handles[lane].data(),
                                                              pages);
        const HostKVAllocationConstView view =
            host_kv_arena_[0]->view(*entry.host_mtp_kv).subview(0, pages);
        shard_a_.decoder->mtp_cache()->page_pool().copy_from_host(view, destination,
                                                                  shard_a_.device.stream);
    }
    for (std::size_t index = 0; index < 2; ++index) {
        shards[index]->device.bind_to_current_thread();
        CUDA_CHECK(cudaStreamSynchronize(shards[index]->device.stream));
    }
    if (kv_trace_enabled()) {
        const char* state_name = state == RecallState::Frontier  ? "frontier"
                                 : state == RecallState::PromptEnd ? "prompt-end"
                                                                   : "shared";
        std::fprintf(
            stderr, "[tp2-kv] recall lane=%u pages=%u boundary=%u state=%s runs=%u/%u ms=%.1f\n",
            lane, pages, boundary, state_name,
            shard_a_.decoder->text_kv.page_pool().contiguous_run_count(
                shard_a_.kv_lane_handles[lane]),
            shard_b_.decoder->text_kv.page_pool().contiguous_run_count(
                shard_b_.kv_lane_handles[lane]),
            std::chrono::duration<double, std::milli>(Clock::now() - recall_start).count());
    }
    shard_a_.device.bind_to_current_thread();
    // A prompt-end recall leaves the entry standing where that prompt ended: the device pools hold
    // its KV and the state at its end, and the tail the entry had generated past it is gone from the
    // device. The history itself stays: `tokens` is the conversation and `frontier` is how far the
    // device came back, and the prefix scan needs the whole history to find where the next prompt
    // diverges from this one - that boundary is what the divergence anchor is frozen at, and a
    // history clipped to the restored frontier hides it behind the recall. The slabs the recall read
    // still carry the KV this history describes, which is what a capture into the entry pairs with.
    entry.frontier = boundary;
    mark_device_lane(sessions_, static_cast<std::size_t>(&entry - sessions_.data()),
                         static_cast<std::int32_t>(lane));
    entry.lru_clock = ++session_lru_clock_;
    ++session_recalls_;
}

void TP2GenerationCore::session_capture_shared_state(std::size_t index, std::uint32_t position,
                                                     bool from_device, const void* const* frozen,
                                                     const void* dflash_frozen, std::uint32_t lane) {
    if (index >= sessions_.size() || position == 0) { return; }
    SessionEntry& entry = sessions_[index];
    // The image is the state of the tokens before 'position' paired with the KV the slabs hold
    // there, so the slabs have to reach the position and no image at or past it may exist yet. A
    // device-resident entry is a candidate: a recall that brought it back to a shallower boundary
    // left its slabs standing at the extent it was evicted with, and that is exactly the anchor's
    // case - the conversation this prompt diverged from is the one the slabs still carry.
    if (position <= entry.host_shared_end || position > entry.host_kv_end) { return; }
    Shard* const shards[2] = {&shard_a_, &shard_b_};
    for (std::size_t shard_index = 0; shard_index < 2; ++shard_index) {
        if (entry.host_shared_state[shard_index] == nullptr) { return; }
    }
    // The draft half of the anchor has no use without its own image, and a partial write would tear
    // an image a previous capture published under the same host_shared_end, so the requirement is
    // checked before the first copy, not during.
    const bool owns_draft = shard_a_.dflash_round != nullptr;
    if (owns_draft &&
        (entry.host_dflash_shared[0] == nullptr || (!from_device && dflash_frozen == nullptr))) {
        return;
    }
    for (std::size_t shard_index = 0; shard_index < 2; ++shard_index) {
        Shard& shard = *shards[shard_index];
        if (from_device) {
            shard.device.bind_to_current_thread();
            copy_lane_state(shard.lane_state_geometry, shard.state_backing.data,
                            static_cast<std::int32_t>(lane),
                            static_cast<std::byte*>(entry.host_shared_state[shard_index]->data()) +
                                static_cast<std::size_t>(lane) * shard.lane_state_geometry.image_bytes,
                            cudaMemcpyDeviceToHost, shard.device.stream);
            // The device draft ring sits at this boundary by construction: the walk restores it
            // there before the first chunk, which is when the from-device capture runs.
            if (shard.dflash_round != nullptr) {
                store_dflash_image(shard, entry.host_dflash_shared[shard_index]->data(),
                                   entry.host_dflash_shared[shard_index]->size(), lane);
            }
            continue;
        }
        if (frozen[shard_index] == nullptr) { return; }
        std::memcpy(static_cast<std::byte*>(entry.host_shared_state[shard_index]->data()) +
                        static_cast<std::size_t>(lane) * shard.lane_state_geometry.image_bytes,
                    frozen[shard_index], shard.lane_state_geometry.image_bytes);
        if (shard.dflash_round != nullptr) {
            // The frozen checkpoint's draft half is one lane's compact image while the session slab
            // holds one image per lane, so only the slab side is a straight offset.
            const std::size_t draft_lane_bytes = shard.dflash_round->lane_context_image_bytes();
            std::memcpy(static_cast<std::byte*>(entry.host_dflash_shared[shard_index]->data()) +
                            static_cast<std::size_t>(lane) * draft_lane_bytes,
                        dflash_frozen, draft_lane_bytes);
        }
    }
    entry.host_shared_end = position;
    // A capture writes this entry: the boundary it now carries is the freshest thing the catalog
    // holds, and evicting it in the same round would throw that work away before any conversation
    // could use it. Recency is what the LRU approximates, so the write refreshes it.
    entry.lru_clock = ++session_lru_clock_;
    if (session_trace_enabled()) {
        std::fprintf(stderr,
                     "[tp2-session] anchor entry=%zu position=%u resident=%d tokens=%zu\n", index,
                     position, entry.device_lane >= 0 ? 1 : 0, entry.tokens.size());
    }
}

void TP2GenerationCore::session_recall(std::uint32_t lane, std::span<const TokenId> prompt_tokens,
                                       std::span<const MediaSpan> media) {
    RetentionState& lane_state = retention(lane);
    if (session_capacity_ == 0) { return; }
    // The conversation this prompt is compared against is the only one that can still be told where
    // the two stopped matching, and the running prefill is the last walk that sees the shared state.
    // execute_walk reads this before it advances anything.
    lane_state.anchor_session = kNoSession;
    // How deep the resident lineage can start this prompt, mirroring what the reuse scan below
    // finds: the live state at its frontier, a device snapshot, or a host checkpoint. Zero means
    // the prompt shares nothing the device can reuse, and keeping the resident session would leave
    // the catalog describing a conversation the device no longer holds.
    std::size_t active_shared = 0;
    if (lane_state.active_session != kNoSession) {
        const SessionEntry& entry = sessions_[lane_state.active_session];
        const std::size_t common  = std::min(entry.tokens.size(), prompt_tokens.size());
        while (active_shared < common &&
               entry.tokens[active_shared] == prompt_tokens[active_shared]) {
            ++active_shared;
        }
        // The resident entry's media identity has to license the same prefix the tokens do, or the
        // recall would hand this prompt a conversation whose KV shows another picture.
        active_shared = media_prefix_cap(entry.media, media, active_shared);
    }
    std::uint32_t resident_depth = 0;
    if (lane_state.cached_state_valid) {
        if (lane_state.live_state_valid && lane_state.active_session != kNoSession) {
            const std::uint32_t frontier = sessions_[lane_state.active_session].frontier;
            if (frontier != 0 && frontier <= active_shared) { resident_depth = frontier; }
        }
        for (std::size_t slot = 0; slot < kReuseSnapshotCount; ++slot) {
            const std::uint32_t boundary = lane_state.cached_boundaries[slot];
            if (boundary <= active_shared && boundary > resident_depth) {
                resident_depth = boundary;
            }
        }
        for (const auto& checkpoint : shard_a_.host_checkpoints) {
            if (checkpoint.valid[lane] && checkpoint.position[lane] <= active_shared &&
                checkpoint.position[lane] > resident_depth) {
                resident_depth = checkpoint.position[lane];
            }
        }
    }

    // A host-resident entry carries three frozen GDN states: the one its evicted frontier stood on,
    // the one its last completed prefill froze at the entry's own prompt end, and the one frozen
    // where another conversation diverged from it. The frontier is only reachable by a prompt that
    // reproduces the generated tail token for token; a client that re-renders the answer it was
    // handed stops at the first generated token and uses the prompt-end image instead; and a
    // conversation that only shares a stable prefix - a system prompt - uses the divergence image.
    // The deepest reachable boundary that still beats the resident lineage wins; a shallower one
    // would spend a PCIe round trip to save fewer tokens than staying put.
    //
    // Every boundary has to leave at least one token to forward: the first sample's logits come from
    // the column the walk ends on, so a prompt that is exactly the recalled history would restore
    // the whole thing and then prefill it again from zero. Taking that recall also discards the
    // device lineage and the checkpoint ring, which is strictly worse than not taking it.
    std::size_t best_host       = kNoSession;
    std::uint32_t best_frontier = 0;
    RecallState best_state      = RecallState::Frontier;
    for (std::size_t index = 0; index < sessions_.size(); ++index) {
        const SessionEntry& entry = sessions_[index];
        if (index == lane_state.active_session || entry.device_lane >= 0 || entry.tokens.empty()) {
            continue;
        }
        const std::size_t common = std::min(entry.tokens.size(), prompt_tokens.size());
        std::size_t shared       = 0;
        while (shared < common && entry.tokens[shared] == prompt_tokens[shared]) { ++shared; }
        shared = media_prefix_cap(entry.media, media, shared);
        // The lane-local boundary is offered under the same rules as the entry's own images, plus the
        // token tag: it is the state of a prefix, so it applies to whichever conversation carries that
        // prefix, and to none whose tokens differ. It needs its draft pair to be loadable.
        const std::uint32_t boundary_position = shard_a_.reuse_boundary_position[lane];
        std::uint64_t boundary_tag            = 0;
        if (boundary_position != 0 && boundary_position <= entry.tokens.size()) {
            boundary_tag = 1469598103934665603ULL;
            for (std::size_t position = boundary_position > 8 ? boundary_position - 8 : 0;
                 position < boundary_position; ++position) {
                boundary_tag = (boundary_tag ^ static_cast<std::uint64_t>(entry.tokens[position])) *
                               1099511628211ULL;
            }
        }
        // NINFER_TP2_REUSE_BOUNDARY=0 ignores the lane-local boundary: the A/B switch the exactness
        // check needs, so the same prompt reuses at the shared block instead.
        const char* boundary_env    = std::getenv("NINFER_TP2_REUSE_BOUNDARY");
        const bool boundary_enabled = boundary_env == nullptr || boundary_env[0] != '0';
        const bool boundary_ready =
            boundary_enabled && shard_a_.reuse_boundary_state != nullptr &&
            (shard_a_.dflash_round == nullptr || shard_a_.reuse_boundary_dflash != nullptr);
        static constexpr const char* kReachName[4] = {"frontier", "prompt-end", "shared",
                                                      "boundary"};
        const std::uint32_t offered[4] = {
            entry.host_frontier_end, entry.host_prompt_end, entry.host_shared_end,
            (boundary_ready && boundary_tag != 0 &&
             shard_a_.reuse_boundary_tag[lane] == boundary_tag)
                ? boundary_position
                : 0};
        std::uint32_t reach = 0;
        RecallState via     = RecallState::Frontier;
        for (std::size_t kind = 0; kind < 4; ++kind) {
            // Every image is capped by the extent the slabs were actually filled to, so a recall
            // only reads pages this entry really carries. At a store the frontier is that extent,
            // and the prompt-end and shared images are taken at or behind it, so no image needs a
            // wider bound. An entry the device no longer holds *and* that was never stored - the
            // shape of publishing a conversation whose opening prompt the resident one already
            // answered, once the two generated tails part - carries no slabs at all: its frontier
            // still names the boundary it reached on the device, and offering it here restored pages
            // that do not exist. The frontier image is offered at the position it was filled at
            // rather than at the entry's frontier, so an entry that took later turns on the device or
            // came back to a shallower boundary still names a state it really holds.
            const std::uint32_t ceiling = entry.host_kv_end;
            if (offered[kind] == 0 || offered[kind] > shared ||
                offered[kind] >= prompt_tokens.size() || offered[kind] > ceiling) {
                continue;
            }
            if (offered[kind] > reach) {
                reach = offered[kind];
                via   = static_cast<RecallState>(kind);
            }
        }
        if (session_trace_enabled()) {
            std::fprintf(stderr,
                         "[tp2-session]   entry %zu session=%s tokens=%zu frontier=%u "
                         "frontier_end=%u kv_end=%u prompt_end=%u host_prompt_end=%u shared_end=%u "
                         "shared=%zu reach=%u via=%s resident=%d\n",
                         index,
                         session_key_label(entry.session ? &*entry.session : nullptr).c_str(),
                         entry.tokens.size(), entry.frontier, entry.host_frontier_end,
                         entry.host_kv_end, entry.prompt_end, entry.host_prompt_end,
                         entry.host_shared_end, shared, reach,
                         reach == 0 ? "none" : kReachName[static_cast<std::size_t>(via)],
                         entry.device_lane >= 0 ? 1 : 0);
        }
        if (reach == 0) { continue; }
        if (best_host == kNoSession || reach > best_frontier) {
            best_host     = index;
            best_frontier = reach;
            best_state    = via;
        }
    }
    if (best_host != kNoSession && best_frontier > resident_depth) {
        const std::size_t previous = lane_state.active_session;
        const bool drop_previous   = previous != kNoSession && !session_store_active(lane);
        // Restoring before dropping keeps the recalled entry's index valid: nothing has been erased
        // yet, and the drop below only shifts indices the restore is already done with.
        session_restore(sessions_[best_host], best_frontier, best_state, lane);
        lane_state.active_session       = best_host;
        lane_state.live_state_valid     = true;
        lane_state.cached_prompt_tokens = sessions_[best_host].tokens;
        lane_state.cached_media         = sessions_[best_host].media;
        // A recalled session has no device state snapshots of its own: those belong to the lineage
        // it displaces. Its frontier state is in the device pool already, which is what LiveState
        // means to the scan below. The checkpoint ring belongs to that lineage too, and its
        // positions can sit inside the recalled history while its states do not.
        lane_state.cached_boundaries.fill(0);
        invalidate_host_checkpoints(lane);
        lane_state.cached_state_valid = true;
        lane_state.reuse_source       = ReuseSource::None;
        if (drop_previous) { session_drop(previous); }
        // The scan below compared this prompt against the recalled entry's own history - the restore
        // kept it - so the boundary the two diverge at is that entry's to keep, not the one that just
        // left the device. The recall retires the ring it read from, which takes the outgoing
        // conversation's block boundary with it: this prompt names its own.
        lane_state.anchor_session          = lane_state.active_session;
        lane_state.block_anchor_position   = 0;
        lane_state.block_anchor_prefill_id = 0;
        if (session_trace_enabled()) {
            std::fprintf(
                stderr,
                "[tp2-session] recall frontier=%u resident_depth=%u tokens=%zu entries=%zu\n",
                sessions_[lane_state.active_session].frontier, resident_depth,
                sessions_[lane_state.active_session].tokens.size(), sessions_.size());
        }
        return;
    }
    // The resident lineage is only worth keeping while this prompt is a later turn of it, and the
    // last completed prefill's end is what proves that: the resident entry's prompt_end is the token
    // count of the prompt the device pools were built from, so a prompt that still shares those
    // tokens contains that whole prompt. A prompt that only matches a host checkpoint inside the
    // shared system prefix - the shape of a client switching conversations - is a different
    // conversation: the resident entry has to reach its host slabs before the prefill below
    // overwrites the device pools, or the conversation is lost.
    const bool resident_continues = lane_state.active_session != kNoSession &&
                                    sessions_[lane_state.active_session].prompt_end != 0 &&
                                    active_shared >= sessions_[lane_state.active_session].prompt_end;
    if (resident_depth > 0 && resident_continues) {
        // The resident conversation already serves this prompt better than any host slab; the reuse
        // scan below turns its depth into a prefill start point.
        if (session_trace_enabled()) {
            std::fprintf(stderr,
                         "[tp2-session] continue active_shared=%zu resident_depth=%u frontier=%u "
                         "prompt_end=%u entries=%zu\n",
                         active_shared, resident_depth, sessions_[lane_state.active_session].frontier,
                         sessions_[lane_state.active_session].prompt_end, sessions_.size());
        }
        return;
    }
    // A different conversation, or one no host entry can serve: preserve the resident session when
    // the budget allows. The prefix-reuse lineage survives the switch - lane_state.cached_prompt_tokens and
    // every checkpoint at or before this prompt's shared prefix still describe tokens this prompt
    // holds - so the scan below keeps whatever they cover instead of re-prefilling the whole system
    // prefix on every turn of a switch.
    const std::size_t previous = lane_state.active_session;
    bool stored                = previous == kNoSession || session_store_active(lane);
    if (!stored) {
        // The host budget could not take the outgoing conversation. Shed the entries the retention
        // weights rank lowest and try again: the resident entry is never a candidate, so this only
        // trades conversations the catalog considers weaker for the one the prefill is about to
        // replace. Without it the outgoing conversation was dropped the first time its slabs did
        // not fit, and its next turn re-prefilled a whole history it had already paid for.
        while (session_evict_one()) {
            if (session_store_active(lane)) {
                stored = true;
                break;
            }
        }
    }
    // session_drop keeps lane_state.active_session on the same entry across an eviction, so the resident
    // index is re-read here instead of trusting the copy taken before the loop: a victim the loop
    // erased from in front of it shifts every later index.
    const std::size_t resident = lane_state.active_session;
    if (!stored) {
        // Only the resident is left, so its own KV is what the host budget cannot hold and no
        // eviction can ever make room. The conversation goes - the prefill below overwrites the
        // device pools - but it is reported rather than dropped silently.
        std::fprintf(stderr,
                     "[tp2-session] drop frontier=%u tokens=%zu weight=%u: the resident session "
                     "alone exceeds the host session budget\n",
                     sessions_[resident].frontier, sessions_[resident].tokens.size(),
                     sessions_[resident].retention_weight);
    }
    // The divergence was measured against the resident entry's history, and the eviction just put
    // that entry's KV in the slabs, so the state the walk starts from is its state at the same
    // boundary. A session the host budget could not keep has no slab to pair the anchor with.
    lane_state.anchor_session            = stored ? resident : kNoSession;
    lane_state.block_anchor_position     = 0;
    lane_state.block_anchor_prefill_id   = 0;
    if (!stored) { session_drop(resident); }
    lane_state.active_session   = kNoSession;
    lane_state.live_state_valid = false;
    lane_state.reuse_source     = ReuseSource::None;
    ++session_full_prefills_;
    if (session_trace_enabled()) {
        std::fprintf(stderr,
                     "[tp2-session] switch active_shared=%zu resident_depth=%u stored=%d "
                     "entries=%zu\n",
                     active_shared, resident_depth, stored ? 1 : 0, sessions_.size());
    }
}

void TP2GenerationCore::session_publish(
    const std::vector<TokenId>& history, std::uint32_t frontier,
    const models::qwen3_5::PreparedContextCache& cache_hints, std::span<const MediaSpan> media,
    std::uint32_t lane) {
    RetentionState& lane_state = retention(lane);
    if (session_capacity_ == 0) { return; }
    // Served and forgotten. Re-prefilling a history this short costs less than the catalog slot it
    // would hold, and the slot is what a returning conversation needs; a client that fires a one-off
    // title or summary call beside every new session would otherwise evict the sessions themselves.
    if (history.size() < session_retention_floor_tokens_) { return; }
    // A conversation the client named owns exactly one catalog entry. The name is the client's
    // statement that two requests are the same conversation, so a rewrite of its history - a folded
    // context, a re-rendered tail, a prompt that lost its last byte - replaces that entry instead of
    // leaving a variant beside it. Keeping the variants is what let one conversation fill every slot
    // while the conversation it displaced lost the only entry its next turn could recall.
    const bool names_conversation = cache_hints.session_key.has_value();
    // The entry this request continues, found by matching the history rather than by trusting the
    // name: one entry per conversation only holds while the next turn can tell which entry the
    // conversation stands on. A turn continues the entry it reproduces bar a rewritten tail; a
    // one-off call beside the conversation opens on instructions of its own, so the prefix it shares
    // stops at the boilerplate the two happen to have in common and covers neither history. Matching
    // content is also what lets a conversation take its slot back after such a call took it first.
    std::size_t parent      = kNoSession;
    std::size_t parent_span = 0;
    if (names_conversation) {
        for (std::size_t index = 0; index < sessions_.size(); ++index) {
            const SessionEntry& entry = sessions_[index];
            // An entry another lane is running cannot be replaced here, and an entry carrying another
            // client's name is that conversation's slot: content alone does not prove this client
            // means it.
            if (entry.device_lane >= 0 && entry.device_lane != static_cast<std::int32_t>(lane)) {
                continue;
            }
            if (entry.session && !(*entry.session == *cache_hints.session_key)) { continue; }
            const std::size_t shared = shared_history_prefix(history, entry.tokens);
            if (shared < session_retention_floor_tokens_) { continue; }
            const std::size_t span = std::min(history.size(), entry.tokens.size());
            if (shared + kSessionRewrittenTailTokens < span) { continue; }
            if (parent == kNoSession || span > parent_span) {
                parent      = index;
                parent_span = span;
            }
        }
    }
    const std::size_t named_owner =
        names_conversation ? session_owner_index(*cache_hints.session_key) : kNoSession;
    // A named request that continues an entry adopts it, even one an earlier one-off call left
    // unnamed; a named request that continues nothing opens the conversation only while the name has
    // no entry at all. A request that continues nothing beside a named entry publishes anonymously
    // and leaves that entry alone - it still runs, it just does not claim the slot.
    const bool bind_name = names_conversation && (parent != kNoSession || named_owner == kNoSession);
    // An entry published without a name is an ordinary continuation, whatever retention the route
    // asked for on the name's behalf.
    models::qwen3_5::PreparedContextCache anonymous_hints;
    const models::qwen3_5::PreparedContextCache* const published_hints =
        bind_name ? &cache_hints : &anonymous_hints;
    // Only the conversation the device still holds may be updated in place. A different one that
    // reaches this point must get an entry of its own, or it would overwrite the resident history
    // and leave the conversation that owned it unreachable - which is what a short request beside a
    // new session used to prevent by taking the resident slot for itself.
    const bool extends_resident =
        lane_state.active_session != kNoSession &&
        sessions_[lane_state.active_session].device_lane == static_cast<std::int32_t>(lane) &&
        history.size() >= sessions_[lane_state.active_session].tokens.size() &&
        std::equal(sessions_[lane_state.active_session].tokens.begin(), sessions_[lane_state.active_session].tokens.end(),
                   history.begin());
    if (extends_resident) {
        // The resident entry is this request's parent by content, so the name belongs on it. Another
        // entry carrying the name is a stale claim: it is released before this entry takes the name,
        // which also frees the slabs it held and keeps the resident reference below valid.
        const bool adopt = bind_name && parent == lane_state.active_session;
        // Another entry carrying the name is a stale claim, and dropping one below the resident entry
        // shifts it; the returned index is where the resident entry now sits.
        if (adopt) {
            lane_state.active_session =
                session_prune_keyed(*cache_hints.session_key, lane_state.active_session);
        }
        SessionEntry& entry = sessions_[lane_state.active_session];
        entry.tokens.assign(history.begin(), history.end());
        entry.media.assign(media.begin(), media.end());
        entry.frontier  = frontier;
        entry.lru_clock = ++session_lru_clock_;
        bind_entry_session(entry, adopt ? *published_hints : anonymous_hints);
        // The reuse scan compares against the resident history, so it has to carry the generated
        // tail too: that is what lets a continued turn start at the committed frontier instead of
        // replaying the previous answer.
        lane_state.cached_prompt_tokens = entry.tokens;
        lane_state.cached_media.assign(media.begin(), media.end());
        lane_state.live_state_valid     = true;
        return;
    }
    // A named request that continues an entry stands on that entry: its next turn has to find it
    // still there, and the slot it occupies is the conversation's own, so the capacity check below
    // counts it once whether it is rewritten or appended beside.
    const bool replaces_parent = bind_name && parent != kNoSession;
    if (session_trace_enabled()) {
        std::fprintf(stderr,
                     "[tp2-session] publish key=%s named=%d bind=%d owner=%zu replaces=%d entries=%zu "
                     "history=%zu held_by=%zu\n",
                     names_conversation
                         ? session_key_label(&*cache_hints.session_key).c_str()
                         : "-",
                     names_conversation ? 1 : 0, bind_name ? 1 : 0,
                     replaces_parent ? parent : named_owner, replaces_parent ? 1 : 0,
                     sessions_.size(), history.size(), named_owner);
    }
    if (replaces_parent) {
        // The entry this request continues is rewritten where it stands rather than dropped and
        // replaced by a copy of itself. What the entry is worth is its host slabs: the KV of the
        // prefix this history reproduces, and the divergence image this prefill just captured before
        // the walk reached the frontier. A copy would arrive with none of it and destroy the captured
        // image with the entry, so a conversation that folded or re-rendered its tail would re-prefill
        // its whole history on every return - which is exactly the reuse the entry was holding.
        if (bind_name) { parent = session_prune_keyed(*cache_hints.session_key, parent); }
        SessionEntry& entry = sessions_[parent];
        // Only the part of the slabs this history reproduces is still theirs: past 'shared' the tokens
        // differ, so the KV there describes the conversation this request diverged from. The images
        // go with it - each is offered at the position it was filled at, so an image that no longer
        // sits inside the shared prefix is dropped rather than left mislabelled.
        const std::uint32_t shared = static_cast<std::uint32_t>(
            std::min(shared_history_prefix(history, entry.tokens),
                     static_cast<std::size_t>(entry.host_kv_end)));
        entry.host_kv_end = shared;
        if (entry.host_frontier_end > entry.host_kv_end) { entry.host_frontier_end = 0; }
        if (entry.host_prompt_end > entry.host_kv_end) { entry.host_prompt_end = 0; }
        if (entry.host_shared_end > entry.host_kv_end) { entry.host_shared_end = 0; }
        entry.tokens.assign(history.begin(), history.end());
        entry.media.assign(media.begin(), media.end());
        entry.frontier  = frontier;
        entry.lru_clock = ++session_lru_clock_;
        // The prompt-end image belongs to the history this rewrite replaced, and the walk finishing
        // this prefill publishes the boundary that replaced it.
        entry.prompt_end = 0;
        bind_entry_session(entry, *published_hints);
        if (session_trace_enabled()) {
            std::fprintf(stderr,
                         "[tp2-session] rewrite entry=%zu shared=%u kv_end=%u frontier_end=%u "
                         "prompt_end=%u shared_end=%u history=%zu\n",
                         parent, shared, entry.host_kv_end, entry.host_frontier_end,
                         entry.host_prompt_end, entry.host_shared_end, history.size());
        }
        lane_state.active_session = parent;
        mark_device_lane(sessions_, parent, static_cast<std::int32_t>(lane));
        lane_state.cached_prompt_tokens = entry.tokens;
        lane_state.cached_media.assign(media.begin(), media.end());
        lane_state.live_state_valid     = true;
        return;
    }
    if (bind_name) {
        // Every other entry carrying the name is a stale claim - a one-off call that took the name
        // first, a variant left by an older publish - now that the name lands on this lineage.
        session_prune_keyed(*cache_hints.session_key, kNoSession);
    }
    const std::uint32_t publisher_weight = private_retention_weight(published_hints->retention);
    if (sessions_.size() >= session_capacity_ && !session_evict_one(publisher_weight)) {
        // The catalog is full of entries the device cannot help with right now. This conversation
        // still runs - the device pools hold it - it just has no recalling entry.
        return;
    }
    SessionEntry entry;
    entry.tokens.assign(history.begin(), history.end());
    entry.media.assign(media.begin(), media.end());
    entry.frontier  = frontier;
    entry.lru_clock = ++session_lru_clock_;
    bind_entry_session(entry, *published_hints);
    sessions_.push_back(std::move(entry));
    lane_state.active_session = sessions_.size() - 1;
    mark_device_lane(sessions_, lane_state.active_session, static_cast<std::int32_t>(lane));
    lane_state.cached_prompt_tokens = sessions_[lane_state.active_session].tokens;
    lane_state.cached_media.assign(media.begin(), media.end());
    lane_state.live_state_valid     = true;
}

GenerationResult TP2GenerationCore::execute(Request& request, OutputSink* sink,
                                            const CancellationView& cancellation) {
    // The live-frontier shortcut is only sound while the device GDN state sits exactly where the
    // catalog says it does. A walk that throws before it publishes leaves the state somewhere the
    // catalog cannot name, so the guard retires both the claim and the resident entry before the next
    // request reads them. A walk that already published its terminal frontier named exactly where the
    // device stopped, so that throw cost the request, not the prefix: the catalog keeps the entry and
    // the next prompt can still recall it.
    const std::uint32_t lane = static_cast<std::uint32_t>(active_lane_);
    retention(lane).terminal_published = false;
    try {
        return execute_walk(request, sink, cancellation);
    } catch (...) {
        if (!retention(lane).terminal_published) { session_invalidate_active(lane); }
        throw;
    }
}

TP2GenerationCore::TurnAdoption
TP2GenerationCore::adopt_generated_turn(models::qwen3_5::PreparedPromptData& data,
                                        std::uint32_t prompt_tokens, bool trace,
                                        std::uint32_t lane) const {
    const RetentionState& lane_state = retention(lane);
    TurnAdoption adoption;
    adoption.prompt_tokens = prompt_tokens;
    // The histories this prompt could be continuing, each paired with the position the answer it
    // wrote begins at.
    //
    // The lane's own lineage is the only candidate while it is alive: it is the state this lane's KV
    // actually holds, and this is exactly what a single-lane route has always run. The catalog is the
    // fallback the batched route leaves behind - a lane hands its pages back and drops its lineage
    // when a request retires (release_lane_kv), so the lane that serves the next turn of a
    // conversation is routinely not the lane that served the last one, and the entry left in the
    // catalog is then the only copy of that turn, and the very entry session_recall() is about to
    // restore. Only entries recall can restore are worth adopting: one still resident on another lane
    // is skipped there (a live session must not be stolen), and one with no host slab has nothing to
    // restore.
    std::vector<AdoptionSource> sources;
    if (!lane_state.cached_prompt_tokens.empty()) {
        // Where the answer this lineage generated begins. The slot-0 snapshot is the boundary the
        // cached lineage's last prefill ended on, which is exactly the position the answer starts at,
        // and it survives the catalog letting the conversation go. The session field is chunk-start
        // bookkeeping, so it is only the fallback for a walk that left no snapshot of its own.
        std::size_t turn_begin = lane_state.cached_boundaries[0];
        if (turn_begin == 0 && lane_state.active_session != kNoSession) {
            turn_begin = sessions_[lane_state.active_session].prompt_end;
        }
        sources.push_back(
            AdoptionSource{lane_state.cached_prompt_tokens, turn_begin, true, false, -1});
    } else {
        for (std::size_t index = 0; index < sessions_.size(); ++index) {
            const SessionEntry& entry = sessions_[index];
            if (entry.device_lane >= 0 || entry.host_kv_end == 0) { continue; }
            sources.push_back(AdoptionSource{entry.tokens, entry.prompt_end, false, true,
                                             static_cast<std::int64_t>(index)});
        }
    }
    // Where the client's rendering and each history part, deepest first. Measured before anything is
    // replaced: it is both the reuse decision the scan would have made and, for the trace, the only
    // measurement of a re-rendered answer that survives the adoption itself.
    std::size_t divergence = 0;
    const std::vector<AdoptionCandidate> candidates =
        adoption_candidates(data.token_ids, sources, divergence);
    adoption.divergence = divergence;
    // A multimodal request is left alone: its token array carries vision runs whose positions a
    // replacement would have to keep, and a media turn never comes back as pure text. Its divergence
    // is measured above all the same, because the reuse trace reports it.
    if (data.has_media()) { return adoption; }
    for (const AdoptionCandidate& candidate : candidates) {
        const std::span<const TokenId> history = sources[candidate.source].history;
        const std::size_t turn_begin           = sources[candidate.source].turn_begin;
        const std::size_t turn_end             = history.size();
        // An adoption only pays while it deepens the prefix the scan can reach. It charges for that
        // in the client's coordinates: the walk runs the tokens this history generated in place of
        // the replayed ones, so the prompt stops agreeing with the client's own rendering at the
        // divergence, and every later turn of the conversation is measured from there. When the
        // client's rendering already agrees with the whole history this entry recorded, the live
        // frontier sits inside that agreement and the scan takes it with no replacement at all: the
        // splice could only trade positions the client will keep reproducing for ones it never will.
        // A tool-calling conversation hits that exactly - the template closes the turn with framing
        // the raw sampled tokens do not carry - and the replacement shortened its prompt behind the
        // answer it had already paid for, so every following turn matched only up to the divergence
        // and restarted on a host checkpoint.
        if (candidate.shared + 1 >= turn_end) { continue; }
        // The replay ends where the message after the turn begins. A prompt without that boundary -
        // an encoded token stream, or a turn the template folds into a neighbouring message - cannot
        // be compared here and keeps today's behaviour.
        std::size_t replay_end = 0;
        for (const auto& boundary : data.message_boundaries) {
            if (boundary.has_value() && *boundary > turn_begin) {
                replay_end = *boundary;
                break;
            }
        }
        if (replay_end <= turn_begin || replay_end > data.token_ids.size()) { continue; }
        const std::span<const TokenId> generated_span =
            std::span<const TokenId>(history).subspan(turn_begin, turn_end - turn_begin);
        const std::span<const TokenId> replayed_span =
            std::span<const TokenId>(data.token_ids).subspan(turn_begin, replay_end - turn_begin);
        const std::string generated = frontend().decode_tokens(generated_span, true);
        const std::string replayed  = frontend().decode_tokens(replayed_span, true);
        const bool same_turn        = same_rendered_turn(generated, replayed);
        if (trace) {
            // The case that decides the whole reuse story: the prompt matched past the previous prompt
            // end and then stopped, because the client re-rendered the turn this history generated.
            // Where the bytes part decides whether the turn is the same one, so print both sides. The
            // source says which history this candidate was: -1 is the lane's own lineage, anything
            // else is the catalog entry at that index.
            std::fprintf(stderr,
                         "[tp2-diverge] prompt=%u cached=%zu prev_prompt=%zu shared=%zu in_turn=%zu "
                         "to_turn_end=%zu replay=%zu same_turn=%d source=%lld",
                         prompt_tokens, history.size(), turn_begin, candidate.shared,
                         candidate.shared > turn_begin ? candidate.shared - turn_begin : 0,
                         turn_end - candidate.shared, replayed_span.size(), same_turn ? 1 : 0,
                         static_cast<long long>(sources[candidate.source].id));
            // Where the two texts part, in bytes: the prefixes below are capped, and a divergence deep
            // inside a long turn is invisible in them.
            std::size_t difference = 0;
            const std::size_t common = std::min(generated.size(), replayed.size());
            while (difference < common && generated[difference] == replayed[difference]) {
                ++difference;
            }
            std::fprintf(stderr, " diff_at=%zu", difference);
            print_reuse_span("generated", *this, generated_span);
            print_reuse_span("replayed", *this, replayed_span);
            const std::size_t window = difference > 24 ? difference - 24 : 0;
            std::fputs(" around_generated=\"", stderr);
            print_escaped_bytes(generated, window, 48);
            std::fputs("\" around_replayed=\"", stderr);
            print_escaped_bytes(replayed, window, 48);
            std::fputs("\"", stderr);
            std::fprintf(stderr, "\n");
        }
        if (!same_turn) { continue; }
        // An exact replay needs nothing replaced: the scan below already reads this history's tokens.
        if (generated_span.size() == replayed_span.size() &&
            std::equal(generated_span.begin(), generated_span.end(), replayed_span.begin())) {
            return adoption;
        }
        // Same turn, so the tokens this history owns are the ones its KV holds; replacing the replay
        // with them lets the prefix scan below take the deepest boundary of this lineage - normally
        // its own frontier - instead of re-prefilling the whole answer. The replacement is allowed to
        // be a few tokens longer than the replay, so it still has to fit the context the request was
        // admitted against. token_types and positions stay as prepared because only the media route
        // fills them, and that route never reaches this point.
        if (data.token_ids.size() - replayed_span.size() + generated_span.size() >
            options_.max_context) {
            continue;
        }
        std::vector<TokenId> spliced;
        spliced.reserve(data.token_ids.size() - (replay_end - turn_begin) + (turn_end - turn_begin));
        spliced.insert(spliced.end(), data.token_ids.begin(),
                       data.token_ids.begin() + static_cast<std::ptrdiff_t>(turn_begin));
        spliced.insert(spliced.end(),
                       history.begin() + static_cast<std::ptrdiff_t>(turn_begin), history.end());
        spliced.insert(spliced.end(),
                       data.token_ids.begin() + static_cast<std::ptrdiff_t>(replay_end),
                       data.token_ids.end());
        data.token_ids = std::move(spliced);
        adoption.adopted       = true;
        adoption.prompt_tokens = static_cast<std::uint32_t>(data.token_ids.size());
        return adoption;
    }
    return adoption;
}

PrefixReusePath TP2GenerationCore::reuse_path(std::uint32_t reuse,
                                              std::uint32_t block_frontier,
                                              std::uint32_t lane) const noexcept {
    const RetentionState& lane_state = retention(lane);
    if (reuse == 0) { return PrefixReusePath::Root; }
    if (block_frontier != 0 && reuse == block_frontier) {
        return PrefixReusePath::SharedStablePrefix;
    }
    if (lane_state.active_session != kNoSession) {
        const SessionEntry& entry = sessions_[lane_state.active_session];
        if (reuse == entry.frontier) { return PrefixReusePath::PrivateEndpoint; }
        // The boundary in front of the answer this lineage generated. A client that re-rendered that
        // answer cannot extend it and re-prefills from the generation opener instead.
        if (reuse == entry.prompt_end) { return PrefixReusePath::PrivateResponseReplay; }
    }
    // Everything else is a retention boundary: a grid or tail anchor this core wrote, or a state
    // another conversation was seen to diverge at.
    return PrefixReusePath::PrivateLongAnchor;
}

// The scan's internal source in the published vocabulary (§17 of docs/maintainer/prefix-reuse.md).
ReuseSourceKind TP2GenerationCore::reuse_source_kind(ReuseSource source) noexcept {
    switch (source) {
    case ReuseSource::DeviceSnapshot: return ReuseSourceKind::Device;
    case ReuseSource::HostCheckpoint: return ReuseSourceKind::HostSlab;
    case ReuseSource::LiveState: return ReuseSourceKind::LiveState;
    case ReuseSource::None: break;
    }
    return ReuseSourceKind::None;
}

// Copies the GDN state this lane's device slot holds right now into the ring slot the group owns,
// tagged with the frontier it captures. It rides the shard stream, so it sees exactly the tokens the
// enclosing loop has enqueued and none of the later ones, and the ring slot it lands in is only
// revisited by a later prefill's checkpoint at the same index.
void TP2GenerationCore::capture_reuse_boundary(std::uint32_t lane) {
    RetentionState& state = retention(lane);
    if (state.reuse_boundary_target == 0 || state.reuse_boundary_captured) { return; }
    const std::uint32_t position = state.reuse_boundary_target;
    for (Shard* shard : {&shard_a_, &shard_b_}) {
        if (shard->lane_state_geometry.image_bytes == 0) { return; }
        if (shard->reuse_boundary_state == nullptr) {
            shard->reuse_boundary_state = std::make_unique<HostBuffer>(
                static_cast<std::size_t>(lanes_) * shard->lane_state_geometry.image_bytes,
                HostPinning::Pageable);
        }
        shard->device.bind_to_current_thread();
        copy_lane_state(shard->lane_state_geometry, shard->state_backing.data,
                        static_cast<std::int32_t>(lane),
                        static_cast<std::byte*>(shard->reuse_boundary_state->data()) +
                            static_cast<std::size_t>(lane) *
                                shard->lane_state_geometry.image_bytes,
                        cudaMemcpyDeviceToHost, shard->device.stream);
        if (shard->dflash_round != nullptr) {
            const std::size_t lane_bytes = shard->dflash_round->lane_context_image_bytes();
            if (shard->reuse_boundary_dflash == nullptr) {
                shard->reuse_boundary_dflash = std::make_unique<HostBuffer>(
                    static_cast<std::size_t>(lanes_) * lane_bytes, HostPinning::Pageable);
            }
            store_dflash_image(*shard, shard->reuse_boundary_dflash->data(),
                               shard->reuse_boundary_dflash->size(), lane);
        }
        shard->reuse_boundary_position[lane] = position;
        shard->reuse_boundary_session[lane]  =
            state.active_session == kNoSession ? 0 : state.active_session + 1;
    }
    // The eight tokens before the boundary are the image's identity: it is that prefix's state
    // whoever wrote it, and a rewritten entry whose tokens changed must not load it.
    std::uint64_t tag          = 1469598103934665603ULL;
    const auto& prefix_tokens  = state.cached_prompt_tokens;
    if (position <= prefix_tokens.size()) {
        for (std::size_t index = position > 8 ? position - 8 : 0; index < position; ++index) {
            tag = (tag ^ static_cast<std::uint64_t>(prefix_tokens[index])) * 1099511628211ULL;
        }
    } else {
        tag = 0;
    }
    for (Shard* shard : {&shard_a_, &shard_b_}) { shard->reuse_boundary_tag[lane] = tag; }
    state.reuse_boundary_captured = true;
    if (const char* probe = std::getenv("NINFER_TP2_REUSE_TRACE");
        probe != nullptr && probe[0] == '1') {
        std::fprintf(stderr, "[tp2-reuse] boundary lane=%u pos=%u session=%zu\n", lane, position,
                     state.active_session);
    }
}

void TP2GenerationCore::snapshot_host_checkpoint(Shard& shard, std::uint32_t frontier, HostRing ring,
                                                 std::uint32_t lane) {
    if (shard.host_checkpoints.empty()) { return; }
    ++prefill_host_writes_;
    const std::size_t grid = shard.host_checkpoint_grid_slots;
    const std::size_t tail = host_checkpoint_tail_slots_;
    std::size_t begin      = 0;
    std::size_t count      = grid;
    bool appended          = false;
    switch (ring) {
        case HostRing::Grid: break;
        case HostRing::Tail: begin = grid; count = tail; break;
        case HostRing::Divergence:
            begin = grid + tail;
            count = host_checkpoint_divergence_slots_;
            break;
        case HostRing::Block:
            begin = grid + tail + host_checkpoint_divergence_slots_;
            count = host_checkpoint_block_slots_;
            break;
        case HostRing::PromptEnd:
            // The prompt end is written once per completed prefill and never rotates, so it gets one
            // slot per lane appended after every lane's slice instead of a place in the grid. A place
            // in the grid would have to be carved out of the position stride, and the ring layout is
            // only per_lane wide when the configured slot budget is large enough (see the ring sizing
            // in build_shard), so a fixed offset inside a slice can alias a neighbour's slot.
            begin    = static_cast<std::size_t>(lanes_) * host_checkpoint_slots_per_lane_ +
                       static_cast<std::size_t>(lane);
            count    = 1;
            appended = true;
            break;
    }
    if (!appended) {
        // Each lane owns its own slice of the ring, laid out exactly like the whole ring at lanes=1,
        // so a write never evicts another lane's checkpoints.
        begin += static_cast<std::size_t>(lane) * host_checkpoint_slots_per_lane_;
    }
    if (count == 0 || begin + count > shard.host_checkpoints.size()) { return; }
    const std::size_t index = ring == HostRing::Tail   ? shard.host_checkpoint_tail_next[lane]
                              : ring == HostRing::Grid ? shard.host_checkpoint_next[lane]
                                                       : 0;
    shard.device.bind_to_current_thread();
    Shard::HostCheckpoint& checkpoint = shard.host_checkpoints[begin + index];
    // Overwriting the slot already destroys the checkpoint it held, so retire it before the copy is
    // enqueued: a prefill that throws between here and the publish sweep must not leave a torn buffer
    // behind a flag that still says the last completed prefill wrote it. Only the sweep below, which
    // runs after the prefill finished, makes a checkpoint usable again.
    checkpoint.valid[lane] = false;
    checkpoint.position[lane]   = frontier;
    checkpoint.prefill_id[lane] = retention(lane).host_checkpoint_live_id;
    // The slot belongs to this lane, so its buffer *is* the lane's compact state image: the lane
    // is expressed by the slot index this ring's layout gives it, never by a byte offset into the
    // slot. The slot is sized for exactly one such image, so a mismatch here would mean the ring
    // partition and the state geometry disagree and the copy would run off the allocation.
    if (checkpoint.buffer->size() < shard.lane_state_geometry.image_bytes) {
        throw std::logic_error(
            "TP-2 host checkpoint slot is smaller than one compact lane state image");
    }
    copy_lane_state(shard.lane_state_geometry, shard.state_backing.data,
                    static_cast<std::int32_t>(lane),
                    static_cast<std::byte*>(checkpoint.buffer->data()),
                    cudaMemcpyDeviceToHost, shard.device.stream);
    // The masked draft's context at the same frontier rides the same slot. A chunk boundary is
    // exactly where the prefill sink has finished committing that chunk, so the ring reaches this
    // frontier; the frontier is recorded so a restore can refuse a checkpoint whose draft half does
    // not describe the position it names.
    if (shard.dflash_round != nullptr && checkpoint.dflash_buffer != nullptr) {
        // The same partition check as the target half above: a slot that cannot hold one lane's
        // draft ring would make the copy below run off the allocation.
        if (checkpoint.dflash_buffer->size() < shard.dflash_round->lane_context_image_bytes()) {
            throw std::logic_error("TP-2 host checkpoint slot is smaller than one lane's draft ring");
        }
        checkpoint.dflash_frontier[lane] = retention(lane).dflash_context_frontier;
        shard.dflash_round->copy_context_to_host(
            static_cast<std::byte*>(checkpoint.dflash_buffer->data()), shard.device.stream,
            static_cast<std::int32_t>(lane));
    }
    if (ring == HostRing::Tail) {
        shard.host_checkpoint_tail_next[lane] = (index + 1) % count;
    } else if (ring == HostRing::Grid) {
        shard.host_checkpoint_next[lane] = (index + 1) % count;
    }
}

void TP2GenerationCore::publish_lane_checkpoints(std::uint32_t lane) {
    const std::uint64_t live_id = retention(lane).host_checkpoint_live_id;
    for (Shard* shard : {&shard_a_, &shard_b_}) {
        for (auto& checkpoint : shard->host_checkpoints) {
            if (checkpoint.prefill_id[lane] == live_id) { checkpoint.valid[lane] = true; }
        }
    }
}

// Opens a request's Vision prefill session on top of the startup plan. One session owns the items
// the request still has to encode: an item that lies entirely inside a reused prefix is already in
// the KV - its embeddings were scattered when that prefix was first prefilled - so it is dropped
// from the plan and its host patch payload is released. A request whose reused prefix covers every
// item (the common case for a chat turn that carries an image in its history) gets no session at
// all. The caller keeps `plan` alive as long as the session lives, because the session binds it by
// reference. A batch executor calls this once per lane: prefill is serial per lane, so the single
// startup arena is reused sequentially and only one session is ever alive.
std::unique_ptr<qwen::execution::VisionPrefillSession> TP2GenerationCore::open_vision_session(
    qwen::PreparedPromptData& data, std::uint32_t reuse,
    qwen::execution::VisionPrefillPlan& plan) {
    if (!data.has_media()) { return nullptr; }
    if (!vision_workspace_ || vision_arena_ == nullptr) {
        throw std::invalid_argument("multimodal request without a Vision workspace plan");
    }
    const auto& vision_config = shard_b_.model->config().vision.value();
    auto control_plan = std::make_shared<qwen::VisionControlPlan>(
        qwen::plan_vision_control(data, vision_config));
    std::size_t max_merged = 0;
    std::size_t first_item = control_plan->items.size();
    plan.uses.reserve(control_plan->items.size());
    for (std::size_t index = 0; index < control_plan->items.size(); ++index) {
        const qwen::VisionItemControlPlan& item = control_plan->items[index];
        if (item.merged_count > vision_workspace_->max_merged_tokens) {
            throw std::invalid_argument(
                "image needs " + std::to_string(item.merged_count) +
                " vision tokens but this TP-2 route admitted only " +
                std::to_string(vision_workspace_->max_merged_tokens) +
                " at startup (see the [mem] vision ledger line)");
        }
        if (item.token_end <= reuse) {
            // Already encoded by the prefill that owns the reused prefix; the patch payload is not
            // needed again, so drop it instead of holding it for the request's lifetime.
            data.media_payloads[index].reset();
            continue;
        }
        if (first_item == control_plan->items.size()) { first_item = index; }
        plan.uses.push_back(qwen::execution::VisionUseSpan{
            .begin               = item.token_begin,
            .end                 = item.token_end,
            .prepared_item_index = static_cast<std::uint32_t>(index),
            .control_index       = 0,
        });
        max_merged = std::max(max_merged, item.merged_count);
    }
    if (plan.uses.empty()) { return nullptr; }
    const auto first      = static_cast<std::uint32_t>(first_item);
    plan.max_merged_count = max_merged;
    plan.control          = std::make_shared<const qwen::VisionControl>(
        qwen::build_vision_control(data, *control_plan, first));
    for (qwen::execution::VisionUseSpan& use : plan.uses) {
        use.control_index = use.prepared_item_index - first;
    }
    shard_b_.device.bind_to_current_thread();
    return std::make_unique<qwen::execution::VisionPrefillSession>(
        shard_b_.device, *shard_b_.parameters,
        DeviceSpan{vision_arena_->base(), vision_arena_->capacity()}, *vision_workspace_, data,
        plan, vision_handoff_peak_bytes_);
}

GenerationResult TP2GenerationCore::execute_walk(Request& request, OutputSink* sink,
                                                 const CancellationView& cancellation) {
    const bool streaming = request.consumer_mode == OutputConsumerMode::Streaming;
    auto& data = qwen::PreparedPromptAccess::mutable_view(request.prompt);
    const auto& token_ids = data.token_ids;
    const std::uint32_t lane = static_cast<std::uint32_t>(active_lane_);
    RetentionState& lane_state = retention(lane);
    // A multimodal request runs its Vision prefill on the Vision shard and then decodes through the
    // same speculative window as a text request. A chat turn that carries an image keeps that image
    // in every later turn's history, so treating "has media" as "no speculation" would cost the whole
    // conversation its MTP rounds; the request-level flag stays a property of the session, not of the
    // prompt.
    const bool media = data.has_media();
    // The walk's own prompt length. adopt_generated_turn below may replace the client's rendering
    // of the last turn with the tokens this lineage generated, which changes the count.
    std::uint32_t prompt_tokens = static_cast<std::uint32_t>(token_ids.size());
    const std::int32_t vocab =
        qwen::execution::dimension(shard_a_.model->config().text.vocab_size);
    const std::int32_t hidden =
        qwen::execution::dimension(shard_a_.model->config().text.hidden_size);
    // The embedding and lm_head are packed to `vocab` rows, but only the first `public_tokens` of them
    // name something the tokenizer can spell; the rows in between are padding. Every selection over
    // these logits is therefore bounded by the public domain, so a padding row can never be sampled,
    // corrected, or licensed.
    const std::int32_t public_tokens =
        static_cast<std::int32_t>(shard_a_.model->resources().public_token_count);

    // Constrained tool-call decoding. The declared-name grammar turns the tool-call region into a
    // logit mask, so a name the request did not declare is unreachable at the sampling layer instead
    // of being rejected after the fact. The mask is applied outside the captured verify graph, to
    // the logits the graph hands back, so the graph itself is unchanged.
    std::shared_ptr<qwen::frontend::ToolCallConstraint> tool_constraint;
    if (data.tool_call_output != nullptr) {
        tool_constraint = frontend_->make_tool_call_constraint(data.tool_call_output);
    }
    const bool tool_constrained = tool_constraint != nullptr;
    // The logits domain is the packed embedding row count, which the artifact contract allows to be
    // wider than the tokenizer public domain the constraint table covers; build_mask() excludes the
    // rows in between. A table wider than the logits domain means the artifact and tokenizer disagree.
    const std::size_t logits_domain = static_cast<std::size_t>(vocab);
    if (tool_constrained && tool_constraint->vocab_size() > logits_domain) {
        throw std::logic_error("TP-2 tool-call constraint vocabulary " +
                               std::to_string(tool_constraint->vocab_size()) +
                               " exceeds the logits domain " + std::to_string(logits_domain));
    }

    GenerationResult result;
    result.prompt = request.summary;
    result.timings.prepare_seconds = request.prepare_seconds;
    const Clock::time_point start = Clock::now();
    // The timing events belong to shard A's device, so bind it before creating them.
    shard_a_.device.bind_to_current_thread();
    Tp2RoundTiming& timing = tp2_timing();
    timing.init();
    timing.reset();
    // Prefill-stage split, reported under NINFER_TP2_TIMING. The stages are separated by device
    // synchronizations so a host-side segment cannot hide device work, which perturbs the absolute
    // numbers a little; this is a diagnostic mode, not a measurement of the production path.
    const bool prefill_trace = [] {
        const char* env = std::getenv("NINFER_TP2_TIMING");
        return env != nullptr && env[0] == '1';
    }();
    // Prefix-reuse trace. It shows what the boundaries offered, what the prompt actually matched,
    // and - when a client re-rendered the answer it was handed - the bytes either side of the
    // divergence, which is what decides whether the two describe the same turn.
    const bool reuse_trace = [] {
        const char* env = std::getenv("NINFER_TP2_REUSE_TRACE");
        return env != nullptr && env[0] == '1';
    }();
    std::uint32_t prefill_chunks = 0;
    prefill_host_writes_         = 0;
    Clock::time_point scan_done       = start;
    Clock::time_point state_done      = start;
    Clock::time_point walk_done       = start;

    // Cross-session recall, before the prefix scan reads the device lineage. A prompt that belongs
    // to another conversation makes the resident session swap out here, so by the time the scan
    // below compares against lane_state.cached_prompt_tokens the pools already hold the conversation this
    // prompt continues. A prompt that continues the resident conversation is left untouched.
    // A client that re-renders the answer it was handed does not have to pay for it: when the
    // replay describes the same turn, the tokens this lineage generated are the ones its KV holds,
    // and the scan below then takes the boundary this lineage actually reached instead of the
    // prompt end. See adopt_generated_turn.
    //
    // This runs before the recall on purpose. The recall decides whether the conversation is still
    // this prompt's lineage from how far the two agree, and a replayed answer is exactly the case
    // where the client's rendering agrees only up to a re-tokenised token deep inside it: comparing
    // the tokens the walk will really forward keeps a conversation whose answer came back as text
    // resident, instead of retiring it over a divergence the adoption is about to remove. When
    // nothing is adopted the tokens are untouched and the recall sees exactly what it always did.
    const TurnAdoption adoption = adopt_generated_turn(data, prompt_tokens, reuse_trace, lane);
    if (adoption.adopted) {
        prompt_tokens                 = adoption.prompt_tokens;
        request.summary.prompt_tokens = prompt_tokens;
        result.prompt.prompt_tokens   = prompt_tokens;
    }

    // The media identity this prompt carries, taken once. `token_ids` above is a view of the same
    // vector, and adoption only rewrites the count, so the spans index into what the walk publishes.
    const std::vector<MediaSpan> prompt_media = collect_media_spans(data, data.token_ids.size());
    session_recall(lane, token_ids, prompt_media);

    // Prompt-prefix reuse. The KV pages hold the K/V of every position the last completed prefill
    // wrote, and the state snapshots hold the matching GDN states, so a prompt that extends the
    // previous prompt can skip the shared prefix and prefill only its suffix. The single-device
    // route gets this from the context cache; TP-2 runs with that cache disabled, so the core keeps
    // the boundaries it already owns. The deepest boundary at or before the shared prefix wins; the
    // final prompt token is always forwarded, because its logits drive the first sample.
    std::uint32_t reuse       = 0;
    std::size_t reuse_slot    = 0;
    std::size_t shared_prefix = 0;
    lane_state.reuse_source             = ReuseSource::None;
    if (lane_state.cached_state_valid && !lane_state.cached_prompt_tokens.empty()) {
        const std::size_t common = std::min(lane_state.cached_prompt_tokens.size(), token_ids.size());
        while (shared_prefix < common &&
               lane_state.cached_prompt_tokens[shared_prefix] == token_ids[shared_prefix]) {
            ++shared_prefix;
        }
        // Token ids cannot see media: every image merges to the same placeholder ids, so two prompts
        // showing different pictures compare equal above. Cap the boundary at the first media item
        // whose identity differs, or the reuse would keep KV written for another picture.
        shared_prefix = media_prefix_cap(lane_state.cached_media, prompt_media, shared_prefix);
        // A host checkpoint is a state plus the KV before it, and both are only this prompt's while
        // the lineage agrees on the tokens before it: the state was taken over the tokens of the
        // prompt that wrote it, and every prefill since kept the KV consistent only up to the prefix
        // it reused. The shared prefix of the last pair is therefore the bound for every older
        // frontier, and a checkpoint past it can never become usable again - a later prompt would
        // have to match a token this one already diverged from. Prune on that bound, then take the
        // deepest survivor. Older checkpoints matter: a prefill that only walked the last few tokens
        // of a fully reused prompt leaves no checkpoint of its own.
        Shard* const shards[2] = {&shard_a_, &shard_b_};
        for (Shard* shard : shards) {
            for (auto& checkpoint : shard->host_checkpoints) {
                if (checkpoint.position[lane] > shared_prefix) { checkpoint.valid[lane] = false; }
            }
        }
        // Every boundary the lineage can offer, deepest wins. Any of them is usable while the state
        // and KV beside it are this prompt's own prefix - which the shared prefix above proves - and a
        // boundary inside a prefill chunk is no exception: the suffix this walk chunks from it rounds
        // differently from a from-scratch walk, so the two can part at an exact logit tie. That is the
        // tolerance a bounded recall has always had, and why a recall owes a from-scratch walk its
        // boundary crossing rather than its whole trajectory. Rounding down to the prefill grid
        // instead would re-prefill up to a whole chunk on every turn of a conversation that continues
        // in place - the common agent case. Decode writes the position grid too (see the decode loop),
        // so a boundary now exists inside the answer this lineage generated as well; what no boundary
        // inside it can offer is a restart *after* a replay that does not describe this turn.
        //
        // The masked draft stays live on such a boundary as well. Its ring belongs to the walk that
        // froze it, so it proposes the block that walk would have proposed; the draft only proposes,
        // and the target verify licenses every emitted token either way.
        const auto take = [&](std::uint32_t position, std::size_t slot, ReuseSource source) {
            if (position == 0 || position > shared_prefix || position >= prompt_tokens ||
                position <= reuse) {
                return;
            }
            reuse         = position;
            reuse_slot    = slot;
            lane_state.reuse_source = source;
        };
        // The live GDN state is the deepest boundary a continued conversation can offer: it sits
        // exactly at the resident entry's frontier, and the device KV holds the whole history before
        // it. Reusing it copies nothing at all - but only a prompt that extends it has a column to
        // forward, and the final prompt token must still be forwarded to produce the logits that
        // drive the first sample.
        if (lane_state.live_state_valid && lane_state.active_session != kNoSession) {
            take(sessions_[lane_state.active_session].frontier, 0, ReuseSource::LiveState);
        }
        for (std::size_t slot = 0; slot < kReuseSnapshotCount; ++slot) {
            // A slot the host ring serves has no device plane; the same boundary is offered below
            // through its checkpoint instead.
            if (shard_a_.state_snapshots[slot].data == nullptr) { continue; }
            take(lane_state.cached_boundaries[slot], slot, ReuseSource::DeviceSnapshot);
        }
        for (std::size_t index = 0; index < shard_a_.host_checkpoints.size(); ++index) {
            const auto& checkpoint = shard_a_.host_checkpoints[index];
            if (checkpoint.valid[lane]) {
                take(checkpoint.position[lane], index, ReuseSource::HostCheckpoint);
            }
        }
    }
    if (prefill_trace) { scan_done = Clock::now(); }
    if (reuse_trace) {
        std::size_t valid_checkpoints = 0;
        for (const auto& checkpoint : shard_a_.host_checkpoints) {
            valid_checkpoints += checkpoint.valid[lane] ? 1U : 0U;
        }
        // The deepest rewind slot this lane still carries. A configuration that keeps no rewind slot
        // reports zero instead of reading past the boundary array.
        std::uint32_t rewind_boundary = 0;
        for (std::size_t slot = 1; slot < kReuseSnapshotCount; ++slot) {
            rewind_boundary = std::max(rewind_boundary, lane_state.cached_boundaries[slot]);
        }
        const char* source = lane_state.reuse_source == ReuseSource::HostCheckpoint ? "host"
                             : lane_state.reuse_source == ReuseSource::LiveState    ? "live"
                             : lane_state.reuse_source == ReuseSource::DeviceSnapshot ? "device"
                                                                            : "none";
        const std::uint32_t previous_prompt =
            lane_state.active_session != kNoSession ? sessions_[lane_state.active_session].prompt_end : 0;
        std::fprintf(stderr,
                     "[tp2-reuse] prompt=%u cached=%zu shared=%zu replay_split=%zu adopted=%d "
                     "prev_prompt=%u prefill_end=%u rewind=%u host=%zu/%zu stride=%u -> reuse=%u "
                     "slot=%zu src=%s\n",
                     prompt_tokens, lane_state.cached_prompt_tokens.size(), shared_prefix,
                     adoption.divergence, adoption.adopted ? 1 : 0, previous_prompt,
                     lane_state.cached_boundaries[0], rewind_boundary, valid_checkpoints,
                     shard_a_.host_checkpoints.size(), host_checkpoint_stride_, reuse, reuse_slot,
                     source);
    }

    // Tag the checkpoints this walk leaves behind and start the ring at the first stride multiple
    // past the reused boundary: a checkpoint at the boundary itself would only duplicate the device
    // snapshot the walk starts from. The new checkpoints become usable when this prefill completes.
    lane_state.host_checkpoint_live_id           = host_checkpoint_next_id_++;
    std::uint32_t next_host_checkpoint = 0;
    if (host_checkpoint_stride_ != 0) {
        next_host_checkpoint = host_checkpoint_stride_;
        while (next_host_checkpoint <= reuse) { next_host_checkpoint += host_checkpoint_stride_; }
    }

    // Multimodal request: one Vision session owns the items this request still has to encode, on top
    // of the startup plan. An item that lies entirely inside a reused prefix is already in the KV -
    // its embeddings were scattered when that prefix was first prefilled - so the session covers only
    // the suffix, and a request whose reused prefix covers every item (the common case for a chat
    // turn that carries an image in its history) runs with no session at all. Its chunks still bind
    // the prompt's 3-axis RoPE table, which is a property of the prompt rather than of the encoding.
    // The plan must outlive the session, which binds it by reference.
    qwen::execution::VisionPrefillPlan vision_plan;
    std::unique_ptr<qwen::execution::VisionPrefillSession> vision_session =
        open_vision_session(data, reuse, vision_plan);

    // Reset per-request state on both shards. The KV pages and execution row 0 are materialized
    // once at startup (build_shard) and reused in place, so a reused prefix needs no KV work at
    // all; only the GDN state must be restored to (or reset at) the prefill frontier.
    auto begin_gdn_state = [&](Shard& shard) {
        shard.device.bind_to_current_thread();
        switch (lane_state.reuse_source) {
        case ReuseSource::None:
            // A lane only owns its own slot, so a reset must not touch its neighbours.
            zero_lane_state(shard.lane_state_geometry, shard.state_backing.data,
                            static_cast<std::int32_t>(lane), shard.device.stream);
            return;
        case ReuseSource::LiveState:
            // The device state already sits at this boundary: a restored session put it there, or
            // the conversation that just decoded left it exactly at its frontier. Nothing to copy.
            return;
        case ReuseSource::DeviceSnapshot:
            copy_lane_state(shard.lane_state_geometry, shard.state_backing.data,
                            static_cast<std::int32_t>(lane), shard.state_snapshots[reuse_slot].data,
                            cudaMemcpyDeviceToDevice, shard.device.stream);
            return;
        case ReuseSource::HostCheckpoint:
            // The slot belongs to this lane, so its buffer *is* the lane's compact state image:
            // the lane is expressed by the slot index, never by a byte offset into the slot.
            copy_lane_state(shard.lane_state_geometry, shard.state_backing.data,
                            static_cast<std::int32_t>(lane),
                            static_cast<std::byte*>(
                                shard.host_checkpoints[reuse_slot].buffer->data()),
                            cudaMemcpyHostToDevice, shard.device.stream);
            return;
        }
    };
    begin_gdn_state(shard_a_);
    begin_gdn_state(shard_b_);
    // The masked draft's context is restored to that same boundary through the same sources. It
    // cannot be left for the walk to rebuild: the walk starts at `reuse`, and the skipped prefix
    // produces no target residual, so a boundary that does not restore these bytes is a hole. A
    // captured conversation is only offered a boundary whose draft image exists (see the session
    // slab allocation), so every source here has one.
    auto begin_dflash_state = [&](Shard& shard) {
        if (shard.dflash_round == nullptr) { return; }
        shard.device.bind_to_current_thread();
        switch (lane_state.reuse_source) {
        case ReuseSource::None:
            shard.dflash_round->zero_context();
            return;
        case ReuseSource::LiveState:
            // The live ring already reaches this boundary: a recall restored it, or the conversation
            // that just decoded left it flushed to its frontier at publish.
            return;
        case ReuseSource::DeviceSnapshot: {
            const std::size_t lane_bytes = shard.dflash_round->lane_context_image_bytes();
            shard.dflash_round->copy_context_from_device(
                DeviceSpan{static_cast<std::byte*>(shard.dflash_snapshots[reuse_slot].data) +
                               static_cast<std::size_t>(lane) * lane_bytes,
                           lane_bytes},
                shard.device.stream, static_cast<std::int32_t>(lane));
            return;
        }
        case ReuseSource::HostCheckpoint: {
            const Shard::HostCheckpoint& checkpoint = shard.host_checkpoints[reuse_slot];
            if (checkpoint.dflash_buffer == nullptr ||
                checkpoint.dflash_frontier[lane] != checkpoint.position[lane]) {
                throw std::logic_error(
                    "TP-2 DFlash2 checkpoint does not carry the draft context at its frontier");
            }
            shard.dflash_round->copy_context_from_host(
                static_cast<const std::byte*>(checkpoint.dflash_buffer->data()), shard.device.stream,
                static_cast<std::int32_t>(lane));
            return;
        }
        }
    };
    begin_dflash_state(shard_a_);
    begin_dflash_state(shard_b_);
    lane_state.dflash_context_frontier = reuse;
    if (prefill_trace) {
        CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
        CUDA_CHECK(cudaStreamSynchronize(shard_b_.device.stream));
        state_done = Clock::now();
    }

    // The sink is the client's connection, and a connection can die mid-answer. That must cost the
    // request, not the conversation: the states this walk still has to publish describe the device
    // exactly where it will stop, so record the first transport failure, write nothing more to the
    // sink, and rethrow once the terminal state is on the books (execute reads terminal_published to
    // tell a failed request apart from a prefix it can no longer name).
    bool sink_failed = false;
    std::exception_ptr transport_error;
    if (streaming) {
        try {
            sink->start(GenerationStart{.prompt = request.summary, .reused_prompt_tokens = reuse});
        } catch (...) {
            sink_failed     = true;
            transport_error = std::current_exception();
        }
    }
    result.reused_prompt_tokens   = reuse;
    result.prefix_reuse_path =
        reuse_path(reuse, data.context_cache.leading_instruction_frontier.value_or(0), lane);
    // The serial walk finds its own boundary instead of asking the ledger, so the one fact it
    // establishes is whether it reused at all. Filling both axes here keeps them defined on this
    // route too, which is what record_reuse() checks before counting either of them.
    result.reuse_source = reuse_source_kind(lane_state.reuse_source);
    result.reuse_reason = reuse != 0 ? ReuseReasonKind::Used : ReuseReasonKind::NoAnchorAtOrBefore;

    // Sampling config, device-resident, for ops::sample.
    ops::SamplingConfig sampling_config = make_sampling_config(request.sampling);
    auto sampling_scope = shard_a_.workspace->scope();
    auto& ws_a = *shard_a_.workspace;
    auto& ws_b = *shard_b_.workspace;
    // Reserved on the packed row count, which is an upper bound over every token domain ops::sample
    // is called with below.
    const std::size_t sampling_ws = ops::sampling_workspace_capacity_bytes(vocab, 1, 1);
    auto sampling_buf_a = ws_a.alloc_bytes(sizeof(ops::SamplingConfig) + sampling_ws, 256);
    auto sampling_buf_b = ws_b.alloc_bytes(sizeof(ops::SamplingConfig) + sampling_ws, 256);
    // A configured penalty reads the request's committed-token counts through this array, and both
    // ops::sample and the speculative accept kernel add to it as they produce every token. Reset it:
    // the arena hands back the previous request's bytes. With no penalty the op reads no counts at
    // all, so the array is only created when a request actually asks for one.
    if (sampling_config.presence_penalty != 0.0F || sampling_config.frequency_penalty != 0.0F) {
        const Tensor counts = ws_a.alloc(DType::I32, {public_tokens});
        shard_a_.device.bind_to_current_thread();
        CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), shard_a_.device.stream));
        sampling_config.token_counts = static_cast<std::int32_t*>(counts.data);
    }
    shard_a_.device.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(sampling_buf_a.data, &sampling_config,
                               sizeof(ops::SamplingConfig), cudaMemcpyHostToDevice,
                               shard_a_.device.stream));
    shard_b_.device.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(sampling_buf_b.data, &sampling_config,
                               sizeof(ops::SamplingConfig), cudaMemcpyHostToDevice,
                               shard_b_.device.stream));
    const auto* sampling_a = static_cast<const ops::SamplingConfig*>(sampling_buf_a.data);
    const auto* sampling_b = static_cast<const ops::SamplingConfig*>(sampling_buf_b.data);
    Tensor logical_pos_a = ws_a.alloc(DType::I32, {1});
    Tensor logical_pos_b = ws_b.alloc(DType::I32, {1});
    (void)sampling_b;
    (void)logical_pos_b;

    // Constrained-decoding scratch. The mask is computed on the host (declared-name trie over the
    // decoded vocabulary pieces) and uploaded once per constrained step. The device buffer is
    // allocated in the request scope, below every round watermark, so it cannot disturb a captured
    // verify graph.
    const std::int32_t constraint_columns =
        dflash2_enabled_ ? static_cast<std::int32_t>(dflash_drafts_) + 1
        : mtp_enabled_  ? static_cast<std::int32_t>(mtp_drafts_) + 1
                        : 1;
    Tensor tool_mask_dev;
    // The prefill's first token needs a buffer of its own: apply_token_mask requires the mask shape
    // to equal the logits shape, a speculative run's tool_mask_dev is [vocab, drafts + 1], and the
    // prefill samples a single [vocab, 1] column.
    Tensor tool_mask_first;
    std::vector<std::uint8_t> tool_mask_one;
    std::vector<std::uint8_t> tool_mask_columns;
    std::size_t constraint_fed = 0;
    if (tool_constrained) {
        tool_mask_dev   = ws_a.alloc(DType::U8, {vocab, constraint_columns});
        tool_mask_first = ws_a.alloc(DType::U8, {vocab, 1});
        tool_mask_columns.resize(static_cast<std::size_t>(vocab) *
                                 static_cast<std::size_t>(constraint_columns));
    }
    auto constraint_advance = [&]() {
        if (!tool_constrained) { return; }
        const std::string_view raw = request.output.raw_content_text();
        if (raw.size() > constraint_fed) {
            tool_constraint->feed(raw.substr(constraint_fed));
            constraint_fed = raw.size();
        }
    };
    // The mask binds only while the committed output is content: the tool parser reads no
    // reasoning channel, so the grammar must not constrain the thinking phase either.
    auto constraint_live = [&]() {
        return tool_constrained && !request.output.in_reasoning();
    };

    auto& ctx_a = *shard_a_.context;
    auto& ctx_b = *shard_b_.context;

    // Accumulates the answer whether or not the sink still accepts it: the walk's own accounting has
    // to stay complete for the terminal publication below.
    auto publish_preview = [&](bool terminal) {
        if (terminal) { (void)request.output.preview_terminal(request.budget.limit_reason()); }
        auto published = request.output.commit_preview();
        for (auto& delta : published) {
            (delta.channel == OutputChannel::Reasoning ? result.reasoning : result.content) +=
                delta.text;
            if (streaming && !sink_failed) {
                try {
                    sink->publish(delta);
                } catch (...) {
                    sink_failed     = true;
                    transport_error = std::current_exception();
                }
            }
        }
    };
    // Called at every exit of the walk, after the terminal publication and after terminal_published
    // is set: the caller sees the transport failure while the catalog keeps what the walk published.
    auto rethrow_transport_failure = [&]() {
        if (transport_error != nullptr) { std::rethrow_exception(transport_error); }
    };

    // The round scratch is the whole pool, because the fold restores it wholesale on the shard's
    // own stream. Both kinds of copy are taken on that stream, so each one captures the GDN state
    // exactly at its boundary: the enclosing loop has enqueued every earlier token's forward and
    // none of the later ones yet.
    auto snapshot_state = [&](Shard& shard, std::size_t slot) {
        shard.device.bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(shard.state_snapshots[slot].data, shard.state_backing.data,
                                   shard.state_backing.bytes, cudaMemcpyDeviceToDevice,
                                   shard.device.stream));
    };
    // The prefix-reuse planes are shared by every lane (one plane per boundary), so a lane writes
    // only its own slice. A whole-plane copy would overwrite the slice another lane's
    // `cached_boundaries` still describes, and a DeviceSnapshot reuse would then restore a state
    // deeper than the boundary it advertises (P2.3 Stage 2). The round scratch keeps the whole
    // plane: the fold restores it wholesale on the shard's own stream.
    auto snapshot_lane_state = [&](Shard& shard, std::size_t slot) {
        // A slot the host checkpoint ring serves is not carved, so there is no plane to freeze here:
        // the prompt-end checkpoint written at publish is that image (see publish_lane_prefill).
        if (shard.state_snapshots[slot].data == nullptr) { return; }
        shard.device.bind_to_current_thread();
        // The device-to-device branch of copy_lane_state copies *into* its first device pointer, so
        // the plane is the destination and the live pool is the source. Passing them the other way
        // round reads the plane back into the pool instead of freezing the pool into the plane, and
        // the plane is never zeroed at startup (only state_backing is), so the walk would then
        // decode from whatever the arena happened to hold.
        copy_lane_state(shard.lane_state_geometry, shard.state_snapshots[slot].data,
                        static_cast<std::int32_t>(lane), shard.state_backing.data,
                        cudaMemcpyDeviceToDevice, shard.device.stream);
    };
    // The draft's still-uncommitted verify window, committed before the target state it belongs to
    // is published. A round leaves the pending staging for the *next* round to commit (its first
    // append_pending), and a finish commits it itself; the paths that end a walk in between are a
    // cancellation and a decode round that the output policy truncated. The pending staging is
    // deliberately not part of any checkpoint or session image, so this is the only way it reaches
    // one, and it is the TP-2 form of the single-device commit's leading
    // enqueue_dflash_context_append (transactions/commit.cpp:305-319).
    auto flush_dflash_context = [&](std::uint32_t frontier) {
        if (shard_a_.dflash_round == nullptr || lane_state.dflash_context_frontier >= frontier) { return; }
        const qwen::execution::ExecutionCore dflash_execution{
            .device           = shard_a_.device,
            .parameters       = *shard_a_.parameters,
            .work             = *shard_a_.workspace,
            .linear_attention = *shard_a_.state,
            .replay_records   = nullptr,
            .io               = shard_a_.io,
            .prefill_hidden   = shard_a_.prefill_hidden,
            .prefill_chunk    = options_.prefill_chunk,
            .proposal_head    = options_.speculative.proposal_head,
        };
        shard_a_.dflash_round->append_pending(dflash_execution, lane_state.dflash_context_frontier, frontier);
        lane_state.dflash_context_frontier = frontier;
    };
    // The host checkpoint ring is written by snapshot_host_checkpoint, which the batch route calls
    // with its own lane as well.

    // Prefill: batched forwards over chunk-sized slices of the prompt suffix, accumulating KV and
    // GDN state in place. A reused prefix starts the walk at its boundary; those positions keep
    // their K/V from the earlier prefill and the GDN state restored above. A chunk reads every
    // weight once, so the weight traffic that dominates a per-token walk is paid once per chunk
    // instead of once per token.
    //
    // The next request's rewind snapshots can only be frozen on a chunk boundary, so a chunk whose
    // end would step over one of the target depths is truncated to end exactly on it. Slot 0 is the
    // prefill end, which is always a boundary.
    // Chunk width from the engine option, clamped to what the cross-device allreduce staging buffer
    // (DevicePair) and the per-chunk activation peak (workspace) can carry.
    const std::uint32_t prefill_chunk = prefill_chunk_width(shard_a_.model->config().text);
    // The tail anchors cover the last few chunk ends of the walk. Their count is bounded by the tail
    // sub-ring, so a narrow chunk cannot flood the ring with anchors that all sit within one chunk of
    // the prompt end.
    const std::uint32_t tail_span =
        std::min<std::uint32_t>(kReuseTailWindow,
                                prefill_chunk * std::max(1U, host_checkpoint_tail_slots_));
    std::array<std::uint32_t, kReuseSnapshotCount> snapshot_at{};
    snapshot_at[0] = prompt_tokens;
    // Rewind snapshots sit on the walk's own chunk boundaries, which are fixed by `reuse`,
    // `prompt_tokens` and `prefill_chunk` alone. They used to be derived from the last observed
    // divergence (`rewind_near_`), which shortened whichever chunk covered the target: the walk's
    // chunking then depended on engine history, so the same prompt prefilled with different chunk
    // widths from one walk to the next and produced different logits. The dense tail ring already
    // keeps a rewind cheap, which is not worth history-dependent arithmetic here.
    for (std::size_t slot = 1; slot < kReuseSnapshotCount; ++slot) {
        const std::uint32_t span = prompt_tokens - reuse;
        std::uint32_t last       = span % prefill_chunk;
        if (last == 0) { last = std::min(prefill_chunk, span); }
        snapshot_at[slot] = (span > last) ? prompt_tokens - last : 0;
        if (snapshot_at[slot] != 0 && snapshot_at[slot] == reuse) {
            // The restored state already sits exactly on this boundary, and the walk never revisits
            // its own starting point, so freeze it before the first chunk. The draft ring was
            // restored to the same boundary by begin_dflash_state.
            snapshot_lane_state(shard_a_, slot);
            snapshot_lane_state(shard_b_, slot);
            snapshot_dflash_state(shard_a_, slot, lane);
            snapshot_dflash_state(shard_b_, slot, lane);
        }
    }
    // Divergence anchor. shared_prefix is where this prompt stopped matching the lineage it
    // inherited, and the next new session - or the next context compression - renders the same stable
    // block and diverges in the same place, so it is the one position a later request is already
    // known to want. Freezing the state there lets that request restart on the boundary instead of at
    // the grid point behind it. The anchor sits kReuseDivergenceMargin tokens *before* the observed
    // divergence rather than on it: the next pair can report a shared prefix a token or two shorter,
    // and an anchor on the exact index would be pruned before it could ever be used. Only a frontier
    // strictly inside this walk is worth freezing: at or below reuse the walk's own starting state is
    // already checkpointed, and at the prompt end the device snapshot holds it. A checkpoint that
    // already sits there stays untouched, which is what keeps a reused stable prefix free of work.
    // Both anchors and their device-side capture are planned by the shared helper, so this walk and
    // the two batched prefills freeze exactly the same boundaries (P2.3).
    const LaneAnchors anchors = plan_lane_anchors(
        lane, token_ids, prompt_tokens, reuse, static_cast<std::uint32_t>(shared_prefix),
        prefill_chunk, data.context_cache.leading_instruction_frontier.value_or(0));
    const bool anchor_divergence        = anchors.anchor_divergence;
    const std::uint32_t anchor_position = anchors.anchor_position;
    const bool block_anchor             = anchors.block_anchor;
    const std::uint32_t block_position  = anchors.block_position;
    // The device state still stands on the boundary this prompt inherited, so freeze it before the
    // first chunk overwrites it: a later conversation opening with the same stable block can then be
    // recalled onto the owner's own slabs instead of prefilling the block again.
    capture_lane_anchor_from_device(lane, reuse, static_cast<std::uint32_t>(shared_prefix));
    // Set when the prefill's own sample already ended the request: the first token can be a stop
    // token, or it can spend the whole output budget. Decode then has nothing left to do.
    FinishReason first_token_finish = FinishReason::None;
    for (std::uint32_t t0 = reuse; t0 < prompt_tokens;) {
        if (cancellation.requested()) {
            // Chunks that finished wrote KV for tokens the prompt really has and left the GDN state
            // at the end of the last one, so the catalog can name where the walk stopped. Publishing
            // that much costs nothing and lets the retry of the same prompt continue from there
            // instead of prefilling its whole history again. Before the first chunk nothing moved:
            // the recall's bookkeeping already describes the device pools, so that case leaves the
            // catalog exactly as the recall left it.
            if (t0 > 0) {
                snapshot_lane_state(shard_a_, 0);
                snapshot_lane_state(shard_b_, 0);
                snapshot_dflash_state(shard_a_, 0, lane);
                snapshot_dflash_state(shard_b_, 0, lane);
                lane_state.cached_boundaries[0] = t0;
                // A cancelled walk reached no rewind boundary, so every rewind slot this lane still
                // carried is dropped: the slot above is where the walk stopped.
                for (std::size_t slot = 1; slot < kReuseSnapshotCount; ++slot) {
                    lane_state.cached_boundaries[slot] = 0;
                }
                lane_state.cached_state_valid   = true;
                // A prefill chunk commits its own draft window inside the forward, so this is a
                // no-op; it is here so the invariant is stated once: nothing is published with an
                // uncommitted draft window.
                flush_dflash_context(t0);
                // The truncated prompt end is a boundary like a completed one, so it goes into the
                // ring's prompt-end slot and the sweep makes it usable: that is what lets the retry of
                // the same prompt stand on it. It is written after the flush so the checkpoint's draft
                // frontier names the same position.
                snapshot_host_checkpoint(shard_a_, t0, HostRing::PromptEnd, lane);
                snapshot_host_checkpoint(shard_b_, t0, HostRing::PromptEnd, lane);
                publish_lane_checkpoints(lane);
                session_publish(
                    std::vector<TokenId>(token_ids.begin(),
                                         token_ids.begin() + static_cast<std::ptrdiff_t>(t0)),
                    t0, data.context_cache, collect_media_spans(data, t0), lane);
                if (lane_state.active_session != kNoSession) { sessions_[lane_state.active_session].prompt_end = t0; }
            }
            (void)request.output.preview_terminal(FinishReason::Cancelled);
            publish_preview(false);
            result.finish_reason = FinishReason::Cancelled;
            result.timings.total_seconds =
                result.timings.prepare_seconds +
                std::chrono::duration<double>(Clock::now() - request.submitted).count();
            // Prompt wall stays admission-relative; the total above now carries preparation, so the
            // two are no longer the same number.
            result.timings.prompt_wall_seconds =
                std::chrono::duration<double>(Clock::now() - start).count();
            // The partial prefill above named the position the device stopped at - its snapshot, its
            // prompt-end checkpoint and the entry's frontier all say t0 - so this walk published a
            // nameable state too, and a dead transport must not retire it.
            lane_state.terminal_published = true;
            rethrow_transport_failure();
            return result;
        }
        std::uint32_t length = std::min(prefill_chunk, prompt_tokens - t0);
        // A multimodal chunk is capped at the boundary of the item it overlaps, so the encoder hands
        // out one item at a time and the scatter below stays one contiguous column range of it. The
        // snapshot clamp still comes last: the boundary a rewind slot must freeze on wins over the
        // Vision cap, and the next chunk simply re-enters the same item.
        qwen::execution::VisionChunk vision_chunk;
        if (vision_session) {
            vision_chunk = vision_session->prepare_chunk(t0, length);
            length       = static_cast<std::uint32_t>(vision_chunk.length);
        }
        for (std::size_t slot = 1; slot < kReuseSnapshotCount; ++slot) {
            const std::uint32_t target = snapshot_at[slot];
            if (target > t0 && target < t0 + length) { length = target - t0; }
        }
        // The grid checkpoints are the only boundaries a prompt that diverges early can still use,
        // and a checkpoint taken at the first chunk end *past* a stride covers more tokens than the
        // stride asks for: a prompt sharing a stable block shorter than that coverage has nothing to
        // restart on and prefills the block again. Ending the chunk on the stride costs one shorter
        // chunk per stride and puts every grid checkpoint exactly on the tokens before it.
        if (next_host_checkpoint > t0 && next_host_checkpoint < t0 + length) {
            length = next_host_checkpoint - t0;
        }
        if (anchor_divergence && anchor_position > t0 && anchor_position < t0 + length) {
            length = anchor_position - t0;
        }
        // The block boundary is a property of the tokens, not of this walk's chunking, so the chunk
        // that would step over it ends on it and the state frozen there is exact.
        if (block_anchor && block_position > t0 && block_position < t0 + length) {
            length = block_position - t0;
        }
            // The boundary is where this prompt stopped agreeing with the lane's history, so the chunk
            // that would step over it ends on it: an image captured there is only loadable if it sits
            // exactly on that token. One extra short chunk per request is the price.
            if (retention(lane).reuse_boundary_target > t0 &&
                retention(lane).reuse_boundary_target < t0 + length) {
                length = retention(lane).reuse_boundary_target - t0;
            }
        qwen::execution::Tp2VisionChunk media_chunk;
        const qwen::execution::Tp2VisionChunk* media_ptr = nullptr;
        if (media) {
            media_chunk.control       = vision_chunk.control;
            media_chunk.embeddings    = &vision_chunk.embeddings;
            media_chunk.positions     = data.positions.data();
            media_chunk.prompt_tokens = data.token_ids.size();
            media_ptr                 = &media_chunk;
        }
        // Scope both shards' workspaces so each chunk starts from a clean arena. The batched
        // forward allocates all of its intermediate activations in each shard's workspace; without
        // a scope the next chunk's arena would build on this one and eventually overflow.
        auto scope_a = ws_a.scope();
        auto scope_b = ws_b.scope();
        Tensor logits_a = ws_a.alloc(DType::BF16, {vocab, 1});
        Tensor logits_b = ws_b.alloc(DType::BF16, {vocab, 1});
        // MTP priming consumes the chunk's final-norm hidden, so the forward hands it back.
        Tensor mtp_input_a;
        if (mtp_enabled_) {
            mtp_input_a = ws_a.alloc(DType::BF16, {hidden, static_cast<std::int32_t>(length)});
        }
        // The masked draft taps this shard's prefill residual at its configured block ids; the
        // sink is empty unless this shard materialized the draft component (shard 0 under a
        // DFlash/DFlash2 backend), so every other route keeps the NullTap forward.
        auto dflash_sink = make_dflash_prefill_sink(shard_a_);
        ctx_a.forward_tp2_prefill(ctx_b, pair_, std::span<const int>(token_ids.data() + t0, length),
                                  static_cast<std::int32_t>(t0), &logits_a, &logits_b,
                                  mtp_enabled_ ? &mtp_input_a : nullptr, nullptr, nullptr,
                                  qwen::TextPhase::Prefill, media_ptr,
                                  dflash_sink ? &*dflash_sink : nullptr, active_lane_);
        if (dflash_sink) {
            // The sink's consumer ran synchronously inside the forward and appended this chunk's
            // absolute positions to the draft ring, so the context frontier tracks the prefill.
            lane_state.dflash_context_frontier = t0 + length;
        }
        if (mtp_enabled_ && t0 + length != prompt_tokens) {
            mtp_prefill_priming(shard_a_, token_ids.data() + t0, length, t0, mtp_input_a, nullptr,
                                false);
        }
        for (std::size_t slot = 1; slot < kReuseSnapshotCount; ++slot) {
            if (snapshot_at[slot] == t0 + length) {
                snapshot_lane_state(shard_a_, slot);
                snapshot_lane_state(shard_b_, slot);
                snapshot_dflash_state(shard_a_, slot, lane);
                snapshot_dflash_state(shard_b_, slot, lane);
            }
        }
        if (host_checkpoint_stride_ != 0) {
            // One checkpoint per stride, tagged with the frontier this chunk actually reached, so a
            // chunk width that does not divide the stride cannot mislabel a state; plus the dense
            // tail window. The end of the prompt is skipped either way: the prompt-end checkpoint
            // the publish below writes holds that same state already.
            const std::uint32_t frontier = t0 + length;
            const bool on_grid           = frontier >= next_host_checkpoint;
            const bool in_tail           = frontier != prompt_tokens &&
                                 static_cast<std::uint64_t>(frontier) + tail_span > prompt_tokens;
            if (on_grid) {
                snapshot_host_checkpoint(shard_a_, frontier, HostRing::Grid, lane);
                snapshot_host_checkpoint(shard_b_, frontier, HostRing::Grid, lane);
                next_host_checkpoint =
                    (frontier / host_checkpoint_stride_ + 1U) * host_checkpoint_stride_;
            } else if (in_tail) {
                snapshot_host_checkpoint(shard_a_, frontier, HostRing::Tail, lane);
                snapshot_host_checkpoint(shard_b_, frontier, HostRing::Tail, lane);
            }
            if (anchor_divergence && frontier == anchor_position) {
                snapshot_host_checkpoint(shard_a_, frontier, HostRing::Divergence, lane);
                snapshot_host_checkpoint(shard_b_, frontier, HostRing::Divergence, lane);
            }
            if (block_anchor && frontier == block_position) {
                snapshot_host_checkpoint(shard_a_, frontier, HostRing::Block, lane);
                snapshot_host_checkpoint(shard_b_, frontier, HostRing::Block, lane);
                // The id names the walk that wrote it, so the eviction can tell this conversation's
                // own block state from a later conversation's rewrite of the same slot.
                lane_state.block_anchor_position   = block_position;
                lane_state.block_anchor_prefill_id = lane_state.host_checkpoint_live_id;
            }
        }
        if (t0 + length == prompt_tokens) {
            // First token: sample from the last chunk's last-column logits.
            ops::set_i32_scalar(logical_pos_a, static_cast<std::int32_t>(prompt_tokens),
                                shard_a_.device.stream);
            // Path parity with the decode loop, deliberately not a behaviour change: the grammar
            // starts in its free-text position, so build_mask() reports the empty prefix as
            // unconstrained (tests/test_tool_call_constraint.cpp pins that) and no mask is applied
            // here today. Applying the decode path's mask at the same site anyway keeps the prefill
            // path from silently depending on that asymmetry if a constrained first position is
            // ever introduced.
            if (constraint_live()) {
                constraint_advance();
                if (tool_constraint->build_mask(logits_domain, tool_mask_one)) {
                    shard_a_.device.bind_to_current_thread();
                    CUDA_CHECK(cudaMemcpyAsync(tool_mask_first.data, tool_mask_one.data(),
                                               tool_mask_one.size(), cudaMemcpyHostToDevice,
                                               shard_a_.device.stream));
                    ops::apply_token_mask(logits_a, tool_mask_first, shard_a_.device.stream);
                }
            }

            Tensor sampled_a = ws_a.alloc(DType::I32, {1});
            ops::sample(logits_a, sampled_a, public_tokens, sampling_a, logical_pos_a,
                        ops::kSamplePurposePrefill, ws_a, shard_a_.device.stream);
            std::int32_t first = 0;
            shard_a_.device.bind_to_current_thread();
            CUDA_CHECK(cudaMemcpyAsync(&first, sampled_a.data, sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, shard_a_.device.stream));
            CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
            abort_if_ar_stalled();
            // The first token is a generated token like any later one, so it has to reach the
            // output policy before it reaches the client. The policy owns the published text, the
            // reasoning/content split and the stop-token decision; a token that bypasses it is never
            // published, and the pools, the session catalog and the published answer then describe
            // histories one token apart - which is exactly what makes a later turn unable to reuse
            // the tail of this one, however exactly the client replays it.
            const TokenId first_token        = static_cast<TokenId>(first);
            const std::uint32_t first_budget = request.budget.remaining();
            if (first_budget == 0) {
                throw std::logic_error("prefill sampled a token with no output budget left");
            }
            const OutputDecision first_decision = request.output.preview_model(
                std::span<const TokenId>(&first_token, 1), first_budget,
                request.budget.limit_reason());
            if (first_decision.accepted_tokens != 1) {
                throw std::logic_error("output policy rejected the prefill's first token");
            }
            request.generated.push_back(first_token);
            request.budget.commit(1);
            publish_preview(false);
            if (first_decision.finished()) { first_token_finish = first_decision.finish_reason; }
            if (mtp_enabled_) {
                // The final MTP column embeds the token just sampled, so the MTP layer's own K/V for
                // the prompt is appended only after the first token exists.
                mtp_prefill_priming(shard_a_, token_ids.data() + t0, length, t0, mtp_input_a,
                                    &sampled_a, true);
            }
        }
        ++prefill_chunks;
        t0 += length;
    }
    // The divergence anchor goes into the owning entry's shared image before session_publish below,
    // which may erase that entry and shift every later index.
    if (anchor_divergence) { capture_lane_anchor_frozen(lane, anchor_position); }
    if (prefill_trace) {
        CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
        CUDA_CHECK(cudaStreamSynchronize(shard_b_.device.stream));
        walk_done = Clock::now();
    }
    computed_prefill_tokens_ += prompt_tokens - reuse;
    record_reuse(result);
    if (vision_session) {
        // Every item the walk overlapped is encoded and its embeddings are in the KV now, so release
        // the host patch payloads and the handoff binding: the decode loop never revisits them.
        vision_session->release_encoded_media_payloads();
        vision_session->retire_handoff();
    }

    // Freeze the GDN state at every boundary this request can offer the next one. The prompt end is
    // the prefill end the walk just reached; the rewind slots were captured at their chunk
    // boundaries.
    // A slot is published only when this prefill actually reached its boundary, so a request that
    // fails mid-prefill leaves the previous request's boundaries and snapshots intact.
    snapshot_lane_state(shard_a_, 0);
    snapshot_lane_state(shard_b_, 0);
    snapshot_dflash_state(shard_a_, 0, lane);
    snapshot_dflash_state(shard_b_, 0, lane);
    // Every lane keeps its own boundary bookkeeping and its own slice of the snapshot planes, so
    // the walk publishes this lane's prompt into both (P2.3 Stage 2). The two must move together: a
    // plane slice written at one boundary while `cached_boundaries` still names a shallower one would
    // let a DeviceSnapshot reuse restore a state deeper than the boundary it advertises. A slot this
    // walk never reached keeps its zero and is never taken.
    lane_state.cached_prompt_tokens.assign(token_ids.begin(), token_ids.end());
    lane_state.cached_media.assign(prompt_media.begin(), prompt_media.end());
    // The generated span is anchored from here on: the prefill's own boundaries stop at this end.
    lane_state.next_generated_checkpoint = prompt_tokens + kReuseGeneratedTailStride;
    for (std::size_t slot = 0; slot < kReuseSnapshotCount; ++slot) {
        lane_state.cached_boundaries[slot] = snapshot_at[slot];
    }
    lane_state.cached_state_valid = true;
    // The prompt end is a boundary like any other: it goes into the ring's prompt-end slot, which is
    // where a store reads the prompt-end image and where a returning turn stands when the
    // configuration keeps no device plane.
    snapshot_host_checkpoint(shard_a_, prompt_tokens, HostRing::PromptEnd, lane);
    snapshot_host_checkpoint(shard_b_, prompt_tokens, HostRing::PromptEnd, lane);
    // The pools now hold exactly this prompt and the live GDN state sits at its end, so the
    // resident catalog entry describes what the walk just wrote. A conversation that had no entry
    // yet gets one here, before decode extends its history.
    session_publish(token_ids, prompt_tokens, data.context_cache, prompt_media, lane);
    // The entry now describes the prompt the device pools were built from, which is what a later
    // request compares against to tell a later turn of this conversation from a switch away.
    if (lane_state.active_session != kNoSession) { sessions_[lane_state.active_session].prompt_end = prompt_tokens; }
    // Only now are this prefill's checkpoints usable: their state is one the walk reached and their
    // KV prefix is one the walk wrote.
    publish_lane_checkpoints(lane);

    if (prefill_trace) {
        const auto millis = [](Clock::time_point from, Clock::time_point to) {
            return std::chrono::duration<double, std::milli>(to - from).count();
        };
        const Clock::time_point report = Clock::now();
        std::fprintf(stderr,
                     "[tp2-prefill] reuse=%u suffix=%u chunks=%u host_writes=%llu | scan=%.2f "
                     "state=%.2f walk=%.2f post=%.2f total=%.2f ms\n",
                     reuse, prompt_tokens - reuse, prefill_chunks,
                     static_cast<unsigned long long>(prefill_host_writes_), millis(start, scan_done),
                     millis(scan_done, state_done), millis(state_done, walk_done),
                     millis(walk_done, report), millis(start, report));
    }

    // Prompt wall time spans prefill and the first-sample step: the first accepted token is
    // produced by the last prefill iteration. That is a stable commit boundary (the sampled token
    // is read back on shard A after both shards finished the forward), matching the single-device
    // Engine's phase accounting. Generation wall time covers the decode rounds.
    const Clock::time_point first_token_time = Clock::now();
    const double prompt_seconds = std::chrono::duration<double>(first_token_time - start).count();
    // The Vision encode is reported separately from the text walk, matching the single-device route;
    // prompt wall time (the response's TTFT) is the whole span either way.
    result.timings.vision_seconds = vision_session ? vision_session->elapsed_seconds() : 0.0;
    result.timings.prefill_seconds =
        std::max(0.0, prompt_seconds - result.timings.vision_seconds);
    result.timings.prompt_wall_seconds = prompt_seconds;

    // Decode: with MTP each round verifies a K-draft window on both shards, records its
    // linear-attention transitions instead of committing them, and folds only the prefix the target
    // and the output policy accept. Without MTP it is a plain one-token autoregressive loop.
    std::int32_t current = request.generated.back();
    std::uint32_t position = prompt_tokens;
    // MTP round state: the anchor token whose transition is still pending, its position, and the
    // final-norm hidden of the column before it (the bridge's hidden input).
    std::int32_t mtp_anchor    = current;
    std::uint32_t mtp_position = position;
    // Per-round scratch starts here: the per-request tensors above (the sampling buffers, the
    // logical positions) must survive every round, and the prefix ends at the request's first
    // round. Every decode round allocates and frees its own scratch above this watermark, so a
    // round that discards its speculative chain by rewinding to it gets the same allocation
    // layout every time - which is what a captured verify graph bakes into its kernel arguments.
    shard_a_.round_base = ws_a.used();
    shard_b_.round_base = ws_b.used();
    // The route and its spare capacity are fixed for the request, so the per-position acceptance
    // histogram is sized once here - the same shape the single-device program publishes
    // (decode.cpp sizes it from the route's draft window).
    const std::uint32_t draft_window =
        dflash2_enabled_ ? dflash_drafts_ : (mtp_enabled_ ? mtp_drafts_ : 0U);
    result.speculative = SpeculativeStats{
        .backend               = dflash2_enabled_ ? SpeculativeBackend::DFlash2
                                 : (mtp_enabled_ ? SpeculativeBackend::Mtp
                                                 : SpeculativeBackend::None),
        .enabled               = draft_window != 0,
        .draft_window          = draft_window,
        .accepted_per_position = std::vector<std::uint64_t>(draft_window, 0),
    };
    bool finished = first_token_finish != FinishReason::None;
    if (finished) { result.finish_reason = first_token_finish; }
    if (sink_failed) {
        // The first token's delta already met a dead connection. Decoding on would only produce text
        // no client can receive, so stop the way a cancellation does and let the terminal publication
        // below name the frontier the device actually reached.
        result.finish_reason = FinishReason::Cancelled;
        finished             = true;
    }
    while (!finished) {
        if (cancellation.requested()) {
            (void)request.output.preview_terminal(FinishReason::Cancelled);
            publish_preview(false);
            result.finish_reason = FinishReason::Cancelled;
            break;
        }
        if (request.budget.remaining() == 0) {
            (void)request.output.preview_terminal(request.budget.limit_reason());
            publish_preview(false);
            result.finish_reason = request.budget.limit_reason();
            break;
        }
        // Scope both shards' workspaces so each decode round starts from a clean arena (see the
        // prefill loop above).
        auto scope_a = ws_a.scope();
        auto scope_b = ws_b.scope();
        std::vector<TokenId> step;
        Tensor round_hidden;
        // Phase marks: 0 round start, 1 after the MTP proposal chain, 2 after the verify forward,
        // 3 after the licensing kernels, 4 after the licenced-token readback, 5/6 around the state
        // fold. The plain loop has no proposal chain and no fold, so it collapses 1 onto 0.
        const Clock::time_point round_start = Clock::now();
        timing.record(0, shard_a_.device.stream);
        if (!mtp_enabled_) { timing.record(1, shard_a_.device.stream); }
        if (dflash2_enabled_) {
            // One masked-draft round, mirroring the single-device sequence
            // (execution/draft.cpp dflash_decode_batch_body): commit the pending verify window of
            // the previous round into the draft ring, propose the masked block, verify the window
            // through the target with the feature sink installed, then accept with DFlash2's sparse
            // semantics and fold only the committed columns.
            Shard& shard = shard_a_;
            auto& round  = *shard.dflash_round;
            const std::uint32_t budget_remaining = request.budget.remaining();
            const std::uint32_t max_by_budget =
                budget_remaining > 1 ? budget_remaining - 1U : 0U;
            const std::uint32_t lane_window = lane_admission_limit();
            const std::uint32_t capacity_left =
                position + 1U < lane_window ? lane_window - position - 1U : 0U;
            // The draft is never declined, whatever boundary the scan took: the ring beside an
            // unaligned boundary belongs to the walk that froze it, so the block it proposes is that
            // walk's, and the target verify licenses every emitted token either way. The proposal
            // budget is what the request has left to spend and what the context has left to hold.
            const std::uint32_t extent =
                std::min({dflash_drafts_, max_by_budget, capacity_left});
            const std::int32_t width = static_cast<std::int32_t>(dflash_drafts_) + 1;
            // extent == 0 is a real round here, exactly as it is on the single-device route
            // (decode.cpp:732 counts it as a fallback step but still runs the round): the window
            // forwards the anchor column, the verify taps its residual into the pending staging, and
            // the next append - or the finish append below - commits it. A plain decode step would
            // leave the draft context one column short of the target frontier, and that frontier is
            // what a checkpoint or a session recall has to reproduce exactly.
            {
                const qwen::execution::ExecutionCore dflash_execution{
                    .device           = shard.device,
                    .parameters       = *shard.parameters,
                    .work             = *shard.workspace,
                    .linear_attention = *shard.state,
                    .replay_records   = nullptr,
                    .io               = shard.io,
                    .prefill_hidden   = shard.prefill_hidden,
                    .prefill_chunk    = options_.prefill_chunk,
                    .proposal_head    = options_.speculative.proposal_head,
                };
                // Hand off the previous verify window: its columns [C, E) are the target residual
                // the draft context is missing, exactly the prepare_ragged_prefix append the
                // single-device route makes at the head of its round.
                round.append_pending(dflash_execution, lane_state.dflash_context_frontier, position);
                lane_state.dflash_context_frontier = position;
                qwen::DFlashDecodeIngress& ingress = round.ingress();
                ingress                            = {};
                ingress.anchors[0]                 = current;
                ingress.execution_frontiers[0]     = static_cast<std::int32_t>(position);
                ingress.context_frontiers[0]       = static_cast<std::int32_t>(position);
                ingress.proposal_extents[0]        = static_cast<std::int32_t>(extent);
                ingress.proposal_valid_columns[0]  = width;
                ingress.target_valid_columns[0]    = static_cast<std::int32_t>(extent) + 1;
                for (std::int32_t column = 0; column < width; ++column) {
                    const std::int32_t offset =
                        std::min(column, static_cast<std::int32_t>(extent));
                    ingress.target_rope_positions[column] =
                        static_cast<std::int32_t>(position) + offset;
                }
                ingress.text_kv_table_rows[0]      = 0;
                ingress.dflash_kv_table_rows[0]    = 0;
                ingress.active_lanes[0]            = 0;
                ingress.state_source_slots[0]      = 0;
                ingress.state_destination_slots[0] = 0;
                ingress.sampling[0]                = sampling_config;
                const qwen::execution::DFlashEnvelopes envelopes{
                    .local  = {0, position},
                    .full   = {0, position},
                    .append = {0, static_cast<std::uint32_t>(width)},
                };
                shard.device.bind_to_current_thread();
                round.propose(dflash_execution, shard.decoder->text_kv, shard.context.get(),
                              dflash_drafts_, envelopes);
                timing.record(1, shard_a_.device.stream);

                qwen::DFlashDecodeState& frame = round.frame();
                // The frame is sized for the whole lane capacity, but this arm drives exactly one
                // lane, so every capacity-sized tensor is trimmed to lane 0 before use: the ops
                // below derive their batch from the tensor shape, and the flattened window views
                // must agree with the frame's element count. The batched arm trims to its live
                // lane count the same way.
                Tensor frame_anchors           = frame.anchors.slice(0, 0, 1);
                Tensor frame_drafts            = frame.draft_tokens.slice(1, 0, 1);
                Tensor frame_frontiers         = frame.execution_frontiers.slice(0, 0, 1);
                Tensor frame_extents           = frame.proposal_extents.slice(0, 0, 1);
                Tensor frame_verify_ids        = frame.verify_ids.slice(1, 0, 1);
                Tensor frame_verify_positions  = frame.verify_positions.slice(1, 0, 1);
                Tensor frame_target_argmax     = frame.target_argmax.slice(1, 0, 1);
                Tensor frame_candidate_ids     = frame.candidate_ids.slice(2, 0, 1);
                Tensor frame_proposal_q        = frame.proposal_q.slice(2, 0, 1);
                Tensor frame_licensed          = frame.licensed_tokens.slice(1, 0, 1);
                Tensor frame_licensed_counts   = frame.licensed_counts.slice(0, 0, 1);
                Tensor frame_accepted          = frame.accepted_drafts.slice(0, 0, 1);
                Tensor window_logits3d         = frame.target_logits.slice(2, 0, 1);
                Tensor window_logits           = window_logits3d.view({vocab, width});
                Tensor window_hidden =
                    frame.target_hidden.slice(2, 0, 1).view({hidden, width});
                ops::speculative_prepare_verify_inputs(
                    frame_anchors, frame_drafts, frame_frontiers, frame_extents, frame_verify_ids,
                    frame_verify_positions, shard_a_.device.stream);
                // The window is assembled in the same pinned buffer the MTP verify uses, so the
                // eager forward reads the verified ids/positions directly.
                auto* window_ids       = static_cast<std::int32_t*>(verify_window_host_->data());
                auto* window_positions = window_ids + width;
                shard_a_.device.bind_to_current_thread();
                CUDA_CHECK(cudaMemcpyAsync(window_ids, frame_verify_ids.data,
                                           sizeof(std::int32_t) * static_cast<std::size_t>(width),
                                           cudaMemcpyDeviceToHost, shard_a_.device.stream));
                CUDA_CHECK(cudaMemcpyAsync(window_positions, frame_verify_positions.data,
                                           sizeof(std::int32_t) * static_cast<std::size_t>(width),
                                           cudaMemcpyDeviceToHost, shard_a_.device.stream));
                // The window is produced on the device, so unlike the MTP path (whose host builds
                // the buffer directly) both copies above are asynchronous into pinned memory.
                // forward_tp2_window then reads that host buffer to build *each* shard's own copy,
                // and shard B's copy runs on a different stream with no ordering against shard A's
                // D2H. Synchronize before the host buffer becomes the forward's input.
                CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
                // The target verify: RecordForReplay leaves the live GDN state untouched, so the
                // fold below can replay exactly the committed columns from the pre-round snapshot.
                // The feature sink captures the window's residuals for the next round's append.
                snapshot_state(shard_a_, kRoundScratchSlot);
                snapshot_state(shard_b_, kRoundScratchSlot);
                qwen::execution::DFlashFeatureSink verify_sink = round.make_verify_sink();
                const ops::CausalAttentionExecutionEnvelope target_envelope{
                    position + 1U, position + static_cast<std::uint32_t>(width)};
                // DFlash2's window is a device-side product, so both shards have to be drained
                // before the paired layer sequence starts: otherwise the in-kernel allreduce
                // handshake can interleave with an op one shard has not finished yet and the
                // window logits come out a few bf16 ulps apart from run to run, which the
                // near-tied last column then turns into a different token. MTP builds its window on
                // the host and never meets this (tp2-plan.md section 3.6, "S1").
                shard_a_.device.bind_to_current_thread();
                CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
                CUDA_CHECK(cudaDeviceSynchronize());
                shard_b_.device.bind_to_current_thread();
                CUDA_CHECK(cudaStreamSynchronize(shard_b_.device.stream));
                CUDA_CHECK(cudaDeviceSynchronize());
                shard_a_.device.bind_to_current_thread();
                // Only the columns that own their position may append KV: the budget clamp pins the
                // trailing columns to the last valid column's position, and unmasked they would all
                // write that one cache slot from the same launch (nondeterministic winner).
                run_verify_window(window_ids, static_cast<std::int32_t>(position), window_logits,
                                  window_hidden, &verify_sink,
                                  static_cast<std::int32_t>(extent) + 1);
                timing.record(2, shard_a_.device.stream);
                // The declared-name mask has to follow the verify forward that produces the logits:
                // run_verify_window writes window_logits, so applying the mask before it would be
                // overwritten and the route would sample undeclared names freely. The window ids the
                // mask is built from were already made host-visible above.
                if (constraint_live()) {
                    constraint_advance();
                    std::fill(tool_mask_columns.begin(), tool_mask_columns.end(), std::uint8_t{1});
                    bool masked = false;
                    std::string drafted_prefix;
                    for (std::int32_t column = 0; column < width; ++column) {
                        if (tool_constraint->build_mask_after(drafted_prefix, logits_domain,
                                                              tool_mask_one)) {
                            const std::size_t base =
                                static_cast<std::size_t>(column) *
                                static_cast<std::size_t>(vocab);
                            std::copy(tool_mask_one.begin(), tool_mask_one.end(),
                                      tool_mask_columns.begin() +
                                          static_cast<std::ptrdiff_t>(base));
                            masked = true;
                        }
                        if (column + 1 < width) {
                            drafted_prefix.append(tool_constraint->piece(
                                static_cast<std::size_t>(window_ids[column + 1])));
                        }
                    }
                    if (masked) {
                        shard_a_.device.bind_to_current_thread();
                        CUDA_CHECK(cudaMemcpyAsync(tool_mask_dev.data, tool_mask_columns.data(),
                                                   tool_mask_columns.size(),
                                                   cudaMemcpyHostToDevice,
                                                   shard_a_.device.stream));
                        ops::apply_token_mask(window_logits, tool_mask_dev,
                                              shard_a_.device.stream);
                    }
                }
                ops::argmax(window_logits, frame_target_argmax, public_tokens,
                            shard_a_.device.stream);
                // DFlash2 acceptance is the sparse 16-candidate rejection sampler, not MTP's greedy
                // matcher: the draft distribution is the selector's proposal q.
                ops::speculative_accept_sparse_drafts(
                    frame_target_argmax, window_logits3d, frame_drafts, frame_candidate_ids,
                    frame_proposal_q, frame_extents, frame_frontiers, frame_anchors,
                    frame_licensed, frame_licensed_counts, frame_accepted, public_tokens,
                    sampling_a, ops::SpeculativeAcceptExecutionEnvelope{false}, ws_a,
                    shard_a_.device.stream);
                timing.record(3, shard_a_.device.stream);
                std::vector<TokenId> licensed_host(static_cast<std::size_t>(width), 0);
                std::int32_t licensed_count = 0;
                CUDA_CHECK(cudaMemcpyAsync(licensed_host.data(), frame_licensed.data,
                                           sizeof(TokenId) * static_cast<std::size_t>(width),
                                           cudaMemcpyDeviceToHost, shard_a_.device.stream));
                CUDA_CHECK(cudaMemcpyAsync(&licensed_count, frame_licensed_counts.data,
                                           sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                           shard_a_.device.stream));
                timing.record(4, shard_a_.device.stream);
                const Clock::time_point sync_start = Clock::now();
                CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
                shard_b_.device.bind_to_current_thread();
                CUDA_CHECK(cudaStreamSynchronize(shard_b_.device.stream));
                shard_a_.device.bind_to_current_thread();
                timing.close_round(
                    std::chrono::duration<double, std::milli>(Clock::now() - sync_start).count());
                if (licensed_count < 1 || licensed_count > width) {
                    throw std::logic_error("TP-2 DFlash2 round produced an invalid licensed prefix");
                }
                step.assign(licensed_host.begin(),
                            licensed_host.begin() + static_cast<std::ptrdiff_t>(licensed_count));
                if (extent == 0) {
                    result.speculative.fallback_steps += 1;
                } else {
                    result.speculative.rounds += 1;
                    result.speculative.drafted_tokens += extent;
                    result.speculative.accepted_tokens +=
                        static_cast<std::uint32_t>(licensed_count - 1);
                    for (std::int32_t i = 0; i < licensed_count - 1; ++i) {
                        result.speculative
                            .accepted_per_position[static_cast<std::size_t>(i)] += 1;
                    }
                }
            }
        } else if (!mtp_enabled_) {
            Tensor logits_a = run_plain_decode_step(current, position);
            timing.record(2, shard_a_.device.stream);
            ops::set_i32_scalar(logical_pos_a, static_cast<std::int32_t>(position + 1),
                                shard_a_.device.stream);
            if (constraint_live()) {
                constraint_advance();
                if (tool_constraint->build_mask(logits_domain, tool_mask_one)) {
                    shard_a_.device.bind_to_current_thread();
                    CUDA_CHECK(cudaMemcpyAsync(tool_mask_dev.data, tool_mask_one.data(),
                                               tool_mask_one.size(), cudaMemcpyHostToDevice,
                                               shard_a_.device.stream));
                    ops::apply_token_mask(logits_a, tool_mask_dev, shard_a_.device.stream);
                }
            }

            Tensor sampled_a = ws_a.alloc(DType::I32, {1});
            ops::sample(logits_a, sampled_a, public_tokens, sampling_a, logical_pos_a,
                        ops::kSamplePurposeDecode, ws_a, shard_a_.device.stream);
            timing.record(3, shard_a_.device.stream);
            std::int32_t next = 0;
            shard_a_.device.bind_to_current_thread();
            CUDA_CHECK(cudaMemcpyAsync(&next, sampled_a.data, sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, shard_a_.device.stream));
            timing.record(4, shard_a_.device.stream);
            const Clock::time_point sync_start = Clock::now();
            CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
            timing.close_round(
                std::chrono::duration<double, std::milli>(Clock::now() - sync_start).count());
            step.assign(1, static_cast<TokenId>(next));
        } else {
            // One window: [anchor, d0, ..., d_{K-1}] at consecutive positions from the anchor's. The
            // MTP layer's column at (anchor - 1) embeds the anchor token and predicts the token after
            // it, so its drafts become the window's columns 1..K.
            const std::vector<TokenId> drafts =
                mtp_propose_window(shard_a_, shard_a_.mtp_anchor_hidden,
                                   static_cast<std::int32_t>(mtp_anchor), mtp_position - 1, ws_a);
            timing.record(1, shard_a_.device.stream);
            const std::int32_t width = static_cast<std::int32_t>(mtp_drafts_) + 1;
            // Every per-round input to the verify goes through the pinned window: the capture path
            // reads it with a memcpy node, and the eager path passes it straight to the forward.
            auto* window_ids       = static_cast<std::int32_t*>(verify_window_host_->data());
            auto* window_positions = window_ids + width;
            window_ids[0]          = static_cast<std::int32_t>(mtp_anchor);
            window_positions[0]    = static_cast<std::int32_t>(mtp_position);
            for (std::int32_t i = 1; i < width; ++i) {
                window_ids[i] =
                    static_cast<std::int32_t>(drafts[static_cast<std::size_t>(i - 1)]);
                window_positions[i] = static_cast<std::int32_t>(mtp_position) + i;
            }
            // The proposal chain's operands are dead now - the window is assembled on the host and
            // the chain's own arena footprint varies with the round's attention route - so both
            // workspaces go back to where this round's fixed scratch belongs. A captured window
            // fixes that place: everything the verify and the accept path allocate is then a
            // function of the round's fixed sequence alone, which is what the graph bakes into its
            // kernel arguments.
            const std::uint32_t visible_end =
                static_cast<std::uint32_t>(mtp_position) + static_cast<std::uint32_t>(width);
            const WindowGraph* reusable = reusable_window_graph(verify_graphs_, visible_end);
            position_arena(ws_a, shard_a_.round_base,
                           reusable != nullptr ? reusable->round_base[0] : shard_a_.round_base);
            position_arena(ws_b, shard_b_.round_base,
                           reusable != nullptr ? reusable->round_base[1] : shard_b_.round_base);
            Tensor window_logits   = ws_a.alloc(DType::BF16, {vocab, width});
            round_hidden           = ws_a.alloc(DType::BF16, {hidden, width});
            Tensor window_drafts =
                ws_a.alloc(DType::I32, {static_cast<std::int32_t>(mtp_drafts_), 1});
            Tensor target_tokens   = ws_a.alloc(DType::I32, {width, 1});
            Tensor current_extents = ws_a.alloc(DType::I32, {1});
            Tensor round_lengths   = ws_a.alloc(DType::I32, {1});
            Tensor round_anchors   = ws_a.alloc(DType::I32, {1});
            Tensor licensed        = ws_a.alloc(DType::I32, {width, 1});
            Tensor licensed_counts = ws_a.alloc(DType::I32, {1});
            Tensor accepted        = ws_a.alloc(DType::I32, {1});
            shard_a_.device.bind_to_current_thread();
            // From the pinned window, not from the proposal chain's arena: that region has been
            // reused by the allocations above.
            CUDA_CHECK(cudaMemcpyAsync(window_drafts.data, window_ids + 1,
                                       sizeof(TokenId) * mtp_drafts_, cudaMemcpyHostToDevice,
                                       shard_a_.device.stream));
            ops::set_i32_scalar(current_extents, static_cast<std::int32_t>(mtp_drafts_),
                                shard_a_.device.stream);
            ops::set_i32_scalar(round_lengths, static_cast<std::int32_t>(mtp_position),
                                shard_a_.device.stream);
            ops::set_i32_scalar(round_anchors, mtp_anchor, shard_a_.device.stream);
            // RecordForReplay records the window's transitions but also advances the live state, so the
            // pre-verify state is snapshotted and the fold replays only the committed columns from it.
            snapshot_state(shard_a_, kRoundScratchSlot);
            snapshot_state(shard_b_, kRoundScratchSlot);
            // The verify window runs Phase::Prefill through the capture-safe window path: the
            // Phase::Verify (decode-equivalent) variant is a known-incomplete TP-2 path (batched
            // GDN/attention faults), so MTP stays approximate. run_verify_window owns the GDN
            // record/replay action around the launch, and replays a captured graph of the whole
            // window when one covers this round's extent.
            run_verify_window(window_ids, static_cast<std::int32_t>(mtp_position), window_logits,
                              round_hidden);
            timing.record(2, shard_a_.device.stream);
            if (constraint_live()) {
                constraint_advance();
                // Column c is drawn after the draft columns before it, and those are exactly the
                // columns a round commits before its first mismatch, so the position the mask is
                // built from is the position that column is consumed at.
                std::fill(tool_mask_columns.begin(), tool_mask_columns.end(), std::uint8_t{1});
                bool masked = false;
                std::string drafted_prefix;
                for (std::int32_t column = 0; column < width; ++column) {
                    if (tool_constraint->build_mask_after(drafted_prefix, logits_domain,
                                                       tool_mask_one)) {
                        const std::size_t base =
                            static_cast<std::size_t>(column) * static_cast<std::size_t>(vocab);
                        std::copy(tool_mask_one.begin(), tool_mask_one.end(),
                                  tool_mask_columns.begin() +
                                      static_cast<std::ptrdiff_t>(base));
                        masked = true;
                    }
                    if (column + 1 < width) {
                        drafted_prefix.append(tool_constraint->piece(
                            static_cast<std::size_t>(window_ids[column + 1])));
                    }
                }
                if (masked) {
                    shard_a_.device.bind_to_current_thread();
                    CUDA_CHECK(cudaMemcpyAsync(tool_mask_dev.data, tool_mask_columns.data(),
                                               tool_mask_columns.size(), cudaMemcpyHostToDevice,
                                               shard_a_.device.stream));
                    ops::apply_token_mask(window_logits, tool_mask_dev, shard_a_.device.stream);
                }
            }
            ops::argmax(window_logits, target_tokens, public_tokens, shard_a_.device.stream);
            ops::speculative_accept_greedy_drafts(
                target_tokens, window_logits, window_drafts, current_extents, round_lengths,
                round_anchors, licensed, licensed_counts, accepted, public_tokens, sampling_a,
                ws_a, shard_a_.device.stream);
            timing.record(3, shard_a_.device.stream);
            std::vector<TokenId> licensed_host(static_cast<std::size_t>(width), 0);
            std::int32_t licensed_count = 0;
            CUDA_CHECK(cudaMemcpyAsync(licensed_host.data(), licensed.data,
                                       sizeof(TokenId) * width, cudaMemcpyDeviceToHost,
                                       shard_a_.device.stream));
            CUDA_CHECK(cudaMemcpyAsync(&licensed_count, licensed_counts.data, sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, shard_a_.device.stream));
            timing.record(4, shard_a_.device.stream);
            const Clock::time_point sync_start = Clock::now();
            CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
            // Both devices must be idle before the host moves the round on: the fold, the state
            // restore and the next round's window all rewrite buffers the peer's verify may still be
            // reading, and the pair's allreduce is the only thing that orders the two devices'
            // kernels against each other.
            shard_b_.device.bind_to_current_thread();
            CUDA_CHECK(cudaStreamSynchronize(shard_b_.device.stream));
            shard_a_.device.bind_to_current_thread();
            timing.close_round(
                std::chrono::duration<double, std::milli>(Clock::now() - sync_start).count());
            if (licensed_count < 1 || licensed_count > width) {
                throw std::logic_error("TP-2 MTP round produced an invalid licensed prefix");
            }
            step.assign(licensed_host.begin(),
                        licensed_host.begin() + static_cast<std::ptrdiff_t>(licensed_count));
            result.speculative.rounds += 1;
            result.speculative.drafted_tokens += mtp_drafts_;
            result.speculative.accepted_tokens +=
                static_cast<std::uint32_t>(licensed_count - 1);
            for (std::int32_t i = 0; i < licensed_count - 1; ++i) {
                result.speculative.accepted_per_position[static_cast<std::size_t>(i)] += 1;
            }
        }
        // Every round converges here before any token reaches the client, which is the last point at
        // which a stalled transport can be caught without serving a partial sum as an answer.
        abort_if_ar_stalled();
        // A speculative round can license more tokens than the request still has budget for; the
        // output policy only ever commits a prefix of what it is shown.
        const std::uint32_t remaining = request.budget.remaining();
        if (step.size() > remaining) { step.resize(remaining); }
        const OutputDecision decision =
            request.output.preview_model(step, remaining, request.budget.limit_reason());
        if (decision.accepted_tokens == 0 || decision.accepted_tokens > step.size()) {
            throw std::logic_error("TP-2 output policy returned an invalid licensed prefix");
        }
        request.generated.insert(request.generated.end(), step.begin(),
                                 step.begin() + static_cast<std::ptrdiff_t>(decision.accepted_tokens));
        request.budget.commit(decision.accepted_tokens);
        committed_decode_tokens_ += decision.accepted_tokens;
        ++decode_rounds_;
        // preview_model already established the (possibly terminal) preview; commit it as-is.
        publish_preview(false);
        if (dflash2_enabled_) {
            // The output policy licenses a prefix of the round's tokens. The verify recorded
            // the window's GDN transitions and advanced the live state; undo that and replay
            // exactly the committed columns, as the single-device resolve does.
            const std::uint32_t committed = decision.accepted_tokens;
            timing.record(5, shard_a_.device.stream);
            for (Shard* fold_shard : {&shard_a_, &shard_b_}) {
                fold_shard->device.bind_to_current_thread();
                CUDA_CHECK(cudaMemcpyAsync(
                    fold_shard->state_backing.data,
                    fold_shard->state_snapshots[kRoundScratchSlot].data,
                    fold_shard->state_backing.bytes, cudaMemcpyDeviceToDevice,
                    fold_shard->device.stream));
            }
            const std::int32_t fold_slots[1]   = {static_cast<std::int32_t>(lane)};
            const std::int32_t fold_columns[1] = {static_cast<std::int32_t>(committed)};
            fold_verify_window(fold_slots, fold_columns, 1);
            // The target frontier advances by exactly the committed columns. The draft ring
            // stays one verify window behind, except when this round ends the request: then it
            // is caught up to the final frontier, which is the single-device rule.
            const std::uint32_t base         = position;
            position                         = base + committed;
            const bool dflash_finished       = decision.finished();
            lane_state.dflash_context_frontier         = dflash_finished ? position : base;
            if (dflash_finished && position > base) {
                const qwen::execution::ExecutionCore dflash_execution{
                    .device           = shard_a_.device,
                    .parameters       = *shard_a_.parameters,
                    .work             = *shard_a_.workspace,
                    .linear_attention = *shard_a_.state,
                    .replay_records   = nullptr,
                    .io               = shard_a_.io,
                    .prefill_hidden   = shard_a_.prefill_hidden,
                    .prefill_chunk    = options_.prefill_chunk,
                    .proposal_head    = options_.speculative.proposal_head,
                };
                shard_a_.dflash_round->append_pending(dflash_execution, base, position);
            }
            current = static_cast<std::int32_t>(step[committed - 1]);
            timing.record(6, shard_a_.device.stream);
        } else if (!mtp_enabled_) {
            current = step.back();
            ++position;
        } else {
            // The output policy licenses a prefix of the round's tokens. The state advances by exactly
            // those columns: their transitions were recorded, not applied.
            const std::uint32_t committed = decision.accepted_tokens;
            timing.record(5, shard_a_.device.stream);
            // Undo the verify's advance: restore the pre-round state, then fold the committed columns.
            for (Shard* shard : {&shard_a_, &shard_b_}) {
                shard->device.bind_to_current_thread();
                CUDA_CHECK(cudaMemcpyAsync(shard->state_backing.data,
                                           shard->state_snapshots[kRoundScratchSlot].data,
                                           shard->state_backing.bytes, cudaMemcpyDeviceToDevice,
                                           shard->device.stream));
            }
            const std::int32_t fold_slots[1]   = {static_cast<std::int32_t>(lane)};
            const std::int32_t fold_columns[1] = {static_cast<std::int32_t>(committed)};
            fold_verify_window(fold_slots, fold_columns, 1);
            // The next anchor is the last committed token; its own transition stays pending, so the
            // next bridge reads the final-norm hidden of the column before it.
            mtp_anchor   = static_cast<std::int32_t>(step[committed - 1]);
            mtp_position = mtp_position + committed;
            const std::int32_t source_column = static_cast<std::int32_t>(committed) - 1;
            shard_a_.device.bind_to_current_thread();
            CUDA_CHECK(cudaMemcpyAsync(
                shard_a_.mtp_anchor_hidden.data,
                static_cast<const char*>(round_hidden.data) +
                    static_cast<std::size_t>(source_column) *
                        static_cast<std::size_t>(hidden) * sizeof(std::uint16_t),
                shard_a_.mtp_anchor_hidden.bytes(), cudaMemcpyDeviceToDevice,
                shard_a_.device.stream));
            timing.record(6, shard_a_.device.stream);
        }
        finished = decision.finished();
        if (finished) { result.finish_reason = decision.finish_reason; }
        if (sink_failed && !finished) {
            // This round's committed columns are already folded into the device state above, so
            // stopping here leaves that state exactly at the frontier the terminal publication below
            // names. Decoding on would only build output for a connection that is already gone.
            result.finish_reason = FinishReason::Cancelled;
            finished             = true;
        }
        timing.committed += decision.accepted_tokens;
        ++timing.rounds;
        timing.round_ms +=
            std::chrono::duration<double, std::milli>(Clock::now() - round_start).count();

        // Decode advances the frontier, and a later turn that re-renders the answer it was handed
        // stops matching inside this generated span. Prefill leaves no boundary here, so without
        // these writes the deepest checkpoint behind a replayed answer sits at the prompt end and
        // the whole answer is re-prefilled. Write the position grid the prefill writes, at the
        // frontier this round actually committed - the same count the final publish names - so the
        // loss a replay cannot avoid is bounded by one stride instead of by the answer. The draft
        // image has to describe the frontier the checkpoint names, so hand it the pending window
        // first; the final boundary is skipped because the prompt-end checkpoint and the live state
        // both already hold it.
        if (host_checkpoint_stride_ != 0 && !finished) {
            const std::uint32_t committed_frontier =
                prompt_tokens + static_cast<std::uint32_t>(request.generated.size()) - 1U;
            if (committed_frontier >= next_host_checkpoint) {
                flush_dflash_context(committed_frontier);
                snapshot_host_checkpoint(shard_a_, committed_frontier, HostRing::Grid, lane);
                snapshot_host_checkpoint(shard_b_, committed_frontier, HostRing::Grid, lane);
                next_host_checkpoint =
                    (committed_frontier / host_checkpoint_stride_ + 1U) * host_checkpoint_stride_;
            }
        }
    }

    // Publish the finished conversation: the pools hold prompt plus committed output and the live
    // GDN state sits at that frontier, which is exactly what a returning turn extends and what an
    // eviction copies out.
    {
        std::vector<TokenId> history = token_ids;
        history.insert(history.end(), request.generated.begin(), request.generated.end());
        // The last sampled token is not forwarded yet, so the device state stops one token short
        // of the history the next turn will send back. A walk that produced no token at all is
        // already a logic error elsewhere; publishing the prompt end keeps the catalog honest.
        const std::uint32_t sampled = static_cast<std::uint32_t>(request.generated.size());
        const std::uint32_t frontier =
            sampled == 0 ? prompt_tokens : prompt_tokens + sampled - 1U;
        // A decode round the walk did not survive to the next round - a cancellation, or a round the
        // output policy truncated - left its committed columns in the pending staging. The published
        // entry names exactly the target frontier, and its draft image has to reach it, so commit the
        // remainder first. A finished round already did this itself.
        flush_dflash_context(frontier);
        session_publish(history, frontier, data.context_cache, prompt_media, lane);
    }
    // Only now are the checkpoints decode wrote usable, on the same terms as the prefill's: their
    // state is one this walk committed and their KV prefix is one it wrote. A walk that threw
    // before this point leaves them invalid, so a torn round can never be recalled.
    {
        Shard* const shards[2] = {&shard_a_, &shard_b_};
        for (Shard* shard : shards) {
            for (auto& checkpoint : shard->host_checkpoints) {
                if (checkpoint.prefill_id[lane] == lane_state.host_checkpoint_live_id) {
                    checkpoint.valid[lane] = true;
                }
            }
        }
    }
    // The catalog now names the frontier the device stopped at, the ring holds this walk's own
    // checkpoints and the live shortcut is honest, so record that and hand a recorded transport
    // failure back to the caller: the request failed, the reusable prefix did not.
    lane_state.terminal_published = true;
    rethrow_transport_failure();
    result.generated_token_ids = std::move(request.generated);
    result.tool_calls          = request.output.take_tool_calls();
    result.tool_call_parse     = request.output.tool_call_parse_diagnostics();
    result.reasoning_tokens    = request.output.reasoning_tokens();
    result.matched_stop_string = request.output.matched_stop_string();
    result.thinking            = request.output.thinking_stats();
    const Clock::time_point finished_at = Clock::now();
    result.timings.decode_seconds =
        std::chrono::duration<double>(finished_at - first_token_time).count();
    result.timings.generation_wall_seconds = result.timings.decode_seconds;
    result.timings.first_token_seconds =
        result.timings.prepare_seconds + result.timings.prompt_wall_seconds;
    // The total span starts where the core accepted the request, matching the single-device route;
    // the serve layer subtracts prepare_seconds from it, so a span that already excluded preparation
    // read back as zero for any generation shorter than its own preparation.
    result.timings.total_seconds =
        result.timings.prepare_seconds +
        std::chrono::duration<double>(finished_at - request.submitted).count();
    timing.prefill_ms = result.timings.prefill_seconds * 1000.0;
    timing.report("decode");
    return result;
}

LoadSummary TP2GenerationCore::load_summary() const {
    LoadSummary summary;
    const auto& model = *shard_a_.model;
    summary.architecture =
        std::string(models::architecture_name(model.config().text.architecture));
    summary.model_name   = model.info().name;
    std::set<std::string> formats;
    for (const auto& weight : model.weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    const auto& stats = model.storage_stats();
    summary.load_seconds         = load_seconds_;
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.host_mapped_bytes    = stats.host_mapped_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    return summary;
}

MemorySummary TP2GenerationCore::memory_summary() const {
    MemorySummary summary;
    summary.device        = options_.device;
    summary.max_context   = options_.max_context;
    summary.kv_capacity   = options_.max_context;
    summary.kv_cache      = options_.kv_cache;
    summary.workspace     = {.capacity_bytes = workspace_bytes_, .used_bytes = 0,
                             .peak_used_bytes = shard_a_.workspace->peak_used()};
    return summary;
}

std::vector<float> TP2GenerationCore::score(qwen::PreparedPrompt&& prompt,
                                           std::uint32_t first_target) {
    const qwen::PreparedPromptData data = qwen::PreparedPromptAccess::take(std::move(prompt));
    const std::vector<TokenId>& tokens  = data.token_ids;
    if (tokens.size() < 2) {
        throw std::invalid_argument("TP-2 causal score needs at least two tokens");
    }
    if (first_target == 0 || static_cast<std::size_t>(first_target) >= tokens.size()) {
        throw std::invalid_argument("TP-2 causal score first_target must be in [1,token_count-1]");
    }
    if (tokens.size() > options_.max_context) {
        throw std::invalid_argument("TP-2 causal score token count exceeds max_context");
    }
    if (data.has_media()) {
        throw std::invalid_argument("TP-2 causal score accepts text tokens only");
    }

    // The shards, the paged KV pool and the DevicePair belong to one walk at a time, and the
    // workspaces are not thread-safe; scoring serializes against generation through the same mutex
    // the request queue uses.
    std::lock_guard<std::mutex> lock(execution_mutex_);

    const std::uint32_t token_count     = static_cast<std::uint32_t>(tokens.size());
    const std::uint32_t predictor_count = token_count - 1U;
    // Token i's logits predict token i+1, so the scored range of predictor columns starts one
    // before the first target -- the mapping the single-device Program::causal_score walk uses.
    const std::uint32_t first_predictor = first_target - 1U;
    const std::int32_t vocab = qwen::execution::dimension(shard_a_.model->config().text.vocab_size);
    const std::int32_t public_tokens =
        static_cast<std::int32_t>(shard_a_.model->resources().public_token_count);
    const std::uint32_t tile =
        std::min<std::uint32_t>(kScoreTile, prefill_chunk_width(shard_a_.model->config().text));
    if (tile == 0) { throw std::logic_error("TP-2 causal score has no prefill chunk width"); }

    // A walk starts from zeroed linear-attention state; the recurrent state is the only thing that
    // carries across positions without being rewritten, so it is the only thing that must be reset.
    // The paged KV needs no reset: every chunk writes positions [first_position, first_position +
    // chunk) before its attention reads them, and the envelope the forward binds ends at the chunk,
    // so an earlier walk's stale tail is never inside this walk's causal window. Both memsets are
    // enqueued on their own device's stream, so the forward that follows is ordered after them.
    shard_a_.device.bind_to_current_thread();
    CUDA_CHECK(cudaMemsetAsync(shard_a_.state_backing.data, 0, shard_a_.state_backing.bytes,
                               shard_a_.device.stream));
    shard_b_.device.bind_to_current_thread();
    CUDA_CHECK(cudaMemsetAsync(shard_b_.state_backing.data, 0, shard_b_.state_backing.bytes,
                               shard_b_.device.stream));
    shard_a_.device.bind_to_current_thread();

    std::vector<float> output;
    output.reserve(token_count - first_target);
    std::vector<std::int32_t> targets;
    targets.reserve(tile);
    std::vector<float> host(tile, 0.0F);

    for (std::uint32_t cursor = 0; cursor < predictor_count;) {
        const std::uint32_t chunk = std::min(tile, predictor_count - cursor);
        // The caller owns the per-column logits buffer, so it is allocated inside this iteration's
        // arena scope: the forward allocates its internals above it and the scope reclaims both. The
        // peer allocates its own copy of the same tile in its arena, which is why build_shard sizes
        // both workspaces for the scoring route.
        auto scope_a = shard_a_.workspace->scope();
        auto scope_b = shard_b_.workspace->scope();
        Tensor logits =
            shard_a_.workspace->alloc(DType::BF16, {vocab, static_cast<std::int32_t>(chunk)});
        try {
            shard_a_.context->forward_tp2_prefill(
                *shard_b_.context, pair_,
                std::span<const int>(tokens.data() + cursor, static_cast<std::size_t>(chunk)),
                static_cast<std::int32_t>(cursor), nullptr, nullptr, nullptr, &logits, nullptr,
                qwen::TextPhase::Prefill, nullptr, nullptr, active_lane_);
        } catch (const std::bad_alloc&) {
            std::fprintf(stderr,
                         "[score] FORWARD OVERFLOW chunk=%u peak_a=%zu used_a=%zu cap_a=%zu | "
                         "peak_b=%zu used_b=%zu cap_b=%zu\n",
                         chunk, shard_a_.workspace->peak_used(), shard_a_.workspace->used(),
                         shard_a_.workspace->capacity(), shard_b_.workspace->peak_used(),
                         shard_b_.workspace->used(), shard_b_.workspace->capacity());
            throw;
        }

        // A predictor column scores only when its target is in the request's range: chunks that end at
        // or before first_target contribute nothing (the prefill that first_predictor does not move),
        // and the chunk holding the last token has one column fewer than chunk. This walk's tile is
        // narrower than the single-device walk's 1024-token chunk, so a request can start several
        // tiles in and leave whole chunks with no scored column at all.
        const std::uint32_t selected   = std::max(cursor, first_predictor);
        const std::uint32_t scored_end = cursor + chunk;
        const std::uint32_t count      = scored_end > selected ? scored_end - selected : 0U;
        std::int32_t scored_columns    = 0;
        if (count != 0) {
            scored_columns = static_cast<std::int32_t>(count);
            Tensor target_ids = shard_a_.workspace->alloc(DType::I32, {scored_columns});
            Tensor logprobs   = shard_a_.workspace->alloc(DType::FP32, {scored_columns});
            targets.assign(tokens.begin() + selected + 1U, tokens.begin() + selected + 1U + count);
            shard_a_.device.bind_to_current_thread();
            CUDA_CHECK(cudaMemcpyAsync(target_ids.data, targets.data(), target_ids.bytes(),
                                       cudaMemcpyHostToDevice, shard_a_.device.stream));
            // Slicing the tile along the column axis keeps the block contiguous, which is what the
            // op's contract requires; the columns before `selected - cursor` belong to a window the
            // chunk shares with the previous stride and are deliberately not scored twice.
            ops::target_logprobs(
                logits.slice(1, static_cast<std::int32_t>(selected - cursor), scored_columns),
                target_ids, public_tokens, logprobs, shard_a_.device.stream);
            CUDA_CHECK(cudaMemcpyAsync(host.data(), logprobs.data, logprobs.bytes(),
                                       cudaMemcpyDeviceToHost, shard_a_.device.stream));
        }
        // The forward's trailing all-reduce runs on the peer's stream, so draining the driver stream
        // is what retires this chunk: it makes both arenas safe to reuse for the next one and it
        // completes the log-probability copy read back below. A chunk that scores nothing still has
        // to drain, because the next chunk reuses the peer tile this one left in flight.
        CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
        abort_if_ar_stalled();
        if (scored_columns != 0) {
            output.insert(output.end(), host.begin(), host.begin() + scored_columns);
        }
        cursor += chunk;
    }

    if (output.size() != static_cast<std::size_t>(token_count - first_target)) {
        throw std::logic_error("TP-2 causal score produced the wrong number of logprobs");
    }
    return output;
}

void TP2GenerationCore::record_reuse(const GenerationResult& result) {
    // The two axes partition one population, so a record that names a storage without naming a use,
    // or a use without a storage, is a broken producer rather than a request worth counting.
    const bool served = result.reuse_source != ReuseSourceKind::None;
    if ((result.reused_prompt_tokens > 0) != served ||
        served != (result.reuse_reason == ReuseReasonKind::Used)) {
        throw std::logic_error("reuse record contradicts itself: tokens=" +
                               std::to_string(result.reused_prompt_tokens) + " source=" +
                               std::to_string(static_cast<int>(result.reuse_source)) + " reason=" +
                               std::to_string(static_cast<int>(result.reuse_reason)));
    }
    switch (result.reuse_source) {
    case ReuseSourceKind::Device: ++reuse_stats_.reuse_source_device; break;
    case ReuseSourceKind::HostSlab: ++reuse_stats_.reuse_source_host_slab; break;
    case ReuseSourceKind::LiveState: ++reuse_stats_.reuse_source_live_state; break;
    case ReuseSourceKind::None: ++reuse_stats_.reuse_source_none; break;
    }
    if (result.reuse_reason == ReuseReasonKind::Used) {
        ++reuse_stats_.reuse_reason_used;
    } else {
        ++reuse_stats_.reuse_reason_no_anchor;
    }
    reuse_stats_.reused_prompt_tokens += result.reused_prompt_tokens;
    reuse_stats_.last_reuse_tokens = result.reused_prompt_tokens;
    reuse_stats_.last_reuse_source = result.reuse_source;
    reuse_stats_.last_reuse_reason = result.reuse_reason;
}

RuntimeStats TP2GenerationCore::runtime_stats() const {
    RuntimeStats stats            = reuse_stats_;
    stats.computed_prefill_tokens = computed_prefill_tokens_;
    stats.committed_decode_tokens = committed_decode_tokens_;
    stats.decode_rounds           = decode_rounds_;
    stats.decode_row_rounds       = decode_rounds_;
    return stats;
}

void TP2GenerationCore::reset_memory_peaks() noexcept {
    shard_a_.workspace->reset_peak();
    shard_b_.workspace->reset_peak();
}

} // namespace ninfer::runtime
