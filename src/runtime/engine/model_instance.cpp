#include "runtime/engine/model_instance.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/startup.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"
#include "product/speculative_options.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {
namespace {

// TP-2 keeps one resident conversation plus this many host-resident ones by default. The host KV
// budget is the real bound on how many fit; this cap only stops the catalog metadata from growing
// without limit, and --max-private-continuations overrides it.
constexpr std::uint32_t kTp2DefaultSessions = 6;

bool is_tp2_generation(const EngineOptions& options) noexcept {
    return options.device_b >= 0 && options.purpose == EnginePurpose::Generation;
}
using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    if (options.vision_item_tokens < kMinimumVisionItemTokens ||
        options.vision_item_tokens > kMaximumVisionItemTokens) {
        throw std::invalid_argument("Engine vision_item_tokens must be within [2048, 16384]");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.lane_context > options.max_context) {
        throw std::invalid_argument("Engine lane_context must be 0 or at most max_context");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    if (options.device_b >= 0 && options.purpose == EnginePurpose::Generation) {
        // Dedicated tensor-parallel (TP-2) generation core: no CUDA Graphs, no context cache, and a
        // page-aligned KV capacity sized to the full context (the core builds its own paged cache
        // from max_context).
        //
        // The core owns its own round loop, so it - not this normalization - is the authority on how
        // many requests it can run at once. Clamp to what it can batch instead of forcing one and
        // silently ignoring what the operator asked for; the clamp lives in one constant that the
        // service layer reads too, so accepted capacity and executed capacity cannot drift apart.
        // `max_pending_requests` is the depth of the FIFO in front of that concurrency and is the
        // operator's to set.
        //
        // The plain, MTP and DFlash2 rounds all carry a batch. `--spec dflash` has no TP-2 context
        // layout and is rejected at the end of this normalization (see below), so no route silently
        // discards lanes: what is left is a straight clamp to the route's batch width, which the
        // operator is told about below instead of inferring it from the accepted capacity.
        const std::uint32_t requested_concurrency = options.max_concurrency;
        options.max_concurrency =
            tp2_generation_concurrency(requested_concurrency, options.speculative.backend);
        if (options.max_concurrency != requested_concurrency) {
            std::fprintf(stderr,
                         "[tp2] --max-concurrency %u exceeds the %u lanes the %s route batches; "
                         "running %u lanes (--max-pending-requests is unchanged)\n",
                         requested_concurrency, options.max_concurrency,
                         product::speculative_backend_name(options.speculative.backend),
                         options.max_concurrency);
        }
        // `--lane-context` is a TP-2 policy knob: a ceiling wider than the logical context is
        // rejected rather than silently clamped, so the operator's number and the ceiling the core
        // advertises never disagree. 0 keeps the whole-pool ceiling (S1 of the kv-sharing plan).
        if (options.lane_context > options.max_context) {
            throw std::invalid_argument("TP-2 lane_context must be 0 or at most max_context");
        }
        // The TP-2 core reads the prefill chunk itself. Clamp the request to the range its
        // cross-device allreduce staging buffer and its per-chunk activation peak can carry
        // (0 selects the default width).
        {
            std::uint32_t chunk = options.prefill_chunk == 0 ? 1024U : options.prefill_chunk;
            if (chunk < 128U) { chunk = 128U; }
            if (chunk > 1024U) { chunk = 1024U; }
            options.prefill_chunk = chunk - chunk % 128U;
        }
        // The overlap splits a chunk at an activation/MMA schedule boundary, so its sub-block width
        // stays 64-aligned. A width wider than half the chunk disables the split instead of narrowing
        // it, which keeps every enabled chunk on a measured width.
        options.prefill_overlap -= options.prefill_overlap % 64U;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        // The TP-2 core drives its own single-request round loop. MTP proposes from the artifact's
        // own nextn head. DFlash2 also owns a draft component the shard split materializes whole on
        // shard 0, and the core implements its masked proposal, target verify, sparse acceptance and
        // GDN fold (tp2-plan.md 3.6, stages B3-B6). Its verify window masks the budget-clamped
        // columns out of the KV append, so the route reproduces a from-scratch walk token for token.
        // DFlash v1 still has no TP-2 context layout, and its draft state would go through the
        // per-layer path this core does not own.
        if (options.speculative.backend == SpeculativeBackend::DFlash) {
            throw std::invalid_argument(
                "TP-2 generation supports --spec mtp and --spec dflash2 (not --spec dflash)");
        }
        // Vision is available on the TP-2 route: the artifact's static shard split places the
        // Vision tower on the shard that holds the vision component (shard 1) and the MTP layer on
        // shard 0, so both fit under the 262,144-token KV ceiling.
        options.use_cuda_graph       = false;
        // The generation core keeps its own prefix-reuse checkpoints in pinned host memory rather
        // than building a context cache, so the host state-image budget survives this reset. The
        // host KV budget and the session count are the inputs of its cross-session retention, so
        // they survive too: a zero host KV budget disables that retention.
        const std::uint32_t host_state_slots = options.context_cache.host_state_slots;
        const std::size_t host_kv_capacity   = options.context_cache.host_kv_capacity_bytes;
        const bool host_kv_pinned            = options.context_cache.host_kv_pinned;
        const std::uint32_t session_capacity =
            options.context_cache.max_private_continuations.value_or(kTp2DefaultSessions);
        const std::uint32_t retention_floor = options.context_cache.session_retention_floor_tokens;
        options.context_cache = ContextCacheOptions{.enabled        = false,
                                                    .host_state_slots = host_state_slots,
                                                    .host_kv_capacity_bytes = host_kv_capacity,
                                                    .session_retention_floor_tokens = retention_floor,
                                                    .host_kv_pinned = host_kv_pinned,
                                                    .max_private_continuations = session_capacity};
    }
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.prefill_chunk        = 1024;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{.enabled = false};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    // TP-2 runs without a context cache but uses two of its budgets: the host state-image ring for
    // its prefix-reuse checkpoints, and the host KV budget plus the session count for its
    // cross-session retention. Those three survive the disabled-cache normalization; every other
    // disabled-cache route has no user for them and drops them.
    const bool tp2_generation = is_tp2_generation(options);
    if (!cache.enabled) {
        if ((cache.device_state_slots && *cache.device_state_slots != 0) ||
            (!tp2_generation && cache.max_private_continuations &&
             *cache.max_private_continuations != concurrency) ||
            (cache.max_shared_prefixes && *cache.max_shared_prefixes != 0) ||
            (cache.max_long_anchors_per_continuation &&
             *cache.max_long_anchors_per_continuation != 0)) {
            throw std::invalid_argument("disabled context cache accepts only root-only capacities");
        }
        cache.device_state_slots = 0;
        if (!tp2_generation) {
            cache.host_state_slots       = 0;
            cache.host_kv_capacity_bytes = 0;
            cache.max_private_continuations = concurrency;
        }
        cache.max_shared_prefixes               = 0;
        cache.max_long_anchors_per_continuation = 0;
        return options;
    }

    cache.device_state_slots            = cache.device_state_slots.value_or(concurrency);
    const std::uint64_t default_private = 2ULL * concurrency;
    cache.max_private_continuations =
        cache.max_private_continuations.value_or(static_cast<std::uint32_t>(default_private));
    cache.max_shared_prefixes = cache.max_shared_prefixes.value_or(
        std::max(concurrency, static_cast<std::uint32_t>(kMaximumExplicitPromptCacheMarkers)));
    cache.max_long_anchors_per_continuation = cache.max_long_anchors_per_continuation.value_or(2U);

    if (*cache.max_private_continuations < concurrency) {
        throw std::invalid_argument(
            "context cache max_private_continuations must cover every active request");
    }
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    const std::uint64_t address_spaces =
        static_cast<std::uint64_t>(*cache.max_private_continuations) + *cache.max_shared_prefixes;
    if (address_spaces > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache address-space capacity exceeds uint32");
    }
    if (*cache.max_long_anchors_per_continuation != 0 &&
        *cache.max_private_continuations >
            std::numeric_limits<std::size_t>::max() / *cache.max_long_anchors_per_continuation) {
        throw std::overflow_error("context cache long-anchor capacity exceeds size_t");
    }
    return options;
}

ModelInstance::ModelInstance(std::unique_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen3_5::make_frontend(
          model->resources(), {.chat_template_path       = options.chat_template_path,
                               .architecture             = model->config().text.architecture,
                               .vision_enabled           = options.enable_vision,
                               .max_context              = options.max_context,
                               .media_cache_bytes        = options.media_cache_bytes,
                               .media_live_bytes         = options.media_live_bytes,
                               .media_preprocess_threads = options.media_preprocess_threads,
                               .vision_item_tokens       = options.vision_item_tokens})),
      capacity(options.max_context) {}

ModelInstance::~ModelInstance() = default;

ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto start = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));
    binding.complete();
    auto model =
        models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<ModelInstance>(std::move(model), options);
    frontend.complete();
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    auto planner    = models::qwen3_5::make_sequence_planner(instance->parameters, device, options);
    auto resolution = resolve_kv_capacity(options.kv_capacity, planner.capacity_curve(),
                                          current_free_device_bytes());
    auto sequence   = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    instance->kv_capacity_resolution = resolution;
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = models::qwen3_5::create_program(instance->parameters, std::move(sequence),
                                                        device, options.startup_observer);
    device.synchronize();
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.context_cost         = std::move(context_cost.summary);
    return {std::move(instance), std::move(summary), std::move(context_cost.model)};
}

} // namespace ninfer::runtime
