#include "runtime/engine/model_instance.h"

#include "ninfer/types.h"

#include <iostream>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::EngineOptions generation_options() {
    ninfer::EngineOptions options;
    options.purpose = ninfer::EnginePurpose::Generation;
    return options;
}

} // namespace

// The TP-2 generation route runs without a context cache, but its generation core stores
// prefix-reuse checkpoints in pinned host memory (one host StateImage per checkpoint), so that
// route keeps the host state-image budget. Any other disabled-cache route has no user for that
// memory and drops it.
int main() {
    int failures = 0;

    {
        ninfer::EngineOptions options           = generation_options();
        options.device_b                        = 1;
        options.max_concurrency                 = 6;
        options.max_pending_requests            = 7;
        options.context_cache.host_state_images  = 16;
        const ninfer::EngineOptions resolved = ninfer::runtime::normalize_engine_options(options);
        failures += check(resolved.device_b == 1, "TP-2 route lost its second device");
        // The core's batch capacity, not a hardcoded one, bounds the route; the queue depth in front
        // of it stays the operator's.
        failures += check(resolved.max_concurrency == ninfer::kTp2GenerationMaxConcurrency,
                          "TP-2 route did not clamp concurrency to the core batch capacity");
        failures += check(resolved.max_pending_requests == 7,
                          "TP-2 route dropped the configured pending-request depth");
        failures += check(!resolved.context_cache.enabled, "TP-2 route kept the context cache");
        failures += check(resolved.context_cache.host_state_images == 16,
                          "TP-2 route dropped the host state-image budget");
        failures += check(resolved.context_cache.device_state_images.value_or(1) == 0,
                          "TP-2 route kept device state-image slots");
        failures += check(resolved.context_cache.host_kv_capacity_bytes ==
                              ninfer::kDefaultHostKvCapacityBytes,
                          "TP-2 route dropped the host KV arena its retention needs");
    }
    {
        // The MTP and DFlash2 rounds both carry a batch (tp2-plan-concurrency.md P2.1b/P2.1c): their
        // TP-2 routes keep their lanes, clamped to the core's ceiling, while `--spec dflash` - the
        // masked draft with no batched TP-2 round - collapses to one lane rather than failing
        // startup.
        ninfer::EngineOptions options = generation_options();
        options.device_b              = 1;
        options.max_concurrency       = 6;
        options.speculative.backend   = ninfer::SpeculativeBackend::DFlash2;
        const ninfer::EngineOptions resolved = ninfer::runtime::normalize_engine_options(options);
        failures += check(resolved.max_concurrency == ninfer::kTp2GenerationMaxConcurrency,
                          "DFlash2 TP-2 route did not keep its lanes");
        failures += check(ninfer::tp2_generation_concurrency(2, ninfer::SpeculativeBackend::DFlash2) == 2,
                          "tp2_generation_concurrency clamped the DFlash2 route too far");
        failures += check(ninfer::tp2_generation_concurrency(6, ninfer::SpeculativeBackend::Mtp) ==
                              ninfer::kTp2GenerationMaxConcurrency,
                          "tp2_generation_concurrency clamped the MTP route too far");
        failures += check(ninfer::tp2_generation_concurrency(6, ninfer::SpeculativeBackend::DFlash) == 1,
                          "tp2_generation_concurrency kept lanes on a masked-draft route");
        failures += check(ninfer::tp2_generation_concurrency(6, ninfer::SpeculativeBackend::None) ==
                              ninfer::kTp2GenerationMaxConcurrency,
                          "tp2_generation_concurrency clamped the plain route too far");
    }
    {
        ninfer::EngineOptions options = generation_options();
        options.device_b              = 1;
        const ninfer::EngineOptions resolved = ninfer::runtime::normalize_engine_options(options);
        failures += check(resolved.context_cache.host_state_images == ninfer::kDefaultHostStateImages,
                          "TP-2 route did not keep the default host state-image budget");
    }
    {
        // CausalScoring is not the TP-2 generation route: no core there keeps host state images.
        ninfer::EngineOptions options          = generation_options();
        options.purpose                        = ninfer::EnginePurpose::CausalScoring;
        options.device_b                       = 1;
        options.context_cache.host_state_images = 16;
        const ninfer::EngineOptions resolved = ninfer::runtime::normalize_engine_options(options);
        failures += check(resolved.context_cache.host_state_images == 0,
                          "a non-TP-2 disabled-cache route kept the host state-image budget");
    }

    {
        // The overlap width snaps to the 64-token schedule boundary the split relies on; zero stays
        // disabled so an opt-out cannot be turned back on by alignment.
        ninfer::EngineOptions options = generation_options();
        options.device_b              = 1;
        options.prefill_overlap       = 200;
        failures += check(ninfer::runtime::normalize_engine_options(options).prefill_overlap == 192,
                          "TP-2 route did not align the prefill overlap width");
        options.prefill_overlap = 0;
        failures += check(ninfer::runtime::normalize_engine_options(options).prefill_overlap == 0,
                          "TP-2 route enabled a disabled prefill overlap");
    }

    {
        // `--lane-context` is the TP-2 multi-lane admission ceiling. Normalization keeps the
        // operator's number and rejects a ceiling wider than the logical context.
        ninfer::EngineOptions options = generation_options();
        options.device_b              = 1;
        options.max_context           = 131072;
        options.max_concurrency       = 4;
        options.lane_context          = 32768;
        const ninfer::EngineOptions resolved = ninfer::runtime::normalize_engine_options(options);
        failures += check(resolved.lane_context == 32768,
                          "TP-2 route dropped the configured lane context ceiling");

        options.lane_context = 262144;
        bool rejected        = false;
        try {
            (void)ninfer::runtime::normalize_engine_options(options);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "TP-2 lane_context above max_context was accepted");
    }

    if (failures != 0) {
        std::cerr << failures << " engine option checks failed\n";
        return 1;
    }
    std::cout << "engine options: ok\n";
    return 0;
}
