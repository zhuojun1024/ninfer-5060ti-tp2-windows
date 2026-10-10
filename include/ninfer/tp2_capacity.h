#pragma once

#include <cstdint>

#include "ninfer/types.h"

namespace ninfer {

// Concurrent requests the dedicated tensor-parallel (TP-2) generation core can run in one decode
// batch, i.e. the number of lanes its round loop can carry (see tp2-plan-concurrency.md).
//
// The core drives its own round loop on the `--devices a,b` route, so it - not the Engine - is the
// authority on how many requests that route can execute at once. Both `normalize_engine_options`
// (runtime/engine/model_instance.cpp) and the service layer's `effective_request_capacity`
// (serve/serve_options.cpp) read this one constant, because the capacity the service accepts and the
// capacity the core actually executes must not drift apart; that drift was the P6 defect the
// service/engine capacity mirror was introduced to fix.
//
// Phase 1 raises this constant as the core gains per-lane batch decode:
//   1 -> single request at a time
//   4 -> the Phase 1 target (C<=4, shared KV pool, no cross-session retention)
// The operator opts into more than one lane with --max-concurrency; the default of 1 keeps the
// historical one-request-at-a-time behaviour.
inline constexpr std::uint32_t kTp2GenerationMaxConcurrency = 4;

// Lanes the TP-2 generation core runs for `requested`, given the speculative backend.
//
// The plain, MTP and DFlash2 routes all carry `min(requested, ceiling)` lanes: P2.1c gave the DFlash2
// round a batched proposal, verify window and fold, one row per live lane, and P2.1b gave the MTP
// round the same shape. `--spec dflash` has no TP-2 context layout at all (the engine rejects it), so
// it is the only speculative backend that collapses rather than feeding every lane through row 0.
// Keeping that rule beside the ceiling means the capacity the service admits and the capacity the
// core executes cannot drift apart.
[[nodiscard]] constexpr std::uint32_t tp2_generation_concurrency(
    std::uint32_t requested, SpeculativeBackend backend) noexcept {
    if (backend == SpeculativeBackend::DFlash) { return 1U; }
    return requested < kTp2GenerationMaxConcurrency ? requested : kTp2GenerationMaxConcurrency;
}

} // namespace ninfer
