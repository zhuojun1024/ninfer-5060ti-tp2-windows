# NInfer Review Defect Fix Plan (2026-09-27)

> This file is the current single active fix plan and also the persistent memory that survives
> context compression. Source: one independent full-architecture review (8 subagents + item-by-item
> manual re-check), with conclusions in the previous conversation round. This file keeps only what to
> fix, how to fix it, how to verify it, and where progress stands. Items that are known and will not
> be changed are in tp2-plan.md §3.3/§3.4/§3.7 and docs/tp2-decisions.md.

## 0. Revisions after re-check (corrections relative to the review report)

- **A3 (reserve_materialization moving prompt too early) reachability downgraded to "nearly
  unreachable"**: resource_manager.h:429 already validates the plan revision with
  program.resource_revision(), and the only "non-cancellation Aborted" at program.cpp:318 compares
  the same revision once more; the revision does not change during a single worker => that Stale
  branch is in fact unreachable. Still handled as a defensive fix (the ordering contradiction is
  real).
- **A5 (P2P transfer) unreachable for the current target**: on this machine p2p_=false (blocked by a
  consumer-grade driver). The fix is to turn it into a correct implementation or an explicit
  rejection, avoiding "silently computing wrong results on machines where P2P is authorized".

## 1. Summary of fix items

| # | Item | Severity | Status |
|---|---|---|---|
| P1 | A1 TP-2 rendezvous id channel limit (MTP K>=3 inevitably exhausts it) | High | Implemented and verified |
| P2 | A2 DevicePair move constructor misses moving watch_ + move assignment leaks | High | Implemented and verified |
| P3 | A4 begin_capture/end_capture not exception-safe | Medium-high | Implemented and verified (compile + dual-GPU artifact) |
| P4 | A3 reserve_materialization moving prompt too early (defensive) | Medium (reachability low) | Implemented and verified (existing Stale assertion still passes) |
| P5 | A6+E1 sampling NaN to INT_MAX out-of-bounds write / token_counts has no length-domain validation | Medium | Safety part implemented and verified; ordering item deferred |
| P6 | B2 serving-layer concurrency capacity splits from engine normalization | Medium | Implemented and verified |
| P7 | F1 graft_components misses Uses/resource references, shared objects can be silently overwritten | Medium-high | Implemented, py_compile passes |
| P8 | A5 P2P allreduce/sendrecv correctness | Latent (in fact a deterministic logic error) | Implemented, not measurable on this machine |
| P9 | B1 worker catch(...) always escalates to permanent Engine failure | Architecture | Not changed after re-check (consistent with §7.4) |
| P10 | F2 GDN input_columns duplicated 48 times per layer (documentation/implementation mismatch) | Low | Implemented and verified |
| - | B5 four release routines consolidated; E3/E4; F3/F4; A7 guards | Low | Recorded, non-blocking |

## 2. Design of each item

### P1 - Rendezvous id channels derived from the graph bucket count (A1)
- Root cause: kArMaxChannels=16 is fixed; every WindowGraph captured for the first time occupies one
  channel; under MTP, verify_graphs_ and mtp_chain_graphs_ each build buckets per
  mtp_graph_profiles(max_context,K). Recomputing the bucket counts verbatim: K=1/2 gives 8+8=16
  (exactly used up); K=3 gives 9+9=18; K=4 gives 11+11=22; K=5 gives 12+12=24. The 17th channel throws
  logic_error, and that bucket can never be captured again afterwards.
- Plan: channel capacity becomes a member that can be reserved after construction.
  - device_pair.h: kArDefaultChannels=16 + member ar_channel_capacity_; add
    void reserve_ar_channels(std::size_t count) (must be called before any create_ar_channel;
    no-op when in-kernel transfer is unavailable).
  - device_pair.cu: capacity replaces 3 allocation sites / 1 check site; implement reserve_ar_channels
    (cudaHostAlloc count id cells anew + cudaMalloc count scalars on both GPUs, then replace the old
    block).
  - tp2_generation_core.cpp: after the graph bucket vectors are built, call
    pair_.reserve_ar_channels(verify_graphs_.size() + mtp_chain_graphs_.size() + decode_graphs_.size()).
- Verification: ninfer_tp_device_pair_test adds a case "after reserving N channels, N consecutive
  creations succeed and the N+1-th throws"; compile ninfer-serve.
- Expected: K>=3 no longer hits the ceiling; K=2 has margin.

### P2 - DevicePair move semantics (A2)
- Root cause: the move constructor initialization list omits watch_ and does not clear other.watch_
  (target watch_ empty -> diagnostics broken; the source still holds the thread while its mapped
  pointer now belongs to the target -> UAF read when the target is destroyed first); move assignment
  overwrites pointers before releasing the target's existing pinned/device buffers (leak).
- Plan: the move constructor moves watch_ and nulls other.watch_; move assignment calls
  stop_ar_watchdog() and releases host_a_/host_b_/arrival_host_*/stall_host_/base_* (released against
  the current device binding) before overwriting, then takes over other and clears other.
- Verification: ninfer_tp_device_pair_test (including the move case) passes.

### P3 - Capture RAII (A4)
- Root cause: three bare calls, pair_.begin_capture(); capture_group(...); pair_.end_capture(); when
  the capture body throws, capturing_ stays true forever -> every later capture fails permanently,
  and later eager sets take the "already captured" branch.
- Plan: tp2_generation_core.cpp gains a local RAII helper (ArCaptureGuard, whose destructor calls
  end_capture), and the three sites call capture_group inside the guard scope. end_capture is a no-op
  for the already-closed state, preserving idempotence.
- Verification: real dual-GPU artifact append/solo cases; a targeted case that throws during capture.

### P4 - reserve_materialization does not consume prompt (A3, defensive)
- Root cause: engine_core.h passes std::move(request->prompt); as soon as Program enters
  start_resource_transaction it moves the prompt away, and the subsequent revision gate fails and
  returns Aborted/stale, leaving the request pending with an empty prompt.
- Plan (implemented): delay the "move" until after Program's revision gate.
  - Program::start_resource_transaction(ResourcePlan&&, PreparedPrompt&, CancellationFlagView):
    on gate failure it returns directly, without touching prompt; only after passing does it call
    PreparedPromptAccess::take(std::move(prompt)).
  - resource_manager keeps its PreparedPrompt&& parameter (the caller authorizes consumption) but
    forwards it to Program as an lvalue, so when Stale is returned the caller's prompt is still
    intact. The engine call site keeps std::move(request->prompt).
  - This fixes the defect while avoiding changes to the 20-odd test call sites (temporaries can still
    bind to &&).
- Verification: compile ninfer_resource_manager_test / ninfer-serve; the existing Stale assertions
  (2418/2434) must still pass.

### P5 - Sampling NaN safety + token_counts contract (A6/E1)
- Root cause: the single-block greedy path starts at bi=INT_MAX, so an all-NaN row outputs INT_MAX and
  does an out-of-bounds atomicAdd(token_counts[INT_MAX]); speculative greedy acceptance likewise uses
  INT_MAX as terminal/id and writes token_counts.
- Implemented: 1) the greedy initial values bi/best_index = 0 in sampling.cuh and
  speculative_round.cuh (0 is a valid id), so all-NaN rows no longer produce INT_MAX. 2) the
  atomicAdd in sampling.cuh is preceded by a [0,token_domain) check.
- Deferred (needs independent oracle acceptance): aligning the NaN ordering of ordered_score_bits /
  sampling_bf16_tile_sort_key and the "ignore NaN" behavior of sampling_better. The tile key's 0 is
  the empty-slot sentinel, and NaN->lowest would conflict with it, so it must be accepted numerically
  together with linear_topk/candidate selection and cannot be changed casually. Recorded; not changed
  this round.
- Verification: ninfer_sampling_test, ninfer_speculative_round_test compile; the targeted all-NaN
  case is under "to be added".

### P6 - Serving-layer capacity takes the normalized engine capacity (B2)
- Root cause: both GenerationService and HttpServer build RequestCapacity/thread pool from
  ServeOptions' max_concurrency+max_pending_requests; the TP-2 route engine normalizes to 1+1, so
  requests admitted by the serving layer are rejected by the engine as overloaded (instead of waiting
  in the FIFO), and the thread pool is also sized by the wrong capacity.
- Plan (implemented): serve_options.h/.cpp adds effective_request_capacity(const ServeOptions&),
  aligned with normalize_engine_options (device_b>=0 -> 1+1); generation_service.cpp and
  http_server.cpp both use it, so the two numbers come from one source.
- Verification: compile ninfer-serve; ninfer_serve_options_test.

### P7 - graft_components misses Uses/resource references (F1)
- Root cause: Entry.binding_objects only scans top-level bindings; uses[].auxiliaries and
  components[].resources likewise reference objects, yet are absent from users. plan_graft's
  "shared object is safe" check therefore misses them: an object referenced by a non-grafted component
  through auxiliary/resource can still be overwritten, violating the promise made in the module
  documentation.
- Plan (implemented): Entry adds object_components (object id -> the set of component names
  referencing it), covering the three reference sites (binding names, Use.parameter, component
  names); plan_graft's foreign check uses it.
- Verification: py_compile (no 3.11 on this machine, so a 3.13 syntax check was used, noted) + the
  existing conversion tests.

### P8 - P2P correctness (A5)
- Re-check conclusion (more severe than the report): P2P allreduce is not merely a race but a **logic
  error**: the two peer copies write into each other in place, since copy1 overwrites data_a with
  data_b while copy2 still has to read the original data_a, so without a temporary buffer no ordering
  can possibly be correct. The P2P branch of sendrecv, for "send_a and recv_a are aliases of each
  other" (explicitly allowed by the header), likewise makes the other side's copy read an overwritten
  source.
- Plan (implemented): delete the two P2P fast paths; P2P hosts instead take the mapped-pinned
  in-kernel transport (change the transport creation condition from !p2p_ && !force_host_staging_ to
  !force_host_staging_, so that P2P hosts are not degraded wholesale to host staging that
  synchronizes on every collective), and payloads beyond the staging threshold go through
  host-staging. p2p_ is retained as capability reporting (p2p_available()); transport no longer
  issues peer copies.
- Verification: there is no P2P on this machine, so the deleted code paths cannot be measured
  locally; capability reporting and the existing move/destructor assertions are retained.

### P9 - Narrowed Engine failure escalation (B1)
- Current state: worker_loop's catch(...) always calls fail_all_locked, and failed_ is never reset.
- Re-check conclusion (carried out): judged to be by design; not changed this round. The request-level
  rejection in engine-architecture §7.3 only happens before Program mutation; exceptions after
  mutation must fail the whole engine per §7.4. worker catch(...) is consistent with the
  documentation; changing it would be a product-level design change and is outside this fix's scope.

### P10 - GDN input_columns duplicated per layer (F2)
- Root cause: recipe.py::prepare creates one object for each auxiliary of each Use. In the GGUF recipe
  the 48 GDN layers each have one gdn/output Use (1 input per layer), so the same 6144xint32
  (24,576 B) input_columns layout is stored 48 times (about 1.125 MiB), contradicting
  docs/maintainer/artifact-container.md §8 "the same vector may be referenced by any number of
  purposes, and the loader binds it only once". The C++ loader (prepare.cpp::Bindings::input_columns)
  already deduplicates by (width, parts), so only the recipe needs fixing.
- Plan (implemented): Recipe.prepare interns auxiliary objects by (format, layout, shape, data),
  allocating a single TensorSpec for identical bytes, with multiple Uses referencing the same object
  id. The numbering/order at the first occurrence is unchanged.
- Verification:
  1) tests/convert/test_recipe.py adds test_identical_auxiliaries_are_stored_once
     (3 Uses share one input_columns -> prepared.auxiliaries is exactly 1, and all three Uses
     reference it).
  2) Real artifact counts (reading the directory JSON):
     - Qwen3.8-27B-GSQ-RCO-IQ3_S-dflash2q4.ninfer: 48 Uses -> 48 objects, same content hash
       1c3c6784c6d0, 24576 B each, 1.125 MiB in total (the pre-fix shape).
     - Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer: 48 Uses -> 1 object (0.023 MiB), proving that the
       "one object referenced by 48 Uses" representation is already accepted by real artifacts and by
       the loader.
     The two reports' objects counts differ by 1296-1249=47=48-1, consistent with that deduplication.

## 3. Verification and commit strategy

- C++ changes: compile the affected targets + the relevant ctest suites (on Windows use
  tools/win_port/test.ps1).
- One commit per item (Conventional Commits, lowercase type); P1/P3/P4 may be merged into a single
  TP-2 fix commit.
- Python changes: python -m py_compile + existing tests.

## 4. Progress

- [x] P1 rendezvous channel reservation (implemented; device_pair.cu/h, tp2_generation_core.cpp, new test)
- [x] P2 DevicePair move semantics (implemented; move ctor/assign + watch_ rebind and restart)
- [x] P3 capture RAII (implemented; ArCaptureGuard x3)
- [x] P4 reserve_materialization prompt (implemented; move delayed until after the Program revision gate)
- [x] P5 sampling NaN / token_counts (safety part implemented; the ordering alignment part is deferred, see the P5 section)
- [x] P6 serving-layer capacity (implemented; effective_request_capacity)
- [x] P7 graft Uses/resource references (implemented; object_components)
- [x] P8 P2P (implemented: removed the incorrect peer-copy fast paths, P2P hosts now use in-kernel/host staging)
- [x] P9 Engine failure escalation (after re-check judged to be by design, not changed this round: 7.3 request-level rejection only happens before Program mutation,
  exceptions after mutation must fail the whole engine per 7.4. worker catch(...) is consistent with the documentation, not changed this round; a product-level design change is required)
- [x] P10 GDN input_columns deduplication (implemented; Recipe.prepare interns auxiliaries by bytes)

### Verification record (2026-09-27, Windows build-win, ctest)

Build: `cmake --build build-win --target ... -j 12` all green (including ninfer-serve).
Tests: `tools/win_port/test.ps1 -Filter 'sampling|speculative_round|serve_options|engine_options|resource_manager|tp_device_pair'`.

Passed:
- ninfer_tp_device_pair_test (34.3 s, PASS). The new cases take effect:
  `reserve channels: overflow refused as expected: ... reserved rendezvous id channels`,
  and `[mem] TP-2 rendezvous id channels reserved: 24`. Existing move / each rung / host-staging cases all pass.
- ninfer_sampling_test (PASS). The newly added `nan_greedy_contract` (all-NaN row -> token 0, counts[0]++,
  an out-of-bounds write would be caught by the GuardedDeviceBuffer guard page) passes.
- ninfer_speculative_round_test (PASS).
- ninfer_sampling_defaults_test (PASS).

Pre-existing failures unrelated to this round's changes (all outside the files changed here, confirmed one by one):
- ninfer_resource_manager_test: "candidate-stratified reuse closure" (MaterializationPlanner ordering).
  Proven with a baseline: after stashing this round's program.h/.cpp + resource_manager.h + that test file and rebuilding, it still fails with the same assertion.
- ninfer_engine_options_test: "TP-2 route kept a host KV arena" (normalize_engine_options in
  model_instance.cpp keeping host_kv_capacity; that file was not changed this round).
- ninfer_serve_options_test: "server reasoning-effort default was not resolved"
  (resolve_semantics in serve/translate.h; not changed this round). The newly added effective_request_capacity assertion passes
  (the failure output contains none of this round's message strings).

Python: `tests/convert` + `tests/artifact` 54 passed in total (including the P7 py_compile and the new P10 case);
`py_compile tools/convert/graft_components.py`, `py_compile tools/convert/recipe.py` pass.
This machine has no Python 3.11 (only 3.13.13 on PATH, and conda has no envs), so 3.13 was used instead.

Incomplete / not locally verifiable:
- The NaN ordering alignment part of P5 (deferred, needs an independent oracle).
- The P2P paths of P8: on this machine p2p_available=0 (blocked by a consumer-grade driver), so the
  deleted peer-copy paths cannot be measured locally.
