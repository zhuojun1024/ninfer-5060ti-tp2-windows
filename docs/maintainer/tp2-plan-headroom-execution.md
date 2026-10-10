# TP-2 headroom optimization execution plan

Basis: the priority table in §6 of tp2-concurrency-headroom.md. Implement item by item, verify item by item, record item by item.

## Execution order and status

| No. | Item | Report priority | Status |
|----|-----|-----------|------|
| 1 | plain batch round timing instrumentation | P1 | Code complete, build+test pass |
| 2 | plain route does not carve the round-scratch state plane | P1 | Code complete, build+test pass |
| 3 | Widen the batch formation window | P0 | Code complete, build+test pass |
| 4 | Interleave a mid-arrival prefill with the decode of live lanes | P0 | Code complete, build+test pass, measured (see below) |
| 5 | Consume the queue during the initial prefill | P0 | **Reverted** after merging with 4: proven a no-op (rationale below) |
| 6 | --lane-context multi-lane default/warning | P1 | Code complete (warning, no hard default), build+test pass |
| 7 | Count queueing time in the TTFT/total accounting | P1 | Code complete, build+test pass |
| 8 | Fix the 262,144/C=1 guidance in docs/serving.md | P2 | Done (1/2/3/4/6/7 are explained in docs/serving.md) |
| 9 | Dynamically shrink the output budget by the remainder / reclaim pages | P2 | **Not changed** after analysis (breaks the all-or-nothing invariant); conclusion goes into the report |
| 10 | Speculative batch chain switched to a per-lane envelope | P2 | **Implemented**: removed the batch's truncation of the host split count + added a batch/solo bitwise parity regression; A/B shows plain is neutral and the acceptance-rate gain did not reproduce (within noise). See "Item C implementation and verification" |

### User's second-round additions (m01111)

| No. | Item | Status |
|----|-----|------|
| A | spec/MTP route prefill interleaving (`pump_spec_round`) | Code complete, build passes, A/B measurement holds (see below) |
| B | spec route `round_ms`/`rest` timing accounting (the measurement premise of A) | Code complete, build passes |
| C | Item 10 Ops kernel split by column (remove the batch's truncation of the host split count) | Code complete + new batch/solo bitwise parity regression test; ops test passes (the old ops binary FAILs that test); A/B measurement complete — plain decode neutral, acceptance-rate gain did not reproduce (see below) |
| D | P3 batched prefill (incidentally eliminating the admission position gap) | Design complete (_temp/20261002-1800_item3_batched_prefill.md) + **measured width cost** (batched window 1.24 vs single-lane 0.587 ms/lane-token, **2.1× slower**) ⇒ **confirmed not to do**, not implemented (see "Item D additional measurement" below) |
| E | Pageable host checkpoint ring (report P3 row) | Code complete (PinnedHostBuffer → HostBuffer(Pageable)), build passes; model tests (lanes all pass / sessions byte-identical to the old baseline); null-pointer regression fixed; latency measured — the two cards save 5.29 GiB of page lock, recall requests +10 ms (within noise) |

### Execution order rationale

6 and 7 were moved ahead of 4/5: 7 is the measurement accounting for the effect of 4/5 (a TTFT without queueing time cannot prove interleaving is effective), and 6 is independent and tiny. This is consistent with the existing practice of "instrument first".

## Verification method (repository verification table)

- C++ changes: build the affected targets (tools/win_port/build.ps1) + the relevant filtered tests of tools/win_port/test.ps1.
- Performance/latency: measured at the claimed scope (this machine, 2×RTX 5060 Ti, serving started with --devices 0,1).
- Documentation changes: links + `git diff --check`.

## Key implementation notes (source locations)

- Driver thread and formation window: src/runtime/engine/tp2_generation_core.cpp:2277-2391,
  kBatchFormationWindow :2250, kBatchFormationMaximum newly added at :2278.
- plain route: execute_plain_batch :3174, pump_decode_round lambda :3405,
  initial admission (S1 loop) :3870-3887, main loop :3890-4030, mid-way admission :3903-3930.
- admit_lane (plain) :3554-3870; spec counterpart :4195-4514.
- State plane carving :851-863; state_snapshots[kRoundScratchSlot] is used only by spec/walk.
- lane-context :443-478 (newly added warning branch).
- TTFT accounting: submit() :2108; two admit_lane sites set queue_wait; two finalize sites sum it;
  execute_walk :7151/:7156/:8021.

## Confirmed physical constraints (these determine which items are worth doing)

- prefill is limited by PCIe (card 2 Gen4 x4, ~7 GB/s): 0.368-0.375 ms/token, a 2.67k tok/s ceiling.
- decode is limited by the DRAM weight stream: about 317 GB/s at C=4.
- Therefore prefill and decode contend for **different** resources, and parallelism is theoretically feasible; but the shared workspace arena and the cross-card in-kernel allreduce rendezvous are software obstacles.
- The serial total work of 4 prefill runs of a 10.8k prompt = 22 s, and no scheduling can be faster; interleaving only changes the latency distribution (each lane's TTFT upper bound drops from "the whole prefill" to "one chunk").

## Item 4/5 design (implemented, plain route)

**Scheduling policy**: at any moment **exactly one** lane is prefilling (preserving the existing serial FIFO order and each lane's current TTFT); a lane that already holds tokens runs **one decode round** between each of that lane's two chunks. Round-robin (one chunk per unfinished lane per turn) was rejected: it would make all lanes finish prefill at the same aggregate wall-clock moment, degrading lane 0's TTFT from ~5.5 s to ~22 s.

The report's "short lane decode 37.7 → 2.5 tok/s = 15x" is a **latency** artifact (tokens/s amortized over the whole blocking prefill), not a per-round cost; chunk-level interleaving fixes exactly this. Total GPU work is unchanged.

**Implementation: a pump callback, not a split state machine**. Lift the decode round body into `auto pump_decode_round = [&]() { ... };` inside `execute_plain_batch` (defined after `active.reserve(lane_capacity);` and before `admit_lane`, so admit_lane can call it directly, with no std::function and no forward declaration). The round body in the main loop is changed to call it;
at the top of admit_lane's chunk loop, after the previous chunk's `scope_a/scope_b` have been destroyed, add `if (!active.empty()) { pump_decode_round(); <restore slot binding> }`.
`admit_lane` remains synchronous and the serial order is unchanged.

**Key invariants**:
- A decode round must start exactly from `shard_a_.round_base`/`shard_b_.round_base` (the captured CUDA Graph has `arena_begin[0/1]` baked in, and `run_plain_decode_step_batch` throws `"TP-2 batched decode CUDA Graph replay found a different workspace layout"` when `workspace->used() != graph->arena_begin[0]`).
  So round_base's capture point is **moved earlier** from "after the reserve loop" to "after the pump lambda definition".
- admit_lane's net ws_a allocation is 0 (mid-way admission today already runs before `position_arena(ws_a, round_base, round_base)`, so any persistent allocation would be rolled back anyway).
- One round re-broadcasts the scalar GDN slot pair (src/models/qwen3_5/execution/text.cpp:2370-2378, the `batch == 1` branch), so after pump returns, restore `set_linear_state_slots(active_lane_, active_lane_)`.
- The single-instance vision session premise **is not broken**: only media-free/media-bearing lanes are serial each, and at any moment there is still exactly one prefill running.

**Scope**: plain route only. The spec route (execute_spec_batch) keeps per-lane serial prefill and is recorded in the report as remaining headroom.

## Item 7 change points (implemented)

- `TP2GenerationCore::Request` (src/runtime/engine/tp2_generation_core.h:102) adds `std::chrono::steady_clock::time_point submitted{};`.
- `submit()` (.cpp:2108) sets `request->submitted = Clock::now();`.
- The two admit_lane sites set `lane.result.engine_timing.queue_wait_seconds = duration<double>(lane.begin - pending.request->submitted).count();`
- The two finalize sites + execute_walk: `first_token_seconds`/`total_seconds` become `prepare_seconds + … + queue_wait` (aligned with the single-card accounting); `prompt_wall_seconds` and `prefill_seconds` **stay** counted from admit (they define the prefill tok/s rate).
  This is mandatory: src/serve/generation_service.cpp:447-457 does `ttft = prepare + max(0, first_token - prepare)` and `total = prepare + max(0, total - prepare)`, i.e. timings.total_seconds **must** include prepare.
- No downstream changes needed: `GenerationEngineTiming::queue_wait_seconds` (src/runtime/engine/request_record.h:70-73) and `| queue <dur>` (src/serve/operational_log.cpp:262-265) already exist.

## Item 5 conclusion (implemented then reverted, do not add it back)

Adding `try_pop_lane_queue()` at the end of the S1 initial admission loop compiles, but analysis proves it is a no-op: the S1 loop admits in `batch.size()` order, and right after the loop finishes the main loop **immediately** runs `while (active.size() < lane_capacity) { try_pop_lane_queue(); ...; admit_lane(...) }`, prefilling the new members at the same wall-clock moment and just as strictly serially, with no TTFT difference — the only difference is which `batch` vector the members sit in. Item 5's premise in the report (the queue is not polled during the initial prefill) is in fact already covered by the existing mid-way admission; the 2+2 split was fixed by item 3, and the residual "a newcomer waits for the current prefilling lane" is an inherent property of single-lane serial prefill that requires P3 batched prefill. Reverted; the file is back to 8082 lines.

## Item 4 measurement results (_temp/20261002-1442_pump.ps1, plain, int8, 245760, C=4)

prompt: long = longSys(160) = 15,310 tokens, short = 54 tokens, max_output=24, reasoning=low.

| Scenario | Short-lane total | Short-lane decode | Mechanism |
|------|---------------|----------------|------|
| Report baseline (no interleaving, B3) | 6.0 s | 2.5 tok/s | The whole prefill blocks |
| Interleaved, chunk 1024 | 7.5 s | 3.1 tok/s | 1 decode round per chunk (~0.55 s) |
| Interleaved, chunk 256 | **2.8 s** | **8.5 tok/s** | 1 round per chunk (~0.19 s) |

- The reverse order (long first, short second) still waits: the short lane queues 9.8-9.9 s, TTFT 9.9 s vs 114 ms when short comes first.
  This is the inherent cost of "only one lane prefills at any moment"; item 4 does not change the admission order.
- The new instrumentation directly proves the mechanism (chunk 256, NINFER_TP2_TIMING=1):
  the batch of the long lane when short comes first prints `prefill_chunks=61 pump_rounds=14` — the short lane gets its 24 tokens and then leaves active, and pump stops accordingly; long first, short second prints `prefill_chunks=61 pump_rounds=1`; four long concurrent prints `prefill_chunks=180 pump_rounds=46`.
- Item 3 takes effect: four long concurrent goes from the report's `batch=2` to `batch=4` (chunk 1024) / `batch=3` (chunk 256, the 50 ms cap still splits the 4th into a second wave); the 2+2 with an 800 ms gap correctly does not merge.
- chunk 256's per-lane prefill rate drops to 1.34-1.52k tok/s (1.69-1.94k at 1024), consistent with the report's cost model: a small chunk trades prefill rate for decode smoothness.
- The client SSE under reasoning + 24 tokens is mostly 0 text deltas and cannot serve as a token clock; the authoritative data are the `done |` and `[tp2-time]` lines.

## Item 9 conclusion (not changed, written into the report)

`reserve_lane_kv` (:3027) makes a **full all-or-nothing** reservation across both cards + the MTP cache before admission; the design comment :3004-3010 says outright "if one pool cannot hold it now, nothing is retained, the caller puts the request back at the head of the queue … the wait is always for another lane to retire, never a deadlock". Shrinking by the remainder or reclaiming pages would make a lane need to **grow** mid-decode, and that route has no mid-request backpressure/preemption (AGENTS.md: no active-request preemption); a safe implementation would require adding an "evict on demand and grow a contiguous KV window" path, which is not the scale of a P2 item. The measured 66 MiB/card idle is the price of the "guaranteed to complete" promise, and its upper bound is set by the client's `max_output_tokens`; the operational lever is `--lane-context` (item 6) or a smaller client budget. The report records it as "analyzed, not changed, with rationale".

## Item 10 attribution measurement (config: 131072 + int8 + --spec mtp --draft-tokens 5)

| Group | Acceptance rate |
|----|--------|
| Single long (15,310) solo C=1 | 46/70 = **65.7%** (4.0 tokens/lane-round) |
| Four long in the same batch C=4 | 45.7 / 54.2 / 34.7 / 38.0% (mean 43.2%, 3.0 tokens/lane-round) |
| Long + short C=2 | long 41.1%, short (54 tokens) **36.2%** |
| Single short solo | 45.3% |
| Two shorts in the same batch / four shorts in the same batch | 54.3 / 55.7% and 53.3 / 55.3 / 43.2 / 42.4% |

Conclusion: the regression is real (wide-window batching 65.7% → 34.7-54.2%, tokens/lane-round −25%), but the discriminating experiment proves "batching itself" is not to blame (a short window in the same batch is no different from solo) ⇒ the mechanism is the **in-batch split policy of the wide window**: `mtp_forward_decode_batch`'s envelope has only one width (tp2_generation_core.cpp:4815-4817 comment: "The envelope steers split policy only … the widest lane bounds every lane's window"). The fix surface is in the Ops kernel (per-column envelope / split policy by column), not in the engine; verify is a captured graph, so changing the signature requires recapture and FP32/FP64-oracle verification of per-token consistency. **Not done this time**, recorded as remaining headroom in the report.
Open question: whether the trigger is "the absolute window is wide" or "the inter-lane position gap" — the short-window experiment can only rule out batching itself, not the former.

## Item A measurement results (spec/MTP route interleaving, _temp/20261002-1745_specpump.ps1)

Config: `--devices 0,1 --max-context 131072 --kv-dtype int8 --max-concurrency 4 --spec mtp --draft-tokens 5 --host-state-slots 32 --host-kv-mib 20480`; long = longSys(160) = 15,310 tokens, short = 54 tokens, max_output=64, reasoning=low. ARM C = short sent first, long +400 ms; ARM B = long sent first, short +400 ms.

| Config | HEAD ARM C short lane | Interleaved ARM C short lane | HEAD ARM C long lane | Interleaved ARM C long lane |
|------|--------------------|--------------------|--------------------|--------------------|
| chunk 1024 | total 8.8 s / decode 3.1 tok/s | **total 3.3 s / 8.9 tok/s** | 9.0 s | 9.5 s |
| chunk 256 | total 11.1 s / decode 2.5 tok/s | **total 2.0 s / 15.7 tok/s** | 11.4 s | 12.1 s |

- ARM B (the later arrival) is unchanged on both sides: short lane queues 7.6 s (chunk 1024) / 10.0 s (chunk 256).
  This is the inherent cost of "only one lane prefills at any moment"; interleaving does not change the admission order.
- The long lane is instead +0.5-0.7 s (pump overhead ~5%); total GPU work is unchanged.
- Mechanism evidence: `[tp2-time] spec-batch … prefill_chunks=16 pump_rounds=6` (the interleaving short lane was pumped for 6 decode rounds across the long lane's 16 chunks).
- Implementation differences from plain: before defining `pump_spec_round`, must **move up** the `shard_[ab]_.round_base` capture, `round` (dflash_round), `dflash_execution`, `lane_extent`, `append_slots/starts/ends`, `mtp_pack`, `mtp_step_host`, `mtp_selectors_h` (admit_lane has zero references to these); after calling it at the top of the chunk loop, must restore `set_linear_state_slots(active_lane_, active_lane_)` (the spec chunk forward does not set the slot itself, relying on the binding at the admit entry; one round re-sends the scalar slot through text.cpp's `batch == 1` path).

## Item B conclusion (spec timing accounting)

The spec route never accumulates `round_ms`, and `[tp2-time] spec-batch`'s `avg_round` is always 0.00, so it cannot serve as per-round cost evidence for item A. Fix:

- Delete the dead member `Tp2RoundTiming::fold_ms` and the 3 sites `timing.fold_ms += timing.elapsed(5, 6);` (fold runs after the synchronization, and reading an unfinished event pair inside a round with `elapsed()` always returns 0.00); `report()`'s field changes from `fold=` to `rest=`, value = `(round_ms - mtp - verify - accept - copy) / n`.
- `pump_spec_round` adds a round-level `round_start` and `timing.round_ms +=` (same accounting as the plain pump).

## Item E design (pageable host checkpoint ring)

Motivation in tp2-concurrency-headroom.md:134-151 §2.5: at C=4/262,144 the ring = 36 slots × 73.4 MiB = 2,642.6 MiB/card (5.1-8.2 GiB of page lock across two cards), and it is the largest block of pinned memory outside the host KV.

Changes (reuse `HostBuffer(HostPinning::Pageable)`, keep no dual path):

- src/runtime/engine/tp2_generation_core.h:195/:200: `HostCheckpoint::buffer`/`dflash_buffer` change from `unique_ptr<PinnedHostBuffer>` to `unique_ptr<HostBuffer>`.
- src/runtime/engine/tp2_generation_core.cpp:929/:1117: change to `std::make_unique<HostBuffer>(bytes, HostPinning::Pageable)`; add `(pageable)` to the `[mem]` line.
- The image transfer path goes back to **raw mapping + length** (consistent with the existing `copy_lane_state` convention): `store_dflash_image(Shard&, void*, std::size_t, lane)`, `load_dflash_image(Shard&, const void*, std::size_t, lane)`, `session_capture_shared_state(..., const void* const* frozen, const void* dflash_frozen, lane)`, with the corresponding `frozen[2]`/`frozen_draft`/`block_draft` locals changed to `const void*` and call sites passing `->data()`/`->size()`. Rationale: ring slots (pageable) and the session slab (pinned) share the same copy path, and PinnedHostBuffer and HostBuffer have no common base class.
- Correctness: the ring is not in any CUDA Graph capture region; all access goes only through `->data()/->size()`; `copy_lane_state` uses cudaMemcpy2DAsync (pageable is legal, at the cost of driver-internal staging + host-thread blocking, i.e. store/recall get slower, not wrong). No test asserts pinning/slot count.
- Verification accounting: build + `tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` + on-machine (ring 35/36 slots at `--host-state-slots 32`) comparison of prefill tok/s and store/recall time.
- Documentation sync: docs/serving.md, docs/tp2-dual-5060ti.md, tp2-concurrency-headroom.md:134-151/:383/:558; tp2-plan-concurrency.md:342's "lanes=4 ⇒ 9.98 GiB/shard" is an outdated draft.

## Item E implementation and verification

Landed as designed above (no dual path): h:195/:200 changed to `unique_ptr<HostBuffer>`; cpp:929/:1117 changed to `make_unique<HostBuffer>(bytes, HostPinning::Pageable)`; the image parameters of `store_dflash_image`/`load_dflash_image`/`session_capture_shared_state` changed to "raw mapping + length", `frozen[2]`/`frozen_draft`/`block_draft` changed to `const void*`; `(pageable)` added to the `[mem]` line. Build BUILD_EXIT=0 (26-28 s, multiple times).

- Model tests (`NINFER_TEST_ARTIFACT=<models-dir>/qwen3_8_27b_swift15_dflash2_final.ninfer`):
  `ninfer_qwen3_5_tp2_lanes_test.exe` all pass (LANES_EXIT=0; mid-batch arrival 0.729842 s / 3.89515 s, against the existing baseline 0.59 s / 3.67 s, same order of magnitude); `..._sessions_test.exe` still fails, but the failing lines are **byte-identical** to the old pinned baseline (_temp/cnt1_sessions_full.log), and Compare-Object over the `^TP-2 ` lines of the two logs shows no difference ⇒ no new failures (that test stops at the first FAIL, and so does the old log).
- **Full model test accounting + HEAD baseline comparison (final binary)**:
  `$env:NINFER_TEST_ARTIFACT='<models-dir>/qwen3_8_27b_swift15_dflash2_final.ninfer'; & tools/win_port/test.ps1 -Filter 'qwen3_5_tp2'`
  → 67% passed out of 6 (load/forward Skipped, sessions Failed, dflash_solo Failed, dflash_append Passed 47.51 s, lanes Passed 69.56 s). After stashing all changes and rebuilding HEAD (464 s), the same filter was re-run and **both failures reproduce byte-for-byte**: sessions' `got [2752 13 198 197 197 92 198 197] expected [467 419 538 13 198 197 197 92]`, dflash_solo's `the recalled walk diverged from the from-scratch walk on its first sample: [59399 475 327 363 62 16 15 15] vs [365 4577 62 16 15 15 15 15]`
  ⇒ both are **pre-existing failures**, unrelated to this round's changes. Evidence _temp/20261002-1745_modelverify_new.md.
- Runtime evidence: `[mem] host checkpoint ring 36 slots x 73.4 MiB/lane/shard (pageable)`, `[mem] host-checkpoints shard 0/1 slots 36 (...) x 73.4 MiB | stride 43776 tok | pageable 2642.6 MiB` (against the old line `... | pinned 734.1 MiB`, 11 slots).
- **Null-pointer regression (fixed)**: when changing the `unique_ptr` handle to "raw pointer + length", `checkpoint.dflash_buffer->data()` became an unconditional dereference on the MTP route where `dflash_buffer == nullptr` (`shard.dflash_round == nullptr`, and dflash_buffer is allocated only inside the `if (dflash_round != nullptr)` loop) ⇒ 0xC0000005 access violation at the wrap-up of a long prompt prefill (deterministic, both reproductions had a wall of 9.9 s). Fixed to take the value after a null check (cpp:2773/:5923), and made the null check of `checkpoint.buffer` explicit (:2768). Lesson: for every `unique_ptr<T>` → raw pointer conversion, each `.get()` must become a null-checked `->data()`.
- **Latency measurement (pinned vs pageable, same harness same generation, differing only in the ring's pinning)**:
  `_temp/20261002-1700_ringio.ps1 -Tag pinned2|pageable2` + `_ringio.mjs` (a 13,559-token prefix + 4 recall requests at 98.2% hit rate, `--max-context 131072 --host-state-slots 32`). The `[mem]` line prints according to the actual pinning (`pinned 2642.6 MiB` / `pageable 2642.6 MiB`). Recall requests (req#2..#5) mean total: pinned 495.5 ms, pageable 505.3 ms (+10 ms, 2%); mean TTFT pinned 296.5 ms, pageable 292.8 ms.
  ⇒ The cost falls within run-to-run noise. The two arms' `[tp2-kv]` store/recall timings are identical (store 125.7-151.4 ms / recall 67.9-80.1 ms) — that measures the paged KV (HostKVArena, pageable anyway), not the ring image, and can serve as corroboration.
- The HostKVArena controlled by `--host-kv-pinned` can still be pinned (that ternary expression was not touched by this item).

## Item C implementation and verification (removing the batch's truncation of the host split count)

Mechanism (design investigation subagent c0f16d5e, report _temp/20261002-1730_item10_ops_envelope.md): on the device side `active_split_count = min(per-column natural split count, value passed from host)` (src/ops/softmax_attention/dense/causal_cache/small_t.cuh:119), while the host-passed value `causal_attention_split_capacity` is truncated by `grid_limit`/`page_limit` when `batch_size > 1` (src/ops/softmax_attention/dense/causal_cache/small_t.cu:229-246). Hence every column in the batch gets a shallower reduction tree than that lane running solo, and the numerics differ bitwise; the wider the window and the larger the batch, the harsher the truncation (Int8/Scale=1/K=5: B=4 cap=10, B=2 cap=20).

Change: `causal_attention_split_capacity` drops the `batch_size` parameter (launch.h:26-33), and the q_heads==24 branch directly `return capacity;` (capacity is itself already the envelope upper bound; the device side still clamps by the per-column natural count, and extra splits on short columns return immediately), with the 11 call sites updated to 4 parameters in sync. The H12/H16 branches already return capacity directly and are unaffected.

Regression test (new): `run_batch_single_parity_case` in tests/ops/softmax_attention/causal_cache.cpp — batch=4, width=4, visible=4104 (>4096 forces the old BF16 code to truncate), 5 storages, comparing "the b-th column in the batch" bitwise against "that lane run solo". Criterion: bitwise equal after the fix.

| Binary | `ninfer_softmax_attention_test` |
|--------|----------------------------------|
| New ops (after fix) | `PASS causal_softmax_attention public-contract correctness`, OPS_EXIT=0; no tolerance-oracle regression (nvfp4 mae=0.00537 / k8v4 mae=0.00537) |
| Old ops (stash revert) | OPS_EXIT=1, `FAIL causal_softmax_attention public-contract correctness`, 1-3 ULP differences across all 5 storages × 4 lanes (e.g. bf16 lane=0 element 4 batched=0x3a20 solo=0x3a1f) |

**A/B measurement (same harness same generation, sole difference = the 6 files under src/ops)**:

| Config | Old ops | New ops |
|------|--------|--------|
| plain decode C=1/C=2/C=3/C=4 (per-lane tok/s) | 36.8 / 35.0·35.9 / 31.9·32.7·33.6 / 29.9·30.6·31.3·32.2 | 37.2 / 35.0·35.8 / 32.0·32.8·33.7 / 29.9·30.6·31.4·32.3 |
| plain `[tp2-time] avg_round` C=1..4 | 27.15 / 13.89 / 9.97 / 7.84 ms | 26.85 / 13.92 / 9.92 / 7.85 ms |
| spec short-arm P4 per-lane round cost mtp+verify | 21.05 ms | 23.57 ms |
| spec long-arm four-long per-lane round cost mtp+verify | 51.65 ms | 50.11 ms |
| spec acceptance short-arm P4 mean | 50.1% | 51.4% |
| spec acceptance long-arm four-long mean | 49.6% | 40.3% |
| spec acceptance **single long (control, no code path change)** | **53.3%** | **41.0%** |

⇒ plain decode is completely neutral; the acceptance-rate differences are entirely within run-to-run noise (the solo control itself drifts by 12 points), and an acceptance-rate gain **cannot be proven**. Conclusion: this item is kept; its value is **numerical consistency** (guarded by the parity test), not a throughput optimization. The wording of report §8.7 and §6 has been rewritten accordingly.

## Item D conclusion (P3 batched prefill, not implemented)

Design report _temp/20261002-1800_item3_batched_prefill.md (161 lines). The only reusable entry point is src/models/qwen3_5/execution/text.cpp:2939-3100 `forward_tp2_window_batch` (per-lane kv row/state slot/valid_columns are all already bound as [batch], width=max(chunkLen)). Two hard constraints:

- With attention batch>1, each lane's width≤16 (causal_softmax_attention.cpp:241-243), and batch>1 never selects the Prompt route (:382) ⇒ a large-T batched prompt forward is unsupported; only small chunks with width≤16 are possible.
- On the TP-2 GDN non-record path, batch>1 forces width==1 (text.cpp:1748-1749, :1917-1918); only `GdnStateAction::RecordForReplay` supports width>1 ⇒ batched prefill must go through record+fold.

Benefit assessment: at width≤16, "fewer kernel launches" does not automatically hold relative to single-lane large-T prefill; the definite benefit is that **the admission position gap is confined within the chunk width** (converging the single-valued envelope of the speculative batch). Moreover `window_batch` goes through the non-overlap layer loop (text.cpp:3076/3078), losing the +11% from L2 AR‖MMA overlap (docs/tp2-decisions.md:281-285).
⇒ Not done this time; report §4 records this conclusion (consistent with §4.2 "batched prefill cannot improve long-prompt aggregate throughput").

### Item D additional measurement (width cost, 2026-10-02, deciding whether the minimal slice is worth writing)

Before writing the slice, first measure the true cost of the slice's **only newly added op** (one B-lane × W-column `forward_tp2_window_batch`, including the in-model GDN record path): `[tp2-time] spec-batch`'s `verify` = `close_round`'s elapsed(1,2), which exactly brackets that call. Config: artifact <models-dir>/qwen3_8_27b_swift15_dflash2_final.ninfer, TP-2 dual card, int8, 131072, C=4, prefill-chunk 1024, `--spec dflash2 --draft-tokens 5|15`, ARM P4 four shorts in the same batch (harness _temp/20261002-1925_w16probe2.ps1, client _temp/20261002-1620_mtpshort.mjs). Note that MTP cannot reach width 16 (src/product/speculative_options.h:41-43 limits draft-tokens≤5); only dflash2 (≤15) can.

Accounting: `++timing.rounds` increments by 1 per submitted column (tp2_generation_core.cpp:3590 and :4815); a round has B lanes each submitting one column ⇒ per-round cost = reported value × B (B=4).

| Config | Per-round avg_round | Per-round mtp | Per-round verify | verify per column |
|---|---|---|---|---|
| W=6 (drafts 5) | 80.9 ms | 17.4 ms | 58.8 ms | 2.45 ms |
| W=16 (drafts 15) | 108.6 ms | 24.0 ms | 79.6 ms | 1.24 ms |

Fit **verify(W,B=4) = 46.3 ms + 0.52 ms/column**; against the report §4.2 measured single-lane prefill fit **39 ms/chunk + 0.549 ms/token**. ⇒ batched prefill W=16,B=4 = **1.24 ms/lane-token**, existing single lane 1024-chunk = **0.587 ms/lane-token**, **batched is 2.1× slower**. Mechanism: the fixed cost (~40 ms of weight stream) is amortized over only W×B=64 columns in a batched round, versus 1024 columns in a single-lane chunk. Break-even needs W×B≈1216 columns/round (W≈300 at B=4, 19× the ops cap of 16 = option B).
⇒ The minimal slice (option A) has an expected negative value; the only interval where it can win is prompt < ~56 tokens (already one chunk).

## Record

- Items 1/2/3/6/7: tools/win_port/build.ps1 full rebuild BUILD_EXIT=0 (712 s, including the msvc_deps_prefix fix); 6/7 incremental BUILD_EXIT=0 (29 s); item 1 (new plain instrumentation) incremental rebuild BUILD_EXIT=0 (28 s, job pwsh-59); `tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` 12/12 pass (6 model-related TP-2 tests skipped for lack of weights).
- Item 4: rebuild after revert BUILD_EXIT=0 (27 s, job pwsh-53); measurement above.
- Item 8: docs/serving.md gained the explanations for 1/2/3/4/6/7 (including measured numbers).
- Item 10: MTP acceptance attribution measurement (`_temp/20261002-1550_mtp.mjs`): S = single long run, A = four long in the same batch, B = long+short (different positions). Note that `--max-context 245760 --kv-dtype int8 --spec mtp` OOMs at startup (`cudaMalloc failed`, 39 s); you must return to the documented accounting `--max-context 131072 --kv-dtype int8`.
- **Second-round final verification (all of this round's changes, worktree uncommitted)**: `tools/win_port/build.ps1` BUILD_EXIT=0 (470 s, full rebuild — ops header changes trigger the whole chain); `tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options|softmax_attention'` **15/15 pass** (307.98 s, including ninfer_softmax_attention_test's 205.62 s batch/solo parity regression, nvfp4 14.26 s, k8v4 15.23 s; the 6 model TP-2 tests skipped with no artifact); `git diff --check` exits 0.
  The solo-run accounting of the model tests with the artifact and the comparison against the HEAD baseline are in "Item E implementation and verification".
