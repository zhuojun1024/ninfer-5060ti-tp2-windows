# PLAN: TP-2 shared KV dynamic reservation (kv sharing) implementation plan

> **This file is the single active plan for the TP-2 shared KV project**, and the persistent memory across context compression; after the context is compressed or the session is resumed, re-read this file before continuing.
>
> - **Baseline**: branch `feat/tp2-concurrency`, `HEAD = 94e6cc64` (11 commits, not pushed); Windows worktree `Documents/workbench/ninfer`, build `build-win` (VS2022 + CUDA 13.3).
> - **Provenance**: this file was solidified from a source-level evaluation (user instruction m00683 "land the document and start implementing", evaluation request m00552 "evaluate dynamic sharing. You may also refer to how a local clone of that fork does it").
> - **Adjacent authorities**: `tp2-plan-concurrency.md` (the main multi-concurrency plan; this file is the continuation of its KV semantics), `docs/serving.md` (external contract), `docs/maintainer/paged-kv-cache.md` (KV physical model and execution row), `docs/maintainer/engine-architecture.md` (single-card execution/scheduling contract), `docs/cli.md` (startup knobs).
> - **Semantics**: shared KV in this project = **one physical pool + full per-request reservation at admission + return at request terminal state + no preemption**. Consistent with the single-card route's KV semantics; it does not include per-round free-slot search, eviction of idle slots, or cross-lane preemption.

---

## 0. Conclusion (evaluation delivered, used as the implementation baseline)

**Worth doing, and the correct convergence direction for TP-2.** Today's TP-2 statically splits one physical KV pool evenly among lanes at startup, which amounts to dividing "the machine's total context" by the concurrency level; whereas the single-card route's semantics are "one shared pool, whoever runs uses it to the full". The two contradict each other under the same product contract:

- Under static even splitting, `--max-context 131072 + --max-concurrency 4` ⇒ 512 pages per lane = **32768 token**. The user's measured (m00498) failing case `prompt 14811 + output 17957 = 32768` exactly fills it up, and the second round directly gets `HTTP 400 context_length_exceeded`;
- Under dynamic sharing the same two requests each need `pages_for_tokens(14811 + 32768) = 744` pages, `2 × 744 = 1488 ≤ 2048` ⇒ **both sessions run at once**; four requests of the same size `2976 > 2048` ⇒ two concurrent, two queued (queued rather than 400);
- Under dynamic sharing a single `prompt 90000` request can also run (static 32768 gets an immediate 400).

Implementation is split into two stages: **S1 text KV dynamic reservation/release (the main body)**, **S2 the ceiling knob `--lane-context`**; MTP mirror (S3) and documentation/acceptance (S4) follow. Completing S1 already covers the user's production route (`--spec dflash2`) and the plain route.

---

## 1. Goals and non-goals

### 1.1 Goals
- **Share one physical KV pool** among lanes: a request running alone can use the whole pool; when multiple requests overlap they are divided by each one's actual need, rather than evenly split at startup.
- At admission, fully reserve pages for the request's **prompt + effective output**; only admit on success; when the pool is insufficient, **queue** (FIFO) instead of returning 400 (only a prompt that itself exceeds the ceiling gets a 400).
- Return physical pages at request terminal state; once the pages go back to the pool's free table they can be used immediately by requests newly arriving in the same batch.
- The single-card route (`--devices` single card / `lanes_ == 1`) behaves **bit-identically**.
- The plain route at temperature 0 is still bit-deterministic; gate 11/11, forward/load, and `tp2_sessions` match the baseline.

### 1.2 Non-goals (not done in the first phase)
- Per-round free-slot search (llama.cpp `find_slot`), eviction of idle slots, preemption, continuous batching.
- Defragmentation (the host path does not require contiguous pages; fragmentation only affects the largest single request that can fit, with no correctness impact).
- Changes to the KV semantics of multimodal lanes (the existing per-lane binding is kept).
- `--kv-capacity` on the TP-2 route is still normalized to `--max-context` by `normalize_engine_options` (see `src/runtime/engine/model_instance.cpp:119`); this plan does not change that.

---

## 2. Current state: static even splitting (code evidence, file:line from the 2026-10-02 `94e6cc64` worktree)

| Location | Fact |
|---|---|
| `src/runtime/engine/tp2_generation_core.cpp:441-443` | `pages_per_lane = pages_for_tokens(options_.max_context) / lanes_; lane_token_capacity_ = pages_per_lane * kPagedKVPageSize;` |
| `src/runtime/engine/tp2_generation_core.h:454-458` | `lane_context_window()`: `lanes_ == 1` returns `options_.max_context`, otherwise returns `lane_token_capacity_ - margin` (with MTP margin = `mtp_drafts_ + 2`, otherwise 1) |
| `tp2_generation_core.cpp:1054-1122` | one-time startup `pool.reserve` + `materialize` + `tables.publish(row, 0, handles)`; lane `l` gets `[l*per_lane, (l+1)*per_lane)`, the last lane gets the remainder |
| `tp2_generation_core.cpp:1081-1089` | A3 startup assertion: `lane_context_window() + write_tail < per_lane * 64` |
| `tp2_generation_core.cpp:1124-1176` | MTP layer KV uses the same static split (shard 0) |
| `tp2_generation_core.cpp:2015-2021` | `submit()` uses `lane_context_window()` to decide the 400 and clamp output |
| Production log | `[mem] shard 0 KV pages 2048 over 4 lanes = 512 pages (32768 tokens) per lane` |

The pool itself **is already dynamic** (this is the fundamental reason the work in this plan is tractable):
- `src/core/paged_kv_cache.h:197` `class DeviceKVPagePool` has its own free page table (`free_page_runs_`/`page_allocated_`, `:262-275`), `reserve()` (`:217`), `materialize()` (`:228`), `capacity_pages()`/`available_pages()` (`:208-211`).
- `src/core/paged_kv_cache.h:339` `class KVExecutionTablePool`'s `publish(row, logical_begin, pages, stream)` (`:353-354`) is a **repeatable, incremental** publication: implemented as host shadow + `cudaMemcpyAsync` (`src/core/paged_kv_cache.cpp:774-833`), so each admit only has to overwrite its own range.
- The single-card route is exactly this semantics: `src/models/qwen3_5/program/storage/kv_store.h:243` `LogicalKVPageStore` repeatedly moves pages between `materialize/dematerialize`.

---

## 3. llama.cpp comparison (a local clone of that fork)

| Mechanism | llama.cpp | This plan |
|---|---|---|
| Switch | `common/common.h:573 bool kv_unified = false;` (static even splitting by default); `include/llama.h:406` | TP-2 defaults to the target semantics (one pool) |
| Static even-split arithmetic | `src/llama-context.cpp:288-304`: `n_ctx_seq = n_ctx / n_seq_max`, truncated down with a warning when not evenly divisible | Exactly isomorphic to NInfer's `:441-443` today |
| Per-slot ceiling | `common/common.h:630 int32_t kv_unified_per_slot = 0;`; `tools/server/server.cpp:153-169` sizes the pool with `n_parallel * kv_unified_per_slot` | S2's `--lane-context` (default 0 = whole pool) |
| Allocator | `src/llama-kv-cache.cpp:751-815 prepare()` whole-batch atomic placement + rollback on failure; `:898-1095 find_slot()` finds cells afresh per ubatch | **Not adopted**: `find_slot` is half of preemption/continuous batching, and re-publishing every round would make the CUDA Graph watermark assertion (`tp2_generation_core.cpp:1551-1555`) fragile |
| Release | `tools/server/server-context.cpp:1667-1690 try_clear_idle_slots()` is bound to preemption with `seq_rm` | Release only at request terminal state, no preemption |
| The cost of static even splitting (its own comment) | `tools/server/server-context.cpp:1420-1434` (`--cache-idle-slots`): under non-unified, clearing a slot frees no reusable space | The same mechanism as the user's m00498 400 |

Conclusion: **take unified's "one pool + ceiling knob", do not take its fresh cell search and eviction**.

---

## 4. Design (option A)

Definition: `need = prompt_tokens + effective_output` (`effective` has already been clamped by `submit()` against the ceiling); `margin = mtp_enabled_ ? mtp_drafts_ + 2 : 1` (following today's `lane_context_window()` margin semantics: MTP's draft chain writes past the longest lane position); `write_tail = mtp_enabled_ ? mtp_drafts_ : 0`.

| # | Rule |
|---|---|
| A1 | At admission, `reserve_pages = pages_for_tokens(need + margin)`, calling `reserve()` once each on both shards' text pools (and, on the MTP route, shard 0's MTP pool); if any fails ⇒ release the part already acquired and **do not admit** |
| A2 | Pool insufficient ⇒ **push the request back to the head of the FIFO** and end this round's admission (do not let a younger request cut in), retry next round; only `prompt > lane_admission_limit()` gets a 400 |
| A3 | After a successful admit, `materialize` + `tables.publish(row, 0, handles, stream)` **covering all of this request's reserved pages**; that lane's window is `kv_window = reserve_pages * kPagedKVPageSize - margin`, with the invariant `kv_window + write_tail <= reserve_pages * 64` (guaranteed by `reserve_pages = pages_for_tokens(need + margin)`; the runtime assertion is retained) |
| A4 | Release = `leases.clear()` (each lease's destructor returns its physical pages to `free_page_runs_`) + the reservation object (after a full materialize `reservation.pages_ == 0`, so its destructor is a no-op); **the execution row need not be cleared** — the window never crosses the published range, but every admit must publish its complete range (the A3 lesson: `src/core/arena.cu:141-155`'s `cudaMalloc` does not zero, so unpublished entries are garbage physical page numbers) |
| A5 | The ceiling is a **policy value** `lane_context_limit_ = min(policy, pool_tokens - margin)`, `pool_tokens = pages_for_tokens(max_context) * 64`; `policy` defaults to the whole pool and is narrowed by `--lane-context` in S2. Taking the `min` is so that **deadlock is impossible**: `need <= limit` ⇒ `reserve_pages <= pool page count` |
| A6 | `lanes_ == 1` takes the original path (one-time startup publish of the whole pool, window = `options_.max_context`), bit-identical |
| A7 | Allocation happens only at the two boundaries **admit / retire** ⇒ zero in-round overhead; the CUDA Graph's `arena_begin` watermark is untouched |

Release timing: a lane returns its pages immediately in `finalize()` after publishing the host session (so requests newly arriving in the same batch can reuse those pages at once); the batch executor's wrap-up returns idempotently once more for all lanes as a safety net.

---

## 5. Change list (S1, target shape)

| File | Change |
|---|---|
| `src/runtime/engine/tp2_generation_core.h` | `Shard` gains `mtp_lane_pages` (`vector<vector<DeviceKVPageLease>>`); delete `lane_token_capacity_`, add `lane_context_limit_`; `lane_context_window()` → `lane_admission_limit()` + `lane_kv_margin()`/`lane_kv_write_tail()`/`lane_kv_window(pages)`; declare `reserve_lane_kv/release_lane_kv/retire_lane_session/requeue_lane_front/lane_need_tokens` |
| `tp2_generation_core.cpp:433-449` | static split arithmetic → `lane_context_limit_` computation (including underflow protection for `pool_tokens - margin`) |
| `tp2_generation_core.cpp:1054-1122` | text pool: when `lanes_ > 1`, only `acquire` each lane's execution row, no reserve/publish; `lanes_ == 1` stays as is |
| `tp2_generation_core.cpp:1124-1176` | MTP pool likewise (shard 0 only) |
| `tp2_generation_core.cpp:2015-2021` | `submit()` switches to `lane_admission_limit()` |
| New functions (near `try_pop_lane_queue`) | `reserve_lane_kv(lane, need_tokens)` / `release_lane_kv(lane)` / `retire_lane_session(lane)` / `requeue_lane_front(pending)` / `lane_need_tokens(pending)` |
| plain batch executor `:2599-2625, 2710-2747, 3010-3012, 3048-3065` | `LaneState` gains `kv_window`; initial-batch and round-boundary admission call `reserve_lane_kv` before admit, on failure push back to the queue head (the initial batch must also erase from `batch`) and `break`; in `finalize`, `retire_lane_session` + `release_lane_kv` after `session_publish`; the driver `catch` releases as a safety net for all lanes |
| spec batch executor `:3239-3260, 3361-3413, 3660-3662, 3721-3756, 3763-3773` | same as above; `lane_window` becomes that lane's own `kv_window` |
| `tp2_generation_core.cpp:6584` | the single-lane walk's window switches to `lane_admission_limit()` (equivalent to today when `lanes_ == 1`) |
| Startup log | `[mem] shard %d KV pages %u over %d lanes = … per lane` → shared-pool wording |
| `docs/serving.md:68-76` | "lanes share one KV pool, so they partition the context ceiling" → dynamic reservation wording |

---

## 6. Risk table

| # | Risk | Level | Handling |
|---|---|---|---|
| R1 | A large-output request starves the others (a client can blow `max_output_tokens` up to the whole pool) | High | S2's `--lane-context`; S1 prints in the startup log "whole pool = N token, max single-request reservation = M pages" |
| R2 | "Pool insufficient" mistakenly falls into the D4 whole-batch failure path | High | A2 makes it explicit: insufficient ⇒ push back to the queue head + `break`, do not throw |
| R3 | The A3 invariant is bypassed (unpublished entries are garbage page numbers) | High | A3 runtime assertion + every admit covering the complete range |
| R4 | The handle vector for host KV / session retention becomes per-request | Medium | Keep the `kv_lane_handles[lane]` semantics as "the pages of that lane's current request"; store/restore both happen within the request's residency, and `pages_for_tokens(frontier) <= reserve_pages` always holds |
| R5 | Timing of release vs. reuse of the same pages by a new request in the same batch | Medium | Release only changes host-side counters; writes to reused pages and host copies execute in order on the same `device.stream` |
| R6 | Fragmentation (the pool suffices but there is no contiguous run) | Medium | No correctness impact (the host path does not require contiguity); document "actual concurrency may be lower than the lane count" |
| R7 | Actual concurrency lower than the lane count (C=4 only fits two 744-page requests) | Medium | Write into `docs/serving.md` and the startup log |
| R8 | The MTP pool is not made dynamic in step ⇒ silent out-of-bounds writes | High | S1 must change the MTP pool at the same time (same `reserve_pages` semantics); or, before S3 is done, push the MTP route's ceiling back to static even splitting (not adopted, changed together instead) |
| R9 | Existing failures such as `tp2_sessions` mistaken for regressions | Low | Line-by-line `Compare-Object` against the baseline (the existing plain shared-system-prompt failure stays unchanged) |

---

## 7. Staged steps

| Stage | Content | Completion criterion |
|---|---|---|
| **S1** | text + MTP pool dynamic reservation/release; `lane_admission_limit_`; three admission paths (plain initial batch / plain round boundary / spec initial batch / spec round boundary) | `ninja` 0; gate 11/11; forward/load 0; `tp2_sessions` matches the baseline; dual-card smoke V1–V6 all green (see §9.2; the single-card `--devices 0` half was not run) |
| **S2** | `--lane-context` (`0` = whole pool) + validation + `docs/cli.md` | the option parses; out-of-range values are rejected; docs synchronized; see §9.4 (done) |
| **S3** | (S1 already includes the MTP pool; here only the remaining mirror) `mtp_page_handles` dead-field cleanup, host checkpoint wording review | no residual references; see §9.3 (done) |
| **S4** | `docs/serving.md` / this file's work log / acceptance record | docs consistent with the code; see §9.1–§9.4 (done) |

---

## 8. Verification plan

### 8.1 Mandatory gates
- `ninja -C build-win` exit 0; `tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` 11/11;
- `ninfer_qwen3_5_tp2_forward_test.exe --artifact …` / `…_load_test.exe --artifact …` exit 0;
- the passed/FAIL lines of `$env:NINFER_TEST_ARTIFACT=…; tp2_sessions_test.exe` match the baseline under `Compare-Object`.

### 8.2 Behavioral acceptance (dual card `ninfer-serve --devices 0,1 --max-context 131072 --max-concurrency 4`)
| # | Scenario | Expectation |
|---|---|---|
| V1 | two requests each `prompt 14811 + max_output_tokens 32768` submitted simultaneously | both 200, and both run to completion (today the second is 400) |
| V2 | four requests of the same size submitted simultaneously | two run, two queue; the queued ones start automatically after the first finishes; no 400/499 |
| V3 | a single `prompt 90000` request (C=4) | 200 (today 400) |
| V4 | a short request arrives while a long request is running | the short request is admitted immediately (P2.2 dynamic admission does not regress) |
| V5 | single card (`--devices 0`) and C=1 | output matches the baseline |
| V6 | plain route temperature 0 | bit-deterministic |

### 8.3 Observation points
- startup log: shared pool page count and "max single-request reservation";
- `[tp2-lane] admit slot=… live=…`;
- when the pool is insufficient, a trace of the push-back to the queue head should appear (optional env switch).

---

## 9. Work log

### 9.1 S1: shared pool dynamic reservation/release (committed `d0931a2d`; baseline `HEAD=94e6cc64`)

**Changes**
- Header: `Shard` gains `mtp_lane_pages`; delete `lane_token_capacity_` → `lane_context_limit_`; `lane_context_window()` is split into `lane_admission_limit()` (`lanes_==1 ? options_.max_context : lane_context_limit_`) + `lane_kv_margin()` (`mtp_drafts_+2` with MTP, otherwise 1) + `lane_kv_write_tail()` + `lane_kv_window(pages)`; 5 new declarations.
- ctor: static split arithmetic → `pool_tokens = pages_for_tokens(max_context) * kPagedKVPageSize`, `lane_context_limit_ = pool_tokens - lane_kv_margin()` (throws when `pool_tokens <= margin`).
- Text/MTP startup blocks: `lanes_ == 1` keeps "one-time reserve+materialize+publish of the whole pool"; `lanes_ > 1` only `acquire`s each lane's execution row (no reserve, no publish), and the logs become `[mem] shard %d KV pool %u pages (%u tokens) shared by %d lanes` / `[mem] shard %d MTP KV pool %u pages shared by %u lanes`. The startup A3 assertion is deleted (the invariant moves into `reserve_lane_kv`).
- `reserve_lane_kv`: first clear any leases possibly left on that lane (self-healing for a failed batch) → `pages_for_tokens(need + margin)` → runtime A3 assertion → `reserve` on both shards one by one (plus shard 0's MTP pool when `mtp_enabled_`), returning `false` immediately if any fails (what was already acquired is returned to the pool by the reservation's destructor) → for each pool `materialize` + `publish(row, 0, handles, stream)`.
- `release_lane_kv`: clear both shards' leases/handles and shard 0's MTP leases/handles; clear that lane's retention (`cached_*`, `cached_state_valid`, `live_state_valid`, `reuse_source`, `anchor_session`, `block_anchor_*`, `invalidate_host_checkpoints`); set all `sessions_[].device_lane == lane` to -1; `active_session = kNoSession`.
- `retire_lane_session`: returns as soon as `session_store_active(lane)` succeeds, otherwise retries with `session_evict_one()`; **does not call `session_drop`** (`sessions_` is a vector, and dropping would invalidate the `active_session` index).
- Executors: `LaneState` gains `kv_window`; initial-batch and round-boundary admission call `reserve_lane_kv` first, on failure `requeue_lane_front` (the initial batch also erases from `batch`, otherwise the driver's wrap-up safety net would publish an empty result) + `break`/`continue`; `lanes[index].kv_window` is written only after `admit_lane` returns true (`admit_lane` starts with `lane = LaneState{}`, which clears it); `finalize` calls `retire_lane_session` + `release_lane_kv` after `session_publish` and before `publish_lane`; the spec round's `capacity_left` switches to that lane's own `kv_window`; `execute_walk` (the single-lane unreachable path) switches to `lane_admission_limit()`; the driver's `catch` releases as a safety net for `0..lanes_-1`.
- `docs/serving.md:68-76` changed to the shared-pool dynamic reservation wording.

**Non-obvious points settled while writing the code**
- Release must also invalidate the lane's "device-resident" claim: the physical pages may be taken by another lane at once, while all three sources of `scan_lane_reuse`/`session_recall` (LiveState/DeviceSnapshot/HostCheckpoint) assume "this lane's device KV still holds that prefix" ⇒ failing to invalidate would silently skip prefill and read another lane's KV.
- On retirement, `session_store_active` first and release second; cross-request reuse then goes through the host slab (next round's `session_recall` → `session_restore` copies it back to new pages); `active_session = kNoSession` makes that host entry a candidate again (`:5120` skips only when `index == active_session`).
- Admission ceiling = whole pool - margin ⇒ as long as `prompt + effective <= lane_context_limit_` it necessarily fits in an empty pool; "pool insufficient" can only mean waiting for another lane to retire, so there is no deadlock.

### 9.2 S1 verification record (2026-10-01/02, baseline `HEAD=94e6cc64`; committed with `d0931a2d`)

**Gates (all passed)**

| Item | Result |
|---|---|
| `ninja -C build-win` | `BUILD_EXIT=0` (43 steps, including `tp2_generation_core.cpp.obj`) |
| `tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` | `100% tests passed out of 11`, 70.54 s, `GATE_EXIT=0` (the 5 artifact-driven cases being Skipped is expected) |
| `ninfer_qwen3_5_tp2_forward_test.exe --artifact …` | exit 0; `duplicated-lane control: lane_gap=0 solo_gap=2.03125 batch_lane0_gap=0 argmax_mismatches=0`; two-lane verify window `max_logit_diff=2.03125 argmax_mismatches=1 near_tie_flips=1 legacy_vs_lane0_diff=0` |
| `ninfer_qwen3_5_tp2_load_test.exe --artifact …` | exit 0; `TP-2 dual-shard load passed devices=0,1 layers=64 shard_gate=[8704,5120] shard_down=[5120,8704] head=[124160,5120] vocabulary-parallel` |

**Dual-card behavioral smoke** (`ninfer-serve --devices 0,1 --max-context 131072 --max-concurrency 4 --spec dflash2 --draft-tokens 5 --host-kv-mib 20480`, port 8099, `POST /v1/responses`)

The startup log changes from the old wording to the new one (for both shards):

```
[mem] shard 0 KV pool 2048 pages (131072 tokens) shared by 4 lanes
```

(old: `[mem] shard 0 KV pages 2048 over 4 lanes = 512 pages (32768 tokens) per lane`)

| Scenario | Result |
|---|---|
| single request `prompt 39804` (> the static ceiling 32768) + `max_output 16` | **HTTP 200**, 27.0 s (the old code necessarily gave 400) |
| two concurrent requests, each `prompt 39804 + max_output 32768` | **both 200**; 27.5 s / 54.7 s ⇒ the second starts automatically after the first retires (queued rather than 400) |
| single request `prompt 89546` + `max_output 16` | **HTTP 200**, 71.6 s (the old code necessarily gave 400) |

The reservation arithmetic is self-consistent: `need = 39804 + 32768 = 72572` ⇒ `pages = pages_for_tokens(72573) = 1134`, `2 × 1134 = 2268 > 2048` ⇒ necessarily serial, matching the observed "the second takes about twice as long". `status=incomplete` is `max_output_tokens=16` hitting the ceiling, not an error. Graph wording `[tp2-graph] plain decode step: eager(exact) | batched: graph | verify step: eager | verify batch: graph | mtp chain: n/a`; `engine ready 30.9 s` / `warmup complete 1.2 s`.

**Behavioral acceptance (V1–V6, dual card `--max-concurrency 4`, `POST /v1/responses`)**

| # | Scenario | Result |
|---|---|---|
| V1 | two concurrent, each `prompt 39804 + max_output 32768` | both 200; 27.5 s / 54.7 s (the second starts automatically after the first retires, not 400) |
| V2 | four concurrent (each `prompt 19968 + max_output 32768`) | all 200; `25474 / 52510 / 38471 / 51652 ms` (two waves), trace `[tp2-lane] admit slot=0 live=2`, no 400/499 |
| V3 | a single `prompt 89546` (C=4) | 200, 71.6 s (necessarily 400 under static splitting) |
| V4 | a short request arrives while a long request is running | the short request is admitted immediately: long 27324 ms / short 23861 ms |
| V5 | C=1 (same binary `--max-concurrency 1`, S1 early-returns entirely at `lanes_ <= 1U`) | consistent with C=4: the same reuse probe gets `cached=5749` at both C=1 and C=4; the `tp2_sessions` lines match the baseline word for word. **The single-card `--devices 0` half was not run** (the TP-2 artifact is a dual-shard layout; single card is a different artifact/layout) |
| V6 | plain route `temperature 0` | the answer channel is bit-deterministic: under all four combinations of 2 sequential runs and 2 concurrent runs × `reasoning.effort` `none`/`xhigh` the message text hashes are all identical (`Blue` / `Red`); the thinking channel is reproducible only within the same batch shape (two sequential runs have the same hash, a concurrent pair has the same hash, but sequential vs concurrent is 178 vs 193 characters) ⇒ consistent with the existing wording that "a different attention envelope within a batch ⇒ ulp-level near-ties can flip", not introduced by S1 |

**`tp2_sessions` baseline comparison** (`NINFER_TEST_ARTIFACT=<models-dir>/qwen3_8_27b_swift15_dflash2_final.ninfer`)

- default plain: `unaligned dialogue passed`, `replayed answer passed`, `FAIL (plain): a conversation behind a shared system prompt diverged from the oracle on its first sample: got [2752 13 198 197 197 92 198 197] expected [467 419 538 13 198 197 197 92]`, `EXIT=1`.
- `NINFER_TEST_ROUTE=dflash2`: `unaligned dialogue (dflash2) passed`, `replayed answer (dflash2) passed: the turn after a re-rendered answer reused 70 of 58 prompt tokens`, `FAIL (dflash2): a recalled conversation diverged from the oracle on its first sample: got [59399 475 327 363 62 16 15 15] expected [365 4577 287 62 16 15 15 15]`, `EXIT=1`.
- Both FAIL strings are **word-for-word identical** to the HEAD baseline recorded at `tp2-plan-concurrency.md:659` (and S1 early-returns entirely at C=1) ⇒ no sessions regression. For the pre-existing problem from the same source see `worklog §3.10` (`dflash_solo`'s `a_continued` first-sample mismatch, already verified via `git stash` to be unrelated to the changes at that time).

**`--spec mtp` multi-lane smoke** (`--spec mtp --draft-tokens 2`, C=4)

- Startup log `[mem] shard 0 MTP KV pool 2048 pages shared by 4 lanes` (the MTP pool hangs only off shard 0, likewise shared); `[tp2-graph] … mtp chain: n/a` (no chain built with multiple lanes, the P3.1 gate in effect).
- M1 a single `prompt 39804 + max_output 16` ⇒ 200 (24.5 s); M2 two concurrent `prompt 19968 + 16` ⇒ both 200 (21.7 / 22.1 s); M3 two concurrent `prompt 39804 + max_output 32768` ⇒ both 200 (25.5 / 50.7 s, pool insufficient ⇒ serial); M4 two concurrent tools requests ⇒ both 200 and each produces a `get_weather` call (the tool grammar works with multiple lanes in a batch).

**Cross-request reuse round trip (S1's core behavior change)**

- Probe `_temp/s1_diag.mjs`: R1 = a long prompt, R2 = replaying R1's `output` array verbatim + a follow-up question (only this shape hits the entry's frontier boundary).
- C=4: `[tp2-session] entry 0 tokens=5750 frontier=5749 kv_end=5749 prompt_end=5718 shared_end=0 shared=5750 reach=5749 via=frontier resident=0` + `recall frontier=5749 resident_depth=0 tokens=5750 entries=1`; usage `cached_tokens=5749`, elapsed 3964 → **511 ms**.
- C=1 control (same binary): likewise `cached=5749` (383 ms) ⇒ the `release_lane_kv` → `session_store_active` → `session_recall` → `session_restore` round trip really does work on the multi-lane route, and matches single-lane behavior (S1's core risk, "reuse silently breaks after release", is disproved).
- Counterexample (pre-existing semantics, not introduced by S1): when the client re-renders the assistant turn itself (`asst('ok')`), `shared=5717` is 1 less than R1's prompt (5718) (the chat template only appends the generation prompt after the last user turn), and `offered[kind] > shared` rejects all three boundaries frontier / prompt_end / shared_end ⇒ `reach=0`, `cached=0`. That criterion is shared by `session_recall` (`tp2_generation_core.cpp:5257-5381`) and `scan_lane_reuse`, unchanged before and after S1.

---

### 9.3 S3: mirror cleanup and wording review (committed `d0931a2d`)

- Delete the dead field `Shard::mtp_page_handles`: it was only populated by `clear`/`reserve`/`push_back` in the single-lane startup block, with no read site anywhere in the repository — host checkpoint export reads `mtp_lane_handles[lane]` (`src/runtime/engine/tp2_generation_core.cpp:5059` and `:5163`), and `mtp_lane_handles[0]` already holds the same batch of handles. The header comment saying "`mtp_pages` and `mtp_page_handles` are the whole-pool lists pinned by the single-lane route and used for host checkpoint export" is correspondingly changed to mention only `mtp_pages` (keeping the whole-pool lease alive) and the per-lane-indexed four-piece set (`mtp_lane_pages`/`mtp_lane_handles`/`mtp_rows`/`mtp_views`).
- Host checkpoint wording review (dual card `--spec mtp --max-concurrency 4` startup log): `[mem] host checkpoint ring 32 slots x 73.4 MiB/lane/shard`, `[mem] host-checkpoints shard 0 slots 32 (grid 3 + tail 3 + divergence 1 + block 1) x 73.4 MiB | stride 43776 tok | pinned 2349.0 MiB` (32 × 73.4 = 2349 ✓), consistent with before S1 — S1 only changes *when* a mirror is taken, not each lane's slot geometry.
- Regression: `ninja -C build-win` exit 0 (43 steps); `tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` 11/11 (70.67 s); `--spec mtp --max-concurrency 1` single-lane startup + one `POST /v1/responses` ⇒ 200 (exactly the startup branch that was changed).

---

### 9.4 S2: the `--lane-context` ceiling knob (committed `d0931a2d`)

**Semantics**: `0` (default) = whole pool, S1 behavior byte-identical; `>0` = how many tokens one lane can admit at most (aligned with llama.cpp `--kv-unified-per-slot`), used to narrow, by policy, "first-come large requests monopolize the whole pool"; the `--max-concurrency 1` route ignores it (a single lane already owns the context) and says so in the startup log.

**Change points**
- `include/ninfer/types.h:181` adds `std::uint32_t lane_context = 0;` (`EngineOptions`, public API).
- `src/serve/serve_options.h:37` same-named field; `src/serve/serve_options.cpp:181` parses `--lane-context N`; `:403` validates `lane_context <= max_context`, otherwise `--lane-context must be 0 or at most --max-context`; the usage text gains `[--lane-context N]` and a one-line description.
- `src/serve/generation_service.cpp:243` maps it into `EngineOptions`.
- `src/runtime/engine/model_instance.cpp:65` (`validate_options`) and `:112` (the TP-2 branch of `normalize_engine_options`) each add the same validation: an out-of-range value **errors rather than being silently clamped**, ensuring the operator's number matches the ceiling the core advertises.
- `src/runtime/engine/tp2_generation_core.cpp:451-463`: after `lane_context_limit_ = pool_tokens - margin`, take the smaller by policy (`lane_context != 0 && lane_context < limit` ⇒ `limit = lane_context`), and print `[mem] TP-2 lane admission ceiling %u tokens (--lane-context %u)`; `:471-478` prints `[mem] --lane-context %u is ignored at --max-concurrency 1: the single lane owns the whole context` when `lanes_ == 1U && lane_context != 0`.
- Tests: `tests/test_serve_options.cpp:164-176` (parses into ServeOptions + rejected when exceeding `--max-context`), `tests/test_engine_options.cpp:107-125` (preserved by normalization + rejected when out of range).
- Docs: `docs/serving.md:851` (new row in the option table), `docs/serving.md:1057-1060` (policy note added to the admission paragraph), `docs/cli.md:279-281` (KV paragraph points to serving).

**Acceptance** (`ninja -C build-win` exit 0, 586 steps; `test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` 11/11, including the two new assertions; script `_temp/run_s2_verify.ps1`, logs `_temp/s2_c4.err` / `_temp/s2_c1.err`)

| Scenario | Command | Result |
|---|---|---|
| startup log (C=4) | `--max-concurrency 4 --lane-context 32768` | `[mem] TP-2 lane admission ceiling 32768 tokens (--lane-context 32768)`, in the same run `[mem] shard 0 KV pool 2048 pages (131072 tokens) shared by 4 lanes` |
| large prompt (39340 tok + max_output 16) | same as above | **HTTP 400** (the ceiling takes effect; a prompt of the same magnitude is 200 under the default `0` — already proved by S1's V3 with 89546 tok) |
| medium prompt (19010 tok + max_output 4096) | same as above | HTTP 200, `status=completed`, 14.6 s |
| startup log (C=1) | `--max-concurrency 1 --lane-context 32768` | `[mem] --lane-context 32768 is ignored at --max-concurrency 1: the single lane owns the whole context` |
| large prompt (39340 tok + max_output 16) | same as above | **HTTP 200** (C=1 ignores the ceiling, the whole pool is available), 26.2 s |

**Not covered**: the fairness/fragmentation benefit of `--lane-context` is not quantified (quantifying R1 is left until there is a real multi-request load).

---

### 9.5 Operational surface additions: 400 message arithmetic, capacity warning, architecture-doc exception note (landed)

**Background**: after S1 replaced "each lane has a fixed 1/lanes of the context" with a shared pool, an operator receiving a 400 could not tell from the message whether it was the "policy ceiling" or the "pool itself", and a long prompt hitting the ceiling and truncating the output produced no log at all. This section adds those three (approved by user m01348).

- **400 message carries the arithmetic** (`src/runtime/engine/tp2_generation_core.cpp:2017-2043`): `submit`'s `context_length_exceeded` splits into two branches —
  - policy-bound (`--lane-context` is the ceiling): `prompt exceeds this lane's context capacity: prompt 39340 tokens > lane ceiling 32768 tokens (--lane-context 32768); raise --lane-context to widen it`
  - pool-bound (default whole pool, **example message**; this branch is reachable only in the narrow band `(pool_tokens − margin, max_context]`, see finding 2, so no end-to-end measurement was done): `… prompt 17200 tokens > lane ceiling 16383 tokens (KV pool 256 pages x 64 tokens = 16384 tokens minus a 1-token write margin, --lane-context 0 = whole pool); raise --max-context to widen it`
- **400s in the frontend preparation stage also carry the token count** (`src/models/qwen3_5/frontend/frontend.cpp:256-263`'s helper gains `prompt_tokens`/`prompt_count_is_capped` parameters, call sites `:849` (`encoded.input_ids.size()`, `at least` when the cap is hit) and `:922` (`token_ids.size()`, the exact value); `src/models/qwen3_5/frontend/processor.cpp:935-940`, `:1066-1071` likewise gain `prompt at least N tokens`): `prepared prompt exceeds Engine max_context 16384: prompt at least 16385 tokens`. This is the only message an operator can see when the prompt exceeds `--max-context` (see finding 2 below); the tokenizer ceiling means it can only give a lower bound, hence `at least`.
- **Capacity warning** (same file `:2044-2055`): when `limit_reason == FinishReason::ContextCapacity` (the ceiling rather than the client determined the output budget), print one line `[tp2-capacity] prompt %u + requested output %u exceeds the %u-token lane context ceiling; the output budget is clamped to %u`; requests that were not truncated print nothing.
- **Architecture-doc exception note** (`docs/maintainer/engine-architecture.md:45-51` gains a paragraph): it states that multi-lane TP-2 is the concrete form "one paged KV pool + reservation of prompt+budget+write margin at admission + queue-head retention when the pool is insufficient + the only 400 is a prompt over the ceiling + `--lane-context` narrowing + no preemption", and points out that fairness is guaranteed by the operator's policy rather than by the scheduler.
- **Doc synchronization**: the 400 paragraph at `docs/serving.md:415-417` gains a sentence on the TP-2 wording (the ceiling is the lane's admission ceiling: whole pool by default, narrowable with `--lane-context`, and the message states which branch is constraining); the guardrail entry at `tp2-plan-concurrency.md:287` notes that the message was later changed to the two arithmetic-carrying branches.

**Acceptance**: `ninja -C build-win` exit 0; gate 11/11; dual-card smoke (`_temp/run_s2c_verify.ps1`, `_temp/run_s2d_verify.ps1`) —

| Scenario | Expectation | Measured (2026-10-02) |
|---|---|---|
| C=4 `--max-context 131072 --lane-context 32768`, prompt 39340 + 16 | 400, policy-bound message | HTTP 400 (72 ms): `prompt exceeds this lane's context capacity: prompt 39340 tokens > lane ceiling 32768 tokens (--lane-context 32768); raise --lane-context to widen it` |
| C=4 `--max-context 131072 --lane-context 32768`, prompt 19010 + 32768 | 200, plus one `[tp2-capacity]` line | HTTP 200 completed (14.5 s), `[tp2-capacity] prompt 19010 + requested output 32768 exceeds the 32768-token lane context ceiling; the output budget is clamped to 13759` (= 32768 − 19010 + 1) |
| C=4 `--max-context 16384` (default whole pool), prompt longer than 16384 | 400, message gives the prompt lower bound | HTTP 400 (38 ms): `prepared prompt exceeds Engine max_context 16384: prompt at least 16385 tokens` |
| C=1 `--max-context 16384`, prompt 9510 + 16384 | 200, the warning prints against the single-lane ceiling | HTTP 200, `[tp2-capacity] prompt 9510 + requested output 16384 exceeds the 16384-token lane context ceiling; the output budget is clamped to 6875` (= 16384 − 9510 + 1) |

**Two things discovered during verification and backfilled**:

1. The **server-side default of `--max-context` is 8192** (`src/serve/serve_options.h:32`). After the first round of scripts removed `--max-context` from the common parameters, the C=4 `--lane-context 32768` server exited immediately at startup (exit 1), first stderr line `ninfer-serve: --lane-context must be 0 or at most --max-context`. So acceptance of the "policy-bound" branch must explicitly pass `--max-context 131072`.
2. **When the prompt exceeds `--max-context`, the frontend preparation stage rejects first** (`src/models/qwen3_5/frontend/frontend.cpp:846`, `src/models/qwen3_5/frontend/processor.cpp:935-940` and `:1066-1071`), so the core's **pool-bound branch is reachable only in the narrow band `(pool_tokens − margin, max_context]`** (band width = `margin − (pool_tokens − max_context)`: exactly 1 token for dflash2 with `margin=1` and zero page-alignment remainder; at most 17 token for MTP with `--draft-tokens 15`), effectively a defensive branch. What is really useful to the operator is the frontend one, which also gained the token count here; since the frontend tokenizer is capped at `max_context + 1` (`encode_rendered_chat(..., impl_->max_context + 1U)`, `EncodeOptions{.max_tokens = encode_limit}`), the count can only give a lower bound, so the message reads `prompt at least 16385 tokens`.

### 9.6 `kReuseSnapshotCount` 2→1: keep only the prefill end-state boundary on the device side (committed `371eded9`, m01730)

**Motivation**: `state_arena = std::make_unique<DeviceArena>((2 + kReuseSnapshotCount) * state_bytes)` (`src/runtime/engine/tp2_generation_core.cpp:836-837`) and `dflash_snapshot_arena = std::make_unique<DeviceArena>(kReuseSnapshotCount * dflash_image)` (`:1026`) show that each shard keeps `kReuseSnapshotCount + 2` full-plane states resident (live + each reuse boundary + round scratch) and `kReuseSnapshotCount` draft rings. Of C=4's 2168 MiB over C=1, 1334 MiB is this batch of snapshots (2 state boundaries 1174.5 + 1 draft ring 160), and only about 834 MiB is multi-concurrency itself.

**Why 1 is the lower bound**: `kRoundScratchSlot = kReuseSnapshotCount` (`src/runtime/engine/tp2_generation_core.h:68`), while `publish_lane_prefill` uses `state_snapshots[0]` to store the prefill end state (`tp2_generation_core.cpp:2514-2525`), and slot 0 is simultaneously the sole source of the prompt end-state mirror in the session slab (`session_store_active` → `entry.host_prompt_state`/`entry.host_dflash_prompt`, `:5079-5100`). Dropping it to 0 would make round scratch and the prefill end state contend for the same plane, so this change only cuts the rewind slots (slot 1..count-1).

**Change points** (one constant + 5 boundary wordings)

| Location | Change |
|---|---|
| `src/runtime/engine/tp2_generation_core.h:55-64` | `kReuseSnapshotCount = 2` → `1`, comment changed to "rewind is not worth a device plane: each slot per shard is yet another full-plane state, the host checkpoint ring already holds the same batch of state, and slot 0 is retained because the session slab reads the prompt end-state mirror, so 1 is the lower bound" |
| `src/runtime/engine/tp2_generation_core.h:274-277` | `dflash_snapshots` comment drops "slot 1 is the rewind behind it" |
| `src/runtime/engine/tp2_generation_core.cpp:1016` | draft ring arena comment "the two reuse boundaries" → "the reuse boundaries" |
| `src/runtime/engine/tp2_generation_core.cpp:6045-6050`, `:6063` | in the `[tp2-reuse]` trace, the out-of-bounds read of `cached_boundaries[1]` becomes a loop taking the deepest rewind slot (`rewind_boundary`) |
| `src/runtime/engine/tp2_generation_core.cpp:6465-6470` | the out-of-bounds write `cached_boundaries[1] = 0` on the cancellation path becomes a loop clearing |

**Semantic consequences**
- All the rewind-slot loops were already `for (slot = 1; slot < kReuseSnapshotCount; ++slot)` (`:6048`, `:6340`, `:6468`, `:6500`, `:6559`), so at count=1 they all spin idle: neither the device-side rewind plane nor the clamping of `snapshot_at[1]` against prefill chunks exists (`:6500-6503`).
- **On C=4 this plane is a dead allocation**: `release_lane_kv` (`:2664`) is called at every request's terminal state when `lanes_ > 1`, clearing `cached_boundaries` and `cached_state_valid`, so the entire boundary scan of `scan_lane_reuse` (`:2331-2400`) is not entered from the second round on (trace `src=none`). C=4's cross-round reuse goes through `session_recall`/`session_restore` and the host slab, unrelated to this change.
- **C=1 is the only semantic change point** (`release_lane_kv` early-returns for `lanes_ <= 1U`): a reuse request landing "within one chunk of the previous round's prompt end, but not equal to the end state" now gets its boundary from the host checkpoint ring's tail sub-ring (`[mem] host-checkpoints shard N slots 34 (grid 24 + tail 8 + divergence 1 + block 1) x 73.4 MiB | stride 8192 tok`, the tail covering the last 8192 token). **The reuse depth is unchanged; only the restore source changes from a device snapshot (D2D) to H2D** (73.4 MiB state + 40 MiB draft/lane).

**Acceptance** (`tools/win_port/build.ps1` BUILD_EXIT=0, incremental 29–30 s; gate `tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` 11/11, 71.48 s)

Memory (same binary and parameters, differing only in `--max-concurrency` and the constant; count=1 logs `_temp/cnt1_c1.err`/`_temp/cnt1_c4.err`, baseline `_temp/s2_c1.err`/`_temp/s2_c4.err`):

| Item MiB | count=2 baseline | count=1 | Delta |
|---|---|---|---|
| C=4 shard 0 `state` | 1174.5 | 880.9 | −293.6 |
| C=4 shard 0 `draft-snap` | 320.0 | 160.0 | −160.0 |
| C=4 shard 0 `free` | 726.0 | 1180.0 | +454.0 |
| C=4 shard 1 `state` | 1174.5 | 880.9 | −293.6 |
| C=4 shard 1 `free` | 1424.0 | 1718.0 | +294.0 |
| **C=4 both cards total** | — | — | **+748** |
| C=1 shard 0 `state`/`draft-snap`/`free` | 293.6/80.0/1996.0 | 220.2/40.0/2108.0 | +112 |
| C=1 shard 1 `state`/`free` | 293.6/2322.0 | 220.2/2394.0 | +72 |
| **C=1 both cards total** | — | — | **+184** |

⇒ **the C=1→C=4 increment drops from 2168 MiB to 1604 MiB** (= 2168 − the 748 saved at C=4 + the 184 saved at C=1; the DFlash2 round line shrinks in step: C=1 `ring 40.0 | context 90.3 | frame 3.4 | proposal 3.3` → C=4 `ring 160.0 | context 211.2 | frame 12.1 | proposal 4.0`).

Behavior (reuse depth unchanged, A/B on the same probe `_temp/cnt1_rewind.mjs`):

| Scenario | count=2 | count=1 |
|---|---|---|
| C=1 rewind band (R2 = FILLER×285 + a different closing sentence, 5431 tok / shared 5418) | `cached=5120` (707 ms), trace `reuse=5120 slot=1 src=device` | `cached=5120` (648 ms), trace `reuse=5120 slot=28 src=host` (`host=5/34 stride=8192`) |
| C=4 two-round recall (`_temp/s1_diag.mjs` probe) | `cached=5749` (511 ms) | `cached=5749` (470 ms) |

Dual-card artifact chain (`_temp/run_cnt1_artifact.ps1`): `tp2_forward_test` exit 0 (`duplicated-lane control: lane_gap=0 solo_gap=2.03125 batch_lane0_gap=0 argmax_mismatches=0`; two-lane verify `max_logit_diff=2.03125 argmax_mismatches=1 near_tie_flips=1`, word-for-word identical to the baseline); `tp2_load_test` exit 0; the full `tp2_sessions_test` log has exactly one FAIL — the plain divergence case, with output byte-identical to the HEAD baseline (`worklog §3.10`) ⇒ no regression (this sessions run was on the plain route; the dflash2 case was not run).

**Not covered**
- Rewind reuse restore changes from D2D to H2D; this change only proves the reuse depth and end-to-end latency are in the same band, without PCIe-bandwidth-level per-item timing.
- The chunk-clamping semantics of vision prompts differ from pure text (the vision cap and snapshot clamp order at `:6490-6503`); no vision reuse case was constructed.
- Rollback: change the constant back to 2 and rebuild (array sizes and slot loops follow automatically).

### 9.7 Moving the prompt end-state plane to the host ring (committed `87f1526d`, m01955)

**Motivation**: after §9.6 cut the device-side reuse boundaries down to only the prefill end state (`state_snapshots[0]`, 293.6 MiB per shard at C=4), the last resident plane existing "for a single consumer" is it: the only readers are `session_store_active` (the session slab's prompt end-state mirror, `src/runtime/engine/tp2_generation_core.cpp:5079-5100`) and C=1's in-lane prompt end-state reuse (the DeviceSnapshot branch of `scan_lane_reuse`, `:2372-2374`). Yet the host checkpoint ring already stores the same batch of state per boundary (`snapshot_host_checkpoint`, `:5736`), it just **deliberately skips the prompt end state** (`frontier != prompt_tokens`: batch `:3098`, walk `:6574`, decode `:7334`, all commented "the device snapshot below holds that same state already"). ⇒ Add a dedicated ring slot to carry it, and the device plane is no longer carved.

**Change points**

| Location | Change |
|---|---|
| `src/runtime/engine/tp2_generation_core.h:149` | `enum class HostRing` gains `PromptEnd` |
| `h:762`, `cpp:5887-5894` | new helper `publish_lane_checkpoints(lane)`: sets `valid[lane]=true` per `prefill_id[lane] == retention(lane).host_checkpoint_live_id`; shared by batch publish (`:2528`), walk publish (`:6850`), and the walk cancellation path (`:6582`) — **the cancellation path previously had no sweep**, and checkpoints written there stayed `valid=false` |
| `cpp:5827-5843` | the `PromptEnd` branch of `snapshot_host_checkpoint`: `begin = lanes_ * host_checkpoint_slots_per_lane_ + lane`, `count = 1`, no rotation (`index=0`), not subject to the lane slice offset (the `appended` flag) |
| `cpp:853-858` | ring size `lanes_ * slots_per_lane + lanes_` |
| `cpp:836-845` | when the ring is enabled the state arena carves only backing + `1..kReuseSnapshotCount` (`state_snapshots[0]` left empty); when the ring is disabled it keeps carving all `2 + kReuseSnapshotCount` |
| `cpp:1035-1050` | when the ring is enabled the draft arena carves `kReuseSnapshotCount - 1` planes (count=1 ⇒ 0, the arena is not created); when the ring is disabled it keeps `kReuseSnapshotCount` |
| `cpp:2524-2528`, `:6839-6850`, `:6580-6582` | the three publish sites write `HostRing::PromptEnd` (the cancellation path writes it **after** `flush_dflash_context(t0)`, ensuring the checkpoint's `dflash_frontier[lane] == position[lane]`, otherwise `restore_lane_dflash` would throw) |
| `cpp:2401-2406`, `:6105-6110` | the two DeviceSnapshot `take` loops skip uncarved planes; the same boundary is automatically supplied by the HostCheckpoint loop (which only looks at `valid[lane]`), so **no new `ReuseSource` is needed**, and the HostCheckpoint branches of `restore_lane_gdn`/`restore_lane_dflash`/`begin_gdn_state`/`begin_dflash_state` are unchanged |
| `cpp:5096-5182`, `:5204` | `session_store_active` instead takes the prompt end-state mirror from the checkpoint with `valid[lane] && position[lane]==entry.prompt_end && prefill_id[lane]==host_checkpoint_live_id` (both state and draft are host→host `memcpy`, **after the stream synchronization**); if none is found and there is no device plane ⇒ `prompt_end_captured=false` ⇒ `entry.host_prompt_end = 0` |

**Why the prompt-end slot goes after all lane slices**: the existing ring layout is wider than `per_lane` when `total <= 3*lanes_` (`cpp:377-403`: divergence/block get a fixed 1 slot each, tail/grid at least 1 each), and in that case a lane's Block slot is **at the same address** as the next lane's Tail slot (the guard at `:5844` lets it through). The default `--host-state-slots 8` at C=4 falls exactly in this range (total=10, per_lane=2, layout 4). Putting the prompt-end slot inside a slice would add another aliasing site, so it goes after all slices (`lanes_ * slots_per_lane + lane`). This is a pre-existing defect, not fixed here, only avoided for the new consumer.

**Semantic consequences**
- The device side no longer has a prompt end-state plane: at C=4, `state` per shard goes 880.9 → 587.2 MiB (only live backing + round scratch remain), shard 0 `draft-snap` 160.0 → 0.0.
- Prompt end-state reuse is instead provided by the host ring: C=1's `scan_lane_reuse` hits HostCheckpoint (trace `src=host`, depth unchanged); C>1's `session_store_active` does a host→host `memcpy` from the checkpoint (previously a device-plane D2H).
- **The cost is pinned host memory**: each prompt-end slot is 73.4 MiB (state) + 160 MiB (draft, allocated as `lanes*lane_image`) ⇒ with `--host-state-slots 32`/C=4, shard 0 pinned goes 7468.8 → 8402.6, shard 1 2348.8 → 2642.6, a total of **+1227 MiB pinned for 748 MiB of device memory**. (The draft buffer's `lanes*lane_image` size is the existing design, while each slot is actually used by only one lane; see "Not covered".)
- Ring disabled (`--no-prefix-reuse` / `--host-state-slots 0`) **fully preserves the old behavior**: everything carved, publishes write device planes, store goes through a device-plane D2H, and `prompt_end_captured` is always true.

**Acceptance** (`tools/win_port/build.ps1` BUILD_EXIT=0, 27 s; gate `test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` 11/11, 70.73 s)

Memory (same binary and parameters, differing only in this change; logs `_temp/cnt1_c4.err`/`_temp/cnt1_c1.err`):

| Item MiB | count=1 (§9.6) | This change | Delta |
|---|---|---|---|
| C=4 shard 0 `state` | 880.9 | 587.2 | −293.7 |
| C=4 shard 0 `draft-snap` | 160.0 | 0.0 | −160.0 |
| C=4 shard 0 `free` | 1180.0 | 1634.0 | +454.0 |
| C=4 shard 1 `state` | 880.9 | 587.2 | −293.7 |
| C=4 shard 1 `free` | 1718.0 | 2012.0 | +294.0 |
| **C=4 device total** | — | — | **+748** |
| C=1 shard 0 `state`/`draft-snap`/`free` | 220.2/40.0/2108.0 | 146.8/0.0/2222.0 | +114 |
| C=1 shard 1 `state`/`free` | 220.2/2394.0 | 146.8/2468.0 | +74 |
| C=4 ring pinned shard 0/1 | 7468.8/2348.8 | 8402.6/2642.6 | +933.8/+293.8 |

⇒ **relative to the count=2 baseline, C=4 frees 1496 MiB across both cards** (shard 0 726.0→1634.0 = +908, shard 1 1424.0→2012.0 = +588); **the C=1→C=4 increment goes 2168 (count=2) → 1604 (§9.6) → 1044 MiB (this change)** (C=1 also frees 372 MiB in the same run: shard 0 +226, shard 1 +146).

Ring layout (`[mem] host-checkpoints`): C=4 `slots 36 (grid 3 + tail 3 + divergence 1 + block 1 + prompt-end 4)`, C=1 `slots 35 (grid 24 + tail 8 + divergence 1 + block 1 + prompt-end 1)`.

Behavior:

| Scenario | Result |
|---|---|
| dual-card artifact chain `_temp/run_cnt1_artifact.ps1` | `tp2_forward_test` exit 0, `duplicated-lane control: lane_gap=0 solo_gap=2.03125 batch_lane0_gap=0 argmax_mismatches=0`, two-lane verify `max_logit_diff=2.03125 argmax_mismatches=1 near_tie_flips=1` word-for-word identical to the baseline; `tp2_load_test` exit 0; the sole `tp2_sessions_test` FAIL is still the plain divergence case, byte-identical to the HEAD baseline. **`TP-2 replayed answer … passed: the turn after a re-rendered answer reused 70 of 58 prompt tokens` is exactly this change's path (prompt end-state reuse)** |
| C=1 rewind band (`_temp/cnt1_rewind.mjs`, `NINFER_TP2_REUSE_TRACE=1`) | `cached=5120` (749 ms), trace `reuse=5120 slot=28 src=host`, `host=5/35 stride=8192`, `prefill_end=5718` ⇒ reuse depth the same as §9.6, source still the host ring |
| C=4 two-round recall (`_temp/s1_diag.mjs`) | `D2-replay cached=5749` (442 ms; 470 ms in §9.6, 511 ms at baseline) |

**Not covered**
- The ring-disabled path (`--no-prefix-reuse` / `--host-state-slots 0`) was not measured end to end, only covered by the gate; under that branch `host_checkpoints` is empty ⇒ `prompt_end_checkpoints` is always null, automatically taking the original device-plane D2H, with `prompt_end_captured` always true.
- The lane slice aliasing defect in the ring layout when `total <= 3*lanes_` remains (hit by the default `--host-state-slots 8` + C=4), not fixed here.
- ~~The draft buffer is allocated per slot as `lanes*lane_image` (160 MiB per slot at C=4) while each slot is used by only one lane ⇒ 120 MiB idle per slot in shard 0~~ (**done**, m02245): each slot's draft half-region becomes a single-lane image + offset 0, see §9.8.
- Rollback: restoring §9.6's device plane makes `first_state_plane`/`first_dflash_plane` always 0 (the PromptEnd slot becomes a redundant copy, with no correctness impact), or simply revert this commit.

### 9.8 Narrowing each slot's draft buffer to a single-lane image (m02245, committed)

**Motivation**: after §9.7 each checkpoint slot serves only one lane (Grid/Tail/Divergence/Block use `lane * host_checkpoint_slots_per_lane_`, PromptEnd uses `lanes_ * host_checkpoint_slots_per_lane_ + lane`), yet the draft half-region is still allocated as `lanes_ * dflash_lane_image` (160 MiB per slot at C=4, while only one 40 MiB image is actually written) ⇒ 120 MiB idle per slot; with `--host-state-slots 32`/C=4, shard 0 wastes `36 x 120 = 4320 MiB` pinned.

**Change points** (11 sites in `src/runtime/engine/tp2_generation_core.cpp` + 1 in `src/runtime/engine/tp2_generation_core.h`)

| Location | Change |
|---|---|
| `cpp:1053` | each slot's draft half-region becomes `std::make_unique<PinnedHostBuffer>(dflash_lane_image, true)`: one compact single-lane image |
| `cpp:5874`, `:5877` | `snapshot_host_checkpoint` gains a guard symmetric with the destination half-region (`checkpoint.dflash_buffer->size() < shard.dflash_round->lane_context_image_bytes()` ⇒ throw `"TP-2 host checkpoint slot is smaller than one lane's draft ring"`), and `copy_context_to_host`'s offset goes to zero |
| `cpp:2499-2501` | the read offset in `restore_lane_dflash`'s HostCheckpoint branch goes to zero |
| `cpp:6251-6254` | the read offset in the walk's `begin_dflash_state` HostCheckpoint branch goes to zero |
| `cpp:5185-5187` | the source offset of the prompt end-state draft memcpy in `session_store_active` goes to zero (the session slab side is still `lane * draft_lane_bytes`) |
| `cpp:5354-5362` | the source offset of the frozen draft memcpy in `session_capture_shared_state` goes to zero (slab side unchanged) |
| `cpp:1089-1090` | the `[mem]` pinned accounting formula `(image_bytes + lanes_ * dflash_image_bytes) * slots` → `(image_bytes + dflash_image_bytes) * slots` |
| `h:188-192` | `HostCheckpoint::dflash_buffer` comment changed to "one compact ring image - for the lane that owns this slot" |
| `cpp:5469` | the env-gated `[tp2-session] entry` trace gains `host_prompt_end=%u` (a permanent diagnostic enhancement, committed together with this change) |
| Unchanged | `store_dflash_image`/`load_dflash_image` at `cpp:1982`/`:1995` (session slab ↔ device ring, the slab still has one image per lane); `restore_lane_dflash`'s DeviceSnapshot branch and the walk's same-named branch (the device snapshot plane is still `lanes_ * lane_image`) |

**Correctness basis**: the same assumption as the destination half-region (`checkpoint.buffer` is already a single-lane `image_bytes`). Under the existing aliasing defect (`total <= 3*lanes_`, hit by the default `--host-state-slots 8` + C=4) the destination half-region is already overwritten by two lanes, and narrowing only makes the draft half-region the same, adding no new class of defect.

**Acceptance** (`tools/win_port/build.ps1` BUILD_EXIT=0, 29 s; gate `test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` 11/11, 70.83 s)

| Item MiB | Before narrowing (§9.7) | This change |
|---|---|---|
| C=4 shard 0 pinned | 8402.6 | **4082.6** (−4320.0 = 36 x 120) |
| C=4 shard 1 pinned | 2642.6 | 2642.6 |
| C=4 per slot | 233.4 | **113.4** (73.4 state + 40 draft) |
| C=1 shard 0/1 pinned | 3969.2/2569.2 | 3969.2/2569.2 (with `lanes_=1` narrowing is the identity transform) |
| C=4 device `free` shard 0/1 | 1634.0/2012.0 | 1634.0/2012.0 (word-for-word unchanged) |

⇒ relative to the count=2 baseline (C=4 pinned 9817.6 = 7468.8 + 2348.8), it is now 6725.2 ⇒ **pinned net down 3092.4 MiB**, while the device side still frees 1496 MiB.

Behavior:

| Scenario | Result |
|---|---|
| dual-card artifact chain `_temp/run_cnt1_artifact.ps1` | `tp2_forward_test`/`tp2_load_test` exit 0 with values word-for-word identical to the baseline; the sole `tp2_sessions_test` FAIL is still the existing plain divergence |
| `NINFER_TEST_ROUTE=dflash2` sessions | `replayed answer … reused 70 of 58 prompt tokens`, `unaligned dialogue` pass; the sole FAIL is the pre-existing `a recalled conversation diverged` token-for-token identical to `worklog §3.10` ⇒ no regression |
| C=1 rewind band (`_temp/cnt1_rewind.mjs`) | `cached=5120` (743 ms), trace word-for-word identical to before narrowing (`reuse=5120 slot=28 src=host`) |
| C=4 two-round recall (`_temp/s1_diag.mjs`) | `D2-replay cached=5749` (477 ms; 442 before narrowing, 470 in §9.6, 511 at baseline ⇒ within noise) |

**Confirming `host_prompt_end`** (§9.7's uncovered item, `NINFER_TP2_SESSION_TRACE=1`): `_temp/cnt1_c4diag.err:30` `[tp2-session] entry 0 tokens=5750 frontier=5749 kv_end=5749 prompt_end=5718 host_prompt_end=5718 shared_end=0 shared=5750 reach=5749 via=frontier resident=0`; `:34` a second `… prompt_end=5772 host_prompt_end=5772 …`. ⇒ At C=4 the prompt end-state mirror really is copied into the session slab by the ring-tail checkpoint via a host→host memcpy (`prompt_end_images && prompt_end_captured` at `cpp:5203` is true).

**lane>0 coverage** (probe `_temp/m02245_lane1.mjs` + `_temp/run_m02245_lane1.ps1`, C=4 + dflash2 + 2 concurrent per round, log `_temp/m02245_lane1.err`):
- The write/store path is covered: all three rounds are `[tp2-lane] batch=2 capacity=4 path=batched` (`:28`, `:40`, `:55`), and the two members of the same batch necessarily occupy two different lanes; both entries' `host_prompt_end` are non-zero (`:41-42`, `:45-46`) ⇒ both lanes' `snapshot_host_checkpoint(PromptEnd)` and `session_store_active`'s state/draft host→host memcpy (draft source offset 0, `cpp:5185-5187`) were executed.
- The read side is unreachable at lane>0: the HostCheckpoint branches of `restore_lane_dflash`/`restore_lane_gdn` are entered only by `scan_lane_reuse` (`cpp:2359`, requiring `lane_state.cached_state_valid`) and the walk path; `cached_state_valid` is only set true by the prefill publish (`:2517`), while `release_lane_kv` clears it at every retirement when `lanes_ > 1` (`:2700`, with the early return at `:2684`) ⇒ at C=4 that branch is unreachable (the same fact as §9.6's "this plane is a dead allocation at C=4"), and the walk path is reachable only when `lanes_ == 1` ⇒ read-side narrowing is an identity transform at lane 0 and unreachable at lane>0, with no uncovered correctness risk.
- The probe's three rounds of `cached=0` are unrelated to this change; they are `session_recall`'s candidacy rule (`cpp:5457-5458`: `offered != 0 && offered <= shared && offered < prompt_tokens.size() && offered <= host_kv_end`): an exactly identical prompt yields `shared == prompt_end == prompt_tokens.size()` (`:45` entry 0 `shared=5718`), which is rejected by `offered >= prompt_tokens.size()` as "at least one token must be left to forward"; a re-rendered assistant turn's prompt diverges one position before the prompt's last token (`shared=5717 < prompt_end=5718`, B side `shared=5323 < 5346`) ⇒ `offered > shared`. ⇒ **A prompt rendered by the server-side chat template cannot land in the band `[prompt_end, frontier)`**; `via=prompt-end` can only be triggered by raw token input that verbatim reproduces the engine prompt's tail (as in the engine-level case `check_replayed_answer_keeps_prompt_end`), which also explains why C=4 recall in §9.7 always reports `via=frontier`; that band is equally unreachable before and after this change, so behavior is unchanged.

**Not covered / rollback**: the ring-disabled path is still covered only by the gate (§9.7); rollback = change `cpp:1053` back to `lanes_ * dflash_lane_image` and restore the 4 read offsets to `lane * lane_context_image_bytes()` (synchronize the accounting at `cpp:1089-1090`).

---

## 10. Open questions and known limitations
- ~~The `--lane-context` default~~ (settled, m01348): default `0` = whole pool, i.e. S1's "a first-come large request may monopolize the whole pool"; when narrowing by policy is needed the operator gives a value explicitly. The default will not change again.
- ~~400 message with lane capacity arithmetic + `prompt + max_output` over-budget warning~~ (**done**, m01348 item 3): see §9.5; the core message splits into "policy-bound / pool-bound" branches, the frontend preparation-stage message gains `prompt (at least) N tokens`, and `[tp2-capacity]` is printed when the output budget is clamped by the ceiling.
- ~~The TP-2 exception note at `docs/maintainer/engine-architecture.md:41`~~ (**done**, m01348 item 3): see `docs/maintainer/engine-architecture.md:45-51`.
- Quantifying fragmentation (the largest request that fits under different sequences of request sizes) has not been measured yet.
- **Known cost (introduced by S1)**: every request's terminal state goes through `retire_lane_session` → `session_store_active`, mirroring that lane's frontier KV + state D2H into the host slab (about 73.4 MiB/shard at full context), and the next round's reuse copies it back to new pages via `session_recall` → `session_restore`. This is the direct consequence of "pages can change owner immediately"; the throughput cost for multi-round sessions remains to be quantified (R4).
- **Known cost (count=1, committed `371eded9`)**: the device side no longer keeps a rewind boundary (`kReuseSnapshotCount = 1`); reuse at C=1 that lands "within the previous round's prompt end chunk" instead gets its boundary from the host checkpoint ring's tail sub-ring, with restore over H2D (73.4 + 40 MiB/lane); the measured reuse depth is unchanged. C=4 is unaffected (that plane was already a dead allocation). Rollback: change the constant at `src/runtime/engine/tp2_generation_core.h:64` back to 2. See §9.6 for details.
- **Known cost (prompt end-state plane moved to the host ring, committed `87f1526d`, m01955)**: the device side no longer keeps a resident prompt end-state plane (at C=4 −293.7 MiB `state` per shard, plus −160.0 MiB `draft-snap` for shard 0); it is instead carried by one non-rotating `PromptEnd` slot per lane at the ring tail; the cost is +1227 MiB pinned host (`--host-state-slots 32`/C=4), already offset by the draft half-region narrowing in §9.8 (net −3092.4 MiB relative to the count=2 baseline). Both C=1's prompt end-state reuse and C>1's session slab mirror now take it from a checkpoint. Rollback: `first_state_plane`/`first_dflash_plane` always 0. See §9.7 for details.
- **Known cost (draft buffer narrowed to a single-lane image, m02245, committed)**: each slot's draft half-region shrinks from `lanes_ * dflash_lane_image` to one single-lane image, C=4 shard 0 pinned −4320 MiB (net −3092.4 MiB relative to the count=2 baseline), device memory unchanged; the read-side path at lane>0 is already unreachable at C=4, and the write/store side is covered by the two-lane concurrency probe. See §9.8 for details.
- **Still unverified**: the single-card `--devices 0` half (the TP-2 artifact is a dual-shard layout, so it does not apply); quantification of fragmentation (the largest request that fits under different sequences of request sizes); R4's multi-round session throughput cost; the vision prompt rewind reuse case.
