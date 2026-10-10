# PLAN: TP-2 multi-concurrency review remediation plan

> **This document merges two independent sets of conclusions into one remediation plan**: ① a **read-only source review** of the TP-2 multi-concurrency changes (session `session-a6dd89fc-759c-435e-8759-2024e33f22c7`, review target = the working tree's diff relative to baseline `d1a9b57b`); ② this session's reproduction and diagnosis of the **user's on-machine log** (the user reported "the second session keeps queueing, then gets interrupted after a while").
>
> - **Re-check baseline**: HEAD `bc3c8a33` (branch `feat/tp2-concurrency`). Every conclusion in the review report was re-located and re-checked on that HEAD, with the result recorded in each item's "HEAD re-check" column; line numbers follow that HEAD.
> - **Numbering convention**: `A*` = review-confirmed defect; `B*` = review risk/fragility point; `C*` = concurrency behavior defect measured in this session; `P0–P3` = remediation phase. The review report's `D1–D5`/`R1–R6` correspond one-to-one with this document's `A/B` (see each item's "review id"). **Note: this document's A/B/C are not the same numbering set as the product decisions `D1–D6` in section 0 of the adjacent plan `tp2-plan-concurrency.md`.**
> - **Adjacent authorities**: `tp2-plan-concurrency.md` (Phase 0 decisions, Phase 1/2 construction drawings, and the item-by-item remediation record in section 12), `docs/serving.md` (external contract), `docs/tp2-decisions.md` (settled).
> - **Scope**: this document only writes "what to fix, why, and how to verify"; it does not repeat the architectural decisions and construction details already frozen in the existing plan.

---

## 0. Conclusions at a glance

The two sets of conclusions cover **two different lines**: the review line is **correctness and robustness** (exception escape from the batch formation section, the unpublished region of the per-lane KV block table, dead code and contradictory documentation), while the measurement line is **concurrency behavior** (no admission of new requests while a batch is running, responses held hostage by the whole queue, requests with tools not concurrent at all). The two lines have **exactly one direct overlap**: review `A1` (a grammar member drags the whole batch back to serial) = measured `C3` (requests with tools are always serial on TP-2); the two are two statements of the same root cause. Separately, `B5` (lane = position within the batch rather than conversation binding) and `B4` (whole-batch failure / whole-catalog invalidation) **constrain** the landing of dynamic admission (`C1`), which is indirect coupling.

**Recommended remediation order**: `A2` (permanent deadlock) → `C2` (response withholding → 499) → `A3` (KV backing assertion) → `A4/A5` + documentation (`B2/B5`) → `C3/A1` (grammar concurrency, the real fix for the user's scenario) → `C1` (dynamic admission) → `B3/B6/B1/B4` cleanup.

| Id | Severity | One-liner | Review id | Relation to the concurrency remediation |
|---|---|---|---|---|
| A1 | Medium | Any grammar member in a batch drags the **whole batch** back to serial; documentation understates the blast radius | D1 | **Directly overlaps** C3; two fix approaches, see section 5 |
| A2 | Medium | Code outside the `try` in `drive_lane_queue` throws ⇒ the TP-2 route **deadlocks permanently** | D2 | New (not covered by this plan) |
| A3 | Medium/Low | The per-lane KV block table publishes only `per_lane` entries; the rest is uninitialized device memory, and an out-of-range access silently corrupts | D3 | New |
| A4 | Low | The "collapses to 1 lane" warning for `--spec dflash` is printed first, then startup immediately throws; comment and documentation contradict | D4 | New |
| A5 | Low | The `refusal = "route"` branch is unreachable (dead code) | D5 | New |
| B1 | Risk | The service capacity invariant (headroom exactly 1) is undocumented; when the pool is full it closes the connection instead of returning HTTP 429 | R1 | New |
| B2 | Risk | With 4 lanes the host checkpoint grid becomes 5.3× coarser (not a correctness defect) | R2 | New |
| B3 | Risk | The multi-lane route still builds `mtp_chain_graphs_` and 12 AR channels, pure waste with misleading logs | R3 | New |
| B4 | Risk | Whole-batch failure + `session_invalidate_all` has a blast radius of the **entire catalog** | R4 | Coupled to C1's redefinition of failure semantics |
| B5 | Risk | `docs/serving.md` says "each lane independently recalls its own conversation", but it is actually "lane = position within the batch" | R5 | **Constrains** C1 (lane ownership becomes less stable after dynamic admission) |
| B6 | Risk | `build.ps1`'s `Repair-MsvcDepsPrefix` is out of scope and touches every C++ source | R6 | New |
| C1 | High | A batch is only formed at batch start and returns only after all lanes finish ⇒ new requests during a run can only wait for the next batch | — | This plan's P2.2 (new) |
| C2 | High | Responses are published per **whole batch**, and the driver thread is held by the entire queue ⇒ an earlier-finished request is dragged to 499 | — | This plan's P0.2 (new) |
| C3 | High | Requests with tools are **not concurrent at all** on TP-2 (grammar has no per-lane mask) | = A1 | This plan's P2.1 (overlap) |

---

## 1. Sources and re-check method

| Source | Content | Authority/nature |
|---|---|---|
| Review session `session-a6dd89fc-759c-435e-8759-2024e33f22c7` | Read-only source review report: `D1–D5` confirmed defects, `R1–R6` risks, a "verified not a defect" list, and a statement of verification limits | Read-only, no files modified |
| This session (user's machine) | Interpretation of the user log `C:/ninfer/serve-win.log`; reproduction with `build-win/apps/ninfer-serve.exe` using the same arguments (port 8099, `NINFER_TP2_LANE_TRACE=1`); source localization | Measured |

Re-check method: for every conclusion in the review report, re-locate it on HEAD `bc3c8a33` with read/grep to line numbers and confirm the code shape; for `A2`/`A3` the trigger mechanism itself was additionally checked (`PreparedPromptAccess::view` does throw; `DeviceArena` does use `cudaMalloc` and does not zero). Wherever a re-check differs from the review report, that is stated explicitly in the item.

---

## 2. Review line: confirmed defects (A1–A5)

### A1 (Medium) Any grammar member in a batch drags the **whole batch** back to serial — the documentation understates the blast radius

**Review id**: D1　**HEAD re-check**: present (line numbers agree)

Evidence: `src/runtime/engine/tp2_generation_core.cpp:2089-2100`

```cpp
bool batchable         = lanes_ > 1;
const char* refusal    = batchable ? nullptr : "route";
if (batchable) {
    for (const auto& member : batch) {
        const auto& data = qwen::PreparedPromptAccess::view(member->request->prompt);
        if (data.tool_call_output != nullptr) {
            batchable = false;
            refusal   = "grammar";
            break;
        }
    }
}
```

`batchable` is a **whole-batch** switch, not a per-member switch. Once hit, `src/runtime/engine/tp2_generation_core.cpp:2119-2126` hands **every** member of the batch to `execute_lane`:

```cpp
} else {
    // P1.4a runs one lane of the batch at a time; ...
    for (std::size_t lane = 0; lane < batch.size(); ++lane) {
        batch[lane]->result =
            execute_lane(*batch[lane], static_cast<std::uint32_t>(lane));
    }
}
```

And `execute_lane` (`src/runtime/engine/tp2_generation_core.cpp:2164-2172`) runs the full single-request walk (`execute` → `execute_walk`). Therefore, under `--max-concurrency 4`, a single tool-grammar request entering the batch drags the other 3 ordinary requests from 2.80× down to 1× as well.

`docs/serving.md:79-80` phrases it as "only a tool-grammar request runs lane by lane instead of as one round", describing the blast radius as one request.

**Why it is a real problem**: grammar requests are the norm rather than an edge case in agent tool loops; the comment at `src/runtime/engine/tp2_generation_core.cpp:2087-2088` (the batched decode step has no per-lane mask, so grammar cannot enter a batched round) is itself correct, and the defect lies only in the granularity of "either the whole batch or fully serial".

**Recommendations**:
1. Short term (stop the bleeding): **pull the grammar members out** of the batch and run them serially on their own, while the remaining members proceed through `execute_plain_batch`/`execute_spec_batch` as usual. Note that the order of `batch` is the lane position (see B5), so after pulling members out the slots of the remaining members must be re-arranged.
2. Target state: implement a per-lane grammar mask in the batched round (see C3 / P2.1), after which the pull-out approach retires.
3. Documentation: change `docs/serving.md:79-80` to "a batch containing a tool-grammar request runs every member lane by lane".

### A2 (Medium) Code outside the `try` in `drive_lane_queue` throws ⇒ the TP-2 route **deadlocks permanently**

**Review id**: D2　**HEAD re-check**: present (line numbers agree)

Evidence: `src/runtime/engine/tp2_generation_core.cpp:2063-2110`

```cpp
void TP2GenerationCore::drive_lane_queue() {
    for (;;) {
        std::vector<std::shared_ptr<PendingRequest>> batch;
        batch.reserve(lanes_);                                    // :2066  may throw std::length_error
        {
            std::unique_lock<std::mutex> queue(lane_queue_mutex_);
            ...
            while (batch.size() < lanes_ && !lane_queue_.empty()) {
                batch.push_back(std::move(lane_queue_.front()));  // :2074  may throw std::bad_alloc
                lane_queue_.pop_front();
            }
            if (batch.empty()) {
                lane_driver_active_ = false;                      // :2078  the only normal reset point
                lane_queue_cv_.notify_all();
                return;
            }
        }
        bool batchable      = lanes_ > 1;
        const char* refusal = batchable ? nullptr : "route";
        if (batchable) {
            for (const auto& member : batch) {
                const auto& data = qwen::PreparedPromptAccess::view(member->request->prompt);  // :2093  may throw
                ...
            }
        }
        ...
        try {                                                     // :2110  ← the only protection starts here
```

`PreparedPromptAccess::view` really does throw: `src/models/qwen3_5/frontend/frontend.cpp:732-735`

```cpp
const PreparedPromptData& PreparedPromptAccess::view(const PreparedPrompt& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    return *prompt.data_;
}
```

State when the exception escapes `drive_lane_queue` before `:2110`:

- `lane_driver_active_` was already set to `true` at `src/runtime/engine/tp2_generation_core.cpp:2039`, and nobody resets it;
- the batch members were already removed from `lane_queue_` by `:2074-2075`, and `member->complete` is `false` for all of them;
- `wait_lanes`'s wait has **no timeout** (`src/runtime/engine/tp2_generation_core.cpp:2035`);
- subsequent requests only take the `lane_driver_active_ == true` waiting branch (`:2034-2041`), so **nobody will ever become the driver again**.

Result: the driver itself receives the exception, the remaining members of the batch and every subsequently queued request **block forever**, and the entire TP-2 route can only be recovered by restarting the process. By contrast, `catch (...)` (`src/runtime/engine/tp2_generation_core.cpp:2127-2146`) already correctly turns "batch failure" into "all members failed"; the problem is only that its coverage is missing one segment.

**Recommendations** (approach 1 preferred):
1. Move the entire segment `:2065-2109` into the `try` (i.e. move the start of `try` before `batch.reserve`), so that a batch formation failure also goes through D4's whole-batch failure path;
2. Or add a scope guard guaranteeing that any exit path resets `lane_driver_active_ = false`, calls `notify_all()`, and uniformly sets unfinished members to `failure`;
3. It is also recommended to give `wait_lanes`'s wait an upper bound sourced from the same place as `--pending-timeout-ms`, as a last resort against this class of invariant breakage.

### A3 (Medium/Low) The per-lane KV block table publishes only `per_lane` entries; unpublished entries are **uninitialized device memory**

**Review id**: D3　**HEAD re-check**: present; publish semantics and the non-zeroed arena were additionally checked

Evidence: `src/runtime/engine/tp2_generation_core.cpp:1034-1064` (text KV) and `:1085-1125` (MTP KV)

```cpp
const std::uint32_t pages    = pages_for_tokens(capacity);
const std::uint32_t per_lane = pages / static_cast<std::uint32_t>(lanes);
...
const std::uint32_t count = (lane + 1 == lanes) ? (pages - assigned) : per_lane;
...
shard.kv_rows.push_back(tables.acquire(lane));
tables.publish(shard.kv_rows.back().handle(), 0, handles, shard.device.stream);
```

`handles` has only `per_lane` entries (the last lane gets the remainder), but that execution row's logical capacity is still the **entire** `max_context`: `src/models/qwen3_5/state/decoder_state.cpp:25-28,50-52` (`logical_pages = page_count(capacity)`, and the pool-level check only requires `physical_page_groups >= logical_pages`).

`publish` only validates the upper bound and does not require a full fill (`src/core/paged_kv_cache.cpp:774-780`), `publish_indices` only `cudaMemcpyAsync`s the published segment (`src/core/paged_kv_cache.cpp:824-833`), and `DeviceArena` allocates with `cudaMalloc` and **does not zero** (`src/core/arena.cu:141-155`). Hence those entries in the device-side block table are **garbage physical page numbers**.

It does not trigger today because three host-side arithmetic checks block it: `src/runtime/engine/tp2_generation_core.cpp:1964-1967` rejects prompts exceeding the window, `src/runtime/engine/tp2_generation_core.h:441-445`'s `lane_context_window()` pins the window at `per_lane*64 - margin`, and the spec path clamps `capacity_left` again every round (`src/runtime/engine/tp2_generation_core.cpp:3444-3449`). All three have **no backing validation**, and they are not named under one shared invariant: once someone relaxes the window (dynamic quotas, reusing another lane's idle pages), the consequence is not an assertion failure but a **silent write to someone else's physical page or outside the pool**. At baseline (lanes=1) `per_lane == pages`, the row is fully filled ⇒ this is a fragility **newly introduced** by this change.

**Recommendations**:
1. Add a startup assertion in `build_shard` that hard-codes the invariant, e.g. `if (per_lane * kPagedKVPageSize < lane_context_window() + 1 + (mtp_enabled_ ? mtp_drafts_ : 0)) throw std::logic_error("TP-2 lane context window exceeds the pages published to its execution row");`
2. Or have `KVExecutionTablePool` record the "published range" so that device-side queries have something to check against;
3. At minimum, add cross-referencing comments at `src/runtime/engine/tp2_generation_core.cpp:1051` and `src/runtime/engine/tp2_generation_core.h:441-445` stating that "window = published pages × 64 − margin" is a hard binding.

### A4 (Low) The "collapse to 1 lane" warning for `--spec dflash` contradicts the implementation/documentation

**Review id**: D4　**HEAD re-check**: present

Evidence: `src/runtime/engine/model_instance.cpp:99-113` prints

```cpp
"[tp2-lane] --max-concurrency %u collapses to 1 lane on the %s route (see tp2-plan-concurrency.md)"
```

but the same function `src/runtime/engine/model_instance.cpp:135-138` immediately terminates startup:

```cpp
if (options.speculative.backend == SpeculativeBackend::DFlash) {
    throw std::invalid_argument(
        "TP-2 generation supports --spec mtp and --spec dflash2 (not --spec dflash)");
}
```

`include/ninfer/tp2_capacity.h`'s `tp2_generation_concurrency` returns 1 only for `DFlash`, so that fprintf is a **dead path**; the comment at `src/runtime/engine/model_instance.cpp:99-103` ("rather than failing startup") and `docs/serving.md:75-76` ("still collapses ... with a warning") both contradict the implementation.

**Recommendations**: delete the fprintf at `src/runtime/engine/model_instance.cpp:107-113` and rewrite the comment as "DFlash v1 is rejected below; this constant returning 1 is only for historical reasons"; change `docs/serving.md:75-76` to "`--spec dflash` is rejected at construction on this route".

### A5 (Low) The `refusal = "route"` branch is unreachable

**Review id**: D5　**HEAD re-check**: present

Evidence: `bool batchable = lanes_ > 1;` at `src/runtime/engine/tp2_generation_core.cpp:2089-2090` is always `true`: the only call site of `drive_lane_queue` is `src/runtime/engine/tp2_generation_core.cpp:2047` (inside `wait_lanes`), and the only call site of `wait_lanes` is `src/runtime/engine/tp2_generation_core.cpp:2000-2001`, guarded by `if (owner_->lanes_ > 1)`.

**Recommendations**: initialize `batchable` directly to `true` (or delete the route branch), keeping only the grammar criterion; sync the comment at `src/runtime/engine/tp2_generation_core.cpp:2083-2084`.

---

## 3. Review line: risks and fragile points (B1–B6)

### B1 The service capacity invariant is undocumented, and it "drops the connection" instead of returning HTTP 429

**Review id**: R1　**HEAD re-check**: present

`src/serve/http_server.cpp:223-229` configures the thread pool with `worker_count = C+P+1` and a queue limit of `C+P`. The httplib customized in this repository makes `enqueue` **return false** when the queue is full (`third_party/cpp-httplib/httplib.h:11704-11723`), and `Server::listen_internal` accepts and enqueues in the **thread that called listen**; on failure it merely records `Error::ResourceExhaustion` and closes the socket (`third_party/cpp-httplib/httplib.h:13652-13737`) ⇒ **the client sees the connection closed and never gets an HTTP 429**.

This path is not reached today only because `src/serve/generation_service.cpp:276-298`'s `acquire_request_lifetime` immediately throws `RequestError(RequestErrorKind::Overloaded, "inference request queue is full")` when `active >= maximum`, keeping in-flight requests ≤ C+P, while the pool capacity is (C+P+1) workers + (C+P) queue slots, giving headroom of exactly 1.

**Recommendations**: add a comment above `src/serve/http_server.cpp:223` pinning the two premises (the accept thread does not occupy a worker; admission failure does not block the thread), and state that "headroom = 1 is deliberate".

### B2 The checkpoint grid becomes 5.3× coarser with 4 lanes

**Review id**: R2　**HEAD re-check**: present (not a defect)

After `src/runtime/engine/tp2_generation_core.cpp:389-397` evenly splits the host checkpoint ring by lane, the number of `grid` slots decreases: default `host_state_slots` + lanes=4 ⇒ `per_lane=8, fixed=2, flexible=6, tail=3, used=5, grid=3`; with `--max-context 131072`, `per_slot = ceil(131072/3) = 43691` ⇒ `host_checkpoint_stride_ = max(8192, round128(43691)) = 43776` (it is 8192 at lanes=1).

It is **not** an "over budget" defect: `lanes_ * floor(total/lanes_) <= total` always holds, the comment at `:380-383` is conservative, and at most `lanes-1` slots are lost relative to the budget. The actual effect is only that grid-style prefix reuse becomes coarser with 4 lanes (short prompts are unaffected, and the tail ring still covers the last 8192 tokens).

**Recommendations**: add a sentence to the retention section of `docs/serving.md`: "with multiple lanes the grid checkpoint spacing becomes coarser in proportion to the lane count".

### B3 `mtp_chain_graphs_` / `mtp_chain_host_` / 12 AR channels are pure waste on the multi-lane route

**Review id**: R3　**HEAD re-check**: present

`src/runtime/engine/tp2_generation_core.cpp:560-569` builds `mtp_chain_graphs_` (12 profiles) whenever `mtp_enabled_`, and `:664-666` reserves 12 AR channels for it. The only user, `mtp_propose_window` (`src/runtime/engine/tp2_generation_core.cpp:1799`, graph selection at `:1812`), is called only at `src/runtime/engine/tp2_generation_core.cpp:6393`, which is inside `execute_walk` (the single-lane serial walk); the multi-lane route goes through `execute_spec_batch`, where MTP chain steps call `ctx_a.mtp_forward_decode_batch` + `ctx_a.mtp_propose_batch` directly and **never enter** `mtp_propose_window`. Meanwhile the startup log (`src/runtime/engine/tp2_generation_core.cpp:650-656`) still prints "mtp chain: eager(bucket)" with multiple lanes, misleading operators.

**Recommendations**: build these two resources (and the AR channel reservation) only when `lanes_ == 1U && mtp_enabled_`, and show `n/a` in the log field with multiple lanes.

### B4 Whole-batch failure + `session_invalidate_all` has a blast radius of the **entire catalog**

**Review id**: R4　**HEAD re-check**: present

The `catch (...)` at `src/runtime/engine/tp2_generation_core.cpp:2127-2146` sets the whole batch to the same failure per D4 and calls `session_invalidate_all()`; `abort_if_ar_stalled` (`src/runtime/engine/tp2_generation_core.cpp:3954-3973`) likewise does `invalidate_host_checkpoints(); session_invalidate_all();` before throwing `RequestError(Unavailable, ...)`. That is, a single allreduce rendezvous failure invalidates the reuse state of **all lanes** (`session_invalidate_all` is defined at `src/runtime/engine/tp2_generation_core.cpp:4107-4110`).

**Recommendations**: confirm whether this is intentional; if so, state explicitly in `docs/serving.md` that "one AR stall discards all reuse state". Decide together with P2.2 (dynamic admission)'s redefinition of failure semantics.

### B5 `docs/serving.md`'s "each lane independently recalls its own conversation" does not match the actual binding

**Review id**: R5　**HEAD re-check**: present (not a correctness defect)

`docs/serving.md:77-79` states that the checkpoint ring and the session catalog are sliced by lane, "so each lane recalls its own previous conversation independently". The actual binding is "lane = position within the batch": the comment at `src/runtime/engine/tp2_generation_core.cpp:2494-2495` states that the slot is simultaneously the position within the batch, the KV execution row, and the GDN state slot, and `:2515` has `lanes[index].slot = static_cast<std::uint32_t>(index);`.

**Confirmed not a correctness defect**: `session_recall` can restore the host-side session onto any lane, and the skip condition at `src/runtime/engine/tp2_generation_core.cpp:4699` (`entry.device_lane >= 0`) ensures two lanes in the same batch do not steal each other's live session; a checkpoint's `valid[lane]` is only set on the lane that wrote it (`publish_lane_prefill` `src/runtime/engine/tp2_generation_core.cpp:2406-2412`). The problem is only that the documentation implies a stable "conversation ↔ lane" binding that does not exist, so the device-state reuse hit rate varies with arrival order.

**Recommendations**: change it to "the checkpoint ring and the session catalog are sliced by lane, so device-resident reuse depends on which lane a request lands on; a recall onto a different lane falls back to the host-resident session". **This is a prerequisite for P2.2**: dynamic admission makes lane ownership even less stable.

### B6 The change to `tools/win_port/build.ps1` is out of scope for this task and forces a full rebuild

**Review id**: R6　**HEAD re-check**: committed together with `6a17bb2c`

The change adds `Repair-MsvcDepsPrefix($BuildDir)`; the fix itself is necessary and correct (the original `rules.ninja`'s `msvc_deps_prefix` was mojibake produced by CMake misdecoding cl's Chinese messages as CP936, which made all ninja dependency records `#deps 0` and stopped header changes from triggering rebuilds). But it is unrelated to TP-2 multi-concurrency, and "touch every C++ source" makes every build trigger a full rebuild.

**Recommendations**: keep the fix (dependency tracking is restored), but restrict the touch to when "a prefix mismatch is detected"; if that is already the behavior, state it in a comment.

---

## 4. Measurement line: concurrency behavior defects (C1–C3)

Reproduction environment: `build-win/apps/ninfer-serve.exe`, port 8099, same arguments as the user's script (`--devices 0,1 --max-concurrency 4 --max-context 131072 --spec dflash2`), `NINFER_TP2_LANE_TRACE=1`. User-side log: `C:/ninfer/serve-win.log`.

### C1 (High) No admission of new requests while a batch is running: batches form only at batch start, and a batch returns only after all lanes finish

**Evidence (code)**: each iteration of the `for(;;)` at `src/runtime/engine/tp2_generation_core.cpp:2063-2158` forms exactly one batch and runs it to completion; `execute_plain_batch` internally runs `while (!active.empty())` (`src/runtime/engine/tp2_generation_core.cpp:2843`), `execute_spec_batch` likewise (`src/runtime/engine/tp2_generation_core.cpp:3408`); the serial route runs each member's entire walk in one go (`:2119-2126`). The batch formation window at `src/runtime/engine/tp2_generation_core.cpp:2058` `constexpr std::chrono::microseconds kBatchFormationWindow{3000};` is only 3 ms ⇒ only requests arriving at "the same instant" can share a batch. `:2105-2109` prints one `[tp2-lane] batch=%zu capacity=%u path=%s%s` per formation, and the log shows only one line per formation, which is direct evidence of "run the whole batch to completion".

**Evidence (measured)**:
- No tools: req#8 (1024 tokens, engine 7.5 s) arrives 11:23:57.861 → `batch=1 capacity=4 path=batched`; req#9 (8 tokens, 1.5 s later) arrives 11:23:59.373 → **another** `batch=1 capacity=4 path=batched`, and does not start until 11:24:05.34: **engine time 127 ms, wall clock 6.1 s**.
- User log: req#2 (tools, 15,163 prompt) monopolizes 11:11:29.4→11:15:52.05 (4m23.4s); req#3/req#4 arrive at 11:11:33 but do not start until 11:15:52.05 (req#4 total 7.9 s).

**Impact**: the second session's TTFT is determined by the previous session's entire generation time — exactly the user's reported "the second session keeps queueing".

**Recommendations**: P2.2 dynamic admission.

### C2 (High) Responses are published per **whole batch**, and the driver thread is held by the entire queue

**Evidence (code)**:
- Publish granularity: `src/runtime/engine/tp2_generation_core.cpp:2147-2156` only does `for (auto& member : batch) member->complete = true; lane_queue_cv_.notify_all();` after the entire batch finishes. Waiters are woken only at the end of each batch ⇒ an earlier-finished lane is held back by the slowest lane in the same batch.
- Driver hostage: in `wait_lanes` (`src/runtime/engine/tp2_generation_core.cpp:2020-2051`) a thread that cannot find a driver **becomes the driver itself** and holds `execution_mutex_` to call `drive_lane_queue` (`:2046-2047`), while `drive_lane_queue` returns only when the **queue is drained** (`:2151-2155`) ⇒ the driver's own request must wait for the whole queue to drain before it gets its result, even if it finished long ago.

**Evidence (measured)**:
- 3×tools arriving simultaneously → `batch=3 capacity=4 path=serialgrammar`: req#5 total 1.4 s, req#6 434 ms, req#7 435 ms, and **all three are only done at 11:22:53.242**.
- User log: req#1 (no tools, 200 prompt/64 out) finishes its entire walk at 11:11:27.86→29.36 (throughput prefill 200 tok + decode 63 tok; `total 773 ms`), but `req#1 done` is recorded at **11:15:59.950**, accompanied by `req#1 response failed during transport | HTTP 499 | client disconnected`.

**Impact**: an earlier-finished request is held for 4m31s ⇒ the client times out/disconnects (HTTP 499); even within the same batch, a fast lane waits for a slow lane. This is a **different** defect from A2: A2 is the exception path jamming the route, C2 is the normal path suppressing responses.

**Recommendations**: P0.2 (decouple driver from request threads + publish on completion).

### C3 (High) Requests with tools are not concurrent at all on TP-2 (= A1's measured face)

**Evidence (code)**: the grammar mask exists only in the serial walk: `src/runtime/engine/tp2_generation_core.cpp:5224-5235` (`tool_constraint` construction), `:5546-5564` (`tool_mask_dev`/`tool_mask_columns`), `:5931-5936` (prefill first-token mask), `:6300-6329` (serial DFlash2 window mask), `:6478-6501` (serial MTP window mask); the batched routes have **none at all**. Any member with tools ⇒ `:2089-2100` sets the whole batch `refusal="grammar"` ⇒ `:2119-2126` runs each member's entire walk.

**Evidence (measured)**:
- tools: req#10 (1024 tokens) 11:24:07.498 → `batch=1 path=serialgrammar`; req#11 (8 tokens, 1.5 s later) 11:24:08.995 → yet another `batch=1 path=serialgrammar`, running only at 11:24:15.989: **engine 337 ms, wall clock 7.0 s**.
- 3×tools in one batch → the whole batch is serial and published as a whole (same measurement as C2).

**Impact**: every dsh request carries tools ⇒ TP-2's concurrency is **always 1**, and `--max-concurrency 4` is completely ineffective for that workload.

**Recommendations**: P2.1 (per-lane grammar mask for batched rounds).

---

## 5. Overlap analysis (the question this document must answer)

| Remediation item in this document | Review conclusion | Relation |
|---|---|---|
| **P2.1 (C3)** | **A1** | **Direct overlap**: same root cause (batched rounds have no per-lane grammar mask). The review describes its **blast radius** from a "defect" viewpoint (one grammar member drags down the whole batch); the measurement describes its **result** from a "behavior" viewpoint (requests with tools are always serial). |
| P2.2 (C1) | none | Complementary. The review report does not cover "no admission of new requests while a batch is running"; the header comment at `src/runtime/engine/tp2_generation_core.cpp:2957-2974` even explicitly says continuous batching is out of scope. |
| P0.2 (C2) | none | Complementary. The review report does not cover "on the normal path responses are published per whole batch / the driver is held by the queue"; `A2` is an adjacent issue on the **exception** path; the two fixes can be merged into the same code segment, but their criteria differ. |
| — | A2/A3/A4/A5/B1/B2/B3/B4/B6 | New items not covered by this remediation plan. |
| Prerequisite for P2.2 | B5 | **Indirect coupling**: once `C1` lands, lane ownership becomes even less stable ⇒ B5's documentation wording must be changed first. |
| Semantic decision for P2.2 | B4 | **Indirect coupling**: dynamic admission changes the batch's member composition, so D4's "whole-batch failure" and the blast radius of `session_invalidate_all` must be redefined together. |

### 5.1 Comparison of the two fixes for A1 and C3

| Approach | Method | Change surface | For the dsh scenario (every request carries tools) | For a mixed batch (1 grammar + N ordinary) |
|---|---|---|---|---|
| Review recommendation (pull-out) | Pull the grammar members out of the batch and run them serially, keeping the rest batched | Small (change `drive_lane_queue`'s dispatch and slot re-arrangement) | **Ineffective**: all members are grammar ⇒ still fully serial | Effective: ordinary members return to 2.80× |
| This plan's P2.1 (per-lane mask) | Assemble a `[vocab, width*lanes]` mask in the batched round, with lane l occupying the column block at `width*l`; advance each lane's constraint state by its accepted tokens; the prefill first token must also be masked | Medium-large (batched round + prefill first token + per-lane constraint objects) | **Effective** | Effective |

Conclusion: the two do not conflict — the pull-out approach can serve as a **short-term stopgap** (P1.4), and the per-lane mask is the **target state** (P2.1). The pull-out approach alone cannot solve the user's current scenario.

---

## 6. Unified remediation plan

### P0 Stopgap (small changes, independently committable)

**P0.1 = A2 (permanent deadlock)**
- Change: move `src/runtime/engine/tp2_generation_core.cpp:2065-2109` into the `try` (or a scope guard); add a timeout fallback to the `wait_lanes` wait.
- Acceptance: construct an admission path where `view` throws (a unit test injecting a prepared prompt with `PreparedPromptData == nullptr` directly, or unit-level injection into the batch formation segment), and confirm the route does not hang and that both the batch members and subsequent requests receive failure results.

**P0.2 = C2 (response withholding → 499)**
- Change:
  1. **Decouple the driver**: change `lane_driver_active_` + `drive_lane_queue` from "the request thread appoints itself driver" to a driver thread owned by the core (started in the constructor, stopped in the destructor), or keep the leader but hand over after each batch if the driver's own request has already completed (`src/runtime/engine/tp2_generation_core.cpp:2020-2051`, `:2147-2156`).
  2. **Publish on completion**: change "uniform `complete = true` after the whole batch finishes" to "publish each member as soon as it completes". The batched routes decide at **round boundaries**: after a round ends, move finished members out of active and publish immediately (plain's `active.swap(live)` and spec's `still` collection already have equivalent compaction logic that can be reused directly); the serial route publishes as soon as each lane finishes.
  3. Add a per-lane publish field to `NINFER_TP2_LANE_TRACE` for regression convenience.
- Acceptance: re-run this session's "long request + short request 1.5 s later" probe (one group with tools, one without) ⇒ the short request's `done` time is close to its own engine time; the user-log-shaped `HTTP 499` no longer occurs.

### P1 Correctness and documentation (review items)

- **P1.1 = A3**: add a startup assertion in `build_shard` (see A3 recommendation 1), plus cross-referencing comments.
- **P1.2 = A4 + A5**: delete the dead path at `src/runtime/engine/model_instance.cpp:107-113`, change the comment at `:99-103` and `docs/serving.md:75-76`; remove the unreachable route branch at `src/runtime/engine/tp2_generation_core.cpp:2089-2090`.
- **P1.3 = B2 + B5 + documentation**: three wordings in `docs/serving.md:75-76`, `:77-79`, `:79-80`; add the coarser-grid explanation to the retention section; `src/serve/serve_options.h:79-82` is still the old wording ("TP-2 normalizes to 1 active + 1 queued") and should be changed as well.
- **P1.4 = A1 mitigation (pull-out)**: perform only if P2.1 is deferred.

### P2 Throughput (the real fix for the user's scenario)

- **P2.1 = C3/A1 (batched per-lane grammar mask)**
  - Layout: `tool_mask_dev = [vocab, constraint_columns * lanes]`, with lane `l` occupying the column block `[width*l, width*(l+1))` (consistent with the existing window column convention `t = column + width*lane`).
  - Constraint state: one `ToolCallConstraint` instance per lane (or an equivalent copyable state), advanced by that lane's accepted tokens in the current round; the prefill first token must also be masked.
  - New pieces needed: a per-column-block variant of `ops::apply_token_mask` (or an equivalent kernel), mask construction and H2D in the batched round, and a per-lane mask path for the prefill first token.
  - Acceptance: concurrent requests with tools show `path=batched`; per-lane oracle compared against the same lane run alone (near-tie criterion); grammar legality checked with tool-call regression cases.
- **P2.2 = C1 (dynamic admission of running batches)**
  - Change: extract the per-lane setup/prefill into callable units (e.g. `admit_lane(PendingRequest&, slot)`), take new requests from the queue at each round boundary and assign freed slots; pre-allocate `lanes`/staging by `lanes_` (`batch_lane_host_`, `mtp_pack`, `accept_host`, etc. are already lane-capacity).
  - Semantics: D4's "whole-batch failure" must be redefined (some members are already running); decide the blast radius of `session_invalidate_all` (B4) at the same time.
  - Acceptance: the user-log-shaped dual-session scenario (the first session is long, the second arrives while it runs) ⇒ the second session begins producing in later rounds of the same batch, and TTFT is no longer equal to the previous session's entire duration; gate 11/11 and the three artifact tests all green.

### P3 Cleanup and observability

- **P3.1 = B3**: build `mtp_chain_graphs_`/`mtp_chain_host_`/AR channels only when `lanes_ == 1 && mtp_enabled_`; show `n/a` in the log field for multiple lanes.
- **P3.2 = B6**: make `build.ps1`'s touch conditional.
- **P3.3 = B1**: capacity invariant comment.
- **P3.4**: the two items already committed this round (`34cb16ca` slicing serial DFlash2 frames by lane 0, `bc3c8a33` view errors carrying shape) are outside this plan's scope, but they are direct 500-failure fixes in the A1/C3 scenario and must be preserved in P2.1's regression.

---

## 7. Acceptance matrix

| Remediation item | Verification method | Criterion |
|---|---|---|
| P0.1 (A2) | Inject a `view`-throwing batch formation path (unit test or temporary injection) | No hang; both batch members and subsequent requests receive failure results; `lane_driver_active_` reset |
| P0.2 (C2) | "Long request + short request 1.5 s later" probe (one group with tools, one without) + user-log-shaped regression | Short request `done` ≈ its engine time; no `HTTP 499`; per-lane publish trace appears |
| P1.1 (A3) | Dual-GPU startup with `--max-concurrency 4 --max-context 131072` | Assertion not triggered; if the window is artificially relaxed, it throws |
| P1.2–P1.3 | Code review + documentation diff | No dead paths; documentation consistent with implementation |
| P2.1 (C3) | Dual-GPU serve smoke: 2×tools concurrently; grammar legality cases | `path=batched`; tool-call output legal; per-lane oracle passes the near-tie criterion |
| P2.2 (C1) | Dual-session arrival-timing probe (long + short arriving mid-run) | The second session starts in later rounds of the first batch; TTFT decoupled from the former's entire duration |
| Full regression | `tools/win_port/test.ps1` filtered by `tp2`/`tp_device`/`engine_options`/`serve_options`; `ninfer_qwen3_5_tp2_forward_test` / `tp2_load_test` / `tp2_sessions_test` (`NINFER_TEST_ARTIFACT`) | gate 11/11; forward/load exit 0; sessions keeps its existing results (including the known shared-system-prompt failure) |

---

## 8. Key points the review has confirmed "are not defects" (to avoid re-reviewing)

| Item | Evidence |
|---|---|
| The P1.8 scalar slot backfill is present at both batched entry points | `src/models/qwen3_5/execution/text.cpp:2361-2365`, `:3047-3051` |
| The graph key design for single-column batches is correct | `src/runtime/engine/tp2_generation_core.cpp:542-549`, `:629-637`; `run_plain_decode_step_batch:1679`, `run_verify_window_batch:1477`; `select_window_graph` requires exact equality of `graph.batch == batch` (`:1233-1239`) |
| AR channels cannot be exhausted | `src/runtime/engine/tp2_generation_core.cpp:664-666` reserves all graph families before any capture |
| The workspace watermark is consistent across graph replays | `:2838-2841` sets `round_base` once; `:2882-2885` rewinds each round; asserts `workspace->used() == graph->arena_begin[i]` before replay (`:1502-1505`, `:1703-1706`) |
| Retired lanes use the original slot rather than the compacted column index | `:2877-2878`, `:2893`, `:3466` |
| The sampling token-count buffer does not alias | `:2533-2550` slices by `base + index * public_tokens`; `:2557` `configs_dev` is `sizeof(SamplingConfig) * lanes_`; contract in `include/ninfer/ops/sampling.h:45-78` |
| `records_for`/`fold_for` have bounds checks | `src/runtime/engine/tp2_generation_core.h:248-259` |
| The wraparound subtraction in `submit` has layered coverage | `:1964` (multiple lanes) and `src/runtime/engine/engine.cpp:335-339` (single lane) |
| `restore_lane_gdn` only touches its own lane | `:2316-2348` (`ReuseSource::None` goes through `zero_lane_state`) |
| `active_lane_` is never read dirty within a batch | 18 references checked; the batched path only writes/reads inside the per-lane prefill loop |
| `state_backing` semantics are correct | `:776-781` `state_backing.bytes == state_bytes` (the pool itself); spec's per-round whole-pool snapshot/rollback does not damage snapshot slots |
| Not binding a GDN state slot for MTP layers is harmless | MTP layers are pure attention+MLP, with no GDN mixer |

---

## 9. Verification limits and open questions

1. **The review is source-level**: the measured numbers cited by the review report (1.76–1.91×@2, 2.80×@4, 1.41× unequal-length pairing, state +293.6 MiB/shard/lane) come from the existing plan and the gate results already recorded in `docs/serving.md`; nothing was rebuilt or run during the review.
2. **ctest cannot cover this change**: `ninfer_qwen3_5_tp2_load/forward/sessions/dflash_solo/dflash_append_test` are **Skipped** by ctest (`tests/models/qwen3_5/test_tp2_forward.cpp:158` needs an explicit `--artifact`, `:163` needs two GPUs), and that test has its own `build_shard_state` (`:120-129`) that **does not go through** `TP2GenerationCore::build_shard` ⇒ it cannot cover A3's block-table publish path. **The real regression gate is the dual-GPU `ninfer-serve --devices 0,1` smoke.**
3. **A byte-identical oracle does not apply to batched routes**: a batch shares one attention envelope determined by the longest lane ⇒ ulp drift can flip near-tie tokens (`docs/serving.md:93-98`). Batched correctness must use the near-tie criterion.
4. **Neither A2 nor A1/C3 can be exposed by existing tests**: the former needs input that throws during batch formation, the latter needs dual GPUs plus grammar and ordinary requests arriving simultaneously. Both need new tests (see the acceptance matrix).
5. **Open**: how D4's "whole-batch failure" should be redefined under dynamic admission (P2.2); whether `session_invalidate_all` should be narrowed to the failing batch (B4); whether the per-lane grammar mask's constraint state needs the entire constraint tree copied per lane (cost to be estimated).

---

## 10. Relation to existing documentation

| Document | Relation |
|---|---|
| `tp2-plan-concurrency.md` | Phase 0 decisions (`D1–D6`) and the Phase 1/2 construction drawings; once this document's A/B/C items land, a construction record should be appended to section 12 of that file, with the status backfilled here |
| `docs/serving.md` | External contract; P1.3 corrects its TP-2 section |
| `docs/tp2-decisions.md` | Settled list; A3/A4/A5 are "implementation does not match documentation" and do not change what is settled |
| `docs/maintainer/paged-kv-cache.md` | Authority for the execution row publish semantics involved in A3 |

---

## 11. Construction record

| Item | Status | Landing location | Verification |
|---|---|---|---|
| P0.1 (A2) | Landed | `src/runtime/engine/tp2_generation_core.cpp:2172-2211`: batch formation, trace, and dispatch are all inside the `try`, so exceptions go down the D4 path instead of escaping the driver thread | Compiles green; gate 11/11 (no throwing input injected separately) |
| P0.2 (C2) | Landed | Driver thread `drive_lane_queue` (`:2135-2218`), publish-on-completion `publish_lane` (`:2097-2130`), thread started in the constructor + `stop_lane_driver()` | "Long request + short request 1.5 s later" probe: the short request's `done` drops from waiting for the whole batch to 1858 ms (after P0.2) and 316 ms (after P2.2); no `HTTP 499` |
| P2.2 (C1) | Landed | `try_pop_lane_queue()` (`:2522-2531`); plain `admit_lane` lambda `:2747-2992` + initial batch `:2994-2996` + round-boundary admission `:3030-3050`; spec `admit_lane` lambda `:3397-3659` + initial batch `:3660-3662` + round-boundary admission `:3721-3741`; both routes `lane_capacity = lanes_` (`:2602`/`:3243`), with all staging (`tool_mask_host`/`accept_host`/`mtp_pack`/`append_*`) pre-allocated by `lanes_` | See below |

P2.2 measurements (dual GPU `--devices 0,1 --max-context 131072 --max-concurrency 4 --spec dflash2`, `NINFER_TP2_LANE_TRACE=1`, log `_temp/p22_serve.log`):

- **plain route**: a short request arriving 1.5 s into a running long request has `done=316 ms` (1858 ms after P0.2), and the long request completes its 4135 ms as usual.
- **tools route** (no longer degraded to serial since P2.1): a short tools request arriving 1.5 s into a running long tools request has `done=898 ms`, with the long request at 3835 ms; the short request produces the legal `call:get_weather({"city":"Paris"})` ⇒ a dynamically admitted lane also goes through the per-lane mask.
- **Four tools requests arriving simultaneously**: 2 form a batch + 2 are dynamically admitted via `admit slot=2 live=3` / `admit slot=3 live=4` ⇒ that round shows `[tp2-lane] grammar masked … window columns of 4 lanes`, and the lane 2/3 mask column blocks left over by P2.1 run end to end.
- **Regression**: `tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` ⇒ `100% tests passed out of 11 (70.73 s)`; `ninfer_qwen3_5_tp2_forward_test.exe --artifact …` and `…_load_test.exe --artifact …` exit 0; `tp2_sessions` is the pre-existing failure from §9.

**P2.2 open**: under dynamic admission D4's "whole-batch failure" is still whole-batch + `session_invalidate_all()` over the entire catalog; B4's blast radius is not narrowed and is left for P3.4/B4 to decide together.

### P1 construction record (P1.1–P1.4)

| Item | Status | Landing location | Verification |
|---|---|---|---|
| P1.1 (A3) | Landed | `tp2_generation_core.cpp`'s `build_shard` gained a startup assertion: when `lanes_ > 1` it requires `lane_context_window() + (mtp_enabled_ ? mtp_drafts_ : 0) < per_lane * kPagedKVPageSize`, otherwise it throws `TP-2 lane context window exceeds the pages published to its execution row`; the comment points out that `DeviceArena` is not zeroed ⇒ an out-of-range request would read uninitialized block-table entries and write to another lane's physical page. The `build_shard` call site was also moved down to after the lane budget computation | Dual-GPU startup with `--max-concurrency 4 --max-context 131072` does not trigger; gate 11/11 |
| P1.2 (A4+A5) | Landed | `src/runtime/engine/model_instance.cpp:97-107`: the dead "collapses to 1 lane" fprintf was deleted and the comment rewritten as "`--spec dflash` is rejected directly at the end of normalization; no route silently drops lanes, and the DFlash rule in the constant is unreachable here"; `tp2_generation_core.cpp` dropped the unreachable `refusal = "route"` branch | Compiles green; gate 11/11 |
| P1.3 (B2+B5+documentation) | Landed | Three wordings in `docs/serving.md` (the lane = position-within-batch binding, the checkpoint grid being about 4× coarser with 4 lanes, dynamic admission); `src/serve/serve_options.h:79-83` changed to `tp2_generation_concurrency(max_concurrency)` (1–4 active lanes) + the correct description of `max_pending_requests` queueing | Documentation diff review |
| P1.4 (A1 pull-out) | Not executed | Superseded by P2.1's real fix: every dsh request carries tools, so pull-out serialization has no benefit | — |

### P2.1 construction record (C3/A1)

- Landing locations: `tp2_generation_core.cpp` plain round `:3097-3102` (mask application), spec round `:3963-3968`/`:4150-4155` (construction + `ops::apply_token_mask`); the grammar refusal branch in `drive_lane_queue` was deleted. One `ToolCallConstraint` per lane, mask tensor `[vocab, window_live]`, lane `l` occupying the column block `[width*l, width*(l+1))` (consistent with the window column convention `t = column + width*lane`), with the prefill first token masked as well.
- Verification: `_temp/p21_serve.log` shows `path=batched` and repeated `[tp2-lane] grammar masked N window columns of M lanes`; in the P2.2 probe, 4 requests with tools show `grammar masked … of 4 lanes` in the same round, and the short request produces the legal `call:get_weather({"city":"Paris"})`; the MTP C=4 smoke (below) likewise returns 200 + `function_call` for `/v1/responses` with tools.

### P3 construction record (B3/B6/B1/B4)

- **P3.1 = B3**: the threshold at `tp2_generation_core.cpp:558-562` became `mtp_enabled_ && lanes_ == 1U`; the startup log's `mtp chain` field shows `n/a` when `!mtp_enabled_ || lanes_ > 1U`; `mtp_propose_window` gained a `mtp_chain_host_ == nullptr` guard (`:1846-1849`). Reachability: the only call site of `mtp_propose_window` is inside `execute_walk`, and `execute_walk` is only called on the `lanes_ == 1` branch.
- **P3.2 = B6**: `tools/win_port/build.ps1:134` gained a comment explaining that the fix was already conditional (it returns directly when the prefix matches, without a full touch + deleting `.ninja_deps`).
- **P3.3 = B1**: the comment at `src/serve/http_server.cpp:224-234` pins the capacity invariant (the accept thread does not occupy a worker; `acquire_request_lifetime` throws `RequestError(Overloaded)` when `active >= max_concurrency + max_pending_requests`) and explains that the headroom of 1 for `worker_count = queued_requests + 1` is deliberate (httplib's `ThreadPool::enqueue` returns false when full, so the client only sees a disconnect rather than 429).
- **P3.4 = B4**: a comment inside `drive_lane_queue`'s catch explains that a blast radius of all lanes is a deliberate choice (one round is one collective); `docs/serving.md` gained "A round is also the unit of failure…".
- Regression: `_temp/fix_gate.log` ⇒ `100% tests passed out of 11 (72.51 s)` (the 5 artifact cases Skipped by design); `ninfer_qwen3_5_tp2_forward_test.exe --artifact …` exit 0; `…_load_test.exe --artifact …` exit 0; `tp2_sessions` see below.

### P0.3 (newly found during P3 verification): MTP multi-GPU warmup failure — peer arena leak in `proposal_argmax`

- Symptom: `--spec mtp --max-concurrency 4` startup warmup reports `TP-2 batched verify CUDA Graph replay found a different workspace layout` (throw point `tp2_generation_core.cpp:1551-1553`); `--max-concurrency 1` is fine; DFlash2 multi-lane is unaffected.
- Root cause: `src/models/qwen3_5/execution/text.cpp:822 TextContext::proposal_argmax` takes a scope only on the local arena (`:823 auto proposal_scope = work_.scope();`), but the branch that splits the proposal head allocates `hidden_peer` (`:867`), `part_peer` (`:876`), `merged_peer` (`:878`) and the scratch for `project(..., peer.work_, ...)` (`:877`) on the **peer's arena**; there is no corresponding scope on the peer side ⇒ each call raises the peer's bump pointer by about 403,456 B and never reclaims it.
- Trigger chain: the multi-lane route runs `max_extent` MTP chain steps per round (`ctx_a.mtp_propose_batch`), and `max_extent` varies with that round's budget (warmup capture round 2, replay round 0) ⇒ when the verify graph is captured `arena_begin[1] = round_base + max_extent * 403456`, and the replay round does not match, so the assertion fails. Step-by-step trace in `_temp/p3_step_trace.log`: before the fix side B goes `81920 → 444416 → 847872` while side A does not move; after the fix side B is constant at `81920`.
- Attribution: a **pre-existing defect**, not introduced by this round's changes — `text.cpp` has only this one addition in the current working tree, and the last commit touching it is `6a17bb2c`. The other peer allocation points in the same file (`:494`/`:932`/`:1050`/`:2137`/`:2814`) all have `auto peer_scope = peer.work_.scope();`; only `:858` was missing it.
- Fix: add one line `auto peer_scope = peer.work_.scope();` after `:858` (with 4 comment lines, +5 lines total).
- Verification: C=4 `--draft-tokens 2` smoke warmup passes (`listening on http://127.0.0.1:8099`), two concurrent `/v1/responses` (a short request with tools + a 1729-token long request) both return HTTP 200, and the one with tools returns a legal `function_call`; C=1 still `mtp chain: graph`, channels 24, HTTP 200; gate 11/11, forward/load exit 0.

### Session-cache single/multi-concurrency consistency (m02934)

Background: `m02832`'s A/B probe found that C=1 reuses the system prefix (`cache 2,048 (74.8%, long anchor)`, TTFT 422 ms), while C≥2 is always `cache 0 (0.0%)`, TTFT 1.6 s. This round compared the wiring and shared helper functions of the three prefill routes item by item — `execute_walk` (C=1), `execute_plain_batch`, `execute_spec_batch` (C≥2) — with the conclusion: two real gaps (GAP A, GAP D), one architectural limit (GAP B, merely recorded), and one false report (GAP C).

| Item | Status | Landing location | Verification |
|---|---|---|---|
| GAP A: the two batched routes lack the walk's anchor mechanism, `entry.host_shared_end` is always 0, and a new session's shared prefix is rejected by `offered > shared` in `session_recall` | Landed | The walk's anchor planning and capture were extracted into three members: `plan_lane_anchors` (`src/runtime/engine/tp2_generation_core.cpp:2435-2531`), `capture_lane_anchor_from_device` (`:2533-2546`), `capture_lane_anchor_frozen` (`:2548-2583`); `LaneReuse` gained `shared_prefix`; the walk now calls them (`:6761`/`:6771`/`:6991`); the plain route wires four places (plan `:3246`, from-device `:3259`, `reuse_path(reuse, anchors.block_frontier, …)` `:3262`, chunk clamp + Divergence/Block write `:3350-3351`, frozen `:3446`), and the spec route is isomorphic (`:3986`/`:4000`/`:4002`/`:4096-4097`/`:4200`) | See below |
| GAP D: when a batched route cancels prefill it unconditionally calls `invalidate_lane_prefill`, losing the partial progress of completed chunks (the walk publishes it) | Landed | Added `publish_partial_prefill` (`:2701-2738`, mirroring the walk's cancel path `:6777-6812`); both cancel branches now publish when `prefilled > 0` and invalidate otherwise (`:3428-3437`, `:4182-4191`); `prefilled` was hoisted outside the chunk loop (`:3282`, `:4019`) | See below |
| GAP B: the walk writes Grid checkpoints in the decode round (`:7616+`), the batched routes do not | Not implemented (recorded as an architectural limit) | Every C≥2 `finalize` goes through `release_lane_kv` → `invalidate_host_checkpoints(lane)`, emptying the ring; only the catalog's three mirror classes (frontier / prompt-end / block-shared) survive across requests. To make the grid boundary at the decode tail usable, `SessionEntry` would need another mirror class, yet the host budget of 10240 MiB/shard only fits 2 sessions of 3960 MiB (`[tp2-session] host budget …`), so the benefit (fewer tail re-prefills when a client re-renders an answer) does not justify the cost | Analysis conclusion |
| GAP C: the frontier semantics published by the two routes are inconsistent | False report | `publish_lane_prefill(lane.prompt_tokens, …)` at plain `:3448` / spec `:4202` and `frontier = prompt_tokens + sampled - 1` at walk `:7647-7655` are the same point in their respective coordinate systems (the last sampled token is no longer forwarded) | Code comparison |

GAP A measurements (after compiling; probes `_temp/m03350_probe.mjs` + `_temp/run_m03350_probe.ps1` / `_temp/run_m03420_probe.ps1`, 5 requests sharing the first 2,722 tokens; before the fix C=2/C=4 all `cache 0`, TTFT 1.6 s):

| Arm | A1 | A2/A3/B1/B2 |
|---|---|---|
| `floor0_c2` (spec dflash2, conc=2, `--session-retention-floor 0`, prompt 2,738) | `cache 0`, 2043 ms | `cache 2,048`, 869/963/937/883 ms |
| `long_c2` (spec, floor 4096, prompt 10,839) | `cache 0`, 7054 ms | `cache 10,240`, 1043/976/964/926 ms |
| `plain_c2` (no `--spec`, floor 0, 2,738) | `cache 0`, 2249 ms | `cache 2,048`, 1203/1229/1387/1276 ms |
| `spec_c4` (conc=4, floor 0, 2,738) | `cache 0`, 2136 ms | `cache 2,048`, 1063/1158/1251/1358 ms |

The long-prompt arm is exactly the user's real scenario (> retention floor): hit 10,240/10,839 = 94.5%. `_temp/m03350_long_c2.err` confirms the chain: A1 stores `anchor entry=0 position=10240`; A2 candidate `… host_prompt_end=10839 shared_end=10240 shared=10823 reach=10240 via=shared` → recall → `reuse=10240 src=live`. The short-prompt arm requires `--session-retention-floor 0` to create an entry at all: `session_publish` early-returns when `history.size() < session_retention_floor_tokens_` (`:5895`); 4096 is user configuration, not a code defect.

GAP D measurements (`_temp/m03470_cancel.mjs` + `_temp/run_m03470_cancel.ps1`, prompt 10,839, C1 disconnects after 2 s ⇒ server-side `cancellation.requested()` hits in the chunk loop):

- spec: after C1 cancels, `anchor entry=0 position=4096`; C2 retries the same prompt with `recall frontier=4096 … reuse=4096 src=live` ⇒ `cache 4,096`, 4414 ms (before the fix that retry was a full prefill, `cache 0`); C3, a new session of the same family, still `cache 10,240`, 966 ms.
- plain: partial progress 5120 ⇒ C2 `cache 5,120`, 3872 ms; C3 `cache 10,240`, 1269 ms.

C=1 regression (`_temp/run_m03480_c1.ps1`, conc=1 / floor 4096 / prompt 2,738): A1 `cache 0` 1933 ms, A2–B2 all `cache 2,048` (686/719/694/810 ms) — behavior is unchanged after the walk switched to the shared helpers.

### Admission budget off-by-one: under C≥2 a clamped request is never admitted (m03586)

Background: the user asked "what happens when 2 sessions are open and the total context exceeds `--max-context`: is output truncated? offloaded to memory and queued? or an outright error?". Tracing the source revealed that `lane_need_tokens` counts one token more than the real KV requirement, and combined with `submit`'s `capacity_output = ceiling - prompt + 1` (deliberately leaving room for "the last sampled token is no longer forwarded"), the reserved page count exceeds the pool page count by 1 ⇒ the request is never admitted.

Mechanism: pool page count `P = pages_for_tokens(max_context)`, pool token count `T = 64P`, `margin = lane_kv_margin()`; for C≥2 and `--lane-context 0` the admission ceiling is `W = T - margin`. `submit` (`src/runtime/engine/tp2_generation_core.cpp:2035-2081`) returns 400 when `prompt > W`, otherwise `effective = min(requested, W - prompt + 1)`. It reserves `pages_for_tokens(prompt + effective + margin)` (`:2800-2802`):

- `prompt + requested ≤ W`: no clamp, `pages ≤ P` ✓;
- `prompt + requested = W + 1`: no clamp (`requested == capacity_output`), `prompt + effective = W + 1` ⇒ `pages = P + 1` ✗;
- `prompt + requested ≥ W + 2`: clamped to `W - prompt + 1` ⇒ likewise `W + 1` ⇒ `pages = P + 1` ✗.

`reserve_lane_kv` is all-or-nothing (`:2804-2831`), and on failure `requeue_lane_front` puts the request back at the head of the queue (`:2922-2926`); the driver thread does not wait while the queue is non-empty (`:2198-2212`) ⇒ busy spin, and the client waits forever. The TP-2 route's `pending_deadline` parameter is unnamed at the definition and ignored (`:2022-2026`), so `--pending-timeout-ms` has no effect on the post-submission waiting phase. The comment at `:2798-2799` ("a request that fits the policy always fits an empty pool … never a deadlock") therefore does not hold.

Fix: change `lane_need_tokens` to `prompt + (budget == 0 ? 0 : budget - 1)` (`:2929-2936`), consistent with the single-GPU plan's `reserved_context_tokens` (`src/models/qwen3_5/program/planning/request_plan.cpp:258-262`) — the last sampled token is no longer forwarded and needs no slot. C=1 does not go through the batched route (`reserve_lane_kv` returns true directly when `lanes_ <= 1U`, `:2805`; the driver thread starts only when `lanes_ > 1U`, `:721`), so the single-concurrency path is byte-identical.

Measurements (`_temp/m03620_cap.mjs` + `_temp/run_m03620_cap.ps1`, dual GPU `--max-context 4096`, `--spec dflash2`, prompt 3,740, `max_output_tokens 4096`):

| Arm | Before the fix | After the fix |
|---|---|---|
| C=2 control (`max_output_tokens 20`, no clamp) | 200 / 2429 ms | 200 / 2517 ms |
| C=2 clamp (`max_output_tokens 4096`) | **No response**: client 30 s timeout (`http=0`), server emits 36,238 lines of `[tp2-lane] batch=1 capacity=2 path=batched` within 30 s, log 1.52 MiB (busy spin) | 200 / 2371 ms, `status=completed`, 7 lane trace lines, log 0 MiB; `[tp2-capacity] … clamped to 356` |
| C=1 clamp (same request) | 200 / 2589 ms, clamped to 357 | 200 / 2451 ms, clamped to 357 (unchanged) |

External behavior (answering the original question): a single request whose prompt exceeds the ceiling ⇒ HTTP 400 `context_length_exceeded`; an output budget exceeding the ceiling ⇒ no error, the output budget is clamped (this has been true since C=1); two sessions whose total exceeds the pool ⇒ the later request stays at the head of the FIFO queue waiting for the previous lane to retire (no truncation, no error; "offload to memory" is a different mechanism, namely S1's session-level host retention). Before the fix, "output budget being clamped" under C≥2 happened to fall into the hang branch above.

Two adjacent gaps found along the way (fixed in the next section): (1) the TP-2 route's `pending_deadline` is ignored ⇒ no `--pending-timeout-ms` enforcement during the queuing phase (the external statement in `docs/serving.md:1039-1045` does not hold for TP-2); (2) a queued request does not check `cancellation` (it is only checked inside the executor, `:3280`/`:4020`/`:6781`/`:7105`) ⇒ after the client disconnects the request still occupies the queue and will still be executed.

### Timeout and cancellation during the queuing phase: enforced by the submitter itself (m03710)

Background: the two adjacent gaps at the end of the previous section. The TP-2 route's `submit` parameter carries `pending_deadline` but it is unnamed at the definition and ignored, and `wait_lanes` only waits for its own members' `complete` ⇒ `--pending-timeout-ms` has no effect at all on the post-submission queuing phase (the external statement in `docs/serving.md:1039-1045` does not hold for TP-2); a queued request also does not check `cancellation` (only the checks inside the executor, `:3280`/`:4020`/`:6781`/`:7105`), so after the client disconnects it still occupies the queue and will still be executed.

Mechanism: the single-GPU route's admission (`src/runtime/engine/engine_core.h:1579-1597`) checks the absolute deadline first and then `cancelled`: timeout ⇒ `RequestError(QueueTimeout, "inference request expired while waiting for admission")` (HTTP 503 `request_queue_timeout`, `src/serve/generation_service.cpp:69-74`), cancellation ⇒ `complete_detached_cancelled` (499). TP-2 moves that same pair of checks to the submitter: `admitted` is the sole handover flag and is only modified under `lane_queue_mutex_` — whoever takes a member out of the FIFO (the driver's batch formation, `src/runtime/engine/tp2_generation_core.cpp:2294-2313`; or the mid-batch `try_pop_lane_queue`, `:2889-2913`) sets it, and `requeue_lane_front` (`:3044-3051`) clears it when putting a member back at the head of the queue. While `admitted == false` the submitter enforces the deadline and cancellation; once set, the executor's existing checks take over.

Neither the deadline nor cancellation can signal into `lane_queue_cv_` (the deadline is not a lock-protected state change, and the cancel flag is an atomic written by the HTTP thread), so `wait_lanes`'s wait is bounded at 250 ms (`kQueuedCancelPollInterval`, `:2149`) and re-checks every round; the deadline takes priority over cancellation, consistent with the single-GPU route. Retirement works the same way as the driver's own pop: `drop_expired_lane` (`:2172`) stores the `QueueTimeout` that the submitter must rethrow (no preview sent), and `drop_cancelled_lane` (`:2164`) emits a terminal Cancelled preview consistent with the executor's first cancellation check. The driver and `try_pop_lane_queue` also refuse to hand a lane to an expired/cancelled member, so a requeued member cannot be preempted into execution while its submitter is asleep.

Measurements (`_temp/m03720_queue.mjs` + `_temp/run_m03720_queue.ps1`; dual GPU `--max-context 4096` (pool 64 pages), `--spec dflash2`, prompt 2,748, `max_output_tokens 1024` (req1 generates for 18 s and fills the pool, and req2 is the same size ⇒ it must queue); req2 is issued 600 ms after req1 starts):

| Arm | Client | Server |
|---|---|---|
| timeout (C=2, `--pending-timeout-ms 2000`) | req2 **HTTP 503**, 2016 ms, `inference request expired while waiting for admission`; req1 200 / 18.3 s / 1024 tok | `publish tokens=0` ⇒ `WARN req#2 failed during generation … HTTP 503 request queue timeout` (req1 is still decoding, proving req2 never got a lane) |
| cancel (C=2, 30000 ms, req2 aborts at 2505 ms) | req2 client AbortError; req1 200 / 18.5 s | `publish tokens=0` ⇒ `req#2 done … cancelled … prompt 0 / output 0` |
| control (C=2, two short requests, both admitted) | two 200s (463/291 ms) | no `publish tokens=0` |
| C=1 regression (two short requests) | two 200s (379/325 ms) | no `publish tokens=0` |

C=1 does not go through this path (`wait_lanes` is only called by `Submission::wait` when `lanes_ > 1`; C=1 queues directly on `execution_mutex_`, with no deadline enforcement, byte-identical to before the change).

## Four-item wrap-up: 413 payloads, KV fragmentation, S1 round-trip cost, C=1 vs C=4 (m03963/m03964)

### 1. HTTP 413 should only come from `--max-request-mib` (fixed)

Root cause: the contract in `docs/serving.md:420-422` is that 413 `request_too_large` is reserved for a raw request body that exceeds `--max-request-mib` before JSON parsing, but the vendored cpp-httplib has a separate 8192-byte limit for `application/x-www-form-urlencoded`: `CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH` (`third_party/cpp-httplib/httplib.h:133-134`) is applied directly in ordinary-handler routes' `Server::read_content` (`httplib.h:13326-13336`), which sets 413 and returns false regardless of `payload_max_length_`, and it happens before any handler ⇒ the application layer cannot intercept it, and the payload even wrongly claims that the configured limit was exceeded.

Fix: the 8 routes with a body (7 POSTs + 1 DELETE, `src/serve/http_server.cpp:492-529`) were moved to httplib's content-reader routes (`HttpServer::register_post`/`register_delete`, `src/serve/http_server.h:77-81`; `buffer_request_body`, `src/serve/http_server.cpp:41-61`). That path goes through `read_content_core` (`httplib.h:13350-13425`), which only checks `payload_max_length_` at `:13409` for the limit, with no form limit and no `parse_query_text` (no DoS amplification); `pre_routing_handler_` (authentication, `httplib.h:13747`) still runs before the body is read. `handle_unrendered_http_error` and `tests/test_http_error_handler.cpp` were unchanged.

Measurements (`_temp/m04130_413.mjs`; dual GPU `--max-request-mib 1`, `--spec dflash2`):

| Request | Before the fix `61F09142…` | After the fix |
|---|---|---|
| 18,234 B, `application/json` (control) | 200 / 2,538 ms | 200 / 2,427 ms |
| 18,234 B, `application/x-www-form-urlencoded` | **413** "exceeds the configured payload limit of 1048576 bytes" (body is only 18 KB) | **200** / 2,357 ms |
| 2,097,202 B, `application/json` | 413 | 413 "…1048576 bytes" |
| 2,097,181 B, `application/x-www-form-urlencoded` | 413 | 413 "…1048576 bytes" |

Streaming regression: SSE works as usual under the content-reader routes (`STREAM http=200 dataLines=20/22 eventLines=20/22 completed=true`, once each for C=1 and C=4).

### 2. KV pool fragmentation: external fragmentation is 0, internal fragmentation = page rounding + holding the entire output budget until retirement

Structurally, admission cannot fail due to fragmentation: `DeviceKVPagePool::reserve` only checks `pages == 0 || pages > available_pages()` (`src/core/paged_kv_cache.cpp:266-270`); a reservation is a pure counter and does not require physical contiguity. Physical contiguity only affects `contiguous_run_count` (`:253-264`) and `materialize`'s adjacency preference (`:288-320`), i.e. it only affects the number of `cudaMemcpy2D` batches in `copy_to_host`/`copy_from_host` (`:576-660`).

A new env-gated trace `NINFER_TP2_KV_TRACE=1` (`src/runtime/engine/tp2_generation_core.cpp:2922-2932`) prints the pool counters, that lane's run count, and the store/recall wall clock at `reserve_lane_kv` (`:3003-3015`), `release_lane_kv` (`:3033-3042`), `session_store_active` (`:5628-5637`), and `session_restore` (`:5765-5777`).

Measurements (`_temp/m04130_kv.mjs` + `_temp/run_m04130_mix.ps1`; dual-GPU user recipe, `--max-context 245760` ⇒ pool 3,840 pages/card, 1,081,344 B/page ≈ 3.96 GiB/card): 34 reserve/release and 18 store/recall operations, with `runs` **all 1/1** (both cards a single contiguous run), sizes covering 1/44/60/86/107/171/234 pages including concurrent interleaving ⇒ no physical fragmentation was observed.

Internal fragmentation (reserved pages vs actually stored pages):

| Request | Reserved pages | Stored pages | Idle |
|---|---|---|---|
| prompt 10,839 + budget 64 | 171 | 170 (frontier 10,857/10,863) | 1 page = 1.03 MiB/card |
| prompt 5,418 + budget 64 | 86 | 85–86 | 0–1 pages |
| prompt 2,738 + budget 64 | 44 | 44 | 0 |
| prompt 10,836 + budget 4096 (stops after 20 tok) | 234 | 170 | **64 pages = 66.0 MiB/card (132.0 MiB for two cards)** |
| prompt 2,735 + budget 4096 (stops after 25 tok) | 107 | 44 | **63 pages = 65.0 MiB/card** |

Concurrency peak: two 234-page requests held simultaneously ⇒ `allocated 468/3840` (12.2%), with `available` dipping to 3,372/3,840 (87.8%). Conclusion: the waste is internal and is the established policy (the whole output budget is reserved up front, held until retirement, never shrunk, never preempted, m01589/m01553), not allocator fragmentation; its visible cost is "requests that could have fit instead queue because the entire budget is reserved", not failure or truncation.

### 3. Throughput cost of the S1 session host round trip

| Operation | Pages | KV bytes/card | ms | KV-only lower bound (two cards combined) |
|---|---|---|---|---|
| store | 170 | 175.3 MiB | 8 runs: 79.0 / median ≈108 / 133.8 | 3.13 GiB/s |
| store | 86 / 85 | 88.7 / 87.7 MiB | 67.3 / 71.9 | 2.4–2.6 GiB/s |
| recall (frontier) | 160 | 165.0 MiB | 6 runs: 60.9 / median 64.5 / 66.6 | 5.00 GiB/s |
| recall (shared) | 160 / 80 | 165.0 / 82.5 MiB | 59.5 / 37.7 | 5.42 / 4.27 GiB/s |

(The KV-only lower bound counts only KV page bytes and excludes the lane state D2H/H2D and dflash draft mirror in the same interval ⇒ the actual bandwidth is higher. store moves only the actual frontier, not the whole reservation.)

Comparison benefit: a 10,845-token cold prefill takes 6.4 s (1.69k tok/s) → after hitting 10,240 tokens, TTFT is 0.45–0.47 s (605 tokens @1.33–1.53k tok/s). One full store+recall round trip ≈ 0.11 + 0.065 = **0.17 s**, i.e. about 3% of the benefit; the cost is fixed and bounded (~0.11 s per retirement, regardless of whether it is later reused).

### 4. C=1 vs C=4 paired single-request measurement

`_temp/m04140_pair.mjs` + `_temp/run_m04140_pair.ps1`; the same prompt (10,845), the same `max_output_tokens 256`, the same user recipe, changing only `--max-concurrency`; 3 consecutive runs + 1 streaming per arm.

| Metric | C=1 | C=4 |
|---|---|---|
| R1 cold TTFT | 6.4 s | 6.4 s |
| R1 cold prefill | 1.69k tok/s (10,845 tok) | 1.70k tok/s (10,845 tok) |
| R1 cold decode | 115.8 tok/s (dflash2 201/269) | 104.9 tok/s (198/283) |
| R1 client total duration | 8,676 ms | 9,215 ms |
| R2 reuse TTFT | 406 ms (cache 10,240, long anchor) | 466 ms (cache 10,240, private endpoint) |
| R2 prefill / client | 1.53k tok/s / 2,599 ms | 1.33k tok/s / 3,001 ms |
| R3 reuse TTFT / client | 405 ms / 2,481 ms | 453 ms / 4,082 ms |
| Device usage (card0/card1) | 14,924 / 14,678 MiB | 15,512 / 15,134 MiB |
| state arena (card1) | 146.8 MiB | 587.2 MiB |
| host checkpoint ring (card1) | 35 slots × 73.4 MiB, stride 10,240, pinned 2,569.2 MiB | 36 slots × 73.4 MiB, stride 81,920, pinned 2,642.6 MiB |
| card1 free | 620.0 MiB | 164.0 MiB |

Conclusion: the multi-lane mechanism has no effect on the prefill throughput of "a single request" (1.69k vs 1.70k tok/s, TTFT 6.4 s both); cold-start end to end is +6% (the decode difference is dominated by dflash2 acceptance-rate variance: 74.7% vs 70.0%, indistinguishable at n=3). The stable differences are the reuse path (C≥2 goes through the host slab: TTFT +15%, 406→466 ms) and device memory (+588/+456 MiB device + 73.4 MiB pinned, card1 free 620→164 MiB). C=4 R3's decode of 80.1 tok/s (acceptance rate 48.5%) shows that decode comparisons at n=3 on a single machine are not controllable; only the prefill and device-memory figures can be read.
