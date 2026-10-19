# Remaining optimization headroom of the TP-2 multi-concurrency implementation

> Subject: the multi-lane batch implementation of `ninfer-serve --devices a,b` (the dedicated TP-2 route).
> Scope: VRAM/memory footprint, prefill/decode speed, TTFT latency.
> Conclusions are based on a source walk plus local measurements (2026-10-02, 2×RTX 5060 Ti, HEAD `e9c8ec17`).
> Existing conclusions (docs/tp2-decisions.md, docs/tp2-dual-5060ti.md, docs/PLAN-tp2-concurrency*.md) are not re-verified; this is incremental only.

## 0. Summary of conclusions

TP-2 multi-concurrency **decode aggregation is already close to the ideal for this implementation**: a 4-lane equal-length batch measures 3.31× (ideal 4×),
and the 17% loss all comes from the part that grows linearly with the column count (single round 26.5 ms → 32.0 ms). The weight stream is already amortized;
there is no order-of-magnitude headroom there.

The remaining, user-visible headroom falls into three classes:

| # | Direction | Measured evidence | Magnitude |
|---|-----------|-------------------|-----------|
| 1 | **TTFT / mixed-load latency**: prefill is serial per lane; a lane that joins mid-flight blocks decode for the entire stretch at the top of a round | 4×10.8k concurrent: the last lane's client TTFT ≈23 s (the engine log reports only 5.5 s); a short request queued behind a long one is blocked for 5.1 s; when the short request is ahead, its decode goes 37.7 → 2.5 tok/s (15×) | Largest and most certain |
| 2 | **VRAM waste**: the round-scratch state plane of the plain route is entirely unused; KV is pre-reserved for the whole output budget and held until the request ends | At C=4, 293.6 MiB/card (587 MiB across two cards) is pure waste; with budget 4096, after emitting only 20 tokens it still holds 64 pages = 66 MiB/card | Medium |
| 3 | **Documentation and measurement**: the "262,144 requires C=1" guidance in `docs/serving.md` is obsolete; the plain batch route has no round timing instrumentation | Today C=4 at 262,144 fp8/MTP K=2 starts normally and runs 4 requests to completion (811/1,769 MiB left); `[tp2-time] plain-batch` is always 0.00 | Low (but blocks further optimization) |

Directions already near the hardware limit, not worth further investment: prefill aggregate throughput (limited by card 2's Gen4 x4 link,
2.67k tok/s ceiling), the decode weight stream (already 75-90% of bandwidth), and the allreduce call count
(reducing the calls necessarily sacrifices weight splitting; see docs/tp2-decisions.md).

**Execution status** (2026-10-02; per-item execution is in the §6 status column, §8 and §9): all P0/P1 items in the §6 list have been implemented and measured;
item 5 was judged to need no change after analysis; item 9 was judged not to change (it would break the reservation invariant); of the four items added in the second round (speculative-route
prefill interleaving, the item 10 Ops fix, P3 batched prefill, pageable host checkpoint ring), three have been implemented and measured
—the value of item 10 is **numerical consistency** (the acceptance-rate gain did not reproduce). P3 batched prefill: measure first, then judge: the batch window
`forward_tp2_window_batch` costs 1.24 ms/lane-token at W=16/B=4, whereas single-lane 1024-chunk is
0.587 ms/lane-token, **batching is 2.1× slower** ⇒ definitively not doing it (breaking even needs W×B≈1216 columns/round, which is
19× the ops limit width ≤ 16). The only remaining headroom is "first relax the ops batch domain before talking about batched prefill" plus hardware (card 2's PCIe slot).

## 1. Current implementation (the structures used by this report)

- One compact decode batch per round, lane count = `--max-concurrency` (1..4), lane identity = position in the batch.
- The driver thread `drive_lane_queue` (src/runtime/engine/tp2_generation_core.cpp:2280-2391) holds
  `execution_mutex_` across the entire batch execution; new requests join via **mid-batch admission** and do not start another batch.
- Routing: plain → `execute_plain_batch` (:3142-3867); MTP/DFlash2 → `execute_spec_batch` (:3868+);
  a single lane goes through `execute_walk` (with lanes>1, `execute_walk` is dead code).
- `admit_lane` (plain :3345-3641) performs that lane's **entire synchronous prefill** inside one lambda:
  the chunk loop `for (t0 = reuse; t0 < prompt_tokens;)` calls `forward_tp2_prefill(...)` (:3468-3605),
  then samples the first token at the end. Comment at :3453-3455, verbatim: "Prefill is serial per lane, so the single startup
  arena is reused sequentially and only one session is ever alive".
- Each lane's KV is reserved dynamically from the shared paged pool (`reserve_lane_kv` :2938-3000);
  the linear attention state arena is slotted per lane (`slot_count = lanes_`, :826).

## 2. VRAM / memory

### 2.1 Measured: device budget per concurrency at the 262,144 ceiling

Configuration: `--max-context 262144 --kv-dtype fp8 --host-state-slots 32 --host-kv-mib 20480`,
no requests after load; nvidia-smi and the startup ledger were read (`_temp/20261002-1350_mem.ps1`).

| Configuration | GPU0 used | GPU1 used | state/card | record/card | ledger free 0/1 |
|------|-----------|-----------|----------|-----------|-----------------|
| MTP K=2 + `--lm-head-draft` C=1 (shipped configuration) | 15,028 | 14,074 | 146.8 | 2.6 | 270.0 / 1,218.0 |
| same, C=2 | 15,178 | 14,222 | 293.6 | 5.1 | 122.0 / 1,070.0 |
| same, C=4 | 15,482 | 14,522 | 587.2 | 10.2 | 0.0 / 770.0 |
| plain C=4 | 14,338 | 14,338 | 587.2 | 0.0 | 952.0 / 952.0 |

(Total VRAM 16,310.6 MiB/card; the ledger's free column is cudaMemGetInfo, which is pessimistic under this driver;
take nvidia-smi as authoritative: C=4 MTP actually leaves 829 / 1,789 MiB.)

**Per-lane incremental cost = (15,482 − 15,028)/3 = 151.3 MiB/card ≈ state 146.8 (2 planes × 73.4) + record ~5.**
The KV pool is fixed by `--max-context` (4,096 pages) and **does not grow with the lane count**.

End-to-end verification (`_temp/20261002-1440_shipped_c4.ps1`, shipped configuration + C=4 + 4 concurrent 243-token requests):
all 4 returned 200, each emitting 256 tokens, wall 5,289 ms, per-lane decode 59.5/54.4/51.8/53.8 tok/s
(aggregate 219.5 tok/s), MTP acceptance 80.0-87.1%, 811 / 1,769 MiB left after the run.

⇒ The `docs/serving.md` guidance "keep `--max-concurrency 1` for the 262,144 configuration; 2-4 lanes need
`--max-context 131072`" is **obsolete guidance**: it was written when the state arena was 293.6 MiB/card/lane
(the era of `kReuseSnapshotCount = 2`). It is currently 146.8 MiB/card/lane, and C=4 starts and runs normally at 262,144.

### 2.2 Waste 1: the round-scratch state plane, entirely unused on the plain route

> **Fixed (item 2)**: the plain route no longer carves that plane. The measured startup ledger is `state 293.6 | round 0.0` (before the change round was also 293.6), saving 293.6 MiB per card. See §8.1.

Plane count of `state_arena` (:839-845):

```cpp
const std::size_t first_state_plane = host_checkpoint_stride_ != 0 ? 1U : 0U;
const std::size_t state_planes      = 1U + shard.state_snapshots.size() - first_state_plane;
```

With the host ring open (the default) `first_state_plane = 1`, so only slot 1 (= `kRoundScratchSlot`) is carved out;
slot 0 (the reuse boundary) is carried by the host ring (explicitly stated by the comment at :2785-2788).

And the **entire set of reference sites** for `state_snapshots[kRoundScratchSlot]` is only:
src/runtime/engine/tp2_generation_core.cpp:4701, :4886, :5079 (`execute_spec_batch`)
and :7761, :7804 (`execute_walk`, and both inside the `dflash2_enabled_` / MTP branches).
`execute_plain_batch` (:3142-3867) has **zero references** (confirmed by a whole-file grep).

⇒ On the plain route that plane is allocated, is memset, and is never read or written. Cost = 73.4 MiB × lanes /card:

| Configuration | Waste/card | Two-card total |
|------|---------|----------|
| plain C=1 | 73.4 MiB | 146.8 MiB |
| plain C=2 | 146.8 MiB | 293.6 MiB |
| plain C=4 | 293.6 MiB | 587.2 MiB |

The fix (a few lines): widen the skip condition at :839 from "does the ring exist" to
`host_checkpoint_stride_ != 0 || !(mtp_enabled_ || dflash2_enabled_)`, i.e. do not carve the scratch plane outside the spec route.

### 2.3 Waste 2: the whole output budget is pre-reserved and held until the request ends

> **Analyzed, not changed (item 9)**: shrinking by the remaining amount would break the all-or-nothing reservation invariant and force lanes to grow mid-flight → deadlock without backpressure. The operational lever is `--lane-context` (item 6). See §8.6.

`lane_need_tokens` (:3113-3116) = `prompt_tokens + (budget == 0 ? 0 : budget - 1)`,
and `reserve_lane_kv` reserves all pages at once at admission time, **never shrinks, never preempts**, until the lane retires.

Measured (prior work, `NINFER_TP2_KV_TRACE=1`):

- 10,836 token prompt + budget 4096, actually emitting only 20 tokens → 234 pages reserved / 170 pages actually written,
  64 pages = **66.0 MiB/card (132 MiB across two cards) idle occupancy**.
- 2,735 + budget 4096, emitting 25 tokens → 107 / 44 = 63 pages = 65.0 MiB/card.

The externally visible symptom is "requests that could have fit are queued", not failure or truncation. Under high concurrency + a large `max_output_tokens` on the
client (tool-style requests often carry a large budget), this directly lowers the effective concurrency.

Possible fixes: shrink the budget dynamically by the remaining amount (release trailing pages), or reclaim pages for lanes that are "finished but not retired",
or set the default `--lane-context` to pool_tokens / max_concurrency (see 2.4).

### 2.4 Risk: `--lane-context` defaulting to 0 lets a single lane monopolize the whole pool

> **Implemented (item 6)**: with multiple lanes and `--lane-context 0`, startup prints a warning (including the ceiling tokens and the lane count); no hard default is applied (a hard default would reject prompts longer than pool/lanes). See §8.8.

`src/serve/serve_options.h:37` `std::uint32_t lane_context = 0;`;
tp2_generation_core.cpp:443-450 sets the ceiling to `pool_tokens - margin` when lanes>1,
and only an explicit `--lane-context` from the operator lowers it.

⇒ Under the default configuration, an early single large request may legitimately occupy almost the entire KV pool, while later lanes sit at the head of the FIFO
waiting for it to retire—**concurrency can degrade to 1 under the default configuration**. This stacks with the "full pre-reservation" of 2.3.
Suggestion: on the multi-lane route, default `--lane-context` to `pool_tokens / lanes_` (or at least print a warning).

### 2.5 Host-side pinned checkpoint ring

`[mem] host-checkpoints shard N slots … x 73.4 MiB | stride … | pinned …`:

| Configuration | slots | stride | pinned/card |
|------|-------|--------|-----------|
| 262,144 C=1 | 35 (grid 24 + tail 8 + div 1 + block 1 + prompt-end 1) | 11,008 tok | 2,569.2 MiB |
| 262,144 C=2 | 36 (grid 8 + tail 7 + …) | 32,768 tok | 2,642.6 MiB |
| 262,144 C=4 | 36 (grid 3 + tail 3 + …) | 87,424 tok | 2,642.6 MiB |
| 245,760 C=4 (dflash2) | 36 | 81,920 tok | 4,082.6 MiB |

This is **non-pageable host memory**, 5.1-8.2 GiB across two cards. The benefit is avoiding heavy re-prefill (a single
10,240 token hit drops TTFT from 6.4 s to 0.45 s); the cost is about 0.17 s per store/recall
(store 3.13 GiB/s, recall 5.00 GiB/s, from prior measurements).

A tunable, not a defect: the ring is pinned for DMA bandwidth; if it were made pageable, the cost is that 0.17 s getting slower,
in exchange for GiB-level host memory. At C≥2 the grid drops from 24 slots to 3 slots (stride 11,008 → 87,424 tok),
i.e. the "grid is about 4× coarser at 4 lanes" described in the docs.

### 2.6 Other

- `--kv-dtype`: int8 is **larger** than fp8 by 48 MiB under the same ceiling (64-element scale group);
  nvfp4/k8v4 are smaller but have a value-precision cost. If you want more context or more headroom, this is the only KV-side knob.
- Weights + CUDA context at 10,448.6-11,528.6 MiB/card are the absolute dominant term and cannot be compressed (except by changing the split scheme).
- No external fragmentation of the KV pool was observed (prior work: 34 reserve/release, 18 store/recall, all runs equal to 1).

## 3. decode speed

### 3.1 Measured scaling curve

plain, bf16 KV, `--max-context 8192`, 243 token prompt, 4 lanes of equal length all hitting the 256 token cap
(`_temp/20261002-1310_scale.ps1`, per-lane rate taken from the `decode X tok/s` field of the `done |` line):

| C | per-lane tok/s | round ms | aggregate tok/s | ideal | efficiency |
|---|---------------|----------|------------|------|------|
| 1 | 37.7 | 26.5 | 37.7 (1.00×) | 1× | 100% |
| 2 | 35.0 / 35.9 | 28.2 | 70.9 (1.88×) | 2× | 94% |
| 3 | 33.5 / 32.1 / 32.8 | 30.5 | 98.4 (2.61×) | 3× | 87% |
| 4 | 30.3 / 32.2 / 31.5 / 30.9 | 32.0 | 124.9 (3.31×) | 4× | 83% |

**Round cost model: round(B) ≈ 26.5 + 1.85 × (B − 1) ms.**
That is, each extra column costs only 1.85 ms more (≈ 7% of a single-lane round); the ideal (purely weight-stream limited) is 0.
C=4 loses 17% of the aggregate gain; halving the marginal column cost would take C=4 to ≈3.62×.

> Difference from the docs: docs/serving.md and tp2-plan-concurrency.md record the plain route aggregate as
> C=2 1.76-1.91×, C=4 2.80×. This run reproduced C=2 (1.88×), but measured **3.31×** for C=4, because this time all four
> lanes were exactly equal length (all hitting the 256 token cap). The docs themselves also say "the cost of a round equals the cost of its longest member:
> a pair whose output lengths differ by 2× only measured 1.41×"—**2.80× conflated round cost with imbalance in lane completion time**.
> The equal-length measurement is the pure upper bound on round cost; real workloads aggregate below this, depending on length dispersion.

### 3.2 Where the marginal 1.85 ms/column goes

The separable parts (estimates, at 448 GB/s and a measured 7 GB/s link):

| Item | Per column/round | Basis |
|----|---------|------|
| GDN linear attention state read/write | ~0.33 ms | the 73.4 MiB/card/lane state image, read+write |
| allreduce payload | ~0.19 ms | hidden 5120 × 2 B × 2 delta × 64 layers = 1.31 MB/column, link ~7 GB/s |
| full attention KV read | ~0.07 ms (499 tok context); ~0.62 ms at 4,543 tok | see below |
| unexplained | ~1.2 ms | see below |

KV reads are not the main cause: taking a single lane's context from 499 → 4,543 token moved the round only from 26.5 → 27.12 ms
(Δ0.62 ms / +4,044 token, i.e. 0.153 µs/token). The KV byte count grew 9× while the round cost grew only 2.3%,
so even at 4.5k context a KV read is only ~2% of a round.

The remaining ~1.2 ms/column **cannot be separated with the existing instrumentation**: the plain batch route has no round timing (see 3.4),
so the candidates are per-lane host-side work (sampling config staging, D2H, commit, publish, session bookkeeping)
and per-column inefficiency in the batch kernels (split policy, grid shape).

### 3.3 Speculative route: batching has a smaller marginal gain

Measured in tp2-plan-concurrency.md:441-462 (not re-measured in this report):

| Route | C=2 | C=4 |
|------|-----|-----|
| plain (equal length) | 1.88× | 3.31× |
| MTP K=5 | 1.48× | 2.26× |
| DFlash2 draft=5 | — | 2.07× |

Reason: single-request speculative decoding already amortizes the weight stream over K+1 columns, so batching can only amortize the lane dimension further.
**This is a design tradeoff, not a defect**: use `--spec` for single-user latency, concurrency for multi-user throughput.

Side effect recorded in the same doc: the MTP batch chain's envelope is **a single value for the whole batch** (max_position + step),
so MTP heads of non-longest lanes drift and the acceptance rate drops (lane1 34/62 vs 36/52 when run alone).
The author attributed it to the envelope; this run did not isolate it independently (it could also be a prompt content difference), so it is listed as to-be-confirmed.
If confirmed, the fix is to pass per-lane positions/envelope into the batch chain (the batch chain is already an eager loop).

### 3.4 Measurement defect: the plain batch route has no round instrumentation

> **Fixed (item 1)**: the plain route has dedicated instrumentation `[tp2-time] plain-batch … decode=/grammar=/sample=/readback=/prefill_chunks=/pump_rounds=`. See §8.2.

`execute_plain_batch` (:3142-3867) has only
`timing.committed += …; ++timing.rounds;` at :3837-3838 and `timing.report("plain-batch")` at :3847,
**with no `timing.record(…)` / `close_round(…)`**. So every plain run with C≥2 prints:

```
[tp2-time] plain-batch rounds=375 committed=375 avg_round=0.00ms mtp=0.00 verify=0.00
accept=0.00 copy=0.00 sync_wait=0.00 fold=0.00 prefill=0.0ms
```

The spec route (:4541/4697/4716/4779/4787/4978/5075/5090-5100) and the single-lane walk (:7340-7902)
both have instrumentation. **Every plain-route aggregate number in the docs comes from this uninstrumented path**,
which both hinders further optimization and makes production problems hard to localize. Fix: copy the 7 events of the spec route.

### 3.5 Recommendations (decode)

1. Add round instrumentation to the plain batch (low cost, unlocks all subsequent decomposition).
2. After decomposing that ~1.2 ms/column, attack the big components:
   - If it is host-side: move each lane's commit/publish/sampling staging off the critical path, or overlap it with the next round's GPU work.
   - If it is the kernel split policy: note that `causal_attention_chunk_tokens` for q_heads≠24 (TP-2 shard geometry)
     always returns 6 (src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp:25-35),
     i.e. on TP-2 no split balancing is done for the batch shape.
3. Do not count on eliminating the weight stream; that is the amortized part and is already a source of gain.

## 4. prefill speed

### 4.1 Cost model

- Single lane 4,543 token: prefill 2,264.6 ms = **0.498 ms/token (2.01k tok/s)**.
- Single lane 10,826 token: 1.99k tok/s; 10,845 token (dflash2 route): 1.69-1.70k tok/s.
- Transport marginal slope 0.368-0.375 ms/token (independent of chunk width) ⇒ **card 2's Gen4 x4 link (~7 GB/s)
  gives a 2.67k tok/s aggregate ceiling**.
- The shipped `--prefill-overlap 256` (N=4, 64-aligned sub-blocks, AR on
  `DeviceContext::collective_stream`) took 0.587 → 0.516 ms/token (−12%);
  that optimization is inside `forward_tp2_prefill` (text.cpp:2661), so the batch route's inline prefill benefits equally.
- Remaining headroom: 0.498 − 0.368 ≈ **0.13 ms/token (26%)**, i.e. hiding AR completely under compute.
  That corresponds to a single-lane ceiling of ~2.7k tok/s.

### 4.2 Why "batched prefill" cannot raise the aggregate throughput of long prompts

The author's estimate in `tp2-plan-concurrency.md:446`: "if prefill were batched too, B=2→1.68×, B=4→2.72×".
**That estimate only holds for prompts shorter than one chunk**:

1. The allreduce bytes per token do not shrink with batching (linear in the T dimension), so the total transfer time is unchanged.
   Since prefill already sits at the link ceiling of 2.67k tok/s, batching cannot raise aggregate throughput.
2. Weight-read amortization has already happened: fitting chunk 512 (0.625 ms/tok) and chunk 1024 (0.587 ms/tok) gives
   ≈0.549 ms per token + ≈39 ms fixed per chunk, i.e. the weight read of a 1024-token chunk is only ~7%.
   Batching can save part of that 7%, which is not the main bottleneck.
3. The kernel level cannot do it either: the batched form requires a per-lane width ≤ `kMaximumVerifyTokens`(16)
   (causal_softmax_attention.cpp:241-242 and :412-413), `kMaximumBatchSize = 8`;
   and the batch=1 prompt kernel uses `positions[0]` as a single base (prompt_nvfp4.cuh:126).
   That is, the existing batched kernels can only do a "B lane × ≤16 token" verify window;
   supporting "B lane × 1024 token" requires **writing a new ragged large-T multi-sequence prompt kernel**.
4. Even if written, the workspace arena would have to grow accordingly: 192 MiB covers only
   ~140 MiB peak of a single-lane 1024-token chunk, and B lanes need ~4×, which would eat the VRAM headroom of 2.1.

**The real value of batched prefill is balancing lane completion time (TTFT), not throughput.**
For long prompts the aggregate ceiling is the 2.67k tok/s given by the link (1.34× the current 1.99k).

### 4.3 Hardware is the largest single lever

Card 2 sits in a chipset Gen4 x4 slot (~7 GB/s). Moving it to a CPU-attached Gen5 x8 slot raises the prefill ceiling
and also helps decode AR. This is already recorded in docs/tp2-decisions.md as the "largest single lever".

## 5. TTFT

### 5.1 Measured (plain, `--max-context 245760 --kv-dtype int8`, C=4)

`_temp/20261002-1310_ttft.ps1`:

**B1: 4 requests with different ~10,826 token prompts sent at the same time**

| Metric | Value |
|------|-----|
| client wall | 22,616 / 22,980 / 23,454 / 23,974 ms |
| engine per-lane TTFT | 5.5 s (all) |
| engine per-lane total | 22.3 / 17.1 / 12.0 / 7.1 s |
| lane admission times | 13:41:23.7 / 29.2 / 34.8 / 40.2 (≈5.5 s apart) |
| aggregate prefill | 1.99k tok/s (= single lane, no batching) |

⇒ prefill is strictly serial: the true client TTFT of the k-th lane ≈ k × 5.5 s.
The lane trace shows that 4 requests arriving almost simultaneously formed only `batch=2 capacity=4`,
and the other two joined mid-flight at `admit slot=2/3`.

**B2: long request first, short request (54 token) 400 ms later**

- Client: long 6,503 ms / short 6,098 ms.
- Short request engine line: `TTFT 69.6 ms | total 1.0s | prefill 796.6 tok/s (54 tok)`,
  i.e. it waited ~5.1 s in the queue and **only started after the long request's prefill completed**.

**B3: short request first, long request 400 ms later**

- Client: short 6,023 ms / long 6,411 ms.
- Short request engine line: `TTFT 84.5 ms | total 6.0s | decode 2.5 tok/s`.
  The first token came out at 84.5 ms, but the remaining 15 tokens were blocked by the long lane's entire inline prefill,
  dropping decode from 37.7 when run alone to **2.5 tok/s (15×)**, and total from <1 s to 6.0 s.

### 5.2 Root cause

> **Implemented (items 3/4/7)**: formation window widened to 3 ms + 50 ms cap, plain route pumps one decode round between chunks, TTFT/total accounting includes queue time. See §8.3/§8.4/§8.8.

1. **Serial prefill during initial admission**: the plain :3647-3659 / spec :4401-4413 loops call
   `admit_lane` sequentially for each member, consuming no queue and doing no decode during that time. The k-th lane's TTFT = k × prefill.
2. **Mid-flight admission blocks decode for an entire stretch at the top of a round**: plain :3695-3725 / spec :4484-4504
   call `admit_lane` inside `while (active.size() < lane_capacity)` at the start of each round; the newcomer's
   **entire prefill** (~5.5 s at 10.8k token) runs synchronously on the compute stream, and
   all surviving lanes' decode stalls completely during it (the 15× collapse of B3).
   Note that admission is only checked at round boundaries of the main decode loop, and is not checked during initial prefill.
3. **The batch formation window is only 3 ms**: `kBatchFormationWindow{3000}` (:2250),
   `lane_queue_cv_.wait_for(queue, kBatchFormationWindow)` (:2291).
   So 4 requests sent at the same time are cut into 2+2; the last two can only start after the first two's initial decode round.
4. **TTFT is measured relative to admission**: the engine's TTFT/total clock starts when the lane is admitted and
   excludes queueing and serialization. **Under TP-2 multi-lane, the client-perceived TTFT can be up to ~4× the logged value**
   (B1: log 5.5 s, client ~23 s). This distorts any capacity planning based on logged TTFT.

### 5.3 Recommendations (TTFT, ordered by value for cost)

1. **Interleave a mid-flight newcomer's prefill with the surviving lanes' decode** (highest value).
   The current behavior is "the entire prefill blocks". Workable intermediate forms:
   - In `admit_lane`'s chunk loop, return to the main loop to run one decode round after each chunk
     (i.e. split `admit_lane` into a state machine of "prepare + push one chunk per round").
     Cost: the chunk granularity determines the blocking duration, and 1024 token ≈ 0.5 s is still too coarse.
   - Under mixed load use a smaller chunk (e.g. 256 → ~0.15 s of blocking), and 1024 when there are no surviving lanes.
     Per the fit in 4.1, chunk 256 costs 0.549 + 39/256 = 0.701 ms/token (−28%),
     but that only happens inside the mixed-load window.
   - More thorough: put the newcomer's prefill on a second stream to truly parallelize with decode. Decode's AR uses only
     ~6% of the link (B=4 round 5.24 MB / 32 ms ≈ 164 MB/s vs a 7 GB/s link), so the link has slack;
     the real contention is the weight stream (both read all weights, sharing 448 GB/s).
     The result is conserved total throughput, but the newcomer's TTFT goes from "wait for a round" to "its own prefill time".
2. **Also check the queue during initial prefill**: in the loop at :3647-3659, call `try_pop_lane_queue` after each member's
   prefill completes, so that B1's 2+2 becomes fully formed at once.
3. **Widen the batch formation window**: change "a fixed 3 ms wait once" to "as long as `batch.size() < lanes_`
   and the queue is non-empty, keep waiting up to a small cap (e.g. 20-50 ms)". When the queue is empty this adds no latency
   (:2284-2287 already sleeps when idle).
4. **Fix the TTFT accounting**: include pre-admission queueing time in the externally reported TTFT (or additionally emit a
   `queue_ms` field); otherwise multi-lane latency data is unusable.
5. **prefix reuse remains the largest single TTFT lever** (already shipped): a 10,240 token hit drops
   cold prefill from 6.4 s to 0.45 s. The known remaining gap: at C≥2, `finalize` does
   `release_lane_kv → invalidate_host_checkpoints(lane)`, wiping the grid ring at the decode tail,
   leaving only the catalog's frontier / prompt-end / block-shared mirrors
   (multi-turn sessions and shared system prefixes still hit; the middle-segment-only-sharing case does not).
   Closing it needs another class of `SessionEntry` mirror, and the host KV budget bounds how many
   fit: the §2.1 configuration (`--host-kv-mib 20480`) leaves 10,240 MiB/shard, two whole-session
   mirrors of about 3,960 MiB, while the shipped `--host-kv-mib 32768` leaves 16,384 MiB/shard, four.

## 6. Recommendation list

| Priority | Item | Type | Expected gain | Cost | Status (§8 is this execution record) |
|--------|-----|------|----------|------|--------------------------|
| P0 | Interleave mid-flight admission's prefill with decode (state machine / small chunk / second stream) | Latency | Order-of-magnitude improvement in mixed-load TTFT; eliminates the 15× decode collapse | Medium-high (scheduler refactor) | **Implemented** (plain route pump callback, one decode round per chunk): at chunk 256, the short lane is 2.8 s / 8.5 tok/s. §8.4; **the second round also covered the speculative route** (spec/MTP pump): at chunk 256, the short lane is 2.0 s / 15.7 tok/s. §9.1 |
| P0 | Also consume the queue during initial prefill | Latency | Burst requests no longer split 2+2 | Low | **No change needed**: the existing mid-flight admission already covers it; the 2+2 split was fixed by the formation window. §8.5 |
| P0 | Widen the batch formation window (keep waiting up to the cap while non-empty) | Latency | Bursts fill the batch | Low | **Implemented** (3 ms window + 50 ms cap): four concurrent went from `batch=2` to `batch=4`. §8.3 |
| P1 | The plain route does not carve the round-scratch plane | VRAM | C=4 saves 293.6 MiB/card (587 MiB across two cards) | Minimal (a few lines) | **Implemented**: startup ledger `state 293.6` (4×73.4), `round 0.0`. §8.1 |
| P1 | `--lane-context` defaults to pool/lanes with multiple lanes, or at least warns | Concurrency | Avoids a single request monopolizing the pool and degrading concurrency | Low | **Implemented (warning only)**: a hard default would reject prompts longer than 1/lanes. §8.8 |
| P1 | Add round instrumentation to the plain batch | Measurement | Unlocks decomposition of the decode marginal cost | Low | **Implemented**: `[tp2-time] plain-batch … decode=/grammar=/sample=/readback=/prefill_chunks=/pump_rounds=`. §8.2 |
| P1 | Add queue time to the TTFT accounting | Measurement | Multi-lane latency data becomes usable | Low | **Implemented**: `ttft`/`total` start from core acceptance, the same accounting as `| queue`. §8.8 |
| P2 | Shrink the output budget dynamically by the remaining amount / page reclamation | VRAM/concurrency | Measured 64 pages = 66 MiB/card idle occupancy | Medium | **Analyzed, not changed**: breaks the all-or-nothing reservation invariant and would introduce mid-flight growth deadlock. §8.6 |
| P2 | Fix the 262,144/C=1 guidance in docs/serving.md | Documentation | Avoids misleading deployments | Low | **Implemented**: and filled in the 1/2/3/4/6/7 descriptions. §8.9 |
| P2 | Switch the speculative batch chain to a per-lane envelope (after confirming the acceptance attribution) | Throughput | MTP C≥2 per-lane acceptance rate | Medium | **Implemented (numerical consistency)**: each column in-batch is bit-identical to standalone decode (new parity regression test); the acceptance-rate gain did not reproduce, plain decode is neutral. §8.7 |
| P3 | Batched prefill (needs a new ragged large-T multi-sequence kernel + a larger workspace) | TTFT balance | Only balances lane completion time; aggregate throughput ceiling 1.34× | Very high | **Measurement confirms not doing it**: with attention batch>1 each lane's width ≤ 16 and GDN must record+fold; the measured batch window W=16/B=4 = 1.24 ms/lane-token vs single-lane 1024-chunk 0.587 ⇒ **2.1× slower** (breaking even needs W≈300). §9.4 |
| P3 | Make the host checkpoint ring pageable / reduce slots | Host memory | Saves GiB-level pinned host memory | Low-medium | **Implemented**: ring slots changed to pageable, saving 5.29 GiB of page-locked memory across two cards; under the same harness the recall request is +10 ms (within noise). §9.5 |
| — | Move card 2 to a CPU-attached Gen5 x8 slot | Hardware | prefill ceiling + decode AR | Hardware swap | Not done |

## 7. How to reproduce

Scripts (in the workspace root `_temp/`, named `<YYYYMMDD-HHmm>_<desc>`):

- `_temp/20261002-1310_scale.ps1` + `_scale.mjs`: C=1..4 equal-length lane decode scaling.
- `_temp/20261002-1310_ttft.ps1` + `_ttft.mjs`: the three B1/B2/B3 TTFT serialization experiments.
- `_temp/20261002-1350_mem.ps1`: VRAM budgets for C=1/2/4 and plain C=4 at 262,144.
- `_temp/20261002-1420_ctx.ps1` + `_ctx.mjs`: prefill and decode at 4,543 token context for C=1/C=4.
- `_temp/20261002-1440_shipped_c4.ps1`: end-to-end verification of the shipped 262,144 + C=4 configuration.
- `_temp/20261002-1442_pump.ps1` + `_pump.mjs`: item 4 interleaving measurements (A=4 long concurrent / B=long-then-short / C=short-then-long / D=2+2),
  once with `--prefill-chunk 1024` and once with `256`.
- `_temp/20261002-1550_mtp.mjs` (with `_mtp5.ps1`): item 10 attribution—S=single long run alone, A=4 long in one batch, B=long+short.
- `_temp/20261002-1620_mtpshort.mjs` (with `_mtpshort.ps1`): item 10 discrimination—short prompts entering the batch at the same instant (C=1/2/4).
- `_temp/20261002-1630_item10.ps1`: item 10 same-generation A/B (`-Bin oldops|newops`, both the long and short arms).
- `_temp/20261002-1745_specpump.ps1` + `_specpump.mjs`: speculative-route prefill interleaving A/B (`-Bin head|pumped`,
  short-first/long-first arms, chunk 1024 and 256).
- `_temp/20261002-1700_ringio.ps1` + `_ringio.mjs`: ring pinned/pageable comparison (`[tp2-kv]` store/recall
  and the TTFT/total of requests with 98.2% recall hits).

Environment notes (pitfall log):

- **There is no `--spec none`**: passing it prints `invalid speculative backend: none` and exits;
  omitting `--spec` is the plain route.
- `--chat-template <models-dir>/chat_template.jinja` must be passed: the artifact's bundled template does not accept
  the Developer role that `instructions` maps to (src/serve/openai_responses_state.cpp:132-141),
  otherwise every /v1/responses is rejected with 400.
- The model is a **positional argument**, and PATH must contain the bin directories of FFmpeg and libcurl.
- **MTP's VRAM accounting**: `--max-context 245760 --kv-dtype int8 --spec mtp` exits 39 s after startup with
  `cudaMalloc failed: cudaErrorMemoryAllocation`; MTP measurements must use the documented accounting
  `--max-context 131072 --kv-dtype int8` (which is what §8.7 of this report uses).
- `_temp/20261002-1310_scale.ps1` builds the client script name from `-Stamp`, so
  `20261002-1310_scale.mjs` must first be copied to `<Stamp>_scale.mjs`, otherwise node cannot find the script and the done line is empty.
- Both cards must be idle; the service process is globally unique, so stop the service before relinking.

## 8. Execution record of this round (2026-10-02)

Executed the §6 list item by item. Source changes are concentrated in `src/runtime/engine/tp2_generation_core.cpp/.h` and `docs/serving.md`;
the build entry point is `tools/win_port/build.ps1` (incremental 27-29 s, `BUILD_EXIT=0`), the test entry point is
`tools/win_port/test.ps1 -Filter 'tp2|tp_device|engine_options|serve_options'` → `100% tests passed out of 12`
(6 model-dependent TP-2 tests are skipped because there are no local weights). Measurement environment: 2×RTX 5060 Ti, `--devices 0,1`, int8,
prompt 15,310 / 54 token, `max_output_tokens 24`, `NINFER_TP2_TIMING=1 NINFER_TP2_LANE_TRACE=1`.

### 8.1 Item 2: the plain route no longer carves the round-scratch state plane

- Basis: `state_snapshots[kRoundScratchSlot]` is referenced only on the speculative route
  (`execute_spec_batch` :4701/:4886/:5079, `execute_walk`'s DFlash2/MTP branches :7761/:7804),
  with zero references in `execute_plain_batch`; `snapshot_state` (whole-pool snapshot :6875-6880) has **no null-pointer guard**,
  so the gate must be exactly `mtp_enabled_ || dflash2_enabled_` (`snapshot_lane_state` has its own null-pointer guard at :6889).
- Measured: startup ledger `shard N … | state 293.6 | … | round 0.0 | … | free 1414.0 of 16310.6 MiB`,
  with `round` dropping from 293.6 before the change to 0.0 ⇒ C=4 saves 293.6 MiB per card. `state 293.6` is the per-lane GDN state
  (4×73.4), unchanged.

### 8.2 Item 1: round instrumentation for the plain batch

Added `Tp2RoundTiming::close_round_plain` / `report_plain`; the plain route no longer borrows the speculative route's
`mtp/verify/accept/copy` phase names. Measured (chunk 256):

```
[tp2-time] plain-batch rounds=69 committed=69 avg_round=112.17ms decode=27.20 grammar=0.00 sample=0.04 readback=0.20 sync_wait=0.03 prefill_chunks=180 pump_rounds=46
[tp2-time] plain-batch rounds=46 committed=46 avg_round=26.76ms decode=15.38 grammar=0.00 sample=0.03 readback=0.10 sync_wait=0.01 prefill_chunks=61 pump_rounds=1
[tp2-time] plain-batch rounds=46 committed=46 avg_round=64.70ms decode=27.89 grammar=0.00 sample=0.04 readback=0.18 sync_wait=0.02 prefill_chunks=61 pump_rounds=14
```

`rounds` counts **committed columns**, so `avg_round` is the per-lane share of the single-round cost (comparable with the speculative route's
`verify` accounting); `pump_rounds` is "the number of decode rounds run between another lane's prefill chunks".

### 8.3 Item 3: widen the batch formation window (3 ms + 50 ms cap)

The formation logic becomes a `take_from_queue()` loop + `formation_deadline = now + 50 ms`: as long as the queue is non-empty and the batch is not full,
keep waiting through another 3 ms window. Measured: four long concurrent requests went from the report baseline `[tp2-lane] batch=2 capacity=4` to
`batch=4 capacity=4` (chunk 1024); at chunk 256 it is still `batch=3` + a second wave of 1 (the 50 ms cap
does not span the 800 ms gap of the 2+2, so they are correctly not merged). A single request only pays one extra 3 ms window.

### 8.4 Item 4: pump one decode round between prefill chunks (plain route)

**Policy**: only one lane is prefilling at any instant (preserving FIFO order and each lane's TTFT), and lanes that already hold tokens
run **one decode round** between two chunks of that lane. Round-robin prefill was rejected: it would make all
lanes finish prefill at the same wall-clock instant, degrading lane 0's TTFT from ~5.5 s to ~22 s.

**Implementation**: the round body is extracted into `auto pump_decode_round = [&]()` inside `execute_plain_batch` (defined after
`active.reserve` and before `admit_lane`, so admit_lane can call it directly); at the top of `admit_lane`'s chunk
loop, after the previous chunk's `DeviceArena::Scope` destructor, it is called and
`set_linear_state_slots(active_lane_, active_lane_)` is restored (the one-column round's `batch == 1`
branch replays the scalar slot pair, src/models/qwen3_5/execution/text.cpp:2370-2378). The `round_base` capture point is moved
after the pump definition (the cached CUDA Graph bakes `arena_begin`; an inconsistent layout throws
`"TP-2 batched decode CUDA Graph replay found a different workspace layout"`).

**Measured** (long 15,310 token, short 54 token submitted 400 ms earlier):

| Case | Short lane total | Short lane decode | Mechanism |
|------|---------------|----------------|------|
| Report baseline (no interleaving, B3 10.8k) | 6.0 s | 2.5 tok/s | entire prefill blocks |
| Interleaved, chunk 1024 | 7.5 s | 3.1 tok/s | 1 round per chunk (~0.55 s) |
| Interleaved, chunk 256 | **2.8 s** | **8.5 tok/s** | 1 round per chunk (~0.19 s) |

- The reverse order (long first, short second) still has to wait: the short lane has `queue 9.8-9.9 s`, TTFT 9.9 s, whereas with short first the TTFT is 114 ms.
  This is the inherent cost of "single-lane serial prefill"; item 4 does not change the admission order.
- The instrumentation proves the mechanism directly: short first (the long lane has 61 chunks, `pump_rounds=14`—the pump stops after the short lane
  takes its full 24 tokens and leaves `active`); long first (`pump_rounds=1`); four long concurrent (`prefill_chunks=180 pump_rounds=46`).
- Cost: chunk 256's per-lane prefill rate is 1.34-1.52k tok/s vs 1.69-1.94k for chunk 1024,
  consistent with the §4.1 cost model (small chunks trade prefill rate for decode smoothness). **Only the plain route was changed**;
  the speculative route still prefills serially per lane (remaining headroom).
- Client SSE deltas are mostly 0 text deltas under reasoning + a small max_output, so they cannot serve as a token clock;
  the authoritative data is the `done |` and `[tp2-time]` lines.

### 8.5 Item 5: consuming the queue during initial prefill—**no change needed**

It was once implemented (a `try_pop_lane_queue()` at the end of the S1 loop) and then reverted: S1 admits in `batch.size()` order, and after the
loop the main loop **immediately** executes `while (active.size() < lane_capacity) { try_pop_lane_queue(); …; admit_lane(…); }`,
so the two prefill new members at the same wall-clock instant and equally strictly serially, with no TTFT difference; the only difference is which
`batch` vector the members live in. The report's premise for this item (the queue is not checked during initial prefill) is in fact already covered by the existing mid-flight admission;
the 2+2 split was fixed by item 3.

### 8.6 Item 9: output budget pre-reservation—**analyzed, not changed**

`lane_kv_pages` (:3023) = `pages_for_tokens(need + lane_kv_margin())`,
`reserve_lane_kv` (:3027) performs a **full all-or-nothing** reservation across both cards + the MTP cache before admission,
`lane_need_tokens` (:3202) = `prompt + budget - 1`. Design comment :3004-3010, verbatim: "The reservation is
all-or-nothing and happens before the request is admitted, so a pool that cannot cover it right now
leaves nothing behind … the wait is always for another lane to retire, never a deadlock."
Shrinking/reclaiming by the remaining amount would make a lane need to **grow** mid-decode, and that route has no mid-request backpressure or preemption
(AGENTS.md: no active-request preemption), so a safe implementation would need to add an "evict on demand + grow a contiguous KV window" path,
which is not P2 magnitude. The 66 MiB/card idle upper bound is determined by the client's `max_output_tokens`; the operational levers are
`--lane-context` or a smaller client budget.

### 8.7 Item 10: per-lane acceptance rate of the speculative batch chain—**numerical mechanism confirmed, acceptance-rate gain did not reproduce; fix implemented**

Configuration: `--max-context 131072 --kv-dtype int8 --spec mtp --draft-tokens 5` (245760+int8 OOMs at startup).
The `mtp accepted X/Y (P%)` newly added to the `done |` line gives the per-lane acceptance rate directly.

**Old ops baseline (with batch truncating the host split count)**:

| Group | lane | Acceptance rate |
|----|------|--------|
| Single long run alone (C=1) | — | 40/75 = **53.3%** |
| Four long in one batch (C=4) | each lane | 48.2 / 64.9 / 39.4 / 45.9% (mean 49.6%) |
| Long + short (C=2, position gap 15k) | long / short | 47.5% / 46.7% |
| Single short run alone | — | 54/145 = 37.2% |
| Two short in one batch (C=2) | each lane | 53.3 / 54.1% |
| Four short in one batch (C=4) | each lane | 47.1 / 52.8 / 53.6 / 46.7% (mean 50.1%) |

**The numerical-level mechanism is confirmed**: on old ops, "the same lane run in-batch" and "run alone" are **bit-different**—
`causal batch/solo parity d256-h24-kv4 bf16 lane=0: element 4 batched=0x3a20 solo=0x3a1f`
(1-3 ULP difference across all 5 storages × 4 lanes, OPS_EXIT=1). The cause is that
`causal_attention_split_capacity` (src/ops/softmax_attention/dense/causal_cache/small_t.cu:217-247)
truncates the host-side split count to `grid_limit/page_limit` when batch>1, while the device side computes
`active_split_count = min(per-column natural split count, host value)` (small_t.cuh:119)—so each column's reduction tree is shortened,
differing from standalone decoding of the same column.

**Fix (A1)**: `small_t.cu` directly `return capacity` (capacity is already the envelope supremum),
`causal_attention_split_capacity` drops its `batch_size` parameter (launch.h:26-29), and 11 call sites are updated in step.
A new regression test `run_batch_single_parity_case` (tests/ops/softmax_attention/causal_cache.cpp,
batch=4 / width=4 / context=4100 / EXACT envelope) pins down "in-batch lane == run alone" bit-for-bit equality:
new ops PASS (`nvfp4 mae=0.00536884`, `k8v4 mae=0.0053744`, no regression against the tolerance oracle), old ops FAIL.

**But the acceptance rate showed no reproducible gain** (same harness, same-generation A/B, the only difference being the 6 files under src/ops):

| Configuration | Old ops | New ops |
|------|--------|--------|
| plain decode C=1/C=2/C=3/C=4 (per-lane tok/s) | 36.8 / 35.0·35.9 / 31.9·32.7·33.6 / 29.9·30.6·31.3·32.2 | 37.2 / 35.0·35.8 / 32.0·32.8·33.7 / 29.9·30.6·31.4·32.3 |
| plain `[tp2-time] avg_round` C=1..4 | 27.15 / 13.89 / 9.97 / 7.84 ms | 26.85 / 13.92 / 9.92 / 7.85 ms |
| spec short arm P4 per-lane round cost mtp+verify | 21.05 ms | 23.57 ms |
| spec long arm four long per-lane round cost mtp+verify | 51.65 ms | 50.11 ms |
| spec acceptance short arm P4 mean | 50.1% | 51.4% |
| spec acceptance long arm four long mean | 49.6% | 40.3% |
| spec acceptance **single long (control, no code-path change)** | **53.3%** | **41.0%** |

⇒ The single-run control itself drifted by 12 points, so the entire acceptance-rate difference in this A/B falls within run-to-run noise and
**cannot prove** an acceptance-rate gain; plain decode is completely neutral, spec round cost −3%/+12%.

**Conclusion**: item 10 is kept, but its value is **numerical consistency**—each column in-batch is bit-identical to standalone decoding, guarded by the parity test;
it is not an acceptance-rate or throughput optimization. The wording of §0/§6 was rewritten accordingly.

- **Open question** (not isolated): whether the trigger for the split change is "the window's absolute width" or "the position gap between lanes"—short-window experiments
  can only rule out batching itself; constructing a wide window with zero position gap requires admission to no longer be serial (i.e. P3 batched prefill, but per §9.4's
  measurements it is 2.1× slower) or an in-kernel probe.
- envelope comment location (reference corrected): it is currently at src/runtime/engine/tp2_generation_core.cpp:4436
  (verbatim: "The envelope steers split policy only; the kernels take their visible set from the
  per-lane positions, and the widest lane bounds every lane's window.").

### 8.8 Item 6 / item 7

- Item 6: with multiple lanes and `--lane-context 0`, startup prints
  `[mem] TP-2 lane admission ceiling 245759 tokens (the whole KV pool): one lane may reserve all of it, so --max-concurrency 4 requests cannot be admitted together until it retires; set --lane-context to cap one lane`.
  No hard default is applied—the default pool/lanes would make prompts longer than 1/lanes fail directly with `context_length_exceeded`.
- Item 7: `submit()` records `request->submitted`, both admit_lane sites set
  `lane.result.engine_timing.queue_wait_seconds`, and both finalize sites plus `execute_walk` change
  `first_token_seconds`/`total_seconds` to `prepare_seconds + … + queue_wait` (aligning with the single-card accounting,
  src/runtime/engine/engine_core.h:873-882). This one is required: src/serve/generation_service.cpp:447-457
  computes `ttft = prepare + max(0, first_token - prepare)`, `total = prepare + max(0, total - prepare)`.
  `prefill_seconds`/`prompt_wall_seconds` **remain** counted from admission (they define the prefill rate).
  Measured: the `done |` line gained a `| queue 7.9s` field, etc., self-consistent with the server-side `total`.

### 8.9 Item 8: docs/serving.md

Added: formation window behavior (item 3), pump interleaving and the measured numbers (item 4), the plain `[tp2-time]` line (item 1),
the `state/round` ledger change (item 2; the original text already noted that round-scratch exists only on the speculative route), the startup warning (item 6),
and the TTFT/total accounting (item 7). Verification: `git diff --check` reports no whitespace errors.

### 8.10 Change list

- `src/runtime/engine/tp2_generation_core.cpp`: 24 hunks, all for the items above (including adding a
  `submitted` timestamp to `TP2GenerationCore::Request` in the `.h`).
- `src/runtime/engine/tp2_generation_core.h`, `docs/serving.md`, this report,
  `tp2-plan-headroom-execution.md` (the per-item execution plan and record).
- Not done (recorded as remaining headroom): P3 batched prefill (§9.4, measured 2.1× slower), swapping card 2's PCIe slot.

## 9. Second-round execution record (2026-10-02, four items added by the user)

User's second-round additions (m01111): prefill interleaving for the speculative route (spec/MTP); the item 10 Ops kernel per-column envelope/split
policy fix; P3 batched prefill (incidentally eliminating the admission position gap); a pageable host checkpoint ring.
The per-item plan and design are in the "User's second-round additions" section of tp2-plan-headroom-execution.md.

### 9.1 Item A: prefill interleaving for the speculative route—**implemented and measured to hold**

The plain route's pump callback (§8.4) was moved to the spec route as-is: the round body of `execute_spec_batch` is extracted into
the `pump_spec_round` lambda (src/runtime/engine/tp2_generation_core.cpp), and at the top of `admit_lane`'s chunk
loop, after "the previous round's chunk scope has returned to the arena", it is called; the main loop keeps only the retire preamble +
mid-flight admission + `pump_spec_round()`. The policy is unchanged: **only one lane prefills at a time**, and lanes already decoding
run one decode round between its chunks. Because only one lane prefills, the single-instance constraint of visual sessions is unaffected;
the pump does not retire/admit, so it will not recurse back into admit_lane.

A/B (the same harness `_temp/20261002-1745_specpump.ps1` + `_specpump.mjs`, binaries differing only in this item):

| Configuration | HEAD (no pump) | With pump |
|------|-----------------|---------|
| chunk 1024, ARM C short lane (54 token) | total 8.8 s / decode 3.1 tok/s | **total 3.3 s / 8.9 tok/s** |
| chunk 256, ARM C short lane | total 11.1 s / decode 2.5 tok/s | **total 2.0 s / 15.7 tok/s** |
| chunk 1024, ARM C long lane (15,310 token) | 9.0 s | 9.5 s |
| chunk 256, ARM C long lane | 11.4 s | 12.1 s |
| ARM B later-arriving short lane (queue) | 7.6 s / 10.0 s | 7.6 s / 10.0 s (unchanged) |

- The direct evidence is the new counters `prefill_chunks`/`pump_rounds`: at chunk 1024 the long lane's prefill is split into
  16 chunks, during which 6 rounds were run for the already-decoding lane (`[tp2-time] spec-batch ... prefill_chunks=16 pump_rounds=6`).
- The long lane is 0.5-0.7 s slower (~5%), which is the insertion cost of the pump rounds.
- **run-to-run jitter is large** (the same configuration re-measured the short lane at 3.3 s → 5.6 s), so the conclusion rests primarily on the `pump_rounds` counter and
  the chunk 256 numbers.
- In ARM B the later-arriving short request still queues (7.6/10.0 s unchanged)—this is the inherent
  cost of the "only one lane prefills at a time" policy, not a defect.

### 9.2 Item B: spec route timing accounting—**implemented**

`Tp2RoundTiming` drops the always-zero `fold_ms` and the 3 `timing.fold_ms += timing.elapsed(5, 6)`
(a query for an unfinished event within a round is always 0), and the 5th field of `report()` becomes `rest=%.2f` =
`(round_ms − mtp − verify − accept − copy) / rounds`; `pump_spec_round` adds the round's
`round_start`/`timing.round_ms +=`, so the spec line now uses the same accounting as the plain line (it used to be always `avg_round=0.00`).

### 9.3 Item C: item 10 Ops fix—see §8.7

The fix is implemented (numerical consistency: each column in-batch is bit-identical to standalone decoding), the acceptance-rate gain did not reproduce; the parity test pins the invariant.

### 9.4 Item D: P3 batched prefill—**confirmed not doing it after investigation + measuring the width cost**

Design investigation conclusion (_temp/20261002-1800_item3_batched_prefill.md, 161 lines):

- The only reusable multi-lane×multi-token entry point is `TextContext::forward_tp2_window_batch`
  (src/models/qwen3_5/execution/text.cpp:2939-3100): it already has per-lane `kv_table_rows`/`state_slots`
  [batch] and per-lane `valid_columns` [batch] (`width=max(chunkLen)`, `valid_columns[i]=chunkLen[i]`
  suffices to express unequal-length chunks). Neither `forward_tp2_prefill` nor `forward_tp2_decode_window_batch` is usable.
- Two hard constraints: with attention batch>1 each lane's **width ≤ 16** (src/ops/softmax_attention/dense/causal_cache/
  causal_softmax_attention.cpp:241-243, kMaximumVerifyTokens=16), and batch>1 **never selects the Prompt route**;
  the TP-2 GDN non-record path **forces width==1** for batch>1 (text.cpp:1748-1749, :1917-1918),
  so batched prefill must go through `GdnStateAction::RecordForReplay` + fold.
- Masking is not the problem (lane isolation relies on per-lane KV rows, the in-lane causal window relies on device-side pos, and the clamped tail columns are not appended);
  the envelope is only a launch/workspace promise, not a mask.
- Value assessment: with width ≤ 16, "fewer kernel launches" does not automatically hold; the **certain gain is only constraining the admission position gap to within the
  chunk width** (converging MTP's single-valued envelope). And `forward_tp2_window_batch` uses the non-overlap layer loop
  (text.cpp:3076/3078), which would lose the +11% of L2 allreduce‖MMA overlap (docs/tp2-decisions.md:281-285).

**Measured width cost (deciding whether the minimal slice is worth writing)**. Before writing the slice, measure the slice's **only newly added op**—one
B lane × W column `forward_tp2_window_batch` (including the model's internal GDN record path)—to get its real cost.
The `verify` of `[tp2-time] spec-batch` = `close_round`'s elapsed(1,2), which exactly encloses that call. Configuration:
artifact <models-dir>/qwen3_8_27b_swift15_dflash2_final.ninfer, TP-2 dual card, int8, 131072, C=4,
prefill-chunk 1024, `--spec dflash2 --draft-tokens 5|15`, ARM P4 four short in one batch (harness
_temp/20261002-1925_w16probe2.ps1). Note that MTP cannot reach width 16
(src/product/speculative_options.h:41-43 limits draft-tokens ≤ 5); only dflash2 (≤15) can.
Accounting: `++timing.rounds` increments by 1 per committed column (src/runtime/engine/tp2_generation_core.cpp:3590 and :4815),
and a round has B lanes each committing one column ⇒ per-round cost = reported value × B (B=4).

| Configuration | avg_round per round | mtp per round | verify per round | verify per column |
|---|---|---|---|---|
| W=6 (drafts 5) | 80.9 ms | 17.4 ms | 58.8 ms | 2.45 ms |
| W=16 (drafts 15) | 108.6 ms | 24.0 ms | 79.6 ms | 1.24 ms |

Fit: **verify(W, B=4) = 46.3 ms fixed + 0.52 ms/column**; against the measured single-lane prefill fit in §4.2 of
**39 ms/chunk fixed + 0.549 ms/token**. ⇒ batched prefill W=16,B=4 = **1.24 ms/lane-token**,
versus the existing single-lane 1024-chunk = **0.587 ms/lane-token**, **batching is 2.1× slower**. Mechanism: the fixed cost (about 40 ms of
weight stream) is amortized over only W×B=64 columns in a batch round, versus 1024 columns for a single-lane chunk (a 16× difference). To break even, batching needs
W×B ≈ 1216 columns/round, i.e. W≈300 at B=4, which is 19× the ops limit `kMaximumVerifyTokens=16`.
⇒ Option A (engine-side minimal slice) has a negative expected value; the only region where it could win is prompt < ~56 token (which is already one
chunk); actually capturing the benefit of batched prefill first requires option B (relaxing the ops batch domain, 400-800 lines, the highest risk),
and §4.2 already argued that even then it cannot raise the aggregate throughput of long prompts (the link ceiling). **This item is wrapped up as not doing it**,
kept as remaining headroom.

### 9.5 Item E: pageable host checkpoint ring—**implemented and measured**

The ring slots were changed from `PinnedHostBuffer` (cudaHostAlloc Portable|Mapped) to `HostBuffer`(`HostPinning::Pageable`),
and the image-passing path was unified as a raw mapping + length (the `frozen` parameter of `store_dflash_image`/`load_dflash_image`/
`session_capture_shared_state`), because the two host buffer kinds have no common base class.
The ring is not inside any CUDA Graph capture region, so being pageable does not affect capture; the HostKVArena controlled by `--host-kv-pinned`
is unaffected (it can still be pinned).

Same-harness A/B (`_temp/20261002-1700_ringio.ps1` + `_ringio.mjs`: a 13,559 token prefix +
4 recall requests with 98.2% hits, `--max-context 131072 --host-state-slots 32`, binaries differing only in the ring's pinning):

| Configuration | `[mem]` line | recall request total (mean of 4) | TTFT mean |
|------|-------------|------------------------------|-----------|
| pinned | `host-checkpoints shard 0/1 slots 36 ... | pinned 2642.6 MiB` | 495.5 ms | 296.5 ms |
| pageable | `... | pageable 2642.6 MiB` | 505.3 ms | 292.8 ms |

⇒ Across two cards, **5.29 GiB (2 × 2642.6 MiB) of page-locked memory becomes ordinary host commit**, at a recall-path cost of ≈ +10 ms
(2%, within run-to-run noise; TTFT actually dropped slightly). The `[tp2-kv]` store/recall timings measure paged KV
(HostKVArena, already pageable) rather than ring images, and are the same in both arms (store 125.7-151.4 ms / recall
67.9-80.1 ms), which serves as corroboration.

Model tests (`NINFER_TEST_ARTIFACT=<models-dir>/qwen3_8_27b_swift15_dflash2_final.ninfer`,
`tools/win_port/test.ps1 -Filter 'qwen3_5_tp2'`): of the 6 items in the final binary, 4 reached a conclusion—lanes **Passed**
(69.56 s), dflash_append **Passed** (47.51 s), sessions Failed, dflash_solo Failed, load/forward
Skipped. After stashing all changes of this round and rebuilding HEAD (464 s), re-testing with the same filter reproduced **both failures verbatim**:
sessions' `got [2752 13 198 197 197 92 198 197] expected [467 419 538 13 198 197 197 92]`,
dflash_solo's `the recalled walk diverged from the from-scratch walk on its first sample:
[59399 475 327 363 62 16 15 15] vs [365 4577 62 16 15 15 15 15]` ⇒ both are **pre-existing failures**, not regressions from this round.

### 9.6 Change list of this round

- `src/runtime/engine/tp2_generation_core.cpp`: spec route pump interleaving (`pump_spec_round` extraction +
  admit_lane calling it in the chunk loop + the main loop keeping only `pump_spec_round()`), timing accounting (drop `fold_ms`, `rest`,
  spec round accumulation), ring made pageable (allocation site, image-passing signatures, `[mem]` line printing the actual pinning).
- `src/runtime/engine/tp2_generation_core.h`: ring slot type and image-passing signatures.
- `src/ops/softmax_attention/dense/causal_cache/`: `launch.h`, `small_t.cu`, `small_t_fp8.cu`,
  `small_t_k8v4.cu`, `small_t_nvfp4.cu`, `causal_softmax_attention.cpp` (dropping batch's truncation of the host split
  count, and removing the batch parameter from `causal_attention_split_capacity`).
- `tests/ops/softmax_attention/causal_cache.cpp`: new batch/solo bit-exact parity regression.
- `docs/serving.md`: spec route interleaving, timing-line accounting, pageable `--host-state-slots`.
- Not done (remaining headroom): P3 batched prefill (§9.4, measured 2.1× slower, requires relaxing the ops batch domain first), swapping card 2's PCIe slot.
