# TP-2 cross-layer + cross-sub-block prefill pipeline (wavefront): pending design

> Status: **pending, not implemented, not scheduled.** This is a paper analysis: no measurement was
> taken for this design. Every baseline number below is either from the 2026-10-02 headroom report
> ([tp2-concurrency-headroom.md](tp2-concurrency-headroom.md) §4) or from the shipped TP-2 measurements
> in [tp2-plan.md](tp2-plan.md) §3.4 and [tp2-dual-5060ti.md](../tp2-dual-5060ti.md).
> Question (user, 2026-10-20): single-request prefill sits at 2.01k tok/s against a 2.67k link floor.
> Can the remaining exposure be removed by pipelining layers across sub-blocks, and what does it cost?

**Verdict: technically feasible; the gain is bounded and the dominant cost is not memory.** The link
floor is 0.368-0.375 ms/token (2.67k tok/s); the shipped N=4 overlap measures 0.516 ms/token on a
1,024-token chunk, so 0.13-0.15 ms/token is still exposed. A perfect wavefront would return all of it
(0.37 ms/token, +39%), but in-kernel allreduce itself occupies SMs, four stages contend for the same
36 SMs and DRAM stream, and the allreduce burst pattern gets worse — so the planning number is
**+8-15% (0.45-0.48 ms/token)**, with +20% as the optimistic case. Cost: **+128 MiB/card of workspace**
(192 → 320 MiB; 384 MiB if a stage needs 48 MiB), four compute streams, three new invariants, and a
code/verification surface that touches every prefill (single-lane and per-lane). Worth opening only
if the Step 1 gate in §8 passes.

## 1. Why the current implementation stops at 0.516 ms/token

- Single lane, 4,543-token prefill: 2,264.6 ms = 0.498 ms/token (2.01k tok/s); 10,826 tokens: 1.99k.
- Transport marginal slope is 0.368-0.375 ms/token, independent of chunk width: two allreduces per
  layer move 2.6 MB/token over card 2's chipset Gen4 x4 link (~7 GB/s) = **2.67k tok/s ceiling**
  ([tp2-decisions.md](../tp2-decisions.md) "Rejected optimizations"; [tp2-pp-prefill-pivot.md](tp2-pp-prefill-pivot.md) §1).
- `--prefill-overlap 256` (N=4 64-aligned sub-blocks, AR on `DeviceContext::collective_stream`) took
  0.587 → 0.516 ms/token (−12%) at chunk 1,024, ~6.3k prompt, median of 7. N is not interpolable:
  N=2 0.554, **N=3 0.612**, N=5 0.547, N=8 0.655; chunk 512 serial 0.625 → 256-wide 0.572 ([tp2-plan.md](tp2-plan.md) §3.4).
- Decomposition of that 0.587: ≈0.368 AR + ≈0.22 compute/launch (arithmetic from the measured slope,
  not a profile). The current overlap hides ≈0.07; **0.148-0.13 ms/token stays exposed**.
- Why the overlap cannot go further, in the current structure:
  1. **The layer tail joins its sub-blocks.** After the two AR stages, `run_layers_tp2_overlap` waits
     on every sub-block's done event on the chunk stream (src/models/qwen3_5/execution/text.cpp:2994-3004),
     so layer L+1's first mixer cannot start before layer L's last residual. There is no cross-layer
     pipelining: the per-layer pattern is mixer→AR, FFN→AR, join, repeat.
  2. **48 of 64 layers are recurrent (GDN)**: their mixer is a per-token serial scan, so the AR cannot
     be issued earlier than the scan finishes, and the sub-blocks inside a layer are scan-dependent.
  3. **The sub-block count is pinned at 4** by measurement (320/384-wide land in a slower scheduling
     tier; N=3 is worse than serial). More parts do not mean more hiding.
  4. The chunk is capped at 1,024 tokens by the 192 MiB workspace (`kWorkspaceBytes`,
     src/runtime/engine/tp2_generation_core.cpp:198; a text-only 1,024-token chunk peaks at 182.3 MiB),
     and a bigger chunk would only lengthen each exposed phase, not remove it.
- The residual is therefore not "more overlap inside a layer" — it is **work from layer L+1 running
  while layer L's allreduce is on the link**.

## 2. Mechanism

The dependency graph is two-dimensional:

```
stage (L, s):  layer L, sub-block s of the chunk (s = 0..3, 256 tokens each)

  (L-1, s) ──residual/hidden──▶ (L, s)
  (L, s-1) ──KV append + GDN scan order──▶ (L, s)
```

Everything else is independent: `x` columns are disjoint per sub-block; each token owns its KV rows;
only the last sub-block writes the chunk's logits. So a wavefront can run

```
stream 0:  L0s0 → L1s0 → L2s0 → …
stream 1:         L0s1 → L1s1 → L2s1 → …
stream 2:                L0s2 → L1s2 → …
stream 3:                       L0s3 → L1s3 → …
```

and the AR of `(L, s)` overlaps the compute of `(L+1, s-1)` and `(L, s+1)`.

Why it cannot reach the floor: in-kernel allreduce is not a DMA engine transfer — its kernels occupy
SMs on both cards, so the "AR time" also consumes compute resources. With four stages in flight the
36 SMs are shared, each stage's GEMM (M=256) inflates, and the 2.5:1 average AR:compute slack is
replaced by a bursty competition. That is the reason the planning band is +8-15% and not +39%.

## 3. Proposed design

### 3.1 Topology

- 4 compute streams (one per sub-block), plus the existing collective stream per shard. Both cards run
  the same stage sequence; the allreduce order must be identical on both sides (rendezvous channels and
  the collective stream already serialize it).
- Per-stream in-flight depth is naturally 1 stage: a stage ends by waiting on its own AR done before
  the next stage of that stream, so the CUDA queue may run deep but the device executes at most 4
  stages (4 streams × 1) plus the AR on the collective stream.
- Stage budget at 256 tokens: per stage AR ≈ 1.5 ms on the link, compute ≈ 0.6-0.9 ms.

### 3.2 Workspace partition

`DeviceArena` already supports a non-owning view over a `DeviceSpan`
(src/core/arena.h:66; `WorkspaceArena = DeviceArena` at :154), and every mixer/FFN/GDN entry point
already takes `WorkspaceArena&` as a parameter. So the 320 MiB becomes:

```
[0, 192 MiB)      shared head: x, cache, rope, positions, logits, mask — existing code unchanged
[192, 320 MiB)    4 × 32 MiB: one private arena per compute stream (non-owning views)
```

- `kWorkspaceBytes` 192 → 320 MiB; 384 if a measured stage needs 48 MiB.
- The head keeps the current semantics (the vision staging path included), so the decode graph
  watermarks and the vision path are untouched.
- A capacity assert on each view plus a poisoned test build (`0xCD`/NaN on alloc) turns an alias or an
  overrun into an immediate logits difference instead of a rare corruption.

### 3.3 Stream dependencies

Only three, and all explicit:

| # | Dependency | Why | How |
|---|---|---|---|
| D1 | `(L-1, s)` → `(L, s)` | residual/hidden | wait on that stream's AR-done event of stage `(L-1, s)` |
| D2 | `(L, s-1)` → `(L, s)` | same-layer KV visibility and GDN scan order | wait on stream `s-1`'s mixer-complete event of layer L |
| D3 | allreduce reads another stream's buffer | collective stream reads the `mix` output | already closed by D1/D2's "wait for own AR done before next stage" — do **not** add another event |

No event is needed for `x` (disjoint columns), KV (per-token rows) or `logits` (last sub-block only).
GDN state is a per-layer slice of the whole-pool buffer, so D2 covers it.

### 3.4 Integration points

1. **`forward_tp2_prefill` return contract**: it must return only after every prefill stream drained
   (today it joins its events into the chunk stream at text.cpp:2994-3004). With four streams an early
   return lets the next lane's `admit_lane` rebind the GDN slot and reuse the workspace while prefill
   work is still in flight — two hazards at once.
2. **Decode graph watermark**: `round_base = ws_a.used()` (tp2_generation_core.cpp:3643/4289) is the
   floor the batched decode graphs were captured with; the head's allocation order and sizes must stay
   byte-identical, or the replay throws "different workspace layout". Private tail views are safe by
   construction; head changes are not.
3. **Mid-batch pump**: the pump calls `forward_tp2_prefill` per chunk between decode rounds
   ([tp2-concurrency-headroom.md](tp2-concurrency-headroom.md) §8.4); it stays chunk-granular and needs
   the drain guarantee from (1).
4. **GDN slot / lane identity**: the slot binding is host-side and only one lane prefills at a time;
   the wavefront does not change that, but it removes the "single stream ⇒ FIFO covers slot reuse"
   argument inside the chunk, so the drain guarantee must be explicit.
5. **MoE stays out of scope**: the overlap gate rejects any layer with `tp_mixer_activation_width != 0`
   (text.cpp:2731-2740) and `run_layers_tp2_overlap` throws on a gather mixer (:2904-2908). A wavefront
   prototype is dense-only; MoE needs the activation-merge path first.
6. **Ring parity must follow the payload sequence, not the stream**: two ring slots are alternated by
   the AR calls; the slot assignment must stay tied to the allreduce sequence so two in-flight stages
   cannot alias a slot.

## 4. Benefits

| # | Benefit | Evidence | Confidence |
|---|---|---|---|
| B1 | Single-request prefill +8-15% (0.516 → 0.45-0.48 ms/token); +20% optimistic; +39% is the no-contention limit | measured baseline + arithmetic decomposition; no prototype yet | Modeled |
| B2 | The batch route's per-lane chunked prefill gets the same gain, because it calls the same `forward_tp2_prefill` — it shortens each lane's TTFT, not aggregate decode | headroom §4.1 ("inside forward_tp2_prefill, so the batch route's inline prefill benefits equally") | High |
| B3 | The only remaining prefill lever that is **bit-identical**: the per-`(L,s)` inputs and allreduce payloads are unchanged (design goal, verified by the existing chunk-split invariant) | `tp2_forward_test` chunk-split invariant | High (if implemented to the dependency graph) |
| B4 | No new CLI/config surface and no product change; it is a refinement of `--prefill-overlap` | — | High |
| B5 | Stacks with the other levers: chunk 2,048 (needs 320-384 MiB anyway), fp8 AR if ever accepted, card-2 x8 slot | §5 of the headroom report / decisions | High |

**Non-benefits, stated explicitly**: decode is unchanged; the 2.67k link ceiling is unchanged (the
wavefront only removes exposure, it does not add link bandwidth); `--prefill-chunk 256` keeps zero
intra-chunk parallelism (`parts < 2` disables overlap); MoE artifacts cannot use it.

## 5. Costs

| # | Cost | Size | Notes |
|---|---|---|---|
| C1 | Workspace | **+128 MiB/card** (192 → 320); +192 MiB/card if a stage needs 48 MiB | shipped 262,144 fp8/MTP K=2 has 697/1,217 MiB free (shard 0/1); after the change 569/1,089 (or 505/1,025). Vision staging stays inside the 192 MiB head |
| C2 | Host CPU per chunk | ~2,000-4,000 launches and ~512 events (estimate) | ~10-20 ms enqueue on one thread, competing with the pump's decode round; events themselves are ~1.5-2 ms |
| C3 | Code surface | ~150-250 lines | arena views, threading the per-stream arena through the layer calls, two cross-stream events, drain/assert contracts, and a gate flag |
| C4 | Measurement and docs | per-stream timing fields, `[wsdiag]` per-view peaks, rewritten `--prefill-overlap` semantics | the headroom report's 26% conclusion must be re-stated in terms of the new structure |
| C5 | Verification | bit-identity sweep (200 runs), poisoned build, injected delay, sanitizer, C=1/C=4 × 1k/8k/100k × plain/MTP/DFlash2 matrix | A-class races are intermittent; the matrix is the only way to make them fail loudly |
| C6 | Opportunity cost | fp8 AR (0.18 ms/token → 3.2-3.5k) has more upside but a cross-layer precision risk; card-2 x8 (~20 GB/s, 1.78× shallow) has more upside but needs hardware | the wavefront is the only bit-identical software path, which is its reason to exist |

## 6. Risks

| # | Risk | Trigger | Effect | Mitigation / gate |
|---|---|---|---|---|
| R1 | Gain smaller than modeled | SM/DRAM contention, in-kernel AR burning SMs, weight access spread over 4 layers | 0-5% instead of 8-15% | Step 1 gate: 2-stream prototype must beat 0.516 by ≥2% with bit-identity; stop otherwise |
| R2 | Arena alias/overrun | a stream's private view used after a rewind, or the head overruns into the tail | silent wrong logits | capacity assert on every view; poisoned test build; per-view `peak_used()` |
| R3 | Missing event (dependency hole) | D1/D2 forgotten for one of the 64 layers | intermittent token flips, only under specific timing | bit-identity ×200 plus an injected `cudaStreamWaitEvent` delay to widen the window |
| R4 | Head layout drift | a new allocation in the head before `round_base` | batched decode graph replay throws "different workspace layout" | keep head allocation order frozen; assert `round_base` unchanged in tests |
| R5 | Return before drain | the drain join is moved/removed | next lane's GDN slot and workspace overlap the previous chunk | make "prefill drained" an explicit postcondition; assert slot/quiescence in tests |
| R6 | Failure atomicity | AR stall mid-wavefront | 4 streams with partial work, un-attributable state | define chunk-level discard; keep the existing all-lanes-fail policy |
| R7 | Ring parity drift | slot chosen by stream instead of payload order | numeric drift vs the current route | bind slot assignment to the AR sequence, and test parity explicitly |
| R8 | MoE unsupported | gather mixer artifact | gate keeps it on the old path | document the dense-only scope; no silent fallback |
| R9 | Event created on the wrong device | events created while shard B is bound | `cudaErrorInvalidResourceHandle` surfacing as a kernel crash | create on shard A after binding; assert device at creation (worklog:2976-2979) |
| R10 | Profiler/sanitizer blind spot | compute-sanitizer and nsys do not see cross-stream races well | false negative in verification | do not rely on tools alone; poison + injection + bit-identity |
| R11 | Power/clock | 4 streams raise sustained utilization | small clock drop late in long prefills | measure; accept if within noise |
| R12 | Regression blast radius | the path is shared by every prefill (single-lane and per-lane) | TTFT regression on all routes | feature flag default off; canary 2-stream first |
| R13 | Interaction with vision | vision prefill already peaks at 182.3 MiB on the head (192.3 MiB before the §4.12 fix) | re-introduces the overflow | vision gate stays on the old path until the head partition is measured |
| R14 | Planning error in the memory budget | 32 MiB/stage too small | arena overflow at runtime | Step 0 measurement; 384 MiB fallback |

## 7. Relationship to settled decisions (do not re-derive)

- **Rejected and out**: prefill sequence parallelism (RS+AG ≡ 2×AR, byte-neutral), multi-block AR,
  copy-engine AR paths, fewer AR calls (would sacrifice weight splitting), the GQA-aware prompt kernel,
  finer sub-block slicing (N=3/5/8 all worse), host-side sub-block interleaving (Round 12, −5%),
  fp8 AR (0.18 ms/token saved, rejected on precision).
- **Round 12's −5%** was a host-side interleave that serialized the layer chain; it is not evidence
  against this design, which keeps the chain and only re-times it across streams.
- **worklog:2972-2975** says "the per-layer dependency chain L,A → L,B → L+1 must be preserved, only
  slot-wise A→B is possible". This design preserves it: `(L, s)` still waits for `(L-1, s)` and
  `(L, s-1)`. The old wording assumed one stream; the dependency graph is the same statement.
- **worklog Round 35c** ("overlap AR with MMA", ~1.5× upper bound) was a different structure (a
  layer-global rendezvous) and is superseded by the measured N=4 (−12%) path; this design is the
  continuation of the measured path, not a revival of that estimate.
- **PP-2 pivot** ([tp2-pp-prefill-pivot.md](tp2-pp-prefill-pivot.md)) is the other prefill lever: 1.7×
  but only for fresh prompts ≥ ~16k and it pays a ~2.9 s swap. The wavefront is per-chunk and works at
  any length; the two are not alternatives for the same workload.
- **Batched prefill** stays rejected ([tp2-concurrency-headroom.md](tp2-concurrency-headroom.md) §4.2,
  §9.4): it is 2.1× slower per lane-token and cannot raise aggregate throughput.

## 8. Staged plan and gates

| Stage | Work | Cost | Gate |
|---|---|---|---|
| Step 0 | Instrument the overlap path: per-stage `peak_used()` of the private arenas, AR vs compute time per stage (`NINFER_TP2_TIMING` fields), event/launch counts | ~1 day, no behavior change | 2 × single-stage peak ≤ 192 MiB? If not, size the head accordingly (320/384) |
| Step 1 | Minimal 2-stream prototype: overlap only layer L's last AR with layer L+1's first mixer (depth 2, no re-partitioning of sub-blocks) | ~1-2 weeks, one flag | bit-identical over 200 runs and ≥2% better than 0.516 at chunk 1,024. **Stop if either fails.** |
| Step 2 | Full 4-stream wavefront with the workspace partition of §3.2 | ~2-4 weeks after Step 1 | ≥8% at chunk 1,024; bit-identical; no regression at C=4 or at 100k depth; vision/MoE paths untouched |
| Step 3 (optional) | chunk 2,048 on the same partitioned arena | small | ≥4% on top, as the earlier arithmetic suggested |

Step 0 is cheap and answers the two unknowns that decide everything: the per-stage arena peak (memory)
and the real AR/compute split at 256 tokens (gain). A negative Step 0 on memory closes the design
without a prototype.

## 9. Open questions

- Is the residual at 4,543 tokens (0.498) launch/overhead-bound rather than AR-bound? The 4,543 and
  6,300-token readings differ by 0.018 ms/token, and chunk 512 shows a smaller overlap benefit — so the
  gain may be length-dependent.
- How much SM time does the in-kernel allreduce actually take? If it is a large share, the max gain is
  lower than +39% by construction.
- Does depth 1 per stream hold when the FFN and the next layer's mixer both become ready?
- What is the smallest safe drain contract for the pump's chunk boundary?
- Can two streams share one private arena with a single "previous use done" event (saves 32 MiB but
  reintroduces the aliasing class)? — currently judged not worth it.

## 10. Provenance

- Analysis date 2026-10-20; no new measurement for this design.
- Baseline: [tp2-concurrency-headroom.md](tp2-concurrency-headroom.md) §4.1 (0.498/0.516/0.587,
  0.368-0.375 slope), §4.2 (batched prefill), §9.4 (batch window cost);
  [tp2-plan.md](tp2-plan.md) §3.4 (N sweep, chunk-split invariant, `--prefill-overlap`);
  [tp2-decisions.md](../tp2-decisions.md) (2.67k ceiling, rejected list);
  [tp2-pp-prefill-pivot.md](tp2-pp-prefill-pivot.md) §1/§6 (link arithmetic, fp8 AR);
  [tp2-dual-5060ti.md](../tp2-dual-5060ti.md) "Device memory budget" (free 697/1,217 MiB, state 293.6 MiB).
- Code: src/models/qwen3_5/execution/text.cpp:2730-2744 (overlap gate),
  :2849 (`run_layers_tp2_overlap`), :2896-3008 (per-layer stages and the tail join),
  :2731-2740/:2904-2908 (MoE gate); src/core/arena.h:66/:154 (non-owning arena view);
  src/runtime/engine/tp2_generation_core.cpp:198 (`kWorkspaceBytes`), :3643/:4289 (`round_base`).
- The code is unmodified; this document only records the pending design, its stated benefits, costs
  and risks, and the gates that would have to pass before implementation.
