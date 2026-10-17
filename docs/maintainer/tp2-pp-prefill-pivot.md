# TP-2 -> PP-2 prefill pivot: measured feasibility (2026-10-17)

Question (user request, 2026-10-17): can a large prefill use a separate path that
temporarily re-lays the resident model from TP-2 to PP-2, runs the prefill, then
re-lays it back to TP-2, leaving the decode path untouched?

**Verdict: technically feasible, and worth ~1.7x end-to-end on fresh prefills of
>= ~16k tokens; a wash at <= ~10k tokens. Not implemented.** It is only viable as
a *swapped* prefill path: a permanent PP-2 (decode included, no swap) is ruled
out by the decode bandwidth floor, see section 7. The pivot is an
in-place re-layout of the resident weights plus a KV/GDN-state permutation - not
a re-allocation and not an artifact change, because TP-2 and PP-2 have the same
per-card byte footprint.

## 1. Why PP-2 attacks the prefill bottleneck where sequence parallelism did not

- The TP-2 prefill bottleneck is the per-layer allreduce: two ARs per layer
  move 2.6 MB/token at the chipset Gen4 x4 link (card 2, ~7 GB/s) =
  **0.368 ms/token / 2.72k tok/s**, and this traffic is irreducible while the
  weights stay split (docs/tp2-decisions.md, Rejected optimizations, 'Prefill
  sequence parallelism ... falsified at the design stage': SP's RS+AG is
  byte-equivalent to the ARs it replaces).
- PP-2 performs **no** cross-card allreduce at all: each card owns a contiguous
  layer range and computes its GEMMs in full. The only cross-card traffic is the
  stage handoff of the hidden state (5120 x W bf16 ~ 10 KB/token = 0.4% of the
  AR stream) and, at swap time, the weight/state permutation.
- The per-card *compute* is unchanged by the pivot. In TP-2 each card computes
  half of all 64 layers; in PP-2 each card computes all of 32 layers - the same
  head x layer product, the same GQA head count per card, the same total MMA
  work. PP-2's per-token wall is therefore `C/2 + handoff` and TP-2's is
  `C/2 + AR_exposed`, i.e. **the pivot buys exactly the exposed AR term**.
- PP-2 is only useful with cross-stage micro-batch (1F1B) pipelining. Naive PP
  serializes the two stages and costs `C` = 2x the TP-2 compute term, which is
  worse than TP-2 at every depth. The pipeline bubbles are (stages-1)/microbatches
  ~ 1/39 with 256-token sub-blocks - negligible.

## 2. Measured TP-2 baseline (shipped build, qwen3_8_27b_nvfp4, plain route)

Single request, `max_tokens 1`, unique nonce, prefix reuse ON, `prompt_ms` from
the response's own `timings`. `--prefill-overlap 256` is the shipped default;
`0` disables the L2 AR/copy overlap.

| prompt (computed) | overlap 256 | overlap 0 |
|---|---:|---:|
| fresh 8,193 | 1883.2 tok/s | 0.531 ms/tok | 1643.8 tok/s | 0.608 ms/tok |
| fresh 32,768 | 1744.6 | 0.573 | - |
| fresh 65,543 | 1483.1 | 0.674 | - |
| fresh 99,993 | 1283.8 | 0.779 | 1261.1 | 0.793 |
| cached 99,328 + 8,540 new @ 100k depth | 812.9 | 1.230 | 854.4 | 1.170 |
| cached 107,520 + 16,224 new @ 108k depth | 794.3 | 1.259 | 864.4 | 1.157 |

Readings:

- The marginal prefill cost at ~100k depth is **1.17-1.26 ms/token**, ~2.3x the
  shallow marginal (0.53). Run-to-run spread across the two server processes is
  ~5%, so the overlap-256 vs overlap-0 difference at depth is inside noise.
- The L2 overlap is worth **0.077 ms/token (14%) at 8k** (0.608 -> 0.531) and
  nothing measurable at depth. Its ceiling is the AR floor.
- Decomposition used below: `C/2(8k) ~ 0.22 ms/token` with `AR_exposed ~
  0.29-0.37 ms/token` (the doc's own pre-overlap split of 0.587 = 0.38 AR + 0.21
  compute, minus the 0.077 the overlap hides, gives 0.309). AR is
  depth-independent, so **the pivot's saving dW ~ 0.33 ms/token is roughly
  constant in depth** while TP-2's own per-token cost grows.

## 3. Swap cost (measured transport, no P2P)

- Weight volume: 9.51 GiB per shard => **4.755 GiB must cross to the peer in each
  direction** per leg; a round trip exchanges 9.51 GiB in total.
- No P2P (`canAccessPeer=0` both ways). Measured `cudaMemcpyPeerAsync`:
  0->1 6.47 GiB/s, 1->0 6.59 GiB/s, both directions concurrently 12.88 GiB/s
  combined. Per-card DMA: device 0 H2D 22.5 / D2H 26.2 GiB/s, device 1 6.66 /
  6.66 GiB/s (card 1 is the x4-slot bottleneck in both directions).
- Direct peer-copy leg: 4.755 GiB / ~6.5 GiB/s ~ **0.73 s**; host-staged leg
  (D2H + H2D through host, 9.51 GiB through the 6.66 GiB/s card) ~ **1.43 s**.
  So `2S ~ 1.5 s` with a device scratch and `~ 2.9 s` if the exchange must be
  staged through host.
- A full-size device scratch is **not** available: free memory at load is
  3,098 MiB/card against 4,755 MiB needed. The swap must therefore run in waves
  (or by cycle-following) with a <=~2 GiB scratch allocated from the free pool;
  the local re-layout inside a card is cheap (device-to-device), only the peer
  transfer is expensive.
- **The pivot is a permutation, not a re-allocation.** A TP-2 card holds half of
  every tensor (9.51 GiB); a PP-2 card holds all of half the tensors (~9.51 GiB).
  Weights, paged KV pool (64 layers x 2 heads vs 32 layers x 4 heads) and GDN
  state pool are all byte-compatible. The planner must confirm per-card byte
  balance across the GDN/attention layer mix and the embedding/lm_head placement;
  any imbalance is small and can be absorbed by the 3 GiB free pool.

## 4. Verdict

End-to-end with the swap included, taking dW = 0.33 ms/token and 2S = 2.9 s
(host-staged; 1.5 s with a scratch-staged exchange in parentheses):

| workload | TP-2 today | PP-2 + swap | speedup |
|---|---:|---:|---:|
| fresh 8,193 | 4.35 s | 4.5 s | ~1.0x |
| fresh 32,768 | 18.8 s | 10.9 s | 1.7x |
| fresh 65,543 | 44.2 s | 25.5 s | 1.7x |
| fresh 99,993 | 79.3 s | 47.8 s | 1.7x |
| cached 99,328 + 8,540 @100k | 10.5 s | 11.9 s | 0.9x |

- Break-even token count `N ~ 2S/dW` = **~8.7k tokens host-staged, ~5k with a
  scratch-staged exchange**. Below it the fixed swap cost dominates; above it the
  pivot converges on ~1.7x and keeps improving with prompt length.
- The pivot barely helps the *incremental* prefill case the user was modelling
  (~10k new tokens at 100k depth: 1.03x), because at that depth TP-2's absolute
  per-token cost is high and dW / W = 0.33 / 1.23 = 27%.
- Its real target is a **long-document / fresh prefills >= 16k tokens**, where it
  is ~1.7x. In a queue of several prefills, staying in PP across the batch
  amortizes the fixed 2S and drops break-even to ~2k tokens/request; that is the
  serving policy to design, not a per-request swap.
- The user's own profile (AI-agent traffic: an 8-24k system prompt, then mostly
  small tool-result increments) is *not* the >= 16k fresh case, but it names two
  cases that are: a full re-prefill after a session switch that lost its KV, and
  a cold start resuming a session. Both are prefix-reuse misses of the whole
  context, where the pivot's 1.3-1.7x applies and the small increments never pay
  a swap. See section 8 for the cheaper alternative that removes these prefills
  instead of speeding them up.

## 5. Implementation outline (if reopened)

1. **PP weight binding.** A second placement for the text parameters where a card
   owns a contiguous layer range with *full* tensors. Anchors:
   `src/runtime/engine/tp2_generation_core.cpp:825` (`build_shard`),
   `src/models/qwen3_5/execution/text.h:350` (`set_shard_config`),
   `src/models/qwen3_5/execution/text.cpp:2106` (`run_layers`) and
   `text.cpp:3248` (single-device prefill loop, already the full-weight path).
2. **Swap planner/executor.** Map every tensor's TP-2 byte range to its PP-2 byte
   range (contiguous row halves for TG/TL-style splits; strided column blocks for
   row-parallel splits), decompose into a wave schedule, allocate the scratch from
   the free pool, run peer copies on the transfer streams. Reuse the A3 pinned
   ring patterns (`src/artifact/materializer.cpp:21-68`) only for the fallback.
3. **PP prefill program.** Stage-split forward (layers 0..31 / 32..63) with 1F1B
   micro-batching over 256-token sub-blocks, hidden-state handoff buffers, KV
   written in the PP view of the (byte-identical) paged pool.
4. **Swap-back.** Weights + KV + GDN state permutation into TP-2 layout before the
   first decode round. The TP decode CUDA graphs already reference the same weight
   addresses; only the contents change, so no re-capture is needed - but the
   orchestrator must guarantee that no TP graph is replayed while the arena holds
   PP contents.
5. **Serving gate.** Mode is chosen per admission window (exclusivity + rollback on
   failure + a minimum-prefill-length and amortization rule), never mid-round.
6. **Verification.** A PP-2 prefill must reproduce the TP-2 route's logits within
   the existing TP-vs-single-GPU tolerance at a fixed prompt (the pivot changes
   only where partial sums are formed), plus prefill tok/s on the curve above.

## 6. Risks / open unknowns

- The swap implementation quality sets the break-even: 1.5 s vs 2.9 s moves it
  from ~5k to ~8.7k tokens. Needs a prototype to settle.
- W4A4/NVFP4 packed layouts: the exchange must respect packed codes and their
  block scales; if a shard split is not cleanly expressible as a row/column half
  in pack space, the permutation gets more expensive (still bounded by the same
  bytes).
- 1F1B correctness with 48/64 GDN (recurrent) layers: the recurrent state is
  card-local and advances in order, which the pipeline preserves, but the
  micro-batch boundary must not break the chunked state-passing chain
  (`src/ops/linear_attention/gated_delta_net/chunked/state_passing.cu:19-24`).
- A swap stalls every in-flight lane for 2S (~3-6 s including the state
  permutation), so it is a latency win paid for with a queue stall.
- Rejected on precision grounds at the design stage: halving the AR bytes by
  transporting the partial sums in fp8 saves only 0.18 ms/token and puts an
  E4M3-scale (~6%) relative error on every layer's residual contribution, which
  does not survive a 64-layer accumulation. This is why removing the AR entirely
  (PP) is the only lever that keeps the arithmetic.

## 7. Permanent PP-2 (decode left in PP, swap deleted) is ruled out

The user asked (2026-10-17) whether PP-2's decode is close enough to TP-2's
(>= 90%: stay in PP forever; < 70%: not worth doing) that the swap could be
deleted. The bound below is arithmetic, not an implementation result, so it needs
no PP prototype and no profiler:

- Batch-1 decode is weight streaming, not compute
  (docs/tp2-dual-5060ti.md:179-187: "the floor of the round time is
  `weight bytes per shard / 448 GB/s`"; the two 5060 Ti have 896 GB/s together,
  one card has 448 GB/s). Each card streams 10.15 GB per forward, so the TP-2
  floor is 22.7 ms (docs/tp2-dual-5060ti.md:373-374).
- The per-card byte and FLOP footprint is **identical** in both layouts: 10.15 GB
  of weights, 32 layer-equivalents (TP: half of 64 layers; PP: all of 32 layers),
  the same KV bytes per card per token. The only difference is cardinality -
  TP-2 streams the two halves **in parallel** on one token, PP-2 streams its two
  layer ranges **serially**.
- Hence PP-2's decode floor is 2 x 22.7 = **45.4 ms/token** against TP-2's
  measured 28.5 ms (35.1 tok/s plain, docs/tp2-dual-5060ti.md:362) or 32.2 ms
  (31.09 tok/s, profiles/bench/tp2_context_curve). The theoretical best ratio is
  28.5 / 45.4 = **0.63**, and it assumes every non-weight-streaming cost
  vanishes - the 2.9 ms of launch gaps, the 128 allreduces at 9-18 us each
  (1.2-2.3 ms), host steps and attention (docs/tp2-dual-5060ti.md:200-204) - which
  does not happen: PP still pays its own per-stage launches plus 64 layer
  handoffs. Realistic PP-2 decode is **0.5-0.6x** of TP-2 plain.
- Speculative rounds do not flip the sign. What does not double is the
  non-weight-streaming share of a round (draft chain, host steps), so MTP K=2
  could reach ~0.7x - still under both thresholds.
- PP-2 decode becomes competitive only with **1F1B across >= 2 independent
  requests**, where both cards stay busy on different micro-batches and
  throughput converges on TP-2's while per-token latency doubles. Single-stream
  agent traffic has no such concurrency.
- **Measured proxy** (2026-10-17, llama.cpp b16-035e227 on the same two cards,
  `D:/LLM/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` IQ3_S, warm, two 128-token reps per
  mode, `-sm layer` = true PP with KV following the layers and no collectives,
  `-sm tensor` = TP): layer split **29.2 / 29.2 tok/s** decode against tensor
  split **43.9 / 44.1 tok/s** => **0.66x**. The same pair reproduced §1's split
  independently: prompt processing 132-138 t/s in PP against 99.6-99.7 in TP,
  i.e. naive PP prefill = 1.33x TP, exactly `C` vs `C/2 + AR`. So a real PP
  implementation lands on the arithmetic bound, not on the 0.9 target, and
  ninfer cannot beat that bound because both layouts hold the same per-card
  bytes.

Consequence: the pivot keeps its swap-back, and decode stays on TP-2 at 100%.
Permanent PP-2 is closed on both the bound and a measurement of real PP code.

## 8. The two cases the user named are KV-retention, not prefill-speed

- Cross-session KV retention already exists in-process: the TP-2 checkpoint ring
  plus the session catalog are sliced by lane, and a recall that lands on another
  lane falls back to the host-resident session (docs/serving.md:87-92), with
  `--host-kv-mib` (32 GiB on the shipped serve.ps1 route) and
  `--host-state-slots` sizing the host retention (docs/serving.md:918-922).
- All of it is process-local and pageable: the Responses store is documented as
  "lost on restart" (docs/serving.md:708-709) and Host KV/State is pageable host
  memory, so a cold start or a ring eviction is a full re-prefill.
- Persisting a session's KV/state to disk (snapshot + restore, with the media
  identity digest it already carries) turns those two cases from "prefill again
  in 8-19 s" into "load and validate", which dominates the pivot's 1.3-1.7x on
  the same scenarios. This is the cheaper lever and is not implemented either.

## Provenance

- Measurements 2026-10-17 on `build-win/apps/ninfer-serve.exe` @ ea1908ae,
  artifact `D:/LLM/qwen3_8_27b_nvfp4.ninfer`, single instance, plain route,
  `--devices 0,1 --max-context 131072 --kv-dtype fp8`.
- Drivers: `_temp/20261017-1520_prefill_depth_probe.ps1` (depth/incremental),
  `tools/win_port/context_curve.ps1` (fresh curve; JSON under
  `_temp/20261017-1540_fresh_curve.json`, `_temp/20261017-1600_fresh_curve_overlap0.json`).
- Transport: `_temp/20261017-0330_tp_swap_bench.cu` (compiled, `tp_swap_bench.exe`).
- PP-vs-TP decode proxy: `D:/llama.cpp/build/bin/llama-cli.exe` b16-035e227,
  `-sm layer` vs `-sm tensor` on the same two 5060 Ti (the fork already implements
  both topologies with KV following the layer split); logs
  `_temp/20261017-1545_llamacpp_sm_ab.log` (first pass),
  `_temp/20261017-1600_llamacpp_sm_ab2.log` (warmup + 2 reps per mode).
