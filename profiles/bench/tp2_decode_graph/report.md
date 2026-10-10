# Round 17 ①: CUDA Graph on plain decode — A/B report

## Three-arm configuration (the only difference is `NINFER_TP2_DECODE_GRAPH`)

`<models-dir>/qwen3_8_27b_w4a4_w8a8.ninfer`, `--devices 0,1`, `--max-context 204800`, `--kv-dtype fp8`,
`--no-prefix-reuse`, `--host 127.0.0.1 --port 3457`, no `--spec` (plain route),
`NINFER_TP2_TIMING=1`; curve `-Sizes 1024,32768 -Repetitions 3 -DecodeTokens 256`.

| arm | meaning | decode tok/s @1018 | @32768 | prefill tok/s @1018 | @32768 | avg_round @32768 |
|---|---|---:|---:|---:|---:|---:|
| `exact` | eager + per-request exact envelope (= the code path before the change) | 31.11 | 29.79 | 1634.6 | 1506.7 | — |
| `0` | eager + bucket envelope (`ordinary_graph_profiles`) | 31.10 | 29.80 | 1635.9 | 1504.9 | — |
| default (graph) | captured graph + bucket envelope | **35.03** | **33.39** | 1628.4 | 1505.0 | 29.93 ms |

- decode: **+12.6%** at 1K, **+12.1%** at 32K (3 reps, sd about 0.01 tok/s).
- Converted to round time: eager 32.1/33.6 ms → graph 28.5/29.9 ms, i.e. about **3.6 ms saved per round** (the same order of magnitude as the 2.91 ms launch gap estimated in `tp2-plan.md`).
- prefill is identical across the three arms (±0.5%, within noise) ⇒ the change affects only decode, as designed.

## Numerical comparison (7 fixed prompts, temperature 0, 160 tokens)

| comparison | result |
|---|---|
| captured graph vs eager bucket envelope (swapping the launch mechanism) | **7/7 exactly identical** |
| eager exact envelope (rerun) vs bucket / vs graph | 7/7 exactly identical |
| eager exact envelope (first run) vs the rest | prompt 0 differs by 13 characters |
| eager exact envelope first run vs rerun with the same configuration | prompt 0 differs by 13 characters |

Conclusions:
- Capture-replay and eager launch are **bitwise identical** (the core correctness criterion passes).
- The bucket envelope matches the exact envelope (consistent with pre-condition 1 of `tp2-plan.md` Round 17).
- The only difference appeared between two independent runs of the same `exact` mode, and **can be attributed to run-to-run non-reproducibility, not to this change**: the route without `--spec` does not guarantee bitwise reproducibility across processes; even repeating the same prompt set inside one process diverges (apparently prefix reuse makes the 8th request take cached-prefill instead of a full prefill).
- Therefore token-level comparison can only serve as "no regression" evidence this round, not as a bitwise criterion; the mode-related difference is **0**.

## Implementation notes

- Added `TextContext::forward_tp2_decode_window` (`text.cpp`): a line-by-line clone of `forward_tp2`, whose only difference is that token/position are written by `copy_i32` (H2D memcpy, capturable) instead of `ops::set_i32_scalar` (host value, not capturable), with the envelope supplied as a parameter.
- `tp2_generation_core` gained `DecodeStepMode{Graph,EagerBucket,EagerExact}` and `std::vector<WindowGraph> decode_graphs_`; `NINFER_TP2_DECODE_GRAPH` = default `graph` / `0` → `EagerBucket` / `exact` → `EagerExact`; `EagerExact` is forced when the allreduce is not in-kernel (P2P/host staging synchronizes the host).
- The buckets come from `qwen::detail::ordinary_graph_profiles(options_.max_context)`, with `visible_begin = profile.min + 1` and `visible_end = min(max_context, profile.max + 1)`; the bucket envelope is bitwise exact (pre-condition 1).
- Startup prints `[tp2-graph] plain decode step: <graph|eager(bucket)|eager(exact)> | verify step: <...>`.

## On disk

`profiles/bench/tp2_decode_graph/`: `curve-{exact,bucket,graph}.json`, `greedy-{exact,bucket,graph,exact2,exact2b}.jsonl`,
`serve-*.log` (containing `[tp2-graph]` and `[tp2-time]`).

## Build script (the issue you pointed out)

`tools/win_port/build.ps1` rewritten: `Start-Process` file redirection instead of pipes + hard timeout + a pipe pre-check for the `ninja` child process +
a pre-check for a running `ninfer-serve` holding the exe; under the sandbox it exits within 8 seconds with a readable reason, and normally an incremental build completes in 37 seconds.
Root cause: `ninja` creates a pipe for each child process and the sandbox refuses pipe creation ⇒ ninja blocks before the first job, with no output, no child processes, and 0 CPU.

## Round 17 ②: AR picks the transfer strategy by tensor size — A/B report

### Strategy

`NINFER_TP2_AR_STRATEGY=size` (default) / `kernel` (the A/B reference, everything goes through the original slicing path).

| payload | transfer | rationale |
|---|---|---|
| ≤ 64 KiB (decode single-token delta = 10 KiB; MTP verify 4 tokens = 40 KiB) | single block; `bump_ar_token` fused into thread 0 of the AR kernel (`fuse_bump`), launches 2→1; thread count still `kArThreads`; dedicated 2×64 KiB mapped staging whose slots are not aliased with prefill slots | latency problem: 128 calls per round, fixed overhead dominates |
| > 64 KiB (prefill) | unchanged: slicing + separate bump + 24 MiB staging | bandwidth problem, already at the PCIe link floor |

Cross-GPU ordering still relies on the mapped-pinned arrival token spinning; the fusion puts the counter read-modify-write into the same kernel, published by thread 0 with
`__threadfence_system()`, so a peer never reads the other side's counter.

### Measurements (same binary, default decode graph, fp8 KV, 204800)

| arm | decode tok/s @1018 | @32768 | avg_round @32768 | prefill tok/s @32768 |
|---|---:|---:|---:|---:|
| `kernel` (reference) | 35.12 | 33.48 / rerun 33.48±0.02 | 29.86 ms | 1506.4 |
| `size` first version (256 threads) | 35.07 | 33.35 | 29.95 ms | 1504.4 |
| `size` final version (1024 threads, kept) | **35.34** | **33.65** | **29.70 ms** | 1505.8 |

- Final version: **+0.63% @1K, +0.51% @32K** (avg_round −0.16 ms).
- The first version with 256 threads was a regression (−0.2%/−0.4%): the gain came from dropping one launch, the loss from shrinking the block. The real cost of the payload is
  the mapped-host peer round trip, and the group loop needs parallelism, so keeping the thread count at `kArThreads` is the correct configuration.
- prefill is the natural control: all three arms hold within 0.1% ⇒ the large-payload path really was not touched.
- Against the +1–2% estimated before implementation, it landed on the **lower edge** (+0.5–0.6%).

### Numerics

`size` (both thread counts) vs `kernel`: 7 fixed prompts **7/7 exactly identical**. The fused path and the slicing path are bitwise identical
because the per-element addition order did not change.

### The arm not implemented (confirmed with you before starting work)

Copy-engine transfer for large payloads: prefill is already at the link floor (10–20 MiB is a bandwidth problem), and its cross-GPU ordering would have to rely on
cross-device `cudaStreamWaitEvent`, and the repository already has a measurement showing the copy engine is slower; the expected gain was ≈0, so it was not implemented.
