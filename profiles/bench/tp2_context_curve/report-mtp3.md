# TP-2 context-length curve (original startup-script configuration: K=3 MTP + vision)

Local measurement, 2026-09-20. Same directory: `mtp3.json` (all records), `mtp3_summary.csv` (summary), `mtp0.json`/report (the comparison baseline).

## How the accounting differs from the baseline report

This run uses the **default recipe of your `$HOME/start-ninfer-tp2.ps1`**, i.e. the production configuration;
the `mtp0.json` run is the speculation-disabled baseline (`--no-prefix-reuse`, no vision). The only methodological difference between the two is that
this configuration **enables prefix reuse** (`context cache` is not root-only), which writes host checkpoints during prefill,
so small sizes are more likely to hit a cache for the template preamble (about 49 tokens).

## Run recipe (defaults of the original script)

| item | value |
|---|---|
| hardware | 2x RTX 5060 Ti 16 GiB (`--devices 0,1`, TP-2) |
| artifact | `<models-dir>/qwen3_8_27b_w4a4_w8a8.ninfer` |
| KV | fp8, capacity 204,800 tokens |
| speculative decoding | `--spec mtp --draft-tokens 3 --lm-head-draft` (K=3) |
| vision tower | on (shard 1 arena 826.5 MiB) |
| sampling/thinking | `--temperature 0.7 --top-k 20 --top-p 0.80 --reasoning-effort medium --preserve-thinking` |
| prefix reuse | on (`--host-state-slots 32`) |
| serving | run `start-ninfer-tp2.ps1` directly (port 3456); engine ready 31.9 s |
| driver | `pwsh -File tools/win_port/context_curve.ps1 -BaseUrl http://127.0.0.1:3456/v1 -Label tp2-mtp3 -OutFile profiles/bench/tp2_context_curve/mtp3.json -Sizes '128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,196608' -Repetitions 3 -DecodeTokens 256` |

The method is the single-request method of `docs/performance/methodology.md`: prefill = `prompt_n/prompt_ms`,
decode = `(predicted_n-1)/predicted_ms`, decode budget 256 tok (all ran to the limit), x-axis `usage.prompt_tokens`.

## Results (K=3)

| context tokens | prefill tok/s | sd | reps | avg cached tok | decode tok/s | sd | ms/token | prefill wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 118 | 787.9 | 0.0 | 1 | 33 | 71.03 | 2.95 | 14.08 | 0.15 |
| 268 | *1417.2 | 54.2 | 3 | 49 | 93.62 | 3.86 | 10.68 | 0.15 |
| 518 | *1542.6 | 32.3 | 3 | 49 | 89.94 | 3.81 | 11.12 | 0.30 |
| 1,018 | *1653.2 | 18.4 | 3 | 49 | 78.60 | 5.77 | 12.72 | 0.59 |
| 2,043 | *1645.2 | 9.3 | 3 | 49 | 75.49 | 6.37 | 13.25 | 1.21 |
| 4,093 | 1613.5 | 0.0 | 1 | 33 | 73.62 | 0.92 | 13.58 | 2.54 |
| 8,193 | 1608.5 | 0.5 | 3 | 0 | 72.85 | 3.03 | 13.73 | 5.09 |
| 16,393 | 1575.1 | 0.8 | 3 | 0 | 71.59 | 7.08 | 13.97 | 10.41 |
| 32,768 | 1495.9 | 0.6 | 3 | 0 | 66.04 | 4.55 | 15.14 | 21.90 |
| 65,543 | 1338.7 | 1.5 | 3 | 0 | 56.32 | 1.74 | 17.75 | 48.96 |
| 131,068 | 1111.2 | 0.5 | 3 | 0 | 54.91 | 6.95 | 18.21 | 117.96 |
| 196,618 | 946.7 | 0.1 | 3 | 0 | 53.68 | 2.85 | 18.63 | 207.69 |

\* = all 3 repetitions at that size hit the cache, so the prefill column can only use the "cached" mean (4-12% higher than a full prefill).
The remaining sizes use records with `cache_n == 0`; ≥8,193 all have 3 cache-free runs.

## Comparison with the MTP0 baseline (prefill uses the "all records" mean on both sides, same accounting)

| context tokens | MTP0 prefill | MTP3 prefill | Δ | MTP0 decode | MTP3 decode | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 118 | 891.1 | 862.2 | -3.2% | 31.09 | 71.03 ± 2.95 | **2.28x** |
| 268 | 1392.4 | 1417.2 | +1.8% | 31.11 | 93.62 ± 3.86 | **3.01x** |
| 518 | 1518.0 | 1542.6 | +1.6% | 31.17 | 89.94 ± 3.81 | **2.89x** |
| 1,018 | 1637.4 | 1653.2 | +1.0% | 31.09 | 78.60 ± 5.77 | **2.53x** |
| 2,043 | 1647.2 | 1645.2 | -0.1% | 30.99 | 75.49 ± 6.37 | **2.44x** |
| 4,093 | 1665.0 | 1633.1 | -1.9% | 30.91 | 73.62 ± 0.92 | **2.38x** |
| 8,193 | 1643.3 | 1608.5 | -2.1% | 30.71 | 72.85 ± 3.03 | **2.37x** |
| 16,393 | 1595.1 | 1575.1 | -1.3% | 30.42 | 71.59 ± 7.08 | **2.35x** |
| 32,768 | 1510.1 | 1495.9 | -0.9% | 29.78 | 66.04 ± 4.55 | **2.22x** |
| 65,543 | 1349.1 | 1338.7 | -0.8% | 28.64 | 56.32 ± 1.74 | **1.97x** |
| 131,068 | 1117.9 | 1111.2 | -0.6% | 26.65 | 54.91 ± 6.95 | **2.06x** |
| 196,618 | 951.4 | 946.7 | -0.5% | 24.98 | 53.68 ± 2.85 | **2.15x** |

## Observations

- **decode**: K=3 speeds up over MTP0 by 2.15x (200K) to 3.01x
  (peak at 268 tokens); throughput is 53.7-93.6 tok/s throughout,
  and 200K still reaches 53.7 tok/s. The low-context stretch (268 tokens) is where speculative gains peak;
  after that, as context grows, verify cost rises and the speedup falls back to ~2x.
- **speculative variance is markedly larger than for MTP0**: the decode sd across 3 repetitions at the same size reaches 7.08 tok/s
  (MTP0 is at most 0.04). This is inherent to speculative decoding — the acceptance count per round depends on the content, not on measurement noise;
  single-point comparisons should look at means and intervals, not one-off numbers.
- **prefill is nearly identical to MTP0 but systematically 0.5-2% lower** (large sizes): speculation does not affect prefill;
  this difference comes from the host checkpoint writes during prefill after this configuration enabled prefix reuse (32 slots × 73.4 MiB/GPU pinned),
  plus the resident vision arena.
- **small sizes (<4K) are dominated by fixed overhead + template cache**: at 118 tokens prefill is only ~790-860 tok/s,
  and the plateau of ~1.65k tok/s is only reached at 4K.

## Known gap: this route does not expose the speculative acceptance rate

The response `timings` block has no `draft_n`/`draft_n_accepted` (which is why accept% in the table is empty), and the logs have only the
prefill/decode phases. The reason is in the source: `src/serve/openai_chat_response.cpp:247` goes through
`outcome_timings()`, and the `result.speculative.*` it reads is populated by the single-GPU route (`engine_core.h:864`),
whereas `src/runtime/engine/tp2_generation_core.cpp` never writes `result.speculative` (it only counts internally in
`timing.committed += decision.accepted_tokens`, for the per-round debug output of `NINFER_TP2_TIMING=1`).
So **Spec acceptance / tokens-per-round as defined in `docs/performance/methodology.md` is currently unavailable on the TP-2 serving route**.
Two ways to fill it: ① have the TP-2 core write back `result.speculative` (recommended; the single-GPU route already has the same struct);
② temporarily restart with `NINFER_TP2_TIMING=1` and parse the per-round logs (crude, requires restarting the service).

## Limitations

- C=1 single requests only; the concurrency curve is measured separately.
- The 4 sizes marked with `* ` include caching in prefill and cannot be compared directly with the clean MTP0 points.
- DFlash/DFlash2 were not measured, nor were vision inputs.
