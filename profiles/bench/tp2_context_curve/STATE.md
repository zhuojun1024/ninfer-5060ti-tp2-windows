# TP-2 context curve run state

Task: measure the prefill and decode context curves (MTP0) on the local TP-2 engine,
capped by the artifact's ~200k context limit.

## Status: DONE (2026-09-20)

Finished 18:40. Files in this directory:
- `report.md`     - run metadata, methodology, result table, observations, limits
- `mtp0.json`     - 36 records + script summary (all repetitions, including the 1-10% faster cached ones)
- `mtp0_summary.csv` - cache-free prefill stats + all-repetition decode stats
- `serve-mtp0.log`- engine log of the benchmark server

Driver: `tools/win_port/context_curve.ps1` (HTTP chat timings, published single-request methodology).

## Result (cache-free prefill, n=1 below 8K / n=3 at and above 8K; decode n=3 everywhere)

| ctx | prefill tok/s | decode tok/s |
|---:|---:|---:|
| 118 | 859.7 | 31.09 |
| 268 | 1265.7 | 31.11 |
| 518 | 1432.0 | 31.17 |
| 1,018 | 1582.2 | 31.09 |
| 2,043 | 1632.6 | 30.99 |
| 4,093 | 1649.3 | 30.91 |
| 8,193 | 1643.3 | 30.71 |
| 16,393 | 1595.1 | 30.42 |
| 32,768 | 1510.1 | 29.78 |
| 65,543 | 1349.1 | 28.64 |
| 131,068 | 1117.9 | 26.65 |
| 196,618 | 951.4 | 24.98 |

## Notes
- `--no-prefix-reuse` still leaves a root-only cache that can hit the ~49-token Qwen template
  preamble; prefill stats therefore use only `cache_n == 0` records, decode is unaffected.
- User's original server (MTP3 + vision, port 3456) was stopped for the run and restarted
  afterwards via a local launcher script (`start-ninfer-tp2.ps1`); it is currently running as a
  DSH-managed background job.


## Round 2 (2026-09-20 evening): original launcher recipe (K=3 MTP + vision)

Ran the same 12 sizes x 3 reps against the user's own server (start-ninfer-tp2.ps1 defaults,
port 3456, K=3, vision, prefix reuse on). Files: `mtp3.json`, `mtp3_summary.csv`, `report-mtp3.md`.

Key numbers (all-reps mean): prefill matches MTP0 within -0.5..-2%; decode 71.0/93.6/89.9/78.6/75.5/
73.6/72.9/71.6/66.0/56.3/54.9/53.7 tok/s -> 2.0-3.0x over MTP0. Decode sd is much larger
(speculation acceptance varies per request).

Finding: TP-2 never fills `result.speculative`, so the API/logs expose no draft/acceptance stats
(`openai_chat_response.cpp:247` reads `result.speculative`, which only `engine_core.h:864` writes).
Server was NOT restarted for this round; the user's config kept running throughout.
