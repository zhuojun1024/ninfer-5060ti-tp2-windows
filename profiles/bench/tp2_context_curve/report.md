# TP-2 context-length curve (prefill / decode, MTP0)

Local measurement, 2026-09-20. Data in the same directory: `mtp0.json` (all records), `mtp0_summary.csv` (summary).

## Environment and recipe

| item | value |
|---|---|
| hardware | 2x NVIDIA GeForce RTX 5060 Ti 16 GiB (TP-2, engine `--devices 0,1`) |
| artifact | `<models-dir>/qwen3_8_27b_w4a4_w8a8.ninfer` (Qwen3.8-27B, w4a4 + w8a8) |
| KV | fp8, capacity 204,800 tokens |
| speculative decoding | off (MTP0, no `--spec` passed) |
| vision tower | off (no `--vision` passed, text-only curve) |
| prefix reuse | `--no-prefix-reuse` (log `context cache | root only`) |
| serve command | `ninfer-serve.exe <artifact> --devices 0,1 --max-context 204800 --port 3457 --host 127.0.0.1 --kv-dtype fp8 --log-level info --no-prefix-reuse` |
| driver command | `pwsh -File tools/win_port/context_curve.ps1 -BaseUrl http://127.0.0.1:3457/v1 -Label tp2-mtp0 -OutFile profiles/bench/tp2_context_curve/mtp0.json -Sizes '128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,196608' -Repetitions 3 -DecodeTokens 256` |

engine ready 29.7 s; after loading, 1184 MiB idle per GPU (capacity 204,800, no vision, no host checkpoint ring).

## Method

Single-request method (same as `docs/performance/methodology.md`): one chat request per size yields both phases at once, and every number is taken from the `timings` block of the response.

- prefill = `prompt_n / prompt_ms`; `prompt_ms` is the prefill wall-clock interval (tokenize/prepare is accounted separately and not included).
- decode = `(predicted_n - 1) / predicted_ms`; the decode budget is 256 tokens and all 36 requests ran to the `length` limit.
- x-axis = `usage.prompt_tokens` (the real context token count).
- Each request carries a unique system/user nonce to break prefix reuse.

**Cache accounting**: a `root only` cache can still hit about 49 tokens of the Qwen template preamble (independent of content),
so prefill counts only records with `cache_n == 0` (≥8,193 all have 3 cache-free runs; ≤4,093 has 1 cache-free run per size);
decode is unaffected by caching and is counted over all 3 repetitions. Records with caching have prefill rates 1-10% higher on average (most visible at small sizes),
and all records are kept in `mtp0.json`.

## Results

| context tokens | prefill tok/s | sd | reps | decode tok/s | sd | reps | decode ms/token | prefill wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 118 | 859.7 ± 0.0 | 1 | 31.09 ± 0.02 | 3 | 32.17 | 0.14 |
| 268 | 1265.7 ± 0.0 | 1 | 31.11 ± 0.02 | 3 | 32.15 | 0.21 |
| 518 | 1432.0 ± 0.0 | 1 | 31.17 ± 0.02 | 3 | 32.08 | 0.36 |
| 1,018 | 1582.2 ± 0.0 | 1 | 31.09 ± 0.01 | 3 | 32.17 | 0.64 |
| 2,043 | 1632.6 ± 0.0 | 1 | 30.99 ± 0.04 | 3 | 32.27 | 1.25 |
| 4,093 | 1649.3 ± 0.0 | 1 | 30.91 ± 0.04 | 3 | 32.36 | 2.48 |
| 8,193 | 1643.3 ± 2.4 | 3 | 30.71 ± 0.03 | 3 | 32.56 | 4.99 |
| 16,393 | 1595.1 ± 1.0 | 3 | 30.42 ± 0.03 | 3 | 32.88 | 10.28 |
| 32,768 | 1510.1 ± 1.2 | 3 | 29.78 ± 0.01 | 3 | 33.58 | 21.70 |
| 65,543 | 1349.1 ± 0.4 | 3 | 28.64 ± 0.02 | 3 | 34.92 | 48.58 |
| 131,068 | 1117.9 ± 0.2 | 3 | 26.65 ± 0.01 | 3 | 37.52 | 117.24 |
| 196,618 | 951.4 ± 0.2 | 3 | 24.98 ± 0.02 | 3 | 40.03 | 206.65 |

## Observations

- **prefill**: 4K-8K is the plateau (1649 tok/s); after that it decreases monotonically with context, reaching 951 tok/s at 200K,
  a 42% drop from the plateau. Above 33K it is essentially a near-linear slow decline with no steps
  (TP-2 chunked prefill uses a fixed step of 1024).
- **decode**: 31.09 tok/s at 1K and 24.98 tok/s at 200K, a 20% relative drop;
  per-token latency rises from 32.17 ms to 40.03 ms. Linear growth of KV reads with context is the main cause.
- **small sizes are dominated by fixed overhead**: 118 tokens give only 860 tok/s, and the plateau is only approached at 4K; decode is nearly flat below 2K.
- **stability**: at ≥8K the run-to-run sd is ≤ 0.2% (prefill) and ≤ 0.1% (decode); the sd at small sizes comes from fixed overhead, not thermal noise.

## Limitations

- C=1 single requests only; prefill/decode under concurrency is a separate curve.
- MTP0 (no speculation); the MTP3/DFlash decode curves need a separate run.
- The vision tower was off, so multimodal prefill is not covered.
