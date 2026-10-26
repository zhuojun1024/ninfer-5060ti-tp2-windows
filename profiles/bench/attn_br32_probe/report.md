# Prompt-attention envelope probe: Br=64 vs Br=32

Local measurement, 2026-10-26. Raw stdout: `br64.txt`, `br32.txt` in this directory.

## Question

At 128k depth ~55-60% of the single-request TP-2 prefill cost is the 16 full-attention layers'
prompt kernel, which runs at 42 TFLOP/s against the engine's own 85.9 TFLOP/s fp8 GEMM. ncu on the
production kernel: tensor pipe 63.6% (busiest), occupancy 33% (1 block/SM, 92,416 B smem, 102
registers), barrier stall 26.3%, math pipe throttle 34.9%. The two internal levers were measured and
closed (FP8 PV 1.176-1.189x, reverted on the quality gate; cross-tile phase pipelining 0.71-0.80x,
register file). The remaining axis is the execution envelope, and the only reachable envelope change
is Br=32 (see `docs/maintainer/tp2-adaptation-worklog.md` sections 7 and 8).

The probe isolates the Br half of the 2-CTA/SM shape's cost, bit-exactly, before anyone spends weeks
on the full rewrite.

## Method

Bench-only kernel (`bench/ops/causal_softmax_attention_bench_br32.cuh`): a copy of
`src/ops/softmax_attention/dense/causal_cache/prompt_fp8.cuh` with only the shape constants changed
(Br 64->32, ProducerWarps 8->4, DConsumers 4->8, PVNtPerWarp 8->4, `bar.sync 1,256` -> `1,128`,
smem 92,416 -> 79,104 B). No production file is touched.

```
ninfer_causal_softmax_attention_bench --entry cached --geometry d256-h12-kv2 --kv-dtype fp8 \
  --tokens 1024 --context 8192,32768,131072,245760 --execution eager --cache cold \
  --warmup 5 --repeat 30 --prompt-br {64,32} --digest
```

`--digest` prints an FNV-1a hash of the whole output buffer, which is how the two shapes are held to
bit identity. Build with `-DNINFER_BUILD_BENCHMARKS=ON` (the flag is OFF in the shipped build-win
cache) and link inside the MSVC environment.

Environment: 2x RTX 5060 Ti 16 GiB, `--devices` not involved (the op bench uses device 0), CUDA
13.3, Release. The resident `ninfer-serve` was stopped for the run. The Br=64 numbers reproduce the
worklog baseline (73,517.7 us at 245,760 and 39,553 us at 131,072) to within 0.5%.

## Result

Medians of 30 cold-cache repeats, one layer, W=1024, production geometry (12 query heads, 2 KV
heads, D=256) and production kv dtype (fp8-e4m3 row-256):

| depth | Br=64 (production) | Br=32 (probe) | ratio | math Br=64 / Br=32 |
|---:|---:|---:|---:|---:|
| 8,192 | 2,638.5 us | 3,054.2 us | **0.864** | 41.51 / 35.86 TFLOP/s |
| 32,768 | 9,947.3 | 11,425.8 | **0.871** | 42.10 / 36.65 |
| 131,072 | 39,169.7 | 44,888.7 | **0.873** | 42.27 / 36.88 |
| 245,760 | 73,526.4 | 83,931.3 | **0.876** | 42.15 / 36.92 |

Output digests are identical at every depth (`cb26fc0c7b915d4d`, `1bb70c61927d76ee`,
`4e9407879cc53c2b`, `c5a1962d6c235802`) and stable across repeated runs, so the probe is exactly the
block-shape change and nothing else.

## Verdict

Halving the query-row block costs **12.6-13.6%**, flat from 8k to 245k depth. The 2-CTA/SM shape
that motivated the probe must also halve Bc (the 32 KB V widening staging does not fit twice), so it
pays a second cost of the same kind, and it can buy back at most the 26% barrier stall — part of
which the section 7 experiment showed is tensor-pipe drain, not reclaimable idling. Break-even needs
>=25% of pure barrier fill. Expected value <= 1.0x, so the envelope rewrite is **not implemented**;
Br=64/Bc=64/16 warps is the interior optimum for this envelope (largest row block that keeps 8
producer + 8 V-widening warps with 4 PV slices; largest key tile that fits 92,416 B of smem).

Residual levers on the deep-context attention term, for the record: the FP8 PV quality trade
(measured 1.18x at the op level, reverted), and the grid tail (192 blocks over 36 SMs = 5.33 waves,
<=4%, needs a workspace-backed persistent work queue). Nothing else was identified.
