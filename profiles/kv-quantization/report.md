# Effect of KV cache quantization on model quality

An empirical report. It answers two questions: **does KV cache quantization break down at long context? and how large is the cost of each quantization tier?**

The measured artifacts are `Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-Vision.ninfer` (21.14 GiB,
TP-2 `--devices 0,1`) and `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` (13.99 GiB, single GPU).
Both are **16 layers of `full_attention` + 48 layers of `linear_attention`**: only 16/64 layers carry a KV
cache, while the recurrent state of the other 48 GDN layers is FP32 and entirely unaffected by `--kv-dtype`.
This is the structural reason the end-to-end effect size is small.

## Conclusions

1. **The error does not grow with context length.** In the fixed-text paired control (the same 1024 tokens are
   scored, only the amount of history visible ahead of them changes), the slope of ΔNLL against log10(history)
   across the five arms is only ~1e-3 nats/decade, |t| < 1.2; the operator-level relative
   attention error is flat from length 6 to 16,384. **There is no knee where "short context is lossless and
   long context collapses".**
2. **There is no catastrophic failure at long context.** The 8-needle probe scores 88/88 at 261,912 tokens; the
   32-needle + 32-character verbatim reproduction scores 32/32 at 261,712 tokens with every character correct.
   The earlier 122K five-format probe was byte-identical.
3. **But the error itself is not small.** At the operator level nvfp4 measures a relative attention error of
   **9.57%**, while int8 is only **0.42%** — a factor of 23. End-to-end perplexity is nearly insensitive to that
   9.6% attention error, which says both that the task is tolerant and that
   **perplexity and retrieval are two rulers that are insensitive to the 4-bit failure mode**.

**Recommendation**: default to `--kv-dtype int8`. It is the only tier on which neither of the two independent
rulers measures a cost, and it uses only 2.3% more VRAM than
`fp8`. Fall back to `fp8` when VRAM is tight; use `nvfp4` only when more than 2× headroom is needed to run
concurrency, charged at **+0.2~0.3% PPL**. `k8v4` is completely dominated by `nvfp4` (same accuracy, 40% more
VRAM), so do not use it.

**Precondition**: at a 262,144 context this model **does not fit** in bf16 (each shard would need 8192 MiB of KV,
but only about 5.1 GiB is actually left, capping it at about 163,840). So "whether to quantize" is not a choice —
"which tier to use" is.

## 1. Measurement method

Four layers, from cheap to expensive, cross-validating each other:

| layer | instrument | cost | what it measures |
|---|---|---|---|
| L0 | `ninfer_softmax_attention_test --quant-sweep` | minutes | the attention output error of the codec itself |
| L1 | `ninfer-perplexity --corpus ... --quick` | ~8 min/arm | end-to-end perplexity (corpus average) |
| L2 | `ninfer-perplexity --text ... --dump-token-nll` | ~7 min/arm | per-token NLL against visible history depth |
| L3 | fixed text block + swept history length (paired) | hours | the only design that separates "history length" from "history content" |
| L4 | prefix-chain retrieval probe | ~25 min | long-context retrieval accuracy (generation side) |

The key property of L0: it isolates the codec from the model. The same K/V is stored into the paged KV in the
target format and in BF16, and the outputs are compared with an FP64 ideal attention; `rel_rmse` is the RMS
relative to the reference output. **Note that its fixture uses uniform random K, so softmax is nearly uniform,
which is the regime where "the error is most easily averaged away"** — it gives a lower bound, not the error of a
real long context.

The key property of L2: `token_nll.<stream>.tsv` gives the NLL of each target, and token `t` is predicted from
`[0,t)`, so `target_index` is exactly the history depth that token actually saw (with `--context >= stream length`
there is only 1 window). Subtracting the per-token NLL of two runs that differ only in KV format cancels the text
term and leaves the contribution of the representation.

The key property of L4: prefix reuse in `ninfer-serve` merges the 11 prefills of a strictly prefix-chained
`[memo] + haystack[0:K] + [question]` into one longest prefill (measured exact re-send **0.9s vs 127.6s cold,
139×**; server log `cache 228,352 (91.4%, long anchor)`). This compresses the 88-sample depth grid from 5.5 h to
25 min, at the cost of dropping the "depth" dimension.

## 2. The physical bytes of each KV format

`docs/maintainer/paged-kv-cache.md`, D256, one token, one head:

| format | bytes | notes |
|---|---|---|
| BF16 | 1024 B | physically BF16 K + **FP16** V |
| INT8-G64 | 528 B | one scale per 64 elements |
| FP8-E4M3FN-row256 | 516 B | one scale per 256-element row |
| NVFP4-G16 | 288 B | groups of 16 elements, 4-bit |
| K8V4 | 402 B | K fixed to FP8-row256, V fixed to NVFP4-G16 (closed asymmetric) |

× 16 layers × 4 KV heads ⇒ whole model per token: bf16 64 KiB, int8 33 KiB, fp8 32.25 KiB,
nvfp4 18 KiB, k8v4 25.125 KiB.

## 3. Operator level (L0)

`rel_rmse` (RMS relative to the BF16 output), fixed-length oracle (6 tokens):

| format | mae | rmse | **rel_rmse** | max_abs |
|---|---|---|---|---|
| int8-g64 | 0.000231766 | 0.000288207 | **0.00417072** | 0.00101571 |
| fp8-e4m3fn-row256 | 0.00141165 | 0.00177748 | **0.0249171** | 0.00655441 |
| nvfp4-g16 | 0.00536884 | 0.00664607 | **0.0957072** | 0.0218758 |
| k8v4 | 0.0053744 | 0.00671739 | **0.0965646** | 0.0243407 |

Length sweep (prefixes of the same K/V, lengths 6 / 64 / 256 / 1024 / 4096 / 16384):

| format | rel_rmse @ 6 → 16384 | conclusion |
|---|---|---|
| int8-g64 | 0.00393 / 0.00408 / 0.00398 / 0.00413 / 0.00434 / 0.00395 | flat |
| fp8-e4m3fn-row256 | 0.02565 / 0.02639 / 0.02525 / 0.02595 / 0.02623 / 0.02380 | flat |
| nvfp4-g16 | 0.09771 / 0.09631 / 0.09186 / 0.09906 / 0.09930 / 0.09280 | flat |
| k8v4 | 0.09768 / 0.09630 / 0.09184 / 0.09904 / 0.09929 / 0.09277 | flat |

**The relative error is independent of KV length**; the absolute error (mae/rmse) falls as 1/√N with length,
exactly canceling the fall in the reference magnitude.

A by-product: **the quantization of V dominates the attention output error.** nvfp4 (both K and V 4-bit) 0.0957
vs k8v4 (K promoted to fp8) 0.0966 — raising the K precision did not change the error. Same caveat as above: the
fixture is uniform random K.

## 4. End-to-end perplexity (L1)

### 4.1 Single GPU 27B, `--quick`, 261,223 scored tokens

`--context 4096 --stride 2048` (124 windows), baseline bf16 = 1.508191 / 4.518551:

| format | mean_nll | Δnll | Δppl% | t (n=4 domain) |
|---|---|---|---|---|
| bf16 | 1.508191 | 0 | 0 | — |
| int8-g64 | 1.510395 | +0.002204 | +0.221% | 1.25 |
| fp8-e4m3-r256 | 1.509566 | +0.001374 | +0.138% | 1.75 |
| nvfp4 | 1.505517 | −0.002675 | −0.267% | −0.49 |
| k8v4 | 1.508828 | +0.000636 | +0.064% | 0.46 |

`--context 16384 --stride 8192` (28 windows), baseline bf16 = 1.461332 / 4.311698:

| format | mean_nll | Δnll | Δppl% | t (n=4 domain) |
|---|---|---|---|---|
| bf16 | 1.461332 | 0 | 0 | — |
| int8-g64 | 1.461336 | **+0.000004** | **+0.0004%** | 0.01 |
| fp8-e4m3-r256 | 1.464230 | +0.002898 | +0.290% | — |
| nvfp4 | 1.463646 | **+0.002314** | **+0.232%** | **3.81** |
| k8v4 | 1.461741 | +0.000409 | +0.041% | — |

**The 4K and 16K rankings contradict each other**, and both are anticorrelated with the L0 accuracy ordering: 4K
says int8 is worst (+0.221%) and nvfp4 best (−0.267%, dominated by the single chinese_reference stream); 16K says
int8 is lossless and nvfp4 and fp8 are worst.
**The 4K ranking is an artifact of short context**; only 16K agrees with the direction of codec accuracy. This
also shows that corpus-average perplexity needs domain-level t statistics with n≥4 to be meaningful.

### 4.2 TP-2 vision artifact, `--quick`, 261,223 scored tokens

`--context 65536 --stride 32768`, baseline bf16 = 1.413411 / 4.109949:

| format | mean_nll | Δnll | Δppl% |
|---|---|---|---|
| bf16 | 1.413411 | 0 | 0 |
| int8-g64 | 1.413121 | −0.000290 | −0.029% |
| fp8-e4m3-r256 | 1.413703 | +0.000292 | +0.029% |
| k8v4 | 1.414427 | +0.001016 | +0.102% |
| nvfp4 | 1.415843 | +0.002432 | +0.244% |

Here nvfp4 is worst, consistent with L0.

**Note that each stream in the `--quick` corpus is only about 65,513 tokens long**, so `--context` of 65536,
131072, 262144 gives **exactly the same** result (measured: bf16 is 1.413411 at all three, fp8 is 1.413703 at
both). In other words "perplexity at 256K context" is an empty comparison on this corpus — the KV never holds
more than 65K positions. To measure long context you must switch to a long corpus (see §5).

## 5. Per-token NLL depth curve (L2)

A **single stream of 257,438 scored tokens** (`--text`, `--context 262144 --stride 131072` ⇒ 1 window), TP-2,
per-token NLL:

| format | context | windows | mean_nll | PPL | KV MiB/shard | tok/s |
|---|---|---|---|---|---|---|
| fp8-e4m3-r256 | 262144 | 1 | 2.029775 | 7.612376 | 4128.0 | 608.6 |
| int8-g64 | 262144 | 1 | 2.030409 | 7.617202 | 4224.0 | 575.7 |
| nvfp4 | 262144 | 1 | 2.032253 | 7.631261 | 2304.0 | 577.7 |
| k8v4 | 262144 | 1 | 2.030734 | 7.619677 | 3216.0 | 610.3 |
| bf16 | 131072 | 2 | — | — | 4096.0 | 755.9 |

The absolute PPL of 7.61 is far above the 4.11 of the 64K `--quick`, because the corpus differs (a single mixed
pg19+wikitext stream vs 4 `--quick` streams) — **only same-stream cross-arm comparisons are valid**.

bf16 OOMs at 262,144 and can only run at 131072, and its window 1 (targets 131072..257437) has only 126,366 tokens
of history, so it cannot be used for the depth comparison. **The claim that "the value measured for bf16 at 131072
can serve as the reference arm for the 262144 runs" was verified separately**: with fp8 run at `--context 131072`
and at `262144`, targets 1..131,008 are bit-identical, and only the 63 depths 131,009–131,071 (the incomplete KV
page at the end of the window) differ, with `max|delta| = 0.455872`, affecting the mean NLL over 1–131,071 by
**4.7e-6**.

Conclusion from the depth curves: **the NLL of every arm falls monotonically with visible history depth** (longer
history is more predictable), and no arm shows a knee where it rises with depth. The per-token NLL data itself is
not committed (about 22 MB); the curves and bucketed statistics are in `kv-depth-curve.html`.

## 6. Fixed-text gold-standard pairing (L3)

Design: pin down the text being scored — the target block is always the last 1024 tokens of the source stream, and
run K exposes only the K tokens preceding it (K takes 13 values: 0, 256, 1024, 4096, 16384, 32768, 65536, 98304,
130048, 163840, 196608, 229376, 256415). For K ≥ 1 the immediate preceding text of the target block is the same
passage every time, **so not even a concatenation seam exists**, and the only variable is how far back the history
can see; at the same x-axis position every arm scores the same token. It has been verified that the last 3000
characters of the 13 inputs are byte-identical.

**Question one: for the same passage, does a longer history make the model better or worse?**
Once the scored text is pinned down, the slope of NLL against visible history is **negative** for every arm (deep
regime ≥16384):

| arm | slope (nats/decade) | se | t |
|---|---|---|---|
| bf16 | −0.168795 | ±0.032161 | −5.25 |
| int8-g64 | −0.109210 | ±0.021811 | −5.01 |
| fp8-e4m3-r256 | −0.119033 | ±0.021986 | −5.41 |
| nvfp4-g16 | −0.119770 | ±0.023345 | −5.13 |
| k8v4 | −0.116536 | ±0.022157 | −5.26 |

Only 6 intervals among the adjacent-depth paired differences are positive (the text becomes slightly harder as
history lengthens), and the largest, int8, is only +0.0081 ± 0.0056 (t=+1.46). ⇒ **"the longer the context the
worse the model" does not hold for this model + this corpus.** The rise in deep PPL seen when running one 257K
natural stream straight through is caused by **text content** (deeper tokens belong to the later book), not by
depth itself.

**Question two: does the cost of KV quantization grow with context?** The slope of the paired differences
themselves:

| comparison | slope (nats/decade) | se | t |
|---|---|---|---|
| fp8 − bf16 | +0.005919 | ±0.007586 | +0.78 |
| int8 − bf16 | −0.002955 | ±0.008596 | −0.34 |
| nvfp4 − bf16 | −0.006089 | ±0.008548 | −0.71 |
| k8v4 − bf16 | −0.002792 | ±0.007226 | −0.39 |
| int8 − fp8 | +0.007373 | ±0.018883 | +0.39 |
| nvfp4 − fp8 | +0.004786 | ±0.005693 | +0.84 |
| k8v4 − fp8 | +0.004298 | ±0.004798 | +0.52 |

Even when positive these slopes are only ~1e-3~1e-2 nats/decade, while the absolute differences themselves are on
the order of 1e-2 nats. ⇒ **The quantization cost is essentially a constant independent of context length.**

Remaining confound: the **content** of the history changes with K (the source stream is four pg19 books
concatenated in order, and when the visible window crosses a book boundary the preceding text switches to another
book). So the adjacent-depth paired difference mixes "more history" and "different history" at once, and
point-wise causality cannot be established; what can be established is the long-term trend. The curve has one
large step at about 60,517 tokens, which is where the window first reaches the beginning of the book that the
target block belongs to.

The full curves, the per-depth table and the slope table are in `goldstandard-paired.html`.

## 7. Long-context retrieval accuracy (L4)

Two probes, int8 and nvfp4, greedy decoding, `temperature 0`. Request body = `[memo] + haystack[0:K] + [question]`,
K increasing from 16,384 to about 262,144, each one a strictly prefixed extension of the previous.

**Soft probe**: 8 needles pinned at the very front of the document (= the hardest, most sensitive 0% depth tier in
the stock NIAH grid), requiring `City = Code` line by line.

| corpus | int8-g64 | nvfp4-g16 |
|---|---|---|
| `PaulGraham_Essays` (English) | 88/88 | 88/88 |
| `Journey_to_the_West` (Chinese) | 88/88 | 88/88 |

All 11 length points (16,396 → 261,912) **score full marks in every cell**.

**Hard probe**: 32 needles + a 32-character random access token reproduced verbatim (all the difficulty lives in
the shared preamble, at zero extra prefill cost).

| length | int8 city code | nvfp4 city code | int8 chars | nvfp4 chars |
|---|---|---|---|---|
| 16,396 → 249,868 | 32/32 | 32/32 | 32/32 | 32/32 |
| 261,712 | 32/32 | 32/32 | 32/32 | 32/32 |

**The result is completely saturated, zero resolution.** The only "failure" is a measurement artifact: at prompt
261,912 the `max_context` is the **total budget of "prompt + output"**, leaving only 232 tokens, while the complete
answer needs 352 output tokens, so the model was hard-cut in the middle of a number (`...Kyoto = 17850\nLagos = 3`).
Moving that point to 261,712 (432 left) restored 32/32.
**This is an operational constraint discovered along the way: with the context opened to 260K there is almost no
room left for output.**

The earlier in-house probe (122,277 tokens, 64 needles) was **byte-identical** across five formats, which also
confirms that the TP-2 path itself is bit-reproducible (re-sending in the same process and re-running in a new
process both agree with a cold start).

⇒ The correct statement is "≤262K, no catastrophic decoding divergence under int8/nvfp4", **not** "KV
quantization has no effect". Retrieval-style tasks have too much logit margin for a 9.6% perturbation of the
attention output to flip the argmax.

Curves are in `niah-probe.html`.

## 8. VRAM and capacity (TP-2, vision artifact)

`--devices 0,1 --max-context 262144 --kv-capacity auto`, each shard has weights+ctx fixed at 10894.6 MiB, total
card capacity 16310.6 MiB:

| kv-dtype | KV MiB/shard | free MiB/shard | KiB/token/shard | 262,144 |
|---|---|---|---|---|
| bf16 | needs 8192 | — | 32.0 | **✗ OOM** |
| int8 | 4224.0 | 924.0 | 16.5 | ✓ |
| fp8 | 4128.0 | 1020.0 | 16.125 | ✓ |
| nvfp4 | 2304.0 | 2844.0 | 9.0 | ✓ |
| k8v4 | 3216.0 | 1932.0 | 12.5625 | ✓ |

bf16 ceiling by bisection: 131072 → KV 4096.0, free 1052; 163840 → KV 5120.0, **free only 28**; 262144 → OOM
(`cudaMalloc failed: cudaErrorMemoryAllocation`). ⇒ **The bf16 KV ceiling is about 163,840 tokens.**

`int8` uses 96 MiB/shard more than `fp8` (+2.3%) — that is the entire cost of "the most accurate quantization
tier".

## 9. Known limitations

1. **Perplexity and retrieval are insensitive to the 4-bit failure mode.** nvfp4 perturbs the attention output by
   9.57% while PPL drops only 0.24%. If the downstream task depends on rare high-impact tokens (long-chain
   reasoning, exact numerics, code identifiers), the guarantees above do not hold — those workloads were not
   measured.
2. **No open-ended generation evaluation.** Everything is teacher-forced NLL or greedy retrieval.
3. **Measured on only one model family**, and both artifacts are `W4A4-W8A8` or IQ3_S, so **the weights are
   already heavily quantized**. Models with bf16 weights may not have the same tolerance.
4. **L0's fixture uses uniform random K**, so softmax is nearly uniform, the regime where "the error is most
   easily averaged away". Real long-context logits are sharper, and L0's conclusion cannot be extrapolated
   directly.
5. **L1's corpus is only about 65K tokens per stream**, so `--context` of 64K or above gives identical results.
6. **L3's history content varies with depth** (see the end of §6).

What to add, ordered by value for money:

| what to add | cost | value |
|---|---|---|
| push L0 length to 262,144 | ~1 h changing constants and memory + minutes to run | currently only reaches 16,384 |
| L1/L2 statistical power 0.5% → 0.1% PPL | 2.5–3 h GPU (4 independent long streams × 5 arms) | the only way to turn "not detected" into a "detected upper bound" |
| switch L0 to real long-context K/V | 2–3 h of development | closes limitation 4, L0's biggest weakness |
| bf16 reference arm out to 262K | impossible | OOM |

## 10. How to reproduce

```bash
# L0 operator-level length sweep (4 formats × 6 lengths)
./build/tests/ninfer_softmax_attention_test --quant-sweep

# L1 end-to-end perplexity, arm by arm
./build/apps/ninfer-perplexity <model>.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json --quick \
  --context 16384 --stride 8192 --kv-dtype int8

# L2 per-token NLL (--context >= stream length ⇒ single window, target_index is the history depth)
./build/apps/ninfer-perplexity <model>.ninfer \
  --text long_document.txt --context 262144 --stride 131072 \
  --devices 0,1 --kv-dtype int8 --dump-token-nll --output <dir>

# L4 prefix-chain retrieval probe (tools/)
python tools/niah_chain_sweep.py --port 18080 --arm int8 \
  --out data/niah_chain_int8.json
python tools/niah_hard_chain.py --port 18080 --arm int8 \
  --out data/niah_chain_hard_int8.json
python tools/niah_probe_report.py

# L2/L3 curves and the gold-standard report (tools/, reads the sweep products under _temp/)
python tools/build_tail_streams.py
python tools/goldstandard_report.py
python tools/kv_depth_report.py
```

The two `niah_*_chain_*.py` scripts need the haystack text of the evalscope NIAH dataset (`--corpus-dir`; this
repository does not vendor those two texts); `goldstandard_report.py` / `kv_depth_report.py` read the ~100 MB
per-token NLL dumps under `_temp/`, and those dumps are not committed, so these two scripts require re-running the
sweep first. The three HTML reports themselves are committed.

The evalscope configuration for NIAH is in `eval/configs/qwen3_8_27b_needle_haystack.yaml`.

## 11. Incidental engine changes

`ninfer-perplexity` originally had only a single `--device`, so artifacts that do not fit on one GPU could not be
evaluated. This work added a TP-2 scoring path to the engine (`TP2GenerationCore::score`, the same public contract
as the single-device `Program::causal_score`) and added `--devices A,B` and `--dump-token-nll` to
`ninfer-perplexity`. A/B cross-check (`tests/models/qwen3_5/test_engine_score_tp2_ab.cpp`, 1537-token repeated
passage): `delta mean_nll = 0.000402097`, `max|logprob delta| = 0.329137`, tolerance 5e-3.

**The two scoring routes are not bit-identical**: the TP-2 head splits the vocabulary across two GPUs, and each
logit is the sum of two half-K partial sums. Same artifact, same KV format, `bf16`, `--quick`, `--context 4096`:
single device 1.508191 / 4.518551, dual device 1.506711 / 4.511865, a difference of about 1.5e-3 — **the same
order as the KV format effect**. So KV formats should only be compared within one route; a difference of that
order across routes must be attributed to the route, not the format.

Details in `docs/perplexity.md` and `docs/maintainer/engine-architecture.md`.
