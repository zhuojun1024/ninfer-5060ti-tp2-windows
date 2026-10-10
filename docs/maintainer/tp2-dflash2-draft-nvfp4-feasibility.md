# Full-draft NVFP4 for DFlash2: source-level feasibility analysis

> Conclusive analysis record (2026-09-24), serving the remaining lever "full-draft NVFP4" of
> [PLAN.md](tp2-plan.md) §3.4.
> Method and evidence: the qtype validators of the consuming ops, the weight binding/layout
> implementation, the converter method registry, the conversion report of the real r66 artifact,
> and this machine's `<models-dir>/W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8` sources plus
> `qwen3_8_27b_w4a4_w8a8_dflash2_q4all.ninfer` (the current best artifact).

## 0. Conclusion (TL;DR)

**Full-draft NVFP4 is "doable" at the source level, but not worth doing; as a device-memory lever it has already been beaten by the current Q4 artifact.**

1. **The gain is overestimated**: relative to the official Q8 draft, full NVFP4 saves **−1090.9 MiB (−1.065 GiB)**, the same order of magnitude as "about −0.95 GiB" in §3.4; but relative to the current r66 `q4all` artifact only **−202.3 MiB** remains, of which **−174.3 MiB comes from the codebook** (not GEMM, and unrelated to the "NVFP4 profile"), the remaining GEMM items together saving only ~28 MiB more.
2. **Tensor by tensor, NVFP4 is larger than Q4**: NVFP4 = 0.5625 B/element, Q4_G64_FP16 = 0.53125 B/element. The draft's gate_up / output / down / kernel_projection / feature_projection are already Q4 in the current artifact, so changing them to NVFP4 is a **capacity increase**; quantizing the fused QKV, still at Q8, to Q4 (the r65 route) saves 4.7 MiB more than NVFP4.
3. **Three hard blockers**: (a) the converter has no float→NVFP4 quantizer, and the draft source is compressed-tensors **dynamic-activation FP8**, with no calibration source for the `activation_input_divisor` that A4 requires; (b) NVFP4 native Weight **accepts only a complete parent**, which conflicts with the row-region sharing topology of the draft's fused QKV; (c) the NVFP4 variant/geometry registries of the consuming ops are closed sets, requiring 4 newly written kernel families (attn_input_proj 3-output, context_kv_materialize, dynamic grouped conv add/prepare, candidate_selector codebook).
4. **The only possible non-device-memory motive** is unifying the draft with the main model in the W4A4 family and getting NVFP4 TMA tensor cores; but that requires an independent activation calibration + performance argument, and the draft decode small-T path already runs SIMT, so the gain is uncertain.

Therefore the recommendation: **do not pursue "full-draft NVFP4" as a device-memory lever**; if device memory is the goal, order by cost-effectiveness: ① codebook quantization (an independent lever, 174 MiB, the deepest change) ② r65 fused QKV→Q4 (79.7 MiB, RowSplit naturally supports regions) ③ leave the rest alone.

---

## 1. draft tensor inventory and current formats

Configuration source `<models-dir>/W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8/config.json`: hidden 5120, intermediate 17408, 5 layers, 32 q heads / 8 kv heads / head_dim 128, sliding_window 2048, conv_kernel_size 2 / conv_group_size 16, selector_rank 256 / selector_top_k 16, target_layer_ids [5,19,33,47,61]. Binding code:
`src/models/qwen3_5/load/dflash.cpp:5-45`, `src/models/qwen3_5/load/dflash2.cpp:5-29`.

The real r66 artifact report `qwen3_8_27b_w4a4_w8a8_dflash2_q4all.ninfer.conversion.json` has `method: grouped_absmax` / `layout: row_split_k128_v1`, with 66 dflash2 objects in total:

| Tensor | Shape | Current format | Count |
|---|---|---|---|
| attention QKV (query+key+value fused; context_key/context_value are shares of key/value) | [6144,5120] | q8_g32_fp16 | 5 |
| mlp gate/up (sharing one storage object) | [34816,5120] | q4_g64_fp16 | 5 |
| attention output | [5120,4096] | q4_g64_fp16 | 5 |
| mlp down | [5120,17408] | q4_g64_fp16 | 5 |
| attention_conv / mlp_conv kernel_projection | [1280,5120] | q4_g64_fp16 | 10 |
| attention_conv / mlp_conv base_kernel | [2,2,5120] | bf16 | 10 |
| input/post_attention/query/key norm | [5120]/[128] | bf16 | 20 |
| feature_projection | [5120,25600] | q4_g64_fp16 | 1 |
| context_norm / final_norm | [5120] | bf16 | 2 |
| candidate_selector hidden_projection | [256,5120] | bf16 | 1 |
| candidate_selector predecessor/successor codebook | [248320,256] | bf16 | 2 |

(context_key/context_value do not get their own objects: the official recipe `tools/convert/official_recipes.py:47-54` calls `recipe.share(prefix+"context_"+role, prefix+role)` for `key/value`, so the two Bindings are Parts of key/value, living inside the same [6144,5120] object.)

## 2. Gain recomputation (MiB)

Three columns: official Q8 draft (all projections Q8, kernel/codebook/norm BF16), current r66 `q4all` (landed in r62–r66), full NVFP4 (the goal of this analysis).

| Component | Official Q8 | r66 q4all | Full NVFP4 | Δ vs Q8 | Δ vs r66 |
|---|---:|---:|---:|---:|---:|
| QKV [6144,5120]×5 | 159.4 | 159.4 | 84.4 | −75.0 | −75.0 |
| gate_up [34816,5120]×5 | 903.1 | 451.6 | 478.1 | −425.0 | **+26.6** |
| output [5120,4096]×5 | 106.3 | 53.1 | 56.3 | −50.0 | **+3.1** |
| down [5120,17408]×5 | 451.6 | 225.8 | 239.1 | −212.5 | **+13.3** |
| kernel_projection [1280,5120]×10 | 125.0 | 33.2 | 35.2 | −89.8 | **+2.0** |
| feature_projection [5120,25600] | 132.8 | 66.4 | 70.3 | −62.5 | **+3.9** |
| codebook ×2 [248320,256] | 242.5 | 242.5 | 68.2 | −174.3 | −174.3 |
| hidden_projection [256,5120] | 2.5 | 2.5 | 0.7 | −1.8 | −1.8 |
| **Total** | **2123.1** | **1234.5** | **1032.2** | **−1090.9** | **−202.3** |

Note: the official Q8 column total of 2123.1 MiB = 2.073 GiB matches the archive's "draft 2.07 GiB/card" bit for bit, which can be taken as a cross-validation of the inventory/byte formula.
The "about −0.95 GiB" of §3.7 should be an estimate that excludes the codebook (−916.6 MiB) or a GB/GiB rounding.

**Key point**: NVFP4's 0.5625 B/element is larger than Q4_G64's 0.53125 B, so after r62–r66 every GEMM already quantized to Q4 becomes a capacity increase when changed to
NVFP4; the only genuinely large saving (codebook, −174.3 MiB) is unrelated to GEMM/profile.

## 3. Per-op NVFP4 support matrix (source evidence)

| draft tensor | Consuming op | Status | Conclusion |
|---|---|---|---|
| QKV [6144,5120] decode | `ops::attn_input_proj` three-output (`execution/draft.cpp:326-328`) | Hard-requires Q8_G32 RowSplit, n=6144 (`ops/wrapper/attn_input_proj.cpp:243-261`); an NVFP4 variant exists only for the 4-output [14336,5120] (`nvfp4_attn_input_plan.cpp:19-88`, `nvfp4_geometry.h:29`) | Needs a newly written 3-output NVFP4 path + N6144K5120 geometry |
| QKV key/value rows [1024,5120] prefill | `ops::context_kv_materialize` (`draft.cpp:157-175`) | Hard-requires Q8_G32 [1024,5120] (`context_kv_materialize.cpp:37-52`); the fused kernel uses 1 byte of codes per 32 elements + FP16 scale, 7 routes (`materialize.cu`, `launch.h:7-28`) | Needs an NVFP4 codec + rewrite of the 7 routes |
| QKV key/value (dflash v1 / MTP) | `ops::linear_pair` (`draft.cpp:203`, `mtp.cpp:87`) | Q8-only (`wrapper/linear_pair.cpp:42-52`) | **Not on the DFlash2 path**, can be left alone |
| gate_up [34816,5120] | `ops::linear_swiglu` | **NVFP4 already supported, and the geometry is hard-wired to N34816K5120** (`wrapper/linear_swiglu.cpp:75-77,105,117-121`; `nvfp4_linear_swiglu_plan.cpp:56-67,148`) | A16 only T≤16 (`plan.cpp:35`); T≥5 requires AllowA4 |
| output [5120,4096] | `linear_dynamic_grouped_conv_add` | Q8/Q5/Q4 only (`dynamic_grouped_conv.cpp:83-124,299-317`); plain NVFP4 `linear_add` only recognizes [5120,6144]/[5120,17408] (`wrapper/linear_add.cpp:118-128,216-227`) | Needs a new NVFP4 fused variant + N5120K4096 geometry |
| down [5120,17408] | same as above | same as above (plain NVFP4 linear_add supports this shape, but the fused conv-add has no NVFP4) | Needs a new NVFP4 fused variant |
| kernel_projection [1280,5120] | `rmsnorm_dynamic_grouped_conv_prepare` | BF16/Q4 only (`dynamic_grouped_conv.cpp:43-55,57-75,236-250`) | Needs a new NVFP4 prepare (partial+reduce) |
| feature_projection [5120,25600] | `ops::linear` | NVFP4 dispatch is a closed (n,k) registry with no 25600 (`nvfp4_dispatch.cpp:9-21`; `nvfp4_geometry.h:44-73`) | Needs a new shape + geometry id |
| hidden_projection [256,5120] | `ops::linear` | same as above, no [256,5120] | 2.5 MiB, gain 1.8 MiB, not worth it |
| codebook ×2 | `candidate_selector_path` | BF16 `Tensor` only (`wrapper/candidate_selector.cpp:110-113`); the parameter type is a bare `Tensor` (`parameters.h:111-114`) | Needs a codebook codec + new selector path + Tensor→Weight + peer |

## 4. The three hard blockers

### 4.1 Converter: no float→NVFP4 quantizer, and no A4 calibration source for the draft

- `tools/convert/methods.py:296-301` only registers `cast_direct` / `grouped_absmax` / `fp8_row_maxabs` /
  `import_encoded`; the only one that can produce NVFP4 is `import_encoded` (`:243-293`), which moves a pre-encoded source byte-for-byte and
  requires `source.format == "nvfp4"`. `grouped_absmax` only accepts integer QuantFormat (`:196-219`),
  and `quantize_matrix` throws directly for a non-QuantFormat (`quantization/groupwise.py:83-85`).
  `docs/maintainer/tensor-formats.md:92-93` explicitly states that there is currently no built-in float→NVFP4 quantizer.
- The draft source is compressed-tensors **dynamic-activation** FP8: `DFlash2-FP8/manifest.json` has
  `activation: float8, symmetric dynamic per-tensor`, with only 20 weight scales and **no input_global_scale**.
  The A4 route requires an `activation_input_divisor` Use for every (parameter,input) (`load/prepare.cpp:44-54`),
  and `import_encoded` injects it only when `AllowA4` holds and the source has an input scale (`methods.py:269-280`).
- If going A16 only: `weight_input.cpp:108-110` will inject divisor=1.0 for NVFP4 A16, but the A16 routes of the individual ops
  restrict small T (e.g. `nvfp4_linear_swiglu_plan.cpp:35` is T≤16 only), so the draft's T=(K+1)×B goes out of bounds at batch≥3 or K>7.
  Covering 1–8 concurrency requires A4, hence a newly added draft activation calibration.

### 4.2 Weight ABI: NVFP4 does not allow sub-regions; the row-view topology of the fused QKV conflicts directly

- `native_weight` throws for any true sub-region of `QuantLayout::BlockScaleK16M128x4` and `RowScale`:
  `"this native Weight input requires a complete FP8/NVFP4 parent"` (`src/core/weight_view.cpp:265-269`,
  `is_complete_weight` requires view.shape == parent.shape, `:175-185`).
- `validate_nvfp4_weight` additionally requires `qdata==payload` and `scales==payload+scale_plane_offset`
  (`src/ops/linear/nvfp4/nvfp4_format.cpp:67-70`); `weight_row_planes` for NVFP4 always returns the parent's
  scale base (`weight_view.cpp:219-223`), i.e. for a sub-region with row_begin>0 even the plane pointer is wrong.
- No NVFP4 Weight with a non-payload base exists in the production path (`native_weight`, `tp_materialize.cpp:146-149`,
  `weight_splitter.cpp:140/187/400` are all at base).
- Yet the draft's QKV is exactly one complete [6144,5120] object, with key/value/context_key/context_value being row ranges
  (the RowSplit branch of `weight_view.cpp:193-199` offsets by `row_begin*per_row_bytes`; `parameters.cpp:231-234`
  binds via `prepare_linear_weight → native_weight`). **Only one family of Q8 RowSplit currently lets this sharing hold.**
- Two ways out:
  1. Split into three complete NVFP4 parents (query [4096,5120], key/value [1024,5120]). This requires changing the share structure of the official recipe,
     and adding a multi-parent NVFP4 branch to `prepare_attn_input_proj_weights` (`ops/weight_input.cpp:165-179`, which currently requires a single contiguous parent + Q8);
     only then can prefill's context_kv_materialize also get a complete parent.
  2. Extend the ABI so `context_kv_materialize`/`linear_pair` directly consume `WeightRowPlanes` (swizzled NVFP4
     scales are already supported), but GEMM weight staging and `validate_nvfp4_weight` both assume row_begin=0, so this is a shared-kernel-layer change.
- An accompanying constraint: **one object can only have one dtype, and QKV is consumed by both decode (attn_input_proj) and prefill
  (context_kv_materialize)** ⇒ it is impossible to make only decode NVFP4 while keeping prefill Q8.

### 4.3 op registry and kernels: closed geometry + hand-written fused kernels

- NVFP4's (n,k) is a closed set: `resolve_nvfp4_geometry` (`nvfp4_geometry.h:44-73`) has 5 ids,
  `nvfp4_dispatch.cpp:9-21` has 10 shapes; A4 TMA descriptors are instantiated only for those 5 ids (`nvfp4_w4a4_tma.cu`).
  The N6144K5120, N5120K25600, N5120K4096, (N1024K5120) that the draft needs are none of them.
- Kernel families that need to be newly written: `attn_input_proj` 3-output (the q4_q5 version is hand-written MMA `q4_q5_attn_input_*`),
  `context_kv_materialize` (codes/scales staging and MMA B-fragment decode in `materialize.cu`, 7 routes),
  the NVFP4 plan/materialized of `dynamic_grouped_conv_add` (the q5/q4 versions delegate to ordinary GEMM + the shared
  `dynamic_conv_finish.cu`), and the NVFP4 partial/reduce of `rmsnorm_dynamic_grouped_conv_prepare`.
- The codebook of `candidate_selector_path` is a bare Tensor (`parameters.h:111-114`) with only a BF16 variant
  (`wrapper/candidate_selector.cpp:110-113`): to quantize it one must first have a codebook codec and preserve the top-k domain,
  and then wire up the Tensor→Weight pipeline and the peer path (`text.cpp:973+`, `draft.cpp:399-407`).

### 4.4 TP-2 position

draft weights are `Replicated shards=0x1` (only on shard 0), the codebook is `0x2` (only on shard 1),
and these two rules come before all shape rules (`load/tp_split_spec.cpp:97-112`) ⇒ the draft's fused QKV
is **never split by shape**. `weight_splitter` itself supports NVFP4 ColumnParallel/RowParallel relayout
(`weight_splitter.cpp:112-190`), but the draft does not use it; codebook quantization would have to cover the selector path on the peer.

## 5. Performance angle

- draft decode's T=(K+1)×B (default K=7, B=1 ⇒ 8). NVFP4's small-T route is SIMT (`nvfp4_shapes.h`,
  gemv/simt/exact at `shapes/n34816_k5120.cu:31-39`), whereas the current q4/q5 already take MMA at small T
  (`SmallTTiled = mma.small_t.tiled` at `linear_swiglu/q4/q4_linear_swiglu_plan.cpp:35-46`).
  ⇒ switching decode to NVFP4 **carries a degradation risk**.
- NVFP4's TMA tensor cores are enabled only at T≥256 (the `select_a4` branch) or T≥5 for A4 MMA; prefill/context append
  widths of 1..2048 can catch part of that. But DFlash2's prefill cost is itself only on the 3–4% order, so the room for gain is limited.
- A4 requires activation quantization (extra workspace + quantizer kernel), run every decode round, further weakening the net small-T gain.

## 6. If pursuing it: modification list (file level)

1. **Converter**: add and register an NVFP4 method in `tools/convert/methods.py` (E2M1 codes + E4M3FN
   scale per 16 K + per-matrix FP32 divisor), aligned with the swizzle / `write_codes` of `tools/artifact/codecs/nvfp4.py`;
   add `tests/artifact/test_codecs.py`. **Plus draft activation calibration** (required by A4) — the repository currently has no draft calibration facility.
2. **Recipe**: change the dflash2 branch of `tools/convert/official_recipes.py:35-46` to NVFP4 on demand; the draft source is FP8,
   so the new method must support FP8→BF16→NVFP4.
3. **Geometry/shapes**: add N6144K5120 / N5120K25600 / N5120K4096 to `nvfp4_geometry.h` (+`resolve_nvfp4_geometry`), `nvfp4_shapes.h`,
   `nvfp4_dispatch.cpp`, `shapes/*.cu`, `sources.cmake`.
4. **attn_input_proj 3-output**: `wrapper/attn_input_proj.cpp`, `ops/weight_input.cpp` (the
   NVFP4/multi-parent branch of `prepare_attn_input_proj_weights`), new kernel + plan.
5. **context_kv_materialize**: `materialize.cu` NVFP4 codec + `require_weight` branch (7 routes).
6. **dynamic grouped conv**: `wrapper/dynamic_grouped_conv.cpp` (`projection_planes`,
   `require_finish_projection_weight`, `kernel_projection_payload_bytes`, dispatch switch, capacity API)
   + `dynamic_grouped_conv/nvfp4/` + the NVFP4 partial/reduce of prepare.
7. **candidate_selector**: codebook codec + new selector path + Tensor→Weight in `parameters.h`/`parameters.cpp`
   + the `text.cpp` peer path.
8. **Verification** (per the AGENTS.md numerical contract): each new kernel gets an independent FP64 oracle (independently decoded from the stored scale),
   route boundaries ±1, graph replay; then add the §3.7/§8.2 acceptance-rate A/B, sampling A/B and the DFlash2 trio.

Rough effort estimate: r62–r66 (4 items, already including op oracle + conversion + A/B) as one baseline unit; this list is about 3–4
units, plus the converter/calibration as a separate unit.

## 7. Recommendation

1. **Do not pursue full-draft NVFP4** (the device-memory motive does not hold: net −202 MiB against the current artifact, 174 of which comes from the codebook).
2. If device memory is still to be freed, order by cost-effectiveness:
   - **codebook quantization** (−174 MiB): an independent lever; in dtype terms **Q4 is smaller than NVFP4**; requires codebook codec +
     top-k domain preservation + Tensor→Weight + peer.
   - **r65 fused QKV→Q4** (−79.7 MiB): RowSplit regions support it naturally, but it requires a family of 4-bit fused materialization kernels +
     a 3-output q4 variant + two sets of oracles; the highest acceptance-rate risk (K projection error accumulates token by token within the 2048 ring).
3. Only when the motive becomes "unify draft and main model in the W4A4 family / take prefill TMA tensor cores" should the project be reopened,
   with activation calibration and a decode net-gain measurement as prerequisites/gates.
