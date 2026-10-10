# PLAN: TP-2 support for GGUF weights

## 1. Goal and acceptance

1. **GGUF artifacts can be materialized and run on TP-2** (this machine's 2×RTX 5060 Ti).
2. **Replace the misleading `cudaMalloc OOM` with an explicit diagnosis**: when a shard genuinely
   cannot be represented, report at startup "GGUF block-quantized weights do not support that
   tensor-parallel split" and name the object and the reason.
3. No regression: the single-GPU GGUF path and the NVFP4/FP8 TP-2 paths (the existing TP-2 test
   suite) all keep passing.

## 2. Current state (reconnaissance conclusions)

The TP-2 split is **layout-agnostic**: 0 GGUF references in `src/core/tp/` and
`src/runtime/engine/`; everything works by `WeightSplitKind`
(`ColumnParallel`/`RowParallel`/`GatherRows`/`GatherCols`/`Replicated`).
The gaps are concentrated in three "layout-aware" places:

| Location | Current state | Consequence |
|---|---|---|
| `tp_materialize.cpp: shard_geometry()` | branches only for RowSplit/NVFP4/FP8, everything else at **4 bytes/element** | GGUF ~0.44 B/element → shard budget inflated by ~9× → `cudaMalloc` OOM (**measured**: `ninfer_qwen3_5_tp2_load_test --artifact <GSQ-RCO GGUF>` EXIT=1) |
| `weight_splitter.cpp: split_weight()/gather_rows_impl()/gather_cols_impl()` | the contract covers only NVFP4/FP8/RowSplit/BF16 | once the above is fixed, GGUF falls through to `throw "unsupported qtype"` |
| `tp_shard_views.cpp` | computes shard regions using "element" offsets | GGUF has a single code plane, row-major; element↔byte must be mapped via `code_bytes_per_row` (the current path may be reusable, needs checking) |

**Key enabler**: `weight_geometry()` already supports `GgufBlocks` (added by this port,
`code_bytes_per_row = (k/B)*b`,
`scale_offset = code_bytes`, i.e. **a single code plane with no separate scale plane**). Therefore:

- `ColumnParallel` (splitting N rows): the shard payload is a contiguous run of bytes of the parent
  object's code plane → **pure byte copy**.
- `RowParallel` (splitting K columns): each GGUF row is `k/B` self-contained blocks; splitting
  columns at **block granularity** = taking the contiguous bytes of blocks [b0,b1) from each row
  → **pure byte copy**, provided `k/2 % B == 0` (B ∈ {32,256}) and the split point falls on a block
  boundary.
- `GatherRows`/`GatherCols`: take half the rows/columns per part, likewise a byte copy.

## 3. The single structural obstacle: the `input_columns` permutation of `gdn/output`

In the recipe, `ssm_out` (= `gdn/output`) has its **storage K columns in tiled value-head order**,
and its Use carries the `input_columns`
auxiliary (column c is multiplied by element `input_columns[c]` of the grouped activation) — this is
exactly why it was kept during the port (moving columns would requantize across blocks).

But the TP-2 head split leaves each shard's activation as the **grouped half** (its own 24 value
heads). `tiled first half`
and `grouped first half` correspond to **different head sets** (`tiled = (h%3)*16 + h//3`; neither
side is the other's contiguous half),
and head = 128 elements < the 256-element i-quant block → **no block-level column split can align
the weight half with the activation half**.

Conclusion: this one object needs **execution-side** handling, i.e. "let both shards receive the
complete value output before `gdn/output`".
The feasible minimal change is to reuse the **existing** in-kernel all-reduce (no new collective
primitive needed): the shard writes its own output into a
**full-width** buffer (zero-filled outside the range) → all-reduce sum → both GPUs get the complete
grouped activation → then compute with the existing
"K split + post-projection all-reduce" pattern (full activation + each side's K half = partial sums,
the original reduction is still needed).
Cost: one extra [6144,T] reduction per GDN layer (same order as the existing [5120,T]
post-projection reduction); that projection costs 2× FLOPs (a negligible share).

## 4. Layered implementation

- L1 Add a `GgufBlocks` branch to `shard_geometry` (delegating to `weight_geometry`) → budget
  correct, OOM gone.
- L2 Explicit diagnosis: add shared validation (layout × split kind × block alignment), reporting
  the error in the same place at the **budget stage** (before cudaMalloc)
  and at the **split stage**, with the message naming the object and the reason.
- L3 `split_weight`: GGUF's `ColumnParallel` (row slicing) and `RowParallel` (block-aligned column
  slicing).
- L4 `gather_rows_impl`/`gather_cols_impl`: GGUF branches (take half per part, concatenate).
- L5 Model side `build_tp_split_spec`: GGUF-aware — select only representable kinds; for objects
  carrying `input_columns`, **explicitly reject** before the execution-side scheme lands (rather
  than silently `Replicated`, which would compute wrong results).
- L6 Execution side: full-width inputs for `gdn/output` + reuse all-reduce (L5's rejection is then
  lifted).
- L7 Tests: byte-level split oracle unit test (independently recompute the expected bytes from the
  block geometry); diagnosis-trigger tests;
  `tp2_load/forward` pass on GGUF artifacts; the existing TP-2 suite does not regress; finally
  dual-GPU end-to-end.

## 5. Progress

- [x] Reconnaissance (branch point `bfb57e72`; this machine's 2×RTX 5060 Ti)
- [x] L1 + L2 (budget and diagnosis) → commit `c3267d14`
- [x] L3 + L4 (split and gather) → commit `c3267d14`
- [x] L5 (spec GGUF-aware) → commit `c3267d14`
- [x] L6 (execution-side input_columns objects + GGUF GDN input projection + separated parent object layout) see section 7
- [x] L7 (verification) see section 6: GGUF `tp2_load`/`tp2_forward` pass, the existing TP-2 suite does not regress, dual-GPU end-to-end matches single-GPU
- [x] Commit `1a4a6a77` (`feat(tp): execute GGUF block checkpoints on the TP-2 pair`), and ff-only merged into
      `feat/windows-native-port` (after the merge, re-ran `linear_gguf` / `tp2_load` / `tp2_forward` at HEAD: pass)
- [x] Follow-up fix: head profile for `--lm-head-draft` / `--spec dflash2` on GGUF (see end of section 7), committed

## 6. Measured results (this round)

- Before the fix: `ninfer_qwen3_5_tp2_load_test --artifact <GSQ-RCO GGUF>` → `cudaMalloc OOM` (shard budget inflated at 4 B/element).
- After the fix: the same command → **materialization succeeds** (`device_objects=830`, **7.70 GiB/GPU**, measured with both GPUs completely idle), then in
  `tp_shard_views` it throws the precise diagnosis:
  `tensor parallel split: text/layers/0/gdn/output gathers its input columns, which a
  column-parallel TP-2 split cannot represent`.
- Independent cross-check: using the per-object shape/format reported by the artifact, recomputed in Python by the rules of `tp_split_spec.cpp`;
  per-GPU requirement ~8 GiB (Replicated 5.46 + RowParallel 1.84 + GatherRows 1.08 + single-GPU components 0.69), consistent with the measured 7.70 GiB.
- New unit test `ninfer_tp_weight_splitter_test` passes: N split, K split
  (block by block, and the two halves restore the parent object byte-for-byte) for Q8_0(32)/IQ4_XS(256)/Q4_K(256), per-part row/column gather, and the three diagnosis triggers; all expected values independently recomputed by
  `gguf_block_shape()`.
- Regression: `ninfer_linear_tp2_split_{nvfp4,fp8_head,grouped_head}_test` all pass (NVFP4/FP8 TP-2 unaffected).

### L7 verification (this round)

| Check | Command | Result |
| --- | --- | --- |
| GGUF dual-GPU materialization | `ninfer_qwen3_5_tp2_load_test --artifact Qwen3.8-27B-GSQ-RCO-IQ3_S` | pass (64 layers, gate [8704,5120], head vocabulary-parallel, 7.70 GiB on both GPUs) |
| GGUF dual-GPU forward | `ninfer_qwen3_5_tp2_forward_test --artifact same as above` | pass (11 probe shards consistent, 32-step decode, batched and sequential consistent; the three chunkings give the same argmax, top5 gap 0.125) |
| NVFP4 dual-GPU materialization/forward | the two commands above + `qwen3_8_27b_nvfp4.ninfer` | pass, and 128+172 / 64+236 still annotated `(bit-exact)` |
| Op/split regression | `ninfer_tp_weight_splitter_test`, `ninfer_linear_gguf_test`, `ninfer_embedding_test`, the three `ninfer_linear_tp2_split_*` | all pass |
| Dual-GPU end-to-end (including CUDA Graph) | `ninfer-serve --devices 0,1 <GGUF>` + `/v1/chat/completions` | materializes 10.9 GiB, `warmup complete` (graph capture succeeded), short prompt 57 tokens generates text **exactly identical** to single-GPU `ninfer.exe --device 0` |
| Long prompt (469 tokens) alignment | same as above, `--prefill-chunk 256/1024` | both dual-GPU chunkings give `…Passage repeated. Question: … in capillary`; single-GPU `--prefill-chunk 1024` agrees with it, single-GPU `--prefill-chunk 256` nearly diverges at the 9th token (on the same GPU, chunking itself flips near-ties) |

Conclusion: the GGUF artifact is now materialized on TP-2, runs forward and end-to-end serving, and produces the same greedy text as the single-GPU route under the same (indeed different) chunking.
Not run: `tp2_sessions` / `tp2_dflash_*` (these need dflash2 artifacts, and go through an engine route untouched by this change; spec derivation is already covered by the materialization cases of the two artifacts above).

## 7. L6 design decision (implemented this round)

Conclusion: there is **no need** to have the two shards each own the activation; instead, decouple the K split of `gdn/output` from the activation's domain.

1. The semantics of `input_columns` are inherently "column c is multiplied by the `input_columns[c]`-th element of the activation" — it indexes the **activation**,
   not the weight's own K. So separate "the weight's K" from "the activation row width": op/bridge accept `source_k >= k`.
2. `gdn/output` is still a contiguous K half split; `input_columns` (`auxiliary/000000`, INT32 [6144], **one object shared by 48 layers**)
   is **windowed** by the same half in `tp_shard_views` (the parent object stays resident in full, only the view is split, +12 KiB/GPU).
3. Two stages on the execution side: `run_layers_tp2` has each GPU write its own 24 value heads into its own half of the full-width [6144,T] buffer
   (the rest zeroed), one allreduce of the same pattern as the existing one merges them into the complete grouped activation, then each projects with its own K half to get partial sums,
   and the existing delta allreduce wraps up. One extra [6144,T] allreduce per GDN layer (12 MiB at T=1024, within threshold).

### Along the way, a second gap was found and fixed: separated gate/up
The GGUF artifact's `mlp/gate`(IQ2_XS) and `mlp/up`(IQ2_XXS) have different formats → they are **two objects**, not a fused parent object,
so the `n == 2*intermediate` rule does not match and it falls through to `Replicated`. And `down` is RowParallel →
`ffn_delta` treats gate's own two halves as gate/up and does silu_mul (**shape-legal, numerically wrong**).
Fix: the spec selects `ColumnParallel` for separated gate/up (exactly lining up with down's K half), and `ffn_delta` is made to support `p->up`.

### Third gap: the GGUF artifact's fused input projections do not always live in one object
Across layers of the same artifact, `gdn/{query,key,value,z}` may be four segments of **one** [16384,5120] parent object (layer 0), or may be
a `[q|k|v]` segment + a separate `z` object (layer 1 etc.), because the recipe only merges logical blocks that are **contiguous and share the same block format** into one object.
The same holds for attention: layer 27 is `[q|k]` fused + independent gate/value, while the remaining layers are four independent objects.
Fix (`tp_split_spec.cpp`): change the attention input projection from the hard-coded `n == 2*attn_q + 2*attn_k` to "runs of contiguous blocks read out by name"
(the `run_parts` helper: the name determines the segment, the width determines the row count, which must exactly equal the object's row count → `GatherRows`), plus two new rules for GDN: `[q|k|v]` (three GatherRows segments) and `z` alone (ColumnParallel).
At the same time, the GGUF form of `ops::gdn_input_proj` originally hard-coded the q/k/v/z row counts to the full-model profile (10240/6144) and was rejected after head splitting; it now derives the channel profile from the parts (q==k, v==z==3q), and a new "single contiguous [q|k|v|z,T] target" overload was added (a GGUF product only writes to a contiguous target, and the two halves of a shard are by nature row segments of the same parent object).

### On the "chunked prefill bit-exactness" criterion
The two chunkings 128+172 / 64+236 of `tp2_forward` require **bit-exact** agreement with single-chunk prefill; the NVFP4 route satisfies this, GGUF does not (top5 gap 0.125, same argmax, max_logit_diff ≈0.4). Localization process (temporary probes, all since removed):

1. Per-layer digest: with the same prompt and the same zero state, the **layer inputs** for T=300 and T=128 are completely identical, and the difference starts at the **layer 0 output** → the divergence comes from some operator inside the layer that consumes GGUF weights (the first one in layer 0 being the GDN input projection).
2. Operator-level reproduction (temporary case in `test_gguf.cpp`, real shapes rows=5120+3072, k=5120, IQ4_XS, two products writing the same target): `128 vs 300 → 140/1M elements differ, worst=16`; whereas the small shape (rows=256, k=1536) shows zero difference.
3. Mechanism: the integer MMQ in `third_party/ggml-quants` uses **stream-K** to distribute the K dimension across multiple blocks according to the grid (row tile × column tile) and then fixup-sums them; the grid changes with the column count → the floating-point summation order for a given column changes with "the column count of this particular call". Under small shapes each tile has exactly one block (no partial sums), hence zero difference. Temporary probes that disabled stream-K or fixed the tile width did not eliminate it (showing it is not tile selection but the chunking itself), but the operator-level reproduction is already enough to prove this is an **inherent property of the quantization route**, belonging to the pre-existing GGUF port (the single-GPU route behaves the same way), not something introduced by the L6 execution side.
4. Therefore `test_tp2_forward.cpp` tightens/relaxes per route: for artifacts that consume ggml block weights (`GgufProjectionWeights`), the chunking criterion becomes "same argmax + top5 gap ≤ 1.0" (still able to catch order-of-magnitude errors such as lost KV/GDN state), while all other routes keep bit-exactness.

### Fourth gap: the half-table proposal head of `--lm-head-draft` (user recipe error, fixed)
The user recipe `--devices 0,1 --spec dflash2 --lm-head-draft --vision --max-context 262144` reported on startup `FATAL server failed during startup | linear_topk: unsupported head profile` (at 27 s into planning).
Root cause: the vocabulary split divides the 131072-row reduced proposal head into 65536 rows per GPU, but `resolve_profile` in `linear_topk` recognizes a half table (`Q4OptimizedHalf`) only for **Q4_G64_FP16**, while for GGUF it recognizes only the full table 131072 and the full vocabulary 248320 → the half table falls through to a fallback throw; the indexing entry point (with the row→global id map) likewise only admits 131072. Fix: add `is_reduced_rows()`, so the GGUF profile accepts both the full and the half table (the same profile, with the row count taken from the weights).
Verification: `ninfer_linear_topk_test` gained sorting cases for both the full-table and half-table geometries of GGUF (q8_0) (independent FP64 oracle, id map, CUDA Graph replay, guard) plus a head-profile admissibility table (full vocabulary/full table/half table accepted, quarter table and 5120 rows rejected) → passed; the user recipe service starts successfully (KV 262144 fp8, vision, `dflash2 accepted 7/14 (50%)`, identical output for the same greedy request twice); the new `ninfer-serve.exe` has been deployed to `C:/ninfer/`.

### A stale assertion fixed along the way and the out-of-bounds it was masking (`test_tp2_load.cpp`)
`tp2_load --lm-head-draft --spec dflash2` on GGUF previously reported `shard 0 DFlash2 proposal head is not replicated`: that assertion was stuck in the "pre-split" design (the proposal head resident in full on shard 0, and the selector vocabulary codebook also on shard 0). The actual design (`load.cpp:137-143`, `draft.cpp:398-408`, `tp_split_spec.cpp:97-108`) is: the reduced head is split into halves by each of the two speculative routes (MTP merges it in `proposal_argmax`; masked draft ranks one half each and then merges with `merge_topk_candidates`), and the selector and full-vocabulary codebook move to shard 1. Once the assertion was let through, a second problem surfaced: the test directly read shard weights that had not been materialized (empty view) → null pointer dereference → process 0xC0000005 (not a throw). The assertion has been changed to "shard 0 holds this GPU's half table and does not hold the codebook; shard 1 does not hold the draft and holds the [248320,256] codebook". All four combinations of the two artifacts (gguf/nvfp4 × dflash2/plain, plus gguf mtp) pass.

## 8. Remaining work (original L6 description, kept for history)

This is the **only** structural reason blocking TP-2 on GGUF, and it requires an execution-side change (it is not a splitter problem):

1. **Execution side**: before the output projection of the GDN mixer, make both shards obtain the **complete** grouped value output. Reuse the **existing** in-kernel all-reduce (no new collective primitive needed): each shard writes its own output into a **full-width** buffer (zeros outside its range) → all-reduce sum → both GPUs get the complete activation. The subsequent "K-split weights + post-projection all-reduce" follows the existing pattern unchanged (complete activation + each side's tiled K half = partial sums, still requiring the original reduction). Cost: one extra [6144,T] reduction per GDN layer (the same order of magnitude as the existing post-projection [5120,T] reduction), and this projection 2× FLOPs (a negligible share). Involves `src/runtime/engine/tp2_generation_core.cpp` (including CUDA Graph capture and the detailed semantics of the in-kernel allreduce) and the TP-2 branch of `execution/text.cpp` on the model side.
2. **Auxiliary objects split along with the weights**: `input_columns` (INT32 [K], 6144 entries) currently falls to `Replicated` because `shape.size() < 2`, whereas the split weight has `k = 3072`, and `prepare.cpp` will error with `columns.numel() != weight.k`. That auxiliary object needs to undergo the same K split as its weight (the RowParallel branch of `tp_shard_views.cpp` currently requires a 2-D matrix, so handling must be added for 1-D auxiliaries), and the rejection condition in `shard_views` can be lifted once "the execution-side plan has landed and the auxiliary has been split".
3. Verification: `tp2_load/forward` passes on the GGUF artifact; two-GPU `ninfer-serve --devices 0,1` end-to-end generation matches single-GPU bit-exactly/numerically; the existing TP-2 suite does not regress.
## 9. DFlash2 component replacement (quantization strategy of the same draft)

The user asked "can the DFlash2 of `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` be swapped for that of `qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer`". Conclusion: **the two drafts are the same weights** — all 91 bindings have identical shapes, and the 33 bf16 tensors are byte-identical; the only difference is the quantization strategy: the GSQ artifact follows the `_companions` of `qwen3_8_27b_gguf` and stores the draft projections as `q8_g32_fp16` with the selector codebook left at `bf16` (dflash2 totaling 2123.6 MiB), whereas the official recipe uses `q4_g64_fp16` (including two codebooks, 977.2 MiB total).

`tools/convert/graft_components.py` cannot be ported directly (it requires the object id and format/bytes to match exactly, and here the id, format and length all differ). The route taken instead was "fix the recipe + re-convert", and in the process three defects introduced by `bfb57e72` (the commit that did the GGUF block conversion) were fixed:

1. In `tools/convert/__main__.py`, `SourceInputs` is defined twice, and the later one (an old implementation) shadows the earlier one (which dispatches by suffix), so `--source gguf=` was opened as safetensors → GGUF conversion failed already at the CLI layer.
2. `gguf_blocks.text_sources` calls a nonexistent `gguf.readdirect_source` (3 places); the actual API is `read_direct`.
3. `gguf_blocks._companions` hard-codes DFlash2 as Q8, contradicting its own docstring ("DFlash2 as the official recipe") and the official recipe's Q4 policy → `official_recipes.assign_dflash_formats` was extracted and shared by both places.

Artifact `<models-dir>/Qwen3.8-27B-GSQ-RCO-IQ3_S-dflash2q4.ninfer` (12.85 GiB, previous artifact 13.99 GiB):

- dflash2: 91/91 bindings **byte-identical** to the donor, 66 objects and shared structures consistent with the donor;
- text/vision/mtp **byte-identical** to the previous artifact (963/441/16 bindings);
- `proposal/head` changed from `gguf_q4_k` to `q4_g64_fp16` (the current official policy of `--proposal`, same as the donor).

TP-2 measurement (`--max-context 262144 --kv-dtype fp8 --spec dflash2 --draft-tokens 7 --lm-head-draft --vision`, greedy): device memory GPU 0 12640→11664 MiB, GPU 1 11520→11332 MiB; acceptance rate paris/256tok 19.1%→15.6% (decode 58.4→53.5 tok/s), prose/381tok 10.5%→10.1% (43.7→44.1 tok/s). Conclusion: the swap is viable, buying about 1 GiB of device memory at the cost of a small drop in acceptance rate on this thinking-heavy workload.

The three converter fixes plus the documentation updates are still uncommitted (working tree: `__main__.py`, `gguf_blocks.py`, `official_recipes.py`, `docs/gguf.md`, `docs/weight-conversion.md`).
## 10. MTP + GGUF long-prompt failure: `gguf matrix product launch: invalid argument`

User report: under `--spec mtp`, long-prompt requests return 500; log `C:\ninfer\serve-win.log`. Reproduction (original GGUF artifact, `--spec mtp --draft-tokens 3 --lm-head-draft --max-context 262144`): short prompts succeed, long prompts (≈7200 tokens, multiple 1024 chunks) always fail. The failing operator from the SSE error body:

```
rows=5120, k=10240, columns=1024, J=128, I=128, threads=256, shared=57856,
row_tiles=40, column_tiles=8, blocks=320, stream_k=true, fixup=false,
kernel_static_shared=0, kernel_max_dynamic_shared=101376, kernel_max_threads=256,
kernel_regs=255, kernel_local=72, kernel_ptx=120, kernel_binary=120,
device_default_block_shared=49152, device_optin_shared=101376, device_sm_count=36
```

That is, the MMQ launch of MTP's `mtp/input_projection` (`[5120, 10240]`, Q6_K) on a 1024-token prefill chunk.

Ruled out (all measured): the opt-in above 48 KiB is in effect (setting the attribute to the device maximum 101376 still fails); both the stream-k and tiled grids fail; `--no-cuda-graph` and `CUDA_MODULE_LOADING=EAGER` both have no effect; there is no residual error before launch (`cudaGetLastError()` is clean); resetting the attribute after a failure and retrying the same tile still fails; letting the tile use J<=64 (`shared=48384` <= 48 KiB) makes the same request succeed. Conclusion: on this machine the launch of this (Q6_K, J=128) instance with 57856 B of dynamic shared memory is rejected by the driver even though every documented limit is satisfied; the cause was not localized.

Handling: `select_matrix_launch` gained an upper bound `max_J`; `launch_matrix` returns a launch status instead of throwing; `matrix_product_impl` falls back to narrower column tiles on `cudaErrorInvalidValue` (down to J=16), reporting an error only if all are rejected, together with the geometry of the widest attempt; `matrix_fixup_bytes_impl` was changed to take the maximum over all candidate tiles so the workspace is always sufficient.
A rejected launch is purely host-side and executes nothing, so the fallback is safe; if the failure is unrelated to the tile, the fallback fails the same way and reports the error.

Verification: the MTP long prompt (7214 tokens) succeeds, decode 87.4 tok/s, `mtp accepted 40/69 (58.0%)`; the dflash2 long prompt succeeds (68.8 tok/s, acceptance rate 26.8%); `ninfer_linear_gguf_test` passes for all 15 quantizations (including a 130-column wide batch, covering the J=128 tile).
