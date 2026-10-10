# TP-2 adaptation work log (2x RTX 5060 Ti, Qwen3.8-27B NVFP4)

Archived chronological record of the TP-2 adaptation: the working plan kept verbatim, from the
initial plan through the final round. Sections 0-5 are the original planning document (goals,
hardware, design decisions, phase plan, risk register, collaboration protocol); section 6 onward is
the round-by-round record, including the delivered benchmark/quantization/context work and the open
MTP-consistency item. Line numbers and round numbers are not sequential: early rounds were logged on
demand, one round number was reused, and later entries were appended under the header that was
current when they were written.

The body was reconstructed from this session's transcript while the plan file was being cleaned up,
so it is the record as it stood at delivery. It is a historical log, not an active authority:
current state, recommended configuration, and measurements live in
[Dual RTX 5060 Ti TP-2 tuning](../tp2-dual-5060ti.md), and remaining work is tracked in
[tp2-plan.md](tp2-plan.md). Five plan snapshots are archived verbatim below under their own
`PLAN.md archive` headers: the 2026-09-21 upstream cherry-pick re-plan, the 2026-09-22
delivery-cleanup rewrite, the 2026-09-24 post-protocol-refactor state, the 2026-09-25
pre-organization state (r67-r70 draft-quantization campaign, TP-2 transport investigation,
TP-2 orchestration/rendezvous work), and the 2026-09-27 pre-organization state (upstream
ninfer-all performance-item port, architecture-wide review, FP8 PV, cross-tile phase
pipelining).

References in the record were rewritten when it was published: the working plan file is now
`tp2-plan.md`, and cross-references cite this worklog by section (`worklog §3.6`) or by round
heading (`worklog Round 11`) rather than by line number, because the archived snapshots shifted the
line numbering. The `PLAN.md archive` headers below are the original snapshot titles and were left
as written.

Evidence artifacts referenced by the record live outside the repository: the WSL build tree
(`$HOME/ninfer`), logs under `$HOME/prof/`, and the driving scripts. Only the retained measurement
routines are published, under `tools/tp_bootstrap/`.

---

# NInfer TP-2 adaptation plan (2× RTX 5060 Ti, Qwen3.8-27B NVFP4)

> This plan is persistent memory across context compression. Update section 6 progress after each step completes.
> Created: 2026-07 (session dates per git log). Status: **planning complete, awaiting Phase 0**.

---

## 0. Goal and completion criteria

**Goal**: run `ninfer-serve` on 2× RTX 5060 Ti (16GB / 448 GB/s / PCIe x8 5.0 each) with tensor
parallelism TP-2, serving the official Qwen3.8-27B NVFP4 artifact (23.7GB, includes text/mtp/dflash2,
vision disabled at startup).

**Completion criteria** (all must be met):
1. The TP=1 path behaves identically to before the change (regression must not break the single-GPU path);
2. Every split Op passes an independent oracle test (against the full Op, FP32/FP64 reference);
3. End-to-end: perplexity evaluation numerics are sound, and the user manually confirms generation quality is normal;
4. Performance floor: MTP0 decode ≥ 28 tok/s (75% of the 37.8 roofline), MTP3 ≥ 70 tok/s;
5. Stability: C=1/2/4 multi-request smoke test shows no OOM, no hangs; if conditions allow, run the 75-request corpus.

**Explicitly out of scope** (in this range): MoE 35B TP, pipeline parallelism, DP, continuous batching extension,
Windows-native port (going through WSL2), new architecture support.

---

## 1. Hardware and environment (measured)

| Item | Value |
|---|---|
| GPU0 | RTX 5060 Ti 16GB, 448 GB/s (128-bit GDDR7 1750MHz), currently occupied by llama-server |
| GPU1 | Tesla T10 16GB (sm_61, cannot run sm_120a, idle, usable as the migration target for the session model) |
| GPU2 | RTX 5060 Ti 16GB, 448 GB/s, currently occupied by llama-server |
| Interconnect | PCIe x8 5.0 only (≈31.5 GB/s/direction), no NVLink |
| Host | Windows; NInfer requires Linux → **WSL2** (confirmed in Phase 0) |
| Host memory pressure | llama-server holds 15.7GB; compile with `-j 4` |
| Artifact | 23.7GB, **does not fit on one GPU** → no local TP-1 comparison baseline |

Performance roofline (decode, bandwidth-bound): 896 GB/s ÷ 23.7GB ≈ **37.8 tok/s** (MTP0 upper bound).
5090 measured efficiency 69–94% of roofline → dual-GPU expectation MTP0 26–34 tok/s, MTP3 70–125, DFlash2 90–180.

---

## 2. Key design decisions (with rationale)

1. **Fixed TP=2, dense 27B only.** No MoE split, no PP/DP. The launch option looks like
   `--devices <name1>,<name2>` (select by GPU name, not index — the T10 sits in between).
2. **Split at load time, artifact format untouched.** converter unchanged; the materialization stage reads the
   full tensor → splits in half → uploads each half. Reuses the official 23.7GB artifact; the user need not re-convert.
3. **Split modes** (per-layer structure of the 27B dense model: norm → mixer → norm → MLP):
   - QKV projection / gate+up (linear_swiglu): **column parallel** (split the output dim, replicate input)
   - o_proj / down_proj (linear_add): **row parallel** (split the input dim, all-reduce the output)
   - softmax attention: split by head; the KV pool stores only local KV heads per GPU; no communication inside attention
   - GDN linear attention: split by head; state [H,D,D] split by head; both chunked prefill and decode
     replay are head-parallel, no internal communication
   - gdn_input_proj / gdn_gating_proj: split by head
   - lm_head: split by vocab, all-gather the logits, single-GPU sampling
   - RMSNorm / bias / small weights: replicated (negligible size), operating on the full hidden after all-reduce
   - MTP / DFlash2 draft layers: the same set of patterns; the selector operates on the post-all-reduce hidden and is safe by construction
4. **Communication: homegrown P2P all-reduce, no NCCL introduced.** 2-GPU all-reduce = 3 D2D
   cudaMemcpyPeer calls + one add (~80KB/call, ~15µs). Two all-reduce points per layer
   (after mixer output, after MLP output) × 48 layers = 96/forward ≈ 1.4ms, acceptable.
   **Fallback**: if WSL2 blocks P2P, switch to host-staging (D2H→H2D via pinned memory),
   latency ×3–5 but function unchanged. Phase 0 measurement decides which path to take.
5. **CUDA Graph deferred.** No graphs in Phases 1–4 (correctness first); Phase 5 introduces dual graphs
   (one per GPU) + cross-device memcpy nodes (if capturable), otherwise the all-reduce stays outside the graph.
6. **Execution model: lockstep per layer.** Batch replicated (TP does not split the batch); the two GPUs execute the same
   Op sequence in parallel (independent streams), the layer-end all-reduce serves as the synchronization barrier. A single Engine worker thread drives both cards.
7. **Physical layer instantiated per shard**: arena / paged KV / state store / round buffers /
   workspace each hold 2 instances; the Program drives the two shards in parallel. This is the largest refactoring surface.
8. **Zero changes to the TP=1 path**: all new code is gated on shard count; the single-GPU path goes through the original route (regression protection).
9. **Frontend / Gateway / Engine request plane unchanged**: they are device-agnostic (that assumption to be verified in Phase 3;
   if hidden single-device dependencies surface, handle them locally).

---

## 3. Phase plan

### Phase 0: environment and baseline (1–2 days, needs a GPU window)
- [ ] Confirm WSL2 is available (`wsl --list`; if unavailable, trigger risk R2's alternative)
- [ ] Install the toolchain inside WSL2: CUDA 13.1, CMake ≥3.28, Ninja, FFmpeg dev, libcurl ≥7.85, Python 3.11
- [ ] Build NInfer TP=1 inside WSL2 (`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`)
- [ ] GPU enumeration: identify the two 5060 Ti by name (the T10 is in between, indices unreliable)
- [ ] **P2P measurement**: a small program measures cudaMemcpyPeer bandwidth between the two cards → decides the communication scheme (R1)
- [ ] Single-GPU efficiency baseline: use Op micro-benchmarks (tools/bench or a temporary harness) to measure linear/attention kernel efficiency on the 5060 Ti, against the 5090 roofline ratio, to calibrate performance expectations
- [ ] Copy the artifact into the WSL2 filesystem (23.7GB, do not load it directly from /mnt/d)
- **Deliverable**: baseline data table (P2P bandwidth, kernel efficiency, WSL2 availability conclusion)

### Phase 1: device layer and communication (~1 week, mostly no GPU needed)
- [ ] core: DevicePair (2× DeviceContext), P2P enable
- [ ] core: all-reduce primitive (P2P version + host-staging fallback, compile-time/run-time switch)
- [ ] core: ShardedTensor view (Tensor + shard index + local geometry)
- [ ] core: dual arena, dual stream/event management
- [ ] Unit test: all-reduce numeric correctness (sum of random tensors checked against CPU)
- **Gate**: compiles + TP=1 regression passes + all-reduce unit test passes

### Phase 2: split Ops (2–3 weeks, some need a GPU window)
For each Op: implement the split entry + oracle test (full Op vs split Op + all-reduce, independent reference).
- [x] linear (**NVFP4 half-size, 5 shapes**: column parallel 3 + row parallel 2): oracle PASS (T=1/8/32, FP64 independent reference); the other qtuples are outside the 27B NVFP4 artifact scope, skipped
- [x] linear_swiglu: decomposed into half-size linear(gate_up [17408,5120]) + silu_mul (column parallel, output rows naturally partitioned, no communication needed); split logic lives in the execution layer (Phase 3)
- [x] linear_add: decomposed into half-size linear(down [5120,8704]) + residual_add + all-reduce (row parallel, all-reduce already verified in the linear oracle); split logic lives in the execution layer (Phase 3)
- [x] attn_input_proj: column parallel QKV [16384,5120]→[8192,5120] (half-size linear already verified); head parallelism needs no kernel change
- [x] softmax_attention: head parallel + local KV (kernel indexes by head, no change needed; KV pool local per GPU, Phase 3 execution layer)
- [x] gdn_input_proj: column parallel [14336,5120]→[7168,5120] (half-size linear already verified); head parallel
- [x] gdn_gating_proj: split by head (head parallel, no kernel change needed)
- [x] gated_delta_net: head parallel + local state (kernel indexes by head, no change needed; state pool local per GPU, Phase 3 execution layer)
- [x] kv_cache append: local pool per GPU (Phase 3 execution layer/Program)
- [x] lm_head: vocab split [248320,5120]→[124160,5120]×2 + all-gather + single-GPU sampling (BF16, Phase 3 execution layer)
- [x] sparse_moe: **skipped** (27B is dense)
- **Gate**: compiles + all Op oracles pass + TP=1 regression passes → **PASS** (half-size linear oracle + tp_device_pair all pass; TP=1 path unchanged)

### Phase 3: load and model integration (1–2 weeks)
- [ ] Materialization: split at load (read full → split in half → upload each separately)
- [ ] Parameters: split geometry
- [ ] Planning: dual-device VRAM budget, dual KV capacity resolution (`--kv-capacity auto` per free space per card)
- [ ] Program: dual State/KV stores, dual round buffers, per-layer lockstep execution + all-reduce points
- [ ] Verify the Frontend/Gateway/Engine device-agnostic assumption
- **Gate**: TP-2 loads the artifact successfully + single-token forward runs (numerics checked against Phase 2 oracles)

### Phase 4: end-to-end correctness (~1 week, GPU window)
- [ ] Single-request generation smoke (greedy, short output, manual readability check)
- [ ] Perplexity evaluation (project perplexity tool, small corpus)
- [ ] C=2 / C=4 multi-request smoke
- **Gate**: user test round 1 passes (or the issue is fixed)

### Phase 5: CUDA Graph (~1 week)
- [ ] Dual graph capture (one per GPU, exact-B decode)
- [ ] Cross-device memcpy into the graph (if capturable); otherwise the all-reduce stays outside the graph
- [ ] Performance comparison: graph on vs off
- **Gate**: graph-path output matches the non-graph path + no performance regression

### Phase 6: speculative decoding (1–2 weeks, GPU window)
- [ ] MTP under TP (draft layers TP + verification batch TP)
- [ ] DFlash2 under TP (selector / proposal head / verification)
- **Gate**: user test round 2 passes

### Phase 7: tuning and delivery (~1 week)
- [ ] prefill chunk / KV dtype / concurrency tuning (per measured values)
- [ ] 75-request corpus benchmark (tools/bench), record makespan and decode tok/s
- [ ] Documentation: add a TP-2 section to README and docs (options, limits, performance)
- **Gate**: user test round 3 (final) passes → goal complete

---

## 4. Risk register

| # | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| R1 | WSL2 virtualization layer blocks P2P | Medium | High (all-reduce degrades to host-staging, latency ×3–5) | Phase 0 measurement; host-staging fallback already in the design |
| R2 | WSL2 unavailable | Low | High (no Linux environment) | Alternative: Linux VM + GPU passthrough; or charter a Windows port (out of scope) |
| R3 | Physical-layer refactor exceeds expectations | Medium | Medium (schedule) | TP=1 path gated for isolation; per-Phase compile gates; small commits |
| R4 | Cross-device CUDA Graph not capturable | Medium | Low (all-reduce outside the graph, ~1.4ms/forward) | Phase 5 already has a fallback |
| R5 | Tight VRAM (~10GB weights/card + KV + workspace) | Medium | Medium | Disable vision; disable DFlash2 if necessary; shrink the prefill chunk; FP8 KV |
| R6 | T10 interferes with device enumeration | Low | Low | Select devices by GPU name |
| R7 | llama.cpp occupies both cards | Certain | High (blocks all GPU tests) | GPU window protocol (see section 5); or the user migrates the session model to the T10 |
| R8 | No TP-1 comparison baseline (23.7GB does not fit on one GPU) | Certain | Medium (weaker correctness-verification chain) | Per-Op oracle + perplexity + user judgment triple backstop |

---

## 5. Collaboration protocol

- **GPU window**: when GPU tests are needed I announce "GPU window needed", the user stops llama-server,
  I test, the user restarts. **Starting/stopping llama.cpp is the user's action only**; I never touch it.
- **Compilation**: I run it inside WSL2, the user need not participate; on compile failure I loop-fix it myself.
- **Test rounds**: user tests → sends logs → I fix → hand back to the user to test; expected 2–4 rounds.
- **Progress**: update section 6 after each step completes; after context compression/session resume, re-read this file first.

---

## 6. Progress

### Round 1 (environment reconnaissance + key decisions)
- [x] WSL2 available: Ubuntu-24.04, WSL 2.7.10, kernel 6.18, passwordless
- [x] WSL toolchain: cmake 3.28.3 / ninja 1.11.1 / gcc 13.3 / git / python3.12; **CUDA 12.8** (/usr/local/cuda, not 13.1, CMake has no version floor, usable)
- [x] sm_120a compile verification passed (nvcc 12.8, -arch=sm_120a kernels runnable)
- [x] Device enumeration inside WSL: measured order **GPU0=5060 Ti, GPU1=5060 Ti, GPU2=T10 (the distractor)** (Round 1's initial ordering was wrong; Round 2 confirmed by name with `pick_devices()`) → device selection must be by name/whitelist, TP device pair = (0, 1)
- [x] **P2P measurement: 0↔2 bidirectional canAccessPeer=0 (WSL2 virtualization layer intercepts)** → all-reduce main path fixed as **host-staging** (pinned memory + CPU-side add, dual-stream parallel D2H/H2D); the P2P path is retained as a runtime-probed optional optimization
- [x] FFmpeg dev / libcurl installed; pkg-config missing (installed this round)
- [x] pkg-config installed; FFmpeg dev + libcurl 8.5 installed (inside WSL)
- [x] Artifact: user provided `<models-dir>\qwen3_8_27b_nvfp4.ninfer` (23,719,715,076 bytes, matches the manifest)
- [x] Repo copied into WSL `$HOME/ninfer` (63MB, excluding .git/models)
- [x] **CMake pitfall 1**: CMake defaults to picking `/usr/bin/nvcc` (12.0) → must explicitly pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` (written into build_wsl.sh)
- [x] **CMake pitfall 2**: nvcc 12.8 reports a shared-memory limit (`0xc000 max`) for sm_120a static device link → **installed CUDA 13.1** (matching the 5090-verified toolchain); under 13.1 the 429/429 full build passed
- [x] **CMake pitfall 3 (critical)**: on the WSL2 driver 591.59, **dynamic `libcudart.so.13` segfaults during CUDA initialization** (`cudaGetDeviceCount`→`pthread_once`→WSL `libcuda`→`libnvidia-ptxjitcompiler` SIGSEGV; A/B control proved only dynamic linking crashes, static linking passes) → added CMake option `NINFER_STATIC_CUDART` (default OFF to preserve product behavior, ON for the WSL build), 5 targets re-linked against `CUDA::cudart_static`
- [x] **Device mapping correction**: the actual WSL order is **0=5060 Ti, 1=5060 Ti, 2=T10** (the 0/1/2 order recorded in Round 1 was wrong); `pick_devices()` matches by name, TP pair = (0,1)
- [x] **Phase 1 complete**: DevicePair + all-reduce (P2P probe + host-staging fallback) + ShardedTensor/shard_view/column_shard/row_shard + unit test `ninfer_tp_device_pair_test` PASS (numeric correctness at 20480B/16B/1MiB, move semantics verified, `p2p_available=0` confirms the host-staging path)
- [x] Full build (product+tests, 672/672) passed; product CLI static cudart confirmed
- [x] **Phase 2 linear (NVFP4) oracle PASS**: `ninfer_linear_tp2_split_nvfp4_test` covers 5 half-size shapes ([7168,5120]/[8192,5120]/[17408,5120] column parallel + [5120,3072]/[5120,8704] row parallel), T=1/8/32 all pass against the FP64 independent reference
- [x] **Production splitter `ninfer::tp::split_weight`** (`src/core/tp/weight_splitter.{h,cpp}`): NVFP4 column split (contiguous byte slices of row blocks) + row split (code strided by row + scale remapped by tile), geometry derived deterministically from (n,k) (scale_offset=align_up(n*k/2,256)). The oracle was refactored to call the production splitter directly → **PASS** (validates the production code path)
- [x] **Phase 2 complete**: fused Ops decomposed into half-size linear + local element-wise Op (linear_swiglu→linear+silu_mul column parallel; linear_add→linear+residual_add+all-reduce row parallel), split logic lives in the execution layer; attention/GDN head parallelism needs no kernel change
- TODO: Phase 3 dual Parameters construction + TP execution coordinator (lockstep + all-reduce) + KV/State pool split + lm_head vocab split + Program/Engine integration

### Round 3 (artifact-measurement correction + Phase 3 load/execution skeleton)
- [x] **Artifact-measurement correction** (the dump tool reads the 23.7GB cold directory directly): the attn/GDN input projections are **FP8_E4M3FN_ROW_BF16 RowScale** (not NVFP4), and are **multi-part concatenated** objects (attn: q6144+k1024+gate6144+v1024; gdn: q2048+k2048+v6144+z6144). A blind N/2 row split would mismatch the heads → changed to **per-part row gather**
- [x] **`gather_weight_rows`** (`src/core/tp/weight_splitter.{h,cpp}`): for a fused parent object, split each part independently in half and concatenate; supports the three geometries NVFP4 / FP8 RowScale / BF16; `WeightSplitKind::GatherRows` + `TPObjectSplit.parts` wired into the materializer; `shard_geometry` gains an FP8 RowScale branch. core compiles
- [x] **Registered half-size FP8 shapes**: `n8192_k5120` (gdn qkv+z shard), `n7168_k5120` (attn qkv+gate shard), wired into fp8_shapes.h / fp8_dispatch.cpp / sources.cmake. ops compiles (Phase 2 already verified the same geometry)
- [x] **Execution-layer refactor**: `TextContext` extracts `mixer_layer` / `mlp_layer` (private) + `single_layer` (composed) + the public `tp_mixer_layer` / `tp_mlp_layer` surface; `run_layers` becomes a loop over `single_layer` (TP=1 behavior unchanged). All-reduce insertion points = the two spots per layer: after the mixer + after the FFN
- [x] Full-build verification (passed after Round 4 fixes)
- TODO: TP execution coordinator + dual KV/State pools + lm_head vocab split + Program/Engine integration (`--devices` option)
### Round 4 (FFN-only split design + dual-shard Model load path)
- [x] **Architecture decision: FFN-only split (first milestone)**. Measurement found that the two fused Ops attn_input_proj / gdn_input_proj hardcode full-model row counts at the wrapper layer (attn: q6144/kv1024/rows14336; gdn: q2048/k2048/v6144/z6144, the projection check + conv + recurrent state all hardcoded). Head splitting would require Op changes (high risk) → replicate attention and the GDN mixer in full on both cards (Ops zero-change, KV/state pools full), split only the FFN (gate_up column parallel [17408,5120]→[8704,5120]x2; down row parallel [5120,17408]→[5120,8704]x2) + lm_head column split (vocab split, logits all-gather). One all-reduce per layer (after the FFN; the mixer output is already complete on both cards). VRAM ~10.3GB/card (text-only), MTP0 roofline ~52 tok/s > 28 target. Attention/GDN head splitting left as a later optimization
- [x] **tp_split_spec.cpp rewritten to FFN-only**: gate_up GatherRows (2 parts), down RowParallel, output_head ColumnParallel, the rest Replicated (including all mixer weights). Removed the conv GatherCols branch (conv is now replicated)
- [x] **tp_shard_views.{h,cpp}** (new): builds shard views directly from PendingWeight.reference.binding.parts + the Directory object shape + the spec (no full single-GPU materialize — 23GB does not fit on one card); parts re-pointed at the shard backing's device parent, halved per split kind. Wired into loading_sources.cmake
- [x] **Dual-shard Model load path**: Model gains a make_shard_model friend factory; load.h declares materialize_model_tp2 (LoadPlan friend); load.cpp implements: after plan_load → build_tp_split_spec → tp::materialize_tp2 (dual-card upload) → dual shard_views → dual make_shard_model
- [x] **Link fixes**: gather_weight_rows/gather_weight_cols were originally inside an anonymous namespace (invisible to tp_materialize.cpp) → renamed *_impl + a namespace-scoped public wrapper
- [x] Full-build verification (including ffn_delta/tp_mlp_delta + tp2 load test, exit 0)
- [x] **All-reduce correctness fix (critical)**: mixer replicated ⇒ the FFN input residual x is **identical** on both cards; all-reducing the full residual would count x twice. The correct approach = all-reduce only the **FFN delta** (the row-parallel down output, before the residual add), then add the shared residual back once. Added `ffn_delta` (ffn.{h,cpp}, writes the delta without the residual add) + `TextContext::tp_mlp_delta` (post-norm + ffn_delta); one all-reduce per layer (FFN delta)
- [x] **TP-2 load smoke test** `ninfer_qwen3_5_tp2_load_test` (tests/models/qwen3_5/test_tp2_load.cpp): plan_load → materialize_model_tp2 → dual execution::Parameters + shard-shape check (gate[8704,5120]/down[5120,8704]/head[124160,5120]/mixer replicated). Measured: the full host-side path passed, the device upload OOMed (both cards 92% occupied by llama.cpp, 0.9GB free; the design needs ~7GB/card, fits after llama.cpp stops)
- TODO: TP execution driver (lockstep dual TextContext + per-layer FFN delta all-reduce + lm_head column-split all-gather sampling) → single-token forward smoke → Program/Engine integration (`--devices` option) → MTP/dflash2 component split

### Round 5 (TP execution driver + single-token forward smoke)
- [x] **TP execution driver `TextContext::forward_tp2`** (text.{h,cpp}, from line 1168): the bind lambda binds replicated mixer state on the dual shards (position 0, envelope{1,1}, batch=1/width=1, GDN state slot 0, KV table row 0, all Scoped* RAII guards) → both cards embed the same token → `run_layers_tp2` (lockstep mixer + per-layer FFN delta all-reduce + residual_add) → both cards run the final rmsnorm → the column-parallel lm_head each computes a half-vocab partial (shard0=[0,V/2), shard1=[V/2,V)) → cudaMemsetAsync zero the remote half of each other's buffer → `pair.allreduce` sums to full-vocab logits (identical on both cards)
- [x] **`forward_tp2_token`**: after forward_tp2, `ops::argmax` (valid_rows=vocab) + D2H + stream sync, returns the token id
- [x] **Single-token forward smoke test** `ninfer_qwen3_5_tp2_forward_test` (tests/models/qwen3_5/test_tp2_forward.cpp + registered in tests.cmake, SKIP_RETURN_CODE 77): materialize_model_tp2 → minimal execution state on dual shards (one tiny PagedKVCache per card: full-attn layer count, capacity 64, one execution-table row; GDN state pool 1 slot all-zero; 1GiB workspace) → dual TextContext cards → bidirectional `forward_tp2_token` → asserts the two shards' argmax agree (the shard-consistency invariant of replicated mixer + all-reduce delta + gather logits)
- [x] Full build GREEN (-j 4, CUDA 13.1, NINFER_STATIC_CUDART=ON)
- TODO: **GPU window** — after the user stops llama-server, run `ninfer_qwen3_5_tp2_forward_test --artifact <models-dir>/qwen3_8_27b_nvfp4.ninfer` to verify single-token forward numeric correctness → Program/Engine integration (`--devices` option, TP prefill/decode variants) → MTP/dflash2 component split

### Round 6 (OOM root-cause fix: per-shard arena capacity)
- [x] **OOM root cause**: `materialize_tp2` initialized each shard's `DeviceArena` with the **full-model** `plan.device_capacity_bytes` (~23.7GB), and the `DeviceArena(capacity)` constructor does `cudaMalloc(capacity)` → each card over-allocated by ~13GB, a 16GB card OOMs necessarily (even with llama.cpp stopped and both cards idle). Fix: added `object_shard_bytes` (resolves per-object shard bytes per spec: Replicated counts full, GatherRows/GatherCols count the sum of parts' half, ColumnParallel counts n/2, RowParallel counts k/2, all resolved via `shard_geometry`) + `shard_capacity_bytes` (sum over the full plan + 256B alignment slack per object); `tp_init` now passes per-shard capacity (estimate ~10.5GB/card + test 1GiB workspace ≈ 11.5GB < 16GB). Build GREEN
- TODO: after the user stops llama-server, re-run the single-token forward smoke (validate the capacity fix + forward numeric correctness) → Program/Engine integration → MTP/dflash2 component split

### Round 7 (OOM second root cause: current-device semantics + forward driver device binding)
- [x] **Second root cause (user observation: GPU2 spiked to 80% then dropped, GPU0 unmoved)**: the `cudaMalloc` inside `DeviceArena(capacity)` lands on the **current device**; `materialize_tp2` ran two consecutive `tp_init` calls without switching devices (DevicePair construction last binds device1) → both shards' weight arenas all allocated on GPU2. Fix: `deviceN.bind_to_current_thread()` before each `tp_init`
- [x] **Forward driver device binding** (text.{h,cpp}): `run_layers_tp2` binds before operating on the peer card each time (6 spots in mixer/delta alloc/mlp_delta/residual_add); `forward_tp2` rewritten — the original bind lambda's RAII guards were destroyed when the lambda returned (the binding never took effect), now lifted to function scope (make_bind returns a BindState + 16 Scoped* guards live until forward ends); removed the leading/trailing `work_.reset()` (which would clobber the caller's logits buffer); bind the corresponding device before each step's operations; `forward_tp2_token` binds its own card before argmax (after forward_tp2 returns, the current device is parked on the peer card)
- [x] **Test build_shard_state**: `device.bind_to_current_thread()` before creating the 3 arenas per card (kv/state/workspace)
- [x] Build GREEN
- TODO: after the user stops llama-server, re-run the single-token forward smoke (expect ~11.5GB per card, output argmax_shard0==argmax_shard1) → Program/Engine integration → MTP/dflash2 component split

### Round 8 (splitter gains FP8 RowScale row split: output_head)
- [x] **New error `weight splitter: unsupported qtype` root cause** (dump7 measured): `text/output_head` is **fp8_e4m3fn_row_bf16[248320][5120]** (not NVFP4), the spec marks it ColumnParallel (vocab row split in half) going through `split_weight`, but split_weight only supports NVFP4/BF16. down is NVFP4 (Phase 2 verified), gate_up goes GatherRows; the only omission was output_head's FP8 row split
- [x] **`slice_fp8_rows`** (weight_splitter.cpp): FP8 RowScale row split = code plane (n×k bytes, 1B per element) + scale plane (n×2 bytes, 1 BF16 per row) as two contiguous copies; shard payload aligned per the `shard_geometry` RowScale layout (256B-aligned after code, then scale), consistent with the upload geometry. `split_weight` gains an FP8_E4M3FN_ROW_BF16 branch (ColumnParallel only, n must be even; 248320/2=124160 satisfies)
- [x] **lm_head half-size FP8 shape registration** (proactive sweep, avoiding the next `fp8 linear: unsupported shape`): `forward_tp2`'s column-parallel lm_head uses [124160,5120] FP8 weights going through `fp8_dispatch`, but the dispatch table only has full-size kFp8N248320K5120. Added `shapes/n124160_k5120.cu` (following n14336_k5120.cu's generic chunk launcher: GEMV@T=1 + SIMT≤11 + A8, Geometry=Fp8Geometry<124160,5120>) + fp8_shapes.h declaration + dispatch table + sources.cmake. The NVFP4-side FFN half-size shapes (kNvfp4N17408K5120 gate/up, kNvfp4N5120K8704 down) were registered in Phase 2; mixer replication uses full-size shapes, no additions needed
- [x] Build GREEN
- TODO: after the user stops llama-server, re-run the single-token forward smoke → Program/Engine integration → MTP/dflash2 component split

### Round 9 (mixed-precision FFN: FP8 column split + half-size shapes + ffn_delta decoupled from fused swiglu)
- [x] **New error `FP8 row-scale supports row splits only` root cause** (dump8/9/10 measured): the text component has `num_hidden_layers=64` (not 48); of the main model's 64 layers, the first 56 layers' FFN is NVFP4 and the last 8 layers (56-63) FFN is **FP8 RowScale** (mixed-precision conversion); dflash2 is a separate 5-layer draft. The spec matches by **shape** (`n==hidden && k==intermediate`→RowParallel) and also marked the 8 FP8 down [5120,17408] as RowParallel, while split_weight's FP8 branch only allows ColumnParallel → throws. The design intent is to split all 64 layers' FFN; FP8 column split must be added
- [x] **`slice_fp8_cols`** (weight_splitter.cpp): FP8 RowScale column split = code plane row-strided copy (k bytes/row, take [col_begin, col_begin+col_count)) + scale plane whole copy (n×2 bytes, 1 BF16 per row, independent of columns); shard payload aligned per the `shard_geometry` RowScale layout. `split_weight` FP8 branch extended: ColumnParallel→`slice_fp8_rows` (output_head), RowParallel→`slice_fp8_cols` (FFN down)
- [x] **FP8 half-size shape registration** (avoiding the next round's `fp8 linear: unsupported shape`): `shapes/n17408_k5120.cu` (gate/up shard, Geometry=Fp8Geometry<17408,5120>), `shapes/n5120_k8704.cu` (down shard, Fp8Geometry<5120,8704>), following n14336/n5120_k17408's generic chunk launcher; fp8_shapes.h declaration + dispatch table + sources.cmake. The corresponding NVFP4-side shapes were registered in Phase 2
- [x] **`ffn_delta` decoupled from fused swiglu** (a deeper blocker, proactively swept): the fused `linear_swiglu`'s FP8/NVFP4 kernels both **hardcode full-size geometry** `N34816K5120` (decode.cu throws when it checks `weight.n != 34816`); the shard gate_up [17408,5120] would die at layer 0. Changed to shape-generic `linear`(gate_up) + `silu_mul` (isomorphic to `ffn`'s mtp branch); `ffn_delta` is only called by TP-2's `tp_mlp_delta`, the TP=1 path (`ffn`) unchanged, byte consistency preserved
- [x] Build GREEN
- TODO: after the user stops llama-server, re-run the single-token forward smoke (expect ~11.5GB per card, argmax_shard0==argmax_shard1) → Program/Engine integration → MTP/dflash2 component split

### Round 10 (shard_views vector objects: matrix-shape check false-positives on 1-D weights)
- [x] **New error `shard view: weight object is not a matrix` root cause** (tp_shard_views.cpp:24): `shard_views` unconditionally called `object_shape` (requiring `shape.size()==2`) on **every** pending weight's part, but norms/biases are 1-D vectors (`input_norm`/`post_attention_norm` [5120], `final_norm` [5120], `a_log`/`dt_bias` [48], `gdn/norm` [128], etc.). These objects are Replicated; their branch only copies `begin/end` and uses `reference.shape`, never touching `n/k` — the matrix check was redundant for them but executed before the kind dispatch
- [x] **Fix**: moved the `object_shape` call from the top of the part loop into the non-Replicated branch (first line of the `else` block). The Replicated branch (norms/biases/mixers/embedding, including 1-D vectors) no longer queries the matrix shape; only GatherRows/RowParallel/ColumnParallel (all 2-D matrices) query it. `view.parts.push_back` stays at the part-loop level (both Replicated and split objects must register their region)
- [x] Build GREEN
- TODO: after the user stops llama-server, re-run the single-token forward smoke → Program/Engine integration → MTP/dflash2 component split

### Round 11 (shard_views GatherRows mapping: up block out of bounds)
- [x] **New error `text/layers/0/mlp/gate: projection region exceeds its parent or logical matrix` root cause** (weight_input.cpp:26): the error names gate, but the actually out-of-bounds part is the same object's **up** part. The `shard_views` GatherRows mapping `row_begin = cum + shard*(block/2)` is wrong — in the gather layout, the shard payload concatenates each part's "this shard's half-rows" in part order (the gather already selected rows with `rb = row_begin + shard*(row_count/2)`); the two shard blocks have the same structure, and a block's offset in the shard row space is just `cum` (the sum of preceding parts' half-rows), independent of shard. The extra `shard*(block/2)` made shard 1's up block row_begin=17408, exceeding the shard parent [17408,5120]'s element domain
- [x] **Fix**: `row_begin = cum` (tp_shard_views.cpp). The gate block cum=0, the up block cum=8704, both shards agree, region legal
- [x] Build GREEN
- TODO: after the user stops llama-server, re-run the single-token forward smoke → Program/Engine integration → MTP/dflash2 component split

### Round 12 (steady-state OOM: 1GiB workspace too large + weight arena measured 14.06GiB)
- [x] **Symptom**: both cards steady at ~14.2GB with no fluctuation in Task Manager; after running about 11 probes, `cudaMalloc failed: out of memory`. The steady state was too high and pierced the 16GB ceiling; the forward's temporary allocations (logits etc.) squeezed the margin out, not a single large-allocation spike
- [x] **Probe dump11 measured the per-shard weight arena** (tools/tp_bootstrap/dump11.cpp, replicating object_shard_bytes/shard_geometry logic, speculative=None): device_objects=659; Replicated 531 = 9358 MiB (mixer projections + token_embedding + norms, cannot be reduced further under FFN-only split); GatherRows 64 = 3357 MiB (gate_up half-split); RowParallel 64 = 1678 MiB (down half-split); **total 14.06 GiB**. dflash2/mtp not in the plan (when speculative=None, config.draft/mtp are not set, confirmed at config.cpp:321/333)
- [x] **Root cause**: steady state = weight arena 14.06GiB + workspace 1GiB + GDN state ~83MiB + KV ~3MiB ≈ 15.15GiB > a 16GB card's actual usable (minus driver/context ~15GiB). Single-token forward peak workspace is only ~600KB (logits [248320,1] BF16 = 496KB + several [N,1] tensors); 1GiB is seriously oversized
- [x] **Fix**: test_tp2_forward.cpp workspace 1GiB → 256MiB (plenty for a single-token smoke), and prints the MiB counts of the kv/gdn/workspace three arenas to ease the next diagnosis. Expected steady state ≈ 14.06 + 0.25 + 0.08 + 0.003 ≈ 14.4GiB, leaving ~0.6GiB of slack
- [x] Build GREEN (/tmp/full_build18.log)
- TODO: after the user stops llama-server, re-run the single-token forward smoke (expect ~14.4GB per card, all 11 probes pass) → if still OOM, further compression is needed (candidates: mixer split / embedding half-split, high risk, last resort) → Program/Engine integration → MTP/dflash2 component split

### Round 13 (shard_geometry missed updating padded_columns: RowParallel down verification failed)
- [x] **New error `text/layers/0 verify columns=1: nvfp4 linear: invalid NVFP4 weight` root cause** (nvfp4_format.cpp:57/64): `shard_geometry` (tp_materialize.cpp) does `WeightGeometry g = full;` then only updates shape/elements/planes, **missing the update to `padded_columns`** (and FP8 RowScale's `group_size`). The RowParallel down [5120,17408]→[5120,8704] shard parent's padded_columns is still 17408; `native_weight` (weight_view.cpp:270) sets `padded_shape[1]=17408` on that basis, but `k=8704` → the `padded_shape[1] != k` check throws. Gate_up [17408,5120] passed because k is unchanged (so the error is on down, not gate_up, but the error message carries the whole layer's verify context)
- [x] **Fix**: `g.padded_columns = k` in `shard_geometry`; FP8 RowScale additionally `g.group_size = k` (consistent with weight_geometry's RowScale branch, weight_view.cpp:131). Correct for NVFP4/FP8/BF16 ColumnParallel/GatherRows (k unchanged) and RowParallel (k halved)
- [x] Added diagnostic prints of the layer-0 gate_up/down Weight fields to the test ([diag] lines), to pinpoint the specific field if it errors again
- [x] Build GREEN (/tmp/full_build18.log)
- TODO: after the user stops llama-server, re-run the single-token forward smoke (expect ~14.4GB per card, all 11 probes pass) → Program/Engine integration → MTP/dflash2 component split

### Round 14 (GDN output projection misclassified as RowParallel: replicated mixer reads half weights OOB)
- [x] **New error `fp8_linear_add_decode.cu:29: cudaErrorIllegalAddress` root cause**: the crash is in the **GDN output projection** (`linear_add`, geometry N5120K17408), and **no `[tp2]` tag was printed at all** (meaning the fault is inside the local mixer, before the first sync). `build_tp_split_spec` (tp_split_spec.cpp) used the shape rule `n == hidden && k == intermediate` to judge the FFN down as RowParallel, but the **GDN output projection's shape is also `[hidden, value_width]`, and when `value_width == intermediate` (this model: 17408) it is exactly the same shape as the FFN down** → misclassified as RowParallel, the shard weight split into `[5120,8704]`; but the GDN mixer is **replicated** (the `linear_add` at `text.cpp:1059` reads the weight as full `[5120,17408]`) → out of bounds → illegal address. The attention output is `[h, query_width]` (≠17408), so the GDN output is the only other `[hidden, intermediate]` weight
- [x] **Fix**: the RowParallel judgment changed to `n == hidden && k == intermediate && has_name("mlp/down")`; the GDN output projection with the same shape but the name `gdn/output` stays **Replicated**. The FFN down binding name is `mlp/down` (load/text.cpp:29), the GDN output is `gdn/output` (load/text.cpp:53); name discrimination is reliable
- [x] Temporary instrumentation (to be deleted): `run_layers_tp2` (text.h) layer-0 four sync labels (local/peer mixer, local/peer ffn) + `ffn_delta` (ffn.cpp) three sync labels (gate_up linear / silu_mul / down linear), to localize the delayed fault to a specific kernel; removed after the smoke passes
- [x] Build GREEN (/tmp/full_build18.log)
- TODO: after the user stops llama-server, re-run the single-token forward smoke (expect ~14.4GB per card, all 11 probes pass) → remove instrumentation → Program/Engine integration → MTP/dflash2 component split

### Round 15 (peer side used card0's weight pointers: cross-device access OOB)
- [x] **New error `text.h:309 (peer mixer sync): cudaErrorIllegalAddress` root cause**: in `run_layers_tp2`, `block = parameters_.text.layers[layer]` is **card0**'s BlockParameters; the peer call `peer.tp_mixer_layer(block, x_peer, ...)` used **card0's weight pointers** (dev0 VRAM) on dev1's stream → WSL2 has no P2P, cross-device pointer access → illegal address. Evidence: `[tp2] local mixer` (card0 weights on card0 stream) printed cleanly, `[tp2] peer mixer` (card0 weights on card1 stream) faulted. `forward_tp2` had the same problem: the peer's embedding/rmsnorm/lm_head all used card0's `*embed_`/`*final_norm_`/`*lm_head_`
- [x] **Fix**: `run_layers_tp2` adds `const auto& block_peer = peer.parameters_.text.layers[layer];`; the peer's `tp_mixer_layer`/`tp_mlp_delta` now use `block_peer`; `forward_tp2`'s peer embedding/rmsnorm/lm_head now use `*peer.embed_`/`*peer.final_norm_`/`*peer.lm_head_` (lm_head is ColumnParallel, card1 uses the upper half of vocab rows). Each shard's Op must use its own shard's weight pointers
- [x] Build GREEN (/tmp/full_build18.log)
- TODO: after the user stops llama-server, re-run the single-token forward smoke (expect ~14.4GB per card, all 11 probes pass) → remove instrumentation → Program/Engine integration → MTP/dflash2 component split

### Round 16 (prompt attention kernel's smem opt-in only set on the current device: peer card launch rejected)
- [x] **New error `causal_cache/prompt.cu:59: cudaErrorInvalidValue` root cause**: after layer 0 (GDN) passed on both cards, the crash was in the **peer** mixer of the first full-attention layer (layer 3). Smem diagnosis: 5060 Ti `smemPerBlock=49152` (48KB default), `smemPerBlockOptin=101376` (99KB opt-in). The prompt kernel needs **98304B (96KB) dynamic smem**, over the 48KB default, within the 99KB opt-in → opt-in required. But `prompt.cu` used `static const cudaError_t attr_bf16 = cudaFuncSetAttribute(...)`, and static only runs on the **first call in the process**, while `cudaFuncSetAttribute` configures the **current device**'s kernel copy → the first was the local (dev0) mixer, dev1's copy never got the opt-in → still the 48KB default → the peer's 96KB launch rejected with `cudaErrorInvalidValue`. The GDN kernels (state_passing/output) set attributes **per call** (non-static), so both cards passed — prompt's `static const` was the odd one out
- [x] **Fix**: `prompt.cu`'s `attr_bf16`/`attr_i8` dropped `static`, now `cudaFuncSetAttribute` every launch (idempotent, per-call cost negligible), matching GDN's per-call pattern. This way, regardless of whether the current device is dev0 or dev1, its kernel copy gets the opt-in
- [x] `prompt.cu` added to build_phase4.sh's sync list
- [x] Build GREEN (/tmp/full_build18.log)
- [x] After the user stopped llama-server and re-ran: layer 0 (GDN) + all 64 layers' FFN passed on both cards (128 `down linear`); the crash point advanced to the lm_head projection

### Round 17 (lm_head and token_embedding weight binding: wrong ColumnParallel assumption)
- [x] **New error `linear: expected [K,T] x [N,K] -> [N,T]` root cause**: after all 64 layers' FFN passed, the crash was in the lm_head projection after the final rmsnorm. The `dump12` probe confirmed `text/output_head` and `text/token_embedding` are the **same physical object obj1** (weight-tied, elements=1271398400=248320×5120). obj1 is a resource object (not a 2-D TensorObject) → `build_tp_split_spec` skips it → no split entry → `shard_views` defaults to **Replicated**. So each shard's lm_head weight is **full [248320, 5120]**, but `forward_tp2` treated it as ColumnParallel (expecting n=124160) → `out.ne[0](124160) != w.n(248320)` → hit that throw exactly
- [x] **Fix**: lm_head binds to the replicated embedding; each shard holds the full [vocab, hidden] weight; hidden_out is identical on both cards (replicated mixer + post-all-reduce FFN delta), so each shard independently computes **full-vocab** logits, exactly identical on both cards, **no all-reduce needed**. `forward_tp2`'s lm_head section changed to `project(hidden_out, *lm_head_, logits, ...)` / `project(hidden_out_peer, *peer.lm_head_, logits_peer, ...)`, removing zero-pad + all-reduce. The full-head FP8 shape n248320_k5120 is already registered (same as TP=1)
- [x] Build GREEN (/tmp/full_build18.log)
- [x] User re-ran: forward **ran end-to-end** (the lm_head fix took effect, no more crash), but the 11-probe verification diverged: for `input_token=151643`, `argmax_shard0=198`, `argmax_shard1=151643` (shard1 returned the input token)

### Round 18 (located the two-card logits divergence: added per-layer x comparison diagnostics + fixed a diagnostic race)
- [x] **Structural analysis**: the TP-2 residual stream `x` itself is **not all-reduced**, only the FFN delta is. By induction: as long as (a) each layer's replicated mixer is bitwise identical on both cards and (b) the all-reduce is correct, `x` stays bitwise-consistent. Checked the `DevicePair::allreduce` (host-staging) implementation, no errors (both cards get a+b)
- [x] **Diagnostic v1 results** (`run_layers_tp2` compares the two cards' x max abs diff after each layer's mixer / after residual_add, layers 0-3 only): layers 0-2 (GDN) post-mixer+post-residual **all 0** (GDN mixer bitwise-consistent on both cards, FFN all-reduce correct); **layer 3 (first full-attention) post-mixer=0 but post-residual=0.265625 (at 3456)**, and data-dependent (appears for probe1=151643, absent for probe2=0)
- [x] **Diagnostic v1 reliability hazard (fixed)**: `cmp_x` used a synchronous `cudaMemcpy` (via the legacy default stream), but `residual_add` enqueues on `ctx_.stream` (`cudaStreamNonBlocking`, device.cu:64). **A non-blocking stream has no implicit synchronization with the legacy default stream** → the post-residual read could race residual_add, reading a mid-flight mix of old and new values. Layer 3's 0.265625 was very likely a **diagnostic artifact** (data-dependent = timing-dependent, exactly the race signature). Logically, post-mixer x identical → h identical → each card computes a different local delta (normal) → after all-reduce both cards get the same sum → post-residual_add x must be identical, so a reliable post-residual must be 0
- [x] **Diagnostic v2**: `cmp_x` does `cudaStreamSynchronize` on both streams before reading (eliminating the race); prints **all 64 layers** (layers 0-3 always printed, the rest only when nonzero). The post-mixer read was already reliable (sync0/sync1 have already synchronized both streams); the post-residual read is now reliable too
- [x] `Tensor::bytes()` checked correct (delta [5120,1] BF16 = 10240B, all-reduce covers the whole buffer), ruling out the "only partially covered" hypothesis
- [x] Build GREEN (/tmp/full_build18.log)
- [x] **Diagnostic v2 results (all 0) + a re-review found a bug in v2 itself**: after the user re-ran, layers 0-3's post-mixer/post-residual were **all 0**, but argmax still diverged (198 vs 151643). Re-reviewing `cmp_x` v2 found a **cross-device memcpy bug**: after `peer.ctx_.bind_to_current_thread()` the current device stayed device1; then `cudaMemcpy(a, x.data, D2H)` read device0's `x` — a synchronous cudaMemcpy resolves the device pointer against the **current device**, WSL2 has no P2P → `a` read cross-device garbage, the "all 0" was untrustworthy. v1's memcpy device binding was correct (bind device0 to copy x first, then bind device1 to copy x_peer); it only lacked the stream sync for post-residual (had the race)
- [x] **Logical contradiction**: if x were truly bitwise-identical after layer 63, then final rmsnorm (deterministic) → lm_head (full weights on both cards, same input) → logits must be bitwise-identical → argmax must be the same. The actual argmax differs → either x is not bitwise-identical (v2 bug's false 0), or there is a bug in the final path
- [x] **Diagnostic v3**: fixed `cmp_x`'s device binding (bind the corresponding device before each memcpy, eliminating cross-device reads); and added `cmp_final` to `forward_tp2`'s final path (compare `hidden_out` after rmsnorm, compare `logits` after lm_head, printing `[tp2-final] hidden_out/logits max_abs_diff=… at …`). One re-run can distinguish "the divergence is in some layer" (some `[tp2-diff]` nonzero) from "the divergence is in the final path" (`[tp2-final]` nonzero while all `[tp2-diff]`=0)
- [x] Build GREEN (/tmp/full_build18.log)
- [x] **Diagnostic v3 results (decisive)**: after the user re-ran, **all 64 layers' `[tp2-diff]` = 0**, **`[tp2-final] hidden_out = 0`, `logits = 0`** (both forwards) → **within a single forward the two cards' logits are bitwise-identical; the TP-2 forward computation is correct**; but `argmax_shard0=198`, `argmax_shard1=151643` still differ
- [x] **argmax determinism check**: the tiled-atomic path uses `argmax_better` (value descending, index ascending) atomic contention, a total order → the fixed point is unique, independent of atomic resolution order → identical logits imply identical argmax; argmax nondeterminism ruled out
- [x] **Root cause = test bug (GDN state not reset)**: the test runs two independent forwards per probe (card0-local, card1-local), but the GDN state pool is zeroed **once** at build only — the first forward updates the GDN state in place, the second forward starts from a **dirty state** → the two forwards' logits differ (L1≠L2) → argmax(L1)≠argmax(L2). Within each forward the two cards are always consistent (the diagnostics proved it); the dirty state only makes the two forwards differ from each other. 151643 = the input token fits exactly the signature of "dirty state shrinks the mixer output, the residual is dominated by the embedding, lm_head maps back to the input token". The test comment claimed fresh state but it was never actually reset
- [x] **Fix**: `ShardState` gains `state_backing` (DeviceSpan); the probe loop calls `reset_state()` before each forward to re-zero both cards' GDN state (bind the corresponding device + cudaMemset). KV needs no reset (at position 0 the bind state has kv_table_rows=0, attention only attends to itself, k/v recomputed from x)
- [x] Build GREEN (/tmp/full_build18.log)
- [x] **User re-run passed (decisive)**: all 11 probes' two-card argmax identical (151643→198/198, 0→49276/49276, …, 248319→177801/177801), all 176 `[tp2-diff]` lines 0, all 44 `[tp2-final]` lines 0, ending with `TP-2 single-token forward passed: all 11 probes shard-consistent` → **TP-2 single-token forward smoke milestone reached**
- [x] **All instrumentation removed**: ffn.cpp sync lambda+call+iostream; text.h run_layers_tp2's sync0-3/cmp_x/dbg+`<cmath>`/`<cuda_bf16.h>`; text.cpp cmp_final+two call sites; test smem print block+gate_up/down/output_head [diag] block (pick_devices' cudaDeviceProp is legitimate device selection, kept). Build GREEN
- [x] **Architecture judgment (critical)**: the Engine's Program is deeply bound to a single card (CUDA Graph decode, KV, context cache, MTP all single-card mechanisms), and TP-2's host-staging all-reduce is **fundamentally incompatible** with exact-batch CUDA Graph → full Engine `--devices` integration is a huge surface needing a CUDA Graph bypass. More valuable in the near term, and a precondition, is **stateful TP-2 decode** (verifying the two cards stay bitwise-consistent across multi-step GDN/KV evolution)
- [x] **`forward_tp2` generalized to stateful**: accepts a `position` parameter (default 0, the existing single-token smoke unchanged). `make_bind` sets the cache/rope position to `position` and the envelope to `{position+1, position+1}` (mirroring the Program ordinary decode convention decode.cpp:307-331); KV row 0 + GDN state slot 0 unchanged. The replicated mixer writes the current token's k/v into its own paged cache and updates the GDN state in place (`tp_mixer_layer`=`mixer_layer`, same path as the single-card `single_layer`) → state accumulates across steps
- [x] **`forward_tp2_token` internal two-card argmax comparison**: argmax both cards' logits; on mismatch `throw std::logic_error` (turns a shard divergence into a hard failure at the sampling boundary). decode calls `card0.forward_tp2_token` once per step (lockstep drives both cards + updates both cards' state simultaneously); no need to zero between the two forwards (unlike the single-token smoke's independent forwards)
- [x] **Test adds a 32-step autoregressive decode**: `ShardState` gains `kv_lease` (holds materialized page ownership); `publish_kv_row` materializes one page per card via materialize_one + `publish_repeated(row0, page, 1, device->stream)` (the H2D copy is enqueued on the device stream, because the attention kernel on the non-blocking stream does not implicitly sync with the legacy stream) + `cudaMemsetAsync` zeroing the GDN state enqueued on the device stream; the decode loop position 0..31, each step `card0.forward_tp2_token(card1, pair, token, step)`, the token from the previous step's sample
- [x] Build GREEN (/tmp/full_build18.log)
- TODO: Program/Engine integration (`--devices`, bypass CUDA Graph, go stateful TP-2 decode) → MTP/dflash2 component split → performance comparison baseline
- [x] **Round 21 test PASSED**: all 11 probes recovered Round 19's real values (198/49276/5328/14/220/381/264/1196/98142/149022/177801, bitwise-consistent on both cards); the 32-step stateful autoregressive decode's per-step two-card argmax agreed; ending `passed`. Round 20's two regression bugs (allreduce cross-stream race + materialize_one empty reservation) confirmed fixed
- [x] **Decode repetition-pattern check = not a bug**: the step 14–23 ≈ step 0–9 loop is the model's real degenerate behavior for "a single-token prompt, no context". Evidence: (a) page size=64 tokens, a single page holds all 32 steps, no growth needed; (b) attention uses the same verified `causal_softmax_attention` op as the single-card production decode, writing the cache_position column and reading 0..position columns each step; (c) **ironclad proof**: step 10 input 2782→output 5877, step 24 input 2782→output 12796 — same input, different history length→different output, proving the GDN recurrent state + GQA KV history both accumulate and are read. The two cards being bitwise-consistent per step is the TP-2 correctness criterion, and it is met
- [x] **TP-2 stateful autoregressive decode milestone reached**: lockstep dual TextContext + per-layer post-FFN all-reduce (explicit cross-stream sync) + replicated lm_head independent full-vocab argmax + stateful paged KV/GDN accumulation, all correct
- [x] **Round 20 first-run FAIL analysis (two independent bugs)**: (a) all 11 probes' argmax 0 (Round 19 had real values 198/49276/…); (b) the decode section's `materialize_one` threw `Paged KV single-page materialization exceeds reservation`
- [x] **Root cause (a) = removing instrumentation removed a load-bearing sync**: `DevicePair` owns its **own** `DeviceContext a_/b_` (created at construction, each with its own non-blocking stream); `allreduce` enqueues D2H copies on `a_.stream`/`b_.stream`; while the FFN kernels run on TextContext's `ctx_.stream`. **Two independent non-blocking streams have no implicit synchronization** → the D2H reads the delta before down-linear completes → stale/garbage delta → wrong logits. Round 19 passed because ffn.cpp's `sync("down linear")` (`cudaStreamSynchronize(ctx_.stream)`) was load-bearing — it guaranteed allreduce was only called after the delta completed; when Round 20 removed the instrumentation it was deleted as a pure diagnostic, exposing the race. Both cards read the same stale data (consistent after the allreduce sum) → argmax still the same, no throw, but the values wrong (all 0). This also explains Round 19's 8000+ line log (11 probe × 2 forward × 64 layers × 3 `[ffn_delta]` sync prints)
- [x] **Fix (a)**: `run_layers_tp2` explicitly `cudaStreamSynchronize(ctx_.stream)` + `cudaStreamSynchronize(peer.ctx_.stream)` before `pair.allreduce` (binding the corresponding device). This is the correct place in the TP-2 protocol — the producer must finish before the cross-stream exchange. The host-staging allreduce is itself a per-layer barrier (internally syncing the two pair streams), so no extra serialization is added
- [x] **Root cause (b) = `materialize_one` requires a nonzero reservation**: `make_empty_reservation()` returns a 0-page reservation; `materialize_one` throws when it checks `reservation.pages_ == 0`. The Program path `reserve(n)` first, then materializes
- [x] **Fix (b)**: the test's `publish_kv_row` does `pool.reserve(1)` first (returns a `std::optional<DeviceKVPageReservation>`, move-only — `std::move(*reserved)` on a `const` optional degenerates to a deleted copy, so const must go) before `materialize_one`
- [x] Build GREEN (/tmp/full_build18.log)
### Round 22 (Program/Engine integration: TP2GenerationCore + --devices CLI)
- [x] **Added `TP2GenerationCore`** (src/runtime/engine/tp2_generation_core.{h,cpp}): the third variant branch of the Engine Core, owning two independent DeviceContexts (shard_a_/shard_b_) + DevicePair + each its own paged KV / GDN state / workspace arena / TextContext. submit() creates the OutputSession + GenerationBudget; wait() drives stateful prefill (per-token forward_tp2) + autoregressive decode (forward_tp2 + ops::sample + OutputSession preview_model/commit_preview); both cards in lockstep, all-reduce after each layer's FFN (run_layers_tp2 has built-in cross-stream sync). lm_head replicated → each shard computes independent full-vocab logits → sampling uses only shard_a_ (both cards' logits are bitwise-identical, proven). memory_summary/runtime_stats/reset_memory_peaks/is_available all implemented
- [x] **Engine::Impl refactor**: a fourth unique_ptr<TP2Core> branch added to the Core variant; the Impl ctor branches on device_b>=0 — under TP-2 it does not construct a single-card device/ModelInstance but make_unique<TP2Core> and copies the Frontend from its frontend() into impl_->frontend_; the single-card path is unchanged. All frontend routing methods (prepare/prepare_tokens/tokenize_text/count_tokens/prompt_capabilities/media_cache_summary) go through impl_->frontend_ + impl_->capacity (under TP-2, active==nullptr, capacity=options.max_context). The destructor only syncs the device when active!=nullptr
- [x] **EngineOptions.device_b** (include/ninfer/types.h): default -1 (single card), >=0 enables TP-2. normalize_engine_options gains a TP-2 branch (concurrency=1, pending=1, kv_capacity=explicit(max_context), speculative={}, vision=false, cuda_graph=false, cache disabled)
- [x] **--devices A,B CLI** (src/serve/serve_options.{h,cpp}): ServeOptions.device_b defaults to -1; --devices parses two comma-separated device indices; generation_service.cpp passes device_b into EngineOptions
- [x] CMake: added tp2_generation_core.cpp to src/runtime/CMakeLists.txt
- [x] Build GREEN (/tmp/full_build_tp2.log)
- TODO: after the user stops llama-server, start ninfer-serve with --devices 0,2 for a live test (expect ~14.4GB per card, normal conversation) → MTP/dflash2 component split → performance comparison baseline
### Round 23 (TP-2 startup OOM: 512MiB workspace too large + 384MiB KV)
- [x] **Symptom**: after the user stopped llama-server, `--devices 0,2 --max-context 8192` started, both cards' VRAM loaded steadily to ~14.4GiB (weight arenas) then a cudaMalloc failed → process exited, VRAM zeroed (FATAL after 1m52s)
- [x] **Root cause**: TP2GenerationCore each shard occupies two more blocks than the verified single-token test — workspace 512MiB (test used 256MiB; the measured single-token forward peak is only ~600KB) + KV cache 384MiB (8192 context = 128 pages × 3MiB/page; the test only tested 64 tokens = 3MiB). Total 14.06+0.51+0.08+0.38 ≈ 15.03GiB > a 16GB card's actual usable (~15GiB) → the KV arena's cudaMalloc failed
- [x] **Fix**: kWorkspaceBytes 512MiB → 128MiB (peak 600KB, 200× margin). Per shard ≈ 14.8GiB, leaving ~0.2GiB of slack
- [x] Build GREEN (/tmp/full_build_tp2.log)
- TODO: user re-runs --devices 0,2 live → if still OOM, compress KV (--max-context 4096 = 192MiB) or split the mixer (high risk, last resort) → MTP/dflash2 component split → performance comparison baseline
### Round 25 (TP-2 startup failure: empty input to tokenizer_config.json)
- [x] **Symptom**: after the OOM fix, the user moved the artifact to `$HOME/models/qwen3_8_27b_nvfp4.ninfer` and re-ran; 29.7s after weight loading finished: FATAL: malformed tokenizer_config.json: parse error ... attempting to parse an empty input
- [x] **Artifact corruption ruled out**: the new path's byte count 23719715076 exactly matches the original /mnt/d path; the weights loaded normally → the artifact is intact
- [x] **Root cause (TP-2 dangling resource pointer)**: FrontendResources' 6 fields are std::string_view, pointing into the load plan's host resource bytes (binder host_object → MaterializationPlan.host_objects). The single-card path's artifact::materialize moves the plan's host bytes into each Model's backing_ (materializer.cpp:169), the addresses unchanged → the string_views stay valid. The TP-2 path's tp::materialize_tp2 only iterates plan.device_objects to build the two shard backings (pure device arenas); the host bytes stay in the plan; when materialize_model_tp2 returns, the plan (data) destructs → the host bytes are freed → the shard Model's resources_ string_views dangle → make_frontend parsing tokenizer_config.json reads empty. The single-card path and the Round 21 test never constructed a Frontend, so it was never exposed
- [x] **Fix**: FrontendResources' six string_view fields changed to std::string (resources.h); the Model owns its resource bytes itself, consistent with the architecture's "immutable Model data owns resources". bind_resources' string_view assignments implicitly convert to string; make_shard_model copies by value → deep copy. The single-card path benefits too (no longer depends on backing address stability). No aggregate initialization of a real FrontendResources inside src; test_frontend.cpp uses a same-named local struct (std::string fields), unaffected
- [x] Build GREEN (job pwsh-5, 285/285 targets, binary build/apps/ninfer-serve; only two pre-existing narrowing warnings at text.cpp:1193, unrelated to this change)
- TODO: after build GREEN, hand to the user to re-run --devices 0,2 --max-context 8192 (model path `$HOME/models/qwen3_8_27b_nvfp4.ninfer`) → expect startup to succeed into serve → the user sends logs to continue debugging
### Round 26 (TP-2 warmup failure: Paged KV materialization exceeds reserved capacity)
- [x] **Symptom**: after the tokenizer fix the engine was ready (30.6s), but 1.59ms into warmup: FATAL: Paged KV materialization exceeds reserved capacity; the log capacity line showed pages 0/0
- [x] **Root cause (two stacked bugs)**: (a) direct cause — execute() calls pool.materialize(reservation, 128, shard.kv_pages) per request, but materialize checks target_page_count > destination.capacity() (paged_kv_cache.cpp:298); shard.kv_pages was an empty vector (capacity 0) → 128 > 0 threw; the caller must reserve(128) first. (b) latent cross-request leak — even with reserve first, per-request reserve(128) accumulates reserved_pages_, and materialize turns 128 pages into a resident lease stored in shard.kv_pages (retained across requests) → the 2nd request's available_pages()=0 → reserve returns nullopt → "reservation failed". The fundamental issue: KV pages are a fixed physical allocation that should be established once at startup and reused in place across requests (max_concurrency=1, block-table mapping constant), not rebuilt per request
- [x] **Fix**: KV page materialization moved from execute() to build_shard() (once at startup): reserve(128) → kv_pages.reserve(128) → materialize → publish execution row 0 (the kv_row lease resident in the Shard). execute() per request only cudaMemsetAsync-clears the GDN state. KV pages reused in place: prefill overwrites position 0..N-1's K/V token by token; stale data outside the frontier is never read (attention only reads ≤ the current position). forward_tp2 fixed-binds kv_table_rows=0 (text.cpp:1199), matching the row 0 published at startup
- [x] Build GREEN (job pwsh-6, binary build/apps/ninfer-serve; only two pre-existing narrowing warnings at text.cpp:1193, unrelated to this change)
- TODO: hand to the user to re-run --devices 0,2 --max-context 8192 → expect warmup to pass, serve ready → the user sends logs to continue debugging
### Round 27 (TP-2 warmup failure: cudaErrorSymbolNotFound @ scalar.cu:10) — diagnosing
- [x] **Symptom**: after the KV fix the engine was ready; warmup FATALed at the first forward_tp2's ops::set_i32_scalar: scalar.cu:10 CUDA_CHECK(cudaGetLastError()) failed: cudaErrorSymbolNotFound: named symbol not found → Aborted (core dumped)
- [x] **Link-layer issue ruled out**: cuobjdump --dump-sass showed the serve and test binaries each have 3191 SASS functions, all containing _ZN6ninfer3ops21set_i32_scalar_kernelEPii; readelf showed both have .nv_fatbin/.nvFatBinSegment/__nv_module_id sections; nm showed the host stub __device_stub__...set_i32_scalar_kernel and ninfer::ops::set_i32_scalar_kernel both T (defined); __sti____cudaRegisterAllv registration symbols 587 each, exactly identical; both binaries timestamped 10:02 (the same build, the test binary not stale). The device code and registration are both present; not missing SASS
- [x] **build_shard first-kernel hypothesis ruled out**: build_shard's pool.materialize/tables.publish only use cudaMemset/cudaMemcpy (paged_kv_cache.cpp:541/831), launching no kernel; publish is just a host_shadow→device cudaMemcpyAsync
- [x] **Key control**: the Round 21 test binary ninfer_qwen3_5_tp2_forward_test did run both cards' forward_tp2_token (card0→card1, card1→card0, 11 probe tokens) and passed → proving that under that binary's structure, kernel launches on device 2 work. The test does not go through TP2GenerationCore (its own build_shard_state, using an ordinary DeviceArena for KV, not uploading the 14GiB weights, not materializing), so Round 26's changes to tp2_generation_core.cpp are not in the test code path
- [x] **The real difference between serve and test**: serve's materialize_model_tp2 (uploading 14GiB of weights to device 2) + build_shard run on device 2 before the first forward; cudaGetLastError is sticky → what truly failed may be an earlier launch on device 2, the error only surfacing at forward_tp2's first CUDA_CHECK. The code has no CUDA_MODULE_LOADING/cudaSetDeviceFlags settings (default LAZY)
- [x] **Decisive experiment (user already ran it, test.log)**: the test binary **passed** — all 11 probes shard-consistent + 32-step autoregressive decode (including host-staging allreduce) shard-consistent → the TP-2 CUDA code itself is correct; a global RDC linking issue ruled out
- [x] **Root cause (WSL2 GPU enumeration drift + hardcoded --devices 0,2)**: test.log line 4 `devices=0,1` — the test's pick_devices() auto-selects the same-name GPU pair; at the moment the test ran, WSL2 enumeration was 0=5060Ti, 1=5060Ti, 2=T10; but the crashing serve used --devices 0,2, and at that moment device 2 was the **T10 (sm_61/7.5)**. The sm_120a fatbin has no image on the T10 → shard_b (device 2) first kernel launch (forward_tp2's 6th set_i32_scalar) reported cudaErrorSymbolNotFound, exactly matching the crash point. WSL2 re-enumerates GPUs on every VM reboot; indices are unstable
- [x] **Fix**: a validate_tp2_devices() guard added at TP2GenerationCore construction (tp2_generation_core.cpp) — both cards must be compute capability 12.0 with the same props.name, otherwise throw invalid_argument and point to 'nvidia-smi -L' for the real indices. Turns WSL2 enumeration drift from a deep cryptic crash into a clear startup-time error
- [x] Build GREEN (job pwsh-8, 285/285; the serve binary at 12:16:32 contains the guard string)
- [x] **User confirmation**: the index problem nailed — after changing to `--devices 0,1` per the hint, the guard passed, the engine was ready (28.6s), warmup hit a new error → into Round 28
### Round 28 (TP-2 warmup failure: output session already has a preview)
- [x] **Symptom**: with --devices 0,1 the engine was ready; 3.8s into warmup: FATAL: output session already has a preview (one of the three preview_ready guards at output_session.cpp:414/542/607)
- [x] **Root cause (TP2GenerationCore decode loop double-preview)**: decode each round first calls `preview_model(step, ...)` — which internally already handles termination (stop token → StopToken, budget exhausted → limit_reason) and sets `preview_ready=true`; then `publish_preview(decision.finished())` calls `preview_terminal(...)` again when finished → inevitably hits the preview_ready guard and throws. The single-GPU engine's correct pattern (engine_core.h:1113-1226) is to directly commit_preview after preview_model; preview_terminal is only for the cancellation path (when preview_model has not yet been called). A short warmup reply hitting a stop token or budget exhaustion triggers it
- [x] **Fix**: the decode loop's `publish_preview(decision.finished())` → `publish_preview(false)` (preview_model already established a preview including the termination state; commit directly). The cancellation path's (lines 299/335) preview_terminal calls are unaffected — on those paths preview_model has not yet been called
- [x] Build GREEN (job pwsh-9)
- [x] **User confirmation**: warmup passed (3.7s), serve startup FATAL: loaded artifact model name must not be empty → into Round 29
### Round 29 (serve startup failure: empty artifact model name)
- [x] **Symptom**: after warmup passed, `server failed during startup | loaded artifact model name must not be empty` (serve_options.cpp:381 resolve_public_model_id); the engine ready log's model name and formats fields were both empty, weights 0 B
- [x] **Root cause (the TP-2 branch never populated the LoadSummary)**: engine.cpp's TP-2 branch (device_b>=0) directly constructs TP2GenerationCore, but Impl::load (LoadSummary) stayed default-constructed — the single-card path fills the LoadSummary via construct_model (model_instance.cpp:207-226); the TP-2 path had no equivalent step → model_name/weight_formats/host_to_device_bytes all empty → resolve_public_model_id threw. The TP-2 core itself holds the shard model (info().name, storage_stats(), weight_data()); the data is all there, just not exposed
- [x] **Fix**: TP2GenerationCore gained load_summary() (taking architecture/model_name/weight_formats/storage_stats from shard_a_.model, same source and structure as the single-card path) + the ctor records load_seconds_; engine.cpp's TP-2 branch does `load = tp2_core->load_summary()`. Incidentally fixed the engine ready log's empty fields and weights 0 B
- [x] Build GREEN (job pwsh-10, serve binary 12:55:47)
- [x] **User confirmation**: serve started successfully (listening on 8088), but the first real request got 404 (a browser GET / curl with no body, a POST-only route) → after using a correct POST request, into Round 30
### Round 30 (real request failure: std::bad_alloc)
- [x] **Symptom**: correct POST (thinking xhigh, max_tokens 64) → req#1 entered execute and ran prefill through (5.0s, prefill 1 tok), then FATAL: std::bad_alloc (HTTP 500 internal_error)
- [x] **Host memory ruled out**: the bad_alloc came from DeviceArena::alloc_bytes capacity exhaustion (arena.cu:215 `if (end > cap_) throw std::bad_alloc()`), not a host malloc failure; the workspace is a DeviceArena (GPU)
- [x] **Located (the ws_b workspace leaking across forwards)**: execute's prefill/decode loops only created `ws_a.scope()` (tp2_generation_core.cpp:311/351), **never created ws_b.scope()** → shard_b's workspace accumulated every forward's intermediate activations (the attention/GDN/FFN [N,1] tensors + the lm_head GEMM scratch), growing only, accumulating across all forwards, eventually blowing the 128 MiB arena → bad_alloc. Warmup (prompt "hi" ≈ 2 tokens) only leaked ~2 forwards and did not blow up; a real xhigh long-prompt request leaked enough forwards to blow up
- [x] **Why the test did not expose it**: the test binary goes through forward_tp2_token (text.cpp:1269 wraps the whole forward in its own `work_.scope()`, resetting the workspace per token); the serve path calls forward_tp2 directly (no scope) → the ws_b leak. The test only ran position-0 probes + 32-step decode, and went through the token path, so it never triggered
- [x] **Fix**: the prefill and decode loops each gained `auto scope_b = ws_b.scope();` (symmetric with the existing ws_a scope) — at the end of each forward/decode round both shards' workspaces return to the starting point. Scope destruction order: scope_b destructs first (declared later), no dependency issue
- [x] Build GREEN (job pwsh-11, serve binary 14:19:46)
- [x] **User confirmed success**: the same POST request returned complete JSON — prompt 58 / output 64 (finish_reason length, hit max_tokens), reasoning_content normal (63 reasoning tokens, thinking xhigh parsed correctly), TTFT 510 us, total 6.3s; decode steady state 10.0 tok/s (the first window 2.6 tok/s includes CUDA lazy loading). **The TP-2 end-to-end chain (prefill + multi-step decode + host-staging allreduce + sampling + thinking output parsing) passed fully for the first time**
### Round 31 (functional verification: multi-request / streaming / stop termination / performance baseline)
- [x] Verification 1 ✅: req#1–#5, five consecutive requests all succeeded; req#1/req#3 ran the same prompt (57/32) repeatedly with normal output — cross-request state reset (KV overwrite + GDN zeroing) correct, no shard divergence
- [x] Verification 2 ✅: req#4/req#5 streaming — SSE chunks complete (reasoning delta → content delta → final chunk with finish_reason+timings → [DONE])
- [x] Verification 3 ✅: req#5 "1+1等于几" → finish_reason stop (output 43, answered "2" and hit the stop token)
- [ ] Supplementary verification: long prompt (~2000 tokens) — covers long-sequence attention (the SmallT multi-split path, never triggered by short prompts) + sustained prefill (a stress test of the ws_b scope fix)
- [x] **Performance baseline (nsys kernel-level profiling, jobs see tools/tp_bootstrap/analyze5.py)**:
  - The nsys profiled request (`"1+1等于几？只回答数字"`, max_tokens=24) measured **completion_tokens=24** (usage-confirmed, analyze5's /24 denominator correct): span 1212.5ms/24 = **50.5 ms/token ≈ 19.8 tok/s** (GPU idle, no interference beyond CUPTI). CUPTI has ~5-10% overhead, the real value is slightly higher (~21-22)
  - **⚠ The user's real run (after Round 33) back-computed from 3 requests' totals is only ~10.5 tok/s (95 ms/token)**, 1.9× off from the profile's 50.5ms/token. Most likely cause: **llama.cpp still running on the 5060 Ti, preempting the GPU** (the user's own report "llama.cpp在占用"; nvidia-smi showed both cards 15GB occupied with no ninfer compute-app). The profile ran with the GPU idle, the user's test ran with llama.cpp active → not directly comparable. llama.cpp must be stopped and re-measured to isolate the allreduce optimization's real gain
  - **Per-token time breakdown (per shard, the two shards in parallel)**:
    - GEMM/attention/GDN compute **33.8 ms (67%)** — memory-bound, near-optimal (lm_head measured 439 GB/s ≈ 5060 Ti peak bandwidth, GEMV efficiency ~75-90%)
    - **Allreduce barrier 13.9 ms (28%)** — the WSL2 no-P2P host-staging per-layer round trip (2 compute-stream syncs + 4 D2H/H2D + CPU add). Standalone benchmark 90us/layer, the real pipeline 217us/layer (the extra ~127us is the compute-stream sync waiting for the FFN kernel to drain + the H2D blocking the residual)
    - Kernel launch overhead **2.8 ms (5%)** — 16918 small gaps (mean 4us)
  - **Conclusion**: GEMM is near the bandwidth floor; the optimization space is the allreduce barrier (a WSL2-specific tax). Under bare-metal P2P the allreduce would drop from 13.9ms to ~1.5ms → ~35 tok/s reachable. **The realistic ceiling of WSL2 TP-2 5060 Ti ≈ 22 tok/s; the 28 tok/s target is probably a bare-metal/P2P target, hard to achieve under WSL2**
  - **Optional optimizations (ROI order)**: ① reduce the allreduce sync count (issue D2H/H2D directly on the compute stream, 6 syncs→2 syncs, saving ~3-5ms/token → +2-3 tok/s); ② CUDA graph to eliminate launch overhead (but the host allreduce cannot enter the graph, limited gain ~2.8ms); ③ accept the status quo (~20 tok/s is already near the WSL2 ceiling)
- [x] User decision: **option 3** (do the allreduce optimization first, then enter the next stage)

## Round 33: allreduce sync optimization (6 syncs → 2 syncs/layer)
- **Motivation**: the host-staging allreduce does 6 `cudaStreamSynchronize` per layer (run_layers_tp2's 2 compute-stream pre-syncs + 2 after the D2H inside the allreduce + 2 after the H2D). Under WSL2 each sync is a round trip through the GPU virtualization layer; it is the bulk of the 13.9 ms/token barrier.
- **Changes** (3 files):
  - `src/core/tp/device_pair.h`: the `allreduce` signature gains `cudaStream_t stream_a, stream_b` (the compute streams)
  - `src/core/tp/device_pair.cu`: the host-staging path issues the D2H/H2D copies on the **compute streams** (no longer using the DevicePair's independent streams). The D2H naturally orders after the FFN kernel (saving the 2 pre-syncs), the H2D naturally orders before the residual_add (saving the 2 H2D syncs) → only 2 D2H syncs remain per layer. The P2P path (unused under WSL2) keeps the pre-syncs + independent streams
  - `src/models/qwen3_5/execution/text.h`: the call sites pass `ctx_.stream, peer.ctx_.stream`, the 2 pre-syncs removed
  - `tests/test_tp_device_pair.cpp`: passes per-device non-blocking streams, adds a `cudaStreamSynchronize` before the readback (mirroring the production path's implicit ordering where residual_add is on the same stream)
- **Correctness argument**: the compute streams are `cudaStreamNonBlocking` (non-blocking), with no implicit ordering against the default stream. In the production path the H2D and the subsequent `residual_add` are on the same compute stream, naturally ordered, no H2D sync needed. The test readback goes through the default stream (`DeviceBuffer::copy_to_host` uses synchronous `cudaMemcpy`), so the test must explicitly sync the compute streams
- [x] Build GREEN (job pwsh-13, 28/28, only pre-existing narrowing warnings); note the WSL-side `$HOME/ninfer` is an independent copy; changed files must be cp-synced first (tools/tp_bootstrap/sync_changed.sh)
- [x] ninfer_tp_device_pair_test **PASS** (p2p_available=0, the host-staging path, 4 size groups including 1MiB)
- [x] User took over testing: llama.cpp occupied the GPUs (nvidia-smi showed each 5060 Ti at 15GB; probe_mem confirmed some were bookkeeping residue), the nsys re-measure was abandoned, the user manually starts/stops serve for live tests
- [ ] User functional regression: real requests with no bad_alloc / correct output / measured tok/s (expect ~22, up from ~20; send the logs back for checking)
- [ ] Enter the next stage: MTP/dflash2 component split, multi-request concurrency smoke, perplexity soundness

## Round 34: in-kernel allreduce + mixer head split (catching/exceeding llama.cpp 31.65 tok/s)
- **llama.cpp mechanism (source read: a local clone of that fork, ar3-opt branch, allreduce.cu)**: small allreduce goes through the "chunked kernel path" — a single kernel runs simultaneously on both GPUs; `cudaHostAllocMapped`+`cudaHostGetDevicePointer` lets the device read/write the pinned host memory directly; **cross-GPU synchronization busy-waits a host arrival flag inside the kernel** (volatile store + `__threadfence_system()`), entirely bypassing the host-side `cudaStreamSynchronize`. This is why it gets ~30µs/AR while we need 2 host syncs (~217µs)
- **WSL2 key precondition measured and verified (tools/tp_bootstrap/mapped_probe.cu)**: `cudaHostAllocMapped`+`cudaHostGetDevicePointer` available; device-b can read the same mapped host memory written by device-a; **in-kernel allreduce 10KB = 0.038 ms/call (38µs), 64 layers = 2.44 ms/token**, vs host-staging 13.9 ms/token
- **Recomputed expectations**: current 50.5ms/token (21 tok/s) → in-kernel AR only 39.0ms (~25.6) → +head split 32.1ms (~31) → +CUDA graph 29.3ms (~34, exceeding llama.cpp)
- **In-kernel AR is a precondition for head splitting**: taking a single AR 217µs→38µs lets head splitting double the AR count (64→128) without losing (4.9ms < the 9.4ms compute saved)

### Step 1: in-kernel allreduce (do and measure first, 21→~25.6 tok/s)
- Modify `src/core/tp/device_pair.{h,cu}`: port llama.cpp's chunked kernel path. `cudaHostAllocMapped` allocates per-device staging + an arrival ring (64B spacing against false sharing); the kernel has three phases: ① local→host_mine ② thread0 writes the arrival token, spins on the peer (`__nanosleep(100)`) ③ reads host_other, `__hadd` writes back the delta (in-place). The two devices' compute streams each launch the kernel once, **no host sync**
- Keep host-staging as the fallback (when mapped is unavailable)
- [x] Implemented the in-kernel AR kernel + DevicePair integration (`ar_inplace_bf16` kernel: Phase1 local→mapped host, Phase2 thread0 writes the arrival token + spins on the peer `__nanosleep(100)`, Phase3 reads the peer + `__hadd` in-place; mapped pinned 1MiB staging + 64B arrival ring; allocated in the ctor, freed in the dtor, moved on move; falls back to host-staging when mapped is unavailable)
- [x] Compile GREEN (142/142) + `ninfer_tp_device_pair_test` PASS (20480/16/1MiB sizes + move semantics, in-kernel path numerically correct)
- [ ] User end-to-end speed test (stop llama.cpp → serve → expect ~25.6 tok/s; confirm 64 consecutive in-kernel ARs per forward don't deadlock, logits correct)

### Step 2: mixer head split (~31 tok/s)
- **Qwen3.8-27B geometry (GGUF metadata already parsed by tools/tp_bootstrap/gguf_meta.py)**: attention 24q/4kv/head_dim256/rotary_dim64; GDN 16k heads/48v heads/head_dim128/inner6144/conv4; 64 layers = 16 full-attn + 48 GDN
- **Per card after splitting**: attention 12q/2kv (qkv 14336→7168 rows, o_proj K 6144→3072); GDN 8k/24v (wqkv+gate 16384→8192 rows, ssm_out K 6144→3072, beta/alpha 48→24); KV cache 2 kv heads per card; GDN state 24 v heads per card
- **llama.cpp split reference (qwen35.cpp)**: attention Q (query+gate fused) split by n_head heads, K/V split by n_head_kv heads, wo row-parallel+AR; GDN wqkv split by head blocks, wqkv_gate split by v heads, ssm_conv1d split by channels, ssm_beta/alpha/dt/a split by v heads, ssm_out row-parallel+AR, recurrent state split by v heads
- **Change surface**: tp_split_spec.cpp (mixer projections to GatherRows/RowParallel) + attn/gdn op geometry parameterization (currently hardcoded full-model geometry) + KV cache 2 kv heads per card + GDN state 24 v heads per card + o_proj/GDN-out allreduce
- [x] Implemented the head split (weights + attention + GDN + KV cache + allreduce)
  - **Load layer** (tp_split_spec.cpp / tp_shard_views.cpp / load/text.cpp): attention fused weights GatherRows {q,k,gate,v}, GDN fused GatherRows {q,k,v,z}, conv GatherCols {q,k,v}, o_proj/ssm_out RowParallel; GDN gating (a/b + a_log/dt_bias) kept replicated (all 48 heads)
  - **Execution layer** (execution/text.{h,cpp}): TextContext gains per-shard TextConfig (head count halved) + set_shard_config/shard_config(); attn_mix/gdn_mix allocate workspace + tensor views with the per-shard config; **fused projections decomposed into the generic ops::linear** (per-shard fused weights [q|k|gate|v] / [q|k|v|z], slice+view after the GEMM, mirroring ffn_delta, bypassing the fused GEMM kernel hardcoded to full-model row counts); gdn_mix allocates g/beta for all 48 heads (gating replicated) then slices to this card's 24 heads feeding the delta net; mixer output projections write the delta (not adding the residual directly)
  - **Driver layer** (tp2_generation_core.{h,cpp}): build_shard constructs the per-shard TextConfig, KV cache (2 kv heads) + GDN state (24 v heads) allocated with per-shard geometry, context->set_shard_config; run_layers_tp2 adds one delta allreduce per layer after the mixer (64 mixer + 64 FFN = 128 AR/token, all ≤10KB going through the in-kernel path)
- [x] Compile GREEN (BUILD_EXIT=0; ninfer-serve rebuilt, including the bind_gdn fix)
  - Load-layer gap-fill ①: src/ops/weight_input.cpp's input_projection dense geometry validation was originally exact values (attn q=={6144,5120}/k=={1024,5120}, GDN q=={2048,5120}/v=={6144,5120}); after head splitting, per-shard shapes (attn q=3072/k=512, GDN q=1024/v=3072) would trip the require and throw; changed to structural validation (attn q=gate, k=v, q=6k; GDN q=k, v=z, v=3q; the four blocks share the hidden dim); both full-model and per-shard shapes pass. The per-shard fused object's rows are contiguous ([q|k|gate|v]/[q|k|v|z] order), so p.projection is a single LinearParameters (not PairedProjectionWeights); the execution layer's std::get<LinearParameters> is correct
  - Load-layer gap-fill ② (the bind_gdn head-count bug): bind_gdn originally bound a/b + a_log/dt_bias to 2*heads (mistakenly assuming the loader passed a per-shard config), but the loader passes the **full-model** config (48 v heads) → bound 96 heads, mismatching the physical a/b object (48,5120), tripping "logical shape differs from Binding coverage". Changed to bind heads (=48, the full-model value); a/b replicated for all 48 heads is correct. The TP-2 load test's shape check passed with this fix (no more binding error)
  - **The TP-2 unit tests (test_tp2_load / test_tp2_forward) updated to head-split assertions**: the load test now checks per-shard mixer shapes (attn q [3072,5120], GDN q [1024,5120]); the forward test allocates KV/GDN state per the per-shard config (heads halved) + calls set_shard_config. Both tests must run with **llama.cpp stopped** (each 2×5060 Ti needs ~14GB of weights; with llama.cpp occupying them cudaMalloc OOMs) — the user stops llama.cpp then runs them, or the user serves directly for a live test
  - Execution-layer gap-fill ③ (gdn_mix three-branch refactor): TP-2 decode goes through Phase::Verify (batch=1, width=1); the original code entered the batched snapshot branch (gdn_projection_snapshot, hardcoded full-model 16384 rows) → per-shard weights tripped "unsupported single-parent profile". Refactored into three branches: (a) batched_verify (Verify with batch>1 or width>1, for speculative decoding) keeps gdn_projection_snapshot/record; (b) T>1 (single-card prefill) keeps the gdn_projection fused GEMM; (c) T=1 (single-token decode, including TP-2) goes through the generic ops::linear on the per-shard fused weights (8192 rows) GEMM, output [q|k|v|z] written to the fused buffer, qkv = the first 6144-row slice as the conv input, z = the last 2048-row slice as the output gate, qc/kc/vc use independent workspace buffers as the conv outputs (avoiding the conv input and output overlapping and being rejected by require_split_nonoverlap). attn_mix has no overlap issue (the fused slices are only read downstream; attention writes independent a/KV buffers). State access consistent (conv_slot(gidx,0)==layer_view(gidx).conv, slot_count=1)
  - Execution-layer gap-fill ④ (conv split row profile registry): causal_conv1d_silu_split's resolve_split_geometry only registered two full-model row profiles (8192/2048/2048/4096 and 10240/2048/2048/6144); the per-shard GDN profile (x=5120, out0=1024, out1=1024, out2=3072) unregistered → tripped "split received an unregistered row profile". Added the CausalConvSplitGeometry::Rows1024x1024x3072 enum value + wrapper registration + the smallt/prefill two launchers' switch cases (template parameters <1024,1024,3072>). The kernel itself is shape-generic (the CausalConvSplitOutput3 template); only the profile needs registering. Files: src/ops/launcher/causal_conv1d.{h,cu}, src/ops/wrapper/causal_conv1d_silu.cpp
  - **Split-layer gap-fill ⑤ (names_by_object used a bool as an object index — this round's key bug)**: when tp_split_spec.cpp built the "physical object index → parameter name" reverse-lookup table it wrote `names_by_object[binding.whole_object].insert(name)`, but `Binding::whole_object` is a **bool** (src/artifact/schema.h:61), implicitly converted to size_t so every whole-object binding landed on key=1. Then `names_by_object.find(idx)` found no name, `has_name("attention/output")` / `has_name("gdn/output")` all returned false, and the two mixer output projections (which should be RowParallel, K dim halved) fell into the final else's **Replicated**, still holding full-width weights per shard (n=5120, k=6144). At runtime `ops::linear(x=[3072,1], w=[5120,6144], out=[5120,1])` threw "linear: expected [K,T] x [N,K] -> [N,T]" because x.ne[0]!=w.k (the diagnostic print confirmed: x.ne=[3072,1] w.n=5120 w.k=6144 out.ne=[5120,1]). Fix: a binding always carries parts (a whole-object binding also stores a part covering the whole object), so the table is uniformly built on `p.object.index`, the whole_object branch deleted. Also explicitly set the untied output_head to Replicated (the TP-2 forward contract: each shard independently computes full-vocab logits, the two bitwise-identical, the head not all-reduced; this model's head shares weights with the token_embedding and is intercepted by the token_embedding branch first). After the fix the only real behavior change is attention/output and gdn/output → RowParallel. Diagnostic tool: a temporary fprintf printing the x/w/out shapes in the ops::linear validate_linear_semantics failure branch (located, to be removed)
  - **Op-support gap-fill ⑥ (attention per-shard head geometry 12/2)**: after TP-2 head splitting, the single-card attention is 12 q heads / 2 kv heads, but causal_softmax_attention's require_causal_geometry only accepts (24,4) and (16,2), and the kernel is instantiated per Geometry template → layer 3 (the first full-attn layer) reported "causal_softmax_attention: unsupported head geometry". Fix: (a) geometry.cuh gained CausalD256H12Kv2 = CausalAttentionGeometry<12, 2, 2> (GroupSize=12/2=6, same as 24/4, so the q→kv contiguous mapping is unchanged; SmallTSplitScale takes 2, consistent with the existing narrow geometry 16/2, and the static_assert 85*scale<=256 is satisfied); (b) require_causal_geometry accepts (12,2); (c) causal_attention_chunk_tokens changed "q_heads==16 → 6" to "q_heads!=24 → 6", consistent with causal_attention_split_capacity's tokens ceiling check (q_heads==24?8:6); (d) the split capacity gained a 12 branch; (e) 8 launcher files (small_t / small_t_fp8 / small_t_k8v4 / small_t_nvfp4, 2 spots each, prompt / prompt_fp8 / prompt_k8v4 / prompt_nvfp4_non_rdc) inserted a 12-head branch after the H24 branch and before the H16 fallback, 13 spots total. The kernel itself is generic in QHeads (QHeads only serves as a grid dim and the QHeads==24 constexpr specialization criterion; 12 goes through the non-24 path), so it is mechanical instantiation
- [x] **Both TP-2 tests passed** (2026-09-18): ninfer_qwen3_5_tp2_load_test → "TP-2 dual-shard load passed devices=0,1 layers=64 shard_gate=[8704,5120] shard_down=[5120,8704] head=[248320,5120] replicated"; ninfer_qwen3_5_tp2_forward_test → "all 11 probes shard-consistent; 32-step autoregressive decode shard-consistent". ninfer-serve rebuilt (21:57). Load layer/execution layer/op layer all aligned to per-shard geometry. Incidental: the load test's original assertion output_head=[124160,5120] (the old ColumnParallel design) changed to [248320,5120] replicated, consistent with the forward's full-vocab logits contract
- [x] **End-to-end live test passed (2026-09-18, llama.cpp stopped, GPUs idle, agent self-test)**:
  - The two TP-2 tests still PASS (load / forward 11 probes + 32-step autoregressive)
  - serve (`--devices 0,1`, port 8088) engine ready 35.2s, warmup 3.1s, 11.5GB per card, no OOM
  - **decode 29.09–29.11 tok/s (34.37 ms/token)**: short request 32 tok, medium request 128 tok, 512 tok long generation, three runs consistent (29.09/29.10/29.09) → **92%** of the llama.cpp baseline 31.65
  - Long prompt (2063 tok) decode **31.53 tok/s** (31.7 ms/token), but prefill token-by-token **69.6s** (TTFT ~70s, 13.6× off from llama.cpp's 403 tok/s) → Round 35's real motivation
  - Generation quality: `1+1=2` reasoning correct, `Paris` hit the stop token (finish_reason=stop), haiku/summary coherent; both cards' logits consistent + output semantically correct
  - **Gap-fill ⑦ (all timings 0)**: TP2GenerationCore never filled `timings.prompt_wall_seconds/generation_wall_seconds/prefill_seconds/decode_seconds/first_token_seconds`, and `computed_prefill_tokens_` only incremented on the last prefill token (recording 1 per request). Filled in per the single-card engine_core.h:826-876 semantics (the first_token boundary = prefill loop end), and the prefill token count changed to `+= prompt_tokens`. After the fix the serve response timings are usable, the log prefill/decode rate lines restored
  - **Gap-fill ⑧ (the device-index hint misled)**: `--devices` takes **CUDA runtime indices** (default CUDA_DEVICE_ORDER=fastest-first), **different** from `nvidia-smi -L`'s PCI order — this machine's CUDA 0,1 are the two 5060 Ti, while nvidia-smi's 1 is the T10 (at 2). The guard's original error hint "run nvidia-smi -L" actually steered the agent onto `--devices 0,2` (=T10). Now changed to directly list the CUDA-visible devices (index=name(sm_xx)), no more referencing nvidia-smi
- [x] **nsys kernel-level breakdown (22:10 profile, llama.cpp stopped)**: wall 33.90 ms/token; dev0 GPU busy 30.99 + idle 2.91; 1009 kernels per forward, 1011 gaps (mean 2.9 µs)
  - **Compute 25.6 ms**: per-card weight traffic ≈10.15 GB/token (gate_up nvfp4 17408×5120×56, down 5120×8704×56, fp8 8 layers each, attn 7168×5120×16 + 5120×3072×16, GDN 8192×5120×48 + 5120×3072×48, head 248320×5120) → 22.7 ms@448GB/s theoretical, measured 89% of peak, near the bandwidth floor
  - **AR 5.38 ms (128 × 42 µs = 15.9%)**: consistent with the bench's 38 µs; the bloat's main causes are the two cards' unequal speeds + the per-layer full barrier
  - **lm_head 2.94 ms/call (8.5%)**: fp8 248320×5120 = 1.27 GB, each card computes the full vocab once (the replicated contract), the single largest optimizable item
  - **Launch gaps 2.91 ms (8.6%)**: 1009 kernels/forward; CUDA graph is the only effective means
- **Next step (this stage's closeout → catch/exceed 31.65)**: ① split lm_head by vocab (124160 rows per card → head 2.94→1.47 ms, then exchange half the logits to shard A for sampling via a pinned buffer, the exchange ~0.06 ms) ≈ −1.4 ms; ② connect the TP-2 decode forward to a CUDA graph (first the AR's host-side token count must become a kernel-incremented arrival flag, otherwise graph replay waits on the same token value and deadlocks) ≈ −2.5 ms. Total ≈30.0 ms → ~33 tok/s, exceeding the baseline by 4%

### Step 3 (later): CUDA graph (~34 tok/s, exceeding llama.cpp)
- TP2GenerationCore currently bypasses the CUDA graph (2.8ms/token launch overhead); after connecting the graph → ~34 tok/s

## Round 36: MTP (decisive exceedance)
- The model carries a nextn head (blk.64.nextn.* in the gguf), unused by llama.cpp
- One forward emits 2-3 tokens → 60+ tok/s

## Round 34 (old version, superseded by the one above): mixer head split (catching llama.cpp 31.65 tok/s)
- **Background (llama.cpp data provided by the user)**: llama.cpp ran **Q6_K (~22 GB, 15% larger than NVFP4's 19.1 GB)**, `-sm tensor -ts 1,1` head split + CUDA graph (`graphs reused=2726`) + MTP off, steady state **31.65 tok/s**; prefill **403 tok/s (2.48 ms/token)**. Our decode 21 tok/s, prefill ~47 ms/token (token-by-token loop, 19× off)
- **Gap located (nsys reconciliation, all matching)**:
  - llama.cpp reads ~11 GB/token per card → compute ~25 ms + allreduce 64×0.05≈3 ms + graph 0 = 31.6 ms ✓
  - We read 13.5 GB/token per card (mixer 7.45 GB read by each of the two cards) → compute 33.8 ms + allreduce 13.9 ms + launch 2.8 ms = 50.5 ms ✓
  - The gap 100% structural: ① mixer replicated (+9.5 ms) ② allreduce sync expensive (+11 ms, half fixed in Round 33) ③ no CUDA graph (+2.8 ms). Our kernel efficiency is higher (396 GB/s = 88% of peak)
- **Model geometry (nsys kernel-shape back-inference, Qwen3.8-27B)**: 64 layers = 16 full-attention + 48 GDN. attention: 24 q heads, 16 k/v heads, head_dim 256 (qkv merged 14336×5120); GDN: in_proj 16384×5120 (q 24×256 + k/v 16×256 + gate 2048). FFN 8704 intermediate dim (55 layers nvfp4 + 8 layers fp8). lm_head 248320×5120 replicated
- **Change surface**:
  1. Weight loading: q/k/v/gate projections split by head (column shard), o_proj/GDN out split by row (row shard); KV cache 8 heads per card
  2. attention kernel: 12 q heads × 8 kv heads per card (GQA 1.5:1)
  3. GDN kernel: 12 q heads × 8 k/v heads per card, recurrent state split by head
  4. allreduce: o_proj output + GDN out output (once per layer, 64/token total, merged with or independent of the FFN delta)
- **Expectation**: per-card compute 33.8 → ~24.7 ms (13.5→9.8 GB), allreduce 13.9 → ~6 ms (after Round 33), wall ≈ 24.7+6+2.8 ≈ 33.5 ms → **~30 tok/s**; if the allreduce compresses to llama.cpp level (0.05 ms) → ~32 tok/s
- [ ] Read the attention/GDN/weight-loading/KV-cache code, design the head split
- [ ] Implement the head split (weight loading + attention + GDN + KV cache + allreduce)
- [ ] Compile + unit tests + user functional regression
- [ ] nsys re-measure decode (target ~30-32 tok/s)

## Round 34 patch: concurrency crash/hang (exposed by the user's real traffic)

- **Symptom (the user's 64k live logs)**: req#2/#3 arrived simultaneously → both HTTP 500 internal error within 6 ms; afterwards req#4/#5's
  `cudaMemsetAsync(shard.state_backing)` reported `cudaErrorIllegalAddress` (the context was already polluted).
  Agent reproduction: **all 3 concurrent requests hung >10 minutes** (three "req started" in the same millisecond, then no progress at all)
- **Root cause (no serialization in the TP-2 core)**: `Engine::submit` goes `core->submit()` for TP-2 (only creates a Submission),
  `GenerationHandle::wait` → `TP2GenerationCore::Submission::wait` → `execute()` **runs directly on the caller (HTTP) thread**.
  The single-card path relies on the EngineCore's worker thread + pending queue for serialization; TP-2 had no equivalent, while serve's own
  `request_capacity` used the **serve options** (≠ the engine-normalized max_concurrency=1/pending=1), so concurrent requests went
  straight into `execute()`. Consequence: two threads share the same workspace **arena** (not thread-safe) → the bump pointers trample each other → wild pointers/illegal access;
  and `DevicePair::allreduce`'s host-side token increment got scrambled → the two cards' arrival flags mismatched → the in-kernel AR **spins forever** (this is the hang)
- **Fix**: `TP2GenerationCore` gained a `std::mutex execution_mutex_`; `Submission::wait` takes the lock before execute (arrival order),
  consistent with the single-card EngineCore's single-worker contract; a request cancelled while queued still enters execute, its first cancellation check is cheap
- **Verification**: 3 concurrent requests completed serially in 2.9s / 5.6s / 8.3s, all HTTP 200, no hang, no illegal access

## Round 34 patch: prompt prefix reuse (the only viable means for long-context multi-turn)

- **Motivation**: TP-2 had no prefix cache; every round recomputed the **entire history** (34 ms/token). At 64k, a 10k history = 5.7 minutes per round, practically unusable
- **Key fact**: KV pages are materialized once at startup, reused in place, never cleared across requests → the K/V at position < the reuse boundary are naturally valid,
  only a snapshot of the GDN/conv state at that boundary is needed
- **Template measurement (temporary `[reuse-diag]` prints of lcp/cached/new, since removed)**:
  - Repeated requests with the same message list: lcp == full length (rendering is deterministic, no per-request variables)
  - Two consecutive dialogue turns: the lcp lands **2–27 tokens before the end** of the previous round's prompt (the previous round's end was the generation-side
    `<|im_start|>assistant\n<think>\n`, while the next round renders that assistant turn as history → a few tail tokens mismatch)
  - Conclusion: **the stably reusable boundary sits very close to the previous round's prompt end**; no need to guess by message structure
- **Design**: `kReuseSnapshotCount = 3` state snapshots per card (live + slot0=prefill end + slot1/2=recursive rewind depths),
  `DeviceArena((1+3)*state_bytes)` ≈ 308 MiB/card (state ≈77 MiB). The rewind depth is **adaptive**:
  it predicts `rewind_near = gap+2`, `rewind_far = 4*near+8` from the previous prompt pair's gap (clamped 4..4096).
  The next request takes the deepest boundary that is "≤ LCP and < the new prompt length"; snapshots are taken in stream order inside the prefill loop (naturally exact to the boundary)
- **Verification (--greedy, guaranteeing token-by-token comparability)**: req#2 cache 48/78 (61.5%), req#3 cache 74/94 (78.7%) TTFT 683 ms;
  the full prefill rebuilt after invalidation and the reuse run had **exactly matching output (MATCH)**; unrelated requests cache 0
- **Leftover**: the gap's explanation (why the gap reaches 18–27 when the reasoning echoes) uninvestigated; currently held by the adaptive depth

## Operations lessons (must obey)

- **Starting ninfer-serve must first kill the old instance and confirm its exit**: each serve process stages 22.6 GB of artifact on the host side;
  two instances at once → host memory blows out (this nearly froze the user's machine), and both cards' VRAM double-occupied
- `tools/tp_bootstrap/serve_run.sh` hardened: pkill -9 → poll-wait 30 s → if still alive, **refuse to start** and print the process table →
  print `free -h` → exec. After startup, triple-confirmation via `pgrep -a -f 'build/apps/ninfer'` + `nvidia-smi` + health that only one instance exists
- WSL `/tmp` gets cleared (lost an nsys report once) → profiling artifacts go to `$HOME/prof/`
- **A WSL distribution powers itself off when the last session ends**: a background serve started via `setsid nohup` cannot be kept alive,
  because WSL, deciding "the last client disconnected", runs `systemctl poweroff` (visible in the journal as
  `Operation canceled @p9io.cpp:258 (AcceptAsync)` → `The system will power off now!` →
  `WSL ... systemctl poweroff did not terminate the instance in 10000 ms, calling reboot(RB_POWER_OFF)`).
  The serve of 2026-09-18 23:53 vanished this way (not a crash, not an OOM; no OOM record in dmesg)
  → **a wsl session must keep existing**: run `serve_run.sh` in the foreground of a background job (the job never ends, the session never breaks);
  do not run a script that would `pkill ninfer-serve` while someone is testing
- **2026-09-19 01:03 second incident (same root cause, hardened)**: `run_r35_tests.sh` carried its own `pkill -9 -f ninfer-serve`,
  while 8088 was then under the supervisor's management → the supervisor immediately restarted the engine (`starting engine`),
  while the test process was loading a second model copy → the user saw "two services", double VRAM/memory occupation.
  Hardening: **test scripts never kill the service themselves**; changed to refuse to run and prompt when `ninfer-serve` is detected;
  stopping the service has a single explicit entry `tools/tp_bootstrap/serve_stop.sh` (**kill the supervisor before the engine**, or it will definitely restart).
  Correct order: `serve_stop.sh` → exclusive testing → `serve_supervise.sh <ctx> <port>`

## Round 35: TP-2 batched prefill (TTFT 19× improvement)

- **Status quo**: TP-2 prefill loops token by token (2063 tokens = 69.6 s, 34 ms/token); llama.cpp batches at 403 tok/s.
  This is the current largest usability bottleneck (the first round's long prompt has no prefix to reuse)
- **Why it could not be done before** (Round 34's three structural blockers):
  1. `forward_tp2` hardcoded single-token ([N,1] tensors, envelope {pos+1,pos+1}, position scalar)
  2. `gdn_mix`'s T>1 branch used the fused `gdn_input_proj` with full-model row counts (16384 rows); per-shard weights of 8192 rows unsupported
  3. `run_layers_tp2`'s mixer/FFN delta allocations live in a scope **outside the layer loop**; at T>1 each layer accumulates 2×hidden×T×2 bytes
     (T=1024, 64 layers = 1.34 GB, far beyond the 128 MiB workspace)
- **Conditions already in place** (earlier gap-fill results, directly reused this round): all per-shard NVFP4 shapes registered and T-generic
  (n7168k5120 attn fused, n8192k5120 GDN fused, n17408k5120 gate+up, n5120k3072 o_proj, n5120k8704 down;
  A16 chunks ≤32 tokens, A4 MMA ≥8 tokens to any length); attention 12/2 geometry in all
  prompt/small_t/fp8/k8v4/nvfp4 launchers; conv split `Rows1024x1024x3072` registered for the smallt+prefill two paths
- **Changes**:
  1. `TextContext::forward_tp2_prefill(peer, pair, ids, first_position, logits, logits_peer)`:
     T tokens' embedding → lockstep layer loop (`Phase::Prefill`) → final norm → lm_head on the **last column only**.
     Binding consistent with the single-card `prefill_impl`: positions=[T] absolute positions, envelope={first+T, first+T}, kv_table_rows=[1]=0,
     `active_sequence_batch_=0` (single-sequence prefill path), state slots=[1]=0
  2. `gdn_mix`: the fused `gdn_input_proj` branch condition tightened to `T > 1 && shard_config_ == nullptr`;
     TP-2 shards' T>1 goes through the same decomposed path as T=1 (generic `ops::linear` + `causal_conv1d_silu_split` + `gated_delta_net`)
  3. `run_layers_tp2`: a per-layer workspace scope added inside the layer loop body
  4. `TP2GenerationCore::execute`: prefill chunked by `kTP2PrefillChunk = 256`;
     reuse snapshots can only be taken at chunk boundaries, so the **chunk length is truncated** to the target depth (`prompt_tokens - rewind_near/far`) so the target lands exactly on a chunk boundary
  5. `DevicePair`: the in-kernel allreduce's mapped pinned staging raised from 1 MiB to 8 MiB (a T=256 delta = 2.6 MB
     exceeds the old ceiling and would fall to the host-staging full-sync path), and vectorized (uint4 + 4-way unrolled reads) + thread count 256→1024
- **Three classes of blockers exposed during implementation (all fixed this round)**:
  1. The fused projection's dim-0 slice window is non-contiguous: `Tensor.ne[0]` is the **innermost contiguous axis**; at T>1 `fused.slice(0,off,n).view(...)`
     directly throws `view requires a contiguous tensor`; likewise the GDN's `g[i*24:(i+1)*24]`/`beta` slices cannot enter
     `gated_delta_net`, and the fused's [q|k|v] slices cannot enter the split conv (both require contiguity).
     Added `copy_row_block()` (`cudaMemcpy2DAsync`, dpitch=the row-block width, spitch=the parent row width) to move the block into an already-allocated contiguous
     workspace buffer; T=1 still goes through the zero-copy alias path
  2. The FP8 A8 / NVFP4 W4A4 activation quantizers only instantiate for K∈{5120,6144,17408}; per-shard K=3072 (o_proj/GDN out) and
     K=8704 (FFN down) threw `unsupported K` (T=1 goes through gemv and does not trigger A8/A4, so it was never exposed before); filled in
     `Fp8Activation{3072,8704}` and `Nvfp4Activation{3072,8704}`
  3. `const auto projection = ...` makes `projection.query` etc. const views, unusable as memcpy targets
- **Verification**: `test_tp2_forward`'s oracle extended to three groups (same weights, same KV rows, same 300-token sequence):
  - **Route equivalence** (T=7, A16, below A4's 8-token threshold): chunked `Phase::Prefill` vs token-by-token `Phase::Verify`,
    same argmax, top5 diff 0.56 — the inherent drift of the bf16 residual stream's reduction reordering across 64 layers (0.4%×√64)
  - **Chunk equivalence** (T=300, A4): the two splits `300→128+172` and `300→64+236` are **bit-identical** to "300 in one go" (max diff 0);
    this is the strongest evidence of KV-page and GDN conv/recurrent state continuation across chunks; `300→256+44` has a top5 diff 0.63 (argmax unchanged) because the 44-column remainder chunk switches to the narrower A4 MMA
    schedule (and its first 3 columns go through the conv's state branch rather than the input tap)
  - **Informational**: A4 batched vs A16 token-by-token, max logit diff 1.99, same argmax; the two shards' logits bit-identical
- **Measured results (64k serve, one round each with `--greedy` and ordinary sampling)**:
  - Fresh 2063-token prompt: **TTFT 1.5 s = 1.38k tok/s** (before the change 69.6 s / ~30 tok/s, **~46×**),
    already exceeding llama.cpp's 403 tok/s (3.4×)
  - Re-asking the same prompt: cache 2051/2063 (99.4%), TTFT **42.7 ms**
  - Decode steady **32.1–32.2 tok/s** (previously 29.1) — the AR staging 8 MiB + uint4 vectorization also takes effect at T=1;
    now **exceeding** the llama.cpp baseline 31.65 tok/s

## Round 35b: prefill deep-dive analysis (adjustable chunk width + allreduce true-cause located)

- **Changes (all kept)**:
  1. TP-2 prefill chunk width changed from hardcoded 256 to **reading the engine option** --prefill-chunk (normalize_engine_options' TP-2
     branch no longer forces the override, changed to clamp into [128,1024] and align to 128); ceiling kPrefillChunkMaximum = 1024.
     serve defaults to 1024, tunable at runtime, no rebuild needed
  2. workspace 128 -> **384 MiB** (the measured 1024-wide-chunk arena peak = **122 MiB**; the old 128 MiB would std::bad_alloc)
  3. DevicePair staging 8 -> **24 MiB**, changed to **double-buffered by call parity**: the peer publishes arrival before reading;
     the original single buffer had a **real cross-call race** ("this card overwrites call k while the peer is still reading call k-1") (previously only not triggered by timing);
     with double buffering, the buffer written at call k has been read by the peer after call k-2, the race entirely eliminated
  4. ar_inplace_bf16 supports **multi-block sharding** (one arrival slot per block, contiguous slices within a block, no coordination needed between blocks);
     the block count is 1..16 by payload (a ≤64 KiB decode payload still uses 1 block for latency)
- **Correctness (test_tp2_forward, all passed)**: KV capacity raised to 2048 tokens; added a **T=1024 one-chunk vs 4x256 chunked**
  case, **bit-identical** to the existing 128/64 chunking (max logit diff 0, top5 gap 0); the two shards' logits bitwise-consistent;
  the 32-step decode sequence unchanged
- **Measured (64k serve, --no-prefix-reuse, 1257-token prompt, chunk sweep)**:

  | chunk | TTFT | prefill |
  |---|---|---|
  | 128 | 1146 ms | 1.15k tok/s |
  | 256 | 1032 ms | 1.33k tok/s |
  | 512 | 990 ms | 1.36k tok/s |
  | 1024 | 976 ms | 1.38k tok/s |
  | 1024 (after multi-block AR) | 954 ms | **1.42k tok/s** |

  I.e.: raising the chunk width from 128 to 1024 only bought **+20~27%**; the expected weight-traffic amortization did not materialize → the bottleneck is not the weight traffic.
- **Root cause (nsys + step-by-step elimination)**: ar_inplace_bf16 takes **48% of the total kernel time** (~60% of a 1024-wide chunk),
  ~3.2 ms per call and perfectly symmetric on the two cards. Item-by-item elimination located the **hardware link asymmetry**:

      index 0  RTX 5060 Ti  pci 01:00.0  Gen5 x8  (~20 GB/s measured)
      index 2  RTX 5060 Ti  pci 08:00.0  Gen4 x4  (~7 GB/s measured)

  **CUDA device 1 (the second 5060 Ti) is plugged into a chipset-assigned Gen4 x4 slot**. TP-2 does two allreduces per layer,
  payload [hidden=5120, T] bf16, write+read round trip = 4xhiddenxT bytes/layer/card; converted per token it is
  **2x64x5120x2Bx2 = 2.6 MB/token**, on 7 GB/s = **~375 us/token, a ~2.67k tok/s ceiling**.
  The measured marginal slope **0.368 ms/token = 2.7k tok/s** — **exactly pinned to this hardware ceiling**, and independent of chunk width.
- **Rejected schemes (all with measured evidence, reverted)**:
  - AR multi-block (1/2/4/8/16): TTFT 859/877/879/879/830/861 ms, **no difference** → not a single SM unable to issue requests
  - Copy-engine data movement + a tiny signal/wait kernel as the rendezvous: **slower** (951 ms vs 857 ms) and the two shards' logits mismatched
    (the rendezvous/double-buffer semantics did not hold), fully reverted; only the kernel path + double buffering + multi-block structure kept
- **Conclusion**: prefill's allreduce is **already pinned to the hardware floor**; continuing to squeeze prefill at the software level has a remaining-gain ceiling of about 1.4x
  (removing the 270 ms/chunk outside the AR), but **the largest single lever is hardware**: move the second card from the Gen4 x4 slot to a CPU-direct
  Gen5/x8 slot, raising the AR ceiling from 2.67k to about 7.6k tok/s.
- **Next step**: per the user's established priorities, prefill ends here; turn to **Round 36 (MTP)** to attack decode (31 -> 60+ tok/s);
The decode AR payload is only 10 KB, unaffected by the Gen4 x4 (measured AR median 9-18 us, improved from 42 us)

## Round 35c: AR ordered-slice pipeline (another 15%)

- **Motivation**: 1024-wide-chunk attribution (dev0, 655 ms window, 98% busy): **AR 413 ms (63.1%)**, fp8_mma 104.7 ms (16.0%),
  nvfp4_w4a4_mma 83.1 ms (12.7%), the rest (silu/state_passing/prepare_wy_wu/residual/attention/conv) ~35 ms.
  AR is the only big item, and it is pure PCIe time. The hypothesis: "PCIe full-duplex → write/read overlap saves another half".
- **Standalone microbenchmark (two 5060 Ti, replicating the engine protocol)**: tools/tp_bootstrap/bench_ar_protocol.cu

  | payload | serial (pre-change protocol) | 4-slice ordered | 8-slice ordered | 16/32 slices |
  |---|---|---|---|---|
  | T=1024 10.5 MB | **3.180 ms** (exactly matching the engine measurement) | 2.724 | **2.704** | 3.84 / 6.24 |
  | T=189 1.94 MB | 0.596 | **0.519** | 0.561 | 0.85 / 1.55 |
  | T=1 10 KB | **0.023** | 0.042 | 0.109 | 0.31 / 0.77 |

- **Key conclusion (refuting the "full-duplex doubling" hypothesis)**: this **Gen4 x4 (through the chipset) link has only ~7.9 GB/s combined bidirectional**,
  not 7.9 GB/s per direction. The physical floor for the 21 MB round trip = 2.66 ms; the **8-slice ordered pipeline reaches 2.704 ms = 98.5%**,
  i.e. AR has no remaining headroom to mine (unless reducing the byte count, which would require changing the parallel layout or lowering precision, both undesirable).
  The slice count must be adaptive: >16 slices get eaten by the per-slice handshake latency (16 slices 3.84 ms, 32 slices 6.24 ms); decode must keep 1 slice.
- **Implementation** (src/core/tp/device_pair.cu, no model-layer changes):
  1. ar_inplace_bf16 gains a **write-order chain**: block b first spins waiting on order[b-1] == token before writing its own slice,
     so slices land in order and the peer can read the front segment early; the arrival-slot protocol is unchanged
  2. ar_slices() policy: payload <=256 KiB uses 1 slice (preserving decode latency), otherwise 512 KiB/slice, capped at 8 slices
  3. The slot array is extended to 2 cache lines (arrival array + order chain), sharing the same cudaHostAlloc
- **Verification and results** (64k serve equivalent config, --no-prefix-reuse, 1257-token fresh prompt, chunk 1024):
  - Correctness: test_tp2_forward all green, **T=1024 one-chunk vs 4x256 still bit-identical** (max logit diff 0, top5 gap 0)
  - **TTFT 954 -> 808 ms (-15.3%)**, prefill **1.42k -> 1.50k tok/s**, decode unchanged at 31.2-31.5 tok/s
- **Prefill remaining-headroom assessment (honest accounting at this round's end)**: a 1024-wide chunk is ~580 ms = AR ~350 ms (60%) + MMA 188 ms (32%)
  + other ~35 ms. AR is already pinned to the link; **the only remaining big lever is "AR/MMA overlap"** (AR uses only PCIe, GEMM only HBM/tensor cores,
  no resource conflict), requiring: moving AR off the compute stream + sub-block pipelining (passing GDN/KV state between sub-blocks per layer) + event sync + ring staging,
  ceiling ~1.5x (prefill -> ~2.1k tok/s), but with cross-card rendezvous deadlock risk, a major refactor
- **The cheaper equivalent lever (hardware)**: move the second card from the Gen4 x4 slot to a CPU-direct
  Gen5/x8 slot, raising the AR ceiling from 2.67k to about 7.6k tok/s
  — the gain exceeds the big refactor above, with zero code risk

## Round 36: MTP (decisive exceedance)
- The model carries a nextn head (blk.64.nextn.* in the gguf), unused by llama.cpp
- One forward emits 2-3 tokens → 60+ tok/s
**Recon conclusion (key): MTP is already fully implemented in this repo; the TP-2 path just turns it off.**
- Single-card path: CLI --spec mtp --draft-tokens K (K=1..5); config.mtp is opened by options.speculative==Mtp,
  loading::bind_mtp binds mtp/* (input_projection [h,2h], embedding_norm, hidden_norm, final_norm,
  the FullAttention block under mtp/layers/0/, token_embedding/output_head shared with the target)
- The op layer is complete: speculative_prepare_verify_inputs/ids, speculative_accept_greedy_drafts (both greedy + sampling paths,
  the workspace capacity function too), mtp_prepare_next_round (round bookkeeping), mtp_pack_fc_input / mtp_split_attn_in (MTP input packing),
  kSamplePurposeSpeculativeAccept/Correction/Bonus
- The execution layer is complete: TextContext already has MTP support, **enabled by passing the MTP KV view + MTP cache at construction**;
  mtp_forward_batch (bridging the first-round proposal) + mtp_forward_ar_step (the following K-1 rounds of autoregressive proposals) + proposal_argmax
- TP-2 status quo: model_instance.cpp:88 in the TP-2 branch **silently drops options.speculative = {}**;
  tp2_generation_core.cpp:118 load.speculative = None; in build_shard enable_mtp=false,
  mtp_physical_page_groups=0, and both TextContext MTP parameters are passed empty_mtp
- **Porting design (key decision)**: **the MTP proposal runs entirely on shard A, with the MTP weights replicated whole to both cards**.
  Rationale: MTP has only 1 layer (weights ~0.2 GiB/card); proposing 1 draft token costs 1/64 of the weight stream (~0.4 ms) + the head (~0.3 ms);
  splitting would save only fractions of a millisecond while introducing peer lockstep + 2 allreduces + shard view rework.
  And the MTP input [enorm(embed(t)) | hnorm(hidden)] is **bitwise-identical** on the two cards (embedding replicated, hidden synced by allreduce;
  the test already verified the two shards' logits are bitwise-consistent), so shard A alone can compute the correct proposal on a single card — **zero allreduce, zero new forward code**.
  Only the verification phase uses both cards (the main model's batched forward).
- **Phased implementation**:
  - **36.1 Loading and state (done and verified)**: normalize_engine_options no longer drops TP-2's --spec mtp (dflash/dflash2 get explicit errors);
     the ctor passes backend/proposal_head to plan_load; tp_split_spec forces mtp/* to Replicated;
     shard 0 turns on enable_mtp and plans and materializes the MTP KV (independent page table + execution row);
     **verification**: test_tp2_load's new assertions (input_projection [5120,10240], attention query [6144,5120] **whole on both cards**)
     pass under --spec mtp, no regression on the no-spec path; the real serve --spec mtp --draft-tokens 3 comes up,
     prompt 55 / decode 31.5 tok/s (the decode loop not yet wired to MTP, as expected).
     Incidental fix: build_r35.sh switched to directory-level rsync (the per-file list missing header sync had bitten twice);
     serve_supervise.sh supports a third parameter to pass through extra serve options
  - **36.2 Proposal (done and verified)**: three wiring points on the single-card side + two "contract traps":
     1. **The MTP input is the post-final-norm hidden** (equal to the lm_head input), not the pre-final-norm residual.
        Evidence: the single-card prefill_chunk passes `xf` (= rmsnorm(x, final_norm)) to mtp_prefill_chunk;
        ordinary_decode_batch's hidden output is likewise post-norm. → forward_tp2/forward_tp2_prefill gained
        an `mtp_input_hidden` out-parameter (optional), returning the post-norm hidden.
     2. **The MTP embedding is shifted right by one**: `plan_mtp_alignment_window` gives `shifted_embedding_begin =
        chunk_begin + 1`, i.e. column i's embedding is token[i+1]; the last column uses the just-sampled token.
        → priming uses the shifted ids; the final chunk's last column is overwritten on-device with the sampled token
        (so the final chunk's priming must move after sampling).
     3. **Column q's output predicts t_{q+2}** (not t_{q+1}): the full-prefix bridge uses
        position = prompt_tokens-1 + next_token = io.token, while speculative_prepare_verify_inputs verifies
        draft0 at frontier+1. → **the acceptance criterion must be "one step lagged"**: step p's draft1 compared against step p+1's
        target argmax. My initial "same-position argmax" criterion was wrong (0/2 was a criterion error, not a wiring error).
     - Implementation: `mtp_propose_window()` (bridge: mtp_forward_batch + K-1 mtp_forward_ar_step calls, word-for-word aligned
       with `mtp_bridge_and_propose`), RoundState (begin/complete_round_state_layout + RoundStateSpec
       {backend=Mtp, draft_window=K, batch_capacity=1}, memset 0 → rope_delta=0,
       backend_kv_table_row=0), TextContext passed the MTP execution view + batch cache +
       set_mtp_proposal_extent(K); `DecoderStateSpec.mtp_kv_heads` (MTP layer replicated → use the **full** kv heads,
       a new field, because the text KV is still split per shard).
     - **Measured**: 64-token greedy generation, draft1 acceptance **52/67 = 77.6%** (normal MTP level) → weights/KV/positions/shift
       all correct. Cost: the proposal is on the critical path and not yet accepted; decode 31.6 → 22.4 tok/s (the payback comes only after the next step's acceptance).
  - **36.3 Verification/commit mechanism (surveyed, to be implemented)** — the single-card design is very clean; TP-2 can copy it:
     1. The verification forward uses `GdnStateAction::RecordForReplay` + `GdnReplayRecords` storage
        (`target_verification.cpp:14`): GDN layers write each column's (q,k,v,{g,beta}) into the record plane, **not touching the live state**
        (`ops::gated_delta_net_replay_record`, text.cpp:1175). Note that `UpdateInPlace` with width != 1 throws directly
        (text.cpp:1031) → multi-column verification must go through RecordForReplay.
     2. `ops::speculative_accept_greedy_drafts(target_tokens, target_logits, drafts, current_extents,
        frontiers, anchors, licensed_tokens, licensed_counts, accepted_drafts, public_token_count,
        sampling, work, stream)` decides the accepted prefix (both the greedy/sampling paths).
     3. `ops::gdn_replay_fold` (include/ninfer/ops/gdn_replay.h, GdnReplayFoldRow{source_state_slot,
        destination_state_slot, commit_columns}) folds the first commit_columns records back into the state slot →
        **commit_columns = accepted + 1**, no snapshot/rollback needed, one forward completes a round.
     4. **Key simplification**: `forward_tp2_prefill(T=K+1, first_position=window start)` is itself the verification forward —
        the envelope/causal mask/position binding all match (every column in the window is valid, no valid_columns filtering needed),
        only two things missing: (a) the **per-column logits** [V,T] are needed (currently only the last column is computed), (b) RecordForReplay is set before and after the call.
        Window = [the just-sampled t_{p+1}, d0, d1, ..., d_{K-1}], positions = [p+1 .. p+K];
        the target's per-column argmax is compared with d_j → accepted prefix; bonus = column j's argmax.
     5. To read (next step): where `GdnReplayRecords`/`GdnReplayFoldPlan` are planned and workspace-allocated
        (program_impl.h:596 replay_records; planning/startup.cpp), and how the accept sampling/
        public_token_count arguments are obtained.
     6. **This phase's acceptance criterion**: the batch-verification accepted count must match 36.2's step-by-step criterion (≈77.6%);
        ultimately "greedy MTP == pure greedy bit-identical" (36.5) is the standard.
  - **36.3/36.4 round semantics (fully derived, implementation checklist below)** — the most important is **state discipline**:
     1. Window = [anchor, d0, ..., d_{K-1}], positions = [frontier .. frontier+K], width T = K+1;
        anchor = the just-sampled token, **its GDN state transition not yet committed** (base state = the one after the token before it).
     2. The verification forward uses RecordForReplay to write only records, not touching state; each column's target argmax is compared with d_j.
     3. `speculative_accept_greedy_drafts`: accepts A drafts, licensed = [d0..d_{A-1}, bonus],
        bonus = column A's argmax (i.e. position frontier+A+1, **it is the next round's anchor and its transition awaits commit**).
     4. `gdn_replay_fold`'s `commit_columns = A+1` (= the licensed count = window columns 0..A, i.e. anchor + accepted drafts),
        consistent with single-card (prefill.cpp:782/792: `committed = accepted_tokens`).
        → the state slots must ping-pong: TP-2 is now **`.slot_count = 1`** (tp2_generation_core.cpp:212),
        which must change to 2 (state_bytes doubles automatically, the arena formula `(1+kReuseSnapshotCount)*state_bytes` unchanged),
        source/destination alternating.
     5. The next round's MTP bridge needs the **hidden at the acceptance boundary** (the final-norm hidden at position frontier+A):
        single-card uses `speculative_select_accepted_hidden(target_hidden, accepted, selected_hidden)` +
        `scatter(selected_hidden, destination_slot, continuation_hidden_store)` (target_verification.cpp:41-44).
        TP-2 version: the verification forward incidentally outputs per-column final-norm hidden [hidden,T], select column A, store it into the
        continuation store indexed by the destination slot (or directly into a [hidden, max_slots] buffer).
     6. Sub-partition checklist (next step to implement):
        (a) `forward_tp2_prefill` gains optional out-parameters: per-column logits [V,T] + per-column hidden [hidden,T] (currently only the last column is projected);
        (b) TP-2 core allocates `GdnReplayRecords` (`plan_gdn_replay_records`, spec: layers=GDN layer count (48),
            record_capacity=B(1), width=T, conv_channels/qk_heads/value_heads/key_dim/value_dim take the **shard** config)
            and `ops::GdnReplayFoldPlan` (the construction paradigm at program_impl.cpp:156);
        (c) the round loop replaces the existing in-place decode loop (key: each token's state transition may happen only once);
        (d) accept's arguments: `token_domain = public_token_count`, `configs = sampling_a` (device-side SamplingConfig),
            the workspace reserved with `speculative_accept_greedy_drafts_workspace_capacity_bytes()`;
        (e) must use `RecordForReplay` (`UpdateInPlace` with width != 1 throws directly, text.cpp:1031).
     7. This round's acceptance: the batch-verification accepted count == 36.2's step-by-step criterion (≈77.6%); 36.5 closes with greedy MTP == pure greedy bit-identical.
  - **36.3 implementation status (code written but blocked on an op whitelist)**:
     - Implemented and compiled: the verification forward's per-column logits/hidden (forward_tp2_prefill new out-parameters),
       GdnReplayRecords + ops::GdnReplayFoldPlan (**one set per card** — verification runs on both cards, so record/fold cannot be shard-0-only),
       mtp_anchor_hidden, priming simplified to "only backfill MTP K/V + grab the last column's hidden",
       the round loop (propose → RecordForReplay verification → accept → output policy → fold → next round's anchor/hidden).
     - **Correction**: at B=1 the fold's use of **in-place** (source=destination=slot 0) is explicitly allowed by the op
       (gdn_replay.h: "A row may be in-place"), so slot_count=2 is **not needed** (original plan item 3 voided).
     - **Blocker**: at startup FATAL "gdn_replay_fold: unsupported record geometry".
       Root cause: fold only registered two **full-size** geometries — src/ops/linear_attention/gated_delta_net/replay.cpp:154-160
       only lets through (48, qk16, val48, conv10240) and (30, qk16, val32, conv8192);
       the record op's own validation at replay.cpp:107 (qk_heads==16 && value_heads in {48,32}) would also block;
       the fold kernel dispatch at recurrent.cu:190-208 only instantiates FoldGeometry48x48 / FoldGeometry30x32.
     - **TP-2 shard geometry**: layers 48, qk_heads 8, value_heads 24, conv_channels 5120
       (half of the full-size 16/48/10240; the template requires kValueHeads % kQkHeads == 0 → 24/8 = 3, the group ratio matches the original geometry,
       and the kernel is generic in head count: recurrent.cuh:447/477/504).
     - **Fix plan**: register a third geometry (48, qk8, val24, conv5120) — the geometry struct + launch_replay_fold dispatch branch +
       the two whitelists (replay.cpp:107, :154); per the op contract, a new instantiation needs an **independent numeric oracle**,
       and end-to-end closes with 36.5's greedy bit-identity.
     - Before the geometry registration **the MTP serve cannot start** (the fold plan construction throws); the service must run without --spec mtp.
  - **36.1 original design points**: normalize_engine_options no longer drops TP-2's --spec mtp; the ctor passes backend to plan_load;
     build_shard turns on enable_mtp for shard A, plans the MTP KV (mtp_physical_page_groups, page allocation + table row publish, isomorphic to the text KV),
     passes the MTP KV view/cache into TextContext; tp_split_spec forces mtp/* objects to Replicated (otherwise they'd be sliced by GatherRows per shape);
     verified by test_tp2_load --spec mtp that both cards load successfully
  - **36.2 Proposal**: on shard A call mtp_forward_batch + K-1 mtp_forward_ar_step;
     oracle: under the same prefix, TP-2's draft tokens must match the single-card engine
  - **36.3 Verification**: the main model's batched forward T=K+1 outputting **per-column logits** (the existing forward_tp2_prefill only computes the last column, needs extension),
     then goes through speculative_accept_greedy_drafts (greedy) / the sampling acceptance path
  - **36.4 Rollback and rounds**: snapshot the GDN state pool before verification (KV is position-addressed; on partial acceptance it is overwritten by later rounds, no rollback needed); restore on rejection;
     mtp_prepare_next_round advances anchor/frontier/budget; multi-token commit goes through preview_model's existing prefix acceptance
  - **36.5 Correctness and performance**: **the greedy MTP token stream must be bit-identical to pure greedy** (the strongest oracle);
     the sampling path's numeric oracle is by spec-probability acceptance; then measure tok/s (K=1/2/3, target 60+)
- [ ] Leftover (non-blocking): the capacity log line shows pages 0/0 under TP-2 (memory_summary not wired to the shard pools);
  the timings fields prompt_ms/predicted_ms all 0 (the TP-2 path doesn't fill them)
- Follow-up: MTP landing → multi-request concurrency smoke (currently max_concurrency=1) → perplexity soundness



## Round 36b: free verification-window optimization, fold-geometry oracle, the "greedy bit-identity" criterion voided

### 1. Free optimization: the verification window no longer re-projects the last column (done and verified)
- forward_tp2_prefill's `logits`/`logits_peer` changed to **optional pointers**: ordinary prefill passes [V,1] (last column only),
  the verification window passes `logits_columns` [V,T] (all columns) with `nullptr, nullptr`. Previously both were computed:
  the last column of `logits_columns` and [V,1] are exactly the same data, yet **lm_head is read again** (a weight stream per card per round).
- The two never-read [V,1] buffers in the MTP branch were also deleted; `forward_tp2_prefill` gained
  an explicit "exactly one of the two" validation.
- Measured (the same 96-token greedy request, prefix already cached): K=3 long 47.99 tok/s; the same criterion pre-change 46.7 tok/s
  — the gain is on the order of the estimated 1.4 ms/round; the rest was masked by the allreduce.

### 2. Independent numeric oracle for `gdn_replay_fold`'s new geometry (done, passed)
- `tests/ops/test_gdn_replay_fold.cpp`: `FoldProfile` gained a `qk_heads` field (previously a file-level constant 16),
  7 new {48, 8, 24, 5120} (TP-2 shard geometry) cases, and corrected the `kProfile` and `Hq=` suffixes.
- `build/tests/ninfer_gdn_replay_fold_test` → `OK gdn_replay_fold` (exit 0). The case carries its own host-side
  independent oracle: the folded recurrent slot, conv history, and guard bytes compared byte-for-byte.
- Conclusion: **the fold (state-commit) mechanism itself is correct**; the new geometry instantiation has no indexing/numeric issues.

### 3. Performance measurements (temperature 0, same prompt, prefix cache hit)
| Route | 96-token request | long request (predicted_n == completion, self-consistent) |
|---|---|---|
| Pure decode (no --spec) | 31.49 tok/s | — |
| MTP K=1 | 41.99 | 42.43 |
| MTP K=2 | **50.47** | **54.03** |
| MTP K=3 | 47.92 | 47.99 |

→ **`--draft-tokens 2` is the current optimum (+71% vs pure decode)**; K=3's third draft has a low marginal acceptance rate,
yet each round pays an extra bridge+AR and a wider verification.

### 4. "Greedy MTP == pure greedy bit-identical" does not hold (characterized; the original 36.5 criterion is void)
- Phenomenon: same prompt, temperature 0, all four routes are **individually deterministic** (repeated requests on the same route are bit-identical), but pairwise different:
  in generation order (reasoning+content) plain vs k1 diverges at the 9th character, plain vs k2 at the 79th, plain vs k3 at the 72nd,
  k2 vs k3 at the 72nd.
- Reasoning: the greedy acceptance op (`speculative_accept_greedy_drafts`; the kernel comment explicitly says it degenerates at temperature<=0 with no penalty
  to "longest matching draft prefix + the divergent column's argmax") guarantees **a committed token is always the verification window's own argmax**.
  So as long as the routes' target logits are the same, the commit stream must be the same; the four routes differ ⇒ **the verification window's target logits
  depend on the window width T**.
- The only input that varies with K is the window width (window content = same anchor + same draft prefix), so this is unrelated to draft quality,
  MTP KV, or fold state (Section 2's oracle + individual determinism rule out state-class defects).
- Root-cause layer: TP-2's "verification" directly reuses `forward_tp2_prefill` (the Phase::Prefill chunk kernel, computing T=K+1 in one go);
  ordinary decoding goes through `forward_tp2` (T=1); the single-card engine has a **dedicated Phase::Verify path**
  (`target_verify_batch`, text.cpp:790 `run_layers(x, Phase::Verify, tap)`). TP-2 has no Verify phase;
  fusing "verification" and "prefill" into one entry costs the verification columns' argmax not being bit-identical to T=1 decoding.
- Characterization: the output is still a **legal greedy continuation under the verification path's own numerics**; near-tie
  argmax flips appear once every ~15-20 tokens; not a state/bookkeeping defect. But it **does not satisfy "cross-route bit-identity"**.
- To restore bit-identity (next step, not done): add a Verify phase to TP-2 (causal phase with valid_columns semantics aligned to the
  decode kernel), or let the verification's **column 0** go through the same projection/attention kernel as T=1. First a minimal localization experiment:
  the verification window's T=2 column-0 logits vs T=K+1 column-0 logits (same state, same token; RecordForReplay width 2 legal),
  quantifying max|Δ| and the per-column argmax agreement rate, to judge whether it is a projection-tile difference or an attention/GDN kernel difference.

### Round 36b ops note
- Switching K requires a serve restart (`--draft-tokens` is a startup parameter): `serve_stop.sh` → `: > serve_supervised.log`
  → a background job runs `serve_supervise.sh 65536 8088 '--spec mtp --draft-tokens N'` → `r36b_wait.sh` waits
  for "listening on" before sending requests (an old log's listening line won't do).
- Measurement scripts: `r36b_fold_test.sh`, `r36b_wait.sh`, `r36b_req.sh <K>`, `r36b_cmp.sh`, `r36b_timings.sh`.

### Round 36c: serving under sampling parameters (user required --temp 0.7 --top-k 20 --top-p 0.80)
- The correct CLI is `--temperature` (no `--temp` alias); top-k ceiling 20:
  `serve_supervise.sh 65536 8088 '--spec mtp --draft-tokens 2 --temperature 0.7 --top-k 20 --top-p 0.80'`
- Measured (request body without temperature/top-k/top-p, using server defaults): 128-token request 50.03 tok/s,
  39-token (stop token ending early) 48.26 tok/s; slightly below greedy K=2's 54.03
  (the sampling acceptance branch builds a truncated distribution per column + random acceptance, and the draft is a one-hot greedy proposal, lowering the acceptance rate).
- Functionally usable: http=200, coherent output; goes through `speculative_accept_greedy_drafts`'s temperature>0 sampling branch
  (kernel comment: distribution-correct rejection sampling against the truncated target distribution).
- Leftover:
  1. `[mtp] round ... rate=hit/checked`'s current semantics is "accepted drafts / rounds", which can exceed 100% and mislead;
     it should be "accepted drafts / proposed drafts" (a single line `mtp_draft_checked_ += mtp_drafts_` suffices).
  2. The TP-2 sampling path still lacks a distribution-correctness statistical oracle (the single-card side has a corresponding test); currently only functional usability + coherent output are verified.

## Round 36d — the user-reported "output dead loop" and the MTP crash

### Conclusion 1: the dead loop is unrelated to MTP (characterized, usable)
- The user's test prompt: "创建一个HTML，内容是SVG绘制一个鹈鹕骑自行车的2D动画" (create an HTML containing a 2D SVG animation of a pelican riding a bicycle).
- MTP route (K=2 + temp0.7/top-k20/top-p0.80, server default thinking xhigh): all 700 tokens fell in reasoning,
  content=0, finish=length, reason=2491 chars; the model stayed stuck on "conceiving/restating the task".
- The same request on the **non-MTP** route: likewise all 700 tokens in reasoning, content=0 (reason=2549) → unrelated to MTP.
- The request body with `"reasoning_effort": "none"` (non-MTP, max_tokens 1200): reasoning=0, content=2893 chars
  (a complete Chinese explanation + a complete single-file HTML), 32.3 tok/s. → the "dead loop" the user saw is **runaway thinking** (the default thinking xhigh
  eating the whole output budget), unrelated to the sampling parameters (the user also observed it existed before adding the parameters).
- Usable means: the request body `reasoning_effort: none|minimal|low`; the server's `--default-thinking-budget N`; a larger max_tokens;
  presence/frequency penalty if needed.

### Conclusion 2: a real crash exists on the MTP route (blocks MTP stability, to fix)
- Scene: the D2H `cudaMemcpyAsync` at `tp2_generation_core.cpp:443` reported `cudaErrorIllegalAddress`.
  This is the first stream sync point after prefill; the truly failing kernel is very likely in the prefill/priming phase, not the proposal itself.
- Reproduction: after starting the MTP K=2 service, the first user request crashed (pelican + reasoning_effort=none, max 1200);
  water-cycle + max 900 + thinking xhigh also crashed twice (~9s). Warmup itself (pos=55/56) was fine.
- compute-sanitizer (memcheck) couldn't catch it: the workspace is one 384 MiB cudaMalloc block; an out-of-bounds landing inside the arena is invisible,
  only going outside the arena becomes an illegal address.
- Next step: use `CUDA_LAUNCH_BLOCKING=1` (tools/tp_bootstrap/r36d_launchblock.sh) so the error is reported directly at that launch's
  CUDA_CHECK, locating the specific op.
### Round 36e — MTP crash root cause and fix (fixed, A/B verified)
- Localization: temporarily inserted cudaStreamSynchronize stage checks into the TP-2 code (prefill forward / priming /
  proposal bridge / ar step / verify / accept / fold), then subdivided inside mtp_prefill_chunk;
  first convergence: prefill forward clean, the final priming reported the error; second convergence: inside priming, mtp_forward_stem.
- Root cause: when mtp_prefill_priming wrote the prompt's last token into the MTP input ids it used
  ids_t.data + (length - 1); Tensor::data is a void*, so the expression is a **byte offset** (a GNU extension),
  and the sampled token's 4 bytes straddled two adjacent int32s, producing an illegal token id >= 2^24;
  ops::embedding fetched a row far beyond the embedding table -> cudaErrorIllegalAddress.
  With a short prompt (warmup) the corrupted int32 happened not to exceed the legal id range, so it only crashed on real requests.
- Fix: changed to static_cast<std::int32_t*>(ids_t.data) + (length - 1) (an element offset).
- Incidental fix 1 (latent out-of-bounds): the MTP KV physical page count was originally the same as the text KV. Referring to the single-card planning
  (planning/startup.cpp's mtp_extra_pages): the speculative window writes mtp_drafts_ more positions past the commit frontier, so the MTP pool needs + ceil(mtp_drafts_/kPagedKVPageSize) pages,
  and the execution row maps only the logical capacity (avoiding a publish out-of-bounds table write).
- Incidental fix 2: the mtp round's rate statistic changed from "accepted drafts / rounds" to "accepted drafts / proposed drafts"
  (mtp_draft_checked_ += mtp_drafts_), no more >100%.
- Verification (MTP K=2 + temp0.7/top-k20/top-p0.80, the same batch of previously certain-crash requests):
  pelican+reasoning none x2 -> http=200, 65/62 tokens, finish=stop, 47-48 tok/s;
  water-cycle max900 -> http=200, 70 tokens, 52.4 tok/s; the service process no longer exits.
  Acceptance sample rate=105/184 (57%), consistent with the pre-fix per-round statistic ~55%, showing the ids corruption had limited impact on the draft.
- Investigation tools (kept): tools/tp_bootstrap/r36d_dbg.sh (self-contained serve start + request burst + status check),
  r36d_verify.sh, r36d_show.sh, r36d_analyze.sh. The debugging stage syncs have been removed from the source.### Round 48 — item-4 end-to-end proof: a 118,869-token prompt succeeded
- The recommended config (plain + fp8 KV + `--max-context 131072`) measured: **prompt_tokens=118,869**, http=200, 96.8 s;
  another case prompt_tokens=19,869, http=200, 19.5 s. => the 131,072 capacity is truly usable (with bf16 KV the ceiling is only 65,536).
- Script `tools/tp_bootstrap/r48_longctx.sh`; evidence written to `docs/tp2-dual-5060ti.md`.
- Note: both `content_chars=0` cases are the known thinking-budget behavior (max_tokens=220 filled by reasoning), not errors.### Round 47 — W1 stability blocker: `rope.cu:189` (reproduced under two independent hypotheses)
- Added a runtime switch `NINFER_TP2_VERIFY_PHASE` (default Prefill, removed after verification) for cheap iteration;
  also added the "Verify phase uses [width, batch]-shaped positions" binding (bound per card).
- Result: still stably crashes at `src/ops/launcher/rope.cu:189 cudaErrorIllegalAddress`, meaning RoPE's sharded width>1
  tensor-layout/workspace assumptions are not met; a deeper rework is needed in the RoPE op and its call sites (the rope position/shape contract).
- Disposition: stopped further probing (high context-budget-exhaustion risk); the core call restored to `Phase::Prefill` and recompiled;
  the Verify wiring in `text.cpp` (the sequence/valid-columns/backend-rows/positions bindings + two shard gates) is kept, disabled by default.
- **W1 conclusion fixed**: token-for-token agreement between MTP and plain requires completing the sharded width>1 Verify path (currently blocked on RoPE),
  infeasible in this session; MTP ships as an experimental feature, with the recommended default plain + fp8 KV + 131k config (verified).### Round 46 — delivery-doc number-criterion corrections
- Found and corrected a prefill criterion error in `docs/tp2-dual-5060ti.md`: the earlier r39 "long prompt" fill script had not taken effect;
  the measured prefill request had only **279 tokens**, so 0.453 s corresponds to **~615 tok/s** (not ~3.3k tok/s); the token count is now annotated in the table.
- Also recorded: the **first request** after a fresh serve includes CUDA graph capture (fp8/131k first request 14.4 s); the doc notes steady-state numbers as the standard.
- Recommended config (plain + fp8 + 131072) steady state: decode 256 tokens 8.10–8.14 s (on par with bf16).### Round 45 — W1 convergence conclusion: TP-2 sharded batched verify is an engine-level gap
- This round's wiring attempts (all implemented and compiling, off by default): `forward_tp2_prefill(..., Phase)`; the Verify phase's
  sequence bindings (batch/width/state-slot/valid-columns/backend-kv-rows, bound per card); `batched_verify` and the GDN
  `ph == Verify` branch gated for shards (going through the decomposed recurrence, numerically consistent between the two phases).
- The failures exposed in sequence: `tensor dimensions must be positive` → `gdn_input_proj_conv_record workspace:
  unsupported single-parent profile` → `residual_add` illegal access → `cudaMemcpy2DAsync` illegal access → **`rope.cu:189`**
  illegal access (warmup hangs under `CUDA_LAUNCH_BLOCKING=1`). => the width>1 Verify path on shards lacks shape/workspace wiring in multiple places.
- Conclusion: W1 (MTP token-for-token parity) is infeasible within this session's budget; MTP ships as an experimental feature (+25% end-to-end throughput /
  +50% pure decode); the recommended default is the plain route.### Round 44 — temporary probe cleanup complete
- Scripts `tools/tp_bootstrap/strip_tp2_probes.py` + `strip_tp2_probe_comments.py` deleted 4 env probe blocks
  (ROUND_DETAIL/PARITY/TOKEN_TRACE/SKIP_FOLD) and 7 leftover comment lines; `grep NINFER_TP2_` hits 0; compile BUILD_EXIT=0.
- Kept intentional behavior: verify still goes through Phase::Prefill (with a TODO noting Phase::Verify is unfinished); MTP is an experimental feature.### Round 43 — fp8 KV quality evidence (W3 completeness raised)
- Recommended config (plain + fp8 KV + 131072 context) 4 prose samples: content 1365/859/1382/1602 (median ~1374),
  longest-segment all 0 in all 4; control bf16 plain baseline 1233/763/1469/1456/1430/1621 (median 1443, all 0-segments 0) => **fp8 KV does not degrade quality**.
  Written into the KV section of `docs/tp2-dual-5060ti.md`.
- Still to do: (1) clean up the 4 temporary probes in `tp2_generation_core.cpp` (lines 830/851/922/953, exact text needed);
  (2) W1's TP-2 batched verify (unfinished; the goal must not be marked complete); (3) the final delivery report.### Round 42 — W5 delivery doc landed
- Added `docs/tp2-dual-5060ti.md`: the recommended config (`--kv-dtype fp8 --max-context 131072 --kv-capacity auto`),
  the benchmark table (prefill/decode, plain vs MTP K=3, VRAM, KV dtype comparison, 131k/262k results), the KV quantization explanation,
  the MTP experimental feature and its deviation root cause, this branch's fix list.
- 8088 currently runs the recommended config (fp8 + 131072 + auto, temp 0.7/top-k 20/top-p 0.80).
- Remaining: clean up the temporary probes in `tp2_generation_core.cpp` (NINFER_TP2_PARITY/ROUND_DETAIL/TOKEN_TRACE/SKIP_FOLD),
  then do the final goal assessment (W1 unmet => must not be marked complete).### Round 41 — W3/W4 resolved: per-device smem opt-in (fp8/nvfp4/int8 KV all usable, 131k context)
- Root cause (high confidence, fixed and verified): the quantized prompt-attention launches in `prompt_fp8.cu:19-22` etc. use an **in-function `static const`**
  calling `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 92,416 B)`. The attribute is a **per-device** function attribute; TP-2's two shards
  each launch, and the second device never opted in => launches requesting >48 KiB dynamic smem report `cudaErrorInvalidValue`.
  The bf16 path's smem <48 KiB needs no opt-in, so only the quantized variants failed.
- Fix: changed to a **per-device cached** opt-in (`cudaGetDevice` + `static bool attr_done[64]`); files changed:
  `prompt_fp8.cu`, `prompt_nvfp4_non_rdc.cu`, `prompt_k8v4.cu` (script `tools/tp_bootstrap/fix_per_device_attr.py`).
- Verification (`r42_fp8check.sh`, `--devices 0,1 --max-context 65536 --kv-capacity auto`, 256-token request):
  bf16 8.153 s / fp8 8.091 s / nvfp4 8.093 s / int8 8.140 s, all http=200, the capacity line showing KV 65,536 tokens <dtype> respectively.
- Larger context (`r43_bigctx.sh`): **fp8 + `--max-context 131072` OK** (KV 131,072 tokens, 13,904 MiB/card, request 8.102 s);
  **int8 + 131072 OK** (13,952 MiB, 8.088 s); `fp8 + 262144` OOM-failed (expected, KV needs ~2× the VRAM).
- Conclusion: **2× context (65k→131k) under fp8/int8 KV keeps VRAM and throughput essentially unchanged** => the recommended config changed to
  `--kv-dtype fp8 --max-context 131072 --kv-capacity auto`.### Round 40 — W3 scope located: all quantized prompt-attention variants fail at the same point
- Measured: TP-2 + `--kv-dtype nvfp4` fails at `prompt_nvfp4_non_rdc.cu:33 CUDA_CHECK(cudaGetLastError()) failed:
  cudaErrorInvalidValue`, the **same class and same point** as fp8's `prompt_fp8.cu:36` => not a single-dtype issue but a shared missing precondition of the quantized prompt
  attention path (most likely: the KV cache's scale plane/view not wired for that path; bf16 has no scale, hence fine).
- The single-card control experiment was **ineffective**: `--devices 0` + `--kv-capacity 32768` was rejected by the CLI (printed help and exited); the single-card device parameter and kv-capacity constraints must first be confirmed with `--help` before redoing; this round recorded the `--help` filtered results in the session.
- Next round's W3 steps: (1) correct the single-card parameters with `--help` and rerun the r40 control; (2) if confirmed TP-2-specific, add minimal diagnostic prints at the quantized KV cache creation
  (scale plane allocation/binding) and the `prompt_*_fp8/nvfp4.cu` launch parameters to locate the missing item.### Round 39 — W2/W4 measured data (banked)
- Environment: 2×(5060 Ti 16GB) TP2, Qwen3.8-27B NVFP4, greedy, `--kv-capacity auto`, `tools/tp_bootstrap/r39_bench.sh`.
- prefill (~1500-token prompt, output 8): plain 0.453 s (≈3.3k tok/s), MTP K=3 0.526 s.
- decode (256 tokens, end-to-end including prefill): plain 8.153 s = **31.4 tok/s**; MTP K=3 6.532 s = **39.2 tok/s** (+25%).
  Note: the earlier pure-decode count (token trace) was plain 31.5 / MTP K=1 42.0 / K=2 50.5–54.0 / K=3 47.9–48.0 tok/s,
  a different criterion from end-to-end; reports must annotate the criterion separately.
- VRAM: plain 13,888 MiB/card; MTP 14,582 / 14,322 MiB (weights 11.1 GiB + KV 65,536 tokens bf16).
- **`--max-context 131072` fails to start under bf16 KV**: `cudaMalloc failed: cudaErrorMemoryAllocation` =>
  a larger context requires KV quantization (fp8/int8) or a lower kv-capacity; this is W3's direct motivation.
- W3 fp8 failure point refined: the kernel launch at `src/ops/softmax_attention/dense/causal_cache/prompt_fp8.cu:27-36`
  reports `cudaErrorInvalidValue`; the same function's `cudaFuncSetAttribute` (`kCausalPromptFp8SmemBytes`) at lines 19-22 succeeded,
  line 22 threw nothing => not an smem overflow. Priority checks: (a) an invalid `cudaStream_t` passed cross-card/cross-context by TP-2;
  (b) the fp8 cache's `k_scale_pages`/`v_scale_pages` not allocated per shard; (c) `q.ne[2]` (tokens) anomalous under shards.### Round 38c — W1 disposition decision (evidence sufficient) and the follow-up route
- New evidence: under `CUDA_LAUNCH_BLOCKING=1` the TP-2 verify-version warmup **hangs** (it crashes when non-blocking) => the batched-verify on the
  head-split shard path has a low-level defect (the classic dual-state of illegal read/write of unmapped memory); the fix depth is uncontrollable.
- Mechanism conclusion (already quantitatively supported by 37g/37h): chunk verify (Prefill kernel) and decode (Verify kernel) have different summation order/rounding,
  per-column 10-20% argmax flips, and the KV rows of accepted tokens are written by the chunk kernel => the drift accumulates permanently => the MTP trajectory keeps diverging
  from plain => long generations degrade (0 repetition). **Conclusion: MTP token-for-token parity requires a full TP-2 batched verify implementation,
  infeasible this round**; MTP stays an opt-in experimental feature, with the limits stated clearly in the docs.
- Delivery criteria (priorities rearranged per objectives 1-5):
  1. MTP: keep usable but labeled experimental; the report gives the measured numbers (K=1/2/3 throughput, quality comparison, known deviations and causes).
  2. W2/W4: complete the prefill/decode benchmark matrix (context × prefill-chunk × MTP on/off) and the VRAM breakdown with the plain route,
     giving the optimal parameters for larger context (prompt cache / max-context / kv-capacity / prefill-chunk).
  3. W3: fix fp8/int8 KV (`prompt_fp8.cu:36` invalid value, the int8 row-copy delayed error); after the fix, rerun r38_kvsweep.
  4. W5: the benchmark doc + the recommended serve config + README/docs updates; clean up temporary scripts and PLAN.md.### Round 38b — the Verify path still fails: CUDA_LAUNCH_BLOCKING needed to locate the real kernel
- After adding the `active_valid_columns_` (I32[1]=tokens) and `active_backend_kv_table_rows_` (I32[1]=0) bindings and rerunning
  `r37_colparity.sh`: the MTP service still crashes, the error point drifting to `text.cpp:91 cudaMemcpy2DAsync` `cudaErrorIllegalAddress`.
- Correction: `copy_row_block` (text.cpp:76-98) is **not a bug** — in that file dim 0 of every activation tensor is the innermost contiguous axis,
  so dpitch/width = rows*element, spitch = parent_rows*element, height = columns is correct; the erroneous edit never took effect; the function was left unchanged.
- Conclusion: `cudaMemcpy2DAsync` is an asynchronous call; its error may be a **delayed manifestation of an earlier kernel fault** (residual_add first, then this point) =>
  next round's first step: start the Verify version with `CUDA_LAUNCH_BLOCKING=1`, get the **real failing kernel and location**, then fix shapes/bindings surgically.
- Enabling Verify only requires adding `qwen::TextPhase::Verify` back to the verify call in `tp2_generation_core.cpp` (currently Prefill + TODO).
- W3 status (measured this round): bf16 4 samples content 1874/1797/0/0 (still 188/206-char 0-segments); fp8 and int8 fail at startup,
  the errors landing at `prompt_fp8.cu:36` (cudaErrorInvalidValue) and `text.cpp:91` (cudaErrorIllegalAddress, suspected same-source delayed error) respectively.### Round 38a — W3 status: the quantized KV paths are unusable (fix required)
- `--kv-dtype fp8` fails at startup: `src/ops/softmax_attention/dense/causal_cache/prompt_fp8.cu:36`
  `CUDA_CHECK(cudaGetLastError()) failed: cudaErrorInvalidValue` (the fp8 causal attention kernel's prompt/prefill launch parameters illegal).
- `--kv-dtype int8` fails at startup: `text.cpp:91 cudaMemcpy2DAsync` `cudaErrorIllegalAddress` (KV row copy source/destination stride mismatch).
- bf16 (default) 4/4 requests http=200, single requests 34.7/37.8/36.4/36.6 s (400+ token prose, MTP K=3).
- Conclusion: W3 is not "configuration will do" but "the two quantized KV paths have bugs under TP-2/head-split geometry"; each needs locating and fixing;
  after the fix rerun `r38_kvsweep.sh` for quality/VRAM/speed comparison.### Round 37h — the root-cause model converges: inter-kernel cumulative drift of bf16 KV + TP-2 missing batched-verify bindings
- Key explanation (consistent with all observations): the KV cache is bf16. The KV rows chunk verify (Phase::Prefill) writes for "accepted tokens"
  are computed by the chunk kernel, while the plain route computes the same token's KV rows with the decode kernel => different bf16 rounding => per-token cumulative drift
  => argmax flips after ~54-80 positions (per-column parity first diff pos=80; earlier single-column probe pos=55).
  This explains 37g's "every column 10-20% inconsistent", "the licensing map is correct yet still diverging", "the first divergence appears at index 3-25".
  Corollary: **as long as verify is not a decode-equivalent kernel, the two routes' KV/state cannot be aligned bit-for-bit**; W1's token-for-token criterion must rely on the Verify phase.
- The single-card MTP path already has width>1 batched verify (reusable as a reference implementation):
  `text.cpp:396-418` the MTP attention's batched branch (using `active_sequence_batch_`/`active_sequence_width_`,
  `active_valid_columns_`, `active_backend_kv_table_rows_`, `batch_mtp_kv_->batch_layer_view(0)`),
  `text.cpp:854-855` is the batched verify binding template: `active_backend_kv_table_rows_` and `active_valid_columns_`.
- TP-2 under the Verify phase lacks exactly this binding set: I bound only batch/width/source-slot, so after MTP attention enters the batched branch
  it dereferences the unbound `active_valid_columns_`/`active_backend_kv_table_rows_` => the crash point is `residual_add` (shape/pointer mismatch).
- Next round's minimal steps (in order): (1) in `forward_tp2_prefill`'s Verify branch, fill in `active_valid_columns_` (I32[width*batch],
  all window columns 1) and `active_backend_kv_table_rows_` (the window columns' backend KV rows, taken from the prefill's existing row table);
  (2) judge with `r37_colparity.sh` (expecting col* diff to drop sharply / be 0); (3) if the GDN's width>1 record path still reports
  `unsupported single-parent profile`, add a shard fallback (reusing the decomposed record path).
- Fallback delivery criteria (if (A) ultimately infeasible): MTP as an opt-in experimental feature, the default release being the plain route's optimal config, and the docs
  stating clearly the MTP numeric floor caused by inter-kernel bf16 drift; but continue (A) first.### Round 37g — per-column parity quantification: every column 10-20% inconsistent => the Verify phase is mandatory (no cheap shortcut)
- The probe upgraded to "per-column T=1 replay": the scratch state is restored once before and once after the probe (guaranteeing only the kernel difference is compared), and `forward_tp2` (= the
  same function as the plain route, Phase::Verify) replays each window column in `mtp_position + j` order, compared with the chunk verify's `target_tokens[j]`.
- Result (K=2, width=3, greedy, single request 400 tokens): col0 same=228/diff=24 (9.5%, first diff pos=80),
  col1 same=200/diff=52 (21%), col2 same=214/diff=38 (15%).
- Conclusion: **chunk verify (Phase::Prefill) and decode (Phase::Verify) are numerically inconsistent per column**, so both draft acceptance and bonus can
  differ from plain => the MTP trajectory diverges => thinking runaway/0 repetition. The licensing map itself is correct (proven in 37d).
- The only correct fix: (A) route the window verification through the Verify phase with width>1. Current block: once enabled, the warmup crashes at
  `src/ops/launcher/residual_add.cu:27 cudaErrorIllegalAddress` (shapes/workspace not derived per width).
  Next step: locate the `text/layers/... verify columns=3` diagnostic context and the mixer/mlp shape derivation under the Verify phase
  (whether the buffers/views are per 1 column while the loop is per 3 columns), fill it in, then rerun `r37_colparity.sh` (expecting col* diff=0) before running `r36h_trace.sh`.
- Probes kept: NINFER_TP2_PARITY (per-column parity), ROUND_DETAIL, TOKEN_TRACE, SKIP_FOLD.### Round 37f — a second-layer gap in the Phase::Verify wiring (shapes/workspace)
- Fix attempt: `batched_verify` gained a `shard_config_ == nullptr` gate (head-split shards lack the full-row-count fused record
  workspace and must go through the shard decomposed route; the gate is kept).
- Result: the warmup gets past the input projection but crashes at **`src/ops/launcher/residual_add.cu:27`** with
  `cudaErrorIllegalAddress` (http=000, no FATAL line) => the Verify phase under width>1 needs TP-2's **tensor shape/workspace**
  wiring (the column counts of the residual/window buffers don't match the width binding).
- Current state: the verify call **reverted to Phase::Prefill** (with a TODO); the MTP service is usable; the `phase` parameter, sequence bindings,
  and shard gate changes are kept in reserve.
- Two continuation routes (next round picks one; recommend (B) first for a verifiable gain):
  (A) Complete the Verify phase wiring: check `residual_add`'s shape/workspace contract and `active_sequence_width_`'s buffer sizes
      in the Verify branch (the run_layers/attn_mix/mlp side of text.cpp), and fill in TP-2's shape derivation and workspace plan.
  (B) Hybrid numeric correction (small step, immediately judgeable): chunk verify is only used to decide draft acceptance; **the last licensed token (the bonus)**
      is recomputed from the same state via T=1 `forward_tp2` (naturally Phase::Verify) before committing; the observed error is exactly the bonus disagreeing with decode.
      Criterion: `r36h_trace.sh`'s first divergence should move significantly later (expecting occasional divergences only at mis-accepted drafts).### Round 37e — Phase::Verify wiring attempt: both two block points precisely located
- Implemented (kept, default off): `forward_tp2_prefill(..., Phase phase = Phase::Prefill)`, phase passed through to
  `run_layers_tp2`; when phase == Verify, bind `active_sequence_batch_ = 1`,
  `active_sequence_width_ = tokens`, `active_linear_state_source_slots_` (I32[1] = 0, bound per card) per the single-card decode practice.
- Block point 1 (resolved): without the sequence binding, `text/layers/0 verify columns=3: tensor dimensions must be positive`.
- Block point 2 (unresolved): after binding, `text/layers/0 verify columns=3: gdn_input_proj_conv_record workspace:
  unsupported single-parent profile` => **TP-2's workspace plan has no profile registered for width>1 record-type GDN ops**.
  Need to check `src/models/qwen3_5/program/planning/startup.cpp` (already has 3 Verify references) and the workspace planning:
  either register the corresponding profile for the TP-2 route by this phase/width, or let the verify record path reuse the single-card's already-registered profile.
- Current code state: the verify call in `tp2_generation_core.cpp` **temporarily reverted to Prefill** (with a TODO), guaranteeing the MTP service can start;
  the phase parameter and binding code are kept for use.
- Alternative (if the profile wiring cost is too high): hybrid numeric correction — use chunk verify only to decide "whether the draft is accepted",
  while **the last licensed token (the bonus) is recomputed from the same state via T=1 `forward_tp2` (Phase::Verify)** and committed;
  the observed error is exactly "the chunk's bonus argmax disagrees with decode" (pos=79: target[0]=13 vs decode 15).
  Cost: one extra T=1 forward per round (~+30-50% target compute), but the bonus immediately matches plain.### Round 37d — W1 root-cause separation complete: the state is correct; the defect is the verify phase's numeric inconsistency
- Round-detail probe (NINFER_TP2_ROUND_DETAIL=1, printing per-round drafts and the verify per-column argmax):
  `[round] pos=70 drafts=[97913,97237] target=[97913,97237,99986]`, `pos=73 drafts=[101729,4960] target=[101729,4960,130621]`,
  `pos=76 drafts=[16,15] target=[16,15,15]` all correct (licensing when target[j]==drafts[j], bonus=target[A]);
  but `pos=79 drafts=[15,96356] target=[13,15,98003]`: **target[0]=13 while decode/plain gives 15 at that position** — MTP committed the wrong token
  (the draft was originally right) => the licensing map is correct; the verify's per-column argmax itself is wrong.
- Parity probe (NINFER_TP2_PARITY=1, verify column 0 vs T=1 decode under the same state): K=1 190/208 rounds agree, K=2 127/179 rounds agree,
  **all first disagreements at absolute position 55**, then ~10-20% of rounds flip on near-ties => pointing at the same position as plain's first divergence (index 3).
- Conclusion: **state/fold/conv/licensing are all correct**; the remaining quality defects and "0 repetition/thinking runaway" come from verify using Phase::Prefill (the T=K+1 chunk kernel)
  being numerically different from decode's **Phase::Verify** (T=1).
- Fix direction (code located): `src/models/qwen3_5/execution/text.cpp:1394` shows TP-2's T=1 decode uses `Phase::Verify`;
  the constraint at `text.cpp:1021` `(ph == Phase::Verify) && (active_sequence_batch_ > 1 || active_sequence_width_ > 1)`
  indicates the Verify phase originally supports width>1 (speculative verification). Next step: switch the MTP window verification from `forward_tp2_prefill` (Phase::Prefill)
  to the width>1 Verify phase entry (check the forward_tp2_* declarations in text.h for a phase parameter or a dedicated batch/verify entry),
  then judge with `r36h_trace.sh` (expecting the first divergence to move sharply later/vanish), then run the 6-sample statistics (expecting content on plain's order, no 0 repetition).
- Temporary probes (removed uniformly after the fix): NINFER_TP2_TOKEN_TRACE, NINFER_TP2_PARITY, NINFER_TP2_ROUND_DETAIL, NINFER_TP2_SKIP_FOLD.### Round 37c — K sweep (trace first divergence) and quality-scan results
- Trace (greedy, compared token-by-token with plain): K=1 first divergence 3 (content 707), K=2 first divergence 10 (content 0), K=3 first divergence 7 (content 401).
  => the divergence is early and happens at every K; K=1/3 still produced body text in that run.
- Quality scan (temp0.7/top-k20/top-p0.80, essay prompt, 4 each, max 1400):
  K=1 content = 0,0,2332,0 (3/4 degraded, longest0 = 0,0,2,0)
  K=2 content = 939,0,1850,0 (2/4 degraded, longest0 = 173,0,111,0)
  K=3 content = 434,1951,1632,1963 (**0/4 degraded**, median 1791; longest0 = 93,158,511,221)
  plain control (6 runs): content 1233/763/1469/1456/1430/1621, median 1443, longest0 all 0.
- Conclusion: K=3 quality is already on plain's order (median 1791 vs 1443), but **every MTP K shows 0-consecutive segments (plain never does)**
  => the MTP route still has quality defects; K=3 is merely the mildest. Before the fix, MTP should use K=3.
- Fold contract check (`include/ninfer/ops/gdn_replay.h:19-47`): commit_columns is the record prefix of [0,T];
  the conv history = tail_3(old_history || conv_record[0:commit_columns]); source_state_slot = the slot used when the row's record was produced.
  Per this contract, my current restore(slot0 <- scratch) + fold(source=dest=0, commit=committed) is correct on paper
  => the remaining deviation is more likely "logit numerics/near-ties" than a state misalignment.
- Next step (decisive oracle): print, in the trace, **the logit value of that token at each position** for both routes (D2H the corresponding element of window_logits/pre-sampling
  logits) and compare the logit difference of the same position and same token: if the difference is tiny but the argmax is a near-tie => a numeric problem (needs a dedicated
  Verify phase to align with decode numerics); if the difference is large => still a state/column semantics error.### Round 37b — correcting 37a: the record plane **does contain the conv** (new hypothesis: record-column vs window-column alignment/off-by-one)
- `src/ops/linear_attention/gated_delta_net/recurrent.cu:113-117` references `records.conv.data` and `states.conv_layer0.data`
  (with `conv_layer_stride_bytes`), showing the record plane has conv records and the fold replays the conv too; my 37a
  "record doesn't contain the conv" judgment came from reading only the first 10 lines of `validate_replay_record` (key/value/gate) and is voided.
- So the restore+fold conv handling **should** be right; the reason the first divergence 4 -> 10 is still inconsistent needs re-investigation. New hypotheses (by priority):
  (h1) **Window-column vs record-column alignment off-by-one**: my verify window is [anchor, d0..d_{K-1}] (width K+1), while the record's
       `width`/column semantics may be K columns "new tokens only"; if so, `commit_columns = committed` advances one column short of the anchor itself
       (the state lags by one column) => wrong from the 2nd round, matching the first divergence at 10. Need to read `GdnReplayRecords`'s spec/width,
       the full `validate_replay_record`, and the window-column -> record-column mapping when the record is written, then decide whether it's
       `commit_columns = committed ± 1` or the verify window changed to K columns excluding the anchor.
  (h2) The record's `rows`/batch dimension doesn't match TP-2's single-row usage (should be 1 at rows=1; needs checking).
  (h3) The fold must synchronize after the verify's stream (the two cards' stream order/dependency): currently the fold and verify are on the same stream and should be ordered,
       but the cross-stream dependency between the restore's memcpy and the fold's kernel needs confirming.
- The judgment tool is unchanged: `r36h_trace.sh` for the first-divergence index (currently 10; old 4; skip-fold 25); one change per run gives a bisection.### Round 37a — W1 key finding (the record plane lacks the conv state)
- `src/ops/linear_attention/gated_delta_net/replay.cpp:129-133`: the record plane has only `key_record`
  [kStateDim, qk_heads, width, rows], `value_record` [kStateDim, value_heads, width, rows],
  `gate_record` [2, value_heads, width, rows] — **no conv/short-convolution record**.
- But the `LayoutRegion linear_conv` at `src/models/qwen3_5/state/state_image.h:34` shows the conv state is part of the state image,
  and verify's `RecordForReplay` advances it over the whole window (K+1 columns).
- Conclusion: `restore + fold` only restores/replays the linear-attention region; the conv region is reverted by the restore but nobody advances it => the conv lags,
  which exactly explains "restore+fold first divergence 10 is worse than the non-restoring skip-fold 25"; with skip-fold holding constant, linear+conv both go by
  K+1, the full-accept rounds happen to be correct, diverging only at the first partial acceptance (≈token 25).
- Correct fix (one of two):
  (i) Extend the recording: let RecordForReplay also record the conv transition and replay_fold also replay the conv region (changing ops + the records structure, generic but heavy work);
  (ii) **Recommended**: after restore+fold, rebuild the conv state from the last `conv_width` **committed** tokens (equivalent to prefill/`state_passing`'s conv priming logic),
       costing only a few tokens and not depending on the recording mechanism.
- Judgment: after the change, run `r36h_trace.sh`, requiring the first divergence index to move sharply later (ideally no divergence within 400 tokens), then the 6-sample statistics.
 ## Round 37 — the long-task objective (the user authorized autonomous progress, no report until done)
 Objective: fix the MTP consistency -> benchmarking -> prefill/decode optimization -> VRAM optimization + KV fp8/q8_0 quantization -> open up
 quality-neutral parameters to support larger context -> deliver usable configs and docs for "dual 5060Ti TP2 extreme optimization".
 ### W1 MTP consistency (criterion: the MTP and plain greedy streams agree token-by-token + 6-sample content on plain's order, no 0 repetition)
 - Root cause confirmed: RecordForReplay advances the live GDN state per window (K+1 columns), and the old code folded again => double advance (first divergence 4).
 - Implemented: kRoundScratchSlot snapshot + restore before the fold (first divergence 4 -> 10, still inconsistent; MTP content still 0).
 - To check: (c1) whether the conv/short-convolution state is inside the record plane (if not => the state lags, consistent with 10 being worse than skip-fold's 25);
   read-only inspection of decoder_state.cpp's linear-attention state/record-plane layout and the records' layer/column coverage.
   (c2) the commit_columns column-start off-by-one. (c3) Fallback: write the live state to scratch without touching it, or UpdateInPlace column by column (correct but slow).
 - Afterwards remove the instrumentation NINFER_TP2_TOKEN_TRACE / NINFER_TP2_SKIP_FOLD and rerun the 6-statistic A/B.
 ### W2 MTP performance/load balancing
- Current GPU0 100% / GPU1 ~72% (the MTP layer + argmax/accept/fold/sampling/policy only on shard A).
- ① lockstep redundant MTP-layer runs on both cards (balanced, wall-clock unchanged) ② split the MTP proposal lm_head + window argmax by vocab (truly shortens
  the critical path) ③ rescan draft-tokens K=1/2/3.
 ### W3 KV cache quantization (fp8/q8_0)
- First check existing capability (--help / docs / the plan_cache dtype path); if supported, measure quality+VRAM+throughput; if not, assess the implementation cost.
 ### W4 VRAM and context
- Quantify each card's VRAM composition (weights/KV/workspace/state arena ((2+kReuseSnapshotCount) copies)/prompt cache slot/page size),
  tune each parameter and record the impact on max-context, giving the recommended --max-context and concurrency.
 ### W5 Benchmark and delivery
- The benchmark matrix (prefill/decode tok/s × context length × MTP on/off × KV quantization on/off, recording commands and hardware).
- Delivery: update README/docs (TP-2 + MTP + quantization + recommended parameters), clean up temporary scripts and instrumentation, and remove PLAN.md at the close.
 ### Round 36j — the snapshot+fold fix: partial improvement (first divergence 4 -> 10), not yet consistent
- Implementation: the header gained `kRoundScratchSlot` (snapshot array `kReuseSnapshotCount + 1`, Arena `(2 + kReuseSnapshotCount)`);
  each round before verify `snapshot_state(a/b, kRoundScratchSlot)`; before the fold, the scratch is copied back to the live state, then the fold
  (`source_state_slot = destination_state_slot = 0, commit_columns = committed`).
- Result (same greedy token trace): the first divergence index 4 (old: the fold stacked on a state verify had already advanced) **-> 10**;
  the old "skip the fold" diagnosis was 25. => the direction is right but the real state is not restored: the fold replay still differs from the real per-column advance.
  Candidate causes: ① whether the conv/short-convolution state is inside the record plane; ② `commit_columns`'s column-start offset 1;
  ③ the alignment of record columns and window tokens. MTP content still 0 (reason 425/493) => not done.
- Next step (bisection on the same harness): (a) restore + skip the fold (verify the restore itself doesn't eat the state);
  (b) restore + the fold with `commit_columns = committed - 1`; (c) include the conv state in the snapshot/replay and retest.
- The criterion is unchanged: the MTP and plain greedy streams agree token-by-token, and the 6-sample content is on plain's order, no 0 repetition.- Supplement (verified): `GdnStateAction` at `src/models/qwen3_5/execution/text.h:32-35` has only the two values `UpdateInPlace` and
  `RecordForReplay`, i.e. `RecordForReplay`'s intent is "record the transition + advance the live state as usual",
  so the correct approach must replay from a **pre-verify snapshot**, not fold again on an already-advanced state:
  the fold row should be `source_state_slot = <scratch>, destination_state_slot = 0, commit_columns = committed`.
- The scratch slot must be a **new** slot (it cannot reuse the `kReuseSnapshotCount` ones: they save the prefill-boundary state for cross-request
  reuse, and decode rounds writing them would break the cache); so the per-card GDN state slot count must be +1 (at the plan_cache / state allocation site).
- Implementation steps: (1) state slot +1; (2) each round before verify, copy slot0 into the scratch; (3) the fold uses source=scratch, dest=0;
  (4) verify with `r36h_trace.sh` that the MTP and plain greedy streams agree **token-by-token** (criterion); (5) remove NINFER_TP2_SKIP_FOLD and
  the token-trace temporary instrumentation; (6) rerun the 6-statistic A/B (criterion: the content median on plain's order, no 0 repetition).
- GPU load observation (user): with MTP on, GPU0 100%, GPU1 ~72%. Reason: `mtp_propose_window` runs only on shard A
  (K+1 MTP layer forwards per round) + all the serial argmax/accept/fold/sampling/policy work on A; B idles after the text mixer lockstep.
  Structurally normal; optimization directions: ① lockstep redundant runs of the same MTP layer on both cards (looks better utilized but not faster);
  ② split the MTP proposal's lm_head and window argmax by vocab (half per card + a small exchange) = truly shortening the critical path.
  Recommend fixing correctness before discussing load splitting (current MTP output is wrong; the performance numbers are meaningless).
 ### Round 36i — root cause confirmed: verify advances the live GDN state + the fold advances it a second time (double advance)
- Differential experiment: `NINFER_TP2_SKIP_FOLD=1` (skip `replay_fold->execute`, everything else unchanged) rerunning the same greedy trace:
  the first divergence index 4 -> **25** (the agreeing prefix lengthened 6-fold).
- Mechanism: `tp2_generation_core.cpp:812-820` runs verify's
  `forward_tp2_prefill` with `set_gdn_state_action(RecordForReplay, &records)`; that path, besides recording the transition, **also writes the window's (K+1 columns) state into the live slot 0**, and then 875-883 fold
  slot 0 with `commit_columns = committed` => the state advances twice (round 1 with full acceptance, 6 columns instead of 3).
  Round 1's tokens are computed before the state is polluted, so the first 3 tokens are right and everything from round 2 is wrong — matching the index=4 first divergence exactly.
  With the fold skipped, the state advances only the window's K+1 columns; the full-accept rounds happen to be correct, wrong only at the first partial acceptance (committed<K+1) => index 25.
- Impact: after the state misalignment the model's thinking runs away/repeats `0`/produces no body text; this is the direct cause of the MTP route's 6/6 degradation (not the sampling parameters, not the token domain).
- Correct fix (next step to implement): each round before verify, snapshot the live state into a scratch slot; after acceptance, run the fold with
  `source_state_slot = scratch, destination_state_slot = 0, commit_columns = committed`,
  so the state is rebuilt only from "committed columns", not depending on whether verify incidentally wrote the live state (one slot per card; the slot capacity needs to be extended accordingly).
- Temporary instrumentation (removed once settled): NINFER_TP2_TOKEN_TRACE (committed-point token tracking), NINFER_TP2_SKIP_FOLD (skip the fold).
- Service state: 8088 = plain (MTP unusable until fixed).
 ### Round 36h — greedy token-stream token-by-token comparison: locating the first divergence
- Method: temporarily added env-gated token tracking at the shared commit point (NINFER_TP2_TOKEN_TRACE=1, printing the plain/mtp absolute sequence position and token id committed per round), the same prompt, temperature 0, one run per route, compared per position.
- Result (index = the generated absolute position):
  plain: 97913 97237 99986 101729 4960 130621 16 ...
  mtp  : 97913 97237 99986 1710   97995 4960  130621 ...
  The first 3 tokens are identical (round 1 of MTP correct); **first divergence index=4 (round 2 of MTP)**;
  MTP committed two tokens at positions 4/5 that the target greedy would not pick; thereafter the stream is 2 positions offset from plain.
- Conclusion: not a numeric butterfly effect (1710 and 101729 are unrelated); the target logits/state are wrong from round 2 on; round 1 uses
  the prefill state and is correct => the suspicion concentrates on **the state advance after round 1's fold** (the record plane's column start / the commit_columns
  multi-column commit semantics / the recording-column vs window-column correspondence).
- Next step (decisive differential): in the MTP rounds force each round to commit only 1 token (fold only 1 column, commit=1) and rerun the same trace:
  if it is token-by-token equal to plain => the single-column path is correct, the defect is the multi-column fold; if it still diverges => verify's column 0/state read itself is wrong.
- Temporary instrumentation: NINFER_TP2_TOKEN_TRACE-gated printing at the commit point in tp2_generation_core.cpp (removed once settled).
- Service state: 8088 = MTP K=2 (temp0.7/top-k20/top-p0.80).
 ### Round 36g — MTP route output degradation: location and the domain experiment
- Statistic A/B (same prompt, temp0.7/top-k20/top-p0.80, 6 each): MTP content median 0 (6/6 degraded, reason 108-2611);
  plain median 1443 (1233/763/1469/1456/1430/1621, 0/6 degraded). Greedy (temperature 0) reproduces it too:
  MTP 900 tokens all thinking content=0; plain reasoning 177 + content 1524. => not sampling variance, not the prompt, not the sampling parameters.
- Token domain experiment: the MTP verification's argmax/accept originally used public_token_count(248077), inconsistent with the TP-2 plain route's vocab(248320).
  After changing to vocab: 2 of 3 dom runs normal (content 639/415); 1 of 4 dom2 runs normal (458) => 3/7 vs 0/6, weakly positive but insufficient.
  Decision: keep vocab (each TP-2 internal route must be on the same distribution); record the "single-device uses public_token_count, TP-2 plain uses vocab"
  domain inconsistency as a TODO: it needs unifying, and changing the plain route must be verified separately.
- Refined hypothesis: plain and MTP's greedy streams diverge only after ~79 characters (≈30 tokens) => not a GDN state misalignment (which would diverge within 1-2 tokens),
  but verify going through the Phase::Prefill T=K+1 chunk kernel, numerically inconsistent with decode's T=1 kernel, the butterfly effect amplifying into thinking runaway.
- Next step: implement a Verify phase for TP-2 numerically equivalent to decode (corresponding to the 36c leftover), or per-column decode recomputation as an oracle, locating the first divergent column.
- Service state: 8088 = MTP K=2 (temp0.7/top-k20/top-p0.80).
 ### Round 36f — MTP route output deviating from the target model (new finding, unfixed)
- User's real test (DSH GUI -> 8088 MTP K=2): the same prompt 3 hours ago (no MTP) was a complete 1000-word article;
  now there are "人工智能的000" (AI's 000), "精准医疗的000" (precision medicine's 000) and hundreds of characters of 0-consecutive segments (token 15 = "0" repetition),
  stopped by the user at 17s/969 tokens. The server log shows `[mtp] round ... anchors=15 accepted=3` repeating at the request's tail.
- Controlled A/B (same prompt, same max_tokens, same sampling parameters, 2 each):
  MTP K=2 -> 278/140 tokens, content 360/0, finish=stop; plain -> 705/1387 tokens, content 1241/1081 (complete articles).
- **The decisive greedy A/B (temperature=0, excluding sampling variance)**:
  MTP K=2 -> 900 tokens all reasoning, content=0 (thinking runaway); plain -> 900 tokens, reasoning 177,
  content 1524 (complete article). => the token stream the MTP route commits is inconsistent with the target model, a real defect (not a sampling-parameter problem).
- Hypotheses (to verify, highest priority first):
  1. verify goes through the Phase::Prefill chunk kernel + RecordForReplay, while decode/plain goes through T=1 forward_tp2;
     the per-column target logits of the window (especially the bonus columns at col>=1) may be inconsistent with the true causal advance (state rolling/phase semantics).
  2. The accept op's current_extents / round_lengths (=mtp_position) / round_anchors semantics bias,
     causing a wrong accepted-prefix selection.
  3. The fold's commit_columns and the record plane's column start (whether the anchor column is counted) mismatch, causing a GDN state misalignment.
- Next step (the decisive judgment experiment): for the same window, compare verify's argmax (window_logits) column by column with the plain path's
  token-by-token decode argmax; the first inconsistent column is the location point.
- Current service state: 8088 switched back to plain (no MTP); MTP is not recommended for testing before the output quality is fixed.

### Round 49 — W1 closed: the root cause is that the shards never wrote the replay records (fixed and verified)
- Method: against a local llama.cpp fork's Qwen3.5 MTP (`src/models/qwen35.cpp`'s `graph_mtp`, `common/speculative.cpp:1324`'s
  `common_speculative_impl_draft_mtp`, the snapshot rewind at `src/llama-memory-recurrent.cpp:193-203`) to set the criterion and structure.
- Gating re-assessment (refuting the 36c/36g phase hypothesis): `Phase` has **no semantics** on head-split shards — `attn_mix`'s phase formal parameter is unused
  (`text.cpp:871-989`), both `ph == Phase::Verify` sites in `gdn_mix` are excluded by `shard_config_ == nullptr`
  (`shard_config_` hits only at 1024/1069/1159 across the file), `mixer_layer`/`mlp_layer`/`tp_mlp_delta`'s `prefill`
  is used only for the NVTX name and the exception text ⇒ changing the phase changes no numerics; 36c's "decode-equivalent Verify phase" plan is void.
- Criterion re-assessment: llama.cpp's verify is also not a decode-equivalent kernel (`n_seq_tokens > 1` goes through chunked `GDN_CH`, `==1` through `GDN_AR`,
  `delta-net-base.cpp:433-446`); an accepted token's KV is written by that window and never revised (server-context.cpp:3992-3998 only trims the tail);
  the bonus token comes from the same K+1 verify (sampling.cpp:697-702); docs/speculative.md:213 explicitly says "use greedy when exact agreement is needed".
  ⇒ "token-by-token agreement with plain" is an unreachable criterion, replaced by "no degradation + same quality tier + speedup".
- **Root cause**: TP-2's head-split shards never produce replay records. Both `gdn_mix` record branches are conditioned on `shard_config_ == nullptr`
  (the fused width>1 record op only registers the full 16384-row fused parent, and the workspace query at `gdn_input_proj.cpp:860-864`
  is exactly the `unsupported single-parent profile` 36c reported), while `tp2_generation_core.cpp:872-886` still runs every round
  "restore the pre-round snapshot + `gdn_replay_fold(commit_columns=committed)`", and the fold consumes a **never-written record plane** ⇒
  the 48 linear-attention layers are replayed from an empty log each round (the recurrent doesn't advance, the conv history is rebuilt from the same empty log); the 16 full-attention layers are fine
  => the first few tokens are coherent, then "0" repetition. Matches the 36f/36g phenomenon.
- **Fix 1** (`gdn_mix` in `text.cpp`): the shard records on its own decomposed route — the conv record = the already-materialized `[q|k|v]`
  raw conv input of that route (copied directly; the channel order and layout match the record plane); the recurrent switches to `ops::gated_delta_net_replay_record`
  (`replay.cpp:107-110` explicitly registers the shard geometry `qk_heads=8, value_heads=24`; the contract specifies its output is bitwise-identical to the normalized `gated_delta_net`) ⇒
  the window logits/acceptance decision unchanged; only the state transition becomes replayable. Also completed the row-profile docs of `causal_conv1d_silu.h`
  (the `(1024,1024,3072)/C=5120` implementation supported it long ago).
- Byproduct (avoiding a wrong turn): `src/ops/launcher/rope.cu:189` is a `CUDA_CHECK(cudaGetLastError())` checkpoint, not the failing kernel;
  under multi-device interleaving the error point need not equal the fault kernel ⇒ 36c's "rope dimension contract → engine-level rank-3→rank-4 rework" premise doesn't hold.
- Verification 1: end-to-end A/B (`r36f_stats.sh`, same prompt, temp0.7/top-k20/top-p0.80, 6 each) MTP content
  1519/1318/1294/1195/1683/0 (median 1306, 6/6 `longest0=0`) vs plain 1165/1452/1546/672/224/1237 (median 1201);
  throughput MTP K=2 steady 44.3-49.2 vs plain 31.6-31.8 tok/s (no regression). The one 0 is a reasoning-budget artifact; plain shows the same.
- **Fix 2** (a newly discovered same-family defect): the small-T kernels in `small_t_fp8.cu`/`small_t_k8v4.cu`/`small_t.cu` also need the >48 KiB
  dynamic smem opt-in but used a **process-level `static`**. Single-token decode uses a 32 KiB tile (below the 48 KiB default), so the defect lay dormant;
  the MTP window T=K+1 switches to a 64 KiB tile, exposing it ⇒ every first request of MTP + quantized KV hit `small_t_fp8.cu:56 cudaErrorInvalidValue`
  (kCausalHeadDim=256: 4*64*256=64 KiB). The four paths unified to the per-(kernel, device) opt-in in `src/ops/common/cuda_smem.h`.
- Verification 2: fp8 KV @131072 + MTP K=2 three-sample content 1530/1481/1708, `longest0` all 0, finish=stop (`r49_fp8mtp_smoke.sh`).
- All regression tests PASS: `gated_delta_net_replay_record`, `gdn_replay_fold` (including the 48×8×24×5120 shard geometry),
  `gdn_input_proj_conv_record`, `gdn_input_proj_conv_snapshot`, `gdn_input_proj`, `tp_device_pair`,
  `softmax_attention`, `kv_cache_append`, `context_kv_materialize`, `sliding_window_attention`,
  `qwen3_5_tp2_load/forward --artifact`; `BUILD_EXIT=0`.
- Service state: 8088 = the recommended config (fp8 KV @131072 + MTP K=2 + temp0.7/top-k20/top-p0.80).

## Round 50 — the proposal head A/B: `--lm-head-draft`

- Origin: `--lm-head-draft` swaps the draft-side output head from the full head (248320-row Q8_G32_FP16) weight-tied with the main head to
  the artifact's indexed proposal head (131072-row frequency short table, Q4_G64_FP16, with a `proposal/token_ids` row→real-id mapping;
  at load time `load.cpp:83-84` directly replaces `mtp/output_head` / `draft/output_head`). Verify still uses the full head,
  so it only changes "what is proposed", not the final output tokens.
- Scripts: `r50_proposal_head_ab.sh` (Chinese long-form), `r50b_code_ab.sh` (Python code). The two routes' configs match:
  `--devices 0,1 --kv-dtype fp8 --max-context 131072 --kv-capacity auto --temperature 0.7 --top-k 20 --top-p 0.80
  --spec mtp --draft-tokens 2`, 6 samples each, serial single requests; only the opt route adds `--lm-head-draft`.
- Chinese long-form: full 45.18 tok/s (43.20-52.25), acceptance 2780/5118=54.3%, content median 1294; opt **50.10** (47.53-55.35),
  2749/5136=53.5%, median 1336; both routes `longest0=0`, no degradation ⇒ throughput +10.9%, acceptance −0.8pp.
- Python code: full 48.59 (45.62-51.07), acceptance 4532/7730=58.6%; opt **49.58** (48.85-54.87), 4326/8142=53.1%;
  both routes finish=length (the 1400-token budget all spent in reasoning), content not comparable ⇒ throughput +2.0%, acceptance −5.5pp.
- VRAM: opt +340 MiB/card (14600→14940, 14338→14678), matching 131072×5120 Q4_G64_FP16 ≈ 0.33 GiB —
  confirming `proposal/head` / `proposal/token_ids` don't start with `text/`, `mtp/`, so the name reverse-lookup table at `tp_split_spec.cpp:51` doesn't include them,
  and they fall into the default branch `Replicated`: each card stores a copy.
- Conclusion: the short table's omissions + the indexed head's Q4 quantization both push its argmax away from the full head (prose −0.8pp, code −5.5pp); the acceptance rate drops by domain;
  but the cheaper head reads still outweigh the loss (prose +10.9%, code +2.0%) ⇒ a throughput tradeoff, not a free lunch. 8088 adopts `--lm-head-draft`
  (the main use is prose/chat); the rollback only removes one flag.
- Criterion pitfall: the TP-2 path's response `timings.draft_n` is always 0; the acceptance rate can only come from the serve log's cumulative counter
  `[mtp] round pos=… accepted=… rate=accepted/drafted` (drafted is exactly rounds×K).
  (Later fix: TP-2 now fills `SpeculativeStats`; the response carries `draft_n`/`draft_n_accepted` directly.)

## Round 51 — the 14.9 GiB/card VRAM composition breakdown

- Method: controlled configuration sweeps (after each startup only `nvidia-smi memory.used` is read, no requests), attribute by the differences between configs; then compute the per-card weight bytes of the 23.7 GB
  artifact with `tools/artifact/reader` + `tp_split_spec`'s split rules. Everything reconciles (error ≤2 MiB).
- Measured (MiB, shard0/shard1): 2048 no-speculation 11872/11872; 2048+MTP K2 12314/12306; 131072 no-speculation 13904/13904;
  131072+MTP K2 14600/14338; 131072+MTP K2+`--lm-head-draft` 14940/14678.
- Config facts (from the artifact config): hidden 5120, vocab 248320, 64 layers (16 full + 48 linear), attn 24 heads /
  **4 KV heads / head_dim 256**, GDN key 16×128, value 48×128, conv kernel 4.
- Key derivation: the text KV = 16 layers × 2 KV heads/card × 256 × K+V × 1 B = **16 KiB/token/card** ⇒ 131072 is exactly 2 GiB/card
  (bf16 needs 4 GiB, which is why it can only reach 65536); the MTP layer KV = 4 heads (no head-split) × 256 × 2 × 1 B
  = 2 KiB/token ⇒ 256 MiB, and only on shard 0 ⇒ this is the sole main reason shard0 is bigger than shard1 (measured difference 262 MiB).
  GDN state = 48 × (128×128×24×4 + 5120×3×2) = 73.4 MiB; arena ×5 = 367 MiB.
- Per-card ledger: sharded text weights 8480 + replicated text weights 2472 (embedding 1213 + lm_head 1213 + norms 46) + MTP weights 430
  + proposal head 341 + text KV 2064 + GDN state 367 + workspace 384 + CUDA context/alignment 137 + MTP KV 262 (shard0 only)
  + replay records ~2 ⇒ 14939/14677, matching the measured 14940/14678.
- Optimizable points (by gain): ① the embedding and output head are each 1213 MiB duplicated on both cards (4.8 GiB combined);
  a vocab-dimension split + one gather saves ~1.2 GiB/card; ② 3 of the state arena's 5 copies are prefix-reuse snapshots and 1 is the round scratch;
  lowering only the snapshot count saves ~220 MiB/card; ③ the workspace is fixed 384 MiB/card, while the single-layer prefill peak is ~140 MiB.
- Incidentally confirmed: this machine's nvidia-smi shows 3 cards; index 1 is a Tesla T10 (0 MiB, unused); the two 5060 Ti are index 0 and 2.

## Round 52 — 262,144 context: the vocab/hidden parallel split + VRAM reallocation

- Objective: the artifact's `max_position_embeddings` = 262,144 tokens. Round 51's ledger gave three directions (vocab parallelism,
  snapshot count, workspace); this round lands all of them and verifies each.
- **Weight splitting** (`tp_split_spec` / `tp_shard_views` / `weight_splitter` / `execution/text.cpp`):
  - `text/output_head` (248320×5120 FP8 RowScale) and `proposal/head` (131072×5120 Q4_G64_FP16) are split by **vocab rows**;
    `text/token_embedding` (same FP8 RowScale) is split by **hidden columns** (not by vocab: ids stay full-domain, ruling out out-of-range rows).
  - Each card computes its own half block, then `write_row_block` + one in-place allreduce merges them back to `[V,T]` before sampling/argmax;
    the merge = two disjoint row blocks + the other half zeroed and summed ⇒ bitwise-exact by construction; sampling and `speculative_accept_greedy_drafts` unchanged.
  - Pits stepped in: `tp_shard_views`'s ColumnParallel must use the shard-local offset (`begin=0, end=(n/2)*k`); the FP8 slice's
    `scale_ne[0]`, `scale_nb[1..3]`, `group`/`group_size` must follow the new row count (otherwise `invalid FP8 weight`);
    at the merge, the peer's half block must be written to `peer.shard_index_*` (writing into your own copy shifts the argmax wholesale by V/2=124160);
    the ids broadcast must be 16-byte padded (allreduce requires %16==0), using the padded buffer's prefix view to keep `ne[0]==ids.ne[0]`.
  - Q4 is an **enumerated (n,k) table** (`q4_dispatch.cpp`); there is no 65536×5120 entry ⇒ added `select_q4_n65536_k5120`
    (isomorphic to the 131072 entry: `gemv_r4_w1_direct` / `ksplit<65536,5120,4|8>` / `mma_r64_c128`).
  - The FP8 embedding gather originally only accepted hidden=5120 ⇒ the kernel templated on D + the `embed_gather_fp8_supports_width` query;
    the `[248320,2560]` domain added.
  - `proposal/head`, `proposal/token_ids` don't start with `text/`, `mtp/` — exactly why Round 50 saw them fall to `Replicated`;
    this round brings `proposal/` into the name table and adds `TpSplitOptions::split_proposal_head`.
- **VRAM reallocation**: prefix-reuse snapshots 3→2 (state arena 367→294 MiB: the chat template's round gaps only need the near rewind;
  measured 16k/65k same-prompt re-entry 0.30/0.33 s); workspace 384→192 MiB (the single-layer prefill peak is ~140 MiB;
  overruns are a reported arena overflow, not an out-of-bounds).
- **Result**: 262144 starts normally; `nvidia-smi` shows **15614 / 15094 MiB** per card (697 / 1217 free); the ledger shard0
  `weights+ctx 11494.6 | kv 4644.2 | state 293.6 | record 2.6 | workspace 192.0`. Prefill 16057 tokens / 10.11 s
  = **1588 tok/s**; decode 512 tokens / 8.68 s = **59.0 tok/s** (223 rounds, 2.3 tokens/round, 64.6% acceptance per draft),
  about 20% above the 46–49 tok/s of the 131072 era (the extrapolation step moved to both cards in parallel); re-entering the same long prompt costs 0.30 s.
- **Bitwise-identical proof**: 5 greedy prompts hash identically at 131072 (pre-optimization `ab-control`) and 262144 (`ab-final`):
  `c8540bca7a3c20de` / `589e247eccd554b7` / `b83495f177889c2a` / `7199385e4eb468b0` / `cc73c73125c6f621`.
- **A dead end worth recording**: dropping the MTP layer KV to nvfp4 saves 228 MiB with an almost unchanged acceptance rate (392/510 vs 391/510),
  but 3 of 5 greedy prompts produce different output — not precision leaking into the output, but the fact that verify+fold and single-token decoding are different execution shapes:
  **change the draft mode and the trajectory of near-ties changes**. In the end the fp8 MTP KV was kept, buying the "bitwise identical" property.
- All tests green: `tp2_load` (plain / `--spec mtp` / `--spec mtp --lm-head-draft`), `tp2_forward`, `embedding`,
  `linear_tp2_split_fp8_head`, the new `linear_tp2_split_grouped_head` (**real** [131072,5120], T=1/2 compared bit-for-bit against the full Op),
  `linear_tp2_split_nvfp4`, `tp_device_pair`.

## Round 11 — (1) exact-batch CUDA Graph on TP-2 verify (and fixing the end-of-round cross-card race)

- Goal: items (1) (exact-batch CUDA graph) + (3) (AR/compute overlap) + (4) (sinking the MTP draft chain) of PLAN §12 Round 11,
  not (2) (NVFP4 weights, a deployment-time variant). This round completed **(1)**.
- **Baseline measurement** (new env-gated phase timing `NINFER_TP2_TIMING=1`, 7 CUDA events + host clock):
  `decode rounds=52 committed=127 avg_round=38.90ms mtp=3.70 verify=34.23 accept=0.08 copy=0.09 sync_wait=0.01 fold=0.00`.
  That is verify 88%, the MTP chain 3.70 ms next; the measured launch gap is 4.4 ms, larger than the 2.91 ms estimate at PLAN:642.
- **Implementation** (`src/core/decode_graph.{h,cpp}`, `src/core/arena.{h,cu}`, `src/models/qwen3_5/execution/text.{h,cpp}`,
  `src/runtime/engine/tp2_generation_core.{h,cpp}`):
  - `DecodeGraphDefinition::capture_group(defs, streams, body)`: **capture on multiple streams simultaneously**, one graph per card; cross-card communication is still
    only the in-kernel AR spin with no host involvement, so the AR protocol needs no change (only the epoch token moves from a kernel argument to a device-side counter).
  - `TextContext::forward_tp2_window(...)`: a capture-safe windowed forward. Each round ids/positions go through a **portable pinned** buffer
    (captured as a memcpy node, re-read on every replay), and the attention envelope is frozen as a capture-time constant.
  - Buckets come from `mtp_graph_profiles(max_context, mtp_drafts_)`: 8 buckets at K=2 / 131072, with the bucket selected each round from the visible upper bound.
  - **Reproducible arena addresses** are a hard prerequisite for capturing: new `DeviceArena::rewind(watermark)` + `position_arena(arena, floor, target)`,
    discarding the proposal chain's occupancy each round back to a fixed watermark; a captured bucket whose watermark is below the current request counts as uncaptured and is re-captured
    (the startup warm-up request has a smaller watermark, so the first real request re-captures once).
  - `NINFER_TP2_VERIFY_GRAPH=0` turns it off, on by default; the eager path and the captured path share **the same** `forward_tp2_window` and
    **the same** bucket envelope, so the A/B differs only in the launch mechanism (which is also the source of confidence in "bitwise identical").
- **Results** (same bench recipe, graph off → on): decode on a short prompt 62.8 → **71.5 tok/s** (+13.9%), 2048 context
  64.1 → **72.4 tok/s** (+13.0%), round time 38.90 → **34.47 ms**, of which verify 34.23 → 29.80 ms (the difference is entirely launch/scheduling;
  the two paths have exactly the same kernel arguments and inputs); prefill 512 unchanged (1312 → 1334 tok/s).
- **Correctness**: greedy A/B over 5 prompts × up to 256 tokens is **byte-identical** between eager and graph (including reasoning_content);
  the graph path is insensitive to instrumentation and reproducible across sessions.
- **A real race fixed along the way (present before this change)**: at first eager and graph disagreed on 3 of 5 prompts, while **any**
  extra synchronization/read-back (even added at the end of a round) made eager agree with graph ⇒ the eager path is timing-sensitive. Root cause: at the end of a decode round
  only `cudaStreamSynchronize(shard_a_.device.stream)` was called, and the host immediately went on to fold / state restore / the next round's window and
  arena reuse, while shard B's verify tail could still be running — the **only** ordering guarantee between the two cards is the AR spin, so there is no ordering at all
  between the peer's tail and the host's next round. The eager path had long masked this by "thousands of launches per round keeping the host behind the GPU"; after switching to graph
  replay the host overtook the GPU and it surfaced reliably. Fix: synchronize both A and B at the end of the round (measured `sync_wait` is still 0.01 ms).
  After the fix eager and graph are byte-identical on all 5 prompts.
- **MTP output quality** (not introduced by this round): compared greedily against the `-Plain` single-token path, only 1 of 5 prompts is byte-identical, the rest flip at near-ties —
  consistent with the property recorded in Round 52 (the verify window and single-token decoding are different execution shapes), not a regression from (1).
- Not done: (4) (de-hosting the draft chain), (3) (prefill AR∥MMA sub-block pipeline).
- **A second defect fixed along the way (a real crash, not introduced by (1))**: in the full test run `ninfer_qwen3_5_tp2_forward_test` reported
  `cudaErrorIllegalAddress` inside the first layer, while HEAD passed with the same binary. Bisection: forcing allreduce through host staging makes it pass ⇒ it was introduced by this round's
  "device-side token" in-kernel AR. Root cause: the token counter is allocated on each device's own device heap, but which shard drives the pair is decided by the caller
  (that test has both shards drive once each, so `stream_a` belongs to device 1), and the kernel dereferenced device 0's raw pointer inside device 1's context.
  Fix: move the token to **mapped pinned host memory** (visible to both devices), with the kernel reading it once per block and broadcasting through shared memory — reading system memory per thread
  would add 4 us per allreduce and 0.5 ms per round. After the fix all 12 related tests are green (including tp2_load / tp2_forward), and eager/graph are still byte-identical.
- **Final measurement (same binary after both fixes, graph off/on)**: avg_round 39.6 → **35.3 ms**, of which verify 35.0 → **30.6 ms**;
  2048-context decode while waiting for output 65.6 → **76.5 tok/s** (+16.6%); prefill_512 is 1419 tok/s on both sides.
- **(4) re-review conclusion (this round's conclusion, do not redo)**: PLAN's description of (4) ("one `cudaStreamSynchronize` per draft", the old
  `tp2_generation_core.cpp:543`) no longer holds in the current code: `mtp_propose_window` has only **1 D2H + 1 sync** for the whole window,
  and per draft only a D2D copy + an increment kernel remain; `mtp_forward_batch` / `mtp_forward_ar_step` / `mtp_forward_core`
  have no host synchronization and no D2H internally (verified by grep). The measured `mtp=3.70~3.75 ms` is real GPU work (two T=1 MTP-layer forwards, weight-bandwidth bound),
  and it runs serially with verify (same stream pair). The only remaining opportunity is "overlapping fold with the MTP chain" ≤0.8 ms (≈2%; fold goes through `replay.cpp` and contains no allreduce,
  so it could in principle use a separate stream), but that requires changing the AR token protocol and the stream ordering; the benefit/risk ratio is unfavorable ⇒ (4) is judged complete and will not be changed. Next: (3).

---

## Round 12 — (3) prefill AR∥MMA sub-block pipeline: implemented, measured, rejected

(3) is the last of the (1)(3)(4) table in PLAN §12 (overlapping prefill allreduce with compute). This round **fully implemented and measured** it,
and the conclusion is that **it brings no benefit on TP-2 — about 5% slower — so it was rolled back entirely**, leaving only this record.

### Implementation (complete and working, not kept)

- `DeviceContext` gains a second stream `collective_stream` (same lifetime and move/dtor semantics as `stream`/`transfer_stream`).
- `forward_tp2_prefill`, when `NINFER_TP2_PREFILL_OVERLAP` is on, phase=Prefill, `pair.in_kernel_allreduce()`,
  the driving card == `pair.a()`, and tokens ≥ 256, splits the chunk into A=[0,ta), B=[ta,T) and advances the two layer-interleaved:
  A.mixer → arm(0) → B.mixer → arm(1) → AR(A.mixer)/AR(B.mixer) (collective stream) → A.res+A.mlp → B.res+B.mlp
  → A.res2 → B.res2. Each AR gets a pair of events: `produced` recorded on the compute stream and waited on by the collective stream;
  after the AR, `done` is recorded on the collective stream and waited on by the compute stream — the dependency chain L,A → L,B → L+1 is preserved,
  and GDN conv/recurrent state and KV visibility are guaranteed by the compute stream's own ordering.
  The sub-blocks are bound with nested `ScopedPositions`/`ScopedEnvelope`: positions/rope are slices of the outer tensor, and the
  envelopes are {first+ta,first+ta} and {visible_end,visible_end} respectively.
- **Key constraint (a pitfall worth remembering)**: the split point must land on an **activation-scheduling block boundary**. The first version used ta=tokens/2 (300→150+150)
  and made the chunk-split invariant of `ninfer_qwen3_5_tp2_forward_test` fail (`300 -> 128+172` with max_logit_diff=1.14~1.60, because the reference "single block of 300" now internally runs 150+150, landing in a different MMA/tile scheduling class);
  after switching to 64-alignment (`ta = max(64, (tokens/2) & ~63)`, 300→128+172) it is **bitwise identical**:
  `300 -> 128+172: max_logit_diff=0`, `300 -> 64+236: 0`, `T=1024 one chunk vs 4x256: 0`,
  and the top5 for 300 and 1024 are numerically identical to the **unsplit baseline** (`348=22.875 621=14.25 …`).
  That is: **layer-interleaved A/B with dual-stream AR is numerically fully equivalent to sequential single-block execution**, which is (3)'s only certain technical benefit.

### Measurement (2072-token single-request TTFT, same binary, prefix cache miss)

| Configuration | run1 | run2 | run3 |
|---|---|---|---|
| `NINFER_TP2_PREFILL_OVERLAP=0` | 1249 ms | 1189 ms | — |
| `NINFER_TP2_PREFILL_OVERLAP=1` | 1258 ms | 1314 ms | 1513 ms (cold start) |

Steady state ≈1219 ms vs ≈1286 ms ⇒ the sub-block pipeline is **about 5% slower** (the direction is consistent across two independent measurements).

### Why the model was wrong

1. The cost model at PLAN:651 treats AR as a **serial** cost ("AR is 48% of prefill; after overlap ≈2.6k tok/s"). Measurement does not support this:
   the two in-kernel AR kernels are symmetric, so when the two cards advance drifted, this card's spin waiting for the peer is already **hidden by the peer's still-running
   compute**, leaving far less overlap headroom than the model estimated.
2. The cost of sub-block splitting is real: after halving T the GEMM tiles get narrower, and **each layer's weights are streamed twice** (prefill is a weight-bandwidth-sensitive shape); measured,
   these two together outweigh the AR benefit.
3. So the final form of (3) is **single-block prefill without sub-block pipelining**; `collective_stream` and the whole pipeline were rolled back,
   and the working tree is clean, back at (1)'s commit `8f6a812e`. If a genuinely compute-bound prefill appears in the future (wider chunks, a faster weight path),
   this section's 64-alignment constraint and event topology can be reused to re-evaluate.

---

### Round 13 — removing the [mtp] spam print in the decode loop

User report: during inference the cmd window was flooded with `[mtp] round pos=… anchors=… accepted=… rate=…/…`, and `--log-level error` could not turn it off.

- **Root cause**: that line was a bare `std::fprintf(stderr, …)` in `tp2_generation_core.cpp`, **writing straight to stderr and bypassing the logger**,
  with no env gate (the same file's `[tp2-time]` is gated by `NINFER_TP2_TIMING=1`). One line per decode round; at the factory configuration of 59.0 tok/s ÷ 2.3 tokens/round that is
  ≈ **26 lines/second**. It was a leftover from the historical `NINFER_TP2_*` probe cleanup, kept only because `r53_decode_analysis.sh` still parsed it.
- **Measured write cost** (same format line, unbuffered `os.write(2, …)`, 20000 lines): `2>NUL` 1.8 µs/line,
  `2>file` 1.5 µs/line, **a real console 20–25 µs/line** (Windows console writes are synchronous and take a hop through conhost).
  ⇒ 26 lines/s × 20 µs ≈ 0.5 ms/s = **0.05%**; 20 µs per round / 35.3 ms = **0.057%**. The throughput impact is negligible.
- **The real risk is not slowness but stalling**: cmd enables QuickEdit by default, so clicking/selecting in the window stops the console consuming output and
  `fprintf` blocks on the generation thread, stalling all inference until the mouse is released; stderr attached to a slow pipe (`2>&1 | tee`) back-pressures the same way.
- **Change**: delete that `fprintf` and the two counters that served only it, `mtp_draft_checked_`/`mtp_draft_hit_`
  (`tp2_generation_core.{h,cpp}`, cpp −7 lines / h −7 +4 lines). The acceptance data is still available:
  `NINFER_TP2_TIMING=1`'s `[tp2-time] decode rounds=R committed=C` ⇒ accepted drafts = C−R,
  acceptance rate = (C−R)/(R·K), K = `--draft-tokens`.
- The statement in `docs/tp2-dual-5060ti.md` that "the acceptance rate comes from the `[mtp] round` counter" was updated to the formula above.
  `tools/tp_bootstrap/r53_decode_analysis.sh` is an archived script from the Linux era (hard-coding the `$HOME/prof/…` log path) and was
  **left unchanged**: its acceptance column will now degenerate to `no mtp`, and it should read from `[tp2-time]` if needed.

---

### Round 14 — the "continuation boundary" (option 2): implemented, falsified by real traffic, withdrawn

The problem the user exposed in an agent session: in steady state `cache` hits 99% yet TTFT is still high, and `cache 0` appears periodically.
Option 2 was implemented first, "push the reuse boundary to the end of the previous round's generated sequence": a new state slot holds the GDN state at the end of decode,
`cached_tokens_` records prompt + `generated[0..G-2]`, boundary = `prompt + G - 1` (the last generated token is the next round's
anchor, with the transition deliberately left unfolded), +73 MiB per card (`state` 293.6 → 367.0 MiB), plus an `NINFER_TP2_REUSE_TRACE=1` diagnostic.

**Measurement (real DSH agent traffic, 36 requests)**: `slot=0` (the end of the previous prompt) hit 35 times, `slot=1` (rewind) once,
and the **continuation slot 0 times**; `continuation > shared` held for 36/36, of which 33/36 had `shared == prefill_end` **exactly equal** —
the client's next prompt is token-for-token identical to the previous prompt until the end, then diverges at the **first generated token**.
Corroborating: req#1 generated 499 tokens, while req#2's prompt is only 203 tokens longer than req#1's (the rendered assistant turn is clearly shorter than the generated stream).
That is, DSH does not send the model's generated token stream back; it re-renders the assistant turn, so "continuing the previous round's output" does not hold at the protocol layer.

**Decision: withdraw** (`tp2_generation_core.{h,cpp}` back to `1cdfab97`), keeping `NINFER_TP2_REUSE_TRACE` —
it is the only means of judging "does the client send history back verbatim" (printing `prompt/cached/shared/prefill_end/rewind -> reuse/slot` per request),
and the `state` ledger goes back to 293.6 MiB/card.

**Three things located in the same round but not yet done**:

1. The first round after a new session / context compression: req#16's prompt of 63,182 shares only 14,768 with the cache (= system prompt + tool definitions),
   while the cache's two boundaries are both near 106k ⇒ `reuse=0`, TTFT **46.9 s** of full recomputation; about 11 s of that is thrown away for nothing
   (adding one more anchor at the system+tools end position after the prefix end would fix it).
2. Large new content returned by tools: the suffix of req#4/5/6 is 8,966 / 14,634 / 8,895 tokens, and the forward pass runs at only 0.95–1.13k tok/s ⇒
   TTFT 8.0 / 13.7 / 8.9 s, with the cache already 82–90% hit, so what remains is **genuinely new computation** ⇒ this is a prefill-throughput problem, not a cache problem
   (cold start is also only 1.35–1.38k tok/s).
3. ~0.28–0.35 s of fixed overhead per request: an 18-token suffix → TTFT 272 ms, a 519-token suffix → 657 ms (the marginal cost is only 0.38 s);
   this is the floor of TTFT.

**Log convention**: `| prefill X tok/s` is `(prompt − cache)/prefill_seconds`, so on a nearly fully-hit request the numerator is only a dozen tokens and the
denominator is dominated by fixed overhead — the user read `prefill 70.6 tok/s` as "prefill from scratch" on that basis. `operational_log.cpp` now also prints the numerator
(`prefill 70.6 tok/s (18 tok)`).

## Round 15 — tool calls returned as text: A/B shows it is unrelated to quantization

**Symptom**: in agent sessions the model occasionally emits tool calls as body text (`<tool_call> <function=todo_write> …`), and the frontend reports
"the tool call format is wrong"; sometimes it recovers after correction.

**Localization**: ninfer's `WARN req#N tool markup returned as text | <reason>` gives four reason classes
(`invalid tool name` / `undeclared tool` / `malformed structure` / `trailing content`). A verbatim-replay probe
(feeding the markup from the screenshot back to the model as-is, once with the newline form and once with the space form) was correctly parsed into `tool_calls` both times; the namespace
grouping round-trip was also correct (declaring `namespace=dsh` for `todo_write` → the model answered with the template-rendered `dsh__todo_write`, and the response was restored to
`name=todo_write, namespace=dsh`) ⇒ neither the parser nor the chat_template is at fault.

**Root cause (model side)**: the request declares only **1** directly callable function (log `tools 1`; the convention is verified: one namespace wrapping 2
functions logs `tools 2`), while the system prompt lists the whole SDK of tools as `tools.read` / `tools.pwsh` / `tools.todo_write` …. The model sometimes emits structured calls
to those "readable but not directly callable" names, or copies them verbatim as dotted `tools.read`, which `enforce_declared_names` rejects and returns as text unchanged.

**A/B (whether the HF → ninfer conversion caused a regression)**: 3 inducing prompts (todo / read / pwsh) + 1 control,
`temperature 0.7`, two rounds per arm, 75 tool requests in total:

| Model | tool requests ok | markup-class failures | hit output cap |
| --- | --- | --- | --- |
| Official `qwen3_8_27b_nvfp4.ninfer` | 38/75 | 27/75 (21 `tools.*` + 6 malformed) | 10 |
| Converted `qwen3_8_27b_w4a4_w8a8.ninfer` | 50/75 | 19/75 (17 `tools.*`/`read` + 1 malformed + 1 undeclared) | 7 |

The markup-class failure rate is 36% vs 25%, not a significant difference (two-proportion z≈1.4, p≈0.16), and the **failure shape is identical** (both copy dotted names like `tools.read`);
the control prompt is 10/10 on both arms ⇒ the conversion caused no regression in tool-calling ability. The other axis is "thinking too long and hitting the output cap",
which the 300-token probe limit amplifies; a real session at 32k will not trigger it.

**Conclusion**: not a ninfer bug, not the chat_template, not quantization; it is model-side copying under a prompt structure where "the only callable tool is run_code, yet the prompt lists the whole SDK
tool name set". ninfer strictly rejecting undeclared calls is correct behaviour — letting them through would let the model bypass run_code
and trigger pwsh/write directly.

**Reusable probe**: `%TEMP%\ab_probe.ps1` + `%TEMP%\probe_t0..t3.json`,
`pwsh -File %TEMP%\ab_probe.ps1 -Tag <name> -Repeats 15`; the script fetches the id from `/v1/models` itself,
classifies into ok / bad-name / malformed / empty, and exports `%TEMP%\ab_<tag>.csv`.

**Also determined along the way**: when the `bin` directories of ffmpeg/libcurl are missing from PATH, ninfer-serve exits silently with `0xC0000135` (DLL not found) and zero output;
launching it from a tool environment must first add those two directories to PATH (see `tools/win_port/serve.ps1:69`).

---

## Round 16 — TP-2 host-side state checkpoints (implemented, pending end-to-end verification)

**Problem** (two measurements after Round 15): the client re-renders history at turn boundaries, so the new prompt is **shorter** than the previous one and the divergence point lands mid-history.
The TP-2 core has only 2 **device-memory** boundaries (the end of the previous prompt + one rewind point), and the criterion `boundary <= shared_prefix` is satisfied by neither
⇒ `reuse=0` ⇒ the whole prompt is recomputed:

| Request | prompt | cached | shared | reuse | TTFT |
| --- | --- | --- | --- | --- | --- |
| req#24 | 12,143 | 12,362 | 11,872 | 0 | 7.8 s |
| req#43 | 57,378 | 58,611 | 43,239 | 0 | 42.0 s |
| req#51 | 74,536 | 90,424 | 57,376 | 0 | 57.6 s |

For req#51 that segment was confirmed with 250 ms NVML sampling: both cards at 98–100% the whole time, **no "only one card is running" and no idle gap**
(the only thing that stays at 0% in the samples is `gpu1` = Tesla T10, which NInfer does not use); the `done` line's `prefill 1.30k tok/s (74,536 tok)`
agrees with the 57.6 s TTFT. Note that the `throughput` window line's `prefill 14.9k tok/s (74,536 tok)` is a different convention
(tokens settled within the window ÷ the 5.0 s window length, `src/serve/operational_log.cpp:340`), not the real prefill throughput;
the real value is on the `done` line (`operational_log.cpp:274-279`, divided by `prefill_seconds`).

**Root cause**: not the KV. The walk starts at the boundary, and KV pages before the boundary are not recomputed anyway (see the comment in
`tp2_generation_core.cpp`). What is missing is the **GDN (linear attention) state** — a recurrent accumulator that can neither be derived from the KV nor shifted; full-attention layers can be truncated freely by LCP
(llama.cpp's `n_past = slot.prompt.tokens.get_common_prefix(input_tokens)`, `tools/server/server-context.cpp:3103`,
which solves only that half), while a hybrid model must fall back on context checkpoints: llama.cpp's `--ctx-checkpoints` defaults to 32,
`--checkpoint-min-step` defaults to 8192 (`common/common.h:611-615`), created before `llama_decode()`
(`server-context.cpp:3508-3518`), and on restore it searches from newest to oldest for a checkpoint `<= LCP`, only falling back to `do_reset` full recomputation if none is found (`:3236-3249`).

**Implementation** (zero device-memory growth, all checkpoints in pinned host memory):

- Reuses the engine's existing host state-image budget `--host-state-slots` (`ContextCacheOptions::host_state_slots`,
  `include/ninfer/types.h:136`): the TP-2 branch does not touch this budget, and other disabled-cache routes keep zeroing it as before
  (`src/runtime/engine/model_instance.cpp:100`, `:135-142`).
- `tp2_generation_core.h/.cpp`: a `Shard::HostCheckpoint` ring (`PinnedHostBuffer`, one copy = `state_backing.bytes` ≈ 73.4 MiB/card);
  the stride starts at `kReuseCheckpointStride = 8192` and is scaled up to a multiple of 128 by `ceil(max_context / slots)` so that
  `slots × stride ≥ max_context` (default 8 slots / 131072 → 16384; the script configures 16 slots → 8192).
- Every time prefill crosses a stride it does a D2H of the state on the same shard stream, recording the frontier and the prefill id;
  in addition to the original 2 device-memory boundaries, the reuse scan also takes the deepest host checkpoint from the **previous prefill** that is ≤ `shared_prefix`;
  `begin_gdn_state` correspondingly takes an H2D branch; the KV path is completely untouched (prefix pages are kept anyway and the suffix is recomputed as before).
- Correctness rule: only checkpoints produced by the **last completed prefill** are used (marked by prefill id) — its `[0, shared)` is token-for-token
  identical to this request's, the same argument as for the device-memory boundaries; older checkpoints are not guaranteed to hold the same tokens at the same position. A cancelled prefill
  (an early `return` on `cancellation.requested()`) does not publish an id, so the checkpoints it wrote into the ring are naturally excluded.
- The startup ledger gains a line: `[mem] host-checkpoints shard N slots S x 73.4 MiB | stride X tok | pinned Y MiB`;
  the `NINFER_TP2_REUSE_TRACE=1` `[tp2-reuse]` line gains `host=<slots> stride=<stride> ... src=host|device`.

**Verification status**: the `ninfer_engine` target compiles (`cmake --build build-win --target ninfer_engine`, BUILD_EXIT=0), and
`git diff --check` is clean. **No end-to-end verification yet**: it requires relinking `ninfer-serve.exe` and restarting the service, and the service is currently running and holds the exe.

**Verification still to do**: after the restart, start a new session and make the second round's prompt shorter than the first with a mid-history divergence, then confirm the `[tp2-reuse]` line shows
`reuse>0` with `src=host`, the `done` line's `cache` goes from 0% to on the order of `shared`, and TTFT drops from ~57.6 s to
`(prompt − checkpoint) / 1.3k + fixed overhead` (with 16 slots / stride 8192, req#51 is estimated at ~13-15 s).

### Round 16.1 — the first restart had no effect + the tail checkpoint window

**After the user changed `--max-context` to 204800 and restarted, it was still slow. Checking the logs showed the old binary was running.**

| Evidence | Value |
| --- | --- |
| `build-win/apps/ninfer-serve.exe` timestamp | `01:39:37` |
| `tp2_generation_core.cpp` modification time | `02:58:20` |
| `host-checkpoints` lines in the log | 0 |
| `src=host` lines in the log | 0 |
| `[tp2-reuse]` format | old (no `host=/stride=/src=`) |

That is: the context was changed and the service restarted, but the exe was not relinked, so Round 16's fix never ran at all.
Only the **final link** is blocked by the running exe (Windows locks a running image and `link.exe` reports LNK1104);
compilation and `ninfer_engine.lib` were always fine, so the library had the new code while the exe was still old. After stopping the service a full build passed in one pass.

**Hard evidence from the log** (old binary, `NINFER_TP2_REUSE_TRACE=1`):

| Request | prompt | cached | shared | reuse | TTFT | Note |
| --- | --- | --- | --- | --- | --- | --- |
| req#9 | 53,167 | 56,462 | 11,872 | 0 | 39.0 s | only 22% shared |
| req#12 | 53,763 | 53,750 | **53,592** | **0** | **39.0 s** | **only 171 tokens apart, yet fully recomputed** |

req#12's two device-memory boundaries are `prefill_end=53,750` and `rewind=53,746`, both **after the divergence point 53,592**,
so `boundary <= shared_prefix` is not satisfied ⇒ `reuse=0`. The rewind slot stopped at the end−4 because `rewind_near_` is predicted from the **previous pair's**
prompt gap, and req#11's pair had `gap == 0` (a complete hit, so `if (gap != 0)` does not update), while the earlier req#10 pair's gap=2
was clamped to 4 by `kReuseRewindMinimum=4`. **The current gap is only known after computing the LCP, so it is structurally unpredictable** ⇒ checkpoint positions must be independent of prediction.

**New: the tail checkpoint window** (`kReuseTailWindow = 8192`). In addition to the stride grid, a host checkpoint is stored at **the end of every chunk** within the last 8192 tokens of
the walk (matching llama.cpp's `near_prompt_end` exception, which is likewise not subject to `checkpoint-min-step`).
Effect (`--prefill-chunk 1024`, 32 slots / stride 8192): req#12's previous prompt was 53,750, whose chunk ends are 1024·k and 53,750; the
tail window gives the 8 checkpoints 46,080 … **53,248** ⇒ the next request with `shared=53,592` hits 53,248 and only needs to recompute
53,763 − 53,248 = **515 tokens ≈ 0.4 s** (was 39.0 s).

**Honest boundaries**: this only helps when "the shared prefix is long and only the tail was re-rendered". The req#12 type (99.7% shared) → 0.4 s;
the req#51 type (77% shared, previous was 90,424) → hits 57,344, recomputes 17,192 ≈ 13 s (was 57.6 s);
the req#43 type (75% shared) → hits 40,960, recomputes 16,418 ≈ 12 s (was 42 s);
while the req#9 type (only 22% shared; the client dropped 44k of history) hits 8,192 and still has to recompute 44,975 ≈ 35 s —
that 41k suffix is genuinely new content and **must be computed, physically**.

**A reminder about the 200k device-memory headroom** (this restart's `[mem]`):

```
[mem] shard 0 capacity 204800 | ... kv 3628.3 | ... free 176.0 of 16310.6 MiB
[mem] shard 1 capacity 204800 | ... kv 3225.0 | vision 826.5 | free 0.0 of 16310.6 MiB
```

Card B has 0.0 MiB of headroom (the vision tower is also on B), so any extra device-memory allocation (large images, graph capture) could OOM;
host checkpoints occupy only host memory and do not add to this. At 131072 card B still had ~1 GB of headroom.

**Build result**: after stopping ninfer-serve, `cmake --build build-win -j 12` passed in one pass, and
`ninfer-serve.exe` / `ninfer.exe` / `ninfer-perplexity.exe` were all relinked (03:13), so the fix is in the binary.

### Round 16.2 — correcting the validity rule (lineage pruning) + cancel invalidates

**Finding**: Round 16's original rule "only accept checkpoints produced by the **last** prefill" breaks the most common scenario. Look at req#11 →
req#12: req#11's prompt was 53,750 with `reuse=53,594`, so **it only advanced 156 tokens itself** ⇒ that prefill would write out only
one chunk end (and that of a skipped prompt) ⇒ **it would leave no checkpoints at all**; whereas req#10's run left the tail checkpoints 46,080…53,248,
which the "last only" id rule discarded wholesale ⇒ req#12 still had `reuse=0`. In other words, under the old rule this fix was a no-op in the most critical
scenario.

**Fix: lineage validity**. Each checkpoint carries `valid`, with two rules:

1. **Prune before selecting**: any checkpoint with `position > shared_prefix` is set `valid = false`. The argument: a checkpoint's state is "the state that the prefill
   which wrote it had over the first p tokens", and it is the current prompt's state only if the entire lineage (previous prefill → current prompt) agrees on the tokens before p;
   `shared_prefix = lcp(cached_prompt_tokens_, prompt)` is exactly that boundary, so an out-of-bounds checkpoint can **never**
   be used again (a later prompt must match a token at which the current prompt has already diverged). After pruning, take the deepest survivor.
2. **Validity on publication**: once a prefill completes successfully, the slots with `prefill_id == this run's live id` are marked `valid` (the id pins down exactly which slots this run wrote,
   so ring wraparound cannot mis-attribute). A prefill cancelled before finishing never executes this step ⇒ the slots it wrote stay unusable.

With this, after req#11 (which advanced only 156 tokens and produced no checkpoints), req#12 can still use the **tail checkpoint 53,248** left by req#10:
`shared = 53,592` covers it ⇒ only 53,763 − 53,248 = **515 tokens ≈ 0.4 s** to recompute.

**A latent hazard fixed along the way**: when a prefill was cancelled midway the original code returned directly while `cached_state_valid_` was still `true`,
so "the previous prefill's boundary + snapshot" was treated as valid — yet the cancelled run had already rewritten the KV and may have overwritten the rewind slot's snapshot ⇒
the next request could get an inconsistent state/KV. The cancel branch now sets `cached_state_valid_ = false` and clears the whole ring, and the next prefill republishes.

**Verification**: `cmake --build build-win -j 12` passes fully (`ninfer-serve.exe` 03:15:15), `ninfer_engine_options_test` passes, and
`git diff --check` is clean. The trace now also prints the count of valid checkpoints: `[tp2-reuse] … host=<valid>/<slots> stride=… -> reuse=… src=host`.

### Round 16.3 — splitting the checkpoint ring into grid + tail sub-rings (with measurement and verification)

**The motivation came from measurement**: the user session's trace reached `host=29/32` at only 77k context, and req#18 hit `host=32/32`, completely full. The reason is that every
prefill contributes up to 8 tail anchors, and the ring is overwritten in position order ⇒ **the position grid (the only means of covering deep divergences) gets squeezed out step by step**,
and it will necessarily get worse at 200k context.

**Change**:

- `kReuseTailCheckpointCount = 8`; the ring is split into two segments: `[0, grid_slots)` holds the position grid (never overwritten by tail anchors),
  `[grid_slots, size)` holds the tail anchors (refreshed by every prefill). `Shard` gains `host_checkpoint_grid_slots` /
  `host_checkpoint_tail_next`, and the two cursors cycle within their own sub-rings.
- The stride is computed from the **grid slot count**: `max(8192, ceil(max_context / (slots - tail_slots)))` ⇒ with 32 slots / 200k that is
  `ceil(204800/24) = 8534 -> 8576 tok` (covering 24 x 8576 = 205,824 ≥ 204,800 ✓).
- The tail window adapts: `tail_span = min(8192, prefill_chunk x tail_slots)`, so a narrow chunk does not flood the tail sub-ring.
- The validity rules are unchanged (mark valid by prefill id on publication, prune by `shared_prefix` before selecting, invalidate on cancel).

**Verification** (stop service → build → start service, all executed by me):

- After stopping the service `cmake --build build-win -j 12` fully succeeded, `BUILD_EXIT=0`, `ninfer-serve.exe` 03:37:40.
- New ledger: `[mem] host-checkpoints shard 0 slots 32 (grid 24 + tail 8) x 73.4 MiB | stride 8576 tok | pinned 2349.0 MiB` ✓
  (stride 8576 is itself evidence that the ring split took effect; the previous version was 8192).
- probe: `openai-chat` 58 tokens → 203 ms, `cache 44 (75.9%)`, decode 76 tok/s ✓.
- `--preserve-thinking` in effect: `req#1 started | ... | preserve thinking` ✓ (the server writes it into the effective semantics by default).

**Measured benefit in this session (user session)** (against the theoretical minimum = prompt − shared): `src=host` hits on req#5/#11/#12/#19;
each request's actual recomputation is within ≤ 1,022 tokens (one chunk) of the theoretical minimum; against the old binary: req#5 10.2 s → 1.3 s,
req#11 37.6 s → 28.6 s. The remaining time is all "genuinely new tokens brought in by client re-rendering/tool results", which must be computed at 1.0–1.5k tok/s.

**Semantics of `--preserve-thinking`** (for A/B): `chat_template.cpp:611` decides whether thinking from finished turns is preserved;
the value rule for the Responses route is in `openai_responses_state.cpp:159-165`: if the request gives it explicitly, use the request's value (and mark a semantic change);
if not given and there **is a parent record**, inherit the parent value; if not given and **no parent record**, use the server default (`translate.cpp:118`). The user's client sends the
full history each round (no `previous_response_id`), so after a restart it is the server default true. Observation point: whether `shared` still collapses midway.

**Port**: the startup script's `$Port` had been changed to 3456, while the client's `.dsh/settings.yaml:18` points to `8099`; restored to 8099.
From now on any port change must be synchronized on both sides, otherwise the client cannot connect.

## Round 15b — tool-call constrained decoding (route A): implementation, verification and two pre-existing defects (fixed)

Continuing Round 15's diagnosis (not quantization, not chat_template, not the parser, not client configuration; llama.cpp does lazy GBNF constrained decoding,
NInfer has no constraint layer). This round landed "route A: the text-based lightweight version", and along with real traffic fixed two **pre-existing** TP-2 defects.

### 1. Route choice (A/B)

The correctness risk of "masking computed wrong when a tool name spans multiple tokens" exists in llama.cpp too, but its implementation keeps it very low:
`llama_grammar_apply_impl` (`src/llama-grammar.cpp:1354`) decodes each candidate token to text with `token_to_piece`, handles a half UTF-8 across tokens with
`llama_partial_utf8`, and then validates "accumulated text + fragment" against the GBNF parse stack — **constraining by text, insensitive to tokenization**,
with a parser that has long production validation. Building a prefix tree over token sequences would be tokenization-sensitive and prone to mis-masking. Hence two routes:

- **Route A (landed this round)**: a state machine accumulates "generated text" (like llama.cpp's `trigger_buffer`), each step decodes candidate tokens to text fragments with the NInfer
  tokenizer (handling a half UTF-8 across tokens) and validates whether "accumulated text + fragment" is a legal prefix of a declared tool/parameter name
  (accelerated by a trie of declared names), otherwise `-INF`. Insensitive to tokenization, correctness risk aligned with llama.cpp, far less code than a full GBNF.
  Cost: each constrained step has an O(vocab) CPU mask computation (~a few ms), but it only applies in the tool-call region (a few tokens), so amortized it is negligible.
- **Route B (Phase 2 option, not done)**: port `llama-grammar.cpp`'s stack parser (~1500 lines) + generate a grammar for Qwen.
  Lowest correctness risk (production-validated), naturally supports full schema validation. Upgrade to B if A still has occasional boundary problems in real traffic, or if full parameter
  schema validation is wanted.

Not doing "return undeclared calls as tool_call too": that would amount to letting the model bypass `run_code` and trigger `pwsh`/`write` directly, breaking the PTC contract.

### 2. Implementation

- **New op `apply_token_mask`**: `include/ninfer/ops/token_mask.h` + `src/ops/kernel/token_mask.cuh` +
  `src/ops/launcher/token_mask.{h,cu}` + `src/ops/wrapper/token_mask.cpp` (registered in `src/ops/basic_sources.cmake`).
  Logical shape: BF16 `logits[rows, columns]` (dim0 contiguous) with a same-shaped U8 mask, element `(v,c)` at `v + c*rows`; positions where the mask is 0 are written as
  `-inf` (`__float2bfloat16(-CUDART_INF_F)`). Contract: rank-2, matching dtype/shape, `ne[2]/ne[3]==1`, contiguous, non-empty, mask read-only,
  the two sides must not alias. **Called outside the CUDA Graph** (the graph only captures the verify forward; `window_logits` comes out of the graph and is masked afterwards).
- **New frontend module `tool_call_constraint.{h,cpp}`**:
  - `ToolCallMaskTable`: one decoded byte sequence per vocab id plus a special-token bit; held by `Frontend` through a `shared_ptr<const Tokenizer>`
    and **built only once** (`build_tool_call_mask_table(tokenizer)` in the `frontend.cpp` constructor); the table has as many rows as the tokenizer's public vocabulary,
    with invalid rows given an empty fragment + special=1.
  - `ToolCallNameTrie` (256-way), `ToolCallGrammar`, `ToolCallGrammarState` (a byte-level state machine with modes
    `Free / FunctionLiteral / FunctionName / FunctionClose / ParameterName / ParameterValue / ToolClose`),
    `ToolCallConstraint` (shared table + immutable grammar + per-request mutable state, hence a **non-const** object constructed per request).
  - Grammar details: the trigger `<tool_call>` uses **rolling matching** (not greedy whole-segment matching); `FunctionClose` uses an `alive_` bitmask to track
    `</function>` and `<parameter=` simultaneously; `</parameter>` uses KMP; any mismatch sets `dead_` (no more constraint at this position, no more legal prefix);
    whitespace is skipped only where `progress_ == 0` in `FunctionLiteral / FunctionClose / ToolClose` (`FunctionName` does **not** skip).
  - **Safety valve**: if at some position no candidate other than pure whitespace can advance the grammar, report "unconstrained" instead of emitting an all-zero mask — whitespace is skipped and does not advance
    the grammar, so masking it would make the model emit spaces until the context is exhausted; an all-zero mask would instead hang the request. `build_mask` therefore returns `advances`
    (true only when the surviving fragment contains a non-whitespace byte).
  - **`dead_` is only read on the probe copy**, so history never pollutes candidate decisions.
- **`OutputSession`**: keeps the raw byte stream of the Content channel (`raw_content_text()`) and exposes `in_reasoning()`. The mask and the tool-call
  parser consume **the same** byte stream ⇒ a tool name spanning tokens, half a UTF-8, or a token straddling a structural literal can never be misjudged.
- **`tool_call_parser.h`**: the Qwen framing constants (`<tool_call>` / `</tool_call>` / `<function=` / `</function>` /
  `<parameter=` / `</parameter>`) are collapsed into `inline constexpr std::string_view`, and the local duplicates in the `.cpp` are removed.
- **`frontend`**: `make_tool_call_constraint` returns a non-const `shared_ptr<ToolCallConstraint>`; it returns empty when the contract is empty, when
  `enforce_declared_names` is off, or when `tools` is empty.
- **`tp2_generation_core`**: plain decode applies the mask before `ops::sample`, and MTP applies it before each verify column's `ops::argmax`;
  each column's mask = the grammar position corresponding to "committed text + drafts before this column" (`build_mask_after`); drafts themselves are not masked (an illegal
  draft disagrees with the masked argmax and is therefore rejected by `speculative_accept_greedy_drafts`); the constraint is active only when `!in_reasoning()`.
  Masking takes effect only in `FunctionLiteral / FunctionName / FunctionClose / ParameterName / ToolClose`; `ParameterValue` and
  free text are not masked.
- **Tests**: `tests/ops/test_token_mask.cpp` (GPU; bit-exact comparison, read-only mask, all-masked, shape/rank/dtype/alias validation; registered in
  `tests/ops/tests.cmake`) and `tests/test_tool_call_constraint.cpp` (pure CPU; lazy triggering, only declared names allowed, multi-token names,
  `>` required after a complete name, parameter names, free values, return to Free, tokens straddling literals, fallback for unspellable names, pure-whitespace dead end, second call,
  packed-domain tail; registered in `tests/models/qwen3_5/tests.cmake`).

### 3. Verification

- `ninfer_token_mask_test` / `ninfer_tool_call_constraint_test` / `ninfer_tool_call_parser_test` all pass; the same batch's
  `ninfer_sampling_test` / `ninfer_argmax_test` / `ninfer_engine_options_test` also pass (6/6).
- `ninfer_qwen3_5_frontend_test` still shows the **pre-existing** failure recorded earlier (`unsupported frontend/chat_template.jinja`,
  unrelated to this change; bisected once by "temporarily disabling the mask-table build" and HEAD fails the same way).
- **Real server end-to-end**: after the user restarted at 11:46, req#51 (`openai-responses` streaming, `tools 1`, prompt 111,124 / output 733,
  `cache 110,158 (99.1%)`, TTFT 1.2 s) **finished normally with no errors** — the domain fix took effect (before the fix this request would always 500).
- **Not done → done later (Round 15c)**: retesting against the Round 15 probe; 45/45 pass on the dflash2 route (see Round 15c at the end).

### 4. Pre-existing defect 1: the TP-2 sampling domain was the packed row count (fixed)

Exposed when the constraint table did not cover the logits domain: the TP-2 core had 4 places passing the **physical row count** `vocab` as the valid domain — `ops::sample` (prefill first
token, plain decode), `ops::argmax` (MTP target), and `speculative_accept_greedy_drafts` (the accept kernel). The contract already separates the two
(`sample` validates `token_domain ∈ [1, physical_rows]`, `argmax`'s parameter is named `valid_rows`, the accept docs say `token_domain`), and the
single-GPU path (`text.cpp:2279/2283`, `decode.cpp:258`, `draft.cpp`'s selector domain) all passes `public_token_count`; that computed-but-never-used
`public_tokens` was exactly the residue of this intent.

Two distinct vocabulary concepts: `resources.public_token_count` is the tokenizer's public vocabulary (the legal id domain), while `config.text.vocab_size`
(248320 on this machine) is the **packed** embedding/logits row count; `frontend/resources.cpp:12` only requires `count <= vocab_size`.
Fix: the declaration is moved next to `vocab`, and all 4 sites pass `public_tokens`; `sampling_workspace_capacity_bytes(vocab, 1, 1)` is kept
(as a capacity upper bound). The constraint mask is likewise generated over the **logits domain** (`build_mask(domain, mask)` / `build_mask_after(prefix, domain, mask)`),
and rows undefined by the tokenizer are zeroed (excluded).

**Impact**: an out-of-range id does not cause an out-of-bounds access (embedding/lm_head both have 248320 rows, so only padding rows are read), but it enters `OutputSession`'s
`Tokenizer::decoded_token` (`output_session.cpp:458/563`, unguarded) which throws `out_of_range` ⇒ a single request 500s. What the TP-2 path lacks is exactly
the guard that single-GPU `program/decode.cpp:256 validate_licensed_tokens` provides. The trigger condition is "a padding row beats all real candidates" (greedy requires it to be
the global maximum; sampling requires it to be in the top-20 and survive top_p/min_p), which under a normal distribution essentially never happens.

**Risk argument (why it is nearly a no-op)**: `token_domain` is only used in kernels as a loop upper bound (`sampling.cuh:33/41/127`), both domains have a
`cap` of `min(20, domain)`, and the RNG draw happens on the truncated candidate set ⇒ the fix only changes results at steps that would have triggered anyway, and normal steps are bit-unchanged.

### 5. Pre-existing defect 2: the penalty count array was always empty (fixed)

`make_sampling_config` left `token_counts` empty, while the penalty term only takes `c_v` from that array (`sampling_device.cuh:252`:
`cnt = c.token_counts ? c.token_counts[v] : 0`, plus the in-round overlay), so the TP-2 path's presence/frequency penalty was completely ineffective for
**cross-round** repetition — only the MTP verify in-round overlay (drafts within the same window) still worked.

The fix follows single-GPU `install_sampling` (`decode.cpp:145-149`): only when the penalty is non-zero, allocate an `I32[public_tokens]`
count array in the request workspace, zero it with `cudaMemsetAsync` (the arena hands back the previous request's bytes), and attach it to `sampling_config`; `ops::sample` (all four
finalize kernels `atomicAdd`) and `speculative_accept_greedy_drafts` accumulate the tokens they produce themselves. When no penalty is configured, no array is allocated and `c_v` is always 0, matching the single-GPU path. This route has no forced-token path, so `increment_token_counts` is not needed.

**Measured A/B** (same seed 12345, same prompt, same parameters; temporarily reverted `64db25f1` to build the pre-fix binary, and each binary ran its own service):

| Run | Before fix | After fix |
|---|---|---|
| penalty = 0 | 532 chars, `apple x30`, longest same-word run 15 | **byte-identical to the pre-fix run** |
| presence = 2.0 | **byte-identical to penalty=0** | different (580 chars, `apple x15`, 69 distinct words) |
| frequency = 2.0 | **byte-identical to penalty=0** | different (717 chars, `apple x9`, longest run 2, 99 distinct words) |

Key points: before the fix all three configurations were **byte-identical** (the penalty was completely ineffective); after the fix the no-penalty path is **byte-identical** to the pre-fix run (no regression);
after the fix the penalty works and frequency=2.0 suppresses repetition completely. The statistics include the model restating the prompt's `apple` occurrences in reasoning (about 16),
so `apple x30` is not purely generated output, but the relative change between the three runs is unaffected. Repeating with the same seed twice is byte-identical ⇒ the difference is attributable to the penalty,
not to randomness.

### 6. Observations not acted on (recorded)

- `tp2_generation_core.cpp`'s `sampling_b` is dead code (only `(void)sampling_b;`), and `sampling_buf_b` exists only for it.
- The TP-2 path should probably gain a `validate_licensed_tokens`-style guard (see the impact in §4).
- The id published by the service's `/v1/models` depends on the startup parameters/artifact (measured with the same artifact: the default startup publishes `qwen3.8-27b`, another run published
  `qwen3.8-27b-w4a4-w8a8`, while the stderr `engine ready` line always prints the latter). **The model in a request must be taken from `/v1/models`**,
  not inferred from logs or hard-coded.

### 7. Operational notes (native Windows / DSH agent sandbox)

- When the DSH file policy is `workspace-write`, **ninja cannot execute any subprocess**: `cmake --build` hangs silently, even for a trivial single-target
  project's `build.ninja` (`ninja -t` / `ninja -n` work fine). Builds must run under `danger-full-access`.
- Using `Tee-Object` (or any pipeline writing to that job's stdout) in a `pwsh` background job hangs halfway once the pipe fills; use
  `cmd /c "... > log 2>&1"` and poll the file instead.
- `ninfer-serve` must be started by the user in their own terminal: a service started with `Start-Process` is "a child process of the calling shell's job object",
  and is reaped when the agent tool call ends (measured: a service started within a single tool call ran to completion and could be killed by me, but across calls it disappears).
  Stopping the service + relinking the exe by the agent is fine (the stopped old process does not hold the exe, and the user restarts after relinking).
- Before compiling, `. ./tools/win_port/vcvars.ps1` is required; running tests requires adding the FFmpeg / curl DLL directories to `PATH`, otherwise the loader reports
  `0xC0000135`.
- Ports: see the end of Round 16.3 for the historical problem (the script's `$Port` and the client's `.dsh/settings.yaml` must stay in sync); this round measured the server on 3456.

## Round 16b — new-session divergence anchor: storing the divergence position in the host ring (the leftover item from Round 14 (1) is fixed)

The waste recorded as Round 14 (1) **did not disappear** after Round 16 landed the host ring. Round 16's ring has two sub-rings: the position grid
(`grid_slots` slots, stride `max(8192, ceil(max_context/grid_slots))`, guaranteeing coverage of the whole context) and the tail anchors
(chunk frontiers within `kReuseTailWindow` = 8192). But **the actual startup did not pass `--host-state-slots`** (`tools/win_port/serve.ps1`
does not contain that option), so the default 8 applied ⇒ `tail=4, grid=4` ⇒ `stride = max(8192, ceil(131072/4)) = 32768`. A new session shares only
system+tools (~14.8k), **which falls before the first grid point**; the inherited prompt's tail anchors are all at its end (~106k) and are all invalidated by the pruning rule
(`position > shared_prefix` ⇒ invalid) ⇒ `reuse=0`, full recomputation.

### Change

A third sub-ring `divergence` was added (`kReuseDivergenceCheckpointCount = 1`), **outside the `--host-state-slots` budget**
(+73.4 MiB pinned host per shard; taking a slot from the grid would push the stride from 32768 up to 43776 with a small ring, hurting the "deep divergence"
scenario instead). During the prefill walk, if `shared_prefix` falls strictly inside this walk's range, the chunk is truncated at that position and a state is frozen by D2H;
the next request that renders the same stable block restarts from there. The hit criterion reuses the existing scan; no new path was added.

The same change also fixed a pre-existing defect in `snapshot_host_checkpoint`: the original implementation started the memcpy after overwriting only `position`/`prefill_id`,
while the publish scan at the end runs only when the walk finishes normally; if prefill throws midway, that slot would keep the **previous prefill's `valid=true`**
with a new `position` and a possibly torn buffer, and later requests would restore a wrong state from it. It now sets `valid = false` before writing.

### Key correction: the anchor must not land on the observed divergence point

The first version failed outright in measurement:

```
B: prompt=35361 cached=35463 shared=17457 host=1/9 -> reuse=47  slot=8 src=host
C: prompt=17472 cached=35361 shared=17456 host=0/9 -> reuse=0   slot=0 src=device
```

B wrote its anchor at 17457, while C and B's shared prefix is 17456 — **one token short**. "The index of the first differing token" depends on how many following characters
the boundary token at the end of the stable block absorbed, and the following text differs between the pair. When the anchor lands exactly on the divergence point, the next pruning
kills it directly (`host=0/9`). Changed to anchoring `kReuseDivergenceMargin = 8` tokens before the divergence point: the cost is recomputing 8 tokens (~5 ms),
buying back the whole prefix.

### Measurement (native Windows, `--max-context 131072`, default `--host-state-slots`)

The scenario geometry deliberately mirrors the real deployment: A = nonce + HEAD(~17.5k) + tailA(~17.5k), B = same nonce + HEAD + tailB, C = same nonce +
HEAD + a short tail. HEAD's end is ~17.5k tokens from B's end, **far beyond the tail window (4096)**, so the tail anchors cannot reach it.

| Request | prompt_n | cache_n | prompt_ms |
|---|---|---|---|
| A_long | 35421 | 44 | 23774 |
| B_deep | 35363 | 0 | 23401 |
| C_short | 23 | **17451** | **71.8** |

Server trace (`NINFER_TP2_REUSE_TRACE=1`):

```
B: prompt=35363 cached=35465 shared=17459 ... host=0/9 stride=32768 -> reuse=0     slot=0 src=device
C: prompt=17474 cached=35363 shared=17458 ... host=1/9 stride=32768 -> reuse=17451 slot=8 src=host
```

C hit `slot=8`, i.e. the **divergence sub-ring** (grid 0-3, tail 4-7, divergence 8), not the tail window; and C's shared
(17458) is one token less than B's (17459), with that drift being absorbed by the 8-token margin. B paid the full 35,363 tokens / 23.4 s (i.e. the cost of every
new session before the change), while C advanced only 23 tokens / 71.8 ms. Startup ledger:

```
[mem] host-checkpoints shard 0 slots 9 (grid 4 + tail 4 + divergence 1) x 73.4 MiB | stride 32768 tok | pinned 660.7 MiB
```

**Benefit**: the second and later new sessions on the same stable block each save ~10.8 s. The first diverging request cannot benefit — its divergence point is unknown before the walk,
and the anchor is written by the request itself.

### Verification and leftovers

- The changed TU compiles on its own (`ninja -C build-win src/runtime/CMakeFiles/ninfer_engine.dir/engine/tp2_generation_core.cpp.obj`),
  and the full `ninja ninfer-serve` link passes.
- End-to-end as in the table above; corroborated both by the server trace and the startup ledger, with no need to compare against the old binary.
- TP-2's ring **still has no automated test coverage**: `test_tp2_forward` only tests `forward_tp2` and does not construct `TP2GenerationCore`.
  Covering it needs a real artifact + two GPUs, so this round again relies on end-to-end measurement.

## Round 17 — decomposing the TTFT floor: the 0.28–0.35 s is not "fixed overhead", half of it grows with prompt length

After Round 16b the remaining question was "where is the 0.28–0.35 s of fixed overhead per request". Conclusion: it is not a constant. At 100k context it decomposes into
~32 ms of Engine constant + ~81 ms of forward that grows linearly with context + ~10 ms of host state restore (new sessions only) + ~70 ms of server-side
rendering/tokenization + ~100 ms of HTTP/JSON/serialization/client.

### Localization tool: `NINFER_TP2_TIMING`'s prefill phase timing

On the TP-2 route `engine_timing` is entirely unfilled (`host_exposed_seconds`, `units` all 0), and `prefill_seconds` is the entire host wall clock from
the top of `execute()` (line 1090) to the first token, so it cannot be attributed. Therefore, under `NINFER_TP2_TIMING=1`, four phases were added —
scan / state / walk / post (with device synchronization inserted between phases, effective only when that environment variable is on): `scan` covers the reuse scan and pruning,
`state` covers GDN state restore, `walk` covers the walk and first-token sampling, and `post` covers snapshot and publish.

12-token suffix, default `--host-state-slots`, same-source comparison (the reuse source is annotated by `NINFER_TP2_REUSE_TRACE`):

| context | src | scan | state | walk | post |
|---|---|---|---|---|---|
| 15.3k | device | 0.02 | 0.55 | 53.83 | 0.03 |
| 15.3k | host | 0.02 | **10.95** | 53.24 | 0.03 |
| 30.5k | device | 0.03 | 0.55 | 65.43 | 0.05 |
| 30.5k | host | 0.02 | **10.94** | 65.36 | 0.05 |
| 45.3k | device | 0.04 | 0.54 | 77.51 | 0.05 |
| 45.3k | host | 0.03 | **11.06** | 77.49 | 0.05 |

- **scan is only 0.01–0.04 ms**: the O(context) host reuse scan (52k integer comparisons) is not the problem; the earlier suspicion based on intercept extrapolation was wrong.
- **The host path's state is a fixed 10.4 ms more expensive** (0.55 → 10.95), independent of context. The 147 MiB H2D is limited by GPU1's Gen4 x4
  (~7 GB/s), which is the link floor.
- **walk = 31.9 ms + 0.81 µs/token × context + 0.79 ms/token × suffix**. The three context points are linear and independent of the reuse source.
- post is 0.03–0.05 ms, negligible.

### The 0.81 µs/token lives in walk: a narrow query block scanning a long KV

The walk processes only 12 tokens, yet the forward grows linearly with context. 30k more context means each card reads about 557 MB more KV, taking 23.7 ms, i.e. an
**effective bandwidth of about 23 GB/s** (5% of that card's 448 GB/s peak). With only 12 queries, the parallelism is insufficient to hide KV read latency. This is a kernel
efficiency problem, not host overhead — the target of item 2.

### The 10.4 ms of host restore is not worth switching to a device plane

Moving the divergence anchor to a device snapshot costs only **+73.4 MiB/card** (one state plane; `state_arena` goes from 4 planes to 5,
293.6 → 367.0 MiB/card; 4.6% of the tighter card's remaining 1580 MiB), while also eliminating the host ring's 9th slot (−73.4 MiB pinned/card).
But the benefit is only **9.9 ms × once per session**: host restore happens only when the reuse boundary is neither the prefill end nor the adaptive rewind point. In Round 17's
scan, 2 of the 3 steady-state probes had `src=host` as an artifact of the probe shape — three rounds of equal-length prompts, a constant pair gap of 12, and the predicted next shared
prefix 15304−12−2 being 2 tokens below the actual shared prefix 15292, so it was discarded by the reachability guard `prompt_tokens - depth >= reuse` as "not reached this round"
(in the trace p2 left `rewind=0` and p3/p4 fell to host slot 6). In real chat the prompt grows each round and the shared prefix even extends into the
assistant reply, so the guard normally holds and the device path is used. **Conclusion: abandon that lever.**

### Server-side rendering/tokenization: 98–99.7% is tokenization, rendering is only 0.1–0.3 ms

`preparation_seconds` is written in the **`request_start`** event (emitted after preparation completes); it is neither missing nor all zero — looking under
`request_done` would misjudge it. Across 7 samples the difference between `total` and `tokenize` is only 0.04–0.34 ms:

| prompt_tokens | prepare total | tokenize | remainder |
|---|---|---|---|
| 15299 | 5.922 | 5.806 | 0.116 |
| 45299 | 29.884 | 29.545 | 0.338 |

That is, `chat_template.render()` is not the cost; **re-tokenizing the whole history for every request** is. `--request-log-jsonl`'s
`timings_seconds.prepare` and `preparation_seconds.total` agree on the chat path (within 0.02 ms), so the main path tokenizes only once —
`Frontend::count_tokens` serves only Anthropic `/v1/messages/count_tokens`.

### Fix: two heap allocations per BPE word

`append_normalized_bpe_ids` (`tokenizer.cpp`) created a new `std::vector<BpeNode>` and a new `std::priority_queue` for every pre-tokenizer word.
A long prompt of about 3–4万 words means over 100,000 heap allocations. Changed to hoist both buffers out of the word loop for reuse
(`nodes.resize` / `heap.clear()`), and to replace `std::priority_queue` with `std::push_heap` / `std::pop_heap` over the same vector
plus a `LaterBpeCandidate`.

This **does not change semantics**: the specified implementation of `std::priority_queue::push` is exactly `c.push_back(v); std::push_heap(...)`,
`pop` is `std::pop_heap(...); c.pop_back()`, and `top()` is `c.front()` — the insertion sequence and comparator are identical,
so the pop order is bit-identical. The `#include <queue>` was retired along the way.

Fixed-content probe, same real artifact, two service processes (before/after):

| cache_n + prompt_n | tokenize before (ms) | tokenize after (ms) | speedup |
|---|---|---|---|
| 1211 | 0.330 | 0.226 | 1.46x |
| 5851 | 1.178 | 0.766 | 1.54x |
| 1812 | 0.856 | 0.439 | 1.95x |
| 7037 | 2.715 | 1.613 | 1.68x |
| 27913 | 13.623 | 6.178 | 2.20x |
| 39213 | 14.568 | 8.329 | 1.75x |
| 67 | 0.064 | 0.056 | 1.13x |

Total 33.3 → 17.6 ms (**1.89x**). At 100k context this item drops from about 38 ms to about 22 ms.

### Verification

- **Bit-equivalent**: the same probe before and after the change gives identical `cache_n + prompt_n` for all 7 requests (including CJK, the combining characters
  e+U+0301 and a+U+0300, and mixed emoji cases), and `prompt_ms` is unaffected.
- `ctest -R qwen3_5_frontend` **passes** (including `test_bpe_merge_order`, `test_boundary_aware_tokenization`).
- A pre-existing environment problem was fixed along the way: the *.jinja files under `tests/fixtures/frontend/` are CRLF in this checkout, while `.gitattributes`
  declares `*.jinja text eol=lf`, so the template sha256 is not in the allowlist and the frontend test exits with `0xC0000409` (uncaught-exception fastfail)
  and **no output at all** — `main` does not catch exceptions, so the failure point is invisible. Normalizing the two fixtures to LF (invisible to git; `git status`
  is still clean) makes the whole suite pass. The crash reproduces identically after stashing the tokenizer change, so it is unrelated to this round.

### Leftovers

- Tokenization still costs about 0.22 µs/token (~22 ms at 100k). Eliminating it entirely requires caching `BoundaryEncodedText` prefixes per added-token segment
  (`encode_with_boundaries` already splits at special tokens and normalization between segments is independent, so splitting by segment is provably equivalent), at the cost of
  introducing a concurrency-protected cache and one prefix-concatenation path in Frontend — too much risk for ~22 ms, so it was not done this round.
- The biggest part of the floor is still walk's 0.81 µs/token (~81 ms at 100k), i.e. item 2.

## Round 18 — a missing attention route for TP-2's narrow chunks: walk's 0.81 µs/token comes from "running a dozen query rows on the prompt kernel" (fixed)

Round 17 recorded walk's context-dependent term (0.81 µs/token, ~81 ms at 100k) as item 2. This round located the root cause: **not insufficient memory bandwidth, but the routing handing
TP-2's split narrow chunk to the prompt kernel**, whose KV-direction parallelism is insufficient to saturate memory.

### Root cause

`ops::causal_softmax_attention` (`src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp`) chooses one of three private routes based on `q_heads`/`width`/`max_visible_keys`: `SmallT`,
`ChunkedSmallT` (both flash-decoding-style split-KV, producing partial_acc/m/l first and then merging), and `Prompt` (a large-T kernel, parallel along query rows, essentially
serial in the KV direction).

- Single-GPU (24 q heads) with `width <= 16` is taken over by the first branch: with fp8 storage, the prompt upper bound at `width >= 9` is only 320 keys, and beyond that it falls to `ChunkedSmallT`.
- `q_heads == 16` (the other split) likewise falls to `ChunkedSmallT` when `max_visible_keys > prompt_visible_keys`.
- **`q_heads == 12` (Qwen3.8-27B's TP-2 split: half of 24q/4kv/head_dim256) has no branch at all**, so beyond `width <= 6` everything falls to the trailing `return Prompt`.

The three routes' implementations already existed and were already instantiated for 12 heads: `causal_attention_chunk_tokens` returns 6 for non-24 heads, `causal_attention_split_capacity`
has a `CausalD256H12Kv2` branch (`small_t.cu:249`), and all five small-T launchers (bf16/i8/fp8/nvfp4/k8v4) carry 12-head dispatch. The 12-head geometry was introduced by
`548a426a feat(tp2)`, while the routing conditions come from the earlier `a7818988` — **the extension was not kept in sync; it is an omission**, not a design tradeoff. 12 query rows
walking 45k keys serially along KV measure about 20 GB/s, i.e. 0.81 µs/token.

### Change

- `causal_softmax_attention.cpp`: relax the long-context split-KV condition from `q_heads == 16` to `(q_heads == 12 || q_heads == 16)`. The `width <= kMaximumVerifyTokens(16)` guard is kept, so large chunks still go to the prompt kernel. The workspace capacity function already re-runs `causal_attention_resolve_route` per width, so the planning side needs no change.
- `tests/ops/softmax_attention/causal_cache.cpp`: register `d256-h12-kv2` in `kGeometries` (previously only 24/4 and 16/2), relax the `q_heads == 16` threshold for long-envelope cases to `!= 24`, and add `verify_route_selection()` asserting route selection directly (12 heads long context → ChunkedSmallT, 12 heads short context → Prompt, 12 heads 6 tokens → SmallT, 24 heads → ChunkedSmallT).
### Measurement

Same probe (12-token suffix, `NINFER_TP2_TIMING=1`'s `[tp2-prefill]` phases), in ms:

| context | walk before | walk after | prompt_ms before | prompt_ms after |
|---|---|---|---|---|
| 15.3k | 53.83 | 44.12 | 64.3 | 55.2 |
| 30.5k | 65.43 | 45.71 | 76.4 | 56.7 |
| 45.3k | 77.51 | 47.46 | 88.7 | 59.0 |

- The context-dependent term **0.81 → 0.111 µs/token (7.3x)**: per 30k of context, from +23.7 ms down to +3.3 ms.
- The constant term did not get worse: `chunks=1`, weights are still streamed once per prefill chunk, and split-KV only re-splits into two segments of 6 tokens inside the attention op.
- Extrapolated to 100k: walk from about 113 ms down to about 44 ms, **about −69 ms**.
- The 1028-token suffix benefits too (35k: 891.7→851.7; 45.3k: 985.1→926.1), though that segment is dominated by per-token weight streaming.

### Verification

- `ninfer_softmax_attention_test` **passes**: the 12/2 geometry is now validated against an **independent FP64 oracle** in all storage types (BF16/INT8/FP8/NVFP4/K8V4), all three block-table mappings, and the A1/A3 (with/without mask) cases; the new long-envelope case is exactly the new route. This Op's public contract explicitly requires that "every cache route be validated directly against an independent mathematical oracle, with cross-route agreement counting only as corroboration", so registering the split geometry in the test matrix is part of that contract.
- End-to-end generation smoke: `17*23` gives 391, the capital of Peru is Lima; normal.
- Full `ctest -j 2`: **126/127 pass**, 4 skipped for a missing real artifact, and the only failure is the pre-existing `ninfer_request_log_test` (`bfec5c03` changed the pretty line but did not update `tests/test_request_log.cpp:487`; unrelated to this round); that stale expectation was filled in as `(300 tok)` along the way and passes after.

### Remaining

- walk's constant term (about 32.9 ms) and suffix term (0.79 ms/token) are untouched; at 100k walk is about 44 ms, already the smaller part of the floor.
- The split geometry's `prompt_visible_keys` (512/1024 keys) reuses the 16-head split's tuning; this round only measured ≥15.3k context, and the crossover at shorter contexts was not measured (the benefit would be small anyway).

---

# PLAN.md archive (archived 2026-09-21, Rounds 48-55 + worklog Rounds 11-19)

Archived verbatim from the working PLAN.md at the point of the upstream cherry-pick re-plan. The completed sections are summarized in the new PLAN.md; this is the full record.

# NInfer TP-2 plan (2× RTX 5060 Ti · Qwen3.8-27B NVFP4)

> **The only active plan**, and the persistent memory across context compression. Last organized: the Round 48 wrap-up.
> - Full history (the original PLAN.md in full, Round 1–48 evidence and failed attempts): `tp2-adaptation-worklog.md`
> - Delivery notes, recommended configuration and measurements: `docs/tp2-dual-5060ti.md`
> - Status: **all of delivery scopes ①–⑤ are complete.** ①'s criterion was changed from "token-for-token identical to plain" (unreachable, see §5.5) to
>   "no regression + same-tier quality + a speedup", and has been met (§5).
> - **Round 52 wrap-up (done)**: vocabulary/hidden parallel splitting (§10's Step 1+2+3) plus memory shuffling pushed the context to the
>   artifact limit **262,144 tokens** (15,614 / 15,094 MiB/card), with greedy output byte-identical to before the optimization.
>   Details in `docs/tp2-dual-5060ti.md` and §10 of this file.
> - **Round 53 (in progress)**: TP-2 vision — "static shard assignment" (MTP only on shard 0 / vision only on shard 1),
>   see §11. Approved by the user; the goal is not to introduce a per-image token limit if memory allows.
> - **§13 (done and measured)**: TP-2 multi-session KV pool — host KV swap-in/swap-out,
>   with only one active KV in device memory at any time and up to 5 inactive sessions in host memory (LRU evicting the oldest), recalled wholesale when a session returns,
>   prefilling only the new suffix. See the §13 progress section.

---

## 1. Goals (the user's original requirements)

1. Fix the MTP consistency problem;
2. Do benchmarking;
3. Look for prefill/decode optimization headroom;
4. Optimize memory usage, enable KV cache fp8/q8_0 quantization, and adjust parameters that cost resources without much impact on performance/quality, in order to enable a larger context;
5. Deliver an inference framework extremely well optimized for this machine's dual-GPU 5060Ti TP-2.

Additional completion conditions from the original plan (recorded in archived §0): zero regression on the TP=1 path, split Ops have independent oracles, end-to-end quality confirmed,
performance floors (MTP0 ≥ 28 tok/s, MTP3 ≥ 70 tok/s), and C=1/2/4 stability.

---

## 2. Completion status

| Goal | Status | Evidence |
|---|---|---|
| ① MTP usable (no regression / same-tier quality / a speedup) | **Done** (§5) | 6 samples content 1519/1318/1294/1195/1683/0 (median 1306, `longest0` all 0) vs plain 1165/1452/1546/672/224/1237 (median 1201); steady-state decode 46–49 vs 31.6 tok/s |
| ② Benchmarking | Done | prefill 279 tokens / 0.453 s (615 tok/s); decode end-to-end (including prefill) 256 tokens: plain 8.153 s (31.4 tok/s), MTP K=3 6.532 s (39.2 tok/s); pure decode: plain 31.5, K=1 42.0, K=2 50.5–54.0, K=3 47.9–48.0 tok/s |
| ③ Memory / KV quantization | Done | Five dtypes (bf16/int8/fp8/nvfp4/k8v4) all usable on TP-2; fp8 quality same tier as bf16 (1365/859/1382/1602 vs 1233/763/1469/1456/1430/1621, no "0" repetition in either); plain memory 13,888 MiB/card |
| ④ Larger context | Done | 65,536 → 131,072 → **262,144 tokens** (fp8 KV, 15,614 / 15,094 MiB/card); end-to-end 118,869-token prompt HTTP 200 / 96.8 s, 65k-token prompt 48.1 s |
| ⑤ Documentation and recommended configuration | Done | `docs/tp2-dual-5060ti.md` (recommended configuration, benchmark tables, KV dtype table, MTP limits and root causes, fix list) |

**Original thresholds not met** (recorded honestly): MTP3 ≥ 70 tok/s was not reached (measured pure-decode ceiling K=2 50.5–54.0);
the TP-2 path has not run a perplexity evaluation (quality evidence instead uses multi-sample A/B on the same prompts); the per-shard arena's
`memory_summary()` still reports `pages 0/0` and `runtime 0 B` (a display-only issue with no functional impact).

---

## 3. Recommended run configuration (8088 currently runs with this)

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2 --lm-head-draft --vision --reasoning-effort medium
```

`--reasoning-effort low|medium|xhigh` (added in Round 53c) is the **process-level default thinking effort**: without the flag, the template's own default applies
(`xhigh` for this artifact), and a request body's `reasoning_effort` takes precedence. Measured on the same question,
`xhigh` thinks for 172 tokens and `medium` for 46.
**`--chat-template` does not exist and this architecture does not support it**: the template is an artifact resource
(`resource/text/chat_template.jinja`) that must be byte-identical to `tokenizer_config.json.chat_template`,
and is then compiled against a sha256 allowlist — only `e84f32a2…` (thinking-toggle) and `c3cf9e34…` (reasoning-effort,
which is what this artifact uses) are accepted. See §11.8.

262,144 is the artifact's `max_position_embeddings`: prefill 1588 tok/s, steady-state decode 59.0 tok/s.
MTP (`--spec mtp --draft-tokens 2 --lm-head-draft`) has been validated on both the bf16 and fp8 KV routes as no-regression and
same-tier quality; dropping those three flags returns to plain.

`--vision` (usable as of Round 53): static shard assignment places MTP on shard 0 and the Vision tower on shard 1,
adding a `vision 413.3` line to the startup ledger (shard 1), with 699 / 951 MiB of headroom on the two cards. Multimodal requests and pure-text requests
go through the same mechanism (MTP + prefix reuse, see §11.6), and cold-start greedy is byte-identical.

---

## 4. Key work completed

- **prefill/decode optimization (goal 3)**: batched prefill (TTFT 19×), block-width sweep and locating the real allreduce cause (Round 35b),
  the ordered AR slice pipeline (+15%, Round 35c), allreduce 6 sync → 2 sync and in-kernel allreduce + mixer head splitting
  (Rounds 33/34, catching up with llama.cpp). Details in the archive.
- **KV quantization fix (the key prerequisite for goal 4)**: quantized KV originally could not start on TP-2 at all. The root cause was that
  `prompt_fp8.cu` / `prompt_nvfp4_non_rdc.cu` / `prompt_k8v4.cu` used a **function-local `static`** to do the
  `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` opt-in (92,416 B > the 48 KiB default); that attribute is a
  **per-device** function attribute, and the device holding the second shard never opted in → `cudaErrorInvalidValue` on every launch.
  After caching it per device all five dtypes became usable (`tools/tp_bootstrap/fix_per_device_attr.py`).
- **Larger context (goal 4)**: fp8/int8 KV makes 131,072 tokens fit within 13.9 GB/card, proven end-to-end with a 118,869-token prompt.
- **MTP state machine fix**: snapshot the GDN state into a new scratch slot (`kRoundScratchSlot`) before each round's verify, restore before fold,
  eliminating the double advance of "verify advances the live state + fold advances it a second time" (first divergence 4 → 10).
- **Temporary instrumentation cleanup**: `NINFER_TP2_ROUND_DETAIL/PARITY/TOKEN_TRACE/SKIP_FOLD/VERIFY_PHASE` all removed
  (`grep NINFER_TP2_ ` returns 0 hits).

---

## 5. Done: ① MTP output degradation (root cause: the split never wrote replay records)

### 5.1 Symptom (before the fix)
MTP was usable and faster (end-to-end +25%, pure decode +50%), but the output degraded: at temp0.7/top-k20/top-p0.80, 6 samples had a
content median of 0 (all 6 showed "0" repetition, `longest0` reaching into the hundreds), while plain's median was 1443 with 0/6 degradations. Greedy reproduced it too.

### 5.2 Root cause (confirmed)
TP-2's head-split shards **never produced replay records at all**: both record branches in `gdn_mix` are conditioned on
`shard_config_ == nullptr` (the fused width>1 record op was only registered for the full 16384-row fused parent, which a shard cannot get). Yet every MTP round still ran
"restore the pre-round snapshot → run `gdn_replay_fold` with `commit_columns`", so fold consumed a **record plane that was never written**:
the recurrent state of the 48 linear-attention layers was replayed from an empty log every round (equivalent to not advancing), and the conv history was rebuilt from the same empty log;
the 16 full-attention layers worked as usual ⇒ the first few tokens were locally coherent and then it collapsed into "0" repetition. This matches the symptom exactly.

### 5.3 Fix (`src/models/qwen3_5/execution/text.cpp` · `gdn_mix`)
- The shard now records on its own decomposed route: the conv record is that route's already-materialized `[q|k|v]` raw convolution input, copied straight into the
  record plane (channel order and layout match; what fold needs is `conv_record[0:committed]`);
- The recurrent path switches to `ops::gated_delta_net_replay_record`: it explicitly registers the split geometry (qk 8 heads / value 24 heads),
  and its contract requires its output to be **bit-identical** to the normalized `gated_delta_net` ⇒ the per-column window logits and the accept decision are unchanged, and only the state transition
  becomes replayable (exactly what fold needs);
- `causal_conv1d_silu.h`'s row profile documentation gains the split profile `(1024,1024,3072) / C=5120` (the implementation already supported it).

### 5.4 Evidence
- End-to-end A/B (`r36f_stats.sh`, same prompt, temp0.7/top-k20/top-p0.80, 6 runs each): MTP content
  1519/1318/1294/1195/1683/0 (median 1306), **6/6 `longest0 = 0` (zero repetition)**; plain 1165/1452/1546/672/224/1237
  (median 1201). The single 0 is a reasoning-budget artefact and appears for plain too (both examples spent the whole budget on reasoning).
- No throughput regression: MTP K=2 steady state 44.3/46.2/45.5/46.1/49.2 tok/s, plain 31.6–31.8 tok/s.
- A second defect (also "process-level static for a per-device attribute"): the small-T kernels in `small_t_fp8.cu`/`small_t_k8v4.cu`/`small_t.cu`
  also need the >48 KiB dynamic smem opt-in, but single-token decode used a 32 KiB tile that happened to stay under the 48 KiB default, so the defect stayed latent;
  the MTP window (T=K+1) switching to a 64 KiB tile exposed it ⇒ `small_t_fp8.cu:56 cudaErrorInvalidValue`. All four paths are now unified on
  `src/ops/common/cuda_smem.h`'s per-(kernel, device) opt-in. Verification: fp8 KV @131072 + MTP K=2, three samples
  content 1530/1481/1708, `longest0` all 0 (`r49_fp8mtp_smoke.sh`).
- All regression tests PASS: `gated_delta_net_replay_record`, `gdn_replay_fold` (including the 48×8×24×5120 split geometry),
  `gdn_input_proj_conv_record`, `gdn_input_proj_conv_snapshot`, `gdn_input_proj`, `tp_device_pair`,
  `qwen3_5_tp2_load --artifact`, `qwen3_5_tp2_forward --artifact`.

### 5.5 Criterion correction (important)
"Token-for-token identical to plain" is an **unreachable criterion** and has been abandoned: the verify window and single-token decoding are inherently different execution shapes
(this engine uses chunked attention/conv, llama.cpp uses a chunked GDN kernel), and the KV of accepted tokens is written by that window and not revised afterwards.
llama.cpp's Qwen3.5 MTP structure is the same, and its documentation likewise only asks for "use greedy when exact consistency is needed". W1's criterion therefore became
**no regression + same-tier quality + a speedup**, and it has been met.

### 5.6 Two rejected diagnoses (recorded so they are not retried)
- **The phase theory**: it was once concluded that "verify runs `Phase::Prefill` and decode runs `Phase::Verify`, so the two kernels round differently". Wrong: `Phase`
  has no semantics on a head-split shard — `attn_mix` does not use its phase parameter at all, and both of `gdn_mix`'s `ph == Phase::Verify` sites
  are excluded by `shard_config_ == nullptr` — so changing the phase changes no value. The original §5.5 plan for a "rank-3 → rank-4
  engine-level layout rework" is void along with it.
- **The `rope.cu:189` theory**: that is a `CUDA_CHECK(cudaGetLastError())` checkpoint, not the failing kernel; with multiple devices interleaved, an asynchronous illegal address
  surfaces at the **first** checkpoint, so the reporting site need not be the failing kernel.

---

## 6. Optional follow-ups (non-blocking)

- **W2 load balancing** (**done**): with MTP enabled, GPU0 at 100% / GPU1 at ~72% is structural (the MTP layer plus all argmax/accept/fold/sampling/policy work lives only on
  shard A; B idles after the text mixer lockstep). Directions: (1) both cards redundantly running the same MTP layer in lockstep (no wall-clock change);
  (2) splitting the MTP proposal's lm_head + window argmax by vocabulary (genuinely shortening the critical path); (3) re-sweeping K=1/2/3.
- **KV dtype sweep completion**: fp8/int8 already have quality and memory data; nvfp4/k8v4 were only validated for startup and throughput.
- **Tool-call constrained decoding** (**done**, route A; diagnosis in Round 15, implementation and verification in worklog Round 15b): align with llama.cpp's lazy
  tool-call grammar, masking the tool name and parameter names after `<function=` to the tools declared by this request (a text-based lightweight version; the full GBNF
  route B is in Round 15). The mask is applied outside the CUDA Graph, MTP takes each verify column's own grammar position with drafts unmasked, and the thinking phase is unconstrained.
- **TP-2 host-side state checkpoints + LCP-truncated reuse for short prompts** (**implemented and end-to-end verified**, diagnosis in Round 16, measurement in Round 16.3): put GDN state snapshots in a pinned host
  ring pool (zero device-memory growth, `--host-state-slots` semantics), stride 8192, with the reuse scan taking the deepest `<= shared_prefix` checkpoint.
  This fixes "the client re-renders a shorter prompt at a turn boundary ⇒ reuse=0 ⇒ the whole thing is recomputed (measured 42.0 / 57.6 s)". Measured `src=host` hits: req#5 10.2 s → 1.3 s,
  req#11 37.6 s → 28.6 s, with the recomputation within one chunk of the theoretical minimum.

---

## 7. Pre-delivery cleanup checklist

- [x] Remove runtime instrumentation (`NINFER_TP2_*`)
- [x] Archive the complete record: `tp2-adaptation-worklog.md`
- [x] Delete `tools/tp_bootstrap/r36b_plan_append.sh` and `r36c_plan_append.sh`, which only appended history text to PLAN.md
- [x] W1 (①: MTP degradation fixed, criterion corrected to "no regression + same-tier quality + a speedup", see §5)
- [ ] Retest concurrency C=1/2/4 and the streaming/stop smoke with the recommended configuration (Rounds 31/34's conclusions were based on the bf16 configuration)
- [ ] Remove this file (PLAN.md) at delivery per AGENTS.md; `tools/tp_bootstrap/` holds about 300 one-off diagnostic scripts, and
      what to keep versus clean up must be decided (among them `build_r35.sh`, `serve_*`, and the `r37+r38+r39+r4x` series are referenced by `docs/tp2-dual-5060ti.md`
      as reproducible processes)

---

## 8. Local environment and operational essentials

| Item | Value |
|---|---|
| Build/run | WSL2; Windows work tree (this repository), build tree `<build-dir>` |
| Build | `bash tools/tp_bootstrap/build_r35.sh` (rsync `src tests apps include bench tools` then `cmake --build build -j 8`; success marker `BUILD_EXIT=0`, about 2–3 minutes) |
| Service | 8088; `serve_stop.sh` → background `wsl -e bash .../serve_supervise.sh <ctx> 8088 '<extra args>'` → poll for "listening on http://127.0.0.1:8088" (30–37 s to load) |
| First request | pays CUDA graph capture (measured 14.4 s); only steady-state numbers are meaningful |
| Constraint | This session must not terminate/restart/start any llama.cpp process; only ninfer processes are managed |
| Toolchain notes | Nested `pwsh` → `wsl -e bash -lc "..."` swallows `$var` and redirections ⇒ write the command into a script file under `tools/tp_bootstrap/` and execute that; the blocking-call limit is 600 s ⇒ use background jobs with ≤420 s polling for long tasks |

---

## 9. Evidence index

- Scripts: `build_r35.sh` (build), `serve_supervise.sh`/`serve_stop.sh` (service), `r37_colparity.sh` (per-column parity),
  `r36h_trace.sh` (per-token divergence), `r38_kvsweep.sh`/`r42_fp8check.sh`/`r44_fp8quality.sh` (KV quantization),
  `r39_bench.sh`/`r43_bigctx.sh`/`r48_longctx.sh` (benchmark/long context), `r45_verify.sh`/`r46_recconfig.sh`/`r47_verify.sh` (acceptance),
  `fix_per_device_attr.py`/`strip_tp2_probes.py`/`strip_tp2_probe_comments.py` (source edits),
  `verify_gdn_replay_fix.sh`/`verify_tp2_artifact.sh` (①'s fix regressions), `r36f_stats.sh` (quality A/B),
  `r49_fp8mtp_smoke.sh` (MTP + fp8 KV smoke)
- Logs: `$HOME/prof/` (`serve_supervised.log`, `kvsweep-out.log`, `r39/bench-out.log`, `r42_fp8check.log`,
  `r43_bigctx.log`, `r44_fp8quality.log`, `r45_verify.log`, `r48_longctx.log`, `kvdiag-out.log`, `ab_stats.log`, `r49_smoke.log`)
- Source change list: `docs/tp2-dual-5060ti.md`'s "Fixes applied on this branch" + §4/§5.3 of this file; per-round file references are in the archive

> Note: ①'s fix changed `src/models/qwen3_5/execution/text.cpp` (`gdn_mix`) and `include/ninfer/ops/causal_conv1d_silu.h` (documentation); the rest of the historical change list is in `docs/tp2-dual-5060ti.md`'s "Fixes applied on this branch".

---

## 10. Done: vocabulary/hidden parallel splitting (Step 1 + 2 + 3) and the 262,144 context

Motivation (the Round 51 memory ledger): each card held a complete `text/token_embedding` (1,213 MiB) and `text/output_head`
(1,213 MiB), one copy per card ⇒ 4.8 GiB total. Splitting along the vocabulary dimension leaves half on each card ⇒ −606 MiB per table per card,
−1.2 GiB/card for both tables; at roughly +65,536 tokens of context per 1 GiB/card saved, 262,144 still needs card 0 to free ~930 MiB more.

**Step 1 (this round)**: `text/output_head` → `WeightSplitKind::ColumnParallel` (per-card row range
`[shard*V/2, (shard+1)*V/2)`).

- Loading: `tp_split_spec` gains `TpSplitOptions::split_output_head`; `materialize_model_tp2` enables it when
  `purpose == Generation && (!speculative_enabled() || proposal_enabled())`. Rationale: with the Full proposal head selected,
  `text/output_head` and `mtp/output_head` are the same artifact object (weight-tied), and the MTP layer only runs on shard 0,
  whose route's proposal argmax needs the full head ⇒ keep it replicated (behaviour exactly as today), leaving the resolution to Step 3 with peer participation.
- Execution: new `TextContext::project_head_tp2(peer, pair, hidden, hidden_peer, logits, logits_peer)`:
  - Uns­plit: two full `project` calls (status quo).
  - Split: each card first `project`s a compact `[V/2, T]`, zeroes the `[V,T]` buffer, writes into its own half with `cudaMemcpy2DAsync`,
    and then `pair.allreduce` sums them (the peer's half is zero) ⇒ both cards get the full `[V,T]` logits, and the sampling/accept logic is unchanged.
  - Three call sites: `forward_tp2` (decode T=1), `forward_tp2_prefill`'s `[V,1]` branch, and verify's
    `logits_columns` (`[V,width]`, which originally only had a buffer on shard A ⇒ one is now opened inside the function for shard B).
- Acceptance (all pass): `ninfer_qwen3_5_tp2_load_test` on all three routes (no spec / `--spec mtp` /
  `--spec mtp --lm-head-draft`), `forward_test` (11 probes + 32-step AR + batched prefill),
  `ninfer_tp_device_pair_test`; the new op oracle `ninfer_linear_tp2_split_fp8_head_test` (real head shape
  [248320,5120] at T=1/3/5, [16384,5120] at T=1/4, all "exact"); 8088 end-to-end greedy comparison
  (`top_k=1` fixed) byte-identical on 5 prompts (`ab-control.jsonl` vs `ab-split.jsonl`, all hashes equal).
- Memory: 14,940/14,678 → **14,334/14,072 MiB/card** (−606 MiB/card).
- Also fixed `weight_splitter`'s FP8 slice descriptors: `slice_fp8_rows` gains `scale_ne[0]`/`scale_nb[1..3]`,
  and `slice_fp8_cols` gains `group_size`/`group` (previously a split Weight and its payload were inconsistent, and only the production path that rebuilds descriptors from the geometry masked the problem).

**Step 2 (done)**: `text/token_embedding` → `WeightSplitKind::RowParallel` (split along the hidden columns, `[V, D/2]` per card). Column splitting was chosen over vocabulary-row splitting: ids are always in the global domain (there are no out-of-range rows), and the merge path is completely isomorphic to the output head's.

- Execution: `TextContext::embedding_tp2(peer, pair, ids, ids_peer, x, x_peer)`: each card looks up its own half of the columns into a compact
  `[D/2, T]`, then uses the newly extracted `merge_local_row_blocks` (shared with the head) to zero, write the half, and allreduce; the text path
  (`forward_tp2`/`forward_tp2_prefill`) passes the peer's ids directly. The MTP stem only runs on shard 0, so a new
  `TextContext::set_tp_peer` (registered by the TP-2 core once both cards are up) broadcasts the ids to the peer (16-byte-aligned padding + allreduce;
  the shape uses a prefix view so `T` matches the output) before merging; it hard-errors when no peer is registered and the table is split.
- Blocker and resolution: the FP8 embedding's registered domain was hard-coded to `[248320,5120]`. The FP8 gather kernel was templated on hidden width
  (`embed_gather.cuh`), the launcher dispatches on width (separate block partitions for 5120/2560), and the wrapper validates with
  `embed_gather_fp8_supports_width` (the vocabulary is no longer constrained, since the kernel is row-independent); the op docs and
  `ninfer_embedding_test` were updated (`Fp8Table` parameterizes width, and 2560 runs the full T sweep + CUDA Graph + alignment cases).
- Acceptance (all pass): `ninfer_embedding_test` (5120+2560 in full), `tp2_forward_test`, both load tests;
  8088 end-to-end greedy comparison of 5 prompts byte-identical to the Step 1 build (`ab-embed.jsonl` vs `ab-split.jsonl`).
  Two problems surfaced that only a real service can expose: `allreduce` requires the byte count to be a multiple of 16; and the broadcast buffer shape must
  equal the number of ids (otherwise the gather's T check fails).
- Memory: 14,334/14,072 → **13,728/13,466 MiB/card** (−606 MiB/card more; −1,212 MiB/card over the two steps).

**Step 3 (done)**: `proposal/head` (131072×5120 Q4_G64_FP16, 341 MiB) split by **vocabulary rows**, 65,536 rows per card
(171 MiB). `proposal_argmax` goes through a `hidden` broadcast → each card computes its own half → `merge_local_row_blocks` assembles
`[131072,T]` → shard 0 argmax + token-id remapping. Blocker: Q4 A16 is an **enumerated (n,k) table** with no 65536×5120 entry;
`select_q4_n65536_k5120` was added (`q4_shapes.h`/`q4_dispatch.cpp`/`sources.cmake` + a new `.cu`). Oracle:
`ninfer_linear_tp2_split_grouped_head_test` is all "exact" when compared bit-for-bit against the full Op at the real [131072,5120], T=1/2.

**Round 52 results (262,144 landed)**:

- Context: 131,072 → **262,144 tokens** (the artifact limit). `nvidia-smi` 15,614 / 15,094 MiB/card (697 / 1,217 free).
- Memory shuffling: state arena 367→294 MiB (prefix reuse snapshots 3→2; `rewind_depths` keeps only `{0, rewind_near_}`,
  measured 0.30/0.33 s to re-enter the same 16k/65k prompt ⇒ sufficient for the chat-turn pattern); workspace 384→192 MiB.
- Performance (t=0, usage counters): prefill 16,057 tokens / 10.11 s = **1588 tok/s**; decode 512 tokens / 8.68 s = **59.0 tok/s**
  (223 rounds, 2.3 tokens/round, 64.6% acceptance per draft), about 20% above the 46–49 tok/s of the 131072 era.
- Bit-identical: 5 greedy prompts hash identically at 131072 (pre-optimization) and 262,144 (`r52_ab.sh final` + `r52_cmp3.sh`).
- Dead-end record: dropping the MTP layer KV to nvfp4 saves 228 MiB with an unchanged acceptance rate (392/510 vs 391/510), but 3/5 prompts change output —
  verify+fold and single-token decoding have different execution shapes, and once the draft mode changes, near-tie choices change; the fp8 MTP KV was therefore kept.
- All regressions green: 3 load routes, `tp2_forward`, `embedding`, `linear_tp2_split_fp8_head`,
  `linear_tp2_split_grouped_head` (new), `linear_tp2_split_nvfp4`, `tp_device_pair`.

---

## 11. In progress: TP-2 vision (static shard assignment)

**Goal**: support `--vision` on the dual-GPU TP-2 route. Using **static assignment** rather than dynamic swap-in/swap-out:
`mtp/*` only on shard 0 (shard 1's current 430.4 MiB copy is pure waste — MTP only ever runs on shard 0 anyway),
`vision/*` and the vision encoding workspace only on shard 1, leaving shard 0 completely untouched. User decision: **if memory suffices, do not introduce a
"per-image token limit" configuration option**.

### 11.1 Why this design (Round 53 argument, done)

Status quo (three independent obstacles, in order of severity):
1. `normalize_engine_options` hard-sets `options.enable_vision = false` in the `device_b >= 0` branch
   (`src/runtime/engine/model_instance.cpp:96`); the serve layer has a second gate
   (`vision_disabled` at `src/serve/generation_service.cpp:313,368`).
2. `TP2GenerationCore` has no vision execution path anywhere (only two pass-through spots, `load.vision` and `frontend.vision_enabled`);
   the encoder session `VisionPrefillSession` exists only on the single-GPU Program path.
3. Memory.

Memory ledger (measured on this machine):
- Component bytes (parsed directly from the directory by `tools/artifact/reader.py`): text 19431.7 / dflash2 2123.6 / mtp 430.4 /
  proposal 340.5 / **vision 282.0** MiB (333 tensors, 27 layers, hidden 1152).
- Vision has no rule in the TP split spec ⇒ the `else` at `tp_split_spec.cpp:157-159` ⇒ `Replicated` (one copy per card).
- Encoding workspace peak (`vision.cpp:88-195` layout + `merged = min(context, kMaximumVisionItemTokens=16384)` at `startup.cpp:720-725`
  ⇒ 65,536 patches): x 144 + patch_bf16 192 + qkv 432 + attention_norm 144 + mlp_up 538 + mlp_norm 144 MiB, which after scope reuse is **about 1.0–1.2 GiB** (estimate,
  ±20%), plus 160 MiB of output handoff.
- Current headroom 697 / 1217 MiB (`nvidia-smi`, 262,144 configuration). After the split:
  - shard 1: 1217 + 430 (MTP moved away) − 282 (vision) = **1365 MiB** → fits the ~1.0–1.2 GiB workspace, leaving 165–365 MiB;
  - shard 0: **stays at 697 MiB, unchanged**.
- Compared with dynamic swap-out: the peak is identical, but static assignment has zero switching, zero pinned host copies, no request-time allocation, and
  failure occurs at startup rather than at request time, so dynamic swap-out was not adopted.

### 11.2 Step A — placement by shard (independently verifiable, done first)

- `tp::TPObjectSplit` gains a `shards` mask (default `0b11`): `tp_materialize.h`.
- `object_shard_bytes`/`shard_capacity_bytes` only count the cards that will receive the object (`tp_materialize.cpp:77-115`).
- `materialize_tp2` does not allocate or copy for cards outside the mask, leaving `storage.device` empty (`tp_materialize.cpp:173-261`).
- `build_tp_split_spec`: `mtp/*` → `shards = 0b01`; `vision/*` → `shards = 0b10`
  (the name prefixes are already determinable in the same loop, so no new option is needed).
- `shard_views` skips objects this card does not have (`tp_shard_views.cpp:38-52`'s `device_parent` would fail).
- `Parameters`/`TextContext` tolerate absence: shard 1 must not dereference MTP weights, and shard 0 must not construct a VisionContext.
- **Acceptance (completed 2026-09-19, all green)**:
  - All 9 regressions pass (3 load routes, `tp2_forward` 11 probes + 32-step AR + batched prefill, `embedding`,
    `linear_tp2_split_fp8_head`, `linear_tp2_split_grouped_head`, `linear_tp2_split_nvfp4`, `tp_device_pair`).
  - Ledger: shard 1 `weights+ctx` **11494.6 → 11064.6 MiB (−430.0)**; shard 0 unchanged.
    `nvidia-smi` 15,614 / **14,664** MiB/card ⇒ shard 1 headroom **1,217 → 1,647 MiB**.
  - Greedy byte-for-byte: `ab-final.jsonl` **IDENTICAL** to the `ab-embed` reference (5/5 hashes equal) ⇒ shard placement is numerically neutral.
  - Implementation: `TPObjectSplit::shards` mask + per-card capacity/upload + order-preserving blanks in `shard_views` +
    `Model::has_weight` + `Parameters` constructing the mtp/vision blocks by presence.
  - Of shard 1's 1,647 MiB, subtracting the 282 MiB of vision weights leaves **1,365 MiB** for the encoding workspace (estimated need 1.0–1.2 GiB).

### 11.3 Step B — the TP-2 vision execution path (implemented, awaiting measurement)

**Execution layer (`execution/text.h|.cpp`)**

- New `Tp2VisionChunk`: `control` (the encoded items this chunk covers, possibly empty), `embeddings`
  (`[hidden, merged]`, belonging to the Vision card), `positions` (the prompt's full `[3, T]` MRoPE table), and `prompt_tokens`.
- `forward_tp2_prefill(..., Phase phase = Phase::Prefill, const Tp2VisionChunk* vision = nullptr)`:
  - `make_bind` builds a **`[tokens, 3]`** rope tensor for each chunk when `vision` is present (the token dimension contiguous,
    matching the single-GPU `[tokens, axes]` binding; `ne[1]==3` is the branch where the RoPE op reads an MRoPE table),
    otherwise it keeps the original `[tokens]` (cache position and rope position still coincide).
  - After `embedding_tp2`, if this chunk intersects an item: take that item's `scatter_indices` **contiguous** column segment within
    `[first_position, first_position+tokens)` (`[visual_begin, +count)`); shard 1 D2D-copies from the handoff tensor into this chunk workspace's `source`, shard 0 zeroes a same-sized
    `staging`, one `pair.allreduce(staging, source)` (adding to zero ⇒ an exact copy) is done, and then both cards each
    `ops::scatter(staging/source, local_indices, x/x_peer)`.
    The cost depends only on the chunk (≤1024 tokens ⇒ ≤10 MiB per call) and does not occupy memory once per image size.
- `PreparedPromptAccess::mutable_view` (`frontend`): the session needs a mutable prompt to release the already-encoded
  host patch payload.

**Core layer (`runtime/engine/tp2_generation_core.*`)**

- Startup: with shard 1 + `enable_vision`, build a plan using the same planner as single-GPU
  (`plan_workspace(config, params, min(capacity, 16384), kWorkspaceBytes)`) and give it a dedicated arena,
  with a new `vision %.1f` column in the ledger.
- Per request: `plan_vision_control` → assemble a `VisionPrefillPlan` (`uses` in full, `control_index==prepared_item_index`,
  `max_merged_count`) → `build_vision_control(data, plan, 0)` → `VisionPrefillSession` (shard 1).
- Prefill loop: `vision_session->prepare_chunk(t0, length)` first truncates at item boundaries, then applies the snapshot boundary
  (the rewind boundary takes precedence, and the next chunk re-enters the same item); pass the `Tp2VisionChunk` to `forward_tp2_prefill`.
- **Multimodal requests and pure-text requests go through the same mechanism** (Round 53b correction, see §11.6): MTP stays enabled as usual and
  prefix reuse works as usual; vision items enter the session only when they "fall outside the reused prefix".
- `vision_seconds` is reported separately, with `prefill_seconds = total − vision_seconds` (matching single-GPU semantics).
- `model_instance.cpp` drops the TP-2 branch's `enable_vision = false` (the DFlash restriction remains);
  the serve-side `--vision` → `EngineOptions.enable_vision` → `LoadOptions.vision` chain already existed.

### 11.4 Step C — end-to-end and the memory verdict (done, passes)

- **Criterion passes**: `load_00/03/07/11.png`, each asked "what number is in the image" → replies `00 / 03 / 07 / 11`
  all correct (`tools/tp_bootstrap/r53_numbers.py`); free-form descriptions of 3 other images match their content
  (concentric rings, colored vertical stripes, a white label box in the top-left corner). prompt 1,092 tokens, TTFT ~870 ms,
  prefill 1.57k tok/s.
- **Memory verdict**: the full 16,384 envelope **does not fit** (the first `cudaMalloc` is rejected; on this machine's WSL2
  `cudaMemGetInfo` is unreliable anyway, so the only criterion can be "was the allocation rejected"). The adaptive loop settles at an
  **item ceiling of 8,192 vision tokens** and an arena of **413.3 MiB** (`encode peak 413.3` / `handoff 80.0`).
  8,192 tokens ≈ 2,900×2,900 px, which already covers real inputs. **No new user-visible configuration was added**;
  a single image beyond the ceiling is explicitly rejected at request time, with the error pointing at the `[mem] vision` ledger line.
- The runtime peak is exactly the same as pure text: shard 0 15,614 MiB (699 free), shard 1 15,360 MiB (951 free) —
  the arena is resident, so image requests no longer make large allocations (the design intent).
- Why not 16,384: `test_vision_workspace.cpp` itself asserts that the largest item's plan is < 827 MiB
  (≈ a linear extrapolation of 8,192's 413 MiB), so it looks more like **fragmentation** preventing a ~800 MiB contiguous block than an outright capacity shortfall.
  Allocating the arena before the KV might obtain 16,384, but that would make the KV/state (4.1 GiB + 293 MiB) fail first instead,
  a worse risk than reward, so "KV first, vision adaptive" was kept.
- Throughput for images/imaging sessions is in §11.6 (68–71.5 tok/s with MTP on, better than the 59 measured for pure text).
- `ninfer_qwen3_5_vision_workspace_test` (which needs `NINFER_TEST_ARTIFACT`) is **not applicable** on this machine:
  it goes through the single-GPU Program, and a single 16 GiB card cannot hold this 27B model, so it OOMs in practice. Its assertion that the largest item plan is
  <827 MiB is cited only as an order-of-magnitude reference, not as evidence for this machine.

### 11.5 Round 53b — fixing "the whole session drops to 33 tok/s after sending an image" (done)

**Symptom and root cause**: in a multi-turn session the client puts the image back into history every turn, so **every subsequent request still carries
media**. The two scope-convergence measures the first version took to reduce risk (`use_mtp = mtp_enabled_ && !media`, and `reuse = 0` for image requests)
therefore made the whole session lose MTP and prefix reuse: decode 33 tok/s and TTFT ~0.8 s every turn.

**Fix: multimodal requests and pure-text requests go through the same mechanism**

- Remove the `!media` condition, keep MTP as usual (back to `mtp_enabled_`), and use the same flag for the decode loop / position advance / decode phase.
- Restore prefix reuse: a vision item enters the `VisionPrefillPlan` and the session only when `token_end > reuse` (outside the reused prefix),
  with `control_index = prepared_item_index - first_item` (matching single-GPU `request_plan.cpp:702-724`
  + `materialization.cpp:295-312`); items inside the reused prefix are released directly from the host patch payload.
- **When the reused prefix covers all items, no session is created** (`vision_plan.uses` empty) — this is exactly the norm
  from the 2nd turn of an imaging session onwards: no re-encoding and no scatter, but **as long as the prompt carries media, every chunk still binds the 3-axis RoPE table**
  (that is a property of the prompt, independent of whether anything is encoded).
- The condition for passing `Tp2VisionChunk` changes from "there is a session" to "the prompt carries media", with control/embeddings filled by the session.

**Measurement (shipped configuration 262,144 + `--vision`, `tools/tp_bootstrap/r53_chat.py`)**

| | prompt | cached | TTFT | decode | Answer |
|---|---|---|---|---|---|
| Turn 1 (image, fresh prefill) | 1,092 | 44 (4%) | 807 ms | **68.0 tok/s** | "07" ✓ |
| Turn 2 (follow-up text question in the same session) | 1,288 | **1,082 (84%)** | **169 ms** | **71.5 tok/s** | "21" ✓ |

- MTP acceptance `rate=170/216 = 78.7%`, higher than the 64.6% measured for pure text in Round 52.
  ⇒ MTP priming using placeholder-token embeddings for image columns (visual embeddings only enter the text KV) has
  **no measurable penalty**, so adding the complexity of "feeding shifted visual embeddings into MTP priming" was dropped.
- Cold-start greedy A/B is **byte-identical** to the golden `embed` (`v53f == embed == v53b`).
- A **pre-existing** characteristic was also pinned down (not introduced here): results are sensitive to "the rewind depth left by the previous request" — in a warm state
  A/B's req0 changes (`v53c=f4f52ac3`; the same history reproduces exactly `v53d==v53e=5b4a9783`; cold start returns to
  golden). What triggers it is a **long pure-text** request reproducing the same shift (`v53g==v53c`), showing it is unrelated to Vision/MTP:
  `rewind_near_` determines the prefill chunk split point → a different FP reduction order flips a greedy token at a near-tie.
  The shipped configuration is `temperature 0.7/top-k 20`, so this is invisible in actual use.
- Pure text is still byte-unchanged: `ab-v53b.jsonl` is **IDENTICAL** to the `ab-embed` reference (5/5).

### 11.6 Known open points

- MTP is enabled for image requests (see §11.5) with 78.7% acceptance: MTP priming uses placeholder-token
  embeddings for image columns, measured with no penalty; making it exact (feeding shifted visual embeddings into `mtp_prefill_priming`) would be about 100 lines of work,
  and there is currently no evidence of benefit.
- The per-image token limit is hard-coded in the adaptive search (8,192); if shard 1's KV or state were to give up a little room later,
  a larger envelope could be reclaimed without changing the interface.
- While vision encoding runs on shard 1, shard 0 idles during encoding (encoding is on the order of a hundred milliseconds, which is acceptable).

### 11.7 Decode bottleneck measurement (Round 53d, measurement only, no changes)

Same text prompt (prompt 75 tokens, output 256, greedy request) under the 262,144 configuration, sweeping the draft window:

| Configuration | decode | Note |
|---|---|---|
| plain (no MTP) | **32.8 tok/s** | 30.5 ms per forward (1 token per call) |
| `--spec mtp --draft-tokens 2` (shipped) | **56.8 tok/s** | ~39–41 ms/round, ~2.0–2.3 tokens/round |
| `--draft-tokens 3` | 58.5 tok/s | within noise of d2, marginally better |
| `--draft-tokens 4` | 55.5 tok/s | the marginal benefit of a longer window is already negative |

Measured during decode (`nvidia-smi -lms 300`): gpu0 **sm 2820 MHz / mem 13801 MHz / 92.8 W / util 100%**,
gpu2 2805 MHz / 13801 MHz / 84.7 W / util 99% ⇒ **full clocks, no power wall hit (180 W), no thermal throttling**,
so this is not a clock/power problem. `nvidia-smi topo -m` shows the two cards are connected as **SYS** (no NVLink/P2P; peer traffic crosses the host bridge).

**Conclusion (bottleneck breakdown)**: one round ≈ 39–41 ms, of which
- **~30.5 ms (about 74%) is one verify forward of the target model**: each card reads ~9.7 GB of weights per call, i.e. ~318 GB/s,
  only ~71% of the 448 GB/s peak; the main suspects for the gap are the per-layer row-parallel cross-card allreduce (over the SYS host bridge)
  + ~150 kernel launches without a CUDA Graph + the actual efficiency of NVFP4 GEMV on the 5060 Ti.
- **~9–10 ms (about 24%) is the MTP draft chain + window assembly + argmax/accept + state restore/fold + per-round host sync**
  (`mtp_propose_window` is a host-serial AR with one sync per draft; d4 being slower means a marginal draft costs ~3 ms).

**Exploitable list (ordered by cost/benefit)**
1. `Phase::Verify` (a decode-equivalent batched GDN/attention) — verify currently runs as approximately `Phase::Prefill`,
   and the acceptance rate is a direct multiplier on tokens/round; this is exactly what the code labels the "known-incomplete TP-2 path".
2. Cross-card collectives: reduce the number of allreduces per layer (merge the attention-out and MLP-down ones into a single call),
   or overlap them with the next layer's weight streaming; P2P is unavailable on consumer drivers, so only count/bytes can be reduced.
3. The 9–10 ms of fixed per-round overhead: move the draft chain to device-side AR (removing the per-draft host sync);
   the TP-2 core has **no CUDA Graph at all** (`cudaGraph` grep = 0), while the single-GPU route has one.
4. Retune NVFP4 GEMV for the 5060 Ti (448 GB/s, fewer SMs): use ncu to check the top kernels' DRAM achievement.
5. Long context: at 262k, fp8 KV reads are about 30% of traffic, and `--kv-dtype k8v4|nvfp4` can cut further (a quality tradeoff).
6. Not viable: overclocking/power wall (already at full clocks, 93/85 W), P2P (blocked by the driver).

### 11.8 Process-level thinking effort (Round 53c) + why there is no `--chat-template`

- **Added** `ninfer-serve --reasoning-effort low|medium|xhigh` (process default, request body takes precedence):
  `serve_options.{h,cpp}` (parsing + usage), `translate.cpp` (`effective_reasoning_effort` fallback chain:
  request field → process default → template default), `generation_service.cpp` (validates at startup that the template supports the level),
  `request_log.cpp` (`server_start` records `default_reasoning_effort`), `docs/serving.md` (flag table +
  explanation), `tests/test_serve_options.cpp` (assertions on parsing / usage / parse semantics; `ninfer_serve_options_test` passes).
  Measured: `--reasoning-effort high` is rejected (message `must be low, medium, or xhigh`); 8088 starts healthy in 11 s;
  the request line shows `thinking medium`; 17×23 answers 391 with 46 thinking tokens (the same question takes 172 at `xhigh`).
- **`--chat-template` is unavailable**, with three independent pieces of evidence:
  1. The product has no such flag (`ninfer-serve: unknown argument: --chat-template`; the CLI `ninfer` does not have it either).
  2. The template is an **artifact resource**: `frontend.cpp:194-202` enforces that
     `tokenizer_config.json.chat_template` and `resource/text/chat_template.jinja` are byte-identical, after which
     `chat_template.cpp:414-424` accepts only two semantics by sha256 — `e84f32a2…` (thinking-toggle) and
     `c3cf9e34…` (reasoning-effort, which is what this artifact embeds, so `reasoning_effort` already worked).
  3. A `chat_template.jinja` placed in the model directory is a **third** template: 28,234 B, sha256 `e57684ba…`,
     `template_version = "qwen3.8-froggeric-v22.5"`, with XML/JSON tool-call formats,
     `auto_disable_thinking_with_tools`, `max_tool_arg_chars`, `preserve_reasoning` aliases, etc. Even adding the
     flag is rejected by the whitelist, because its rendering semantics are not implemented in this engine (the
     frontend is not a general Jinja interpreter, but one hand-written C++ renderer per semantics). Using it
     = adding a new frontend semantics (including tool calls/multimodal/effort aliases), which is a separate
     feature and needs a separately confirmed scope.

---

## 12. In progress: native Windows port (branch `feat/windows-native-port`)

**Goal**: the same artifact, the same set of product code, running natively on Windows — single-GPU CLI/serve first, then a TP-2 spike.

**Platform-surface survey (Round 54, confirmed)**
- Very thin: `mmap`/`madvise`/`epoll`/`pthread`/`NUMA`/`setrlimit`/thread affinity are **all absent**; threads and synchronization use `std::thread`;
  the HTTP server is cpp-httplib, and `src/serve/http_transport.cpp` already has a `#if defined(__linux__)` structure to which a branch can be added directly.
- Code to change:
  - `src/artifact/file_io.cpp`: `::open`/`::pread`/`O_CLOEXEC` → `CreateFileW` + `ReadFile` + `OVERLAPPED`
    positioned reads (mind >4 GiB offsets, `FILE_FLAG_SEQUENTIAL_SCAN`), about 50 lines.
  - `src/product/logging/logging.cpp`, `src/serve/request_log.cpp`: `localtime_r`/`getpid`/`isatty`/`strerror_r`.
  - `src/product/media_acquire/acquire.cpp`: POSIX-socket outbound image fetch → recommend switching to the repo's existing cpp-httplib client.
- Toolchain (verified present): VS 2022 Community `D:\Program Files\Microsoft Visual Studio\2022\Community`
  (`vcvars64.bat` ✓), Windows CUDA **v13.3** (`sm_120a` support ✓; the WSL side is 13.1, keep that difference in mind when comparing).
- **Biggest unknown workload**: whether GCC extensions (`__attribute__`/`__builtin_*`) exist in `.cu`/`.cpp` — needs one MSVC+nvcc
  compile sweep to price.
- **Highest risk (decides whether a Windows TP-2 version is viable)**: the availability and performance of the `cudaHostAllocMapped` dual-GPU in-kernel allreduce
  (`src/core/tp/device_pair.cu:195`) under Windows/WDDM.
- Reference available: the three-GPU branch in a local clone of that fork @ `ar3-opt`, HEAD is
  `92758c647 ggml-cuda: add 3-way AllReduce kernel (on by default) + timing instrumentation` —
  directly relevant to our cross-GPU collective bottleneck (see §11.7).

**Steps**
1. Single-GPU `ninfer` CLI: file IO + logging misc + MSVC/CMake build; run 5 greedy prompts and compare against the Linux results.
2. Single-GPU `ninfer-serve`: HTTP, streaming, media-input behavior comparison.
3. **TP-2 spike**: whether mapped-pinned dual-GPU holds up; measure decode ms per round and compare with Linux (~39 ms/round).
4. Wrap-up: write the Windows build/run into `docs/`, convert scripts to `.ps1`, regression checklist.

**Expectation**: mechanical port 3–5 days; on performance, WDDM will very likely be slightly worse than Linux (we already get only 71% of the 448 GB/s peak,
see §11.7), so the benefit of the port is mainly **deployment convenience**, not speed.

**Progress (Round 54, 1st round, measured)**
- CMake configure got quite far on native Windows: MSVC 19.44.35228 + nvcc 13.3.73 (host=MSVC) + CUDAToolkit 13.3 all recognized successfully,
  `sm_120a` accepted, Threads found. The CMake in use is the pip-installed **4.4.2** (the repo requires ≥3.28; note CMake 4 may raise
  policy errors for `cmake_minimum_required(<3.5)` in `third_party`, add `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` if necessary).
- `cmake/NinferTargets.cmake` contains **no GCC-specific flag at all** (only CUDA's `-lineinfo`), so the previously feared
  "GCC extensions" risk looks much smaller from the build-system side than expected.
- **The only configure blocker**: `find_package(PkgConfig REQUIRED)` at `cmake/Dependencies.cmake:3`.
  It is used to find two things:
  - `FFMPEG` (libavformat/libavcodec/libavutil/libswscale, **unconditionally REQUIRED**): the 60 call sites in
    `src/media/decode/decode.cpp` are the media's actual decoders ⇒ image/video decoding depends on it, **not optional**.
  - `libcurl>=7.85` (only with `NINFER_BUILD_PRODUCT_SUPPORT`): used only by `src/product/media_acquire/acquire.cpp`
    to fetch remote http/https media (26 sites).
- The machine has neither pkg-config/pkgconf nor vcpkg/MSYS2/conda; `C:\WINDOWS\system32\curl.exe` is runtime-only.
  `C:\ffmpeg-8.1.2-full_build` has only `bin`, **no `include`/`lib` (dev files missing)**, so it cannot be used for linking directly.

**Next-step decisions (2nd round)**
1. Get MSVC-usable ffmpeg dev files: prefer gyan.dev's `ffmpeg-release-full-shared` (with `include/`+`lib/*.lib`);
   if direct connection fails, go through a local proxy per project rules; the fallback is a vcpkg source build (more time-consuming).
2. Make `cmake/Dependencies.cmake` platform-aware: keep pkg-config on Unix, use
   `NINFER_FFMPEG_ROOT` + `find_path`/`find_library` on Windows to locate the same set of libraries; handle libcurl the same way, or replace
   `media_acquire`'s fetch with WinHTTP to drop that dependency entirely (that file is itself a POSIX-socket to-change point).
3. Re-configure → first compile sweep, to price `.cu/.cpp` MSVC compatibility.
- Already obtained MSVC-linkable FFmpeg dev (BtbN win64-gpl-shared):
  `D:\ffmpeg-dev\expanded\ffmpeg-master-latest-win64-gpl-shared`, with `include/libavcodec|libswscale` and
  `lib/{avcodec,avformat,avutil,swscale}.lib` all in place. Next, wire it into the Windows branch of `cmake/Dependencies.cmake`
  (`NINFER_FFMPEG_ROOT` + `find_path`/`find_library`, the Unix side keeps using pkg-config).

**Process-occupancy discipline (user requirement)**
- Once `ninfer`/`ninfer-serve` runs on the Windows side, the same 23.7 GB artifact gets loaded a second time (VRAM + host KV/state),
  competing with the WSL-side 8088 service for VRAM and memory. **Keep the WSL 8088 service stopped during the port** (stopped in Round 54: all three GPUs'
  `memory.used` = 0 MiB, health unresponsive); start it temporarily only for cross-platform comparison measurements, and stop it immediately after measuring.
**Progress (Round 54, 1st round): the core library compiles on Windows**

Toolchain VS2022 19.44 + CUDA 13.3 nvcc + Ninja, `-DCMAKE_CUDA_ARCHITECTURES=120a`;
`tools/win_port/configure.bat` + `build.bat` (`NINFER_JOBS`, default 12). Currently apps are off
(`-DNINFER_BUILD_APPS=OFF`), `cmake --build build-win` → `BUILD_EXIT=0`, 10 static libraries
(artifact/core/engine/media_decode/model_loading/model_runtime/nvfp4_non_rdc/ops/runtime_support/text)
+ 394 object files.

Port changes (landed):

| Location | Problem | Handling |
|---|---|---|
| `cmake/Dependencies.cmake`, `src/{media,product}/CMakeLists.txt` | no pkg-config on this machine | platform branch: Windows goes through `NINFER_FFMPEG_ROOT`/`NINFER_LIBCURL_ROOT` + `find_path`/`find_library`; imported targets renamed to `ninfer::ffmpeg`/`ninfer::curl` |
| `CMakeLists.txt` | CUDA 13's CCCL requires `/Zc:preprocessor`; `windows.h`'s min/max macros | add `/Zc:preprocessor` under MSVC (CUDA via `-Xcompiler`) + `NOMINMAX WIN32_LEAN_AND_MEAN` |
| `src/artifact/file_io.{h,cpp}` | `open/fstat/pread/close`, `unistd.h`, `off_t/ssize_t` | `CreateFileW` (sequential scan / `FILE_FLAG_NO_BUFFERING`) + `ReadFile` with `OVERLAPPED` offset for positioned reads; handle type `NativeFileHandle` (`void*` on Windows) |
| `src/ops/linear/nvfp4/nvfp4_w4a4_tma.{cuh,cu}`, `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.{cuh,cu}` | C2719: a 128-byte-aligned TMA descriptor cannot be passed by value | new `Nvfp4TmaDescriptorStaging` (`cudaMallocAsync` + `cudaMemcpyAsync` into device memory, pass a pointer, `_WIN32` only); use a reference inside the kernel body, so both paths share the same code |
| `src/core/uint128.h` (new) | MSVC has no `__int128`/`__uint128_t` | new `Uint128` (`_umul128` or native 128-bit + saturating add/shift-right/compare); changed `runtime/contract/resources.h`, `runtime/engine/context_cache/context_cost.cpp`, `context_cache/materialization_planner.h` |
| `src/core/host_process.h` (new) | `getpid`/`isatty`/`localtime_r` | `host_process_id()`/`host_stderr_is_interactive()`/`host_localtime()`; `context_cost.cpp` already switched to it |
| `src/text/CMakeLists.txt` | utf8proc C2491 (dllimport defined inside a static library) | `target_compile_definitions(ninfer_text PRIVATE UTF8PROC_STATIC)` |
| `src/models/qwen3_5/execution/text.cpp` | C2397 non-constant narrowing (GCC only warns); `void*` arithmetic is a GCC extension | explicit `static_cast`; `static_cast<const std::byte*>` first, then the byte offset |

Lesson: GCC treats "non-constant narrowing" as a warning, MSVC reports C2397 outright, so code that compiles on Linux gets exposed one by one on Windows and needs an explicit cast each time.

Windows-side resources and paths:

- FFmpeg dev: `D:\ffmpeg-dev\expanded\ffmpeg-master-latest-win64-gpl-shared` (BtbN shared, with `include/`+`lib/`).
- libcurl: `D:\curl-dev\expanded\curl-8.22.0_1-win64-mingw` (**only `lib/libcurl.dll.a`, the mingw build**; whether MSVC can link it directly is still to be verified, otherwise switch to vcpkg or WinHTTP).
- Model artifact: `<models-dir>/qwen3_8_27b_nvfp4.ninfer` (23,719,715,076 B, the same artifact as on the WSL side, so it can be used directly for cross-platform comparison).
- Custom template: `<models-dir>/chat_template.jinja` (the same 28,234 B file as on the WSL side).

Next (apps off→on): `src/product/logging/{logging,startup_log}.cpp` (`unistd.h`/`localtime_r`/`isatty`/`sys/ioctl.h` TIOCGWINSZ →
`host_process.h` + `GetConsoleScreenBufferInfo`), `src/serve/request_log.cpp` (`getpid`),
`src/product/media_acquire/acquire.cpp` (POSIX socket + libcurl), `src/serve/http_transport.cpp` (`#if defined(__linux__)` branch);
after that, align the single-GPU `ninfer` CLI output with Linux (greedy baseline hashes see §11), then move on to the TP-2 mapped-pinned spike.
**Progress (Round 54, 2nd round): CLI / serve all compile and link, and can start on Windows**

Configure with `-DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_FFMPEG_ROOT=… -DNINFER_LIBCURL_ROOT=…` →
`BUILD_EXIT=0`, artifacts `build-win/apps/{ninfer,ninfer-serve,ninfer-perplexity}.exe` (~188 MB each);
`ninfer.exe --help` and `ninfer-serve.exe --help` both print normally and exit with code 0 (showing that DLL/CRT/linking are in place).

New fixes this round:

| Location | Problem | Handling |
|---|---|---|
| `CMakeLists.txt` | spdlog's bundled fmt requires `/utf-8` | add `/utf-8` under MSVC (CUDA via `-Xcompiler`) |
| `src/core/host_process.{h,cpp}` | `ioctl(TIOCGWINSZ)`, `gmtime_r` | new `host_terminal_columns()` (`GetConsoleScreenBufferInfo`) and `host_gmtime()`; `src/core/CMakeLists.txt` registers `host_process.cpp` |
| `src/product/logging/{logging,startup_log}.cpp`, `src/serve/request_log.cpp` | `localtime_r`/`isatty`/`getpid`/`unistd.h` | switch to the platform functions in `core/host_process.h` |
| `src/product/media_acquire/acquire.cpp` | `<arpa/inet.h>` and other POSIX headers; `relative.native().starts_with("..")` (Windows is wide-char) | platform branch `<winsock2.h>/<ws2tcpip.h>` + one-time `WSAStartup`; instead test whether the first path component is `path("..")` |
| `src/product/CMakeLists.txt` | unresolved WinSock symbols | link `ws2_32` |
| `cmake/Dependencies.cmake` | `find_library` selects the **static** `libcurl.a` (mingw static archive, MSVC cannot link it) | prefer `find_file(... libcurl.dll.a)` to take the import library, fall back to `find_library` if not found |
| `apps/perplexity/main.cpp` | `gmtime_r` | change to `ninfer::host_gmtime` (that file sits in a global anonymous namespace, so the qualified name is required) |

Verified: MSVC can link and **run** the mingw `libcurl.dll.a` + `libcurl-x64.dll` (probe `tools/win_port/probe_curl_link.bat`,
prints `curl libcurl/8.22.0 LibreSSL/…`, exit code 0), so `media_acquire` keeps using libcurl on Windows and a WinHTTP rewrite is not needed for now.

Runtime environment (important): both exes need `PATH` to contain

    D:\ffmpeg-dev\expanded\ffmpeg-master-latest-win64-gpl-shared\bin;D:\curl-dev\expanded\curl-8.22.0_1-win64-mingw\bin

(`libcurl-x64.dll` must be visible). Before starting, confirm there is no second model process: `nvidia-smi memory.used` is 0 MiB on all three GPUs.

Next: real single-GPU CLI generation (`<models-dir>/qwen3_8_27b_nvfp4.ninfer`) and alignment with the Linux greedy baseline hashes → serve end-to-end →
TP-2 mapped-pinned spike (the key thing to verify is whether `cudaHostAllocMapped` in `src/core/tp/device_pair.cu` is usable under WDDM).
**Progress (Round 54, 3rd round): TP-2 runs on Windows, performance on par with Linux; mapped-pinned spike passes**

End-to-end measurements (`tools/win_port/serve.ps1`, two 5060 Ti GPUs, artifact `<models-dir>/qwen3_8_27b_nvfp4.ninfer`):

| Item | Linux (WSL2) | Windows |
|---|---|---|
| TP-2 pure decode | 32.8 tok/s | 32.9 tok/s |
| TP-2 MTP `--draft-tokens 2` | 56.8 tok/s | 56.9 tok/s |
| Image-turn decode | normal after the 53b fix | 60.4 tok/s, MTP sustained (rate=106/174) |
| Engine load (23.7 GB artifact) | — | 35–42 s |
| Real agent traffic (streaming / tools / images) | — | served, 41–73 tok/s |

KV capacity and VRAM headroom (measured, with `--vision`): capacity 8192 → kv 145 MiB, free 4180/3930 MiB;
capacity 131072 → kv 2322 MiB, free 2002/1580 MiB. **131072 is recommended on Windows** (262144 is the WSL recipe;
under WDDM the headroom is too small).

**Progress (Round 10): same-machine comparison against llama.cpp `ar3-opt` — the reason decode is a draw is not the kernels**

New/modified scripts: `tools/win_port/bench_serve.ps1` (drives any OpenAI-compatible service with the same workload, all numbers taken from the server's self-reported `timings`),
`tools/win_port/serve_llama.ps1` (freezes the llama.cpp recipe tuned on this machine, includes `-Stop`/`-Status`; mutually exclusive with `serve.ps1`, only one engine is resident at a time),
`serve.ps1` gains `-DraftTokens`. Protocol: temperature 0, 256 output, medium thinking, MTP on for both, same prompts, one engine at a time.

| Workload | ninfer | llama.cpp | Ratio |
|---|---|---|---|
| prefill 1.9k | 1,401 tok/s | 764 tok/s | 1.83x |
| prefill 7.2k | 1,667 tok/s | 1,040 tok/s | 1.60x |
| prefill 28.5k | 1,552 tok/s | 1,016 tok/s | 1.53x |
| decode 256 greedy tokens (3 runs) | 56.7 / 56.7 / 56.7 tok/s | 51.7 / 52.6 / 52.7 tok/s | 1.08x |

Per-round budget on the same decode prompt: ninfer K=2 38.2 ms/round, 2.18 committed, 58.8% single-draft acceptance rate; ninfer K=3 40.7 ms, 2.27, 42.4%;
llama.cpp `n_max=3` 49.4 ms, 2.60, 53.8% (llama.cpp reuses a CUDA graph 95–96 times per 256 tokens).

Conclusions (full argument in the Cross-engine comparison section of `docs/tp2-dual-5060ti.md`):

1. **batch=1 decode is a weight-streaming problem, not a compute problem**: `tok/s = committed per round / round duration`, and the round-duration floor = weight bytes per GPU / 448 GB/s.
   ninfer does 10.15 GB/forward per GPU, llama.cpp 8.56 GB (measured with `llama-gguf`, its GGUF 15.94 GiB type distribution: NVFP4 MLP 10.38 GiB,
   q5_K attention/GDN 3.09 GiB, q6_K vocabulary 1.55 GiB, q8_0 output head 1.29 GiB). The two land at 59% / 39% of their respective floors.
   Two 5060 Ti cards total 896 GB/s = half of one 5090 (1,792 GB/s), which is exactly why the same artifact does 71.2 tok/s on a 5090 and ~57 tok/s here:
   **TP-2 is a capacity decision (a 27B does not fit on a single 16 GiB card), not a speed decision**.
2. ninfer's targeted optimizations do take effect where there is compute to spare: 23% faster per round than llama.cpp, 1.5–1.8x faster prefill.
3. Four things eat that advantage: **draft depth** (2.18 vs 2.60; on this artifact K=3 actually drops to 55.7), **FP8 weights read ~26% more bytes**,
   **TP-2 turns off the CUDA graph** (`model_instance.cpp:99` `use_cuda_graph=false`, ~1,009 kernels/forward, ~2.9 ms launch gap)
   and **head splitting doubles allreduce to 128 per token**, the fused projection kernel is bypassed on sharded rows (`text.cpp:1104-1113`),
   and **the MTP chain serializes 7–10 ms per round on the host** (`mtp_propose_window` does one `cudaStreamSynchronize` per draft).
4. Follow-up optimization order (by evidence strength): (a) put an exact-batch CUDA graph on TP-2's verify forward (~7.6%/round);
   (b) switch attention/GDN/embedding/head and the last 8 layers' FFN from FP8 to NVFP4 (−2.6 GB per GPU per token, about +18%, at an accuracy cost);
   (c) reduce the number of collectives (no head splitting, or merge the two ARs of the same layer); (d) overlap the draft chain with the next round's verify or sink it to the device.

**Round 10 addendum: theoretical gain accounting for optimizations ①③④ (model estimate, not measured)**

Itemized accounting for a 38.2 ms / 2.166 committed decode round (=56.7 tok/s): weight streaming 25.6 ms (448 GB/s theoretical 22.7 ms, efficiency 89%)
+ allreduce 1.7 ms (128 × 9–18 µs, already optimized down from 42 µs) + launch gap 2.91 ms (1009 kernels/forward, measured with nsys)
+ MTP chain / accept-fold / host sync 7.7 ms (= round duration − verify 30.5 ms).
Itemized accounting for prefill at 660 ms per 1024-token block (converted to Windows from the worklog's Round 35c figure of 580 ms = AR 350 + MMA 188 + other 35):
AR ≈350 ms (53%), MMA+other ≈310 ms; AR is already at the Gen4 x4 link floor (2.6 MB/token ÷ 7 GB/s → 2.67k tok/s).

| Item | decode theoretical gain | prefill theoretical gain |
|---|---|---|
| ① exact-batch CUDA graph | launch gap 2.91 → 0: 38.2→35.3 ms, 61.4 tok/s (+8.3%) | intra-block gaps amortized, ≈ −0.4% |
| ③ collective overlap (**not removal**, TP needs it) | −1.7 ms (overlap with the draft chain): →33.6 ms, 64.5 tok/s | AR overlapped with MMA: 660→≈390 ms, **≈2.6k tok/s** (capped by the link's 2.67k) |
| ④ draft chain sunk/overlapped (keep ~3 ms on device) | −4.7 ms: →28.9 ms, **74.9 tok/s (+32%)** | not applicable (prefill does not go through MTP) |

Combined theoretical value: decode **≈75 tok/s** (absolute floor 22.7+3.0=25.7 ms → 84 tok/s); prefill **≈2.6k tok/s** (the worklog's own conservative estimate 1.5x → ≈2.1k).
Boundary on ③: the worklog has already done the head-splitting math — going from 64 to 128 ARs costs 4.9 ms but buys back 9.4 ms of compute, a net gain;
so "reduce the number of ARs" would have to give up weight splitting, which always loses on decode — the only correct direction is "overlap".

**VRAM accounting for ①③④** (per GPU; measured on Windows 131072 free 2002/1580 MiB, 262144 recipe free 0/196 MiB):

| Item | Increment/GPU | Mechanism |
|---|---|---|
| ① CUDA graph | ≈ 0–a few MiB (worst case ≈ 192 MiB if a graph-private pool is used) | all arenas are resident from launch (workspace 192.0, state 293.6, KV, weights); the graph only adds exec and node bookkeeping; the hard prerequisite is moving the AR arrival token from mapped host to the device (KiB-scale). Risk seen in this repo before: a `cudaMallocAsync` inside capture becomes a graph memory node (that TMA-descriptor incident), so everything allocated during capture must be pre-allocated |
| ③ AR∥MMA sub-block pipeline | **+190–580 MiB** | 2–4 in-flight sub-blocks each need an activation region (workspace 192 MiB × pipeline depth) and a GDN state snapshot (state 293.6 is made of 4–5 copies, ≈59–73 MiB each). AR staging itself is `cudaHostAllocMapped` host pinned (2×24 MiB × 2 buffers = 96 MiB/GPU), and scaling it only consumes host memory |
| ④ draft chain sunk/overlapped | only removes host sync ≈ 0–tens of MiB; making verify and draft **concurrent** requires a second MTP KV: **+258 MiB** (shard 0 only, 131072) | MTP weights 430 MiB are already replicated on both GPUs, MTP KV already exists (516.2 MiB at 262144, shard 0 only) |

Total **+0.2–0.85 GiB/GPU**: 131072 fits (headroom drops to ≈1.4–1.8 GiB on shard 0, ≈1.0–1.4 GiB on shard 1),
262144 does not fit (free 0/196). Conversely, ② NVFP4-ifying weights would **save** ≈2.15 GiB/GPU — doing ② together with ③ means both speeding up and fitting 262144 back in.

**mapped-pinned spike (go/no-go passed)**: `tools/win_port/spike.ps1` (source `tp_mapped_spike.cu`) measured
peer access 0→1 / 1→0 both 0 (SYS topology, no P2P); `cudaHostAllocPortable|cudaHostAllocMapped` +
`cudaHostGetDevicePointer` succeed on both GPUs and peer sees the **same UVA address**; cross-GPU reads of mapped host memory verify
`sum=match`, bandwidth about 1.4–1.9 GiB/s (local write 5.8 GiB/s). Conclusion: under WDDM, in-kernel mapped-pinned allreduce
is usable and in the same class as WSL2, which is exactly why end-to-end decode is on par with Linux.

**Important environment fact**: `$HOME\.dsh\settings.yaml` already configures the `ninfer-win` provider
(`baseURL: http://127.0.0.1:8099/v1`) and `agent-default-model` points at it. In other words: on the Windows side,
ninfer-serve running on 8099 is this machine's DSH default model service; during testing it already served real streaming/tools/image requests
(`req#4 media 1, prepared 7.91 ms`). The WSL-side 8088 is stopped, and there is only one model process on the whole machine.

**Script lesson**: a service started with `Start-Process` is a "child process of the calling shell's job object" and gets killed along with it
when the agent tool call ends (symptom: the process is there, GPU 0%, the model was never loaded). Therefore `serve.ps1` by default runs in the
**foreground** (the terminal user owns the process, Ctrl+C stops it), and only `-Background` uses `Start-Process`; when testing the service yourself you must use the harness's background job (which survives across calls).

Tools and docs: `tools/win_port/` is now all PowerShell (`vcvars/build/serve/spike/fetch_ffmpeg/fetch_curl`,
the old `.bat` files are deleted); new `docs/windows.md` registered in `docs/README.md`; `docs/windows.md` covers build, run,
platform differences, measured data and limitations (the single-GPU CLI needs ≥20.2 GiB VRAM, and a 16 GiB card on this machine can only go through TP-2).
**Progress (Round 54, 4th round): commit and regression verification**

- Commit: `453aaa6a feat: build and serve natively on Windows with VS2022 and CUDA 13.3` (branch
  `feat/windows-native-port`).
- Linux regression: `build_r35.sh` → `BUILD_EXIT=0` (full rebuild + test targets). Fixed one **test-infrastructure issue**:
  the rsync list in `build_r35.sh` was missing `cmake` and the top-level `CMakeLists.txt`, so the WSL tree still used the old
  `cmake/Dependencies.cmake` (`PkgConfig::FFMPEG` vs the new `ninfer::ffmpeg`) and configuration failed; now added
  (note: that script syncs itself, so a change only takes effect on the **second run**).
- Affected Linux tests: `context_cost`, `resource_manager`, `pretty_logging`, `context_cost_measure`
  all pass; `artifact|nvfp4|qwen3_5` 25 of 26 pass.
- **The only failure is a pre-existing issue (unrelated to this branch)**: `ninfer_qwen3_5_frontend_test` throws
  `unsupported frontend/chat_template.jinja (sha256 821b1c036748885c26b552d2a174878fe06199c390ccf612a6169326338cbaa5)`
  — i.e. the template digest whitelist introduced by 53c rejects the test fixture; `src/models/qwen3_5/chat_template.cpp` and `tests/`
  are both unmodified on this branch, so it fails on HEAD as well.
- Lesson: before relinking an exe on Windows you must stop the service first (a running exe locks the file → `LNK1104`).

**Remaining gap (user decided not to do it)**: the "run the single-GPU ninfer CLI" item cannot be literally completed on this machine — the CLI supports only a single GPU, while that
27B NVFP4 artifact needs about 20.2 GiB of resident weights (10.1 GiB/shard × 2), which does not fit on a 16 GiB card (already written into the
limitations section of `docs/windows.md`). To actually reach end-to-end, a small `.ninfer` test artifact would have to be built (conversion toolchain +
synthetic small checkpoint), which is extra work; the current CLI binary itself has been verified runnable (`--help`), and it shares the same
engine/loader as serve (already verified end-to-end).
**Progress (Round 54, 5th round): the Windows test suite passes, and one real port defect was caught**

- New `tools/win_port/test.ps1`: put FFmpeg/curl's `bin` on PATH first, then run ctest. **This is mandatory**: the test exes load
  avcodec/swscale/libcurl at runtime, and when PATH is missing the Windows loader pops a "DLL not found" dialog for each exe instead of reporting a test failure.
- The full Windows test target set compiles and links (253 targets, `BUILD_EXIT=0`). Platform branches added to make the tests pass:
  `test_pretty_logging` (`pipe/dup/dup2/read` → `_pipe/_dup/_dup2/_read` + `_setmode(_O_BINARY)`),
  `test_context_cost`/`test_request_log` (`getpid` → `ninfer::host_process_id()`), `tests/artifact/fixture.h`
  (`mkdtemp` → unique-directory-name loop), `test_gdn_replay_records` (`std::aligned_alloc` → `_aligned_malloc`),
  `test_host_timing` (`#undef near`: a leftover empty macro in winnt.h), 3 instances of `constexpr ... std::sqrt` (MSVC does not fold them),
  1 explicit `std::array` CTAD, and `<array>` added to 4 files.
- The only platform exclusion: the CUDA fault injection in `ninfer_artifact_materialization_test` depends on GNU ld `--wrap`, and the MSVC linker has no
  equivalent; `tests/artifact/tests.cmake` uses `if(NOT MSVC)` to exclude that source file and the `--wrap` option, all other checks run as usual.
- Python tool platform branches: `tools/artifact/file_io.py` (`os.sysconf`/`posix_fadvise`/`fdatasync` used when present, otherwise
  degraded: dropping the page cache becomes a no-op, `fsync` as fallback; new `read_at`/`write_at` wrapping `os.pread`/`os.pwrite`),
  `reader.py`/`writer.py`/`convert/sources/safetensors.py` switched to use them.
- **Real product defect (caught by the test suite)**: Windows `open_handle` used only `FILE_SHARE_READ`, which made it **impossible to rewrite/replace/delete the file while reading the artifact**
  (POSIX read semantics allow it), manifesting as `ninfer_artifact_reader_test` throwing
  `ios_base::failbit`. Changed to `FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE`, aligned with POSIX read semantics.
- Regression: Windows `artifact` 3/3 pass; Linux 23/23 pass (`artifact|context_cost|pretty_logging|request_log|
  host_timing|resource_manager|kv_cache|gdn|attn_input_proj|gated_delta`), Linux full build `BUILD_EXIT=0`.
**Progress (Round 54, 6th round): fixed a real Windows A4 TMA defect (CUDA Graph capture)**

- Symptom: `ninfer_linear_nvfp4_a4_test` throws `cudaErrorIllegalInstruction` on Windows; `CUDA_LAUNCH_BLOCKING=1`
  localizes it to `cudaGraphLaunch` at `src/core/decode_graph.cpp:144` — i.e. the failure happens in the **post-capture CUDA Graph**.
- Root cause: in round 4, to work around MSVC C2719, the tensor map was changed to "`cudaMallocAsync` device staging + `cudaMemcpyAsync` from a **stack** host object".
  That route does not hold under graph capture: the stream-ordered allocation becomes the graph's memory node (the address is only
  determined at instantiation), while the pageable source pointer is long invalid at replay → TMA reads a garbage descriptor → illegal instruction.
  Linux is unaffected (there it is `__grid_constant__` pass-by-value).
- Fix: on Windows, store the descriptor in a **mapped pinned host arena** (`cudaHostAllocMapped|Portable` +
  `cudaHostGetDevicePointer`, 32 slots rotating; the host writes directly, TMA reads through the same UVA address), and the setup path
  **issues no CUDA call at all**, so it is capture-safe; the kernel side publishes the descriptor to the
  tensor-map proxy with `fence.proxy.tensormap::generic.release/acquire.gpu` (PTX ISA 8.3, SM_90+). Both the linear and linear_swiglu A4 kernels were changed together.
- Result: `ninfer_linear_nvfp4_a4_test` passes on Windows; the Windows linear/TMA surface is 30/30, the same Linux surface 29/29.
- Full Windows suite (124 items) was 121 pass/3 fail before the fix, and a4 is now fixed; `ninfer_context_kv_materialize_test` hit a 300 s timeout when the whole round
  ran serially but passes when run alone (judged interference, not a defect); `ninfer_qwen3_5_frontend_test` has the same
  pre-existing template-whitelist problem as on Linux.
**Progress (Round 54, 7th round): evidence and ruling on the single-GPU CLI clause**

- Facts: on Windows `ninfer.exe` runs and handles arguments correctly (`ninfer_cli_options_test` **exits 0** on that platform), and the loading/engine/KV/graph/sampling beneath the CLI
  are all the same engine already verified end-to-end by the TP-2 server (Windows and Linux performance on par).
- Criterion: the machine's only artifact (27B NVFP4) needs ~20.2 GiB of resident weights for a single GPU vs 16.3 GiB of VRAM, and there is **no pre-allocation capacity check**
  (grep found no "does not fit"-style pre-check), so forcing a single-GPU run would make WDDM page the missing ~4 GiB to main memory instead of failing cleanly —
  exactly the risk the user asked to avoid, so it is **not executed**.
- Conclusion: that clause is a hardware constraint (not a port gap). To satisfy it literally, a small artifact would have to be built: there is no torch on the Windows side (miniconda
  3.13 does not have it installed), the WSL side has the project's py311 environment and can convert, but a tiny-model recipe still has to be written — a separate sub-task, pending the user's decision
  on whether to invest.
**Progress (Round 54, 8th round): plan for the single-GPU CLI small-artifact sub-task (cross-compression memory)**

Goal: produce on this machine a `.ninfer` small enough to fit on a single 16 GiB card, then use the Windows `ninfer.exe` to run single-GPU
"load → generate → sample" end-to-end, providing measured evidence for the "single-GPU CLI works" clause.

Confirmed thresholds and facts (do not re-investigate):
- The machine's only artifact: `<models-dir>/qwen3_8_27b_nvfp4.ninfer` (27B NVFP4, single GPU needs ~20.2 GiB > 16.3 GiB); inside the engine
  `src/runtime` there is **no** pre-allocation capacity pre-check, so forcing a single GPU gets paged by WDDM → attempting it is forbidden.
- Windows Python is miniconda 3.13, **no torch**; the WSL side has no miniconda/py311 (in AGENTS.md that refers to the upstream author's machine).
- Converter entry point `python -m tools.convert` (`tools/convert/__main__.py`), official recipes are only 27B-class
  (`qwen3_8_27b_nvfp4` etc.), so a tiny model needs a self-written recipe (`--recipe my_recipe.py[:configure]` or `--override`).
- Docs: `docs/weight-conversion.md` (recipe/source/override semantics, safetensors sources).

Proposed steps (each independently verifiable, revertible on failure):
1. Install **CPU-only** torch (avoid the 2.5 GB CUDA wheels): prefer `pip install torch --index-url
   https://download.pytorch.org/whl/cpu`; if direct connection fails, use a local proxy. If not viable, switch to
   a "pure-numpy synthetic `LogicalSource`" and bypass torch.
2. Read the `LogicalSource` interface in `tools/artifact/*.py` + the usage in `tools/convert/sources/safetensors.py` to determine what the synthetic source
   must provide (tensor names/shapes/bytes).
3. Following the validation in `src/models/qwen3_5/config.cpp` and HF naming, write a tiny config (e.g. 2 layers, hidden 1024,
   head_dim 128, q/kv heads satisfying the divisibility constraint), generate random weights (numpy/torch, fp32/bf16 is fine).
4. Convert a small artifact with `tools/convert` (first try `--override` changing only source/output paths, write a tiny recipe if necessary).
5. Run single-GPU on Windows: `ninfer.exe <tiny.ninfer> --prompt "..." --max-new 8`, record load time, whether generation happened, exit code;
   if the engine has hard constraints on tiny shapes (static asserts/blocksize), raise the config to the smallest workable values per the errors.
6. On success, write it into `docs/windows.md` (single-GPU CLI measured data) and commit; if it does not converge for a long time, record the clause as a hardware+toolchain limitation.
**Progress (Round 54, 9th round): sub-task prerequisites ready**

- CPU-only torch already installed in `build-win\torch-venv` (gitignored, base environment untouched): **torch 2.14.0+cpu / numpy 2.5.3** ✓
  (`build-win\torch-venv\Scripts\python.exe`).
- Converter interface confirmed: `python -m tools.convert --model <checkpoint directory> --recipe <official name|file[:func]> --out X.ninfer
  [--components text] [--override file.py] [--source NAME=PATH] [--name NAME] [--device DEVICE]`;
  `--model` is the "main checkpoint/config + default resources".
- Next (by cost, low to high): ① first run a tiny checkpoint directly with the official recipe (`--recipe qwen3_8_27b_nvfp4
  --components text`); if it maps layer by layer driven by config, it may succeed with zero changes; ② if not, write a tiny recipe/override;
  ③ put the generation script at `tools/win_port/tiny_model.py` (synthetic tiny config + random bf16 weights, HF naming).

---

## Round 11: implementation of ①③④ (TP-2 dual-GPU optimization)

Goal (explicitly requested by the user): implement the three items ①③④ from the PLAN §12 table on `feat/windows-native-port`, **do not do ②** (NVFP4-ifying weights,
which requires a new artifact + per-shape Op admission and has been judged a deployment-time variant rather than a runtime switch).

Definitions of the three items and their theoretical values (accounting at PLAN:640-667; baseline decode 38.2 ms/round, 2.166 committed, 56.7 tok/s; prefill 1552 tok/s@28.5k):

| Item | Content | decode theoretical | prefill theoretical |
|---|---|---|---|
| ① | exact-batch CUDA graph on the TP-2 verify forward (currently disabled by `model_instance.cpp:99`) | launch gap 2.91→0 ms: 35.3 ms / 61.4 tok/s | intra-block gaps amortized ≈ −0.4% |
| ③ | collective and compute **overlap** (not removing AR; removing AR loses, see PLAN:655) | −1.7 ms (AR overlapped with the draft chain) | AR∥MMA: 660→≈390 ms / ≈2.6k tok/s |
| ④ | MTP draft chain sunk/overlapped (one cudaStreamSynchronize per draft, see tp2_generation_core.cpp:543) | −4.7 ms → 28.9 ms / **74.9 tok/s** | not applicable |

Per-round budget accounting (PLAN:642-644): weight streaming 25.6 ms (448 GB/s floor 22.7) + AR 1.7 ms (128 calls) + launch gap 2.91 ms
(1009 kernels/forward) + **MTP chain/accept/fold/host sync ≈ 8.0 ms**. prefill 660 ms per 1024-token block = AR ≈350 + MMA/other ≈310.

VRAM accounting: total **+0.2–0.85 GiB/GPU**; the 131072 recipe (free 2002/1580 MiB) fits, 262144 (free 0/196) does not.

### Confirmed interface facts (do not re-investigate)

- `forward_tp2_prefill` (src/models/qwen3_5/execution/text.cpp:1766) is TP-2's only chunked forward entry point, and the decode round's verify
  also goes through it (Phase::Prefill, see tp2_generation_core.cpp:999-1005). It has only three host dependencies:
  `copy_i32(ids.data(), b.ids, stream)` (:1826), `ops::fill_i32_positions(b.positions, first_position, stream)` (:1827),
  `b.envelope = {visible_end, visible_end}` (:1823). Device-driven/graph-ification must turn the first two into device tensors and
  turn envelope into a per-bucket capture constant.
- The single-GPU path's graph mode can be copied directly: src/models/qwen3_5/program/graph_execution.h (capture_graph/run_prepared),
  program/decode.cpp:24-63 (one **pinned** host↔device memcpy each at the head and tail of the captured body, program_impl.cpp:62 allocates
  PinnedHostBuffer), program/planning/graph_profiles.cpp:62-89 mtp_graph_profiles (capacity 131072, at K=2
  ends = 124/508/2044/4092/8194/16386/32764 → **8 buckets**, envelope captured at the bucket maximum, the per-round real position carried by a device tensor).
- `DecodeGraphDefinition::capture` / `DecodeGraphExecutable::instantiate|launch` are at src/core/decode_graph.cpp:59-146,
  platform-independent, so TP-2 can use them directly (one definition/executable per GPU).
- AR is `cudaHostAllocMapped|Portable` host pinned + kernel spin (src/core/tp/device_pair.cu:90-143);
  the two GPUs just each capture and then launch alternately, no change to the AR protocol needed.
- Windows has no nsys/ncu (under C:\Program Files\NVIDIA Corporation there is only App/FrameView), so performance attribution can **only use
  in-engine CUDA events**.

### Execution order (each step independently testable and revertible)

1. **Measure**: add env-gated phase timing (host clock + CUDA events) to the TP-2 decode round, confirm the attribution of the 8.0 ms MTP chain/accept/fold,
   and reproduce the 56.7 tok/s baseline.
2. **④**: de-hostify the draft chain — keep drafts on the device, feed window tokens directly to the device-side forward, drop redundant D2H/H2D and per-draft synchronization.
3. **①**: capture the whole round into two graphs (one per GPU), pinned ingress/egress, captured per the mtp_graph_profiles buckets; each round only fills ingress + launches.
4. **③**: overlap prefill's AR with MMA (sub-block pipeline; needs two sets of activation regions and two GDN state snapshots, see the VRAM accounting at PLAN:663).
5. Verification and documentation: greedy A/B (token-by-token identical to before the change) + `bench_serve.ps1` measurements + update docs/tp2-dual-5060ti.md / worklog.

### Progress

- [x] Plan written to disk (this section).
- [ ] Step 1 measurement.

#### Round 11 reconnaissance conclusions (subagent read-only investigation, do not repeat)

- **Layer loop**: `run_layers_tp2` is a template at text.h:411-458 (not in text.cpp). The per-layer order is fixed:
  `tp_mixer_layer` (dual GPU, row-parallel output projection writing partial deltas) → **AR#1** (text.h:436) → `residual_add`×2 →
  `tp_mlp_delta` → **AR#2** (text.h:449) → `residual_add`×2. Within a layer there is **no host synchronization/readback at all** (grepping the whole
  `src/ops` directory for `cudaStreamSynchronize|DeviceToHost` gives zero matches).
- **allreduce** (src/core/tp/device_pair.cu:289-395): signature `allreduce(void* a, void* b, bytes, stream_a, stream_b)`;
  on the in-kernel path the host side only issues two kernels back to back with no synchronization; the epoch is `token = ++ar_call_` (monotonically
  non-decreasing, device_pair.h:64), double-buffered by `token&1` (24 MiB/GPU), with an inter-block write-order chain `order_mine[b-1]==token` and the
  peer spinning on `arrival_other[b]!=token`.
  **Graph hazard**: token is a **kernel argument**, so capture freezes it, and replay reusing an old token lets the peer's spin pass early (reading the
  previous round's stale data) → before going to graph, token must become a **device-side counter** (`increment_i32_scalar` or an in-kernel increment).
- **positions**: `fill_i32_positions` accepts only a host base (src/ops/kernel/position.cuh:7); `offset_i32_positions`
  supports **[1] device delta broadcast to [T]** (kernel: `dst[i]=src[i]+delta[0]`) → the device-ified path = static arange + device base.
- **fold**: `GdnReplayFoldPlan::execute(std::span<const GdnReplayFoldRow>, stream)` does **host-side validation and packing**, with
  `commit_columns` going into `__grid_constant__` kernel arguments (replay.cpp:266-325, recurrent.cu:104-128); the header itself says
  "changing the host row descriptor requires re-capture" → **fold cannot go into the graph with a varying commit count**, so it stays eager this round.
- **state accounting**: state pool `slot_count=1`, live backing = **73.4 MiB**; the 293.6 MiB in PLAN = live + 3 snapshots
  (2 prefix-reuse + 1 round scratch). Each round's snapshot+restore = 2×73.4 MiB D2D/GPU (not 293.6×2).
- **accept**: `speculative_accept_greedy_drafts` already outputs a device-side `accepted`, but **nobody reads it**; the commit count always comes from the host output policy.
- **host dependencies of forward_tp2_prefill** (text.cpp:1766-2021): `copy_i32(ids)` H2D (:1826),
  `fill_i32_positions(b.positions, first_position)` (:1827), `envelope{visible_end,visible_end}` (:1798/:1823,
  host-side decision of attention routing/chunk/split); the `Phase::Verify` branch (:1931-1965) is **not** taken on MTP rounds (those go through Prefill).
- `set_i32_scalar` only supports a host value (scalar.cu:9); `assign_i32_scalar` is a D2D memcpy; `increment_i32_scalar` is device-side.
- **VRAM of the verify window**: T=3 at K=2, activation peak only a few MB (the largest item is `logits_columns` [248320,3] BF16 = 1.5 MB),
  so the graph-private workspace does **not** need 192 MiB (the worst-case estimate at PLAN:662 is overly conservative).
- **Prerequisite for ③'s sub-block pipeline**: all op shapes are generalized (arbitrary T), but (a) the token range is bound once per chunk by forward_tp2_prefill
  (positions/envelope/kv_table_rows), (b) the GDN conv/recurrent state advances in source/destination slot order,
  (c) the full-attention sub-block 2 query must see the K/V already appended by sub-block 1 ⇒ the per-layer dependency chain L,A→L,B→L+1 must be preserved;
  what can be done is **slot-wise A→B pipelining** (while AR uses the link, the other sub-block uses the SMs).
- **Pitfall (already fixed)**: when adding CUDA event timing to the decode round, an event is bound to the **current device at creation time**; if that is not
  the same device as `shard_a_.device.stream`, `cudaEventRecord` returns `cudaErrorInvalidResourceHandle`,
  and that error is **latched** into the next unrelated `cudaGetLastError()` (manifesting as a crash in `set_i32_scalar` at `scalar.cu:10`).
  Fix: create the events only after binding the shard A device, and on a timing-call failure discard the error and self-disable.






## Round 11 progress: ① complete (measured), plus one real cross-GPU race fixed

### Step 1 measurement (complete)

Added env-gated phase timing to tp2_generation_core.cpp (NINFER_TP2_TIMING=1, 7 CUDA events + host clock).
Baseline (graph off = eager path, same recipe as bench_serve.ps1):

    decode rounds=52 committed=127 avg_round=38.90ms mtp=3.70 verify=34.23 accept=0.08 copy=0.09 sync_wait=0.01 fold=0.00

I.e. 38.9 ms/round = MTP chain 3.70 + verify 34.23 + accept 0.08 + copy 0.09 + host policy/wrap-up (remainder about 0.8 ms).
verify accounts for 88%, and its launch/scheduling overhead measures larger than the 2.91 ms at PLAN:642; the MTP chain at 3.70 ms is the second-largest item.

### Step 3 ① exact-batch CUDA Graph (complete)

Implementation locations: src/core/decode_graph.{h,cpp}, src/core/arena.{h,cu}, src/models/qwen3_5/execution/text.{h,cpp},
src/runtime/engine/tp2_generation_core.{h,cpp}.

- DecodeGraphDefinition::capture_group(defs, streams, body): capture multiple streams simultaneously; each graph records only the kernels on its own stream;
  cross-GPU ordering still relies solely on the in-kernel AR spin, with no host involvement.
- TextContext::forward_tp2_window(...): the capture-safe variant of the prefill forward. Per-round inputs (ids/positions) go through
  portable pinned host buffers (captured as memcpy nodes, re-read on every replay), and the attention envelope becomes a capture-time constant.
- Buckets come from mtp_graph_profiles(max_context, mtp_drafts_) (8 buckets at K=2, 131072, dispatched by visible upper bound).
- The arena layout must be reproducible round by round: new DeviceArena::rewind(watermark) and position_arena(arena, floor, target).
  Each round first discards the proposal chain's arena usage and returns to a fixed watermark so that the addresses frozen in the graph hold; at replay it
  aligns to the captured watermark (a lower rewind, a higher alloc_bytes; below the request's watermark it errors out). If a captured bucket's watermark is below the current request's,
  it is treated as uncaptured and re-captured (warm-up's watermark is smaller, so the first real request re-captures once).
- `NINFER_TP2_VERIFY_GRAPH=0` disables it (for A/B), on by default; the eager reference path and the capture path share the same window function
  and the same bucket envelope, so the A/B compares only the launch mechanism.

Measured (same bench recipe, graph on/off):

| Metric | ① off (eager) | ① on (graph) | Difference |
|---|---|---|---|
| avg_round | 39.6 ms | 35.3 ms | **−4.3 ms** |
| verify | 35.0 ms | 30.6 ms | **−4.4 ms** |
| decode_at_2048 (64 tokens on both sides) | 65.6 tok/s | 76.5 tok/s | +16.6% |
| prefill_512 | 1419 tok/s | 1419 tok/s | on par |
| sync_wait | 0.01 ms | 0.01 ms | — |

(tok/s is also affected by the draft acceptance rate determined by this round's text, so the round time and the equal-output 2048 comparison are what count.)
The 4.4 ms verify reduction is entirely launch/scheduling overhead (both paths have exactly the same kernel arguments and inputs).
Numerical correctness: greedy A/B over 5 prompts × up to 256 tokens, eager and graph byte-identical (including reasoning_content);
the graph path is insensitive to whether instrumentation is present, and reproducible.

### Also fixed a real cross-GPU race (pre-existing before the change, not introduced by ①)

Investigation found: the uninstrumented eager path disagreed with graph on 3 of 5 prompts, while any extra synchronization/readback
(even at the end of the round) makes it agree with graph — i.e. the eager path is timing-sensitive. Root cause: at the end of a decode round only shard A
was synchronized (cudaStreamSynchronize(shard_a_.device.stream)), after which the host immediately does fold / state restore / the next round's window and
arena reuse, while shard B's verify tail may still be running; the only ordering guarantee between the two GPUs is the AR spin.
Fix: synchronize both A and B at the end of the round (tp2_generation_core.cpp); measured sync_wait is still 0.01 ms (B lags only very slightly).
After the fix, eager and graph are byte-identical on all 5 prompts.

### Second defect also fixed: token ownership in in-kernel allreduce (real crash)

In the full test run with ①, ninfer_qwen3_5_tp2_forward_test reported cudaErrorIllegalAddress (inside the first layer), while the same
binary on HEAD passed. Bisected: forcing allreduce through the host-staging path passes ⇒ it was introduced by this round's change to a "device-side token" in-kernel AR.
Root cause: the token counter was originally allocated on each device's own device heap, but which shard drives the pair is decided by the caller (that test
has each of the two shards drive once, so stream_a belongs to device 1), and the kernel dereferenced a raw pointer of device 0 inside device 1's context.
Fix: token moved to mapped pinned host memory (visible to both devices), and on the kernel side each block reads it once and broadcasts via shared memory
(if every thread read system memory directly, a single allreduce would cost ~4 us more and a whole round ~0.5 ms more). After the fix the 12 related tests are all green
(including tp2_load / tp2_forward), and eager and graph remain byte-identical.

### Open (out of scope this time)

MTP output quality: compared greedily against plain decode (-Plain, the single-token TP-2 path), only 1 of the 5 prompts is byte-identical,
the rest flip at near-ties. This is the **acceptance behavior** already documented in docs/tp2-dual-5060ti.md §MTP: the verify window and single-token decoding are
different execution shapes (chunked attention/conv routing), so per-column logits necessarily differ, as they do for llama.cpp's Qwen3.5 MTP.
**Note**: do not explain it again with "Phase::Prefill causes the approximation" — the same document already records that as a rejected older diagnosis (Phase has no
semantic effect on the head-split shards). This is not a regression from ①.

### Next steps

- [x] Step 1 measurement.
- [x] Step 3 ① (exact-batch graph) + end-of-round dual-GPU synchronization.
- [x] Step 2 ④ re-review: **this item, as originally described, no longer needs doing**. The basis for ④ in PLAN ("one cudaStreamSynchronize per draft",
  referring to the old tp2_generation_core.cpp:543) no longer holds in the current code: mtp_propose_window does only **1 D2H + 1
  sync** for the whole window, with each draft left with just one D2D copy + one increment kernel; mtp_forward_batch / mtp_forward_ar_step / mtp_forward_core
  contain no host synchronization or D2H at all (grep-verified). The measured mtp=3.70~3.75 ms is **real GPU work** (two T=1 MTP layer forwards,
  weight-streaming-bound), and it is serial with verify. The only residual opportunity is overlapping fold with the MTP chain ≤0.8 ms (≈2%, fold goes through replay.cpp, not including
  allreduce, which in theory could use a separate stream), but that would require touching the AR token protocol/stream ordering ⇒ the benefit-to-risk ratio is not worth it ⇒ ④ judged complete, no further changes.
- [x] Step 4 ③: prefill AR overlapped with MMA — **implemented, measured, rejected** (see below).
- [x] Step 5: retest and documentation update (③'s conclusion written into this file and the worklog).

## Round 12: ③ prefill AR∥MMA sub-block pipeline (implemented, measured to give no benefit, reverted)

### Implementation (fully functional, ultimately not kept)

- `DeviceContext` gains a second stream `collective_stream` (same lifetime as `stream`/`transfer_stream`).
- `forward_tp2_prefill` takes the two-sub-block pipeline when `NINFER_TP2_PREFILL_OVERLAP` is on and phase=Prefill, `pair.in_kernel_allreduce()`,
  driving card == `pair.a()`, tokens ≥ 256: A=[0,ta), B=[ta,T); per layer the order is
  A.mixer → arm(0) → B.mixer → arm(1) → AR(A.mix)/AR(B.mix) → A.res+A.mlp → B.res+B.mlp → A.res2 → B.res2,
  each of the 4 ARs paired with an event pair (`produced` on the compute stream, `done` on the collective stream),
  sub-block binding via nested `ScopedPositions`/`ScopedEnvelope` (positions/rope are slices of the outer tensor,
  envelopes are {first+ta,first+ta} and {visible_end,visible_end} respectively).
- **Key constraint (pitfall)**: the split point must land on an activation scheduling block boundary. The initial version with ta=tokens/2 (300→150+150) makes
  the chunk-split invariant of `tp2_forward_test` fail (max_logit_diff=1.14/1.60); switching to 64-alignment
  (ta=max(64,(tokens/2)&~63), 300→128+172) gives **bit-identical** results:
  `chunking 300 -> 128+172: max_logit_diff=0`, `64+236: max_logit_diff=0`,
  `T=1024 one chunk vs 4x256: max_logit_diff=0`, and the top5 for 300/1024 is bit-identical to the **unsplit** baseline ⇒
  interleaving A/B across layers and dual-stream AR are numerically fully equivalent (this is the only definite technical benefit of ③).

### Measured (same binary, no cached prefix, single request with a 2072-token prompt, TTFT end-to-end)

| Config | run1 | run2 | run3 |
|---|---|---|---|
| `NINFER_TP2_PREFILL_OVERLAP=0` | 1249 ms | 1189 ms | — |
| `NINFER_TP2_PREFILL_OVERLAP=1` | 1258 ms | 1314 ms | 1513 ms (cold start) |

Steady state ≈ 1219 ms vs ≈ 1286 ms ⇒ overlap is **~5% slower**.

### Conclusion (why we are not doing it)

1. The original cost model (PLAN:651, AR accounting for ~48% of prefill) treated AR as **serial** overhead, which measurement does not support: the two
   kernels of in-kernel AR are symmetric, and when the two cards advance with drift, this card's time spent spinning on the peer is already **masked by the
   compute the peer is still running**, so the overlappable margin is far smaller than the model estimated.
2. The cost of sub-block splitting is real: halving T narrows the GEMM tiles and forces **weights to be streamed twice per layer** (prefill is a shape
   sensitive to weight streaming); measured, these two A/B items together outweigh the AR benefit.
3. Hence ③ is not an effective optimization on TP-2: **the final solution remains a single-block prefill without sub-block pipelining**; `collective_stream`
   and the whole pipeline implementation have been reverted (the working tree is cleanly back at 8f6a812e).
4. If a genuinely compute-bound prefill shape ever appears (e.g. a wider chunk, a faster weight path), it can be re-evaluated;
   the 64-alignment constraint and event topology from this section are directly reusable then.

---

### Round 13 — Remove the [mtp] spam print in the decode loop (done)

The decode loop in `tp2_generation_core.cpp` had a bare `std::fprintf(stderr, "[mtp] round pos=…")`:
it bypasses the logger (so `--log-level error` cannot turn it off) and has no env gating, one line per round (≈26 lines/s).
Measured write cost: `2>NUL` 1.8 µs/line, file 1.5 µs/line, **real console 20–25 µs/line**
⇒ 20 µs per round / 35.3 ms = **0.057%**, so the throughput impact is negligible; the real risk is that a QuickEdit selection in cmd makes the console
stop consuming and blocks the generation thread (not slow, but stuck).

That print and the two counters used only by it, `mtp_draft_checked_`/`mtp_draft_hit_`, have been deleted;
the acceptance rate is now derived from `NINFER_TP2_TIMING=1`'s `[tp2-time] decode rounds=R committed=C`:
number of accepted drafts = C−R, acceptance rate = (C−R)/(R·K). `docs/tp2-dual-5060ti.md` has been updated accordingly.

---

### Round 14 — Prefix reuse "continuation boundary": measured, falsified and withdrawn (done)

Implemented option 2 (continuation boundary: extend `cached_tokens_` to the end of the previously generated sequence + a new state slot), validated with real DSH traffic (36 requests):
`slot=0` 35 times, `slot=1` (rewind) 1 time, **continuation slot 0 times**; 33/36 rows have `shared == prefill_end` exactly equal —— the client's next prompt
is token-for-token identical to the previous one to the end, then diverges at the first generated token (it does not send the generated stream back, but re-renders the assistant turn).
Conclusion: this option gives zero benefit for this client, and has been **withdrawn** (the two engine files are back at `1cdfab97`, `state` back to 293.6 MiB/card);
`NINFER_TP2_REUSE_TRACE=1` is kept (per request it prints `prompt/cached/shared/prefill_end/rewind -> reuse/slot`),
as it is the only way to tell whether "the client sends history back verbatim".

**Next candidates (user deferred)**: ① add an anchor at the end of the system+tools prefix, fixing the 46.9 s first round after a new session/compaction (could save ~11 s);
② prefill throughput (suffix 0.95–1.13k tok/s, cold start 1.35–1.38k); ③ ~0.3 s fixed overhead per request.

---

### Round 15 — Constrained decoding of tool calls (route A landed)

**Symptom**: when DSH talks to NInfer it occasionally emits a tool call as body text (`<tool_call> <function=todo_write> …`), and the frontend warns
"tool call form is wrong"; the same DSH against llama.cpp never does this.

**Located (diagnostic details in worklog Round 15)**: it is not quantization (official `nvfp4` and converted `w4a4_w8a8` with 75 tool requests each, marked-class failures
27/75 vs 19/75, z≈1.4 not significant), not chat_template, not the parser, not client configuration. The root cause is a **difference in engine policy**: llama.cpp does
lazy GBNF constrained decoding on declared tools (`common/chat.cpp:1286` generates a `<function=NAME>` rule per tool, `server-common.cpp:1332`
injects the generation parameters), so undeclared names are unreachable at the sampling layer; NInfer only rejects them at the parsing layer with
`enforce_declared_names` and otherwise returns the text verbatim.

**Decision (routes A/B)**: llama.cpp's grammar validates by **text** (`token_to_piece` + `llama_partial_utf8` + GBNF stack) and is
insensitive to tokenization; building a prefix tree over the token sequence would be tokenization-sensitive and easy to mask wrongly. So two routes:

- **Route A (landed this round)**: a state machine accumulates the generated text, and each step uses the tokenizer to decode candidate tokens into pieces (handling a
  half UTF-8 sequence across tokens), and validates whether 「accumulated text + piece」 is a legal prefix of a declared tool name/parameter name, otherwise it sets `-INF`. It is insensitive to tokenization, its correctness risk
  matches llama.cpp, and it is far less code than full GBNF.
- **Route B (Phase 2 option, not done)**: port the stack-based parser of `llama-grammar.cpp` + generate a grammar for Qwen, which naturally supports full parameter
  schema validation. Upgrade only if A still shows occasional boundary problems on real traffic, or if full schema validation is wanted.

**Status**: route A is implemented and passes unit tests. The implementation lives in `include/ninfer/ops/token_mask.h` (+ `src/ops/{kernel,launcher,wrapper}`),
`src/models/qwen3_5/frontend/tool_call_constraint.{h,cpp}`, `frontend.{h,cpp}`, `output_session.{h,cpp}`,
`tool_call_parser.h`, `src/runtime/engine/tp2_generation_core.cpp`; `ninfer_token_mask_test` /
`ninfer_tool_call_constraint_test` / `ninfer_tool_call_parser_test` pass, and a real serving `tools 1` request works end-to-end.
Key points: the mask is only active in the tool-call structure region (`ParameterValue` and free text are not masked), only activates in the non-thinking phase, and is applied **outside** the CUDA Graph;
MTP takes each verify column's own grammar position, and drafts are not masked (an illegal draft disagrees with the masked argmax and is rejected).
Implementation details, boundary handling and verification are in worklog Round 15b.

**Not doing**: "returning undeclared names as tool_call too" would be letting the model bypass `run_code` and directly trigger `pwsh`/`write`, breaking the PTC contract.

**Two pre-existing defects fixed in the same round** (details in worklog Round 15b):

- TP-2 passed the **physical row count** as the valid domain to `ops::sample` / `ops::argmax` / `speculative_accept_greedy_drafts`, so it could sample
  packed rows undefined by the tokenizer (248320 rows vs the public vocabulary); all now pass `public_token_count`, bit-identical on normal steps.
- `make_sampling_config`'s `token_counts` was always null ⇒ presence/frequency penalty was completely ineffective against cross-round repetition; counts are now
  built per request. A/B measured (same seed, one serving start each for the pre-fix/post-fix binaries): before the fix the three penalty configurations produced **byte-identical** output,
  after the fix `frequency=2.0` pushes repetition from "longest run 15" down to "2", and the no-penalty path is byte-identical.

**Remaining (optional)**:

1. Real-model end-to-end retest: against the 75 probes of Round 15, confirm undeclared names no longer appear.
2. Lightweight diagnostics: print the tool name the model actually wrote in the WARN, so one can tell online whether it is `tools.read` or `todo_write`.
3. Route B (full GBNF / parameter schema validation).
4. Optional hardening: add a `validate_licensed_tokens`-style guard to TP-2.

---

### Round 16 — TP-2 host-side state checkpoints + short-prompt reuse truncated by LCP

**Status**: implemented and end-to-end verified (`src/runtime/engine/tp2_generation_core.{h,cpp}`, `src/runtime/engine/model_instance.cpp`,
`tests/test_engine_options.cpp`); `ninfer_engine` compiles, the new option-contract unit tests pass; measured `src=host` hits (req#5 10.2 s → 1.3 s, req#11 37.6 s → 28.6 s, see worklog Round 16.3).

**Problem (follow-up measurement from Round 15)**: the client re-renders history at turn boundaries, and the new prompt is **shorter** than the previous one (measured −219 / −1,233 /
−15,888 tokens), with the divergence point in the middle of the history. The TP-2 core keeps only 2 **VRAM** boundaries (the end of the previous prompt + one rewind point),
and the criterion requires `boundary <= shared_prefix`; both boundaries fall after the divergence point ⇒ `reuse=0` ⇒ **the entire prompt is recomputed in full**:
req#43 57,378 tokens / TTFT 42.0 s, req#51 74,536 tokens / TTFT 57.6 s (NVML sampling confirms both cards are at 98–100% for those 59 s,
consistent with the `done` line's `prefill 1.30k tok/s (74,536 tok)`).

**The root cause is not KV**: the walk starts from the boundary, and KV pages before the boundary are never recomputed anyway (comment at `tp2_generation_core.cpp:1042-1045`).
What is missing is the **GDN (linear attention) state**: it is a recurrent accumulator, which can neither be derived back from the KV nor translated; the K/V of full-attention
layers can be truncated and reused freely, but linear layers cannot. This is exactly why llama.cpp additionally implements context checkpoints (PR #15293): its
`n_past = slot.prompt.tokens.get_common_prefix(input_tokens)` (`tools/server/server-context.cpp:3103`) only handles the
attention layers, and hybrid models fall back on checkpoints — `n_ctx_checkpoints = 32`, `checkpoint_min_step = 8192` (`common/common.h:611-615`),
created before `llama_decode()` (`server-context.cpp:3508-3518`); on restore it searches from newest to oldest for a checkpoint `<= LCP`, and only if none is found does it
`do_reset` a full recompute (`:3236-3249`).

**Plan (zero VRAM increment)**: checkpoints live in **pinned host memory**, and the engine already has this semantics: `--host-state-slots` (called
"complete Host StateImages" in the docs table, `src/serve/serve_options.cpp:220-223`), `HostStatePool` +
`StateImageStore(device_pool, host_pool, capacity)` (`program_impl.cpp:125-144`), and `HostStatePin` at startup;
the TP-2 core already uses `PinnedHostBuffer` (`tp2_generation_core.cpp:279`).

1. One copy of state = **73.4 MiB/card** (`state_bytes`; the ledger `state 293.6 = (2+2)×73.4`). 16 slots = 2.3 GB host,
   32 slots = 4.7 GB host, **VRAM +0**; one restore = 73.4 MiB H2D per card (pinned, ≈5–20 ms), negligible against tens of seconds.
2. During prefill, asynchronously copy `state_backing` into the ring host pool at `kReuseCheckpointStride = 8192` (aligned with llama.cpp's `checkpoint_min_step`),
   and record the frontier position.
3. The reuse scan expands from "2 VRAM boundaries" to "2 VRAM boundaries + host checkpoints", taking the deepest `<= shared_prefix`
   (`tp2_generation_core.cpp:891-897` is the extension point); `begin_gdn_state` (`:976-987`) gains a host→device branch,
   and the KV path is untouched.
4. **Correctness rule (lineage validity)**: checkpoints carry `valid`; before each selection, any with `position > shared_prefix` is invalidated (an out-of-range one
   can never be used again), and when a prefill publishes successfully, the slots written by it are marked valid with the prefill id. It **must not** only recognize "the
   previous prefill" — if the previous one only ran a very short suffix (full reuse), it leaves no checkpoint of its own, and that most common small-tail divergence would still give `reuse=0` (Round 16.2).
   Also, when a prefill is cancelled mid-way, clear `cached_state_valid_` and empty the whole ring (the KV taken by the cancellation and `cached_prompt_tokens_` are no longer consistent).
5. A 131k context at 8k stride yields at most 16 checkpoints, so 16 slots suffice to cover a full context (excess is evicted ring-style);
   32 slots are recommended for 200k.
6. **Tail window**: within the last 8192 tokens, additionally store one copy at the end of each chunk (aligned with llama.cpp's `near_prompt_end` exception), covering
   the most common case of "only the tail is re-rendered" (measured req#12: only 171 tokens differ yet a full recompute of 39.0 s → ~0.4 s after hitting the tail checkpoint).

**Acceptance**: two consecutive rounds in the same session, the second prompt shorter than the first with divergence in the middle ⇒ `[tp2-reuse]` reports `reuse > 0` and coming from a host
checkpoint, `cache` hit rate rises from 0% to the `shared` order, TTFT drops from ~57 s to `(prompt − checkpoint)/1.3k + fixed overhead`;
regression: existing TP-2 cases + the new "short prompt hits a host checkpoint" scenario; `git diff --check` clean.

---

## 13. Done: TP-2 multi-session KV pool (host KV swap in/out)

**Goal**: multiple chat sessions (A/B/C…) alternate on the same TP-2 engine. At any moment only one active session's KV is in VRAM;
the KV + state of up to 5 inactive sessions are kept in pinned host memory, and when a session comes back it is recalled wholesale (H2D) with only the new suffix prefilled;
beyond 5, the oldest is evicted (LRU).

**Background and status quo (evidence)**

- The TP-2 core (`src/runtime/engine/tp2_generation_core.{h,cpp}`) bypasses the whole context cache of the single-card route
  (Program/ResourceManager/checkpoint/resource transactions); `model_instance.cpp`'s `normalize_engine_options`
  resets `context_cache` to disabled when `device_b >= 0` (keeping only `host_state_slots`) ⇒
  `--max-private-continuations` / `--host-kv-mib` are currently **silently ineffective** on the TP-2 route (no error).
- State maintained by the TP-2 core itself: single-row paged KV (`kv_table_rows = 1`, pool built for max_context, **the previous prefill's
  KV stays resident in VRAM**, new requests reclaim pages directly), 2 GDN state snapshots (VRAM, end of prefill + 1 rewind),
  a host checkpoint ring (Round 16: GDN state only, single lineage, stride 8192).
- Round 16's ring solves "the client re-renders a shorter history within the same session" (GDN state recall + KV recomputed by interval from the checkpoint position);
  this plan solves "**cross-session** KV retention" (KV + state recalled wholesale, zero recompute). The two are orthogonal, and the ring is kept.
- Quantities (Qwen3.8-27B: 16 full-attention layers × 4 KV heads × head_dim 256, plus 48 GDN layers):
  text KV 16.125 KiB/token/card (FP8, TP-2 VRAM ledger); GDN state 73.4 MiB/card/session; MTP KV 2 KiB/token (shard 0).
  Swap in/out cost (PCIe 4.0, ~25 GB/s measured basis): 32K session ≈ 40 ms; 128K ≈ 160 ms;
  negligible against a full prefill (200k ≈ 90–160 s).

**Key design decisions**

1. **No preemption (v1)**: swap in/out happens only at **request boundaries** — the active session's response has been fully generated and streamed out, and
   the swap happens when a new session request arrives. A's client sees zero stall; B's TTFT pays only one extra swap-in.
   "B interrupting A mid-generation" would require pausing the lockstep decode loop, saving partial output, and later resuming A; listed as v2.
2. **Zero new VRAM**: the device KV pool and GDN state pool are unchanged (they still serve only one active session).
3. **Exact token matching**: after tokenizing the incoming prompt, scan the directory (≤6 entries) for the longest exact prefix match;
   if the rendering is inconsistent ⇒ full prefill (graceful degradation, same as today).
4. **Keep the Round 16 ring**: it serves in-prefill checkpoints for the same session; when memory is tight, the 32 slots can be reduced
   (8 slots ≈ 1.2 GiB, coarser stride, more recompute).
5. **Two-phase swap in/out + verification**: D2H (both cards) → verify → free VRAM pages → H2D (both cards) → verify → map.
   If any phase fails, the system is in a consistent state (each session in either VRAM or memory), and the next request can retry after the error.
   Following the commit/abort semantics of the single-card route's ResourcePlan, the TP-2 version is simplified to two phases + verification.

**Work items (in dependency order)**

| # | Work item | Location | Size |
|---|---|---|---|
| 1 | **Host KV arena** (one per shard): pinned host storage mirroring the device KV page geometry (page-major `[X,P,H,N]`, P=64); one slab per session, allocated by that session's actual frontier page count (not pre-allocated full to max_context). Budget: 5 sessions × 204800 tokens × 16.125 KiB/token/card ≈ 16 GiB/card, 32 GiB total (i.e. `--host-kv-mib 32768`); plus GDN state 5 × 73.4 MiB × 2 cards ≈ 0.7 GiB | `tp2_generation_core.cpp` | ~1–1.5k LOC |
| 2 | **Session directory**: replaces the `cached_prompt_tokens_`/`cached_boundaries_`/`cached_state_valid_` single-lineage trio. ≤6 entries (1 active + 5 inactive), each: token history (204800 × 4 B ≈ 0.8 MiB, negligible), frontier, residency flag (device/host), host KV slab, host GDN state, MTP KV (shard 0), LRU clock | same | ~500 LOC |
| 3 | **Session matching**: extend the existing reuse scan (from `tp2_generation_core.cpp:953`) — generalize the linear token comparison against a single `cached_prompt_tokens_` into scanning the directory for the longest exact prefix | same | ~200 LOC |
| 4 | **Store transaction (D2H)**: triggered at request boundaries (on admitting a new session). Both cards' KV pages D2H → verify → GDN state D2H (reusing the existing `PinnedHostBuffer` mechanism) → MTP KV D2H → free VRAM pages → mark host-resident. Copy before free, so peak VRAM does not increase | same | ~500 LOC |
| 5 | **Restore transaction (H2D)**: the reverse. Map VRAM pages → KV H2D → GDN state H2D → MTP KV H2D → verify → mark device-resident | same | ~500 LOC |
| 6 | **LRU eviction**: when the directory is full (5 inactive) and a new session arrives, free the oldest session's host KV + state + history | same | ~200 LOC |
| 7 | **prefill integration**: after a restore, prefill only the new suffix from the session frontier. The existing prefill already supports starting from a reuse point (the `reuse` token); wire up "reuse point = session frontier + restored pages already mapped" | same | ~300 LOC |
| 8 | **Option plumbing**: `model_instance.cpp` passes `host_kv_capacity_bytes` / `max_private_continuations` through to TP2GenerationCore (or add TP-2-specific options); startup validation (arena ≥ session count × capacity × bytes per page) + one more line in the `[mem]` ledger | `model_instance.cpp`, `serve_options` | ~300 LOC |
| 9 | **Tests**: add `tests/models/qwen3_5/test_tp2_sessions.cpp` (following the small-model fixture of `test_tp2_forward.cpp`): A prefill→done → B prefill→done → **A continues** (assert recall hit, not a full prefill; output **bit-identical** to the from-scratch prefill oracle — identical KV bytes imply identical attention) → C/D/E/F (assert A is LRU-evicted) → A returns (assert full prefill) | `tests/` | ~800–1k LOC |
| 10 | **Documentation**: `docs/tp2-dual-5060ti.md` (product semantics + recommended configuration) + startup script comments | `docs/` | — |

**Acceptance criteria**

- Functionality: in multi-session alternating scenarios `[tp2-session]` diagnostics report hit/miss/eviction; on a hit, TTFT ≈ swap-in cost + new-suffix
  prefill (32K session ≈ 40 ms + suffix), rather than a full prefill.
- Numerics: a recalled session's output is bit-identical to the from-scratch prefill oracle (identical KV bytes ⇒ identical attention).
- Regression: all existing TP-2 cases pass (`qwen3_5_tp2_forward --artifact` etc.); `git diff --check` clean.
- Resources: zero new VRAM (nvidia-smi ledger unchanged); pinned host ≈ 32 GiB (KV) + 0.7 GiB (state) + the existing ring
  4.7 GiB. **Confirm physical memory ≥ 48 GiB before startup** (WMI on this machine cannot read total memory, so check with Task Manager);
  if insufficient, step `--host-kv-mib` down (on the 204,800-token basis, 16384 = 2 full-context sessions).

**Risks and caveats**

- 32 GiB of pinned memory: WDDM/Windows `cudaHostAlloc` has a practical ceiling; the existing ring already pins 4.7 GiB and works fine,
  so 32 GiB needs empirical confirmation.
- `max_pending_requests = 1` (enforced by TP-2): only one waiting request at a time, no impact for chat scenarios.
- The vision workspace is a per-request temporary arena, not part of session state, and does not participate in swap in/out.
- Session token history ≤ 0.8 MiB/session, negligible.

**Phasing**

- **v1**: all of items 1–10 above (request-boundary swap in/out + 5-session LRU + exact matching).
- **v2 (optional)**: preemption during generation (pause/resume lockstep decode, continue streaming partial output).

**Effort estimate**: ~4–5k LOC C++ + tests, on the order of weeks. The physical transfer can follow the single-card route's
`src/models/qwen3_5/program/transactions/materialization.cpp` D2H/H2D transactions, but the TP-2 core is an independent
code path, so this is a port rather than an enablement.

**Progress (Round 55, done and measured passing)**

Landed (the code part of work items 1–10):

- **Host KV arena**: reuses the existing `HostKVArena` (`src/core/host_kv_arena.h`) and
  `DeviceKVPagePool::copy_to_host/copy_from_host`; TP-2 does not write its own D2H paging loop. One
  arena per shard, capacity = `--host-kv-mib / 2`, **constructed only on the first store** (a single-session load never pins). Shard 0's
  arena registers both the text and MTP geometries. A session slab is allocated by that session's frontier page count
  (`pages_for_tokens(frontier)`), not pre-allocated to max_context. The device KV pool stays fully materialized at startup
  with an unchanged page table; swap in/out only overwrites data, with zero new VRAM.
- **Session directory**: `TP2GenerationCore::SessionEntry` = tokens (prompt + committed generation), frontier,
  device/host residency flag, per-shard host KV slab, per-shard GDN state image, MTP slab, LRU clock.
- **Matching**: `session_recall` runs at the start of execute, before the prefix scan. It computes the exact LCP for each directory entry;
  a host entry requires `LCP >= frontier` (whole-recall semantics); candidates are compared by frontier depth, and only those deeper than the **resident lineage's
  reusable depth** are swapped. The resident depth is derived from the live frontier / device snapshot / host checkpoint, from the same source as the scan;
  when the depth is 0 the resident session is stored first and then the new one created (otherwise the directory entry would be silently overwritten by the new session's content).
- **Store/restore transactions**: `session_store_active` / `session_restore`. The two shards each do D2H/H2D of KV + GDN state + MTP KV on their local stream,
  and only after both cards synchronize are the residency flags changed. On a store failure (insufficient budget) the entry is discarded
  and the path degrades to a full prefill; on a restore failure, `execute`'s catch invalidates the resident entry.
- **LRU**: directory capacity = `--max-private-continuations` (TP-2 default 6, including the resident one); when full, the oldest non-resident entry is evicted.
- **prefill integration**: the scan gains a **LiveState** candidate (reuse = session frontier, state not copied);
  `ReuseSource{None,DeviceSnapshot,HostCheckpoint,LiveState}` replaces `reuse_from_host_`;
  `cached_prompt_tokens_` now includes committed generated tokens (published at the end of decode), so a continuation round in the same session can also reuse from the
  frontier. Note frontier = prompt + generated − 1: the last sampled token has not been forwarded yet.
- **Correctness boundaries**: ① on a restore/new session, invalidate the host checkpoint ring (the ring's state belongs to the displaced lineage and its position may
  still fall within the recalled history); ② any exception thrown from the walk is caught by `execute`, which invalidates the resident entry and clears
  `cached_state_valid_`; ③ the same for a cancelled prefill (other host entries are kept).
- **Option plumbing**: the TP-2 branch of `normalize_engine_options` keeps `host_kv_capacity_bytes` and
  `max_private_continuations`, and the disabled-cache validation is skipped for this route; `--host-kv-mib 0` disables session retention.
  The startup ledger gains `[mem] host sessions capacity ...`; `NINFER_TP2_SESSION_TRACE=1` prints
  recall / evict decisions. `tools/win_port/serve.ps1` exposes `-HostKvMiB` (default 32768) and
  `-PrivateContinuations` (default 6).
- **Tests**: `tests/models/qwen3_5/test_tp2_sessions.cpp` (registered, `SKIP_RETURN_CODE 77`).
  Requires `NINFER_TEST_ARTIFACT` + two same-name sm_120a cards. Scenario: oracle (retention off) full prefill alignment →
  A/B/A recall and assert `reused_prompt_tokens == frontier` → output token-for-token identical to the oracle → fill the directory
  to trigger LRU eviction of A → A returns and asserts `reused_prompt_tokens == 0` with output still identical. Device scanning picks a same-name
  pair only among sm_120a devices (CUDA on this machine also enumerates a Tesla T10), so foreign devices no longer make the test skip permanently.

**Measured (Round 55)**

- End-to-end test `ninfer_qwen3_5_tp2_sessions_test`: artifact `<models-dir>/qwen3_8_27b_w4a4_w8a8.ninfer`,
  two 5060 Ti (CUDA 0/2), `max_context 2048`, `prefill_chunk 256`, greedy, running two routes:
  plain + bf16 KV, and MTP + fp8 KV (the recommended serve configuration). Exit code 0, 200 s, output
  `TP-2 session retention (plain|mtp) passed: recall reused 71 prompt tokens bit-identically;
  LRU eviction forced a full prefill`. On the MTP round shard 0's arena prints `2 layouts` (shard 1 is `1`),
  proving MTP's own KV slab is swapped in/out with the session. With `NINFER_TP2_SESSION_TRACE=1` the decision chain and directory state are self-consistent:
  `new session entries=0` (A) → `new session entries=1` (B, A stored) →
  `recall frontier=71 resident_depth=0 tokens=72 entries=2` (A recalled) → `new session entries=2` (C) →
  `new session entries=3` + `evict frontier=55 tokens=56` (evict B) → `new session entries=3` +
  `evict frontier=95 tokens=96` (evict A) → A returns and takes `new session` (full prefill). Only on the first store does
  `host KV arena shard N: 64.0 MiB pinned, 1 layouts` appear, confirming a single-session load does not pin.
- Pinned memory ledger: `ninfer-serve --devices 0,1 --max-context 204800 --kv-dtype fp8 --host-kv-mib 32768
  --max-private-continuations 6` measured at startup
  `[mem] host sessions capacity 6 | host KV 16384.0 MiB/shard | retention enabled` +
  `[tp2-session] host budget 16384.0 MiB/shard holds 5 of 5 full-context sessions (3225.0 MiB each)`.
  204,800 × 16.125 KiB = 3225.0 MiB/shard, matching the documentation formula bit for bit; `--host-kv-mib 32768` + the default directory of 6
  (1 resident + 5 host) is the recommended setting. VRAM usage was 0 throughout (the arena was never pinned), and the GPU returned to zero after shutdown.
- Numerical consistency is carried by the test's token-for-token alignment: the recall walk and the oracle's full prefill produce exactly the same greedy output,
  showing that both the KV bytes and the GDN state image are swapped in correctly; re-prefilling after LRU eviction also returns to the same sequence.

**Known boundaries**

- Budget basis: shard 0's arena carries both the text and MTP slabs, so MTP eats part of shard 0's
  budget under a symmetric split; if capacity is insufficient, entries are LRU-evicted, and `--host-kv-mib` can be raised if needed.
- A session where client re-rendering gives `LCP < frontier` is not recalled (graceful degradation to a full prefill); partial
  reuse within the same lineage is still handled by the Round 16 ring + device snapshot.
- Running the model tests requires ffmpeg and libcurl's `bin` on PATH (same for `ninfer-serve`), otherwise the process exits with
  `STATUS_DLL_NOT_FOUND`(0xC0000135); `tools/win_port/serve.ps1` sets this for you.
- **Pre-existing artifact difference (not introduced by §13)**: `test_tp2_forward` on `qwen3_8_27b_w4a4_w8a8.ninfer`
  does not satisfy chunk-split invariance (300 -> 256+44, top5 gap 1.0625, max logit diff 11.53), while on
  `qwen3_8_27b_nvfp4.ninfer` (the artifact recorded in the docs) it is bit-identical (gap 0, diff 0). That test uses only the low-level
  shard fixture and does not construct `ninfer::Engine`, so it is unrelated to §13. `ninfer_qwen3_5_tp2_load_test` passes on the new
  artifact.
- **Pre-existing defect (not introduced by §13, not fixed)**: on the TP-2 route `bf16` KV + `--spec mtp` crashes on the very first prefill
  (`prompt.cu:63` `cudaErrorInvalidValue`). It reproduces at both `--max-context` 2048 and 8192, with the startup ledger showing
  `host sessions capacity 0` (session retention off), so it is unrelated to the session-pool change; `fp8` + MTP works fine,
  which is why the test's MTP round uses fp8. The entry point for investigation is the launch parameters of bf16 prompt-attention (`prompt.cu`).

### 13.1 Follow-up hardening of session restore (Round 18, implemented and measured)

The user asked: fix session-restore hazards 1/2/5, and investigate "the system prompt cache never hits at all".

- **Hazard 1 (prefill cancellation loses progress)**: on cancellation, publish the completed chunk prefix into the directory (when `t0 > 0`,
  `snapshot_state` + `cached_boundaries_` + `session_publish(tokens[0..t0), t0)`), so resending the same prompt
  continues from that prefix instead of starting over from 0. The test adds `run_cancelled`: cancel before the 2nd chunk, resend and assert
  `reused_prompt_tokens == 256` and token-for-token agreement with the oracle.
- **Hazard 2 (prompt-end state image makes reclamation fail)**: the two optional state images become best-effort — an allocation failure only warns,
  and the session is still recallable via the frontier image (`host_prompt_end` is set only when both images are obtained).
- **Hazard 5**: deleted `host_valid`, which was written but never read.
- **System prompt cache** (mechanism added this round, three items):
  1. Grid checkpoint alignment: the end of a prefill chunk is truncated to `next_host_checkpoint`, so grid checkpoints land exactly on stride
     multiples (previously they landed at the end of the first chunk after the stride, and when the shared prefix was shorter than that coverage there was no reusable boundary).
  2. Degenerate-recall guard: the host scan requires `reach < prompt_tokens` (there must be tokens left to forward); otherwise, after restoring the whole
     history it would still have to prefill from 0, and it also clears the device lineage and the checkpoint ring.
  3. Shared-prefix state image (`host_shared_state` / `host_shared_end`): on session store, if the new prompt diverges exactly at the
     walk start (`reuse == shared_prefix`), freeze that boundary state into the stored session's own image; if the divergence point
     falls inside the walk, copy the divergence anchor's state into that image. Afterwards any session sharing the same stable block
     can recall this entry (KV uses that entry's own slab, state uses this image). `[tp2-session]` gains
     `shared_end=` and `via=shared`.
- Measured (plain route, exit 0): `entry 0 tokens=672 frontier=671 prompt_end=664 shared_end=512
  shared=512 reach=512 via=shared` → `recall frontier=512` → `reuse=512 src=live`; even with an earlier small,
  unrelated request taking over the device pool, the recall still works.

**Open issue (important: not fixed, do not treat as delivered correctness)**: after adding the assertion "recall walk vs from-scratch oracle",
we found that the `shared_c` scenario (the 3rd session sharing the same system prompt) has its recall walk diverge from from-scratch prefill at the
7th/8th greedy token (becoming repeated tokens); whereas the same boundary going through the "device KV still present" reuse path (`src=host`
for `shared_b`) is token-for-token identical to the from-scratch oracle. This round pinned the scope down with device-side read-back:

- KV round trip is **byte-exact**: after restore, reading device pages back via D2H and comparing with the slab gives `diff=0 first=<end>` (8 pages / 16 MiB).
- **The entire device page content (including the tail not covered by the host transfer) is exactly the same at the two samplings, store and recall**:
  `store entry=1 frontier=671 dev=56163be625b88263` == `restore boundary=512 dev=56163be625b88263`.
  That is, the device KV at the recall point is the same content already validated by the from-scratch oracle in the `shared_b` scenario.
- The three state-image copies (capture / restore / ring slot) have the same checksum; each branch of `begin_gdn_state` touches only `state_backing`.
- Two engines' from-scratch output for the same prompt is token-for-token identical (`shared_a_first` vs the oracle's `shared_a`),
  so this is not a reproducibility difference between two engines.

**Root cause located (Round 19, wording revised)**: after printing the pre-sampling top-1/top-2 with `NINFER_TP2_DEBUG_LOGITS`, it was confirmed that the
divergence **already exists in the last prefill column** (not something accumulated round by round during decode):

```
shared_c  oracle   reuse=0   src=0  prefill pos=639  top1=54 4.750000 top2=1646 4.562500 gap=0.187500
shared_c  recall   reuse=512 src=3  prefill pos=639  top1=54 4.406250 top2=220  4.406250 gap=0.000000
shared_b  oracle   reuse=0   src=0  prefill pos=639  top1=1703 6.468750 top2=4263 5.468750
shared_b  recall   reuse=512 src=2  prefill pos=639  top1=1703 6.218750 top2=4263 5.250000
shared_a  oracle   reuse=0   src=0  prefill pos=639  top1=197 4.781250 top2=220 4.593750
shared_a  this eng reuse=0   src=0  prefill pos=639  top1=197 4.593750 top2=220 4.562500
```

The key is the last two lines: **two Engine instances each doing one from-scratch prefill of the same prompt also give different logits**
(4.781250 vs 4.593750), while their 8 greedy tokens are still identical (the test has been asserting this all along). `shared_b` goes through the
existing ring path that **restores no KV at all** (`src=2`), and it is likewise off by ~0.25. Therefore:

- It is not the host slab recall losing data (proved byte-exact last round: KV round trip `diff=0`, whole-page device content identical at store/recall);
- It is not the shared-prefix image being wrong;
- Rather, **prefill numerics are not reproducible across Engine instances**, at a magnitude of ~0.2 (4% for logit≈5, far larger than a reduction order
  could produce; uninitialized reads caused by workspace/arena history are suspected). `shared_c` merely happens to have an oracle gap of only
  0.1875, and the first token flipped by this offset appears at the 7th.

**Root cause (Round 19 wrap-up, settled)**: printing the width of the last prefill chunk as well makes it clear — for the same
prompt and the same `reuse=0` from-scratch walk, the logits drift with the **width of the last chunk**:

```
#3  oracle    shared_a  reuse=0 chunk=550+90 snap1=550  top1=197 4.781250
#19 this eng  shared_a  reuse=0 chunk=542+98 snap1=542  top1=197 4.593750
#30 this eng repeat shared_a  reuse=0 chunk=582+58 snap1=582  top1=197 5.156250
#9  oracle    shared_c  reuse=0 chunk=590+50 snap1=590  top1=54  4.750000
#23 recall    shared_c  reuse=512 chunk=512+128 snap1=0 top1=54  4.406250
```

`snap1 = prompt_tokens - rewind_near_`, and `rewind_near_` is derived from the engine's history (the previous request's divergence position),
**so the same prompt gets different chunk boundaries in every walk**, and the prefill's last-column logits therefore differ by
0.3~0.56 (6~11% for logit≈5, far beyond what floating-point reordering can produce). That is:

- Recall / host slab / shared-prefix image are **all innocent** (KV and state have both been proven byte-identical);
- The real source of divergence is "**the suffix chunk width varies with history ⇒ the same prompt's prefix computation is not reproducible**";
- The earlier judgment that "both sides have the same chunk" was wrong: `shared_c`'s oracle is `590+50`, and the recall is `512+128`.

**Fix direction**: decouple the chunk plan from the ring's rewind target — the chunk width is decided only by `(reuse, prompt_tokens)`
(fixed `prefill_chunk`, `min(prefill_chunk, remaining)`), and ring checkpoints land only on those fixed boundaries.
"Putting the snapshot close to the frontier" was originally a small optimization to save one rewind, at the cost of making the mathematical result depend on history,
and `kReuseTailCheckpointCount`'s dense tail window already bounds the rewind cost to within a few dozen tokens, so this optimization is not worth keeping. After the change, the chunk plan for the same prompt is reproducible, and the recall walk can again assert token-for-token.

Diagnostic code: `tp2_debug_logits` (`NINFER_TP2_DEBUG_LOGITS`) and `tp2_debug_hash_*` /
`NINFER_TP2_DEBUG_KV`, off by default, to be deleted once the fix is complete.

---

## Round 17: plain decode CUDA Graph + AR transfer strategy keyed by tensor size (in progress)

**Goal (explicitly requested by the user)**: put the two items below into the plan and implement them, running an A/B and reporting after each.

| # | Item | Theoretical benefit | Status and basis |
|---|---|---|---|
| ① | Put the plain (`--spec` default) single-token decode round on an exact-batch CUDA Graph | Remove the launch/scheduling overhead of ~1000 kernels per round. Round 11 measured `verify` at 4.4 ms/round less on MTP verify (same order of kernel count) | The plain route (`tp2_generation_core.cpp:1463-1494`) is fully eager; `forward_tp2` passes host token/position via `set_i32_scalar`, which is **not capturable**. MTP verify already has a graph (Round 11 ①) |
| ② | `DevicePair::allreduce` switch transfer strategy by payload size | On the decode side, 128 ARs per round at 9–18 µs each (1.2–2.3 ms, 4–7% of a round); on the prefill side a single AR is already at the link floor | Currently there is only one in-kernel mapped-pinned spin path (`device_pair.cu:338-444`), with only the slice count adaptive to size (`ar_slices`) |

### Key prior conclusions (do not re-investigate)

1. **Envelope buckets are numerically exact**. `small_t_fp8.cuh:147-179`: `window = last_pos + 1` is computed from the **device-side positions**; `active_split_count = min(default_splits(window), split_count)`, and the split key intervals are all derived from `window`; the host-side `logical_capacity = envelope.max_visible_keys` is used only for out-of-range protection (`last_pos >= logical_capacity ⇒ write_neutral`). `causal_small_t_launch_capacity` (`small_t.cu:78-94`, together with the tier endpoint `{128,160,512,4096,5000,8198,16390}` in `small_t.cuh:81-94`) takes the supremum of `default_splits` within the envelope interval, so as long as the host's `splits ≥ default_splits(window)`, the bucket envelope and the exact envelope are **bit-identical** ⇒ the single-card `ordinary_graph_profiles` bucket approach can be carried over to TP-2.
2. **`Phase` has no numerical effect on the TP-2 shards**: the Verify branches at `text.cpp:1233` and `:1399` are both blocked by `shard_config_ == nullptr`; in `attn_mix`/`gdn_mix`, `ph` is used only for the nvtx name.
3. **But `active_sequence_batch_` does have a numerical effect**: from `text.cpp:1154`, only when `active_sequence_batch_ != 0` does it take the batched attention route. `forward_tp2` (plain decode) sets batch=1/width=1, whereas `forward_tp2_window` (used by MTP verify) does **not** set batch/width ⇒ **plain decode on a graph cannot directly reuse `forward_tp2_window`**; the bindings of `forward_tp2` must be replicated.
4. `use_cuda_graph=false` at `model_instance.cpp:99` affects only the single-card Program route and is unrelated to TP-2.
5. `--spec` defaults to `SpeculativeBackend::None` (`speculative_options.h:11-16` only parses an explicitly passed value); `tools/win_port/serve.ps1 -Plain` is the plain route.

### ① Implementation design

- Add `TextContext::forward_tp2_decode_window(...)` (`models/qwen3_5/execution/text.{h,cpp}`): line-for-line isomorphic to `forward_tp2`, with only capturability changes — `ids` and `cache/rope positions` are read from pinned host via a `copy_i32` memcpy node (no longer host values via `set_i32_scalar`), and the envelope becomes a capture parameter; the other bindings (batch=1, width=1, `kv_table_rows=0`, state slots=0, `Phase::Verify`, final norm, `project_head_tp2`) are copied exactly.
- `TP2GenerationCore`: rename `VerifyGraph` to `WindowGraph` and reuse the same `round_base/arena_begin/arena_bytes` bookkeeping; add `decode_graphs_` (buckets from `qwen::detail::ordinary_graph_profiles(max_context)`, `visible_begin = min+1`, `visible_end = min(max_context, max+1)`, same origin as the single-card ordinary decode graph) and `decode_window_host_` (pinned `[token, position]`).
- Only graph it when `pair_.in_kernel_allreduce()` (P2P / host-staging synchronize the host and are not capturable).
- Env switch `NINFER_TP2_DECODE_GRAPH`: unset/other = capture graph; `0` = eager + **bucket** envelope (pure launch-mechanism A/B); `exact` = eager + exact envelope (= HEAD behavior, envelope A/B).
- Sampling stays outside the graph (sampling parameters change per request, and `logical_pos_a` and the tool mask are both host-driven); the graph covers only forward; the logits buffer is allocated by the caller before the graph in a fixed order, replicating the verify graph's positional convention.
- Acceptance: greedy token-for-token consistency across the three configurations + `bench_serve.ps1` measured tok/s + `NINFER_TP2_TIMING=1`'s `avg_round`.

### ② Implementation design

- Size-keyed transfer strategy (`NINFER_TP2_AR_STRATEGY`: `size` default by size / `kernel` forces the current path and serves as the A/B reference):
  - **Small payload (≤ 64 KiB, covering decode's 10 KiB and verify's 40 KiB)**: in-kernel path + fold `bump_ar_token` into the AR kernel (with a single block, thread 0 increments, publishes via `__threadfence_system()`, broadcasts through shared memory), saving one launch per AR; staging switches to an independent 2×64 KiB mapped-pinned region, and the parity slot no longer aliases the prefill slot. Thread count stays `kArThreads`.
  - **Large payload (> 64 KiB, covering prefill's 10 MiB)**: the sliced path is left untouched (measured to be already at the link floor).
- **Unimplemented copy-engine large-payload arm** (D2H → `cudaEventRecord` → peer `cudaStreamWaitEvent` → H2D to scratch → device-side add): abandoned for two reasons — first, its cross-card ordering must rely on a cross-device `cudaStreamWaitEvent` (the "copy engine is slower" measurement at `device_pair.cu:352-357` was of the old host-staging path, insufficient to reject on that basis, but in bandwidth terms the copy engine and SM writes use the same PCIe link); second, the real lever for this item is not the transfer method but **the count of 128 calls per round**, as demonstrated by the untouched prefill reference arm.
- The numerical contract is unchanged: BF16 element-wise `__hadd`, addition order unchanged (measured 7/7 bit-identical).

### Progress

- [x] Plan written down (this section)
- [x] ① Implementation (forward_tp2_decode_window + DecodeStepMode + decode_graphs_)
- [x] ① Build + numerical comparison + A/B: decode +12.6% @1K / +12.1% @32K (31.11→35.03, 29.79→33.39 tok/s, ~3.6 ms saved per round); prefill unchanged; graph vs bucket 7/7 bit-identical. See profiles/bench/tp2_decode_graph/report.md
- [x] ② Implementation (payload ≤ 64 KiB single block + bump folded into the AR kernel + dedicated 2×64 KiB staging; > 64 KiB uses the original sliced path; NINFER_TP2_AR_STRATEGY=size|kernel)
- [x] ② Build + numerical comparison + A/B: decode +0.63% @1K / +0.51% @32K (35.12→35.34, 33.48→33.65 tok/s, avg_round 29.86→29.70 ms); prefill is a natural control, unchanged in all three arms; token 7/7 bit-identical. The thread count must stay kArThreads (the 256-thread version is a −0.2%/−0.4% regression). See profiles/bench/tp2_decode_graph/report.md

---

# PLAN.md archive (archived 2026-09-22, upstream cherry-pick re-plan + DFlash2 TP-2 rounds)

Archived verbatim from the working PLAN.md when it was reduced to the current state, after the
upstream cherry-pick re-plan, the DFlash2 TP-2 rounds (B1-B7, verify CUDA Graph, MTP draft-chain
graph, declined-draft fix) and the delivery cleanup. Historical record only: the current state and
the remaining work live in `PLAN.md` at the repository root, product behavior and measurements in
[tp2-dual-5060ti.md](../tp2-dual-5060ti.md).

# NInfer TP-2 plan (2× RTX 5060 Ti · Qwen3.8-27B NVFP4)

> **The only active plan**, and the persistent memory across context compaction. Compiled: 2026-09-21 (upstream cherry-pick re-plan).
> - Full historical record (Round 1–48 round-by-round evidence and failed attempts): `tp2-adaptation-worklog.md`
> - Full text of the previous PLAN.md (Round 48–55 + worklog Round 11–19, archived word for word): end of the same file
> - Delivery notes, recommended configuration and measured data: `docs/tp2-dual-5060ti.md`; Windows native port: `docs/windows.md`

---

## 1. Key information

- **Branch**: `feat/windows-native-port` (Windows native + WSL2 dual tree; merge base with origin/master `e360c4c0`,
  upstream is now 20 commits ahead, see §4).
- **Hardware**: 2× RTX 5060 Ti 16G (SYS topology, no P2P, 448 GB/s/card). WSL2 build tree `$HOME/ninfer`,
  Windows working tree the repository root (build tree `build-win`).
- **Main workload**: Qwen3.8-27B NVFP4, TP-2 (weights split in half by shard, locally built half geometry shapes); MTP draft
  (K=2, `--lm-head-draft`) uses full geometry shapes (NVFP4 swiglu TMA / linear_add TMA / Q8).
- **Status**: delivery goals ①–⑤ all complete; vision, the Windows port, the multi-session KV pool, and decode graph + AR strategy are all landed and measured.
- **Key numbers** (262,144 configuration, WSL): prefill 1,588 tok/s; decode 59.0 tok/s (MTP K=2);
  Windows 131,072 configuration decode 56.9 tok/s (on par with Linux).

**Recommended runtime configuration** (same recipe for WSL 8088 / Windows 8099, with `--max-context 131072` on the Windows side):

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2 --lm-head-draft --vision --reasoning-effort medium
```

`--reasoning-effort low|medium|xhigh` is the process-level default thinking effort (the request body takes precedence); `--chat-template` **does not yet
exist** on this branch (templates are compiled from a sha256 whitelist, with only two semantics, `e84f32a2…`/`c3cf9e34…`) — the new official artifact (template `a497db9e…`) and `--chat-template` only become
available after the §4 Tier 1 cherry-pick.

### Environment and operations notes

| Item | Value |
|---|---|
| Build (WSL) | `bash tools/tp_bootstrap/build_r35.sh` (rsync + `cmake --build build -j 8`, success marker `BUILD_EXIT=0`, about 2–3 minutes; the script syncs itself, so a change only takes effect on a second run) |
| Build (Windows) | `tools/win_port/configure.bat` + `build.bat` (VS2022 + CUDA 13.3, `-DCMAKE_CUDA_ARCHITECTURES=120a`) |
| Serving | WSL 8088 (`serve_supervise.sh`); Windows 8099 (`tools/win_port/serve.ps1`, foreground by default; a self-test service must use a harness background job, since a process started with `Start-Process` is killed when the tool call ends) |
| Runtime PATH (Windows) | FFmpeg (`D:\ffmpeg-dev\…\bin`) and libcurl (`D:\curl-dev\…\bin`) must be on PATH, otherwise `STATUS_DLL_NOT_FOUND` |
| artifact | `<models-dir>/qwen3_8_27b_nvfp4.ninfer` (23.7 GB, **old** artifact, template `c3cf9e34…`); the new official artifact (template `a497db9e…`) must be extracted per §4 Tier 1 before it can be loaded |
| Process discipline | Only one model process on the whole machine at a time (WSL 8088 and Windows 8099 are mutually exclusive); stop the service before relinking the Windows exe (LNK1104) |
| Toolchain | A nested `pwsh` → `wsl -e bash -lc` swallows `$var` and redirections ⇒ write commands into a script file under `tools/tp_bootstrap/` and execute that; a single blocking call is capped at 600 s ⇒ use background jobs for long tasks |

---

## 2. Summary of completed work (see the archive for details)

| Item | Conclusion |
|---|---|
| ① MTP consistency | Root cause = the shard did not write replay records (fold consumed an empty log); fixed, and the criterion was corrected to "no degradation + same quality tier + speedup" and met (archive §5) |
| ② Benchmark | prefill 615→1,588 tok/s; decode plain 31.4 / MTP K=2 50.5–54.0 → 59.0 tok/s (archive §2) |
| ③④ device memory / KV / context | five KV dtypes usable (fixed the per-device smem opt-in defect); 65,536 → 131,072 → **262,144** (vocab/hidden parallel split, −1.2 GiB/card) (archive §10) |
| ⑤ Documentation | `docs/tp2-dual-5060ti.md` + `docs/windows.md` |
| Vision (Round 53) | static shard division of labor (MTP→shard 0, vision→shard 1); multimodal and pure text use the same mechanism (MTP + prefix reuse); single-image cap 8,192 tokens, no new configuration (archive §11) |
| Windows port (Round 54) | VS2022 + CUDA 13.3 native build; CLI/serve/TP-2 all working, TP-2 performance on par with Linux (decode 32.9/56.9 tok/s); TMA descriptor mapped-pinned fix (C2719 + CUDA Graph capture, two in a row) (archive §12) |
| ①③④ optimization (Round 11–12) | ① exact-batch graph landed (verify −4.4 ms/round, decode +16.6%); ③ the AR∥MMA sub-block pipeline measured 5% slower once implemented, **rejected and reverted**; ④ after review the original premise did not hold, judged complete (archive Round 11/12) |
| Round 13–16 | [mtp] spam-print cleanup; the prefix-continuation boundary scheme was empirically falsified and withdrawn; tool-call constrained decoding route A landed (token mask, applied outside the graph); host-side state checkpoint (stride 8192, 0 incremental device memory) landed |
| Multi-session KV pool (Round 55/18) | host KV swap-in/swap-out + 5-session LRU + shared-prefix mirror; the KV restored by recall is byte-identical to device, and results agree with from-scratch within chunk-boundary rounding (an exact tie can still flip the first sample, see §3.1) (archive §13/13.1) |
| Round 17 | plain decode exact-batch graph (+12.6%) + AR transfer strategy split by payload size (+0.63%) landed |
| Round 19 | the shared_c recall divergence root cause settled (chunk width depends on engine history via `rewind_near_`); the fix landed and was verified (§3.1) |
| Round 20 | chunk plan decoupled from ring rewind (1a/1b/1c): the same prompt no longer depends on engine history, plain/mtp pass bit-for-bit; boundary sensitivity quantified (the last block width is not a variable, the boundary position is; 120 decode steps with zero flips); host KV arena switched to pageable backing (`--host-kv-pinned` pins pages only; a rejected pin falls back automatically) |
| DFlash2 (single-card baseline; not adapted to TP-2) | fully implemented and qualified on the **single 5090** route (`docs/maintainer/dflash.md`, `docs/performance/*`, bench corpus); the TP-2 route **deliberately rejects** `--spec dflash2` (`model_instance.cpp` normalization error; worklog §36.1 and `:2438` explicitly retain this restriction); bringing it to dual cards is "MTP/dflash2 component split"-level work, not scheduled, see §3.6 |

**Unmet original thresholds (honest record)**: MTP3 ≥ 70 tok/s was not reached (pure decode ceiling K=2 50.5–54.0);
the TP-2 path has not run a perplexity evaluation (quality evidence uses same-prompt multi-sample A/B); the per-shard arena's
`memory_summary()` still reports `pages 0/0` (a display-only issue).

---

## 3. Outstanding items

### 3.1 Round 19/20: decoupling the chunk plan from the ring rewind target (completed)

Root cause (settled): the width of the last prefill chunk was determined by `rewind_near_` (derived from the previous
request's divergence position) ⇒ the chunk boundaries of the same prompt differ on every walk ⇒ the last-column prefill
logits drift by 0.3–0.56 (6–11% for logit≈5, far beyond floating-point reordering magnitude). Recall / host slab / shared-prefix
mirror are all exonerated (KV and state have been proven byte-identical).

Fix direction: chunk width is determined only by `(reuse, prompt_tokens)` (fixed `prefill_chunk`,
`min(prefill_chunk, remaining)`); ring checkpoints land only on these fixed boundaries; abandon the small optimization of
"placing snapshots close to the frontier" (`kReuseTailCheckpointCount`'s dense tail window already bounds the rewind cost
to a few dozen tokens).

**Progress (Round 20, first half landed)**: `snapshot_at[1]` was changed to "the start of the last chunk"
(`span = prompt_tokens - reuse`, `last = span % prefill_chunk`), and the chunk plan no longer reads `rewind_near_`.
Two consecutive runs measured:

- prefill logits are bit-identical across all walks (`#3` and `#19` both `chunk=512+128 top1=197 4.937500`,
  `#9` `chunk=512+128 top1=54 4.406250`) ⇒ **the result for the same prompt no longer depends on engine history**;
- before the fix the oracle's `shared_c` was 4.750000 and recall was 4.406250; now the two are the same `4.406250/4.406250`
  ⇒ the former recall divergence is gone;
- `rewind_near_` is now write-only, i.e. dead code pending deletion (along with `kReuseRewindMinimum` / `kReuseRewindMaximum`).

**Remaining (second half)**: the test case now fails one level down — `shared_a_second` reuses 647 (a per-token decode
checkpoint, falling off the chunk grid), the suffix's first block is `647+17`, while from-scratch is `640+24`, so it still
flips tokens. Fix: **recall boundaries may only land on the `prefill_chunk` grid** (647 → 640, 71 → 0), and the test
assertions change to 640/0 accordingly. Feature value is not damaged: the cross-session shared system prompt boundary 512
is already on the grid; what is given up is only sub-grid reuse of ≤1 block (at the cost of recomputing ≤255 tokens).

**Measured result (alignment implemented)**: after adding the `position % prefill_chunk == 0` filter to the recall boundary
and running once: `shared_c` (512, on the grid) oracle and recall are both `chunk=512+128 top1=54 4.406250` ⇒ exactly
identical; `shared_a_second`'s 647 is downgraded to 512 (the oracle's 640 device snapshot is filtered the same way, both
sides landing on 512) ⇒ the chunk plans on the two sides agree. Cost and side effects:

- the reuse loss is **bounded** (≤1 chunk = ≤255 tokens), because the aligned fallback takes a checkpoint "at some
  chunk boundary no later than the frontier";
- but when **not a single aligned checkpoint exists** (e.g. a small 71-token session, where `opening`'s 64 is also not a
  multiple of 256), reuse drops to 0 = full recompute, so the repository-archived "host slab recall" test case no longer
  covers that feature. That is exactly the current failure point: `a recalled conversation reused 0 prompt tokens, expected 71`.

⇒ A more precise rule (to implement next): **prefer the deepest checkpoint that is grid-aligned and exists; when there is
none, fall back to the deepest checkpoint of any kind**. That way 647→512 (aligned on both sides, exact), while 71 still
keeps 71 (host slab recall for small sessions is not sacrificed).

**1c implemented (two-channel selection)**: first select the deepest among aligned candidates only; if that channel selects
nothing (the whole lineage falls inside the first chunk), fall back to "the deepest among arbitrary candidates". Measured:
the plain route is all green: `recall reused 71 prompt tokens bit-identically`, while the 512 cascade scenario stays exactly
identical.

**Acceptance (met)**: both the plain and mtp routes each output
`TP-2 session retention (plain|mtp) passed: recall reused 71 prompt tokens bit-identically; LRU eviction
forced a full prefill`; the `NINFER_TP2_DEBUG_LOGITS` (including the temporary `pc=` field) and `NINFER_TP2_DEBUG_KV`
diagnostic code, plus the `rewind_near_` / `kReuseRewindMinimum/Maximum` dead code, have all been deleted (residual
references in `tp2_generation_core.cpp` are 0, and the test cases are still all green).

**Round 20 supplementary conclusion (quantifying boundary sensitivity)**: running the same batch of 640-token prompts with
`prefill_chunk ∈ {128,256,384,512,640}` (that value quantized by `% 128`) gives:

- **the last block width itself is not a variable**: with last block 128 (pc=128/256/512), 256 (pc=384), 640 (one of the
  pc=640 prompts), the logits are bit-identical in all three cases (`197@4.937500`, `54@4.406250`);
- **the real variable is "where the last block starts"**: the only difference appeared in the other pc=640 prompt — the last
  block was cut into `504+136` instead of `512+128`, and the logit jumped from 6.218750 to 7.687500 (Δ=1.47, far larger
  than the bf16 step 0.0625), but argmax is unchanged and the token sequence is unchanged by a single character;
- **zero flips at the token level**: 5 chunkings × 3 prompts × 8 decode rounds = 120 steps, argmax identical throughout;
  repeated copies of the same prompt within one run are also bit-identical (reproducible across processes);
- ⇒ the 0.3~0.56 is not quantization noise but a last-column reduction-order difference caused by chunk boundaries
  (magnitude can reach ~1.5); it has almost no effect on greedy results, but an **exact tie** (`shared_c` has 54 and 220
  both 4.406250, gap=0) can flip — which also explains the cascade scenario's FAIL under a non-production
  `prefill_chunk` (the changed grid changed the recompute span, not a defect). The test case's 256 is now fixed,
  and a comment states that it is bound to the recall alignment grid.

**host KV arena made pageable (completed incidentally this round)**: `HostBuffer` (`src/core/arena.{h,cu}`) pages by
default (`std::malloc`, can be swapped out by the OS); only `HostPinning::PreferPinned` goes through `cudaMallocHost`, and
**a rejected pin falls back to pageable automatically**; `HostKVArena` logs the real tier
(`host KV arena shard 0: 64.0 MiB pageable`); the CLI gains `--host-kv-pinned`; docs are updated in step
(`docs/serving.md`, `docs/tp2-dual-5060ti.md`, and the earlier over-strong "per-token identical" wording corrected).
End-to-end verification (`ninfer-serve --devices 0,1 --host-kv-mib 4096`, 7 different
system prompts forcing eviction): the default output is `host KV arena shard 0/1: 2048.0 MiB pageable`, with
`--host-kv-pinned` the output is `2048.0 MiB pinned`, neither run has `refused`, and device memory is released cleanly after
the service is stopped ⇒ a 4 GiB pool can be created and the switch really takes effect. ⇒ `--host-kv-mib` no longer
requires pinned residency and will not fail wholesale because a pin was rejected; the commit charge is unchanged
(pageable ≠ less committed).

**Nasty trap hit and fixed**: the TP-2 normalization in `src/runtime/engine/model_instance.cpp` rebuilds
`ContextCacheOptions` with designated initializers, copying only the fields it explicitly lists ⇒ new fields are silently
dropped (this time `host_kv_pinned` failed exactly that way, showing up as `--host-kv-pinned` having no effect while the
log still reported `pageable`). Adding a field to that struct requires updating this spot as well.
worklog §36.1 records the same class of accident in the same function (back then `--spec` was dropped once too) ⇒ this is the
second time for the same function; when modifying `ContextCacheOptions` the designated initializer in `model_instance.cpp`
must be changed along with it.

### 3.2 Pre-delivery wrap-up (archive §7)

- [ ] Re-test concurrency C=1/2/4 plus streaming/stop smoke with the recommended configuration (the Round 31/34 conclusions were based on the bf16 configuration)
- [ ] On delivery, remove this file (PLAN.md) per AGENTS.md; the ~300 one-off diagnostic scripts under `tools/tp_bootstrap/`
      need a keep/cleanup scope decision (`build_r35.sh`, `serve_*`, the `r37+r38+r39+r4x` series and
      `docs/tp2-dual-5060ti.md` are referenced as the reproducible flow)

### 3.3 Windows single-card CLI tiny-artifact subtask (user decision; prerequisites ready)

Goal: produce a `.ninfer` small enough to fit on a single 16 GiB card, and run "load → generate → sample" single-card with
the Windows `ninfer.exe`.
Prerequisites (Round 54.9): CPU-only torch in `build-win\torch-venv` (torch 2.14.0+cpu); the converter interface is confirmed.
Next steps: ① run the official recipe straight on a tiny checkpoint (`--recipe qwen3_8_27b_nvfp4 --components text`);
② if that does not work, write a tiny recipe/override; ③ put the generation script at `tools/win_port/tiny_model.py`
(synthesize a tiny config + random bf16 weights, HF naming); ④ on success, write it into `docs/windows.md` and commit.

### 3.4 Existing defects (unfixed, low priority)

- The TP-2 route with `bf16` KV + `--spec mtp` crashes on the very first prefill (`prompt.cu:63` `cudaErrorInvalidValue`);
  fp8 + MTP is fine (the recommended configuration is unaffected). Investigation entry point: the launch parameters of
  bf16 prompt-attention.

### 3.5 Deferred / optional (not scheduled)

- Round 15 remainder: end-to-end re-test of tool-call constrained decoding on the real model (75 probes); WARN diagnostics;
  route B (full GBNF); TP-2 `validate_licensed_tokens` guard
- Round 14 candidates (deferred by the user): ① add an anchor at the end of the system+tools prefix (fixes the 46.9 s first
  round after a new session/compression, saves about 11 s); ② prefill throughput (suffix 0.95–1.13k tok/s, cold start
  1.35–1.38k); ③ ~0.3 s fixed overhead per request
- archive §11.6: MTP priming uses placeholder embeddings for image columns (~100 lines, no evidence of benefit); the
  single-image cap 8,192 is hard-coded (yielding shard 1 can regain a larger envelope); shard 0 is idle during vision
  encoding (acceptable)
- archive §6: KV dtype sweep completion (nvfp4/k8v4 verified only startup and throughput, no quality data)

---

### 3.6 Plan for adapting DFlash2 to TP-2 (not scheduled; conclusions from the §7 investigation)

**Status**: DFlash2 is fully implemented and qualified on the **single 5090** route (mathematics and state in
`docs/maintainer/dflash.md`, implementation in `execution/draft.cpp`, `load/dflash{,2}.cpp`, performance and corpus in
`docs/performance/*`); the TP-2 route originally **deliberately rejected** `--spec dflash2` (`model_instance.cpp` throws
`TP-2 generation supports --spec mtp only`, the reason being that the dual-shard loader does not power up the draft
components), and the restriction was explicitly retained (worklog §36.1 `:713`, `:2438`). **B5 has now opened up that
rejection** (see the B5 result below); `--spec dflash` (v1) is still rejected.

**Single-card benefit evidence (ready, cost = reading the docs)**: the repository has already published a single-5090, C=1
DFlash2 K=7 vs MTP3 comparison (`docs/performance/qwen3.8-27b.md:275-307`):

- per-request phase rates: the `nvfp4` tier +64.5% / +21.1% / +19.2% (three AIME long-decode fixtures), Code +36.6%,
  Translation +33.1%, Structured +62.3%, Story −3.8%; the `groupwise-int` tier is mostly negative (Story −35.6%,
  AIME15 −11.1%) ⇒ the benefit **depends on the weight format**;
- whole corpus (C=1): `nvfp4` decode rate **+19.5%**, makespan **−22.7%**; `groupwise-int` −9.1% / +12.1%;
- mechanism: DFlash2 proposes more per round (tokens/round 3.5–6.5 vs MTP3 2.7–3.7), but has a lower acceptance rate
  (35–78% vs 37–91%);
- the document's own caveats: the two campaigns used different revisions and artifacts, and output lengths differ randomly,
  so it is **not an isolated backend comparison**, and **no new MTP3 baseline was run**, nor does it represent answer
  correctness; groupwise-int additionally has an AIME30 repetition-loop outlier; DFlash2 K=15
  and concurrent DFlash2 are **unpublished**;
- no single-card control is possible on this machine: the smallest 27B artifact is 17.4 GB (> 16 GiB), and
  `qwen3_8_27b_w4a4_w8a8_dflash2.ninfer` is 24.8 GB ⇒ **the published campaign can only serve as the benefit evidence**.

**Inference**: this machine's route is w4a4/w8a8 (close to the `nvfp4` tier) ⇒ the benefit is positive and the direction
supports adaptation. But DFlash2's benefit comes from "more proposals per round", and on dual cards the per-round draft cost
(5 layers × two cards + feature gathering) and the verify cost both rise ⇒ whether this +19.5% can be retained depends on
the implementation efficiency of B1–B4, **which is the real risk of this item**. The archive's expectation for dual cards is
**DFlash2 90–180 tok/s** (worklog `:60`), as the B7 acceptance target.

**Existing surfaces directly reusable (MTP already paved them)**: the shard placement mechanism (`tp_split_spec.cpp:87-94` +
`tp_shard_views.cpp:42-61`'s "one card holds it, the other keeps an empty view", plus `has_weight` tolerating missing
weights); the loading seam (`load.cpp:63-85,125-155`); **the entire verification loop** (`forward_tp2_window`, Verify
binding, per-column logits/hidden, `run_verify_window`, RecordForReplay+fold, `speculative_accept_sparse_drafts`); the
independent draft KV geometry precedent (`state/decoder_state.h:20-29`); and the graph bucketing
`dflash_graph_profiles` (`graph_profiles.h:9`, `graph_profiles.cpp:91`, already used in `graphs.cpp:375,399`).

**Places that need new design (breaking the single-card assumptions)**:
1. **Conditioning features across cards**: the proposal condition is a projected concatenation of the target's 5 blocks'
   residuals (`DFlashFeatureSink` in `draft.cpp:59-89`), while TP-2 splits the 64 layers across two cards ⇒ a cross-card
   handoff is needed every step, and the code **has no design for it yet**;
2. **Selector and vocabulary**: `candidate_selector_path` reads the **full-vocabulary** codebook directly (pred/succ each
   248320×256 BF16, about 254 MiB in total), and the proposal head shares weights with the target lm_head (`load.cpp:83`)
   ⇒ TP-2's existing practice of splitting the text head by vocab would break the selector's full-vocab top-k ⇒ the
   codebook must be **copied in full**;
3. **Proposal loop**: MTP's "right-shift the embedding by one + overwrite the last column on device + host serial AR" is
   MTP-specific (`tp2_generation_core.cpp:760-802,1004-1042`); DFlash2 is one non-causal masked block + sparse rejection
   sampling and must be redone (the verification surface does not);
4. **Per-card budget**: the draft measures **2.07 GiB** under the actual encoding of the ready artifact (payload difference
   between `qwen3_8_27b_w4a4_w8a8_dflash2.ninfer` and the same-source artifact without dflash2 = 2,226,792,960 B; 3.59 GiB
   if pure BF16), plus 5 layers of local ring K(BF16)+V(FP16) fixed at 40 MiB with no per-token growth (`dflash.md:274-289`).

**Decision points (settle before writing code)**:
- **D1 ownership**: copy everything, or split by head/FFN? **Measured conclusion: copying is not viable** — under the
  full-context configuration (262144 tok / fp8 / MTP K=2) free is only 697 / 1217 MiB (`docs/tp2-dual-5060ti.md:329-337`),
  while copying needs 2.07 GiB/card; DFlash2 and MTP are also **mutually exclusive backends**, and choosing DFlash2 frees
  shard 0's MTP weights 430 MiB + KV 516 MiB, which still only brings free to about 1.6 GiB. After splitting, about
  1.04 GiB per card is viable, but shard 1 has only about 0.2 GiB left. **Use a small context in the evaluation phase
  first**: on this machine `--max-context 4096` measured free 4344 MiB (§3.1 acceptance record), which bypasses this
  bottleneck for now;
- **D2 quantization**: what format to use for the draft weights (a bf16 copy has no headroom ⇒ quantization or splitting is
  required); first look at the actual precision of the companion weights in the artifact;
- **D3 feature handoff**: capture per block on the target side and then all-gather (5×5120 BF16 per step), or have the
  proposing card hold half each?
- **D4 proposal location**: follow MTP's "propose only on shard 0" (in which case both the features and the codebook must be
  complete on shard 0), or have the two cards each propose half?

**B1 result (completed)**: `dflash2/*` is classified as shard-local in split spec (`shards=0x1`, the same rule as
`mtp/*`), and the `tp_split_spec.h` contract comment is updated in step; incidentally a real defect was fixed: the draft
branch in `parameters.cpp` lacked the shard-existence check (MTP/Vision both have it), so on shard 1 it constructed
parameter blocks from nonexistent weights ⇒ `projection weight must be a matrix`.
After the fix `ninfer_qwen3_5_tp2_load_test` is all green on three cases (`--spec dflash2` head copy,
`--spec dflash2 --lm-head-draft` head split by vocab, `--spec mtp --lm-head-draft` no regression), and the test additionally
asserts that shard 0 holds the whole draft (feature_projection
[5120,25600], draft query [4096,5120], predecessor codebook [248320,256]), while shard 1 does not pay those bytes. The load
test goes straight through the loader and not through Engine option normalization ⇒ the generation path's `--spec mtp only`
gate can stay until B5 is done.

**B2/B6 design points (investigation conclusions)**:

- the capture plumbing is small: `run_layers_tp2` already calls `tap.capture_layer` per layer after the two all-reduces, and
  the two `forward_tp2_*` entry points merely hard-code `NullTap` ⇒ adding an optional `DFlashFeatureSink*` branch is
  enough, and the layer loop needs no change;
- the two cards' residuals are **bit-identical** (all three allreduces do the BF16 local+peer add on the local side) ⇒
  capturing shard 0 alone is enough;
- TP-2 chunk ≤ 1024 < S=2048 ⇒ "store only the last active window" never triggers, so the features buffer only needs one
  chunk's worth;
- **state movement that must be added**: the draft's local context ring is currently not in `state_backing`/host
  checkpoint/host KV slab, and prefix reuse skips the shared prefix (skipped segments get no target residual) ⇒ without
  moving state, the draft context is necessarily missing after recall/reuse; pending features must also be flushed before
  the checkpoint.

**Feasibility conclusion and potential problems (investigation summary)**: functionally viable (loading already measured),
but three things must be solved or confirmed first:

1. **reduced proposal head vocab splitting conflicts with DFlash2 single-card top-k (first, most certain)**: `proposal_head`
   defaults to Optimized, and `tp_split_spec.cpp:123-129` makes it ColumnParallel ⇒ shard 0 has only 131072/2 = 65536 rows;
   while `propose_dflash2_batch` (`draft.cpp:356-370`) does `linear_topk` over the entire reduced vocabulary on shard 0, and
   that operator matches profiles by exact row count (248320 / 131072) ⇒ 65536 throws `unsupported head profile` outright,
   and the DFlash2 path has **no** cross-card merge of any kind (MTP merges with `merge_local_row_blocks` + allreduce in
   `text.cpp:782-825`). **Fixed**: `load.cpp:137-145` makes DFlash2 keep the whole proposal head (`split_proposal_head`
   gains `!dflash2()`); workspace capacity is computed from shard-local parameters (`planning/startup.cpp:631-636`), and
   once the head becomes whole it grows along automatically;
2. **wide-window verify falls off tiny-T**: the target attention's tiny-T kernel implements only T=1..6 (all geometries) /
   T=7..8 (24 heads only), while TP-2 single-card is a 12/2 geometry and MTP's `draft_tokens` is already clamped to 1..5
   ⇒ W≤6 lands right on the fast path. DFlash2 allows K=1..15 ⇒ **W≥7 goes through ChunkedSmallT/Prompt (runnable, but a
   different kernel/slow path)** ⇒ the local benefit should first be estimated at K≤5 (tokens/round drops from 3.59 to
   about 3.0 ⇒ the performance expectation drops from ~108 to about ~90 tok/s), and the actual cost of W≥7 must be measured;
3. **the draft side does not lack wide windows**: `sliding_window_attention` covers T=1..16, window 2048/4096, and DFlash2's
   dedicated fused operator goes to T≤48 ⇒ the bottleneck is only on the target verify side;
4. **Correction: `dflash_graph_profiles` is meaningless on TP-2** — TP-2 forces `use_cuda_graph=false`, `prepare_graphs`
   returns on its first line, and these buckets are never captured; TP-2 uses its own `WindowGraph` +
   `select_window_graph` (`tp2_generation_core.cpp:868-914`, bucketed by `visible_end`, with eager and capture sharing the
   same `forward_tp2_window`) ⇒ the "hook up dflash_graph_profiles" line in the phase breakdown is deleted.

**State surface and timing (investigation summary; highest-risk area)**:

- **the runtime state surface is zero**: the TP-2 core has no `StateImageDevicePool`, no draft local ring, no
  pending/prefill feature buffer (on the device side there is only the GDN pool `tp2_generation_core.cpp:500-523`); the
  single-card side has a ready layout to port (`state/state_image.h:26-30`, `state_image.cpp:117-131,159-164`,
  `startup.cpp:139-165`).
- **Places to touch**: `build_shard` creates the ring and counts it into the state arena slot; `snapshot_host_checkpoint`
  (`:2022-2057`, currently only D2H `state_backing`) gains the ring and the frontier; session slab registration
  (`session_ensure_host_slabs` `:1153-1273`, imitating
  `host_mtp_kv`); `SessionEntry` (`h:190-225`) gains a draft mirror and frontier; four copy points (store `:1296-1317`,
  restore `:1352-1369`, device snapshot `:2008-2013`, round scratch `:2496-2497`); and a new `DFlashFeatureSink*` entry
  for `forward_tp2_prefill/window` (`text.h:196-213` currently has none).
- **Two hard timing constraints**: ① before checkpoint/publish, pending features must be flushed (on single GPU this relies on
  `commit.cpp:305-319`), whereas TP-2's `session_store_active` at request boundaries copies only text KV + GDN + MTP KV,
  neither flushing nor moving the draft face; ② **prefix reuse is inherently in conflict with draft context**: skipping the
  shared prefix produces no target residual for `[0,reuse)` (`execute_walk` starts from reuse, `:2132`), the draft context
  gets holes that the existing mirrors cannot patch back ⇒ either bring the draft context into the reuse boundary state, or
  abandon reuse on this route.
- **Acceptance criterion**: TP-2 has no Verify phase (MTP was pushed back to Prefill back then and recorded as "not
  byte-identical"), so DFlash2 on TP-2 can only be accepted by "no degradation + same-tier quality + speedup", and
  **cannot** be accepted by byte parity.
- **Risk of reproducing a historical trap**: per-device smem opt-in still has four process-level statics
  (`q8_dynamic_grouped_conv_add_materialized.cu:46` and others), which happen to cover the draft's finish path;
  Round 49's "first few tokens normal, then 0 repetition loop" is the silent manifestation of the same class of state
  misalignment.

**Phase breakdown (each phase accepted independently)**:
- **B1 Load and placement**: lift the rejection + `tp_split_spec` gains `dflash2/*` rules (feature_projection/codebook copied,
  context K/V split by head) + per-card draft config/state/plan (`tp2_generation_core.cpp:436-523`) ⇒ acceptance: both cards
  materialize successfully, `ninfer_qwen3_5_tp2_load_test` passes;
- **B2 Context materialization**: two steps. **B2a (done)**: both `forward_tp2_*` entries accept an optional
  `DFlashFeatureSink*`; in prefill `begin` → layer loop → `capture_positions` → `consume_prefill_chunk`, in window
  `begin` → layer loop (batch mode), `NullTap` is still the default. **B2b (done, `ae83d4a6`)**: shard 0
  allocates `prefill_features[hidden*L, chunk]` / `prefill_positions[chunk]` /
  `pending_features[hidden*L, K+1, 1]` and a zero-initialized draft local ring (`plan_cyclic_kv_cache`, lane=1)
  according to the single-GPU persistent layout; the prefill loop assembles the sink: `layers` takes
  `draft.target_layer_ids`, the consumer walks lane 0, whole chunk, through `dflash_append_context` (scratch already
  scoped). The gate is untouched ⇒ at runtime it still always walks `NullTap`, and both plain and
  mtp sessions pass. **Four open items**: ① the ring is not yet included in `session_store_active`/host slab/checkpoint (B6), the
  lane is fixed at 0, and prefix reuse will leave holes in the draft context; ② the workspace peak has not been measured
  (consumer ~60 MB at chunk=1024); ③ verify's batch fields are not wired (B3/B4), `pending_features` is allocated
  but unused; ④ **acceptance (TP-2-only, done)**:
  `tests/models/qwen3_5/test_tp2_dflash_append.cpp` drives the sink via the loader-only path (`plan_load(DFlash2)` +
  `materialize_model_tp2`, bypassing the Engine, gate untouched) ⇒ the prefill last-column logits are byte-identical
  with and without the sink, and identical between the two shards; consumer once per round, `captured_mask=0x1f`, positions exactly
  `[0,1024)`; the ring structure is sound (all live written, nonfinite=0, all capacity slack zero); direct append and the
  sink path are byte-identical; one 1024 call vs two 512 calls are byte-identical; reproducible across
  processes (fnv1a=0x1a53fd824cd4e360). arena 90.1 MiB, **workspace peak 113 MiB (59% of 192 MiB,
  same with and without the sink)** ⇒ open item ② closed. **Still unproven**: the numerical correctness of the ring — the single-GPU
  oracle was confirmed unconstructible by three pieces of evidence
  (loader unconditionally binds text, `Parameters` unconditionally prepares the whole text stack, the whole model does not fit on one GPU);
- **B3 Proposal forward**: masked block 5-layer sliding + dynamic convolution executed per D1 ⇒ acceptance: the proposal is
  deterministic and reproducible for the same features, and **under greedy decoding
  DFlash2 and plain produce exactly identical output token sequences** (speculation only changes speed, not the greedy result; this also covers
  the correctness of acceptance and folding);
  **B3 result (this round, TP-2-only; uncommitted)**: new exit `dflash_propose_batch` (`execution/draft.cpp`, `program/context.h`)
  reuses the production `propose_dflash2_batch` (no rewriting of the math; running the masked block + `linear_topk` +
  `candidate_selector_path` together is reuse of production code, not a newly written selector). Two TP-2 gaps were found on the
  path to making it work: ① `text/token_embedding` is RowParallel
  (`load/tp_split_spec.cpp:117-122`), but the masked draft needs the full-width embedding ⇒ new `TextContext::embedding_full_width`
  (reuses `embedding_tp2`, same path as the MTP stem, `execution/text.cpp`), and `propose_dflash2_batch` uses it when
  `DFlashBatchContext.tp_card` is non-null; ② after the cross-card allreduce finishes, `DevicePair::allreduce` leaves the
  peer device as current (`core/tp/device_pair.cu:440,459`), so the selector's [256,5120] BF16 projection (which needs opt-in dynamic smem)
  sets attributes on the wrong device instance ⇒ after embedding, the proposal must rebind the local card. Test
  `test_tp2_dflash_append` was extended to prefill→append→proposal: K=7 produces draft[7], candidate[16,7], proposal_q[16,7], query
  positions `[1023,1031)`; each row's 16 candidates are mutually distinct and fall in the common token domain, draft ∈ that row's
  candidates, greedy proposal_q is an exact one-hot; two proposals on the same
  ring and a fresh prefill→append→proposal chain are byte-identical (fnv1a=0xbad27a494a9bc853, consistent across two independent
  processes; K=5 gives 0xbee487264ca8ffb8). The per-card draft weights measured as quantized tier (layers/feature_projection=q8_g32_fp16,
  codebook=bf16, tied output head=fp8_e4m3fn), shard 0's draft block (including the shared tied head 1213 MiB) 3655.4 MiB;
  proposal frame 4.4 MiB, instantaneous workspace peak 4.4 MiB (2% of 192 MiB), event timing **7.95 ms/iteration (K=7), 7.93 ms/iteration (K=5)**. **Not done this round**: the selector's
  Engine publication, verify/accept/fold, session/checkpoint state (B4–B6); therefore the plan's "greedy DFlash2 == plain" end-to-end acceptance cannot be established this round,
  and must not be claimed before B5.
- **B4 Round assembly (done, `b4e84056`)**: the selector math was already running in B3 with the production `propose_dflash2_batch`; this phase
  converges the whole round into the production component
  `program/dflash_round.{h,cpp}` (draft persistent context + decode frame of exact B + **independent proposal arena** + publishing K drafts /
  candidate ids / proposal q / query ids+positions), collapses core's six shard fields into a single `dflash_round`, changes the sink factory to a delegation, and
  single-GPU and dual-GPU share `dflash2_proposal_workspace_bytes()`. **Arena ownership pitfall proven fixed**: after poisoning shard 0's resident `prefill_hidden`,
  proposing again ⇒ the pattern survives byte-identically, the shard workspace count is unchanged, and the component arena peak is 4.4 MiB (K=7)/3.3 MiB (K=5),
  exactly matching the planner budget. Acceptance: the assembled round reproduces the B3 hashes byte-for-byte (K=7 `0xbad27a494a9bc853`, K=5 `0xbee487264ca8ffb8`, across binaries/processes),
  one proposal 7.95 ms; context 90.4 MiB (0.34 MiB more than B3, because the component allocates `pending_features` per production `lanes=K+1=8`, currently read by nobody);
- **B5 result (done, worktree uncommitted)**: DFlash2's masked draft has now run the full round on TP-2 — "prefill sink → append_pending →
  masked proposal → target verify (with feature sink) → sparse rejection sampling → GDN fold → terminal append" — with the gate opened in
  `model_instance.cpp` (`--spec dflash2` allowed, `--spec dflash` still rejected).
  - Changes: `program/dflash_round.{h,cpp}` gains `make_verify_sink()`/`append_pending()`/`draft_window()`/
    `feature_lanes()`; `runtime/engine/tp2_generation_core.{h,cpp}`'s `run_verify_window` gains a sink parameter, the constructor establishes
    DFlash2 flags/draft count/context frontier, `build_shard` builds GDN records + `replay_fold` for DFlash2 (width=drafts+1), the
    decode branch is implemented in the order of the single-GPU `execution/draft.cpp:dflash_decode_batch_body`, and extent==0 falls back to a plain step;
    `src/runtime/engine/model_instance.cpp:97-112` opens the gate and turns off context cache/host checkpoint/session retention for this route
    (the draft ring has not yet entered B6's state mirror).
  - Key defect (fixed this round): the verify window's ids/positions are device-side products; after D2H into the pinned buffer they were **not
    synchronized** before `forward_tp2_window` read them on the other card's stream (MTP's window is written directly by the host, so the existing
    path never exposed this) ⇒ before the fix the output degraded into repeated fragments; after adding one
    `cudaStreamSynchronize` it is coherent.
  - Acceptance: K=7/K=5 `ninfer_qwen3_5_tp2_dflash_append_test` hashes unchanged; `load_test` three cases, `sessions_test` all green;
    on `ninfer-serve --devices 0,1 --max-context 4096 --kv-dtype fp8 --greedy --no-prefix-reuse` all 5 prompts are coherent
    (longest same-word repetition=1), and repeated requests for the same prompt are byte-identical (determinism).
  - **"DFlash2 == plain token-for-token" does not hold, and is impossible in this engine**: window forward and single-token decode are different execution
    shapes, and per-column logits differ (`docs/tp2-dual-5060ti.md:172-176` already states this for MTP). This round's measured comparison (same artifact, same prompt, greedy 128
    tokens): MTP K=2 and plain diverge at a common prefix of 320 characters; DFlash2 **K=2 at 390 characters (better than MTP K=2)**, K=7 at 207
    characters (shorter as the window widens). Therefore B5's acceptance was changed to "no worse than MTP and no degradation", which is satisfied; strict
    token-for-token identity is recorded as an **unestablishable** item.
  - Measured (4096 context, fp8 KV, greedy, 65 prompt tokens→128 output tokens): plain 35.1 tok/s; MTP K=2 62.8 tok/s
    (56 rounds/127 commits=2.27 token/round, acceptance rate 63.4%); DFlash2 K=2 51.0 tok/s (58 rounds, acceptance rate 59.5%); **DFlash2 K=7
    63.6 tok/s (41 rounds/127 commits=3.10 token/round, acceptance rate 30.0%)**, 1.81× plain, on par with MTP K=2. It did not reach §3.6's
    108 tok/s roofline estimate — DFlash2's verify is currently **eager** (the feature sink cannot enter the graph); same-width verify costs about 4 ms/round more than MTP,
    and the proposal about 6.4–6.8 ms/round. Memory: shard 0 `weights+ctx 13878.6 MiB`, free 1764 MiB (draft 3655.4 MiB);
    the two proposal heads (default Full and `--lm-head-draft`) agree in output and rate;
- **B6 result (done, worktree uncommitted; retention stays disabled)**: DFlash2's masked draft context (local cyclic K/V ring +
  its absolute frontier) now goes through all of TP-2's state channels together with the target GDN/KV, but session retention is **not** reopened: the
  "recall byte-identical" required by the retention acceptance does not hold for this route on synthetic prompts (see below). Changes:
  - `program/dflash_round.{h,cpp}`: new `context_image_bytes()` and
    `copy_context_to_host/from_host/to_device/from_device` — pack the ring layer-major (K then V, tightly packed by layer extent,
    skipping alignment pitch) into a flat image shared by the host mirror and the device snapshot;
  - `runtime/engine/tp2_generation_core.{h,cpp}`: `HostCheckpoint` gains `dflash_buffer`+`dflash_frontier`,
    `Shard` gains `dflash_snapshots[0..1]`+arena, `SessionEntry` gains `host_dflash`/`host_dflash_prompt`/
    `host_dflash_shared`; `build_shard` accounts by actual allocation; `session_ensure_host_slabs` forces allocation of
    `host_dflash` for shards owning a draft (on failure the session is rejected); `session_store_active`/`session_restore`/
    `session_capture_shared_state` move target GDN and the draft ring in pairs by `RecallState` (Frontier/PromptEnd/Shared);
    `snapshot_host_checkpoint` records `dflash_frontier`; `execute_walk` gains `begin_dflash_state` (None→zero,
    LiveState→no-op, DeviceSnapshot/HostCheckpoint→restore by image, HostCheckpoint validates `dflash_frontier == position`);
    before publish/cancel/publish, `flush_dflash_context` commits pending staging; `extent==0` remains a "real round" (otherwise the last column's
    draft residual is missing and a hole is left at the frontier); the reuse scan is opened for DFlash2 (removing the residual `!dflash2_enabled_` guard);
  - `tests/models/qwen3_5/test_tp2_sessions.cpp`: route parameterized `plain|mtp|dflash2` (by default only the first two run), DFlash2 uses
    `draft_tokens=7` + fp8 KV.
  - Acceptance: `ninfer_qwen3_5_tp2_dflash_append_test` K=7 `0xbad27a494a9bc853`, K=5 `0xbee487264ca8ffb8` unchanged;
    `ninfer_qwen3_5_tp2_load_test` three cases pass; `ninfer_qwen3_5_tp2_sessions_test` plain/mtp all green.
  - **Recall boundary diagnosis (this round, diagnose before acting)**: DFlash2's recall boundary comes from the reuse scan at `tp2_generation_core.cpp:1965-2045`;
    the aligned scan only accepts `boundary % reuse_grid == 0` (`reuse_grid = min(max(prefill_chunk,64), maximum) = 256`), but the
    fallback branch at `:2021-2045`, when `reuse==0`, **drops the grid restriction** and rescans the same batch of boundaries (live frontier / `cached_boundaries_` /
    host checkpoints), so prompt-end recall pins reuse at 64 (64 is not an integer multiple of 256). The prefill chunk loop
    (`for (t0 = reuse; …)` at `:2455` and `length = min(prefill_chunk, prompt_tokens - t0)` at `:2489`) **anchors the chunk
    at reuse**, so the recall suffix's first block has only 24 columns, whereas the same segment in a from-scratch walk has 88 columns (one block). The fp8 target logits
    flip at a near-tie in the synthetic prompt (same token pair 220/198), and the greedy answer diverges in the rerendered output (`[2752 11 220 …]` vs
    `[2752 11 198 …]`). The column-width difference does not flip plain/MTP, so they pass. Three diagnostic measurements:
    - chunk 256 + retention ON: the grid-aligned `shared_b`@512 (DeviceSnapshot) matches the oracle round by round and passes; the failure is at prompt-end 64.
    - chunk 128 (effective value after normalization): even the grid-aligned `shared_b` flips at the last token (198 vs 220) ⇒ whether the boundary is on the grid
      is **not** the only factor; the chunk width itself also participates.
    - forcing DFlash2 to `extent=0` throughout (running only target rounds): `shared_a_first` diverges immediately (`got [197 197 92 198 198 198 1464 198]`
      vs `expected [197 197 92 198 695 197 197 92]`) ⇒ the target side running alone **also** does not reproduce the oracle's window rounds.
    **Mechanism conclusion**: DFlash2's generated tokens depend on each round's verify window layout, and the window layout is driven by "proposal → sparse acceptance → commit prefix";
    the chunk width at the recall point changes the suffix walk's window sequence, the acceptance counts change accordingly, and the output flips at the near-tie. So "anchor chunks at absolute
    boundaries and truncate only the first block at the reuse point" cannot make recall for prompt<chunk identical to from-scratch (the first block at `:2489` is still `[reuse, prompt)`),
    nor can "draft declared invalid" guarantee byte identity (see the `extent=0` diagnosis above).
  - **Fallback contract (implemented, option B chosen this round)**: when the recall boundary is not on the prefill grid, `tp2_generation_core.cpp:2044-2052` sets
    `dflash_draft_declined_` (`reuse != 0 && reuse % reuse_grid != 0`), and every DFlash2 round for that request has `extent=0`
    (`:2778-2788`), i.e. only target rounds run; target-side KV/GDN reuse continues **as usual** (the state mirror is still restored via `begin_dflash_state`). The contract is exposed through
    `include/ninfer/types.h`'s `GenerationResult::draft_context_declined`, asserted by
    `test_tp2_sessions.cpp`'s `draft_declined`/`compare_recall`. Rationale: the draft only proposes; every token still goes through target
    verify; an invalid draft context only costs acceptance rate and does not affect the validity of the target's own tokens.
  - **Failed item (decided not to reopen retention)**: the grid-aligned `shared_b`@512 recall diverges from the from-scratch walk in a fixed shape:
    engine `[1703 220 248046 198 248045 198 248045 198]` vs oracle `[1703 220 248046 198 248045 198 248045 220]`
    — **the first 7 tokens and each verify round's base/extent/licensed agree item by item; only the last round's token differs**; that is, **the recall path
    itself is clean** (state, KV/GDN, boundaries, round sequence all reproduce), and the divergence lands on DFlash2's window's last **`extent=0` clamp round**
    (all 8 columns clamped to the same position at the anchor, `:2784-2826`). That round is **not reproducible even on the same binary**: `b6-acc2-dflash2.log`
    (FAIL) and `b6-rep1/rep2.log` (PASS) are results of the same build; the same binary run 5 times in a row (`build-win/b6-det-1..5.log`) gives
    1 pass (`b6-det-1`) and 4 failures (`b6-det-2..5`), and the 4 failures' `got` are **character-for-character identical**
    (`[1703 220 248046 198 248045 198 248045 198]`, oracle last digit 220) ⇒ the failure is **biased** rather than purely random.
    ⇒ DFlash2's generation in this engine is **not reproducible**: at the near-tie of the clamp round there is a race/unsynchronized read (the same input flips 198/220),
    which shares the root cause of "DFlash2 window ≠ single-token decode" under the window-shape heading, but **it is an implementation defect, not a pure numerical shape limitation**.
  - **Conclusion (one line)**: **not a recall boundary/state defect, but the non-reproducibility of DFlash2's verify window clamp round (`extent=0`)** —
    the recall path reproduces state/KV/GDN/boundaries/round sequence (the first 7 tokens and each round's base/extent/licensed agree item by item), and the flip happens only in
    the window's last clamp round, and that round also flips randomly on the **same binary** (1 failure in 3 runs). Per §3.6's decision rule DFlash2 does not reach the
    retention property ⇒ DFlash2's retention disable at `model_instance.cpp:105-118` **stays disabled**; this route does not claim "recall byte-identical"; and
    the contract assertion for `NINFER_TEST_ROUTE=dflash2` will still FAIL when the clamp round flips (this route is not in the default route set).
  - **Wrap-up contract**: in the final state `NINFER_TEST_ROUTE=dflash2` yields `[mem] host sessions capacity 0 | retention disabled`
    and **exit 0** (`route_keeps_retention` + `compare_recall`): retention off ⇒ no cross-session recall occurs, every recall scenario expects reuse of 0
    tokens (returns the whole prefill for the session), and the only allowed non-zero reuse is prefix reuse of the shared system prefix via device snapshot (`shared_b` 512); plain/mtp
    keep their original byte-reuse assertions. The `draft_context_declined` fallback contract remains (it is a correct improvement: it only loses acceptance rate, not correctness).
  - **To reopen retention later, first solve**: byte-level reproducibility of DFlash2's verify window **`extent=0` clamp round**. This round clarified the KV/attention
    index chain of that round and **refuted the initial guess** (that the clamp columns read the KV of the previous round's rejected proposals):
    - The clamp columns are constructed in `src/ops/kernel/speculative_round.cuh:33-37`: `verify_ids[j>extent]=anchors[row]`,
      `positions[j>extent]=base_positions[row]+extent` ⇒ the clamp columns are an **exact copy of column `extent`** (same token, same position);
    - The cache slot and RoPE position are bound to the same clamp array (`src/models/qwen3_5/execution/text.cpp:2193-2194`, both `cache0`/`rope0`
      use `bind0.positions`), and the KV slot is derived from that position via the page table ⇒ the clamp columns read **column `extent`'s own slot**, not a stale row elsewhere;
    - The write side is masked: `small_t_fp8.cuh:94-98,191` (`valid_tokens=min(valid_columns[batch]-column_begin,TokenTile)`, only the first
      `extent+1` columns write KV); the read window's upper bound is the clamped `last_pos+1` (`:147-159`); sink/append are likewise masked by `target_valid_columns`
      (`execution/draft.cpp:578-660`, `program/dflash_round.cpp:339-385`) ⇒ **the clamp round is deterministic within the target window and correctly masked**;
    - So the only cross-run variable left is TP-2's **cross-card allreduce transport**: on this machine `cudaDeviceCanAccessPeer(0↔1)==0` (probe
      `build-win/tmp_peer_probe.cu`), which goes through the in-kernel mapped-host
      arrival-token handshake at `src/core/tp/device_pair.cu:95-170` (merging the window `[V,8]` logits is the `ar_slices==5` multi-block path, `:200-205,448-466`), the only
      asynchronous cross-card mechanism on this route. An A/B was tried this round: temporarily adding a `host` value to `NINFER_TP2_AR_STRATEGY` to force host staging (`:492-527`),
      but that path is **not a valid control** — all 5 runs FAIL, and even the first "fresh conversation" scenario produces garbage (`got [15 15 …]`,
      `expected [548 271 1919 5686 …]`, outputs that differ completely across runs), indicating that host-staging's D2H does not cover the production kernel (that fallback
      path has long been unused and is broken); hence the A/B conclusion is **invalid**, and the cross-card transport suspicion is **neither ruled out nor confirmed**. The temporary
      switch has been reverted (`device_pair.cu` has no diff in the final worktree) and is not left in the source;
    - Before the verdict lands, **do not change the clamp-round semantics** (that would amount to changing the fallback/acceptance semantics), nor open retention; until then do not do §3.6's B7 speedup.
  - **This round's wrap-up (trial (b) failed ⇒ route returns to construction-time rejection; supersedes lines 429–432 above and earlier versions of this item)**: per the product decision, option (b) was
    implemented first (remove DFlash2 from the **reuse scan itself**, set `host_checkpoint_stride_=0` at construction, allow the route, assert `reused=0` in tests), but **measurement proved prefix reuse
    is not the cause**, so all (b) changes were reverted (`git checkout 61cd270d -- src/runtime/engine/tp2_generation_core.cpp/h
    src/runtime/engine/model_instance.cpp tests/models/qwen3_5/test_tp2_sessions.cpp docs/serving.md`), the route is restored to **construction-time rejection**,
    and the B1–B6 implementations and tests are kept as before, with the `draft_context_declined` contract and B6's draft state-transfer code untouched; the rejection message and comments, `docs/serving.md`,
    and the test header comment rationale have been synchronized to "under the same configuration the verify walk differs run to run", no longer attributing it to prefix reuse.
    **Negative evidence**: `NINFER_TEST_ROUTE=dflash2` run 5 times in a row gives `P F P P F` (`build-win/b6c-dflash2-1..5.log`), the failure still being
    `shared_b` (`test_tp2_sessions.cpp:406`) with `got [1703 220 248046 198 248045 198 248045 198]` vs `…220` — **character-for-character identical** to when reuse was
    on; and :401's reuse expectation (DFlash2 = 0) also passes in the failing runs, showing those runs indeed had no reuse. ⇒ The previous round's conclusion "locked to the reuse walk"
    is **refuted by this experiment**, and B5's premise "DFlash2 is deterministic when it does not participate in reuse" likewise does not hold.
    **Why this settles it**: after cleanly removing DFlash2 from the reuse scan (`reused=0`), the bad value is character-for-character identical to when reuse was on, and the failure occurs between **two full
    prefills** ⇒ reuse is not the cause. New localization: in the same process, with the same set of options, **two Engine instances' DFlash2 walks give different results** — the oracle
    instance (constructed first) is stable at `220` in 10 runs, while the engine instance (constructed later) flips 6 times in 10; the earlier 10 "with reuse" runs likewise error only on the engine
    instance (`build-win/b6-det-1..5.log`, `b6-fin-1..5.log`). plain/MTP in the same scenarios are byte-identical in 5+2 runs; the clamp-round index chain and the bit-exact
    allreduce probe (`tests/test_tp_device_pair.cpp`: 64 data swaps per size + 200-deep queueing, including 20 KiB / 1 MiB / 80 KiB window layers /
    2433024 B `[V,8]` 5-slice merge; `DevicePair(0,1)` all 5 sizes byte-identical PASS, 3.5 s, `p2p_available=0`) have all been excluded ⇒ what remains
    is "the part of the state or timing that differs run to run under the same configuration".
    **Preconditions for re-allowing (next experiment)**: in a failing run, align the engine's and oracle's `shared_b` round by round — compare round by round the licensed columns, verify-window logits and KV/GDN state in `tp2_generation_core.cpp`'s
    proposal/verify/argmax section (:2813-2941); and swap the two engines' construction order
    (oracle built later) to see whether the difference follows "the instance constructed later", distinguishing allocation-layout-related from walk-itself-related. After fixing, restore the reuse assertions per route.
    Other acceptance (measured on the (b) binary; after revert plain/MTP behavior is unchanged): plain 2/2, mtp 2/2 byte-reuse assertions pass; append K=7
    `0xbad27a494a9bc853`, K=5 `0xbee487264ca8ffb8` unchanged; load passes; after removing `[mem]` lines from the 5 dflash2 logs, passing runs are byte-identical
    to each other (SHA256 `324835E9…`), failing runs identical to each other (`8CF800…`).
    Performance: **decode tok/s not re-measured** — this round's changes only touch option normalization and the reuse scan, and do not touch any decode/attention/head/kernel file; reuse only
    affects prefill, so the TP-2 DFlash2 K=7 63.6 tok/s vs plain 35.1 tok/s recorded in B5 (=1.81×; 4096 context, single request, greedy) is unaffected.
    **Single-instance determinism ground-truth experiment (next round's item 1, done)**: new probe `tests/models/qwen3_5/test_tp2_dflash_solo.cpp`
    (target `ninfer_qwen3_5_tp2_dflash_solo_test`) — **a single process constructs only one Engine**, uses that engine's own answers to continue and replay the session scenario's request sequence
    (opening → other_a → a_continued → other_b/c/d → a_continued(evicted) → shared_a → shared_b), printing each walk's tokens and fnv1a digest.
    After temporarily opening the route, **an independent process run 10 times in a row** (`build-win/b6f-solo-1..10.log`), `shared_b` (reused=512) is in order:
    7 times `[1703 220 248046 198 248045 198 248045 198]` (digest `0x4bcc3994a5efba7d`),
    1 time `… 248045 220]` (`0x4bcc3f94a5efc4af`), 1 time `… 248045 248046]` (`0x49045194a1360b45`).
    ⇒ **the single-instance, single-process product form itself is non-deterministic**, and there is more than one wrong value ⇒ it is not a harness artifact of "two Engines coexisting in one process", **it is genuinely a product defect**,
    and it also shows that this token's top-2/3 logits differ by only ~1 bf16 ulp. The gate therefore stays closed; the probe is treated as SKIP(77) when the route is rejected, and remains in the tree
    as an acceptance tool for re-allowing.
    **Next steps (items 2–4)**: (2) swap the two instances' construction order (oracle built later) to see whether the difference follows "the instance constructed later", and in a failing run do
    engine/oracle round-by-round alignment (`tp2_generation_core.cpp:2813-2941`'s proposal/verify/argmax section); (3) use poisoning to check candidate buffers one by one
    (draft ring / pending_features / round arena / GDN records / snapshot slots / paged KV) to find the one not fully overwritten; (4) replace the oracle with
    **Experiment A (round-sequence split, done, decisive)**: print each DFlash2 round's proposal window and target licensed prefix per round (temporary hook `NINFER_TP2_DIAG_ROUNDS`,
    already reverted), run the solo probe 6 times in a row (`build-win/b6h-round-1..6.log`):
    - Every time it is **48 rounds, the round-by-round text fully identical, only the last round (`shared_b`'s 5th round, position=646) differs**;
    - That round's **draft window is byte-identical** across the 6 runs: `window=[248045 248045 248045 248045 248045 248045 248045 248045] licensed=1`;
    - Only the **target's licensed/committed token differs**: 5 times `step=[198]`, 1 time `step=[248046]`.
    ⇒ By the split rule: **the perturbation is on the target verify side, not the proposal/draft side**; and it manifests on only this one round (the other 47 rounds fully identical, showing the whole state trajectory is
    byte-identical, even the acceptance prefixes). That round's window is all the same id (draft collapse/clamp round), and `licensed=1` means the target itself replaced the draft ⇒ what is voted on is **that round's column-0 logits
    argmax**, flipping between `198` and `248046` with a gap of about 1 ulp.
    **Judgment and remaining possibilities**: the state trajectory (KV/GDN/ring) is byte-identical ⇒ not state transport, not reuse, not the draft path, not cross-card allreduce (bit-exact probe); the suspicion falls on
    a **non-deterministic read/reduction in this round's target window computation that affects only the last column's logits** (uninitialized or previous-round-residue-contaminated workspace, verify window buffer,
    `verify_window_host_`, or that clamped row under an all-identical window). Next steps (not done this round): poisoning (B, write patterns to candidate buffers to see which is not overwritten), construction-order
    swap and plain-oracle control group (C), time-boxed `compute-sanitizer --tool initcheck` (D); A has already narrowed the range to "the last round's column-0 logits", and
    the fastest next step is to dump that column's top-k logits and all intermediates for that round (`window_hidden` hash, KV row hashes) and compare run by run.
    **Experiment 1b (last-round logits/hidden dump, both attempts failed)**: added a temporary hook `NINFER_TP2_DIAG_LOGITS` (prints that round's column-0 top-5 and
    the hash of `target_hidden`); all 6 runs fail at the **first round**: `tp2_generation_core.cpp:2924: CUDA_CHECK(cudaMemcpyAsync(host_logits.data(),
    frame.target_logits.data, logit_count * sizeof(float), D2H, ...)) failed: cudaErrorInvalidValue`; first using the view pointer (`window_logits.data`),
    then the base tensor pointer (`frame.target_logits.data`), the result is the same ⇒ `vocab * width * sizeof(float)` does not match that tensor's actual allocation (very likely it is not
    `[vocab, width]` fp32 but a local half / bf16 / merge result containing only the last token). **To make the "input buffer vs computation internals" split, one must first obtain that tensor's own
    byte count or element type** (the Tensor's size/bytes fields, or probing by shrinking the copy length step by step). Logs: `build-win/b6j-logits-1..6.log` (view-pointer version),
    `b6k-logits-1..6.log` (base-tensor version).
    **This round stops deep-digging with a bounded wrap-up, and the gate stays closed.** Remaining possibilities (all unverified): (i) a **workspace read in that round's target window that is uninitialized or contaminated by the previous round's residue**
    (especially the all-identical window, the `licensed=1` branch/clamp path differing from other columns); (ii) a **cross-stream write-read ordering gap** in the two-card window forward (timing-related ⇒ low frequency, present in every round,
    visible only in the last round because the logits happen to tie); (iii) non-deterministic ordering in that round's head/merge reduction. New means needed to distinguish them: obtain the real byte counts of `frame.target_logits` /
    `frame.target_hidden` and hash across runs; run the solo probe under a time-boxed `compute-sanitizer --tool initcheck`; or do a per-kernel
    race check (nsys/sanitizer) on the window forward.
    The temporary hooks have been cleanly reverted with `git checkout -- src/runtime/engine/tp2_generation_core.cpp src/runtime/engine/model_instance.cpp` (the worktree, for these files
    **Diagnostic round (1c tensor facts + artifact comparison)**:
    (i) **Tensor facts** (`src/models/qwen3_5/program/round_buffers.cpp:214-218`): `target_logits` = **BF16** `{output_rows=248320, columns, batch=1}`,
    `target_hidden` = **BF16** `{hidden=5120, columns, batch}`; the `vocab` in the view at `tp2_generation_core.cpp:2844` is the **full vocabulary 248320**.
    The first two dumps used `vocab*width*4` (assuming fp32), which overruns by 2×; `cudaErrorInvalidValue` came from that; after changing to **2 bytes/element the copy succeeds**.
    (ii) **But that dump's content does not agree with the target's decision**: the host-side top-5's maximum is only 7.6–12.8, with ids differing each time, while the same round's `frame.target_argmax` is stably `[198 …]`
    ⇒ the bytes read are **not the rows the argmax reads** (a write-side layout/offset problem). So the split "input buffer differs vs computation internals non-deterministic" **still does not hold**, and
    the hashes obtained are insufficient for a conclusion (cross-run variation only shows the buffer contains bytes that were never written). The new means needed next: read the head's write side
    (`project_head_tp2` / `merge_local_row_blocks`) to confirm the stamped row range and layout, then hash only **the rows the decision actually reads**.
    (iii) **Artifact comparison** (same binary, same scenario, gate temporarily opened to run the solo probe, 10 runs per artifact; `build-win/b6n-{a,b,c}-1..10.log`):
    a) `qwen3_8_27b_w4a4_w8a8_dflash2` (user conversion): 9× `[1703 220 248046 198 248045 198 248045 198]` (digest `0x4bcc3994a5efba7d`)
    + 1× tail token `6558` (`0x4bb38194a5c5b9d5`) ⇒ **flips**;
    b) `qwen3_8_27b_w4a4_dflash2` (the same model's other w4a4): **10/10 fully identical** (`0xad284a4b774b1cc3`) ⇒ no flip;
    c) **official artifact** `qwen3_8_27b_nvfp4` (`README.md:18`, `docs/performance.md:22`): **4 distinct outputs in 10 runs** (6× `2523`, 1× `248046`, 1× `3710`, 2× `7734`)
    ⇒ **the official artifact also flips**. Per the preset interpretation rule: **the engine defect "same binary + same input ⇒ same output" is confirmed by the official artifact, and is unrelated to the user's conversion**; artifact b being stable only means the exposure surface
    relates to that artifact's numerics/tie positions. The last-round candidates have accumulated 7 distinct tokens (198/220/248046/6558/2523/3710/7734) ⇒ the logit gap there is near the bf16 quantization minimum
    step (this dump could not give trustworthy numbers).
    **Diagnostic round 2 (head write side + valid-row dump)**:
    (a) **Head write-side facts**: `src/models/qwen3_5/execution/text.cpp:1608-1641` (`project_head_tp2`) — `vocab = dimension(config_.vocab_size)` (=248320),
    `local = dimension(lm_head_->weight.n)` (=124160, half per card); when `local != vocab`, each card projects only its own half into `partial [local, columns]`,
    then `merge_local_row_blocks(...)` (:1640) stamps/sums the two halves by vocabulary offset into the full `[V,T]`; `program/round_buffers.cpp:214-218` defines
    `target_logits` as **BF16 {output_rows=248320, columns, batch}**.
    (b) **Valid rows**: `tp2_generation_core.cpp:1888-1892` — lm_head is packed by `vocab` rows, but only the first `public_tokens` rows are real (measured at runtime **248077**);
    `src/ops/wrapper/argmax.cpp:37`'s `argmax(logits,out,valid_rows,stream)` requires rank-2 `[vocab,T]` (:45) and uses the same valid_rows
    ⇒ the argmax at :2911 only scans **[0, 248077)**. The previous round scanned `[0,248320)` and read packed rows the head never writes; that is why the dump disagreed with `target_argmax`.
    (c) **Still inconsistent after scanning only valid rows** (`b6p-logits-1..8.log`): 8 runs of `pos=646 valid=248077 hidden=5120`, host top0
    `31074/217312/125935/217282/62082/…` (bf16 `0x40e3–0x4126`, `gap_ulps=2..18`), while the last token in all 8 runs is `198`, `target_argmax=[198 …]`;
    `logit_hash`/`hidden_hash` **differ on every run**.
    (d) **Double-copy discriminator** (`b6q-logits-1..4.log`): copying twice in a row within the same round and synchronizing, 4/4 `same=1` (hash and top0 fully identical) ⇒ **not a concurrent-writer overwrite (not a race)**,
    but rather the address/slot I read is **not that block of memory which argmax actually reads**.
    ⇒ The split still does not hold. The next step must first confirm on the model side which tensor/arena slot the window verify head writes logits into (`forward_tp2_window` entry,
    `program/speculative/target_verification.cpp:18-41`, the slot binding of `round.frame()`), then use that real pointer for cross-run hashing; or run a time-boxed
    `compute-sanitizer --tool initcheck`. The hook has been reverted; the gate stays closed.
    **Diagnostic round 3 (P1 "re-run the same round from the same state", did not hold)**: immediately after the :2911 argmax, run once more `round.make_verify_sink()` +
    `run_verify_window(...)` + `ops::argmax(...)`, comparing `frame.target_argmax` between the two (temporary hook `NINFER_TP2_DIAG_RERUN`, reverted).
    4 runs (`build-win/b6t-rerun-1..4.log`): all 48 rounds are `equal=1`, **but the probe itself changed the walk** — the `shared_b` output went from
    `[1703 220 248046 198 248045 198 248045 198]` to `[1703 220 16 15 15 15 15 15]`, and the walk ends early at `pos=643`
    ⇒ the second verify is not a "re-run from the same state": although the KV writes are idempotent, `make_verify_sink()`'s residual collection has **append** semantics, and the round's pending staging
    was written twice ⇒ **`equal=1` cannot serve as evidence that "the computation is internally deterministic"** (honest conclusion: this criterion did not hold).
    ⇒ For P1 to hold, there must first be a **round-state restore primitive covering pending feature staging** (the existing `snapshot_state(..., kRoundScratchSlot)` only covers
    the KV/GDN category), or the pending staging must be reconstructible before the second run; this is exactly the concrete new instrument the next step needs.
    **Diagnostic round 4 (P1b same-sink re-run and initcheck, neither held)**:
    (a) **P1b**: the second verify reuses the round's own sink (no new `make_verify_sink()`), and the self-check **fails** — 4 runs (`build-win/b6w-rerun-1..4.log`)
    still show `shared_b` as `[1703 220 16 15 15 15 15 15]` (correct value `[… 248045 198]`), and the trajectory collapses as before; `mismatch=20` and identical across all 4 runs (systematic rather than random)
    ⇒ the **state difference between the two is caused by the probe itself** (the second verify rewrote state that later rounds actually read; reusing the sink is not enough to eliminate it), so `mismatch`
    **cannot** be taken as evidence of "internal computational non-determinism". To establish P1, there must first be a **round-state restore primitive covering KV accounting/pending staging**.
    (b) **initcheck is unavailable on this machine**: `compute-sanitizer --tool initcheck` (`build-win/b6v-initcheck.log`) reports
    `Failed to initialize WDDM debugger interface. Please run EnableDebuggerInterface.bat as an administrator` and `Device not supported`,
    `ERROR SUMMARY: 4 errors` all of these two kinds; the probe itself still PASSes normally (43 s, original `shared_b` value, digest `0x4bcc3994a5efba7d`) ⇒ P3 is excluded in this environment
    (requires administrator privileges + device support). ⇒ **Neither P1 nor P3 holds; the gate stays closed**, and work stops under the bounded wrap-up.
    **Diagnostic round 5 (S1 "thorough synchronization" experiment, **hypothesis holds**)**: only in the DFlash2 branch, insert a one-shot thorough synchronization before `run_verify_window(...)` (`tp2_generation_core.cpp:2908`)
    **Wrap-up round (S1 fix landed + final-binary acceptance; conclusion: not fully stable, the gate stays closed)**:
    **Wrap-up round 2 (D root-cause probe + N=50 fast-fail protocol; not passed, the gate stays closed)**:
    (1) **D (one level deeper into the window forward)**: read `src/models/qwen3_5/execution/text.cpp:2162-2222` — each card binds the window with **its own arena** (`make_bind` → H2D
    `copy_i32(ids/positions)` on that card's own stream), immediately followed by `embedding_tp2` + `run_layers_tp2` (paired layer sequence, peer exchange via
    the in-kernel mapped-host handshake of `src/core/tp/device_pair.cu`), with **no host-side staging/event in between**; the two cards' streams were already drained by the bounded per-round synchronization before entering the window
    ⇒ **there is no "minimal necessary dependency" left to add on the host side**; the residual non-determinism lies in the paired in-kernel protocol/reduction (requires per-kernel ordering inspection, or running the sanitizer in an environment with a supported device / administrator privileges).
    (2) **N=50 fast-fail protocol** (`build-win/b8-solo-*.log`, the script stops at the first inconsistency): **fails on run 10** — digest `0x4bb38194a5c5b9d5`,
    tokens `[1703 220 248046 198 248045 198 248045 6558]` (the other 9 runs all `0x4bcc3994a5efba7d`). Aggregating three batches after bounded synchronization: `b6y`
    (12 runs 0 failures) + `b7b` (12 runs 1 failure) + `b8` (failure on run 10) = **34 runs, 2 failures ≈ 5.9%** (95% Wilson interval roughly 1.6%–19%),
    against a no-synchronization baseline of roughly 20–30% ⇒ this synchronization is a **real but incomplete fix**.
    **Statistical bound (honest)**: even a future 0/50 would only bound the per-run flip rate at ≲6% with 95% confidence, not "proven deterministic"; this round's measured ~6% already directly refutes stability.
    (3) **Not released**: the construction-time rejection stays, sessions/serving docs/refusal wording/solo-to-acceptance were all left unchanged; **K=7 decode tok/s not measured**.
    **Next step**: do a per-kernel read/write ordering inspection of the `run_layers_tp2`/`device_pair` in-kernel handshake (who read data the peer had not finished writing), or move to a supported device/administrator environment to run initcheck;
    alternatively, evaluate option B from (2), "near-tie protection" (a semantic addition that must separately declare trigger rate/extra cost/acceptance).
    (1) **Landed**: write S1's thorough synchronization (both cards `cudaStreamSynchronize` + `cudaDeviceSynchronize`, only in the DFlash2 branch, before `run_verify_window`,
    `src/runtime/engine/tp2_generation_core.cpp`) into the working tree as **bounded per-round synchronization** (no env hook; MTP/plain unaffected).
    (2) **Final-binary 12-run protocol** (`build-win/b7b-solo-1..12.log`): **11/12 byte-identical** — `shared_b` is
    `[1703 220 248046 198 248045 198 248045 198]` throughout, loghash `175A56CC3A333FA4` (11 runs), **z-10 still flips to** `[… 248046]`
    (loghash `6F0194D3FCF00981`). Against the S1 batch (`b6y-sync-1..12`), which was 12/12 at the time ⇒ this synchronization significantly reduced the flip rate below the baseline (2–3 variants in 10 runs)
    (merged across both batches, only 1 in 24), but **did not eradicate it**: the cross-stream / cross-card write-read ordering gap is **one of the main causes**, and residual sources remain.
    (3) **Other acceptance items in the same batch (all passed)**: sessions plain `recall reused 71 prompt tokens bit-identically … passed`, mtp `passed`,
    append K=7 `0xbad27a494a9bc853`, K=5 `0xbee487264ca8ffb8`, load `TP-2 dual-shard load passed devices=0,1 layers=64
    head=[124160,5120] vocabulary-parallel`, `ninfer_tp_device_pair_test` **PASS**.
    (4) **Not released**: the construction-time rejection stays (`model_instance.cpp` back at HEAD), the sessions dflash2 expectation, `docs/serving.md`, the refusal wording,
    and solo-probe-to-acceptance were all left unchanged; **K=7 decode tok/s not measured** (the fix did not pass acceptance, so it does not proceed to performance acceptance).
    **Next step**: inspect `forward_tp2_window` per kernel for read/write ordering (which kernel read data the peer had not finished writing), or run
    `compute-sanitizer --tool initcheck` in an environment with administrator privileges / a supported device; only after 12/12 stability, do the release checklist in (4).
    ——`shard_a_.device.bind_to_current_thread(); cudaStreamSynchronize(a); cudaDeviceSynchronize();`, the same once for `shard_b`, then bind back to a — temporary hook
    `NINFER_TP2_DIAG_SYNC` (**reverted**). The solo probe **run 12 times in a row** (`build-win/b6y-sync-1..12.log`): **12/12 byte-identical** —
    `shared_b` is `[1703 220 248046 198 248045 198 248045 198]` throughout, digest is `0x4bcc3994a5efba7d` throughout, and after removing the `[mem]` lines, the entire log's
    first 16 SHA256 hex digits are `175A56CC3A333FA4` throughout. Controls (same binary/probe/artifact, **without** synchronization): `b6f` (10 runs, 3 distinct last tokens), `b6n-a` (10 runs, 2 distinct).
    ⇒ **Confirmed (at least a main cause) is the cross-stream / cross-card write-read ordering gap before the DFlash2 verify window**, in the same category as the unsynchronized D2H that B5 fixed
    (the MTP window is written directly by the host, so it never exposes this).
    **Next step (item 2, not done, requires a new round)**: replace this "thorough synchronization" with **precise dependencies** (events, or the existing synchronization primitives in `pair_`/`adopt_peer`, effective only for DFlash2),
    then verify against the release checklist: solo 10 consecutive runs byte-identical, sessions dflash2 5/5 pass, append K=7 `0xbad27a494a9bc853` / K=5 `0xbee487264ca8ffb8` unchanged,
    load passes, and a K=7 decode tok/s figure is provided (expected microsecond-level overhead, no measurable loss); only when all green, remove the construction-time rejection, update serving.md/PLAN/refusal wording, and convert the solo probe into an acceptance case.
    This round's temporary synchronization and gate-opening changes have all been reverted; the gate stays closed.
    back to HEAD); after rebuild, `NINFER_TEST_ROUTE=dflash2` refusal exits 0, the solo probe exits 77.
    A plain engine in the same process serves as the control group, to determine whether it is "two DFlash2 instances interfering with each other" or "any second instance is affected".
    The temporary route-opening changes have been restored: `model_instance.cpp` restores the construction-time rejection and is rebuilt and verified (`dflash2 exit 0`, solo probe `exit 77`).
- **B6 state and retention (see the B6 results above)**: the draft ring/pending features are now saved and restored with session recall (reusing MTP's host slab
  precedent), `dflash_graph_profiles` is wired in; "recall bit-identical after session switch" does not hold for this route, so retention stays disabled;
- **B7 performance acceptance**: dual-card throughput against the 90–180 tok/s target + comparison with MTP + no regression for plain/mtp.
- **B7 pre-scouting (completed, not implemented; deliverable only after the route is released)**: the reason DFlash2 verify is currently eager is **not** capture-illegal (the single-card route has already captured the same
  `DFlashFeatureSink` into a CUDA Graph, `draft.cpp:712-718` + `program/graphs.cpp:373-421`), but rather 4 places that are not wired up:
  ① the master switch only takes effect for MTP (`tp2_generation_core.cpp:374`); ② the verify bucket is only built in the MTP branch (`:387-400`);
  ③ `capture_verify_graph` does not take a sink (`:910-912`, **must be added**, otherwise there is no scatter node in the graph, `pending_features` silently loses columns, and only the acceptance rate drops);
  ④ `:979-984` has a `throw` justified by "an un-solidified host capture call", whose premise has been falsified by the single-card route.
  Biggest risk: building the bucket makes the **eager branch also switch to the bucket envelope** (`:960-971`), and DFlash2's output is sensitive to the verify window layout ⇒ one must first do a greedy byte-identical A/B of "only switching the envelope", then graph on/off, otherwise two variables change at once.
  Criterion: the graph arm's verify ≈ 30.6 ms (measured for MTP at the same width), 4~4.5 ms/round lower than eager, roughly +15% throughput at equal output; hard precondition `pair_.in_kernel_allreduce()`; if the envelope change is rejected, capture is unworkable and has no equivalent substitute (the only movable part is the <0.1 ms D2H+sync).

**Root-cause conclusion and fix (round A, 2026-09-22; supersedes the "remaining possibilities" of the diagnostic rounds above)**: the root cause of the non-reproducible clamped round has been located and fixed —
**TP-2's `forward_tp2_window` never bound `active_valid_columns_`/`active_sequence_batch_`** ⇒ `attn_mix` takes the non-batch branch and KV append
is completely unmasked (`small_t_fp8` is instantiated with `Masked=false` ⇒ `valid_tokens=TokenTile`, and every column of the window writes KV). Whereas DFlash2's budget-clamped window pins the tail columns at
the **same absolute position** of the last valid column (`speculative_round.cuh:36` `positions[off]=base+(j<=extent?j:extent)`; the TP-2 side `tp2_generation_core.cpp:2821-2826`
is synonymous) ⇒ within the same window forward multiple warps concurrently write **the same paged-KV slot** (FP8 code + scale torn byte by byte, last-writer-wins) ⇒ that slot's contents are non-deterministic between runs ⇒
this round and all subsequent logits that read that slot jitter ⇒ `target_argmax[0]` flips at near-ties. This mechanism fully explains every known fact: flips appear only in clamped rounds
(`extent<k`, i.e. the budget tail rounds); failures are biased (warp interleaving is biased); the wrong value is variable (any of the 8 columns' K/V can win); construction order affects the flip rate (timing/L2 state);
the official artifact flips as well; S1's synchronization only changes the interleaving statistics without eradicating it; single-card is qualified (`target_verify_batch_impl` binds the mask ⇒ the tail columns do not write KV); MTP/plain are stable
(MTP window positions are contiguous `:3002-3008`, plain T=1, neither has a shared slot); two Engine instances have different flip rates. **Excluded this round**: the `ar_inplace_bf16` handshake
(double-buffered parity + spin-publish gating, a closed timing loop), `argmax` (fixed total-order CAS, bit-deterministic), `speculative_accept_sparse_drafts`
(`raw_greedy`/greedy-without-penalty reads `target_tokens` directly; the pipeline path's `col>extent` early exit makes the finalize count exactly extent+1, and at extent=0 it reads the slot it wrote itself),
the split reducer (fixed order, no atomic). **Fix (landed, under acceptance)**: ① the `attn_mix` else branch consumes `active_valid_columns_` in the same form as the batch branch;
② `forward_tp2_window`/`run_verify_window` gain a new `valid_columns` (= `target_valid_columns[0]` = extent+1; clamped columns are forbidden from writing KV and their logits are zeroed),
passed only by DFlash2 calls; MTP passes 0 (no mask bound, the original route unchanged bit-for-bit).
**Round A acceptance (2026-09-22, all green)**: ① load ✓; ② append hashes **unchanged** (K=7 `0xbad27a494a9bc853`; the prefill/proposal side was unaffected by the fix,
so no re-baselining is needed); ③ sessions plain/mtp ✓ (recall 71 bit-for-bit, no regression); ④ the solo probe in a separate process run **10/10 consecutive, bit-identical** (digest is
`0x4bcc3994a5efba7d` throughout, all 9 walk tokens identical; before the fix solo flipped ~25%/run, and the probability of 10 identical runs is only ~5%, so the evidence is sufficient) — among which
the direct evidence for **recall reproducing token by token** is the in-probe pairing: `a_continued` (reuse 71, draft declined) and `a_continued_evicted` (prefill from scratch)
are identical across all 8 tokens; ⑤ `NINFER_TEST_ROUTE=dflash2` real scenario run **5/5 all passed** (note: at the time all three recall cases landed on the declined boundary, and the assertion, per the B6 interim
contract, only pinned the first sample; shared_c@512 — i.e. the B6 flip position — was also only pinned on the first sample). Round B has **tightened** the declined boundary and shared_c@512 **to full-token comparison**
as the release regression; if all pass, then "DFlash2 recall reproducing token by token from-scratch" is concluded, and the root cause of the B6 "failed items" is this defect. The temporary changes used for verification (gate release, crossing the sessions tripwire, switching route_keeps_retention to true)
have all been reverted, and the official tree keeps the gate closed. **Upgrade points left for B**: remove the gate and tripwire; `route_keeps_retention(DFlash2)` can be switched to true (the retention property has been empirically demonstrated);
the declined boundary of `draft_declined` is still only pinned on the first sample, and an attempt can be made to tighten it to full-token comparison.

**Round B (release, 2026-09-22)**: the gate and the sessions tripwire have been removed (`model_instance.cpp` keeps only the DFlash v1 rejection, wording changed to "--spec mtp and
--spec dflash2"); sessions defaults to three routes, `route_keeps_retention` is consolidated to all-on (its ternary expectation branch is deleted along with it); solo is converted to an acceptance case (removing the rejection
skip, adding an in-run recalled-vs-evicted first-sample assertion). **Tightening experiment (key new evidence)**: after tightening shared_c@512 and the declined boundary from a first-sample pin to
full-token comparison, shared_c@512 (the B6 flip position) **passes on all tokens** — the round A fix empirically holds at the original fork position; but the declined boundary's rerender scenario
**stably forks** at the 3rd
token (5/5 identical token by token: got `[2752 11 220 16 24 24 15 82]` vs oracle `[2752 11 198 220 220 16 15 15]`), not residual
jitter. Mechanism: the declined request runs a target-only round (1-valid clamped window) vs the oracle's all-draft round (8-valid window) = different execution shapes; the attention KV reduction split
varies with the window tail position, the written KV rows permanently carry a ulp difference, and near-tie trajectories each stay stable per the "a change in draft pattern causes drift" class (the same class as MTP≠plain and the tp2 doc's "a
changed draft pattern shifts the trajectory on which near-ties are resolved"). **Conclusion**: the declined boundary reverts to a first-sample pin (the boundary-crossing
logits = the property that retention genuinely carries), and its tail reproducibility is guaranteed across processes by the solo digest; the aligned boundary keeps full-token comparison. append K=7/K=5 and ring
hashes are at baseline (`0xbad27a494a9bc853`/`0xbee487264ca8ffb8`/`0x1a53fd824cd4e360`). Docs have been synchronized (tp2-dual-5060ti.md gains a DFlash2
subsection, three rows in the Verification table; serving.md's TP-2 limitations section rewritten).
**Round B acceptance (all green)**: sessions default three routes exit 0; `NINFER_TEST_ROUTE=dflash2` 5/5 all passed; solo 10/10 all with the same digest
`0x4bcc3994a5efba7d`; append K=7/K=5 hashes at baseline; load ✓. **K=7 decode tok/s (bench_serve synthetic load) = 20.2–20.7 tok/s**,
against MTP K=2 66–72 tok/s under the same load; the breakdown shows the **per-round cost ~48.8 ms/round is consistent with the B5 basis (no regression)**, and the difference is entirely in acceptance rate (synthetic fill text
~1.0 tok/round vs the B5 basis's 4096-context greedy 3.10 tok/round = 63.6 tok/s). Another finding: DFlash2's verify step is still eager (MTP is already graph,
log `[tp2-graph] verify step: eager`) — i.e. C (B7 graph wiring) is the first lever. `serve.ps1` gains `-Spec mtp|dflash2`, making the released route usable end to end.

**Theoretical performance gap (single-card 5090 vs dual-card TP-2, DFlash2 K=7; roofline synthesis, not measured)**:

- **Hard ceiling = bandwidth ratio 2×**: the two cards total 896 GB/s = half of one 5090's 1,792 GB/s (`docs/tp2-dual-5060ti.md:106-108`).
  Measured 10.15 GB per shard per forward stream ⇒ floor 10.15/448 ≈ **22.7 ms/round**; the 5090 floor ≈ 20.3/1792 ≈ 11.3 ms.
- **The two routes' measured efficiency is comparable**: TP-2 MTP K=2 is 38.2 ms/round ≈ 59% of floor; 5090 DFlash2 192.5 tok/s,
  acc 37.0% ⇒ 3.59 token/round ⇒ 18.6 ms ≈ 61% of floor ⇒ the gap is mainly just the bandwidth ratio.
- **DFlash2-specific disadvantage**: draft (2.07 GiB) is only on shard 0 ⇒ 2.22/448 ≈ **5.0 ms/round** (1.24 ms on the 5090, four times).
- **DFlash2-specific advantage**: MTP's single largest cost item on TP-2 is the **7–10 ms host-serial draft chain**
  (same document `:127-128`); DFlash2 produces all proposals in parallel in one non-causal masked block forward ⇒ this cost disappears.
- **Estimate**: (22.7 verify + 5.0 draft + 2.9 launch gap + ≈2.5 allreduce) ≈ **33 ms/round ÷ 3.59 ≈ 108 tok/s**
  ⇒ about **0.56×** of the 5090, but **≈1.9×** this machine's MTP K=2 at 57 tok/s —— clearly larger than the +19.5% on a single card,
  because the chain that gets eliminated accounts for the largest share of a TP-2 round.
- Correction to the archived **90–180 tok/s**: the 90 end agrees with the roofline; **180 is unreachable** —— even with the launch gap, allreduce and
  draft all driven to zero, the theoretical ceiling is only 3.59/0.0227 ≈ **158 tok/s**.
- **D1 corollary**: splitting the draft across two cards saves only about 1.5 ms/round (≈5%), a gain far smaller than the complexity ⇒ **D1 is done for memory only, not for speed**.
  The speed levers are restoring CUDA graph (−2.9 ms), reducing/fusing allreduce, and lowering draft weight precision.

**Round C (B7 verify CUDA Graph wiring, 2026-09-22)**: all 4 unwired spots from the B7 pre-reconnaissance were changed: the master switch `:374` now takes effect for `mtp||dflash2`;
the verify bucket is built in both branches (DFlash2 uses `dflash_graph_profiles`, target envelope `{1, min(max_context, planned.max + draft_window + 1)}`);
`capture_verify_graph` takes `sink`/`valid_columns` and forwards them into the capture body; the `throw` was removed. The clamped extent goes through an int at the tail of the pinned buffer
(`[ids(W), positions(W), valid(1)]`), re-read on every replay by the captured graph via a memcpy node —— **`set_i32_scalar` cannot be used**: it bakes the value into
the launch arguments of `set_i32_scalar_kernel`, so replays stay clamped to the extent at capture time forever. Two-stage A/B (moving one variable at a time): ① "switch only the envelope" (eager +
bucket envelope) greedy byte-identical A/B all green (solo digest `0x4bcc3994a5efba7d` ×2 + sessions dflash2 passes) ⇒ the bucket variable is harmless; ② graph on/off revealed
**the graph path flips 2/6** (the shared_b tail token taking the triple `248045 198/220/13962` is exactly the fingerprint of the unmasked race from round A) while eager is 0/13+.
**C2 root cause**: the `capture_verify_graph` signature had the `sink`/`valid_columns` parameters added first, but **the `forward_tp2_window` call inside the capture body did not forward them** ——
the default arguments `nullptr/nullptr` let it compile silently ⇒ the captured sequence = the old window with "no mask + no sink" ⇒ the clamped KV append race fixed in round A came back to life on the graph route
(KV tearing + `pending_features` silently missing columns). Fix = the capture call forwards `sink, valid_columns` (tp2_generation_core.cpp `capture_verify_graph`).
**C4 acceptance (after the fix)**: solo graph path in a separate process **10/10 bit-identical** (all `0x4bcc3994a5efba7d`, 2/6 flips before the fix), `NINFER_TP2_VERIFY_GRAPH=0`
eager hits the same golden values (graph≡eager bitwise), sessions dflash2 4/4 passes (recall 71 bitwise), sessions default three routes pass (no plain/mtp regression),
append K=7/K=5/ring three golden values unchanged. **Lesson**: staged edits of the "signature first, body forwarding later" kind must
land as a pair; default arguments on the capture path are a silent-degradation trap (the wrong shape is legal, and the result only drifts at near-ties).
**C5 performance (bench_serve synthetic load, graph/eager A/B on the same build)**: equal-output 128-token decode@8192 **6267→5627 ms (-10.2%, i.e. -4.9 ms/round)**,
20.3→22.6 tok/s; decode_short 20.6→23.0 tok/s (+11.7%); prefill flat across all three segments; MTP K=2 control 67.9/64.8 tok/s (within the B baseline band of 66–72), no regression.
The per-round time difference is the same order as the MTP window graph's -4.3 ms (the B7 criterion of −4~4.5 ms/round is met); synthetic-text acceptance ~1 tok/round ⇒ throughput gain +11% (the +15% criterion was extrapolated from the MTP basis,
so it is reasonable under the measured basis).

**Round E (graph-ifying the MTP draft chain, in progress)**: goal = capture the MTP draft chain (measured 7–10 ms/round at K=2, the largest single cost item in docs/tp2-dual-5060ti.md:127-128)
in its entirety into a CUDA Graph (same as llama.cpp: the whole round including the chain in one graph). Reasons for choosing it: purely a launch-mechanism change, **does not touch the numerical contract** (round C already proved graph≡eager bitwise),
and it reuses all of round C's machinery (WindowGraph/buckets/pinned inputs/memcpy nodes). Steps: ① reconnoiter the capture surface of `mtp_forward_batch`/`mtp_forward_ar_step` (text.cpp) ——
per-round values (the two `set_i32_scalar` at :1119/:1121 must be changed to pinned passing, the lesson from round C), host synchronization points, stability of sampling/state addresses; ② the whole chain in one graph/bucket
(K steps + D2D hidden relay + `increment_i32_scalar` + sampling), per-round inputs go through pinned (anchor, position) + stable device tensors
(`mtp_anchor_hidden`/sampling state/MTP KV slab), `drafts` output as a device tensor + one D2H per round; the bucket can reuse ordinary_graph_profiles (the envelope is a routing hint,
width does not change the numerics —— proved empirically by C1); ③ graph on/off byte-identical A/B (`NINFER_TP2_MTP_CHAIN_GRAPH=0` control) + sessions three routes + append golden values unchanged;
④ bench MTP K=2 control (baseline 38.2 ms/round, 67.9/64.8 tok/s); ⑤ documentation + commit. Decision criterion: the drafts coming out of the chain stay bit-identical ⇒ window/acceptance/trajectory all untouched;
expected 38.2→~30 ms/round, MTP decode +15–20%.
E2 progress: `tools/win_port/r52_ab.ps1` (same five r52 prompts, greedy top_k=1, hash=sha256(content||reasoning)[:16]) has been created, **pre-E golden values** (current build,
--spec mtp 131072 fp8 KV): req0 `9f4abde908470928`(stop/130), req1 `a956f217d6a6250c`(length/436), req2 `847d706198441ac9`(length/608),
req3 `49800780a7666b44`(stop/69), req4 `46698bd3a4f115ee`(stop/634). Acceptance = all three arms agree (pre == eager(NINFER_TP2_MTP_CHAIN_GRAPH=0) == graph).
Capture-surface conclusion: `mtp_forward_batch/ar_step`→`mtp_forward_core`→`stem/tail` are all device ops with no synchronization; the only per-round values are the three set_i32_scalar at :1119-:1121
(token/position/position+1) → change to pinned+copy_i32; the envelope is a host structure entering the launch plan → bucketize (the kernel's active-splits are derived from device positions and
the reducer comes from the same source ⇒ widening is harmless by construction, the C1 precedent); stem's embedding_tp2 (column split) + split-head proposal_argmax run on the pair ⇒ capture_group two streams.

**Round E results (acceptance complete, pending bench/docs/commit)**: implementation = pinned `[anchor, position, position+1, drafts(K)]` buffer + memcpy node, `mtp_chain_body`
a single implementation shared by all three paths (eager×2 + capture), `capture_mtp_chain_graph` capture_group two streams, watermark reusing the verify window's select/reusable/launch;
three-state switch `NINFER_TP2_MTP_CHAIN_GRAPH` (unset=graph / `0`=eager(bucket) / `exact`=eager(exact), same convention as `NINFER_TP2_DECODE_GRAPH`;
`DecodeStepMode` renamed to `StepLaunchMode`, shared by both). **Acceptance evidence: the five hashes of the three arms graph / eager(bucket) / eager(exact) are bit-identical; and a git-stash
bisection (old HEAD build vs new build) is bit-identical ⇒ the change is golden-value neutral and graph capture is bitwise lossless; exact per-step envelope == bucket full-width envelope ⇒ the chain-step numerics are
neutral with respect to the envelope (re-confirmed as in C1)**. Clean golden values (reproduced 5 times, across two builds): req0 `5b4a978335a95cbb`(stop/186), req1 `2072b4db667b861d`(length/449),
req2 `3ce32871d0a13d34`(length/571), req3 `56fd6e3da523651d`(stop/200), req4 `cc73c73125c6f621`(stop/629).
**Lesson (root cause of the initial pre-baseline outlier)**: the first probe of the comma-precedence bug had earlier sent one long mangled concatenated request to the same server, and the
37-token template prefix KV that its prefill computed under a different chunking shape was reused by the golden sample (the "KV rows carry the reduction shape's ulp" contract) ⇒ near-tie drift throughout; located here after the stash bisection cleared the change.
**Greedy golden values must be collected on a clean server (zero prior requests)** —— written into the header comment of `r52_ab.ps1`.
**Round E regression addendum (the zero-byte advance trap)**: sessions exposed it —— on one side of the capture body the scratch is all scope-local allocation (net increment 0), and on the replay side `alloc_bytes(0)` was
rejected by `DeviceArena::alloc_bytes` ("arena allocation must be nonzero"); the serving flow's net increment is nonzero so the golden values were all green, while the session flow (recall/chunked re-prefill)
is net zero and blew up on the first replay. The three window/chain replay advance points now treat 0 as a no-op following the `position_arena` `target > used` precedent; sessions passes all three routes
(plain/mtp/dflash2, with mtp's `2 layouts` property retained).
**Round E performance verdict** (`NINFER_TP2_TIMING=1` A/B on a fixed greedy trajectory, four runs on the same trajectory of 94 rounds/180 commits): chain step eager 3.73/3.69 → graphed 3.50/3.50 ms;
whole round 34.30/34.09 → 33.99/33.95 ms ⇒ capture removes ~0.2–0.3 ms/round of host launch gap (decode ≈ +0.7%), **not** the +15–20% estimated in E5: the old "7–10 ms chain"
was a subtractive decomposition of slower rounds; the measured chain is 3.5–3.7 ms of real MTP layer GPU work, and the launch gap was already largely hidden. bench_serve's cross-run decode rate cannot resolve the difference (the nonce
changes the content ⇒ acceptance rate/round count drift), so the component A/B is authoritative.
- Uncertainties: acceptance and token/round are determined by the mathematics and transfer across cards (residuals bit-identical), but the local draft is w4a4 rather than
  nvfp4 ⇒ acceptance will differ slightly; the efficiency anchor is taken from MTP rounds; prefill and first-round jitter are not counted.

**DFlash2 22tok/s root cause (proved empirically, 2026-09 diagnostic round)**: not a draft-quality problem but the **draft-decline guard being falsely triggered by the template prefix cache**.
The reuse scan first takes a grid-aligned boundary (`reuse_grid = clamp(prefill_chunk, 64, ·)`=256); when no alignment is found, the **unaligned fallback**
(tp2_generation_core.cpp:2159-2184) takes the deepest unaligned boundary (the chat template prefix of ~37-57 token always hits) ⇒
`dflash_draft_declined_ = reuse % reuse_grid != 0` (:2193) ⇒ `extent=0` every round (:2923) ⇒ not a single draft enters acceptance,
and only 1 target token is committed per round = 22-25 tok/s. Empirical proof (diagnostic dump, same K=7 serving): first request on a cold cache `extent=7`, 0-3 accepted per round,
2.29 tok/round (proposal quality normal, on par with the document's story tier); the 2nd request (template prefix already cached) **`extent=0 count=1` every round** = 1.00 tok/round.
The guard's design intent (:2186-2192 comment): when the round follows a heterogeneous chunking walk, its proposals cannot be licensed against a from-scratch walk's window. The crux: **the unaligned fallback only covers
the "shared prefix shorter than one chunk" case (clip<256 token), sacrificing the entire draft (decode ~2×) to save that little prefill**; moreover :2100-2102
design comment itself states "rounding down recomputes at most one extra chunk". Fix directions (1+3): when clip≤1chunk, round down to keep the round canonical (draft online),
for deep clip keep the decline but make it **visible** (TP-2 wires up speculative stats + draft_context_declined display).
"Only repetition loops hit it" is a misreading: on a cold cache normal text hits it too (proved by dump), and the other measurements were in the declined state (extent=0, nothing acceptable).
Swapping the artifact is ineffective ✓ consistent with the conclusion (the decline is a runtime policy).

**Fixes 1+3 landed (this round)**:
- **Visibility (1)**: TP-2 now fills `GenerationResult::speculative` (the round-loop preamble presets per route
  `accepted_per_position`/draft_window/backend, after which the dflash and MTP branches each accumulate rounds/drafted/
  accepted/per-position, on the same basis as single-card decode.cpp:139/732) ⇒ the response's `timings.draft_n`/
  `draft_n_accepted` are directly usable; on decline a single line `[tp2-draft] masked draft declined: …` is printed to stderr.
  Pitfall: TP-2's `Request` is not a `RequestRecord` (no `speculative_stats`), and that vector was never allocated ⇒ a direct
  `vec[i]+=1` wrote out of bounds and hard-crashed sessions (all output lost, exit 1); changed to accumulate into `result.speculative` with the length preset.
- **Deep fix (3)**: the non-grid fallback (the reuse scan fallback, the only entry point that produces unaligned reuse) is gated for DFlash2:
  when `position <= max(reuse_grid, 16×budget.remaining())`, that boundary is not taken; instead the aligned scan result (usually 0) is used to replay
  the clip —— the round keeps a canonical chunking walk and the draft stays online throughout. Cost model: clip recompute ~0.6 ms/token (prefill ~1.6K tok/s)
  vs the draft saving ~19 ms/generated token ⇒ with the 16× margin it is always a net win; only beyond that is mid-chunk reuse + decline (visible) kept.
- **Evidence**: `ninfer_qwen3_5_tp2_sessions_test` passes all three routes (EXIT=0; the test mirrors the new policy:
  `draft_rounds_down()` + a shallow-clip assertion that reuse==0 + `compare_recall`'s third category "replay from scratch" pins only the boundary's first sample);
  dflash2 serving + `greedy_probe` (7 requests sharing a template prefix, K=7) measured 3.17 committed per round
  (rounds=30/committed=95), `draft_n` 187–276, accept 53–67, **no decline line**, prefill 188–388 ms
  ⇒ up from ~22 tok/s to ~71 tok/s (1.00 tok/round before the fix).
- **Contract note**: a from-scratch replay's recall in a hot engine is not guaranteed to match the cold oracle token for token (`tools/win_port/r52_ab.ps1` already
  records this: cached KV rows carry the reduction shape of the prefill that produced them, at ulp level) —— the verifiable new contract is "the draft stays online + the boundary's first sample agrees".

**Artifact status (checked)**: this machine's `<models-dir>/qwen3_8_27b_w4a4_w8a8.ninfer` does **not** contain DFlash2 (components are only text/vision/mtp,
parameters=1422), but the same directory already has `qwen3_8_27b_w4a4_w8a8_dflash2.ninfer` (parameters=1513, objects=1218) ⇒
the evaluation **does not require re-conversion**; just swap the artifact.

**Unknowns still to be measured**: the true free memory of a native Windows build (the ledger's `free` comes from `cudaMemGetInfo`, which under-reports by about 1 GiB on WSL2);
whether splitting the draft by head/row is numerically equivalent (no corresponding split test currently); the workspace peak at K=15; the target's 5-layer residual cross-shard gathering
per-step cost; the latency of the draft's per-layer allreduce; the equivalence of sparse rejection sampling under TP-2 sharded logits.

## 4. Upstream cherry-pick plan (first tier + second tier)

**Background** (2026-09-21 analysis): origin/master is ahead of this branch by 20 commits (merge base `e360c4c0`,
upstream HEAD `9e163eee`). The local main workload is NVFP4 TP-2; upstream's Q4/Q5 tuning, sparse_moe, and Q5 bench/test
commits are irrelevant to this machine and are not scheduled (see the session record for the full per-commit assessment). **The new official artifact (`6cc95cc5`) is incompatible with the local frontend**:
the new artifact embeds a maintained-edition template (sha256 `a497db9e…`) that is not in the local `CompiledChatTemplate::resolve()`
whitelist (`e84f32a2…`/`c3cf9e34…`), so loading throws `unsupported frontend/chat_template.jinja`.
All local changes to the shared nvfp4 code are Windows-specific TMA descriptor staging (`#if defined(_WIN32)`), orthogonal to upstream's
performance changes but in the same files.

**Prerequisites**

- [x] Handle uncommitted changes in the worktree (Round 18 session-resume hardening: `tp2_generation_core.{h,cpp}`,
      `test_tp2_sessions.cpp`, measured but uncommitted) —— committed by `4890f554` (code) and `ad1336a1` (docs)
- [ ] Record the baseline: `tools/win_port/bench_serve.ps1` decode/prefill + op-level NVFP4 bench
      (WSL, a4 graph mode, T=1024/1536/2048)

### First tier: product compatibility (frontend jinja chain, picked in order)

| # | Commit | Content | Expected local conflict |
|---|---|---|---|
| 1 | `a2b7ed11` | Log simplification (3 files) | None |
| 2 | `b9219f3f` | Introduce the llama.cpp Jinja engine into `third_party/llama-jinja/` (~6,000 lines) + the `ninfer_jinja` target | None (new directory) |
| 3 | `98dada0e` | Execute custom Jinja chat templates (76 files): `chat_template.cpp` summary whitelist → Jinja execution, new `prompt_layout`, CLI `--chat-template`, serve/translate kept in sync | `chat_template/frontend/processor/tokenizer/prompt_layout` locally unmodified, should apply cleanly; **`src/serve/*` has local Windows port changes, conflict expected** (keep the local platform branches + adopt the upstream semantic changes); `include/ninfer/types.h`, `apps/cli/*` need review |
| 4 | `8eaed538` | Preserve template literal content (Refs #258) | Same files as 3, small |
| 5 | `6cc95cc5` | Model card SHA/manifest updated to the maintainer template | None (docs only) |

**Acceptance (per step)**

- Dual-platform (Windows + WSL) build `BUILD_EXIT=0`
- `ninfer_qwen3_5_frontend_test` (currently failing on both platforms — the fixture template is rejected by
  the whitelist — should turn green after the cherry-pick), `ninfer_jinja_test`,
  `tests/text/test_chat_templates.py` (needs Python ≥3.11: Windows miniconda 3.13 or WSL python3)
- End to end: the new artifact (new SHA, template `a497db9e…`) loads on 8088/8099; `reasoning_effort`,
  tools, and multimodal rendering work; greedy A/B against the old artifact with the same prompt (plain-text
  chat should be equivalent; tool-call formatting may differ, record the differences)

### Second tier: NVFP4 performance (in upstream order; all must be re-measured on the dual 5060 Ti)

| # | Commit | Content | Local impact |
|---|---|---|---|
| 1 | `1d8587bc` | W4A4 activation scale planes written per tile, one tile read per TMA request; `select_a4` returns `Nvfp4A4Route` (GEMM + layout as a pair) | Affects the local TMA routes (linear_add T≥1024, swiglu TMA, attn/gdn input proj); MMA routes unaffected. The upstream per-shape floor (34816x5120 from 256, the rest from 1024) targets full-geometry shapes; local half-geometry shapes take MMA, so no floor change is needed |
| 2 | `05507ab0` | Fused SwiGLU TMA epilogue uses `silu_approx` (2 files: `math.cuh` + epilogue) | Directly acts on the MTP draft path (full-geometry swiglu TMA). **Not bit-exact** (the model's usual in-band fuzziness is ~1.3×, but it removes spurious tail zeros) ⇒ greedy output may flip at near-ties; the acceptance criterion is "no degradation + same quality tier" rather than bit-exactness |
| 3 | `5f5fccab` | The W4A4 TMA route accepts token counts not divisible by 256 (the last M tile may be partially filled; scale planes zero-padded; the epilogue clamps the padding lanes) | Local linear_add/swiglu TMA currently requires `tokens % 256 == 0`, so ragged prefill tail blocks all fall back to MMA; after the cherry-pick ragged widths take the TMA route. Upstream deliberately leaves the fused SwiGLU route untouched (it depends on the landing order of silu_approx) |

**Conflict expectation**: the three touch the same files as the local Windows TMA descriptor staging
(`Nvfp4TmaDescriptorStaging`, tensormap fence, `#if defined(_WIN32)` branches) — `nvfp4_w4a4_tma.cuh`,
`nvfp4_w4a4.cu`, `nvfp4_linear_swiglu_w4a4_tma.cuh` — but they are orthogonal: upstream changes the
layout/routing logic, local changes how descriptors are passed, so both are kept when merging.

**Acceptance (per step)**

- Dual-platform build; all op tests pass (including the upstream-added routing boundary cases: 1023/1024/1025, 255/256/257/512)
- Numerics: `1d8587bc`/`5f5fccab` are exact changes ⇒ with `NINFER_OP_REPORT_STATS=1` the %.17g bytes match on
  both sides (upstream baseline 3285/3290 records); `05507ab0` is not bit-exact ⇒ the upstream measured
  criterion + local greedy A/B (no degradation)
- Re-measure on the dual 5060 Ti: op-level NVFP4 bench (a4 graph, T=1024/1025/1536/2048) + `bench_serve.ps1`
  end to end, against the baseline; the upstream 5090 numbers (−8.7%/−10.8%/−17.5%) serve only as direction
- After all three are done: full dual-platform test run (the pre-existing `ninfer_qwen3_5_frontend_test` problem
  is fixed by the first tier)

**Not scheduled (kept on record)**: the Q4/Q5 A16 tuning series (`bb844c43`/`beedffa0`/`d3c125ed`/`b39de4d5`/`a9a0d10a`/
`5b4303c0`/`9e163eee`), Q5 bench/test (`cb30e070`/`f76e19c0`), `dc58675f` (sparse_moe, MoE-specific),
`028eb61e` (Q8 cache policy, optional), `f9c4a04b` (state cache bench scenario, optional).

---

## 5. Progress

- [x] Prerequisites (worktree handling + baseline record) — committed in Round 18 as `9123217c`; PLAN archived as `9a1e5ae3`
- [x] First tier-1 `a2b7ed11` → `322217ae` (clean)
- [x] First tier-2 `b9219f3f` → `c6c49b2d` (clean, 17 files +5966)
- [x] First tier-3 `98dada0e` → `119e7f12` (7 conflicts resolved, 75 files)
- [x] First tier-4 `8eaed538` → `dd985028` (clean, 19 files)
- [x] First tier-5 `6cc95cc5` → `8a086f40` (clean, 15 files)
- [x] Second tier-1 `1d8587bc` → `1ed82861` (clean, 22 files; Windows staging retained)
- [x] Second tier-2 `05507ab0` → `2a7ec6c5` (clean, 2 files)
- [x] Second tier-3 `5f5fccab` → `4b1e8507` (2 conflicts resolved, 19 files)
- [x] WSL build (BUILD_EXIT=0, all 212 targets pass; merge artifact fix committed as `75831a03`)
- [ ] WSL op tests + byte equality + frontend fixture test (**WSL2 CUDA driver crash**: PTX JIT segfault inside
  `cudaGetDeviceCount()`, the GPU is held by a Windows service; the jinja test passing proves the binary is
  sound, op tests moved to the Windows side)
- [x] Windows build (full build-win2 + BUILD_TESTING=ON, BUILD_EXIT=0, 697 targets; `localtime_r`→`localtime_s`
  fix `474daf92`)
- [x] Windows op tests (NVFP4 A4/A16, frontend, jinja all pass; with `NINFER_OP_REPORT_STATS=1` the error
  metrics are within tolerance, 268 records)
- [x] End-to-end A/B verification (`tools/win_port/verify_cherry_pick.ps1`; old vs new binary, same config,
  same script)

### 5.4 e2e A/B verification results (2026-09-21, dual 5060 Ti, 3456 service)

Same configuration (launch script MaxContext 204800, draft-tokens 3, KV fp8), same method (bench_serve.ps1):

| Item | Old binary | New binary | Δ |
|---|---|---|---|
| prefill_2048 | 574.3 tok/s | 587.9 tok/s | +2.4% |
| prefill_8192 | 1,568.9 tok/s | 1,555.4 tok/s | −0.9% |
| prefill_32768 | 1,498.3 tok/s | 1,488.2 tok/s | −0.7% |
| decode_short | 71.6 tok/s | 72.5 tok/s | +1.3% |
| decode_at_8192 | 70.9 tok/s | 75.0 tok/s | +5.8% |

**Conclusion**: the cherry-pick causes no degradation. prefill is flat (within noise), decode rises slightly
(+1.3%~+5.8%). The decode token counts differ (old 118/128 vs new 84/83) because of jinja template
rendering/sampling randomness, so throughput is not strictly comparable, but the direction is flat to slightly
up. Correctness is normal (both sides generate replies with reasoning). Results:
`profiles/bench/upstream_cherry_pick/bench-{old,new}.json`.

### 5.1 `98dada0e` conflict resolution points (local `--reasoning-effort` feature reconnected to the upstream jinja design)

- `serve_options.h`: `enable_thinking`/`preserve_thinking` changed to `std::optional<bool>` (the template owns
  the defaults), keeping the local `default_reasoning_effort`.
- `engine.cpp`: adopt the upstream `sampling_mode` based on `info.starts_in_reasoning` (decided after
  rendering), keep the local `frontend_`/`capacity`; **re-add** `Engine::prompt_capabilities()` (deleted
  upstream, needed by the local `--reasoning-effort` startup validation) + the `engine.h` declaration.
- `translate.cpp`: merge the upstream kwargs merging mechanism (`merge_boolean` + template
  `reasoning_effort`) with the local `effective_reasoning_effort` computation + effort validation switch
  (only accept low/medium/xhigh/none supported by the template); the signature keeps 3 params (request,
  server, capabilities).
- `translate.h`: `ResolvedPromptSemantics` regains the `effective_reasoning_effort` field + the 3-param
  signature.
- `generation_service.cpp`/`request_log.cpp`/`test_serve_options.cpp`: keep the local `prompt_capabilities_` +
  `default_reasoning_effort` validation/logging; the tests regain the `prompt_capabilities` definition + all
  switch to 3 params.
- `src/text/CMakeLists.txt`: keep both the local `UTF8PROC_STATIC` (Windows fix) and the upstream
  `ninfer_jinja` target.

### 5.2 `5f5fccab` conflict resolution points (Windows TMA descriptor staging × upstream partial M tile)

- `nvfp4_w4a4_tma.cu`: adopt the upstream ceiling division `(tokens+kBlockM-1)/kBlockM` + the new `tokens`
  argument, keep the local `#if defined(_WIN32)` staging branch (`staged.get()` + `tokens`).
- `nvfp4_w4a4_tma.cuh`: the kernel signature gains the upstream `int token_count` parameter, keeping the local
  `#if defined(_WIN32)` descriptor reference + tensormap publish/acquire block; the kernel body already bounds
  with `token_count` (auto-merge).

### 5.3 Build script fix

- The rsync list in `tools/tp_bootstrap/build_r35.sh` was missing `third_party` (`b9219f3f` adds
  `third_party/llama-jinja/`), so the WSL side lacked the `ninfer_jinja` target and CMake configuration
  failed; `third_party` has been added.

## 6. The official NVFP4 artifact's DFlash2 acceptance rate is higher: localization and a synthetic "official components + local text" artifact

### 6.1 Goal

User observation: the dflash2 acceptance rate of the official artifact `<models-dir>/qwen3_8_27b_nvfp4.ninfer`
is clearly higher than that of the self-converted artifact
`<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2.ninfer`, raising the suspicion that the official artifact was
optimized. Requested: merge "the official artifact's mtp+dflash2+vision" with the weights from
`<models-dir>/W4A16/NVFP4/W4A4+W8A8` into a new artifact for comparison testing.

### 6.2 Confirmed facts (evidence: directly parsing the v3 directories of both artifacts)

| Item | Official `qwen3_8_27b_nvfp4.ninfer` | Self-converted `qwen3_8_27b_w4a4_w8a8_dflash2.ninfer` |
|---|---|---|
| metadata.name | `qwen3.8-27b` | `qwen3.8-27b-w4a4-w8a8` |
| recipe | built-in `qwen3_8_27b_nvfp4` | `<models-dir>/w4a4_family_recipe.py` |
| dflash2 source | maintenance-machine BF16 directory `.../Qwen3.8-27B/dflash2` | `W4A16\NVFP4\W4A4+W8A8\DFlash2-FP8` |
| vision bindings | bf16 195 / q4 27 / q5 54 / q6 1 / q8 2 (+162 non-parameter objects) | completely identical |
| mtp bindings | bf16 7 / q8 3 (+6) | completely identical |
| dflash2 bindings | bf16 45 / q8 11 (+35) | completely identical |
| dflash2 config | `target_layer_ids=[5,19,33,47,61]`, `selector_rank=256`, `selector_top_k=16`, `conv_kernel_size=2` | field-for-field identical |
| text bindings | nvfp4 56 / fp8 74 | nvfp4 42 / fp8 88 |
| proposal | q4 1 + int32 1 | same |

Conclusion: **the official artifact has no "optimization" on mtp/dflash2/vision**; the three components'
formats, object counts and component configuration are field-for-field identical to the self-converted artifact.
There are only two real differences: (a) the dflash2 **source checkpoint** (raw BF16 vs an FP8 intermediate,
both ultimately stored as `q8_g32_fp16`, 11 Q8 parameters each); (b) the per-layer representation of text (the
official rule being "nvfp4 when mlp and layer<56, fp8 otherwise", while the self-conversion imports each
layer's already-stored encoding from the source checkpoint). No BF16 DFlash2 checkpoint exists locally (only
`DFlash2-FP8`, `Qwen3.8-27B-DFlash2-EXL3-5.0bpw` and a few ggufs), so the official draft cannot be reproduced
from the source weights; but the official artifact itself carries that draft's Q8 bytes.

### 6.3 Design decision: per-object byte grafting, not re-running the conversion

Added `tools/convert/graft_components.py`. Each logical binding in a v3 directory points to one object, and an
object's `format/shape/layout/bytes` is determined by the recipe independently of the conversion batch;
measured, the two artifacts' vision/mtp/dflash2 binding objects show **0 missing, 0 format differences, 0 byte
differences** (vision 279, mtp 10, dflash2 56 objects, 1.46 GiB total), so the donor object bytes can be
overwritten directly onto the corresponding positions of the base payload (`entry_payload_start +
object.offset`), with no dequantization, no second quantization, and no payload reordering. The tool also
writes `metadata.graft` and an optional `metadata.name` (JSON section length unchanged, space-padded),
refreshes the 16-byte `artifact_id`, and SHA-256 cross-checks every grafted object in both directions. The
alternative of "dequantize to safetensors and re-convert with the recipe" was not taken: it would add another
quantization round-trip on Q8, and text would have to be re-converted, 22 GB.

### 6.4 Steps and progress

1. [x] Parse both artifact directories, confirm component consistency and overwritability (see 6.2).
2. [x] Implement `tools/convert/graft_components.py`.
3. [x] Generate `<models-dir>/qwen3_8_27b_w4a4_w8a8_official_parts.ninfer`: 411 objects, 2836.0 MiB
   (bf16 274 / q4 54 / q5 54 / q6 1 / q8 28); the in-tool SHA-256 cross-checks all agree.
4. [x] Verification: the Engine loads normally (`engine ready` + `listening`, name
   `w4a4-w8a8-official-parts`); grafted objects 411/411 equal the donor; of the 807 non-grafted objects, 410
   sampled (4.0 GiB) equal the base; the bindings/components tables are unchanged.
5. [x] Measured acceptance rates (same prompt, greedy, 500-token body, dflash2 K=7, `--lm-head-draft`):
   body phase official 44.6% / user 52.9% / grafted 52.2%; thinking phase see §6.6.
6. [x] Conclusion: the synthetic artifact follows the text rather than the official components ⇒ the
   acceptance-rate difference does not come from mtp/dflash2/vision. Truly reproducing the official draft from
   the source weights still requires obtaining its BF16 DFlash2 source checkpoint.

### 6.6 Measured results (same prompt, greedy, `--spec dflash2 --draft-tokens 7 --lm-head-draft`)

| Artifact | Phase | req1 | req2 | req3 | Mean |
|---|---|---|---|---|---|
| official `qwen3_8_27b_nvfp4.ninfer` | body | 42.3% | 44.5% | 47.1% | 44.6% |
| user `qwen3_8_27b_w4a4_w8a8_dflash2.ninfer` | body | 50.7% | 54.0% | 53.9% | **52.9%** |
| grafted `..._official_parts.ninfer` | body | 47.3% | 55.3% | 54.0% | **52.2%** |
| official | thinking (first round, 400 tokens all thinking) | 31.8% | 29.4% | 26.1% | 29.1% |

Thinking phase, all three artifacts re-measured together (`build-win/acc3-*.jsonl`):

- official thinking-phase: 346/1066=32.5% | 347/1057=32.8% | 336/1137=29.6% => mean 31.6% | p1=111
- user thinking-phase: 351/1034=33.9% | 369/905=40.8% | 368/906=40.6% => mean 38.4% | p1=102
- grafted thinking-phase: 352/1027=34.3% | 368/912=40.4% | 368/906=40.6% => mean 38.4% | p1=102

Re-measurement with sampling defaults (`build-win/acc4-*.jsonl`, 4 requests per artifact, 800 tokens, thinking
on, temperature 0.7 / top-k 20 / top-p 0.8):

- official sampled: 39.2% | 37.5% | 29.0% | 42.9% => mean 37.2%
- user sampled: 39.8% | 34.9% | 31.4% | 40.8% => mean 36.7%
- grafted sampled: 45.5% | 38.1% | 47.1% | 42.3% => mean 43.2%

A single run scatters over 29–47%, and with 4 samples the standard error of the difference is about 3–4
percentage points, so this group can only show that "the synthetic artifact is no worse than the other two"; it
is not enough to assert that the official one is highest or lowest; the two greedy groups (body 52.2% vs 52.9%,
thinking 38.4% vs 38.4%) are the low-variance evidence.

Conclusion (component dimension): under the two low-variance settings — greedy body and greedy thinking — the
synthetic artifact (official mtp/dflash2/vision + self-converted text) has essentially the same acceptance rate
as the self-converted artifact, and the official artifact is lowest in both ⇒ **the mtp/dflash2/vision components
are not the source of the acceptance-rate difference**; the difference is determined jointly by the text
(target) weights and the prompt content. Reproducing the "official is higher" the user observed requires
re-measuring with its real system prompt, sampling settings and a longer budget.

Observation: for every serve, the req#1 acceptance rate is clearly lower than req#2/req#3 (e.g. grafted 47.3%
vs 55.3%/54.0%), showing that the first request is affected by cold start; comparisons should look at
req#2/req#3. In the greedy body phase official is lowest and grafted and user are essentially identical
(≈52–53%), i.e. the components are not the source of the difference. The "official is higher" the user observed
most likely occurs in real sessions dominated by reasoning, and must be re-measured with the real system prompt
and a longer budget.

## 7. TP-2 two-card device-memory imbalance: plan A (dflash2 proposal head split) and plan B (selector moved down)

### 7.1 Problem and ledger (measured, dflash2 K=7 + vision, --max-context 102400, after subtracting the ~1 GiB of driver over-reporting in the ledger)

| | GPU0 (shard 0) | GPU2 (shard 1) |
|---|---|---|
| Actual usage | 14636 MiB | 13434 MiB |
| KV (fp8) | 1612.5 | 1612.5 |
| Non-KV fixed overhead | **13023.5** | **11821.5** |

KV must hold the same token count on both cards (split by head), and the ceiling is set by the heaviest card:
a 1202 MiB difference ≈ **76k tokens**; after balancing the ceiling rises from about 190k to about 228k
(**+38k tokens**). At 262144, GPU0 needs 17151 (over by 840 ⇒ OOM) while GPU2 needs only 15949 — consistent
with the "GPU0 OOM, GPU2 has room" phenomenon. KV slope 1612.5/102400 = 15.75 KiB/token/card.

### 7.2 Why "moving whole components" gains nothing (arithmetic conclusion)

Of the fixed overhead, only the dflash2 draft (artifact 2123.6 MiB, measured 1840 extra on shard 0) and vision
(826.5) are exclusive to a single card; together 2746.5 exceeds the symmetric per-card headroom of 1373: with
draft→shard1 and vision→shard0 the ceiling is unchanged (14195), and moving only one layer is worse.
**The only thing that pays is "splitting open" a block.**

### 7.3 Plan A (in progress): split `proposal/head` by vocabulary along the dflash2 route

- Gain: the reduced proposal head Q4 [131072,5120] at 340 MiB split half-and-half ⇒ **about −170 MiB per
  card**, context +~10.8k tokens; no loss of precision (candidates are bit-identical to the whole head), no
  loss of speed (an extra KB-scale exchange per round).
- Current state and gate: `split_proposal_head` in `load.cpp:131-148` excludes dflash2; `draft.cpp:362-385`
  uses the fused `ops::linear_topk` and assumes the whole head. The precedent is MTP's `proposal_argmax`
  (text.cpp:796-836) + `merge_local_row_blocks`.
- Design: add a small op `ops::merge_topk_candidates` (inputs are two columns of [16,U] candidates each,
  merged by "descending score, smaller global id first on ties", with rules consistent with the `linear_topk`
  contract); during proposal each card runs `linear_topk` on its own half table, uses `DevicePair::allreduce`
  to broadcast hidden to the peer, then merges the two candidate columns into a [32,U] exchange and finally
  picks the top-16 with the new op. The global top-16 ⊆ the union of the two halves' top-16, and the tie rule
  matches ⇒ bit-exact.
- Files to change: `load.cpp` (gate + comments), `execution/draft.cpp` (split branch),
  `program/dflash_round.cpp` (`dflash2_proposal_workspace_bytes`), `program/planning/startup.cpp` if needed
  (peer-side capacity), `load/tp_split_spec.{h,cpp}` (comments), and the new op plus its oracle test.

### 7.4 Acceptance criteria for plan A

1. The new op's oracle test (bit-exact comparison against a host naive sort) and
   `ninfer_linear_tp2_split_grouped_head_test` pass.
2. **Digests unchanged**: `ninfer_qwen3_5_tp2_dflash_append_test` is still `0xbad27a494a9bc853` (K=7) /
   `0xbee487264ca8ffb8` (K=5); `ninfer_qwen3_5_tp2_dflash_solo_test` is still `0x4bcc3994a5efba7d`.
3. Ledger: before the change `proposal/head` was replicated on both cards, so **each card drops by half, 170
   MiB** (see the measurement in §7.7).
4. Spot-check greedy dflash2: output byte-identical to before the change with the same command and prompt (see
   §7.7).

### 7.7 Plan A implementation results (completed)

**Changes**: `ops::merge_topk_candidates` (new op + oracle test), `linear_topk` supports the 65536-row
half-table profile, `TextContext::proposal_topk_tp2`, the `draft.cpp` split branch, `startup.cpp` text
workspace reservation, `load.cpp` gate opened, comments synchronized (`tp_split_spec.{h,cpp}`).

**Ledger A/B (same artifact `qwen3_8_27b_w4a4_w8a8_dflash2.ninfer`, same command `--spec dflash2
--draft-tokens 7 --lm-head-draft --vision --devices 0,1 --max-context 102400`, the only variable being the
gate on/off)**:

| Item | Gate off (replicated) | Gate on (half table) | Diff |
|---|---|---|---|
| shard 0 `weights+ctx` | 13878.6 | 13708.6 | **−170.0** |
| shard 1 `weights+ctx` | 12036.6 | 11866.6 | **−170.0** |
| shard 0 `free` | 136.0 | 308.0 | +172.0 |
| shard 1 `free` | 1338.0 | 1508.0 | +170.0 |
| nvidia-smi GPU0 / GPU2 | 15158 / 13954 | 14986 / 13784 | −172 / −170 |

The ceiling is set by shard 0: +170 MiB ÷ 16.13 KiB/token ≈ **+10.8k tokens** (consistent with the design
expectation). The `dflash2 proposal` round workspace also drops from 7.4 → 4.4 MiB (the half-table
`linear_topk` staging is smaller).

**Spot check (greedy, temperature 0, max_tokens 500, same prompt)**:

| Item | Gate off | Gate on |
|---|---|---|
| completion_tokens | 385 (stop token) | 385 (stop token) |
| dflash2 accepted | 252/931 (27.1%) | 252/931 (27.1%) |
| decode tok/s | 66.3 | 66.9 |
| Output text sha256 (first 32) | `2765879ab69653a592f330a335025ff9` | `2765879ab69653a592f330a335025ff9` |
Byte-identical output + identical acceptance counts ⇒ the split path's candidates are bit-equivalent to the whole head (greedy determinism).

**Pitfall hit (fixed)**: the candidate payload of `linear_topk` is **column-contiguous** (`merge.cu:82` writes `column * 16 + rank`, `merge.cu:22` likewise); the new Op initially indexed it as `rank * columns + column`, so at U=7 whole columns were misaligned, the spot-check acceptance rate dropped to 7.4% and tok/s halved. After the fix it recovered to exactly matching the pre-change state, and this layout was written into the Op contract and the oracle test.

### 7.8 A latent hazard in already-committed code (fixed together with work item B)

`DevicePair::allreduce` is the **only** collective communication between the two cards, its semantics are **BF16 element-wise addition** (`add_bf16x8`/`__hadd` in `src/core/tp/device_pair.cu:21,95`), and it requires the byte count to be a multiple of 16 (`device_pair.cu:209`). Scheme A's `proposal_topk_tp2` used it to move **I32 candidate ids and FP32 scores**: memset 0 on one side and adding is numerically a copy, but interpreted as BF16, any 16-bit lane that falls in a signaling-NaN pattern (0x7C01–0x7FFF / 0xFC01–0xFFFF) gets quieted and changes the bit pattern — the low 16 bits of an id (<248077) or either half-word of an FP32 score can hit it, a low-probability but real **silent error**. This round's byte-identical E2E only shows that this time nothing was hit.

**Fix** (without touching `allreduce` itself): `TextContext` owns its own pinned host staging (2 MiB) + events and performs **byte-wise** cross-card transfers as `cudaMemcpyAsync` D2H → `cudaEventSynchronize` → H2D; both A's union exchange and B's selector transfer go through it. The cost is a few host event synchronizations per round (µs level, a round is about 47 ms, <0.1% of it); `DevicePair` and the MTP/single-device routes are left completely untouched.

**TODO (not fixed this round, verification cost is high)**: the MTP route around `text.cpp:486` likewise uses allreduce to move I32 proposal ids, the same class of sNaN bit-pattern risk.

**Phase-B measurement (selector on shard 1, host staging still 5 synchronizations)**: proposal 7.946 → 8.418 ms (+5.9%); K=7 `0xbad27a494a9bc853` and K=5 `0xbee487264ca8ffb8` both unchanged; shard 0 draft block 3655.4 → 3410.4 MiB (−245.0). After hidden (71.7 KB BF16) was moved to in-kernel allreduce: synchronizations 5→4, staging 73 KB→1.4 KB; the remaining 4 can only be eliminated with a byte-wise device-side copy variant of `DevicePair`.

## 8. Work item B: move the DFlash2 selector to shard 1 (done, acceptance in §8.9)

Goal: move `dflash2/candidate_selector/*` (codebook 121.25 MiB each + hidden_projection 2.5 MiB ≈ 245 MiB) from shard 0 to shard 1 ⇒ shard 0 loses 245 MiB ⇒ limit +~15.6k token; zero accuracy loss is the hard threshold.

Design:
1. `tp_split_spec.cpp`: add `dflash2/candidate_selector/` → `shards=0x2` **before** the `mtp/|dflash2/|vision/` rule; the rest of dflash2 stays put.
2. Parameter ownership: promote the selector from `DraftParameters::selector` to top-level `Parameters::dflash_selector`, and decide on-card presence by `source.has_weight(...)` (the shard-local precedent of vision/MTP) ⇒ absent on shard 0, present on shard 1, present on single device (single device keeps running locally).
3. `TextContext::dflash_selector_tp2`: send hidden/candidates/scores/anchors/frontiers to the peer, run `project` + `candidate_selector_path` on the peer, bring drafts/proposal_q back; for sampling use host-side `host_ingress.sampling` copied directly H2D to the peer (not through BF16 addition).
4. Workspace: add a `dflash_selector_peer` reservation to the text plan (peer-side buffers + the selector's own `candidate_selector_path_workspace_capacity_bytes`), decided by "this card has a selector and no draft".
5. Do not change selector Op semantics/arithmetic.

Acceptance: append K=7 `0xbad27a494a9bc853` / K=5 `0xbee487264ca8ffb8`, solo EXIT=0, ledger A/B (shard 0 −245, shard 1 +245), throughput/acceptance A/B (report any >2% regression honestly).

### 7.5 Scheme B (TODO): move `dflash2/candidate_selector` (245 MiB) to shard 1

The selector is needed only once per draft round (its hidden input is ~KB level); moving it to shard 1 rebalances about 245 MiB (+~15.6k token), at the cost of one PCIe round trip per round, which first needs micro-benchmark quantification; `dflash2/feature_projection` (132.8 MiB) is analogous, but it serves prefill (long prompts must ship features), so it has lower priority.

### 7.6 Other options (not adopted, kept on file)

- Uneven TP (KV head 1:3 + FFN 3:1) would swallow 1202 MiB in one go (+38k token), but the part in `tp_split_spec.cpp:138-143` is currently hard-coded to a bisection, and merge/collective also assume half blocks — an explicit product change.
- `--spec mtp` (draft only 148 MiB) opens 262144 with zero code (a line measured in the repo), at the cost of switching the draft backend.
- KV quantization classes (nvfp4/k8v4/int8) lose accuracy, excluded; turning off `--vision` only frees the light card and does not raise the limit under dflash2.


### 6.5 Acceptance

- The new tool depends only on the Python standard library; `git diff --check` is clean.
- The synthetic artifact loads and serves, and component byte ownership is as above.
- The acceptance-rate comparison gives three reproducible numbers (same prompt and sampling settings).


### 8.9 Landing results (acceptance of sendrecv and B, all measured)

- **New `DevicePair::sendrecv`** (`core/tp/device_pair.{h,cu}`): the existing in-kernel kernel was templated into `ar_exchange<kAdd>`; with `kAdd=false`, phase 3 **writes bit-wise** the peer's staged bytes into `recv`; staging, arrival token, slice and double-buffering are all reused ⇒ steady state has **zero host synchronization**, it can go into a CUDA Graph, and it is lockstep-compatible. The unit test `ninfer_tp_device_pair_test` gained adversarial cases (signaling-NaN bit-pattern lanes, I32/FP32 bit patterns, 960B/64B/1MiB three sizes) **PASS**; on this machine `p2p_available=0` ⇒ exactly the in-kernel path is taken.
- **A's hazard is fixed**: the candidate union exchange of `proposal_topk_tp2` now goes through `sendrecv` (I32 ids / FP32 scores no longer pass through BF16 addition).
- **B landed**: `tp_split_spec.cpp` assigns `dflash2/candidate_selector/*` to shard 1; `dflash_selector_tp2` replaces the original host-synchronized channel with two **packed** exchanges (forward draft state, reverse selector output); `peer_stage_copy` and its pinned/event state were deleted; workspace reservations were synchronized accordingly.
- **Acceptance numbers**: ledger shard 0 `weights+ctx` 13708.6 → **13462.6 (−246.0)**, shard 1 11866.6 → **12112.6 (+246.0)** (the sum of the two cards is unchanged); greedy output on the same prompt has sha256 `2765879ab69653a592f330a335025ff9`, **exactly identical** to when the selector was on shard 0, with the same acceptance count `252/931 (27.1%)`; decode **66.7 tok/s** (Phase-A baseline 66.3–66.9, regression <0.5%); append digests K=7 `0xbad27a494a9bc853` and K=5 `0xbee487264ca8ffb8` unchanged; proposal time 8.097/8.037 ms (A baseline 7.946/7.962, +1.3~1.7%, not measurable end to end).
- Gain: shard 0 headroom +246 MiB ⇒ context limit about **+15.6k token** (16.13 KiB/token/card).

## 9. TP-2 dual-card spin hang investigation (2026-09-23): root cause settled and fixed — cross-card false sharing of the token counters

- **Symptom** (encountered by the user several times + twice on the r56 finish5 A/B arms): `ninfer-serve` is alive (/health 200), no error logs, both cards at SM 100%, the host round loop no longer advances; the log freezes before `[tp2-time] rounds=1` for the new request ⇒ **the hang is inside the new request's first DFlash2 round** (at the 6 collective calls of propose or at verify graph replay). Same symptom as Round 34's concurrency scramble, different root cause.
- **Reproduction and forensics**: pure Op queue pressure for 40 rounds (~32k collectives, deep queues, mixed large/small paths) scored zero hits; **graph replay pressure** (new `check_graph_queue`: capture-queue replayed 20 times + interleaved eager collectives, bit-exact verification) hit on run 30 within 40 rounds (scene in `build-win/r57d/hang-stress-30.err`); 40 serve-level arms (base/finish5 alternating, 7 requests per arm) scored zero hits ⇒ the amplifier is the **concurrency window of the dual-card token bump** (graph replay compresses the two counter self-increments to their closest; the production verify graph fires 128 fused bumps back to back per round, likewise).
- **Scene signature** (`NINFER_TP2_AR_WATCHDOG=1` stall dump, normally zero mapped access): `calls=944 last=2097152 tok=[1549,1548] arrA=[1549×4 …] arrB=[1548×4 …]`. A full audit of the call sequence shows it is **constructively paired** (all 13 collective call sites are decided by a single host and launched as a pair on the two streams; the decode step/verify/mtp chain three graph units all use `capture_group` to capture the two streams as a group; DFlash2 rounds take the eager `dflash_propose_batch` path), yet **one side's counter misses one bump** — the lost-update site is exactly the same cache line of the two cards' concurrent RMW.
- **Root cause**: `token_host_` is a single 8-byte allocation with `token_b_ = token_a_ + 1`: on every collective each card RMWs its own int (the small path via the fuse bump kernel's self-increment, the large path via `bump_ar_token`), mapped host memory is not coherent across devices, and **either side's whole-line write-back wipes the increment the other side just landed** (observed shape `tok=[N+1, N]`, i.e. the latter was wiped). Once the count is off by one, **every** collective has both sides spinning forever on `arrival_other[b] != token` (the decode small path is a single block, has no order chain, and the spin point is unique), with no visible error anywhere — the host thread sits at the end-of-round `cudaStreamSynchronize` and never reaches the next CUDA_CHECK. llama.cpp's host-staging token has always used a 64B gap to prevent false sharing (llamacpp-notes:156); in this implementation arrival/order/staging are all allocated per card independently (no cross-card write sharing, safe), only the token pair violates that discipline, and it is also the only cross-card write-shared cache line in the whole design.
- **Fix**: `kArTokenStrideBytes = 64`, giving the two counters their own cache lines each (`core/tp/device_pair.cu`), with zero changes to the data path; the watchdog reads adapt to the stride.
- **Acceptance (2026-09-23, 2×RTX 5060 Ti, sm_120a, Windows native)**: ① `ninfer_tp_device_pair_test` PASS (including `check_graph_queue` bit-exact in three forms); ② after the fix, graph pressure **60 rounds 0 hangs** (before the fix the same battery hit 1 in 40 rounds); ③ the serve bit-consistency control arm: `drafted=2843 accepted=699 (24.59%)`, per prompt 344/109, 399/102, 337/110, 396/100, 495/87, 379/104, 493/87 **identical item by item** to the r55/r56 baseline, throughput 56.6 tok/s (baseline 56.4, no regression).
- **Retained tools**: `DevicePair::start_ar_watchdog` (`NINFER_TP2_AR_WATCHDOG=1`, dumps tok/arr/order only when the call count stalls; signature decoding: tok unequal = count mismatch; tok equal and both arr published = visibility; mixed old/new slots = protocol race), `tests/test_tp_device_pair.cpp::check_graph_queue`, `tools/tp_bootstrap/r57_spin_stress.ps1` (Op-level stress loop), `tools/tp_bootstrap/r57_spin_repro.ps1` (serve-level reproduction loop, alternating artifacts, automatic forensics on hang).

---

# PLAN.md archive (archived 2026-09-24, post-TP2-protocol-refactor state)

Archived verbatim from the working PLAN.md when it was reduced to key information and remaining work, after the TP-2 rendezvous protocol refactor (2026-09-23), the DFlash2 K sweeps and prefill cost attribution (2026-09-23), the DFlash2 draft quantization experiments r54-r57 (2026-09-23), the DSH/agent-loop prefix-reuse fixes and the TP-2 first-token publish fix (2026-09-24), and the DFlash2 draft re-quantization campaign r62-r66 (2026-09-24). Historical record only: the current state and the remaining work live in `PLAN.md` at the repository root, product behavior and measurements in [tp2-dual-5060ti.md](../tp2-dual-5060ti.md).

# NInfer TP-2 plan (2× RTX 5060 Ti · Qwen3.8-27B NVFP4)

> **The only active plan**, and the persistent memory across context compression. Organized: 2026-09-22 (rewritten
> as the pre-delivery wrap-up version after the upstream cherry-picks and the DFlash2 rounds were completed).
> - Full historical record: `tp2-adaptation-worklog.md` —— contains two full PLAN.md verbatim archives (the 2026-09-21 upstream
>   cherry-pick re-planning version; the full text of this version before the 2026-09-22 reorganization).
> - Delivery notes, recommended configuration and measured data: `docs/tp2-dual-5060ti.md`; Windows native port: `docs/windows.md`.

---

## 1. Key information

- **Branch**: `feat/windows-native-port` (Windows native + WSL2 dual tree). All 20 commits by which origin/master leads have been
  cherry-picked (frontend jinja chain + the three NVFP4 performance items), with no end-to-end A/B degradation.
- **Hardware**: 2× RTX 5060 Ti 16G (SYS topology, no P2P, 448 GB/s/card). WSL2 build tree `$HOME/ninfer`;
  Windows working tree at the repository root (build tree `build-win`).
- **Main workload**: Qwen3.8-27B NVFP4, TP-2 (weights split in half by shard, locally built half-geometry shapes). Draft backends: `--spec mtp`
  (K=2, `--lm-head-draft`) or `--spec dflash2` (K=7); `--spec dflash` (v1) is rejected at construction time.
- **Current state**: delivery goals ①–⑤ all complete; vision, the Windows port, the multi-session KV pool, decode graph + AR strategies are all landed and measured;
  DFlash2 is released on TP-2 (B1–B6, verify CUDA Graph, MTP draft chain graphing, declined-draft guard fix).
- **Key numbers** (262,144 configuration, WSL): prefill 1,588 tok/s; decode 59.0 tok/s (MTP K=2); Windows 131,072 configuration
  decode 56.9 tok/s (on par with Linux). DFlash2 K=7 at 4096 context in the greedy tier is about 71 tok/s (on par with MTP K=2);
  the roofline-corrected ceiling ≈108 tok/s (the 180 of the original 90–180 archive is unreachable).

**Recommended run configuration** (same recipe for WSL 8088 / Windows 8099, Windows side `--max-context 131072`):

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2 --lm-head-draft --vision --reasoning-effort medium
```

`--reasoning-effort low|medium|xhigh` is the process-level default reasoning effort (the request body takes precedence); `--chat-template` becomes
available with the first echelon (the template is executed by the embedded Jinja).

### Environment and operations notes

| Item | Value |
|---|---|
| Build (WSL) | `bash tools/tp_bootstrap/r55_build.sh` (rsync + `cmake --build build_dyn -j 8`, success marker `BUILD_EXIT=0`, about 2–3 minutes; the WSL build tree is now `build_dyn` (Ninja), `build_r35.sh` still points at the deleted `build/` and is unusable) |
| Build (Windows) | `tools/win_port/configure.bat` + `build.bat` (VS2022 + CUDA 13.3, `-DCMAKE_CUDA_ARCHITECTURES=120a`) |
| Serving | WSL 8088 (`serve_supervise.sh`); Windows 8099 (`tools/win_port/serve.ps1`, foreground by default; self-test serving must use a harness background job, processes started with `Start-Process` are killed when the tool call returns) |
| Runtime PATH (Windows) | FFmpeg (`D:\ffmpeg-dev\…\bin`) and libcurl (`D:\curl-dev\…\bin`) must be on PATH, otherwise `STATUS_DLL_NOT_FOUND` |
| artifact | `<models-dir>/qwen3_8_27b_nvfp4.ninfer` (23.7 GB, old artifact, template `c3cf9e34…`); the new official artifact (template `a497db9e…`) becomes available with the first echelon |
| Process discipline | Only one model process on the whole machine at any time (WSL 8088 and Windows 8099 are mutually exclusive); stop the service before relinking the exe on Windows (LNK1104) |
| Toolchain | Nested `pwsh` → `wsl -e bash -lc` swallows `$var` and redirections ⇒ write commands as script files under `tools/tp_bootstrap/` and execute those; a single blocking call is capped at 600 s ⇒ use background jobs for long tasks |
| GPU/WSL state (from 2026-09-23) | The machine sees 3 cards: nvidia-smi order 0/2 = 5060 Ti, 1 = Tesla T10; with `--devices 0,1` the ledger reports 16310.6 MiB for both shards ⇒ the CUDA ordinals pick up the two 5060 Ti cards (mismatched with the nvidia-smi order of the T10), TP-2 pairing is unaffected. On the WSL side CUDA failed across the board that day (unchanged tests for cast/q4/q6/q8/q5 all failed, presumably CUDA init crashing once the T10 became visible) ⇒ tests were moved to Windows `tools/win_port/test.ps1`, and the WSL build_dyn is used only for compile verification |

---

## 2. Completed summary (details in the archive)

- ① MTP consistency: root cause = shards did not write replay records (fold consumed an empty log), fixed; the criterion is "no degradation + same-tier quality + actual speedup".
- ② Benchmarks: prefill 615→1,588 tok/s; decode plain 31.4 / MTP K=2 50.5–54.0 → 59.0 tok/s.
- ③④ KV / context: five KV dtypes usable; 65,536 → 131,072 → 262,144 (vocabulary/hidden parallel split, −1.2 GiB/card).
- ⑤ Documentation: `docs/tp2-dual-5060ti.md` + `docs/windows.md`.
- Vision: static shard division of labor (MTP→shard 0, vision→shard 1), single-image limit 8,192 token, no new configuration.
- Windows native port: VS2022 + CUDA 13.3 build; CLI/serve/TP-2 all working, performance on par with Linux; TMA descriptor
  mapped-pinned fix.
- Optimizations: exact-batch verify graph (−4.4 ms/round), plain decode graph (+12.6%), AR transfer split by payload (+0.63%),
  MTP draft chain graphing (−0.2~0.3 ms/round); AR∥MMA sub-block pipelining measured 5% slower, rejected and reverted.
- Round 19/20: chunk planning decoupled from ring rewind (1a/1b/1c), results for the same prompt no longer depend on engine history; the host KV arena
  is pageable by default (`--host-kv-pinned` pins it, a rejected pin falls back automatically).
- Multi-session KV pool: host KV swap-in/swap-out + 5-session LRU + shared-prefix mirroring; recalled KV is byte-identical to device.
- Upstream cherry-picks: first echelon 5 items (jinja frontend chain + `--chat-template`), second echelon 3 items (NVFP4 TMA routing,
  SwiGLU epilogue, non-256-divisible tokens) all landed; e2e A/B no degradation (decode +1.3%~+5.8%).
- Official NVFP4 artifact acceptance-rate investigation: the synthetic artifact (official mtp/dflash2/vision + self-converted text) matches the self-converted one, the official one is lowest
  ⇒ the difference comes from the text weights and is unrelated to the components.
- TP-2 memory imbalance: scheme A (reduced proposal head split by vocabulary, −170 MiB/card) and scheme B (selector moved to shard 1,
  −246/+246 MiB) are both landed, with byte-identical output.

---

## 3. Outstanding items

### 3.1 Pre-delivery wrap-up

- [ ] Re-test concurrency C=1/2/4 and streaming/stop smoke with the recommended configuration (Round 31/34 conclusions are based on the bf16 configuration).
- [ ] On delivery, remove this file (PLAN.md) per AGENTS.md; decide the keep/cleanup scope of the ~300 one-off diagnostic scripts under
      `tools/tp_bootstrap/` (`build_r35.sh`, `serve_*`, the `r37+r38+r39+r4x` series and the `docs/tp2-dual-5060ti.md` references are
      reproducible procedures).

### 3.2 Windows single-card CLI tiny artifact subtask (user decision; prerequisites ready)

Goal: produce a `.ninfer` small enough to fit on a single 16 GiB card and run "load → generate → sample" on one card with the Windows `ninfer.exe`.
Prerequisites (Round 54.9): the CPU build of torch is in `build-win\torch-venv` (torch 2.14.0+cpu); the converter interface has been confirmed.
Next steps: ① run the official recipe directly on a tiny checkpoint (`--recipe qwen3_8_27b_nvfp4 --components text`);
② if that fails, write a tiny recipe/override; ③ put the generation script at `tools/win_port/tiny_model.py` (synthetic tiny config +
random bf16 weights, HF naming); ④ on success, write it into `docs/windows.md` and commit.

### 3.3 Unfixed defects and known limitations

- The TP-2 route with `bf16` KV + `--spec mtp` crashes on the very first prefill (`prompt.cu:63` `cudaErrorInvalidValue`);
  fp8 + MTP works (the recommended configuration is unaffected). Entry point for investigation: the launch parameters of bf16 prompt-attention.
- The MTP route around `execution/text.cpp:486` likewise uses a BF16-add allreduce to move I32 proposal ids ⇒ lanes landing on
  signaling-NaN bit patterns get silently rewritten (DFlash2's equivalent exchange has been switched to byte-wise `DevicePair::sendrecv`; MTP has not,
  the verification cost is high).
- The MTP3 ≥ 70 tok/s threshold is not reached (the pure decode ceiling at K=2 is 50.5–54.0).
- The TP-2 route has not run the perplexity evaluation (quality evidence is A/B multi-sampling on the same prompt).
- The per-shard arena's `memory_summary()` still reports `pages 0/0` (a display-only issue).

### 3.4 Deferred / optional (unscheduled)

- Round 15 remainder: end-to-end re-test of tool-call constraints with the real model (75 probes); WARN diagnostics; route B (full GBNF);
  TP-2 `validate_licensed_tokens` guard.
- Round 14 candidates (deferred by the user): ① append an anchor at the end of the system+tools prefix (fixes the 46.9 s first round of a new session/after compression, saves about 11 s);
  ② prefill throughput (suffix 0.95–1.13k tok/s, cold start 1.35–1.38k); ③ ~0.3 s fixed overhead per request.
- Archive §11.6: MTP priming uses placeholder embeddings for image columns (~100 lines, no evidence of benefit); the single-image limit 8,192 is hard-coded
  (yielding shard 1 would regain a larger envelope); shard 0 idles during visual encoding (acceptable).
- Archive §6: complete the KV dtype sweep (nvfp4/k8v4 were only verified for startup and throughput, no quality data).
- To truly replicate the official artifact's DFlash2 draft from source weights, one must first obtain its BF16 source checkpoint (this machine only has FP8/EXL3/GGUF).

### 3.5 Environment-blocked items not done (covered by alternative paths)

- [ ] WSL-side Op tests + byte consistency + frontend fixture tests: **WSL2 CUDA driver crash** (PTX JIT inside `cudaGetDeviceCount()`
      segfault, GPU held by a Windows service); the jinja tests passing proves the binary is fine, Op tests were moved to the Windows side.
      Windows side is green (NVFP4 A4/A16, frontend, jinja; `NINFER_OP_REPORT_STATS=1` error metrics within tolerance).

### 3.6 DFlash2 on TP-2 (main body complete; remainder unscheduled)

**Status**: `--spec dflash2` is released on TP-2 (`--spec dflash` v1 is still rejected at construction time). Completed: load/sharding (B1),
feature sink context materialization (B2a/B2b), masked proposal forward (B3), assembled round `program/dflash_round` (B4), whole-round hookup
(B5), draft ring moved with session/checkpoint (B6), verify window CUDA Graph (round C), MTP draft chain graphing (round E),
reduced proposal head split by vocabulary (scheme A), selector moved to shard 1 (scheme B). New `DevicePair::sendrecv`
does byte-wise cross-card exchange, eliminating the sNaN bit-pattern risk of scheme A using BF16-add allreduce to move I32/FP32. Golden values: append K=7
`0xbad27a494a9bc853` / K=5 `0xbee487264ca8ffb8`; ring `0x1a53fd824cd4e360`; solo digest
`0x4bcc3994a5efba7d`.

**Root cause settled (round A, 2026-09-22)**: `forward_tp2_window` never bound `active_valid_columns_` /
`active_sequence_batch_` ⇒ the window wrote KV for every column; while DFlash2's budget-clamped window pins the tail columns at the same absolute position ⇒ in the same forward
multiple warps concurrently write the same paged-KV slot, content is nondeterministic across runs ⇒ argmax flips at near-ties (the earlier S1 "fully synchronized" only lowered the
probability). Fix: the window accepts `valid_columns` (clamped columns are forbidden to write KV, their logits are zeroed), passed only by DFlash2, the MTP/plain routes
are bit-unchanged. After the fix, recall reproduces from-scratch token by token, the retention property is empirically demonstrated and open (see the archived A/B/C round evidence).

**Recall contract**: grid-aligned boundaries (`reuse % reuse_grid == 0`) get a full token comparison; non-aligned fallback (declined boundaries) only pins the
boundary first sample, tail reproducibility is guaranteed by the solo digest across processes; declined-draft state is exposed via
`GenerationResult::draft_context_declined`.

**Declined-draft guard fix**: the non-grid fallback (template prefixes ~37–57 token always hit) used to set `dflash_draft_declined_` every round ⇒
`extent=0` ⇒ only 1 token committed per round (~22 tok/s). Now a gate is added for DFlash2: when `position <= max(reuse_grid, 16×remaining)`
no non-aligned boundary is taken and an aligned scan replay clip is used instead ⇒ drafts stay online the whole time; measured 3.17 committed per round, about 71 tok/s. The synthetic-load
numbers of 20–25 tok/s in the archive predate this fix.

**Remainder**:

- [ ] B7 performance acceptance: dual-card throughput against the target (roofline-corrected ≈108 tok/s, the 90 end matches, 180 unreachable) + comparison with MTP +
      no regression for plain/mtp, and record the recommended K for `--spec dflash2`.
- [ ] Unknowns to measure: the true free of the Windows native build (the ledger `free` comes from `cudaMemGetInfo`, which under-reports by about 1 GiB on WSL2);
      numerical equivalence of splitting the draft by head/row; the workspace peak at K=15; the per-step cost of converging the target's 5 residual layers across shards;
      the latency of the draft's per-layer allreduce; the equivalence of sparse rejection sampling under TP-2 sharded logits.

**Design constraints (do not re-investigate)**: replicating the full draft is infeasible — at full context (262144 / fp8 / MTP K=2) free is only
697/1217 MiB, while the draft is 2.07 GiB/card; after splitting it is about 1.04 GiB per card. The proposal condition is the residual concatenation projection of the target's 5 blocks,
and TP-2 cuts the 64 layers across two cards ⇒ a cross-card handoff is needed (both cards' residuals are bit-identical, so capturing shard 0 alone suffices). The candidate selector reads the full-vocabulary
codebook directly (pred/succ each 248320×256 BF16, about 254 MiB), so it must be whole. DFlash2 and MTP are mutually exclusive: choosing DFlash2 frees
shard 0's MTP weights 430 MiB + KV 516 MiB. The per-round cost breakdown, the D1–D4 decisions and the negative conclusions of each round are in the archive.

### 3.7 DFlash2 draft quantization experiments (2026-09-23; decision made: keep opt-in)

**Goal**: verify how much "changing only the conversion recipe, not the Ops" can shrink the TP-2 DFlash2 draft and how much acceptance rate it costs.

**Conclusion**: it can, but only one block can be touched — the draft's fused gate/up SwiGLU can run at Q4, **saving shard 0 450 MiB**, with K=7 acceptance
unchanged (24.77% vs 24.59%), K=2 about −2.5pp (1.6σ, sample insufficient to conclude), probe throughput flat. `mlp/down` and
`attention/output` cannot be changed: the draft's fused dynamic grouped-conv finish hard-codes Q8 (see below).

**Decision (2026-09-23, confirmed by the user)**: keep it as an **opt-in override**, do not change the official recipe (`official_recipes.py:35-46`
still pins mtp/dflash/dflash2 at Q8); the new artifact is only an experimental piece, the product route continues with the Q8 draft.

**Distribution-consistency conclusion**: this engine's speculative decoding is distribution-exact — every token emitted by the greedy branch is the target's own argmax
(`src/ops/kernel/speculative_round.cuh:120-142`), and the sampling branch is rejection sampling with masked residual correction (`:144-214`, algorithm description
`:216-228`), with clamped columns not writing KV (`docs/tp2-dual-5060ti.md:251-256`) ⇒ **the draft weights (including their quantization) only change the proposal
distribution q and the acceptance rate, not the model output distribution**. But "bit-identical output trajectory" is broken: window shape differences flip at near-ties (an existing property, precedent
`worklog Round 52`), and measurements show the greedy text length changed for both K values.

**Scripts and artifacts**:

- override: `tools/tp_bootstrap/r54_draft_mlp_override.py` (`dflash2/layers/*/mlp/gate|up` → `q4_g64_fp16`,
  `grouped_absmax`); the recipe is still `<models-dir>/w4a4_family_recipe.py`, draft source `W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8`.
- Conversion: `tools/tp_bootstrap/r54_convert_mlp4.ps1` (332 s, 1218 objects, 1 file).
- New artifact: `out/qwen3_8_27b_w4a4_w8a8_dflash2_gateup4.ninfer` (24,338,661,380 B, −451.6 MiB vs baseline).
- A/B arms: `tools/tp_bootstrap/r54_dflash2_arm.ps1` (start service → greedy_probe 7 prompts × 160 token → aggregate → stop service);
  inspection tool `tools/tp_bootstrap/r54_draft_quant_inspect.py`, dry-run `tools/tp_bootstrap/r54_draft_quant_dryrun.py`.

**Ledger (`--max-context 131072`, DFlash2 K=7 + vision + host KV 32 GiB, same binary)**:

| arm | shard 0 `weights+ctx` | shard 0 `free` | shard 1 |
|---|---:|---:|---:|
| baseline (gate/up q8) | 13462.6 | 102.0 | 12112.6 |
| gate/up q4 | **13012.6** | **552.0** | 12112.6 |

⇒ shard 0 limit +450 MiB ≈ **+28.5k token** (16.125 KiB/token/card).

**Acceptance rate (greedy, each arm with an independently cold-started service)**:

| arm | K=7 drafted/accepted | K=7 acc | K=7 tok/s | K=2 drafted/accepted | K=2 acc | K=2 tok/s |
|---|---|---:|---:|---|---:|---:|
| baseline | 2843 / 699 | 24.59% | 54.9 | 1012 / 603 | **59.58%** | 52.0 |
| gate/up q4 | 2826 / 700 | **24.77%** | 57.2 | 1035 / 591 | 57.10% | 51.8 |

(tok/s is converted from the probe time of 7×160 token, including prefill, ±5% noise.)

**Failure record (important)**: the first version also dropped `mlp/down` to Q5; loading was fine but warmup reported
`FATAL warmup failed | linear dynamic grouped conv add: invalid projection_weight`. Root cause:
`require_finish_projection_weight` in `src/ops/wrapper/dynamic_grouped_conv.cpp:61-73` hard-requires
`Q8_G32_FP16 / RowSplit / group 32` — that finish fuses down/output with the dynamic convolution delta, and the draft's
`mlp/down` (5120×17408) and `attention/output` (5120×4096) have no Q5/Q4 variants. That intermediate artifact has been deleted.

**Follow-up levers (unscheduled; ordered by cost-effectiveness)**:

- [x] **A Q5 variant of the fused dynamic-conv finish** (relaxing the Q8-only check in `dynamic_grouped_conv.cpp`):
      landed (r56, 2026-09-23). `mlp/down` −172 MiB, `attention/output` −42 MiB (5 draft layers total).
      Implemented as the "Q5 GEMM materialization + shared finish kernel" route: **reuse the already-compliant `select_q5_a16_launch` Q5 A16
      GEMM, no new GEMM kernel** (the original "needs a new kernel" estimate was too high) — added q5 shape `n5120_k4096`
      (`src/ops/linear/q5/shapes/n5120_k4096.cu`, mirroring selector n5120_k17408) + dispatch registration;
      added `src/ops/dynamic_grouped_conv/q5/` (plan/materialized, route name
      `dynamic_grouped_conv_add.q5.*.materialized_bf16`); the finish kernel was extracted from q8 materialized into a shared
      `src/ops/dynamic_grouped_conv/dynamic_conv_finish.cu` (shared by q8/q5, arithmetic bit-unchanged). The wrapper
      validation and non-overlap checks dispatch by qtype (Q8_G32_FP16 / Q5_G64_FP16 RowSplit, Q5 includes the qhigh high-plane check);
      the capacity API `linear_dynamic_grouped_conv_add_workspace_capacity_bytes` gained a leading qtype parameter (aligned with
      the `linear_workspace_capacity_bytes` convention, callers in `draft.cpp`/`dflash_round.cpp` followed along); the public contract
      (`include/ninfer/ops/dynamic_grouped_conv.h`) was synchronized to the dual-codec semantics; bench gained `--qtype q8|q5`.
      **Verification**: the Op oracle `ninfer_linear_dynamic_grouped_conv_add_test` passes (4.38 s: Q8/Q5 ×
      C∈{4096,17408} all W/B domains 120 shapes × graph replay, FP64 oracle independently decoded by stored scale);
      `ninfer_linear_q5_a16_test` passes (3.11 s, the new 5120×4096 full-route boundary vs FP64 oracle);
      WSL build_dyn and Windows build-win compile.
      **Experimental artifact converted** (2026-09-23, CONVERT_EXIT=0, 285.8 s, 1218 objects):
      `<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_finish5.ninfer` (23,449.4 MiB, **−213.3 MiB** vs the baseline
      23,662.7, consistent with the exact calculation; exactly 10 objects `q5_g64_fp16`: `dflash2/layers/*/attention/output`
      [5120,4096] ×5, `dflash2/layers/*/mlp/down` [5120,17408] ×5, all other draft tensors unchanged).
      override `tools/tp_bootstrap/r56_draft_finish_override.py`, conversion script
      `tools/tp_bootstrap/r56_convert_finish5.ps1` (r54 recipe, `--device cpu`).
      **r56 measurement** (2026-09-23, Windows 8099, K=7 greedy probe 7×160, 2×5060 Ti, logs `build-win/r56/`):
      ledger shard 0 free **102.0 → 314.0 MiB** (weights+ctx 13462.6 → 13250.6 = −212.0 MiB device-side,
      artifact −213.3, 1.3 absorbed by allocation granularity), shard 1 unchanged 810.0 ✓ locality holds;
      acceptance base **24.59%** (699/2843) → finish5 **25.13%** (703/2798), **+0.54pp** (p0/p2/p3 identical round by round to
      base, p1/p4/p5/p6 text trajectories drift so the denominator shrinks; no sign of decline); throughput 56.4 → 56.2 tok/s (−0.4%).
      Determinism: the base arm reproduces the r55 baseline byte-for-byte across a rebuild (2843/699, all text lengths identical) — after the shared finish refactor the
      Q8 route has zero drift. Verdict: **keep opt-in**; before promoting it to a contract feature the three-piece verification at the end of §3.7 is still owed.
      **Known flake (existing issue, the user confirms hitting it occasionally several times)**: the first finish5 arm hangs probabilistically mid-stream at req#2
      (both cards spin at SM 100% waiting for handoff, /health 200, no error in the logs; desktop processes ruled out), a rerun passes;
      the hang mechanism awaits a dedicated investigation (deferred by the user, log kept at `build-win/r56/serve-finish5-hang1.log`); **that dedicated item was taken over by §3.8** (after the second reproduction on 2026-09-23 it was settled as device-side spinning, handling in §3.8).
- [x] **New `feature_projection` profile** (5120×25600, 132.8 MiB): landed (r55, 2026-09-23).
      Correction: that tensor goes through the plain `linear` Op (`project` → `ops::linear`), not `linear_add`.
      Added `src/ops/linear/q5/shapes/n5120_k25600.cu` (mirroring selector n5120_k17408: T=1 simt_r8_c4,
      T=2–6 ksplit, T≤24 simt_r8_c8, otherwise mma_r64_c128) + dispatch table registration + a compliance test case
      (`tests/ops/linear/test_q5_a16.cpp` 5120×25600, seed 183U, FP64 oracle). opt-in override
      `tools/tp_bootstrap/r55_draft_feature_proj_override.py` (`dflash2/feature_projection` → Q5).
      Exact gain: that object is shard 0's full local copy (tp_split_spec's dflash2/* Replicated shards=0x1),
      Q8 132.81 MiB → Q5 82.03 MiB (row_split_k128_v1: 400 groups/row × 42 B + scale), **−50.8 MiB**
      (the original estimate of −63 was too high). Conversion script `tools/tp_bootstrap/r55_convert_featproj5.ps1` (r54 recipe +
      r55 override, `--device cpu`, no GPU needed). WSL build_dyn compiles (BUILD_EXIT=0).
      **Experimental artifact converted** (2026-09-23, CONVERT_EXIT=0, 299 s, 1218 objects):
      `<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_featproj5.ninfer` (23,611.9 MiB,
      measured −50.8 MiB vs the baseline, consistent with the exact calculation; `dflash2/feature_projection` confirmed
      `q5_g64_fp16:[5120,25600]`, all other draft tensors remain Q8). A/B baseline:
      `<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2.ninfer` (23,662.7 MiB).
      **r55 measurement** (2026-09-23, Windows 8099, K=7 greedy probe 7×160, 2×5060 Ti, logs `build-win/r55/`):
      Op compliance `ninfer_linear_q5_a16_test` passes (5120×25600 full-route boundary vs FP64 oracle, 2.9 s);
      ledger shard 0 free **102.0 → 152.0 MiB** (weights+ctx 13462.6 → 13412.6, device-side −50.0 MiB,
      allocation granularity ate 0.8), shard 1 unchanged 810.0 ✓ locality holds; acceptance base **24.59%** (699/2843) →
      featproj5 **24.40%** (696/2853), **−0.19pp** (the same band as r54 gate/up Q4's ±0.2pp ⇒ unchanged);
      throughput 56.7 → 56.4 tok/s (−0.5%). Probe determinism: base reproduces byte-for-byte across a rebuild (2843/699, all text lengths identical);
      changing the draft necessarily drifts the text slightly (FP ties move with the round cut, same phenomenon as r54 gateup4: 834→837, 795→784;
      featproj5 drifts 4/7 prompts), the target greedy stream is still the verification oracle. Verdict: **keep opt-in**,
      before promoting it to a contract feature the three-piece verification at the end of §3.7 is still owed.
- [ ] **selector codebook quantization** (242.5 MiB BF16, not a GEMM): requires a new selector path (codebook codec + preserving the top-k domain).
- [ ] **Full draft NVFP4** (about −0.95 GiB): first add a BF16/FP8→NVFP4 method to the converter (currently only `import_encoded` can produce NVFP4),
      then add the NVFP4 profiles for the three items above; `linear_pair` and the draft version of `attn_input_proj` are still Q8-only, and full quantization must touch them.

**Combined experimental artifact (r57, 2026-09-23)**: the three verified quantizations combined into the final experimental piece
`<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_draftall.ninfer` (22,947.1 MiB, **−715.6 MiB** vs the baseline
23,662.7, = exact sum of 451.6+50.8+213.3; 21 objects: gate/up ×10 `q4_g64_fp16`,
feature_projection/down/output ×11 `q5_g64_fp16`; `attention/{query,key,value,context_*}`
[6144,5120], the draft versions of `attn_input_proj`/`linear_pair` etc. are still Q8 — part of lever #4's scope).
override `tools/tp_bootstrap/r57_draft_all_override.py` (with count validation) +
`tools/tp_bootstrap/r57_convert_draftall.ps1` (CONVERT_EXIT=0, 290 s).
**r57 measurement** (Windows 8099, K=7 greedy probe 7×160, logs `build-win/r56/`): ledger shard 0
weights+ctx 13462.6 → **12748.6** (−714.0 MiB, granularity absorbed 1.6; the 2 MiB difference from the sum of the individually measured levers 712.0
comes from their differing granularity rounding), free 102.0 → **816.0 MiB** (≈ +45k token limit); shard 1 unchanged 810.0 ✓;
acceptance **25.3%** (706/2791) vs baseline 24.59% (no decline); throughput 56.9 tok/s (+0.9%).
Named `qwen3.8-27b-w4a4-w8a8-draftall`.

**Verification that must be added before promoting this to a contract-protected feature (not done this time)**:

- [ ] Re-baseline the DFlash2 three-piece suite against the new artifact: `ninfer_qwen3_5_tp2_dflash_append_test` (K=7/K=5 golden values),
      `ninfer_qwen3_5_tp2_dflash_solo_test` (digest), `ninfer_qwen3_5_tp2_sessions_test` (retention assertions).
- [ ] Sampling mode (temperature>0) acceptance/throughput A/B: distribution exactness is guaranteed by the algorithm, but a changed q affects the acceptance rate under sampling.
- [ ] The Op oracle for Q4 gate_up at that shape, plus the "independently decoded by stored scale" check (AGENTS.md's numerical contract).

---

### 3.8 TP-2 rendezvous protocol refactor (2026-09-23; incident-driven, in progress)

**Incident (second reproduction, and localized to device-side spinning)**: 2026-09-23 21:02:56, Windows 3456, `qwen3.8-27b-w4a4-draftall`
(245760 / k8v4 / DFlash2 K=5 / 4-session host KV). req#34 froze after prefill 2,189 tok + decode 158 tok:
both cards SM 100%, memory steady at 12,962/12,558 MiB, `/health` 200, exactly 1 host core continuously saturated, zero log output;
req#35–#39 (the client retries roughly every 5 minutes) were accepted but never advanced ⇒ 6 requests silently swallowed (`--max-pending-requests 16`).
Log kept at `build-win/serve-hang-2026-09-23.log` (30,559 B), scene snapshot `build-win/live-serve-snapshot.log`.
**After killing the process both cards still reported 100% / 2805 MHz / 22–25 W / 0 MiB with no compute process** ⇒ what is stuck is a device-side spinning kernel
(the low-power signature of `__nanosleep` polling). Process termination does not reclaim it immediately: killed at 21:45, at 21:47 both cards still 100%/2805 MHz/~25 W, at 21:49 they fell back on their own to 180 MHz/0%/4–7 W ⇒ further evidence of device-side spinning; confirm both cards are back to 0% before measuring (confirmed this time, no reboot needed).

**Relation to the afternoon fix (conclusion: not a missed patch)**: the deployed `C:\ninfer\ninfer-serve.exe` = `build-win/apps/ninfer-serve.exe`
(202,497,024 B, 14:23:32), the binary contains the literals `NINFER_TP2_AR_WATCHDOG` and `graph queue[` ⇒ the 14:50–14:51
batch of commits (`12760cb8` watchdog → `f8d33f13` cache-line fix → `fe96c13d` graph replay test) was already in the tree at link time;
the 14:24–14:30 r57e (graph pressure 60/60 PASS) and r57f (serve recheck) ran exactly that. ⇒ This is **another path of the same class of phenomenon**:
`src/core/tp/device_pair.cu:191-197` only plugged the one missing bump of "two counters on the same cache line overwriting each other".

**Three design defects (root-cause judgment)**:

- **D1 the rendezvous identity = a self-incrementing counter each card RMWs** (`device_pair.cu:208`, `:116-119`): it treats "the two cards permanently lockstep" as
  an unverifiable, unrecoverable distributed invariant. And on this platform it cannot be fixed with atomics — `atomicAdd_system()` requires
  `hostNativeAtomicSupported`, which PCIe consumer cards do not have (llama.cpp `ggml/src/ggml-cuda/allreduce.cu:56-58`).
- **D2 the wait is strict equality with no upper bound** (`:153-158`): the only failure mode is livelock, and it can take down the whole service.
- **D3 the rendezvous lands in the least controllable coherence domain** (two GPUs writing the same mapped host memory), about 128 times per token.

**Key decisions (against llama.cpp ar3-opt and upstream `allreduce.cu`; details in `tp2-llamacpp-notes.md`)**:

1. **Borrow "who issues the number", not the architecture.** Adopt a host-authoritative id, but do **not** adopt llama.cpp's graph-level meta architecture
   (about 81 segments of host lockstep per decode step): measured on the same machine, 3 cards gave 44.5 tok/s vs this engine's 2 cards at about 100 tok/s,
   and it violates this repository's "Models own finite execution composition" ownership.
2. **Id delivery reuses a mechanism this engine already has**: each round the host writes a pinned `base` → an in-graph memcpy node copies it into a device scalar →
   the kernel computes `id = base + call_index` (`call_index` is a capture parameter, constant across replays). Same route as `valid_columns`
   (`src/models/qwen3_5/execution/text.h:203-216`, `tp2_generation_core.cpp:998-1006`).
   **It must go through the memcpy node and must not be passed by value as a kernel argument** — llama.cpp's form, once captured into a graph, replays a frozen value, the spin condition is
   satisfied by the previous round's leftover value ⇒ it reads without waiting and silently computes wrong (harder to find than a hang).
3. **The large-payload event path is only an A/B candidate**, not a presumed replacement: sliced is already at the link floor (2.70 vs 2.66 ms); upstream's
   copy-engine fixed overhead is about 80 µs against SM kernel stage A's about 30 µs, hence its 1 MB threshold.
4. **Add what neither side has**: bounded spinning + failure semantics + a degradation path. Partial merge (`ggml-backend-meta.cpp:2126-2180`,
   verified) is non-associative under BF16 and conflicts with the bit-wise digest contract, so it is not done this round.

**Change list**:

**Phase 1 —— stop the bleeding (protocol untouched, can ship independently)**
- [ ] `device_pair.cu`: add a `clock64()` upper bound to the spin (env `NINFER_TP2_AR_TIMEOUT_MS`, default 2000, 0=off); after the timeout,
      `__threadfence_system()` + set a mapped `stalled` flag and make all blocks exit; extend the existing `ArWatchState`
      (`:236-244`) to carry that flag.
- [ ] `allreduce`/`sendrecv` return status; after a timeout the host drains both streams, resets arrival/order/parity, fails that request and
      **poisons that session context** (requiring a fresh prefill, to avoid continuing with half-written KV), while the service stays alive.
- [ ] A per-round host-side invariant check: `token_a_ == token_b_` (both mapped counters are directly host-readable), and on mismatch dump and fail.
- [ ] Acceptance: fault injection (artificially make one side miss one collective / lose one publish) must manifest as "that request 5xx + service alive + the next request is normal";
      real agent traffic for ≥1 h continuously with no permanent hang; prefill/decode A/B regression ≤1%.

**Phase 2 —— make the id authoritative (delete the whole desync class)** ✅ Done (2026-09-23)
- [x] `device_pair.{h,cu}`: delete `token_host_/token_a_/token_b_`, `bump_ar_token`, `fuse_bump` and
      `ar_size_keyed_`/`NINFER_TP2_AR_STRATEGY`. The landed form is stricter than the original plan, and all three deviations are intentional:
      - **One id channel per captured graph** (`create_ar_channel()`, channel k occupies `[(k+1)<<32, …)`), rather than "one pinned cell per device":
        each round launches three windows once each, and sharing one cell would let the next write clobber it before the previous launch's memcpy node executed, crossing values.
        Each channel = one pinned cell + one device scalar per device (the source and destination of the in-graph memcpy node).
      - **Ids are 64-bit throughout** (arrival/order slots changed to `unsigned long long`): a 32-bit truncation would make different channels' id segments collide,
        and then a stale slot could satisfy a future spin again — exactly the class being eradicated.
      - **eager goes through 64-bit kernel arguments** (starting at `1<<62`, a separate interval), not a mapped cell: eager calls are enqueued back to back and
        sharing one cell would likewise read the later id. The kernel only does `token = base_dev ? *base_dev + call_index : value`,
        and `call_index` is a capture-time constant, so baking it into the argument is safe (baking the id is not).
      - The hooks are `begin_capture(channel)/end_capture()/arm_round(channel)`: `end_capture` records `calls` itself (one less source of truth);
        `ar_token_skew()` is deleted (the counter concept is gone) → `ar_last_id()` serves error messages and the watchdog;
        `clear_ar_stall()` only clears the trip flag — **ids never repeat ⇒ a stale slot cannot satisfy a future spin ⇒ the stalled round is recoverable in place**,
        which is the substantive gain of Phase 2 over Phase 1.
- [x] `src/core/decode_graph.{h,cpp}`: **unchanged** (the hooks are called explicitly by the caller, no implicit `cudaStreamIsCapturing` determination is used).
- [x] `src/runtime/engine/tp2_generation_core.{h,cpp}`: `WindowGraph` gains `ar_channel`; the three capture points are wired to the hooks
      (`capture_verify_graph`/`capture_decode_graph`/`capture_mtp_chain_graph`); `launch_window_graph` calls `arm_round` first.
      The 9 model-side `allreduce/sendrecv` call sites are indeed **untouched, not a single line**.
- [x] Acceptance (2026-09-23):
      `ninfer_tp_device_pair_test` PASS — including the reworked `check_graph_queue`: **each replay uses fresh operands** (otherwise id reuse cannot be detected at all,
      old operands would let a stale slot "pass" just the same) + `arm_round` every round;
      `tp2_dflash_solo` (solo digest), `tp2_dflash_append` (K=7/K=5), 3× `linear_tp2_split` all PASS;
      `tp2_sessions` still fails on the **byte-identical pre-change** sequence (a pre-existing ulp non-contract, undisturbed);
      service-level injection 3000: `HTTP 503 | TP-2 allreduce stalled at rendezvous id 4294968076`
      (`= (1<<32) + 1548` ⇒ exactly channel 1's segment, proving the base really is delivered through the in-graph memcpy node) ⇒ the next request is HTTP 200;
      the same round's `decode 120.4 tok/s / dflash2 accepted 782/1,200 (65.2%)` exactly matches Phase 1's 120.0/65.2% (no performance regression);
      the watchdog dump shows both `12884914887` (channel 3) and `4611686018427390359` (eager `1<<62`+) ⇒ multiple channels and both intervals coexist without interfering.

**Phase 3 —— large-payload event path (decided by A/B)** ❌ Rejected (2026-09-23, settled by measurement)
- [x] Decided to measure: **do not write the full implementation**, because the **floor** of the candidate path already exceeds the **entire cost** of the current path. The event path and the sliced path move the same bytes
      (per-device D2H local delta + H2D peer delta, both over the same PCIe), so it suffices to first measure the copy engine ceiling for that payload shape.
      Probe: `NINFER_TP2_AR_COPY_BENCH=1 build-win/tests/ninfer_tp_device_pair_test.exe`
      (`check_copy_engine_ceiling`: chunked as planned `chunk = clamp(nbytes/4, 512 KiB, 2 MiB)`, **two copy streams per device so both directions are in flight at once**;
      the serial single-stream version measured 3.315 vs 3.316 ms bidirectional ⇒ the copy engine does not get faster from bidirectional concurrency, so that figure is a reliable floor).
      Measured: **10 MiB → 3.32 ms/call (6.33 GB/s for both directions combined); 24 MiB → 8.86 ms (5.68 GB/s)**.
      The current sliced path measured **2.70 ms** for the same 10 MiB delta (≈7.4 GB/s, 98.5% of the 2.66 ms link floor).
      ⇒ Even before accounting for event fences (about 80 µs/call fixed) and the extra device round trips the add input needs, the candidate path is already ~23% slower:
      "prefill throughput ≥ current sliced path" cannot be satisfied. The second condition (removing the 8-slice write-order chain) likewise fails to hold — the write-order chain is exactly why the current path
      reaches the link floor, and removing it only makes things slower.
      Scope note: the prefill per-layer payload is pinned at 10 MiB by `--prefill-chunk 1024`, independent of prompt length, so a micro-measurement by payload shape suffices to decide
      (the current path is already within 1.5% of the link floor ⇒ end-to-end there cannot be a >1.5% gain, while the candidate is 23% worse at the transport layer alone).
- [x] Conclusion: **"otherwise do nothing"** per the plan — do not implement the event path, do not touch the sliced path or the write-order chain. If the platform changes later (PCIe bidirectional bandwidth more asymmetric,
      or a machine where SM access to mapped host is slower), this probe can be rerun to recheck.

**Phase 4 —— partial merge feasibility audit** ✅ Audit closed (2026-09-23): **the shape does not exist, the threshold is not triggered, do nothing**
- [x] Audit scope: 9 model-side call sites (`text.cpp:409/475/822/893/950/952/2288`, `text.h:496/509`). Criterion: whether there exist
      **two independent partials each reduced and then fed into the same elementwise ADD** — only that shape can be compressed into "add the two local partials first → one collective → one add".
      Site-by-site classification:
      | Call site | Actual shape | Mergeable |
      |---|---|---|
      | `text.h:496` mixer delta → residual_add | row-parallel partial → one allreduce → `x += delta` | **No**: the immediately following MLP (`text.h:505`) reads the **in-place updated** `x`, the two adds are strictly dependent |
      | `text.h:509` MLP delta → residual_add | same as above | **No**: same as above; the next layer's mixer again depends on this layer's result |
      | `text.cpp:409 / 475 / 822 / 893 / 2288` | one side **zero-padded**, the other a local data copy ⇒ actually replicate/gather (add-over-zeros) | **No**: no two independent partials exist; and `__hadd(x,0)==x` holds bitwise, so these were never partial-sum reduction points to begin with |
      | `text.cpp:950 / 952` | two back-to-back `sendrecv` (ids + scores) | **No**: the result is a top-k **selection merge**, not an elementwise ADD |
- [x] Conclusion: **the shape does not exist** ⇒ per the "evaluate only if present" threshold it does not enter evaluation, **the FP64 oracle is not triggered**, no implementation change. Incidental records (audit by-products, all judged as do-nothing):
      - Changing the 5 replicate-class sites to `sendrecv` is bitwise equivalent (peer is all zeros), but the cost is dominated by bytes moved and the in-kernel add is nearly free ⇒ no measurable gain.
      - Merging `950/952` into one call would require packing two dtypes, I32/FP32; what is saved is only one handshake (tens of µs per round / 37 ms round time ⇒ ~0.1%).
      - Corollary: of the roughly 128 collectives per decode round, 96 are 48 layers × 2 **true** partial-sum reductions, and every result has a consumer depending on it
        ⇒ **irreducible without changing the math**; this also explains why the transport optimization space is only in the cost of each collective (Phases 1–3 have closed it off).

**Progress**

- [x] **Phase 1a transport layer (2026-09-23)**: `device_pair.{h,cu}` gains a `%globaltimer` deadline (`NINFER_TP2_AR_TIMEOUT_MS`, default 2000, 0=off), a mapped `stall` flag per side,
      cooperative exit (thread 0 publishes via shared memory, the whole block returns together, avoiding `__syncthreads` deadlock), `ar_stalled()/ar_token_skew()/clear_ar_stall()`
      (reset = take the larger of the two counters + clear arrival/order slots), and a test fault injection `set_ar_fault_skip_peer_call()`.
      New case `tests/test_tp_device_pair.cpp::check_ar_timeout` (skipping one peer launch) measured: **gave up after 2000 ms, token skew 1 →
      bitwise correct after reset**, `ninfer_tp_device_pair_test` PASS (all other allreduce/sendrecv/graph replay/move cases pass).
      ⚠️ **1a cannot ship alone**: the timeout kernel returns before phase 3, the caller only gets the local partial sum ⇒ it would turn a "loud hang" into a "silent wrong answer"; it was completed in the same batch as 1b.
      Also fixed a serious defect in 1a itself: the kernel entry first reads the trip flag on **both** sides and exits immediately if already tripped — otherwise after the first timeout the token stays permanently skewed, and each of the ~128 subsequent reductions in the same round would burn 2 s (≈4 minutes/round) before host detection.
- [x] **Phase 1b engine layer (2026-09-23)**: `TP2GenerationCore::abort_if_ar_stalled()` (`tp2_generation_core.cpp`, declared in `tp2_generation_core.h`) checks at **each round's convergence point**
      (before `request.budget.remaining()`/output policy, before any token is delivered to the client) and after sampling the first prefill token; the trigger condition uses **only the trip flag** —
      the plain decode path only drains shard A, and a mirrored shard lagging 1~2 reductions behind is normal, so judging by token skew would false-positive a 503; the skew is only written into the error message as evidence;
      after triggering, drain both cards → `clear_ar_stall()` → `invalidate_host_checkpoints()` + `session_invalidate_active()` → `RequestError(Unavailable)` ⇒ **HTTP 503 + the next request re-prefills**.
      Fault injection entry: `NINFER_TP2_AR_FAULT_SKIP_PEER_CALL=<n>` (env, read at construction; for diagnosis).
      **Service-level acceptance** (the new build was deployed to `C:\ninfer\ninfer-serve.exe`, hash `001446D4…` at acceptance; injection 3000): `[ar-watch] calls=3114 … tok=[34054,34053]` (skew 1)
      → `WARN req#2 failed during generation | HTTP 503 | service unavailable` → error body `TP-2 allreduce stalled (token skew 1)…`
      → after reset, dump `tok=[34054,34054]`, arrival/order all 0 → **the next two requests got HTTP 200 within 1 s**. Phase 1 complete.
      Lesson recorded: the original text of this acceptance survives only in this section and that session — `C:\ninfer\serve-win.log` was overwritten by the subsequent clean restart; future acceptances should use `--request-log-jsonl` first or copy the log first.
      **Regression matrix** (artifact `<models-dir>/qwen3_8_27b_w4a4_dflash2_draftall.ninfer`; every item was run twice, "after the change / git-stash to pre-change HEAD"):
      `ninfer_tp_device_pair_test` (including the new injection case) **PASS**; `ninfer_qwen3_5_tp2_dflash_solo_test` (solo digest) **PASS**; `ninfer_qwen3_5_tp2_dflash_append_test` (K=7/K=5 hashes) **PASS**; 3× `ninfer_linear_tp2_split_*` **PASS**;
      `tp2_sessions_test` (plain recall sequence divergence), `dflash2_real_test` (1.7 s cudaMalloc OOM), `dflash_real_test` (0xc0000409 fast fail) — these three **reproduce byte-for-byte / with the same code on pre-change HEAD** ⇒ pre-existing issues unrelated to this change
      (sessions is the ulp non-contract noted in the worklog as "a hot engine replaying from zero is not guaranteed to match the cold oracle token by token"; the other two are artifact/environment class, and the docs do not list these two case names).
- [x] **Phase 2 (2026-09-23)**: see the Phase 2 checklist above (id authority-ization complete, committed as `de74d966`/`b9c5d11c`/`634df875`).
- [x] **Phase 3 (2026-09-23)**: see the Phase 3 checklist above — **measured rejection, do nothing** (copy engine floor 3.32 ms vs current path 2.70 ms).
- [x] **Phase 4 (2026-09-23)**: see the Phase 4 checklist above — **the shape does not exist, threshold not triggered, do nothing** (9 call sites classified individually, FP64 oracle not triggered).
      **Deployment note**: `build-win/apps/ninfer-serve.exe` is **not** synchronized automatically to `C:\ninfer\`, it must be copied manually; the PE timestamp makes the hash differ on every link, so compare by "the deployed binary and the build-win output have equal `Get-FileHash`" (this round was synchronized, `108AD767…`).
- [x] Killed the stuck process and archived the log (2026-09-23; device memory cleared immediately; the residual spinning kernel subsided on its own after about 2 minutes)
- [x] Pre-measurement confirmation: both cards' idle usage already back to zero (2026-09-23 21:49, no restart needed)
### 3.9 DFlash2 DraftTokens (K) sweep (2026-09-23; conclusion: **keep K=5**)

**Question**: what `--draft-tokens` value is optimal for `qwen3_8_27b_w4a4_dflash2_draftall.ninfer` on the MTP + DFlash2 route.
**Method**: `tools/tp_bootstrap/r59_draft_tokens_sweep.ps1` — for each K, start ninfer-serve with the **exact same production parameters**
(245760 / k8v4 / `--spec dflash2` / `--lm-head-draft` / temp 0.7 / top-k 20 / top-p 0.8 / vision / preserve-thinking),
changing only `--draft-tokens`; for each K, warm up once first, then run 5 times each on 3 prompt classes (reason reasoning chain / prose descriptive text / code code), `max_tokens=256`.
Metrics come from the request log's `decode X tok/s` and `dflash2 accepted a/b`; tokens/round = output ÷ (drafted/K), rounds/s = decode ÷ tokens/round.

| K | decode tok/s | acceptance rate | tokens/round | rounds/s | reason | prose | code |
|---|---|---|---|---|---|---|---|
| 3 | 83.8 | 58.4% | 2.78 | 30.2 | 93.4 | 76.0 | 82.2 |
| **5** | **96.4** | 48.5% | 3.45 | 27.9 | 113.9 | 81.6 | **93.6** |
| 7 | 96.1 | 36.9% | 3.62 | 26.6 | **117.8** | **83.2** | 87.4 |
| 9 | 89.1 | 30.1% | 3.74 | 23.8 | 113.1 | 72.9 | 81.3 |

(The remaining values of the 8-point coarse sweep: K=1 → 60.3, K=11 → 84.6, K=13 → 85.4, K=15 → 81.8 tok/s, all below the K=3~7 plateau.)

**K=6 supplementary test + interleaved recheck** (5/6/7 started in rotation for 3 rounds, 18 samples per K in total; comparing within the same round cancels machine state drift — both K=5/7 rose from c1 to c3 as the machine warmed up):

| K | pooled decode | acceptance rate | tokens/round | rounds/s (three rounds) | reasoning chain | descriptive text | code |
|---|---|---|---|---|---|---|---|
| 5 | 93.3 | 46.3% | 3.34 | **27.9 / 28.0 / 27.9** | 112.1 | 77.4 | **90.4** |
| 6 | 89.7 | 40.9% | 3.48 | **25.7 / 25.7 / 25.7** | 111.8 | 75.7 | 81.6 |
| 7 | **96.9** | 37.4% | 3.65 | 26.6 / 26.4 / 26.6 | **123.2** | **78.2** | 89.4 |

Same-round pairing (c1/c2/c3): K=5 88.1 / 93.8 / 98.0, K=6 87.6 / 94.0 / 87.5, K=7 94.4 / 96.8 / 99.6 ⇒ **K=7 is no lower than K=5 in all three rounds (+7% / +3% / +2%, the gap narrowing as the machine warms up)**, and K=6 is below K=7 in all three rounds.

**K=6 is a reproducible per-round cost dip**: its rounds/s is 25.7 (±0.0) in all three rounds, below both the narrower-window K=5 (27.9) and the wider-window K=7 (26.6); acceptance rate and tokens/round both sit normally in between the two ⇒ it is not noise, and it is not in the startup ledger (`[mem]`'s `record` grows monotonically with K, 5.1/6.0/6.8 MiB, and `[tp2-graph]` is identical across the three K). Suspected kernel boundary effect from bucketing/alignment by window width, **to be investigated** (clue: `NINFER_TP2_TIMING=1` to get per-phase per-round timing).

**Mechanism**: increasing K raises tokens/round (2.78 → 3.74), but the acceptance rate decays rapidly with position (58% → 30%), while the wider window makes each round more expensive
(rounds/s 30.2 → 23.8) ⇒ the net gain tops out at K=5~7.
**Conclusion**: **K=5 and K=7 differ by only a few percentage points, with K=7 slightly better**; **do not use K=6** (dominated by both neighbors); K≤3 and K≥9 are clearly worse.
- In the interleaved pairing K=7 is ≥ K=5 in all three rounds (+7% / +3% / +2%, narrowing to ~+1.6% after warm-up), and the driver is **reasoning chain +10%** (123.2 vs 112.1);
  code/tool-call class is the reverse, with K=5 slightly higher (90.4 vs 89.4), and descriptive text is nearly identical.
- In the single-round coarse measurement the two are tied at 96.4 vs 96.1 ⇒ the difference is ≤ a few percentage points, within sample fluctuation; a decision would need a longer paired measurement.
- Leaning: customer traffic is dominated by thinking ⇒ **recommend changing `--draft-tokens` from 5 to 7** (which is also the binary's default value of 7 for dflash2, consistent with the `append K=7` golden value);
  if the share of tool calls rises, the gap between 5 and 7 will shrink further.
**Boundaries**: single machine, single artifact, 3 prompt classes, temp 0.7, 256-token generation; the acceptance rate is strongly content-dependent, and the conclusion may move with a different corpus.
Device memory is not a differentiator (K=3→9 differs by only ~10 MiB free per card; K=15 also starts normally at 245760).
### 3.10 K sweep for the pure MTP backend (`--spec mtp`) (2026-09-23; conclusion: **K=3/4 tied, K=5 clearly worse; still use DFlash2**)

**Method**: the same production recipe and interleaved method as §3.9 (3 rounds × K=3/4/5 × 18 samples per K), only changing `--spec` to `mtp`:

| K | decode tok/s | acceptance rate | tokens/round | rounds/s (three rounds) | reasoning chain | descriptive text | code |
|---|---|---|---|---|---|---|---|
| 3 | 79.4 | 52.7% | 2.58 | **30.8 / 30.8 / 30.8** | 88.5 | **73.8** | **75.9** |
| 4 | **79.7** | 45.0% | 2.80 | 28.5 / 28.5 / 28.5 | **92.4** | 71.0 | 75.7 |
| 5 | 74.5 | 37.2% | 2.86 | 26.1 / 26.1 / 26.1 | 88.3 | 67.7 | 67.5 |

**Conclusion**: MTP's optimum falls at **K=3~4 (79.4 vs 79.7, a statistical tie)**, with K=5 6.5% lower; by class, K=4 is better on long reasoning chains (92.4 vs 88.5), K=3 is better on descriptive text (73.8 vs 71.0), and code is a tie.
But **at the same K, DFlash2 is uniformly ahead**: at K=5, 74.5 vs 93.3 (-20%), and the gap widens with K (MTP's tokens/round rises only slightly, 2.58→2.86, while DFlash2 goes 3.34→3.65),
plus MTP's K=5 cap leaves no further gain to trade for ⇒ **keep using `--spec dflash2`**.
**Incidental clarification**: the startup script's comment on lines 60-61, "TP-2 only supports mtp (dflash/dflash2 are rejected)", **is out of date** — both this round and §3.9 ran dflash2 successfully on TP-2.
**Methodological corroboration**: each backend's rounds/s varies by only ±0.1 across its three rounds (MTP 30.8 / 28.5 / 26.1; DFlash2 27.9 / 25.7 / 26.6),
showing that "per-round cost" is a deterministic function of K, while decode round-to-round fluctuation comes from acceptance rate/content; interleaved pairing is the correct way to make this kind of comparison.

### 3.11 prefill comparison: DFlash2 K=7 / MTP K=3 / no speculation (2026-09-23; conclusion: **speculation costs almost no prefill overhead, DFlash2 is ~3-4% slower**)

**Method**: `tools/tp_bootstrap/r60_prefill_spec_compare.ps1` — the three configurations use the **same production parameters**, differing only in the spec-related switches
(`--spec dflash2 --draft-tokens 7 --lm-head-draft` / `--spec mtp --draft-tokens 3 --lm-head-draft` / all three removed);
each configuration gets a **new process** (no cross-configuration reuse), the two lengths use **different text**, and the log confirms `cache 0%` (the 0.6% for the 6481 one is the shared chat template prefix, ~39 tok).
To rule out the "later in the same round runs hotter" ordering artifact, besides 2 forward-order rounds (dflash2→mtp→plain) one **reverse-order** round was run (plain→mtp→dflash2).

| prompt | DFlash2 K=7 | MTP K=3 | no speculation | TTFT (dflash2 / mtp / plain) |
|---|---|---|---|---|
| ~1,673 tok | 1,473 tok/s | **1,593** | 1,567 | 1.10 / 1.05 / 1.10 s |
| ~6,481 tok | 1,743 tok/s | 1,807 | **1,817** | 3.70 / 3.60 / 3.55 s |

(3 measurements per group, within-configuration repeatability ±1%; in the reverse-order round dflash2 ran last and was still the lowest ⇒ the ordering artifact is ruled out. The rate rising with prompt length = the per-request fixed overhead being amortized.)

**Conclusion**: enabling speculation **costs almost no prefill overhead** (within 1%), MTP and no speculation are essentially tied; **DFlash2 is consistently 3-4% slower** (still ~3% after deducting the 0.6% prefix hit difference).
On the whole request it is negligible: on the decode side DFlash2 leads MTP by ~20% (§3.10), far exceeding this 3-4% ⇒ **production keeps DFlash2 (K=5 or 7)**.
**Not localized**: the exact source of DFlash2's 3-4% (no per-phase attribution was done; to investigate, use `NINFER_TP2_TIMING=1` to break down prefill per-phase time).

### 3.12 DFlash2 K=7 prefill depth curve (0 → 239k; 2026-09-23)

**Method**: `tools/tp_bootstrap/r61_prefill_depth_curve.ps1` — a single process, a single document grown step by step (each request sends segments 1..i),
relying on prefix reuse so each step **prefills only the newly added segment**, hence the N in the log's `prefill X tok/s (N tok)` is the "number of newly added tokens" (verified `N = prompt − cache`),
i.e. **the instantaneous rate within that depth interval**. K=7 / dflash2 / production parameters, 17 steps in total, ~14.1k tokens per step, max_tokens=1 to isolate prefill.

| Depth interval | Rate | Fit | Cumulative time |
|---|---|---|---|
| 0-14,116 | 1,810 | 1,785 | 7.8 s |
| 13,312-28,180 | 1,610 | 1,590 | 17.0 s |
| 27,648-42,244 | 1,440 | 1,428 | 27.1 s |
| 41,984-56,308 | 1,310 | 1,297 | 38.1 s |
| 55,296-70,372 | 1,200 | 1,191 | 50.7 s |
| 69,632-84,436 | 1,090 | 1,098 | 64.3 s |
| 83,968-98,500 | 1,000 | 1,018 | 78.8 s |
| 98,304-112,564 | 931 | 949 | 94.1 s |
| 111,616-126,628 | 908 | 891 | 110.6 s |
| 125,952-140,693 | 822 | 838 | 128.5 s |
| 140,288-154,758 | 797 | 791 | 146.7 s |
| 154,624-168,823 | 738 | 749 | 166.0 s |
| 167,936-182,888 | 700 | 712 | 187.4 s |
| 182,272-196,953 | 684 | 678 | 208.9 s |
| 196,608-211,018 | 653 | 647 | 231.0 s |
| 210,944-225,083 | 622 | 618 | 253.7 s |
| 224,256-239,148 | 597 | 593 | 278.7 s |

**Fit**: per-token cost `1/rate = 5.25e-4 + 5.01e-9 · depth` seconds (17 points, all errors <3%)
⇒ extrapolating to depth 0 gives **1,905 tok/s**, at 245,760 gives **569 tok/s**; **at full context attention already accounts for 70% of the per-token cost** (shallow depths are dominated by fixed overhead).
**Practical conclusion**: filling a 239k context actually takes **278.7 s** (of which 248,072 tokens are truly computed — 3.7% more than 239,148, tail recomputation caused by the 1024-chunk checkpoint granularity),
an effective rate of 890 tok/s (first segment 1,810 → last segment 597, a 3.0× drop).
**Reproduction and corroboration**: an independent rerun matches the first 16 points point by point (±1%); the `cache` figures show prefix reuse is still nearly perfect at 239k depth (the cache boundary = the previous prompt minus ~1 chunk),
i.e. the context cache ring is usable at full depth.

### 3.13 Cost attribution and optimization headroom for deep-context prefill (2026-09-23; analysis, no code changed)

**Question**: §3.12 measured prefill rate 1,905 → 569 tok/s (3.3×) — is there optimization headroom.

**Architectural fact** (`<models-dir>/W4A16/NVFP4/W4A4/config.json`): Qwen3.5 is **hybrid attention** — of 64 layers only **16 are full attention**
(24 Q / 4 KV / head_dim 256; TP-2 single card 12 Q / 2 KV), while the other **48 layers are linear attention (GDN) whose cost is depth-independent**.
⇒ The entire slope comes from 16/64 layers alone; if all 64 layers were full attention, at the same depth it would be ≈185 tok/s (currently 569, the hybrid structure has already bought 3.1×).

**Measured decomposition** (§3.12 fit): per-token cost = **0.525 ms (depth-independent, 1,905 tok/s ceiling) + 5.012e-9·d**;
at 245,760 the two account for **30% / 70%** respectively ⇒ that 70% is the **gain budget for "any attention-side optimization"**: a perfect kernel gives at most 3.3×, realistically about 1.3–1.8×.

**Which kernel production prefill takes** (verified): `causal_softmax_attention.cpp:382-385` only gives ChunkedSmallT for `(q_heads==12||16) && width<=16`,
so a prefill of width 1024 falls to the **Prompt** route ⇒ `causal_attention_prompt_k8v4_kernel` (flash-style: Br=64 rows, Bc=64 KV columns, 16 warps,
FP8 tensor cores for QK^T, FP16 for PV), but **grid = (tokens/64, 12 Q heads)**: **each block processes only one Q head**, while a single card has only 2 KV heads
⇒ 6 Q heads share 1 KV head yet each traverses the same KV segment independently (6× request-level redundancy; L2 may absorb part of it, needs profiling to decide).

**Estimated traffic and efficiency** (anchored on the engine's own `[mem]` ledger: kv 3015 MiB/shard @245760 ⇒ 12.87 KB/token/shard,
of which the full-attention part is 16 layers×2 KV heads×256 dim×1.5 B = 12,288 B ✓):
at 245,760 depth and a 1024-token chunk, KV traffic ≈ 16 row tiles × 12 Q heads × 384 B × 245,760 ≈ **290 GB/chunk/card**;
measured attention time ≈1.26 s/chunk ⇒ **~230 GB/s (51% of peak)**; attention compute ≈4.95e13 FLOP/chunk ⇒ **~39 TFLOP/s (~40% of the FP8 roofline)**.
⇒ Neither roofline is saturated, and there is 6× request-level redundancy ⇒ there is real headroom.

**Optimization list (ordered by gain; all estimates, to be settled by profiling)**:
1. **GQA-aware prompt kernel**: let Q heads sharing the same KV head share one KV data stream (e.g. 6 heads × 16 rows/block), traffic 290 GB→48 GB/chunk;
   if it becomes bandwidth-bound (~400 GB/s) ⇒ attention 0.12 ms/token, end-to-end at 245k could reach **~1,550 tok/s (2.7×)**;
   if stopped by the FP8 compute floor (~60–70% roofline) ⇒ more realistically **~830–1,000 tok/s (1.5–1.8×)**. About 1.3× at 50k, about 1.05× at 10k.
2. **Raise the bandwidth/occupancy the Prompt kernel already achieves** (KV tile double buffering, prefetch, occupancy): at most ~2× on the attention term.
3. **Generality note**: the gain is concentrated in long contexts; **the short-prompt bottleneck is the 0.525 ms/token fixed term** (48 linear attention layers + MLP + W4A4 GEMM), which is a separate line of work.
4. **Not levers**: a smaller KV dtype (already 1.5 B/element, and with a quality cost), TP splitting, the GQA ratio (already 4 KV heads), the chunk cap (`kPrefillChunkMaximum=1024` is the established TP-2 design).

**Next step (recommended first)**: have `ncu` capture `causal_attention_prompt_k8v4_kernel` (at depth ≥100k) to look at **DRAM throughput / L2 hit rate / tensor pipe utilization**
— from that decide whether the 6× redundancy really reaches DRAM (determining whether to do "reduce traffic" or "raise occupancy"). Note that the worklog has already fixed a same-family 12-head routing hole (covering only the width≤16 verify window);
the large-chunk one is a leftover.

**Addendum: the PCIe topology (Gen5 x8 + Gen4 x4) has been considered, and is the main cause of the "depth-independent term" (2026-09-23 cross-validation)**

worklog Round 35b/35c (`worklog Round 35b-35c`) already measured this machine's topology and localized it:

- Card 0 (`pci 01:00.0`) is **Gen5 x8, measured ~20 GB/s**; card 1 (CUDA device 1, `pci 08:00.0`) is **chipset Gen4 x4, measured ~7 GB/s** (`nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max` reproduces: device 0 = 5/x8, device 2 = 4/x4).
- nsys attribution: `ar_inplace_bf16` takes **48%** of total kernel time (about 60% within 1024-wide blocks); 2 allreduces per layer, payload [hidden 5120, T] bf16, write+read round trip = 4·hidden·T bytes/layer/card
  ⇒ **2.6 MB/token**, at 7 GB/s = **0.368 ms/token ≈ 2.67k tok/s AR hardware ceiling** (the measured marginal slope lands exactly on it, independent of block width).
- Key disproof: this Gen4 x4 (via chipset) link has **only ~7.9 GB/s in both directions combined** (not per direction); the 21 MB round trip has a physical floor of 2.66 ms, and 8-slice ordered pipelining already achieves **2.704 ms = 98.5%**.
  (Phase 3 on this machine independently reproduces the same number: 10 MiB payload 2.70 ms ✓) ⇒ **there is no software headroom left on the AR side**; multi-block AR makes no difference, and copy-engine rendezvous is slower and gives inconsistent results — both measured and rejected.

**Cross-validation with this section's fit**: my measured depth-independent term `a = 0.525 ms/token` (1024 block = 538 ms) **agrees** with the worklog's nsys attribution at ~1.3k depth (AR ~350 + MMA ~188 + other ~35 = 573 ms)
⇒ the composition of that fixed term is **AR ~65% + MMA ~34% + other ~6%**. Note that worklog run was only ~1.3k deep, so its attribution naturally covers only the depth-independent part and does not conflict with this section's attention term.

**This corrects §3.13's wording**: the fixed term is **not** "another optimizable compute line" but **PCIe overhead already pinned to the link floor**.
The shallow-depth ceiling 1,905 tok/s = 1/(0.34 + 0.18 + 0.03) ms, of which 0.34 ms/token has no software headroom left. Two levers:

| Lever | Scope | Gain | Cost/Risk |
|---|---|---|---|
| Move card 1 to a CPU-attached Gen5 x8 slot | Hardware | shallow/medium context **1,905 → ~3,390 tok/s (1.78×)**; at 245k 569→655 (1.15×); at 14k 1,810→2,740 (1.51×) | Zero code; needs a free motherboard slot |
| Overlap AR with MMA (ring staging + event sync + sub-block pipelining) | Software | ceiling ~1.5× (shallow ceiling ~4,760 tok/s once AR is hidden) | Large refactor, cross-card rendezvous deadlock risk |
| GQA-aware prompt kernel (see above) | Software | deep context 1.5–1.8× | Kernel engineering |

The three are complementary: hardware buys the shallow layers, the kernel buys the deep layers. The root cause of P2P being unavailable is also this slot — `canAccessPeer=0` (worklog:200), and `nvidia-smi topo` shows the two cards as **SYS** (different root complexes), so peer traffic must cross the host bridge, which is also the origin of the host-staging design (`device_pair.h:114`).

### 3.14 Two software levers pending implementation (2026-09-23; L1 has started)

Per §3.13's accounting, the two prefill cost terms each have a lever, and they are **complementary** (one acts on `b·d`, one on `a`):

| # | Lever | Term | Estimated gain | Prerequisite/Risk |
|---|---|---|---|---|
| **L1** | **GQA-aware prompt kernel**: let the 6 Q heads sharing the same KV head share one KV data stream (currently `grid=(tokens/64, QHeads=12)`, each block processes 1 Q head, and a single card's 2 KV heads are each read 6 times) | `b·d`: deep-context attention, **70%** at 245k | deep context **1.5–1.8×** (245k: 569 → ~830–1,000 tok/s) | First pass the measurement gate to confirm whether the 6× redundancy really reaches DRAM; only scheduling and data reuse change, **the physical accumulation order is untouched ⇒ can be bitwise identical to the current kernel**. **(2026-09-23 measurement gate disproved it ⇒ cancelled, see below)** |
| **L2** | **Overlap AR with MMA** (ring staging + event sync + sub-block pipelining) | `a`: the depth-independent term, of which AR ≈ 65% | shallow ceiling **~1.5×** (1,905 → ~4,760 tok/s ceiling) | Cross-card rendezvous deadlock risk, a large refactor (already assessed in worklog Round 35c) |

> The hardware lever (moving card 1 from the chipset Gen4 x4 slot to a CPU-attached Gen5 x8) is not in the table: zero code but needs a motherboard slot, gain 1.78× (shallow/medium context), stackable with L1/L2.

**L1 execution order**:
1. **Measurement gate (do first; if it fails, do not touch the code)**: have ncu capture `causal_attention_prompt_k8v4_kernel` (production geometry 12q/2kv/d256/k8v4, KV depth ≥100k), look at
   `dram__bytes_read.sum`, `lts__t_sector_hit_rate.pct`, tensor pipe utilization, achieved occupancy.
   - High L2 hit rate and DRAM already near peak ⇒ the 6× redundancy is absorbed by L2 ⇒ the lever shrinks, switch to "raise occupancy/lower latency";
   - DRAM traffic ≈ 6× unique KV bytes and throughput near peak ⇒ the redundancy is real ⇒ proceed to step 2.
2. **Design**: change the block's row dimension from "1 head × 64 rows" to "6 heads × R rows" (R derived backward from the smem budget; the Q/K/V/P/stats items are already accounted for in `prompt_k8v4.cuh`),
   the KV tile still streams in order with Bc=64; **the KV tile order and accumulation order seen by each row is unchanged** ⇒ the goal is bitwise identity with the current kernel.
3. **Implementation**: change only `prompt_k8v4` (the production dtype); the prompt kernels of the other four dtypes keep the current path for now (isomorphic, to be synchronized later).
4. **Verification**: Op-level FP32/FP64 oracle (12/2/d256 deep envelope) + synchronize `causal_attention_workspace_capacity_bytes` + model-level golden values (`tp2_dflash_solo`/`append` digests) + rerun the §3.12 depth curve as the performance comparison.

**L1 measurement gate conclusion (2026-09-23, disproved ⇒ do not change the kernel)**: using the existing Op-level benchmark `bench/ops/causal_softmax_attention_bench` (production shard geometry `d256-h12-kv2` was added to it this time — the Op and tests always supported it, the bench lagged) measured at W=1024 (Prompt route), h12-kv2, 245,760 depth single layer:

| KV dtype | Time | Achieved math | Unique KV bytes |
|---|---|---|---|
| bf16 | 82,927 µs | 37.4 TFLOP/s | **482.0 MB** |
| int8 | 83,207 µs | 37.2 | 248.5 MB |
| fp8 | **73,782 µs** | 42.0 | 242.9 MB |
| nvfp4 | 82,270 µs | 37.7 | 135.6 MB |
| k8v4 (production) | 73,910 µs | 41.9 | 189.2 MB |

**KV bytes span 3.5×, yet elapsed time differs by only 13%**: fp8 has 28% more bytes than k8v4 but takes the same time; nvfp4 has 28% fewer bytes yet is 11% slower; bf16 has 2.55× the bytes and is only 12% slower
⇒ **the kernel is not KV-bandwidth-bound but compute/ALU-bound** ⇒ removing the 6× request redundancy yields no time gain, and **L1 is cancelled as originally designed** (kernel code untouched, consistent with the gating agreement).

**Ceiling established along the way**: on the same machine at T=1024, `ninfer_fp8_linear_add_bench` (A8, K=17408) **85.9 TFLOP/s**, `ninfer_q4_linear_swiglu_bench` (W4A4) **44.1 TFLOP/s**;
attention's QK^T and PV are each about 21 TFLOP/s (41.9 combined). Compared with a pure FP8 GEMM, each attention GEMM reaches only 24%,
but k8v4 and fp8 taking the same time shows **the gap is not only in MMA dtype** but in kernel structure (k8v4's PV goes through FP16, Hadamard rotation/row scaling, softmax, smem traffic).
⇒ If attention is to be attacked anyway, the only form is **compute-path efficiency** (FP8 PV + fewer ALU passes), and **a profiler is a prerequisite**: this machine **has no Nsight Compute (ncu) installed**, and black-box timing has reached its limit.
Expectation: attention fully eliminated ⇒ 569 → ~1,900 tok/s at 245k (3.3× upper bound); realistically halved ⇒ ~1.4×.

**Baseline on record** (W=1024, h12-kv2, k8v4, single-layer median): 8,192 → 2,637 µs; 32,768 → 9,924; 65,536 → 19,630; 131,072 → 39,602; 245,760 → 73,880.
Linearity is excellent, and ×16 layers = 1.18 s/chunk, only 6% off the §3.12 model-level 1.26 s/chunk ⇒ two independent measurements corroborate each other.
Re-run: `ninfer_causal_softmax_attention_bench.exe --entry cached --geometry d256-h12-kv2 --kv-dtype k8v4 --tokens 1024 --context 8192,32768,65536,131072,245760 --execution eager --cache cold`.
**Build note**: `-DNINFER_BUILD_BENCHMARKS=ON` makes the default target fail on Windows (`bench/context_cost/model_context_fixture.cpp:400` uses `__int128`, which MSVC does not support),
so this time only the target was built (`cmake --build build-win --target ninfer_causal_softmax_attention_bench`); that cache switch has been reset to OFF and does not affect normal builds.

**L1 follow-up (2026-09-23): profiler is ready, but blocked by driver permissions**

- Nsight Compute 2025.4.1 installed (winget `Nvidia.Nsight.Compute`), `ncu --version` is normal; but non-admin processes are denied reading performance counters:
  `ERR_NVGPUCTRPERM` (target device 0). Two options to unlock: ① run ncu from an **administrator** PowerShell;
  ② NVIDIA Control Panel → Desktop → Enable Developer Settings → Developer → Manage GPU Performance Counters → allow all users access → Apply (persistent).
- Attribution obtainable without counters (completed):
  * **W sweep** (depth 65,536, k8v4, h12-kv2): W=256 → 31.7, 512 → 41.8, 1024 → 41.8, 2048 → 45.4 TFLOP/s
    ⇒ **W=1024 is already in the plateau region**; raising the prefill chunk cap from 1024 to 2048 is worth only ~9%, not a lever.
  * **Same-machine GEMM ceiling** (T=1024): FP8 A8 **85.9**, W4A4 **44.1** TFLOP/s; attention plateaus around 45 (QK+PV combined),
    while QK^T / PV are each about 21–23 ⇒ QK^T(FP8) ≈ 25% of its dtype ceiling, PV(FP16) ≈ 50%.
  * The 4-bit K path (nvfp4-K vs fp8-K) costs +11%; bf16 (no dequantization at all) is instead 12% slower ⇒ **MMA dtype dominates more than dequantization**.
- **Candidate change (not to be made until profile confirms it; gating observed)**: raise the prompt kernel's KV tile width `Bc` from 64 to 128
  (QK^T's N dimension doubles, KV loop iterations halve; smem grows from ~99 KB to ~165 KB, still fitting in a single block).

**L1 experiment record (2026-09-23): both low-cost paths are blocked dead by a hardware wall, kernel has been reverted**

ncu measurements for `causal_attention_prompt_k8v4_kernel` (W=1024, 65,536 depth, k8v4, h12-kv2):

| Metric | Value | Meaning |
|---|---|---|
| Compute (SM) Throughput | **63.54%** | compute-dominated |
| Tensor pipe | **63.5% (highest)** | tensor pipe is the busiest resource |
| **DRAM Throughput** | **1.09%** | KV traffic is essentially all absorbed by L2 |
| L2 / L1 throughput | 10.10% / 40.43% | neither is a bottleneck |
| Issue Slots Busy / IPC | 30.89% / 1.39 | not issue-limited |
| Largest stall | waiting on math pipe 4.0/11.5 cycles (34.4%) | tensor pipe queueing |
| Occupancy | theoretical = actual **33.33%** (16/48 warps) | 1 block/SM |
| Block Limit | registers 1 / smem 1 | dual limitation |
| Registers/thread · dynamic smem | **100** · 85.12 KB (configured 102.40 KB) | — |

⇒ **"cut 6× KV redundancy" is thoroughly falsified** (DRAM 1%); the real bottleneck is the tensor pipe + 33% occupancy.

Two experiments (both passing oracle correctness first):
1. **Warps 16→32** (double the warps within the same smem, register cap consequently pressed to 64): correctness **PASS**, performance **−12%** (65k: 19.63→22.17 ms; 245k: 73.88→82.43 ms).
   The cause is structural: a 1024-thread block is hard-capped at ≤64 registers/thread by the 64K register file, while the kernel needs ~100 ⇒ spilling. **Dead end**.
2. **Bc 64→128** (widening the KV tile improves the MMA shape, smem 85,120→151,808 B): **fails to launch**,
   `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` returns `cudaErrorInvalidValue` ⇒ consumer Blackwell's **per-block dynamic smem limit is ~100 KB**. **Dead end**.

**Conclusion**: this kernel has already pressed against this machine's hardware boundary (tensor pipe busiest, DRAM idle, occupancy simultaneously locked by "register file + 100 KB smem").
Everything remaining is larger in scope with limited payoff: tile restructure to Br=32/256 threads aiming for 2–4 blocks/SM (register-dominated), switching PV to FP8, compressing non-tensor FP32 instructions (11% of instructions, ncu hints ~3.6%).
Upper bound: attention term about 1.2–1.4× ⇒ 245k end-to-end ~1.1–1.3×, 50k ~1.05–1.1×.
⇒ **Cost-effectiveness ordering is now clearer: hardware slot move (shallow/mid 1.78×, zero code) > L2 (AR/MMA overlap, shallow ~1.5×) > attention restructure (deep context, smallest payoff, largest risk).**

**Incidental fix**: `bench/context_cost/model_context_fixture.cpp`'s `__int128` changed to an equivalent uint64 overflow check (neither of the two accumulator terms overflows on its own, so only their sum needs checking),
`-DNINFER_BUILD_BENCHMARKS=ON` now builds normally on Windows (verified with the default target `cmake --build build-win`).

---

## 4. DSH/agent-loop prefix reuse fix (2026-09-24, in progress)

**Problem**: in the agent loop where DSH hooks into ninfer, after some round generates a large tool call (writing a large file, 24k token),
the next round must re-prefill those 24k tokens (measured req#3: prompt 34,050 / cache 9,216 / prefill 24,834 tok / TTFT 17.0s).

**Established (measured with NINFER_TP2_REUSE_TRACE=1 / NINFER_TP2_SESSION_TRACE=1)**:
1. The engine saved the previous round's full history (`cached` = prompt+generated), but `shared` equals the previous round's **prompt** length
   (332/331/315 across three cases), i.e. **the very first generated token already does not match** → it can only restart from the end of the previous round's prompt.
2. Replaying `reasoning_content` as-is via `/v1/chat/completions` reproduces the same result → not DSH-specific.
3. The artifact's embedded template applies `|trim` to `reasoning_content`, eating the first space the model generated (measured:
   `"ABCD"` and `"  ABCD  "` render to exactly the same prompt length); and the model's first reasoning token carries a leading space,
   so no text replay can reproduce it token by token.
4. When the TP-2 path builds its own frontend it **fails to pass `chat_template_path`** (`tp2_generation_core.cpp:472-480`),
   so `--chat-template` is silently ignored on that path (measured: the developer role still reports `artifact:chat_template.jinja`).
5. In the reuse scan, "alignment boundary first" shadows a deeper unaligned live frontier, and the fallback is taken only when `reuse == 0`;
   and DFlash2's fallback carries a cost gate (dropping the draft ≈ 3× slower decode vs. the prefill saved).

**Results (2026-09-24)**:
- [x] A `tp2_generation_core.cpp`: TP-2 frontend passes through `chat_template_path`. Measured: after deploying the new binary and restarting with
      `--chat-template`, the `developer` role goes from 400 to 200 (the override truly takes effect).
- [x] C `tp2_generation_core.cpp`: the fallback boundary scan becomes an unconditional second pass, the gate switches to "marginal saving"
      (`position - reuse`), and alignment boundaries remain lowest priority. `dflash2`/`mtp` both `ninfer_qwen3_5_tp2_sessions_test` PASS.
- [x] D `tests/models/qwen3_5/test_tp2_sessions.cpp`: new resident-session consecutive-two-round case (1000-token prompt;
      round two expects plain/mtp = 1007, dflash2 = 768). Without C, plain/mtp stops at 768, so the case catches that change.
- [x] E build (`nm`/ninja, test targets + `ninfer-serve` and syncing `C:\ninfer\ninfer-serve.exe`) + tests +
      online re-test; evidence: `profiles/sessions-{dflash2,mtp,plain-trace,all}.log`, `profiles/reuse-trace*.log`.
- [x] F docs: `docs/tp2-dual-5060ti.md` gains resident consecutive-two-round, gating and `--chat-template` notes.
- [~] B template `|trim` → `rstrip('\n')`: **reverted (falsified by measurement)**. With the template replaced by a maintenance template, replaying
      `reasoning_content` as-is still yields `shared == the previous round's prompt length`, consistent with the artifact template's result.

**Residual root cause (newly found, unfixed)**: text-protocol replay cannot reproduce token by token. At `max_tokens=1`, measured
`completion_tokens=1 / reasoning_tokens=0 / reasoning_content empty`; k=2 gives `" need"`, k=3 gives
`" need answer"`, while replay always stops at `shared=72` (the previous round's prompt length). That is, **the first generated token is not in the published
reasoning text** (or the replay's re-tokenization differs from the sampling boundary), so on DFlash2 "reuse the tail of the previous answer" is unreachable for any text client.
Add the gate on top of that (dropping the draft ≈ 3× decode vs. the prefill saved): under a 32k output budget, recomputing 24.8k tokens (17 s)
is the **rational choice** under the cost model.

**Optional levers**: client output budget (`remaining` must be < saving/16 ≈ 1.5k tokens to trigger reuse); let DFlash2 keep the draft
under unaligned reuse (requires first solving the reproducibility of the `extent=0` clamped round); or take a continuation protocol carrying token ids.

**Pre-existing failure (unrelated to this change)**: `NINFER_TEST_ROUTE=plain` at `shared_a_first` with `reuse=0/src=none`
differs from the cold oracle's first token in the from-scratch walk (`[2752 13 198 …]` vs `[365 2798 349 …]`). The trace proves that scenario and
all its preceding reuse decisions (0/71/512) match the test expectations; the new scan path cannot pick any boundary when `shared_prefix=0`.

## 5. TP-2 first generated token not published (2026-09-24, fixed)

**Symptom**: at `max_tokens=1`, `completion_tokens=1 / reasoning_tokens=0 / no text`; `max_tokens=k` publishes text for only k−1 tokens;
when replaying the previous round's answer byte by byte (chat-completions returning `reasoning_content`+`content` as-is, with/without tools, different templates — all the same),
`shared` always equals the previous round's prompt length (measured 67 / 306 / 72). So on TP-2, "reuse the tail of the previous answer" is unreachable for any text client,
and `cache` in the serving log is always `floor(previous round prompt/1024)×1024`.

**Root cause**: at the end of prefill in `src/runtime/engine/tp2_generation_core.cpp`, `request.generated.push_back(first)` records the token directly,
**without going through the output session** (`preview_model`/`commit_preview`). The single-GPU path commits the first token sampled in prefill via
`commit_pending`→`preview_model` (`src/runtime/engine/engine_core.h`), and TP-2 is the only path that missed it; therefore
`generated` (the sequence indexed by the KV/state/session directory) has one more token than the "published text".

**Fix**: after prefill sampling, `preview_model({first}, budget.remaining(), limit_reason())` → `budget.commit` →
`publish_preview(false)`; if that decision already terminated the request (stop token or exhausted budget), record it in `first_token_finish` and skip the decode loop.
This also incidentally fixes "the engine keeps generating when the first token is a stop token" and the first token not being subject to grammar constraints (the former is fixed, the latter recorded as a separate leftover item).

**Verification** (evidence: the `[tp2-reuse]` trace in `C:\ninfer\serve-win.log`, `profiles/fixtok-*.log`):
- `max_tokens=1` → `reasoning_tokens=1`, `R="We"` (0 / empty before the fix).
- Replaying a 40-token answer: `shared=107` (= 67 prompt + 40 generated; 67 before the fix).
- Replaying a 1400-token answer: `reuse=1470 src=live` = the entire previous round's answer (DFlash2 declines the draft for that 1-token request per the gate, negligible cost).
- `ninfer_qwen3_5_tp2_sessions_test`: dflash2 / mtp both PASS (no regression in existing assertions).
- New case `check_first_token_published`: with thinking on and an output budget of 1 token, `reasoning_tokens` must be 1.

**Residual (unfixed)**: the gate still decides whether to take an unaligned boundary by `max(reuse_grid, 16 × remaining budget)`, so under DSH's 32,768 budget a long answer is still recomputed in full
(the rational choice under the cost model); prefill's first sample still does not have the tool grammar mask applied (an asymmetry between the TP-2 prefill and decode paths, a separate leftover item).

**Incidental observation (unfixed, separate)**: sending the same request 3 times in a row to the serving instance with `temperature: 0`, the first 4 tokens show two outcomes
(`"The user wants me"` / `"We need to respond"`). `src/serve/translate.cpp:38-88` overrides the service default with the request's temperature,
so the temperature really is applied; DFlash2's sparse acceptance (`speculative_accept_sparse_drafts`) combined with a random seed per request (`translate.cpp:51-57`)
means decoding is not guaranteed to be bitwise argmax even at temperature 0 — which is exactly why `--greedy` ("force temperature 0 (exact argmax)") exists.
When reproducible A/B is needed, add `--greedy`; this is independent of this fix and was not changed.

## 6. Reuse gate repriced by "the session's own turns" (2026-09-24, changed)

**Problem**: the budget in the gate `position - reuse <= max(reuse_grid, 16 × request.budget.remaining())` is the client-set cap
(DSH fixes it at 32,768). When budget ≥ 16K, `16 × budget >= max_context`, the unaligned fallback branch is **mathematically unreachable** — and it is the
only entry point for tail reuse (there is no decode checkpoint on the grid), meaning under that machine's production config it can never trigger, even if the saving is the whole context.

**Change**:
- `SessionEntry` gains `generated_tokens_total` / `generated_turns`; after each round they are accumulated by `request.generated.size()`
  (counting only rounds that generated ≥1 token, ensuring a non-zero mean).
- The gate switches to `expected = min(budget, total/turns)`; with no completed round (first-round request, or retention off) it falls back to budget (the original worst case).
- Semantically it only loosens (`min(budget, mean) <= budget`), so any boundary previously accepted is still accepted.

**Verification**:
- Test `ninfer_qwen3_5_tp2_sessions_test`: dflash2 / mtp EXIT 0. New scenario = 5 short rounds + 1 long round (60 tokens) + 1024-budget
  continuation round, expecting reuse straight to the frontier; two further scenario self-checks fail explicitly with numbers if they ever stop landing in the take branch. Evidence `profiles/gate-{dflash2,mtp}.log`.
- Online (serve + trace): 20 short rounds (mean 2) + 1 2000-token long round → continuation round under **budget 32,768**: `reuse=2442 src=live`,
  `cache 2,442 (99.3%)`, **TTFT 57.9 ms** (prefill only 17 tokens); the old rule's threshold 16 x 32,768 = 524,288 would necessarily skip.
  The same request declines the draft per the gate, decode 22.5 tok/s, output 37 tokens ⇒ cost about 1.2 s < about 1.6 s saved, so this note is a net gain.

**Known boundary (deliberately kept)**: the mean used for pricing includes burst rounds, so whether "the short round immediately after a burst" can reuse depends on that session's mean —
it is still skipped when the mean > saving/16 (a risk-neutral conclusion, not a bug). The current DSH session mean is about 1,735, and its 24.8k tail is still skipped;
flipping this requires making the ratio 16 a configurable option, or taking step 3 (keeping the masked draft under unaligned reuse).

## 7. Unaligned reuse no longer declines the draft: masked draft in-line throughout (2026-09-24, changed)

**The premise was falsified by measurement**: step 2's gate was built on "unaligned reuse must drop the masked draft". Measurement shows this premise does not hold —
after keeping the draft the ring differs only at the ulp level and the acceptance rate is almost unchanged.

**Measurement (same artifact, serve recipe + trace; 20 short rounds mean 2 + one 2000-token long round, continuation budget=32768)**:
- Same reuse result: `reuse=2442 src=live`, `cache 2,442 (99.3%)`, TTFT 58 ms, prefill only 17 tokens;
- Declining the draft (old behavior): `decode 22.5 tok/s`, output 37 tokens;
- Keeping the draft (new behavior): `decode 87.1 tok/s`, `dflash2 accepted 28/63 (44.4%)`.
Declining the draft never protected correctness (target verify always issues a token for each token), only the contract of "bitwise reproduction of the from-scratch walk".

**Change**:
- `extent` no longer goes to zero because a boundary is unaligned; `dflash_draft_declined_`, `GenerationResult::draft_context_declined`,
  the stderr hint and the print in the solo test are deleted.
- The reuse scan is merged into a single pass: grid filtering and the gate are deleted, **the deepest boundary wins**; `reuse_grid` is deleted along with them.
- Step 2's `SessionEntry::generated_tokens_*` and session-mean pricing are deleted along with them (the premise is gone, the gate loses meaning).

**Contract (strict-version experiment to delimit)**: boundary on the grid ⇒ token-by-token identical to the oracle; boundary inside a chunk ⇒ only boundary crossing is guaranteed.
Temporarily changing `compare_recall` to "require token-by-token equality everywhere" and running it fails at `a conversation behind a re-rendered answer`:
`got [2752 11 220 16 24 24 15 82]` vs `expected [2752 11 198 220 220 16 15 15]` —— the first two tokens match, the third diverges,
consistent with the old comment's prediction that "the third token must diverge", showing the divergence comes from unaligned prefill chunking (different window splits), not from the draft mode.
Keeping the draft in fact reduces the divergence source from two (mode + chunking) to one (chunking).

**Verification**:
- `ninfer_qwen3_5_tp2_sessions_test`: dflash2 / mtp EXIT 0; plain on this machine fails at the existing scenario "a conversation behind a
  shared system prompt" (`got [2752 13 198 ...]` / `expected [365 2798 349 ...]`), bit-for-bit identical to the same failure recorded in the previous session.
  This commit is a **behavior-equivalent refactor** for plain/mtp (the old gate only took effect when `dflash2_enabled_`, and both already took the deepest boundary), so it is unrelated to that failure.
- `ninfer_qwen3_5_tp2_dflash_solo_test` (after rebuild): two independent processes both digest `0x4bcc3994a5efba7d`, PASS, matching the documented value
  ⇒ the grid-aligned path is undisturbed (the solo scenario itself contains no unaligned reuse).
- served route (step-3 binary): `reuse=2442 src=live`, `cache 99.4%`, TTFT 51-57 ms, prefill 14 token, no draft-declined hint.
  Two independent restarts of the same binary give bit-identical results (`output 52`, `accepted 31/147 (21.1%)`, `decode 53.9 tok/s`) ⇒ that binary is deterministic,
  and 21% is not run-to-run jitter (the earlier "temperature 0 is nondeterministic" conclusion applies only to that scenario at that time).
- **Acceptance rate is a property of the text, not of this change**: same reuse boundary, same binary, changing only the continuation content —— prose 53.9 tok/s / 21.1%,
  a numeric list 175.8 tok/s / 99.6% (446/448). Therefore `44.4%` (the experiment run, numeric type, output 37) and `21.1%` (final version, prose, output 52)
  are measurements of two different texts and cannot serve as an A/B; the true same-walk A/B is "declining the draft 22.5 tok/s vs keeping it 87.1 tok/s / 44.4%" (same conversation,
  same scan, only extent toggled).
- Why the two builds produced different text: deleting the gate also changed the reuse boundary of **early short rounds** (the old gate required `saving > max(1024, 16×expected)`;
  the off-grid saving of a short-round prompt is far below 1024, so the old code fell back to the aligned boundary; the new code takes the frontier) ⇒ different prefixes ⇒ a 2-token short answer may differ at the tie
  ⇒ content-level cascade divergence (observed: old 2,459/37 vs new 2,456/52). Both are legitimate walks.

## 8. DFlash2 draft second-stage quantization r62–r66 (2026-09-24; project opened, not started)

**Background**: §3.7 (r54–r57) lowered the draft to "the lowest qtype supported by the consuming op at the time", after which every draft component sits
at the op-level lower bound. On 2026-09-24 a per-op supportability audit of the remaining space was done (evidence chain: the consuming ops' qtype validators), concluding that
**no item can land in the current build by "changing only the recipe"** —— the remaining space all requires new op variants. This campaign does the four most cost-effective of them
(codebook and cyclic FP8 explicitly deferred, see below).

**Audit conclusions (each item's precision lock point)**:

| Component | Current | Lock point (consuming op) | Verdict |
|---|---|---|---|
| mlp/gate_up [34816,5120] | q4 | `linear_swiglu` q4 plan shape closes exactly (`q4_linear_swiglu_plan.cpp:33`) | already at the bottom, leave alone |
| mlp/down [5120,17408], attention/output [5120,4096] | q5 | fused `linear_dynamic_grouped_conv_add` validator only admits Q5_G64/Q8_G32 (`wrapper/dynamic_grouped_conv.cpp:60-95`) | **r63**: new q4 variant |
| feature_projection [5120,25600] | q5 | plain `ops::linear`; the q4 registry has no {5120,25600} (`q4_dispatch.cpp:13-25`) | **r62**: new q4 shape |
| context_key [6144,5120] | q8 | decode: `attn_input_proj` three-output overload hard-asserts `require_q8_rowsplit` (`wrapper/attn_input_proj.cpp:243-261`); prefill: `context_kv_materialize` hard-asserts Q8_G32 [1024,5120] (`context_kv_materialize.cpp:37-52`) | **r65**: new q4 variants in both places |
| kernel_projection [1280,5120] | bf16 | `rmsnorm_dynamic_grouped_conv_prepare` hard-asserts BF16+Contiguous+no scale (`wrapper/dynamic_grouped_conv.cpp:39-52`) | **r64**: new q4 prepare |
| codebook [248320,256]×2 (shard 1) | bf16 | `candidate_selector_path`'s only bf16 variant (`wrapper/candidate_selector.cpp:110-113`); and the parameter is a bare `Tensor` not a `Weight` (`parameters.h:113`) ⇒ also requires changing the Tensor→Weight pipeline + TP-2 peer path | **deferred** (deepest change + highest acceptance-rate sensitivity: 256-dim inner-product selection) |
| cyclic K=BF16/V=FP16 | — | locked in three places: `CyclicKVCache` layout, the SWA check (`sliding_window_attention.cpp:60-62`), and `context_kv_materialize`'s `validate_cache` (:98-99); only windows {2048, 4096} are registered | **deferred** (~55 MiB, largest engineering effort, worst cost-effectiveness) |

**Target gain** (shard 0, exact values; GPU0 free 607 → ~860 MiB ≈ +16k token KV cap):

| Item | Object | Current → Target | Saving |
|---|---|---:|
| r62 | feature_projection | q5 82.03 → q4 66.4 | **15.6** |
| r63 | attention/output ×5 + mlp/down ×5 | q5 344.5 → q4 278.9 | **65.6** |
| r64 | kernel_projection ×10 | bf16 125.0 → q4 33.2 | **91.8** |
| r65 | context_key ×5 | q8 159.4 → q4 79.7 | **79.7** |
| total | | | **252.7** |

**Decision (following §3.7)**: all four are opt-in overrides, not changes to the official recipes (`official_recipes.py:35-46` continues to pin
mtp/dflash/dflash2 to Q8, and to exclude kernel_projection/codebook/hidden_projection from quantization); the artifact is an experimental build.
Before upgrading to a contract-protected feature, the three-part set at the end of §3.7 must be added (append K=7/K=5 golden values, solo digest, sessions retention assertions)
+ sampled-mode A/B + the "independently decode by stored scale" check for the Q4 op oracle.

### 8.1 Per-item change list

**r62 — feature_projection → q4 (smallest surface, plumb the whole chain first)**
- New `src/ops/linear/q4/shapes/n5120_k25600.cu` (selector mirrors `src/ops/linear/q5/shapes/n5120_k25600.cu`:
  T=1 simt_r8_c4, T=2–6 ksplit, T≤24 simt_r8_c8, rest mma_r64_c128; draft prefill width ≤2048 × batch).
- Changes: `q4_shapes.h` declaration + `q4_dispatch.cpp` registry gains {5120,25600} + `src/ops/linear/q4/sources.cmake`.
- Test: `tests/ops/linear/test_q4_a16.cpp` gains 5120×25600 (independent seed, FP64 oracle, all route boundaries).
- override `tools/tp_bootstrap/r62_draft_feature_q4.py` (`dflash2/feature_projection` → q4_g64_fp16,
  `grouped_absmax`) + dry-run + conversion script (r57 recipe + `--device cpu`).
**r63 — attention/output + mlp/down → q4 (q4 variant with fused conv-add)**
- Key structural fact (audit finding): the GEMM part of the q5 variant delegates to the ordinary q5 registry
  (`select_q5_a16_launch`); the convolution + residual tail is the qtype-agnostic shared
  `dynamic_conv_finish_launch` (`q5_dynamic_grouped_conv_add_materialized.cu:11-20`)
  ⇒ **the q4 variant needs no new fused kernel, only two new ordinary q4 GEMM shapes + a q4 plan directory**
  (smaller than the original estimate of "a new fused-kernel family").
- New `src/ops/linear/q4/shapes/n5120_k4096.cu`, `n5120_k17408.cu` (mirroring the same-named q5 shapes).
- New `src/ops/dynamic_grouped_conv/q4/` (plan/materialized, mirroring `q5_dynamic_grouped_conv_add_plan.cpp`:
  the projection delegates to `select_q4_a16_launch(5120, input_rows, tokens)`, the tail uses the shared finish;
  route names `dynamic_grouped_conv_add.q4.*.materialized_bf16`).
- Changes: `wrapper/dynamic_grouped_conv.cpp` (`require_finish_projection_weight`/`projection_planes`
  admit Q4_G64 group 64, no qhigh; `linear_dynamic_grouped_conv_add` gains a q4 dispatch branch);
  `include/ninfer/ops/dynamic_grouped_conv.h` three-codec semantics synced; the capacity API already had a
  qtype first parameter (added in r56), only the q4 branch was added.
- Tests: `tests/ops/test_linear_dynamic_grouped_conv_add.cpp` adds Q4 × C∈{4096,17408} over the full W/B domain
  (2..128 tokens) × graph replay, FP64 oracle independently decoded per stored scale (r56 precedent: 120 shapes 4.38 s).
- Override `tools/tp_bootstrap/r63_draft_finish_q4.py` (`/mlp/down`, `/attention/output` → q4) + count check (5 each).

**r64 — kernel_projection → q4 (the q4 variant of prepare)**
- New q4 prepare partial kernel ([1280,5120] q4 dequantization GEMM, split-K) + q4 plan (mirroring the routes of
  `bf16_dynamic_grouped_conv_prepare_plan.cpp`: tokens≤48 use R16C{8,16,32,48}S8, the rest R32C{32,64}S4).
- Changes: `wrapper/dynamic_grouped_conv.cpp` (`require_kernel_projection_weight` admits Q4_G64;
  `rmsnorm_dynamic_grouped_conv_prepare` gains q4 dispatch) + `sources.cmake`.
- Tests: `tests/ops/test_dynamic_grouped_conv_prepare.cpp` adds the q4 arm: prepare = rmsnorm∘GEMM∘(base+delta)
  as an independent FP64 reference implementation, full T domain (2..128).
- **Numerical-property warning**: this GEMM's product is the dynamic convolution weight itself (error in weight
  space, not activation space), sensitivity may be higher than the ordinary GEMM of r62/r63 ⇒ the acceptance-rate
  gate must pass; **fallback = keep bf16** (this item is optional, failure does not block the campaign, total gain drops to 160.9).
- Override `tools/tp_bootstrap/r64_draft_kernel_q4.py` (`/attention_conv/kernel_projection`,
  `/mlp_conv/kernel_projection` → q4; the official recipe currently excludes these two names with `continue`,
  dry-run confirmed 10 objects hit).

**r65 — context_key → q4 (two ops, the largest item, done last)**
- decode path: new `src/ops/attn_input_proj/q4/` plan + 6 dflash2 schedules (mirroring `kDFlash2Routes`:
  SmallT ≤48, MmaR16C64K128, MmaR32C32K128, MmaR32C64K128, MmaR32C64, MmaR64C128; draft decode T=(K+1)×B ≤64,
  the directory still covers up to kAnyCols per the closure convention).
- prefill path: new q4 variant of `context_kv_materialize` (all 7 routes KSplit16/KSplit24/Mma32/Mma80/Mma96/Fused64/Mma64
  are needed — prefill widths 1..2048 all reach them; q4 dequantization = 2 code/byte + scale every 64, versus every 32 for the current q8;
  new file `materialize_q4.cu` + launch dispatched by qtype).
- Changes: `wrapper/attn_input_proj.cpp` (three-output overload gains a Q4 branch: `require_q4_rowsplit` [6144, hidden]);
  `context_kv_materialize.cpp` (`require_weight` admits Q4_G64 group 64, scale bytes rows×80×2; dispatch by qtype).
- Tests: `tests/ops/test_attn_input_proj.cpp` adds the dflash2 three-output q4 route boundaries; materialize test: 5 layers ×
  the full W/B profile, ring contents compared element-by-element against a naive reference (q4 GEMM → key_norm rmsnorm → RoPE → ring write).
- **Recipe note**: the official recipe `recipe.share(prefix+"context_key", prefix+"key")` (`official_recipes.py:52-54`)
  binds context_key/context_value to share the same [6144,5120] object with query/key/value ⇒ the override assigns to a
  shared object, and the dry-run must confirm "exactly 5 objects become q4_g64_fp16" (mirroring the r57 count check).
- **Acceptance-rate risk**: the highest of this batch (the K projection error accumulates token by token inside the 2048-window
  ring, affecting every subsequent decode step in the window) ⇒ the gate must pass; fallback on failure = keep q8.
- Override `tools/tp_bootstrap/r65_draft_contextkey_q4.py`.

**r66 — merged experimental artifact**: the four items r62–r65 combined (override `r66_draft_q4_all.py` with a 26-object count
check: feature 1 + down 5 + output 5 + kernel 10 + context_key 5), full acceptance after conversion.

### 8.2 Acceptance criteria per step (gates)

1. **op oracle** (AGENTS.md numerical contract): each new shape/route against an independent FP32/FP64 naive reference; packed
   inputs **independently decoded per stored scale** (do not trust the scale semantics the converter emitted); route boundaries
   (T split points ±1) fully covered; graph replay covered.
2. **Acceptance-rate A/B** (following the §3.7 method): greedy probe 7×160, K=5 and K=7, an independent cold-started server per arm;
   the baseline arm = the active r57 artifact (`<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_draftall.ninfer`, r57 measured acc K=7 25.3% /
   56.9 tok/s); **pass line |Δacc| ≤ 1.0pp and Δtok/s ≥ −3%** (r54–r57 observed band ±0.5pp / ±1%).
   Each step additionally adds one combined intermediate arm (previously passed items + this item) to isolate interaction effects.
3. **Determinism**: the baseline arm reproduces byte-identically across rebuilds (r57 precedent: 2843/699, text length all identical);
   the new arm's acceptance rate must be explained by probe-text drift (tie flips) rather than a systematic drop.
4. **Memory check**: `nvidia-smi` + the startup `[mem]` ledger, device-side saving differing from the exact MiB computation by
   ≤ the allocation granularity (precedent ~1.6 MiB).
5. **Upgrade gate (after r66)**: rebaseline the DFlash2 trio against the new artifact (`ninfer_qwen3_5_tp2_dflash_append_test` K=7/K=5 golden values,
   `ninfer_qwen3_5_tp2_dflash_solo_test` digest, `ninfer_qwen3_5_tp2_sessions_test` retention assertion) +
   sampling-mode (temp 0.7) acceptance rate/throughput A/B (r54 precedent: under sampling K=2 once lost −2.5pp at 1.6σ).

### 8.3 Rollback points

- **C++ side**: all additive (new shape files, new q4 plan directories, new validation branches); the existing q5/q8/bf16 routes keep their
  bit patterns (r56 precedent: zero drift on the Q8 routes after the shared-finish refactor) ⇒ rollback = git revert, no artifact format change.
- **Artifact side**: each item has its own independent override, the official recipe is untouched ⇒ rollback = re-convert with the previous
  override set (~300 s); the old artifact can always be run by the new engine (the engine is a superset).
- **Item-level rollback**: if r64 or r65 fails the acceptance rate ⇒ that item keeps its current precision and the campaign continues
  (r64 failure: total gain 160.9; r65 failure: 220.5; both fail: 140.0).

### 8.4 Execution order and dependencies

0. Baseline re-measurement: acceptance rate + throughput of the active r57 artifact at K=5/7 (confirm agreement with the 25.3%/56.9 recorded
   in §3.7, no environment drift).
1. **r62** (smallest surface: exercise the whole chain shape→dispatch→oracle→override→conversion→A/B, validate the methodology).
2. **r63** (reuses r62's dispatch infrastructure + r56's q5 plan template).
3. **r64** (independent op; can be developed in parallel with r63, A/B run sequentially — process discipline: only one model process machine-wide at a time).
4. **r65** (largest item; by then the acceptance-rate data of r62–r64 is in hand, so an early stop-loss is possible if a systematic downward trend has appeared).
5. **r66**: merged artifact + sampling A/B + trio rebaseline + memory check (GPU0 free 607 → ~860; GPU2 unchanged) +
   decide whether to request promotion to a contract feature (current decision: stay opt-in).

**Build/ops conventions** (following the §1 environment table): Windows build-win (WSL CUDA is dead, compile verification only); tests built
per target (`cmake --build build-win --target ninfer_linear_q4_a16_test` etc.); conversion `--device cpu` (~300 s, no GPU needed);
serve 8099 via a harness background job; stop the service before relinking, manually sync the exe into `C:\ninfer\` and compare with `Get-FileHash`.

### 8.5 Progress

**Step 0 baseline re-measurement (2026-09-24, done)**: the active r57 artifact (`<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_draftall.ninfer`),
the current HEAD binary, `--max-context 131072`, greedy probe 7×160, an independent cold-started server per arm
(`tools/tp_bootstrap/r62_baseline_arms.ps1`, logs and JSONL under `build-win/r62/`):

| K | drafted / accepted | acc | tok/s |
|---|---|---:|---:|
| 7 | 2578 / 736 | 28.55% | 61.1 |
| 5 | 1968 / 711 | 36.13% | 61.0 |

**Inconsistent with the §3.7 archive (K=7 25.3% / 56.9), attributed**: after §3.7 there were two commits that changed numbers/changed the walk —
`ea6204fa fix(ops): restore accurate silu in nvfp4 swiglu` (changes model numerics ⇒ the text trajectory changes) and the §4–§7 TP-2 first-token
publishing + reuse-scan changes; the acceptance rate is a property of the text (§7 has argued this), and its absolute value drifts with the binary.
⇒ **this campaign uniformly uses the r57 artifact from this re-measurement as the
baseline**; the §3.7 numbers are historical only. Side product: on this machine the current binary has equal throughput at K=5 and K=7 (61.0 vs 61.1).

**Step 1 r62 feature_projection q4 (done; conclusions and gates in §8.6)**

- [x] **op variant**: `src/ops/linear/q4/shapes/n5120_k25600.cu` + registration in `q4_shapes.h` / `q4_dispatch.cpp` /
      `src/ops/linear/q4/sources.cmake`. **Deviation (intentional)**: the selector uses q4's coarse buckets (`T=1 → simt_r8_c4`,
      `T≤8 → ksplit<5120,25600,8>`, `T≤24 → simt_r8_c8`, otherwise `mma_r64_c128`), not the per-T capacity buckets written in §8.1
      (2..6). Rationale: q4's `Capacity` is only a compile-time **mask column upper bound**, the tile width is unified to 8 by `(Capacity+7)/8*8`,
      and per-T instantiation only changes one staging loop bound (dead code on the masked path) ⇒ pure code bloat; the convention of the
      existing q4 shapes is coarse buckets.
- [x] **oracle**: `ninfer_linear_q4_a16_test` **PASS** (12.1 s). Two new 5120×25600 groups: `Comparison::Full`
      (T∈{1..7,24,25}, element-by-element) and `Comparison::Sampled` (T∈{1..9,15,16,17,23,24,25,26,32,33,63,64,65,127,128,129,2048}),
      both groups include graph replay; the FP64 naive reference decodes independently per stored scale.
- [x] **override + dry-run**: `tools/tp_bootstrap/r62_draft_feature_q4.py` (**cumulative**: r57's three levers +
      feature_projection→q4; asserts gate/up/down/output 5 each) + `tools/tp_bootstrap/r62_draft_q4_dryrun.py`.
      Dry-run parse: `q4=11` (10 gate/up + 1 feature), `q5=10` (down + output), `q8=25`, `bf16=45`
      ⇒ differs from the r57 artifact by exactly one object, feature_projection.
      **Explanation (deviation from §8.1)**: §8.1 describes each override as "changing only this item", but §8.2 requires the baseline arm = the r57 artifact
      ⇒ each step's arm must be a **cumulative artifact (previously passed items + this item)**, otherwise it falls back to the official Q8 recipe and
      does not test the product path. Being cumulative also lets "this step vs the previous step" isolate the item's incremental effect.
- [ ] **conversion**: `tools/tp_bootstrap/r62_convert_feature4.ps1` (r57 recipe + this override, `--device cpu`)
      → `<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_featproj4.ninfer` (in progress).
- [x] **A/B + determinism**: after the rebuild the baseline arm reproduces byte-identically (this step's C++ is all additive) + r62 arm gate — conclusions and numbers in §8.6.

**Subsequent steps (not started)**

- [ ] 2 r63 finish projection q4 (two shapes + q4 plan + wrapper + oracle + override + conversion + A/B)
- [ ] 3 r64 kernel_projection q4 (prepare kernel + wrapper + oracle + override + conversion + A/B)
- [ ] 4 r65 context_key q4 (attn_input_proj q4 + materialize q4 + two wrappers + oracle + override + conversion + A/B)
- [ ] 5 r66 merged artifact + sampling A/B + trio rebaseline + memory check + upgrade decision

### 8.6 Execution record (2026-09-24; progress anchor across context compaction)

**Methodology correction (important, followed by later steps)**: §8.2's "greedy 7×160 probe |Δacc| ≤ 1.0pp" was judged **content-confounded** — as soon as
the draft weights change, the target's tie flips change the text, and the acceptance rate follows the text (§7 gave the extreme example of 21% vs 99.6%). The same pair of artifacts for r62:

| Metric | Baseline (r57 artifact) | r62 artifact | Δ |
|---|---:|---:|---:|
| greedy K=7 acc | 28.55% | 28.64% | +0.09pp |
| greedy K=5 acc | 36.13% | 34.02% | **−2.11pp (over the gate)** |
| sampling K=5 acc (3 classes × 6 runs × 256 token, pooled) | 44.12% | 45.66% | **+1.54pp** |
| sampling K=7 acc (baseline measured later: 15 runs × 256 token) | 33.90% | 36.58% (r63 artifact) | +2.68pp |

Sampling breakdown (reason / prose / code): +4.3 / +0.1 / +1.5 pp ⇒ the two metrics have opposite signs ⇒ the greedy difference is text drift, not q degradation.
**New gate method**: `tools/tp_bootstrap/r62_sampling_ab.ps1` (sampling acceptance rate, content-averaged, insensitive to machine drift) as the acceptance-rate
arbiter; the greedy probe is used only for the determinism check "baseline reproduces byte-identically across rebuilds"; memory is checked with the `[mem]` ledger.
**Determinism**: two independent cold-started servers on the r62 artifact are byte-identical; after the rebuild (r62+r63+r64 code) the baseline arm's K=7 JSONL has
exactly the same SHA256 as the baseline before the change (`D61C1112…`) ⇒ both the engine and the artifact are deterministic.

**Step 1 r62 feature_projection q4 (done: passed)**

- op: `src/ops/linear/q4/shapes/n5120_k25600.cu` + registration in `q4_shapes.h` / `q4_dispatch.cpp` / `sources.cmake`.
  The selector uses q4's **coarse buckets** (T=1 → simt_r8_c4, T≤8 → ksplit<5120,25600,8>, T≤24 → simt_r8_c8, otherwise mma_r64_c128),
  not the per-T capacity buckets written in §8.1 (2..6): q4's `Capacity` is only a compile-time masked column upper bound, the tile is unified to 8 by
  `(Capacity+7)/8*8`, and per-T instantiation only changes one staging loop bound (dead code on the masked path) ⇒ pure code bloat.
- oracle: `ninfer_linear_q4_a16_test` PASS (12.1–14.7 s; Full T∈{1..7,24,25} + Sampled up to 2048, including graph replay).
- override `r62_draft_feature_q4.py` (**cumulative**: r57's three levers + feature→q4) + dry-run `r62_draft_q4_dryrun.py`:
  q4=11 (10 gate/up + 1 feature) / q5=10 / q8=25 / bf16=45.
- conversion `r62_convert_feature4.ps1` → `..._dflash2_featproj4.ninfer` (375 s, 1218 objects); file −16,384,000 B.
- memory: shard 0 `weights+ctx` 12750.6 → 12734.6 MiB, free 816 → 832 MiB.

**On the cumulative design of the overrides**: §8.1 describes each override as "changing only this item", but §8.2's baseline is the r57 artifact ⇒ each step's arm must be a
cumulative artifact (previously passed items + this item), otherwise it falls back to the official Q8 recipe and does not test the product path. The r62→r63→r64 overrides are all cumulative,
so **the r64 artifact is itself the merge candidate for r62+r63+r64** (r66 is merely the formal artifact with added count checks).

**Step 2 r63 attention/output + mlp/down q4 (done: passed)**

- Structural finding: the q5 variant's GEMM delegates to the ordinary q5 registry, and the convolution + residual tail is the qtype-agnostic shared `dynamic_conv_finish_launch`
  ⇒ the q4 variant **needs no new fused kernel**, only two new q4 shapes + one q4 plan directory (smaller than the §8.1 estimate).
- New: `src/ops/linear/q4/shapes/n5120_k4096.cu`, `n5120_k17408.cu`; four files under `src/ops/dynamic_grouped_conv/q4/`;
  q4 branches added to the wrapper's `projection_planes` / `require_finish_projection_weight` / capacity dispatch / execution dispatch;
  `bench/ops/linear_dynamic_grouped_conv_add_bench.cu` supports `--qtype q4` (the route name `…q4.ksplit_exact…` has been measured).
- oracle: `ninfer_linear_q4_a16_test` PASS (12.7 s); `ninfer_linear_dynamic_grouped_conv_add_test` PASS
  (7.9 s: Q8/Q5/Q4 × C∈{4096,17408} over the full W/B domain + graph replay).
- override `r63_draft_finish_q4.py` dry-run: q4=21 (1 feature + 10 gate/up + 5 down + 5 output) / q8=25 / bf16=45.
- conversion `r63_convert_finish4.ps1` → `..._dflash2_finish4.ninfer` (378 s, 1218 objects); file 22,947.1 → 22,865.8 MiB
  = **−81.25 MiB** (= r62 15.625 + r63 65.625, exactly consistent).
- A/B: `tools/tp_bootstrap/r62_step_arms.ps1 -Tag finish4` (greedy K=7/K=5 + sampling K=7/K=5) — **passed**:
  greedy K=7 28.55%→27.03% (−1.52pp, still content-confounded), greedy K=5 36.13%→35.80% (−0.33pp ✓),
  sampling K=7 33.90%→36.58% (**+2.68pp**), sampling K=5 44.12%→46.21% (**+2.09pp**) ⇒ both sampling metrics rose clearly.
  To fill in the sampling baseline, the baseline artifact's K=7 sampling was measured retroactively (15 runs, 33.90%).
- memory (shard 0 `[mem]`): `weights+ctx` 12750.6 → **12668.6 MiB**, free 816 → 898/904 MiB;
  measured −82.0 MiB vs predicted −81.625 MiB (difference 0.375 MiB = artifact allocation granularity, the same constant offset as in the r62 step).

**Step 3 r64 kernel_projection q4 (op done, conversion/A-B pending)**

- Implementation choice (deviation from §8.1): instead of writing a "q4 prepare partial kernel", **reuse the ordinary q4 GEMM that already passed the oracle** —
  add `src/ops/linear/q4/shapes/n1280_k5120.cu`; `…/q4/q4_dynamic_grouped_conv_prepare_plan.cpp` does
  rmsnorm → ordinary q4 linear (writing the [1280,tokens] coefficient matrix into the workspace) → new reduce
  (`q4_dynamic_grouped_conv_prepare_reduce.cu`: replaces the bf16 reduce's FP32 split-K partial-sum input with the BF16 coefficient matrix).
- Cost and rationale: the coefficient matrix is materialized as BF16 (the bf16 route keeps FP32 partial sums), adding about half a BF16 ulp; that intermediate is not an observable boundary
  (AGENTS.md), and is far smaller than the quantization error of the q4 weights themselves. The test therefore gives the Q4 arm its own criterion (prepared budget 1.5× bf16),
  measured worst ratios rel_l2 0.62× / gross 0.75× of budget, finish uses the shared budget 0.74× / 0.68×.
- Capacity: prepare's public capacity API has no qtype, and the q4 route reserves the same amount as the query (the bf16 split-K partial sum is always larger than q4's BF16 coefficient),
  preserving the `peak_used == query` invariant.
- oracle: `ninfer_linear_q4_a16_test` PASS; `ninfer_dynamic_grouped_conv_prepare_test` PASS (bf16 + q4 two arms,
  label prefixes now carry the codec for easier localization).
- conversion: **folded into the r66 merge** (`r64_convert_kernel4.ps1` is no longer run separately: the cumulative r66 artifact = r62+r63+r64,
  which would be a byte-identical file to running r64 alone, saving one 6-minute conversion and 23 GB of intermediates); r64's gate is thus measured on the r66 artifact.

**Step 4 r65 context_key q4 (decision: deferred this time)**

- Fact check (more accurate than the §8.1 audit): the draft's `attention/{query,key,value,context_key,context_value}` in the artifact are
  **all aliases of the same [6144,5120] object** (1 per layer × 5 layers), so "context_key ×5 = 79.7 MiB" actually means demoting 5 fused QKV
  objects as a whole to q4, and both consumers must be changed **simultaneously** for it to load:
  - decode: the `attn_input_proj` three-output overload and `prepare_attn_input_proj_weights` both hard-require Q8 (`weight_input.cpp:176`);
  - prefill: `context_kv_materialize`'s fused kernel (`materialize.cu`, 482 lines) hand-writes staging/MMA/score write-back with **1 signed byte per 32 elements + FP16 scale**,
    and fuses the key's norm+rope+ring-cache write with the value's fp16 write-back.
- Conclusion: r65 is not "one more shape", but writing a family of 4-bit fused materialization kernels (7 routes) + a 3-output q4 variant + two sets of oracles.
  Relative to the 79.7 MiB gain, the remaining budget and risk of this campaign are not worth it; §8.3 itself also allows item-level rollback ("that item keeps its current
  precision") and schedules it last.
- This delivery is therefore = **r62 + r63 + r64 (173 MiB / 252.7 MiB = 68%)** + the r66 merge; r65's audit and roadmap are retained.

**Step 5 r66 (done: passed)**

- override `tools/tp_bootstrap/r66_draft_q4_all.py` (single cumulative file, count check over 31 selections: feature 1 + gate/up 10 +
  down 5 + output 5 + kernel 10; gate/up shares one stored object per layer ⇒ 26 objects promoted).
- dry-run: q4 31 / q8 25 / bf16 35 (10 of the r57 artifact's bf16 45 were kernel projections moved into q4).
- conversion `r66_convert_q4all.ps1` → `<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_q4all.ninfer` (326 s, 1218 objects).
- **File-level accounting (exactly equal)**: 24,061,771,780 → 23,880,318,980 B = **−181,452,800 B = −173.05 MiB**
  = 16,384,000 (r62) + 68,812,800 (r63) + 96,256,000 (r64), byte-for-byte as predicted.
- Object formats: `q4_g64_fp16` 81 (r57 artifact 71 + 10), `bf16` 569, `q5_g64_fp16` 54, `q8_g32_fp16` 12.
- A/B (`r62_step_arms.ps1 -Tag q4all`) — **passed**:

  | Metric | Baseline r57 artifact | r66 q4all artifact | Δ | Gate |
  |---|---:|---:|---:|---|
  | greedy K=7 acc | 28.55% | 26.95% | −1.60pp | over the line (content-confounded, see the §8.6 method correction) |
  | greedy K=5 acc | 36.13% | 35.96% | −0.17pp | ✓ |
  | sampling K=7 acc (15×256) | 33.90% | 35.05% | **+1.15pp** | ✓ |
  | sampling K=5 acc (15×256) | 44.12% | 47.84% | **+3.72pp** | ✓ |

  **Why the greedy probe cannot be compared across arms (direct evidence, not inference)**: `greedy_probe.ps1` recorded the full text, and a per-line SHA256 comparison of the baseline and
  q4all artifacts' K=7 probes shows: **6 of the 7 prompts have different text** (lengths 779/788, 768/789, 877/890, 741/778, 804/785, 804/798),
  only idx 5 is byte-identical. The mechanism is the one recorded in §7: a changed draft proposal length changes the verify batch shape ⇒ the target's numerical path changes ⇒ a near-tie greedy
  decision flips ⇒ the text changes ⇒ the denominator of the acceptance rate is now different content. So the greedy 7×160 difference is "two numbers over different texts".
  **Honest caveat**: on the only text-stable one, idx 5, the K=7 acceptance rate still drops 3.75pp (672 drafted), suggesting that the q4 draft at K=7 may be slightly
  weaker on that content; but the pooled sampling (7711 drafted total at K=7) is the more reliable statistic and rises +1.15pp.

  Both sampling gates pass; the greedy K=7 overage appears at every one of r62/r63/r64 (−1.5 ~ −2.1pp) while sampling simultaneously rises,
  which is the content confusion recorded in §8.6, not monotonic degradation (if it were degradation, sampling should fall as well).
- **Memory check (shard 0 `[mem]`)**: `weights+ctx` 12750.6 → **12576.6 MiB** (−174.0 MiB, predicted file-level −173.05 MiB,
  difference 0.95 MiB ≤ allocation granularity 1.6 MiB); free 816 → **990 MiB** (K=7) / 996 MiB (K=5); shard 1 unchanged (810 → 810 MiB @K=5).
  The "GPU0 free 607 → ~860" written in §8.4 is an absolute number from the old §3.7 archive (a different capacity/KV configuration); this time's measured increment agrees with the prediction.
- Trio: `tools/tp_bootstrap/r66_suites.ps1`. **FFmpeg's and libcurl's `bin` must be added to PATH before running**,
  otherwise `solo`/`sessions` fail immediately at load time with `0xC0000135` (STATUS_DLL_NOT_FOUND) — `append` does not depend on the media path, so only it is unaffected.
  Measured trio (`NINFER_TEST_ARTIFACT` pointing at the q4all artifact, unless noted):

  | Suite | Result | Evidence |
  |---|---|---|
  | `append` K=7 | PASS (37.9 s) | ring replay byte-identical; proposal deterministic `fnv1a=0xda91572dd83980bd`; workspace peak 113.3/192 MiB |
  | `append` K=5 | PASS (37.5 s) | as above, `fnv1a=0x5a3629fb79be1cd3` |
  | `solo` two processes | PASS (41.1 / 41.8 s) | two independent processes have the same digest `0x4bcc3994a5efba7d` |
  | `sessions` | **FAIL (plain route, pre-existing before this campaign)** | see below |

  The `sessions` failure: `FAIL (plain): a conversation behind a shared system prompt diverged from the oracle
  on its first sample: got [2752 13 …] expected [365 2798 …]`. **Reproduced token-for-token identically (`got`/`expected`) with the unmodified baseline artifact (r57 `draftall`) on the same binary**
  ⇒ this failure is unrelated to the q4 levers of this campaign (the plain route does not load the draft at all).
  It does not block this delivery (draft q4 is covered by append at both K + solo digest + op oracle + sampling A/B), but **it blocks "upgrading to a contract feature"**:
  the upgrade gate requires the trio to be all-green, and one of them has a pre-existing failure unrelated to this campaign ⇒ the conclusion is to stay opt-in (see below).

  **Evidence from the changed route itself (measured retroactively)**: running sessions targeted at `NINFER_TEST_ROUTE=dflash2`, both baseline and q4all **PASS**
  (164.8 s each, `TP-2 session retention (dflash2) passed: recall reused 71 prompt tokens bit-identically;
  LRU eviction forced a full prefill`) ⇒ the failure exists only on the plain route, and matches the baseline artifact token for token.

**Upgrade decision (§8.4 step 5)**: **stay opt-in, do not request promotion to a contract feature**. Two reasons:
1. In the trio, the plain route of `sessions` has a failure **pre-existing before this campaign** (token-for-token reproducible with the baseline artifact); the upgrade gate requires the trio to be all-green,
   so until that failure is fixed there is no basis for upgrading;
2. §8.2's literal gate "greedy |Δacc| ≤ 1.0pp" was not passed at K=7 (−1.60pp) — although it has been proven that this metric is not comparable across arms (6/7 prompts have different text),
   changing the gate to be sampling-led counts as a "gate revision", which should be decided separately from "upgrade".
   The q4 artifact continues to be offered as an explicit override + dedicated artifact (`…_q4all.ninfer`), with neither the engine nor the official recipe touched.
**The W4A4 family's equivalent final artifact (user's added request, 2026-09-24)**

- Script `tools/tp_bootstrap/r66_convert_q4all_w4a4.ps1` → `<models-dir>/qwen3_8_27b_w4a4_dflash2_q4all.ninfer`
  (`--model`/`--source quantized` changed to `W4A16/NVFP4/W4A4`, the draft still `W4A4+W8A8/DFlash2-FP8`;
  this matches the provenance of the existing `…_w4a4_dflash2_draftall.ninfer` — the two families differ only in the text/vision/MTP source).
- dry-run `r62_draft_q4_dryrun.py --base <models-dir>/W4A16/NVFP4/W4A4 --draft <models-dir>/W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8`:
  q4 31 / q8 25 / bf16 35, the same solution as the w8a8 family (the r57 baseline of both families was generated by `r57_draft_all_override.py`, and the draft source is the same).
- **Measured**: 18,874,128,644 → 18,692,675,844 B (−181,452,800 B, byte-for-byte the same amount as the w8a8 family);
  report `name=qwen3.8-27b-w4a4-q4all`, 1590 objects, format counts exactly the same as the w8a8 artifact (q4 81 / bf16 569 / q5 54 / q8 12).
- **Cross-family verification (object-level SHA256)**: **all 66 dflash2 objects of the two q4all artifacts are byte-identical** (including the stored objects of the 31 q4 selections)
  ⇒ the draft quality is identical to the w8a8 artifact that already passed A/B, and the acceptance-rate evidence transfers directly; 8 of the 16 objects under `text/layers/0/*` differ
  (the two families really do differ only in the text/vision/MTP source).

---

# PLAN.md archive (archived 2026-09-25, draft-Q4 official-recipe promotion + TP-2 transport/orchestration work)

Archived verbatim from the working PLAN.md when it was reduced to key information and remaining work, after the DFlash2 draft QKV/codebook q4 campaign r67-r70 (2026-09-24/25) and the official-recipe promotion (r69, 2026-09-25), the TP-2 low-frequency transport-failure investigation (§3.10, 2026-09-25), and the TP-2 orchestration and rendezvous work (§3.11, 2026-09-25). Historical record only: the current state and the remaining work live in `PLAN.md` at the repository root, product behavior and measurements in [tp2-dual-5060ti.md](../tp2-dual-5060ti.md).

# NInfer TP-2 plan (2× RTX 5060 Ti · Qwen3.8-27B NVFP4)

> **The single active plan**, and also the persistent memory across context compaction. Organized: 2026-09-24 (before the archival cleanup the
> full text went into the worklog; this file keeps only the
> key information and unfinished items).
> - Full history: `tp2-adaptation-worklog.md` — contains three verbatim PLAN.md archives (the 2026-09-21 upstream
>   cherry-pick replanning version; the 2026-09-22 delivery wrap-up version; the 2026-09-24 pre-cleanup full text, including §3.5–§3.14 and §4–§8
>   all round records).
> - Delivery notes, recommended configuration and measured data: `docs/tp2-dual-5060ti.md`; Windows native port: `docs/windows.md`.

---

## 1. Key information

- **Branch**: `feat/windows-native-port` (Windows native + WSL2 dual tree). All 20 commits by which origin/master was ahead have been
  cherry-picked (frontend jinja chain + three NVFP4 performance items), with no end-to-end A/B regression.
- **Hardware**: 2× RTX 5060 Ti 16G (SYS topology, no P2P, 448 GB/s per card). WSL2 build tree `$HOME/ninfer`;
  Windows working tree (this repository, build tree `build-win`).
- **Main workload**: Qwen3.8-27B NVFP4, TP-2 (weights split in half per shard, locally built half-geometry shapes). Draft backend: `--spec dflash2`
  (K=5/7, K=7 is the binary default) or `--spec mtp` (K=2, `--lm-head-draft`); `--spec dflash` (v1) is rejected at construction time.
- **Current state**: delivery goals ①–⑤ all complete; vision, the Windows port, the multi-session KV pool, and the decode graph + AR strategy have all landed and been measured;
  DFlash2 has been released on TP-2 (unaligned reuse no longer discards drafts); the TP-2 rendezvous protocol refactor is complete (host-authoritative id + bounded spin +
  timeout failure semantics).
- **Key numbers** (262,144 configuration, WSL): prefill 1,588 tok/s; decode 59.0 tok/s (MTP K=2); Windows 131,072 configuration
  decode 56.9 tok/s (on par with Linux). DFlash2 K=7 is about 71 tok/s in the 4096-context greedy setting; the 245,760/k8v4 (draftall artifact)
  scan gives K=5 96.4 / K=7 96.1 tok/s (interleaved recheck 93.3 / 96.9); the roofline-corrected ceiling ≈108 tok/s (the 90–180 in the
  original archive is not attainable at 180).

**Recommended run configuration** (WSL 8088 / Windows 8099 same recipe, Windows side `--max-context 131072`):

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2 --lm-head-draft --vision --reasoning-effort medium
```

`--reasoning-effort low|medium|xhigh` is the process-level default thinking effort (the request body takes precedence); `--chat-template` is available
as it is picked up by the first tier (the template is executed by embedded Jinja). Measured 2026-09-23: DFlash2 (K=5/7) decode leads MTP by ~20%, prefill cost within 1%
⇒ for throughput use `--spec dflash2 --draft-tokens 7` (the scan recipe is 245,760/k8v4, see archived §3.9–§3.11 for the basis).

### Environment and ops essentials

| Item | Value |
|---|---|
| Build (WSL) | `bash tools/tp_bootstrap/r55_build.sh` (rsync + `cmake --build build_dyn -j 8`, success marker `BUILD_EXIT=0`, about 2–3 minutes; the WSL build tree is now `build_dyn` (Ninja), `build_r35.sh` still points at the deleted `build/` and is unusable) |
| Build (Windows) | `tools/win_port/configure.bat` + `build.bat` (VS2022 + CUDA 13.3, `-DCMAKE_CUDA_ARCHITECTURES=120a`) |
| Serving | WSL 8088 (`serve_supervise.sh`); Windows 8099 (`tools/win_port/serve.ps1`, foreground by default; a self-test service must use a harness background job, processes started with `Start-Process` get killed when the tool call ends) |
| Runtime PATH (Windows) | FFmpeg (`D:\ffmpeg-dev\…\bin`) and libcurl (`D:\curl-dev\…\bin`) must be on PATH, otherwise `STATUS_DLL_NOT_FOUND` (the trio's solo/sessions hit `0xC0000135` at load time without them) |
| artifact | `<models-dir>/qwen3_8_27b_nvfp4.ninfer` (23.7 GB, old artifact, template `c3cf9e34…`); the new official artifact (template `a497db9e…`) is available with the first tier; experimental quantized artifacts `…_dflash2_draftall.ninfer` (r57, −715.6 MiB) and `…_dflash2_q4all.ninfer` (r66, −173.05 MiB, one each for the w8a8 + w4a4 families) |
| Process discipline | Only one model process machine-wide at a time (WSL 8088 and Windows 8099 are mutually exclusive); stop the service before relinking the Windows exe (LNK1104); `build-win/apps/ninfer-serve.exe` is not automatically synced into `C:\ninfer\`, it must be copied manually and compared with `Get-FileHash` |
| Toolchain | Nested `pwsh` → `wsl -e bash -lc` swallows `$var` and redirections ⇒ write commands as script files under `tools/tp_bootstrap/` and then execute them; a single blocking call is capped at 600 s ⇒ use background jobs for long tasks |
| GPU/WSL state (from 2026-09-23) | The machine exposes 3 cards: nvidia-smi order 0/2 = 5060 Ti, 1 = Tesla T10; `--devices 0,1` ⇒ the CUDA ordinals pick up the two 5060 Ti cards (misaligned with nvidia-smi order), the TP-2 pairing is unaffected. WSL-side CUDA fails across the whole stack ⇒ tests moved to Windows `tools/win_port/test.ps1`, and WSL build_dyn is only used for compile verification |

---

## 2. Completed summary (one line each, details in the archive)

- ① MTP consistency: root cause = the shard did not write a replay record (fold consumed an empty log), fixed; the criterion is "no degradation + same quality tier + speedup".
- ② Baseline: prefill 615→1,588 tok/s; decode plain 31.4 / MTP K=2 50.5–54.0 → 59.0 tok/s.
- ③④ KV / context: five KV dtypes available; 65,536 → 131,072 → 262,144 (vocabulary/hidden-layer parallel split, −1.2 GiB per card).
- ⑤ Documentation: `docs/tp2-dual-5060ti.md` + `docs/windows.md`.
- Vision: static shard division of labor (MTP→shard 0, vision→shard 1), single-image limit 8,192 tokens, no new configuration.
- Windows native port: VS2022 + CUDA 13.3 build; CLI/serve/TP-2 all working, performance on par with Linux; TMA descriptor
  mapped-pinned fix.
- Optimizations: exact-batch verify graph (−4.4 ms/round), plain decode graph (+12.6%), AR transmission split by payload (+0.63%),
  MTP draft chain graphified (−0.2~0.3 ms/round); the AR∥MMA sub-block pipeline measured 5% slower, rejected and rolled back.
- Round 19/20: chunk plan decoupled from ring rewind (1a/1b/1c), the result for the same prompt no longer depends on engine history; the host KV arena
  is paginable by default (`--host-kv-pinned` locks pages, and a rejected lock automatically falls back).
- Multi-session KV pool: host KV swap-in/swap-out + 5-session LRU + shared prefix mirroring; recalled KV is byte-identical to device.
- Upstream cherry-pick: first tier 5 items (jinja frontend chain + `--chat-template`), second tier 3 items (NVFP4 TMA route,
  SwiGLU epilogue, non-256-divisible tokens) all landed; e2e A/B no regression (decode +1.3%~+5.8%).
- Official NVFP4 artifact acceptance-rate investigation: synthetic artifacts match self-converted ones, the official artifact is lowest ⇒ the difference comes from the text weights, unrelated to components.
- TP-2 memory imbalance: scheme A (reduced proposal head split by vocabulary, −170 MiB/card) and scheme B (selector moved to shard 1,
  −246/+246 MiB) both landed, output byte-identical; new `DevicePair::sendrecv` does the byte-wise cross-card exchange, fixing the BF16-add
  allreduce sNaN bit-pattern risk of moving I32/FP32.
- TP-2 rendezvous protocol refactor (2026-09-23, incident-driven): device-side spin incident reproduced a second time (both cards at SM 100%, 6 requests silently swallowed;
  after killing the process both cards still spun for about 2 minutes) ⇒ Phase 1 bounded spin (env `NINFER_TP2_AR_TIMEOUT_MS`, default 2000) + timeout failure
  semantics (HTTP 503 + session poisoning forcing a re-prefill, the service survives and the next request is normal); Phase 2 host-authoritative 64-bit id (one
  id channel per captured graph, the id is delivered via an in-graph memcpy node — it cannot be passed by value as a kernel argument, and a stale slot never satisfies a future spin), committed
  `de74d966`/`b9c5d11c`/`634df875`; Phase 3 large-payload event path **measured and rejected** (copy engine floor 3.32 ms vs the current
  sliced 2.70 ms, already 23% slower without counting the event fence); Phase 4 partial merge audit **the fixed form does not exist** (9 call sites classified
  one by one). Regression matrix zero regressions (3 pre-existing failures reproduce byte-identically at the HEAD before the change).
- K scan and prefill cost attribution (2026-09-23): DFlash2 K=5≈K=7 (interleaved pairs slightly favor K=7; **do not use K=6** — three rounds held
  25.7 rounds/s, a reproducible per-round cost dip dominated from both sides); pure MTP K=3/4 a tie (79.4/79.7) and at the same K behind
  DFlash2 by ~20% ⇒ **production continues with `--spec dflash2`**; enabling speculation costs almost no prefill (within 1%), DFlash2 prefill is consistently
  3–4% slower (the exact source not attributed); prefill 1,905→569 tok/s (0→245k) = a depth-independent fixed term (0.525 ms/token, of which
  AR ≈65% is already at the Gen4 x4 link floor with no software headroom) + a depth term (at 245k, attention accounts for 70% of the per-token cost); the GQA 6×
  request redundancy was **disproved** by the measurement gate (KV bytes varying 3.5× across elapsed time differ by only 13%; ncu: DRAM 1%, tensor pipeline 63.5% busiest, occupancy pinned at 33% by the register
  file + ~100 KB smem) ⇒ L1 (GQA-aware prompt kernel) cancelled, kernel code untouched.
- DFlash2 draft quantization r54–r57 (2026-09-23, opt-in): draft gate/up→Q4 (shard 0 −450 MiB), fused finish projection
  mlp/down + attention/output→Q5 (shared finish kernel + new q5 shape, −213.3 MiB), feature_projection→Q5
  (−50.8 MiB); the merged experimental artifact `…_dflash2_draftall.ninfer` −715.6 MiB, K=7 acceptance rate 24.59%→25.3% (no drop),
  throughput on par ⇒ **decision: stay opt-in**, official recipe unchanged.
- DSH/agent-loop prefix reuse (2026-09-24): the TP-2 route silently ignoring `--chat-template` was fixed (frontend
  `chat_template_path` forwarded, developer role 400→200); the template `|trim` assumption was **measured and disproved** (after overriding the maintenance template,
  replaying as-is still stopped at the previous round's prompt length); a resident two-round test case and documentation were added.
- First generated token not published (2026-09-24): the TP-2 prefill first token used to be `push_back`ed straight into the ledger, bypassing the output session (the only
  route that missed it) ⇒ now `preview_model → commit → publish`, and `max_tokens=1` publishes normally (reasoning_tokens=1);
  replaying a 40/1400 token answer goes straight to `src=live` (previously it always stopped at the prompt length); new test case `check_first_token_published`.
- Reuse gating and draft discarding (2026-09-24): the premise "unaligned reuse must discard the masked draft" was **disproved by measurement** (same round: discarding 22.5 vs
  keeping 87.1 tok/s, acceptance rate almost unchanged) ⇒ the gate, `draft_context_declined`, `reuse_grid` and session-average pricing were removed,
  and the single-pass scan's deepest boundary wins; contract: grid alignment is token-for-token consistent with the oracle, and only boundary crossing is guaranteed within a chunk.
- DFlash2 draft second-round quantization r62–r66 (2026-09-24, opt-in): feature_projection (r62), finish projection
  mlp/down + attention/output (r63, q4 reusing the ordinary GEMM + shared finish, no new fused kernel needed), kernel_projection
  (r64, reusing the ordinary q4 GEMM + new reduce) → q4; the r66 merged artifact `…_dflash2_q4all.ninfer` −173.05 MiB (byte-level
  accounting exactly equal), sampling acceptance rate K=5 +3.72pp / K=7 +1.15pp (both pass); greedy K=7 −1.60pp proven to be cross-arm text
  drift confusion (6 of 7 probes have different text); r65 (context_key) deferred after measurement; both the w8a8 + w4a4 family artifacts have been produced, cross-family
  66 dflash2 objects byte-identical. **Decision: stay opt-in** — the upgrade is blocked by the pre-existing plain sessions failure (§3.3) and the gate-revision
  decision (§3.7).

---

## 3. Unfinished items

### 3.1 Pre-delivery wrap-up

- [ ] Re-measure concurrency C=1/2/4 and streaming/stop smoke with the recommended configuration (the Round 31/34 conclusions were based on the bf16 configuration).
- [ ] On delivery remove this file (PLAN.md) per AGENTS.md; decide the retention/cleanup scope of the ~300 one-off diagnostic scripts under `tools/tp_bootstrap/`
  (`build_r35.sh`, `serve_*`, the `r37+r38+r39+r4x` series and the references in `docs/tp2-dual-5060ti.md` as
  reproducible procedures).

### 3.2 Windows single-GPU CLI tiny-artifact subtask (user decision; prerequisites ready)

Goal: produce a `.ninfer` small enough to fit on a single 16 GiB card, and run "load → generate → sample" end to end on a single GPU with Windows `ninfer.exe`.
Prerequisites (Round 54.9): CPU-only torch in `build-win\torch-venv` (torch 2.14.0+cpu); the converter interface has been confirmed.
Next steps: ① run the official recipe directly on a tiny checkpoint (`--recipe qwen3_8_27b_nvfp4 --components text`);
② if that fails, write a tiny recipe/override; ③ put the generation script at `tools/win_port/tiny_model.py` (synthetic tiny config +
random bf16 weights, HF naming); ④ on success write it into `docs/windows.md` and commit.

### 3.3 Unfixed defects and known limitations

- The TP-2 route `bf16` KV + `--spec mtp` crashes on the first prefill (`prompt.cu:63` `cudaErrorInvalidValue`);
  fp8 + MTP is normal (the recommended configuration is unaffected). Investigation entry point: the launch parameters of bf16 prompt-attention.
- The MTP route near `execution/text.cpp:486` likewise uses a BF16-add allreduce to move I32 proposal ids ⇒ when it lands on a
  signaling-NaN bit pattern it is silently rewritten (DFlash2's equivalent exchange has been changed to `DevicePair::sendrecv` byte-wise; MTP has not been changed,
  and the verification cost is high).
- The MTP3 ≥ 70 tok/s threshold is not reached (pure decode ceiling K=2 50.5–54.0).
- The TP-2 route has not been through a perplexity evaluation (the quality evidence is same-prompt multi-sample A/B).
- The per-shard arena's `memory_summary()` still reports `pages 0/0` (a display-only issue).
- Temperature 0 does not guarantee a bitwise argmax: DFlash2 sparse acceptance + a per-request random seed (`translate.cpp:51-57`) make decoding
  nondeterministic even at temperature 0 ⇒ add `--greedy` when a reproducible A/B is needed (existing behavior, unchanged).
- `ninfer_qwen3_5_tp2_sessions_test` plain route "a conversation behind a shared system prompt" first-sample divergence:
  a pre-existing failure (reproduced token-for-token with both the r57 baseline artifact and the r66 q4all artifact, `got [2752 13 198 …]` vs `expected [365 2798 349 …]`;
  the dflash2/mtp routes pass) ⇒ blocks "promoting draft q4 to a contract feature" (the trio must be all-green).

### 3.4 Deferred / optional (unscheduled)

- Round 15 remainder: end-to-end re-measurement of tool-call constraints on the real model (75 probes); WARN diagnostics; route B (full GBNF);
  the TP-2 `validate_licensed_tokens` guard.
- Round 14 candidates (user deferred): ① add an anchor at the end of the system+tools prefix (fixes the 46.9 s first round after a new session/compaction, saving about 11 s);
  ② prefill throughput (suffix 0.95–1.13k tok/s, cold start 1.35–1.38k); ③ a ~0.3 s fixed overhead per request.
- Archive §11.6: MTP priming uses placeholder embeddings for the image columns (~100 lines, no evidence of benefit); the single-image limit 8,192 is hardcoded
  (yielding shard 1 would regain a larger envelope); shard 0 is idle during visual encoding (acceptable).
- Archive §6: completing the KV dtype scan (nvfp4/k8v4 only verified startup and throughput, no quality data).
- To truly replicate the official artifact's DFlash2 draft from the source weights, one would first need to obtain its BF16 source checkpoint (this machine has only FP8/EXL3/GGUF).
- r65 context_key → q4 (−79.7 MiB): all 5 of the draft's fused attention QKV objects alias the same [6144,5120] object, and both consumers,
  decode (`attn_input_proj` three-output) and prefill (`context_kv_materialize` the 7-route fused kernel),
  hard-require Q8 ⇒ requiring a family of 4-bit fused materialization kernels + a three-output q4 variant + two sets of oracles; acceptance-rate risk is the highest of this batch
  (the K projection error accumulates token by token inside the 2048-window ring); fallback on failure = keep q8.
- L2: AR and MMA overlap (ring staging + event sync + sub-block pipeline): shallow-layer ceiling ~1.5× (1,905 → ~4,760 tok/s ceiling);
  a large refactor, with cross-card rendezvous deadlock risk (already assessed in worklog Round 35c).
- If attention is still to be attacked: only compute-path efficiency remains (FP8 PV + reducing the non-tensor FP32 passes), and profiler permission must first be unlocked
  (ncu 2025.4.1 is installed, `ERR_NVGPUCTRPERM` is rejected; an administrator PowerShell or NVIDIA Developer Settings can allow
  GPU performance counters). Expected attention item ~1.2–1.4× (end-to-end at 245k ~1.1–1.3×).
- Hardware lever: move card 1 from the chipset Gen4 x4 slot (~7 GB/s) to a CPU-direct Gen5 x8 slot (~20 GB/s): zero code,
  1.78× at shallow/medium context, 1.15× at 245k; requires a free slot on the motherboard.
- The DFlash2 K=6 per-round cost dip (three rounds held 25.7 rounds/s, below both the narrower K=5 (27.9) and the wider K=7 (26.6)):
  suspected kernel boundary effect from bucketing/aligning by window width; use `NINFER_TP2_TIMING=1` to get per-phase per-round timings.
- The exact source of DFlash2 prefill being consistently 3–4% slower (no per-phase attribution done; `NINFER_TP2_TIMING=1` breaks prefill into per-phase timings).

### 3.5 Items blocked by the environment (covered by alternative paths)

- [ ] WSL-side op tests + byte-identity + frontend fixture tests: **WSL2 CUDA driver crash** (`cudaGetDeviceCount()` PTX JIT
  segfault, the GPU is held by a Windows service); the jinja tests passing proves the binary is fine, so the op tests were moved to the Windows side.
  Windows side already passes (NVFP4 A4/A16, frontend, jinja; the error metric under `NINFER_OP_REPORT_STATS=1` is within tolerance).

### 3.6 DFlash2 on TP-2: remaining acceptance and design constraints

- [ ] B7 performance acceptance: dual-card throughput against the target (roofline-corrected ≈108 tok/s) + comparison with MTP + no regression for plain/mtp, and record
  the recommended K for `--spec dflash2`.
- [ ] Unknowns still to be measured: the true free memory of the Windows native build (the ledger's `free` comes from `cudaMemGetInfo`, and WSL2 under-reports by about 1 GiB);
  the numerical equivalence of splitting the draft by head/row; the workspace peak at K=15; the per-step cost of gathering the target's 5-layer residual across shards;
  the latency of the draft's per-layer allreduce; the equivalence of sparse rejection sampling under TP-2 sharded logits.

**Design constraints (do not investigate again)**: copying the draft in full is infeasible — at full context (262144 / fp8 / MTP K=2) free is only
697/1217 MiB, whereas the draft is 2.07 GiB per card; after splitting it is about 1.04 GiB per card. The proposed condition is a concatenated residual projection of the target's 5 blocks;
TP-2 splits the 64 layers across two cards ⇒ a cross-card handoff is needed (the two cards' residuals are bit-identical, so capturing shard 0 alone suffices). A candidate selector reads the
full-vocabulary codebook directly (pred/succ each 248320×256 BF16, about 254 MiB) and must be complete. DFlash2 and MTP are mutually exclusive: choosing DFlash2
frees shard 0's MTP weights 430 MiB + KV 516 MiB. The per-round cost composition, the D1–D4 decisions and the per-round negative conclusions are in the archive.

### 3.7 draft quantization: upgrade gate (stay opt-in until satisfied)

- [ ] Fix the plain route `sessions` pre-existing failure (§3.3) — the trio must be all-green before upgrading. The r66 q4all artifact has been measured: `append`
  K=7/K=5 PASS (golden values rebaselined), `solo` two processes with the same digest `0x4bcc3994a5efba7d`, `sessions` fails only on plain;
  the dflash2 route run targeted PASSes (164.8 s, recall 71 tokens byte-identical).
- [ ] Gate-revision decision: make the acceptance-rate arbitration sampling-led — the literal gate "greedy |Δacc| ≤ 1.0pp" was not passed at K=7 (−1.60pp),
  proven to be cross-arm text drift confusion (6 of 7 probes have different text; on the only text-stable one, idx 5, −3.75pp, a small-sample caveat);
  pooled sampling K=5/K=7 both rose (+3.72pp / +1.15pp). The gate revision and the upgrade are decided separately.
- (Q4 op oracle: the new shapes 5120×25600 / 5120×4096 / 5120×17408 / 1280×5120 are already covered by
  `ninfer_linear_q4_a16_test` / `ninfer_linear_dynamic_grouped_conv_add_test` with independent decoding per stored scale,
  including graph replay.)

Remaining levers (unscheduled):
- selector codebook quantization (242.5 MiB BF16, not a GEMM): requires a new selector path (codebook codec + preserving the
  top-k domain); and the parameter is a bare `Tensor` rather than a `Weight`, so the Tensor→Weight pipeline + the TP-2 peer path must also be touched.
- Full-draft NVFP4: **a source-level feasibility audit has been done and the decision is not to proceed** (see `tp2-dflash2-draft-nvfp4-feasibility.md`).
  Key points: relative to the active r66 `q4all` artifact it would save only another 202.3 MiB, of which 174.3 comes from the codebook (not a GEMM); NVFP4 per element
  is 0.5625 B > Q4's 0.53125 B, so converting the already-Q4 gate_up/output/down/kernel/feature to NVFP4 would increase size; three
  hard blockers = the converter has no float→NVFP4 quantizer and the draft source (dynamic activation FP8) has no A4 activation-divisor calibration, the NVFP4 native Weight
  only accepts a complete parent (conflicting with the fused QKV's row-view sharing), and every consuming op's NVFP4 geometry/kernel is a closed set
  (`linear_pair` only serves dflash v1/MTP, not on the DFlash2 path). Re-open only if the motivation shifts to "unify the W4A4 family / prefill
  TMA".
### 3.9 DFlash2 draft q4 second increment: ② fused QKV→q4 / ① codebook quantization (started 2026-09-24)

> Basis: audit of `tp2-dflash2-draft-nvfp4-feasibility.md` (all-draft NVFP4 was rejected; only these two real VRAM levers remain).
> Decision: both remain opt-in override + dedicated artifact; the official recipe and the engine's default path are untouched.

**② Fused QKV → q4 (−79.7 MiB) — key design correction: no new fused kernel needed**

The audit found that the q4 GEMM registry **already** contains `{4096,5120}`/`{1024,5120}` (`src/ops/linear/q4/q4_dispatch.cpp:14-15`), and that RowSplit q4's
row-region is supported end-to-end by `native_weight` (`src/core/weight_view.cpp:271-284`) ⇒ once the fused [6144,5120] QKV object is lowered to q4,
both Q8-only consumers can route around it with existing ops:

- decode (`draft.cpp:326`): Q8 goes through the fused `attn_input_proj` three-output; q4 instead goes through **3× `ops::linear`**
  (query/key/value row views, shapes already in the registry).
- append/prefill (`draft.cpp:157`): Q8 goes through the fused `context_kv_materialize`; q4 instead goes through the **existing per-layer generic branch**
  (2× `ops::linear` + `rmsnorm`/`rope`/`kv_cache_append_prefix`), and that branch's reuse of the dflash2-trimmed context is fixed
  (the original implementation was correct only for dflash v1; v1 is already rejected at construction). Cost: ~20 extra launches per round, estimated ≪1%, verified by the A/B gate.

Change list: `parameters.h`/`parameters.cpp` (DraftBlockParameters gains query/key/value row views; `query_key_value` becomes
`std::optional`, constructed only for the Q8 parent), `draft.cpp` (branch by qtype + generic-branch fix), override
`tools/tp_bootstrap/r67_draft_qkv_q4.py` (**exactly 5 objects** count check; the official recipe's share puts
query/key/value/context_key/context_value in the same object).
Gates: the stock Q8 route is bit-identical; the existing q4 op oracle already covers the shapes used (no new kernel oracle needed); VRAM ledger −79.7 MiB;
greedy K=5/7 + sampling acceptance rate A/B; the DFlash2 three suites.

**② Progress (2026-09-24)**: code landed, `BUILD_EXIT=0`; the stock Q8 artifact (`…_w8a8_dflash2_q4all.ninfer`)
append regression **PASS** (ring/proposal digest bit-identical to baseline, zero perturbation of the Q8 route). The r67 experimental artifact has been converted:
`<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_qkv4.ninfer` (1218 objects, 379.5 s; payload **−79.69 MiB**,
q4 81→86 / q8 12→7, exactly 5 `[6144,5120]` q4 objects; dry-run selection q4 46 / q8 10 / bf16 35).
The qkv4 artifact's append functional verification **PASS** (`TEST_EXIT=0`: direct/split append bit-identical, repeat bit-identical,
K=7 proposal deterministic, ring K identical value-for-value to the Q8 baseline, V difference at the q4 quantization magnitude ≤1.5%); proposal cost
7.036 vs 6.854 ms/proposal (+2.7% on the proposal step, ≈+0.25% end-to-end, pending A/B measurement).
Incidentally fixed the append test's draft-weight ledger: the fused QKV parent and its context row views are the same allocation,
now counted once per parent (qkv4 arm layers **843.9 MiB**; q4all arm on the same basis **923.6 MiB**, difference 79.7 MiB
consistent with the artifact byte difference — the old basis counted the parent 3 times, inflating gross by 318.7 MiB).
Remaining: greedy K=5/7 / sampling acceptance rate A/B, solo digest and the sessions three suites, and end-to-end throughput A/B.

**②-e Acceptance A/B (2026-09-25, same binary; baseline = r66 `q4all` artifact)**:

| Metric | q4all (Q8 QKV) | qkv4 | Δ |
|---|---|---|---|
| Greedy K=7 | 26.95% (720/2672) | 27.07% (722/2667) | +0.12 pp |
| Greedy K=5 | 35.96% (712/1980) | 34.85% (703/2017) | −1.11 pp |
| Sampling K=7 (15 run pooled) | 33.87% (2678/7907) | 33.23% (2663/8014) | −0.64 pp |
| Sampling K=5 (15 run pooled) | 46.50% (2667/5735) | 44.34% (2626/5922) | −2.16 pp (standard error of the difference ≈0.92 pp) |
| Greedy throughput | 59.0 / 59.2 tok/s | 59.4 / 57.9 tok/s | within noise |

Sampling K=5 > K=7 matches expectation (a shorter draft has a higher acceptance rate). The **first** K=7 run of the q4all sampling arm crashed at request 14:
`tp2_generation_core.cpp:3119`'s D2H `cudaMemcpyAsync` reported `cudaErrorIllegalAddress` (sticky — the fault was in an earlier
DFlash2 verify/accept kernel, and the reporting point is merely the first API boundary); a **rerun of the same arm passed 15/15**, and that artifact's draft
execution path is untouched by this change (Q8 QKV still goes through the fused `attn_input_proj`/`context_kv_materialize`), so it was judged to be
a **pre-existing TP-2 race flake** (same class in worklog history: Round 36d `tp2_generation_core.cpp:443` D2H, same symptom),
recorded for dedicated investigation; this round's conclusions use the rerun data.

**Three suites (qkv4)**: append-K7 / append-K5 / solo all **exit=0**; the solo digest is identical across two processes
(`0x4bcc3994a5efba7d`), and **bit-identical** to q4all — under greedy acceptance the draft only changes the number of tokens produced per round, not the token sequence,
so this is expected; the append proposal digest differs from q4all (`0xf9622ee5cd45709d` vs `0xda91572dd83980bd`), as expected.
`sessions` under the **default (all three routes)** reports `FAIL (plain): a conversation behind a shared system prompt diverged …`,
but that failure is **unrelated to this change and to draft precision**: the q4all arm's output tokens are **bit-identical**, and the pre-change
`sessions.log` (baseline artifact) in `build-win/r66/` and `sessions-baseline.log` are the same token string; narrowing the route to
`NINFER_TEST_ROUTE=dflash2` makes `sessions-dflash2-*.log` PASS every time. ⇒ The sessions gate for this campaign is always run on the dflash2 route (as it was for r66).

**②-f 30-rep high-power re-test (2026-09-25, same binary, adjacent rounds of the two arms) — overturns the 15-rep conclusion above**:

| Metric | q4all | qkv4 | Δ | se(Δ) |
|---|---|---|---|---|
| Sampling K=7 | 35.13% (16234/46207) | 34.99% (16208/46326) | **−0.14 pp** | ≈0.31 pp |
| Sampling K=5 | 45.42% (15865/34927) | 45.25% (15852/35029) | **−0.17 pp** | ≈0.38 pp |
| proposal cost (CUDA events) | 6.851 ms | **7.044 / 7.031 ms** | **+0.19 ms (+2.8%)** | deterministic |

The acceptance-rate cost is **not measurable** (both levels <0.5σ); the 15-rep "−2.2 pp" is the same magnitude as the baseline's own 0.6–1.1 pp run-to-run drift.
The three suites rerun under the **current binary** still all pass (`SUITES_DONE failures=0`, 5/5 exit=0; solo digest bit-identical to baseline;
append proposal digest K=7 `0x8103f572fb2d2f99` / K=5 `0x96582e7289dd2702`). ②'s real cost is only the deterministic
proposal step +2.8% (+0.19 ms/round; at ~45 ms/round that is ≈ **+0.4% end-to-end**).

**② Conclusion (corrected)**: ② is a "**−79.69 MiB for ≈+0.4% round time, no acceptance-rate cost**" neutral-to-positive lever, not the
"losing 1.3–3.2% throughput" judged in the previous round. Compared with ① (−178.09 MiB, zero cost), ① is strictly better; whether ② enters
the default depends on whether the VRAM pressure is worth that 0.4% round time. (Methodology: acceptance-rate verdicts in this campaign must use 30-rep; the 15-rep
inter-arm noise floor is ≈±2 pp.)

**① selector codebook quantization (−174.3 MiB, non-GEMM)**

Status quo: the codebook is a bare `Tensor` (`parameters.h:111-114`), the only bf16 path (`src/ops/candidate_selector/bf16/`,
`wrapper/candidate_selector.cpp:110-113`), 248320×256×2 = 242.5 MiB; TP-2 places it on shard 1
(`tp_split_spec.cpp:97-104`) and the peer runs the selector on its behalf (`text.cpp:973+`). The selection domain is a top-16/248320 inner product over 256 dims.
Design: add a codebook codec (target level TBD: Q4_G64 is smallest per element, NVFP4 has one scale per 16 weights and better ordering fidelity),
keep the top-k domain; `SelectorParameters` (`parameters.h:116-119`)'s codebook is upgraded from a bare `Tensor` to a representation carrying a qtype,
and is then independently decoded by the quantized overload of `candidate_selector_path` using the stored scales. Two ABI constraints identified:
① the loader currently pins the format exactly with `b.direct(..., BF16)`, and `Binder::binding` throws when `exact_format` does not match
(`src/artifact/binder.cpp:53`) ⇒ the codebook parameter must become "not format-pinned" and choose its representation from the artifact's actual geometry;
② under TP-2 the codebook is on shard 1 and consumed by the peer's `dflash_propose_batch` (`text.cpp:973+`), so the quantized representation must likewise cross the peer.
First do an op-level oracle (top-16 set/scores vs the bf16 baseline), then end-to-end A/B (acceptance rate is a hard gate — quantization error directly changes the proposal set).

**① Progress (2026-09-25)**: the op-layer implementation is complete. The format-agnostic plan was moved up from `bf16/` to
`src/ops/candidate_selector/plan.{h,cpp}` (new `candidate_selector_path_q4_dispatch`, with the workspace non-overlap check
now also covering both codebook planes); new `src/ops/candidate_selector/q4/candidate_selector_path_q4.cu`, decoding independently
per Q4_G64_FP16 RowSplit — `code(r) = nibble (r&1) of byte (r>>1)` (nibble semantics `(n^8)-8`),
scale = `fp16(scales + token*8 + (r>>6)*2)` (`padded_columns=align(256,128)=256`, `code_bytes_per_row=128`,
`scale_offset=align(n*128,256)=n*128` verified against `weight_geometry`); the public header gains a `Weight` overload,
and the wrapper validates qtype/layout/group/n/k/scale_dtype and payload non-overlap. `ninfer_ops` and
`ninfer_candidate_selector_test` both compile and link (exit 0). The test gains a q4 oracle arm: on the host it constructs
`value = code * fp16_scale` and packs the nibbles, the FP64 oracle **independently decodes from the stored scales**, covering K=3 (Direct)/
K=7 (Lattice) × B∈{1,3,8}. That test has run **PASS** (`OK candidate_selector_path`, `TEST_EXIT=0`) —
q4 decoding and the FP64 oracle (independently decoded from the stored fp16 group scales) are bit-identical on both routes.
**①-b Progress (2026-09-25)**: loader / parameters / peer are wired through. `load/dflash2.cpp`'s codebook changed from
`b.direct(..., BF16)` to `b.parameter(...)` (no format pinning; `direct` in `bindings` was already
a special case of `parameter(..., {}, format)`, so only the format pin was removed); new `SelectorCodebook{dense, weight, quantized}`
representation and `Prepare::codebook()` (choosing by artifact geometry between BF16→`tensor()` and Q4→`prepare_linear_weight().weight`,
throwing on an illegal format); new `execution/selector.h::run_candidate_selector()` as the **single dispatch point**,
with `draft.cpp` (the shard holding the selector) and `text.cpp` (the TP-2 peer) both going through it, so the quantized representation crosses the peer naturally
with `SelectorParameters`. **stock BF16 codebook regression**: the q4all artifact's append test digest is **bit-identical** to baseline
(proposal `0xda91572dd83980bd`, ring `0x371f934bfbd6c7ec`), `TEST_EXIT=0`.
The r68 experimental artifact `<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_cb4.ninfer` has been converted (1218 objects, 379.8 s;
payload **−178.09 MiB**, q4 81→83 / bf16 569→567, exactly 2 `[248320,256]` q4 codebooks; dry-run
q4 33 / bf16 33 / q8 25; does not include r67's QKV change). The first implementation at load time hit `Model::input(WeightId)` requiring
"exactly one Use" (`model.cpp:23-28`) while the codebook is a bare direct parameter — changed to taking the Weight from the full
parent directly via `native_weight(bound.view)` (no Use needed), still with the same geometry check. **cb4 end-to-end verification**: append test
`TEST_EXIT=0` (direct/split append, ring repeat, proposal determinism all pass), and
`peer selector: on shard 1` went from **245.0 MiB → 66.9 MiB** (−178.1 MiB, consistent with the artifact byte difference),
proposal cost **6.854 ms/proposal** (baseline 6.851, within noise).

**①-c Result (2026-09-25, same binary; baseline = q4all)**: sampling pooled (15 run × 3 prompt classes).

| Metric | q4all | cb4 | Δ |
|---|---|---|---|
| Sampling K=7 | 34.62% (2697/7790) | 34.37% (2691/7829) | −0.25 pp |
| Sampling K=5 | 45.59% (2649/5810) | 43.40% (2609/6012) | −2.19 pp (se ≈0.91 pp) |

The baseline itself is drifting too: the same q4all artifact was K=7 33.87% / K=5 46.50% in the r67 round and 34.62% / 45.59% this round
⇒ the 15-rep pooled inter-arm drift is about ±1 pp, and K=5's −2.2 pp is only ~2.4σ, **not enough to conclude**. The three suites **all pass**
(append-K7/K5, solo×2, sessions-dflash2, `SUITES_DONE failures=0`); the solo digest is bit-identical to baseline
(the greedy sequence does not change with the draft); append's K=5 proposal digest is **identical** to q4all (`0x5a3629fb79be1cd3`),
K=7 differs (`0x766be9d7643131e2`) ⇒ codebook quantization has very little effect on candidate selection for this fixture.
**①-c High-power re-test (30-rep, 90 run/arm, same binary) — the 15-rep −2.2 pp was noise**:

| Metric | q4all | cb4 | Δ | se(Δ) |
|---|---|---|---|---|
| Sampling K=7 | 35.77% (16329/45644) | 35.74% (16309/45629) | **−0.03 pp** | ≈0.32 pp |
| Sampling K=5 | 44.36% (15754/35510) | 45.95% (15928/34666) | **+1.59 pp** | ≈0.38 pp |

The 15→30-rep estimates of the two arms themselves drifted 1.4–2.6 pp, showing that the inter-arm noise of 15-rep pooling is ~±2 pp, so the earlier r67/r68
"−2 pp acceptance-rate cost" was noise of that magnitude and cannot be used as grounds for rejection. Under 30-rep there is no difference at K=7, and at K=5 the direction is actually positive
(+1.6 pp is more likely a residual arm effect than a real improvement, but at least there is **no measurable cost**).

**① Conclusion (recommend adoption)**: 178.09 MiB (artifact **−0.75%**, runtime peer selector 245.0→66.9 MiB)
for zero measurable acceptance-rate cost and zero proposal overhead; op oracle, three suites, and the stock BF16 bit-identical regression all pass. **The comparison between ② and ① was exactly the
misjudgment caused by the A/B noise floor (~±2 pp)** — both have a real effect on acceptance rate smaller than the resolution of this campaign's 15-rep scheme, so
the "save VRAM, lose throughput" conclusion simply does not hold for ① (① is refuted), while for ② a re-test is required before concluding (see below).

**r69 promoted to official recipe + final artifact (2026-09-25, confirmed by the user)**: `tools/convert/official_recipes.py::_optional`
now carries the complete DFlash2 draft Q4 set — r62-r66's MLP (gate/up/down), attention output, the two dynamic convolution kernels,
`feature_projection`, plus **②**'s fused QKV (`dflash2/layers/*/attention/{query,key,value}`, one
`[6144,5120]` parent object per layer, consumed via row views) and **①**'s two selector codebooks; all restricted to `dflash2/`,
with dflash v1 and MTP still Q8 (`DFLASH2_Q4_NAMES`/`DFLASH2_Q4_ROLES`/`DFLASH2_CODEBOOKS`).

The final artifact `<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer` (**no `--override`**, 1218 objects, 385.7 s,
`CONVERT_EXIT=0`): compared parameter by parameter with the `q4all` baseline, **exactly 17 parameters** changed format (15 QKV + 2 codebooks,
all → q4), and the remaining **1486 parameters' formats and shapes are fully identical** ⇒ the official recipe exactly reproduces the r66 artifact, final = r66 + ① + ②.
payload **−257.77 MiB** (q4 81→88 / q8 12→7 / bf16 569→567).

**r69 final acceptance (30-rep, 90 run/arm, adjacent rounds of the two arms)**:

| Metric | q4all (r66 baseline) | final (r66+①+②) | Δ | se(Δ) |
|---|---|---|---|---|
| Sampling K=7 | 35.09% (16223/46228) | 34.79% (16188/46530) | **−0.30 pp** | ≈0.31 pp |
| Sampling K=5 | 44.99% (15828/35181) | 44.88% (15812/35233) | **−0.11 pp** | ≈0.37 pp |

Both levels ≤1σ ⇒ **no measurable acceptance-rate cost**. The three suites **all pass** (`SUITES_DONE failures=0`: append-K7/K5, solo×2,
sessions-dflash2); the solo digest is bit-identical to baseline (`0x4bcc3994a5efba7d`); runtime **peer selector 66.9 MiB**
(baseline 245.0), proposal cost **7.029 ms** (baseline 6.851, i.e. ②'s +0.18 ms ≈ +0.4% round time).
⇒ **The final artifact is releasable**: it saves **257.77 MiB (−1.08%)** versus the r66 baseline, at no acceptance-rate cost and ~0.4% round-time cost.

**Two low-frequency failures during the process (unrelated to this change, pending dedicated investigation)**: ① in the r67 round, the q4all arm's K=7
`cudaErrorIllegalAddress` (a rerun of the same arm all passed, judged a pre-existing TP-2 race); ② in this round, the final arm's K=5 first run, request 14,
`HTTP 503 service unavailable` with no FATAL log (a rerun passed 90/90). Both appeared only in sampling arms of ~90 consecutive requests,
about 2 out of ~15 arms in total.

**Final artifacts for the remaining Qwen3.8-27B variants in r69/r70 (2026-09-25)**: the same official recipe (no override) + the shared draft source
`<models-dir>/W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8`, one artifact per variant, each containing `text,vision,mtp,dflash2`:

| Variant | Source | Final artifact | Size | Text representation | Conversion |
|---|---|---|---|---|---|
| W4A4 | `W4A16/NVFP4/W4A4` | `qwen3_8_27b_w4a4_dflash2_final.ninfer` | 18.42 GB | nvfp4 256 / fp8 2 | 336 s |
| Swift-1.5 | `Swift-1.5-Qwen3.8-27b-NVFP4` | `qwen3_8_27b_swift15_dflash2_final.ninfer` | 21.58 GB | fp8 130 / nvfp4 128 | 413 s |
| ThinkingCap | `ThinkingCap-Qwen3.8-27B-NVFP4A4-AWQ` | `qwen3_8_27b_thinkingcap_dflash2_final.ninfer` | 22.52 GB | fp8 146 / nvfp4 112 | 388 s |

All four artifacts (including w4a4+w8a8) have fully identical draft profiles (q4 88 / q8 7 / bf16 567), and the append smoke test **all PASS**:
`peer selector on shard 1 = 66.9 MiB`, proposal cost 7.03–7.08 ms, K=7 proposal determinism.

**ThinkingCap's tokenizer pitfall (fixed)**: that export was serialized by a different `tokenizers` version, triggering **three** validation failures
in the engine frontend (the first two were exposed immediately by the append smoke test; the third triggers only inside `compile_chat_template` at **serve startup**):

1. `tokenizer.json`'s Split pre-tokenizer uses `\p{L}+` (without `\p{M}`) and `trim_offsets=true` → the engine's Qwen
   tokenizer supports only the canonical form, reporting `pre_tokenizer.Split is not supported`.
2. `tokenizer_config.json` lacks `added_tokens_decoder` (
   required by [tokenizer.cpp:288](../../src/models/qwen3_5/frontend/tokenizer.cpp)) → `missing field
   added_tokens_decoder`.
3. `tokenizer_config.json` lacks `add_bos_token`, while
   [frontend.cpp:196](../../src/models/qwen3_5/frontend/frontend.cpp) writes `value("add_bos_token", true)` —
   **a missing field defaults to true**, so it hits the prefix-semantics validation → `FATAL: tokenizer_config.json does not match Qwen3.5
   tokenizer prefix semantics`. The canonical export writes `add_bos_token: false` explicitly, so the other three artifacts are unaffected.

Fix: that export's `vocab.json`/`merges.txt` are **byte-identical** to the canonical export, and the canonical export's 33
`added_tokens_decoder` entries are **fully consistent** with its `tokenizer.json`'s added_tokens (the 13
`additional_special_tokens` are all among them), so the canonical `tokenizer.json` was substituted, and
`added_tokens_decoder`/`add_bos_token`/`additional_special_tokens` were merged into that export's own
`tokenizer_config.json` (keeping its model-specific fields), with the conversion script asserting the engine frontend's three rules internally. See
`tools/tp_bootstrap/r70_convert_thinkingcap.ps1`. **End-to-end verification**: `serve.ps1 -Spec dflash2 -DraftTokens 7`
started successfully (`engine ready | qwen3.8-27b-thinkingcap | 33.9s`, `listening on 127.0.0.1:8098`, no FATAL), and
passed an actual Q&A over HTTP `/v1/chat/completions` (HTTP 200, 50 completion tokens).

**Execution order**: ② first (changes concentrated, verifiable with the existing oracle), ① in parallel doing design/op oracle.

### 3.8 Prefix reuse: remaining issues (2026-09-24)

- Text-protocol replay cannot guarantee per-token reproduction: the first generated token problem has been fixed with "the first generated token is not published" (replay answers now go straight to
  `src=live`), but when the client re-tokenizes differently or the sampling boundary differs, or the boundary falls inside a chunk (the contract only guarantees boundary crossing),
  tail reuse still degrades; the only lever that guarantees per-token is a continuation protocol carrying token ids (not scheduled).
- TP-2 prefill's first sample does not apply the tool-grammar mask (the prefill and decode paths are asymmetric, a separate leftover item).

### 3.11 TP-2 orchestration and handshake: three concrete pieces of work (started 2026-09-25)

> Origin: the user asked "is it worth doing an overall rewrite with the goal that 'all components support low-coupling TP-2'". The conclusion is **not to do an overall rewrite**,
> only three concrete things with clear benefit. Quantitative basis: the shard plan is already centralized (`tp_split_spec.cpp` 175 lines, shard mask at only 2 sites),
> and the bulk of the giant `execute_walk` (1412 lines/12 lambdas) is the session and checkpoint state machine rather than TP; the bottleneck is target verify.

**② Graph wiring: review conclusion = already done, what remains is B7 acceptance**

The worklog 4128-4131 I based this on (2026-09-22 "B7 pre-reconnaissance (done, not implemented)") **is out of date**. All four places are wired in the live code:

| Listed by the reconnaissance | Live code |
|---|---|
| ① The master switch only takes effect for MTP | `tp2_generation_core.cpp:374` = `(mtp_enabled_ || dflash2_enabled_) && in_kernel_allreduce()` ✓ |
| ② The verify bucket is only created in the MTP branch | `:387-397` both routes create `verify_window_host_`; `:398-409` profiles go through
`dflash_graph_profiles(DFlash2,…)` ✓ |
| ③ `capture_verify_graph` takes no sink | `:954-958` the parameters now include `DFlashFeatureSink* sink` + `valid_columns` ✓ |
| ④ The throw at `:979-984` | Has been replaced by the forwarding in the capture body `:980-986` (the comment states exactly this conclusion) ✓ |

**Measured confirmation**: this session's DFlash2 service log `build-win/r69/sampling-r69-final-k7.log` prints
`[tp2-graph] plain decode step: graph | verify step: graph | mtp chain: graph` ⇒ production is already running inside graphs.

**Remaining work = B7 acceptance** (the criterion from worklog 4133): paired A/B of `NINFER_TP2_VERIFY_GRAPH=0/1`, verify ≈30.6 ms,
4–4.5 ms/round lower than eager, equal-output throughput ≈+15%. Gate: two-arm throughput + proposal cost + three suites.

**① Close out implicit invariants: two concrete goals (explicitly not rewriting `execute_walk`)**

**1a Eliminate the "silent default argument" class** — corresponding to two real incidents (worklog 4131 sink missed wiring, 4135 `valid_columns`/
`active_valid_columns_` unbound causing a paged-KV same-slot tear):
`TextContext::forward_tp2_window` (`text.h:217-221`)'s last three parameters
`hidden_columns=nullptr` / `sink=nullptr` / `valid_columns=nullptr` — **each one changes semantics**:
the header's own comment (`:211-216`) says "nullptr leaves every column appending", i.e. omitting it = silent degradation to no mask.
Approach: **delete the three defaults**, with the two call sites (`:984` capture, `:1031` eager) passing explicitly; and collect both sides' arguments into a single read-only
struct, making "capture and eager pass different arguments" impossible at the type level (currently kept in sync by copy-paste in two places).
Gate: `BUILD_EXIT=0` + three-suite digest bit-identical.

**1b Make in-window give-up identifiable** (§3.10(d)): in-window spin give-up now manifests as an uninformative illegal address
(injection 700 reproduced it byte for byte). The goal is to be able to determine "give-up in window" rather than "illegal address" on failure.
Gate: injection 700's error is identifiable, and the clean arm emits zero new output.

**③ Naming and declaration payloads**

- The shard mask is now down to **2 sites** (`tp_split_spec.cpp:101,109`); change them to named role constants (e.g. `Shard::DraftHome`).
- Centralize the declaration of the payloads and contracts of the 4–6 handshake protocols (candidate ids/scores union, MTP `pack_forward`/`back`, draft state, AR).

**Order and gates**: ③ → 1a → ②B7 acceptance → 1b. ③/1a touch only structure and parameters, not numerics ⇒ the three-suite digest must be bit-identical;
② uses paired A/B via `NINFER_TP2_VERIFY_GRAPH`; 1b uses the injection point.

**Progress (2026-09-25, this batch implemented and verified)**

- **1a done**: `TextContext::forward_tp2_window`'s last three semantic parameters had their defaults removed
  (`hidden_columns`/`sink`/`valid_columns`), and the header states "each one changes semantics; omitting it means silent degradation";
  the two call sites (capture `tp2_generation_core.cpp:984`, eager `:1031`) already passed explicitly ⇒ it is now a compile-time constraint.
  The subsequent step of collecting both sides' arguments into one struct was not done (diminishing returns, see below).
- **③ done (naming part)**: `src/core/tp/tp_materialize.h` gains `kLocalShard=0x1`/`kPeerShard=0x2`,
  and `tp_split_spec.cpp`'s two bare masks become named roles ⇒ the repo-wide `0x1U/0x2U` count is zero. **The payload-declaration part was not done**.
- **1b done and verified**: `ar_exchange`'s give-up path **no longer leaves the output as the stale value from the previous round** — new
  `ar_abandon_output()`, with all three `stalled` early returns (entry check / block-order wait / arrival wait) first zeroing
  this block's `out` slice. Rationale: that staging buffer is reused round after round, and a replayed captured graph reuses the same arena memory, so leaving old values lets downstream
  treat stale data as an **index** → illegal address inside the window, and no host-side check is fast enough (§3.10 already proved this).
  Zero is a legal token id/candidate/logit, so that round can complete, and the round's convergence point's `abort_if_ar_stalled()` turns it into the designed 503.
  **Verification**: `r72_stall_injection.ps1 -Calls @(560,700)` ⇒ both report `503,ok illegal=False`
  (before the fix, 700 was the **byte-for-byte reproduced** `tp2_generation_core.cpp:3119 cudaErrorIllegalAddress` + process exit);
  zero cost on the healthy route (the fill only executes on the timeout path).
- **Zero numerical impact from ①②③**: `r66_suites.ps1` (dflash2 route, final artifact) **5/5 exit=0**,
  digest **bit-identical** to before the change (K=7 `0x8103f572fb2d2f99`, K=5 `0x96582e7289dd2702`,
  solo `0x4bcc3994a5efba7d`, ring `0xb1a1fe192b36817a`).
- **② B7 acceptance (`r81_verify_graph_ab.ps1`, final artifact, K=7, 6 reps × 3 prompt classes = 18 requests/arm, 4608 tokens/arm)**:

  | Statistic | graph | eager | Δ |
  |---|---|---|---|
  | Per-request decode mean | 76.22 tok/s | 71.90 tok/s | +4.33 (+6.0%, se=4.39, **t=0.99 not significant**) |
  | Per-request decode median | 72.2 | 70.5 | +1.7 |
  | **Aggregate throughput** (Σtoken / Σ request total) | **72.68 tok/s** | **69.09 tok/s** | **+3.60 (+5.2%)** |

  Both arms' `[tp2-graph]` lines confirm the only difference is verify (`verify step: graph` vs `eager`, plain/mtp chain identical).
  ⇒ **the graph really is running and brings about +5% end-to-end decode** (direction matches the worklog's +15% estimate but the magnitude is smaller);
  per-request statistics cannot resolve it at this sample size, and the aggregate basis is usable but still a single round; settling it needs ≥24 reps/arm.

**③ done (payload-declaration part)**: `text.h`, after `set_tp_peer`, gains a **handshake protocol master table** — 9 protocols listing one by one
payload type/shape, transfer method and contract (row-block gather, I32 id broadcast, hidden broadcast ×3, candidate merge, selector
forward/back packing, vision embedding broadcast), and stating the criterion: **the payload decides the transport** — `allreduce` is only for BF16-safe
bit patterns and only as "zero-buffer broadcast / disjoint gather", while `sendrecv` is the channel for ids/FP32/payloads that must be byte-identical.

**And the declaration is turned into an executable check**: `dflash_selector_tp2`'s forward packing first computes
`forward_payload = ids+scores+anchors+frontiers`, and after packing asserts
`packed == forward_payload` (a mismatch in packing/unpacking order between the two sides or a field added/removed throws immediately).
**That assertion immediately caught a real inconsistency**: the packing accumulator missed the final `packed += frontiers.bytes()`
(no accumulation after the last memcpy — a "looks right but wrong the moment anyone depends on the accumulator" shape), now fixed.

**Measured gain outside the batch (natural give-up fails safely for the first time)**: in the first three-suite run after the ③ change, `solo-a` hit a
**natural** give-up (`TP-2 allreduce stalled at rendezvous id 4294967296` = channel 1, base 0, i.e. the first collective of the second graph).
This is the **3rd natural occurrence** in this campaign (the first two are in §3.10: r67's illegal address and r69's 503),
and **this time it was a clean failure**: the process survived, reported the designed failure message, and the watchdog produced no dump because it was not enabled. A rerun was **5/5 exit=0**,
with digest bit-identical to before the change ⇒ this occurrence is a pre-existing intermittent, not introduced by this change; at the same time it is live evidence that 1b, on the natural path (not just the injection path),
turns "a crash" into "a designed failure".

**Not done (intentionally left by this batch)**: 1a's "collect both sides' arguments into one struct" (the two call sites are already explicit and consistent, which suffices at compile time).

### 3.10 Dedicated investigation of low-frequency TP-2 transport failures (2026-09-25)

**Goal**: characterize and, if feasible, root-cause the two low-frequency engine failures under long consecutive runs of the sampling arm.

**Evidence (two instances, the only two in the whole campaign; confirmed by scanning all `.out`/`.log` under `build-win/**`)**

1. **IllegalAddress (2026-09-25 08:20:59, r67 q4all K=7 arm, req#14)**:
   ```
   tp2_generation_core.cpp:3119: CUDA_CHECK(cudaMemcpyAsync(licensed_host.data(),
     frame.licensed_tokens.data, sizeof(TokenId)*width, D2H, shard_a_.device.stream))
     failed: cudaErrorIllegalAddress
   ```
   This is a **misattributed report**: the real out-of-bounds kernel is earlier, and the reporting point is merely the first API boundary at the end of the round. The log has no 503/stall record.
   A rerun of the same arm passed 15/15 and 90/90; the same artifact behaved this way before the change too.
2. **HTTP 503 (2026-09-25 10:34:39, r69 final K=5 arm, req#14)**: request body
   `{"code":"service_unavailable","message":"TP-2 allreduce stalled at rendezvous id 8590057052: the request
   was failed and the reusable context was discarded"}`. **Not a crash**: this is the designed behavior of bounded spin
   (`tp2_generation_core.cpp:1410 abort_if_ar_stalled()`). id = `2<<32 + 122460` ⇒ channel index 1
   (the 2nd created graph), base accumulated 122460 collectives, consistent with "14 requests × K=5" rounds × collectives per round.

**Ruled out**
- Residual environment variables: HKCU/HKLM and the process environment have **no** `NINFER_*`; fault injection
  `NINFER_TP2_AR_FAULT_SKIP_PEER_CALL` is off (`device_pair.cu:327-331` does not count at 0, zero overhead),
  and the timeout is the default **2000 ms** (`device_pair.cu:481`).
- Clock: `ar_now_ns()` uses `%globaltimer` (ns, device-consistent), deadline semantics correct (`device_pair.cu:113-123`).
- Boundaries: `ar_exchange`'s group/tail loops, parity slots (`(token&1)*slot_bytes`), stall flags
  (`stall_host_` 128 B = 2×64 B, `ar_stalled()` reads flags[0]/flags[16]) and the store helper's `lane<=k`
  were checked item by item with no out-of-bounds; `count_bytes <= slot_bytes` always holds (small path 64 KiB / large path 48 MiB — 2026-09-26
  expanded from 24 MiB, see that entry at the end of the document — consistent with their respective guards).

**Mechanism judgment**: one failed spin leaves **that round's local partial sum not updated by the peer** (the kernel returns early),
while the host only checks the trip flag at the **per-round convergence point** (`abort_if_ar_stalled()` call sites 2751/3317).
If corrupted data is consumed as an **index** (candidate/anchor/frontier/argmax) before detection, it becomes a sticky
illegal address ⇒ the two manifestations may be **the same event observed at two moments** (clean 503 vs a crash first).

**Verification results (2026-09-25)**

1. **Not reproduced by long runs**: `tools/tp_bootstrap/r71_ar_stress.ps1` (final/K=5, watchdog enabled, 600 requests,
   continuing to collect after the first failure) — **0 failures** ⇒ the natural rate is below ~1/1350 requests, so long runs cannot be used to investigate it.
2. **Injection reproduces it deterministically, and confirms "the same event"**: `tools/tp_bootstrap/r72_stall_injection.ps1` uses the existing injection point
   `NINFER_TP2_AR_FAULT_SKIP_PEER_CALL=N` to skip one peer launch at a specified collective index:

   | Injection index | Landing point | Result |
   |---|---|---|
   | 20 | warmup (eager) | `FATAL warmup failed … stalled at rendezvous id 1<<62+129`, engine exits, no crash |
   | 280 / 300 / 330 / 360 / 400 / 440 / 480 | after warmup, prefill (eager) | **clean 503 + next request ok** (7/7) |
   | 560 | inside a captured graph (`graph-ch2`) | clean 503 |
   | **700** | inside the verify window | **`tp2_generation_core.cpp:3119 cudaErrorIllegalAddress`**, the **byte-for-byte same error** as the production incident, process exits, connection cut off |

   The watchdog dump also gives baseline facts: warmup ends at `calls=268` (real requests start at 269), and the last healthy
   collective has `arrA==arrB` on both sides (6 slices, large-payload path).
3. **Bisection localizes it: the fault occurs inside the window**. After temporarily inserting an
   `abort_if_ar_stalled()` probe after the failing branch's `run_verify_window(...)` (`3100`) and rerunning injection 700: the probe **did catch the trip**,
   and the error moved from 3119 to **`tp2_generation_core.cpp:1419` (`cudaStreamSynchronize` inside `abort_if_ar_stalled`)**
   ⇒ the illegal access **has already happened inside the verify window (a captured graph)**, and no host-side check is fast enough. The probe has been reverted.
4. **Tool boundary (negative conclusion, important)**: `CUDA_LAUNCH_BLOCKING=1` is **incompatible** with this transport — side A's launch becomes blocking,
   while side B's kernel only launches after it returns ⇒ the 1st collective necessarily gives up (self-deadlock). `compute-sanitizer`
   likewise (serializes kernels; after raising the timeout to 120 s, warmup still gives up) ⇒ **this transport cannot be attributed at kernel level with launch-serializing
   tools**, only by source-level probe bisection.

**Defect characterization (proved)**: after a bounded-spin give-up, the partial sum of this round's collectives is **local only** (the peer's contribution is missing), yet the round
keeps running. Phase 1b/2's mitigation only detects at the **round convergence point** (`abort_if_ar_stalled()`, call sites 2751/3317),
and when the give-up lands **inside a captured window** it is **too late**: the window's own kernel using the inconsistent data as an **address** crashes it
(reproduced). Only a give-up landing in eager prefill becomes a clean 503 + recovery on the next request.

**Trigger source (open)**: natural give-up cannot be reproduced on demand (0 in 600 requests; ~2700 requests → 2 in the whole campaign). The production 503's
rendezvous id (`2<<32+122460`) is also inside a **captured channel**, consistent with the reproduction. It requires >2 s of host/device delay,
or a real desync (id reuse/mismatch). Residual environment injection and any change to the default timeout have been ruled out.

**Recommended handling (pending decision)**
- (a) **Catch the trigger source (implemented; default now enabled)**: sampling A/B (`r62_sampling_ab.ps1`), three suites (`r66_suites.ps1`),
  and long runs (`r71_ar_stress.ps1`) all enable `NINFER_TP2_AR_WATCHDOG=1`; on failure `tools/tp_bootstrap/ar_watch.ps1`
  saves the log as `*-FAILED.log`, which **cannot be overwritten by a rerun**, and writes the interpreted dump to `*-FAILED.arwatch.txt`.
  **The save gate is now "the log really has a dump"** (`Test-ArWatchDump`, 2026-09-25): failures without a dump — test assertions,
  bad artifacts, port contention — are already fully described by the ordinary run log, and saving them only scatters `*-FAILED` files that need manual cleanup
  (15 of this campaign's 20 came from exactly that).
  Interpretation rules classify by the **magnitude of the id difference** (the array retains each side's **last written** value): a whole group of 0 on one side ⇒ that side
  **never reached the arrival write** (the kernel did not run); difference 1 ⇒ **OFF-BY-ONE** (the two sides number the same collective differently);
  a difference of a whole block (hundreds) ⇒ **BLOCK-LEVEL** (the two sides are running on different id blocks, exactly the shape of "a captured graph read a cell the host had
  already re-armed"); the rest are general SKEW. Also improved `device_pair.cu`'s watchdog: it dumps only when **the device really trips** (reading both stall flags, same source
  as `ar_stalled()`) — idleness between requests also stops the host counter, which used to be mistaken for a stall; the polling interval was tightened from 500 ms to
  **25 ms**, because after a trip there are often only milliseconds before the fault and exit, and a slower rate misses it.
  **Verification**: the clean arm has 0 ar-watch lines and no FAILED file; the injection arm (`FAULT_SKIP_PEER_CALL=700`) exits 1 and leaves
  `FAILED.log` + the interpretation `OFF-BY-ONE: A=1<<62+439 B=1<<62+440`
  (**the only retained raw dump**: `build-win/ar-evidence/r73-watchdog-gate-injection.arwatch.txt` + the same-named `.log`;
  the other 18 historical `*-FAILED.*` were cleaned up — r82 was 5 sets of false alarms caused by a packing assertion I myself wrote wrong, r76 was the
  old-harness port-contention artifact already characterized in §3.10, r75/r83 duplicate r73 or are summarized in §3.11); the three suites with the watchdog enabled are
  **5/5 exit=0**, proposal digest bit-identical to before the change (K=7 `0x8103f572fb2d2f99` / K=5 `0x96582e7289dd2702`);
  two-point gating verification: idle (3 requests + 4 s interval) **0 dumps**, injected trip **2 dumps with correct interpretation**.

  **The trigger source is still not caught by natural long runs**: the 900-request long run with the watchdog enabled (`r77`) **900/900 succeeded, 0 failures**, cumulative
  natural requests 1500 (`r71` 600 + `r77` 900) with zero events, supporting a natural rate ≲1/1500. (Note also: `r76`'s
  "request 3 connection cut off" was an artifact of **an old harness still running and contending for the same port** — the server log shows two
  clients running at once, and what actually killed serve was the old harness's wrap-up; after confirming the environment was clean, a rerun produced `r77`.)
- (b) **Make give-up safe**: host-side checks have been refuted (too slow); either let the transport stop consuming after a give-up
  (poison/abort semantics), or have the in-window indexing ops bound their inputs — both are architecture-level changes, and require first knowing the exact op.
- (c) **Stop-loss (implemented 2026-09-25)**: default timeout **2000 ms → 10000 ms**
  (`device_pair.cu:511`, env `NINFER_TP2_AR_TIMEOUT_MS` still overrides, 0 disables).
  The criterion and tradeoff are written in the code comment: 10 s is ~3700× the **slowest legal collective (~2.7 ms full-width prefill payload; 9–18 µs on the decode side)**;
  this bound exists only to break a **permanent** desync (which no value can resolve), so it must be far above anything a healthy round can produce —
  **a second-scale stall is a host/driver event (WDDM scheduling, paging, driver contention), not a slow kernel**. The two costs are asymmetric:
  a false trip costs **an entire request plus its re-prefill**, while loosening it only makes something that was going to fail anyway wait longer, and the natural event rate is about
  **3/1500 requests**. It is still a bounded spin, so a true desync will not hang the service indefinitely.
  **Verification**: injection 700 with the default value elapsed 73.7 s, with explicit `=2000` 65.6 s ⇒ difference **8.1 s ≈ 10−2 s**,
  proving the new default takes effect; both are still `503,ok illegal=False` (failure semantics unchanged, no crash). Guard script
  `tools/tp_bootstrap/r85_timeout_default_check.ps1`.
- (d) **Diagnosability (partly implemented)**: in-window give-up originally manifested as an uninformative illegal address; 1b's abandon-output
  fix has turned it into an identifiable 503 (see §3.11). What remains is collecting trigger-source data ⇒ see (e).
- (e) **Diagnostic thread enabled by default (implemented 2026-09-25)**: `NINFER_TP2_AR_WATCHDOG` is **on by default** (explicit `=0` disables).
  Rationale: the only relevant hot-path overhead — `note_ar_call`'s per-collective `calls.fetch_add` — **already executes unconditionally**
  (gated only on `watch_ != nullptr`, and the inline transport necessarily allocates `watch_` when it comes online), so the switch only controls that thread; the thread
  reads the host counter once every 25 ms plus (when idle) the two mapped stall flags, **calls no CUDA API, does not synchronize a stream, and does not touch the
  arrival/order rows the kernel is spinning on** (the 32 slots are read only after a trip is confirmed).
  **Also rate-limited the big dump to once per stall**: previously only the first line `[ar-watch] stalled at` was gated, and the big dump repeated one line every 25 ms
  (after the timeout was raised to 10 s, one event would print ~400 lines); what the interpretation needs is the state **at the moment of detection**, and repetition adds no information.
  **Verification** (`tools/tp_bootstrap/r87_watchdog_default_check.ps1`, injection 700): default ⇒ `stall_lines=1
  detail_lines=1` (would have been ~400 before the change), `NINFER_TP2_AR_WATCHDOG=0` ⇒ `0/0`; both are still `503,ok illegal=False`.
  The healthy arm has **0 ar-watch lines** (a clean run leaves no log).
  **Measured backing** (`tools/tp_bootstrap/r88_watchdog_ab.ps1`, final artifact K=7, 18 requests/arm, 4608 tokens/arm):
  on = **75.67 tok/s** aggregate (per-request mean 79.52, n=30), off = **73.14 tok/s** (76.71, n=31)
  ⇒ Δ = **+3.5% favoring "on"**, within this basis's noise floor and physically implausible in direction (the same basis previously measured graph vs eager
  at only +5.2%) ⇒ **no measurable cost**, consistent with the code analysis.
  Incidentally: `r72_stall_injection.ps1` was changed to "default to 1, but respect an explicit outer setting", otherwise its own `=1` line would override
  `=0` and the disable path could not be verified.

---

**TP-2 host-staging fallback rewrite + explicit staging capacity (2026-09-26)**

**S1 · Fallback reduction rewrite** (`src/core/tp/device_pair.cu`)

- **Old**: the fallback path added element by element with a scalar `__hadd` loop; measured 10 MiB payload **25.8–32.6 ms (0.31 GB/s)**,
  the dominant cost of the fallback (pure transfer of the same size is only 3.1–5.9 ms); 50 MiB payload **160.8 ms**.
- **New**: SSE2 vector add, 8 BF16 per iteration. BF16→FP32 is a left shift by 16 (BF16 is just the high half of FP32); after adding in FP32
  it rounds back with RNE as `cuda_bf16.hpp`'s `__float2bfloat16` does (including normalizing NaN to `0x7fff`); payloads above 32 MiB
  are sharded across up to 8 host threads by an atomic counter.
- **Structure**: `2×D2H → dual-stream sync → reduction → 2×H2D`. Briefly implemented 1 MiB chunked pipelining (add overlapped with H2D), which
  measured **about 1 ms slower** than a single large copy (the copy-engine overhead of 20 small H2Ds exceeds the overlap gain), so it reverted to the single-copy structure.
- **Measured** (`NINFER_TP2_AR_STAGING_BENCH=1`, force the fallback pair, min over multiple rounds):

  | Payload | Old (scalar add + pure transfer) | New allreduce min | Pure transfer min |
  |---|---|---|---|
  | 10 MiB | ≈ 35.5 ms (32.4 + 3.1) | **5.2–8.0 ms** | 3.1–5.9 ms |
  | 50 MiB | ≈ 175.7 ms (160.8 + 14.9) | **20.5 ms** (8 threads) / 25.0 ms (forced 1 thread) | 14.9 ms |

  ⇒ about **5–7×** improvement at 10 MiB and about **8×** at 50 MiB. This machine's copy-engine path is itself bimodal (the same 10 MiB two-way pattern
  jumps between 3.1 and 5.9 ms), hence the ranges in the table.
- **Basis for the 32 MiB thread threshold**: at 10 MiB the 8-thread add measured 4.1–4.4 ms, actually slower than single-thread 2.1–2.5 ms
  (on Windows each `std::thread` creation costs a few hundred microseconds ×7); at 50 MiB the 8-thread 8.4 ms beats single-thread 12.0 ms.
  The threshold sits between the two, so the production-width 10 MiB payload creates no threads (the same-process A/B of the two configurations shows identical 10 MiB times).
- **Diagnostic hooks**: `NINFER_TP2_AR_FORCE_HOST_STAGING=1` (read at construction; in-kernel is available on this machine, and without this hook
  the fallback path would never execute) and `NINFER_TP2_AR_ADD_THREADS` (overrides the reduction thread count, for same-process A/B).
- **Bit-exact verification**: `test_tp_device_pair.cpp::check_host_add_patterns` takes 736 targeted bit-pattern pairs
  (ties 12 / inf 84 / nan 247, including ±0, denormals, exact ties in both directions, overflow to inf, qNaN/sNaN with both signs)
  really through one transfer, and both the in-kernel and fallback paths are bit-identical to the host oracle `__float2bfloat16(a+b)`.

**S2 · staging capacity 24 → 48 MiB + explicit chunk clamp**

- `kInKernelArBytes` 24 → **48 MiB** (per card per parity slot); the cost is +48 MiB pinned per card, +96 MiB for both cards
  (the fallback/inline pair is now 2×48 MiB per card, and allocated only when `!p2p_`), with no change to the device memory budget.
- New `DevicePair::in_kernel_allreduce_bytes()`; `TP2GenerationCore::prefill_chunk_width(config)` derives the
  chunk width backwards from "the widest per-token allreduce payload of the model":
  `max(hidden_size, gdn->value_width(), attention->query_width()) × 2 B × T ≤ staging`. The three chunk
  computation sites (`build_shard`'s `feature_columns`, `make_dflash_prefill_sink`, `execute_walk`) all go through it.
  Previously "payload ≤ staging" was only a numerical coincidence of 5120×1024×2 = 10 MiB < 24 MiB; raising `kPrefillChunkMaximum`
  or widening the model would silently fall into the fallback path.
- **Verification**: `ninfer_tp_device_pair_test` **PASS** — the new 25 MiB case (`staging_bytes=50331648`, stays
  in-kernel; under the old 24 MiB it would fall back) and the 50 MiB case (exceeds staging ⇒ fallback + sharded reduction) are both bit-exact; the forced
  fallback pair additionally covers 3 MiB+16 (non-whole-chunk width), 50 MiB, a 20 KiB×32 deep queue, and sendrecv cases.

**Pre-existing problem found during verification (not introduced here)**: `ninfer_qwen3_5_tp2_dflash_solo_test` fails stably on
`<models-dir>/qwen3_8_27b_swift15_dflash2_final.ninfer` — `a_continued` (reused=71) first sample
`[59399 475 327 363 …]` does not match from-scratch `a_continued_evicted` `[365 4577 62 16 …]`.
After `git stash`ing this batch's four source files and rebuilding at HEAD, it **fails identically, token for token** ⇒ unrelated to S1/S2; repeating after the change
gives the same token set ⇒ deterministic, not intermittent (unlike §3.10's natural give-up intermittent). Not characterized in this batch; needs separate
investigation (candidates: the artifact does not match the test baseline, or a pre-existing defect in the recall path). Also: media-path tests like solo/sessions
need FFmpeg on PATH (on this machine `C:\ninfer`), otherwise a load-time `0xC0000135`.

**Still not done**: the fallback's lower bound is now this machine's copy-engine rate (10 MiB two-way ≈ 3.1 ms, inline transfer of the same payload ≈ 2.7 ms).
Replacing the fallback's D2H/H2D with "kernel writes directly to mapped pinned" was the next step recorded at the time; after review in the next section it was rejected and changed to the staging
ladder.

### TP-2 staging ladder: progressive degradation of mapped reservations (2026-09-26)
**Trigger**: the previous batch (S1+S2) recorded the fallback's lower bound as "this host's copy-engine rate", and listed "replace the fallback's D2H/H2D with kernels writing directly to mapped pinned" as an optional follow-up. The user chose to take that route; before starting, its premises were re-reviewed, with the conclusion being rejection, replaced by a staging ladder.

**Re-review (why in-kernel direct-write fallback was rejected)**

- **Tiny gain**: for the same payload (20 MiB bidirectional per card) the copy engine is only about 14% slower than the kernel path (the fallback's sendrecv pure transfer 3.09-3.14 ms vs in-kernel 2.70 ms, the latter also including the peer handshake). Of the fallback's 5.2 ms total, 2.1 ms is CPU addition (S1 already at the floor), so switching the transport is worth ≤8% overall.
- **Unreachable in production**: there are only two ways to trigger the fallback — payload > staging, or mapped pinned unavailable. The former is unreachable in this product: the widest prefill payload is the gather activation 6144 x 1024 x 2 = 12 MiB, the vocab-parallel head merge in production prefill is `[V,1]` (`tp2_generation_core.cpp:2722` passes `logits_columns = nullptr`), and the verify window is `[V, W<=16]` <= 7.6 MiB; the latter (mapped pinned unavailable) precisely means the device cannot address host memory, in which case in-kernel direct write is impossible by definition.

**What was implemented instead**: `DevicePair`'s staging reservation goes through a **ladder**, largest rung first: 48 → 24 → 12 MiB (per card, per parity slot, i.e. 2 x rung per attempt). `reserve_in_kernel_staging()` tries rung by rung, releasing a failed rung before dropping one rung; if all fail there is no in-kernel transfer as before and everything goes through the fallback. A new env `NINFER_TP2_AR_STAGING_MIB` caps the ladder, used to reproduce a constrained host from a script. When a rung below full is selected it prints one line `[mem] TP-2 in-kernel allreduce staging N of 48 (larger rungs did not map)`.

**Basis for the 12 MiB lower bound**: 12 MiB is the capacity at which "no payload in this product is narrowed" — the largest eager allreduce is exactly 1024 token x 6144 columns x 2 = 12 MiB flat, and the widest **in-graph** payload is the full-window merge of the vocab-shard head `V x (K+1) x 2` = 248320 x 16 x 2 = 7.6 MiB (K=15 upper bound). One rung lower would require narrowing the prefill chunk, or letting the in-graph merge fall into the fallback — and the fallback must synchronize two compute streams per collective, which is **illegal during capture**. So the ladder stops at 12 MiB; smaller windows leave the host on the fallback (correct, just slower), rather than using a transport that would blow up during capture.

**Key regression point**: the slot stride when launching the kernel was originally the constant `kInKernelArBytes` (48 MiB). Once the rung is selectable it must follow `in_kernel_bytes_`, otherwise the parity slot would be offset by 48 MiB and written outside a mapped region only 2 x rung in size. The test covers this with a payload "exactly equal to the rung" (12 MiB onto a 12 MiB slot, writing to the last byte of each slot).

**Capture-time guard**: added `require_capturable_payload()` — when capture is on, a payload exceeding staging directly throws `std::logic_error` (explaining that the fallback's stream synchronization cannot go into the graph), rather than silently switching transports and reporting an obscure capture error at the driver layer.

**Verification**

- `ninfer_tp_device_pair_test` **PASS**: the 24 MiB rung (20 MiB in-kernel, 25 MiB out of range via the fallback), the 12 MiB rung (12 MiB full-load allreduce + sendrecv, 4 MiB x 16 deep queue, 12 MiB+16 out of range), the 8 MiB cap (below the lower bound ⇒ `in_kernel=0`, all fallback), and the default pair still reporting the 48 MiB / 25 MiB cases unchanged.
- **End to end**: `ninfer_qwen3_5_tp2_dflash_append_test` (swift15 dflash2 artifact, chunk 1024) under `NINFER_TP2_AR_STAGING_MIB=12` is **exit=0**, all bit-exact assertions pass (sink logits bit-identical, direct/split append bit-identical, ring reproducible), proposal cost 7.057 ms vs baseline 7.045 ms (within noise); the log's first line confirms the 12 MiB rung was selected. Under the 12 MiB rung the chunk is still 1024 (`fits = 12 MiB / 12288 = 1024`), so the narrowing branch is still unreachable on the current model — it is an explicit invariant kept for wider models/smaller rungs.
- Full tree `cmake --build build-win` 243/243 passes.

**Still not done**: hosts below 12 MiB can still only use the copy-engine fallback (in-kernel direct write is impossible there); if a model with `V x (K+1) x 2 > 12 MiB` appears in the future, capture will be explicitly rejected by `require_capturable_payload`, requiring either re-runging or narrowing the verify window on the core side.

### TP-2 host-staging fallback parity race (2026-09-26, found during ladder verification)

**Finding**: while verifying the ladder's "below the lower bound" outcome (= no in-kernel transfer, every collective goes through the fallback), `ninfer_qwen3_5_tp2_dflash_append_test` (swift15 dflash2 artifact, chunk 1024) failed in the DFlash2 proposal stage with `device_pair.cu:1133 CUDA_CHECK(cudaStreamSynchronize(stream_b)) failed: cudaErrorIllegalAddress`; both `NINFER_TP2_AR_FORCE_HOST_STAGING=1` and `NINFER_TP2_AR_STAGING_MIB=8` stopped at the same place.

**Localization** (four discriminating configurations, same artifact/config):

| Config | Transport | Result |
|---|---|---|
| `NINFER_TP2_VERIFY_GRAPH=0` | in-kernel | PASS ⇒ eager verify is innocent |
| `NINFER_TP2_DECODE_GRAPH=exact` | in-kernel | PASS ⇒ EagerExact is innocent |
| `NINFER_TP2_AR_FORCE_HOST_STAGING=1` | fallback | **FAIL** (illegalAddress) |
| same + `CUDA_LAUNCH_BLOCKING=1` | fallback | PASS ⇒ **it is a race, not an addressing error** |

**Root cause**: the fallback's pinned staging is **a single shared buffer**, while adjacent calls run on two **mutually independent** compute streams. One call does D2H of both halves into `[a | b]`, writes the sum back to `[a]`, then both H2Ds upload from `[a]` (one each for device a and device b). There is **no ordering whatsoever** between the previous call's H2D (stream_b reading `[a]`) and the next call's D2H (stream_a writing `[a]`): device b receives a false sum "partially overwritten by the next layer's raw delta". In-kernel transfer has long had parity double buffering for cross-call reuse (`host_mine_base + (token & 1) * slot_bytes`), the fallback did not. The polluted hidden propagates downstream and can manifest either as "target logits changed after installing the sink" (what I saw when adding diagnostics changed the timing) or as a downstream out-of-bounds index ⇒ `illegalAddress`.

**Why existing tests did not catch it**: every pair-test case is separated by a synchronizing copy (implicitly waiting on both streams), while the fallback-rung deep-queue case is only 20 KB — H2D is about 10 µs, shorter than the host's work between two calls, so it never hits the overlap window.

**Fix**: added `DevicePair::host_staging()`: the buffer becomes "two parity slots × `[a | b]` per slot", taking slots alternately per call; on growth it first `cudaStreamSynchronize`s both streams before releasing the old buffer (otherwise an in-flight H2D would read freed pinned memory). Both the fallback's `allreduce` and `sendrecv` were moved onto it (they share the same slot sequence, which is exactly what their interleaved calls need).

**Evidence**: before the fix this route hit `illegalAddress` twice; after the fix both routes `FORCE_HOST_STAGING` and `STAGING_MIB=8` are **exit=0** with all bit-exact assertions passing; the pair test gained a "4 MiB deep queue" case (in-kernel 16 times / fallback 8 times) as the regression case for this race (the 20 KB rung cannot catch it), and the full tree builds 243/243. The fallback rung's proposal cost is 7.80/7.97 ms vs in-kernel 7.05 ms, consistent with the expectation that the fallback requires host synchronization per collective.

### PLAN §4.2 re-review: zero_pages' "HeadMajor out-of-bounds write" is a false positive (2026-09-26)

**Request**: fix PLAN.md §4.2 (at the time the only "confirmed latent bug: `zero_pages` HeadMajor out-of-bounds write").

**Conclusion**: this latent bug does not exist, `zero_pages` is correct under both plane orders; what needed fixing was the **analysis** and the **contract documentation**.

- The original reasoning's premise was "H is the innermost" (treating `{leading, 64, physical_pages, head_extent}` as numpy's memory order), hence reading `nb[3]` as the element size and concluding "each row advances by i*elem_size".
- `Tensor::set_contiguous_strides` (`src/core/tensor.cpp:50-57`) multiplies dimension by dimension starting from `nb[0]=dtype_size` ⇒ **`ne[0]` is the innermost**. Under HeadMajor `nb[2] = 64*leading*es` (bytes per page) and `nb[3] = physical_pages*nb[2]` (head stride), so `cudaMemset2DAsync`'s `dpitch=nb[3]` is exactly "one head per row", and row i covers head i's page interval; the last byte `first*nb2+(H−1)*nb3+count*nb2 ≤ H*nb3` = the end of the plane, and `dpitch = physical_pages*nb2 ≥ count*nb2` satisfies that API's pitch constraint ⇒ **no out-of-bounds**.
- The 1D `cudaMemsetAsync(base+first*nb2, 0, count*nb2)` suggested in the original text only zeroes head 0, and is wrong.

**Criterion**: `tests/test_kv_cache.cpp` gained `exercise_zero_pages` — both plane orders, first filling the entire pool with a deterministic bit pattern, running `zero_pages` over the middle run `[2,5)` and the last page `[7,8)`, then comparing **all bytes** per plane against the expectation (all 0 inside the run, byte-identical outside the run). Measured: the current implementation **PASS**; temporarily replacing the implementation with the suggested 1D memset gives **FAIL** (5 cases, including the existing `HeadMajor selective zero left page payload bytes`) ⇒ this case really does adjudicate the matter. The temporary change has been reverted (`git diff -- src/core/paged_kv_cache.cpp` is empty).

**Incidentally**: comments were added to `PagedKVPlaneOrder` in `src/core/paged_kv_cache.h` (the enum name refers to the slowest dimension; the shapes and page/head strides of both orders), writing the cause of this misreading into the contract; PLAN §4.6's conclusion and statistics were updated in sync.

---

### L2 (AR∥MMA overlap) re-review: can in-kernel AR co-run with SM-saturating compute (2026-09-27)

**Background**: the L2 entry in §3.14 (worklog:5291) previously had only Round 35c's cost model (AR ≈65% depth-independent term, ceiling ~1.5×) and one measured rejection in Round 12 (worklog:1425, two sub-blocks interleaved on the host side, −5%). The 2026-09-26 NINFER_TP2_AR_PAYLOAD_DIVISOR measurement (PLAN §5.4) gives the opposite premise: AR is **strictly serial** at 324 ms/block (compute 344 ms, AR taking **48%**), and the ceiling as AR bytes→0 is ≈2.98k tok/s (**+94%, about 1.94×**). That is, L2's gain premise holds, and the reason ① given by Round 12 ("AR's spinning is already hidden by the peer's compute") does not hold.

**The question this discrimination answers**: when AR is moved to the second stream, will SM-saturating compute make the AR kernel **unable to launch** (starve it)? If so, L2 is a hardware wall (can be rejected); if not, and it merely slows down, then Round 12's −5% is a scheduling problem (cannot be rejected).

**Method and reproduction** (tools/tp_bootstrap/bench_ar_overlap.cu, standalone CUDA, no engine dependency): replicate the production protocol (ordered-slice write to mapped pinned host → send arrival id → spin waiting for the peer id → read back and add, <<<8,1024>>>), with payload 5 MiB / 10 MiB (chunk 512 / 1024's mixer delta); AR alone is **1.43 / 2.70 ms/call** (consistent with the 2.70 ms recorded in Round 35c, already at the link floor). Both cards open a second stream to run AR while the compute stream saturates the SMs; cross-stream events measure ar_start_offset (AR's launch delay relative to compute) and ar_inflate (inflation relative to exclusive).

    nvcc -O3 -std=c++17 -arch=sm_120a -o build-win/bench_ar_overlap.exe tools/tp_bootstrap/bench_ar_overlap.cu
    .\build-win\bench_ar_overlap.exe 0 1 16 60 <blocks_per_sm> <chain> <style>

(style=0 FP32-FMA saturating, style=1 HBM streaming saturating; chain = how many kernels the compute side issues at a time.)

**Results** (16 collectives/arm; AR was **never starved**: ar_start_offset was 0.008–0.34 ms throughout):

| compute shape | 5 MiB inflation | 10 MiB inflation | Verdict |
|---|---:|---:|---|
| single long kernel, 1 blk/SM | +2.0% | +14.5% | fully hidden |
| single long kernel, 8 blk/SM (FMA / HBM) | +152…184% / +204% | +101…109% / +104% | partial |
| 2 kernels, 8 blk/SM (HBM) | +199% | +102% | partial |
| **64 kernels, 8 blk/SM (FMA / HBM)** | **+33.5% / +18.3%** | **+8.7% / +7.2%** | **fully hidden** |

**Conclusions**:
1. "compute saturates the GPU ⇒ the AR kernel cannot launch" is **disproved**: under a 288-block long kernel, AR's 8 blocks still launch in ~0.02 ms. L2's wording "no resource conflict" does not strictly hold (AR's transfers are issued by SM instructions, and inflate 2–3× under a long kernel), but that is a **granularity** problem: the engine's compute is "a string of small kernels per layer", landing in the 64-kernel bucket, where AR inflates only **7–18%** and finishes within the window.
2. The contradiction between Round 12's −5% and the divisor's +94% ceiling most likely comes from Round 12's event/dual-stream topology, not from hardware. ⇒ L2's status changes from "rejected" to "**scheduling redo pending**" (PLAN §3.4 and tp2-decisions.md updated in sync).

**Limitations**: the synthetic compute is not a real tensor-core prefill GEMM; this experiment only proves "co-running is feasible and strongly correlated with granularity", and does not reproduce Round 12's sub-block dependency topology, so it cannot prove "an engine redo will certainly pay off" — that needs an engine-level A/B (next step).

### L2 engine-level A/B: sub-block count + event topology (2026-09-27)

**Implementation** (all code changed this time):
- DeviceContext gains a second compute stream collective_stream (same lifetime as stream/transfer_stream; device.{h,cu} and test_device.cpp updated in sync).
- TextContext::run_layers_tp2_overlap (execution/text.{h,cpp}): split the prefill chunk into equal-width sub-blocks, and per layer lay out "all sub-blocks' mixer → each sub-block's AR (collective stream) → each sub-block's residual → all sub-blocks' FFN → each sub-block's AR → residual"; each AR uses a pair of events to interlock the collective stream and the compute stream (produced recorded on the compute stream, done recorded on the collective stream), and positions/rope/envelope are bound to sub-block slices via ScopedPositions/ScopedEnvelope. Option `--prefill-overlap W` (EngineOptions::prefill_overlap, default 256, 0 disables) — **width** is the independent variable rather than block count (see the fine sweep below); enabled only in Phase::Prefill, with no DFlash sink, no vision, in-kernel transport, no gather mixer in any layer (tp_mixer_activation_width==0), and when the chunk is an integer multiple of that width and contains at least two blocks.
  (The implementation first used the environment variable NINFER_TP2_PREFILL_OVERLAP=N for experiments; when pinning the setting it was replaced by the option above and the environment variable deleted.)

**Correctness**: ninfer_qwen3_5_tp2_forward_test --artifact …swift15_dflash2_final.ninfer, both N=2 and N=4 exit 0, and chunking 300 -> 128+172 / 64+236 and T=1024 one chunk vs 4x256 both give max_logit_diff=0 (bit-identical).

**Measurements** (this host's 2×5060 Ti, ninfer-serve running swift15, --prefill-chunk 1024, about 7,160 token single request, taking prompt_ms from the server's timings block, 5 reps per setting, service restarted between settings):

| N | prompt_ms (reps) | prefill |
|---:|---|---:|
| 0 (baseline, two independent launches) | 4557/4268/4222 and 4292/4251/4180/4178/4175 | **1,677–1,714 tok/s** |
| 2 | 4213/3957/3892 | 1,708–1,839 |
| **4** | 3780/3669/3698/3819/3778 | **1,875–1,951 (median 1,905)** |
| 8 | 4748/4702/4723/4595/4596 | 1,516–1,558 |

Per-token prefill time: baseline ≈0.587 ms, N=2 ≈0.560 (−4.6%), **N=4 ≈0.522 (−11.0%)**, N=8 ≈0.650 (+11%).
⇒ the optimum is at **N=4**; cutting finer narrows the GEMM tile past the point of being worth it.

**N fine sweep (same batch, chunk 1024, prompt_n≈6,284, 7 reps, taking medians)**: the span between N=2 and N=8 was too large, so after adding N=3/N=5 the curve is

| N | sub-block width | ms/token (median) | vs baseline |
|---:|---|---:|---:|
| 0 (baseline) | — | 0.587 | — |
| 2 | 512 | 0.554 | −5.6% |
| **3** | **320+320+384** | **0.612** | **+4.2% (worse than baseline)** |
| **4** | **256×4** | **0.516** | **−12.1%** |
| 5 | 192×4+256 | 0.547 | −6.8% |
| 8 | 128×8 | 0.655 | +11.6% |

N=4 reproduces across two independent batches (0.5151 / 0.5172, 0.4% apart), a reliable optimum. **But N does not interpolate**: N=3's 320 width (tail block 384) is slower than baseline, showing that a sub-block width landing in some activation/MMA scheduling bucket slows everything down, so the setting must be pinned as "sub-block width = 256", not extrapolated linearly in N.

**Fixed-width strategy verified at chunk 512** (same batch, prompt_n≈6,283, 7 reps taking medians):

| Config | sub-block width | ms/token | vs same-setting baseline |
|---|---|---|---:|---:|
| chunk 1024 serial | — | 0.587 | — |
| chunk 1024 + overlap | 256×4 | **0.516** | **−12.1%** |
| chunk 512 serial | — | 0.625 | — |
| chunk 512 + overlap | 256×2 | **0.572** | **−8.4%** |

⇒ "fixed 256 width, enabled only when the chunk contains at least two blocks" holds at both settings; chunk 512 with overlap (0.572) is even faster than the chunk 1024 serial baseline (0.587). chunk 512 serial is itself 6.5% slower than chunk 1024, and overlap recovers most of that.

**Conclusion**: Round 12's −5% was a problem with **its scheduling shape** (two sub-blocks interleaved on the host side + event topology), not with "AR and MMA being unable to co-run"; after switching to "one produced/done event pair per sub-block + a separate collective stream", the same chunk width nets **+11% prefill** with values bit-identical. This is self-consistent with the 2026-09-26 divisor arm's "AR is a strictly serial 48% of the block time": under N=4 part of it is recovered.
(Not done: the chunk 768→3 sub-blocks setting — TP-2's prefill chunk is clamped to ≤1024 by normalize_engine_options, so a 256 width can cut at most 4 blocks, and chunk 640/896 are not integer multiples of 256 and automatically fall back to serial, hence 768 is the only visible but untested block count; also outstanding are long context (32k–128k) and the multi-request regime, and the interaction of overlap with §3.6's give-up semantics.)

## Round 15c — the dflash2 route's mask ordering bug (third reproduction of tool-marker leakage)

**Symptom**: the agent session again emitted tool calls as body text (three `<function=tools.pwsh>` in a row), with the server's `stopReason=stop` and no structured tool_calls; the request declared only 1 tool (`run_code`), while the system prompt listed the whole set of SDK tool names.

**Root cause**: `--spec dflash2`'s verify branch wired the constraint mask in the wrong order. `apply_token_mask(window_logits, ...)` was written **before** `run_verify_window(...)`, and the latter is exactly the target verify forward that produces `window_logits` — the mask was immediately overwritten, and constrained decoding on that route never took effect (equivalent to having no constraint). `git blame`: the MTP placement was put in the right position by `71ac9883` (the constrained decoding feature), while the dflash2 placement was copied before the forward when `c08a6624` (*feat(tp2): run the DFlash2 loop end to end and open the option gate*, 09-21) wrote the new loop.

**Fix**: move the whole `if (constraint_live()) {...}` block to after `run_verify_window` and before `ops::argmax` (aligning with MTP), and delete the redundant `cudaStreamSynchronize` inside the block (`window_ids` is already synchronized above). Re-reviewing the four placements (prefill first token / plain decode / MTP verify / dflash2 verify), they are now all at "after logits are produced, before sampling or acceptance".

**Deployment**: after rebuilding, `build-win/apps/ninfer-serve.exe` was manually synced to `C:\ninfer\ninfer-serve.exe`, with `Get-FileHash` equal on both sides (`99BCF3077CC6B37025DFCDBCD7935F81CC43FD718B89442DE743A1D84B160D37`); `ninfer_tool_call_constraint_test` / `ninfer_tool_call_parser_test` pass.

**Acceptance (making up Round 15b's "not done")**: `%TEMP%\ab_probe.ps1` (with `-BaseUrl` and `-NoProxy` added) hits this `--devices 0,1 --spec dflash2 --draft-tokens 5` service on port 3456:

| Variant | n | ok | failure classes |
| --- | --- | --- | --- |
| todo / read / pwsh | 15 + 15 + 15 | 45 | none |
| control | 5 | 5 | none |

Tool requests **45/45 hit `run_code`**, with `bad-name` / `malformed` / `empty` all 0. Against Round 15's "unconstrained" baseline (marker-class failures 19–27/75, i.e. 25–36%): if the true failure rate were still 25%, the probability of 45/45 passing is about 2.6e-6 ⇒ constrained decoding is in effect on that route.

**Remaining**: the single-card route (without `--devices`) still has **no** constrained decoding at all (`make_tool_call_constraint` / `apply_token_mask` are only called by `tp2_generation_core.cpp`), and the same class of leakage will reproduce 100% there; the plan is in `PLAN-single-gpu-tool-mask.md`.
## Round 15d —— constrained decoding on the single-GPU route (landed; dflash2 partially covered)

**Background**: constrained decoding was previously implemented only in the TP-2 core (the call sites of `make_tool_call_constraint` and `apply_token_mask`
are all in `tp2_generation_core.cpp`), while the single-GPU route (`ProgramImpl` + `program/*`) lacked it entirely, and the same class of leak necessarily reproduces there
(the model emits an undeclared name → the parser rejects it → it is used verbatim as body text).

**Approach (mask as a per-replay device input)**: sampling on the single-GPU route is inside a CUDA Graph (`ordinary_batch_body` contains
`ops::sample`, and the MTP/DFlash verify is captured likewise), while the mask depends on already-generated text (host-side state) and cannot be computed at capture time.
Hence: a resident `U8 [vocab, columns]` device buffer (column count = `max_concurrency` without spec; with spec
`max_concurrency*(draft_window+1)`, accounted in `device_reservation_bytes`); the in-graph kernel contains only `apply_token_mask`;
before **each** replay, the H2D is done out of graph on the **same stream**; when unconstrained, maintaining an "all-ones whole block" invariant lets the upload be skipped.
The body is not split, and no separate graph is captured per constraint state.

Landing points: new `program/tool_call_mask.h` (`live`/`advance`/`build_single`/`build_after`/`piece`, semantics aligned verbatim with
TP-2); `program/decode.cpp` (ordinary body masks before `ops::sample` + host-side mask fill and H2D for all three decode kinds);
`execution/text.cpp`'s `target_verify_batch` (between `project` and `argmax`); `prefill.cpp` (last column);
`speculative/mtp.cpp`, `execution/draft.cpp`, `speculative/target_verification.cpp`;
`program/planning/startup.{h,cpp}` (resident buffer and accounting); `program.cpp/.h`, `program_impl.{h,cpp}`,
`graphs.cpp`, `dflash_round.cpp`, `storage/context.cpp`, `transactions/commit.cpp` (clear pointer on lane release/reuse);
`runtime_types.h`, `model_instance.h`, `request_record.h`, `engine_core.h` (construct the constraint with Frontend before the prompt is moved,
`attach_tool_call_mask` after materialize).

**Real-machine acceptance** (artifact `Qwen3.8-27B-GSQ-RCO-IQ3_S-dflash2q4.ninfer`, single GPU 8099, `--max-context 8192
--kv-dtype fp8`; 12.2 GiB of weights on a 16 GB card FATALs at startup at 32k, so it must be lowered to 8k. Probe as in Round 15/15c):

| Route | Before fix | After fix |
| --- | --- | --- |
| ordinary (no spec) | 87/90, `bad-name` 3 | 88/90, `bad-name` **0** |
| mtp (draft-window 5) | not tested | **90/90, failures all 0** |
| dflash2 (draft-window 5) | 42/45, `bad-name` 3 (6.7%) | 59/60, `bad-name` 1 (1.7%) |

**Known gap (not done this round)**: single-GPU dflash2 only covers the anchor column —— `dflash_decode_batch` captures append→propose→
verify→accept entirely into one graph, and the mask for columns ≥1 needs "the draft before that column", which is a device product inside the same graph and is not returned to host,
leaving no seam for in-graph readback. By contrast, TP-2 is eager propose + D2H `verify_ids` + **capturing only verify**, so the draft is
visible to host. Zeroing it out requires splitting append+propose out of the verify graph (capture topology rework + dflash2 decode performance re-measurement). Current choice: keep the
status quo, and use `--spec mtp` when starting an agent on a single GPU (measured 90/90); the anchor-column mask semantics are correct (the syntactic position of that column is determined solely by committed
text), kept as a profitable degradation.

**Unexplained items**: after the fix, 2 `empty` in the ordinary arm (1 benign with `incomplete=max_output_tokens`; 1 at 756 ms with no
`incomplete`). The mask necessarily does not take effect in the Free state, and `raw_content` is empty at the first token ⇒ mechanically it cannot be caused by the mask; I lean toward
a probe misclassification artifact of "reasoning item only, no message item". Neither is counted as `bad-name`.

**Build/unit tests**: `cmake --build build-win --target ninfer-serve -j 12` passes; `ninfer_tool_call_constraint_test` /
`ninfer_tool_call_parser_test` / `ninfer_token_mask_test` / `ninfer_qwen3_5_runtime_mechanisms_test` /
`ninfer_qwen3_5_context_store_test` / `ninfer_qwen3_5_state_image_layout_test` / `ninfer_decode_graph_test` /
`ninfer_public_api_test` pass; `ninfer_qwen3_5_frontend_test` / `vision_workspace_test` /
`engine_options_test` give `0xC0000135` on load (the working tree lacks the FFmpeg runtime DLLs), unrelated to this change.

**Deployment status**: not deployed to `C:\ninfer\` (that still holds the TP-2 fix accepted in Round 15c, hash `99BCF307…`);
the new build is at `build-win/apps/ninfer-serve.exe`.

---

# PLAN.md archive (archived 2026-09-27, upstream perf port, architecture review, FP8 PV, phase pipelining)

Archived verbatim from the working PLAN.md when it was reduced to key information and remaining work, after the upstream ninfer-all performance-item port and the architecture-wide review (2026-09-26), and the two closed attention projects: FP8 PV (§6) and cross-tile phase pipelining (§7) (2026-09-27). Historical record only: the current state and remaining work live in PLAN.md at the repository root, product behavior and measurements in [tp2-dual-5060ti.md](../tp2-dual-5060ti.md).

# NInfer TP-2 plan (2× RTX 5060 Ti · Qwen3.8-27B NVFP4)

> **The single active plan**, and the persistent memory across context compression. Compiled: 2026-09-25 (the full text before this compilation went into the worklog; this file keeps only
> key information and outstanding items).
> - Full history: `tp2-adaptation-worklog.md` —— contains four verbatim full-text archives of PLAN.md (2026-09-21 upstream
>   cherry-pick re-planning version; 2026-09-22 delivery wrap-up version; 2026-09-24 post-TP-2 rendezvous-protocol-rework version; 2026-09-25 the full
>   text before this compilation, including the r67–r70 draft quantization campaign, the TP-2 transport low-frequency failure investigation, and the TP-2 orchestration/rendezvous work).
> - Delivery notes, recommended configuration and measured data: `docs/tp2-dual-5060ti.md`; Windows native port: `docs/windows.md`.
> - Settled-decision quick reference (rejected options, root causes, methodology rules, worklog pointers): `docs/tp2-decisions.md` —— mandatory reading before re-investigating a settled topic.

---

## 1. Key information

- **Branch**: `feat/windows-native-port` (Windows native + WSL2 dual tree). All 20 commits by which origin/master is ahead have been
  cherry-picked (frontend jinja chain + three NVFP4 performance items), with no end-to-end A/B regression.
- **Hardware**: 2× RTX 5060 Ti 16G (SYS topology, no P2P, 448 GB/s per card). WSL2 build tree `$HOME/ninfer`;
  Windows working tree = the repository root (build tree `build-win`).
- **Main workload**: Qwen3.8-27B NVFP4, TP-2 (weights split in half per shard, locally built half-geometry shape). Draft backend: `--spec dflash2`
  (K=5/7, K=7 is the binary default) or `--spec mtp` (K=2, `--lm-head-draft`); `--spec dflash` (v1) is rejected at construction time.
- **Current state**: delivery goals ①–⑤ all complete; vision, the Windows port, the multi-session KV pool, decode graph + AR policy are all landed and measured;
  DFlash2 is now allowed on TP-2 (unaligned reuse no longer discards the draft); the TP-2 rendezvous protocol rework is complete (host-authoritative id + bounded spin +
  timeout-failure semantics); the full DFlash2 draft Q4 set has been promoted into the official recipe (r69, user-confirmed, final artifacts releasable); the verify graph
  B7 acceptance passed (aggregate +5.2%).
- **Key numbers** (262,144 configuration, WSL): prefill 1,588 tok/s; decode 59.0 tok/s (MTP K=2); Windows 131,072 configuration
  decode 56.9 tok/s (on par with Linux). DFlash2 K=7 is about 71 tok/s in the 4096-context greedy mode; the 245,760/k8v4 (draftall artifact)
  scan gives K=5 96.4 / K=7 96.1 tok/s (interleaved re-check 93.3 / 96.9); roofline-corrected upper bound ≈108 tok/s (the 180 of the originally archived 90–180 is
  unreachable). The final r69 artifact saves 257.77 MiB versus r66 (−1.08%), with no measurable acceptance-rate cost (≤1σ), and a round-time cost of ~+0.4%
  (proposal step +0.19 ms).

**Recommended run configuration** (same recipe on WSL 8088 / Windows 8099, with `--max-context 131072` on the Windows side):

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2 --lm-head-draft --vision --reasoning-effort medium
```

`--reasoning-effort low|medium|xhigh` is the process-level default thinking effort (request body takes precedence); `--chat-template` is available as soon as the first tier is picked up
(the template is executed by the embedded Jinja). Measured 2026-09-23: DFlash2 (K=5/7) decode leads MTP by ~20% with prefill cost within 1%
⇒ for throughput use `--spec dflash2 --draft-tokens 7` (the scan recipe is 245,760/k8v4, rationale in the archive).

**Final artifacts** (r69/r70, all converted with the official recipe and no `--override`, all containing text/vision/mtp/dflash2, draft profile identical across the four:
q4 88 / q8 7 / bf16 567): `<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer`,
`…_w4a4_dflash2_final.ninfer` (18.42 GB), `…_swift15_dflash2_final.ninfer` (21.58 GB),
`…_thinkingcap_dflash2_final.ninfer` (22.52 GB).

### Environment and operations notes

| Item | Value |
|---|---|
| Build (WSL) | `bash tools/tp_bootstrap/r55_build.sh` (rsync + `cmake --build build_dyn -j 8`, success marker `BUILD_EXIT=0`, about 2–3 minutes; the WSL build tree is now `build_dyn` (Ninja), `build_r35.sh` still points at the deleted `build/` and is unusable) |
| Build (Windows) | `tools/win_port/configure.bat` + `build.bat` (VS2022 + CUDA 13.3, `-DCMAKE_CUDA_ARCHITECTURES=120a`) |
| Serving | WSL 8088 (`serve_supervise.sh`); Windows 8099 (`tools/win_port/serve.ps1`, foreground by default; a self-test service must use a harness background job, since a process started with `Start-Process` is killed when the tool call ends) |
| Runtime PATH (Windows) | FFmpeg (`D:\ffmpeg-dev\…\bin`) and libcurl (`D:\curl-dev\…\bin`) must be on PATH, otherwise `STATUS_DLL_NOT_FOUND` (the solo/sessions trio gives `0xC0000135` at load time without them) |
| artifact | `<models-dir>/qwen3_8_27b_nvfp4.ninfer` (23.7 GB, older artifact, template `c3cf9e34…`); the new official artifact (template `a497db9e…`) is available with the first tier; r69/r70 final artifacts see above |
| Process discipline | Only one model process machine-wide at a time (WSL 8088 and Windows 8099 are mutually exclusive); stop the service before relinking the Windows exe (LNK1104); `build-win/apps/ninfer-serve.exe` is not automatically synced to `C:\ninfer\`, so copy it manually and compare with `Get-FileHash` |
| Toolchain | Nested `pwsh` → `wsl -e bash -lc` swallows `$var` and redirections ⇒ write the command as a script file under `tools/tp_bootstrap/` and execute that; a single blocking call is capped at 600 s ⇒ use background jobs for long tasks |
| GPU/WSL state (since 2026-09-23) | The machine sees 3 cards: nvidia-smi order 0/2 = 5060 Ti, 1 = Tesla T10; `--devices 0,1` ⇒ the CUDA ordinals pick up the two 5060 Ti (mismatched with the nvidia-smi order), and TP-2 pairing is unaffected. Full-segment CUDA errors on the WSL side ⇒ tests move to Windows `tools/win_port/test.ps1`, and the WSL build_dyn is used only for compile verification |

---

## 2. Completed summary (one line each, details in the archive)

- ① MTP consistency: root cause = shard did not write the replay record (fold consumed an empty log), fixed; criterion is "no regression + same quality tier + speedup".
- ② Benchmark: prefill 615→1,588 tok/s; decode plain 31.4 / MTP K=2 50.5–54.0 → 59.0 tok/s.
- ③④ KV / context: five KV dtypes usable; 65,536 → 131,072 → 262,144 (vocab/hidden parallel split, −1.2 GiB per card).
- ⑤ Documentation: `docs/tp2-dual-5060ti.md` + `docs/windows.md`.
- Vision: static shard split of duties (MTP→shard 0, vision→shard 1), single-image cap 8,192 tokens, no new configuration.
- Windows native port: VS2022 + CUDA 13.3 build; CLI/serve/TP-2 all working, performance on par with Linux; TMA descriptor
  mapped-pinned fix.
- Optimizations: exact-batch verify graph (−4.4 ms/round), plain decode graph (+12.6%), AR transport split by payload (+0.63%),
  MTP draft chain graphed (−0.2~0.3 ms/round); AR∥MMA sub-block pipelining measured 5% slower, rejected and reverted.
- Round 19/20: chunk planning decoupled from ring rewind (1a/1b/1c), results for the same prompt no longer depend on engine history; the host KV arena
  is pageable by default (`--host-kv-pinned` is required to pin, and a rejected pin falls back automatically).
- Multi-session KV pool: host KV swap-in/swap-out + 5-session LRU + shared-prefix mirror; recalled KV is byte-identical to device.
- Upstream cherry-pick: the 5 first-tier items (jinja frontend chain + `--chat-template`), the 3 second-tier items (NVFP4 TMA routing,
  SwiGLU epilogue, non-256-divisible tokens) all landed; no e2e A/B regression (decode +1.3%~+5.8%).
- Official NVFP4 artifact acceptance-rate investigation: synthetic artifacts match self-converted artifacts, and the official artifact is the lowest ⇒ the difference comes from the text weights, independent of components.
- TP-2 device-memory imbalance: option A (reduced proposal head split by vocabulary, −170 MiB per card) and option B (selector moved to shard 1,
  −246/+246 MiB) both landed, output byte-identical; new `DevicePair::sendrecv` does byte-exact cross-card exchange, eliminating the sNaN bit-pattern risk of
  BF16-add allreduce moving I32/FP32.
- TP-2 rendezvous protocol rework (2026-09-23, incident-driven): Phase 1 bounded spin (env `NINFER_TP2_AR_TIMEOUT_MS`) + timeout-failure
  semantics (HTTP 503 + session poisoning and re-prefill); Phase 2 host-authoritative 64-bit id (delivered via an in-graph memcpy node); Phase 3 large-payload
  event path measured and rejected; Phase 4 partial-merge audit concluded the settled form does not exist. Regression matrix shows zero regressions.
- K scan and prefill cost attribution (2026-09-23): DFlash2 K=5≈K=7 (**do not use K=6** —— reproducible per-round cost dip); pure MTP
  K=3/4 tie and lag DFlash2 by ~20% at the same K ⇒ production continues `--spec dflash2`; enabling speculation costs almost no prefill cost; prefill
  1,905→569 tok/s (0→245k) = depth-independent fixed term (AR ≈65%, already at the Gen4 x4 link floor, no software headroom) + depth term (attention accounts for 70%
  at 245k); the GQA 6× request redundancy was **falsified** by the measurement gate ⇒ L1 cancelled.
- DFlash2 draft quantization r54–r66 (2026-09-23/24, opt-in → official): draft gate/up/down/output/kernel/feature projections
  q4 (r62–r66, −173.05 MiB); draftall artifacts −715.6 MiB; acceptance-rate adjudication settled as 30-rep sampling-dominated (15-rep inter-arm noise
  ~±2 pp).
- DSH/agent-loop prefix reuse (2026-09-24): the TP-2 route silently ignoring `--chat-template` is fixed (frontend
  `chat_template_path` passthrough); the template `|trim` assumption was **measured and refuted**; new resident two-round case and documentation.
- First generated token not published (2026-09-24): TP-2 prefill first token now does `preview_model → commit → publish`; replay
  answers go straight to `src=live`; new test case `check_first_token_published`.
- Reuse gating and draft discard (2026-09-24): the premise "unaligned reuse must discard the masked draft" was **refuted by measurement** (same round: discarding 22.5 vs
  keeping 87.1 tok/s) ⇒ delete the gating, `draft_context_declined`, `reuse_grid` and session-average pricing; contract: grid alignment is
  token-for-token identical with the oracle, and within a chunk only boundary crossing is guaranteed.
- ② Fused QKV→q4 (r67, 2026-09-24/25): no new fused kernel is needed (decode uses 3× `ops::linear` row views; append uses the
  existing per-layer generic branch); qkv4 artifact −79.69 MiB, all three suites pass; 30-rep acceptance rate has no measurable cost; the real cost is only the proposal step
  +2.8% (~+0.4% end-to-end).
- ① selector codebook→q4 (r68, 2026-09-25): format-agnostic plan + q4 op oracle (FP64) decoding independently by stored scale;
  the loader no longer pins the format; the single dispatch point is `run_candidate_selector()`; peer selector 245.0→66.9 MiB; cb4 30-rep
  acceptance rate shows no cost.
- r69 official recipe promotion + final artifacts (2026-09-25, user-confirmed): the full DFlash2 draft Q4 set (15 QKV + 2 codebook → q4)
  enters `tools/convert/official_recipes.py::_optional`; final artifacts vs r66: acceptance rate ≤1σ, all three suites pass, peer
  selector 66.9 MiB ⇒ releasable (saves 257.77 MiB, round-time cost ~0.4%).
- r70 four-family final artifacts (2026-09-25): one artifact each for the w4a4 / swift15 / thinkingcap variants, append smoke tests all pass; ThinkingCap
  tokenizer pitfalls fixed (three frontend validations: Split pre-tokenizer, `added_tokens_decoder`, `add_bos_token`; canonical
  tokenizer.json replacement + config merge; serve startup + HTTP Q&A pass).
- TP-2 orchestration and rendezvous (2026-09-25): `forward_tp2_window` eliminates implicit default parameters (1a); shard mask named constants + handshake
  protocol master table + packed assertions (③, the assertion caught a real accumulator omission); 1b give-up abandon-output (in-window give-up goes from
  byte-reproducible illegal address to a designed 503); B7 acceptance: verify graph aggregate +5.2% (6 reps/arm, ≥24 needed to settle).
  Zero numerical impact on the full batch, and the three-suite digest is bit-identical.
- TP-2 transport low-frequency failure investigation (2026-09-25): characterization —— when a give-up lands inside a captured window the kernel uses inconsistent data as an address and crashes,
  and the host-side check comes too late; it does not reproduce in long runs (≤1/1500), while injecting 700 reproduces it byte-for-byte. Remediation implemented: watchdog on by default (large dumps rate-limit one
  per stall), timeout default 2000→10000 ms, abandon-output; the trigger source is undecided.

---

## 3. Outstanding items

### 3.1 Pre-delivery wrap-up

- [ ] Re-test concurrency C=1/2/4 and streaming/stop smoke with the recommended configuration (the Round 31/34 conclusions are based on the bf16 configuration).
- [ ] On delivery, remove this file (PLAN.md) per AGENTS.md; decide the retention/cleanup scope of the roughly 300 one-off diagnostic scripts under
  `tools/tp_bootstrap/` (`build_r35.sh`, `serve_*`, the `r37+r38+r39+r4x` series and the references in `docs/tp2-dual-5060ti.md` are
  reproducible procedures).

### 3.2 Windows single-GPU CLI tiny artifact subtask (user-decided; prerequisites ready)

Goal: produce a `.ninfer` small enough to fit on a single 16 GiB card and run "load → generate → sample" on a single GPU with the Windows `ninfer.exe`.
Prerequisites (Round 54.9): CPU build of torch in `build-win\torch-venv` (torch 2.14.0+cpu); converter interface confirmed.
Next steps: ① run the official recipe directly on a tiny checkpoint (`--recipe qwen3_8_27b_nvfp4 --components text`);
② if that fails, write a tiny recipe/override; ③ put the generation script at `tools/win_port/tiny_model.py` (synthetic tiny config +
random bf16 weights, HF naming); ④ on success, write it into `docs/windows.md` and commit.

### 3.3 Unfixed defects and known limitations

- The TP-2 route `bf16` KV + `--spec mtp` crashes on the first prefill (`prompt.cu:63` `cudaErrorInvalidValue`);
  fp8 + MTP is fine (the recommended configuration is unaffected). Investigation entry point: the launch parameters of bf16 prompt-attention.
- The MTP route likewise uses BF16-add allreduce to move I32 proposal ids around `execution/text.cpp:486` ⇒ when it lands on a
  signaling-NaN bit pattern it is silently rewritten (DFlash2's equivalent exchange has already switched to byte-exact `DevicePair::sendrecv`; MTP has not,
  and the verification cost is high).
- The MTP3 ≥ 70 tok/s threshold is not met (pure decode ceiling K=2 50.5–54.0).
- The TP-2 route has not been run on the perplexity evaluation (quality evidence is multi-sample A/B on the same prompt).
- The per-shard arena's `memory_summary()` still reports `pages 0/0` (a display-only issue).
- Temperature 0 does not guarantee bitwise argmax: DFlash2 sparse acceptance + per-request random seed (`translate.cpp:51-57`) make decoding
  nondeterministic even at temperature 0 ⇒ add `--greedy` when reproducible A/B is needed (existing behavior, unchanged).
- `ninfer_qwen3_5_tp2_sessions_test` plain route "a conversation behind a shared system prompt" first-sample divergence:
  a pre-existing failure (reproduced token-for-token on the r57 baseline artifact and the r66 q4all artifact; the dflash2/mtp routes pass) ⇒ it blocks "all three suites green" (the sessions
  gate now runs on the dflash2 route).
- TP-2 prefill's first sample does not apply the tool-syntax mask (the prefill and decode paths are asymmetric; a separate outstanding item).
- Prefix reuse: text-protocol replay cannot guarantee token-for-token reproduction (client re-tokenization and sampling boundaries differ, or the boundary falls inside a chunk —— the contract only
  guarantees boundary crossing); the only lever that guarantees token-for-token is a continuation protocol carrying token ids (unscheduled).
- Two low-frequency failures (unrelated to changes, all pass on rerun, investigation in §3.6): ① the r67 round q4all arm K=7 `cudaErrorIllegalAddress`
  (req#14); ② the r69 round final arm K=5 first run req#14 HTTP 503 with no FATAL log.

### 3.4 Deferred / optional (unscheduled)

- Round 15 remainder: end-to-end re-test of tool-call constraints on the real model (75 probes); WARN diagnostics; route B (full GBNF);
  the TP-2 `validate_licensed_tokens` guard.
- Round 14 candidates (user-deferred): ① append an anchor at the end of the system+tools prefix (fixes the 46.9 s first round after a new session/compression, saving about 11 s);
  ② prefill throughput (suffix 0.95–1.13k tok/s, cold start 1.35–1.38k); ③ ~0.3 s fixed overhead per request.
- Archive §11.6: MTP priming uses placeholder embeddings for image columns (no evidence of gain); the single-image cap 8,192 is hardcoded (yielding shard 1 can
  retake a larger envelope); shard 0 is idle during visual encoding (acceptable).
- Archive §6: completing the KV dtype scan (nvfp4/k8v4 only verified startup and throughput, no quality data).
- To truly replicate the official artifact's DFlash2 draft from the source weights, one must first obtain its BF16 source checkpoint (this machine only has FP8/EXL3/GGUF).
- L2: AR and MMA overlap —— **implemented and measured: +11% prefill (N=4)**. Implementation: the prefill chunk is cut into N sub-blocks (64-aligned),
  and each sub-block's collective goes on the second stream (`DeviceContext::collective_stream`), overlapping the next sub-block's mixer/FFN
  (`run_layers_tp2_overlap`, option `--prefill-overlap W` = sub-block width, default 256, 0 disables). Measured (chunk 1024, ~6.3k prompt, median of 7 reps): N=0 0.587 ms/token, N=2 0.554, **N=3 0.612 (worse than baseline)**,
  **N=4 0.516 (−12%, reproduced across two batches)**, N=5 0.547, N=8 0.655 ⇒ the optimum is indeed at **N=4 (sub-block 256)**, and **N cannot be interpolated**
  (320/384 fall into slower scheduling tiers); settling should use "sub-block width = 256" rather than extrapolating from N (chunk 512 re-check: serial 0.625 →
  256-wide overlap 0.572, −8.4%, and faster than the chunk 1024 serial baseline 0.587); numerics are bit-identical to the single block (tp2_forward_test's
  chunk-split invariant is still exact at N=2/N=4). Three prerequisites of evidence: the divisor arm in §5.4 (when AR is a 48% block of strictly serial),
  `tools/tp_bootstrap/bench_ar_overlap.cu` (AR is not starved by compute that saturates the SMs, inflating by only 7–18% at engine granularity),
  and Round 12's −5% being merely a problem with its host-side scheduling shape. **Remaining**: the choice of default value/N, independence from the decode
  path, and the failure semantics of the cross-card rendezvous (see §3.6).
- If attention is attacked further: only the compute-path efficiency remains (FP8 PV + compressing the non-tensor FP32 passes), and profiler permission must first be unlocked (ncu
  2025.4.1 is installed, `ERR_NVGPUCTRPERM` is rejected; an administrator PowerShell or NVIDIA Developer Settings allows GPU
  performance counters). Expect the attention item ~1.2–1.4× (245k end-to-end ~1.1–1.3×).
  **Started 2026-09-27 as §6** (including quality risk, variants and gates).
- Hardware lever: move card 1 from the chipset Gen4 x4 slot (~7 GB/s) to a CPU-direct Gen5 x8 slot (~20 GB/s): zero code, 1.78× at shallow/medium context,
  1.15× at 245k; requires a free motherboard slot.
- DFlash2 K=6 per-round cost dip (a constant 25.7 rounds/s across three rounds, below K=5 (27.9) and K=7 (26.6)): suspected kernel boundary effect from bucketing/aligning
  by window width; use `NINFER_TP2_TIMING=1` to get per-phase per-round timings and investigate.
- The exact source of DFlash2 prefill being consistently 3–4% slower (no per-phase attribution done; `NINFER_TP2_TIMING=1` splits prefill per phase).
- B7 settling: the verify graph A/B is 6 reps/arm (aggregate +5.2%, per-request t=0.99 not significant); ≥24 reps/arm are needed to settle.
- 1a residue: fold both sides' actual arguments into a single struct (the two call sites are already explicit and consistent, diminishing returns).
- Items not measured during the DFlash2 TP-2 period: the true free memory of the Windows native build (the ledger's `free` comes from `cudaMemGetInfo`, and WSL2 under-reports
  by about 1 GiB); numerical equivalence of splitting the draft by head/row; the K=15 workspace peak; the per-step convergence of the target's 5 residual layers across shards
   overhead; the latency of the per-layer allreduce of the draft; the equivalence of sparse rejection sampling under TP-2 sharded logits.
- Gate revision decision (separate from the promotion): the literal gate “greedy |Δacc| ≤ 1.0pp” did not pass at K=7 (−1.60pp, proven to be confounded by cross-arm text
  drift); the promotion was completed as user-confirmed (r69), and 30-rep sampling-dominated adjudication is the de-facto method — if the gate is ever formally revised, do it in this one place.

**Design constraints (do not re-investigate)**: copying the draft in full is not viable — under full context (262144 / fp8 / MTP K=2) free is only
697/1217 MiB, while the draft is 2.07 GiB/card; after splitting, about 1.04 GiB per card. The proposed condition is the residual concatenation projection of the target 5 blocks;
TP-2 splits 64 layers across two cards ⇒ a cross-card handoff is required (the residuals are bit-identical on the two cards, so capturing shard 0 alone suffices). The candidate selector reads the
full-vocabulary codebook directly (about 254 MiB), which must be whole. DFlash2 and MTP are mutually exclusive: choosing DFlash2 can free shard 0's MTP weights
430 MiB + KV 516 MiB. The per-round cost composition, the D1–D4 decisions and the negative conclusions of each round are in the archive.

### 3.5 Items blocked by the environment (covered via alternative routes)

- [ ] WSL-side op tests + byte-identical + frontend fixture tests: **WSL2 CUDA driver crash** (PTX JIT
  segfault inside `cudaGetDeviceCount()`, GPU occupied by a Windows service); the jinja test passing proves the binary is sound, so the op tests were moved to the Windows side. The Windows side
  already passed (NVFP4 A4/A16, frontend, jinja; the `NINFER_OP_REPORT_STATS=1` error metrics are within tolerance) ⇒ the WSL build_dyn
  serves only as compile verification.

### 3.6 TP-2 transport low-frequency failure (remaining)

- **Characterization (proven)**: after a bounded-spin give-up, the partial sums of each collective in this round are **local only** (the peer's contribution missing), while the round
  keeps running; when the give-up lands **inside a captured window**, the window's own kernel uses the inconsistent data as an **address** ⇒
  illegal address, and no host-side check is in time (injection 700 reproduces it byte-for-byte); only a give-up in eager prefill turns into a
  clean 503 + recovery on the next request.
- **Trigger source undecided**: natural long runs (cumulative 1500 requests, watchdog on) 0 events, natural rate ≲1/1500; it requires >2 s of
  host/device delay, or one real desync (id reuse/mismatch). The watchdog is on by default; on failure it additionally saves `*-FAILED.log` /
  `*.arwatch.txt` (interpretation by the magnitude of the id difference: a whole group of 0 = that side never reached the arrival write; 1 = OFF-BY-ONE; a whole
  block = BLOCK-LEVEL; otherwise SKEW).
- **(b) making give-up safe (pending decision)**: host-side checks have been falsified; either poisoning/abort semantics after give-up, or making operators that index inside the
  window bound their inputs — both are architecture-level changes and require pinning down the exact operator first.
- **Tool boundary**: `CUDA_LAUNCH_BLOCKING=1` / compute-sanitizer are incompatible with this transport (side A's launch becomes blocking, and side B's
  kernel only launches after it returns ⇒ the 1st collective necessarily gives up, self-deadlock) ⇒ launch-serializing tools cannot be used for
  kernel-level attribution; only source-level probe bisection works.


---

## 4. Architecture deep read and defect analysis (2026-07-18, appended)

> This section is an independent full-architecture deep read (without using subagents), appended on top of the old TP-2 plan, without modifying the text above.

### 4.1 Scope of the deep read

Read through the three architecture authority documents in `docs/maintainer/` (engine-architecture, paged-kv-cache, resource-scheduling),
and read the C++ core layer by layer: `src/core` (paged_kv_cache, host_kv_arena, tp/device_pair, arena,
linear_attention_state, gdn_replay_records), `src/runtime` (engine_core, scheduler,
tp2_generation_core, contract, kv_capacity), `src/serve` (generation_service, http_server/transport,
openai/anthropic routes, translate), `src/ops` (sampling, causal_attention, kv_cache_append),
`src/models/qwen3_5` (execution/text, program/decode, frontend/output_session, prefix_identity,
dflash_round), and cross-checked the known defects of §3.3 above against git history.

### 4.2 Determinism / latent bugs

**Re-check result (2026-09-26): the only “confirmed latent bug” in this section is a false positive and has been withdrawn — both plane
orders of `zero_pages` are correct.** The derivation and the executable criterion follow.

- **〔Withdrawn〕`src/core/paged_kv_cache.cpp` — the “out-of-bounds write” in the HeadMajor branch of `zero_pages` (original text 2026-07-18)**:
  The original text asserted that for the HeadMajor layout (plane shape `[X,64,N_phys,H]`)
  `cudaMemset2DAsync(dst=base+first*nb2, dpitch=nb3, width=count*nb2, height=H)` shifts the i-th row forward by
  `i*elem_size`, that the total write range exceeds the target block by (H−1)×elem_size bytes, and suggested changing it to a 1D
  `cudaMemsetAsync(base+first*nb2, 0, count*nb2)`.
  **Re-check: the premise was wrong — the order of `ne` is not numpy-like “last dimension innermost”.**
  `Tensor::set_contiguous_strides` (`tensor.cpp:50-57`) multiplies dimension by dimension starting from `nb[0]=dtype_size` ⇒
  **`ne[0]` is innermost**, while `plan_device_kv_page_pool` (`paged_kv_cache.cpp:90-93`) builds for HeadMajor
  `{leading_extent, kPagedKVPageSize, physical_pages, head_extent}` ⇒
  `nb[2] = 64*leading*es` (**bytes per page**), `nb[3] = physical_pages*nb[2]` (**head stride**),
  `ne[3] = head_extent` (number of heads; the existing case already pins `ne[2]=pages`/`ne[3]=heads`, see
  `tests/test_kv_cache.cpp:247-251`). Hence `dpitch=nb[3]` is exactly “one head per row”, the i-th row covers
  `[base+first*nb2+i*nb3, +count*nb2)` = the page range of head i (row 0 no longer “exactly covers”, all rows are equidistant), and the
  last byte `first*nb2+(H−1)*nb3+count*nb2 ≤ H*nb3` = the end of the plane (run length ≤ total pages) ⇒ **no out-of-bounds**;
  moreover `dpitch = physical_pages*nb2 ≥ count*nb2 = width`, satisfying `cudaMemset2DAsync`'s pitch constraint.
  The original text mistook the initial shape list for memory order, and that is why it read `nb[3]` as the element size; the 1D memset it suggested **only clears head 0**,
  leaving the pages of the other heads at their original values, which is not a correct implementation.
  **Criterion** (new `exercise_zero_pages` in `tests/test_kv_cache.cpp`): for each of the two plane orders take the middle run
  `[2,5)` and the last page `[7,8)`, first fill the whole pool with a deterministic bit pattern, run `zero_pages`, then compare against the expected value **byte-for-byte** per plane
  (all 0 inside the run, byte-for-byte unchanged outside the run). The current implementation **PASS**es; after temporarily swapping the implementation for the 1D memset suggested by the original text it **FAIL**s
  (4 new cases + the existing `selective zero left page payload bytes`, 5 in total) ⇒ this case can indeed decide the matter.
  At the same time, a comment was added to `PagedKVPlaneOrder` in `paged_kv_cache.h` stating “the enum name refers to the slowest dimension; the shapes and the
  page/head strides of the two orders”, to avoid reproducing the same misreading.

### 4.3 Verification of the known defects in §3.3 above

①②③④⑧ are **all fixed** in the current code (evidence in 4.2 and the parenthetical notes of the §3.3 entries); ⑤ (give-up safety) is mitigated but remains an open architecture issue;
⑥⑦⑨⑩ are known limitations/undecided and were not expanded on this time.

### 4.4 Design defects and latent risks (architecture level)

1. **Dual execution core drift risk**: TP-2 goes through a separate `TP2GenerationCore`, bypassing the EngineCore scheduler/CUDA Graph/
   ResourceManager (single-request lockstep). Request lifecycle, capacity, cancellation, prefix reuse, checkpoint ring and similar logic are implemented twice across the two
   paths and must be kept semantically consistent long-term — this has already repeatedly produced prefill/decode asymmetry defects (tool masks, first-token timing,
   MTP ids swapping are all instances). The biggest maintainability risk.
2. **TP-2 in-kernel allreduce give-up safety still open**: when the timeout lands inside a captured window the kernel zeroes the local output slice to avoid
   inconsistent data being used as an address, but the round still completes with mixed real/zero data, relying on host `ar_stalled()` + a convergence check to turn into a failure.
   Mitigated (bounded spin + watchdog + `abort_if_ar_stalled`); “making give-up itself safe” is pending decision.
3. **The host-staging CPU allreduce fallback is a performance cliff**: when the in-kernel transport is unavailable it falls back to host staging, where the element-wise add is completed in a
   CPU host loop with both streams synchronized, extremely slow for large prefill payloads. The fallback is documented, but it constitutes a marked performance cliff.
4. **TP-2 prefix reuse capacity hard cap**: the device reuse snapshot is fixed at `kReuseSnapshotCount=2` (73 MiB/shard per slot),
   deeper divergence relies on the host checkpoint ring (one PCIe round trip). An explicit capacity tradeoff; the reuse cost in deep-divergence scenarios is structurally raised.

### 4.5 Suspicions ruled out after investigation

- **`queue_cv_.wait_for(1ms)` at the tail of the `engine_core.h` worker loop**: initially judged to add 1 ms of latency to every decode round,
  re-check **does not hold** — every execution branch (control/prefill/decode) does `continue` straight back to the loop top, and the 1 ms wait is reached only on the “no execution
  unit (action==Wait)” idle path; when busy it is a tight loop.
- `host_kv_arena` (first-fit + coalesce + consistent plan/apply offsets), `kv_capacity` (overflow checks complete),
  `sampling` (greedy/stochastic dual path, RNG pure-function replayable under graph), `causal_attention` (online-softmax
  split merge, envelope validation), `output_session` (UTF-8 incremental decoding, stop matching, reasoning channel separation,
  preview/commit), `prefix_identity` (FNV-style digest) were all reviewed; the implementations are rigorous and no determinism defect was found.

### 4.6 Conclusion

Overall engineering quality is high: boundary validation, overflow checks, lifetime ownership, and numerical oracle conventions are in place, and all the known defects of §3.3 are fixed in the current code.
The only “confirmed latent bug” this time (the `zero_pages` HeadMajor out-of-bounds write) has been **withdrawn** per the re-check in §4.2 — the original text used the wrong `ne` order,
the code is correct for both plane orders, and a byte-level case has been added; among the **4 architecture-level risks**, “TP-2 dual-core drift” and “TP-2 give-up
safety” most deserve follow-up investment (the former see the implementation record in §4.7, the latter is still undecided).

### 4.7 Investigation of solutions for §4.4.3 (host-staging CPU allreduce performance cliff) (2026-09-26)

> An investigation of the available options for point 3 of §4.4 (falling back to host staging when the in-kernel transport is unavailable, element-wise add in a CPU host loop + synchronized dual streams,
> extremely slow for large prefill payloads). Investigation and evaluation only; no code changed.

**Current-state inventory (code evidence)**

- Three tiers of transport (`src/core/tp/device_pair.cu`): in-kernel (mapped pinned staging + arrival id,
  `815-871`; the current production path, measured to be within 2% of the Gen4 x4 link's bidirectional bandwidth lower bound: `809-814` comment) → P2P
  (`872-895`, requires `cudaDeviceCanAccessPeer`, probe fails under WSL2 at `542-552`) → host staging fallback
  (`896-928`).
- There are only two fallback trigger conditions: (a) `cudaHostAllocMapped` fails at construction (`in_kernel_available_==false`,
  `553-623`); (b) payload > `kInKernelArBytes` (24 MiB, `249`).
- Trigger (b) is **unreachable** under the current product parameters: the prefill chunk is clamped at `kPrefillChunkMaximum=1024`
  (`tp2_generation_core.cpp:164,2545-2547`), the largest payloads are mixer/MLP delta
  5120×1024×2 = 10 MiB and gather activation 6144×1024×2 = 12 MiB, both < 24 MiB. But “payload ≤ staging”
  is only a numerical coincidence — the chunk clamp (`2545-2547`) only references `kPrefillChunkMaximum` and is not derived from
  `in_kernel_bytes_`; it is an implicit invariant rather than an explicit constraint. The realistic trigger is (a): environments where non-WSL2 / mapped-pinned
  is unavailable.
- Cliff cost breakdown (`896-928`): ① **scalar single-threaded CPU `__hadd` loop** (`919-921`) — single-threaded about
  0.5–2 GB/s, about 5–20 ms per 10 MiB payload (2.7 ms for the same payload in-kernel), the dominant term; ② `cudaStreamSynchronize`
  on the two compute streams (`913-914`) — a full pipeline stall per layer; ③ **not graph-capturable** —
  `tp2_generation_core.cpp:385,451` downgrades the MTP chain, decode, and verify graphs all to EagerExact when `in_kernel_allreduce()==false`.
  PCIe traffic is the same as in-kernel (2× payload per card), so the cliff is not a bandwidth problem,
  it is a “CPU compute + full synchronization + no graph” problem.
- Amplification: 2–3 collectives per layer (`text.h:564,576,589`) × 64 layers, 128–192 per chunk;
  with the fallback the per-chunk allreduce cost rises from ~0.35 s to the 2–4 s range.
- Test coverage gap: the largest-payload case in `tests/test_tp_device_pair.cpp` is 2433024 B (`666`),
  the >24 MiB fallback path and the mapped-pinned-unavailable path **both have no existing case** (`51` comment claims the three paths are bit-exact,
  but the fallback path has never been executed on this machine).

**External corroboration (web)**

- The standard drivers for consumer dual-card PHB topologies **do not enable P2P**; even bare metal requires a patched open kernel module (BAR1)
  ([vLLM forum: dual 5090 SHM vs P2P](https://discuss.vllm.ai/t/dual-rtx-5090-tp-2-shm-vs-patched-bar1-p2p-cumem-single-pass-2-7-mean-point-estimate/2870):
  enabling P2P gains only +2.67% on average, workload-dependent; another set of multi-round measurements was −5–7%). WSL2 is Hyper-V PCI passthrough +
  a Microsoft-managed kernel, so a patched module is not viable. → The P2P route **does not apply** to this product's target (WSL2 dual 5060 Ti),
  and even on bare metal the gain is only a few percent.
- Without P2P, NCCL uses the **SHM transport** (the GPU writes shared host memory over PCIe and the peer GPU reads it back,
  [Demystifying NCCL](https://arxiv.org/html/2507.04786v1);
  [NCCL issue #1838](https://github.com/NVIDIA/nccl/issues/1838)) — isomorphic to NInfer's in-kernel
  mapped-pinned route, which shows that the in-kernel path is already the standard solution for this environment and there is no better host-side transport route.

**Option evaluation**

| # | Option | Effect | Cost/risk | Conclusion |
|---|---|---|---|---|
| S1 | **Fix the fallback itself**: change the CPU add to SIMD + multithreading (AVX2/AVX-512, add in the float domain and round back to BF16 RNE to preserve bit-exactness), chunk the payload + double-buffered pipelining (overlap D2H(i+1) with add(i)) | Fallback goes from a “~5–10× cliff” to “link-limited”, the same order as in-kernel; introduces no extra loss beyond the graph downgrade | The change is confined to the host-side loop in `device_pair.cu`; bit-exactness can be verified directly with the existing three-path consistency assertion | **Recommended (primary)** |
| S2 | **Enlarge in-kernel staging**: `kInKernelArBytes` 24→48/96 MiB (cost is only host pinned RAM) + derive the prefill chunk clamp from `in_kernel_bytes_`, turning “payload ≤ staging” into an explicit invariant | Closes trigger (b) for good, immune to wider models/larger chunks | One constant + one clamp line; very low risk | **Recommended (secondary)** |
| S3 | P2P (bare-metal BAR1 patch) | A few percent on bare metal, workload-dependent | Not viable under WSL2; introduces a driver dependency | **Ruled out** |
| S4 | copy-engine event-style large-payload path | — | Already ruled out by measurement: `device_pair.cu:813-814` “copy engines were slower”; the ceiling probe in `test_tp_device_pair.cpp:422-427` (`NINFER_TP2_AR_COPY_BENCH`) confirms sliced in-kernel is already within 2% of the link lower bound | **Ruled out (existing internal evidence)** |
| S5 | Payload quantization (delta to FP8, halving bytes) | Halves bandwidth | Changes the numerical semantics of the residual, requires a full oracle qualification; in-kernel is already at the link lower bound, no gain for the current target | **Out of current scope, not done** |

**Recommendation**

- Do **S1 + S2** as one batch (diff confined to `device_pair.cu` + the clamp in `tp2_generation_core.cpp` +
  `tests/test_tp_device_pair.cpp`): the fallback becomes link-limited, trigger (b) is closed, the chunk clamp becomes explicit.
  Keep the graph downgrade (EagerExact) — host synchronization is an inherent property of the fallback, and after S1 the only degradation left in non-WSL2 environments is “no graph”,
  no longer a cliff.
- Verification: add a >24 MiB payload case to `test_tp_device_pair.cpp` (exercising the fallback path, reusing the three-path bit-exact assertion);
  add a test hook that can force the fallback (e.g. an env that skips mapped-pinned initialization), and A/B time the fallback path before and after.

**Bare-metal Windows applicability**: the conclusion holds as-is. Under a typical desktop PHB topology P2P is likewise not authorized (closed-source drivers
refuse P2P for GeForce; authorization is based on NVIDIA's chipset allow-list, and consumer chipsets are not on it; [club-3090 six-card model](https://github.com/noonghunna/club-3090/blob/master/docs/PCIE_P2P.md)),
the transport layering matches WSL2, and the S1+S2 benefit/cost holds item by item. The only fork: if P2P happens to be authorized (workstation
chipsets on the allow-list / PIX switch topology), `DevicePair` automatically takes the P2P path, the host-staging cliff does not exist, and S1+S2 are unnecessary;
the correct improvement in that scenario is kernel-based P2P exchange (graph-capturable), which is beyond the current target scope.

**Implementation result (2026-09-26): S1 + S2 landed**

- **S1** (`src/core/tp/device_pair.cu`): the fallback reduction was changed to SSE2 vector adds (8 BF16/iteration; BF16→FP32 shift left,
  FP32 add, RNE round-back per `__float2bfloat16`, and NaN canonicalized to `0x7fff`) + sharding across host threads for 32 MiB and above
  (up to 8). The structure is `2×D2H → dual-stream sync → reduction → 2×H2D` — the 1 MiB chunked pipeline implemented first measured
  **about 1 ms slower** than a single large copy (the CE overhead of 20 small H2Ds > the overlap gain), so it was reverted to a single copy.
  Measured (`NINFER_TP2_AR_STAGING_BENCH=1`): 10 MiB payload old ≈ 35.5 ms (scalar add 32.4 + transfer 3.1) →
  new **5.2–8.0 ms** (5–7×); 50 MiB old ≈ 175.7 ms → new **20.5 ms** (8×).
  Basis for the 32 MiB thread threshold: at 10 MiB, 8 threads add in 4.1–4.4 ms, slower than single-threaded 2.1–2.5 ms (Windows
  thread creation costs a few hundred microseconds), while at 50 MiB, 8 threads take 8.4 ms, beating single-threaded 12.0 ms.
- **S2** (`device_pair.{h,cu}` / `tp2_generation_core.{h,cpp}`): `kInKernelArBytes` 24 → 48 MiB;
  added `DevicePair::in_kernel_allreduce_bytes()` and `TP2GenerationCore::prefill_chunk_width()`,
  and the three chunk computation sites are all now derived backwards from the staging capacity, so “payload ≤ staging” has changed from a numerical coincidence into an explicit invariant.
- **Verification**: `ninfer_tp_device_pair_test` PASS. Added 25 MiB (stays in-kernel; under the old 24 MiB it would have fallen back),
  50 MiB (over staging ⇒ fallback + sharding), a forced-fallback comparison of 3 MiB+16 / 50 MiB / deep queue / sendrecv, plus
  736 directed bit patterns (ties 12 / inf 84 / nan 247) — both transport paths are bit-exact.
- **Trigger re-check (2026-09-26 continued)**: replacing the fallback's D2H/H2D with “kernel-direct write to mapped pinned” was
  rejected by measurement and reachability review — for the same payload (20 MiB bidirectional per card) the copy engine is only about 14% slower than the kernel path
  (3.14 vs 2.70 ms), and replacing it is worth only ≤8% to the fallback overall (2.1 ms of the fallback total is the CPU add, where S1 has already reached the floor);
  and **a size-triggered fallback is unreachable in the production configuration**: the largest prefill payload is 10–12 MiB (hidden 5120 / gather 6144
  × chunk 1024), the vocabulary-parallel head merge is `[V,1]` (production prefill passes `logits_columns = nullptr`) and the
  verify window's `[V, W≤16]` is ≤ 8 MiB, all within 48 MiB. The hosts that can trigger the fallback are precisely the “mapped pinned
  unavailable” class — where kernel-direct write is by definition impossible.
- **Changed to S6 · staging ladder**: reserve a stepwise degradation 48→24→12 MiB (two parity slots/card; 12 MiB is the level at which "no payload of this product
  narrows it", and the widest capturable payload is the vocabulary-shard head merge at 7.6 MiB), preserving the in-kernel
  transport and letting S2's chunk clamp adapt automatically to a lower level, replacing the original “all or nothing → host
  synchronization on every collective”. `NINFER_TP2_AR_STAGING_MIB` lets tests reproduce a constrained host.
- **Incidental fix (found during the ladder verification)**: the fallback's pinned staging had no parity, and two independent compute streams shared one block ⇒
  the next call's D2H overwrote the half being uploaded by the previous call's H2D, and the device received a partial false sum (manifesting as logits changes and downstream
  even `illegalAddress`). Changed to two parity slots + synchronizing both streams before growth; added a 4 MiB deep-queue regression case (the 20 KB level
  could not catch it because the H2D is too fast).

---

## 5. Porting plan for upstream ninfer-all performance items (appended 2026-09-26)

> Source: `infer-all` — the multi-fork integration line (base `ashalliants/ninfer-3090` v0.11.0,
> mostly written by Warlax); its commits are replayed one by one after the baseline moves, so commit hashes are not shared with local; after aligning with `git patch-id --stable`,
> of the 1770 non-merge commits on ninfer-all master, 1113 share a patch with local and 657 are missing locally (of which 119 are perf/bench).
> This section only collects items that benefit **2×5060 Ti TP-2**, ordered by benefit/cost.
>
> Criterion (§1 and `docs/tp2-dual-5060ti.md`): decode is bandwidth-bound by the per-shard weight stream (10.15 GB/forward ÷ 448 GB/s
> ≈ 22.7 ms lower bound, measured round 38.2 ms); there are also 128 allreduces/token (9–18 µs), the MTP draft chain host-serial
> 7–10 ms, launch gap ~2.9 ms (1009 kernels/forward), and a hard 16 GiB device-memory constraint. Ordering: reduce weight bytes per card >
> reduce launch/kernel count ≈ reduce communication count > raise prefill compute > pure compute optimization.

### 5.1 Execution order and status

| # | Item | Upstream commit | Status |
|---|---|---|---|
| 1 | per-device `cudaFuncSetAttribute` cache (the second card's smem limit was not raised) | `9adeb90c` (fixes 19 sites) | **Done** 2026-09-26 |
| 2 | 170 SM hardcoding → runtime device SM count | `d7bbcf57`, `7afc8e17` | **Done** (3 code sites; 1 awaiting measurement) |
| 3 | Reduce kernels/stack frames: RDC-off + attention gate folding + INT8 scale via smem | `ea74dca4`, `77c9cc39`, `a941c9f4` | 3a **not supported by measurement, reverted**; 3b/3c deferred (see below) |
| 4 | Reduce weight bytes per card: embedding/head and MTP experts Q4/Q6 | `1da31697`, `f1982279`, `f8c75edd` | Pending decision: this artifact is FP8 vocabulary/head, the equivalent action = change the FP8 tensors that must be streamed to NVFP4 (~17%, needs a quality A/B) |
| 5 | verify segment: small-T tensor-core kernels + GDN record window staging | `ce2df46b`, `21df3069` | 5b ported (Op level −22~27%, not measurable end-to-end); 5a not done after re-check |
| 6 | Big items (need product decisions): KV codec / device route profile / fast prompt kernel | `eb9f7a23`/`ad26b362`/`2ba10da2`, `0b2f8384`/`81861a07`, `4303e604` | TODO |

### 5.2 Scope and verification per item

**1. per-device `cudaFuncSetAttribute`**
- Fact: this attribute is an attribute of “that function on the current device”; a function-local `static` makes the first device record the result and the second device skip the call.
  Local already has the correct implementation `src/ops/common/cuda_smem.h::ensure_max_dynamic_shared_memory` (keyed by (kernel, device),
  degrading to an unconditional call once the slots are full, still correct), but only 4 `small_t` sites are wired up.
- To fix (10 function-local `static`s, schedule permits up to 99 KiB > 48 KiB):
  `src/ops/attn_input_proj/bf16/bf16_attn_input_gemm_mma.cu:59`,
  `src/ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu:34`,
  `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu:295`,
  `src/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu:33`,
  `src/ops/linear/bf16/bf16_launch.cuh:42`, `src/ops/linear/bf16/shapes/n256_k5120.cu:13`,
  `src/ops/linear/fp8/fp8_launch.cuh:59`, `src/ops/linear_add/bf16/bf16_linear_add_gemm_mma.cu:51`,
  `src/ops/linear_add/fp8/fp8_linear_add_a8.cu:32`, `src/ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu:36`.
- Three duplicate implementations of the same mechanism are likewise consolidated into that helper (logically equivalent): the `static bool attr_done[64]` in `prompt_fp8.cu:27`,
  `prompt_k8v4.cu:22`, `prompt_nvfp4_non_rdc.cu:19`, and the call on every launch in `prompt.cu:28-35`.
- Approach: change them all to go through `ensure_max_dynamic_shared_memory(reinterpret_cast<const void*>(kernel), bytes)`,
  keeping the existing `if constexpr (>48 KiB)` condition.
- Verification: `cmake --build build-win -j` passes; `git grep 'static const cudaError_t'` is zero; the related op tests
  (`ninfer_tp_device_pair_test`, attention/linear suites) pass; running a path containing a >48 KiB kernel on the second card no longer
  returns `cudaErrorInvalidValue`.

**Implementation result (2026-09-26)**

- All 10 function-local `static`s switched to `ensure_max_dynamic_shared_memory`:
  `ops/attn_input_proj/bf16/bf16_attn_input_gemm_mma.cu`,
  `ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_materialized.cu`,
  `ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu`,
  `ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu`, `ops/linear/bf16/bf16_launch.cuh`,
  `ops/linear/bf16/shapes/n256_k5120.cu`, `ops/linear/fp8/fp8_launch.cuh`,
  `ops/linear_add/bf16/bf16_linear_add_gemm_mma.cu`, `ops/linear_add/fp8/fp8_linear_add_a8.cu`,
  `ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu`.
- 4 duplicate implementations consolidated into the same helper: `prompt.cu` (previously called on every launch), `prompt_fp8.cu`,
  `prompt_k8v4.cu`, `prompt_nvfp4_non_rdc.cu` (previously each had its own `static bool attr_done[64]`).
- `git grep 'static const cudaError_t'` is now 0; all sites uniformly include `#include "ops/common/cuda_smem.h"`.
- Evidence: `pwsh -File tools/win_port/build.ps1 -SkipProbe` full build exit 0 (192 steps, 89 s); all 17 affected
  suites green (device, tp_device_pair, rmsnorm, gated_rmsnorm, gdn_gating, softmax_attention,
  rmsnorm_rope, attn_input_proj, gdn_input_proj, dynamic_grouped_conv×2, linear_fp8_a16/a8,
  linear_bf16_a16, linear_add_bf16/fp8, linear_swiglu_fp8, 317 s).
- **Real-artifact verification (2026-09-26)**: ran the dual-card artifact `<models-dir>/qwen3_8_27b_swift15_dflash2_final.ninfer` (20.10 GB):
  `ninfer_qwen3_5_tp2_dflash_append_test` **PASS** (47 s, both cards executing the forward + DFlash2 proposal simultaneously);
  `ninfer_qwen3_5_tp2_dflash_solo_test` **PASS** on `<models-dir>/qwen3_8_27b_w4a4_dflash2_q4all.ninfer` (41 s).
  Both failing items are **pre-existing and already recorded** and unrelated to this round's changes: the plain divergence of `tp2_sessions_test`
  under the default three routes (worklog:6117 specifies that this gate uses `NINFER_TEST_ROUTE=dflash2`); the stable divergence of `tp2_dflash_solo_test`
  on the swift15 artifact (worklog:6548 records “after `git stash` it was rebuilt at HEAD and failed identically token by token”, the candidate cause being that this artifact
  does not match the test baseline). ⇒ The path of >48 KiB kernels on the second card is now covered by a real two-card forward.
- Follow-up candidates (not done, within item 3's scope): there are still per-launch calls to an unconditional `cudaFuncSetAttribute`
  (`context_kv_materialize/materialize.cu`, `linear/gguf/ggml_bridge_mmq.cuh`,
  `linear/nvfp4/nvfp4_w4a4_tma.cu`, `linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu`,
  `chunked/{output,prepare_wy_wu,state_passing}.cu`, `linear_topk/{fp8,q4,q8}_m64.cu`). They are correct, but every
  launch goes through the driver; TP-2's ~1009 launches per forward add up on the host side, and it is worth consolidating once there is timing evidence.

**2. 170 SM hardcoding**
- Sites and approach: `src/ops/launcher/rmsnorm.cu:20` (`kRmsPrefetchBlocks`), `src/ops/launcher/rope.cu:16`,
  `src/ops/linear_attention/gated_delta_net/chunked/output.cu:9` (`kRtx5090SmCount`),
  `src/ops/softmax_attention/dense/causal_cache/small_t.cu:66,230`,
  `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:33,42` → query the device SM count at runtime (cached per device index,
  falling back to 170 on query failure). 5060 Ti has 36 SMs ⇒ 170 blocks ≈ 4.7 waves, and rounding/prefetch/split-K all mismatch.
- Verification: existing tests for the target ops; A/B timing of rmsnorm / gdn gating / chunked output / small_t on the 5060 Ti.

**Implementation result (2026-09-26)**

- Added `src/ops/common/device_sm_count.h::device_sm_count(fallback = 170)`: caches `cudaDevAttrMultiProcessorCount` by device index,
  clears the error and falls back to 170 on query failure (consistent with upstream `d7bbcf57`'s convention).
- Three code sites changed to a runtime query (on a 170-SM device the value is unchanged, behavior bit-identical):
  `ops/launcher/rmsnorm.cu` deletes `kRmsPrefetchBlocks = 170`, and both thresholds use `device_sm_count()`;
  `ops/launcher/rope.cu`'s `kLargeBlockWaveCapacity = 1020` becomes `kLargeBlockCtasPerSm(6) *
  device_sm_count()`; `gated_delta_net/chunked/output.cu`'s `kTargetCtas = 170 * 4` becomes
  `device_sm_count() * kCtasPerSm`.
- Two sites **only comments changed, no code**: the 170s at `small_t.cu:66,230` appear only in comments and the code uses a literal CTA
  target aligned with “one wave” (160/320); `bf16_gdn_gating_proj_plan.cpp:33,42` likewise only in comments, and notes “the launcher
  independently enforces actual-device residency”, with the SM count already passed in by the caller. Changing 160/320 to SM-derived values would
  change the existing behavior on the 5090, so a 5060 Ti A/B with batch_size>1 is required first.
- Evidence: build exit 0; all 18 related suites green (rmsnorm, rmsnorm_pack_tail, gated_rmsnorm, gated_delta_net,
  gdn_gating, gdn_gating_proj, rope, rmsnorm_rope, linear_tp2_split×3, 61 s). These tests already run on this machine's
  two 5060 Tis, so the new paths (36 / 216 / 144) have been accepted against the Op oracle.
- Runtime coverage of the 36 SM path: the 2026-09-26 dual-card real-artifact runs (append PASS, solo@q4all PASS) executed both cards'
  forwards simultaneously on the 36-SM 5060 Ti, so the new values (36 / 216 / 144) have run through the real decode/verify paths.
- TODO: A/B timing on the 5060 Ti (rmsnorm thresholds, rope block selection, chunked GDN wave target), and measurement of the small_t
  160/320 targets.

**Implementation result (2026-09-26): 3a not supported by measurement, reverted; 3b/3c deferred**

- **3a disabling RDC (`ea74dca4`) — reverted**. Using the same metric as upstream (`STACK:` from `cuobjdump --dump-resource-usage`,
  i.e. the number of functions with stack frames) to compare `apps/ninfer-serve.exe` in this tree:
  - Before (`ninfer_cuda_archive` = `CUDA_SEPARABLE_COMPILATION ON` + `CUDA_RESOLVE_DEVICE_SYMBOLS ON`):
    4405 functions, 231 with `STACK>0`, total stack frames 8328.
  - After (both removed, whole-program device code): 4395 functions, **269** with `STACK>0` (+16.5%), total 9200.
    By family: rope_fixed yes→no, gdn 8→0, q8 13→8 (improved); small_t_i8 90→150, q4 10→13 (worse).
  - Upstream's “1155→927” comes from their tree; this tree is a net +38, the opposite direction. TP-2's decode is bandwidth-bound by the per-shard weight stream,
    in-kernel instruction efficiency is not on the critical path; this machine has no `.ninfer` artifact, so an end-to-end A/B cannot be done, and therefore a
    full-rebuild-scale build change should not be introduced without evidence. `git checkout` was used to revert `cmake/NinferTargets.cmake`, `src/ops/CMakeLists.txt`,
    `src/ops/linear/gguf/sources.cmake`, and a full rebuild restored them.
  - A full `ctest` run in the same period (RDC-off build, 135 items): 4 host-side failures (`chat_templates` = Python-side GBK
    encoded emoji error, `resource_manager`, `engine_options`, `serve_options`, all pre-existing failures unrelated to this change),
    plus a 300 s timeout in `context_kv_materialize_test` (still times out when re-run alone). That timeout was not attributed, and is one of the reasons for the revert.
  - Note: `src/ops/CMakeLists.txt` records that the nvfp4 warp-specialized kernel is archived separately without RDC because `setmaxnreg` is broken by RDC;
    once RDC is actually turned off in the future, that archive and `ninfer_cuda_archive` will no longer differ.
- **3b folding the attention gate into the reduce epilogue (`77c9cc39`) — deferred**. Local
  `src/models/qwen3_5/execution/text.cpp` has 3 separate `ops::sigmoid_mul` (613/762/1507). Folding it in requires adding an optional gate parameter to the
  causal attention Op and implementing it in each route's epilogue (spanning `include/ninfer/ops/softmax_attention.h`
  and several `.cuh`). Benefit: one fewer launch per layer, at 1009 launches ≈ 2.9 ms gap ⇒ ~2.9 µs/launch × 28 layers ≈ 0.08 ms,
  0.2% of the 38.2 ms round. Benefit below risk, not done for now.
- **3c INT8-family attention scale via shared memory (`a941c9f4`) — deferred**. Sites are the computed-lane
  `__shfl_sync` at `prompt_i8.cuh:223/224/261/262/297/298/387` and `small_t_i8.cuh:331/332/440/441/540`. Two reasons for deferring: the recommended
  configuration is `--kv-dtype fp8` (which goes through `*_fp8.cuh` rather than the i8 family); and after this tree turned RDC off the
  small_t_i8 stack frames actually increased, indicating that family's lowering in this tree differs from upstream, so SASS attribution of whether there really is an out-of-line
  call is needed before deciding the change.
  - **Revert verification (2026-09-26)**: after the revert a full rebuild (533 s, exit 0), `ninfer-serve.exe` re-measured 4405 functions /
    **231** `STACK>0` / 4486 resource records, matching the pre-change values one by one ⇒ the metric is reproducible in this tree, and +38 is indeed caused by turning RDC off.
    Also: `ninfer_context_kv_materialize_test` times out at 300 s under the RDC build as well ⇒ that timeout is a **pre-existing issue** unrelated to this change;
    the other 4 host-side failures out of 135 (chat_templates / resource_manager / engine_options / serve_options)
    are likewise pre-existing.

**3. Reduce kernels/stack frames**
- `ea74dca4`: remove `CUDA_SEPARABLE_COMPILATION ON` and `CUDA_RESOLVE_DEVICE_SYMBOLS ON` from `ninfer_cuda_archive` in `cmake/NinferTargets.cmake`
  (upstream measured stack-frame kernels 1155 → 927).
- `77c9cc39`: fold the attention output gate into the causal reduce epilogue (one fewer launch per layer, bit-identical).
- `a941c9f4`: read Q/K/V scale from shared memory for INT8-family attention, removing the computed-lane shuffle.
- Verification: Op tests + whole-model greedy byte-identity + decode timing.

**4. Weight bytes**
- `1da31697` (transcode W8 vocabulary/output head at load time, −644 MiB each on 27B), `f1982279` (`--embedding-q6`),
  `f8c75edd` (`--mtp-experts-q4`). §1 already named about 1/4 of the per-token traffic as 1 byte/element, and repacking saves
  ~2.6 GB/shard/token.
- Verification: artifact contract tests + perplexity A/B + decode tok/s.
- **Applicability re-check (2026-09-26, reading `<models-dir>/qwen3_8_27b_swift15_dflash2_final.ninfer.conversion.json` and
  `<models-dir>/w4a4_family_recipe.py`)**: upstream `1da31697`'s object is “W8G32 token embedding / output head”,
  whereas this artifact's recipe assigns both `text/token_embedding` and `text/output_head` as **`fp8_e4m3fn_row_bf16`**
  (1 B/element, the head with `AllowA8`), and the remaining matrices are NVFP4 (128) or FP8 (130). ⇒ That upstream “W8→Q4 load-time
  transcoding” **does not hold** for this artifact; the equivalent action would be moving these two FP8 matrices (plus the FP8 attention/GDN projections and
  the last eight FFN layers that must be streamed every token) to NVFP4/Q4, which is a **recipe/conversion-layer** change with a quality cost.
- Magnitude: `output_head` = 248320×5120 × 1 B ≈ 1.27 GB total, halved per shard ≈ 0.64 GB/shard/forward, ~6% of 10.15 GB;
  changing to NVFP4 saves about 3% of the round ⇒ decode about +3% (embedding is only capacity, the gather does not stream the whole table). The quality cost and the
  acceptance-rate cost require a separate A/B per the repository rules, so this item should be measured first before deciding whether to change the recipe.

**5. verify segment**
- `ce2df46b` small-T tensor-core kernels (MTP verify / cohort, 1–32 columns); `21df3069` GDN record window
  `cp.async` staging. verify forward ≈30 ms / 38.2 ms round.
- Verification: Op oracle + verify graph timing + greedy byte-identity.

**Implementation result (2026-09-26)**

- Attribution (§5.4): of the 33.9 ms round, verify 29.1 (86%), proposal 3.7 (11%); verify's per-shard weight stream lower bound
  is 22.7 ms ⇒ fixed headroom 3.9 ms (T=1 measured 26.6 ms), and 2 more columns add only 2.5 ms (T=3 measured 29.1 ms).
- **5b `21df3069` GDN record window staging: ported and verified** (`recurrent.cuh`, +92 lines; uses `cp.async` to prefetch the
  window's key/query/this CTA's value rows/gate into shared memory, then performs the same operations in place).
  - Op level (`ninfer_gdn_replay_bench --profile 27b --component recurrent`, B=1): T=3 cold L2
    22.1 → 16.2 µs (−27%), warm 16.1 → 16.3 µs (within noise); T=8 cold L2 36.5 → 28.3 µs (−22%),
    warm 28.5 → 27.5 µs (−3%).
  - End-to-end: the MTP route shows **no measurable difference** — verify phase of the same workload 29.07 → 29.13 ms (0.2%); decode tok/s
    going from 70.4 → 76.4 is acceptance-rate drift (committed/round 2.39 → 2.58, in the same proportion as tok/s), not speed. The DFlash2
    route, converted from T=8's ~8 µs/layer × 48, is only about 0.4 ms, below the phase-timing resolution. So this item claims only an **Op-level** conclusion.
  - Correctness: `gated_delta_net_test` (op oracle), `gated_delta_net_replay_record_test`,
    `gdn_replay_fold_test`, `gdn_replay_records_test`, `gdn_input_proj_conv_record_test` all green;
    the real dual-card artifact `tp2_dflash_append_test` (including the proposal/accept summary) PASS ⇒ bit-identical.
- **5a `ce2df46b` small-T tensor-core kernels: after reviewing the scope, judged not applicable to the TP-2 mainline**. That commit lands only on the **q4/q5
  route** (`linear/q4/q4_small_t_mma`, `linear/q5/q5_small_t_mma`, `linear_add/q5`, `linear_swiglu/q4`,
  `linear_topk/q4`, `attn_input_proj/q4_q5`, `gdn_input_proj/q4_q5`, about 25 files / 1500+ lines including new tests),
  whereas the body of the dual-card `swift15` artifact is NVFP4 (128 matrices) + FP8 (130), with q4/q5 covering only the draft weights ⇒ at most it touches part of the 3.7 ms
  proposal segment; and the “small-T inefficiency” at 1–32 columns manifests on this machine as verify's 2.5 ms column-extension cost (7% of the round). ⇒ Not done.
- Conclusion and ordering suggestion: the only thing with double-digit headroom is **weight bytes** (22.7 / 33.9 ms = 67% of the stream), i.e. the generalized form of item 4
  (moving the FP8 tensors that must be streamed every token to NVFP4/Q4; the document estimates −2.6 GB/shard ≈ −5.8 ms ≈ 17%), at the cost of quality/acceptance rate,
  and a perplexity A/B must be done first before deciding whether to change the recipe.

**6. Big items (need product decisions, not changed for now)**
- KV codec: `eb9f7a23` rk4v4-e8 (280 B/token/head), `ad26b362` rk2v4-e8 (216 B), `2ba10da2` rk8v4 G32,
  `9218b67b` root-code table lookup; the current fp8 KV = 16.125 KiB/token/shard.
- Device route profile: `0b2f8384` + `81861a07` (the 5090 profile would enable FP16 P·V and the fast prompt kernel; a 36 SM
  card needs recalibration).
- fast prompt kernel: `4303e604` (`--fast-prefill-kernel`) + `NINFER_PREFILL_ALIGN` whole-wave alignment.

### 5.3 Explicitly not done

- MoE five-piece set (`5fd7c463`/`b2ba37d2`/`d63162c8`/`ba1284d4`/`f1806721`): we run the dense 27B.
- T2 ternary, INT8 activation route (`--prefill-a8`/`--mlp-a8-decode`), W4A8: decode is bandwidth-bound, these are compute routes.
- The whole sm_86/3090 tuning set; `82631f68`'s crossing byte pipelining (< 256 KiB skipped, each TP-2 allreduce is only ~20 KB).
- Structured output / vision / WebUI / disk KV / DirectStorage / D3D12 residency (feature items, not this round's perf).
- Weight bytes (the generalized form of 4 in this section: moving the FP8 tensors that must be streamed per token to NVFP4/Q4): **the user decided on 2026-09-26 to drop it for now**,
  because it requires recipe/conversion-layer changes + quality and acceptance-rate costs, and the benefit (documented estimate ~17%) is not enough to go first.
- NCCL (`D:\nccl-windows` on this machine = a NCCL 2.29.7 build from SystemPanic/nccl-windows, containing
  `build/bin/nccl.dll`, `nccl_static.lib`, `install/`): **not integrated**. Without P2P, NCCL uses the SHM transport
  (the GPU writes shared host memory over PCIe and the peer reads it back), isomorphic to NInfer's in-kernel mapped-pinned mechanism; and
  the copy-engine ceiling probe in `tests/test_tp_device_pair.cpp:425-429` has confirmed that the sliced in-kernel path is within 2% of the link's raw
  lower bound. NCCL's only structural advantage over this path is P2P/NVLink, which is unavailable on this machine (it can be enabled only with a patched BAR1, and external measurements
  value it at only +2.67% / −5~7%), and each collective would need a separate launch (128/token, launches alone ~0.2 ms), plus it would introduce a 53 MB DLL
  and a hard dependency. Keep it only as an **external reference meter** (whether the 20 KB allreduce landing is also within 9–18 µs).

### 5.4 TP-2 baseline (2026-09-26, this machine's 2×5060 Ti, dual-card artifact swift15)

- Configuration: `tools/win_port/serve.ps1 -Model <models-dir>/qwen3_8_27b_swift15_dflash2_final.ninfer` (default recipe:
  `--devices 0,1 --kv-dtype fp8 --max-context 131072 --spec mtp --draft-tokens 2 --lm-head-draft --vision
  --reasoning-effort medium`, sampling 0.7/20/0.80, `--host-kv-mib 32768`).
- Startup: `engine ready | qwen3.8-27b-swift1.5 | total 45.6s | weights 9.66 GiB`; per shard
  `weights+ctx 11050.6 / 10900.6 MiB`, KV 131,072 (fp8) taking 2322 / 2064 MiB, free 2446 / 2026 MiB.
- `tools/win_port/bench_serve.ps1` (one-off, non-streaming, numbers taken from server-side timings):

| workload | prompt | prefill | decode |
|---|---|---|---|
| prefill_2048 | 1,832 tok | 1,206 tok/s | — |
| prefill_8192 | 7,153 tok | 1,654 tok/s | — |
| prefill_32768 | 28,528 tok | 1,603 tok/s | — |
| decode_short | 23 tok | — | **75.2 tok/s**（119 tok） |
| decode_at_8192 | 7,154 tok | — | **71.2 tok/s**（128 tok） |

- Purpose: as the A/B baseline for items 4/5. Note: decode goes through MTP K=2 + `--lm-head-draft`, the acceptance rate drifts per round with sampling,
  so a single result can only judge differences at the ≥5% level; deciding on smaller changes requires repetition (the worklog's acceptance-rate adjudication uses 30-rep).
- Artifact format (`*.conversion.json` + `w4a4_family_recipe.py`): both `text/token_embedding` and `text/output_head`
  are `fp8_e4m3fn_row_bf16`; 128 NVFP4, 130 FP8, 567 BF16, 88 Q4G64, 54 Q5G64,
  1 Q6G64, 7 Q8G32.

**verify segment attribution (2026-09-26, server-side stage timings from `NINFER_TP2_TIMING=1`)**

| segment | per round (ms) | share |
|---|---:|---:|
| avg_round | 33.92 | 100% |
| **verify** | **29.07 / 29.90** | **~86%** |
| mtp (proposal) | 3.69 / 3.77 | ~11% |
| accept + copy + sync_wait + fold | 0.37 | ~1% |

(two independent samples: `rounds=36 committed=86` and `rounds=24 committed=63`, i.e. 2.39 token/round ⇒ 70.5 tok/s,
consistent with the §5.4 bench_serve numbers.)

- Conclusion: item 5's direction is right — verify is the only segment worth touching, proposal has only 11% left.
- But the ceiling must be computed clearly: verify's per-shard weight stream is 10.15 GB ⇒ 22.7 ms bandwidth lower bound, measured 29 ms ⇒ **the non-bandwidth headroom
  is only ~6.3 ms (18% of the round)**. The small-T tensor-core kernels and GDN staging can only claim part of that 6.3 ms.
- The next step is to distinguish how much of that 6.3 ms is “small-T GEMM efficiency / 128 collectives / in-graph kernel gaps”,
  and then decide whether 5a (`ce2df46b`'s new kernel family) or 5b (`21df3069` GDN staging) goes first.

**Measured upper bound on AR payload bytes (2026-09-26, diagnostic switch `NINFER_TP2_AR_PAYLOAD_DIVISOR`)**

New diagnostic (`device_pair.{h,cu}`, default 1 = bit-identical, `ninfer_tp_device_pair_test` still PASS): make each
in-kernel collective exchange only 1/N of the partials, leaving the remaining ranges at the destination buffer's original value (in-place allreduce = this card's partial).
The numbers are necessarily wrong; the point is only to set an upper bound for “compressing the AR payload”. Same workload (prefill 8192, `bench_serve`):

| divisor | prefill | block time/1024 tok | decode verify | conclusion |
|---:|---:|---:|---:|---|
| 1 | 1,533.9 tok/s | 668 ms | 29.5 / 30.6 ms | baseline |
| 2 | **2,025.7** (+32%) | 506 ms | within noise | |
| 4 | **2,414.3** (+57%) | 424 ms | 28.1 / 28.9 ms | |

- The two differences give the same solution: **AR ≈ 324 ms, compute ≈ 344 ms** (÷1→÷2 saves 162 ms = AR/2; ÷2→÷4 saves another 81 ms =
  AR/4). That is, AR accounts for **48%** of the prefill block time, consistent with the old cost model.
- ⇒ **The ceiling of any “AR bytes → 0” = 344 ms/block ≈ 2.98k tok/s (+94%)**; ÷2 gets 1/3 of it, ÷4 gets about 60%.
- **No decode gain**: ÷4 only lowers verify from 29.5 to 28.1 ms (≈4%, and within round noise) — small payloads are latency-bound,
  bandwidth is only ~3 µs of each 9–18 µs.
- Comparison with hardware: moving card 2 from Gen4 x4 to a direct x8/x16 slot = link ×2~3.5 = **a free ÷2~÷3.5**, no quality cost,
  zero code ⇒ higher priority than compression; the two do not stack (they share the 344 ms compute floor).
- Criterion for “dictionary-mapping compression”: a lossless dictionary cannot achieve 2× on bf16 activations; reaching ÷2/÷4 requires lossy quantization (FP8 block scale /
  INT4 codebook), so it falls under the S5 “payload quantization” entry of PLAN §5.3 and requires rebuilding the oracle plus a quality/acceptance-rate A/B.

---

## 6. FP8 PV: attention compute path efficiency (started 2026-09-27 → reverted the same day, CLOSED)

> **Reverted (2026-09-27)**: the implementation passed all quality gates (G0–G3) but reached only 1.179× at the op level (target ≥1.3×), so per the user's pre-authorized
> rule it was reverted and the fp8 criterion 1.2e-2 restored; the patch sits alongside it at the repository root: `PLAN-fp8pv-v1b.patch` (`git apply`
> restores the whole implementation). **The most valuable output of this round is the bottleneck localization**:
> barrier stall 34.2% + math pipe throttle 25.2% ⇒ the headroom is in the **phase structure** rather than the PV dtype (see the end of §6).
> The suggested follow-up “cross-tile phase pipelining” (bit-exact, zero quality cost) was not chartered.

**Goal**: Switch production prefill PV from the FP16 tensor path to FP8, and compress the non-tensor FP32 passes. Scope is deep-context prefill (at 245k, attention accounts for 70% of per-token cost, §3.13 fit). Expected attention term **1.2–1.4×** ⇒ 245k end-to-end **~1.1–1.3×**, 50k ~1.05–1.1×. Decode small-T kernels are irrelevant to this lever.

### 6.1 Basis (settled, do not re-investigate)

- L1 measurement gate (worklog §3.13/§3.14 + L1 experiment log): KV bytes span 3.5× while time differs by only 13% ⇒ **compute/ALU bound**, not KV-bandwidth bound; "cut 6× KV redundancy" is therefore refuted (DRAM 1.09%).
- ncu (`causal_attention_prompt_k8v4_kernel`, W=1024, 65,536 depth, h12-kv2): **tensor pipe 63.5% (busiest resource)**, Compute 63.54%, DRAM 1.09%, occupancy 33.33% (1 block/SM; registers 100/thread + dynamic smem 85.12 KB, dual-limited), non-tensor FP32 instructions 11% (ncu estimates ~3.6% runtime).
- Tensor pipe budget (per KV tile per Q block): QK^T 2.10 MFLOP @FP8 (25% of the 85.9 TFLOP/s ceiling) + PV 2.10 MFLOP @FP16 (50% of ~43) ⇒ **PV takes 2× the pipe time of QK^T**; PV to FP8 ⇒ pipe work ×0.67.
- **Production dtype is `--kv-dtype fp8`** (recipe recommended in §1, shipped in `docs/tp2-dual-5060ti.md`), the production kernel is `prompt_fp8.cuh`; k8v4 is merely the dtype of the 245,760 sweep ⇒ **zero new rounding on the V side**: the FP8 MMA reads today's E4M3 codes directly, equivalent to today's "exact widening to FP16". This is the key simplification of this plan relative to the k8v4 viewpoint.
- Already tried and dead (do not redo): warps 16→32 (−12%, register spill; 1024 threads are pinned to ≤64 registers/thread by the 64K register file); Bc 64→128 (consumer Blackwell per-block dynamic smem cap ~100 KB, `cudaFuncSetAttribute` returns `cudaErrorInvalidValue`); W=2048 chunk (worth only ~9%, W=1024 is already in the plateau region); GQA cutting 6× KV traffic (DRAM 1%).

### 6.2 Design (variant V1a, do first)

1. **P: one E4M3 scale per (row, tile)**: `psc = P_tile_max/448 = exp2((bm − nm)·scale_l2)/448`. `bm` (tile max) and `nm` (new running max) are already being computed by the producer ⇒ **scale is free**; tile max maps to 448, range fully utilized. `p_s` changes from FP16 to E4M3 (8 KB → 4 KB).
2. **Accumulator absorbs the scale ratio**: `alpha' = alpha · psc_prev/psc_cur` (once per row, computed by the producer, broadcast with `alpha_s`; new row array `psc_s[Br]`); final round `out = acc/(l · psc_last)` (one multiply per row, free).
3. **V passthrough**: the 8 V-worker warps no longer do FP16 widening; instead they **transpose/swizzle-write the E4M3 codes into the B staging** — the FP8 B operand (k32×n8) requires a `[n=d][k=key]` row layout (`ldmatrix_x2` non-transposed has row dim = n), while the cache is `[key][d]`, so the worker side needs one 8×8 byte transpose (in-register shuffle + swizzled write).
4. **v_scale ownership (the only open choice)**:
   - **V1a (do first)**: the worker side requantizes `code × v_scale` to E4M3 (one extra 2^-4 relative rounding); the logic of P and the denominator `l` stays exactly as-is (`l` is still accumulated from unweighted FP32 P) ⇒ minimal change, no coupling.
   - **V1b (alternative)**: fold `log2(v_scale)` into the score (`score + log2v_s[col]`, multiply → add), so the tile max automatically includes v_scale; but the denominator `l` must be computed separately with unweighted exp2 ⇒ **MUFU doubled**; only consider if V1a quality is insufficient.
5. **Non-tensor pass compression** (same batch as 1–4): the V widening pass disappears entirely; P's `__float2half_rn` becomes an E4M3 conversion; the production kernel's smem drops from 92,416 B to **~72 KB** (VStage 32 KB removed, P halved, adding V's B staging and the psc row).

### 6.3 Quality risk (the main kill point for this lever)

E4M3 has only ~2^18 dynamic range (2^-9 … 448), while the dynamic range of `P = exp2(score − m)` is distribution-determined. After the per-tile scale maps each tile's max to 448, **small P within a tile can still fall into subnormal or 0**: with a 245k context, if attention is near-flat (P ~ 4e-6), an entire row's P may be crushed to zero ⇒ normalization distorted. Unlike the L1 "cut traffic" lever, **this lever is not numerically transparent**: it necessarily alters the greedy trajectory (precedent: when MTP cache tried nvfp4, 3 of 5 goldens changed text) ⇒ it is "quality for speed", and must be adjudicated with a declared tolerance and sampled acceptance, not accepted as "bit-identical".

### 6.4 Gates and execution order (measure before touching code; if G1 fails, stop and record the rejection)

| Gate | Content | Criterion |
|---|---|---|
| **G0** | Re-capture ncu on the production fp8 kernel (d256-h12-kv2, W=1024, depth ≥100k) + fp8 baseline depth curve (8k→245k) | tensor pipe still the busiest resource; occupancy/DRAM same shape as the k8v4 profile |
| **G1** | Numerical study of P quantization error: PV relative error under per-tile E4M3 scale for real/synthetic score distributions, covering both ends (flat/spiky) at 245k | tail loss within acceptable range (tentatively PV output relative error on the same order as fp8 KV) |
| **G2** | Op-level independent FP32/FP64 naive oracle @ production geometry deep envelope; declare the P quantization boundary and tolerance (V1a also records the second V quantization) | production route compared directly against the oracle |
| **G3** | Model-level golden digests (`tp2_dflash_solo`/`append`) + sessions + 30-rep acceptance rate | acceptance rate has no measurable cost, or the cost is accepted by the user |
| **G4** | Depth curve A/B + `bench_serve.ps1` prefill row | attention term delivers 1.2–1.4× |

**Files**: `src/ops/softmax_attention/dense/causal_cache/prompt_fp8.{cuh,cu}`, `tests/ops/softmax_attention/causal_cache.cpp`, `bench/ops/causal_softmax_attention_bench.cu`. The k8v4 / nvfp4 / bf16 / i8 prompt kernels keep FP16 PV untouched (k8v4 is only a sweep dtype).

**G0 result (2026-09-27, PASS)**

- fp8 baseline depth curve (`ninfer_causal_softmax_attention_bench --entry cached --geometry d256-h12-kv2 --kv-dtype fp8 --tokens 1024 --execution eager --cache cold`, single-layer median): 8,192 → 2,645.3 µs; 32,768 → 9,949.4; 65,536 → 19,690.7; 131,072 → 39,673.7; 245,760 → **73,521.8 µs** (math 42.15 TFLOP/s, qk/pv 21.07 TFLOP/s each). Linearity excellent, same tier as the k8v4 baseline (73,880 µs), slightly faster.
- ncu re-capture of `causal_attention_prompt_fp8_kernel` (d256-h12-kv2, W=1024, 65,536 depth, grid 192, block 512): **Tensor pipe 63.7% is the busiest resource**, Compute (SM) 63.67%, **DRAM 2.23%**, L1 39.79%, L2 14.67%, IPC 1.26, Issue Slots 28.09%; **registers 102/thread + dynamic smem 92.42 KB (configured 102.40) ⇒ 1 block/SM**; largest stall = waiting on the math pipe 4.4/12.7 cycles (**34.9%**, ncu's Est. Local Speedup 34.9%, original text "all active warps execute their next instruction on a specific, oversubscribed math pipeline").
- ⇒ **Same shape as the k8v4 profile**: the bottleneck is the tensor pipe + 1 block/SM, not bandwidth. The premise holds, the lever is live.

**G1 result (2026-09-27, PASS + design re-decision)**

Simulation (per-tile E4M3 scale, V row scale of fp8 KV, D=32, Bc=64, 4 n×σ×spike combinations; error = output rel-L2):

| Scenario (P distribution) | V requantization alone | E4M3 per-tile | E5M2 per-tile | E4M3 global 1/448 | P mass zeroed |
|---|---:|---:|---:|---:|---:|
| n=8192 σ=1 no spike | 2.610% | 2.967% | 6.073% | 4.813% | 0% |
| n=65536 σ=1 with spike | 1.778% | **2.415%** | 4.194% | 3.059% | 0% |
| n=245760 σ=1 no spike | 2.760% | **2.954%** | 5.965% | 3.805% | 0% |
| n=245760 σ=4 with spike | 2.552% | **2.552%** | 2.551% | 2.650% | 0% (global 0.358%) |

- **Conclusion 1 (per-tile scale is required)**: global 1/448 is clearly worse at long context (+0.9pp) and starts losing mass (0.358% zeroed); per-tile E4M3's zeroed mass ≈ 0%, **the "long-context tail collapse" I originally feared does not hold** (each 64-key tile carries its own scale, so the intra-tile dynamic range is bounded).
- **Conclusion 2 (E4M3 > E5M2)**: E5M2's 2-bit mantissa (12.5% relative error) outweighs its exponent-range advantage, ~2× worse at long context ⇒ use E4M3.
- **Conclusion 3 (the cost of P quantization itself is small)**: E4M3 per-tile adds only ~0–0.6pp relative to "V requantization alone"; **the real cost is on the V side (1.8–3.4%)**.
- ⇒ **The design is re-decided from V1a to V1b (V stays bit-exact)**: fold `log2(v_scale)` into the weighted score. After re-derivation V1b turns out to be **cheaper**: only 1 extra add per element (`u+w`) + 1 fmax, and the denominator `l` is recovered with `Σ P'_q·inv_scale` (1 multiply per element) under **the same shift**, requiring no second exp2 (MUFU does not double — measured SFU occupancy is only ~3%, doubling would be harmless anyway).
- **Algebraic simplification (important)**: let `u=score·scale_l2`, `w=log2(v_scale)`, `P'=exp2(u+w−m)`, `psc=exp2(bm'−nm')/448`; then the accumulator and the denominator share the same rescale **`alpha' = exp2(bm'_prev − bm'_cur)`** (it depends only on the weighted maxima of adjacent tiles; the running-max terms cancel entirely) ⇒ final round `out = acc / l`, **no extra psc wrap-up multiply needed**.

**Final design (V1b, being implemented)**

1. worker 8 warps: no more FP16 widening; **transpose the staged V E4M3 codes into the B staging laid out as `[d][key]`** (the FP8 B operand requires row dim = n = d and 16 bytes within a row = 16 keys; `ldmatrix_x2` non-transposed, k step 32 ⇒ `PVKs = Bc/32 = 2`), and compute `log2(v_scale)` and `inv_scale` per key (64 each, negligible spread over 256 threads).
2. producer: in the score loop compute the tile max `bm'` of `u+w` (1 add + 1 fmax/element); after `P'=exp2(u+w−nm')` quantize to E4M3 by `psc` and store into `p_s` (Br×Bc×1 B); accumulate `l` with `P'·inv_scale`.
3. consumer: after `acc *= alpha'` (computed per row by the producer and broadcast, reusing `alpha_s`), run the FP8 PV mma (both P and V codes are E4M3).
4. smem: 92,416 → **~72 KB** (drop VStage 32 KB, P halved 4 KB, add the transposed V staging 16 KB and the small log2/inv/psc arrays).

**Implementation result (2026-09-27, scheme V1b landed; uncommitted)**

- `src/ops/softmax_attention/dense/causal_cache/prompt_fp8.cuh` (+253/−94, the only substantive change):
  - smem **92,416 → 72,448 B**: drop `v_f16` (32 KB widening staging), `p_s` changed from `__half` to **E4M3 uint8** (8→4 KB), add `v_t` (the transposed `[d][key]` V codes, 16 KB; the raw `v_fp8` staging is kept), `lv_s/iv_s[Bc]`, `bmp_s[Br]`, drop `running_m_s`; still 1 block/SM (registers 102→96).
  - producer: `bm'=max(u+w)` (1 fma + 1 fmax/element, reusing `partial_m_s` to merge the two halves); `P=exp2(fma(score,scale_l2,lv−bm'))·448` (tile max maps exactly to 448); `code=__nv_cvt_float_to_fp8(P,SATFINITE,E4M3)`; `l += P·(1/v_scale)` (same exp2 as the code, no second one); `alpha'=exp2(bm'_prev−bm'_cur)`; P's swizzle changed to 16-byte granularity (row 64 bytes).
  - worker 8 warps: no more widening; instead a **byte-wise 8×8 register transpose** (three-stage 64-bit masked exchange) moves the V codes into `v_t[d][key]`, placed at `((key_group>>1)^(j&3))<<4 | ((key_group&1)<<3)`.
  - consumer 16 warps: `PVKs=Bc/32=2`; A=P via `ldmatrix_x4`, B=V via **non-transposed** `ldmatrix_x2` ⇒ `mma_fp8_e4m3`.
  - `prompt_fp8.cu` unchanged (the smem constant is in the header, launch syncs automatically).
- **Necessary corrections during implementation** (all written into code comments):
  (a) the denominator must carry the same 448 factor (the mma consumes `448·P` ⇒ `l += (448·P)·iv`; 448 cancels in acc/l, so the final round is still `out=acc/l`);
  (b) use the per-tile `bm'` directly as the shift (do not introduce a running max): storing codes is equivalent and saves one multiply per element;
  (c) for `v_scale<=0` take `lv=0,iv=1`: the softmax weight of an all-zero V row must still enter the denominator, so it must not take `-inf/0`;
  (d) **an empty tile (whole block masked, `bm'=-inf`) must use `alpha'=1` and must not update `bmp_s`** — a real bug measured: before the fix, T=65/keys=128 and dflash W=9/16 produced non-finite; after the fix, 0 non-finite.

**G2 (op-level oracle): PASS, but the criterion must be relaxed.** `ninfer_softmax_attention_test` all PASS (0 fail / 0 non-finite). P→E4M3 is the only new semantic boundary; measured worst **rel-L2 2.309e-2 / gross-abs 2.258e-2** (max_ref 1.0), 1.92× over the original 1.2e-2 threshold ⇒ the fp8 criterion changes to **3.2e-2 / 9.0e-3 / 2.4e-2** (~1.4× margin, below the theoretical upper bound of a single E4M3 weight 2^-4=6.25e-2), and a comment in `tests/ops/softmax_attention/causal_cache.cpp` records the boundary and the measured numbers.
**⇒ This is about 2× the output perturbation of the old fp8 route, a cost G3 must adjudicate.**

**G4 (performance): not met.** `ninfer_causal_softmax_attention_bench --entry cached --geometry d256-h12-kv2 --kv-dtype fp8 --tokens 1024 --execution eager --cache cold`:

| Depth | before µs | after µs | speedup | math before→after (TFLOP/s) |
|---|---:|---:|---:|---:|
| 8,192 | 2,645.3 | 2,252.4 | 1.174× | 41.40 → 48.63 |
| 32,768 | 9,949.4 | 8,449.2 | 1.178× | 42.09 → 49.56 |
| 65,536 | 19,690.7 | 16,730.0 | 1.177× | 42.20 → 49.68 |
| 131,072 | 39,673.7 | 33,359.0 | 1.189× | 41.73 → 49.63 |
| 245,760 | 73,521.8 | 62,493.5 | 1.176× | 42.15 → 49.59 |

ncu (same G0 config): Tensor pipe **63.7% → 50.1%**, registers 102→96, dynamic smem 92.42→72.45 KB, DRAM 2.23→2.68%, L1 39.79→45.65%, still 1 block/SM. ⇒ PV tensor time indeed dropped 1.5× (absolute 0.637→0.425), but wall clock is only 1.18× faster: the kernel has gone from "tensor-exclusive" to "tensor and everything else at half each", **and the remaining 2–3% gap is on the non-tensor side** — that is, the other half, §6.2 item 5 "compress the non-tensor FP32 passes", which has not been done yet.

**G3 (model-level adjudication): PASS, but coverage has a gap.** 30-rep acceptance rate A/B (`tools/tp_bootstrap/r62_sampling_ab.ps1`, artifact `qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer`, dflash2 k=7, temperature 0.7/top_k 20/top_p 0.8, 3 kinds of short prompt × 30 rep = 90 requests/arm, same prompt set in both arms):

| Arm | Samples | Acceptance | committed/round | drafts/round | Arm wall clock |
|---|---:|---:|---:|---:|---:|
| change (FP8 PV) | 90 | **35.79%** | 3.4265 | 6.779 | 355.1 s |
| base (FP16 PV) | 90 | **35.50%** | 3.4083 | 6.783 | 366.8 s |

Δ=**+0.29pp** (well within the 30-rep ±1.4pp / 15-rep ±2pp noise, and the direction favors the change) ⇒ **no measurable model-level degradation**.
golden: `tp2_dflash_solo` (q4all artifact) **PASS** (digest `0x19047f8ccaf5707f`, 43.3 s); `tp2_sessions` dflash2 official r69 artifact **PASS** (164.3 s); `tp2_dflash_append` PASS but **its KV is BF16 ⇒ it does not go through prompt_fp8, regression only**; `sessions` on the swift15 artifact FAILs but **the baseline FAILs too** (reproduced by rerunning after stash, failing earlier) ⇒ existing artifact mismatch, not a break by the new kernel.

**Gap (must remember)**: the acceptance-rate arm uses ~50-token short prompts and is decode-dominated; this lever's scope is **deep-context prefill**, and "P quantization in deep context has no quality cost" is **not tested** by this gate (G2's op-level upper bound rel-L2 ≤2.309e-2 covers the deep envelope, but only at op level).

**G3 deep-context supplementary test (2026-09-27): no degradation within scope either.** Long-prompt variant `build-win/g3_long_ab.ps1` (gitignored, reusing r62's serve recipe and aggregation convention, only replacing the 3 short prompts with 1 long filler, **each rep carries a new nonce** to defeat prefix cache ⇒ full prefill every time), artifact `qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer`, dflash2 k=7, actual prompt_n ≈ **28.5k** (single-request prefill ≈18.5–18.9 s ⇒ ≈1,550 tok/s):

| Arm | prompt_n | Samples | Acceptance | committed/round | drafts/round |
|---|---:|---:|---:|---:|---:|
| change 10 | 28,537 | 10 | 39.96% | 3.7317 | 6.84 |
| base 10 | 28,538 | 10 | 38.57% | 3.6434 | 6.85 |
| change 30 | 28,533 | 30 | **38.09%** | 3.6025 | 6.83 |
| base 30 | 28,534 | 30 | **38.97%** | 3.6717 | 6.86 |

Δ(10)=+1.39pp, Δ(30)=**−0.88pp** (the sign flips from 10→30 ⇒ noise; |Δ|<1.4pp threshold); no non-finite, all four arms exit 0 ⇒ **in the lever's true scope (deep-context prefill) there is likewise no measurable quality cost**.

**G4 end-to-end (same serving session, 1 sample per cell, below the single-sample decision threshold)**:

| workload | change | base | Δ |
|---|---:|---:|---:|
| prefill_2048 (1,835 tok) | 1,288.0 tok/s | 1,299.9 | −0.9% |
| prefill_8192 (7,160 tok) | 1,560.4 | 1,594.0 | −2.1% |
| prefill_32768 (28,534 tok) | **1,550.5** | **1,529.4** | **+1.4%** |

PLAN §5.4 itself states that a single `bench_serve` run can only resolve ≥5% ⇒ the first two cells are within noise; the deep-context +1.4% is in the same direction as the op-level 1.176×, but **below the decision threshold, so a formal G4 verdict still needs multiple reps**. (The decode row drifts with the sampled text and is unrelated to the change.)

**Overall ledger (2026-09-27 wrap-up)**: on the quality side all green — short-prompt arm Δ=+0.29pp, deep-context arm Δ=−0.88pp, both within noise, three goldens PASS on valid artifacts; on the performance side op-level **1.176×**, end-to-end 28.5k **+1.4%** (single sample). **Cost**: the fp8 route's op criterion 1.2e-2 → 3.2e-2 (~2× output perturbation), which is a **substantive relaxation of an existing quality gate**, while the end-to-end gain bought is currently only on the order of ~1.4% ⇒ whether to keep it depends on whether the follow-up non-tensor-side compression can push op-level above ~1.3×.

**User decision (2026-09-27): choose A — keep the change, do one round of "bounded non-tensor-side compression"**, targeting op-level **≥1.3×** (only then is end-to-end ~+3%, making the trade defensible); **if it falls short, revert and restore the 1.2e-2 threshold**.

Hard constraints for this round (otherwise the just-passed G3 evidence is void): **only bit-exact optimizations are allowed** — do not reassociate floats, do not change the semantics of fma/sub/exp2/cvt/mask; the bit-exact criteria are ① `ninfer_qwen3_5_tp2_dflash_solo_test` (q4all artifact) digest still `0x19047f8ccaf5707f`, ② the op unit test's worst error numbers (rel-L2 2.309e-2 etc.) unchanged verbatim. Occupancy is already hardware-locked (96 registers × 512 threads, 72.45 KB smem ⇒ 1 block/SM; warps 16→32 and Bc 64→128 both measured and rejected) ⇒ do not touch structural refactoring.

Starting point of this round: op-level 1.176× (245,760 depth = 62,491.4 µs), ncu Tensor 50.1% / L1/TEX 45.65% / DRAM 2.68%. To do: first get the pipe breakdown and stall detail from ncu to pin the bottleneck; the candidates are combining P's 16 one-byte scattered writes into 8 16-bit writes (adjacent bytes in the same column, bit-exact), the address budget, and the worker transpose only if ncu points at it (loading V directly from global to save the raw `v_fp8` 16 KB and one smem round trip; high risk).

**Bounded compression round result (2026-09-27): fell short of 1.3×, reverted per the user's pre-authorized rule.**

ncu breakdown (`causal_attention_prompt_fp8_kernel`, ctx=65536, reg 96, dyn smem 72.45 KB): on the pipe-elapsed basis Tensor(FP) **50.14%**, ALU 12.52%, FMA 7.59%; on the instruction-rate basis **LSU 37.39%**, Tensor 24.17%, ALU 12.27%. Stalls (cyc/issued inst, total 12.38): **Barrier 4.23 (34.2%)**, **Math Pipe Throttle 3.12 (25.2%)**, Wait 1.47 (11.9%), MIO 0.76, Short Scoreboard 0.69, Long Scoreboard 0.19.
⇒ **The bottleneck is the phase structure of the tensor pipe, not non-tensor instruction throughput.**

- This round did only the single bit-exact optimization (P codes: 16 one-byte scattered writes → 8 16-bit writes, swizzle address computation 16→8): the solo digest is still `0x19047f8ccaf5707f` and the op unit test quality numbers are unchanged verbatim ⇒ bit-exactness holds;
- but it has **zero wall-clock benefit**: `sm__inst_executed` −2.58%, LSU down correspondingly, 245,760 depth 62,481.0 → 62,358.7 µs (1.002×; still **1.179×** relative to the original FP16 PV, 9.3% short of 1.3×) ⇒ instruction throughput is hidden, it is not the limiter;
- candidate 2 (worker transpose → direct global load) and candidate 3 (merge redundant `lv_s/iv_s` reads) were refuted by ncu: Long Scoreboard 0.19, MIO 0.76, and per tile the producer is ≈250 inst/thread vs worker ≈100 (the worker reaches the barrier first and just waits) ⇒ speeding it up yields zero gain.

**Revert execution (2026-09-27)**: `prompt_fp8.cuh` and `tests/.../causal_cache.cpp` were `git checkout`ed back to HEAD, and the fp8 criterion was restored to **1.2e-2**; the combined patch is archived at the repository root as `PLAN-fp8pv-v1b.patch` (23,614 B: the complete FP8 PV implementation + P packing, touching the two files `prompt_fp8.cuh` and `causal_cache.cpp`; `git apply PLAN-fp8pv-v1b.patch` has been verified to apply cleanly). Bench re-measurement after the revert: 8,192 = 2,645.9 µs; 65,536 = 19,731.1; 245,760 = **73,646.7** (back to the FP16 PV baseline, same tier as the pre-change 73,521.8). The working tree has only the `PLAN.md` change left.

**Conclusion and recommended next step**: this round's rejection point is not "FP8 PV is not worth it" but that **trading a relaxed quality gate for 1.2% end-to-end was the wrong lever** — the real headroom is the **34.2% barrier stall**, i.e. the phase serialization of the tile loop (producer QK+softmax ∥ worker transpose → barrier → consumer PV → barrier; with 1 block/SM the producer phase has only 8 warps issuing mma). **Cross-tile phase pipelining** (double-buffer `p_s`, overlapping tile t's PV with tile t+1's QK) is **bit-exact and zero quality cost**, and it acts on the already-reverted FP16 PV baseline ⇒ strictly better than this round's FP8 PV trade. **Now opened (2026-09-27, user decision, see §7).**

---

## 7. Cross-tile phase pipelining: eating the barrier stall (opened 2026-09-27 → rejected the same day, CLOSED)

**Goal**: change the tile loop of `causal_attention_prompt_fp8_kernel` from phase-serial to cross-tile pipelined, overlapping tile t's PV with tile t+1's QK^T/softmax, directly eating the barrier stall located in §6. **The acceptance basis is the opposite of §6: this lever introduces no new numerical boundary, so it is accepted on bit-identity (solo digest + op unit test numbers unchanged verbatim), with no quality gate and no criterion relaxed.**

**Baseline (2026-09-27, HEAD `d21febf6`, FP8 PV reverted)**: `ninfer_causal_softmax_attention_bench --entry cached --geometry d256-h12-kv2 --kv-dtype fp8 --tokens 1024 --execution eager --cache cold` (warmup 5 / repeat 30, single-layer median): 8,192 → 2,643.6 µs; 32,768 → 9,949.2; 65,536 → **19,689.3**; 131,072 → 39,553.2; 245,760 → **73,517.7 µs** (math 42.15 TFLOP/s, qk/pv 21.08 each), same tier as the post-revert re-measurement in §6.

### 7.1 Basis

- §6's last ncu round (FP8 PV version, ctx=65,536): **Barrier 4.23 cyc/inst (34.2%) first**, Math Pipe Throttle 3.12 (25.2%), Wait 1.47, MIO 0.76, Short Scoreboard 0.69; pipe-elapsed basis Tensor(FP) 50.14%.
- Phase structure (`prompt_fp8.cuh:232-415`): each KV tile is `producer QK^T+softmax (8 warps) ∥ worker V widening (8 warps) → __syncthreads → all 16 warps PV → __syncthreads`; the producer phase has only 8 warps issuing mma, and the PV phase must wait for the producer to fully arrive; 1 block/SM (96 registers × 512 threads + dynamic smem) ⇒ there is no second block to fill the phase holes. The same file :379-413 shows tile t+1's KV `cp.async` prefetch is **already** overlapped onto tile t's PV ⇒ the remaining gap is the **serialization of the two compute phases, QK/softmax and PV**, not prefetch.
- **Key uncertainty**: §6's 34.2% is the FP8 PV version's number (it was only after the PV tensor time halved that barrier was pushed to first place). Whether the reverted FP16 PV baseline likewise has barrier as its primary stall **must be re-measured by G0** — the comparable k8v4 profile (worklog §3.13) had math pipe at 34.4% in first place at the time. This is exactly why the G0 gate exists.

### 7.2 Gates and execution order (measure before changing; if G0 fails, reject)

| Gate | Content | Criterion |
|---|---|---|
| **G0** | Re-capture ncu `--set full` on the current FP16-PV baseline (fp8 KV, d256-h12-kv2, ctx=65,536), requiring stall classification + pipe-elapsed breakdown | barrier-class stall first or second and ≥15% (phase serialization genuinely exposes appreciable time) ⇒ proceed to G1; otherwise **record a rejection, do not touch code** |
| **G1** | Bit-exact cross-tile phase pipelining (the concrete form determined by G0's stall attribution, constrained by §7.3) | no float reassociation, no change to fma/exp2/cvt/mask semantics or accumulation order; ncu re-measurement shows barrier markedly lower |
| **G2** | Bit-exact criteria | ① `ninfer_qwen3_5_tp2_dflash_solo_test` (`<models-dir>/qwen3_8_27b_w4a4_dflash2_q4all.ninfer`) digest bit-identical to the **pre-change HEAD build**; ② `ninfer_softmax_attention_test` all PASS with the worst error numbers unchanged verbatim (fp8 criterion stays 1.2e-2) |
| **G3** | Performance | depth curve 8k→245k no regression vs baseline, with a resolvable improvement at 245,760 |

**Files**: `src/ops/softmax_attention/dense/causal_cache/prompt_fp8.cuh` (the only substantive change; `prompt_fp8.cu` only if launch constants need syncing), `tests/ops/softmax_attention/causal_cache.cpp` (only if the criterion text needs changing). The k8v4/nvfp4/bf16/i8 prompt kernels are untouched this round.

### 7.3 Design constraints (already measured, do not redo)

- **Hard smem budget constraint**: current dynamic smem **92,416 B**, opt-in cap **101,376 B** (measured in G0; Bc=128's 151,808 B directly returns `cudaErrorInvalidValue`, worklog §3.13) ⇒ only **8,960 B** of incremental budget. Double-buffering `v_f16` (32 KB) or `k_fp8`/`v_fp8` (16 KB each) does not fit at all; the design must first shrink or replace these buffers rather than simply doubling them.
- **Registers/occupancy locked**: 102 registers × 512 threads + 92.4 KB smem ⇒ 1 block/SM (`regsPerSM = 65,536` ⇒ ≤128 registers/thread); warps 16→32 measured at −12% (spill).
- **Movable points**: `v_f16` (32 KB, V widening staging) is the largest single item, and §6 already argued that the worker phase's ~100 inst/thread reaches the barrier far earlier than the producer ⇒ it both consumes the whole budget and is off the critical path, making it the most likely source of headroom this round.

**G0 result (2026-09-27, PASS + key attribution)**

- Baseline rerun (HEAD `d21febf6`, ctx=65536): Tensor 63.6% (busiest), Compute 63.60%, DRAM 2.28%, L1 39.80%, registers 102/thread, dynamic smem 92.42 KB, 1 block/SM, Warp Cycles/Issued Inst **12.71**.
- Stall classification (cyc/issued-inst, same basis as §6): **Math Pipe Throttle 4.44 (34.9%)**, **Barrier 3.34 (26.3%)**, Wait 1.68 (13.2%), Selected 1.00, Short Scoreboard 0.62, Not Selected 0.60, MIO 0.52, Long Scoreboard 0.12, No Instruction 0.16, Branch Resolving 0.14. ⇒ **on the reverted FP16 PV baseline, barrier is still the second stall (26.3%)**, far above the 15% G0 threshold ⇒ phase pipelining is viable, proceed to G1.
- **PC-level attribution** (`ncu --page source --print-source sass`, 1,724,394 warp-stall samples, of which barrier 452,953):
  | Where the barrier stalls | SASS PC (in-function offset) | Samples | Share |
  |---|---|---:|---:|
  | after the tile-end `__syncthreads()` (0x5b00) | UMOV (0x5b40, loop back-edge) | 262,286 | 58% |
  | after the producer/worker → PV `__syncthreads()` (0x4bd0) | BRA.U (0x4bf0) | 177,073 | 39% |
  | after the producer's first internal `bar.sync 1,256` (0x3760) | LDS (0x3770) | 9,422 | 2.1% |
  | after the producer's second internal `bar.sync 1,256` (0x4740) | BRA (0x4750) | 3,933 | 0.9% |
  ⇒ **97% of the barrier stall is at the two block-wide `__syncthreads()` (phase serialization); the producer's internal named barriers account for only 3%**. Conclusion: what needs eating is this phase boundary, not softmax's two partial reductions.
- Note: ncu attributes a `.DEFER_BLOCKING` barrier wait to the first instruction after the barrier, so every PC in the table above is the successor of a `BAR.SYNC`.
- Device parameters (`cudaDeviceProp`, sm_120): `sharedMemPerBlockOptin = 101,376 B`, `sharedMemPerSM = 102,400`, `regsPerSM = 65,536` (512 threads ⇒ ≤128 registers/thread). Current dynamic smem 92,416 ⇒ **only 8,960 B of headroom**.

**G1 design (2026-09-27, decided per G0 attribution)**: split the tile loop into two **resident warp roles**, one `__syncthreads`/tile:
- **A (warps 0–7, producer)**: QK^T + softmax(t) → `p_s[t&1]`, `alpha_s[t&1]`, `running_m/l`; also prefetches tile t+1's K codes and K scale.
- **B (warps 8–15, consumer)**: first does **PV(t−1)** (reading `p_s[(t−1)&1]`, `alpha_s[(t−1)&1]`, `v_f16`), then **V widening(t)** (`v_fp8(t)`→`v_f16`), also prefetching tile t+1's V codes and V scale.
- Wrap-up: after the loop B does PV(K−1) and writes the output.
- **Overhead beyond zero smem increment**: `p_s` 8,192→16,384 (double buffer), `alpha_s` 256→512 ⇒ 92,416→**100,864** (512 B left); `k_fp8`/`v_fp8`/`v_f16`/`k_scale_s`/`v_scale_s` **stay single-buffered** — their prefetches are issued by the group that consumes them after it finishes reading the current tile, hiding latency across one phase via cp.async's asynchrony.
- **Bit-exactness argument**: each output element is still accumulated by a unique warp with the same set of operands (same P slice, same V slice) in the same order (`acc*=alpha` then the four k=0→3 mmas), the tile order and the `running_m/l` recurrence are entirely unchanged; the only change is that **the same work is executed by a different warp** (PV is covered by 8 warps × 16 n-tiles each instead of 16 warps × 8 each), and softmax's partial reduction is still done by the original 8 producer warps with the original `bar.sync 1,256` ⇒ no reassociation, no change in operation semantics.
- **Known risks**: ① B's `acc[16][4]`=64 accumulator registers (originally 32) may hit `__maxnreg__(120)`; ② the phase relationship between the 5 named barriers (A 3 + B 2) and the 1 `__syncthreads` must be exact, otherwise deadlock or `v_f16` corruption.

**G1/G3 result (2026-09-27): rejected. All three configurations slower than baseline, lever CLOSED.**

(G1's own criterion "barrier markedly lower" was met: 3.34 → 2.23 cyc/inst; **the rejection happened at G3** — all three implementations regressed to 0.71–0.80× at 65,536/245,760, with instruction count up, a structural cost rather than a tuning problem, so work stopped and the rejection was recorded.)

Implementation (reverted): `prompt_fp8.cuh` split into two resident roles + one `__syncthreads` per tile (`p_s`/`alpha_s` double-buffered, smem 92,416 → **100,864 B**, 512 B left). **Correctness is not an issue**: `ninfer_softmax_attention_test` all PASS (fp8 criterion 1.2e-2 unchanged, 0 fail / 0 non-finite). The rejection is purely for performance/structural reasons.

| Config | Structure | 65,536 µs | vs baseline | 245,760 µs | vs baseline |
|---|---|---:|---:|---:|---:|
| baseline (HEAD) | 8 producer ∥ 8 Vdeq → 16-warp PV | **19,689** | 1.00× | **73,518** | 1.00× |
| V1 | 8 producer + 8 consumer (PV 32→64 acc/thread, 512 threads) | 24,527~24,640 | **0.80×** | 92,160~92,219 | **0.80×** |
| V2 | 24 warps (8 producer + 16 consumer, 768 threads, `__maxnreg__(80)`) | 27,836 | **0.71×** | 104,193 | **0.71×** |

**V1's ncu attribution** (same G0 config, `--set full` + per-PC sampling):
- **barrier 3.34 → 2.23 cyc/inst (the target was indeed eaten)**; but math pipe throttle 4.44 → **5.24**, long scoreboard 0.12 → **0.77**, short scoreboard 0.62 → 0.88, wait 1.68 → 1.85 ⇒ total latency 12.71 → 13.31.
- **Executed instruction count +17.4%** (2,222.6 M → 2,608.9 M): of which `LDL` **0 → 40.4 M** (runtime local spill), IADD +54.8 M, LOP3 +37.3 M, IMAD +31.5 M, IMAD.SHL +29.7 M, BRA +34.8 M, S2R +23.8 M, `BSSY/NOP/BSYNC.RECONVERGENT` +16.2 M each (address recomputation and branch management). `LDSM.16.M88.4` (P operand) −6.34 M is the only substantive saving.

**Root cause (structural, reproducible)**: the whole-block PV accumulator = Br×D = 64×256 fp32 = **16,384 registers = 25% of the SM register file**.
- V1 squeezes PV into 8 warps (256 threads) ⇒ acc per thread 32 → **64**; the hard cap at 512 threads is 128 registers/thread (65,536/512). Measured with `__maxnreg__(128)`, ptxas uses all 128 and **still spills**: the prologue `STL`s ~20 loop-invariant addresses to the stack and `run_pv` `LDL`s them back on every call (static 48 STL/LDL; `REG:128 LOCAL:0` is a cuobjdump false report).
- V2 restores PV to 16 warps (32 acc/thread) at the cost of 768 threads ⇒ the per-thread cap becomes 65,536/768 = 85, and registers are allocated in 8/thread granules ⇒ `__maxnreg__(84)` directly returns `cudaErrorLaunchOutOfResources` (24×32×88 = 67,584 > 65,536); dropping to 80 (61,440) pushes the producer path (`score[4][4]` + softmax ≈90 registers) into heavier spilling ⇒ 0.71×.

⇒ **"Free up 8 warps to issue QK" has no solution on this machine**: either PV's acc doubles and hits the 128/thread wall, or adding threads pushes the cap to ≤80 and hits the producer working-set wall. The baseline "8 producer ∥ 8 Vdeq → 16-warp PV" is precisely the only viable packing under this machine's envelope of "register file + 100 KB smem + 1 block/SM". Moreover, the fact that total stall rises after barrier drops to 2.23 shows: **part of the phase serialization is the tensor pipe draining at the phase tail** (the rise in math throttle is another face of the same phenomenon), not reclaimable idling.

**Revert execution and G2 re-check (2026-09-27)**: `src/ops/softmax_attention/dense/causal_cache/prompt_fp8.cuh` has been `git checkout`ed back to HEAD (`git status` has only the plan file left); after rebuilding, bench re-measurement gives 65,536 = 19,728.6 µs, 245,760 = 74,132.0 µs (same tier as baseline, math 42.13 TFLOP/s), `ninfer_softmax_attention_test` all PASS (exit 0). **Digest criterion**: `tp2_dflash_solo` (q4all artifact) = **`0xad284a4b774b1cc3`**, consistent across two independent processes, PASS; the reverted source is byte-identical to HEAD, so this value is the baseline value (§6's recorded `0x19047f8ccaf5707f` is the **FP8 PV version's** value — §7.2's initial citation of it was a misremembering and has been changed to "bit-identical to the pre-change HEAD build"; the working-tree historical value `0x4bcc3994a5efba7d` predates the two frontend changes `6e01537a`/b7d352e8, and the three do not contradict each other).

The working tree has only the plan file left. **This section is CLOSED and will not be reopened** — unless the execution envelope itself changes (e.g. Br=32 lowering acc/thread to 16, or a >100 KB block-level smem budget appearing so that `v_f16` can also be double-buffered).
