# Anatomy of llama.cpp multi-GPU tensor parallelism (CUDA) — read-only analysis of a local clone of that fork

> Analysis target: a local clone of that fork (read-only, no writes/builds performed).
> Annotation convention: [fact] = code/documentation/git text I read, with file:line; [inference] = code-based reasoning, not verified by running.
> All line numbers refer to the current HEAD content of that file.

---

## 0. Executive summary (read this part first)

1. **This is not upstream llama.cpp.** It is a fork of zhuojun1024/llama.cpp, currently on the ar3-opt branch, and HEAD is a custom commit that specifically optimizes AllReduce for a *3-GPU no-P2P* topology (README.md:20-56).
2. **The ggml_cuda_op_mul_mat row-split path assumed in the task no longer exists.** Upstream deleted the CUDA row-split implementation on 2026-07-06 with commit 74976e1ae ('CUDA: remove -sm row, refactor cuBLAS', PR #24216); `--split-mode row` on CUDA now directly throws 'device %s does not support split buffers' (llama-model.cpp:1118-1119), and only SYCL still keeps split buffer.
3. **Current tensor parallelism follows a completely different architecture:** `--split-mode tensor` + the ggml_backend_meta "virtual device" abstraction (upstream PR #19378, commit d6f303004). Tensor parallelism is **graph-level**: weight/KV splitting is done by the meta buffer, and the insertion of allreduce between operators is implemented by the meta backend cutting subgraphs in the graph; ggml-cuda.cu only provides the **communication primitives** and does not participate in splitting.
4. **Cross-GPU communication uses neither P2P nor cudaMemcpyPeerAsync nor NCCL (Windows default)**, but **pinned host staging + in-kernel spin (device-side spin)** or **copy-engine + event** (allreduce.cu:13-37). The fork additionally adds a default-on 3-way single-kernel allreduce (allreduce.cu:229-322, 1874-1983).
5. Each layer, each token has **2 allreduce barriers** (attn_output row parallel + ffn_down row parallel); there is no communication-computation overlap between layers. The fork's own diagnostic constant says '~81 subgraphs per decode step' (ggml-cuda.cu:1189-1196).
6. In tensor mode the KV cache is **split to each GPU by KV head, not replicated, with no cross-GPU KV copies** (llama-model.cpp:526-527 + llama-kv-cache.cpp:233-234, 1278-1284).

---

## 1. Version and topology (question 1)

### 1.1 git metadata

[fact] git log -1:

    92758c6475d48021fb32ab875a6dcec651e53df7
    Sat Sep 12 12:59:04 2026 +0800
    ggml-cuda: add 3-way AllReduce kernel (on by default) + timing instrumentation
    HEAD -> ar3-opt, fork/ar3-opt

- Branch: ar3-opt (locally there are also all / ar3-dual / ar3-ring / dflash2-tensor / local / master, etc.).
- remotes: fork https://github.com/zhuojun1024/llama.cpp , origin https://github.com/ggml-org/llama.cpp .
- Upstream base point: origin/master = 41fc7584f (2026-09-10 15:44:40 +0200, 'scripts : use sed instead of grep for version parsing (#28700)'); the fork's merge records 824a41e11 'Merge branch master into all', 924cd0af7 'Merge branch all into ar3-opt'.
- git describe: b10825-91-g92758c647.

[fact] Local changes (git status): only 1 file is modified, **uncommitted**:

    ## ar3-opt
     M common/arg.cpp

Content (around common/arg.cpp:883-893): the default selection of the mmproj device is no longer pinned to params.devices.front() under LLAMA_SPLIT_MODE_TENSOR, because under tensor parallelism mmproj has to go through tensor_split:

    -    if (params.mmproj_use_gpu && params.mmproj_device == nullptr && !params.devices.empty()) {
    +    if (params.mmproj_use_gpu && params.mmproj_device == nullptr && !params.devices.empty()
    +        && params.split_mode != LLAMA_SPLIT_MODE_TENSOR) {

Other than that the worktree is clean (no changes to the remaining files).

[fact] HEAD commit 92758c647 changes 4 files: README.md, ggml/src/ggml-cuda/allreduce.cu (+645), ggml/src/ggml-cuda/ggml-cuda.cu (+257), tools/server/server-context.cpp (+28).

[fact] The fork's self-described topology and goal (README.md:20-51): the goal is '3-GPU inference on a mixed PCIe topology without P2P (two RTX 5060 Ti + one Tesla T10)'; it gives 3 AllReduce route branches: ar3-opt (3-way kernel for small tensors + copy-engine for large tensors), ar3-dual (two 2-GPU pipelines), ar3-ring (ring), all (no 3-GPU AR). Measured table (README.md:46-51): two-pipeline ~37 tok/s decode / ~575 tok/s prefill, 3-way kernel ~44.5 tok/s / ~665 tok/s (Qwen3.8-27B, draft-mtp).
> Note: the user's task says "dual GPU", but this checkout is actually tuned for **3 GPUs**; the CUDA internal AR primitive itself only supports **2 devices**, and 3 devices are composed from two 2-device pipelines.

### 1.2 Which split modes are supported, and which code path each takes

[fact] Enum (include/llama.h:198-203):

| Value | Name | Meaning |
|---|---|---|
| 0 | LLAMA_SPLIT_MODE_NONE | Single GPU |
| 1 | LLAMA_SPLIT_MODE_LAYER | Split by layer + KV follows the layer |
| 2 | LLAMA_SPLIT_MODE_ROW | Old row split (the comment still says 'use tensor parallelism if supported') |
| 3 | LLAMA_SPLIT_MODE_TENSOR | New, tensor parallelism (meta device) |

CLI parsing: common/arg.cpp:2812-2823 (none|layer|row|tensor), help text common/arg.cpp:2808-2811. The documentation docs/multi-gpu.md:22-27 explicitly marks row as **Deprecated** and tensor as **EXPERIMENTAL**.

**(a) layer (pipeline parallel)**
- Each GPU takes a contiguous span of layers; the KV cache lands on the same GPU as its layers.
- Relevant code: src/llama-context.cpp:428-433 decides whether to enable the scheduler's pipeline parallel (split_mode() == LLAMA_SPLIT_MODE_LAYER && offload_kqv && n_gpu_layers > n_layer_all && !has_tensor_overrides()).
- **No collective communication between layers**.

**(b) row (old row split) —— dead on CUDA**
- Path: src/llama-model.cpp:1095-1120 make_gpu_buft_list(), which when split_mode == ROW asks the backend reg for the 'ggml_backend_split_buffer_type' proc address; if it cannot get it, it throws std::runtime_error('device %s does not support split buffers') (llama-model.cpp:1118-1119).
- [fact] That symbol is no longer in the CUDA backend registry: ggml_backend_cuda_reg_get_proc_address() at ggml/src/ggml-cuda/ggml-cuda.cu:5953-5971 only exports ggml_backend_comm_init / comm_free / comm_allreduce_tensor / register_host_buffer / unregister_host_buffer / ggml_backend_get_features.
- [fact] Across the whole repository only SYCL still has ggml_backend_sycl_split_buffer_type (ggml-sycl.cpp:1466, registered at 7029-7030).
- [fact] The commit that deleted it is 74976e1ae 'CUDA: remove -sm row, refactor cuBLAS (#24216)' (2026-07-06), and git merge-base --is-ancestor 74976e1ae HEAD is true.
- [fact] The function ggml_cuda_op_mul_mat **no longer exists** anywhere under ggml/src/ggml-cuda/: grep only hits the typedef void (*ggml_cuda_op_mul_mat_t)(...) at ggml-cuda.cu:1601 plus the single-GPU kernel entry points at mmvf.cu:731 / mmvq.cu:1522. ggml_cuda_mul_mat (ggml-cuda.cu:2084) is now a pure single-GPU implementation; the ggml_cuda_mul_mat(ctx, &src0_slice, ...) inside ggml_cuda_mul_mat_id (ggml-cuda.cu:2315) is a **single-GPU intra-GPU expert slice** (used with --n-cpu-moe), not multi-GPU splitting.
> In other words: the answer to the task's question "how do ggml_cuda_op_mul_mat / ggml_cuda_mul_mat_batched split weight rows across GPUs" in the current checkout is —— **these functions are gone, and the CUDA backend no longer splits across GPUs itself**.

**(c) tensor (currently the only usable tensor parallelism)**
- Device creation: src/llama.cpp:158-220 llama_prepare_model_devices() —— when split_mode == LLAMA_SPLIT_MODE_TENSOR, it wraps all non-CPU devices into **one Meta device** (ggml_backend_meta_device(devs, n, llama_meta_device_get_split_state, &ud), llama.cpp:176-179 / 217-220), and the model only sees 1 device.
- Context constraints: src/llama-context.cpp:3683-3695 —— TENSOR forces flash_attn = ENABLED (AUTO gets changed to ON, explicit OFF errors out directly); single-device use warns.
- Under TENSOR, backend sampling is not supported (llama-context.cpp:1225-1236), --fit is not supported (common/fit.cpp:182-183), and the row weight-allocation adjustment is not supported (common/fit.cpp:476-477).
- Architecture whitelist: src/llama-model.cpp:354-355 llm_arch_supports_sm_tensor(arch); the docs list the failure list at docs/multi-gpu.md:89-93 (MoE/SSM/RWKV and several other classes).
- Code path: the **graph layer** is dominated by ggml/src/ggml-backend-meta.cpp, the **weight layout** is decided by the split-state callback at src/llama-model.cpp:371-839, and **communication** is provided by ggml/src/ggml-cuda/allreduce.cu + ggml-cuda.cu:965-1527.

---

## 2. Operator-level implementation of tensor parallelism (question 2)

### 2.1 How weights are split across GPUs: not an operator split, a buffer split

[fact] The splitting of each tensor is given by the callback llama_meta_device_get_split_state() (src/llama-model.cpp:371-839). It regex-matches tensor names and returns a ggml_backend_meta_split_state{axis, ne[], nr[], n_segments}:

| Tensor | split axis | Evidence |
|---|---|---|
| blk.N.attn_q / attn_k / attn_v .weight | **axis 1** (output dim/head dim) | llama-model.cpp:511-513 |
| blk.N.attn_qkv.weight | axis 1 | llama-model.cpp:517-519 |
| blk.N.attn_output.weight | **axis 0** (input dim) | llama-model.cpp:529-531 |
| blk.N.ffn_up / ffn_gate(_exps) .weight | axis 1 | llama-model.cpp:561-563 |
| blk.N.ffn_down(_exps).weight | **axis 0** | llama-model.cpp:570-572 |
| cache_k_lN / cache_v_lN | **axis 0** | llama-model.cpp:526-527 |
| output.weight | axis 1 (unless output_mirrored / dsv4) | llama-model.cpp:581-586 |
| everything else (q/k bias, qk_norm, sinks… as needed) | axis 0/1/MIRRORED | llama-model.cpp:514-575 |
| fallback | MIRRORED (fully replicated on every GPU) | llama-model.cpp:594 |

This is exactly Megatron-style **column-parallel (QKV/FFN-up, axis 1) + row-parallel (attn_output/FFN-down, axis 0)**, and the matmul output of the row-parallel side is marked GGML_BACKEND_SPLIT_AXIS_PARTIAL (partial sum, needs allreduce).

[fact] The split ratio comes from tensor_split and is aligned to quantization block/head granularity: llama-model.cpp:786-836 (find the cut points via tensor_split_scan, round down to the granule), with the granularity given by get_split_granularity() (597-784) —— for example KV weights/KV cache use granularity_kv = granularity_q / n_gqa (755-761), ensuring the Q head count on each GPU is exactly n_gqa times its KV head count. There is also a rotation mechanism that makes different layers alternately round up/down for better balance (llama-model.cpp:453, 459, 462, 796, 815-818).

[fact] The physical slicing happens at the meta buffer layer: ggml-backend-meta.cpp:1180-1250 computes ne/nb/view_offs for each backend's t_ij (simple tensor, i.e. the piece on that GPU); ggml_backend_meta_buffer_type_alloc_buffer allocates one buffer per backend (1687-1701). The meta buffer interface is at 1659-1671.

### 2.2 "Combining partial sums": graph-level subgraph boundary + comm_allreduce, not a reduce inside ggml-cuda

[fact] The meta backend's graph_compute:
1. First compute split_state node by node over the whole graph; any node with axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL is a **subgraph boundary** (ggml-backend-meta.cpp:2182-2222, the test is at 2192).
2. In the loop, first let **all** backends run cgraph_main[i] (2434-2441), then call comm_allreduce for each backend's **last node** of that subgraph (2443-2454):

       if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
           backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
       }
       if (!backend_allreduce_success) { allreduce_fallback(i); }   // 2456-2461

3. If a backend does not provide a comm implementation, it falls back to the meta's generic butterfly: pairwise ggml_backend_tensor_copy_async + GGML_OP_ADD (2317-2431, n_reduce_steps = ceil(log2(n_devs)) see 1810).

[fact] comm_allreduce is a backend-registered proc address; on the CUDA side it is ggml_backend_cuda_comm_allreduce_tensor (ggml-cuda.cu:1521-1527), dispatched by the function pointer try_allreduce chosen by comm_init:

| Priority | Implementation | Location |
|---|---|---|
| 1 | NCCL (ncclAllReduce, FP32 for small tensors / BF16 for large tensors) | ggml-cuda.cu:1009-1084, 1444-1474 |
| 2 | internal (self-written pinned-host AR) | ggml-cuda.cu:1290-1363, 1408-1442 |
| 3 | none → meta butterfly fallback | ggml-cuda.cu:1404-1406, 1439-1441 |

[fact] Platform defaults (Linux goes NCCL, everything else/Windows goes internal):

    // ggml-cuda.cu:1494-1501
    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
    #if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
    #else
        ggml_backend_cuda_comm_init_internal(ret);
    #endif
    }

NCCL is disabled under virtual devices (1446-1453). GGML_CUDA_NCCL is a CMake **default ON** (ggml/CMakeLists.txt:210), but on Windows, if the NCCL package is not found it only prints a warning (ggml/src/ggml-cuda/CMakeLists.txt:174-182).

### 2.3 The mechanism of internal AllReduce (core): pinned host + kernel spin / copy-engine + event

[fact] The header comment of ggml/src/ggml-cuda/allreduce.cu:13-37 directly defines the two strategies:
- **chunked kernel path (small tensors, decode)**: one kernel writes the data to pinned host and does the local add at the same time; **cross-GPU synchronization is inside the kernel** (busy-wait on host-memory flags).
- **copy-engine path (large tensors, prefill)**: chunked D2H + H2D cudaMemcpyAsync + one small device-side add kernel; **cross-GPU synchronization is outside the kernel** (CUDA event).
- It explicitly writes 'targets setups without NVLink, where data is exchanged between the GPUs by staging it through pinned host memory over PCIe' (allreduce.cu:18-19).

[fact] Synchronization primitives (allreduce.cu:40-66):
- Each (slot, rank) has an int arrival token in pinned host memory; the writer does *(volatile int*)p = token, and the reader spins on *(const volatile int*)p until the token is equal (allreduce.cu:61-66).
- The writer does __threadfence_system() before publishing the token (release semantics); the comment explains why atomicAdd_system is not used: 'requires hostNativeAtomicSupported, which is unavailable on PCIe-attached consumer GPUs without NVLink' (allreduce.cu:52-58).
- One arrival int per 64 B to avoid false sharing (allreduce.cu:68-71); 8 blocks, each block an independent token (allreduce.cu:73-76, 101-106); __nanosleep(100) in the spin (allreduce.cu:288-292).
- Each token is a monotonically increasing AR call sequence number that does not wrap (31-bit int), saving a per-round reset (allreduce.cu:44-49). Ring depth GGML_CUDA_AR_POOL_SIZE = 2 (allreduce.cu:401-407).

[fact] The pinned host allocation is cudaHostAlloc(..., cudaHostAllocPortable | cudaHostAllocMapped) + cudaHostGetDevicePointer (allreduce.cu:441-474), i.e. mapped memory that the CPU does not touch and the GPU reads/writes directly.

[fact] The 2-device chunked path main loop (allreduce.cu:1265-1475):
- Check type/alignment/contiguity; if unsupported, return false and let the upper layer fall back (1303-1334);
- When F32 and nbytes >= bf16_threshold (default 1, i.e. always on), go through the **F32→BF16 on-the-wire compression round trip** (1282-1289, 604-607);
- Loop over chunks (each chunk capped at GGML_CUDA_AR_MAX_BYTES = 1 MB, allreduce.cu:411, 1417), launching the bidirectional kernel per chunk per device (1441-1462), running on the **caller's compute stream** (the comment at allreduce.cu:1411-1416 explicitly says this is a barrier);
- At the end records ev.ker (1467-1469).

[fact] The 2-device copy-engine path (allreduce.cu:1116-1229): stage 1, each GPU does D2H to its own pinned buffer, recording an event per chunk (1158-1167); stage 2, each GPU waits for the peer's chunk event, does H2D to its own dev_tmp, records host_large_read_done, then cudaEventRecord(h2d) + compute stream cudaStreamWaitEvent(h2d), and runs the add kernel (1188-1217). Thresholds GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 1 MB, single-shot cap GGML_CUDA_AR_COPY_MAX_BYTES = 32 MB (419, 415); above 32 MB the outer chunker slices it (1231-1245).

[fact] The fork's 3-way single kernel (allreduce.cu:229-322): each device casts its own shard to T_wire and writes it into its own pinned slot, then **each block spins waiting for the tokens of the other two peers**, and finally reads the three slots and adds them in a **fixed device order** ((w0+w1)+w2) —— the comment says this guarantees the three-GPU result is **bit-identical** (allreduce.cu:220-227, 301-321). Call site ggml_cuda_ar_allreduce3way (1727-1822), on by default, disabled with GGML_CUDA_AR3WAY=0 (1902-1938); a single tensor must fit into one rank slot (1 MB) otherwise it returns false and falls back to the 2-pipeline (1751-1755).

[fact] The 3-device composition of two 2-device pipelines: (dev0,dev1) and (dev0,dev2) share the pivot dev0 (ggml-cuda.cu:1408-1435); the sequence is AR(dev0,dev1) → AR(dev0,dev2) → host-staged broadcast dev0→dev1 (allreduce.cu:1939-1975, broadcast kernel 324-395). Large tensors go through a dedicated 3-device copy-engine (allreduce.cu:894-1115, 1888-1900). If the broadcast fails it falls back to ggml_backend_tensor_copy_async (1972-1975).

### 2.4 The actual extent of P2P / cudaMemcpyPeerAsync / NCCL usage

[fact] cudaDeviceEnablePeerAccess is **only called when the GGML_CUDA_P2P environment variable is present** (ggml-cuda.cu:391-405); a NCCL build also implicitly enables peer access (ggml-cuda.cu:605-609, 611-640, VMM allocation requires an explicit cuMemSetAccess).
[fact] cudaMemcpyPeerAsync only appears in the generic tensor copy cpy_tensor: the buffer version ggml-cuda.cu:820-843 (line 834), the backend async version 2746-2801 (line 2784). Both **directly return false** under #ifdef GGML_CUDA_NO_PEER_COPY (831-835 / 2781-2785), and set props.caps.events to false (5306-5311) with event_new returning nullptr (5819-5842). GGML_CUDA_NO_PEER_COPY is controlled by the CMake option GGML_CUDA_NO_PEER_COPY, **default OFF** (ggml/CMakeLists.txt:203; definition at ggml/src/ggml-cuda/CMakeLists.txt:143-145).
[fact] **The AllReduce path does not use peer copy at all**: internal AR only does host staging; the NCCL path uses ncclAllReduce. That is, on a machine without P2P, as long as GGML_CUDA_P2P is not enabled, there will be no D2D peer traffic on the machine (unless the meta butterfly fallback goes through tensor_copy_async, and that path does reach the cudaMemcpyPeerAsync at 2784).

---

## 3. Number and timing of cross-GPU communications (question 3)

### 3.1 Count

[fact] Communication points = the PARTIAL boundaries in the graph (ggml-backend-meta.cpp:2188-2192). The rule for producing PARTIAL is in handle_mul_mat (ggml-backend-meta.cpp:577-608):
- src0 axis 1 split + src1 MIRRORED → output axis 0 split (column parallel, **no communication**);
- src0 axis 0 split + src1 axis 0 split → **PARTIAL** (row parallel, **needs allreduce**).

[fact] The two row-parallel matmuls of a standard decoder layer: attn_output.weight axis 0 (llama-model.cpp:529-531) and ffn_down.weight axis 0 (570-572). Therefore:
> **Every transformer layer, every token = 2 cross-GPU allreduces** (one after attn_output, one after ffn_down); about 2*n_layer per token for n_layer layers.

[fact/corroboration] The fork's subgraph timing diagnostics write the slot count as **81**, the original comment being 'one per meta subgraph, ~81 per decode step' (ggml-cuda.cu:1189-1196), R = 81 (1196). The form 81 = 2×40+1 fits "40 layers, 2 boundaries per layer" exactly (the fork's model is Qwen3.8-27B).
[inference] If the model is MoE, PARTIAL also comes from ffn_down_exps; if it is a linear-attention hybrid architecture, row-parallel layers such as ssm_out each contribute one boundary too. The exact layer count/model will differ.

[fact] Communication volume: each AR is a summation over the **entire hidden dimension** (the row-parallel output, shaped like [n_embd, n_tokens]); during decode n_tokens is small (the comment gives the example '80 KiB', allreduce.cu:1751-1753), during prefill it is large.

### 3.2 Timing and synchronization

| Scenario | Form of synchronization | Evidence |
|---|---|---|
| Small tensors (chunked kernel) | **in-kernel spin** (volatile + __threadfence_system + __nanosleep), the kernel runs on the compute stream and is a hard barrier | allreduce.cu:61-66, 276-299, 1411-1416 |
| Large tensors (copy-engine) | **stream events** (app/cpy/h2d/ker event pool), the add kernel on the compute stream, H2D on a dedicated non-blocking stream | allreduce.cu:434-439, 562-567, 1137-1224 |
| ring wraparound | **host-side cudaEventSynchronize** (POOL_SIZE=2, performed on every AR from the 3rd one on, waiting on the *previous* AR's ker event) | allreduce.cu:537-550 |
| 3-way kernel | in-kernel spin waiting for two peers | allreduce.cu:280-299 |
| Whole-device synchronization | ggml_backend_meta_synchronize synchronizes backend by backend; the llama layer llama_synchronize | ggml-backend-meta.cpp:1957-1962 |

[fact] No cudaStreamSynchronize appears in the AR path; the only host-side blocking points are the cudaEventSynchronize on ring wraparound (537-550) and cudaEventElapsedTime at the timing nodes.

### 3.3 Overlap of communication and computation

[fact] The overlap mechanism **exists only within a single AR**; there is no cross-layer/cross-subgraph communication-computation overlap:
- The copy-engine path lets D2H/H2D run on a dedicated AR stream via the copy engine while the add kernel runs on the compute stream; the comment explicitly writes 'This keeps the compute engine free while large transfers are in flight, which matters for prefill-sized tensors' (allreduce.cu:29-34); the AR stream uses cudaStreamCreateWithFlags(cudaStreamNonBlocking) (617).
- The chunked path is **deliberately** placed on the compute stream (allreduce.cu:1411-1416): the comment says 'since AR is a barrier here, same-stream ordering subsumes any cross-stream event handshake... and skips the cross-stream scheduling overhead that was hurting the small-tensor (tg) latency'.
- The meta backend is **lockstep subgraph by subgraph**: first all GPUs run subgraph i (async launch), then allreduce, then subgraph i+1 (ggml-backend-meta.cpp:2434-2461). There is no mechanism to overlap the computation of subgraph i+1 with the communication of subgraph i (they are on the same per-device stream, hence inherently serial).
- The 3-way kernel must wait for the slowest peer (README.md:44), so -ts needs to be balanced.

---

## 4. CUDA graph (question 4)

### 4.1 Compilation and switches
- GGML_CUDA_GRAPHS is default ON (CMakeLists.txt:170); when enabled, add_compile_definitions(GGML_CUDA_USE_GRAPHS) (ggml/src/ggml-cuda/CMakeLists.txt:123-125), and common.cuh:1261-1263 maps it to USE_CUDA_GRAPH.
- At runtime it can be turned off with GGML_CUDA_DISABLE_GRAPHS (common.cuh:1292).

### 4.2 Capture/reuse locations
- Entry point ggml_backend_cuda_graph_compute() (ggml-cuda.cu:4681-4752).
- The real work is done by ggml_cuda_graph_evaluate_and_capture() (ggml-cuda.cu:4446-4662): cudaStreamBeginCapture(..., cudaStreamCaptureModeRelaxed) (4732) → execute (or capture) node by node → cudaStreamEndCapture (4635) → cudaGraphInstantiate (4650) → cudaGraphLaunch (4656).
- Updating an existing instance uses cudaGraphExecUpdate (2903), and on failure it destroys and rebuilds (2910-2920).

### 4.3 Cache key and reuse criterion
- **key = the pointer of cgraph->nodes[0]** (ggml-cuda.cu:2852-2854), stored in a per-CUDA-context std::unordered_map<const void*, ggml_cuda_graph> (common.cuh:1461-1489).
- After a hit, it is only reused directly if cgraph->uid == graph->uid (2862-2867); otherwise it compares node structure node by node + each src's data pointer + ne/nb (2877-2895), and any change requires recapture.
- So the exact answer to "what key is it cached by" is: **indexed by the first node's address + an all-graph node-by-node shape/pointer snapshot for the equivalence decision**, and **not** a parameterized 'envelope'/batch bucket. [inference] Since llama reuses the same batch of ggml_context/buffer, the first node's address is stable for the same ubatch shape → the same graph can be reused long-term; a change in ubatch/batch size (e.g. prefill chunk vs single-token decode) triggers a property change → recapture.

### 4.4 warmup and "how many times can one generation reuse it"
[fact] (ggml-cuda.cu:4690-4722):
- First call: property change → **execute directly, no capture** (use_cuda_graph = false);
- Second call with unchanged properties → warmup_complete = true, and this time it **captures and launches** (4701-4708);
- Every time after that: use_cuda_graph = true, cuda_graph_update_required = (instance == nullptr) → only cudaGraphLaunch to reuse (4716-4719).
> **【inference】** If one round of requests does R decode steps, then roughly 1 warmup direct execution + R−1 captures/reuses (of which the 2nd is capture+launch). Because decode is one call per token with an unchanged shape, in steady state **every token reuses the same instance**.
- Eviction: scans every 5 s, deletes after 10 s of disuse (common.cuh:1471-1481).
- **Note that in tensor mode it is "one CUDA graph per subgraph"**: the meta backend cuts the whole graph into ~81 subgraphs, each subgraph calls graph_compute once per backend → about 81 graphs per device, replaying 81 times per decode step (evidence ggml-cuda.cu:1189-1196).

### 4.5 host-side memcpy needed before replay
[fact] Input tensors (tokens/pos/kq_mask/k_idxs/…) are written with ggml_backend_tensor_set[_async] before the replay (examples src/llama-graph.cpp:73, 81, 98, 104, 113, 142, 172). The meta device distributes them according to split_state: MIRRORED → one write per backend; axis-split → assembled chunk by chunk onto each GPU with set_tensor_2d_async (ggml-backend-meta.cpp:1867-1910).
[fact] Two write paths on the CUDA side:
- buffer synchronous version: cudaMemcpyAsync(..., cudaMemcpyHostToDevice, cudaStreamPerThread) **immediately followed by cudaStreamSynchronize(cudaStreamPerThread)** (ggml-cuda.cu:784-790) —— host blocks, and the source pointer is directly the caller's memory.
- backend async version: cudaMemcpyAsync(..., cuda_ctx->stream()) (ggml-cuda.cu:2706-2713), attached to the compute stream, hence naturally ordered with the subsequent graph launch, **no extra synchronization needed**.
[fact] **Whether it is pinned**: by default it is not pinned —— the source of cudaMemcpyAsync is ordinary host memory (e.g. llama_batch/std::vector). pinned only in these places:
- ggml_cuda_host_malloc() uses cudaMallocHost, can be disabled by GGML_CUDA_NO_PINNED (ggml-cuda.cu:1545-1561), and determines the host_buffer cap in device props (5306);
- AR's staging/arrival is always pinned+mapped (allreduce.cu:451-465).
[fact] Output side: meta's get_tensor_async does D2H from **each GPU** chunk by chunk for axis-split tensors and then assembles them on the host (ggml-backend-meta.cpp:1912-1955) —— this is also how logits (split along the vocab axis 1) are collected in tensor mode; MIRRORED tensors are only read from device 0 (1945-1950).
[fact] KV cache writes are an **in-graph op** (cpy_k/cpy_v → GGML_OP_SET_ROWS and the like), not a host memcpy.
[fact] These H2D transfers will not be inside the CUDA graph capture: they happen before ggml_backend_graph_compute (host thread) and do not participate in the capture.

---

## 5. Multi-GPU distribution of the KV cache (question 5)

[fact] KV physical tensor: ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) (src/llama-kv-cache.cpp:233-234), i.e. [head_dim × n_head_kv, kv_size, n_stream].
[fact] The split-state callback marks cache_k_lN / cache_v_lN as **axis 0** (llama-model.cpp:526-527), that is, **split along n_embd_k_gqa = split by KV head**; the granularity uses granularity_kv = granularity_q / n_gqa (llama-model.cpp:755-761) to ensure the cut points fall on head boundaries.
[fact] On read, get_k()/get_v() return a 4D view [head_dim, n_head_kv, n_kv, n_stream] (llama-kv-cache.cpp:1278-1284 / 1301-1315), and then ggml_permute(q/k/v, 0, 2, 1, 3) moves heads to axis 2 (src/llama-graph.cpp:2607-2611); the FA output shape is {v->ne[0], q->ne[2], q->ne[1], q->ne[3]} (ggml/src/ggml.c:5525-5527), so heads are on axis 1.
[fact] handle_flash_attn_ext asserts that Q is split on axis 2, and K/V are **either split on axis 2 or entirely MIRRORED** (ggml-backend-meta.cpp:773-791).
> **Conclusion (fact): in tensor/row mode the KV cache needs no cross-GPU replication** —— each GPU only holds its own portion of KV heads, does the attention for its own heads locally, and the partial results are merged by the allreduce of attn_output. **There is no cross-GPU cudaMemcpy of KV**.
[fact] Exceptions (the cache/state of these models is forced to MIRRORED, i.e. every GPU redundantly replicates the full KV):
- dsv4's cache_k/cache_v and dsv4_*_state_* (llama-model.cpp:474-478);
- indexer cache (501-503), PLE table (505-508);
- LFM2/LFM2MOE's cache_r/cache_s (546-549).
[fact/note] The documentation says 'SPLIT_MODE_TENSOR + KV cache quantization' errors out at startup (docs/multi-gpu.md:87, 121), but in this version of the code I **did not find that explicit check** (the TENSOR-related checks in llama-context.cpp are only the flash_attn requirement at 3683-3695 and the backend sampling at 1225; there is no occurrence of TENSOR in llama-kv-cache.cpp either). [inference] That documentation entry may be stale, or the restriction is expressed indirectly elsewhere (meta split-state assertions / FA kernel capabilities) —— unverified.

---

## 6. What it "does not do" + Windows/WDDM/no-P2P limitations (question 6)

### 6.1 Things it explicitly does not do (compared against what a self-written engine might do)

| Item | Current state | Evidence |
|---|---|---|
| **Communication-computation fusion (in-kernel allreduce / fusing the reduce into the GEMM epilogue)** | **No**. AR is a standalone kernel, and there are 2 barriers per layer. The fork's 'in-kernel' only means *the cross-GPU spin inside the communication kernel*, not fusion with computation | allreduce.cu:229-322; ggml-backend-meta.cpp:2434-2461 |
| **Overlapping communication with the next layer's computation (communication/computation overlap)** | **No**. The meta is lockstep subgraph by subgraph; AR and compute are on the same stream (small tensors) or overlap only with the copies of the same AR | ggml-backend-meta.cpp:2434-2461; allreduce.cu:29-36, 1411-1416 |
| **Graph cache parameterized by envelope/bucket** | **No**. The key is the first node pointer + a node-by-node property snapshot; a batch-size change triggers a warmup reset + recapture | ggml-cuda.cu:2852-2894, 4712-4715 |
| **host checkpoint ring / activation ring** | **No**. The AR ring has only 2 slots, and it is used only for communication staging | allreduce.cu:401-407 |
| **Cross-GPU P2P/NVLink direct AR** | **No**. AR always goes through pinned host (only cpy_tensor has peer copy, and peer access is not enabled by default) | allreduce.cu:18-19; ggml-cuda.cu:391-405, 2784 |
| **Activation compression (FP8/INT8)** | Only an F32→BF16 round trip (can be disabled) | allreduce.cu:604-607, 1282-1289 |
| **Quantized KV + tensor parallelism** | Not implemented/not verified (the docs claim it errors, no check found in the code) | docs/multi-gpu.md:87,121 |
| **backend sampling under tensor parallelism** | Explicitly unsupported, falls back to CPU | llama-context.cpp:1225-1236 |
| **--fit automatic memory sizing under tensor parallelism** | Explicitly unsupported | common/fit.cpp:182-183, 476-477 |
| **row split on CUDA** | Deleted | commit 74976e1ae; llama-model.cpp:1118-1119 |
| **Efficient AR for >3 GPUs** | internal AR only supports 2 devices; 3 devices rely on two 2-device pipelines; >3 devices fall back to the meta butterfly (pairwise tensor copy + ADD, the slowest) | allreduce.cu:575-590; ggml-cuda.cu:1408-1441; ggml-backend-meta.cpp:2317-2431 |
| **Multiple virtual devices on the same physical GPU** | It is handled (compares physical device; the same physical GPU goes D2D instead of peer copy) | ggml-cuda.cu:824-830, 2774-2780 |

### 6.2 Limitations/fallback paths under Windows + WDDM + no P2P (SYS topology)

[fact]
1. **AR backend selection**: non-Linux (including Windows) defaults to **internal** (pinned host) rather than NCCL (ggml-cuda.cu:1494-1501); when NCCL is not compiled it warns and falls back to internal (1466-1471); NCCL is disabled under virtual devices (1446-1453).
2. **All cross-GPU data goes through pinned host, two PCIe passes** (allreduce.cu:18-19, 441-474) —— the inevitable cost under SYS/no-P2P topologies, and also the motivation for the fork's 3-way kernel.
3. **P2P must be explicitly enabled** with GGML_CUDA_P2P (ggml-cuda.cu:391-405); the docs warn that on some motherboards/BIOS (IOMMU) it crashes or produces corrupted output (docs/multi-gpu.md:104-112, 127).
4. **Compile-time switch for peer copy** GGML_CUDA_NO_PEER_COPY (default OFF, ggml/CMakeLists.txt:203): when enabled, cpy_tensor directly returns false (ggml-cuda.cu:831-835, 2781-2785), device props events = false (5307-5311), event_new returns nullptr (5826-5829), and the feature reports NO_PEER_COPY (5918-5920). The author added this switch precisely for no-P2P platforms.
5. **Admission conditions for the internal AR**: must have cc >= Volta (needs __nanosleep) and n_devices == 2, otherwise init fails, returns nullptr and falls back (allreduce.cu:575-590). 3 devices are assembled from two 2-device pipelines sharing a pivot (ggml-cuda.cu:1408-1435); if that also fails it falls back to the meta butterfly (1437-1441) —— a generic implementation that is pure tensor_copy_async + GGML_OP_ADD (ggml-backend-meta.cpp:2317-2431), which becomes a host round trip with no P2P and has the worst performance.
6. **HIP/MUSA directly do not support** internal AR: allreduce.cu:1985-2004 provides empty implementations (return nullptr/false), with comments explaining that HIP lacks cudaHostAllocPortable/Mapped, cudaHostGetDevicePointer, __nanosleep.
7. **Windows-related build patches**: the fork has commit 6c98116ed 'ggml-cuda: fix winsock.h (v1) conflict with NCCL headers on Windows' (README.md:35; there is a comment at CMakeLists.txt:259).
8. **WDDM**: grep for WDDM across the whole repository gives **zero hits** —— there is **no WDDM-specific branch/fallback** in the code. [inference] Under WDDM, pinned memory is limited, page-locked staging for cudaMemcpyAsync is more expensive, and GPU submission through the command queue adds latency; this implementation handles none of that specifically; the only indirect mitigation is defaulting to internal (avoiding NCCL's problems on WDDM).

### 6.3 Environment variables/constants quick reference (for aligning with a self-written engine)

| Name | Default | Purpose | Location |
|---|---|---|---|
| GGML_CUDA_ALLREDUCE | platform default (Linux=nccl, others=internal) | Select nccl/internal/none | ggml-cuda.cu:1494-1514 |
| GGML_CUDA_AR3WAY | on | =0 disables the 3-way kernel | allreduce.cu:1907-1908 |
| GGML_CUDA_AR_TIMING | off | Print ar_timing/sg_timing/ar3way | ggml-cuda.cu:1097, 1337; allreduce.cu:1918 |
| GGML_CUDA_P2P | off | cudaDeviceEnablePeerAccess | ggml-cuda.cu:391 |
| GGML_CUDA_DISABLE_GRAPHS | off | Disable CUDA graph | common.cuh:1292 |
| GGML_CUDA_NO_PINNED | off | Disable pinned host buffer | ggml-cuda.cu:1546, 5306 |
| GGML_CUDA_AR_COPY_THRESHOLD | 1 MB | chunked→copy-engine threshold | allreduce.cu:419, 595 |
| GGML_CUDA_AR_BF16_THRESHOLD | 1 (always on) | F32→BF16 on-the-wire compression | allreduce.cu:604-607 |
| GGML_CUDA_AR_COPY_CHUNK_BYTES | 0 (heuristic nbytes/4) | copy chunk size | allreduce.cu:598-603, 552-560 |
| LLAMA_SERVER_TIMING | off | server timing | README.md:56; server-context.cpp changes |
| GGML_CUDA_NO_PEER_COPY | CMake OFF | Disable peer copy/event | ggml/CMakeLists.txt:203 |
| GGML_CUDA_NCCL | CMake ON | Compile NCCL | ggml/CMakeLists.txt:210 |
| GGML_CUDA_GRAPHS | CMake ON | Compile CUDA graph | CMakeLists.txt:170; ggml/CMakeLists.txt:209 |

---

## Appendix: index of all key line numbers in this document

- split mode enum: include/llama.h:198-203
- layer pipeline decision: src/llama-context.cpp:428-433
- row requires split buffer: src/llama-model.cpp:1095-1120
- CUDA reg proc address (no split_buffer): ggml/src/ggml-cuda/ggml-cuda.cu:5953-5971
- TENSOR creates the Meta device: src/llama.cpp:158-220
- TENSOR context constraints: src/llama-context.cpp:3683-3695
- split-state callback: src/llama-model.cpp:371-839 (tensor_config 473-595, segments 597-669, granularity 671-784, scan 786-836)
- meta buffer slicing/allocation: ggml/src/ggml-backend-meta.cpp:1180-1250, 1659-1701
- subgraph splitting + allreduce scheduling: ggml/src/ggml-backend-meta.cpp:2182-2222, 2434-2461
- generic butterfly fallback: ggml/src/ggml-backend-meta.cpp:2317-2431
- meta input/output distribution: ggml/src/ggml-backend-meta.cpp:1867-1955
- mul_mat split rules: ggml/src/ggml-backend-meta.cpp:577-608
- FA/KV split rules: ggml/src/ggml-backend-meta.cpp:773-791
- CUDA comm context / NCCL / internal dispatch: ggml/src/ggml-cuda/ggml-cuda.cu:965-1084, 1290-1527
- 3-device pipeline composition: ggml/src/ggml-cuda/ggml-cuda.cu:1408-1435
- internal AR implementation: ggml/src/ggml-cuda/allreduce.cu:13-37, 61-76, 108-322, 397-712, 894-1229, 1265-1475, 1631-1705, 1722-1983
- P2P/peer copy: ggml/src/ggml-cuda/ggml-cuda.cu:391-405, 605-648, 820-843, 2746-2801, 5306-5311, 5819-5842
- CUDA graph: ggml/src/ggml-cuda/ggml-cuda.cu:2818-2925, 4446-4662, 4681-4752; ggml/src/ggml-cuda/common.cuh:1261-1296, 1459-1489
- subgraph diagnostics (81): ggml/src/ggml-cuda/ggml-cuda.cu:1189-1196
- KV tensors and views: src/llama-kv-cache.cpp:233-234, 1266-1316; src/llama-graph.cpp:2602-2611; ggml/src/ggml.c:5525-5527
- documentation: docs/multi-gpu.md:22-27, 37, 80-93, 104-112, 118-127; README.md:20-56
