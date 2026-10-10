# TP-2 Execution Design (2× RTX 5060 Ti, Qwen3.8-27B NVFP4)

## 0. Goal
Get TP-2 decode of 27B NVFP4 working on dual 5060 Ti under WSL2 (C=1..8),
MTP0 ≥28 tok/s, MTP3 ≥70 tok/s. The TP=1 path behavior is unchanged.

## 1. Sharding Table (27B dense, hidden=5120, 48 layers)

| Weight | Full Size | Sharding Method | Per-GPU Size | Communication |
|---|---|---|---|---|
| QKV (attn) | [16384,5120] | column-parallel (shard N) | [8192,5120] | none (head-parallel) |
| GDN qkv+z | [14336,5120] | column-parallel (shard N) | [7168,5120] | none (head-parallel) |
| o_proj (attn) | [5120,6144] | row-parallel (shard K) | [5120,3072] | all-reduce (sum) |
| GDN output | [5120,6144] | row-parallel (shard K) | [5120,3072] | all-reduce (sum) |
| gate_up (MLP) | [34816,5120] | column-parallel (shard N) | [17408,5120] | none (silu_mul local) |
| down (MLP) | [5120,17408] | row-parallel (shard K) | [5120,8704] | all-reduce (sum) |
| lm_head | [248320,5120] | column-parallel (shard N) | [124160,5120] | all-gather (concat) |
| token_embedding | [248320,5120] | replicated | same as full size | none |
| RMSNorm/bias | [5120] etc. | replicated | same as full size | none |

2 all-reduce points per layer (mixer output + MLP output) × 48 layers = 96 times/forward.
all-reduce implementation: host-staging (pinned buffer + CPU addition); P2P is unavailable (WSL2).

## 2. Execution Flow (decode, T=1..8)

```
for each GPU in {0, 1}:
    x = embedding(ids)          // replicated, identical on both GPUs
for layer in 0..47:
    for each GPU in {0, 1}:
        x = rmsnorm(x)
        if full_attention:
            qkv = linear(x, QKV_shard)       // [8192, T] column-parallel
            q, k, v, gate = split(qkv)       // local head sharding
            attn_out = causal_softmax_attention(q, k, v, gate)  // head-parallel
            out = linear(attn_out, o_proj_shard)  // [5120, T] row-parallel (partial sum)
        else:  // GDN
            qkvz = linear(x, GDN_shard)      // [7168, T] column-parallel
            gdn_out = gated_delta_net(qkvz)  // head-parallel
            out = linear(gdn_out, GDN_out_shard)  // [5120, T] row-parallel (partial sum)
    allreduce(out)                        // sum → identical on both GPUs
    for each GPU in {0, 1}:
        x = rmsnorm(x)
        gu = linear(x, gate_up_shard)     // [17408, T] column-parallel
        act = silu_mul(gu[:8704], gu[8704:])  // [8704, T] local
        delta = linear(act, down_shard)   // [5120, T] row-parallel (partial sum)
    allreduce(delta)                      // sum → identical on both GPUs
    x += delta                            // residual (identical on both GPUs)
x = rmsnorm(x, final_norm)                // replicated
for each GPU in {0, 1}:
    logits_half = linear(x, lm_head_shard)  // [124160, T]
allgather(logits_half) → logits           // [248320, T]
sample(logits)                            // single-GPU sampling
```

## 3. Key Design Decisions

### 3.1 Fused Op Decomposition
The 27B dense FFN uses the fused Ops linear_swiglu + linear_add (dispatch hardcodes the full shape).
TP-2 does not register a new fused kernel; instead it **decomposes** them into:
- linear_swiglu → linear(gate_up_shard) + silu_mul (column-parallel, no communication needed)
- linear_add → linear(down_shard) + residual_add + all-reduce (row-parallel)
The sharding logic lives in the execution layer (TextContext); the Op layer keeps the already-verified half-size linear.

### 3.2 KV/State Pools
- Attention KV: head-parallel; each GPU stores the KV of its own heads (local pool, no communication)
- GDN state: head-parallel; each GPU stores the state of its own heads (local pool, no communication)
- Per-GPU KV capacity = full capacity / 2 (proportional to the number of heads)

### 3.3 lm_head Vocab Sharding
- Each GPU computes [124160, T] logits (BF16 linear; the generic GEMM has no shape restriction)
- all-gather (concat along the vocab dimension) → [248320, T]
- single-GPU sampling (GPU 0)

### 3.4 Synchronization Strategy
- 2 all-reduce points per layer; before each all-reduce the streams of both GPUs must be synchronized
- the host-staging all-reduce does D2H internally (implicitly waiting for compute to finish)
- no explicit cudaStreamSynchronize is needed (the D2H memcpy already blocks until the source is ready)
- in practice it is needed: cudaStreamSynchronize both streams before all-reduce to ensure compute has finished

### 3.5 TP=1 Unchanged
- all TP code is gated on shard_count > 1
- with TP=1, the behavior of Parameters/TextContext/Program is byte-identical to before the change

## 4. Implementation Phases

### 4.1 Weight Sharding Tool (host side)
- Input: the complete NVFP4/BF16 weight payload
- Output: two shard payloads (column sharding = contiguous byte slice; row sharding = strided + scale remapping)
- Reuse the slice_nvfp4_rows/slice_nvfp4_cols logic of the Phase 2 oracle
- BF16 weights: simple row/column slicing (no scale plane)

### 4.2 Dual-Parameters Construction
- Shard the complete weights from the Model → two Parameters (each holding half-size weights)
- Each Parameters' Weight points to the device memory of its corresponding GPU
- Replicated weights (norm/embedding) are held once per GPU

### 4.3 TP Execution Coordinator
- New class TPTwoTextContext: holds two TextContexts + a DevicePair
- run_layers_tp(): lockstep layer by layer; on each layer both GPUs run their local computation, synchronizing at the all-reduce points
- Modify the ordinary_decode_batch entry: with TP-2 it goes through TPTwoTextContext

### 4.4 KV/State Pool Sharding
- Independent KV pool per GPU (capacity = full capacity / 2)
- Independent GDN state pool per GPU
- Program holds two KV stores + two state stores

### 4.5 lm_head + Sampling
- Each GPU computes half-vocab logits → all-gather → single-GPU sampling
- Broadcast the sampling result to both GPUs (or sample only on GPU 0 and pass the result through the host)

### 4.6 Program/Engine Integration
- Program holds a DevicePair + dual Parameters + dual KV/state pools
- The Engine detects the TP configuration at startup (--tp 2)
- With TP=1 it takes the original path (unchanged)

## 5. Verification
- Per-step oracle: TP-2 output vs TP-1 output (same input)
- Single-token forward numerically compared against the Phase 2 oracle
- perplexity evaluation (small corpus)
- user test rounds

## 6. Risks
- R1: all-reduce latency (host-staging ~10-20μs each × 96 times = ~1-2ms/forward)
  → acceptable under the 37.8 tok/s ceiling (~26ms per token)
- R2: dual-GPU synchronization overhead (stream sync × 96 times)
  → use cudaEvent instead of stream sync to reduce the overhead
- R3: KV pool capacity halved → context length is limited
  → 27B on 16GB cards; KV capacity is itself limited, and after TP-2 each card gives 8GB to KV
- R4: performance loss from decomposing the fused Op (linear + silu_mul vs the fused linear_swiglu)
  → correctness first; performance tuning comes in Phase 7
