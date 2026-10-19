# GGUF block formats

NInfer serves Qwen3.8-27B GGUF releases that give every tensor its own ggml quantization type,
such as ISTA-DASLab's [GSQ-RCO models](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF),
without converting their weights to another format. The converter recipe `qwen3_8_27b_gguf` copies
each quantized tensor's blocks into the artifact unchanged (see
[weight conversion](weight-conversion.md#a-mixed-precision-qwen38-27b-gguf)), and the runtime
multiplies them in place.

## Formats

All fifteen block types llama.cpp writes for dense models: `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`,
`Q5_K`, `Q6_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`,
`IQ4_NL` and `IQ4_XS`, stored as the `gguf_*` formats of the `gguf_blocks_v1` layout
([tensor formats](maintainer/tensor-formats.md#35-gguf-block-formats),
[storage layouts](maintainer/storage-layouts.md#6-gguf_blocks_v1)). A projection may mix types: the
parts of a fused projection that share a type are multiplied together, the others one by one against
the same activation.

## Products

Every product quantizes its BF16 activation to ggml's `q8_1` numbers, one scale and one sum per 32
values, as llama.cpp does, and accumulates in FP32; the weights' represented values are exactly those
of `ggml-quants.c`.

- **Up to eight columns** (decode, speculative verification, small batches), the vector kernel
  (`src/ops/linear/gguf/ggml_bridge_vec.cuh`) decodes each 32-value slice of a row once into int8
  words and dots it with every column. A warp owns two rows; a `[gate; up]` pair of one type is one
  launch that writes `silu(gate) * up`.
- **More columns** (prompts) take llama.cpp's integer tensor-core kernel (`mul_mat_q`, vendored
  unmodified in `third_party/ggml-quants` under its MIT license) with its Ampere tile table.
  `IQ1_M` has no such kernel; its wide products dequantize to BF16 and run cuBLAS.
- **Token table** rows are dequantized exactly.

The scales of `IQ2_XXS`, `IQ2_XS`, `IQ2_S` and `IQ3_XXS` are applied in FP32 where llama.cpp's
vector kernels round an integer rescale, so a decode step's logits can differ from llama.cpp's in the
last bits; prompts use llama.cpp's own arithmetic.

The GGUF head of the reduced proposal table materializes BF16 logits per chunk and ranks them through
the same grouped K-split reduction the fused producers use.

## Tensor parallelism

A converted artifact also runs on the dual-GPU tensor-parallel route (`ninfer-serve --devices
0,1`). The split follows the block layout instead of a repacked format:

- An output-row split (the vocabulary head, and a `gate`/`up` parent stored as one object) keeps
  whole rows, so each shard stores contiguous code-plane rows.
- A K split (the FFN `down` projection) keeps whole 256-element blocks, so each shard holds valid
  blocks and the partial products are reduced as usual.
- The release stores a projection whose parts carry different types as separate objects rather
  than one parent (`[q|k]` or `[q|k|v]` plus a `z` object, four separate attention blocks, two
  separate `gate`/`up` objects). The split spec reads the logical blocks an object carries from
  its binding names, halves each of them, and concatenates the halves, and `ops::gdn_input_proj`
  derives its channel profile from the parts instead of the full model's.
- The reduced proposal head of speculative decode is rank-split like any other output-row split,
  and `linear_topk` serves both of its geometries from the one GGUF profile: the whole table and
  the half one shard materializes.
- Some layers of the GSQ-RCO release store their grouped value projection in tiled value-head
  order and describe that order with an `input_columns` column map whose entries address heads of
  both shards. That projection therefore runs in two stages: each shard leaves its own heads in
  its half of the activation, the pair all-reduces the disjoint halves, and both shards then
  project the assembled activation with their half of the K.

The integer tensor-core kernel distributes a matrix's K dimension over its grid with stream-K,
and that partition follows the grid shape, so one output column's rounding depends on how many
columns the call carries. A chunked prefill consequently does not reproduce a single-chunk
prefill bit for bit on this route, and a near-tie token can resolve differently;
[`ninfer_qwen3_5_tp2_forward_test`](../tests/models/qwen3_5/test_tp2_forward.cpp) holds a
block-quantized artifact to the sampled token plus the leading-logit bound, and keeps its
byte-exact criterion for the other formats.

## Serving

A converted artifact starts like any other; MTP, DFlash2 and Vision work as with the official
artifact. Text, token table, output head and MTP head keep their GGUF blocks; Vision and DFlash2
use the official formats, so a draft converted here carries the same Q4_G64_FP16 projections and
selector codebooks as one converted by `qwen3_8_27b`:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_gsq_rco_iq3_s.ninfer \
  --model-id qwen3.8-27b --kv-dtype k8v4 --spec mtp --draft-tokens 3
```

The converter recipe, the artifact and the products above are qualified by
[`tests/ops/linear/test_gguf.cpp`](../tests/ops/linear/test_gguf.cpp), which compares every block
type against the FP64 product of its exactly dequantized weights.
