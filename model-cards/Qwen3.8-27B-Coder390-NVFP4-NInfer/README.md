---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: finetune
tags:
  - ninfer
  - qwen3.8
  - nvfp4
  - w4a4
  - w8a8
  - blackwell
  - multimodal
  - mtp
  - dflash2
  - community
---

# Qwen3.8-27B Coder390 (community merge) - NInfer NVFP4 W4A4-W8A8

This package is published by [nerkyor](https://huggingface.co/nerkyor), not by this fork, and its
[model card](https://huggingface.co/nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-NVFP4-NInfer)
is the authority for the model itself: training lineage, evaluations, every quantization tier and
every runtime. This directory is the record this fork keeps of the one package it verifies and
recommends, so that its identity, provenance and the way to run it on the dual RTX 5060 Ti TP-2
route are versioned together with the engine that runs it.

## Package

Paths are relative to the source repository; the package lives under `NVFP4-NInfer/W4A4-W8A8/`.

| File | Bytes | SHA256 |
|---|---:|---|
| `Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-MTP-DFlash2.ninfer` | 23,773,675,264 | `cccfbf0ba3ec4e86837e8e6481ec32ee3e10636633e421429f19ea3ea97926e2` |
| `vision-bf16.safetensors` | 921,497,224 | `d7defc90994f7bbc58a8feb8d3c4144895babb434a9be9f47e88ae3646e1df33` |
| `Qwen3.8-27B-Coder390-W4A4-W8A8-NInferV2-NOT-FOR-V3.ninfer` | 20,752,831,236 | `01a4440f77dcdac66906e1bb061e1c38c6b236b805ff7bcb644aedf90f06a038` |
| `Qwen3.8-27B-Coder390-W4A4-W8A8-MTP-DFlash2-NInferV2-NOT-FOR-V3.ninfer` | 23,477,696,516 | `099b2d00fa3982d601218d6587dc25222cc21ef708de7dab8b2e2c0bee60da74` |

Only the first row is needed to serve the model. `vision-bf16.safetensors` is the BF16 source of the
vision component and is not read at run time. The `NInferV2-NOT-FOR-V3` files use the V2 container
and exist for the Tesla V100 `ninfer-v100` branch: this engine reads V3 only and rejects them, so do
not rename them into V3 file names.

The package is a single V3 `.ninfer` carrying text, vision, a Q8 MTP head, the DFlash2 draft and the
proposal head, so no external draft model is needed. The vision component was added on 2026-10-09:
333 tensors, 295.7 MB of quantized weights, described by the publisher as patch embedding Q6,
attention q/k/v and MLP fc1 Q4, remaining projections Q5, merger Q8, norms and biases BF16. The
text, MTP and DFlash2 weights are unchanged by that addition. Pass `--vision` to accept images;
without it the package runs text-only.

`artifact-manifest.json` records the package's identity, provenance and inventory, read from the
package's own directory rather than from the publisher's card. Its internal metadata name is
`qwen3.8-27b-coder390-w4a4-w8a8-mtp-dflash2`, the container reports artifact id
`0d236ca0a8c5482d91d85306e5c691cb`, and the publisher's own commands pass
`--model-id qwen3.8-27b-coder390-w4a4-w8a8`, which overrides that name in `/v1/models`.

## Download and verify

```bash
hf download nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-NVFP4-NInfer \
  --include 'NVFP4-NInfer/W4A4-W8A8/Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-MTP-DFlash2.ninfer'
```

`SHA256SUMS` in this directory is the publisher's own file, copied verbatim; its entries are bare
file names, so run it inside the directory that holds the download.

## Run it

The publisher's command for this package, on one card, with the `--model-id` recommended there:

```bash
ninfer-serve Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-MTP-DFlash2.ninfer \
  --host 127.0.0.1 --port 19931 --device 0 \
  --model-id qwen3.8-27b-coder390-w4a4-w8a8 \
  --max-context 102400 --kv-capacity 819200 --max-concurrency 8 \
  --kv-dtype fp8 \
  --spec dflash2 --draft-tokens 8 --lm-head-draft
```

On the pair this fork targets, both cards serve one model, and the wrapper script supplies the TP-2
flags:

```powershell
$env:NINFER_MODEL = "<models-dir>\Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-MTP-DFlash2.ninfer"
pwsh -File tools/win_port/serve.ps1 -Spec dflash2 -DraftTokens 7
```

`--lm-head-draft` and `--vision` are on by default there, and `-Plain` turns off speculation and
vision. MTP accepts 1-5 draft tokens and DFlash2 1-15, and the two are mutually exclusive per
launch. The 27B weights do not fit one 16 GiB card, so the TP-2 route splits them; the measurements
below and [README.md](../../README.md#performance) cover this pair, including the per-card memory
ledger.

## Measured on this pair

Measured on 2026-10-11 with [`tools/win_port/bench_serve.ps1`](../../tools/win_port/bench_serve.ps1)
against the local equivalent of this package -
`Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-Vision.ninfer`, which is these weights plus the
vision component rebuilt with this fork's recipe - on two RTX 5060 Ti cards (`--devices 0,1`) at
`--max-context 225280 --kv-dtype fp8`, one request at a time. Prefill is a one-token request over a
filler prompt, decode is a 128-token answer after a 7,200-token prompt, every number comes from the
server's own `timings` block, and every prompt carries a nonce, so prefix reuse cannot fake prefill.
Each arm ran four times.

| Route | Prefill 1.9k | 7.2k | 28.6k | Decode | vs plain |
|---|---:|---:|---:|---:|---:|
| plain, no draft | 1,796 | 1,943 | 1,772 | 35.9 | 1.0x |
| MTP, K=3 | 1,797 | 1,917 | 1,753 | 71-79 | ~2.1x |
| DFlash2, K=5 | 1,618 | 1,629 | 1,557 | 81-96 | ~2.5x |
| DFlash2, K=7 | 1,595 | 1,626 | 1,556 | 87-90 | ~2.5x |

All figures are tok/s. Prefill cells are the mean of the runs (four for the 1.9k column, two for the
others) and spread by under 2%; decode cells are the range over the runs that produced the full 128
tokens, which moves with the accepted-draft rate rather than with measurement noise. The draft buys
decode with prefill: DFlash2's own forward pass costs about 15% of prefill throughput at every
prompt size and MTP's about 1%, and DFlash2 K=5 and K=7 are within each other's spread. The
published file stores the same weights with a Q8 DFlash2 draft instead, 0.946 GiB larger, so these
numbers belong to the rebuild named above. The per-card memory ledger, the multi-request matrices
and the tuning record are in [README.md](../../README.md#performance) and the
[TP-2 note](../../docs/tp2-dual-5060ti.md).

## Publisher measurements

One RTX PRO 6000 Blackwell, 2026-10-09, NInfer `64492cb`, `--vision`, image prompt, one request at a
time:

| Route | Accept length | Decode |
|---|---:|---:|
| MTP, 3 draft tokens | 3.19 | 164 tok/s |
| DFlash2, 7 draft tokens | 3.74 | 195 tok/s |

Those are the publisher's numbers, on one card and one engine revision; this fork's dual-card
numbers are above. Every tensor the package binds under `dflash2/` is BF16 (the norms and the two
selector codebooks) or Q8 (the draft projections). Rebuilding the same weights with this fork's
converter applies the shipped recipe, which assigns Q4 to those projections and saves about
0.946 GiB per copy ([DFlash2 draft weights](../../README.md#dflash2-draft-weights)).

## License and attribution

Apache-2.0, as declared by the source repository; the model is a post-training of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). The weights are published by
nerkyor and are not redistributed here: this directory holds only this card, its license, the
checksums and the manifest. The engine is [Neroued/ninfer](https://github.com/Neroued/ninfer); this
fork adds the dual-card TP-2 route.
