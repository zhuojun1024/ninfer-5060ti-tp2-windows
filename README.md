# NInfer

**English** | [简体中文](README.zh-CN.md)

> Dual RTX 5060 Ti tensor-parallel-2 serving, natively on Windows.

This repository is the dual-GPU fork of [Neroued/ninfer](https://github.com/Neroued/ninfer). It
serves Qwen3.8-27B NVFP4 across two RTX 5060 Ti (16 GiB) cards with a tensor-parallel-2 (TP-2) route
and builds natively on Windows; the same sources build and run on Linux and WSL2. Upstream NInfer is
a from-scratch C++/CUDA inference engine for a single RTX 5090, and that single-GPU route remains
upstream's.

The server runs text, image, and video prompts through OpenAI- and Anthropic-compatible HTTP APIs.
The runtime is deliberately specialized: one TP-2 pair, one resident model, a startup-fixed capacity
of one to eight active requests, and one compact decode round that batches up to four queued
requests. The CLI is single-GPU, so on two 16 GiB cards it cannot hold this model at all (the 27B
NVFP4 weights need about 20.2 GiB resident); the server route is the one this fork uses.

## Hardware and artifacts

The verified configuration is two identical RTX 5060 Ti 16 GiB cards (compute capability 12.0,
`sm_120a`) in one machine, on Windows 11 with Visual Studio 2022 (MSVC 19.44) and CUDA 13.3. The
pair is what makes the model fit: the 27B NVFP4 artifact needs about 20.2 GiB of resident weights,
10.1 GiB per TP-2 shard. The build rejects CUDA architectures other than `sm_120a`, and the TP-2
core fails fast when the two devices are not identical.

Six artifacts are listed below: five official upstream artifacts and one community merge. The
recipes below use Qwen3.8-27B NVFP4, the artifact verified on this pair; another artifact fits the
same pair only if one TP-2 shard fits in 16 GiB.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.8-27B Coder390 (community merge) | `nvfp4` W4A4-W8A8 | `Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-MTP-DFlash2.ninfer` | [Qwen3.8-27B-Coder390 NVFP4](https://huggingface.co/nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-NVFP4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-27B † | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B † | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B † | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

† Not tested on the dual RTX 5060 Ti pair; the Qwen3.6 series artifacts are listed for
completeness.

Each v3 `.ninfer` artifact carries model configuration, encoded weights, logical bindings and
frontend resources. Runtime execution uses those facts with the implemented model and Op
capabilities. You can also [convert your own weights](docs/weight-conversion.md), reuse an official
recipe or choose another supported mixture of formats.

The current engine requires v3 artifacts. Existing official v2 downloads can be [upgraded
locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact) without downloading the weights
again.

## Quick start (native Windows)

Prerequisites:

- Visual Studio 2022 with the C++ desktop workload (`vcvars64.bat`);
- CUDA toolkit 13.3 (`nvcc`), with CMake 3.28 or newer and Ninja on `PATH`;
- an FFmpeg **development** prefix (headers plus import libraries) for media decoding, and a libcurl
  prefix for remote media acquisition. `tools/win_port/fetch_ffmpeg.ps1` and
  `tools/win_port/fetch_curl.ps1` download both, and each prints the prefix and the environment
  variable that makes it the default.

The [Windows native build](docs/windows.md) guide has the full prerequisite list, the platform notes
carried by the port, and the test commands.

Build the product binaries:

```powershell
git clone https://github.com/zhuojun1024/ninfer-5060ti-tp2-windows.git
cd ninfer-5060ti-tp2-windows

pwsh -File tools/win_port/build.ps1 -Configure     # Release, sm_120a -> build-win/apps/*.exe
pwsh -File tools/win_port/build.ps1                # incremental build
```

`build.ps1` imports the `vcvars64` environment itself, so no cmd wrapper is needed. `-Apps 0` builds
the core libraries only, and `-Tests 1 -Benchmarks 1` add the suite and the benchmarks, which a
plain `-Configure` turns off. There is no install target or packaged binary distribution; run NInfer
from its build tree. Python tools run independently of CMake; the standalone HBM probe has its own
[build command](tools/README.md#standalone-hbm-probe).

Download the artifact and verify it:

```powershell
pip install -U huggingface_hub     # provides the `hf` CLI
hf download neroued/Qwen3.8-27B-nvfp4-NInfer --local-dir models
(Get-FileHash models\qwen3_8_27b_nvfp4.ninfer -Algorithm SHA256).Hash
# 74D2C57145E6FF11D1D2FAA79594477F9BC903A611AF1FB20218189FBBB77D82
```

The repository is public and ungated. The artifact is 23.7 GB, and the same download brings its
`SHA256SUMS`, `artifact-manifest.json`, and model card; pass the file name instead of the whole
repository to fetch only the artifact.

Start the TP-2 server:

```powershell
$env:NINFER_MODEL        = "models\qwen3_8_27b_nvfp4.ninfer"
$env:NINFER_FFMPEG_ROOT  = "D:/ffmpeg-dev/expanded/ffmpeg-master-latest-win64-gpl-shared"
$env:NINFER_LIBCURL_ROOT = "C:/curl"     # fetch_curl.ps1 prints the prefix it unpacked

pwsh -File tools/win_port/serve.ps1            # 127.0.0.1:8099, 131072-token KV, MTP K=2, vision
pwsh -File tools/win_port/serve.ps1 -Status    # processes, health, VRAM
pwsh -File tools/win_port/serve.ps1 -Stop      # stop every ninfer-serve process
```

Keep that terminal open: a `-Background` child belongs to the launching shell's job object and dies
with it. The script puts the FFmpeg and libcurl `bin` directories on `PATH`, because that is where
the loading DLLs live, and it refuses to start when a server is already running. `-MaxConcurrency 4`
batches up to four queued requests, `-Context 262144` raises the KV capacity, `-Plain` drops MTP and
vision, `-Spec dflash2 -DraftTokens 7` selects the masked-draft route, and `-HostKvMiB 0` disables
cross-session KV retention. `-Model` (or `NINFER_MODEL`) selects another artifact.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

### The same server without the wrapper script

```powershell
build-win\apps\ninfer-serve.exe models\qwen3_8_27b_nvfp4.ninfer `
  --devices 0,1 `
  --max-context 262144 `
  --kv-capacity auto `
  --kv-dtype fp8 `
  --spec mtp --draft-tokens 2 --lm-head-draft `
  --temperature 0.7 --top-k 20 --top-p 0.80
```

`--devices 0,1` are CUDA runtime indices, not the `nvidia-smi` PCI-bus order; a wrong pair reports
the CUDA-visible device list. Executable `--help` is the current option authority and [HTTP
serving](docs/serving.md) documents the request contract.

### Linux and WSL2

The same product build with the Linux toolchain: CUDA 13.1 is the validated development toolkit
there, plus `pkg-config`, the FFmpeg development libraries (`libavformat`, `libavcodec`,
`libavutil`, `libswscale`) and `libcurl >= 7.85`.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --devices 0,1 --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 \
  --spec mtp --draft-tokens 2 --lm-head-draft
```

`cmake --preset release` configures the same product build; `cmake --preset dev` also enables tests
and benchmarks and finds a Python 3 interpreter. Both presets use `build/` and explicitly reset the
build options, and machine-specific compiler and Python paths belong in the ignored
`CMakeUserPresets.json`. See [build organization and
configuration](docs/maintainer/build-system.md).

## Memory and configuration

`--max-context` is each sequence's logical ceiling, and `--kv-capacity` sizes the shared Main Text
KV pool that active requests and retained prefixes draw from; `auto` resolves the largest legal
capacity at startup from the memory left after weights, keeping 1 GiB of sizing headroom. The TP-2
lanes share that one pool dynamically: a request is admitted once the pages its own prompt plus
output budget need are free, so a lane that runs alone may use the whole ceiling instead of a fixed
1/lanes slice, and only a prompt longer than the pool's admission ceiling is rejected as
`context_length_exceeded`.

Verified headroom on the two 16 GiB cards with fp8 KV:

| Configuration | KV | Free after weights |
|---|---:|---|
| `--max-context 262144`, no vision | 4,128 MiB | 697 MiB (shard 0) / 1,217 MiB (shard 1) |
| `--max-context 131072`, `--vision` | 2,322 MiB | 2,002 MiB (shard 0) / 1,580 MiB (shard 1) |
| `--max-context 8192`, `--vision` | 145 MiB | 4,180 MiB (shard 0) / 3,930 MiB (shard 1) |

fp8 KV costs 16.125 KiB per token per shard. The 262,144-token configuration reaches the artifact's
full `max_position_embeddings` and is what the [TP-2 note](docs/tp2-dual-5060ti.md) recommends; the
shipped Windows `serve.ps1` recipe uses 131,072 with `--vision`, because the vision workspace on top
of a 262,144-token KV pool leaves too little VRAM for comfort under WDDM.

### DFlash2 draft weights

A DFlash2 artifact carries a second, masked draft model next to the target weights. Every shipped
conversion recipe stores that draft's projections in `q4_g64_fp16` and its selector codebooks in
BF16, so the draft occupies 1,155.3 MiB instead of the 2,123.6 MiB an all-Q8 draft takes. Two
conversions of the same model that differ only in the draft objects are 968.3 MiB (0.946 GiB) apart.

Those bytes are freed on the shard that carries the masked draft, which is the shard that bounds
context capacity, and it is the heavier of the two, so the pair ends up more even and the saving
converts straight into KV: at 16.125 KiB per token, 968.3 MiB is about 61,000 more fp8 tokens there.
An artifact whose `dflash2/` bindings still name `q8_g32_fp16` objects carries the older draft and
pays that memory for nothing; re-converting with a shipped recipe reclaims it. No flag or extra
script is involved, because the format set is fixed by `tools/convert/official_recipes.py`:

- Q4: the fused attention QKV parent (one shared `[6144, 5120]` object per layer, which
  `context_key`/`context_value` are views of), attention output, MLP gate/up/down, both dynamic-conv
  kernel projections and `dflash2/feature_projection` — 31 stored objects, 909.8 MiB;
- BF16: the two selector codebooks (242.5 MiB) and the norms;
- Q8: DFlash v1 and MTP drafts.

The set is the campaign's qualified result rather than a per-recipe choice. Each step was A/B-tested
on this pair; the merged artifact measured +1.15pp at K=7 and +3.72pp at K=5 over the draft it
replaced on pooled sampling acceptance, and the promotion into the recipes landed within 1σ over 30
repetitions. A greedy probe cannot compare two artifacts at all, because a changed draft changes the
proposed text and therefore the target's near-ties.

Convert and check:

```bash
python3 -m tools.convert --model /path/to/NVFP4/W4A4-W8A8 --recipe qwen3_8_27b \
  --source dflash2=/path/to/NVFP4/W4A4-W8A8/DFlash2-FP8 --components text,vision,mtp,dflash2 \
  --proposal --out models/my_qwen.ninfer
python3 -m tools.artifact.inspect models/my_qwen.ninfer --objects --bindings
```

Quantizing the selector codebooks as well saves a further 178.09 MiB, but it measured about 2pp less
acceptance on a 33k-token prompt at K=7, and those bytes land on the selector's shard rather than
the masked draft's, so they buy no context. The shipped recipes keep the codebooks in BF16.

### Quantizing a DFlash2 draft yourself

The same lever is available as an explicit override, `tools/convert/overrides/dflash2_draft_q4.py`,
for a conversion whose base recipe does not already select it: an artifact you rebuild from your own
recipe, or one you want to compare against an all-Q8 draft. Start with a plan that writes nothing:

```bash
python3 -m tools.convert --model /path/to/NVFP4/W4A4-W8A8 --recipe my_recipe.py \
  --source dflash2=/path/to/NVFP4/W4A4-W8A8/DFlash2-FP8 --components text,vision,mtp,dflash2 \
  --override tools/convert/overrides/dflash2_draft_q4.py --dry-run
```

`--dry-run` applies the base recipe, the proposal head and the override, prints every selection as
`name [begin,end) format method`, and writes nothing; `--out` is required only without it. Drop the
flag and add `--out models/my_qwen.ninfer` to convert. The default entry function `configure`
assigns Q4 to the draft projections and leaves every other parameter at whatever the base recipe
chose; the codebook lever is a second entry point:

```bash
  --override tools/convert/overrides/dflash2_draft_q4.py:configure_with_codebooks
```

The override fails loudly when the model has no `dflash2/feature_projection`, has no draft layers,
or has a layer missing one of the nine projections it quantizes, so a partial conversion cannot pass
unnoticed. The general mechanism is
[changing part of a recipe](docs/weight-conversion.md#change-part-of-a-recipe).

## Performance

Every number below was measured on this pair (`--devices 0,1`) with `ninfer-serve`, one request at a
time unless a concurrency is named. The [Dual RTX 5060 Ti TP-2](docs/tp2-dual-5060ti.md) note holds
the full matrices, the memory ledgers and the tuning record; the [performance
index](docs/performance.md) holds upstream's single-GPU measurements on an RTX 5090.

### Single-request serving

Measured on the 2026-10-11 build with
[`tools/win_port/bench_serve.ps1`](tools/win_port/bench_serve.ps1) against
`Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-Vision.ninfer` (the Coder390 artifact above with its
vision component) at `--max-context 225280 --kv-dtype fp8`, one request at a time. Prefill is a
one-token request over a filler prompt; decode is a 128-token answer after a 7,200-token prompt.
Every number comes from the server's own `timings` block, and every prompt carries a nonce, so
prefix reuse cannot fake prefill. Each arm ran four times.

| Route | Prefill 1.9k | 7.2k | 28.6k | Decode | vs plain |
|---|---:|---:|---:|---:|---:|
| plain, no draft (`-Plain`) | 1,796 | 1,943 | 1,772 | 35.9 | 1.0x |
| MTP K=3 `--lm-head-draft` | 1,797 | 1,917 | 1,753 | 71-79 | ~2.1x |
| DFlash2 K=5 `--lm-head-draft` | 1,618 | 1,629 | 1,557 | 81-96 | ~2.5x |
| DFlash2 K=7 `--lm-head-draft` | 1,595 | 1,626 | 1,556 | 87-90 | ~2.5x |

All figures are tok/s. Prefill cells are the mean of the runs (four for the 1.9k column, two for the
others) and spread by under 2%; the decode cells are the range over the runs that produced the full
128 tokens. That range is the accepted-draft rate moving with the text, not measurement noise:
answers from a short prompt measured 66-84 (MTP K=3), 84-104 (DFlash2 K=5) and 85-97 (DFlash2 K=7)
tok/s over outputs of 43-181 tokens. The draft buys decode with prefill - DFlash2's own forward pass
costs about 15% of prefill throughput at every prompt size, MTP's about 1% - and DFlash2 K=5 and K=7
are within each other's spread.

Memory at this capacity, `nvidia-smi` per card: 13,688 MiB for the plain arm (`-Plain` also drops
vision and the sampling overrides), 14,820 / 14,638 MiB for MTP K=3, 14,954 / 14,884 MiB for DFlash2
K=5 and 14,960 / 14,886 MiB for K=7. Engine load is a one-off 40 s (plain) to 44 s (DFlash2) and is
outside these numbers. Earlier figures taken under other protocols - the 16,057-token TTFT, the
prefix-reuse latency and the llama.cpp cross-engine comparison - are in the [TP-2
note](docs/tp2-dual-5060ti.md), which keeps every measurement next to the protocol it was taken
under.

### Multi-request serving

`--max-concurrency` (1..4 on this route) is how many queued requests the core batches into one
decode round; the default is 1. A lone request is admitted after a 3 ms formation window, and a
burst that arrives inside that window shares one round, because the window stays open until the
round is full or 50 ms have passed. A request that arrives while a round is running joins a later
round of the same batch as soon as a lane retires.

| Route | Aggregate decode at C=2 | at C=4 |
|---|---:|---:|
| plain, int8 KV, 131,072 | 1.76-1.91x | 2.80x |
| `--spec dflash2` | — | 2.07x |
| `--spec mtp --draft-tokens 5` | 1.48x | 2.26x |

A round costs as much as its longest member, so the aggregate gain depends on how evenly the batch
members finish: a pair whose outputs differ two-fold measured 1.41x. The round itself is cheap to
widen — four equal-length lanes measured 26.5 ms at one lane and 32.0 ms at four, i.e. 3.31x of the
ideal 4x, with a per-column marginal of about 1.85 ms (7% of a one-lane round). Four lanes at
`--max-context 262144 --kv-dtype fp8` with `--spec mtp --draft-tokens 2 --lm-head-draft` all
completed with 829 MiB (shard 0) and 1,789 MiB (shard 1) free, and each lane costs about 147 MiB of
linear-attention state per card.

A lane that is prefilling does not stall the lanes beside it: after each prefill chunk the core runs
one decode round for the lanes that already hold a token, so a 54-token request admitted 400 ms
before a 15,310-token one finished in 2.8 s at 8.5 tok/s with `--prefill-chunk 256` (7.5 s and 3.1
tok/s at the default 1024).

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations. On the TP-2 route the checkpoint ring and the session
catalog are sliced by lane, so device-resident reuse depends on which lane a request lands on and a
recall onto a different lane falls back to the host-resident session.

See [Resource scheduling and context
cache](docs/maintainer/resource-scheduling-and-context-cache.md) for the algorithm and [Serve TTFT
benchmark](tools/bench/ttft/) for public-HTTP coverage of hot reuse, Host resume, eviction, shared
prefixes, scheduling boundaries, and multimodal load.

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. These are model-capability scores from the upstream single-GPU run; the TP-2 pair serves the
same artifacts and reaches the artifact's full 262,144-token ceiling. Each score is one sample per
problem; model cards contain the correct/total counts and evaluation notes.

## Docker

The image builds the same sources, so the TP-2 route runs inside it, but the two-card container path
is not verified here. Build on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Pass both devices and the same TP-2 profile:

```bash
docker run --rm \
  --gpus '"device=0,1"' \
  --publish 8099:8099 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --port 8099 \
  --devices 0,1 \
  --max-context 262144 \
  --kv-capacity auto \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 2 --lm-head-draft \
  --temperature 0.7 --top-k 20 --top-p 0.80
```

## Capabilities and limits

The official artifacts provide the following capabilities, with optional components enabled at
startup:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8, FP8, NVFP4, and K8V4 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

An artifact converted with `qwen3_8_27b_gguf` keeps every tensor in its GGUF release's own ggml
block type instead; see [GGUF block formats](docs/gguf.md).

The 35B-A3B target additionally supports DFlash with draft windows from one to fifteen for Text and
image/video Vision prompts. Qwen3.8-27B artifacts with the DFlash2 companion weights support `--spec
dflash2 --draft-tokens 7` for the same Text/Vision Engine path, with draft counts 1..15 and either
full or optimized proposal heads.

The product boundary remains intentionally small:

- two identical RTX 5060 Ti 16 GiB cards on the TP-2 server route (or one RTX 5090 on upstream's
  single-GPU route) and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress (the TP-2 pair
  batches up to four queued requests on its plain, DFlash2 and MTP rounds and runs one at a time
  otherwise);
- no request preemption, priority/QoS, active-request swapping, weight offload, or distributed
  serving; multi-GPU is limited to the two-card TP-2 server route;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- model architectures and format/shape combinations use explicitly implemented native paths;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

## Startup notes

GPU residency is fixed at process startup, so `--devices 0,1`, `--spec` and `--vision` are all
chosen once and held for the process lifetime. Qwen3.6-35B-A3B DFlash can be combined with Vision;
it accelerates generated-text decode after multimodal prefill, not Vision encode itself.

## Documentation

- [Documentation index](docs/README.md)
- [Windows native build](docs/windows.md)
- [Dual RTX 5060 Ti TP-2](docs/tp2-dual-5060ti.md)
- [Settled TP-2 decisions](docs/tp2-decisions.md)
- [HTTP serving](docs/serving.md)
- [CLI](docs/cli.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Weight conversion and custom recipes](docs/weight-conversion.md)
- [GGUF block formats](docs/gguf.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## Relationship to upstream

This repository is a fork of [Neroued/ninfer](https://github.com/Neroued/ninfer). It keeps the
upstream engine, the `.ninfer` artifact format, the converter recipes, the model cards and the
published artifacts, and adds the dual-GPU tensor-parallel-2 (TP-2) route for two 16 GiB cards plus
a native Windows build. Upstream is the authority for the engine itself; the TP-2 additions, their
measurements and the full adaptation record are documented in [Dual RTX 5060 Ti TP-2
tuning](docs/tp2-dual-5060ti.md), [Settled TP-2 decisions](docs/tp2-decisions.md) and the [TP-2
adaptation worklog](docs/maintainer/tp2-adaptation-worklog.md).

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
