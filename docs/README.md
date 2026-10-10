# NInfer documentation

Start with the [project README](../README.md) to build NInfer, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, and tool calls |
| [Performance](performance.md) | RTX 5090 measurement coverage, per-model serving results, methodology, and publication rules |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [GGUF block formats](gguf.md) | GGUF releases with a ggml type per tensor: the fifteen block formats, their products and serving |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [Windows native build](windows.md) | VS2022 + CUDA 13.3 build, native run recipe, verified Windows/WDDM behavior, TP-2 mapped-pinned findings, and platform differences |
| [Dual RTX 5060 Ti TP-2](tp2-dual-5060ti.md) | tensor-parallel-2 server configuration, memory budget, and measurements on two RTX 5060 Ti cards |
| [TP-2 settled decisions](tp2-decisions.md) | quick reference of settled decisions, rejected approaches, and root causes from the TP-2 campaign, with worklog pointers |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

[Engine architecture](maintainer/engine-architecture.md) is the single top-level reference. The
other references own narrower contracts:

| Document | Responsibility |
|---|---|
| [Engine architecture](maintainer/engine-architecture.md) | model/config/weight ownership, loading-to-execution flow, requests, scheduling, transactions and graphs |
| [Build system](maintainer/build-system.md) | CMake targets, explicit source ownership, CUDA compilation boundaries, presets and developer configuration |
| [Artifact container](maintainer/artifact-container.md) | v3 directory, objects, logical bindings, Uses, resources and file framing/sharding |
| [Numeric formats](maintainer/tensor-formats.md) | represented values, codes/scales, conversion arithmetic and numerical interpretation |
| [Storage layouts](maintainer/storage-layouts.md) | packing, plane offsets, padding, encoded sizes and view addressing |
| [Qwen3.5 model](maintainer/qwen3_5-model.md) | Dense/MoE mathematics, instance config, logical parameters, MTP, Vision and state semantics |
| [DFlash and DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | candidate selection, retention, materialization and Device/Host checkpoint policy |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Linear benchmark](maintainer/linear-benchmark.md) | pure Linear measurement, metrics and suites |
| [Linear tuning and reports](maintainer/linear-tuning.md) | tuning ranges, priority points, dispatch tradeoffs and final performance report format |

### TP-2 campaign records

The 2x RTX 5060 Ti TP-2 fork added the records below. They are the planning authority and campaign
history for that route, not contracts for the single-GPU product: the references above still own
the shared machinery.

| Document | Responsibility |
|---|---|
| [TP-2 plan](maintainer/tp2-plan.md) | remaining work, closed items, environment notes, and the architecture-level risks still open |
| [TP-2 design](maintainer/tp2-design.md) | early TP-2 execution design: the shard split table and execution flow |
| [TP-2 concurrency plan](maintainer/tp2-plan-concurrency.md) | multi-lane concurrency plan: batching, admission, and the interfaces it introduced |
| [TP-2 concurrency review remediation](maintainer/tp2-plan-concurrency-review-remediation.md) | review findings on the multi-lane work and their remediation |
| [TP-2 KV sharing plan](maintainer/tp2-plan-kv-sharing.md) | shared KV pool semantics behind multi-lane admission |
| [TP-2 headroom execution plan](maintainer/tp2-plan-headroom-execution.md) | execution steps of the concurrency headroom campaign |
| [TP-2 concurrency headroom](maintainer/tp2-concurrency-headroom.md) | headroom measurements and what remains |
| [TP-2 GGUF plan](maintainer/tp2-plan-gguf.md) | GGUF block formats on the TP-2 route |
| [TP-2 fix plan](maintainer/tp2-plan-fixes.md) | review defect fixes and their verification record |
| [TP-2 DFlash2 draft NVFP4 feasibility](maintainer/tp2-dflash2-draft-nvfp4-feasibility.md) | why an all-draft NVFP4 artifact was rejected |
| [llama.cpp TP notes](maintainer/tp2-llamacpp-notes.md) | comparison notes against a local llama.cpp TP build |
| [TP-2 adaptation worklog](maintainer/tp2-adaptation-worklog.md) | chronological round-by-round record, including five verbatim plan snapshots |

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.
