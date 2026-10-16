# Dual RTX 5060 Ti TP-2 tuning (Qwen3.8-27B NVFP4)

This note records the tensor-parallel-2 (TP-2) adaptation of this engine for two RTX 5060 Ti
(16 GiB) cards running `Qwen3.8-27B` NVFP4. Every number was measured on this machine with
`ninfer-serve` on `--devices 0,1`, one request at a time unless a concurrency is named.
[Current measurement](#current-measurement) is the reading on the current build; the sections after
it are the tuning record, and each table keeps the protocol and the artifact it was measured under.
The retained measurement routines live in
[`tools/tp_bootstrap/`](../tools/tp_bootstrap/README.md).

## Recommended configuration

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2 --lm-head-draft
```

The full `max_position_embeddings` of the artifact (262,144 tokens) now fits: 15,614 MiB on shard 0
and 15,094 MiB on shard 1, i.e. 697 / 1,217 MiB still free per card (`nvidia-smi`). Splitting the
vocabulary- and hidden-parallel shard weights is what closed the gap; with the reduced state and
workspace pools alone, fp8 KV topped out near 229,376 tokens. `--kv-dtype fp8` costs 16.125 KiB per
token per shard. `--spec mtp --draft-tokens 2` is the speculative route that still fits the full
context and adds sustained decode over the plain route ([Current
measurement](#current-measurement) has the current four-route table) and is
verified non-degenerate (see the MTP section); `--lm-head-draft` selects the artifact's reduced
proposal head for the draft side; drop those two flags for the plain route. Every one of those
choices is numerically transparent: the greedy output at 262,144 is byte-identical to the pre-split
build at 131,072 (see Verification).

## Current measurement

The four serving routes on the current build, measured 2026-10-11 with
[`tools/win_port/bench_serve.ps1`](../tools/win_port/bench_serve.ps1) against
`Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-Vision.ninfer`, started through
[`tools/win_port/serve.ps1`](../tools/win_port/serve.ps1) with `--max-context 225280 --kv-dtype fp8
--vision-item-tokens 8192 --host-kv-mib 32768`, one request at a time. Prefill is a one-token
request over a filler prompt (1.9k / 7.2k / 28.6k prompt tokens); decode is a 128-token answer
after a 7.2k-token prompt. Every number is the server's own `timings` block, and every prompt
carries a nonce, so prefix reuse cannot fake prefill. Each arm ran four times.

| Route | Prefill 1.9k | 7.2k | 28.6k | Decode | VRAM shard 0 / 1 | Load |
|---|---:|---:|---:|---:|---|---:|
| plain, no draft (`-Plain`) | 1,796 | 1,943 | 1,772 | 35.9 | 13,688 MiB | 40-41 s |
| MTP K=3 `--lm-head-draft` | 1,797 | 1,917 | 1,753 | 70.6-79.1 | 14,820/14,638 MiB | 43 s |
| DFlash2 K=5 `--lm-head-draft` | 1,618 | 1,629 | 1,557 | 80.8-95.6 | 14,954/14,884 MiB | 43 s |
| DFlash2 K=7 `--lm-head-draft` | 1,595 | 1,626 | 1,556 | 87.2-90.1 | 14,960/14,886 MiB | 43-44 s |

Prefill and decode are tok/s; the prefill cells are the mean of the runs - four for the 1.9k column,
two for the others - which spread by under 2%, and the decode cells are the range over the runs that
produced the full 128 tokens. The spread
inside an arm is the accepted-draft rate following the text: answers from a short prompt measured
66-84 (MTP K=3), 84-104 (DFlash2 K=5) and 85-97 (DFlash2 K=7) tok/s over outputs of 43-181 tokens.
DFlash2's own forward pass costs about 15% of prefill throughput at every prompt size; MTP's costs
about 1%.

The engine's memory ledger at `--max-context 225280` puts the artifact at 9.51 GiB of weights
(plain), 10.1 GiB (MTP K=3) and 10.6 GiB (DFlash2), and leaves 1,602 / 478 / 340 / 334 MiB free on
shard 0 for the four routes in table order. DFlash2 K=5 and K=7 are the fastest decode of the four
and are what the draft's memory buys; at this capacity the plain route is the only one that could
still reach 262,144 tokens, which is why the configuration above is an MTP one. The plain arm runs
`-Plain`, which also drops vision and the sampling overrides, so only its decode column is directly
comparable.

The tables below keep the protocols and artifacts they were measured under; this section is the
current reading.

The loader staging overlaps the artifact read with both shard uploads instead of a serial
read → split → per-upload-sync chain: 2026-10-17, six interleaved A/B samples on the same host
(22.09 GB NVFP4 artifact, `--max-context 131072`, one instance at a time, engine `ready` line,
stash/rebuild arm switch): baseline 45.0/47.0/47.1 s, pipelined 29.8/37.9/38.3 s (median 47.0 vs
37.9 s), with the startup report showing `staging peak` 256 MiB (`four-slot portable-pinned ring`,
[`docs/windows.md`](windows.md#verified-behavior-on-windows)). The `Load` column above predates
this change.

## Benchmarks

| Configuration | KV capacity | Memory per GPU | Prefill | Decode |
|---|---|---|---|---|
| plain, bf16 KV | 65,536 | 13,888 MiB | 0.453 s (279 tok, 615 tok/s) | 8.153 s (31.4 tok/s) |
| MTP K=3, bf16 KV | 65,536 | 14,582 / 14,322 MiB | 0.526 s | 6.532 s (39.2 tok/s) |
| plain, fp8 KV | 65,536 | - | - | 8.091 s |
| plain, nvfp4 KV | 65,536 | - | - | 8.093 s |
| plain, int8 KV | 65,536 | - | - | 8.140 s |
| plain, bf16 KV | 131,072 | out of memory | - | - |
| **plain, fp8 KV** | **131,072** | **13,904 MiB** | - | **8.102 s** |
| plain, int8 KV | 131,072 | 13,952 MiB | - | 8.088 s |
| MTP K=2 fp8, `--lm-head-draft` | 131,072 | 14,940 / 14,678 MiB | - | 46-49 tok/s |
| plain, fp8 KV (pre-split build) | 262,144 | out of memory | - | - |
| **MTP K=2 fp8, `--lm-head-draft`, shard split** | **262,144** | **15,614 / 15,094 MiB** | **1,588 tok/s** | **59.0 tok/s** |
| DFlash2 K=2 fp8 | 4,096 | - | - | 51.0 tok/s |
| **DFlash2 K=7 fp8** | **4,096** | **-** | **-** | **63.6 tok/s** |

The 65,536 and 131,072 rows use the historical protocol (prefill: 279-token prompt, 8-token output;
decode: 256-token output reported end-to-end, so it understates sustained decode). The 262,144 rows
use `temperature 0` per request and the response's `usage` token counts, one request at a time. The
two DFlash2 rows use the B5 basis (4,096-token context, 65-token prompt, 128-token greedy output,
one request at a time) and anchor [DFlash2 throughput](#throughput) below:

- **Prefill**: a 16,057-token prompt produced its first token in 10.11 s (1,588 tok/s); a 65k-token
  prompt returned a complete answer in 48.1 s end to end (~1.35k tok/s of prompt), against 96.8 s for
  a 118,869-token prompt at the previous 131,072 ceiling.
- **Decode**: a 512-token answer took 8.68 s (59.0 tok/s over 223 MTP rounds, 2.3 committed tokens per
  round, 64.6% per-draft acceptance); a 64-token answer measured 63.6 tok/s. The previous ceiling's
  sustained decode was 46-49 tok/s, so doubling the context also moved throughput ~20% forward: the
  split halves each shard's proposal-head read and lets both shards draft in parallel.
- **Prefix reuse**: repeating an identical 16,057-token prompt returned in 0.30 s (0.33 s for the
  65k-token one) against 10.11 s / 48.1 s fresh, so one rewind snapshot behind the prefill end covers
  the chat-turn pattern (see Resident memory). A prompt that shares only the client's stable system
  prompt and tool definitions with the conversation before it -- a new session, or the first request
  after a context compression -- is covered too: the core freezes that divergence in the host ring, so
  the next such prompt reuses the whole stable block instead of restarting from zero. With a
  20k-token stable block and a 35k-token predecessor, the first divergent prompt walked 35,363 tokens
  in 23.4 s, and the one after it reused 17,451 and walked 23 tokens in 71.8 ms. A prompt whose only
  reachable boundary is a *shallow* shared-prefix image -- the shape a client that fires a title or
  summary request beside every new conversation produces -- is covered as well: the walk behind that
  recall freezes where the block really ends on the recalled conversation's own slabs, so the next
  conversation of the family starts there instead of at the shallow boundary. The *first* conversation
  of a family cannot wait for that observation, so what it freezes is the boundary the prompt names
  itself: a rendered chat prompt carries the frontier where its leading instruction block ends, the
  walk that prefills it writes the state there, and the conversation's eviction hands it to the
  session slabs. The anchor sits on the prefill chunk grid rather than a fixed distance behind the
  boundary, so what it freezes is the state a from-scratch walk has at that token count and not one
  this walk produced by being cut short. In a served reproduction of the reported shape -- three
  sessions, each opening with the same ~7k-token stable block behind its own title request, at the
  served 1024-token prefill chunk -- the first main request walked 7,037 tokens in 4.4 s, the second
  reused 6,145 of 7,039 (87.3%) at 817 ms TTFT, and the third reused 7,020 (99.8%) at 172 ms on the
  divergence anchor that walk left behind, where the same client shape reused 33 tokens of a
  10,279-token prompt before the fix. The served route also declines to retain a conversation whose
  whole history is shorter than `--session-retention-floor` (default 2048): a client that fires a
  one-off title or summary call beside every new session would otherwise fill the private catalog with
  conversations too short to be worth a slot and evict the sessions the catalog exists for, and a
  corpus that short costs less to prefill again than the slot is worth. The Engine itself keeps
  retaining everything unless a caller sets the option. A conversation also never adopts another
  one's entry: the catalog keeps one entry per conversation, only the conversation the device pools
  hold is updated in place, and a recall clears that claim on every other entry - so a displaced
  conversation keeps its own history on its slabs instead of being overwritten by whoever arrives
  next. A capture counts as a use for the LRU as well, so a boundary frozen this round is not
  evicted before the conversation it was frozen for can recall it.
- **Concurrency**: two simultaneous streaming requests both completed (24 and 27 chunks, 1.99 s).

The startup after CUDA-graph capture is a one-off 14.4 s on this configuration; it is outside the
numbers above.

## Cross-engine comparison: llama.cpp `ar3-opt`

Its own tensor-parallel implementation — graph-level sharding, the allreduce transports, and why
neither `-sm row` nor NCCL applies on this host — is analysed source by source in
[llama.cpp TP notes](maintainer/tp2-llamacpp-notes.md).

The reference engine on this machine is a tuned llama.cpp build (`C:\llama_ar3-opt`, source branch
`ar3-opt` at `C:\llama.cpp`), so both engines were measured back to back on the same two cards with
the same protocol and exactly one engine resident at a time: one non-streaming chat completion at a
time, `temperature 0`, 256 generated tokens, `--reasoning-effort medium`, MTP speculative decoding
on both sides, identical prompt text, and every number taken from the server's own `timings` block.
Prefill rows send a fresh nonce-prefixed filler prompt with `max_tokens 1`, so prefix caching cannot
fake them. ninfer runs `--devices 0,1 --max-context 131072 --kv-dtype fp8 --spec mtp
--draft-tokens 2 --lm-head-draft`; llama.cpp runs the owner's recipe via
`tools/win_port/serve_llama.ps1` (`-sm tensor -ts 1,1 -c 262144 --spec-type draft-mtp
--spec-draft-n-max 3 -fa on -ctk q8_0 -ctv q5_0`). Script `tools/win_port/bench_serve.ps1`,
raw results `build-win/bench-ninfer.json` and `build-win/bench-llama.json`.

| Workload | ninfer | llama.cpp | Ratio |
|---|---|---|---|
| prefill, 1.9k-token prompt | 1,401 tok/s | 764 tok/s | 1.83x |
| prefill, 7.2k-token prompt | 1,667 tok/s | 1,040 tok/s | 1.60x |
| prefill, 28.5k-token prompt | 1,552 tok/s | 1,016 tok/s | 1.53x |
| decode, 256 greedy tokens, 3 reps | 56.7 / 56.7 / 56.7 tok/s | 51.7 / 52.6 / 52.7 tok/s | 1.08x |

Round breakdown for the identical decode prompt:

| | ninfer K=2 | ninfer K=3 | llama.cpp n_max=3 |
|---|---|---|---|
| round time | 38.2 ms | 40.7 ms | 49.4 ms |
| committed tokens per round | 2.18 | 2.27 | 2.60 |
| per-draft acceptance | 58.8% | 42.4% | 53.8% |

**Decode does not separate because batch-1 decode is weight streaming, not compute.** Each engine
reads every weight once per forward, so `tok/s = committed tokens per round / round time`, and the
floor of the round time is `weight bytes per shard / 448 GB/s`. ninfer streams 10.15 GB per shard per
forward; llama.cpp streams 8.56 GB (`llama-gguf` reports 15.94 GiB for its GGUF: 10.38 GiB NVFP4 MLP,
3.09 GiB q5_K attention/GDN, 1.55 GiB q6_K embedding, 1.29 GiB q8_0 head). The two engines land at
38.2 ms and 49.4 ms per round, 59% and 39% of their respective floors, once the draft chain and the
host-side steps are counted. Two 5060 Ti together have 896 GB/s, half of one RTX 5090's 1,792 GB/s,
which is why this artifact decodes at 71.2 tok/s on a 5090 and at ~57 tok/s here: TP-2 is a capacity
decision (the 27B artifact does not fit on a single 16 GiB card), not a speed one.

Engine work is where ninfer is ahead - a 23% shorter round than llama.cpp and 1.5-1.8x the prefill
throughput - and four things consume that on decode:

- **Draft depth.** llama.cpp commits 2.60 tokens per round against 2.18 here. Raising
  `--draft-tokens` to 3 does not pay on this artifact: the round grows to 40.7 ms while acceptance
  falls to 42.4%, so decode drops to 55.7 tok/s. Each extra draft is host-serialized
  (`mtp_propose_window` synchronizes per draft) and adds only ~0.1 committed tokens.
- **Weight bytes.** Roughly a quarter of the per-token traffic is stored at one byte per element:
  attention/GDN projections, embedding, output head and the last eight layers' FFN are FP8 where the
  reference GGUF spends 0.56-0.69 B/element. Repacking them as NVFP4 would remove ~2.6 GB per shard
  per token, at a quality cost.
- **TP-2 structural losses.** This route runs with `use_cuda_graph = false`
  (`src/runtime/engine/model_instance.cpp:99`), so a verify forward launches ~1,009 kernels and pays
  ~2.9 ms of launch gaps; splitting heads doubles the collectives to 128 allreduces per token
  (~2.6 MB round trip, 9-18 us each); the fused `linear_swiglu`/`attn_input_proj` kernels are
  registered for full-model rows only and are bypassed on shard-local rows. llama.cpp captures the
  whole round, draft chain included, in CUDA graphs (95-96 replays per 256-token answer). Both launch
  gaps are recovered since: the verify window and the MTP draft chain are each captured per envelope
  bucket (see "Verify CUDA graph" and "MTP draft-chain CUDA graph").
- **Draft chain on the host.** Two drafts cost 7-10 ms of the 38.2 ms round on top of the ~30 ms
  verify forward, so the MTP chain, not the kernels, is the largest single overhead. Recovered since:
  the chain is one captured sequence per envelope bucket (see "MTP draft-chain CUDA graph").

Caveat: llama.cpp ran with the owner's exact command line and therefore its default batch/ubatch; a
larger `-ub` was not tried and could move its prefill numbers.

## KV cache quantization

`--kv-dtype` supports `bf16`, `int8`, `fp8`, `nvfp4`, and `k8v4`. All of them work on the plain
TP-2 route, but the quantized variants needed two fixes (below): the quantized prompt-attention
kernels, and the quantized small-T decode kernels used by a multi-column (MTP) window, both opt in to
more than 48 KiB of dynamic shared memory, and in both cases the opt-in was a function-local
`static` that configured only the first shard's device.

`bf16` KV does not combine with `--spec mtp` on this build. The first prefill dies in the bf16
prompt-attention kernel (`src/ops/softmax_attention/dense/causal_cache/prompt.cu:63`,
`cudaErrorInvalidValue`) on both cards. It reproduces at `--max-context` 2,048 and 8,192, and with
the cross-session pool switched off, so it is independent of that feature; fp8 KV, the shipped
recipe, runs MTP correctly, which is the route `ninfer_qwen3_5_tp2_sessions_test` exercises it on.

The capacity is usable end to end: a **65k-token** prompt returned HTTP 200 after 48.1 s on the
shipped 262,144-token fp8 configuration, and a **118,869-token** prompt took 96.8 s at the earlier
131,072-token ceiling, against a 65,536-token ceiling for bf16 KV on the same cards.

fp8 KV does not measurably change output quality. Four 1,400-token budget samples of the same essay
prompt produced content lengths 1365 / 859 / 1382 / 1602 (median ~1374) with no repeated-token runs,
against 1233 / 763 / 1469 / 1456 / 1430 / 1621 (median 1443) for the bf16 route at 65,536 tokens.
The fp8 KV route is also *exactly* reproducible across builds: every golden prompt below matches byte
for byte at 131,072 and at 262,144, across the shard splits and the reduced arena sizes.

The MTP layer's own cache follows `--kv-dtype` as well. That cache is draft-only -- every draft is
verified by the target -- so quantizing it is a tempting 228 MiB at 262,144 tokens, and it was measured
rather than assumed. With the MTP cache at nvfp4 and the target KV still fp8, the same cumulative
acceptance counter read 392/510 against 391/510 (77.2% against 76.9%) over the golden workload, but
three of the five golden prompts produced different greedy text. The mechanism is not the draft cache's
precision leaking into an emitted token: the verify-and-fold path is a different execution shape from
single-token decode, so a changed draft *pattern* shifts the trajectory on which near-ties are resolved.
The shipped configuration therefore keeps the MTP cache at the target dtype and pays the 228 MiB, which
is the price of the byte-identical guarantee below.

## Multi-token prediction (MTP)

`--spec mtp --draft-tokens K` (K in 1..5) drafts K tokens with the MTP layer and verifies them in a
single window forward, which is ~25% faster end-to-end and ~50% faster in sustained decode at K=2-K=3.

MTP is not token-for-token identical to the plain route, and no engine gives that. A verify window
is a different execution shape from a single-token decode -- here the chunked attention/convolution
route, in llama.cpp the chunked GDN kernel -- so its per-column logits differ, and the KV rows written
for accepted tokens carry those differences permanently. llama.cpp's Qwen3.5 MTP has the same property
and its documentation says exact output matching requires greedy sampling. The relevant criterion is
therefore that MTP is not *worse*, and it now qualifies: six samples of the same essay prompt at
temperature 0.7 / top-k 20 / top-p 0.80 produced content lengths 1519 / 1318 / 1294 / 1195 / 1683 / 0
(median 1306, no repeated-token runs) on the MTP route against 1165 / 1452 / 1546 / 672 / 224 / 1237
(median 1201) on the plain route; the single zero is a reasoning-budget artefact that the plain route
shows too. Sustained decode is 46-49 tok/s at K=2 against 31.6 tok/s plain, and the combination
with the recommended KV configuration is clean as well: three samples on fp8 KV at 131,072 tokens
produced content 1530 / 1481 / 1708 with no repeated-token runs.

This paragraph replaced a defect, not a limitation. A head-split shard was not recording its linear
attention transitions: the two record-producing branches in `gdn_mix` were gated on
`shard_config_ == nullptr`, because the fused width>1 record ops are registered for the full
16384-row fused parent only. The round nevertheless restored the pre-round state and ran
`gdn_replay_fold` over those record planes, which had never been written -- so every MTP round
replayed an empty transition log into the 48 linear-attention layers, which stopped advancing (and had
their convolution history rebuilt from the same empty log), while the 16 full-attention layers kept
working. That is exactly the observed signature: locally coherent output for a few tokens, then
collapse into `0` repetition. The shard now records on its decomposed route: the convolution record
is the raw pre-convolution window that route already materializes as `[q|k|v]`, and the recurrence
uses `ops::gated_delta_net_replay_record`, which is registered for the shard geometry (8 key heads,
24 value heads) and whose live outputs are defined to be bit-identical to the normalized
`gated_delta_net` it replaces, so window logits and acceptance are unchanged.

An earlier diagnosis of this item was wrong and is worth recording: the verify window's phase was
blamed for the drift. `Phase` has no semantic effect on a head-split shard -- `attn_mix` ignores its
phase argument entirely, and both `Phase::Verify` gates in `gdn_mix` are excluded for shards -- so
switching the window to `Phase::Verify` could not have changed a single number. The reported fault
site was likewise a red herring: `src/ops/launcher/rope.cu:189` is a `CUDA_CHECK(cudaGetLastError())`
checkpoint, and with two devices interleaved an asynchronous illegal address surfaces at whichever
checkpoint runs first, not necessarily at the faulting kernel.


### Verify CUDA graph

The MTP verify window is captured as one CUDA graph per device (src/core/decode_graph.h, capture_group
begins capture on both device streams and records each stream into its own definition), so a round's
~1,000 kernels are replayed from ~10 host calls instead of being launched one by one. The window's host
inputs (token ids, positions) are written into a portable pinned buffer that the graph's leading memcpy
nodes re-read on every replay, and the attention envelope is baked per capture: the eight buckets come
from mtp_graph_profiles(capacity, draft_tokens), so a round picks the bucket whose visible range covers
it. The captured body allocates from the round's arena watermark, which DeviceArena::rewind restores
before each round; a bucket captured at a lower watermark (the startup warm-up request) is re-captured
at the first real request's.

NINFER_TP2_VERIFY_GRAPH=0 runs the same window eagerly. Both routes call the same forward_tp2_window
with the same bucket envelope, so the switch changes only how the kernels reach the GPU:

| Metric (bench_serve.ps1, fp8 KV, context 131072, K=2) | eager | graph | delta |
|---|---|---|---|
| round time (NINFER_TP2_TIMING=1) | 39.6 ms | 35.3 ms | -4.3 ms |
| of which verify | 35.0 ms | 30.6 ms | -4.4 ms |
| 2048-token-context decode (64 tokens both) | 65.6 tok/s | 76.5 tok/s | +16.6% |
| prefill, 512-token prompt | 1419 tok/s | 1419 tok/s | flat |

Decode throughput also depends on the draft acceptance rate, which the round's own text decides, so the
round time and the equal-output 2048-token run are the trustworthy comparisons. The whole 4.4 ms is
launch and scheduling overhead: the verify's kernels, parameters and operands are identical in both
routes, and five prompts x up to 256 greedy tokens are byte-identical between them.
The graph's replay is also insensitive to added instrumentation, which is what made the round-boundary
synchronization defect below visible in the first place.

## DFlash2 masked draft

`--spec dflash2 --draft-tokens K` (K in 1..15) runs the masked-draft round: a proposal over the
target's captured prefill features, one target verify window over `K+1` columns, sparse acceptance
against the target's own greedy tokens, and a GDN record/replay fold. The draft component (5 ring
layers + selector) is materialized whole on shard 0 next to the 40 MiB context ring; the verify window
and fold run on both shards exactly like the MTP window.

The route is bit-reproducible across a recall while the walk keeps the oracle's execution shape. A
recall that lands on the prefill grid does: it re-walks the same chunks a from-scratch prefill of the
same prompt walks, so every verify window verifies the same columns and the KV rows they write are
the same bytes. That
property needed one fix worth naming: a budget-clamped window (the tail of a request, where only a
prefix of the proposal can be licensed) pins its trailing columns on the last licensed column's
absolute position, and those duplicate columns used to append KV like every other column - several
warp writes racing the position owner's slot inside one launch, with torn FP8 codes. The window now
binds the column mask `target_valid_columns` already carried (`extent + 1` columns own their slot), so
clamped columns neither append KV nor publish logits. `ninfer_qwen3_5_tp2_dflash_solo_test` is the
acceptance: independent processes must print one identical digest over a nine-walk sequence, and
its recalled walk must match the evicted (from-scratch) walk of the same prompt token for token.

A recall whose boundary is not a multiple of the prefill chunk (256 by default) walks its suffix from
wherever the boundary sits, so its chunks are not the oracle's and it does not owe the oracle the
whole trajectory - only the boundary crossing. The state and KV beside the boundary are still the
conversation's own prefix (the scan proved the tokens before it agree), so the first sample is the
same logits, and the tolerance covers the exact ties that can resolve differently later. The masked
draft stays live on such a boundary rather than being declined: the ring beside the restored state
belongs to the walk that froze it, so it proposes the block that walk would have proposed, the draft
only proposes, and the target verify licenses every emitted token regardless - the guarantee every
speculative round already gives. Keeping it live is what the route wants. On the walk the decline used to run target-only - the same
conversation and the same reuse boundary, only the draft toggled - keeping it decoded at 87 tok/s
with 28 of 63 drafts accepted, against 22.5 tok/s declined. What it accepts follows the text it is
predicting, not the route: the same served recipe and the same 2,442-token reuse decode a prose
answer at 54 tok/s (31 of 147 accepted) and a number list at 176 tok/s (446 of 448).

### Throughput

The route's controlled numbers are measured on the served TP-2 path, one request at a time, fp8 KV.
The 4,096-token-context rows are the B5 basis - a 65-token prompt and a 128-token greedy output -
and the two controls are there to read the route's gain:

| Route | Context | Decode | Acceptance | Notes |
|---|---|---|---|---|
| plain (control) | 4,096 | 35.1 tok/s | - | |
| MTP K=2 (control) | 4,096 | 62.8 tok/s | 63.4%, 2.27 tokens/round | |
| DFlash2 K=2 | 4,096 | 51.0 tok/s | 59.5% | |
| **DFlash2 K=7** | 4,096 | **63.6 tok/s** | 30.0%, 3.10 tokens/round | 1.81x plain, on par with the MTP control |
| DFlash2 K=7, drafts live on unaligned reuse | 4,096 | ~71 tok/s | 3.17 tokens/round | the released configuration |
| DFlash2 K=5 / K=7, `draftall` artifact | 245,760 | 96.4 / 96.1 tok/s | - | interleaved recheck 93.3 / 96.9 |

Two properties of the route show up as spread rather than as one figure, and both are measured in
the paragraph above: what the draft accepts follows the text being predicted, and a declined draft
costs most of the route's speed.

What bounds the route is bandwidth. A forward stream reads 10.15 GB per shard, so no round can
finish in less than 10.15/448 = 22.7 ms on TP-2, against 20.3/1,792 = 11.3 ms for the same stream
on a single 5090; with the verify window still eager (the feature sink cannot enter the capture)
and the proposal at 6.4-6.8 ms per round, the roofline estimate in
[tp2-plan.md](maintainer/tp2-plan.md) §3.6 is 108 tok/s, and K=7's 63.6 tok/s does not reach it.
Two route-specific terms sit on top of that floor: the draft's 2.07 GiB lives on shard 0 alone,
about 5.0 ms per round against 1.24 ms on the 5090, while MTP pays 7-10 ms per round for the
host-serialized draft chain DFlash2 does not have. The platforms are otherwise comparable in
efficiency - the 5090's 192.5 tok/s at 37.0% acceptance is 3.59 tokens per round in 18.6 ms, 61% of
that card's floor, and TP-2's MTP K=2 is 38.2 ms per round at 59% of its own - so the roughly 2x
gap between them is the bandwidth ratio rather than an implementation difference. Single-5090
numbers for the same route are tabulated in [qwen3.8-27b.md](performance/qwen3.8-27b.md).

### Verify CUDA graph

The DFlash2 verify window is captured exactly like the MTP one (one graph per device per envelope
bucket, now from `dflash_graph_profiles`), with one difference in how per-round values reach it: the
clamp extent cannot ride in a kernel argument. `set_i32_scalar` bakes its value into the setter
kernel's launch arguments, so every replay would clamp to whatever extent the capture round saw. The
extent therefore travels through the pinned window buffer's trailing int (`[ids, positions, valid]`),
which the graph's memcpy nodes re-read on every replay exactly like ids and positions. The feature
sink travels with the capture as well: its scatter addresses are baked, and its per-round lane and
column tensors are re-read by the kernels on every replay. `NINFER_TP2_VERIFY_GRAPH=0` runs the same
window eagerly against the same bucket envelope, so the switch changes only how the kernels reach
the GPU.

The acceptance is byte-exact: the solo digest must be identical across independent processes under
both routes, and it is - 6/6 graph runs and the eager run agree on `0x4bcc3994a5efba7d`. The first
wiring attempt failed exactly this test and is worth recording: the capture's call forwarded neither
the sink nor the extent (default arguments compiled quietly), so every replay ran the old unmasked,
sink-less window and the budget-clamped rounds were back to the torn-KV race above - 2 of 6 runs
flipped the tail near tie through precisely that race's value set (`198/220/13962` at the shared
walk's last sample), while eager never moved. A default argument on a capture path is a silent
downgrade: the captured shape is legal, every replay runs it, and only near ties in the output show
the difference.

Measured with bench_serve.ps1 against the served configuration (synthetic fill text, which this
acceptance rejects to about one committed token per round): the equal-output 128-token decode at an
8192-token context drops from 6267 ms eager to 5627 ms graph (-10.2%, about -4.9 ms per round),
20.3 -> 22.6 tok/s; the short decode moves 20.6 -> 23.0 tok/s. Prefill is unchanged
(1107/1565/1504 -> 1154/1564/1499 tok/s at 2048/8192/32768-token prompts) and the MTP K=2 control
stays in its 66-72 tok/s band (67.9/64.8), so the wiring costs neither route. The -4.9 ms round-time
delta matches the MTP window's -4.3 ms; the smaller throughput gain than MTP's +16.6% is the
synthetic text's low acceptance, which leaves less output per round to speed up.

### Draft precision and the selector codebooks

The masked draft's projections run Q4_G64_FP16; its selector codebooks (predecessor and successor,
[248320, 256] each) keep the stored BF16 representation. That split is measured, not inherited:
quantizing the codebooks to Q4 saves 178.09 MiB and costs long-context acceptance.

The comparison ran three artifacts over the same ~32,948-token prompts (a scenario fixture with
140,000 characters of NIAH document prepended, 512 output tokens, stochastic
0.6/0.95/20/0/1.0/0, C=1, K=7, fp8 KV, 4 fixtures x 5 seeds = 20 requests per arm):

| Draft representation | Pooled acceptance | Paired delta vs Q4 (95%) |
|---|---:|---:|
| Stored BF16 draft (baseline) | 27.89% | +2.13 +/- 2.10 pp |
| Q4 projections + Q4 codebooks | 26.00% | - |
| Q4 projections + BF16 codebooks | 28.38% | +2.49 +/- 1.71 pp |
| BF16 feature_projection and context K/V, Q4 codebooks | 27.52% | +1.90 +/- 1.39 pp |

Restoring the two codebooks alone returns acceptance to the stored artifact's level (28.38% against
27.89%, a difference inside the interval); restoring the context-conditioning path (feature_projection
plus the ten context K/V views) does the same. The two do not add: doing both measured 27.75%. Q8 is
not a middle option the Engine offers - SelectorCodebook accepts BF16 or Q4_G64_FP16 and rejects any
other representation at load - and it would not buy much: storing Q8-quantized values in a BF16
codebook (relRMS 0.0057 against the BF16 values, where Q4 is 0.110) measured -0.29 +/- 0.51 pp.

The 178.09 MiB lands on shard 1, which holds the selector, not on shard 0, which holds the masked
draft. At 65,536 tokens the Q4 artifact's shards hold 11976.6 and 11132.6 MiB of weights+context;
BF16 codebooks move the second to 11310.6 and leave the first unchanged, so the precision costs no
context on the shard that bounds it (free 2848 against 2922 MiB with the vision arena enabled). The
context-conditioning restore is the one that does cost: its 119 MiB belongs to shard 0, about 7.6k
tokens of fp8 KV at 16.1 KiB per token.

The draft's Q4 conversion is a pure quantization of the same weights: every changed tensor correlates
0.9938-0.9944 with the stored representation (relRMS 0.109-0.112, the noise a 4-bit group-64 scale
predicts), and restoring all 88 changed tensors reproduced the baseline artifact's per-request
counters exactly, so the acceptance differences above are not an artifact of how the mixed artifact
was built. The delivered artifact is `rloo351-mixed-mtp-dflash2-vision.ninfer` (21.14 GiB, metadata
name `rloo351-mixed-mtp-dflash2-vision`), text+mtp+dflash2+vision with the codebooks restored. Its
source is the Q4 draft plus vision artifact, which is not retained under a separate name - the
campaign graft builds it from the stored draft and the W4A4 donor:

```bash
python -m tools.convert.graft_bindings \
  --source <Q4-draft+vision artifact>.ninfer \
  --override rloo351-mixed-mtp-dflash2.ninfer \
  --binding dflash2/candidate_selector/predecessor_codebook \
  --binding dflash2/candidate_selector/successor_codebook \
  --name rloo351-mixed-mtp-dflash2-vision \
  --out rloo351-mixed-mtp-dflash2-vision.ninfer
```

The acceptance comparison itself ran through `tools/bench/run_serve_spec_ab.py` (its `run`
subcommand per artifact, then `compare` over the two summaries) with the fixtures, seeds, padding
and sampling profile listed above.

## MTP draft-chain CUDA graph

The MTP chain (split embedding, MTP-layer stem, the autoregressive hidden relay, the two draft
proposals) ran as three host-serial forward calls per round. It is now one captured sequence per
envelope bucket and per device, launched together through `capture_group` on the two shards'
streams - the structure llama.cpp uses for its whole round. The three per-round scalars reach the
chain from the pinned buffer's leading ints (`[anchor, position, position+1]`) through
graph-capturable memcpy nodes: `set_i32_scalar` bakes its value into the setter kernel's launch
arguments, so a replay would otherwise run at the capture's position. The autoregressive position
advances on the device, the hidden relay moves on the device, and the drafts return through one D2H
copy into the same pinned buffer. The captured body is the eager route's body verbatim.

Replay bookkeeping exposed one real trap. A captured body's workspace advance is replayed as
`alloc_bytes(captured_delta)` per side, and a side whose scratch is entirely scope-unwound nets
zero - a legitimate value that `DeviceArena::alloc_bytes` rejects ("arena allocation must be
nonzero"). The served flow nets a small positive peer-side delta and never saw it; the session flows
(recall, chunked re-prefill) net zero and died at the first replay. All three window/chain replay
advances now treat zero as a no-op, exactly as `position_arena` already did with its strict
`target > used` comparison.

`NINFER_TP2_MTP_CHAIN_GRAPH` selects the launch mode with the same convention as
`NINFER_TP2_DECODE_GRAPH` (the shared enum is now `StepLaunchMode`): unset captures, `0` runs the
same body eagerly against the bucket envelope, `exact` reproduces the pre-graph per-step envelopes.
`exact` is also forced without the in-kernel allreduce transport.

The acceptance is byte-exact and the performance claim is smaller than projected. Five greedy goldens
are byte-identical across all three modes and against the pre-change build (git-stash A/B), the
sessions suite passes on all three routes, and the deterministic-trajectory
`NINFER_TP2_TIMING=1` A/B (identical 94-round / 180-token walks in all runs) measures the chain step
at 3.73/3.69 ms eager versus 3.50/3.50 ms captured, the round at 34.30/34.09 -> 33.99/33.95 ms. The
capture removes about 0.2-0.3 ms of host launch gap per round (~+0.7% decode), not the projected
15-20%: the earlier "7-10 ms chain" figure was a subtraction from a slower round, and the
instrumentation shows 3.5-3.7 ms of real MTP-layer work whose launch gaps were already mostly
hidden behind it. `bench_serve` decode rates cannot settle this either - its unique nonces change
the sampled content, which moves draft acceptance and the round count - so the component A/B above
is the measurement of record.

## Proposal head

`--lm-head-draft` switches the draft head from the weight-tied full output head (248,320 rows,
Q8_G32_FP16) to the artifact's indexed proposal head (131,072-row frequency shortlist, Q4_G64_FP16,
plus a row-to-token-id map). Target verification still uses the full head, so this changes only what
the draft proposes, never which token is emitted. A/B at the recommended fp8/131,072 MTP K=2
configuration, two six-sample runs, one request at a time:

| Workload (1,400-token budget) | Route | Decode tok/s, median (range) | Draft acceptance | Content median |
|---|---|---|---|---|
| Chinese essay | full | 45.18 (43.20-52.25) | 54.3% (2780/5118) | 1294 |
| Chinese essay | `--lm-head-draft` | **50.10** (47.53-55.35) | 53.5% (2749/5136) | 1336 |
| Python coding | full | 48.59 (45.62-51.07) | 58.6% (4532/7730) | budget-limited |
| Python coding | `--lm-head-draft` | **49.58** (48.85-54.87) | 53.1% (4326/8142) | budget-limited |

Acceptance is derived from the `NINFER_TP2_TIMING=1` decode line
(`[tp2-time] decode rounds=R committed=C`): accepted drafts are C - R out of R*K, where K is
`--draft-tokens`. The response's `timings.draft_n` and `draft_n_accepted` now carry the same counts
on this route. Both routes were
non-degenerate on the essay prompt (no repeated-token runs); the coding prompt exhausted the token
budget in reasoning on every sample, so its content is not comparable.

The shortlist and the indexed head's Q4 quantization both move its argmax away from the full head's,
so acceptance falls -- 0.8 points on prose and 5.5 on code. The cheaper head read still wins on both
workloads, but the margin shrinks as acceptance drops, so this is a throughput trade rather than a
free win.

The head is 131,072 x 5,120 Q4_G64_FP16 (341 MiB) and only shard 0 ever samples from it, so the pair
now holds one vocabulary half each (65,536 rows, 171 MiB) and reassembles the draft logits before the
argmax. That needed a Q4 route for the half shape: the Q4 A16 dispatch is an enumerated (n, k) table
and no admitted shape had an admitted half, so `select_q4_n65536_k5120` was added next to the
131,072-row entry. The merge is exact -- disjoint row blocks, one in-place all-reduce -- and
`ninfer_linear_tp2_split_grouped_head_test` runs both shard halves against the full-weight Op at
exactly this shape (T=1 and T=2) and compares them bit for bit.

## Cross-session KV pool

TP-2 keeps exactly one conversation in the device KV pool. A request that belongs to another
conversation copies the resident session's KV, GDN state and MTP cache into pinned host memory and
copies the returning session back, so an alternating chat client pays one PCIe round trip instead
of a full prefill. The catalog holds the resident conversation plus `--max-private-continuations - 1`
host-resident ones (six entries by default) and evicts the least recently used when it is full.

Enable it by giving the host KV budget, which is split evenly between the two shards:

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 <...> \
  --host-kv-mib 32768 --max-private-continuations 6
```

`--host-kv-mib 32768` is 16 GiB per card, enough for five 204,800-token fp8 conversations at
16.125 KiB/token/card, and the startup ledger states that arithmetic back
(`[tp2-session] host budget 16384.0 MiB/shard holds 5 of 5 full-context sessions (3225.0 MiB each)`).
The arena is built on the first eviction, so a single-conversation workload never allocates it. Its
backing is pageable, which keeps the pool creatable when the commit limit is tight and lets the OS
reclaim it under pressure; `--host-kv-pinned` pins it instead when the faster transfers are worth a
permanent lock, and a refused pin falls back to pageable rather than failing the pool.
`--host-kv-mib 0` disables the pool entirely and restores the previous behaviour.

Observable behaviour:

- A returning conversation reports `reused_prompt_tokens == frontier` (its whole stored history) and
  the prefill only walks the new suffix. The frontier is one token short of the history, because the
  last sampled token is not forwarded until the next round.
- A conversation that continues in place - the agent loop, where no other conversation runs in
  between - reaches its committed frontier, off the prefill grid, so the tail of the last answer is not
  prefilled a second time. Decode writes the same host checkpoint grid the prefill writes, at the
  frontier each round actually committed, so a prompt that cannot reach that frontier still restarts
  inside the generated span: a replay the engine cannot adopt costs at most one stride
  (`kReuseCheckpointStride`, 8192 tokens) instead of the whole answer.
- Every generated token is published, including the one the prefill itself samples, so a client that
  sends an answer back the way it received it re-tokenizes to the stored tokens and reaches the frontier above
  rather than stopping at the previous prompt. The first token is committed through the output policy like every
  later one; a token that bypassed it would be missing from the answer and would leave the pools and the session
  catalog describing a history one token longer than any client could send back.
- A recall restores byte-identical KV and GDN state, so a returning conversation reproduces a
  from-scratch prefill of the same prompt token for token while its boundary sits on the prefill grid.
  A boundary inside a chunk re-walks the suffix from there: its first sample is the same logits, but an
  exact tie later can resolve differently, so the session tests assert that crossing rather than the
  whole generated tail.
- A conversation the LRU budget evicted reports `reused_prompt_tokens == 0` and is prefilled from
  zero again.
- `NINFER_TP2_SESSION_TRACE=1` prints every recall and eviction with its frontier, and
  `NINFER_TP2_REUSE_TRACE=1` prints per request what the scan offered (`prompt`, `cached`,
  `shared`, `prev_prompt`, `replay_split`, `adopted`, the checkpoint census) and what it took
  (`reuse`, `slot`, `src`). When the client's rendering parts from the resident history inside the
  span this lineage generated, the same switch prints the bytes either side of that divergence and
  whether the two describe the same turn - the measurement that decides whether a replayed answer is
  reusable at all.

Constraints and limits:

- Host memory: the KV budget is pageable by default and still counts as private commit; add one GDN
  state image per host-resident session per shard (~73 MiB each, pinned) and the pinned prefix-reuse
  checkpoint ring. Confirm at least 48 GiB of physical memory before asking for 32 GiB, and lower
  `--host-kv-mib` when in doubt: at 204,800 tokens a
  full-context fp8 conversation is 3,225 MiB per shard, so 32768 holds the five host-resident
  conversations the default catalog admits and 16384 holds two.
- Shard 0's arena also carries the MTP layer's own KV (~2 KiB/token), so a symmetric split spends a
  little of shard 0's budget on it.
- A client that re-renders the answer it was handed normally differs from the generated bytes only in
  the framing the template rewrites: it trims the reasoning and the content and it writes its own
  newline runs around the reasoning close, so a replay loses the leading whitespace of the model's
  reasoning while still describing the same turn. The walk adopts the span this lineage generated for
  such a replay - the tokens, not the client's rendering, are what its KV holds - and the conversation
  then reaches the frontier it would have reached by echoing exactly.
- A replay that describes a *different* turn is left alone: a reordered argument object, an added
  default, an edited line or a missing first token keeps the client's own tokens, and the walk restarts
  at the deepest boundary it still shares - the state frozen at the previous prompt's end when the
  divergence sits at the first generated token, or a decode-time grid checkpoint inside the answer.
- The reuse scan takes the deepest boundary the lineage offers, on or off the prefill grid, and the
  request's output budget never enters into it: a client that asks for its model's whole context still
  keeps the tail it already generated. An off-grid boundary leaves the suffix chunked from a different
  start, so such a walk guarantees its boundary crossing rather than its whole trajectory, and it
  keeps the masked draft live (see the masked-draft section).
- `--chat-template FILE` reaches this route: the core builds the serving frontend with the option, so a
  maintained template can replace the artifact's embedded one.
- Session switching happens at request boundaries only. On the default one-lane route a request
  that arrives while another is generating waits for it. On the batched route (`--max-concurrency`
  `2..4`, plain, DFlash2 and MTP rounds) up to that many queued requests share one decode round.
  Cross-session retention stays on: the checkpoint ring and the session catalog are sliced by lane, so
  each lane recalls its own previous conversation independently. A Vision request encodes its media
  inside the round on the Vision shard, and a reused prefix additionally requires matching media
  identity -- a checkpoint records each image's digest, grid, and consumer spans -- so two prompts that
  differ only in image content never share cached KV.

## Device memory budget

Measured by starting the engine under controlled configurations and reading
`nvidia-smi --query-gpu=memory.used` after load with no requests. CUDA devices 0 and 1 are nvidia-smi
indices 0 and 2; nvidia-smi index 1 is an unrelated idle Tesla T10. At the shipped configuration
(262,144 tokens, fp8 KV, MTP K=2, `--lm-head-draft`) the per-card totals are 15,614 MiB (shard 0,
which also owns the MTP layer) and 15,094 MiB (shard 1): 697 and 1,217 MiB still free of 16,311 MiB.

The engine prints one ledger line per shard at startup -- the blocks it allocates, in MiB, at the
requested ceiling:

```
[mem] shard 0 capacity 262144 | weights+ctx 11494.6 | kv 4644.2 | state 293.6 | record 2.6 | round 0.0 | workspace 192.0 | free 0.0 of 16310.6 MiB
[mem] shard 1 capacity 262144 | weights+ctx 11494.6 | kv 4128.0 | state 293.6 | record 2.6 | round 0.0 | workspace 192.0 | free 196.0 of 16310.6 MiB
```

| Component | shard 0 | shard 1 |
|---|---:|---:|
| Weights + CUDA context (ledger `weights+ctx`) | 11,494.6 | 11,494.6 |
| Text KV cache, fp8, 262,144 tokens | 4,128.0 | 4,128.0 |
| MTP layer KV cache, fp8, 262,144 tokens | 516.2 | 0.0 |
| Linear-attention state arena (2 live + 1 round scratch + 2 reuse snapshots) | 293.6 | 293.6 |
| Program workspace arena | 192.0 | 192.0 |
| MTP replay records and round anchor | 2.6 | 2.6 |
| **Sum of the engine's blocks** | **16,627** | **16,111** |

The per-token costs are the ones to plan with: text KV is 16 full-attention layers x 2 local KV heads
(of four, head-parallel) x head_dim 256 x K+V x 1 byte = 16.125 KiB per token per shard, 4,128 MiB at
262,144 tokens; MTP KV is 2 KiB per token on shard 0 only, because that layer is replicated rather
than head-split. The ledger's `free` column is `cudaMemGetInfo`, which this WSL2 driver under-reports
(it reads 0.0 while nvidia-smi shows 697 MiB free and the engine starts and serves reliably); the sum
of the rows therefore exceeds the reported device total of 16,310.6 MiB by ~1 GiB. Treat the row
sizes and the nvidia-smi totals as the budget, not the `free` column.

Three things had to give to reach 262,144 tokens, all of them memory-only and all of them verified
numerically transparent:

- **Vocabulary and hidden parallelism of the shard weights.** `text/output_head` (248,320 x 5,120,
  FP8 row-scale), the reduced `proposal/head` (131,072 x 5,120, Q4_G64_FP16) and
  `text/token_embedding` (248,320 x 5,120, FP8 row-scale) used to be replicated on both shards,
  4.8 GiB of duplicated tables on the pair. The two heads are now split by vocabulary row and the
  embedding by hidden column, so each shard holds half of each table and the pair assembles the
  result before sampling or drafting: ~1.2 GiB per card for the head and embedding, 170 MiB for the
  proposal head.
- **Prefix-reuse snapshots 3 -> 2.** The state arena held five 73.4 MiB planes: two live buffers, the
  MTP round scratch, and three prefix-reuse snapshots at rewind depths 0 / near / far. The chat-turn
  pattern the snapshots exist for needs the near rewind only, and the measurement above confirms it
  (identical 16k and 65k prompts re-enter in 0.30-0.33 s).
- **Workspace 384 -> 192 MiB.** The arena's per-layer prefill peak is ~140 MiB at the maximum prefill
  chunk of 1,024 tokens, so 192 MiB still leaves headroom; a larger request is a reported arena
  overflow, not corruption.

One block stays deliberately replicated: the MTP layer's own weights (430 MiB), because the draft path
runs on shard 0 alone and handing it a collective per draft step would cost more than the memory. The
KV cache stays fp8: nvfp4 or k8v4 KV (`--kv-dtype`) would free further memory at a value-precision
cost, and int8 is 48 MiB *larger* than fp8 at the same ceiling because of its 64-element scale groups.
The context cache is disabled on this route -- the core owns the prefix-reuse snapshots instead.

## Verification

| Check | Command | Result |
|---|---|---|
| Greedy identity of the whole stack | `r52_ab.sh final` then `r52_cmp3.sh` | 5/5 prompts byte-identical to the pre-split build at 131,072 (`ab-embed`, itself identical to the pre-optimization `ab-control`) |
| Route loading | `ninfer_qwen3_5_tp2_load_test` (plain, `--spec mtp`, `--spec mtp --lm-head-draft`, `--spec dflash2`, `--spec dflash2 --lm-head-draft`) | expected shard shapes: head 124,160 / 248,320 rows, embedding `[248320,2560]`, proposal head `[65536,5120]` |
| TP-2 execution | `ninfer_qwen3_5_tp2_forward_test` | 11 probes shard-consistent, 32-step decode, prefill chunk-split invariant |
| Vocabulary-parallel head | `ninfer_linear_tp2_split_fp8_head_test` | exact at `[248320,5120]` and `[16384,5120]` |
| Grouped proposal head | `ninfer_linear_tp2_split_grouped_head_test` | exact at `[131072,5120]`, T=1 and T=2 |
| Half-width FP8 embedding | `ninfer_embedding_test` | full sweep at `[248320,2560]` |
| NVFP4 split Linear | `ninfer_linear_tp2_split_nvfp4_test` | passes |
| Collective staging | `ninfer_tp_device_pair_test` | passes |
| DFlash2 masked-draft round | `ninfer_qwen3_5_tp2_dflash_append_test` (`--k 7`, `--k 5`) | sink logits bit-identical with and without the sink; ring sane and reproducible (fnv1a `0x1a53fd824cd4e360`); direct/split append bit-identical; proposal deterministic (fnv1a `0xbad27a494a9bc853` at K=7, `0xbee487264ca8ffb8` at K=5) |
| DFlash2 walk determinism | `ninfer_qwen3_5_tp2_dflash_solo_test` (independent processes) | one digest (`0x4bcc3994a5efba7d`) over 10 captured-graph runs with identical walk bodies, byte-identical again under `NINFER_TP2_VERIFY_GRAPH=0`; the recalled walk matches the from-scratch walk of the same prompt on its first sample in every run |
| DFlash2 verify CUDA graph | solo A/B graph vs `NINFER_TP2_VERIFY_GRAPH=0`, `bench_serve.ps1` served route | walk digests byte-identical across routes (before the capture-forwarding fix 2 of 6 graph runs flipped the tail near tie); equal-output 128-token decode -10.2% wall (-4.9 ms/round), prefill flat, MTP K=2 control in its band |
| MTP draft-chain CUDA graph | r52 greedy goldens in all three launch modes + git-stash pre-change A/B, `ninfer_qwen3_5_tp2_sessions_test`, `NINFER_TP2_TIMING=1` trajectory A/B | 5/5 byte-identical across graph/eager(bucket)/eager(exact) and the pre-change build; sessions pass on all three routes (after teaching the replay advances to accept a zero workspace delta); identical 94-round walks measure the chain step 3.7 -> 3.5 ms and the round 34.2 -> 34.0 ms |
| Multi-lane admission and scheduling | `ninfer_qwen3_5_tp2_lanes_test` (`NINFER_TEST_ARTIFACT`; plain, `--max-concurrency 4`) | four lanes in one batch each keep their own KV row and state slot (four distinct 64-token prompts each spend their budget; the same prompt x4 with prefix reuse off agrees token for token); six 600-token prompts share a 32-page pool that holds three lanes and all complete through the requeue path; an already-expired deadline is retired as `RequestError(QueueTimeout)` ("expired while waiting for admission") and a queued cancellation as `FinishReason::Cancelled` with no tokens, after which a fresh request still succeeds; a request arriving mid-batch (P2.2) finishes in 0.59 s against the running request's 3.67 s; destroying the engine with a request in flight stops and joins the driver without hanging |
| Cross-session KV retention | `ninfer_qwen3_5_tp2_sessions_test` (`NINFER_TEST_ARTIFACT`; all three routes) | every route: a returning conversation recalled 71 prompt tokens and matched the oracle token for token where the draft pattern survives (the switch scenario on the prefill grid and the cancellation retry included); the LRU-evicted one reported `reused_prompt_tokens == 0`. A recall anchored inside a chunk pins the boundary crossing (first sample) instead of the whole trajectory. The MTP round's shard-0 arena reports `2 layouts` against shard 1's `1`, so its own KV slab travels with the session. One scenario pins the shallow-recall case: a conversation recallable only 8 tokens in has to leave the stable block's real end (500 - 8) on its own slabs, and the next conversation of that family reused 492 prompt tokens -- 8 before the fix -- on the `dflash2` and `mtp` routes (the `plain` route stops earlier on the artifact-sensitive first-sample divergence above). A second scenario pins the block a prompt names itself: after the conversation's eviction hands the frozen frontier to its slabs and a probe prunes the ring around it, the next conversation of the family reused at least nine tenths of the first one's prompt (the served reproduction above records the same effect as 7,017 of 7,039 tokens). One more pins the client-disconnect case: a request whose sink throws mid-answer still leaves the prompt end the walk published recallable, so the conversation's next turn reused 408 prompt tokens instead of prefilling its history. It pins the boundary and not a bit-exact answer: a recall re-walks its suffix from a different chunk split than a from-scratch walk uses, and on this artifact the resulting near tie resolves either way - the same retention-off oracle answered differently under two chunkings - which is the bounded-recall tolerance documented beside this row |

"Exact" is bit for bit: the split Op output equals the same Op run with the full weight on the same
device, which is the property the merge relies on. `temperature 0, top_k 1` is what makes the greedy
comparison meaningful; the served configuration samples (`0.7 / 20 / 0.80`).

## Fixes applied on this branch

- Vocabulary-parallel and hidden-parallel shard weights: `text/output_head` and the reduced
  `proposal/head` split by vocabulary row, `text/token_embedding` by hidden column, with the pair
  merging the reduced block before sampling or drafting. That required shard-local `ColumnParallel`
  view offsets in `tp_shard_views.cpp`, FP8 slice descriptors that follow the new row count
  (`scale_ne[0]`, `scale_nb[1..3]`, `group`/`group_size` in `weight_splitter.cpp`), an offset
  correction for the peer half during the merge, a 16-byte-padded ids broadcast (the collective
  requires a multiple of 16 bytes), and a grouped-format (Q4/Q5/Q6/Q8) row slice plus `shard_geometry`
  knowledge of the `RowSplit` planes.
- A Q4 A16 route for the half proposal head (`select_q4_n65536_k5120`), because the Q4 dispatch is an
  enumerated shape table and no admitted shape had an admitted half.
- A `[248320, 2560]` FP8 embedding-gather domain: the gather kernel was templated on the hidden width
  and the wrapper now queries `embed_gather_fp8_supports_width` instead of hard-coding 5,120.
- The startup `[mem]` ledger line per shard, next to the existing `capacity | KV` line.



- `copy_row_block` and the ids plumbing: a `void*` byte-offset bug corrupted window ids in the MTP
  warmup (`cudaErrorIllegalAddress`).
- MTP rounds snapshot the pre-verify GDN state and replay the fold at the committed width, so the
  verify no longer double-advances the recurrent state.
- Head-split shards now produce the replay records that fold consumes (convolution record from the
  decomposed route's pre-convolution window; recurrence via `gated_delta_net_replay_record` at the
  shard geometry). Without them the fold replayed an empty log every round and MTP degenerated.
- Quantized prompt attention (`prompt_fp8.cu`, `prompt_nvfp4_non_rdc.cu`, `prompt_k8v4.cu`) now
  perform the `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` opt-in **per device**; a
  function-local static configured only the first shard's device, so the second device rejected every
  >48 KiB launch with `cudaErrorInvalidValue` and no quantized KV dtype could start on TP-2.
- The same defect existed in the small-T decode kernels (`small_t_fp8.cu`, `small_t_k8v4.cu`,
  `small_t.cu`), where it stayed latent because a single-token decode uses a 32 KiB window that fits
  the 48 KiB default. The MTP window (`T = K+1`) switches to a 64 KiB tile that does not, so
  `--spec mtp` failed on every first quantized-KV request with `small_t_fp8.cu:56
  cudaErrorInvalidValue`. All four quantized/bf16 small-T routes now share one per-device opt-in
  authority (`src/ops/common/cuda_smem.h`).
- The in-kernel allreduce's arrival tokens (the monotonic per-call counter that keeps a captured
  sequence's handshake honest) now live in **mapped pinned host memory** instead of on a device heap.
  The caller chooses which shard drives a pair, so an allreduce's stream is not necessarily device A's
  -- the TP-2 forward test drives the same pair from either shard -- and a device allocation was then
  dereferenced from the other device's context (`cudaErrorIllegalAddress` inside the first layer).
  The kernel reads the token once per block and publishes it through shared memory, because a
  per-thread read of a system-scope location costs ~4 us per allreduce, ~0.5 ms per round.
- TP-2 decode rounds now synchronize **both** devices at the round boundary, not just shard A. The
  host folds the verified columns, restores the pre-round GDN state and rebuilds the next round's
  window right after that sync, while shard B's verify tail could still be running; the pair's
  in-kernel allreduce is the only thing that orders the two devices' kernels against each other, so
  the peer's tail was unordered against the host's next round. It was latent in the eager path (where
  ~1,000 launches per round kept the host behind the GPU) and only became deterministic once the
  verify turned into a graph replay. The added sync costs 0.01 ms because shard B trails by less than
  one kernel, and eager and graph now agree byte for byte on every prompt.
- Every token selection on the TP-2 route (`ops::sample` for the prefill token and for plain decode, the
  MTP target `ops::argmax`, and `speculative_accept_greedy_drafts`) is bounded by the public tokenizer
  vocabulary instead of the packed embedding row count. A padding row could be selected, corrected or
  licensed, and the resulting id then failed tokenizer decoding and killed the whole request; the
  single-GPU route and the draft selector already passed `public_token_count`.
- Presence and frequency penalties reach the sampler on the TP-2 route. `make_sampling_config` left
  `token_counts` null and the penalty only reads that array, so across rounds both penalties were inert
  and only the MTP verify window's round-local overlay still applied. The request now owns an
  `I32[public_token_count]` count array, reset with the request, that `ops::sample` and the speculative
  accept kernel increment as they produce tokens.
- The host-staging fallback's reduction is vectorized and split across host threads. The fallback
  hands its payload to the copy engines and used to reduce with a scalar `__hadd` loop at 0.31 GB/s,
  an order of magnitude under the link it feeds. It is now an SSE2 pass (eight BF16 per iteration:
  the widening is a shift, the FP32 sum rounds back to BF16 nearest-even with the NaN
  canonicalization `__float2bfloat16` performs) that a payload past 32 MiB splits across up to eight
  host threads. Measured at a 10 MiB payload on the 2x RTX 5060 Ti host: the retired loop cost
  25.8-32.6 ms on its own against a 3.1-5.9 ms two-way copy floor, and the collective now completes in
  5.2-8.0 ms; at 50 MiB it is 20.5 ms against the retired 160.8 ms loop plus 14.9 ms of transfers.
- The cross-device all-reduce staging buffer is 48 MiB per device per parity slot (was 24 MiB), and the
  prefill chunk is derived from it rather than clamped only by the model's maximum:
  `DevicePair::in_kernel_allreduce_bytes()` exposes the capacity and
  `TP2GenerationCore::prefill_chunk_width` bounds the chunk by the model's widest all-reduce payload
  per token - `max(hidden_size, GDN value_width, attention query_width)` BF16 columns. A payload past
  the buffer used to change transport silently; the chunk narrows instead, and the draft's feature
  staging and the prefill sinks are sized from the same width.
- The staging reservation walks a ladder (48, 24, 12, 6 MiB per parity slot) rather than asking for the
  full size once. A host whose mapped-host window cannot hold the top rung keeps the in-kernel transport
  at the largest rung both devices map, and the chunk clamp above narrows the prefill to that capacity:
  dropping to the host-staging fallback instead would synchronize both compute streams on every
  collective, which is the larger penalty. The 6 MiB floor still carries a 512-token chunk at the widest
  activation this route reduces. `NINFER_TP2_AR_STAGING_MIB` caps the ladder so a constrained host is
  reproducible from a script.
- The prefill all-reduce overlaps compute. A prefill chunk is split into equal sub-blocks of
  `--prefill-overlap` tokens (default 256; `0` disables) and each sub-block's in-kernel collective
  runs on a second stream while the next sub-block's mixer/FFN keeps the compute stream busy: the
  producer records an event on the compute stream, the collective stream waits on it, and the compute
  stream waits on the collective's done event before the residual add. Measured on 2x RTX 5060 Ti at a
  6.3k-token prompt with a 1024-token chunk, 7 repetitions: **0.587 ms/token with the overlap off
  versus 0.516 at 256 (-12%)**. The width rather than the sub-block count is the knob because the gain
  is non-monotone in width - 320/384 costs 0.612 and 128 costs 0.655, both worse than doing nothing -
  and a chunk is split only when it holds at least two whole sub-blocks, so a width the schedule
  handles badly can never make a chunk slower. At a 512-token chunk the same 256-wide split is
  0.625 → 0.572 ms/token (-8.4%), faster than the 1024-token serial chunk (0.587). The result stays
  bit-identical to the lockstep loop
  (the `ninfer_qwen3_5_tp2_forward_test` chunk-split invariant covers it with the overlap enabled).
- The host-staging fallback's pinned staging is double buffered by call parity. The two devices' compute
  streams are independent, so one shared buffer let the next collective's download overwrite the half the
  previous collective's upload was still sending from: a device could receive a partial sum, which the
  DFlash2 route turned into changed logits and then an illegal address (the pair test hid it because every
  case synchronized between collectives). Growth now synchronizes both streams before releasing the
  previous buffer, which an in-flight upload could still be reading.

## Work log

[TP-2 adaptation work log](maintainer/tp2-adaptation-worklog.md) is the chronological round-by-round record:
design decisions, failed attempts, and the evidence behind the measurements above. It is a
historical log, not an active reference; the current state and the open item are as described in
this note.
