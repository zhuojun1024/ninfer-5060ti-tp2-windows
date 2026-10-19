# NInfer TP-2 Plan (2× RTX 5060 Ti · Qwen3.8-27B NVFP4)

> **The single active plan**, and the persistent memory across context compression. Organized: 2026-09-27 (before
> reorganization the full text went into the worklog; this file retains only
> key information and the tasks that actually need handling).
> - Full history: `tp2-adaptation-worklog.md` — contains verbatim archives of the full text of five PLAN.md
>   revisions (2026-09-21 upstream cherry-pick re-planning revision; 2026-09-22 delivery wrap-up revision; 2026-09-24
>   post-rendezvous-protocol-refactor revision; 2026-09-25 post-r67–r70 draft quantization and transport/orchestration
>   work revision; 2026-09-27 the full text before this reorganization, covering upstream performance-item porting,
>   full-architecture close reading, FP8 PV and cross-tile phase pipelining, two rounds of closed projects).
> - Delivery notes, recommended configuration and measured data: `docs/tp2-dual-5060ti.md`; Windows native port: `docs/windows.md`.
> - Settled-decision quick reference (rejected options, root causes, methodology rules, worklog pointers): `docs/tp2-decisions.md` — must-read
>   before re-investigating an already-settled topic.

---

## 1. Key information

- **Branch**: `feat/windows-native-port` (Windows native + WSL2 dual tree). All 20 commits by which origin/master leads
  have been cherry-picked (frontend jinja chain + three NVFP4 performance items), with no end-to-end A/B degradation.
- **Hardware**: 2× RTX 5060 Ti 16G (SYS topology, no P2P, 448 GB/s per card). WSL2 build tree `$HOME/ninfer`;
  Windows working tree `<repo-root>` (build tree `build-win`).
- **Primary workload**: Qwen3.8-27B NVFP4, TP-2 (weights split in half per shard, locally built half-geometry shape). Draft backends: `--spec dflash2`
  (K=5/7, K=7 is the binary default) or `--spec mtp` (K=2, `--lm-head-draft`); `--spec dflash` (v1) is rejected at construction time.
- **Current status**: delivery goals ①–⑤ all complete; vision, the Windows port, the multi-session KV pool, and the decode graph + AR strategy are all
  landed and measured; DFlash2 is released on TP-2 (unaligned reuse no longer discards the draft); the TP-2 rendezvous protocol refactor is complete
  (host-authoritative id + bounded spin + timeout-failure semantics); the Q4 set of the DFlash2 draft projection has been promoted into the official
  recipe (r69); of these, the selector codebook has been reverted to unquantized (measured at 33k context ≈ −2 pp acceptance, and its 178.09 MiB
  falls in the shard where the selector lives, so it does not constitute a context-capacity bottleneck; evidence in
  [tp2-dual-5060ti.md](../tp2-dual-5060ti.md#draft-precision-and-the-selector-codebooks));
  verify graph B7 accepted (aggregate +5.2%).
- **Key numbers** (262,144 configuration, WSL): prefill 1,588 tok/s; decode 59.0 tok/s (MTP K=2); Windows 131,072 configuration
  decode 56.9 tok/s (on par with Linux). DFlash2 K=7 at 4096 context in the greedy tier ≈ 71 tok/s; 245,760/k8v4 (draftall artifact)
  sweep K=5 96.4 / K=7 96.1 tok/s (interleaved recheck 93.3 / 96.9); roofline-corrected ceiling ≈ 108 tok/s, an upper bound that
  also leaves the proposal step out (the 180 of the originally archived 90–180 is unreachable). The r69 final artifact saves 257.77 MiB relative to r66 (−1.08%), with no measurable cost in
  acceptance (≤1σ, short-context A/B; the long-context 33k recheck shows about −2 pp for the codebook half), at a round-time cost of ~+0.4%
  (proposal step +0.19 ms). After the codebook revert to BF16, that difference returns to ≈79.7 MiB (converted value, throughput not re-measured).

**Recommended runtime configuration** (same recipe for WSL 8088 / Windows 8099, with `--max-context 131072` on the Windows side):

```
./build/apps/ninfer-serve <model>.ninfer --devices 0,1 \
  --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 --port 8088 \
  --spec mtp --draft-tokens 2 --lm-head-draft --vision --reasoning-effort medium
```

`--reasoning-effort low|medium|xhigh` is the process-level default thinking effort (request body takes precedence); `--chat-template` becomes available
with the first-tier cherry-pick (templates are executed by the embedded Jinja). Measured 2026-09-23: DFlash2 (K=5/7) leads MTP by ~20% in decode at a prefill cost within
1% ⇒ for throughput use `--spec dflash2 --draft-tokens 7` (the sweep recipe is 245,760/k8v4; rationale in the archive).

**Final artifacts** (r69/r70, all converted with the official recipe without `--override`, all containing text/vision/mtp/dflash2, with identical draft
profiles across the four: q4 88 / q8 7 / bf16 567): `<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer`,
`…_w4a4_dflash2_final.ninfer` (18.42 GB), `…_swift15_dflash2_final.ninfer` (21.58 GB),
`…_thinkingcap_dflash2_final.ninfer` (22.52 GB).

### Environment and operations essentials

| Item | Value |
|---|---|
| Build (WSL) | `bash tools/tp_bootstrap/r55_build.sh` (rsync + `cmake --build build_dyn -j 8`, success marker `BUILD_EXIT=0`, about 2–3 minutes; the WSL build tree is now `build_dyn` (Ninja), `build_r35.sh` still points at the deleted `build/` and is unusable) |
| Build (Windows) | `tools/win_port/configure.bat` + `build.bat` (VS2022 + CUDA 13.3, `-DCMAKE_CUDA_ARCHITECTURES=120a`) |
| Serving | WSL 8088 (`serve_supervise.sh`); Windows 8099 (`tools/win_port/serve.ps1`, foreground by default; a self-test service must use a harness background job, since a process started by `Start-Process` is killed when the tool call ends) |
| Runtime PATH (Windows) | FFmpeg (`D:\ffmpeg-dev\…\bin`) and libcurl (`D:\curl-dev\…\bin`) must be in PATH, otherwise `STATUS_DLL_NOT_FOUND` (the solo/sessions trio without them fails with `0xC0000135` at load time) |
| artifact | `<models-dir>/qwen3_8_27b_nvfp4.ninfer` (23.7 GB, old artifact, template `c3cf9e34…`); the new official artifact (template `a497db9e…`) becomes available with the first tier; r69/r70 final artifacts see above |
| Process discipline | Only one model process machine-wide at any moment (WSL 8088 and Windows 8099 are mutually exclusive); stop the service before relinking the exe on Windows (LNK1104); `build-win/apps/ninfer-serve.exe` is not automatically synced to `C:\ninfer\` and must be copied manually and compared with `Get-FileHash` |
| Toolchain | Nested `pwsh` → `wsl -e bash -lc` swallows `$var` and redirections ⇒ write commands as script files under `tools/tp_bootstrap/` and execute those; a single blocking call is capped at 600 s ⇒ use background jobs for long tasks |
| GPU/WSL state (from 2026-09-23) | 3 cards visible to the machine: nvidia-smi order 0/2 = 5060 Ti, 1 = Tesla T10; `--devices 0,1` ⇒ CUDA ordinals pick up the two 5060 Ti (misaligned with nvidia-smi order), TP-2 pairing is unaffected. CUDA errors across the board on the WSL side ⇒ testing moved to Windows `tools/win_port/test.ps1`, and WSL build_dyn serves only compile validation |

---

## 2. Closed projects (do not reopen)

> Details in the 2026-09-27 full-text archive of `tp2-adaptation-worklog.md` and in `docs/tp2-decisions.md`; this file no longer expands on them.

- **Upstream ninfer-all performance-item porting**: per-device `cudaFuncSetAttribute` (10 function-local statics + 4 duplicate implementations
  consolidated) and the 170 SM hardcoding (3 places changed to runtime device SM count) are landed; 3a, closing RDC, measured as unsupported and
  reverted; 5b GDN record window staging is ported (Op level −22~27%, not measurable end-to-end); 5a small-T tensor-core on review only covers q4/q5 and does not apply
  to the TP-2 mainline; project 4 (changing the FP8 tensors to be streamed into NVFP4/Q4) was abandoned by user decision. Remaining items in §3.8.
- **Full-architecture close reading**: the only "confirmed latent bug" (`zero_pages` HeadMajor out-of-bounds write) was **revoked** by byte-level review
  (`ne[0]` is the innermost, both plane orders are correct); the full-architecture close reading verified that known defects ①②③④⑧ of the old §3.3 are
  all fixed in the current code; the host-staging allreduce performance cliff has been landed via S1+S2 (fallback 5–8× faster, `kInKernelArBytes`
  48 MiB, chunk clamping derived backwards from staging capacity). Remaining
  architectural hazards in §3.7.
- **FP8 PV (2026-09-27)**: the implementation passed all quality gates (short prompt Δ=+0.29pp, deep context Δ=−0.88pp, both within noise, op-level
  output perturbation about 2×), but was only 1.176× at the op level (target ≥1.3×) ⇒ reverted per the user's pre-authorization rule, and the fp8
  criterion restored to 1.2e-2; the patch is kept as `docs/maintainer/patches/fp8-pv-v1b.patch`. **Key output**: barrier stall 34.2% + math pipe throttle 25.2% ⇒
  the headroom is in the phase structure, not in the PV dtype.
- **Cross-tile phase pipelining (2026-09-27)**: G0 confirmed that on the post-revert FP16-PV baseline the barrier is still the second stall (26.3%,
  of which 97% falls on two full-block `__syncthreads()`); but both resident-role implementations regressed to 0.71–0.80× — the full-block PV accumulator's
  16,384 registers = 25% of the SM register file, so splitting warps inevitably hits the 128-registers/thread wall or the producer working-set wall ⇒ the current
  "8 producer ∥ 8 Vdeq → 16-warp PV" packing is already optimal under the 1 block/SM envelope. **CLOSED, do not reopen**, unless
  the execution envelope changes (e.g. Br=32 brings acc/thread down to 16, or a >100 KB block-level smem budget appears).
- **TP-2 cross-session stable-block reuse: shallow recall swallows the real divergence anchor (2026-09-27 fix + regression scenario)**: the failure symptom
  is "a new session's system prompt cache does not take effect". Root-cause chain: DSH sends title requests in parallel for every new session, and the title shares
  only the thinking preamble (≈41 tok) with the long session, so the long session is left with a shallow mirror (anchor=33), which becomes the only recallable
  boundary for subsequent main requests; and `session_restore` truncates the recalled entry's `tokens` to the recall boundary, with `cached_prompt_tokens_`
  shortened accordingly ⇒ `shared_prefix` is pinned at the boundary, `anchor_position > reuse` never holds, and the real divergence point (end of the system
  prompt ≈9,292) never gets an anchor written ⇒ the first three new sessions each pay a full prefill. Fix (`src/runtime/engine/tp2_generation_core.{h,cpp}`): recall no longer truncates history
  (`frontier` only governs the device KV coverage range); a new `SessionEntry::host_kv_end` records the slab's valid range (shallow recall does not shorten
  the slab); `session_capture_shared_state` allows writing device-resident entries that still hold that prefix's slab; anchor ownership changed from "the previous
  session that was swapped out" to "the entry compared in the reuse scan" (`anchor_session_`); and on store, old mirrors that the slab can no longer cover are discarded.
  Evidence: `ninfer_qwen3_5_tp2_sessions_test` gained a "shallow mirror recall → stable-block-tail anchor" scenario, and the `dflash2`/`mtp` routes are
  all green on `qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer` — W1 reuses only 8 (shallow mirror) while `shared=500`, and the same walk
  anchors 492 into the recalled entry's own slab (`[tp2-session] anchor entry=0 position=492 resident=1 tokens=636`),
  W2 reuses 492; in the same scenario before the fix, `tokens=8`/`cached=8` and W2 reused 8 (A/B reproduced, see the 23:45 run).
  Note: the `plain` route still sits at the pre-existing "shared system prompt first-sample divergence" of §3.3, unrelated to this change (a byte-identical
  failure before and after the fix).
- **TP-2 cross-session stable-block reuse: the second session is still slow (prompt carries its own block-boundary anchor; 2026-09-28 fix + server-side reproduction)**: the previous round's
  "shallow recall anchor" only covered "after the second one" — the first session to share a stable block is itself the one that has to walk the block through, and observation is always one step
  late, so in the user's log the three main requests were still `cache 33`, TTFT 6.2s. This round made two fixes:
  1. **Anchor ownership** (`execute_walk` in `tp2_generation_core.cpp`): instead of only honoring the entry compared in the reuse scan, scan all
     host-resident entries, and pick the one sharing the deepest prefix with the inbound prompt whose slab covers `shared - margin` (validated by `host_kv_end`),
     excluding `anchor_session_` itself; when the shallow mirror sits on the title entry and the deep history on the main entry, the anchor thus lands on the deep entry.
  2. **The block boundary comes from the prompt itself**: `Frontend::prepare_context_cache` unconditionally writes the frontier of the leading instruction block
     into the new field `PreparedContextCache::leading_instruction_frontier` (independent of `allow_engine_automatic_shared_prefixes`,
     and the mark policy of the single-card context cache is unchanged); the TP-2 core forces `frontier - margin` to be a chunk boundary during prefill and writes it
     into the new block ring slot (`kReuseBlockCheckpointCount = 1`, outside `--host-state-slots` just like the divergence slot),
     recording `block_anchor_position_/block_anchor_prefill_id_`; when a session is swapped out (`session_store_active`), it finds that checkpoint by
     position+prefill_id and copies the state (including the DFlash2 draft mirror) into that entry's `host_shared_state`,
     making it the recallable boundary for the next session of the same family (`host_shared_end`).
  3. Evidence: server-side reproduction (`build-win/apps/ninfer-serve.exe`, `--max-context 65536 --host-state-slots 32`, swift15
     artifact, `--prefill-chunk 1024`, three sessions each with one title request): main request 1 = 7,037 tok full prefill, TTFT 4.4s; main request 2
     reuses **6,145/7,039 (87.3%), TTFT 817ms** (the anchor lands on a 1024-aligned chunk boundary, so what is frozen is "the state walking from scratch would also
     reach", at the cost of at most one chunk of reuse depth); main request 3 reuses 7,020 (99.8%), TTFT 172ms (taking the
     observational anchor left by the previous round's walk). Trace: `[tp2-session] anchor entry=2 position=6145` (transferred on swap-out) →
     `[tp2-reuse] prompt=7039 shared=7028 -> reuse=6145 src=live`. Regression: `ninfer_qwen3_5_tp2_sessions_test` gained a "prompt carries its own block boundary → swap-out transfer → same-family session reuse ≥90%" scenario, `dflash2`/`mtp`
     all green, `plain` still at the pre-existing first-sample divergence (unrelated to this change).
  4. Operations: the user's `--max-context 262144` pushed device memory to `free 0.0 MiB`; on this machine as currently configured (desktop/browser occupying the GPU), even the **old binary**
     OOMs at startup the same way (lowering `--host-state-slots` to 31 so that pinned matches the old version still OOMs), which is unrelated to this change; lower it to
     `--max-context 131072` or free up GPU usage when needed.
  5. **Short sessions do not occupy a quota slot (2026-09-28, follow-on fix)**: each DSH session produces two entries (title ~160 tok + main ~10k tok),
     so `--max-private-continuations 3` in practice only retains about one session, and hence session A's continuation (req#7) got `cache 0`. Added
     `ContextCacheOptions::session_retention_floor_tokens` (engine default 0 = retain all) + server default 2048 +
     `--session-retention-floor` override; `session_publish` neither creates nor updates an entry for histories shorter than the threshold (requests are still served as usual).
     Pitfall: TP-2 normalization **rebuilds** the `ContextCacheOptions` allowlist fields in `model_instance.cpp`, and the new field was dropped at the time
     (manifesting as the rule not taking effect) — it has been added to that initializer list. Verification (full rebuild + server-side A/B): after the warmup/title requests
     `entries=0` (no quota slot taken), the topic request `reuse=3` goes through the ring; req#4 `cache 5,121/5,609`(91.3%)/TTFT 468ms,
     req#6 `cache 5,590/5,606`(99.7%)/TTFT 126ms. Also: this repository's incremental dependency scan is unreliable, so after changing `include/ninfer/types.h`
     or an engine header you must do a `--clean-first` full rebuild (both false failures this round were caused by this).
  6. **Session entry lifecycle (2026-09-28, same-round fix)**: the threshold rule of item 5 removed the title entry next to each new session,
     and the "update the resident entry in place" branch had depended on it as the victim — so the next session directly overwrote the previous session's entry, and A's history was
     successively erased by B and C (server-side reproduction: `entries=1`, back to A `reuse=0`). Three fixes: ① `session_publish` only updates in place when
     `history` has the previous entry's tokens as a prefix (an extension of the same session), otherwise it creates a new entry; ② a new
     `mark_device_resident` enforces "at most one entry claims residency" (previously two claimed residency at once → the eviction scan could only pick the entry that had just frozen
     the boundary, which turned the engine test red); ③ `session_capture_shared_state` refreshes `lru_clock` after success (written means used, so
     a just-frozen boundary is not evicted in the same round). Verification: `dflash2`/`mtp` all green; server-side A→B→C→back to A reproduces `entries` 0→1→2,
     back to A `reuse=5590/5625 (99.4%)` (before the fix, `reuse=0` full prefill).

- **Vision aggregation ceiling 32,768 → 131,072 merged tokens (2026-09-28, user request)**: the symptom was that the user's 1080p screenshot session was
  rejected with `HTTP 400 media_budget_exceeded: vision budget exceeded` at "16 already in history + 8 added this round = 24 images".
  Root cause: the whole-request aggregation upper bound `kMaximumPromptVisionTokens` (`src/models/qwen3_5/frontend/prepared_prompt.h`) was
  32,768 merged token (= 131,072 raw patch = 33.5 MP aligned pixels), while the client resends the entire session history every round and the images in the history are
  counted again every round ⇒ effectively equivalent to "at most 16 1080p images / at most 37 720p images per session", and once over the limit every subsequent request in that session got 400.
  The change only raises the aggregation upper bound (the per-item ceiling of 16,384 and the encoding workspace planning are unchanged ⇒ **device memory usage unchanged**):
  `kMaximumPromptVisionTokens = 131'072` and the comment now states that it only determines "the retained patch budget / media live lower bound / admission check";
  in sync, `docs/maintainer/qwen3_5-model.md` (524,288 raw patch / 131,072 merged token), `docs/serving.md`
  (new envelope + `--media-live-mib` requires ≥ 12,288 B/merged token = 1.5 GiB, and the default 2048 satisfies it),
  and the fixture in `tests/test_request_log.cpp`. Evidence: after a `--clean-first` full rebuild (769 steps)
  `ninfer_request_log_test` / `ninfer_qwen3_5_frontend_test` / `ninfer_public_api_test` pass;
  `ninfer_qwen3_5_vision_workspace_test` `cudaMalloc` OOMs on this machine's 16 GB card (the worklog already notes "not applicable on this machine", and its assertions
  concern only the per-item upper bound, unrelated to this change); end-to-end (`build-win/apps/ninfer-serve.exe`, 1920×1080 image = **2,040** merged
  token each): 24 images → HTTP 200 (prompt 49,022), 64 images → HTTP 200 (prompt 130,702), 65 images → HTTP 400
  `media_budget_exceeded` ("vision raw patches exceed processor budget") ⇒ the boundary falls exactly at 131,072.
  Cost: the host-side media live lower bound goes 384 MiB → 1.5 GiB, and a single request worst-case retains 1.5 GiB of BF16 patch (within the 2 GiB live budget the media cache
  shrinks from ~1 GiB to ~512 MiB); KV/context is still determined by `--max-context`. Deployed `C:\ninfer\ninfer-serve.exe`
  (SHA256 `29D8758E548E581B4BAF38EDD6C56E5451F43CDC3E2F1615EA6DF4523708F2C1`, identical to the `build-win` artifact).

---

## 3. Pending tasks

> Ordered by priority/actionability; closed projects in §2.

### 3.1 Pre-delivery wrap-up

- [ ] Re-test concurrency C=1/2/4 and streaming/stop smoke with the recommended configuration (the conclusions of Round 31/34 are based on the bf16 configuration).
- [ ] On delivery, remove this file (PLAN.md) per AGENTS.md; decide the keep/cleanup scope of the ~300 one-off diagnostic scripts under
  `tools/tp_bootstrap/` (`build_r35.sh`, `serve_*`, the `r37+r38+r39+r4x` series, and the `docs/tp2-dual-5060ti.md` references as
  the reproducible procedure).

### 3.2 Windows single-card CLI tiny artifact subtask (user decision; prerequisites ready)

Goal: produce a `.ninfer` small enough to fit on a single 16 GiB card and get "load → generate → sample" working on a single card with the Windows `ninfer.exe`.
Prerequisites (Round 54.9): CPU-only torch in `build-win\torch-venv` (torch 2.14.0+cpu); the converter interface is confirmed.
Next steps: ① run the tiny checkpoint directly with the official recipe (`--recipe qwen3_8_27b_nvfp4 --components text`);
② if that fails, write a tiny recipe/override; ③ put the generation script at `tools/win_port/tiny_model.py` (synthetic tiny config +
random bf16 weights, HF naming); ④ on success, document in `docs/windows.md` and commit.

### 3.3 Unfixed defects and known limitations

- The TP-2 route with `bf16` KV + `--spec mtp` crashes at the first prefill (`prompt.cu:63` `cudaErrorInvalidValue`);
  fp8 + MTP is fine (the recommended configuration is unaffected). Investigation entry point: the launch parameters of bf16 prompt-attention.
- On the MTP route, around `execution/text.cpp:486`, the allreduce that adds in BF16 likewise carries I32 proposal ids ⇒ when a value lands in a
  signaling-NaN bit pattern it is silently rewritten (DFlash2's equivalent exchange has already been changed to `DevicePair::sendrecv` byte-for-byte; MTP has not been changed, and
  verification is costly).
- The MTP3 ≥ 70 tok/s threshold is not met (pure-decode ceiling K=2 50.5–54.0).
- The TP-2 route has not run a perplexity evaluation (the quality evidence is same-prompt multi-sample A/B).
- The per-shard arena's `memory_summary()` still reports `pages 0/0` (a display-only reporting issue).
- Temperature 0 does not guarantee bit-exact argmax: DFlash2 sparse acceptance + a per-request random seed (`translate.cpp:51-57`) make decoding
  nondeterministic even at temperature 0 ⇒ add `--greedy` when a reproducible A/B is needed (existing behavior, unchanged).
- The `plain` route of `ninfer_qwen3_5_tp2_sessions_test` diverges at the first sample for "a conversation behind a shared system prompt":
  a pre-existing failure (reproduced token-by-token on the r57 baseline artifact and the r66 q4all artifact; the dflash2/mtp routes pass) ⇒ it blocks "all three green" (the sessions
  gate currently runs on the dflash2 route).
- The first sample of TP-2 prefill does not apply the tool syntax mask (prefill and decode paths are asymmetric; a separate outstanding item).
- Prefix reuse: text-protocol replay cannot guarantee token-by-token reproduction (client re-tokenization and sampling boundaries differ, or the boundary falls inside a chunk — the contract only guarantees boundary crossing); the only lever guaranteeing token-by-token identity is a continuation protocol carrying the token id (not scheduled).
- Two low-frequency failures (unrelated to the change, all pass on re-run, investigation in §3.6): ① in r67 the q4all arm K=7 `cudaErrorIllegalAddress` (req#14); ② in r69 the final arm K=5 first run req#14 HTTP 503 with no FATAL log.

### 3.4 Deferred / optional (not scheduled)

- Round 15 remaining: tool-call-constrained real-model end-to-end re-test (75 probes); WARN diagnostics; route B (full GBNF); TP-2 `validate_licensed_tokens` guard.
- Round 14 candidates (user deferred): ① append an anchor at the end of the system+tools prefix (fixes the 46.9 s first round after a new session/compaction, saves about 11 s); ② prefill throughput (suffix 0.95–1.13k tok/s, cold start 1.35–1.38k); ③ per-request ~0.3 s fixed overhead.
- Archive §11.6: MTP priming uses a placeholder embedding for image columns (no evidence of benefit); single-image cap hardcoded at 8,192 (yielding shard 1 can reclaim a larger envelope); shard 0 idle during visual encoding (acceptable).
- Archive §6: KV dtype sweep completion (nvfp4/k8v4 only verified startup and throughput, no quality data).
- To faithfully reproduce the official artifact's DFlash2 draft from the source weights, its BF16 source checkpoint must first be obtained (this machine only has FP8/EXL3/GGUF).
- L2: AR and MMA overlap — **implemented and measured: +11% prefill (N=4)**. Implementation: the prefill chunk is split into N sub-chunks (64-aligned), each sub-chunk's collective goes on a second stream (`DeviceContext::collective_stream`), overlapping with the next sub-chunk's mixer/FFN (`run_layers_tp2_overlap`, option `--prefill-overlap W` = sub-chunk width, default 256, 0 disables). Measured (chunk 1024, ~6.3k prompt, median of 7 reps): N=0 0.587 ms/token, N=2 0.554, **N=3 0.612 (worse than baseline)**, **N=4 0.516 (−12%, reproduced in two batches)**, N=5 0.547, N=8 0.655 ⇒ the optimum is indeed at **N=4 (sub-chunk 256)**, and **N is not interpolable** (320/384 land in a slower scheduling tier); the tiering should be set by "sub-chunk width = 256" rather than extrapolated by N (chunk 512 re-check: serial 0.625 → 256-wide overlap 0.572, −8.4%, and faster than the chunk 1024 serial baseline 0.587); numerically bit-identical to the single chunk (tp2_forward_test's chunk-split invariant remains exact at N=2/N=4). Three pieces of prior evidence: the divisor arm in the 2026-09-27 archive §5.4 (when AR is 48% of a strictly serial chunk), `tools/tp_bootstrap/bench_ar_overlap.cu` (AR is not starved by compute that saturates the SMs, inflating only 7–18% at engine granularity), and Round 12's −5% being merely a problem of its host-side scheduling shape. **Remaining**: choice of default/N, independence from the decode path, and the failure semantics of cross-card rendezvous (see §3.6).
- If attention is still to be attacked: FP8 PV and cross-tile phase pipelining have both been tried and disabled (see §2) ⇒ under the current execution envelope there are no untried attention levers; continuing requires first changing the envelope (larger smem budget / different Br) or unlocking profiler permissions (ncu 2025.4.1 is installed, `ERR_NVGPUCTRPERM` is refused; administrator PowerShell or NVIDIA Developer Settings allows GPU performance counters).
- Hardware lever: move card 1 from the chipset Gen4 x4 slot (~7 GB/s) to a CPU-direct Gen5 x8 slot (~20 GB/s): zero code, 1.78× on shallow/medium context, 1.15× at 245k; requires a free slot on the motherboard.
- DFlash2 K=6 per-round cost dip (a constant 25.7 rounds/s across three rounds, below K=5 (27.9) and K=7 (26.6)): suspected kernel boundary effect from bucketing/alignment by window width; use `NINFER_TP2_TIMING=1` to obtain per-phase per-round timings for investigation.
- Exact source of DFlash2 prefill being consistently 3–4% slower (no per-phase attribution done; `NINFER_TP2_TIMING=1` to break prefill down per phase).
- B7 tiering: verify graph A/B at 6 reps/arm (aggregate +5.2%, per-request t=0.99 not significant); ≥24 reps/arm is needed to set the tier.
- 1a residue: fold the actual arguments on both sides into one struct (the two call sites are already explicit and consistent; diminishing returns).
- Items not actually measured during the DFlash2 TP-2 period: the real free under a Windows native build (the ledger `free` comes from `cudaMemGetInfo`, WSL2 under-reports by about 1 GiB); numerical equivalence of drafting by head/row split; K=15 workspace peak; per-step overhead of gathering the target's 5-layer residual across shards; latency of the per-layer allreduce for drafting; equivalence of sparse rejection sampling under TP-2 sharded logits.
- Gate revision decision (separate from the promotion): the literal gate "greedy |Δacc| ≤ 1.0pp" was not passed at K=7 (−1.60pp, proven to be confounded by cross-arm text drift); the promotion was completed per user confirmation (r69), with 30-rep sampling as the de facto deciding method — if the gate is ever formally revised, do it in this one place.

**Design constraints (do not re-investigate)**: copying the whole draft is infeasible — at full context (262144 / fp8 / MTP K=2) free is only 697/1217 MiB, while the draft is 2.07 GiB/card; after splitting it is about 1.04 GiB per card. The proposed condition is a residual concatenation projection for the target's 5 blocks; TP-2 splits the 64 layers across two cards ⇒ a cross-card handoff is needed (the two cards' residuals are bit-identical, so capturing shard 0 alone suffices). The candidate selector reads the full-vocabulary codebook directly (about 254 MiB), which must be complete. DFlash2 and MTP are mutually exclusive: choosing DFlash2 frees shard 0's MTP weights 430 MiB + KV 516 MiB. Per-round cost composition, D1–D4 decisions and the negative conclusions of each round are in the archive.

### 3.5 Environment-blocked items (already covered by alternative paths, no further action needed)

- WSL-side op tests + byte-identity + frontend fixture tests: **WSL2 CUDA driver crash** (PTX JIT segfault inside `cudaGetDeviceCount()`, GPU held by a Windows service); the jinja tests passing proves the binary is fine, and the op tests were moved to the Windows side. The Windows side has passed (NVFP4 A4/A16, frontend, jinja; `NINFER_OP_REPORT_STATS=1` error metrics within tolerance) ⇒ the WSL build_dyn is only compile validation.

### 3.6 TP-2 transport low-frequency failures (remaining)

- **Characterization (proven)**: after a bounded-spin give-up, the partial sums of this round's collectives exist **only locally** (the peer's contribution is missing), while the round keeps running; when the give-up falls **inside a captured window**, the window's own kernel uses the inconsistent data as an **address** ⇒ illegal address, and no host-side check can be in time (injecting 700 reproduces byte-identically); only a give-up in eager prefill comes out as a clean 503 + recovery on the next request.
- **Trigger source undecided**: natural long runs (cumulative 1500 requests, watchdog enabled) show 0 events, natural incidence ≲1/1500; it requires a >2 s host/device delay, or a real desync (id reuse/mismatch). The watchdog is enabled by default; on failure it separately saves `*-FAILED.log` / `*.arwatch.txt` (interpretation by the magnitude of the id difference: a whole-group 0 = that side never reached the arrival write; 1 = OFF-BY-ONE; whole-block = BLOCK-LEVEL; otherwise SKEW).
- **(b) Making give-up safe (pending decision)**: the host-side check has been falsified; either poisoning/aborting semantics after a give-up, or range-constraining the inputs of in-window indexing operators — both are architecture-level changes and require first pinning down the exact operator.
- **Tooling boundary**: `CUDA_LAUNCH_BLOCKING=1` / compute-sanitizer are incompatible with this transport (the A side's launch becomes blocking, the B side's kernel only launches after it returns ⇒ the 1st collective necessarily gives up, self-deadlock) ⇒ launch-serializing tools cannot be used for kernel-level attribution; only source-level probing with binary search is possible.

### 3.7 Architecture-level hazards (long-term, not scheduled)

- **Dual execution core drift risk**: TP-2 goes through a separate `TP2GenerationCore`, bypassing the EngineCore scheduler/CUDA Graph/ResourceManager (single-request lockstep). Request lifecycle, capacity, cancellation, prefix reuse, checkpoint ring and similar logic are implemented twice across the two paths and must be kept semantically consistent long-term — this has repeatedly produced prefill/decode asymmetry defects (tool mask, first-token timing, MTP ids swap are all instances). The largest maintainability hazard.
- **TP-2 in-kernel allreduce give-up safety is open**: the host-side check has been falsified, see §3.6(b); "making give-up itself safe" is an architecture-level change and requires first pinning down the exact operator.
- **host-staging CPU allreduce fallback**: S1+S2 already turned it from a performance cliff into link-limited; the graph degradation (EagerExact) is retained — host synchronization is inherent to the fallback, and the only degradation not attributable to the WSL2 environment is "no graph".
- **TP-2 prefix reuse capacity hard cap**: the device reuse snapshot fixes `kReuseSnapshotCount=2` (73 MiB/shard per slot), and deeper divergence relies on the host checkpoint ring (one PCIe round trip); the reuse cost in deep-divergence scenarios is structurally raised.

### 3.8 Remaining items from porting upstream ninfer-all

- **Upstream `origin/master` `e31bc99b` (2026-09-26) / `dev` `a012e2bc` (2026-09-28) performance items (2026-09-29 source assessment): no TP-2 merge value** — linear unification + sliced-K, nvfp4 W4A4 TMA, q4/q5 dispatch all show no benefit (decode is weight-stream bandwidth-bound; the main GEMM's half geometry goes through MMA, not TMA; mainline NVFP4 is not q4/q5); the Jinja chat template is already merged (including local enhancements). The decision and commit ids are in the "Upstream merge-value assessments" of [tp2-decisions.md](../tp2-decisions.md); do not re-evaluate until upstream passes `e31bc99b` and touches the mainline.
- **Upstream `origin/master` = `dev` = `68c54356` (2026-10-05 assessment, baseline move triggered re-evaluation)**: of the 73 commits after crossing the line, only the jinja cache boundary fix `68c54356` has merge value, and it has **been ported** — the source byte ranges folded away by `|trim`/`strip` are now retained as `TemplateOutput::boundary_mappings`, and `MessagePartBoundary` / `LeadingInstructionBoundary` breakpoints are no longer silently dropped by `prepare_context_cache` (the production templates `tools/chat_templates/qwen3_{6,8}.jinja` apply `|trim` to every message's content and `reasoning_content`, and multi-line system blocks and `tool_result` endings carrying a newline are the norm). The change touches `third_party/llama-jinja/jinja/{string.h,string.cpp,value.cpp}`, `src/text/jinja.{h,cpp}`, `src/models/qwen3_5/frontend/prompt_layout.cpp`; the regression cases are folded into `tests/text/test_jinja.cpp` and `tests/models/qwen3_5/test_frontend.cpp` (20 expression scenarios × system/user breakpoints). In the same assessment batch, `75a89050` was also ported: `abort()` and the `start_request` rollback synchronize the compute stream before returning lane buffers/pages/execution rows (locally the synchronization is placed before `retain_aborted_continuation`, because cataloging a cancelled continuation reads the same batch of device buffers), and the same class of gap is closed on the TP-2 route (`drive_lane_queue`'s whole-batch failure path drains both shard streams before returning each lane's KV pages). The rest remain reference implementations: `b9114396` context cache + preemption rewrite (different product contract), the numerical policy in the SM derivation refactor (rope/rmsnorm were ported on 09-26; the MoE prefill grid cap and attention split budget await 5060 Ti A/B), and the attention/linear half-geometry perf batch (following the 09-29 no-benefit criterion). Decisions and local deviations are in [tp2-decisions.md](../tp2-decisions.md).

- **3b attention gate folded into the reduce epilogue** (upstream `77c9cc39`): deferred — one fewer launch per layer ≈0.08 ms, 0.2% of the 38.2 ms round; benefit below risk.
- **3c INT8-family attention scale via shared memory** (upstream `a941c9f4`): deferred — production uses `--kv-dtype fp8`; in this tree, disabling RDC actually increases small_t_i8 stack frames, so SASS attribution is needed first to determine whether out-of-line calls really exist.
- **5060 Ti A/B timing of SM-derived values**: rmsnorm threshold, rope block selection, chunked GDN wave target, and the small_t 160/320 targets (the latter changing derived values would change existing 5090 behavior, so a 5060 Ti A/B with batch_size>1 is required first).
- **Project 6 major items (need product decisions, not changing yet)**: KV codec (`eb9f7a23`/`ad26b362`/`2ba10da2`), device route profile (`0b2f8384`+`81861a07`, 36-SM cards need recalibration), fast prompt kernel (`4303e604`).
- **Converging unconditional `cudaFuncSetAttribute` at launch**: `context_kv_materialize/materialize.cu`, `linear/gguf/ggml_bridge_mmq.cuh`, `linear/nvfp4/nvfp4_w4a4_tma.cu` and others still call the driver on every launch; correct but worth converging along with timing evidence.
---

## 4. Round 54: four improvements to the TP-2 session cache (implementation plan)

> Source: anthropic 0-hit diagnosis (stream-500 → session_invalidate_active chain) + comparative analysis of single-card vs TP-2 cache mechanisms (upstream 4201b5d2).
> Baseline branch: feat/windows-native-port @ 92d7ab57 (the running server C:\ninfer\ninfer-serve.exe is from the same source; local master has only 1102 lines of early TP-2, do not use).
> Status: **plan organized, not implemented.** ④ can land first and independently (production pain point).

### 4.1 Environment constraints

- Working branch feat/windows-native-port; git checkout that branch before implementing.
- The two 5060 Ti are occupied by the production server (:3456) → integration tests need a service-stop window (confirm with the user first): test binary takes :3456 → probe → restore.
- Probe scripts use a _temp/ timestamp prefix and are deleted on success.

### 4.2 ④ count-tolerant anthropic stream encoder (smallest, independent)

Goal: an adoption splice changing the prompt count no longer 500s, and session state is no longer destroyed.
- src/serve/anthropic_messages_response.cpp L225-230: in start(generation) remove the throw and set input_tokens_ = generation.prompt.prompt_tokens; (streaming_start_usage's input - reused arithmetic is automatically correct).
- Audit the final usage (L91/L352 message_delta uses outcome.*): start and final chunk agree (both are the engine count).
- Audit the analogous throw for prompt_progress in openai_chat_response.cpp L287-295: return_progress_ is off by default → record as a known limitation; make it equally tolerant if the change is small.
- Out of scope (recorded): deeper fixes so that a serve render failure does not destroy engine state.
- Test: 1620-scenario probe (divergent echo stream → 200 + cache_read = reuse after splice; a subsequent non-stream then hits its own boundary); message_start and message_delta usage.input_tokens agree.

### 4.3 ① session_key + retention weight (engine-layer foundation)

Goal: the TP-2 catalog entry carries a session key and a retention class; eviction is weight-based.
- src/runtime/engine/tp2_generation_core.cpp:
  1. SessionEntry + std::optional<CacheSessionKey> session + weight (with key = LiveSession 16, without = RecentPrivate 4; constants aligned with the generic cache's private_retention_weight).
  2. session_recall: matching stays a pure token scan; after a hit, update the binding according to incoming hints.
  3. session_publish: take the key from data.context_cache().session_key (the plumbing already exists); if another place with the same key is already bound → rebind + demote the old entry by 4 (aligned with the generic cache's publish_session/demote_replaced_session publication_order rules).
  4. session_evict_one(): weight-aware LRU (evict lowest weight oldest first; when all are 16, evict oldest).
- No serve-layer changes (openai-responses already passes the key; anthropic synthesizing a key is a separate follow-up).
- Tests: unit (eviction order 16 vs 4 / same weight oldest / all 16 oldest); integration (openai A for 2 rounds + pressure B..G, A survives and hits on the next round); regression (all 4 anthropic behaviors unchanged).

### 4.4 ② Capacity-aware admission (eliminate silent drops)

Goal: when host KV is insufficient, explicitly reject (503 + Retry-After) rather than silently dropping the session.
- session_store_active() (L1751): feasibility check before D2H.
- Switch path (drop_previous): on failure first retry after freeing a slab via ① weight eviction; still failing → a new exception ContextCapacityUnavailable thrown from submit()/execute_walk — incoming is rejected, **the current resident entry is not destroyed** (key invariant).
- generation_service: catch → HTTP 503 + Retry-After: 1, label "context capacity temporarily unavailable".
- Startup capacity check: session_capacity_ > number of host full-size sessions + 1 → operational log WARNING.
- No new flag. Protocol contract change → schema tests + docs/serving.md updated together.
- Test: after N ( > host capacity) large sessions, the N+1th concurrent switch → 503 (not silent); recovery one by one after idling; rejection does not destroy the resident (verified via logs).

### 4.5 ③ Shared-prefix entry (two phases)

Goal: multiple sessions sharing a stable prefix (system prompt + tools) cache it once and are protected.
- Phase A (retention protection only, do first):
  1. SharedPrefixEntry: token prefix (digest key) + refcount + dual-shard host slab replicas; refcount>0 is not evictable, ==0 is evictable oldest-first.
  2. session_publish detection: shares >= --shared-prefix-floor (new flag, default 4096, 0 = disabled) with an existing entry → create/update a shared entry at the divergence boundary, refcount += 1.
  3. Eviction releases a reference; evictable once refcount → 0.
  4. recall unchanged (each entry materializes its own slab); the shared entry doubles as a recall anchor for new sessions.
- Phase B (storage dedup, optional): the entry stores only suffix KV; recall = shared prefix H2D + suffix H2D; grid covers the suffix; snapshot at the divergence point. Only if multi-agent deployments prove A insufficient.
- Test (A): two sessions with the same 30k system+tools → 1 shared entry; after eviction pressure, a third session with the same system can stand on the shared boundary; refcount lifecycle.

### 4.6 Order and dependencies

④ (independent, serve-only) → ① (engine foundation) → ② (depends on ①'s weight eviction) → ③A (depends on ①②).

### 4.7 Risks

- ② changes the serve error surface → protocol contract update (schema tests + docs).
- ①'s rebinding semantics must match the generic cache publish_session to keep behavior consistent across modes.
- ③A publish adds bounded token comparisons (within catalog scale, negligible).
- TP-2 files are locally unique (upstream has no TP-2) → no upstream merge conflicts; serve-layer files are evolving upstream → do a rebase check on the serve layer before implementing.

### 4.8 Progress and measured results

- [x] **④ stream encoder tolerance** —— implemented (`anthropic_messages_response.{h,cpp}`) + unit test cases (`test_anthropic_schema.cpp`) + on-machine A/B.
- [x] **① session key + retention weight** —— implemented (`tp2_generation_core.{h,cpp}`: SessionEntry.session/retention_weight, bind_entry_session/demote_session_owners, weight-aware session_evict_one).
- [x] **② capacity-aware admission** —— implemented, but **the planned 503 rejection was not adopted**, see below.
- [ ] **③A shared-prefix entry (retention protection only)** —— not implemented (design in 4.5; larger in scope, see below).
- [ ] ③B storage dedup (optional) —— not implemented.

#### ④ On-machine A/B (decisive reproduction)

Deterministic trigger: at template render time `reasoning_content` is trimmed but the body is not, whereas `same_rendered_turn` trims both; therefore **adding trailing whitespace to the echoed assistant body** makes the same turn look equivalent to the engine while the tokenization and generated bytes differ → triggers `adopt_generated_turn` splicing → prompt count changes.

| Round | Old binary (C:\ninfer\ninfer-serve.exe.bak-20260929-1722) | New binary |
|---|---|---|
| turn 1 | 200, cache 0 | 200, cache 8375 |
| turn 2 (splice) | **SSE error: "Anthropic stream prompt count differs from Engine start"** | 200, cache 8441 (99.7%) |
| turn 3 | 200, **cache 0** (entry destroyed by the 500 → full re-prefill) | 200, cache 8375 |

That is: the root-cause chain of the production 0-hit loop is fully reproduced, and it disappears after the fix. Unit test `ninfer_anthropic_schema_test` passes (including the added splice case).

#### ① On-machine verification (named session vs anonymous pressure)

12 anonymous anthropic sessions (no session key, weight 4) + 1 named openai-responses session (store:true, weight 16, clock=1) fill the catalog (9 slots):

- All 4 eviction traces are `weight=4`; the clock=1 named session was never selected (direct evidence that the weight works).
- The named session's turn2 recall is `cached=8460/8485 = 99.7%` (it survives after the pressure and can still be recalled deeply).

#### ② On-machine verification (`--host-kv-mib 2048`, 1024 MiB/shard)

- The catalog self-limits to the 7 entries the host can actually hold (rather than the catalog cap of 9).
- All 13 requests return 200; all evictions are `weight=4`; **zero** `[tp2-session] drop ... alone exceeds the host session budget`.
- The named session still hits at 99.7%.

#### ② Deviation from the plan (to be recorded)

The plan originally was: store fails → evict by weight and retry → still fails → throw `RequestError(Unavailable)` → 503. During implementation we found: **the eviction loop drains all evictable entries**; failing after the loop ⇔ only the resident itself remains ⇔ it alone exceeds the host budget (e.g. `--host-kv-mib` smaller than a single session's capacity). Rejecting at that point would cause a **permanent failure in which every switch returns 503** (because no switch can succeed, and therefore no entry is ever evicted). So it was changed to:

1. store fails → evict by weight and retry (covers the genuinely common "host filled up by weak sessions" case, which is the main cause of silent drops);
2. still fails (the resident itself is over budget) → keep the original drop, but **unconditionally print** `[tp2-session] drop ...` to stderr (into the server operations log), no longer silent.

Also fixed an index defect found during implementation: erasing entries inside a loop in `session_evict_one()` shifts the vector, so the captured `previous` index becomes stale; changed to re-read `active_session_` after the loop (`session_drop` maintains it).

#### ③A Why it was not implemented

- The status quo already has **partial coverage**: TP-2's per-entry divergence image (`host_shared_state`/`host_shared_end`) already provides a recall anchor for cross-session shared prefixes; in measured traces `reach=8448 via=shared` is visible. ③A's incremental value is "the shared prefix still survives after the sessions are all evicted" (session-independent shared entries).
- That increment requires new storage and recall branches (shared entries and their host slabs, divergence-point snapshots, a recall branch sourcing from shared entries, reference-counted lifetime), a fairly large change; after ④①② fixed the production zero-hit and silent-drop issues, scheduling it separately is safer (avoids a half-finished feature entering the serving engine).
- Suggested next step: first implement the "session-independent shared-prefix checkpoint pool" per 4.5 phase A, reusing the existing host checkpoint restore-then-prefill path rather than duplicating SessionEntry's entire host state.

#### Pre-existing failures (unrelated to this round's changes, attributed)

`ninfer_resource_manager_test` (candidate-stratified reuse closure) and `ninfer_serve_options_test` (server reasoning-effort default) **fail at baseline** on this branch: after `git stash`-ing all of this round's changes, rebuilding and running them, both output exactly the same failure messages. The former is precisely the single-GPU context cache eviction/reuse closure selection issue and can serve as a lead for an independent single-GPU-side defect.

### 4.9 TP-2 reuse collapse every other round: adoption's coordinate shift (fixed)

**Symptom (production log `C:\ninfer\serve-win.log`, two anthropic sessions switching back and forth twice)**: all 31 requests succeeded, no zero hits, but the same session's growth phase **strictly alternates**: `private endpoint` (98-100%) ↔ `long anchor` (83-93%), and the shallower round prefills 4,000–9,500 extra tokens.

**Reproduction (rebuilt from the exported session's real systemPrompt + 30 tools, `_temp/20260929-1955_tool_replay.mjs`)**

| Round | Cache before fix | Cache after fix |
|---|---|---|
| 1 | 0 (full) | 0 (full) |
| 2 | 51,917 | 51,852 |
| 3 | **51,200** | **51,969** |
| 4 | 52,151 | 52,109 |
| 5 | **51,910** | **52,212** |
| 6 | 52,381 | 52,386 |

**Root cause (trace evidence: `shared`, `replay_split`, `adopted`, `prev_prompt`)**

Every other round shows `adopted=1`: `adopt_generated_turn` replaces "the same turn re-rendered by the client" with tokens generated by this lineage. When the client render **already covers the entry's entire history** (`shared == cached`), this replacement **does not deepen any reusable boundary** — the scan ceiling is `shared`, and the deepest boundary (frontier = `turn_end - 1`) is already within `shared`. Its only effect is to shorten the prompt and shift the entry's coordinates forward as a whole.

So in the next round the client resends at its own coordinates, the common prefix reaches only `replay_split` (e.g. 52,152, shallower than the `prompt_end` 52,177 recorded in the previous round) → `resident_continues` fails → falls into `switch` (D2H the session to host, `live_state_valid_=false`) → re-prefills from the shallower host checkpoint. **`switch` is only the symptom, not the cause.**

**Fix**: `adopt_generated_turn` only performs the replacement when it can deepen the reachable prefix — `if (shared + 1 >= turn_end) { return adoption; }`. When the divergence falls inside the answer (`shared < turn_end - 1`), it still replaces as before, and in that case it genuinely advances the reachable prefix from `shared` to `turn_end`.

**Measured**: the trace shows `continue` throughout, `adopted=0`, `src=live` every round (hitting the frontier directly), `shared == cached` (the entry stores the client's own tokens, no longer shifted).

**Rejected alternative (recorded to avoid retrying)**: change the session-continuation threshold from `prompt_end` to the "client render agreement point" (`adoption.divergence`, already computed every round). Measured, it did make every round take `continue`, but **the reuse depth did not change at all** (the shallow rounds were still 51,200 / 51,906, because the scan ceiling was still `shared`), and it introduced a hazard: the first round's entry may have `divergence` of only 1, degrading the judgment to "any prompt counts as a continuation", so when another session arrives it **will not** store the resident to host; and once an entry loses residency and has no host copy (`host_kv_end=0`, `frontier>0`), `session_recall` still treats its frontier as a candidate and `session_restore` dereferences an empty `host_kv`. Reverted.

**Known residual**: when the divergence really falls inside the answer (long-answer scenarios, where adoption has real benefit), the replacement still shifts the entry coordinates forward, and the next round's match is capped at the divergence point; the divergence point usually **has no snapshot** (the walk never passes through it, since it lies inside the answer), so the next round can only land on the nearest checkpoint before it. This is the tradeoff of "reuse a bit more this round, a bit less next round"; whether it is a net gain requires long-answer measurement to decide; eliminating it would require the walk to leave a checkpoint at the splice divergence point (or moving session-continuation judgment and matching wholly to client coordinates), which is a separate design.

**Evidence and tests**: `_temp/20260929-*_replay_*.out.txt` (control replay), `C:\ninfer\serve-guard.err.log` (trace), `build-win/adopt_guard.log` (build); `ninfer_turn_replay_test` passes. **Test coverage (added this round, partial)**: `tests/models/qwen3_5/test_tp2_sessions.cpp` adds `check_replayed_answer_keeps_prompt_end`, which uses `engine.prepare(PromptInput)` through the frontend rendering path (producing `message_boundaries`) and asserts "the round after the replayed answer still reuses the previous round's entire prompt" (measured `reused 70 of 58`). **But it cannot isolate this section's guard**: removing the guard still passes — with a text answer the client echo is token-for-token identical to the entry, so it takes adoption's "exact replay" early return and the guard never participates; triggering the guard requires whitespace inherent to the template's framing (tool-call rounds in production), which this test path cannot construct, so the guard's regression protection still relies on live replay.

### 4.10 Session cache crash: entries without a host copy are still treated as recall candidates (fixed)

**Discovery**: while doing interleaved two-session replay with exported sessions (the two sessions **had exactly identical opening messages**, which was precisely the usage pattern in Claude Code at the time), the server **crashed outright** on the third request (A's second round) — the process vanished with no output at all.

**Trigger chain (fully recorded in the trace)**

1. A1 full prefill, publishes entry 0 (device-resident, never written to host).
2. B1's prompt is **byte-identical** to A1 → `active_shared == prompt_end` → takes `continue` (correct: reuse the same KV span) → but the generated tail differs from A's → `session_publish` hits the "not extends_resident" branch → creates entry 1 and `mark_device_resident(entry1)` → **entry 0 becomes non-resident, yet has no host copy at all** (trace: `entry 0 ... kv_end=0 ... resident=0`).
3. When A2 arrives the recall loop still treats entry 0's `frontier=51805` as a candidate (`reach=51805 via=frontier`) → `session_restore` goes to `host_kv_arena_->view(*entry.host_kv[i])` with `pages_for_tokens(51805)` → **dereferences a null slab** → crash.

**Root cause**: the recall loop only capped `host_shared_end` (kind 2) at "no more than the slab's actual fill range"; `frontier` (kind 0) and `host_prompt_end` (kind 1) used `UINT32_MAX`. On the normal path store syncs `host_kv_end` to `frontier`, so the problem is invisible; but once an entry loses device residency **without a store**, that invariant breaks (`session_publish` only does `mark_device_resident` when creating a new entry, and does not save a copy for the entry it displaces).

**Fix**: all three candidates are now uniformly capped by `entry.host_kv_end`. At store time `host_kv_end == frontier` and the other two mirrors are already within it, so behavior is exactly unchanged for legitimate entries; for entries without a stored copy, all three candidates are 0 → automatically skipped.

**Measured** (same two-session probe, 4 rounds × 2 sessions interleaved)

| | A1 | B1 | A2 | B2 | A3 | B3 | A4 | B4 |
|---|---|---|---|---|---|---|---|---|
| Before fix | 0 | 51,200 | **crash** | — | — | — | — | — |
| After fix | 0 | 51,200 | 51,728 | 51,720 | 51,925 | 51,967 | 52,041 | 52,096 |

8/8 requests succeeded, hit rate 99.0–99.9%; the trace shows A2's recall loop reports `reach=0 via=none` for both entries without stored copies, so it no longer recalls them and instead continues via the resident (`reuse=51728 src=device`).

**Not done**: entries without a stored copy are now merely "inert" — `reach` stays 0 until evicted by weight/LRU. They are already harmless, and an active drop would still need to determine whether they are anchors etc., so the change was kept minimal.

**Test coverage (added this round)**: `tests/models/qwen3_5/test_tp2_sessions.cpp` adds `check_unaligned_dialogue`, which uses the raw-token path to construct two sessions with "the same opening prompt, different continuations" (the first entry loses residency and has no host copy) and then brings the first session back. **Reverse verification**: with the recall cap reverted to `UINT32_MAX`, the test terminates with `0xC0000005` (access violation) — exactly this section's crash; with the cap in place it passes and agrees with the oracle.

### 4.11 TP-2 multi-concurrency: adoption of re-rendered answers fails (fixed)

**Scope**: this section fixes only the **third** symptom of the original report ("content it decoded itself still has to be re-prefilled by itself"). The first and second symptoms (the system prompt is not reused between the two sessions; the second session re-prefills the system prompt) **are not in this section**: they are blocked by `session_recall`'s skip on `entry.device_lane >= 0` (`tp2_generation_core.cpp:6157-6158`), i.e. the **unimplemented feature** of "while the first session is still running, another session shares the same stable block" (§4.8 ③A, marked unimplemented at :368). That skip has been confirmed in `docs/PLAN-tp2-concurrency-review-remediation.md:251-259` (review items B5/R5) as "confirmed not a correctness defect", and `docs/serving.md:87-100` states "device-resident reuse depends on which lane a request lands on".

**Symptom (production log `C:\ninfer\serve-win.log`, `--max-concurrency 2`)**: req#7–#10's cache **equals exactly the previous round's prompt length** (45,099 / 51,629 / 53,785) rather than the previous round's frontier (51,387 / 51,896 / 54,065), with the difference exactly equal to the previous round's generation count (6,289 / 268 / 281). req#5/#6, by contrast, got the previous round's frontier (29,427 / 29,576).

**Root cause**: the batch executor runs `retire_lane_session(lane.slot); release_lane_kv(lane.slot);` at request terminal state (plain `tp2_generation_core.cpp:3438-3439`, spec `:4233-4234`). When `lanes_ > 1`, `release_lane_kv` clears that lane's `cached_prompt_tokens`/`cached_boundaries`/`cached_state_valid`/`live_state_valid` and `invalidate_host_checkpoints` (`:3147-3184`). But the only mechanism that aligns "the answer sent back by the client" with "the tokens the engine sampled itself", `adopt_generated_turn`, reads only the lane's own `cached_prompt_tokens` (the early return at the original `:6387`) ⇒ already cleared ⇒ adoption silently fails. So `session_recall` can only take boundaries by token-by-token common prefix; `offered[0] = entry.frontier` is rejected by `offered[kind] > shared` (`:6178-6179`), leaving only `host_prompt_end`. req#5/#6 did not expose this because the client echo was token-for-token identical to the generated tokens (`shared == frontier`), so adoption was not needed at all.

**Why it is not "the client lost the reasoning"**: req#7's prompt 45,099 = 13,492 (system prompt) + 31,314 (replay) + 293 (new user round), and the total replay length exactly equals req#4's generation count (31,314) ⇒ the replayed token count did not shrink; the divergence is only at the start of the generation segment.

**Divergence mechanism**: `src/models/qwen3_5/frontend/output_session.cpp:260-289` emits the reasoning channel as-is (the content channel strips leading whitespace at `:246-258` via `strip_content_leading`, reasoning does not), and `src/runtime/engine/engine_core.h:709-724` accumulates deltas directly, so the `reasoning` returned by the engine can begin with a newline; the template `<models-dir>/chat_template.jinja:331` writes `reasoning_content = reasoning_content | trim`, stripping it ⇒ divergence at token 0 of the generation segment. This is exactly the divergence that `same_rendered_turn` tolerates (`src/runtime/engine/turn_replay.h:56` compares `trim(generated_head)` with `trim(replayed_head)`) and that `adopt_generated_turn` is supposed to eliminate.

**Fix**: candidate selection was extracted into the pure function `adoption_candidates(incoming, sources, divergence)` in `src/runtime/engine/turn_replay.h`; when the lane's own lineage is empty, `adopt_generated_turn` falls back to the catalog: it takes only `(entry.tokens, entry.prompt_end)` from entries with `entry.device_lane < 0 && entry.host_kv_end != 0` (the ones recall can restore) as candidates, stably sorts by descending `shared` (ties by descending `history.size()`), and runs each through the original `same_rendered_turn` + splice logic. When the lane lineage is non-empty, behavior is byte-identical (only one candidate is produced, and it shadows the catalog). This path covers both "cross-lane continuation" (req#7: previous round on lane1, this round lands on lane0) and "same-lane continuation" (req#8–#10).


**Measured**: `ninfer_turn_replay_test` passes (including the new `check_adoption_candidates` assertions); the two-lane `check_replayed_answer_across_lanes` (lanes=2) passes (`reused 242 tokens, past the 168-token prompt it was built from`); the single-lane `check_replayed_answer_keeps_prompt_end` is still `reused 70 of 58`, consistent with §4.9, showing single-lane behavior is unchanged. Regression (`NINFER_TEST_ROUTE=plain`, `_temp/20261002-2129_tp2final.ps1`): `ninfer_qwen3_5_tp2_lanes_test` 5/5 passes; `ninfer_qwen3_5_tp2_sessions_test`, after three passed, stops at the §3.3 pre-existing "shared system prompt first-sample divergence" (`got [2752 13 198 197 197 92 198 197] expected [467 419 538 13 198 197 197 92]`), a line byte-identical to `_temp/20261002-1541_tp2model_sessions.log` before the change ⇒ this change adds no new failure (that failure is already recorded as pre-existing at :195-197). **The production-log fingerprint (req#7–#10's cache == the previous round's prompt length) was not re-verified by server-side replay this round**: reproducing it requires the original session's systemPrompt + tool set; this delivery rests on the self-consistency of the log numbers (replay token count exactly equal to the generation count) and the code localization of the divergence mechanism.

**Test coverage**

- `ninfer_turn_replay_test` adds `check_adoption_candidates`: catalog fallback, lane lineage taking precedence and shadowing the catalog, sources that are neither resident nor stored being ignored, ordering by agreement and ties by longer history, four kinds of boundary guards (`turn_begin == 0`, `history.size() <= turn_begin`, `incoming.size() <= turn_begin`, empty history), and rejection when `shared < turn_begin`. **This is the regression protection for this fix**: before the fix the "lineage empty" branch produced no candidates at all.
- `tests/models/qwen3_5/test_tp2_sessions.cpp` adds `check_replayed_answer_across_lanes` (lanes=2, thinking enabled, client echoes reasoning + content). **But it does not isolate this fix**: the measured trace is `[tp2-reuse] ... shared == cached ... adopted=0 ... src=live`, the client replay is token-for-token identical to the engine tokens, and `session_recall` takes `entry.frontier` directly, the same value before and after the fix. The reason is the same as §4.9: the client prompt's tokens come from template rendering, rendering is canonicalizing, and the client cannot create token divergence; divergence can only come from non-canonical bytes the model itself sampled. So it serves only as a replay guard for the multi-lane route, as its comment states. **An isolation trick tried and abandoned** (verified this round): injecting leading whitespace into the replayed assistant body — ineffective, `<models-dir>/chat_template.jinja:244` writes `render_content(...) | trim` for the assistant body (`:331` likewise `| trim` for reasoning), so leading/trailing whitespace is removed during rendering; the only thing that survives is a whitespace change **inside** the body, and that makes `same_rendered_turn` (`src/runtime/engine/turn_replay.h:56` compares the trimmed head/body) judge them as different turns ⇒ the divergence a client can construct and `same_rendered_turn`'s tolerance domain are mutually exclusive, and forcing divergence end-to-end is infeasible under this problem.

**Not done**: `release_lane_kv` is left as-is. The catalog already covers the real scenarios; keeping lane lineage matters only when `session_capacity_ == 0`, and at that point the pages have already been released and cannot be reused anyway.

### 4.12 TP-2 vision prefill blows out the 192 MiB text workspace (fixed)

**Symptom**: with two lanes concurrent, `C:\ninfer\serve-win.log` shows `[tp2-lane] batch of 2 failed: text/layers/4 prefill columns=1024: bad allocation` ⇒ HTTP 5xx.

**Localization**: unrelated to concurrency. Single lane + 4 images reproduces on a clean server, with byte-identical offsets (`[arena] out of memory: 5916672 bytes at offset 195712256 does not fit 201326592 bytes (used 195712256, peak 195712256)`); the text-only two-lane case merely then hits the §3.6 transport stall — the two are distinct defects.

**Root cause**: `forward_tp2_prefill` in `src/models/qwen3_5/execution/text.cpp` keeps each shard's handoff staging (`hidden × count × 2 = 5120 × 1024 × 2 = 10.0 MiB`, plus a `source` on the peer side) alive from `ops::scatter` until the end of that forward pass, so its 10.0 MiB stacks on top of the layer peak. The measured layer peak for a text-only 1024-wide chunk is **182.3 MiB** (arena capacity `kWorkspaceBytes = 192ULL << 20` at `src/runtime/engine/tp2_generation_core.cpp:197`), plus 10.0 MiB ⇒ 192.3 MiB, so the first 1024-wide chunk of 100% vision tokens (measured `t0=13221 vision_tokens=1024`) overflows at layer 4 (GDN/linear_attention, the `mixer_layer` wrapper at `text.cpp:2015-2019`). `staging = 5120 × 1024 × 2` is exactly on the order of the rejected 5,916,672 B, and the offset 195712256 falls inside the layer loop.

**Fix**: the handoff dies immediately after scatter, so both arenas' `scope()` calls were moved down to the start of the vision block (`handoff_scope_a` / `handoff_scope_b` at `text.cpp:2633-2634`), so that `staging`/`source`/`indices`/`indices_peer` are released before the layer loop. Safety: `ar_exchange<true>` only reads and writes this device's own buffers; cross-device exchange goes through `DevicePair`'s mapped pinned host staging and never writes the peer arena (`src/core/tp/device_pair.cu:1224-1232,1267-1301`; the host-staging fallback path also does `cudaStreamSynchronize` on both streams, `:1313-1336`), while the `ops::scatter` that consumes it and the kernel that produces it are on the same stream. After the release the media block peak returns to 182.3 MiB (9.7 MiB headroom).

**Measured** (`_temp/20261008-0134_tp2-badalloc/`, `build-win`, 2×RTX 5060 Ti, `--devices 0,1 --max-concurrency 3 --prefill-chunk 1024 --vision-item-tokens 8192`):

- Single lane + 4 images (16,315-token prompt): before the fix `bad allocation`, after the fix `done | prefill 1.01k tok/s (16,315 tok) | decode 91.4 tok/s`.
- Two lanes (lane1 text + lane2 4 images): no prefill overflow, the pure-vision chunk `t0=5035 len=1024 vision_tokens=1024` peaks at 182.3 MiB; it then still hits the §3.6 stall (both of the batch return 503 together), a distinct defect.
- Two lanes, text only: both lanes `done` (decode 65.0 / 49.8 tok/s).
- Across all three cases the `[arena] out of memory` count is 0, and the largest lane0 peak in `[wsdiag]` is 182.3 MiB (the same chunk was 195.7 MiB before the fix).

**Incidental**: the insufficient-capacity branch in `src/core/arena.cu` now prints the rejected size, offset, capacity, and current/peak usage (previously it was a bare `throw std::bad_alloc()`); this localization depended entirely on that information.

### 4.13 TP-2 multi-lane window graph first launch: implicit upload blocks the host and causes a collective timeout (fixed)

**Symptom**: with two lanes concurrent, `C:\ninfer\serve-win.log` shows `[tp2-lane] batch of 2 failed: TP-2 allreduce stalled at rendezvous id 12884901888: the request was failed and the reusable context was discarded` ⇒ two `request_error` in the same millisecond, both lanes HTTP 503. After §4.12 fixed the vision workspace, this is the second independent defect in the same log.

**Scope localization**: single lane (batch=1) never triggers it, two lanes (batch=2) always do; `NINFER_TP2_VERIFY_BATCH_GRAPH=0` (routing batched verify through eager) makes it disappear completely, while `NINFER_TP2_DECODE_BATCH_GRAPH=0` has no effect ⇒ the culprit is the multi-lane **verify batch capture graph** (channel 3 = the first batch=2 window), unrelated to the decode batch graph.

**Localization chain** (every step obtained with temporary instrumentation, all of which was removed before commit):
1. Node census (`cudaGraphGetNodes` + `cudaGraphNodeGetType` + `cudaGraphKernelNodeGetParams`): the graphs on both sides of batch=2 each contain **all 178** `ar_exchange` kernels (`def0 nodes=2731 kerns=2191 ar=178 | def1 nodes=2725 kerns=2186 ar=178`) ⇒ it is not "the peer graph is missing collective kernels".
2. Kernel entry/participation counts (probes written at `+32/+40/+48/+56` of each side's 64 B stalled cache line): side A entry 7224 / participated 7109, side B entry 7114 / participated 7108 ⇒ B did enter channel 3's #0..#2, but gave up at entry all three times; the only condition for giving up at entry is `*stall_mine != 0 || *stall_peer != 0`, and side B has `flagsB=0` ⇒ B's giving up can only come from A's flag, and that flag is set only after A spins out `ar_timeout_ns_ = 10 s` at #0 (`src/core/tp/device_pair.cu:941`) ⇒ **B's stream starts executing the same graph ≥10 s after A's stream**.
3. Host timestamps (`[ar-launch]` printing milliseconds at each step of `launch_window_graph`, `[ar-mark]` placing a marker kernel at the head of each side's graph using mapped pinned buffers): `[ar-launch] host_ms=217113215 ch=3 bindA=0 launchA=10003 bindB=0 launchB=36 bindEnd=0 total=10039` (all previous launches on channel 2 were 0–1 ms); side A's graph-head marker fires **1 ms** after that launch call (A is already running), side B's graph-head marker is **+10016 ms** later.
4. Changing `NINFER_TP2_AR_TIMEOUT_MS` to 60000 and reproducing: `launchA=60006 ms`, B's marker +60002 ms, still 503 ⇒ **the blocking duration tracks the collective timeout ceiling exactly**, showing that what blocks is not "slow upload" but that the upload/first-launch and that graph's own collective wait are locked on each other.

**Root cause**: `cudaGraphLaunch` performs an implicit upload first for an executable that **has never been uploaded**, and the upload requires the target device to drain. `launch_window_graph` (`src/runtime/engine/tp2_generation_core.cpp:1419-1432`) launches shard A and then shard B with no synchronization between the two launches (the comment explicitly says "Nothing here may synchronize the host"). Once A's graph is submitted it starts running on the device and stops at channel 3's first collective waiting for the peer; the host, however, is stuck in `graph.executable[0].launch(...)` waiting for that implicit upload to complete, and the upload must wait for the device to drain — the device is occupied by that very graph, so it can only wait for the collective timeout. The peer's `executable[1].launch(...)` is not issued until 10 s later ⇒ B's stream is 10 s late ⇒ A times out and sets the flag, B gives up at entry, and `abort_if_ar_stalled()` throws. **This is not a concurrency-correctness problem, nor a device problem; a single lane's first launch blocks just the same (except that graph does not depend on a peer, so the blocking duration equals its own execution time).**

**Fix**: added `TP2GenerationCore::install_window_graph(WindowGraph&)` (declared in `src/runtime/engine/tp2_generation_core.h`, defined in `src/runtime/engine/tp2_generation_core.cpp` before `capture_verify_graph`), which unifies the identical tails (both sides `instantiate` + `graph.captured = true`) of the 5 capture functions (`capture_verify_graph`/`capture_verify_batch_graph`/`capture_decode_graph`/`capture_decode_batch_graph`/`capture_mtp_chain_graph`) into: both sides `instantiate` → both sides `executable.upload(stream)` → both sides `device.synchronize()` → `graph.captured = true`. The upload happens during capture while the device is idle; measured `instantiate` 8–27 ms, `upload_sync` 2–5 ms; every replay thereafter is a pure enqueue. This adopts the existing single-GPU idiom (`upload` + `synchronize`) from `src/models/qwen3_5/program/graphs.cpp:67-75`, and deliberately **does not** use the "warm-up launch" at :85-89 of the same file — TP-2's warm-up would actually execute 178 collectives and interact with the peer.

**Measured** (`_temp/20261008-0134_tp2-badalloc/`, `build-win`, 2×RTX 5060 Ti, `--devices 0,1 --max-concurrency 3 --prefill-chunk 1024 --spec dflash2 --vision-item-tokens 8192`):

- Before the fix (with instrumentation): `[ar-launch] ch=3 launchA=10003`, both lanes 503 at the same time, `[ar-watch] stalled ... id=12884901888`.
- After the fix (with instrumentation): all 308 `[ar-launch]` lines ≤1 ms (including each channel's first launch), all 616 `[ar-mark]` lines align line-by-line between the two sides (0–2 ms apart), 0 lines of `[ar-watch]`; both lanes complete (lane1 text + lane2 4 images): lane1 `192 events/51 deltas @19.6 s`, lane2 `918/291 @29.4 s`, ledger has 0 `request_error`, `decode 103.4 tok/s`, `prefill 3263 tok/s` (the same case was 5.4 tok/s before the fix).
- After the fix (clean build, `run_regression.ps1` three groups, `NINFER_TP2_*` unset throughout):
  - Single lane + 4 images: `stream-end 763 events/240 deltas @25.7 s`, `prefill 16,315 token / TTFT 16.7 s / decode 8.6 s`, markers 0, `request_error` 0.
  - Two lanes + 4 images (the original reproduction case): lane1 `192/51 @19.5 s`, lane2 `918/291 @29.0 s`; req#2 `prompt 16,315 / TTFT 17.5 s / decode 11.1 s`, markers 0, `request_error` 0.
  - Two lanes, text only: lane1 `192/51 @14.8 s`, lane2 `325/94 @17.0 s`; req#2 `prompt 12,211 / TTFT 12.4 s`, markers 0, `request_error` 0.

**Incidental**: the reason the "low-frequency transport stall" recorded in §3.6 reproduces in this case is that the batch verify window's first launch necessarily goes through implicit upload; after the fix that first-launch latency is gone as well. The prints added during diagnosis (`[ar-cap]/[ar-arm]/[ar-launch]/[ar-mark]/[ar-watch]` and the kernel probes) are **all temporary instrumentation and were removed before commit**; the final commit contains only `install_window_graph` and its 5 call sites.
### 4.14 TP-2 client disconnect: a transport exception retires the reusable prefix, so the next same-session request does a full prefill (fixed)

**Symptom**: req#93 in `C:\ninfer\serve-win.log` is `cache 0 (0.0%)`, `prefill 1.11k tok/s (126,001 tok)`, TTFT 1m59.4s, yet the session directory holds a record sharing 124,759 tokens with it that cannot be recalled.

**Scope localization**: among the 6 business-level zero hits in the same log only req#93 is a real defect (req#2 is the process's first real session, with an empty directory; req#9 shares only a very short prefix with its candidate; the rest are small 177/400-token requests). The dividing line is "the client hits stop inside the engine walk": req#92 (`cancelled during transport`, HTTP 499) is immediately followed by req#93, whereas req#64/#80, which also hit stop but where the engine had already returned normally (`response failed during transport`), did not destroy the cache.

**Localization chain** (the timeline comes from the DSH session record `session-f8ad33fd-1280-4192-88fb-f892b6e332a7` decompressed frame-by-frame from zstd and aligned millisecond-by-millisecond with the serve log):
1. All four `turn/end` are user aborts; turn3's last step ends at 21:44:51.165 == the same millisecond as `req#92 cancelled`, and turn4 step1 ends at 21:47:20.900 == `req#93 done` (`cache 0`, `prefill 126,001 tok`).
2. Text discriminator: `render_request_failure` at `src/serve/operational_log.cpp:311-320` prints `" cancelled during "` for `RequestFailureClass::ClientDisconnected`, and `render_response_failure` (same file :322-327) prints `" response failed during "` ⇒ req#92's exception was thrown from inside the engine walk, while req#64/#80 went through the response rendering path (the engine had already returned).
3. Recall scan for req#93: all 5 directory entries have `kv_end=0 resident=0`, `switch active_shared=0 resident_depth=0 stored=1 entries=5` ⇒ all entries have `host_kv_end == 0`, and the recall ceiling is always 0 (`src/runtime/engine/tp2_generation_core.cpp:6209-6213`), regardless of how large `shared` is.
4. Index shift: req#92 scanned 6 entries (5 printed + the active one 125054 skipped), and after evicting 112533, 125054 moved up to index 4 while the newly created entry took index 5; req#93 scanned 5 entries with 125054 still at index 4 ⇒ the newly created entry was deleted, while 125054 survived with `device_lane=-1`, `host_kv_end=0`.

**Root cause**: during req#92's decode, `preview_terminal(Cancelled)+publish_preview(false)` writes to the already-disconnected connection; `ClientDisconnected` (`src/serve/http_transport.h:22`) is thrown from `src/serve/openai_responses_http.cpp:424-427` and propagates out of `execute_walk`, hitting `execute`'s catch-all ⇒ `session_invalidate_active` → `session_drop` deletes the entry just published. That walk's recall had taken `resident_continues`' `continue` branch (`tp2_generation_core.cpp:6277-6290`, which does not write a host slab), and the true resident 125054's `host_kv_end` had never been written ⇒ after the new entry is deleted, 125054 remains in the directory but with a recall ceiling of 0, permanently unreachable, so req#93 can only do a full prefill.

**Fix**: `RetentionState` gains `terminal_published` (`src/runtime/engine/tp2_generation_core.h:1015`). `execute` clears the bit at the start of each round (:6408-6409), and the catch-all only calls `session_invalidate_active` when "the walk did not publish a terminal state" (:6411-6414) — a walk that published a terminal state before throwing has already written the device-stopped frontier into the directory, and its cost is this request, not the prefix. `execute_walk` no longer unwinds immediately on a transport exception: the first exception from `sink->start`/`sink->publish` is recorded (:7059-7067, :7155-7161), after which nothing more is written to the sink and decode is stopped as a cancellation (:7626, :8143), and terminal publishing proceeds as usual (ring checkpoint validation + entry frontier), rethrowing after setting `terminal_published` at both exits (:7343, :8212). When the walk throws before publishing, the original full invalidation still applies, so safety is unchanged.

**Measured** (new `check_transport_failure_keeps_prefix`, `tests/models/qwen3_5/test_tp2_sessions.cpp:875`, with `DisconnectingSink` throwing on the second delta, asserting that the next round's `reused_prompt_tokens` == the prompt end published by the failed walk (408); before the fix it was 0):
- Before the fix (after `git stash`-ing the two src files and rebuilding): dflash2 `FAIL: reused 0 prompt tokens, expected ... at 408` (`_temp/20261009-0025_baseline_dflash2.log`), plain likewise (`_temp/20261008-2359_prefixcheck_plain.log`).
- After the fix (documented baseline artifact `<models-dir>/qwen3_8_27b_swift15_dflash2_final.ninfer`): dflash2 `passed: reused 408` (`_temp/20261009-0145_final_swift15_dflash2.log`), mtp `passed: reused 408` (`_temp/20261009-0205_final_swift15_mtp.log`); each route is left with only one recorded pre-existing failure, with byte-identical tokens (dflash2 = the `a recalled conversation` recorded at `worklog §3.10`; mtp = the `a conversation behind a shared system prompt` recorded at `tp2-plan-concurrency.md:660`).
- After the fix (plain + the q4 artifact): the case gives `reused 408` and is token-for-token identical to the from-scratch oracle (`_temp/20261008-2335_tp2_sessions_plain.log`).

**Note**: when run with `<models-dir>/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3-dflash2-q4.ninfer`, `run_scenario`'s "shallow stable block" assertion fails (reused 0, expected 8); stashing all changes back to HEAD fails identically (`_temp/20261009-0035_head_dflash2.log`), unrelated to this fix, and that artifact is not this test suite's documented baseline artifact.

**Incidental**: `Repair-MsvcDepsPrefix` in `tools/win_port/build.ps1` fixed the mojibake in `build-win/CMakeFiles/rules.ninja`'s `msvc_deps_prefix` (a pre-existing build defect recorded in §3); since then, editing headers correctly triggers recompilation.
