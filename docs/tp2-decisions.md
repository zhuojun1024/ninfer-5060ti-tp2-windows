# TP-2 (2× RTX 5060 Ti) settled decisions

Quick reference of settled conclusions from the 2× RTX 5060 Ti TP-2 campaign
(Qwen3.8-27B NVFP4). Check this file before re-investigating any topic: entries
record what was tried, what the evidence showed, and where the full record lives.

Historical reference only: current state and remaining work live in
[tp2-plan.md](maintainer/tp2-plan.md). Product behavior and measurements live in
[tp2-dual-5060ti.md](tp2-dual-5060ti.md). Two shorthands are used throughout:
`worklog §N` and `worklog Round N` cite
[tp2-adaptation-worklog.md](maintainer/tp2-adaptation-worklog.md) — the
chronological round-by-round record — by section and round heading rather than by
line number, and `tp2-plan.md §N` cites [tp2-plan.md](maintainer/tp2-plan.md).

## Architecture and ownership

- **Fixed TP=2, dense 27B only.** No MoE split, no PP/DP; `--devices <name1>,<name2>`
  selects by GPU name, not index. (worklog §0–§2)
- **Load-time weight splitting; converter untouched.** Materialization reads the
  complete tensor, splits, and uploads per shard; reuses the official 23.7 GB
  artifact. (worklog §2) Since 2026-10-17 the materialization overlaps the
  artifact read with both shard uploads through a four-slot 64 MiB portable-pinned
  ring (one event per device context per slot; events are context-affine so each
  slot builds one under each shard's device), replacing the serial read → split →
  per-upload-sync chain. Measured 2026-10-17, six interleaved A/B samples,
  stash/rebuild arm switch, one instance at a time: engine `ready` median 47.0 s
  (baseline 45.0/47.0/47.1) vs 37.9 s (pipelined 29.8/37.9/38.3), −19%; the load
  summary reports `staging peak` 256 MiB as engagement proof. Details:
  [windows.md](windows.md#verified-behavior-on-windows).
- **DFlash2 and MTP are mutually exclusive backends.** Choosing DFlash2 frees
  shard 0's MTP weights (430 MiB) + KV (516 MiB). (worklog §3.6)
- **Whole-copy draft is infeasible.** At full context (262,144 / fp8 / MTP K=2)
  free is only 697/1,217 MiB vs a 2.07 GiB/card draft; after splitting ~1.04
  GiB/card. The proposal condition is a 5-block residual concatenation
  projection ⇒ cross-card handoff is required (residuals bit-identical across
  cards; only shard 0 captured). The candidate selector reads the full-vocab
  codebook directly (~254 MiB) and must stay whole. Do not re-investigate.
  (worklog §3.6; tp2-plan.md §3.4)
- **`--spec dflash` (v1) is rejected at construction time** — product decision;
  DFlash2 and MTP are the only supported draft backends. (worklog §3.6)
- **`--chat-template` is now a generic Jinja interpreter (this entry is
  superseded).** As of the first-tier cherry-pick (`b9219f3f` import llama-jinja
  + `98dada0e` execute custom jinja, 2026-09-21) the frontend compiles and runs
  arbitrary Jinja via the in-tree `third_party/llama-jinja` runtime, and
  `--chat-template FILE` loads a custom template (verified in source
  2026-09-29). The earlier "hand-written renderer + sha256 whitelist" state
  (worklog §11.8) is obsolete.
- **Vision: single-image cap 8,192 tokens**, static shard split (MTP→shard 0,
  vision→shard 1); images above the cap are rejected at request time with a
  pointer to the `[mem] vision` ledger. (worklog §11.4)

## Rejected optimizations (measured)

- **FP8 PV (2026-09-27): implemented, passed every quality gate, reverted.** It
  cleared G0-G3 (short-prompt acceptance Δ=+0.29 pp, deep-context Δ=−0.88 pp, both
  within the 30-rep noise) but reached only **1.176×** at the op level against the
  ≥1.3× bar (end-to-end ≈+1.4%, single sample) while widening the fp8 op tolerance
  1.2e-2 → 3.2e-2 (~2× output perturbation). Per the user's pre-authorization it was
  rolled back and the 1.2e-2 gate restored; the full implementation sits at
  `docs/maintainer/patches/fp8-pv-v1b.patch`. The lasting result is the attribution:
  **barrier stall 4.23 cyc/inst (34.2%) + math pipe throttle 3.12 (25.2%)** ⇒ the
  headroom is in the phase structure, not the PV dtype. (worklog 2026-09-27 archive
  §6; tp2-plan.md §2)
- **Cross-tile phase pipelining (2026-09-27): rejected.** G0 confirmed barrier is
  still the second stall on the reverted FP16-PV baseline (3.34 cyc/inst, 26.3%; 97%
  of it after the two block-wide `__syncthreads()`), but both resident-role
  implementations fell to **0.71–0.80×**: the whole PV accumulator is Br×D = 64×256
  fp32 = 16,384 registers = 25% of the SM register file, so moving PV to 8 warps
  doubles acc/thread into the 128-register wall, and a 768-thread variant (24 warps)
  pushes the producer past its working-set wall. The baseline "8 producer ∥ 8 Vdeq →
  16-warp PV" packing is optimal in the 1 block/SM envelope. Closed unless the
  envelope changes (Br=32, or a >100 KB block-level smem budget). (worklog 2026-09-27
  archive §7; tp2-plan.md §2)
- **Upstream ninfer-all items closed by measurement (2026-09-26).** Per-device
  `cudaFuncSetAttribute` and the 170-SM hardcoding were fixed (3 sites made
  runtime-derived; two stayed comments only, because the derived value would change
  existing 5090 behavior without a 5060 Ti A/B); item 3a disabling RDC was reverted
  (this tree gained +38 stack-frame functions, opposite to upstream); 5b GDN record
  staging was ported but is Op-level only (−22~27%, unmeasurable end to end); 5a
  small-T tensor-core covers only q4/q5 and does not apply to the TP-2 NVFP4/FP8
  trunks; item 4 (re-quantizing the streamed FP8 tensors to NVFP4/Q4) was dropped by
  user decision. (worklog 2026-09-27 archive §5; tp2-plan.md §2, §3.8)
- **Prefill AR∥MMA sub-block pipeline (R12 host-side chunk interleave): ~5%
  slower, rolled back.** That implementation, not the lever: a re-implementation
  with per-sub-block events and a dedicated collective stream nets **+11%**
  (see the L2 entry below). R12's stated reason — the in-kernel AR spin is
  "already masked by the peer's compute" — was also contradicted by the
  2026-09-26 divisor arm (AR is 48% of block time and strictly serial).
  (worklog Round 11-12)
- **Multi-block AR: no difference** (TTFT 859/877/879/879/830/861 ms for
  1/2/4/8/16 blocks) — not a single-SM issue. (worklog Round 35b)
- **Copy-engine rendezvous (small payloads): slower (951 vs 857 ms) and shard
  logits disagree** — rendezvous/double-buffer semantics do not hold. (worklog
  Round 35b)
- **Large-payload event path (Phase 3): rejected.** Copy-engine floor 3.32 ms vs
  sliced 2.70 ms; the event fence alone is +23%. (worklog §3.8)
- **Copy-engine large-payload arm (design stage): not implemented.** The real
  lever is the 128 calls/round count, not the transfer mechanism; the prefill
  control arm was left untouched and proved it. (worklog Round 17)
- **GQA-aware prompt kernel (L1): falsified at the measurement gate.** ncu on
  the production prompt kernel: DRAM 1.09% (KV traffic absorbed by L2), tensor
  pipeline 63.5% busiest, occupancy locked at 33% (100 regs + ~100 KB smem per
  thread). Warp doubling: −12% (register spilling). Bc 64→128: cannot launch.
  The 6× KV-request redundancy is not a DRAM cost. (worklog §3.14)
- **Reducing AR call count requires giving up weight splitting** ⇒ always a loss
  in decode. The only correct direction is overlap. (worklog §12)
- **Prefill allreduce is at the hardware floor.** Card 2 sits in a chipset
  Gen4 x4 slot (~7 GB/s) ⇒ 2.67k tok/s ceiling; the measured marginal slope
  (0.368 ms/token) sits exactly on it, independent of chunk width. The largest
  single lever is hardware: move card 2 to a CPU-direct Gen5 x8 slot. (worklog
  Round 35b)
- **Overclock/power wall and P2P are infeasible** (already at full clock,
  93/85 W; the consumer driver masks P2P, `canAccessPeer=0`, SYS topology
  across root complexes). (worklog §11.7, §3.13)
- **Prefill sequence parallelism (2026-10-16): falsified at the design stage, not
  implemented.** Splitting tokens across the pair does not reduce the prefill AR
  bottleneck: the RS+AG pairs a Megatron-SP layout needs are byte-equivalent to the
  ARs they replace (per layer: RS(mixer-out)+AG(next QKV)+RS(FFN-out)+AG(gate_up)
  = 4S ≡ 2×AR), so net PCIe bytes ≈ 0 while halving each payload (10→5 MiB) would
  only lower mapped-pinned transfer efficiency. Compute savings are also capped:
  attention/GDN stay head-sharded (each card must process all T tokens for its own
  heads), so token splitting saves only FFN/norm/Q-projection FLOPs (~30–40% of a
  layer) and 16 attention layers would need a cross-card KV gather (~8 KB/token).
  The blocker is GDN ownership: 48 of 64 layers are GDN whose chunked state-passing
  is a per-head serial chain across chunks (`src/ops/linear_attention/gated_delta_net/chunked/state_passing.cu:19-24`)
  with state owned by the head's card; token splitting forces either full-activation
  cross-card handoff per layer (erasing the compute saving) or a mid-sequence
  cross-card break of the serial chain (~1.6 MB state/conv handoff + per-layer
  serialization). Do not re-investigate without a hidden-dimension or GDN-state
  redesign. (feasibility analysis 2026-10-16; PLAN.md §A1)
- **Adaptive speculative extent per request (2026-10-17): implemented, measured on the
  production route, reverted.** The idea: replace the startup-fixed draft window with a
  per-round extent chosen from measured per-position acceptance
  (`E(k)=1+Σ_{i<k}q_i`, EWMA α=0.25 over rounds) divided by a per-envelope round-time model
  (EWMA bucket per envelope size; linear prior 35+2.5k ms). Two policy shapes were measured on
  the MTP K=5 route, 4-concurrent mixed batch (lane prompts 64/1536/6144/8192, 512 max tokens,
  20 waves) with server-side equals (env `NINFER_TP2_ADAPTIVE_SPEC=1`); the interleaved A/B
  campaign ran 4 cycles × 8 waves per arm, fresh server process per arm-half:
  - *Per-lane selfish policy (v1): −6.5% aggregate* (57.3 vs 61.3 tok/s). The round costs what
    its widest lane drafts, so a lane drafting short while others stay wide free-rides — fewer
    expected tokens per round at identical round cost. Trace confirmed collapse toward
    window 0/1. Per-lane extent optimization is structurally wrong under a shared envelope.
  - *Batch-uniform window policy (v2), the corrected shape*: aggregate +2.6 ± 5.1 tok/s paired
    (decode-only +2.4 ± 6.4) — not significant, below the ≥3.5% retention bar; measured
    acceptance rose (0.43 vs 0.385 pooled) but the wider-window cost it saves is already small
    (verify column marginal ≈1.85 ms).
  - *Solo regression: −7.1 ± 1.0 tok/s (−9.7%), consistent in all 4 cycles.* A single request on
    the 4-lane route still runs the batch executor (`columns=1`), where round cost is
    substantially width-sensitive; the policy shrank the solo window below its optimum. A
    columns≥2 gate would remove this but leaves the unproven ~+2.6 tok/s batch effect.
  Reverted per the pre-set <3.5% rule. Do not re-open unless a workload with structurally
  broader per-position acceptance spread (e.g. much larger K, or acceptance-collapsed batches)
  is the actual serving profile. Interleaved A/B protocol: cold solo probe first (post-wave
  solo runs sink ~10 tok/s from in-process batch state in a server-version-independent way),
  fresh process per arm-half, paired-by-cycle analysis.
  (PLAN.md §A2; measurement JSONs under `_temp/20261017-0115_a2_*`; bench driver
  `_temp/20261017-0115_a2_interleaved_ab.ps1`)
- **Device-plane divergence anchor (+73.4 MiB/card): abandoned.** Saves only
  ~9.9 ms per session; in real chat the reachability guard usually holds and
  the device path is taken. (worklog Round 17)

## Upstream merge-value assessments

- **Upstream `origin/master` through `e31bc99b` (2026-09-26) and `origin/dev`
  through `a012e2bc` (2026-09-28): assessed 2026-09-29, no TP-2 merge value.**
  Checked the perf candidates against the settled bottleneck - decode is
  weight-stream bandwidth-bound (GEMV 75-90%, lm_head ~439 GB/s ≈ 5060 Ti peak;
  36 SM, worklog §5.2):
  - *Linear unification + sliced-K MMA* (09-26 batch `229c1832`…`ecbc3357`): no
    gain. Sliced-K does not cut weight traffic on a bandwidth-bound GEMM; the
    NVFP4 trunk is W4A4 while upstream's sliced-K targets the A16 route
    (upstream deleted W4A4); TP-2 half-geometry halves N (column-parallel) or K
    (row-parallel), shrinking the split-K space.
  - *nvfp4 W4A4 TMA tweaks* (`1d8587bc`/`05507ab0`/`5f5fccab`): already
    cherry-picked (second tier) but no TP-2 gain - every main GEMM is a
    half-geometry shape that runs the MMA route, never TMA (all 5 half shapes
    lack `nvfp4_a4_tma_route`; only the 5 full-geometry shapes have it).
  - *q4/q5 dispatch tuning* (`d3c125ed`/`beedffa0`/`bb844c43`/`a9a0d10a`/
    `b39de4d5`/`9e163eee`/`5b4303c0`/`594930e7`): not on the trunk (mainline is
    NVFP4; q4/q5 only in the DFlash2 draft path, also bandwidth-bound). Already
    "not scheduled" (worklog §5).
  - *Custom Jinja chat templates* (`98dada0e` + `b9219f3f`): a feature, not a
    perf lever - already merged (first tier) with local extensions;
    `--chat-template FILE` runs via the in-tree `third_party/llama-jinja`
    runtime. Nothing to merge.
  Real levers remain those in tp2-plan.md (allreduce, AR∥MMA overlap N=4, CUDA
  graph, card-2 Gen5 slot). Re-assess only if `origin/master` moves past
  `e31bc99b` in a way that touches the trunk. (source-verified 2026-09-29)

- **Upstream `origin/master` = `origin/dev` = `68c54356` (2026-10-05): the
  cache-boundary trimming fix is ported; the rest is still reference-only.**
  Re-assessed 2026-10-05 after the baseline moved (73 upstream commits from the
  `e360c4c0` merge base, 270 local ones).
  - *Ported - `68c54356` "fix(frontend): preserve cache boundaries through
    template trimming".* A source byte range collapsed by `trim`/`strip` is now
    retained through string composition (`string_part::collapsed_source_end`,
    `cut_bytes(begin, end, collapse_removed)`, `append_boundary`), published as
    `TemplateOutput::boundary_mappings`, and consulted by `source_boundary`.
    Before it, every trimmed part lost its `source_offset`, so
    `MessagePartBoundary` and `LeadingInstructionBoundary` markers resolved to
    nothing and `prepare_context_cache` dropped them silently (`if (!resolved)
    continue`). That is the common case, not an edge case: the production
    templates (`tools/chat_templates/qwen3_{6,8}.jinja`) apply `|trim` to every
    message's rendered content and to `reasoning_content`, so a system block or
    `tool_result` ending in a newline lost its requested frontier - the caller's
    `cache_control` had no effect and the span was re-prefilled. Genuinely
    ambiguous frontiers (repeated text, non-exact transforms, slice endpoints)
    still resolve to no boundary, and a boundary landing inside one BPE token is
    still rejected, so a wrong-frontier reuse is not introduced. Local
    deviations: `strip()` keeps the local manual Unicode trim (upstream
    `a8e212ac`'s `trim_utf8` refactor is a separate item), `value.cpp` keeps the
    local `_WIN32` `localtime_s` branch, and the continuation truncation in
    `chat_template.cpp` also drops collapsed boundaries that fall past the
    retained end (upstream leaves the new vector unclamped while
    `Tokenizer::encode_with_boundaries` rejects a byte boundary beyond the text
    outright). Evidence: upstream's 20-case regression (Unicode spaces, macros,
    slices, `tojson`, an injected BPE merge `x`+`y`) added to
    `test_frontend.cpp` plus three `boundary_mappings` assertions in
    `test_jinja.cpp`; 13 of the 14 resolvable cases fail without the fix, and
    `ninfer_jinja_test` / `ninfer_qwen3_5_frontend_test` /
    `ninfer_prompt_input_test` / the OpenAI+Anthropic schema tests /
    `ninfer_prepare_ragged_prefix_test` pass with it. `ninfer_chat_templates_test`
    fails on its recorded host-side cause (worklog §5.2), identically before and
    after the port.
  - *Ported - `75a89050` "fix(runtime): wait for compute before request cleanup".*
    `ProgramImpl::abort` and the `start_request` rollback returned the lane's
    buffers, pages and execution row to the pools without waiting for the compute
    stream, so a cancel that arrived after activation but before the first prefill
    unit had waited for its uploads and initialization could free memory an
    in-flight kernel was still writing. Local placement differs from upstream in
    one respect: the sync sits before `retain_aborted_continuation`, because
    catalogueing the aborted continuation reads the same device buffers and
    upstream has no retention branch at that point. The same class was audited and
    closed on the production route: `TP2GenerationCore::drive_lane_queue`'s
    whole-batch failure path returns every lane's KV pages right after a round
    that threw, and a stalled collective only leaves its kernel on the bounded
    in-kernel expiry, so both shard streams now drain before the pages go back. No
    sync is needed at the other release sites: `finish` is reachable only from
    `Lifecycle::Finishable` (set after the round's host-visible results were
    consumed), `commit`'s cancelled row only bumps the lane revision while the
    release happens in `finish`/`abort`, the materialization abort settles the
    transfer stream through `context_completion_` under the same open context
    transaction (so admission cannot interleave before the upload completes), and
    `fail_all_cleanup` runs on the terminal `failed_` path where the pool is never
    handed out again. Evidence: full build; `ninfer_qwen3_5_tp2_lanes_test`
    (2xRTX 5060 Ti, `NINFER_TEST_ARTIFACT=<models-dir>/qwen3_8_27b_swift15_dflash2_final.ninfer`,
    five scenarios including queued cancellation and a driver stop with a request
    in flight) and `ninfer_tp_device_pair_test` (in-kernel AR stall/give-up) exit
    0, the materialization/resource-manager/engine-options/HTTP host tests pass,
    and the frontend/jinja suites still pass. No deterministic reproduction of the
    in-flight cancel race exists; the change is an ordering guarantee on a path
    that is otherwise rare.
  - *Everything else on the trunk remains reference-only.* `b9114396` replaces
    the context cache and adds preemptive scheduling (176 files), a different
    product contract; the SM-count refactor
    (`a667efdd`/`417eb3d6`/`e621c7d6`) reaches this fork only as the
    `device_sm_count()` numeric policy (rope/rmsnorm done 2026-09-26; the MoE
    prefill grid cap and the attention split budgets still need a 5060 Ti A/B);
    the attention/linear perf batch (`23b0997d`…`7f6aafed`) is half-geometry MMA
    under the same no-gain argument as the 2026-09-29 assessment; `1cfdb4d6`/
    `84cf93e4` are no-ops at the local CUDA 13.1 toolchain. Inventory in tp2-plan.md
    3.8. (source-verified 2026-10-05)

## Withdrawn approaches (falsified by real traffic)

- **"Continuation boundary" (reuse to the end of the last generation):
  withdrawn.** DSH re-renders the assistant turn instead of echoing the
  generated token stream; the continuation slot was used 0/36 in real traffic.
  (worklog Round 13-14)
- **Template `|trim` → `rstrip('\n')`: rolled back.** Even with a maintained
  override template, verbatim replay of `reasoning_content` still stops at the
  previous prompt length. (worklog §4)
- **Text-protocol replay cannot guarantee token-exact reproduction.** The first
  generated token is not in the published reasoning text (`max_tokens=1` ⇒
  empty `reasoning_content`); with the DFlash2 cost gate (decline draft ≈ 3×
  decode vs saved prefill), recomputing is the rational choice. The only lever
  for token-exact reuse is a continuation protocol carrying token ids
  (unscheduled). (worklog §4, §6)

  *Byte-exactness withdrawn; the generated span is adopted instead.* The
  template's `|trim` removes the leading whitespace of the model's reasoning
  (worklog §4), so a replayed turn parts from the resident history at
  its first token while describing the same turn. When the two renderings agree
  - same trimmed reasoning, same trimmed content, byte-identical tool-call
  region - the walk replaces the replay with the tokens this lineage generated
  and reaches its frontier again; a replay that describes a different turn keeps
  the old fall-back, and decode now writes the checkpoint grid so that fall-back
  costs at most one stride. `turn_replay.h` owns the guard,
  `adopt_generated_turn` the replacement, and `NINFER_TP2_REUSE_TRACE=1`
  reports `adopted=` and the divergence bytes per request.
- **"Unaligned reuse must decline the masked draft": premise overturned.** Same
  round: declining 22.5 vs keeping 87.1 tok/s with near-identical acceptance ⇒
  the gate, `draft_context_declined`, `reuse_grid` and session-mean pricing
  were removed; a single scan with the deepest boundary wins. (worklog
  §3.6 background)
- **DFlash2 TP-2 option (b) (2026-09-22): implemented, then rolled back.** It
  was not the cause of the solo-probe instability; the route went back to
  construction-time rejection until the clamping-round root cause was fixed
  (B5). (worklog §3.6)

## Measurement and methodology rules

- **Acceptance-rate verdicts need a 30-rep pool.** The 15-rep inter-arm noise
  floor is ~±2 pp; two "−2 pp cost" verdicts from 15-rep pools were later shown
  to be noise. (worklog §3.9)
- **Greedy cross-arm comparison is invalid.** Draft proposal length changes the
  verify batch shape ⇒ target numerics change ⇒ text drifts (6/7 probe prompts
  differ). Greedy probes are deterministic regression checks only; pooled
  sampling is the acceptance-rate statistic. (worklog §8.6)
- **`CUDA_LAUNCH_BLOCKING=1` / compute-sanitizer are incompatible with the
  TP-2 in-kernel AR** — the A-side launch becomes blocking, the B-side kernel
  waits for it ⇒ the first collective always gives up (self-deadlock).
  Kernel-level attribution requires source-level probe bisection. (worklog
  §3.10)
- **Boundary sensitivity:** the last chunk width is not the variable; the
  boundary position is; 120-step decode shows zero flips. (worklog §2)
- **B7 (verify graph) settling needs ≥24 reps/arm.** Current A/B: 6 reps/arm,
  aggregate +5.2%, per-request t=0.99 (not significant). (tp2-plan.md §3.4)
- **The Q4 selector uses coarse T buckets, not per-T capacity buckets** —
  per-T instantiation is pure code bloat under the mask path. (worklog
  §8.6)
- **Draft-quantization overrides are cumulative by design** — each arm must be
  the cumulative artifact, otherwise it falls back to the official Q8 recipe
  and the test is not of the product path. (worklog §8.6)

## Root causes (settled)

- **MTP consistency (①): shards never wrote replay records** (fold consumed an
  empty log). Fixed; criteria were no-regression + same quality tier + speedup.
  (worklog Round 36f)
- **Spin hang (2026-09-23): token counter cross-card false sharing.** Fixed;
  `DevicePair::start_ar_watchdog` retained as a diagnostic. (worklog §8.9)
- **Clamping rounds: `forward_tp2_window` never bound
  `active_valid_columns_` / `valid_columns`** ⇒ paged-KV same-slot tearing.
  Fixed. (worklog §3.6)
- **DFlash2 22 tok/s: the decline guard misfired on the template prefix**
  (~37–57 tokens always hit the unaligned fallback ⇒ `extent=0` every round).
  Fixed with a 16×-budget cost gate. (worklog §3.6)
- **TP-2 decode double preview:** `preview_model` already handles termination;
  the decode loop was publishing a second preview. (worklog Round 28)
- **Early GDN test failures: test bug** (GDN state pool not reset between the
  card-local double forwards). (worklog Round 18)
- **Bounded-spin give-up inside a captured window:** the round keeps running
  with locally-only partial sums; the window's kernels consume the inconsistent
  data as addresses ⇒ illegal address; no host-side check can make it in time.
  Only eager-prefill give-ups become clean 503s. Mitigations in place:
  abandon-output (1b), watchdog (default on), timeout default 2000→10,000 ms
  (a second-level stall is a host/driver event, not a slow kernel). (worklog
  §3.10; tp2-plan.md §3.6)
- **Two low-frequency failures (r67 IllegalAddress req#14; r69 HTTP 503
  req#14) are the same event class.** Natural rate ≲1/1,500 requests; long runs
  (1,500 requests, watchdog on) have not caught the trigger. (tp2-plan.md §3.3, §3.6)
- **The two low-frequency transport failures are a one-collective rendezvous skew
  (settled 2026-10-06).** The in-kernel allreduce published one id per slice and spun
  until the peer's slot equalled it, while the engine legitimately lets the mirror shard
  run a call or two behind (only shard A is drained where a round reads its sample), so
  whenever the fast side completed a collective and published the next one before the
  slow side sampled, the wait became unsatisfiable and the 10 s bound turned it into the
  usual 503. The natural dumps pin it: arrival slot 0 differs by exactly one id between
  A and B with every other slot identical, and `last=352` - the `dflash_selector_tp2`
  back exchange payload (`align16(5*4 + 16*5*4)`) - i.e. two `sendrecv`s with no
  device work between them, the tightest handoff the transport ever sees. Fixed by
  banking the arrival and write-order slots and rotating the bank with the id
  (`NINFER_TP2_AR_BANKS`, power of two in 1..8, default 4; 1 is the pre-bank geometry),
  so the bank still holds the id the slow side is waiting for. Acceptance: the
  `check_ar_skew` case in `ninfer_tp_device_pair_test` (hold injection: 1 bank must
  still give up, 4 banks must complete bit exact) and
  the `NINFER_TP2_AR_FAULT_HOLD_POLL` skew acceptance run (1 bank must fail - the warmup with a
  stall message or a request with 503 and a watchdog dump - and 4 banks must start and serve).
  Injection: `NINFER_TP2_AR_FAULT_HOLD_POLL=<eager serial>:<ns>[:self]` (0 = every eager
  collective; captures are excluded, since a hold recorded into a captured graph would be
  replayed by every launch of it); both fault hooks log `[ar-fault]` when they land.
  (device_pair.{h,cu}; worklog §3.10)
- **BF16-add allreduce is bit-safe only for BF16 payloads.** I32/FP32 ids can
  carry signaling-NaN bit patterns; use `DevicePair::sendrecv` for byte-exact
  cross-card exchange. DFlash2's equivalent paths are fixed; the MTP route at
  `execution/text.cpp:486` is not (verification cost high). (tp2-plan.md §3.3)
- **Official NVFP4 artifact's lower acceptance rate vs synthetic/self-
  converted:** the difference is in the text weights, not the components.
  (worklog 2026-09-27 archive)
- **Solo-probe determinism: ~6% flip rate** refutes "byte-identical across
  runs" for the dflash2 solo probe. (worklog §3.6)
- **The sessions plain-route failure is pre-existing** (the r57 baseline
  artifact reproduces it token-exactly; dflash2/mtp routes pass) ⇒ blocks the
  three-suite all-green; the sessions gate runs on the dflash2 route. (worklog
  §8.6; tp2-plan.md §3.3)

## Tool and environment boundaries

- **WSL2 CUDA is unusable for execution** (PTX JIT segfault in
  `cudaGetDeviceCount()`; GPU held by Windows services) ⇒ the WSL tree is
  compile-verification only; all tests run on the Windows side
  (`tools/win_port/test.ps1`). (tp2-plan.md §3.5)
- **Three GPUs visible:** nvidia-smi order 0/2 = 5060 Ti, 1 = Tesla T10;
  `--devices 0,1` picks the two 5060 Tis by CUDA ordinal (offset from
  nvidia-smi order). (tp2-plan.md §1)
- **Windows PATH must include the FFmpeg and libcurl `bin` dirs** or
  solo/sessions die at load with `0xC0000135` (append does not use the media
  path). (tp2-plan.md §1)
- **Windows libcurl link:** `find_library` picks the static `libcurl.a`
  (MSVC cannot link it); prefer `find_file(... libcurl.dll.a)`. (worklog
  §12)
- **Process discipline:** one model process machine-wide (WSL 8088 vs Windows
  8099 mutually exclusive); stop the service before relinking the exe
  (LNK1104); `build-win/apps/ninfer-serve.exe` is not auto-synced to
  `C:\ninfer\`. (tp2-plan.md §1)
- **Watchdog cost model:** `note_ar_call` was already unconditional; the
  thread polls host counters every 25 ms, calls no CUDA API, touches no
  stream; large dumps are rate-limited to one per stall. A/B: no measurable
  cost. (worklog §3.10)
- **The frontend test fixture rejection is a pre-existing failure** (fixture
  template rejected by the chat-template whitelist; expected to turn green
  after the cherry-pick). (worklog §4)

## Deferred (settled as "not now")

- **Full draft NVFP4: not advancing** (source-level audit;
  [tp2-dflash2-draft-nvfp4.md](maintainer/tp2-dflash2-draft-nvfp4-feasibility.md)). Only ~202.3 MiB
  left vs r66 q4all, of which 174.3 MiB is the non-GEMM codebook; NVFP4
  0.5625 B/elem > Q4 0.53125 B/elem, so re-quantizing already-Q4 tensors would
  grow them; three hard blockers (no float→NVFP4 quantizer in the converter and
  no A4 activation calibration for the dynamic-FP8 draft source; NVFP4 native
  Weight only takes full parents; consumer op NVFP4 geometries are closed
  sets). Reopen only with a W4A4-family / prefill-TMA motivation. (worklog
  §3.7)
- **Ring K/V quantization (CyclicKVCache): deferred.** Locked in three places
  (`CyclicKVCache` layout, SWA validation, `context_kv_materialize`
  `validate_cache`); ~55 MiB, worst cost/benefit. (worklog §8)
- **L2 (AR∥MMA overlap): retested with a corrected schedule; +11% prefill.**
  The prefill chunk is split into N 64-aligned sub-blocks and each sub-block's
  collective runs on a second stream (`DeviceContext::collective_stream`) while
  the next sub-block's mixer/FFN keeps the compute stream busy
  (`run_layers_tp2_overlap`, `--prefill-overlap W` / `EngineOptions::prefill_overlap`,
  default 256, `0` disables). Chunk 1024, ~6.3k-token prompt, 7 reps (median): N=0 0.587 ms/token, N=2
  0.554, **N=3 0.612 (slower than doing nothing)**, **N=4 0.516 (−12%,
  reproduced across two batches)**, N=5 0.547, N=8 0.655. N=4 (256-wide
  sub-blocks) is the real optimum, and **N does not interpolate** — 320/384
  falls into a slower activation/MMA schedule class — so a policy must key on
  the 256-token sub-block width, not extrapolate N. Re-verified at chunk 512:
  0.625 serial → 0.572 with 256-wide sub-blocks (−8.4%). Bit-identical to the
  single-block loop (tp2_forward_test chunk-split invariant exact at N=2 and
  N=4). This supersedes R12's −5% and the "unscheduled" status. (worklog §3.14
  + the two 2026-09-27 entries)
- **Gate revision (sampling-led acceptance verdict): separate from the
  promotion.** The literal greedy gate `|Δacc| ≤ 1.0pp` failed at K=7
  (−1.60 pp, proven cross-arm text drift); the promotion is done (r69,
  user-confirmed); 30-rep pooled sampling is the de facto method. Formalize in
  one place if/when. (worklog §8.6; tp2-plan.md §3.4)
- **TP-2 -> PP-2 prefill pivot (2026-10-17): measured, not implemented.** A
  prefill-only path that temporarily re-lays the resident weights from TP-2 to
  PP-2 removes the per-layer allreduce entirely (PP has no cross-card GEMM split)
  at zero change to the per-card compute (each card runs all of 32 layers vs half
  of 64; both ~0.21-0.24 ms/token shallow), so it buys exactly the exposed AR
  term dW ~ 0.33 ms/token, depth-independent. The swap is an in-place permutation
  (TP-2 and PP-2 have the same per-card bytes: weights 9.51 GiB, and the KV/GDN
  pools are byte-compatible), 4.755 GiB each direction per leg at the measured
  6.5 GiB/s peer rate staged through the <=3 GiB free pool in waves: 2S ~ 1.5-2.9 s.
  Measured TP-2 baseline (nvfp4, overlap 256): fresh 8k 0.531 ms/token, 32k 0.573,
  64k 0.674, 100k 0.779; marginal at 100k depth (a cached prefix + 8.5k new)
  **1.17-1.23 ms/token**. Expected end-to-end: fresh 32k-100k ~1.7x (79.3 s ->
  47.8 s at 100k), break-even ~5-8.7k tokens, a wash for the ~10k-new-token-at-100k
  case (1.03x). Amortized over a queue of prefills the break-even drops to
  ~2k/request. Reopen when a long-document / fresh-prefill workload is the actual
  serving profile; the effort is a full-weight PP binding, a 1F1B PP prefill
  program, a KV/GDN swap-back permutation and a serving gate.
  (docs/maintainer/tp2-pp-prefill-pivot.md)
- **Conversation identity and one catalog entry per conversation (2026-10-17).** The
  private-continuation catalog keyed slots only by token prefix: any perturbation of a conversation
  - folding, a trimmed tail, a re-render - appended a new entry and merely demoted the old one to
  weight 4, so variants of one conversation filled the `--max-private-continuations` slots and
  evicted the other conversation, paying host memory for no reuse. The serve layer now extracts a
  conversation identity (`x-session-id` > `x-claude-code-session-id` >
  `x-bili-plugin-conversation` beside `x-bili-plugin` > body `prompt_cache_key`/`session_id`;
  1..256 bytes) on all three routes and passes it as the existing LiveSession key. The engine keeps
  one entry per name: a publish locates the entry the request continues by content - the entry,
  carrying that identity or still anonymous, whose stored history shares the longest prefix with the
  incoming history, covering all but a short rewritten tail - and rewrites that entry in place (the
  resident case simply extends it) instead of dropping and re-appending it. That distinction is the
  whole reuse benefit: the entry's host slabs hold the prefix KV and the divergence image this
  request's own prefill just snapshotted, so replacing the record threw the fresh image away and the
  next turn had to re-prefill the full prompt. The rewrite folds the new history over the old record
  and clamps the host bookkeeping to the shared prefix (`host_kv_end`, plus every offered boundary
  above it reset), so a boundary the engine can no longer reach is never offered again. A related
  correctness fix gives the frontier state image its own position (`SessionEntry::host_frontier_end`):
  an entry's `frontier` moves when the entry is recalled or extended while its `host_state` image
  stays where it was captured, so offering that image at the entry's current frontier would have
  restored a deeper state onto a shallower boundary. A request that carries the name but continues
  none of that history (a review or sub-agent
  call) publishes anonymously and leaves the entry alone; a conversation whose slot was taken that
  way takes it back on its next turn, replacing the one-off entry. Eviction gained a publisher-weight
  gate, so an anonymous request cannot displace a named conversation. Deterministic harness (three
  alternating conversations, one folded rewrite, one unrelated request carrying a session's header,
  one title call that opens a name before the conversation does, chat route): catalog size stayed at
  the 3-entry capacity with one entry per conversation, the folded rewrite reported `replaces=1`, the
  unrelated request `bind=0` with no eviction, and the title call's entry was replaced by the
  conversation's own turn and extended afterwards. Route probes confirmed the header precedence, the
  `x-bili-plugin` marker guard, the body fallback, and that the Responses route keeps its response-ID
  key when no header is present.
  - Live check (2026-10-17; DSH + bili, Claude Code + bili, Claude Code alone, all against a traced
    3456 instance): the header contract held - DSH arrives under `x-session-id` with bili's
    `session-<uuid>` (with `x-bili-plugin-conversation` and `prompt_cache_key` carrying the same
    value), Claude Code under `x-claude-code-session-id`, and no `x-bili-session-id` appeared, so no
    billion-context change is needed. DSH also exposed the failure the first design left: its session
    spends the name on a two-message title call first, and the agent turns that follow continue a
    different history. Guarding the name by `--session-retention-floor` alone let that title call
    keep the slot, so the real turns published anonymously and the conversation still accumulated
    variants - the original symptom. Matching the continued history instead of trusting the floor is
    what fixed it.
  - Interleaved live check (same traced 3456 instance, two Claude Code conversations alternating):
    the slot discipline held - one entry each, four `rewrite` lines, never more than two entries -
    but **every returning turn re-prefilled the entire prompt (cache 0)**. The trace showed that the
    returning history shares all but the engine's one-token generation opener, so none of the three
    boundaries the entry offered (frontier image 7,471, prompt-end image 7,470, divergence image 0)
    lay at or below the shared prefix 7,463, and the divergence image the prefill itself captured at
    6,144 was destroyed by the very publish that replaced the entry. Rewriting in place keeps it.
    Re-measured after the fix: each conversation's third turn reuses 9,216 of 12,930 tokens (71.3%,
    `private endpoint`) at TTFT 2.5 s, against 6.4 s for the same 12,930-token prompt one turn
    earlier; the deterministic harness reports 10/10 assertions with the rewrite path exercised
    seven times and 3,072-7,168 tokens reused after each rewrite.
  - Three-agent live confirmation (2026-10-17, same 3456 instance): DSH+bili (`/v1/responses`,
    `x-session-id`), Claude Code+bili and Claude Code alone (`/v1/messages`,
    `x-claude-code-session-id`) each opened (~2.7k/14.6k, 26.7k, 24.6k tokens) and then took turns
    in the same order twice more. Each conversation kept exactly one entry (entries reached 5 of 6,
    including two leftovers from the earlier probe, and **no eviction ran at all**), and after its
    opening turn **every return reused 96.5-99.9% of its prompt** - 14,807/15,117 (97.9%),
    26,813/27,115 (98.9%), 25,231/25,271 (99.8%) on the first return, then 96.9-99.9% through the
    second and third rounds. TTFT fell from 7.9-15.5 s on the cold turn to 0.16-1.3 s on the
    returning ones. DSH's title call still opens the name first and the real turn still takes it
    back (`bind=0` then `replaces=1`, `rewrite entry=2 shared=0`), which now costs nothing because
    the entry that could not help is the one being rewritten.
  (docs/serving.md §Conversation identity, docs/tp2-dual-5060ti.md §Cross-session KV pool;
  _temp/20261017-1700_partA_harness.mjs, _temp/20261017-1925_partA_interleave_reuse.mjs)
- **Genuinely open items** (tracked in tp2-plan.md §3, not settled): bf16 KV + MTP
  first-prefill crash; MTP I32 sNaN risk; MTP3 ≥70 tok/s; perplexity on TP-2;
  trigger source of the low-frequency transport failures; B7 settling
  (≥24 reps/arm); sessions plain-route failure; upstream 3b/3c and the 5060 Ti
  SM-count A/B timings (tp2-plan.md §3.8).
