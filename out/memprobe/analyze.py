"""Reconstruct the per-GPU device memory composition of the running ninfer-serve process.

Inputs:
  - artifact <models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_draftall.ninfer (weight bytes per object)
  - tp_split_spec.cpp classification rules (applied verbatim)
  - runtime arena formulas from tp2_generation_core.cpp / planning code (hand-computed here)

Outputs: per-shard weight bytes + per-shard runtime blocks + reconciliation vs nvidia-smi.
"""
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tools.artifact.reader import Artifact

ARTIFACT = pathlib.Path(r"<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_draftall.ninfer")

# --- text geometry (from artifact text config) ---
HIDDEN = 5120
VOCAB = 248320
ATTN_Q = 24 * 256      # query width
ATTN_K = 4 * 256       # kv width
GDN_K = 16 * 128
GDN_V = 48 * 128
INTER = 17408
N_FULL_ATTN_LAYERS = 16   # layer_types: every 4th layer
N_GDN_LAYERS = 48


def align_up(x, a=256):
    return (x + a - 1) // a * a


def mib(b):
    return b / 1048576.0


def main():
    with Artifact(ARTIFACT) as a:
        d = a.directory
        names_by_object = collections.defaultdict(set)
        for name, binding in d.bindings.items():
            if not name.startswith(("text/", "mtp/", "dflash2/", "proposal/", "vision/")):
                continue
            if "parts" in binding:
                for part in binding["parts"]:
                    names_by_object[part["object"]].add(name)
            elif "object" in binding:
                names_by_object[binding["object"]].add(name)

        shard0 = 0
        shard1 = 0
        by_rule = collections.Counter()
        by_component = collections.Counter()
        for oid, names in sorted(names_by_object.items()):
            o = a.by_id.get(oid)
            if o is None or not hasattr(o, "shape"):
                continue  # resource objects are host-only
            b = o.bytes
            primary = sorted(names)[0]
            top = primary.split("/")[0]
            if any(n.startswith("dflash2/candidate_selector/") for n in names):
                rule = "selector->s1"
                by_rule[rule] += b
                by_component[(top, rule)] += b
                shard1 += b
            elif any(nm.startswith("mtp/") for nm in names):
                # config.mtp is only bound when --spec mtp (config.cpp:321); this route is
                # dflash2, so the MTP component is not materialized at all.
                rule = "mtp->not loaded"
                by_rule[rule] += b
                by_component[(top, rule)] += b
            elif any(nm.startswith(p) for nm in names for p in ("dflash2/", "vision/")):
                rule = "vision->s1" if any(nm.startswith("vision/") for nm in names) else "dflash2->s0"
                by_rule[rule] += b
                by_component[(top, rule)] += b
                if rule == "vision->s1":
                    shard1 += b
                else:
                    shard0 += b
            elif len(o.shape) < 2:
                rule = "rank1->rep"
                by_rule[rule] += b
                by_component[(top, rule)] += b
                shard0 += b
                shard1 += b
            else:
                n, k = int(o.shape[0]), int(o.shape[1])
                if primary.endswith("token_embedding"):
                    rule = "embedding->half"
                elif primary.endswith("proposal/head"):
                    rule = "proposal/head->half"
                elif primary.endswith("output_head"):
                    rule = "output_head->half"
                elif n == 2 * ATTN_Q + 2 * ATTN_K and k == HIDDEN:
                    rule = "attn in->half"
                elif n == 2 * GDN_K + 2 * GDN_V and k == HIDDEN:
                    rule = "gdn in->half"
                elif n == 4 and k == 2 * GDN_K + GDN_V:
                    rule = "gdn conv->half"
                elif n == HIDDEN and k == ATTN_Q and primary.endswith("attention/output"):
                    rule = "attn out->half"
                elif n == HIDDEN and k == GDN_V and primary.endswith("gdn/output"):
                    rule = "gdn out->half"
                elif n == 2 * INTER and k == HIDDEN:
                    rule = "gate/up->half"
                elif n == HIDDEN and k == INTER:
                    rule = "ffn down->half"
                else:
                    rule = "other->rep"
                by_rule[rule] += b
                by_component[(top, rule)] += b
                if rule.endswith("half"):
                    shard0 += b // 2
                    shard1 += b // 2
                else:
                    shard0 += b
                    shard1 += b

    print("=== per-shard weight bytes (from artifact, per tp_split_spec) ===")
    print(f"shard 0 (nvidia-smi GPU 0): {shard0:>15,} B = {mib(shard0):>9.2f} MiB = {shard0/2**30:.3f} GiB")
    print(f"shard 1 (nvidia-smi GPU 2): {shard1:>15,} B = {mib(shard1):>9.2f} MiB = {shard1/2**30:.3f} GiB")
    print(f"total weights in artifact : {d.payload_bytes:>15,} B = {mib(d.payload_bytes):.2f} MiB")
    print("\n--- rule totals (MiB, whole-object bytes) ---")
    for (rule), v in sorted(by_rule.items()):
        print(f"  {rule:<18} {mib(v):>10.1f}")
    print("\n--- component x rule (MiB) ---")
    for (top, rule), v in sorted(by_component.items()):
        print(f"  {top:<10} {rule:<18} {mib(v):>10.1f}")

    # --- runtime blocks (formulas from engine source) ---
    capacity = 245760  # --max-context; explicit KV capacity on the TP-2 route
    page = 64
    pages = (capacity + page - 1) // page

    # text KV, k8v4 (Fp8KeyNvfp4Value): K = 256x1B + 2B scale; V = 128B + 16B scale, per head
    kv_heads_shard = 4 // 2
    kv_k = pages * page * 256 * kv_heads_shard * 1          # data
    kv_k_scale = pages * page * 2 * kv_heads_shard          # fp16 scale, 1 per 256
    kv_v = pages * page * 128 * kv_heads_shard              # nvfp4 packed
    kv_v_scale = pages * page * 16 * kv_heads_shard         # u8 scale, 1 per 16
    kv_bytes = align_up(kv_k) + align_up(kv_k_scale) + align_up(kv_v) + align_up(kv_v_scale)
    kv_bytes *= N_FULL_ATTN_LAYERS
    kv_bytes += align_up(pages * 4)  # execution block table [pages, 1] i32

    # GDN state pool: conv 5120x3x2B + recurrent 128x128x24x4B per layer; 4 planes (2 live + 2 snapshots)
    state_plane = N_GDN_LAYERS * (align_up(5120 * 3 * 2) + align_up(128 * 128 * 24 * 4))
    state_arena = (2 + 2) * state_plane

    # GDN replay records: width = draft_tokens+1 = 8, capacity 1, outer = 48 layers
    W = 8
    rec_conv = align_up(5120 * W * N_GDN_LAYERS * 2)
    rec_key = align_up(128 * 8 * W * N_GDN_LAYERS * 2)
    rec_val = align_up(128 * 24 * W * N_GDN_LAYERS * 2)
    rec_gate = align_up(2 * 24 * W * N_GDN_LAYERS * 4)
    rec_bytes = rec_conv + rec_key + rec_val + rec_gate

    # DFlash2 (shard 0 only): context = prefill features + positions + pending + ring
    target_layers = 5
    feat = HIDDEN * target_layers
    prefill_cols = 1024  # prefill chunk
    prefill_feat = align_up(feat * prefill_cols * 2)
    prefill_pos = align_up(prefill_cols * 4)
    pending_feat = align_up(feat * 8 * 1 * 2)
    ring_padded = 2048  # sliding window, already multiple of 128
    ring = 5 * (align_up(128 * ring_padded * 8 * 2) + align_up(128 * ring_padded * 8 * 2))
    dflash_context = prefill_feat + prefill_pos + pending_feat + ring
    dflash_image = ring  # ring payload is the reuse-boundary image
    draft_snap = 2 * dflash_image

    # DFlash2 frame (round state, dflash2 backend, window 7, batch 1)
    frame = (
        4 * 256                                   # token/pos/rope_pos/rope_delta scalars
        + align_up(VOCAB * 2)                     # step logits bf16
        + 3 * 4 + 2 * 256                         # kv table rows + produced_count scalar (approx)
        + align_up(1536)                          # DFlashDecodeIngress (approx 1.5 KiB)
        + align_up(576)                           # DFlashDecodeEgress
        + 5 * align_up(8 * 4)                     # id arrays [8,1]
        + align_up(16 * 7 * 4)                    # candidate_ids [16,7,1] i32
        + align_up(16 * 7 * 4)                    # proposal_q fp32
        + align_up(2 * 7 * 4)                     # append_positions/counts
        + align_up(VOCAB * 8 * 2)                 # target_logits [V,8] bf16
        + align_up(HIDDEN * 8 * 2)                # target_hidden
        + align_up(HIDDEN * 2)                    # target_continuation_hidden
    )
    # proposal workspace: q4/q5/q8/bf16 linear ops report zero transient capacity;
    # the SWA/selector scratch is small at 8 columns -> bounded by the largest activation
    # re-materialization the proposal forward performs (draft mlp up 17408x8x2B x2, qkv 3*5120x8x2B).
    proposal_ws = align_up(2 * INTER * 8 * 2) + align_up(3 * HIDDEN * 8 * 2) + align_up(HIDDEN * 8 * 2) + 256

    workspace = 192 << 20  # kWorkspaceBytes, both shards

    # Vision workspace (shard 1): max_merged_tokens 8192, patches 32768.
    # build_workspace_layout uses scoped lifetimes: the patch/position/attention/mlp/merger
    # regions are serial, so the encode peak is x + max(branch peaks) (q4/q5/q8 linear scratch = 0).
    patches, tokens = 8192 * 4, 8192
    vis_x = align_up(1152 * patches * 2)
    vis_patch_peak = align_up(patches * 2 * 4) + align_up(512 * patches * 2) + 0  # pos_ids + patches
    vis_pos_peak = align_up(patches * 2 * 4) + align_up(4 * patches * 4) + align_up(4 * patches * 4)
    vis_attn_peak = align_up(3 * 1152 * patches * 2) + align_up(1152 * patches * 2)
    vis_mlp_peak = align_up(4304 * patches * 2) + align_up(1152 * patches * 2)
    vis_merger_peak = align_up(1152 * patches * 2)
    vis_encode = vis_x + max(vis_patch_peak, vis_pos_peak, vis_attn_peak, vis_mlp_peak,
                             vis_merger_peak)
    vis_handoff = align_up(192 << 20) + align_up(tokens * 5120 * 2)  # offset + handoff
    vision = max(vis_encode, vis_handoff)

    blocks_common = {
        "kv (k8v4, 245760 tok)": kv_bytes,
        "gdn state arena (4 planes)": state_arena,
        "gdn replay records (w=8)": rec_bytes,
        "text workspace arena": workspace,
    }
    blocks0 = dict(blocks_common)
    blocks0["dflash2 context (ring 40MiB)"] = dflash_context
    blocks0["dflash2 reuse snapshots (x2)"] = draft_snap
    blocks0["dflash2 frame"] = frame
    blocks0["dflash2 proposal ws"] = proposal_ws
    blocks1 = dict(blocks_common)
    blocks1["vision workspace (8192 tok)"] = vision

    print("\n=== runtime device blocks (MiB) ===")
    labels = [k for k in blocks0] + [k for k in blocks1 if k not in blocks0]
    for label in labels:
        b0, b1 = blocks0.get(label, 0), blocks1.get(label, 0)
        print(f"  {label:<34} shard0 {mib(b0):>9.2f}   shard1 {mib(b1):>9.2f}")

    total0 = shard0 + sum(blocks0.values())
    total1 = shard1 + sum(blocks1.values())
    print(f"\n=== reconstructed engine totals ===")
    print(f"shard 0: weights {mib(shard0):>9.2f} + blocks {mib(sum(blocks0.values())):>9.2f} = {mib(total0):>9.2f} MiB")
    print(f"shard 1: weights {mib(shard1):>9.2f} + blocks {mib(sum(blocks1.values())):>9.2f} = {mib(total1):>9.2f} MiB")
    gpu0_used = 15444.0  # nvidia-smi MiB (fresh reading)
    gpu2_used = 15036.0
    print(f"\nnvidia-smi used: shard0 {gpu0_used:.0f} MiB, shard1 {gpu2_used:.0f} MiB")
    print(f"residual (CUDA context + driver + alignment): shard0 {gpu0_used - mib(total0):.1f} MiB, "
          f"shard1 {gpu2_used - mib(total1):.1f} MiB")
    print(f"\ntrue free per nvidia-smi: GPU0 {16311 - gpu0_used:.0f} MiB, GPU2 {16311 - gpu2_used:.0f} MiB, "
          f"GPU1(T10) 16160 MiB")


if __name__ == "__main__":
    main()
