import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tools.artifact.reader import Artifact

HIDDEN, ATTN_Q, ATTN_K = 5120, 24 * 256, 4 * 256
GDN_K, GDN_V, INTER = 16 * 128, 48 * 128, 17408

with Artifact(pathlib.Path(r"<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_draftall.ninfer")) as a:
    d = a.directory
    names_by_object = collections.defaultdict(set)
    for name, binding in d.bindings.items():
        if not name.startswith(("text/", "mtp/", "dflash2/", "proposal/", "vision/")):
            continue
        if "parts" in binding:
            for part in binding["parts"]:
                names_by_object[part["object"]].add(name)
        else:
            names_by_object[binding["object"]].add(name)

    for oid, names in sorted(names_by_object.items()):
        o = a.by_id.get(oid)
        if o is None or not hasattr(o, "shape"):
            continue
        n, k = (int(o.shape[0]), int(o.shape[1])) if len(o.shape) >= 2 else (0, 0)
        if any(x.startswith(("dflash2/candidate_selector/", "mtp/", "vision/")) for x in names):
            continue
        matches = (
            names and (sorted(names)[0].endswith("token_embedding")
                       or sorted(names)[0].endswith("proposal/head")
                       or sorted(names)[0].endswith("output_head")
                       or (n == 2 * ATTN_Q + 2 * ATTN_K and k == HIDDEN)
                       or (n == 2 * GDN_K + 2 * GDN_V and k == HIDDEN)
                       or (n == 4 and k == 2 * GDN_K + GDN_V)
                       or (n == HIDDEN and k == ATTN_Q and any(x.endswith("attention/output") for x in names))
                       or (n == HIDDEN and k == GDN_V and any(x.endswith("gdn/output") for x in names))
                       or (n == 2 * INTER and k == HIDDEN)
                       or (n == HIDDEN and k == INTER)))
        if not matches:
            print(f"{o.bytes/1048576:8.2f} MiB  {list(o.shape)!s:<20} {o.format:<14} {sorted(names)}")
