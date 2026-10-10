import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tools.artifact.reader import Artifact

with Artifact(pathlib.Path(r"<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_draftall.ninfer")) as a:
    d = a.directory
    names_by_object = collections.defaultdict(set)
    for name, binding in d.bindings.items():
        if not name.startswith(("dflash2/", "proposal/")):
            continue
        if "parts" in binding:
            for part in binding["parts"]:
                names_by_object[part["object"]].add(name)
        else:
            names_by_object[binding["object"]].add(name)

    # per-layer template: name with layer index stripped
    import re

    by_template = collections.defaultdict(lambda: [0, 0, ""])
    for oid, names in sorted(names_by_object.items()):
        o = a.by_id.get(oid)
        if o is None or not hasattr(o, "shape"):
            continue
        primary = sorted(names)[0]
        tmpl = re.sub(r"layers/\d+", "layers/N", primary)
        # per-shard placement
        if primary.startswith("dflash2/candidate_selector/"):
            place = "shard1 (selector)"
        elif primary.startswith("dflash2/"):
            place = "shard0 (draft)"
        else:
            place = "both (proposal)"
        by_template[(tmpl, o.format, place)][0] += o.bytes
        by_template[(tmpl, o.format, place)][1] += 1
        by_template[(tmpl, o.format, place)][2] = list(o.shape).__str__()

    total = collections.Counter()
    for (tmpl, fmt, place), (b, cnt, shape) in sorted(by_template.items(), key=lambda kv: -kv[1][0]):
        print(f"{b/1048576:9.2f} MiB  x{cnt:<3} {fmt:<14} {shape:<18} {place:<20} {tmpl}")
        total[place] += b
    print()
    for place, b in total.items():
        print(f"{place:<20} {b/1048576:9.2f} MiB")
