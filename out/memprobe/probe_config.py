"""Dump the text/vision/draft component configs of the running artifact."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tools.artifact.reader import Artifact

p = pathlib.Path(r"<models-dir>/qwen3_8_27b_w4a4_w8a8_dflash2_draftall.ninfer")
with Artifact(p) as a:
    d = a.directory
    print("artifact_id:", a.artifact_id.hex())
    print("payload_bytes:", d.payload_bytes)
    print("components:", sorted(d.components.keys()))
    for name in ("text", "mtp", "vision", "dflash2", "proposal"):
        if name in d.components:
            print(f"\n===== component {name} =====")
            print(json.dumps(d.components[name], indent=1, ensure_ascii=False)[:6000])
