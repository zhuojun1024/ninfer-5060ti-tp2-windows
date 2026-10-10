"""Cheap long-context retrieval sweep: one prefix chain per (arm, subset).

Every request is a strict extension of the previous one:

    [needles] + haystack[0:K] + [question]

so ninfer-serve's prefix reuse turns 11 full prefills into a single prefill of
the longest length (verified: an exact repeat costs 0.9s vs 127.6s cold, and a
+32K extension costs 50.5s vs ~157s cold).

The needles sit at the very front, so the distance between the fact and the
question is exactly K - the hardest position in a NIAH grid. Eight independent
needles make the score a rate (0/8 ... 8/8) rather than a coin flip.

Usage:
  python profiles/kv-quantization/tools/niah_chain_sweep.py --arm int8 --port 18080 \
      --out profiles/kv-quantization/data/niah_chain_int8.json

Both haystack texts come from the evalscope NIAH dataset
(AI-ModelScope/Needle-in-a-Haystack-Corpus), which this repository does not vendor.
Point --corpus-dir at a copy of that dataset's two text files.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[3]
CORPUS_FILES = {
    "english": "PaulGraham_Essays.txt",
    "chinese": "Journey_to_the_West.txt",
}
DEFAULT_CORPUS_DIR = ROOT / "_temp" / "niah_corpus"
DEFAULT_TOKENIZER_PATH = r"<models-dir>/Qwen3.8-27B-NVFP4"

# 11 visible-length targets, spanning the 128K and 260K points the suite used.
# The last entry is the *total* prompt budget: max_context is 262144, and the
# preamble + question eat ~192 tokens, so the haystack stops just short of it.
LENGTHS = (16384, 32768, 49152, 65536, 98304, 131072, 163840, 196608, 229376, 249856, 261700)

NEEDLES = (
    ("Amsterdam", "48173"),
    ("Brisbane", "60412"),
    ("Cairo", "73528"),
    ("Dublin", "15096"),
    ("Edinburgh", "82961"),
    ("Florence", "30457"),
    ("Geneva", "97204"),
    ("Helsinki", "55830"),
)
PREAMBLE = (
    "Reference sheet. Each line records the vault code of one office.\n"
    + "".join(f"The vault code for the {city} office is {code}.\n" for city, code in NEEDLES)
    + "\n"
)
QUESTION = (
    "\n\nQuestion: The reference sheet at the very beginning of this document lists "
    "the vault code of eight offices. For each of those eight offices, state its "
    "vault code. Answer with one line per office in the form `City = Code`, and "
    "nothing else."
)


def load_tokenizer(path: str):
    from modelscope import AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(path)
    return tok


def build_haystack(text: str, tokens: int, tok) -> str:
    """Corpus text long enough for `tokens` tokens, truncated at a token boundary."""
    encoded = tok(text, add_special_tokens=False)["input_ids"]
    if len(encoded) >= tokens:
        return tok.decode(encoded[:tokens], skip_special_tokens=True)
    # Grow by whole copies until long enough, then truncate.
    grown = text
    while len(encoded) < tokens:
        grown = grown + "\n" + text
        encoded = tok(grown, add_special_tokens=False)["input_ids"]
    return tok.decode(encoded[:tokens], skip_special_tokens=True)


def ask(prompt: str, port: int) -> tuple[str, float, int, str]:
    body = json.dumps(
        {
            "model": "qwen38-tp2",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 700,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        payload = json.loads(resp.read())
    elapsed = time.time() - started
    choice = payload["choices"][0]
    content = choice["message"]["content"] or ""
    finish = str(choice.get("finish_reason") or choice.get("stop_reason") or "")
    return content, elapsed, int(payload.get("usage", {}).get("prompt_tokens") or 0), finish


def score(prediction: str) -> tuple[int, list[str]]:
    """Count offices whose code appears correctly paired in the answer."""
    hits, missing = 0, []
    for city, code in NEEDLES:
        # Accept "City = Code", "City: Code" or a line holding both.
        pattern = rf"{city}[^0-9\n]{{0,12}}{code}"
        if re.search(pattern, prediction, flags=re.IGNORECASE):
            hits += 1
        else:
            missing.append(city)
    return hits, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--out", required=True)
    parser.add_argument("--subsets", default="english,chinese")
    parser.add_argument("--depth", default="front", choices=("front", "back"),
                        help="front = needles first (hardest), back = needles just before the question")
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR,
                        help="directory holding the evalscope NIAH haystack text files")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_PATH)
    args = parser.parse_args()

    tok = load_tokenizer(args.tokenizer)
    overhead = len(tok(PREAMBLE + QUESTION, add_special_tokens=False)["input_ids"])
    subsets = [s for s in args.subsets.split(",") if s]

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    done = {(r["subset"], r["depth_mode"], r["target_tokens"]) for r in records}
    if done:
        print(f"resuming {out_path.name}: {len(done)} point(s) already collected", flush=True)

    for subset in subsets:
        text = (args.corpus_dir / CORPUS_FILES[subset]).read_text(encoding="utf-8")
        print(f"\n=== {args.arm} / {subset} ({args.depth}) ===", flush=True)
        for target in LENGTHS:
            if (subset, args.depth, target) in done:
                print(f"  target={target:>7} skip (already collected)", flush=True)
                continue
            hay = build_haystack(text, max(target - overhead, 1024), tok)
            if args.depth == "front":
                prompt = PREAMBLE + hay + QUESTION
            else:
                prompt = hay + "\n\n" + PREAMBLE + QUESTION
            try:
                prediction, elapsed, prompt_tokens, finish = ask(prompt, args.port)
            except Exception as exc:  # keep the chain alive; record the failure
                print(f"  target={target:>7} FAILED: {exc}", flush=True)
                records.append(
                    {
                        "arm": args.arm, "subset": subset, "depth_mode": args.depth,
                        "target_tokens": target, "prompt_tokens": 0,
                        "latency_s": 0.0, "hits": -1, "total": len(NEEDLES),
                        "missing": [c for c, _ in NEEDLES], "prediction": f"ERROR: {exc}",
                    }
                )
                out_path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
                continue
            hits, missing = score(prediction)
            records.append(
                {
                    "arm": args.arm,
                    "subset": subset,
                    "depth_mode": args.depth,
                    "target_tokens": target,
                    "prompt_tokens": prompt_tokens,
                    "latency_s": round(elapsed, 1),
                    "hits": hits,
                    "total": len(NEEDLES),
                    "missing": missing,
                    "finish_reason": finish,
                    "truncated": finish == "length",
                    "prediction": prediction,
                }
            )
            print(
                f"  target={target:>7} actual={prompt_tokens:>7}  {hits}/{len(NEEDLES)}"
                f"  {elapsed:>7.1f}s  missing={','.join(missing) or '-'}",
                flush=True,
            )
            out_path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
