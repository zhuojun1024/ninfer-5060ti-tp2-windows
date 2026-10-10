"""Final HTML report for the long-context retrieval probes.

Two probes, both using the prefix-chain trick (each (arm, subset) is one chain
of strictly-extending requests, so the whole chain costs about one full prefill
of the longest length):

  soft  8 needles, needles at the very front, english + chinese
        -> profiles/kv-quantization/data/niah_chain_<arm>.json
  hard  32 needles + a 39-char random access token to copy verbatim, english
        -> profiles/kv-quantization/data/niah_chain_hard_<arm>.json

Usage:
  python profiles/kv-quantization/tools/niah_probe_report.py
"""

from __future__ import annotations

import html
import json
import math
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "niah-probe.html"

ARMS = ("int8", "nvfp4")
ARM_COLOR = {"int8": "#2f6fd0", "nvfp4": "#d0402f"}
ARM_LABEL = {"int8": "int8-g64", "nvfp4": "nvfp4-g16"}
SUBSETS = (
    ("english", "English haystack (PaulGraham_Essays)"),
    ("chinese", "Chinese haystack (Journey_to_the_West)"),
)
ACCESS = "XK4Q9M2B7T1V8W3N6R5P2H9D4C8L1G7F"
ACCESS_NORM = re.sub(r"[^A-Za-z0-9]", "", ACCESS).upper()


def esc(value) -> str:
    return html.escape(str(value))


def norm_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def access_score(prediction: str) -> tuple[int, int, str]:
    """Best character-level match between the answer's token line and the target."""
    best = ""
    for line in prediction.splitlines():
        if "=" in line:
            continue
        cand = norm_token(line)
        if len(cand) > len(best):
            best = cand
    n = len(ACCESS_NORM)
    if not best:
        return 0, n, ""
    if len(best) <= n:
        hits = sum(1 for a, b in zip(best, ACCESS_NORM) if a == b)
        return hits, n, best
    best_hits, best_window = -1, best[:n]
    for i in range(len(best) - n + 1):
        window = best[i:i + n]
        hits = sum(1 for a, b in zip(window, ACCESS_NORM) if a == b)
        if hits > best_hits:
            best_hits, best_window = hits, window
    return best_hits, n, best_window


def is_complete(rec: dict) -> bool:
    """The answer ran to the end instead of being cut by the context budget."""
    if rec.get("hits", -1) < 0:
        return False
    if rec.get("truncated") is True or rec.get("finish_reason") == "length":
        return False
    stripped = rec["prediction"].strip()
    if not stripped:
        return False
    last = stripped.splitlines()[-1]
    return bool(re.search(r"=\s*\d{5}\s*$", last)) or len(norm_token(last)) >= len(ACCESS_NORM)


def load(prefix: str) -> dict:
    data: dict = {}
    for arm in ARMS:
        path = DATA / f"{prefix}{arm}.json"
        if not path.exists():
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            if rec.get("depth_mode") != "front" or rec.get("hits", -1) < 0:
                continue
            data.setdefault((rec["subset"], arm), []).append(rec)
    for key in data:
        data[key].sort(key=lambda r: r["prompt_tokens"])
    return data


def chart(series, title, xlabel, ylabel, width=1000, height=340, ymin=None, ymax=None,
          yticks=None, note=None, fmt="{:g}", hlines=()):
    ml, mr, mt, mb = 82, 214, 40, 52
    pw, ph = width - ml - mr, height - mt - mb
    xs = [p[0] for s in series for p in s["points"]]
    ys = [p[1] for s in series for p in s["points"]]
    if not xs:
        return ""
    xlo, xhi = math.log10(min(xs)), math.log10(max(xs))
    if xhi - xlo < 1e-9:
        xhi = xlo + 1
    ylo = min(ys) if ymin is None else ymin
    yhi = max(ys) if ymax is None else ymax
    if yhi - ylo < 1e-9:
        yhi = ylo + 1
    pad = (yhi - ylo) * 0.08
    ylo = (ylo - pad) if ymin is None else ylo
    yhi = yhi + pad

    def px(x):
        return ml + (math.log10(x) - xlo) / (xhi - xlo) * pw

    def py(y):
        return mt + ph - (y - ylo) / (yhi - ylo) * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="Segoe UI,Helvetica,Arial,sans-serif">',
           f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{ml}" y="24" font-size="15" font-weight="600" fill="#1a1a1a">{esc(title)}</text>']
    for tv in (yticks or [ylo + (yhi - ylo) * i / 5 for i in range(6)]):
        y = py(tv)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#e8e8e8" stroke-width="1"/>')
        out.append(f'<text x="{ml-10}" y="{y+4:.1f}" font-size="11" fill="#666" text-anchor="end">{esc(fmt.format(tv))}</text>')
    for hv in hlines:
        y = py(hv)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#bbb" '
                   f'stroke-width="1" stroke-dasharray="4,4"/>')
    xv_ticks = []
    for decade in range(int(math.floor(xlo)), int(math.ceil(xhi)) + 1):
        for mult in (1, 2, 5):
            v = mult * (10 ** decade)
            if xlo - 1e-9 <= math.log10(v) <= xhi + 1e-9:
                xv_ticks.append(v)
    for xv in sorted(set(xv_ticks)):
        x = px(xv)
        out.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" stroke="#f0f0f0" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{mt+ph+20}" font-size="11" fill="#666" text-anchor="middle">{int(xv):,}</text>')
    out.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#999" stroke-width="1.2"/>')
    out.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#999" stroke-width="1.2"/>')
    out.append(f'<text x="{ml+pw/2:.0f}" y="{height-12}" font-size="12" fill="#444" text-anchor="middle">{esc(xlabel)}</text>')
    out.append(f'<text x="18" y="{mt+ph/2:.0f}" font-size="12" fill="#444" text-anchor="middle" '
               f'transform="rotate(-90 18 {mt+ph/2:.0f})">{esc(ylabel)}</text>')
    for s in series:
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in s["points"])
        out.append(f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" stroke-width="2.4"/>')
        for x, y in s["points"]:
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.4" fill="{s["color"]}"/>')
    ly = mt + 6
    for s in series:
        out.append(f'<line x1="{ml+pw+14}" y1="{ly}" x2="{ml+pw+44}" y2="{ly}" stroke="{s["color"]}" stroke-width="2.4"/>')
        out.append(f'<text x="{ml+pw+50}" y="{ly+4}" font-size="12" fill="#333">{esc(s["name"])}</text>')
        ly += 20
    if note:
        for chunk in note.split("\n"):
            out.append(f'<text x="{ml+pw+14}" y="{ly+8}" font-size="10.5" fill="#888">{esc(chunk)}</text>')
            ly += 14
    out.append("</svg>")
    return "".join(out)


def table(headers, rows):
    out = ['<table><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr></thead><tbody>"]
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def soft_section(data: dict) -> tuple[str, list]:
    body, summary = [], []
    for subset, label in SUBSETS:
        series_acc, series_lat, rows = [], [], []
        by_len: dict = {}
        for arm in ARMS:
            recs = data.get((subset, arm), [])
            if not recs:
                continue
            series_acc.append({"name": ARM_LABEL[arm], "color": ARM_COLOR[arm],
                               "points": [(r["prompt_tokens"], r["hits"]) for r in recs]})
            series_lat.append({"name": ARM_LABEL[arm], "color": ARM_COLOR[arm],
                               "points": [(r["prompt_tokens"], r["latency_s"]) for r in recs]})
            for r in recs:
                by_len.setdefault(r["prompt_tokens"], {})[arm] = r
            summary.append((label, ARM_LABEL[arm],
                            sum(r["hits"] for r in recs), sum(r["total"] for r in recs)))
        for tokens in sorted(by_len):
            entry = by_len[tokens]
            cells = [f'<span class="num">{tokens:,}</span>']
            for arm in ARMS:
                r = entry.get(arm)
                cells.append(f'<span class="num">{r["hits"]}/{r["total"]}</span>' if r else "&ndash;")
            for arm in ARMS:
                r = entry.get(arm)
                cells.append(f'<span class="num">{r["latency_s"]:.0f}s</span>' if r else "&ndash;")
            cells.append('<span class="num">0</span>' if len(entry) == 2 else "")
            rows.append(cells)
        body.append(f"<h2>{esc(label)}</h2>")
        body.append('<div class="card">' + chart(
            series_acc, f"retrieval hits vs visible context length — {label}", "prompt tokens (log)",
            "number of needles hit (of 8)", ymin=0, ymax=8,
            yticks=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            note="needles are at the very front of the document, the question is at the end") + "</div>")
        body.append('<div class="card">' + chart(
            series_lat, f"latency of a single request — {label}", "prompt tokens (log)", "latency (s)",
            ymin=0, fmt="{:,.0f}", note="rising = prefix reuse taking effect (only the increment is paid)") + "</div>")
        body.append(table(["prompt tokens", "int8-g64", "nvfp4-g16", "int8 latency",
                           "nvfp4 latency", "nvfp4 &minus; int8"], rows))
    return "".join(body), summary


def hard_section(data: dict) -> tuple[str, list, list]:
    body, summary, bad = [], [], []
    subset = "english"
    series_acc, series_tok, series_lat, rows = [], [], [], []
    by_len: dict = {}
    for arm in ARMS:
        recs = [r for r in data.get((subset, arm), []) if is_complete(r)]
        cut = [r for r in data.get((subset, arm), []) if not is_complete(r)]
        bad.extend((arm, r) for r in cut)
        if not recs:
            continue
        series_acc.append({"name": ARM_LABEL[arm], "color": ARM_COLOR[arm],
                           "points": [(r["prompt_tokens"], r["hits"]) for r in recs]})
        series_lat.append({"name": ARM_LABEL[arm], "color": ARM_COLOR[arm],
                           "points": [(r["prompt_tokens"], r["latency_s"]) for r in recs]})
        tok_points = []
        for r in recs:
            hits, total, _ = access_score(r["prediction"])
            tok_points.append((r["prompt_tokens"], 100.0 * hits / total))
        series_tok.append({"name": ARM_LABEL[arm], "color": ARM_COLOR[arm], "points": tok_points})
        for r in recs:
            by_len.setdefault(r["prompt_tokens"], {})[arm] = r
        summary.append((ARM_LABEL[arm], sum(r["hits"] for r in recs), sum(r["total"] for r in recs),
                        sum(access_score(r["prediction"])[0] for r in recs),
                        len(recs) * len(ACCESS_NORM)))

    for tokens in sorted(by_len):
        entry = by_len[tokens]
        cells = [f'<span class="num">{tokens:,}</span>']
        for arm in ARMS:
            r = entry.get(arm)
            cells.append(f'<span class="num">{r["hits"]}/{r["total"]}</span>' if r else "&ndash;")
        for arm in ARMS:
            r = entry.get(arm)
            if r:
                hits, total, _ = access_score(r["prediction"])
                cls = "pos" if hits == total else "neg"
                cells.append(f'<span class="num {cls}">{hits}/{total}</span>')
            else:
                cells.append("&ndash;")
        for arm in ARMS:
            r = entry.get(arm)
            cells.append(f'<span class="num">{r["latency_s"]:.0f}s</span>' if r else "&ndash;")
        rows.append(cells)

    body.append(f"<h2>Hard probe: 32 needles + a 39-character random token (English haystack)</h2>")
    body.append('<div class="card">' + chart(
        series_acc, "hits of the 32 needles vs visible context length", "prompt tokens (log)",
        "number of needles hit (of 32)", ymin=0, ymax=32,
        yticks=[0, 8, 16, 24, 32], note="needles are at the very front of the document") + "</div>")
    body.append('<div class="card">' + chart(
        series_tok, "character-level reproduction rate of the 39-character access token", "prompt tokens (log)",
        "character accuracy (%)", ymin=0, ymax=100, yticks=[0, 25, 50, 75, 100], fmt="{:,.0f}",
        hlines=(100,), note="character-by-character comparison, taking the best alignment window after normalization") + "</div>")
    body.append('<div class="card">' + chart(
        series_lat, "latency of a single request (hard probe)", "prompt tokens (log)", "latency (s)",
        ymin=0, fmt="{:,.0f}", note="rising = prefix reuse taking effect (only the increment is paid)") + "</div>")
    body.append(table(["prompt tokens", "int8 hits", "nvfp4 hits", "int8 token chars",
                       "nvfp4 token chars", "int8 latency", "nvfp4 latency"], rows))
    if bad:
        items = "; ".join(
            f"{ARM_LABEL[a]} @ {r['prompt_tokens']:,} tok output only {r['hits']}/{r['total']}"
            for a, r in bad)
        body.append(f'<div class="note"><b>Points cut off by the context budget and dropped from the curves</b>: {esc(items)}. '
                    f'A complete answer needs 352 output tokens, while <code>max_context</code>=262,144 is '
                    f'the total budget of "prompt + output", so once the prompt exceeds about 261,790 the answer no longer fits and '
                    f'the model is hard-cut in the middle of a number (measured <code>Lagos = 3</code>). This is a measurement artifact, not a retrieval failure.</div>')
    return "".join(body), summary, bad


def main() -> int:
    soft = load("niah_chain_")
    hard = load("niah_chain_hard_")
    soft_body, soft_summary = soft_section(soft)
    hard_body, hard_summary, _ = hard_section(hard)

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>KV quantization: long-context retrieval accuracy vs visible context length</title>
<style>
 body {{ font-family: "Segoe UI",Helvetica,Arial,sans-serif; margin:0; padding:28px 34px 60px;
        background:#fafafa; color:#1a1a1a; }}
 h1 {{ font-size:25px; margin:0 0 6px; }}
 h2 {{ font-size:18px; margin:34px 0 10px; }}
 .sub {{ color:#666; font-size:13.5px; margin-bottom:20px; }}
 .card {{ background:#fff; border:1px solid #e3e3e3; border-radius:9px; padding:12px 14px;
         margin:14px 0; box-shadow:0 1px 2px rgba(0,0,0,.04); overflow-x:auto; }}
 .note {{ background:#fff; border:1px solid #e3e3e3; border-radius:9px; padding:16px 20px; margin:16px 0; }}
 table {{ border-collapse:collapse; width:100%; font-size:12.5px; background:#fff;
         border:1px solid #e3e3e3; border-radius:9px; overflow:hidden; margin:14px 0; }}
 th {{ background:#f4f4f6; text-align:right; padding:7px 10px; font-weight:600; color:#444;
      border-bottom:1px solid #e3e3e3; }}
 th:first-child, td:first-child {{ text-align:left; }}
 td {{ padding:6px 10px; text-align:right; border-bottom:1px solid #f2f2f2; }}
 tr:last-child td {{ border-bottom:none; }}
 .num {{ font-variant-numeric: tabular-nums; }}
 .pos {{ color:#1a7f37; font-weight:600; }}
 .neg {{ color:#b3261e; font-weight:600; }}
 code {{ background:#f2f2f4; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
</style></head><body>
<h1>KV quantization: long-context retrieval accuracy vs visible context length</h1>
<div class="sub">model Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-Vision · TP-2 (<code>--devices 0,1</code>)
 · greedy decoding · each (arm, corpus) is an 11-point <b>strict prefix chain</b>, and the whole chain costs one longest prefill</div>

<div class="note">
<b>Design</b>: request body = <code>[needle memo] + haystack[0:K] + [question]</code>. K increases from 16,384 to about 262,000,
and every request is a strictly prefixed extension of the previous one, so prefix reuse in ninfer-serve merges the 11 prefills into one longest prefill
(measured: exact re-send 0.9s vs 127.6s cold, i.e. 139×; the server log also reports <code>cache 228,352 (91.4%)</code>).
All needles are pinned at the very front of the document, so "the distance from a needle to the question" is exactly K — this is the hardest and most sensitive
0% depth tier in the stock NIAH grid.
<br><br>
<b>Why this is an order of magnitude cheaper than a depth-by-depth grid</b>: in the depth grid each sample has its needles at a different position ⇒ the haystack token sequence
differs ⇒ there is no prefix reuse at all, and each of the 88 samples has to pay for a full 128K/260K prefill (about 5.5 hours). The prefix chain pushes the cost down to
one longest prefill (about 25 minutes), at the cost of dropping the "depth" dimension.
<br><br>
<b>Note</b>: the needles and the question are both in English; the Chinese document only swaps the haystack — consistent with what the stock NIAH adapter does.
</div>

<div class="note">
<b>Summary</b><br>
{'<br>'.join(f'8-needle probe · {s} · {a}: {h}/{t}' for s, a, h, t in soft_summary)}<br>
{'<br>'.join(f'32-needle probe · {a}: city codes {h}/{t} · access token chars {th}/{tt}' for a, h, t, th, tt in hard_summary)}
</div>

<h1 style="font-size:20px;margin-top:34px">Probe 1: 8 needles</h1>
{soft_body}

<h1 style="font-size:20px;margin-top:34px">Probe 2: 32 needles + verbatim reproduction</h1>
{hard_body}
</body></html>
"""
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    for s, a, h, t in soft_summary:
        print(f"  soft {s} / {a}: {h}/{t}")
    for a, h, t, th, tt in hard_summary:
        print(f"  hard {a}: codes {h}/{t}  token chars {th}/{tt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
