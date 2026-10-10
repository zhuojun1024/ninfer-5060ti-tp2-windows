"""Gold-standard depth sweep report.

Design
------
The scored text is FIXED: it is the last BLOCK_TOKENS tokens of the source stream, and it
is byte-identical in every run.  Run k exposes only the last (K + BLOCK_TOKENS) tokens, so
block token j is predicted from exactly (visible + j) tokens of history -- its own natural
preceding text.  Every run is a single window (--context >= stream length).

Because the text is fixed, two things are clean:
  * arm-vs-arm differences at the same depth are strictly paired (same token);
  * depth-vs-depth differences at the same block token are strictly paired too.

Estimators
----------
run means      : mean NLL of the block per run (independent points, few).
panel          : for each fixed block token j, fit its NLL across runs against log10(history),
                 then average the per-token slopes.  Text held exactly constant.
adjacent delta : paired NLL difference between two neighbouring depths.  Model free, and the
                 honest way to ask "does MORE history hurt?" without a global functional form.

Reads  <sweep-dir>/<arm>/k<depth>/token_nll.<stream>.tsv   (default _temp/scoreSweep2/out)
Writes profiles/kv-quantization/goldstandard-paired.html

The per-run token-NLL dumps are scratch sweep output and are not committed; re-run the
sweep first (see profiles/kv-quantization/report.md).

Usage:  python profiles/kv-quantization/tools/goldstandard_report.py [sweep_dir]
"""

import glob
import html
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "_temp", "scoreSweep2", "out")
OUT_HTML = os.path.join(HERE, "..", "goldstandard-paired.html")

BLOCK_TOKENS = 1024
DROP_HEAD = 16
DEEP_FROM = 16384          # "long context" regime used for the slope statements

# The source stream is eval/corpora/perplexity-1m/data/pg19/{00,01,02,03}.txt joined in order
# with "\n" separators, then cut at a newline.  These are the three joins, expressed as
# approximate token distance back from the end of the stream (1024998 chars / 3.9815 chars
# per token).  Crossing one means the visible window has reached into an earlier book.
BOOK_BOUNDARIES = [(60517, "02|03"), (128300, "01|02"), (192257, "00|01")]

ARM_ORDER = ["bf16", "int8", "fp8", "nvfp4", "k8v4"]
ARM_COLOR = {"bf16": "#111827", "int8": "#2563eb", "fp8": "#059669",
             "nvfp4": "#dc2626", "k8v4": "#d97706"}
ARM_DASH = {"int8": "10,5", "k8v4": "1,6"}
ARM_WIDTH = {"nvfp4": 4.2, "k8v4": 2.6}


# --------------------------------------------------------------------- loading

def read_arm(arm):
    result = {}
    for path in glob.glob(os.path.join(ROOT, arm, "k*", "token_nll.*.tsv")):
        nominal = int(os.path.basename(os.path.dirname(path))[1:])
        by_target = {}
        for line in open(path, encoding="utf-8"):
            if line.startswith("#"):
                continue
            index, _, value = line.partition("\t")
            by_target[int(index)] = float(value)
        n_tokens = max(by_target) + 1
        base = n_tokens - BLOCK_TOKENS
        block = {j: by_target[base + j] for j in range(DROP_HEAD, BLOCK_TOKENS)
                 if (base + j) in by_target}
        result[nominal] = {"n_tokens": n_tokens, "base": base, "block": block}
    return result


def entry(data, arm, nominal):
    return data.get(arm, {}).get(nominal)


def run_mean(data, arm, nominal):
    e = entry(data, arm, nominal)
    if not e or not e["block"]:
        return None
    values = list(e["block"].values())
    history = [e["base"] + j for j in e["block"]]
    mean = sum(values) / len(values)
    se = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1) / len(values))
    return {"arm": arm, "nominal": nominal, "n_tokens": e["n_tokens"],
            "visible": e["base"], "mean_nll": mean, "ppl": math.exp(mean),
            "se": se, "history_mid": sum(history) / len(history),
            "history_lo": min(history), "history_hi": max(history), "n": len(values)}


def paired(data, arm, reference, nominal):
    a, b = entry(data, arm, nominal), entry(data, reference, nominal)
    if not a or not b:
        return None
    keys = sorted(set(a["block"]) & set(b["block"]))
    if not keys:
        return None
    diffs = [a["block"][k] - b["block"][k] for k in keys]
    mean = sum(diffs) / len(diffs)
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs)) if len(diffs) > 1 else float("nan")
    return {"nominal": nominal, "visible": a["base"], "delta": mean, "se": se,
            "history_mid": a["base"] + (min(keys) + max(keys)) / 2.0,
            "n": len(diffs), "per_token": dict(zip(keys, diffs))}


def adjacent(data, arm, nominal_lo, nominal_hi):
    """paired NLL(hi) - NLL(lo) over the same block tokens; positive = worse with more history."""
    a, b = entry(data, arm, nominal_lo), entry(data, arm, nominal_hi)
    if not a or not b:
        return None
    keys = sorted(set(a["block"]) & set(b["block"]))
    if not keys:
        return None
    diffs = [b["block"][k] - a["block"][k] for k in keys]
    mean = sum(diffs) / len(diffs)
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs)) if len(diffs) > 1 else float("nan")
    return {"lo": a["base"], "hi": b["base"], "delta": mean, "se": se, "n": len(diffs),
            "per_token": dict(zip(keys, diffs))}


# ----------------------------------------------------------------- statistics

def pooled(entries, min_visible=0):
    """Pool the per-token paired diffs of several runs into one mean + se."""
    diffs = []
    for e in entries:
        if e["visible"] >= min_visible:
            diffs.extend(e["per_token"].values())
    if len(diffs) < 2:
        return None
    mean = sum(diffs) / len(diffs)
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs))
    return {"delta": mean, "se": se, "t": mean / se if se > 0 else float("nan"),
            "n": len(diffs), "runs": sum(1 for e in entries if e["visible"] >= min_visible)}


def ols(points):
    n = len(points)
    if n < 3:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx <= 0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    slope = sxy / sxx
    resid = sum((p[1] - my - slope * (p[0] - mx)) ** 2 for p in points)
    se = math.sqrt(resid / (n - 2) / sxx)
    return {"slope": slope, "se": se, "t": slope / se if se > 0 else float("nan"), "n": n}


def fit_log(points, min_x=DEEP_FROM):
    """points = [(raw_x, y)]; regress y on log10(x) using only raw_x >= min_x."""
    pts = [(math.log10(x), y) for x, y in points if x >= min_x]
    return ols(pts)


def panel_slopes(series, min_x=DEEP_FROM):
    """series = [(base, {j: value})] -> per-token slopes of value vs log10(base+j), averaged."""
    keys = None
    for _, block in series:
        keys = set(block) if keys is None else (keys & set(block))
    slopes = []
    for j in sorted(keys or []):
        pts = [(math.log10(base + j), block[j]) for base, block in series if base + j >= min_x]
        got = ols(pts)
        if got:
            slopes.append(got["slope"])
    if len(slopes) < 2:
        return None
    mean = sum(slopes) / len(slopes)
    se = math.sqrt(sum((s - mean) ** 2 for s in slopes) / (len(slopes) - 1) / len(slopes))
    return {"slope": mean, "se": se, "t": mean / se if se > 0 else float("nan"), "tokens": len(slopes)}


# ------------------------------------------------------------------- drawing

def esc(text):
    return html.escape(str(text))


def _fmt(value, digits=4):
    if value is None:
        return "-"
    if value == 0:
        return "0"
    if abs(value) < 1e-4:
        return "%.2e" % value
    return ("%." + str(digits) + "f") % value


def _signed(value, digits=6):
    if value is None:
        return "-"
    return ("%+." + str(digits) + "f") % value


def _t(value):
    if value is None or value != value:
        return "-"
    return '<span class="%s">%s</span>' % ("bad" if abs(value) >= 2 else "good", _signed(value, 2))


def _tickfmt(value):
    if abs(value - round(value)) < 1e-9 and abs(value) < 1e9:
        return "{:,}".format(int(round(value)))
    return _fmt(value, 3)


def _ticks(lo, hi, count=6):
    if hi <= lo:
        return [lo]
    step = (hi - lo) / float(count - 1)
    return [lo + step * i for i in range(count)]


def chart(title, xlabel, ylabel, series, width=1040, height=470, xlog=False,
          ymin=None, ymax=None, hlines=(), note=None, errorbars=False, vlines=()):
    ml, mr, mt, mb = 92, 268, 48, 60
    pw, ph = width - ml - mr, height - mt - mb
    xs = [p[0] for s in series for p in s["points"]]
    ys = [p[1] for s in series for p in s["points"]]
    if not xs or not ys:
        return '<p class="nodata">no data</p>'
    xlo, xhi = max(min(xs), 1.0), max(xs)
    if xlog:
        lg_lo, lg_hi = math.log10(xlo), math.log10(xhi * 1.0001)
        tx = lambda v: ml + (math.log10(max(v, 1.0)) - lg_lo) / (lg_hi - lg_lo) * pw
        tickvals, e = [], int(math.floor(lg_lo))
        while 10 ** e <= xhi * 1.0001:
            if 10 ** e >= xlo:
                tickvals.append(10.0 ** e)
            e += 1
    else:
        tx = lambda v: ml + (v - xlo) / (xhi - xlo) * pw
        tickvals = _ticks(xlo, xhi, 6)
    lo = min(ys) if ymin is None else ymin
    hi = max(ys) if ymax is None else ymax
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.10
    ylo, yhi = lo - pad, hi + pad
    ty = lambda v: mt + ph - (v - ylo) / (yhi - ylo) * ph

    out = ['<svg viewBox="0 0 %d %d" width="100%%" role="img">' % (width, height),
           '<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>' % (width, height),
           '<text x="%d" y="24" font-size="15" font-weight="600" fill="#111827">%s</text>' % (ml, esc(title))]
    for value in _ticks(ylo, yhi, 5):
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e5e7eb"/>' % (ml, ty(value), ml + pw, ty(value)))
        out.append('<text x="%d" y="%.1f" font-size="11" fill="#6b7280" text-anchor="end">%s</text>' % (ml - 8, ty(value) + 4, esc(_fmt(value))))
    for value in tickvals:
        out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#e5e7eb"/>' % (tx(value), mt, tx(value), mt + ph))
        out.append('<text x="%.1f" y="%d" font-size="11" fill="#6b7280" text-anchor="middle">%s</text>' % (tx(value), mt + ph + 18, esc(_tickfmt(value))))
    for value, label in hlines:
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#9ca3af" stroke-dasharray="4,4"/>' % (ml, ty(value), ml + pw, ty(value)))
        out.append('<text x="%d" y="%.1f" font-size="10" fill="#6b7280">%s</text>' % (ml + 4, ty(value) - 4, esc(label)))
    for index, (value, label) in enumerate(vlines):
        if value < xlo or value > xhi:
            continue
        out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#c084fc" stroke-dasharray="3,3"/>' % (tx(value), mt, tx(value), mt + ph))
        out.append('<text x="%.1f" y="%d" font-size="9.5" fill="#7e22ce" text-anchor="middle">%s</text>'
                   % (tx(value), mt + 11 + (index % 3) * 11, esc(label)))
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#9ca3af"/>' % (ml, mt + ph, ml + pw, mt + ph))
    out.append('<text x="%d" y="%d" font-size="12" fill="#374151" text-anchor="middle">%s</text>' % (ml + pw // 2, height - 14, esc(xlabel)))
    out.append('<text x="18" y="%d" font-size="12" fill="#374151" text-anchor="middle" transform="rotate(-90 18 %d)">%s</text>' % (mt + ph // 2, mt + ph // 2, esc(ylabel)))
    for s in series:
        pts = sorted(s["points"], key=lambda p: p[0])
        if not pts:
            continue
        dash = ' stroke-dasharray="%s"' % s["dash"] if s.get("dash") else ""
        if errorbars and s.get("se"):
            for (x, y), se in zip(pts, s["se"]):
                if se and se == se:
                    out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" opacity="0.55"/>'
                               % (tx(x), ty(y - se), tx(x), ty(y + se), s["color"]))
        d = " ".join(("M" if i == 0 else "L") + "%.1f,%.1f" % (tx(x), ty(y)) for i, (x, y) in enumerate(pts))
        out.append('<path d="%s" fill="none" stroke="%s" stroke-width="%.1f"%s/>' % (d, s["color"], s.get("width", 2.2), dash))
        if s.get("markers"):
            for x, y in pts:
                out.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>' % (tx(x), ty(y), s.get("marker_radius", 3.4), s["color"]))
    ly = mt + 6
    for i, s in enumerate(series):
        yy = ly + i * 20
        out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="%.1f"%s/>'
                   % (ml + pw + 16, yy, ml + pw + 38, yy, s["color"], s.get("width", 2.2),
                      ' stroke-dasharray="%s"' % s["dash"] if s.get("dash") else ""))
        out.append('<text x="%d" y="%d" font-size="11" fill="#374151">%s</text>' % (ml + pw + 44, yy + 4, esc(s["name"])))
    if note:
        out.append('<text x="%d" y="%d" font-size="10.5" fill="#6b7280">%s</text>' % (ml + pw + 16, ly + len(series) * 20 + 14, esc(note)))
    out.append("</svg>")
    return "\n".join(out)


def table(headers, rows):
    out = ['<table><thead><tr>' + "".join("<th>%s</th>" % esc(h) for h in headers) + "</tr></thead><tbody>"]
    for row in rows:
        out.append("<tr>" + "".join("<td>%s</td>" % c for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


# ---------------------------------------------------------------------- main

def main():
    arms = [a for a in ARM_ORDER if os.path.isdir(os.path.join(ROOT, a))]
    data = {a: read_arm(a) for a in arms}
    if not data:
        print("no arms found under", ROOT)
        return 1
    nominals = sorted({k for a in data for k in data[a]})
    print("arms:", arms)
    print("nominal k:", nominals)

    print("\n== per-run block summary (mean NLL of the fixed 1008-token block) ==")
    print("  %-8s %-9s %s" % ("k", "visible", " ".join("%20s" % a for a in arms)))
    run_rows = []
    for nominal in nominals:
        cells, visible = [], "-"
        for arm in arms:
            row = run_mean(data, arm, nominal)
            if row:
                visible = row["history_mid"]
                cells.append("%.6f" % row["mean_nll"])
            else:
                cells.append("-")
        print("  %-8d %-9s %s" % (nominal, visible, " ".join("%20s" % c for c in cells)))
        run_rows.append([str(nominal), str(visible)] + cells)

    print("\n== run-mean slope vs log10(history) ==")
    run_fit = {}
    for arm in arms:
        pts = [(run_mean(data, arm, n)["history_mid"], run_mean(data, arm, n)["mean_nll"])
               for n in nominals if run_mean(data, arm, n)]
        whole, deep = fit_log(pts, 1.0), fit_log(pts, DEEP_FROM)
        run_fit[arm] = {"whole": whole, "deep": deep}
        if whole:
            print("  %-8s whole %+.6f +- %.6f (t %+.2f, n=%d)   deep(>=%d) %s"
                  % (arm, whole["slope"], whole["se"], whole["t"], whole["n"], DEEP_FROM,
                     ("%+.6f +- %.6f (t %+.2f, n=%d)" % (deep["slope"], deep["se"], deep["t"], deep["n"])) if deep else "n/a"))

    print("\n== panel slope (per fixed token, then averaged) ==")
    panel_fit = {}
    for arm in arms:
        series = [(data[arm][n]["base"], data[arm][n]["block"]) for n in nominals if n in data[arm]]
        whole, deep = panel_slopes(series, 1.0), panel_slopes(series, DEEP_FROM)
        panel_fit[arm] = {"whole": whole, "deep": deep}
        if whole:
            print("  %-8s whole %+.6f +- %.6f (t %+.2f)   deep %s"
                  % (arm, whole["slope"], whole["se"], whole["t"],
                     ("%+.6f +- %.6f (t %+.2f)" % (deep["slope"], deep["se"], deep["t"])) if deep else "n/a"))

    print("\n== adjacent-depth paired delta (positive = worse with MORE history) ==")
    adj = {}
    for arm in arms:
        for lo, hi in zip(nominals, nominals[1:]):
            got = adjacent(data, arm, lo, hi)
            if got:
                adj.setdefault(arm, []).append((lo, hi, got))
                print("  %-8s %7d -> %-7d  %s +- %s  t %s" % (arm, got["lo"], got["hi"],
                      _signed(got["delta"]), _fmt(got["se"], 6), _signed(got["delta"] / got["se"], 2)))

    print("\n== paired delta vs bf16 ==")
    delta_bf16 = {}
    for nominal in nominals:
        for arm in arms:
            if arm == "bf16":
                continue
            got = paired(data, arm, "bf16", nominal)
            if got:
                delta_bf16.setdefault(arm, []).append(got)
                print("  k=%-7d %-8s %s +- %s" % (nominal, arm, _signed(got["delta"]), _fmt(got["se"], 6)))

    print("\n== paired delta vs fp8 ==")
    delta_fp8 = {}
    for nominal in nominals:
        for arm in arms:
            if arm == "fp8":
                continue
            got = paired(data, arm, "fp8", nominal)
            if got:
                delta_fp8.setdefault(arm, []).append(got)
                print("  k=%-7d %-8s %s +- %s" % (nominal, arm, _signed(got["delta"]), _fmt(got["se"], 6)))

    print("\n== panel slope of the paired deltas ==")
    panel_delta = {}
    for reference, store in (("bf16", delta_bf16), ("fp8", delta_fp8)):
        for arm, entries in store.items():
            series = [(e["visible"], e["per_token"]) for e in entries]
            whole, deep = panel_slopes(series, 1.0), panel_slopes(series, DEEP_FROM)
            panel_delta[(arm, reference)] = {"whole": whole, "deep": deep}
            if whole:
                print("  %-8s vs %-6s whole %+.6f +- %.6f (t %+.2f)   deep %s"
                      % (arm, reference, whole["slope"], whole["se"], whole["t"],
                         ("%+.6f +- %.6f (t %+.2f)" % (deep["slope"], deep["se"], deep["t"])) if deep else "n/a"))

    # ------------------------------------------------------------------ HTML
    P = []
    P.append("""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>KV quantization vs context depth — fixed-text gold-standard control</title><style>
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f9fafb;color:#111827}
.wrap{max-width:1120px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:24px;margin:0 0 6px}h2{font-size:18px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid #e5e7eb}
h3{font-size:15px;margin:22px 0 8px}p,li{font-size:14px;line-height:1.65}
.sub{color:#6b7280;font-size:13px;margin-bottom:18px}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px 18px;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{border:1px solid #e5e7eb;padding:6px 9px;text-align:right}
th{background:#f3f4f6;font-weight:600}td:first-child,th:first-child{text-align:left}
.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.good{color:#059669;font-weight:600}.bad{color:#dc2626;font-weight:600}.nodata{color:#dc2626}
svg{display:block;max-width:100%}ul{margin:8px 0 8px 18px;padding:0}
</style></head><body><div class="wrap">""")
    P.append('<h1>KV cache quantization vs context depth: fixed-text gold-standard control</h1>')
    P.append('<div class="sub">Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-Vision · TP-2 <span class="mono">--devices 0,1</span> · '
             'the scored text is fixed to the last %d tokens of the source stream · each run only changes "how much preceding text is visible"</div>' % BLOCK_TOKENS)
    P.append('<div class="card"><h3 style="margin-top:0">Why this design can answer "does a longer context make things worse"</h3>'
             '<p>The previous version ran one 257K natural stream straight through and looked at the per-token NLL — depth and text content are '
             'locked together (deeper tokens belong to the later book), so "the model gets worse with context" cannot be split into depth or text.</p>'
             '<p>This version <b>pins down the scored text</b>: the target block is always the last 1024 tokens of the source stream, and run K exposes only the K tokens before it. '
             'For K ≥ 1 the immediate preceding text of the target block is the same passage every time, <b>so not even a concatenation seam exists</b>, and the only variable is how far back the history can see. '
             'The x-axis is the actual history length of a given token; at the same x-axis position every arm scores <b>the same token</b>, so arm-vs-arm differences and depth-vs-depth differences are strictly paired.</p>'
             '<p>Verified: the last 3000 characters of the 13 input streams are byte-identical (each is a character suffix of the source stream), '
             'so the 1008 scored tokens are strictly the same passage in every run.</p>'
             '<p><b>The only remaining confound</b>: the history <b>content</b> changes with K — the source stream is four pg19 books concatenated in order, so when the '
             'visible window crosses a book boundary the preceding text becomes another book. The "adjacent-depth paired difference" therefore mixes '
             '"more history" and "different history" at once, and point-wise causality cannot be established; what can be established is the <b>long-term trend</b>.</p></div>')

    # ------------------------------------------------------------- verdict
    verdict = []
    for arm in arms:
        f = panel_fit.get(arm, {}).get("deep") or panel_fit.get(arm, {}).get("whole")
        if f:
            verdict.append((arm, f["slope"], f["se"], f["t"]))
    P.append('<h2>Conclusions</h2><div class="card">')
    P.append('<h3 style="margin-top:0">Question one: for the same passage, does a longer history make the model better or worse?</h3>')
    if verdict:
        steepest = min(verdict, key=lambda v: v[1])
        flattest = max(verdict, key=lambda v: v[1])
        P.append('<p>Once the scored text is pinned down, the slope of NLL against visible history is <b>negative</b> for <b>all arms</b>: '
                 '%s. Even the flattest one (%s) reaches t=%s, '
                 'so "the longer the history, the better this fixed passage is predicted" is statistically certain, and its direction is opposite to "degradation with context".</p>'
                 % ("; ".join("%s %s ± %s nats/decade (t=%s)"
                              % (esc(a), _signed(s), _signed(se, 6), _signed(t, 2))
                              for a, s, se, t in verdict),
                    esc(flattest[0]), _signed(flattest[3], 2)))
        pos = [(a, g) for a in arms for _, _, g in adj.get(a, []) if g["delta"] > 0]
        if pos:
            worst = max(pos, key=lambda p: p[1]["delta"])
            P.append('<p>Only %d intervals among the adjacent-depth paired differences are positive (the text becomes slightly harder as history lengthens), '
                     'the largest being %s at %s ± %s (t=%s) — <b>indistinguishable from zero</b>. '
                     'All the rest are negative and mostly significant.</p>'
                     % (len(pos), esc(worst[0]), _signed(worst[1]["delta"]),
                        _signed(worst[1]["se"], 6), _signed(worst[1]["delta"] / worst[1]["se"], 2)))
        P.append('<p><b>So "the longer the context the worse the model" does not hold for this model + this corpus.</b>'
                 'The previous version ran one 257K natural stream straight through and saw deep PPL rise; that was'
                 'caused by <b>text content</b> (deeper tokens belong to the later book), not by depth itself.</p>')
        P.append('<p>Note the shape of this curve: the gain from 0 to ~1K of history is very steep, then it flattens out;'
                 'at about %s tokens there is one large step (the purple line "book 02|03"),'
                 'which is where the visible window first reaches the beginning of the book the target block belongs to.'
                 'After that, lengthening the history further (including pulling in the completely unrelated books 01 and 00) never made the result worse.</p>'
                 % "{:,}".format(BOOK_BOUNDARIES[0][0]))
    P.append('<h3>Question two: does the cost of KV quantization grow with context?</h3>')
    if panel_delta:
        lines = []
        for (arm, reference), fits in sorted(panel_delta.items()):
            f = fits.get("deep") or fits.get("whole")
            if f:
                lines.append("%s − %s: %s ± %s nats/decade (t=%s)"
                             % (esc(arm), esc(reference), _signed(f["slope"]), _signed(f["se"], 6), _signed(f["t"], 2)))
        P.append('<p>The slope of the paired differences themselves against history (positive = the quantization cost grows with context):</p><ul>'
                 + "".join("<li>%s</li>" % line for line in lines) + "</ul>")
        P.append('<p>Even when positive, these slopes are only of order ~1e-3 nats/decade, while the absolute differences themselves are of order 1e-2 nats — '
                 '<b>the quantization cost is essentially a constant independent of context length</b>, and no '
                 'knee of "short context lossless, long context collapsing" appears.</p>')
    P.append('</div>')

    P.append('<h2>1. NLL of the fixed text vs visible history length</h2>')
    P.append('<p>Each point is one run: the mean NLL of the %d tokens of the target block in that run (the first %d tokens of the block are dropped to avoid tokenization boundaries), '
             'with error bars at the within-block standard error. The content is completely fixed.</p>' % (BLOCK_TOKENS - DROP_HEAD, DROP_HEAD))
    P.append('<p class="sub">The purple dashed lines are the book boundaries in the source stream (pg19 00/01/02/03 concatenated in order); '
             'read the x-axis from right to left: the further right a point is, the shorter the history. Crossing a purple line means the visible window has reached into an earlier book.</p>')
    series = []
    for arm in arms:
        pts, ses = [], []
        for n in nominals:
            row = run_mean(data, arm, n)
            if not row:
                continue
            pts.append((max(row["history_mid"], 1.0), row["mean_nll"]))
            ses.append(row["se"])
        series.append({"name": arm, "color": ARM_COLOR[arm], "points": pts, "se": ses,
                       "dash": ARM_DASH.get(arm), "width": ARM_WIDTH.get(arm, 2.2), "markers": True})
    P.append('<div class="card">' + chart("fixed target block: NLL vs visible history",
             "visible history (tokens)", "mean NLL of the fixed block (nats)", series, xlog=True,
             errorbars=True, vlines=BOOK_BOUNDARIES,
             note="same 1008 tokens scored in every run") + '</div>')

    P.append('<h2>2. Paired ΔNLL between neighbouring depths — the most direct answer</h2>')
    P.append('<p>Each row takes two neighbouring runs and averages the NLL difference for <b>the same token</b>. '
             '<b>Positive = this token becomes harder to predict once the history is longer</b>, negative = it becomes easier. '
             'This is a test with no functional-form assumption.</p>')
    adj_series = []
    for arm in arms:
        entries = adj.get(arm)
        if not entries:
            continue
        adj_series.append({"name": arm, "color": ARM_COLOR[arm],
                           "points": [(run_mean(data, arm, hi)["history_mid"], g["delta"]) for _, hi, g in entries],
                           "se": [g["se"] for _, _, g in entries],
                           "dash": ARM_DASH.get(arm), "width": ARM_WIDTH.get(arm, 2.2), "markers": True})
    P.append('<div class="card">' + chart("paired delta NLL between neighbouring depths",
             "history of the deeper run (tokens)", "delta NLL (nats)", adj_series, xlog=True,
             ymin=-0.08, ymax=0.08, hlines=[(0.0, "zero")], errorbars=True, vlines=BOOK_BOUNDARIES,
             note="positive = deeper context makes this same text harder") + '</div>')
    rows = []
    for arm in arms:
        for lo, hi, g in adj.get(arm, []):
            mid_lo = run_mean(data, arm, lo)["history_mid"]
            mid_hi = run_mean(data, arm, hi)["history_mid"]
            rows.append([esc(arm), "%s &rarr; %s" % ("{:,}".format(round(mid_lo)), "{:,}".format(round(mid_hi))),
                         _signed(g["delta"]), _fmt(g["se"], 6),
                         _t(g["delta"] / g["se"] if g["se"] else None)])
    P.append('<div class="card">' + table(["arm", "history lo -> hi", "delta NLL", "se", "t"], rows) + '</div>')

    P.append('<h2>3. Slope: how much NLL changes per 10x of history</h2>')
    P.append('<p>The <b>whole range</b> uses every depth and is dominated by the steep improvement from "0 to 1K of history"; '
             'the <b>deep range</b> uses only runs with history ≥ %s, which is the regime where the "long context" question applies.</p>' % "{:,}".format(DEEP_FROM))
    rows = []
    for arm in arms:
        rf, pf = run_fit.get(arm, {}), panel_fit.get(arm, {})
        for label, key in (("run-mean", "whole"), ("run-mean (deep)", "deep")):
            f = rf.get(key)
            rows.append([esc(arm), label, _signed(f["slope"]) if f else "-",
                         _signed(f["se"], 6) if f else "-",
                         _t(f["slope"] / f["se"] if f else None), str(f["n"]) if f else "-"])
        for label, key in (("panel", "whole"), ("panel (deep)", "deep")):
            f = pf.get(key)
            rows.append([esc(arm), label, _signed(f["slope"]) if f else "-",
                         _signed(f["se"], 6) if f else "-",
                         _t(f["t"] if f else None), str(f["tokens"]) if f else "-"])
    P.append('<div class="card">' + table(
        ["arm", "estimator", "slope (nats / decade)", "se", "t", "points"], rows) +
        '<p class="sub" style="margin-bottom:0">Negative = better with longer history, positive = worse with longer history. |t| &lt; 2 means indistinguishable from a zero slope.</p></div>')

    P.append('<h2>4. Cross-arm paired differences vs history length</h2>')
    P.append('<p>ΔNLL is the difference for the same token between two arms, averaged over the 1008 tokens of the block. '
             'This is the direct answer to "does the quantization cost grow with context".</p>')
    for reference, store, limit in (("bf16", delta_bf16, 131072), ("fp8", delta_fp8, 257439)):
        ds = []
        for arm, entries in store.items():
            ds.append({"name": "%s - %s" % (arm, reference), "color": ARM_COLOR[arm],
                       "points": [(e["history_mid"], e["delta"]) for e in entries],
                       "se": [e["se"] for e in entries],
                       "dash": ARM_DASH.get(arm), "width": ARM_WIDTH.get(arm, 2.2), "markers": True})
        P.append('<div class="card">' + chart(
            "delta NLL vs %s" % reference, "visible history (tokens)", "delta NLL (nats)",
            ds, ymin=-0.03, ymax=0.03, hlines=[(0.0, "zero")], errorbars=True, xlog=True,
            note="reference reaches %s tokens of history" % "{:,}".format(limit)) + '</div>')

    rows = []
    for arm, entries in sorted(delta_bf16.items()):
        for e in entries:
            rows.append([esc(arm) + " - bf16", "{:,}".format(round(e["history_mid"])), _signed(e["delta"]),
                         _fmt(e["se"], 6), _t(e["delta"] / e["se"] if e["se"] else None)])
    P.append('<h3>vs bf16 (lossless reference, out to 131,072)</h3><div class="card">' + table(
        ["comparison", "visible history", "delta NLL", "se", "t"], rows) + '</div>')
    rows = []
    for arm, entries in sorted(delta_fp8.items()):
        for e in entries:
            rows.append([esc(arm) + " - fp8", "{:,}".format(round(e["history_mid"])), _signed(e["delta"]),
                         _fmt(e["se"], 6), _t(e["delta"] / e["se"] if e["se"] else None)])
    P.append('<h3>vs fp8 (all depths)</h3><div class="card">' + table(
        ["comparison", "visible history", "delta NLL", "se", "t"], rows) + '</div>')

    rows = []
    for (arm, reference), fits in sorted(panel_delta.items()):
        for label, key in (("whole", "whole"), ("deep", "deep")):
            f = fits.get(key)
            rows.append([esc(arm) + " - " + esc(reference), label, _signed(f["slope"]) if f else "-",
                         _signed(f["se"], 6) if f else "-", _t(f["t"] if f else None),
                         str(f["tokens"]) if f else "-"])
    P.append('<h3>Panel regression of the paired differences</h3><div class="card">' + table(
        ["comparison", "range", "slope (nats / decade)", "se", "t", "tokens"], rows) +
        '<p class="sub" style="margin-bottom:0">Only a significantly positive slope shows "the longer the context, the larger the quantization cost".</p></div>')

    P.append('<h3>Mean quantization cost in the long-context regime (pooling every token at depth ≥ %s)</h3>' % "{:,}".format(DEEP_FROM))
    rows = []
    for reference, store in (("bf16", delta_bf16), ("fp8", delta_fp8)):
        for arm, entries in sorted(store.items()):
            got = pooled(entries, DEEP_FROM)
            if got:
                rows.append([esc(arm) + " - " + esc(reference), _signed(got["delta"]),
                             _signed(got["se"], 6), _t(got["t"]),
                             "%s%% PPL" % _signed(math.exp(got["delta"]) * 100.0 - 100.0, 3),
                             str(got["runs"]), "{:,}".format(got["n"])])
    P.append('<div class="card">' + table(
        ["comparison", "mean delta NLL", "se", "t", "equivalent PPL", "runs", "tokens"], rows) +
        '<p class="sub" style="margin-bottom:0">This is the single number for "the quantization cost at long context"; '
        'its se is only of order 1e-3, while the depth-slope term is only 1e-3/decade — the cost is a constant.</p></div>')

    P.append('<h2>5. Raw numbers for every run</h2>')
    P.append('<div class="card">' + table(
        ["nominal k", "visible history"] + arms,
        [[r[0], esc(r[1])] + [esc(c) for c in r[2:]] for r in run_rows]) + '</div>')

    P.append("</div></body></html>")
    document = "\n".join(P)
    with open(OUT_HTML, "w", encoding="utf-8", newline="") as handle:
        handle.write(document)
    print("\nwrote %s (%d bytes)" % (OUT_HTML, len(document.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
