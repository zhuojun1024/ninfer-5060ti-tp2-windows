#!/usr/bin/env python3
"""Build the KV-quantization-vs-context-depth HTML report.

Inputs (all produced by ninfer-perplexity --dump-token-nll and the L0 oracle):
  <curve-dir>/<arm>/token_nll.*.tsv      target_index -> nll   (default _temp/ppl_curve)
  <curve-dir>/<arm>/report.json          run metadata
  _temp/*l0*sweep*.log                   L0 operator-level sweep

These are scratch sweep outputs; the repository keeps the generated HTML, not the raw
per-token dumps.  Re-run the sweep first (see profiles/kv-quantization/report.md).

Usage:  python profiles/kv-quantization/tools/kv_depth_report.py [curve_dir] [out.html]
Output: profiles/kv-quantization/kv-depth-curve.html
"""

import glob
import html
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
CURVE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "_temp", "ppl_curve")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "kv-depth-curve.html")
_l0_logs = sorted(glob.glob(os.path.join(REPO, "_temp", "*l0*sweep*.log")), key=os.path.getmtime)
L0_LOG = _l0_logs[-1] if _l0_logs else os.path.join(REPO, "_temp", "l0_length_sweep.log")

ARM_META = [
    ("bf16", "bf16 (lossless reference)", "#111827", "6,3"),
    ("int8", "int8-g64", "#2563eb", ""),
    ("fp8", "fp8-e4m3-r256", "#059669", ""),
    ("nvfp4", "nvfp4-g16", "#dc2626", ""),
    ("k8v4", "k8v4", "#d97706", ""),
]
COLOR = {a: c for a, _, c, _ in ARM_META}
LABEL = {a: l for a, l, _, _ in ARM_META}
DASH = {a: d for a, _, _, d in ARM_META}


# ---------------------------------------------------------------- data loading
def read_token_nll(arm):
    """Return {target_index: nll} for one arm."""
    paths = sorted(glob.glob(os.path.join(CURVE, arm, "token_nll.*.tsv")))
    if not paths:
        return None
    out = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                out[int(parts[0])] = float(parts[1])
    return out or None


def read_report(arm):
    path = os.path.join(CURVE, arm, "report.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def cumulative(nll):
    """Return (sum_by_index, mean_by_index) dicts for a {index: nll} mapping."""
    total = 0.0
    run = {}
    mean = {}
    for t in sorted(nll):
        total += nll[t]
        run[t] = total
        mean[t] = total / t
    return run, mean


def parse_l0(path):
    """Parse the operator-level sweep log into {codec: [(length, mae, rmse, rel, maxabs)]}."""
    sweep = {}
    single = {}
    if not os.path.exists(path):
        return single, sweep
    pattern = re.compile(
        r"^(?P<codec>[a-z0-9\-]+) quantized-vs-bf16 "
        r"(?P<kind>oracle quality|length sweep)"
        r"(?: length=(?P<len>\d+))? "
        r"mae=(?P<mae>[\d.eE+-]+) rmse=(?P<rmse>[\d.eE+-]+) "
        r"rel_rmse=(?P<rel>[\d.eE+-]+) max_abs=(?P<maxabs>[\d.eE+-]+)$"
    )
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(line.strip())
            if not match:
                continue
            codec = match.group("codec")
            row = (
                float(match.group("mae")),
                float(match.group("rmse")),
                float(match.group("rel")),
                float(match.group("maxabs")),
            )
            if match.group("kind") == "oracle quality":
                single[codec] = row
            else:
                sweep.setdefault(codec, []).append((int(match.group("len")),) + row)
    for codec in sweep:
        sweep[codec].sort()
    return single, sweep


# ------------------------------------------------------------------- svg chart
def _ticks(lo, hi, count=6):
    if not (hi > lo):
        return [lo]
    raw = (hi - lo) / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = magnitude * 10
    for candidate in (1, 2, 2.5, 5, 10):
        if (hi - lo) / (candidate * magnitude) <= count * 1.6:
            step = candidate * magnitude
            break
    ticks = []
    value = math.ceil(lo / step) * step
    while value <= hi + step * 1e-9:
        ticks.append(value)
        value += step
    return ticks


def _fmt(value, digits=4):
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 1e-3:
        return "{:.3g}".format(value)
    return ("{:.%df}" % digits).format(value)


def _tickfmt(value):
    """Axis labels: keep integers short ('10', '1,000') and small values precise."""
    if value == 0:
        return "0"
    if abs(value) >= 1 and float(value).is_integer():
        return "{:,.0f}".format(value)
    if abs(value) >= 1000 or abs(value) < 1e-3:
        return "{:.3g}".format(value)
    return "{:.4f}".format(value)


def chart(title, xlabel, ylabel, series, width=1020, height=440, xlog=False,
          ymin=None, ymax=None, hlines=(), legend_cols=1, note=None):
    """Render one line chart as inline SVG."""
    ml, mr, mt, mb = 84, 232, 46, 56
    pw, ph = width - ml - mr, height - mt - mb

    points = [p for s in series for p in s["points"]]
    if not points:
        return '<p class="empty">no data for: %s</p>' % html.escape(title)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    lo = min(ys) if ymin is None else ymin
    hi = max(ys) if ymax is None else ymax
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad

    def tx(v):
        if xlog:
            return ml + (math.log10(v) - math.log10(x0)) / (math.log10(x1) - math.log10(x0)) * pw
        return ml + (v - x0) / (x1 - x0) * pw

    def ty(v):
        return mt + ph - (v - lo) / (hi - lo) * ph

    parts = ['<svg viewBox="0 0 %d %d" class="chart" role="img">' % (width, height)]
    parts.append('<text x="%d" y="24" class="ctitle">%s</text>' % (ml, html.escape(title)))

    for tick in _ticks(lo, hi):
        y = ty(tick)
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>' % (ml, y, ml + pw, y))
        parts.append('<text x="%d" y="%.1f" class="tick" text-anchor="end">%s</text>'
                     % (ml - 8, y + 4, html.escape(_tickfmt(tick))))

    if xlog:
        decade = 10 ** math.floor(math.log10(x0))
        while decade <= x1:
            if decade >= x0:
                x = tx(decade)
                parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" class="grid"/>' % (x, mt, x, mt + ph))
                parts.append('<text x="%.1f" y="%d" class="tick" text-anchor="middle">%s</text>'
                             % (x, mt + ph + 20, html.escape(_tickfmt(decade))))
            decade *= 10
    else:
        for tick in _ticks(x0, x1):
            x = tx(tick)
            parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" class="grid"/>' % (x, mt, x, mt + ph))
            parts.append('<text x="%.1f" y="%d" class="tick" text-anchor="middle">%s</text>'
                         % (x, mt + ph + 20, html.escape(_tickfmt(tick))))

    for value, label, color, dash in hlines:
        if value < lo or value > hi:
            continue
        y = ty(value)
        style = ' stroke-dasharray="%s"' % dash if dash else ""
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-width="1.4"%s/>'
                     % (ml, y, ml + pw, y, color, style))
        parts.append('<text x="%d" y="%.1f" class="hlabel" fill="%s">%s</text>'
                     % (ml + pw + 8, y + 4, color, html.escape(label)))

    for s in series:
        pts = s["points"]
        if not pts:
            continue
        d = " ".join("%s%.1f %.1f" % ("M" if i == 0 else "L", tx(p[0]), ty(p[1]))
                     for i, p in enumerate(pts))
        style = ' stroke-dasharray="%s"' % s.get("dash") if s.get("dash") else ""
        parts.append('<path d="%s" fill="none" stroke="%s" stroke-width="%s"%s/>'
                     % (d, s["color"], s.get("width", 1.9), style))
        for p in pts:
            if p[0] in s.get("markers", ()):
                parts.append('<circle cx="%.1f" cy="%.1f" r="%s" fill="%s"/>'
                             % (tx(p[0]), ty(p[1]), s.get("marker_radius", 3.2), s["color"]))

    parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="axis"/>' % (ml, mt, ml, mt + ph))
    parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="axis"/>' % (ml, mt + ph, ml + pw, mt + ph))
    parts.append('<text x="%d" y="%d" class="axtitle" text-anchor="middle">%s</text>'
                 % (ml + pw // 2, height - 8, html.escape(xlabel)))
    parts.append('<text x="16" y="%d" class="axtitle" text-anchor="middle" transform="rotate(-90 16 %d)">%s</text>'
                 % (mt + ph // 2, mt + ph // 2, html.escape(ylabel)))

    ly = mt + 6
    for s in series:
        if not s["points"]:
            continue
        parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="2.4"%s/>'
                     % (ml + pw + 14, ly - 4, ml + pw + 44, ly - 4, s["color"],
                        ' stroke-dasharray="%s"' % s["dash"] if s.get("dash") else ""))
        parts.append('<text x="%d" y="%d" class="legend">%s</text>'
                     % (ml + pw + 50, ly, html.escape(s["name"])))
        ly += 20
    if note:
        parts.append('<text x="%d" y="%d" class="note">%s</text>'
                     % (ml + pw + 14, ly + 8, html.escape(note)))
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------ statistics
def buckets(max_index, count=46):
    """Geometric bucket edges over [1, max_index]."""
    edges = []
    ratio = (max_index / 1.0) ** (1.0 / count)
    value = 1.0
    while value < max_index:
        edges.append(int(round(value)))
        value *= ratio
    edges.append(max_index)
    clean = []
    for edge in edges:
        if not clean or edge > clean[-1]:
            clean.append(edge)
    return clean


def trend_fit(stats):
    """Weighted least squares of bucket mean against log10(depth).

    Returns (slope per decade, slope standard error, number of buckets) or None.
    """
    rows = [s for s in stats if s[2] >= 64]
    if len(rows) < 8:
        return None
    xs = [math.log10(0.5 * (lo + hi)) for lo, hi, _, _, _ in rows]
    ys = [mean for _, _, _, mean, _ in rows]
    ws = [float(n) for _, _, n, _, _ in rows]
    sw = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    if sxx <= 0:
        return None
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    residual = sum(w * (y - (intercept + slope * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    dof = max(1, len(rows) - 2)
    sigma2 = residual / dof
    se = math.sqrt(sigma2 / sxx) if sigma2 > 0 else 0.0
    return slope, se, len(rows)


def bucket_stats(nll_a, nll_ref, edges):
    """Mean and standard error of (nll_a - nll_ref) inside each bucket."""
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        deltas = [nll_a[t] - nll_ref[t] for t in range(lo, hi) if t in nll_a and t in nll_ref]
        if not deltas:
            continue
        n = len(deltas)
        mean = sum(deltas) / n
        if n > 1:
            var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
            se = math.sqrt(var / n)
        else:
            se = 0.0
        rows.append((lo, hi, n, mean, se))
    return rows


def main():
    nll = {}
    runs = {}
    for arm, _, _, _ in ARM_META:
        data = read_token_nll(arm)
        if data:
            nll[arm] = data
            runs[arm] = read_report(arm)
    if not nll:
        print("no token NLL data found under", CURVE, file=sys.stderr)
        return 1

    # An arm only reports *full-history* depths for its first window. bf16 cannot hold a
    # 262144-token KV pool, so it runs at a smaller context and its later windows score
    # targets with truncated history -- those must not enter a depth comparison.
    stream_len = max(max(data) for data in nll.values()) + 1
    full_max = {}
    for arm, data in nll.items():
        report = runs.get(arm) or {}
        context = report.get("execution", {}).get("context_tokens")
        full_max[arm] = (min(int(context), stream_len) - 1) if context else max(data)
        nll[arm] = {t: v for t, v in data.items() if t <= full_max[arm]}

    cumulative_sum = {}
    cumulative_mean = {}
    for arm, data in nll.items():
        cumulative_sum[arm], cumulative_mean[arm] = cumulative(data)

    max_index = {arm: max(data) for arm, data in nll.items()}
    ref = "bf16" if "bf16" in nll else "fp8"
    ref_max = max_index[ref]

    # ------------------------------------------------------------- L0 evidence
    l0_single, l0_sweep = parse_l0(L0_LOG)
    codec_color = {
        "int8-g64": "#2563eb",
        "fp8-e4m3fn-row256": "#059669",
        "nvfp4-g16": "#dc2626",
        "k8v4": "#d97706",
    }
    # nvfp4 and k8v4 land within 3.2e-5 of each other (same V codec), so give them
    # different stroke styles or one curve completely hides the other.
    codec_style = {
        "int8-g64": {"width": 2.2},
        "fp8-e4m3fn-row256": {"width": 2.2, "dash": "10,5"},
        "nvfp4-g16": {"width": 4.2},
        "k8v4": {"width": 2.6, "dash": "1,6"},
    }
    l0_rel_series = []
    l0_abs_series = []
    for codec, rows in l0_sweep.items():
        color = codec_color.get(codec, "#6b7280")
        style = dict(codec_style.get(codec, {}))
        style["name"] = codec
        style["color"] = color
        style["markers"] = tuple(r[0] for r in rows)
        style["marker_radius"] = 5.0 if codec == "nvfp4-g16" else 2.6
        l0_rel_series.append(dict(style, points=[(r[0], r[3]) for r in rows]))
        l0_abs_series.append(dict(style, points=[(r[0], r[2]) for r in rows]))

    # --------------------------------------------------------- E2E PPL curves
    ppl_series = []
    for arm, _, _, _ in ARM_META:
        if arm not in nll:
            continue
        last = max_index[arm]
        pts = []
        step = max(1, last // 900)
        for t in range(1, last + 1, step):
            pts.append((t, math.exp(cumulative_mean[arm][t])))
        pts.append((last, math.exp(cumulative_mean[arm][last])))
        ppl_series.append({"name": LABEL[arm], "color": COLOR[arm],
                           "points": pts, "dash": DASH[arm] or None})

    # ------------------------------------- cumulative Delta mean NLL vs depth
    delta_series = []
    for arm in ("int8", "fp8", "nvfp4", "k8v4"):
        if arm not in nll or arm == ref:
            continue
        pts = []
        for t in range(1, ref_max + 1, max(1, ref_max // 900)):
            if t in cumulative_sum[arm]:
                pts.append((t, (cumulative_sum[arm][t] - cumulative_sum[ref][t]) / t))
        if pts:
            pts.append((ref_max, (cumulative_sum[arm][ref_max] - cumulative_sum[ref][ref_max]) / ref_max))
        delta_series.append({"name": "%s - %s" % (LABEL[arm], LABEL[ref]),
                             "color": COLOR[arm], "points": pts,
                             "dash": DASH[arm] or None})

    # -------------------------------------- cumulative *relative* PPL gap (%)
    # The headline chart: a flat line means "quantization costs a constant amount";
    # a rising line means "quantization gets worse as the context grows".
    def rel_curve(arm, reference, last):
        pts = []
        step = max(1, last // 900)
        for t in range(1, last + 1, step):
            if t in cumulative_mean[arm] and t in cumulative_mean[reference]:
                pts.append((t, (math.exp(cumulative_mean[arm][t]
                                         - cumulative_mean[reference][t]) - 1.0) * 100.0))
        if last in cumulative_mean[arm] and last in cumulative_mean[reference]:
            pts.append((last, (math.exp(cumulative_mean[arm][last]
                                        - cumulative_mean[reference][last]) - 1.0) * 100.0))
        return pts

    rel_series = []
    for arm in ("int8", "fp8", "nvfp4", "k8v4"):
        if arm not in nll or arm == ref:
            continue
        rel_series.append({"name": "%s - %s" % (LABEL[arm], LABEL[ref]), "color": COLOR[arm],
                           "points": rel_curve(arm, ref, ref_max), "dash": DASH[arm] or None})

    # ------------------------------------------------ bucketed instantaneous
    edges = buckets(ref_max)
    bucket_series = []
    bucket_rows = []
    for arm in ("int8", "fp8", "nvfp4", "k8v4"):
        if arm not in nll or arm == ref:
            continue
        stats = bucket_stats(nll[arm], nll[ref], edges)
        bucket_rows.append((arm, stats))
        bucket_series.append({
            "name": "%s - %s" % (LABEL[arm], LABEL[ref]), "color": COLOR[arm],
            "points": [(0.5 * (lo + hi), mean) for lo, hi, _, mean, _ in stats],
            "dash": DASH[arm] or None,
        })

    # -------------------------------- cross-arm bucketed delta over the full range
    wide_ref = "fp8" if "fp8" in nll else ref
    wide_max = min(max_index.get(a, 0) for a in ("int8", "fp8", "nvfp4", "k8v4") if a in nll)
    wide_edges = buckets(wide_max)
    wide_series = []
    wide_rows = []
    for arm in ("int8", "nvfp4", "k8v4"):
        if arm not in nll:
            continue
        stats = bucket_stats(nll[arm], nll[wide_ref], wide_edges)
        wide_rows.append((arm, stats))
        wide_series.append({
            "name": "%s - %s" % (LABEL[arm], LABEL[wide_ref]), "color": COLOR[arm],
            "points": [(0.5 * (lo + hi), mean) for lo, hi, _, mean, _ in stats],
            "dash": DASH[arm] or None,
        })

    rel_wide_series = []
    for arm in ("int8", "nvfp4", "k8v4"):
        if arm not in nll:
            continue
        rel_wide_series.append({"name": "%s - %s" % (LABEL[arm], LABEL[wide_ref]),
                                "color": COLOR[arm],
                                "points": rel_curve(arm, wide_ref, wide_max),
                                "dash": DASH[arm] or None})

    # ------------------------------------------------------------------ tables
    def arm_table():
        out = ['<table><thead><tr><th>arm</th><th>depth covered</th>'
               '<th>mean NLL (full)</th><th>PPL (full)</th>'
               '<th>&Delta;NLL vs ref</th><th>&Delta;PPL %</th>'
               '<th>KV MiB/shard @262144</th></tr></thead><tbody>']
        for arm, _, _, _ in ARM_META:
            if arm not in nll:
                out.append('<tr class="missing"><td>%s</td><td colspan="6">no data</td></tr>'
                           % html.escape(LABEL[arm]))
                continue
            last = max_index[arm]
            mean_nll = cumulative_mean[arm][last]
            ppl = math.exp(mean_nll)
            if arm in cumulative_sum and ref in cumulative_sum:
                span = min(last, ref_max)
                d_nll = (cumulative_sum[arm][span] - cumulative_sum[ref][span]) / span
                d_ppl = (math.exp(cumulative_mean[arm][span])
                         / math.exp(cumulative_mean[ref][span]) - 1.0) * 100.0
                d_nll_s, d_ppl_s = _fmt(d_nll, 6), "{:+.3f}%".format(d_ppl)
            else:
                d_nll_s = d_ppl_s = "&mdash;"
            kv = ""
            report = runs.get(arm)
            if report:
                exec_ = report.get("execution", {})
                kv = "%s / ctx %s" % (exec_.get("kv_dtype", "?"), exec_.get("context_tokens", "?"))
            out.append('<tr><td class="arm"><span class="swatch" style="background:%s"></span>%s</td>'
                       '<td>1 &ndash; %s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
                       % (COLOR[arm], html.escape(LABEL[arm]), f"{last:,}",
                          _fmt(mean_nll, 6), _fmt(ppl, 6), d_nll_s, d_ppl_s, html.escape(kv)))
        out.append("</tbody></table>")
        return "".join(out)

    def bucket_table(rows, label):
        out = ['<table class="tight"><thead><tr><th>depth range</th><th>tokens</th>'
               '<th>mean &Delta;NLL</th><th>std. error</th><th>mean &Delta;PPL %</th>'
               '<th>t</th></tr></thead><tbody>']
        for arm, stats in rows:
            out.append('<tr class="group"><td colspan="6">%s vs %s</td></tr>'
                       % (html.escape(LABEL[arm]), html.escape(label)))
            for lo, hi, n, mean, se in stats:
                if n < 2:
                    continue
                t_stat = mean / se if se > 0 else 0.0
                out.append('<tr><td>%s &ndash; %s</td><td>%s</td><td>%s</td><td>%s</td>'
                           '<td>%s</td><td>%s</td></tr>'
                           % (f"{lo:,}", f"{hi:,}", f"{n:,}", _fmt(mean, 6), _fmt(se, 6),
                              "{:+.4f}%".format((math.exp(mean) - 1.0) * 100.0),
                              "{:+.2f}".format(t_stat)))
        out.append("</tbody></table>")
        return "".join(out)

    def checkpoint_table(depths):
        """Delta NLL / Delta PPL % at a few round depths, vs the best reference available."""
        out = ['<table><thead><tr><th>depth</th>']
        present = [a for a, _, _, _ in ARM_META if a in nll]
        for arm in present:
            out.append('<th>%s</th>' % html.escape(LABEL[arm]))
        out.append("</tr></thead><tbody>")
        for p in depths:
            usable = [a for a in present if p <= max_index[a]]
            if len(usable) < 2:
                continue
            reference = ref if p <= ref_max else wide_ref
            if reference not in usable:
                continue
            out.append('<tr><td><b>%s</b></td>' % f"{p:,}")
            for arm in present:
                if p > max_index[arm]:
                    out.append("<td>&mdash;</td>")
                    continue
                if arm == reference:
                    out.append('<td class="ref">PPL %s</td>' % _fmt(math.exp(cumulative_mean[arm][p]), 5))
                    continue
                if p not in cumulative_sum[reference]:
                    out.append("<td>&mdash;</td>")
                    continue
                d_nll = (cumulative_sum[arm][p] - cumulative_sum[reference][p]) / p
                d_ppl = (math.exp(cumulative_mean[arm][p] - cumulative_mean[reference][p]) - 1.0) * 100.0
                cls = ' class="bad"' if abs(d_ppl) >= 0.1 else ""
                out.append('<td%s>%s<br><span class="small">%s</span></td>'
                           % (cls, "{:+.3f}%".format(d_ppl), _fmt(d_nll, 6)))
            out.append("</tr>")
        out.append("</tbody></table>")
        return "".join(out)

    # ------------------------------------------------------------------ verdict
    def growth(arm, stats):
        """Compare the first and last third of the buckets."""
        usable = [s for s in stats if s[2] >= 512]
        if len(usable) < 6:
            return None
        third = len(usable) // 3
        early = usable[:third]
        late = usable[-third:]
        e = sum(s[3] for s in early) / len(early)
        l = sum(s[3] for s in late) / len(late)
        e_ppl = (math.exp(e) - 1.0) * 100.0
        l_ppl = (math.exp(l) - 1.0) * 100.0
        return e, l, e_ppl, l_ppl

    verdict_lines = []
    trend_lines = []
    for arm, stats in bucket_rows:
        fit = trend_fit(stats)
        if not fit:
            continue
        slope, se, k = fit
        t_stat = slope / se if se > 0 else 0.0
        ppl_per_decade = (math.exp(slope) - 1.0) * 100.0
        trend_lines.append(
            "<li><b>%s</b> vs %s over 1&ndash;%s: &Delta;NLL trend <b>%s per decade of depth</b> "
            "(&plusmn;%s, t = %s) &mdash; about %+.4f%% PPL per decade over %d buckets.</li>"
            % (html.escape(LABEL[arm]), html.escape(LABEL[ref]), f"{ref_max:,}",
               _fmt(slope, 6), _fmt(se, 6), "{:+.2f}".format(t_stat), ppl_per_decade, k))
    for arm, stats in wide_rows:
        fit = trend_fit(stats)
        if not fit:
            continue
        slope, se, k = fit
        t_stat = slope / se if se > 0 else 0.0
        ppl_per_decade = (math.exp(slope) - 1.0) * 100.0
        trend_lines.append(
            "<li><b>%s</b> vs fp8 over 1&ndash;%s: &Delta;NLL trend <b>%s per decade</b> "
            "(&plusmn;%s, t = %s) &mdash; about %+.4f%% PPL per decade over %d buckets.</li>"
            % (html.escape(LABEL[arm]), f"{wide_max:,}",
               _fmt(slope, 6), _fmt(se, 6), "{:+.2f}".format(t_stat), ppl_per_decade, k))
    print("--- trend of Delta NLL vs log10(depth) ---")
    for line in trend_lines:
        print("  " + re.sub(r"<[^>]+>", "", line).replace("&Delta;", "D").replace("&plusmn;", "+/-")
              .replace("&ndash;", "-").replace("&mdash;", "--"))

    for arm, stats in bucket_rows:
        g = growth(arm, stats)
        if g:
            verdict_lines.append(
                "<li><b>%s</b>: low-depth buckets average %s &Delta;NLL (%+.4f%% PPL), "
                "high-depth buckets %s (%+.4f%% PPL).</li>"
                % (html.escape(LABEL[arm]), _fmt(g[0], 6), g[2], _fmt(g[1], 6), g[3]))
    for arm, stats in wide_rows:
        g = growth(arm, stats)
        if g:
            verdict_lines.append(
                "<li><b>%s</b> vs fp8 over the full 1&ndash;%s range: low-depth %s, "
                "high-depth %s &Delta;NLL (%+.4f%% &rarr; %+.4f%% PPL).</li>"
                % (html.escape(LABEL[arm]), f"{wide_max:,}", _fmt(g[0], 6), _fmt(g[1], 6),
                   g[2], g[3]))

    l0_note = ""
    if l0_sweep:
        sample = l0_sweep.get("nvfp4-g16") or next(iter(l0_sweep.values()))
        l0_note = ("operator-level relative error is flat: %s spans %s at length %s and %s at "
                   "length %s." % (html.escape("nvfp4-g16"), _fmt(sample[0][3], 4),
                                   f"{sample[0][0]:,}", _fmt(sample[-1][3], 4), f"{sample[-1][0]:,}"))

    doc = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>KV cache quantization error vs context depth</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin:0; padding:32px 40px 80px; background:#f7f8fa; color:#16181d;
        font:15px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",Roboto,sans-serif; }}
h1 {{ font-size:26px; margin:0 0 6px; }}
h2 {{ font-size:19px; margin:38px 0 10px; padding-bottom:6px; border-bottom:2px solid #e3e6ec; }}
h3 {{ font-size:16px; margin:24px 0 8px; }}
.sub {{ color:#5b6270; margin:0 0 22px; }}
.card {{ background:#fff; border:1px solid #e3e6ec; border-radius:10px; padding:18px 20px;
         margin:16px 0; box-shadow:0 1px 2px rgba(16,24,40,.04); }}
.chart {{ width:100%; height:auto; display:block; }}
.ctitle {{ font-size:15px; font-weight:600; fill:#16181d; }}
.tick {{ font-size:11.5px; fill:#6b7280; }}
.axtitle {{ font-size:12.5px; fill:#3f4653; }}
.legend {{ font-size:12.5px; fill:#2b3140; }}
.note {{ font-size:11.5px; fill:#6b7280; }}
.hlabel {{ font-size:11.5px; }}
.grid {{ stroke:#eceef3; stroke-width:1; }}
.axis {{ stroke:#c3c8d2; stroke-width:1.2; }}
table {{ border-collapse:collapse; width:100%; margin:10px 0 4px; font-size:13.5px; }}
th,td {{ border-bottom:1px solid #e9ebf0; padding:7px 10px; text-align:right; }}
th {{ background:#f2f4f8; font-weight:600; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
tbody tr:hover {{ background:#fafbfd; }}
tr.group td {{ background:#eef1f6; font-weight:600; text-align:left; }}
tr.missing td {{ color:#9aa1ae; font-style:italic; }}
.swatch {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:7px; }}
.kv {{ font-family:ui-monospace,Consolas,monospace; font-size:12.5px; background:#f2f4f8;
       padding:1px 5px; border-radius:4px; }}
.verdict {{ background:#fff; border-left:4px solid #2563eb; border-radius:8px;
            padding:16px 20px; margin:16px 0; }}
.verdict ul {{ margin:8px 0 0; padding-left:20px; }}
.verdict li {{ margin:5px 0; }}
.warn {{ border-left-color:#dc2626; }}
.empty {{ color:#9aa1ae; font-style:italic; }}
td.ref {{ color:#6b7280; }}
td.bad {{ background:#fef2f2; font-weight:600; }}
.small {{ color:#8b93a1; font-size:11.5px; }}
code {{ background:#f2f4f8; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
</style></head><body>

<h1>Does the KV cache quantization error grow with context depth</h1>
<p class="sub">model <code>{model}</code> · TP-2 (<code>--devices 0,1</code>) · a single {tokens}-token stream
· per-token NLL · reference arm <code>{ref}</code> (depths 1&ndash;{refmax})</p>

<div class="card">
<h3 style="margin-top:0">Conclusions</h3>
<p style="margin:6px 0 2px"><b>Slope of the error against depth</b> (&Delta;NLL change per decade of depth, from a
least-squares fit weighted by the token count in each bucket; a t value near 0 means "does not grow with depth"):</p>
<ul>{trend}</ul>
<p style="margin:14px 0 2px">Low-depth buckets vs high-depth buckets:</p>
<ul>{verdict}</ul>
<p style="margin:12px 0 0">{l0note}</p>
</div>

<h2>1. Operator level (L0 oracle): the quantization error is independent of KV length</h2>
<p>The same K/V is stored into the paged KV in the target format and in bf16, and the outputs are compared with an FP64 ideal attention.
<code>rel_rmse</code> is the RMS relative to the reference output; <code>rmse</code> is the absolute error.</p>
<div class="card">{l0rel}</div>
<div class="card">{l0abs}</div>
<p><b>Reading it:</b> the relative error (top chart) is almost a flat line from length 6 &rarr; 16,384, meaning "the more tokens are stored in the KV,
the relative distortion of a single attention does not grow". The absolute error (bottom chart) falls as 1/&radic;N with length — because the magnitude of the softmax output
itself shrinks as 1/&radic;N too (more terms take part in the average).</p>
<p><b>Why do the nvfp4 and k8v4 curves coincide?</b> Not an omitted measurement — on this fixture the two differ by at most
<b>3.2e-5</b> (0.03% relative), so on a chart with a 0.09 range they overlap completely. The reason is the storage format itself:
<code>k8v4</code> = K in FP8-row256 + V in <b>NVFP4-G16</b>, <code>nvfp4</code> = K in NVFP4 + V in
<b>NVFP4-G16</b>, i.e. the two have <b>identical V codecs</b> and differ only in K. And this fixture's K/V are uniform random numbers
&rArr; the attention logits are small and softmax is nearly uniform &rArr; the output is approximately the mean of V &rArr; the error is dominated by V and K's codec
barely matters. Real long-context logits are sharper and K matters more, so this chart is the lower bound of the "most favourable regime".</p>

<h2>2. End to end: PPL vs depth</h2>
<p>The same stream, the same token set, each arm run from an empty KV with <code>--context 262144</code> (bf16 can only reach
<code>--context 131072</code> because of VRAM). The curve is cumulative:
<code>PPL(p) = exp( (1/p) &sum;<sub>q&le;p</sub> NLL(q) )</code>.</p>
<div class="card">{ppl}</div>
<div class="card">{table}</div>
<p>Readings at a few round depths (the upper line in each cell is &Delta;PPL% relative to the reference arm, the lower line is &Delta;NLL; the reference arm itself gives the absolute PPL).
For depth &le; {refmax} the reference arm is bf16; deeper than that it switches to fp8.</p>
<div class="card">{checkpoint}</div>

<h2>3. Relative PPL gap vs depth &mdash; the key chart</h2>
<p>Divide the cumulative PPL of each arm by the cumulative PPL of the reference arm. <b>If KV quantization were to "amplify" at long context, this line should keep rising with depth;
if quantization only brings a fixed cost, this line is horizontal.</b> The left chart uses bf16 as reference (1&ndash;{refmax}),
the right chart uses fp8 as reference (1&ndash;{widemax}, because bf16 cannot reach 262K).</p>
<div class="card">{rel}</div>
<div class="card">{relwide}</div>

<h2>4. Cumulative &Delta;NLL vs depth</h2>
<p>Subtract the reference arm token by token, then take the cumulative mean (an absolute quantity, not a percentage). <b>If the quantization error grew with context, these curves should slope up to the right.</b></p>
<div class="card">{delta}</div>

<h2>5. Bucketed &Delta;NLL (logarithmic depth buckets)</h2>
<p>The mean of <code>NLL(arm) - NLL(ref)</code> inside each geometric bucket. The error bars are standard errors; the wider the bucket the more samples,
the smaller the standard error. This chart reveals "whether the penalty grows monotonically with depth" instead of having it hidden by averaging.</p>
<div class="card">{bucket}</div>
<div class="card">{bucket_table}</div>

<h2>6. Full depth range (1&ndash;{widemax}): quantized arms against each other</h2>
<p>bf16 cannot reach 262K, so over the full depth range fp8 is used as the reference and the other quantized formats are compared by their offset from it.</p>
<div class="card">{wide}</div>
<div class="card">{wide_table}</div>

<h2>7. How to reproduce</h2>
<pre class="kv">ninfer-perplexity &lt;model.ninfer&gt; --text _temp/20261008-0920_stream.txt \\
  --context 262144 --stride 131072 --devices 0,1 \\
  --kv-dtype &lt;bf16|int8|fp8|nvfp4|k8v4&gt; --dump-token-nll --output _temp/ppl_curve/&lt;arm&gt;

ninfer_softmax_attention_test --quant-sweep        # L0 length sweep</pre>
<p>The per-token NLL lands in <code>token_nll.&lt;stream&gt;.tsv</code> (<code>target_index&nbsp;nll</code>);
<code>target_index = t</code> means that token is predicted from <code>tokens[0,t)</code>, so the history depth is exactly <code>t</code>.</p>
<p><b>Context-independence check (is the bf16 reference arm usable):</b> fp8 run once at <code>--context 131072</code> (window 0 is the full history)
and once at <code>--context 262144</code> on the same stream gives depths <b>1&ndash;131,008 bit-identical</b>; only the last 63 depths
(131,009&ndash;131,071, i.e. the final incomplete KV page of the 131,072 window) differ, with a maximum single-token difference of 0.456,
an effect of <b>4.7e-6</b> on the mean NLL over 1&ndash;131,071 and 2.1e-5 on the mean of the last depth bucket,
far below the 1e-3&ndash;3e-3 signal this report needs to resolve. So the depths 1&ndash;131,071 measured by bf16@131072 can be used directly
as the reference arm for the 262144 run.</p>

</body></html>
""".format(
        model=html.escape(runs.get("fp8", {}).get("artifact", {}).get("name", "model")),
        tokens=f"{max(max_index.values()):,}",
        ref=html.escape(LABEL[ref]),
        refmax=f"{ref_max:,}",
        widemax=f"{wide_max:,}",
        verdict="".join(verdict_lines) or "<li>no bucket data</li>",
        trend="".join(trend_lines) or "<li>no bucket data</li>",
        l0note=l0_note,
        l0rel=chart("L0 oracle: relative error vs KV length", "KV length (tokens)",
                    "rel_rmse (rmse / reference rms)", l0_rel_series, xlog=True,
                    note="nvfp4 solid, k8v4 dotted: max 3.2e-5"),
        l0abs=chart("L0 oracle: absolute error vs KV length", "KV length (tokens)",
                    "rmse", l0_abs_series, xlog=True, ymin=0.0,
                    note="nvfp4 solid, k8v4 dotted: max 4.4e-5"),
        ppl=chart("Perplexity vs context depth", "history depth p (tokens)",
                  "PPL(p) = exp(cumulative mean NLL)", ppl_series, xlog=True),
        table=arm_table(),
        checkpoint=checkpoint_table([1024, 4096, 16384, 65536, 131072, 196608, 262144]),
        rel=chart("Cumulative PPL gap vs depth (reference: %s)" % LABEL[ref],
                  "history depth p (tokens)", "PPL_arm(p) / PPL_ref(p) - 1  (%)",
                  rel_series + [{"name": "zero", "color": "#9aa1ae",
                                 "points": [(1, 0.0), (ref_max, 0.0)], "dash": "5,4"}],
                  xlog=True),
        relwide=chart("Cumulative PPL gap vs depth (reference: fp8, full range)",
                      "history depth p (tokens)", "PPL_arm(p) / PPL_fp8(p) - 1  (%)",
                      rel_wide_series + [{"name": "zero", "color": "#9aa1ae",
                                          "points": [(1, 0.0), (wide_max, 0.0)], "dash": "5,4"}],
                      xlog=True),
        delta=chart("Cumulative mean NLL difference vs depth", "history depth p (tokens)",
                    "mean(NLL_arm - NLL_ref) over [1,p]",
                    delta_series + [{"name": "zero", "color": "#9aa1ae",
                                     "points": [(1, 0.0), (ref_max, 0.0)], "dash": "5,4"}]),
        bucket=chart("Bucketed mean NLL difference vs depth", "history depth p (tokens)",
                     "mean(NLL_arm - NLL_ref) in bucket",
                     bucket_series + [{"name": "zero", "color": "#9aa1ae",
                                       "points": [(1, 0.0), (ref_max, 0.0)], "dash": "5,4"}],
                     xlog=True),
        bucket_table=bucket_table(bucket_rows, LABEL[ref]),
        wide=chart("Quantized arms vs fp8 over the full depth range", "history depth p (tokens)",
                   "mean(NLL_arm - NLL_fp8) in bucket",
                   wide_series + [{"name": "zero", "color": "#9aa1ae",
                                   "points": [(1, 0.0), (wide_max, 0.0)], "dash": "5,4"}],
                   xlog=True),
        wide_table=bucket_table(wide_rows, "fp8-e4m3-r256"),
    )

    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write(doc)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
