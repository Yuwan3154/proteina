"""Self-check for figures/training_status_report.html as written by build_status_report.py.

Recomputes every expected value from the JSON snapshots on its own and reads each chart's scale from
the chart's own tick labels, so it does not trust the generator's arithmetic. Exits non-zero on the
first failed assertion; prints the sample size behind every PASS.
"""

import json
import math
import re

PAGE = "/Users/Chenxi/SOLab/proteina/.claude/worktrees/distogram-head/figures/training_status_report.html"
ORIG = "/Users/Chenxi/.claude/jobs/2c2943b0/tmp/report/report_7fc1e47.html"
DATA = "/Users/Chenxi/.claude/jobs/2c2943b0/tmp/report"
SEC_OPEN = '<section class="sec" id="convergence-2026-09-21">'
PX_TOL = 0.5
X0, X1, Y0, Y1 = 52.0, 548.0, 14.0, 176.0

SRC = {f: json.load(open(f"{DATA}/{f}")) for f in ("tri_epochs.json", "c2c_steps.json", "tri_val_pal.json")}
XKEY = {"tri_epochs.json": "epoch", "c2c_steps.json": "step"}


def attrs(tag):
    return dict(re.findall(r'([\w-]+)="([^"]*)"', tag))


def median9(pts, w):
    out = []
    for e, _v in pts:
        vals = sorted(v for x, v in pts if e - w // 2 <= x <= e + w // 2)
        n = len(vals)
        out.append((e, vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2))
    return out


def source(src, key, smooth=None):
    if src == "tri_val_pal.json":
        d = SRC[src]
        rows = d["onestep_by_epoch"].get(key) or d["sampling"][key]
        pts = [(r[0], r[1]) for r in rows]
    else:
        pts = [(r[XKEY[src]], r[key]) for r in SRC[src] if XKEY[src] in r and key in r]
    return median9(pts, int(smooth)) if smooth else pts


def on_ladder(step):
    m = step / 10 ** math.floor(math.log10(step) + 1e-12)
    return any(abs(m - c) < 1e-6 for c in (1, 2, 2.5, 5))


def section(t):
    a = t.index(SEC_OPEN)
    return a, t.index("</section>", a) + len("</section>")


page, orig = open(PAGE).read(), open(ORIG).read()
# The two DOCUMENTED edits outside the section (header date; the archived-numbers correction note that
# follows the section) are normalised away; every other outside byte must still be identical.
_date = re.compile(r"convergence status updated \d{1,2} \w{3} \d{4}")
_cav = re.compile(r'\n\n<section class="sec" id="archive-caveat-2026-09-22">.*?</section>', re.S)
assert len(_cav.findall(page)) == 1, "correction note missing or duplicated"
page = _cav.sub("", _date.sub("convergence status updated <DATE>", page, count=1), count=1)
orig = _date.sub("convergence status updated <DATE>", orig, count=1)
pa, pb = section(page)
oa, ob = section(orig)
sec = page[pa:pb]
charts = re.findall(r'<div class="chart-t">(.*?)</div>(<svg.*?</svg>)', sec, re.S)
print(f"charts parsed in section: {len(charts)}")
assert len(charts) == 10, len(charts)

n_poly = n_pts = n_raw = n_dots = n_grid = 0
max_err = 0.0
lines_by_key = {}
cap_chart = helix_chart = val_chart = None
for title, svg in charts:
    # ---- (i) y ticks: lowest label 0, constant step on the ladder, labels sit on their gridlines
    labs = [(float(y) - 3.5, float(v)) for y, v in
            re.findall(r'<text x="45" y="([\d.]+)"[^>]*>(-?[\d.]+)</text>', svg)]
    labs.sort(key=lambda p: p[1])
    vals = [v for _, v in labs]
    assert vals[0] == 0.0, f"{title}: lowest tick {vals[0]}"
    steps = [b - a for a, b in zip(vals, vals[1:])]
    step = steps[0]
    assert all(abs(s - step) < 1e-9 * max(1, step) for s in steps), f"{title}: uneven ticks {vals}"
    assert on_ladder(step), f"{title}: step {step} not on 1/2/2.5/5 ladder"
    top = vals[-1]
    assert abs(labs[0][0] - Y1) < 0.051 and abs(labs[-1][0] - Y0) < 0.051, f"{title}: 0/top not at plot edges"

    def sy(v, top=top):
        return Y1 - v / top * (Y1 - Y0)

    for y, v in labs:
        assert abs(y - sy(v)) < 0.051, f"{title}: tick {v} at {y}, expected {sy(v):.2f}"
    # ---- (iv) every gridline has a tick label at its height
    grids = [float(y) for y in re.findall(
        r'<line x1="[\d.]+" y1="([\d.]+)" x2="[\d.]+" y2="[\d.]+" stroke="var\(--rule\)"', svg)]
    assert len(grids) == len(labs), f"{title}: {len(grids)} gridlines vs {len(labs)} labels"
    for g in grids:
        assert any(abs(g - y) < 0.051 for y, _ in labs), f"{title}: unlabeled gridline at {g}"
    n_grid += len(grids)
    # x scale from the first/last x tick labels (both are exact integer data values)
    xl = [(float(x), float(v.replace(",", ""))) for x, v in
          re.findall(r'<text x="([\d.]+)" y="198"[^>]*>([^<]+)</text>', svg)]
    assert abs(xl[0][0] - X0) < 0.051 and abs(xl[-1][0] - X1) < 0.051, f"{title}: x ends"
    x0, x1 = xl[0][1], xl[-1][1]

    def sx(x, x0=x0, x1=x1):
        return X0 + (x - x0) / (x1 - x0) * (X1 - X0)

    for x, v in xl:
        assert abs(x - sx(v)) < 0.051, f"{title}: x tick {v} at {x}, expected {sx(v):.2f}"
    svg_attr = attrs(re.match(r"<svg[^>]*>", svg).group(0))
    cap = float(svg_attr["data-cap"]) if "data-cap" in svg_attr else None
    assert cap is None or cap == top
    # ---- (ii) polylines inverted through this chart's scale reproduce the JSON values
    polys = []
    for tag, pts_s in [(m.group(0), m.group(1)) for m in
                       re.finditer(r'<polyline [^>]*?points="([^"]+)"[^>]*/>', svg)]:
        a = attrs(tag)
        pix = [tuple(map(float, p.split(","))) for p in pts_s.split()]
        exp = [(x, v) for x, v in source(a["data-src"], a["data-key"], a.get("data-smooth")) if x0 <= x <= x1]
        assert len(exp) == len(pix), f"{title} {a['data-key']}: {len(pix)} px vs {len(exp)} source points"
        for (px, py), (x, v) in zip(pix, exp):
            if cap is None:
                assert v <= top + 1e-12, f"{title}: value {v} above top {top} without a cap"
            e = max(abs(px - sx(x)), abs(py - sy(min(v, top))))
            max_err = max(max_err, e)
            assert e <= PX_TOL, f"{title} {a['data-key']} x={x}: err {e:.3f}px"
        n_poly += 1
        n_pts += len(pix)
        polys.append((a, pix))
        lines_by_key[(title, a["data-key"])] = (a, pix, top)
    for gtag, body in re.findall(r'(<g data-raw="1"[^>]*>)(.*?)</g>', svg, re.S):
        a = attrs(gtag)
        dots = [(float(cx), float(cy)) for cx, cy in re.findall(r'cx="([\d.]+)" cy="([\d.]+)"', body)]
        exp = [(x, v) for x, v in source(a["data-src"], a["data-key"]) if x0 <= x <= x1]
        assert len(dots) == len(exp), f"{title} raw {a['data-key']}: {len(dots)} vs {len(exp)}"
        for (cx, cy), (x, v) in zip(dots, exp):
            e = max(abs(cx - sx(x)), abs(cy - sy(v)))
            max_err = max(max_err, e)
            assert e <= PX_TOL, f"{title} raw {a['data-key']} x={x}: err {e:.3f}px"
        n_raw += len(dots)
    # ---- (iii) every end-dot sits on its polyline's last point
    for ctag in re.findall(r'<circle data-end="1"[^>]*/>', svg):
        a = attrs(ctag)
        (pa_, pix), = [(p, x) for p, x in polys if p["data-key"] == a["data-key"] and p["data-src"] == a["data-src"]]
        assert (float(a["cx"]), float(a["cy"])) == pix[-1], f"{title}: end-dot {a['cx']},{a['cy']} vs {pix[-1]}"
        n_dots += 1
    if cap is not None:
        cap_chart = (title, svg, sx, sy, top, x0, x1, polys)
    if "helix_pos_frac" in title:
        helix_chart = (title, svg, sy)
    if title.startswith("c2c — validation loss"):
        val_chart = (title, svg, sx)
    print(f"  {title[:62]:62s} ticks 0..{labs[-1][1]:g} step {step:g}  x {x0:g}..{x1:g}")

print(f"(i)   PASS  {len(charts)} SVGs: lowest y tick 0, constant ladder step, ticks on their gridlines")
print(f"(ii)  PASS  {n_poly} polylines / {n_pts} points + {n_raw} raw dots reproduce JSON; max err {max_err:.3f}px (tol {PX_TOL})")
print(f"(iii) PASS  {n_dots} end-dots each equal their polyline's last point")
print(f"(iv)  PASS  {n_grid} gridlines, every one labelled")

# ---- (v) train panel cap and arrows
title, svg, sx, sy, top, x0, x1, polys = cap_chart
(a, pix), = polys
vals = [v for x, v in source(a["data-src"], a["data-key"]) if x0 <= x <= x1]
s = sorted(vals)
p99 = s[math.ceil(0.99 * len(s)) - 1]
raw_step = p99 / 5
mag = 10 ** math.floor(math.log10(raw_step))
lad = min(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw_step * (1 - 1e-9))
want_top = math.ceil(p99 / lad - 1e-9) * lad
assert abs(want_top - top) < 1e-9, f"cap {top} != p99 {p99} rounded up {want_top}"
over = [(x, v) for x, v in source(a["data-src"], a["data-key"]) if x0 <= x <= x1 and v > top]
arrows = [attrs(t) for t in re.findall(r'<polygon data-cap-arrow="1"[^>]*/>', svg)]
labels = re.findall(r'<text data-cap-label="1"[^>]*>([^<]+)</text>', svg)
assert len(arrows) == len(over) == len(labels), f"{len(arrows)} arrows, {len(labels)} labels, {len(over)} values over cap"
for arr, lab, (x, v) in zip(arrows, labels, over):
    apex = [tuple(map(float, p.split(","))) for p in arr["points"].split()][-1]
    assert abs(apex[0] - sx(x)) <= PX_TOL, f"arrow at {apex[0]} vs {sx(x)}"
    assert abs(float(lab) - v) <= 0.05, f"label {lab} vs {v}"
print(f"(v)   PASS  n={len(vals)} train values; nearest-rank p99 {p99:.4f} -> ladder step {lad:g} -> cap {top:g}; "
      f"{len(over)} value(s) above cap, {len(arrows)} arrow(s): " + ", ".join(f"{v:.2f}@{x}" for x, v in over))

# ---- (vi) helix band spans 0..0.12; excursion shading spans its step window
title, svg, sy = helix_chart
r = attrs(re.search(r'<rect [^>]*fill="var\(--ok\)"[^>]*/>', svg).group(0))
ytop, ybot = float(r["y"]), float(r["y"]) + float(r["height"])
top_h = [float(v) for v in re.findall(r'<text x="45"[^>]*>([\d.]+)</text>', svg)][-1]
v_hi, v_lo = (Y1 - ytop) / (Y1 - Y0) * top_h, (Y1 - ybot) / (Y1 - Y0) * top_h
assert abs(v_hi - 0.12) * (Y1 - Y0) / top_h <= PX_TOL and abs(v_lo) * (Y1 - Y0) / top_h <= PX_TOL, (v_lo, v_hi)
title, svg, sx = val_chart
r = attrs(re.search(r'<rect [^>]*fill="var\(--crit\)"[^>]*/>', svg).group(0))
xa, xb = float(r["x"]), float(r["x"]) + float(r["width"])
assert abs(xa - sx(11500)) <= PX_TOL and abs(xb - sx(13500)) <= PX_TOL, (xa, xb)
print(f"(vi)  PASS  helix band inverts to {v_lo:.4f}..{v_hi:.4f}; c2c val-panel excursion px {xa:.1f}..{xb:.1f} = steps 11,500..13,500")

# ---- (vii) Sep-4 PNG block (and everything else outside the section) byte-identical to the input page
assert page[:pa] == orig[:oa], "bytes before the section changed"
assert page[pb:] == orig[ob:], "bytes after the section changed"
png = re.compile(r'<section class="sec">(?:(?!</section>).)*?data:image/png;base64,(?:(?!</section>).)*?</section>', re.S)
po, pp = png.findall(orig), png.findall(page)
assert len(po) == len(pp) == 1 and po[0] == pp[0], (len(po), len(pp))
print(f"(vii) PASS  PNG section {len(po[0].encode()):,} bytes identical; outside-section prefix "
      f"{len(page[:pa].encode()):,} B + suffix {len(page[pb:].encode()):,} B identical to {ORIG.rsplit('/', 1)[-1]}")

# ---- (viii) table smoothed values equal the chart lines' last points (and the JSON)
cells = re.findall(r'<td class="num" data-key="([^"]+)" data-smoothed="1">([\d.]+)</td>', sec)
chains = dict(re.findall(r'<td class="num" data-chains="([^"]+)">(\d+)</td>', sec))
assert len(cells) == 6 and len(chains) == 3, (len(cells), len(chains))
onestep_title = [t for t, _ in charts if "one-step" in t][0]
pal = SRC["tri_val_pal.json"]["onestep_by_epoch"]
for key, txt in cells:
    a, pix, top = lines_by_key[(onestep_title, key)]
    line_last = (Y1 - pix[-1][1]) / (Y1 - Y0) * top
    exact = source("tri_val_pal.json", key, a["data-smooth"])[-1][1]
    assert abs(float(txt) - line_last) * (Y1 - Y0) / top <= PX_TOL, (key, txt, line_last)
    assert abs(float(txt) - exact) <= 0.0005 + 1e-12, (key, txt, exact)
    print(f"        {key.rsplit('/', 1)[-1]:40s} table {txt}  line-end {line_last:.4f}  json {exact:.4f}")
for key, n in chains.items():
    rows = pal[key]
    e = rows[-1][0]
    want = sum(r[2] for r in rows if e - 4 <= r[0] <= e)
    assert int(n) == want, (key, n, want)
    print(f"        {key.rsplit('/', 1)[-1]:40s} chains {n} = sum n over epochs {e - 4}..{e}")
print("(viii) PASS  6 smoothed table cells match their line ends (<=0.5px) and the JSON (<=0.0005); 3 chain counts match")
print("ALL CHECKS PASSED")
