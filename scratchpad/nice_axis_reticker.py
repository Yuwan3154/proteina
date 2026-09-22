"""Re-tick every chart in training_status_report.html: round intervals, minimum at the bottom.

The originals were auto-scaled min..max into 5 even ticks, which printed a NEGATIVE bottom tick on
a loss, a distance, an RMSD and a fraction. This does not need the source data: the existing tick
labels give an exact pixel->value map, so each polyline is inverted back to data values, a round
axis is chosen, and everything is re-emitted on the new scale. The curves are unchanged.
"""

import math
import re

H = "/Users/Chenxi/SOLab/proteina/.claude/worktrees/distogram-head/figures/training_status_report.html"
Y0, Y1 = 14.0, 176.0          # plot box top / bottom, unchanged
X0, X1 = 52.0, 548.0


def nice_step(lo, hi, target=5):
    raw = (hi - lo) / target
    if raw <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag * 1.0000001:
            return m * mag
    return 10 * mag


def nice_axis(lo, hi, target=5):
    s = nice_step(lo, hi, target)
    a = math.floor(lo / s + 1e-9) * s
    b = math.ceil(hi / s - 1e-9) * s
    if b <= a:
        b = a + s
    return a, b, s


def decimals(step):
    if step >= 1:
        return 0
    d = -math.floor(math.log10(step))
    # a 2.5e-k step needs one more decimal than 1e-k
    if abs(step * 10 ** d - 2.5) < 1e-9:
        d += 1
    return int(d)


def fit(ticks):
    """(pixel, value) pairs -> (a, b) with pixel = a + b*value, least squares."""
    n = len(ticks)
    sx = sum(v for _p, v in ticks)
    sy = sum(p for p, _v in ticks)
    sxx = sum(v * v for _p, v in ticks)
    sxy = sum(p * v for p, v in ticks)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    return a, b


h = open(H).read()
charts = list(re.finditer(r'(<div class="chart-t">)(.*?)(</div>\s*<svg[^>]*>)(.*?)(</svg>)', h, re.S))
print(f"{len(charts)} charts found")

out = []
last = 0
report = []
for m in charts:
    title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
    svg = m.group(4)

    ticks = [(float(p), float(v)) for p, v in
             re.findall(r'<text x="45" y="([\d.]+)"[^>]*>(-?[\d.]+)</text>', svg)]
    assert len(ticks) >= 3, f"{title}: only {len(ticks)} y ticks"
    # the label sits 3.5px below the gridline it annotates
    ticks = [(p - 3.5, v) for p, v in ticks]
    a, b = fit(ticks)

    polys = re.findall(r'<polyline points="([^"]+)"', svg)
    vals = []
    for p in polys:
        for pair in p.split():
            _x, y = pair.split(",")
            vals.append((float(y) - a) / b)
    lo, hi = min(vals), max(vals)
    nlo, nhi, step = nice_axis(lo, hi)
    dec = decimals(step)

    # --- rebuild y gridlines + labels -----------------------------------------------------
    svg2 = re.sub(r'<line x1="52" y1="[\d.]+" x2="548" y2="[\d.]+" stroke="var\(--rule\)"[^/]*/>', "", svg)
    svg2 = re.sub(r'<text x="45" y="[\d.]+"[^>]*>-?[\d.]+</text>', "", svg2)

    grid = []
    t = nlo
    labels = []
    while t <= nhi + step * 1e-6:
        y = Y1 - (t - nlo) / (nhi - nlo) * (Y1 - Y0)
        grid.append(f'<line x1="52" y1="{y:.1f}" x2="548" y2="{y:.1f}" stroke="var(--rule)" stroke-width="1"/>')
        grid.append(f'<text x="45" y="{y + 3.5:.1f}" text-anchor="end" font-size="10" '
                    f'fill="var(--muted)" font-family="var(--mono)">{t:.{dec}f}</text>')
        labels.append(f"{t:.{dec}f}")
        t += step

    # --- rescale polylines to the new axis -------------------------------------------------
    def remap(mo):
        pts = []
        for pair in mo.group(1).split():
            x, y = pair.split(",")
            v = (float(y) - a) / b
            ny = Y1 - (v - nlo) / (nhi - nlo) * (Y1 - Y0)
            pts.append(f"{float(x):.1f},{ny:.1f}")
        return '<polyline points="' + " ".join(pts) + '"'

    svg2 = re.sub(r'<polyline points="([^"]+)"', remap, svg2)
    svg2 = "".join(grid) + svg2

    out.append(h[last:m.start()])
    out.append(m.group(1) + m.group(2) + m.group(3) + svg2 + m.group(5))
    last = m.end()
    report.append((title, lo, hi, labels))

out.append(h[last:])
open(H, "w").write("".join(out))

print("\n%-58s %-18s %s" % ("chart", "data range", "new ticks"))
for title, lo, hi, labels in report:
    print("%-58s %7.3f..%-8.3f %s" % (title[:58], lo, hi, " ".join(labels)))
