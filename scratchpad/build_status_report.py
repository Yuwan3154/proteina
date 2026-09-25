"""Regenerate the convergence section of figures/training_status_report.html in place.

Reads the page and the three JSON snapshots, rebuilds the whole
<section class="sec" id="convergence-2026-09-21"> ... </section> (cards, tri loss charts, the tri
validation P@L block, c2c charts, callouts) and writes it back. Every byte outside that section is
kept. Idempotent: the section is found by its opening tag and the first </section> after it and is
replaced, never inserted.

Axis rule (user 2026-09-22): every y-axis starts at a hard 0. The tick step is the smallest value on
the 1/2/2.5/5 x 10^k ladder >= max/5 (nice_step of scratchpad/nice_axis_reticker.py); the top is the
smallest multiple of that step >= the plotted max (for the c2c train panel: >= the nearest-rank p99).
Every mark (line, dot, tick, gridline, band, shading, arrow) is placed by that one scale.
"""

import json
import math
import re

PAGE = "/Users/Chenxi/SOLab/proteina/.claude/worktrees/distogram-head/figures/training_status_report.html"
DATA = "/Users/Chenxi/.claude/jobs/2c2943b0/tmp/report/r20260925"
SNAPSHOT = "25 Sep 2026"  # date of the JSON snapshots in DATA; update after a re-fetch

TRI = json.load(open(f"{DATA}/tri_epochs.json"))
C2C = json.load(open(f"{DATA}/c2c_steps.json"))
PAL = json.load(open(f"{DATA}/tri_val_pal.json"))
C2C_CF = json.load(open(f"{DATA}/c2c_steps_confind.json"))          # ConFind c2c twin (same recipe, ConFind maps)
TRI_FT = json.load(open(f"{DATA}/tri_epochs_tri_confindsynth_ft.json"))  # ConFind tri fine-tune (launched 25 Sep)
C2C_SEGMENTS, C2C_CF_SEGMENTS = 11, 6   # wandb segments per run, from the fetch_report_data.py log of 25 Sep

SEC_OPEN = '<section class="sec" id="convergence-2026-09-21">'
CAV_OPEN = '<section class="sec" id="archive-caveat-2026-09-22">'
HEADER_DATE = re.compile(r"convergence status updated \d{1,2} \w{3} \d{4}")
SEC_CLOSE = "</section>"
W, H = 560, 210
X0, X1, Y0, Y1 = 52, 548, 14, 176
LADDER = (1, 2, 2.5, 5, 10)
TAIL = 50        # tri convergence-view start epoch (unchanged from build_convergence.py)
CUT = 3000       # c2c informative-window start step (unchanged from build_convergence.py)
SMOOTH = 9       # one-step P@L centred rolling-median window, epochs (user: keep 9)
P99 = 0.99       # c2c train-panel cap percentile (user decision 3)
HELIX_BAND = (0.0, 0.12)
VAL_EXC = (11500, 13500)   # c2c val-loss excursion window (unchanged from build_convergence.py)
MAE_EXC = (13000, 15000)   # c2c distance-MAE excursion window (unchanged)
LEAK_TRAIN, LEAK_TOTAL = 31, 32   # val_fixed32_max256.txt chains in the TRAIN split, verified 2026-09-22
BLUE, ORANGE = "var(--la)", "var(--tri)"


def _pts(rows, xk, yk):
    return [(r[xk], r[yk]) for r in rows if xk in r and yk in r]


def _fmt(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def y_axis(vmax, target=5):
    """Hard-0 axis on the 1/2/2.5/5 x 10^k ladder: (top, [(tick value, label), ...])."""
    raw = vmax / target
    k = math.floor(math.log10(raw))
    m = next(m for m in LADDER if raw <= m * 10 ** k * 1.0000001)
    if m == 10:
        m, k = 1, k + 1
    step = m * 10 ** k
    dec = max(0, (1 if m == 2.5 else 0) - k)
    n = max(1, math.ceil(vmax / step - 1e-9))
    return n * step, [(i * step, f"{i * step:.{dec}f}") for i in range(n + 1)]


def nearest_rank(vals, q):
    s = sorted(vals)
    return s[math.ceil(q * len(s)) - 1]


def rolling_median(pts, w=SMOOTH):
    """Centred rolling median over an epoch window, truncated at the ends. [(epoch, value)]."""
    out = []
    for e, *_r in pts:
        vals = sorted(p[1] for p in pts if e - w // 2 <= p[0] <= e + w // 2)
        n = len(vals)
        out.append((e, vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])))
    return out


def S(src, key, label, col, pts, width=1.9, dash=None, opacity=None, smooth=None):
    return dict(src=src, key=key, label=label, col=col, pts=pts, width=width, dash=dash,
                opacity=opacity, smooth=smooth)


def legend(series):
    return " ".join(f'<span class="lg"><i style="background:{s["col"]}"></i>{s["label"]}</span>'
                    for s in series)


def chart(title, series, *, xlab=None, xdom=None, cap=None, band=None, band_label="", notes=(),
          raw=(), end_dots=True, leg="", caption="", aria=None):
    """series/raw: lists of S(...). cap: value the axis top is rounded up from; values above the
    resulting top are clamped to the top edge and marked with an up-arrow + value label."""
    allx = [x for s in series for x, _ in s["pts"]]
    ally = [y for s in series for _, y in s["pts"]] + [y for g in raw for _, y in g["pts"]]
    if band:
        ally.append(band[1])
    x0, x1 = xdom if xdom else (min(allx), max(allx))
    top, ticks = y_axis(cap if cap is not None else max(ally))
    assert cap is not None or max(ally) <= top + 1e-12

    def sx(x):
        return X0 + (x - x0) / (x1 - x0) * (X1 - X0)

    def sy(y):
        return Y1 - y / top * (Y1 - Y0)

    def yv(y):
        return min(y, top) if cap is not None else y

    capattr = f' data-cap="{ticks[-1][1]}"' if cap is not None else ""
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="xMidYMid meet" role="img" '
         f'aria-label="{aria or (title + " versus " + (xlab or "epoch"))}"{capattr}>']
    for v, lab in ticks:
        yy = sy(v)
        o.append(f'<line x1="{X0}" y1="{yy:.1f}" x2="{X1}" y2="{yy:.1f}" stroke="var(--rule)" stroke-width="1"/>')
        o.append(f'<text x="{X0 - 7}" y="{yy + 3.5:.1f}" text-anchor="end" font-size="10" '
                 f'fill="var(--muted)" font-family="var(--mono)">{lab}</text>')
    for i in range(5):
        xv = x0 + round((x1 - x0) * i / 4)  # integer tick value, placed at its own position
        anchor = "end" if i == 4 else "middle"  # the last label would overrun the 560 px edge
        o.append(f'<text x="{sx(xv):.1f}" y="{H - 12}" text-anchor="{anchor}" font-size="10" '
                 f'fill="var(--muted)" font-family="var(--mono)">{_fmt(xv)}</text>')
    if band:
        by0, by1 = sy(band[1]), sy(band[0])
        o.append(f'<rect x="{X0}" y="{by0:.1f}" width="{X1 - X0}" height="{by1 - by0:.1f}" '
                 f'fill="var(--ok)" opacity="0.13"/>')
        if band_label:
            o.append(f'<text x="{X0 + 4}" y="{by0 - 3:.1f}" text-anchor="start" font-size="9.5" '
                     f'fill="var(--ok)" font-family="var(--mono)">{band_label}</text>')
    for a, b, lab in notes:
        o.append(f'<rect x="{sx(a):.1f}" y="{Y0}" width="{sx(b) - sx(a):.1f}" height="{Y1 - Y0}" '
                 f'fill="var(--crit)" opacity="0.07"/>')
        o.append(f'<text x="{(sx(a) + sx(b)) / 2:.1f}" y="{Y0 + 11}" text-anchor="middle" '
                 f'font-size="9.5" fill="var(--crit)" font-family="var(--mono)">{lab}</text>')
    o.append(f'<line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="var(--ink-2)" stroke-width="1"/>')
    o.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="var(--ink-2)" stroke-width="1"/>')
    for g in raw:
        style = (f'fill="none" stroke="{g["col"]}" stroke-width="0.8" stroke-opacity="0.4"' if g["dash"]
                 else f'fill="{g["col"]}" fill-opacity="0.3"')
        dots = "".join(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="1.6"/>' for x, y in g["pts"])
        o.append(f'<g data-raw="1" data-src="{g["src"]}" data-key="{g["key"]}" {style}>{dots}</g>')
    for s in series:
        tag = f'data-src="{s["src"]}" data-key="{s["key"]}"' + (
            f' data-smooth="{s["smooth"]}"' if s["smooth"] else "")
        d = " ".join(f"{sx(x):.1f},{sy(yv(y)):.1f}" for x, y in s["pts"])
        op = f' stroke-opacity="{s["opacity"]}"' if s["opacity"] is not None else ""
        da = f' stroke-dasharray="{s["dash"]}"' if s["dash"] else ""
        o.append(f'<polyline {tag} points="{d}" fill="none" stroke="{s["col"]}" stroke-width="{s["width"]}"'
                 f'{op} stroke-linejoin="round" stroke-linecap="round"{da}/>')
        if end_dots:
            lx, ly = s["pts"][-1]
            o.append(f'<circle data-end="1" {tag} cx="{sx(lx):.1f}" cy="{sy(yv(ly)):.1f}" r="2.8" fill="{s["col"]}"/>')
    if cap is not None:
        for s in series:
            for x, y in s["pts"]:
                if y > top:
                    ax = sx(x)
                    o.append(f'<polygon data-cap-arrow="1" points="{ax - 4:.1f},{Y0:.1f} {ax + 4:.1f},{Y0:.1f} '
                             f'{ax:.1f},{Y0 - 8:.1f}" fill="{s["col"]}"/>')
                    anchor, tx = ("end", ax - 6) if ax > X1 - 60 else ("start", ax + 6)
                    o.append(f'<text data-cap-label="1" x="{tx:.1f}" y="{Y0 - 1}" text-anchor="{anchor}" '
                             f'font-size="9.5" fill="{s["col"]}" font-family="var(--mono)">{y:.1f}</text>')
    if xlab:
        o.append(f'<text x="{X0}" y="{H - 1}" font-size="10" fill="var(--muted)">{xlab}</text>')
    o.append("</svg>")
    return (f'<div class="chartbox"><div class="chart-t">{title}</div>{"".join(o)}'
            f'<div class="legend">{leg or legend(series)}</div>{caption}</div>')


def caption_p(text):
    return f'<p class="note" style="margin:.1rem 0 .3rem;font-size:.72rem">{text}</p>'


# ---------------------------------------------------------------- tri loss
tri_tc = _pts(TRI, "epoch", "train_contact")
tri_vc = _pts(TRI, "epoch", "val_contact")
tri_ta = _pts(TRI, "epoch", "train_align")
tri_va = _pts(TRI, "epoch", "val_align")
tri_tc_tail = [p for p in tri_tc if p[0] >= TAIL]
tri_vc_tail = [p for p in tri_vc if p[0] >= TAIL]
tri_first_tail, tri_last_tail = tri_tc_tail[0], tri_tc_tail[-1]
tri_span = int(tri_last_tail[0] - tri_first_tail[0])
tri_drop = (tri_first_tail[1] - tri_last_tail[1]) / tri_first_tail[1] * 100
vt = [y for _, y in tri_vc_tail]
tri_val_min, tri_val_max = min(vt), max(vt)


def _ols(pts):
    """Least-squares slope per epoch and its t statistic (plain OLS standard error)."""
    n = len(pts)
    mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    rss = sum((y - my - b * (x - mx)) ** 2 for x, y in pts)
    return b, b / math.sqrt(rss / (n - 2) / sxx)


tri_val_slope, tri_val_t = _ols(tri_vc_tail)
tri_tr_slope, tri_tr_t = _ols(tri_tc_tail)
tri_val_falling = tri_val_slope < 0 and tri_val_t <= -2   # |t| >= 2: the conventional ~95% bar
tri_last_step = int(next(r for r in TRI if r["epoch"] == tri_last_tail[0])["step"])

T = "tri_epochs.json"
charts_tri = (
    chart("tri — contact-map loss (full run)",
          [S(T, "train_contact", "train", ORANGE, tri_tc), S(T, "val_contact", "validation", BLUE, tri_vc)],
          xlab="epoch")
    + chart(f"tri — contact-map loss (epoch {TAIL}+, the convergence view)",
            [S(T, "train_contact", "train", ORANGE, tri_tc_tail),
             S(T, "val_contact", "validation", BLUE, tri_vc_tail)], xlab="epoch")
    + chart("tri — alignment loss",
            [S(T, "train_align", "train", ORANGE, tri_ta), S(T, "val_align", "validation", BLUE, tri_va)],
            xlab="epoch")
)

# ---------------------------------------------------------------- tri validation P@L
OS, SA = PAL["onestep_by_epoch"], PAL["sampling"]
P = "tri_val_pal.json"
MODEL = "validation_loss/contact_precision_at_L_single_step"
FLOOR = "validation_loss/contact_precision_at_L_noisy_floor"
STRATA = ("_tlow", "_tmid", "_thigh")
COL = {"_tlow": "var(--crit)", "_tmid": "var(--tri)", "_thigh": "var(--ok)"}
NAME = {"_tlow": "t_low (most corrupted)", "_tmid": "t_mid", "_thigh": "t_high (nearly clean)"}
xdom_pal = (0, int(PAL["epoch"]))

one_series, one_raw, table, nmin, nmax = [], [], {}, None, None
for s in STRATA:
    md, fl = OS[MODEL + s], OS[FLOOR + s]
    assert [(p[0], p[2]) for p in md] == [(p[0], p[2]) for p in fl], f"{s}: model/floor epochs or n differ"
    ns = [p[2] for p in md]
    nmin = min(ns) if nmin is None else min(nmin, min(ns))
    nmax = max(ns) if nmax is None else max(nmax, max(ns))
    sm_md, sm_fl = rolling_median(md), rolling_median(fl)
    one_raw += [S(P, FLOOR + s, "", COL[s], [(p[0], p[1]) for p in fl], dash="3 3"),
                S(P, MODEL + s, "", COL[s], [(p[0], p[1]) for p in md])]
    one_series += [S(P, FLOOR + s, "", COL[s], sm_fl, width=1.3, dash="3 3", opacity=0.75, smooth=SMOOTH),
                   S(P, MODEL + s, "", COL[s], sm_md, width=1.9, opacity=1.0, smooth=SMOOTH)]
    e_last = md[-1][0]
    win = [p for p in md if e_last - SMOOTH // 2 <= p[0] <= e_last]
    table[s] = dict(md=sm_md[-1][1], fl=sm_fl[-1][1], e0=win[0][0], e1=win[-1][0], k=len(win),
                    n=sum(p[2] for p in win))

leg1 = "".join(f'<span class="lg"><i style="background:{COL[s]}"></i>{NAME[s]}</span>' for s in STRATA)
leg1 += '<span class="lg" style="opacity:.75">dashed = noised-input floor</span>'
leg1 += '<span class="lg" style="opacity:.75">faint dots = raw per-epoch values (hollow = floor)</span>'
chart_one = chart(
    "tri &mdash; validation one-step P@L by noise level, vs the floor", one_series, xdom=xdom_pal,
    raw=one_raw, end_dots=False, leg=leg1,
    aria="tri validation one-step P at L, stratified by noise level, against the noised-input floor, versus epoch",
    caption=caption_p(
        f"Centred rolling median over {SMOOTH} epochs (window truncated at the ends; the last point&rsquo;s "
        f"window is listed in the table below). Faint dots are the raw per-epoch values; each rests on "
        f"{nmin}&ndash;{nmax} validation chains per noise stratum, so only the multi-epoch trend is meaningful."))

SAMP = [
    ("validation_sampling/contact_precision_at_L_mean", "var(--tri)", "P@L", 2.0),
    ("validation_sampling/contact_precision_at_L2_mean", "var(--la)", "P@L/2", 1.5),
    ("validation_sampling/contact_precision_at_L5_mean", "var(--ok)", "P@L/5", 1.5),
    ("validation_sampling/contact_long_range_precision_at_L5_mean", "var(--crit)", "long-range P@L/5", 1.5),
]
sm = SA["validation_sampling/contact_precision_at_L_mean"]
LEAK = (f"Scored on <code>val_fixed32_max256.txt</code>, of which <b>{LEAK_TRAIN} of {LEAK_TOTAL} chains are "
        f"TRAINING-split chains</b> (verified 2026-09-22), so it measures fit on seen chains, not "
        f"generalisation; the one-step P@L uses the real validation split.")
chart_samp = chart(
    "tri &mdash; sampling P@L (full rollout) &mdash; on training-split chains",
    [S(P, k, lab, c, [(e, v) for e, v in SA[k]], width=w, opacity=1.0) for k, c, lab, w in SAMP],
    xdom=xdom_pal, end_dots=False,
    leg="".join(f'<span class="lg"><i style="background:{c}"></i>{lab}</span>' for _k, c, lab, _w in SAMP),
    aria="tri sampling precision at L, L over 2, L over 5 and long-range, on training-split chains, versus epoch",
    caption=caption_p(f"One point per sampling round ({len(sm)} rounds). Unsmoothed. {LEAK}"))

rows = []
for s in STRATA:
    t = table[s]
    rows.append(
        f'<tr><td style="color:{COL[s]};font-weight:600">{NAME[s]}</td>'
        f'<td class="num" data-key="{MODEL + s}" data-smoothed="1">{t["md"]:.3f}</td>'
        f'<td class="num" data-key="{FLOOR + s}" data-smoothed="1">{t["fl"]:.3f}</td>'
        f'<td class="num"><b>{t["md"] - t["fl"]:+.3f}</b></td>'
        f'<td class="num">{t["md"] / t["fl"]:.1f}&times;</td>'
        f'<td class="num">{t["e0"]}&ndash;{t["e1"]} ({t["k"]})</td>'
        f'<td class="num" data-chains="{MODEL + s}">{t["n"]}</td></tr>')
lo, hi = table["_tlow"], table["_thigh"]
e_end = max(t["e1"] for t in table.values())
last8 = sm[-8:]
l8min, l8max = min(v for _e, v in last8), max(v for _e, v in last8)

BLOCK = f"""
  <h3 id="tri-val-pal">tri &mdash; P@L: one-step on the validation split (by noise level), and full sampling on training-split chains</h3>
  <p class="note" style="margin:.2rem 0 .6rem">
    Two different questions. <b>One-step P@L</b> asks: given a contact map corrupted to noise level
    <i>t</i>, how well does a <i>single</i> denoising step recover the true contacts? <b>Sampling
    P@L</b> asks the question that actually matters at deployment: run the <i>full</i> generative
    rollout from noise and score the contact map it produces. P@L = precision over the top-L
    predicted contacts, L = chain length. <b>Here the sampling metric is scored on training-split
    chains</b> (see its panel), so only the one-step metric is a held-out measurement.
    <b>x-axis is epoch, not step</b> &mdash; <code>trainer/global_step</code> resets on every resume
    and this run has eight of them, so it is not a usable axis.
  </p>

  <div class="grid2">{chart_one}{chart_samp}</div>

  <div class="scroll">
  <table>
    <thead><tr><th>noise stratum</th><th class="num">one-step P@L</th>
      <th class="num">noised-input floor</th><th class="num">gap</th>
      <th class="num">lift</th><th class="num">window, epochs (present)</th><th class="num">val chains</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  </div>
  <p class="note" style="margin:.45rem 0 0">
    Table values are the <b>smoothed</b> ends: the same {SMOOTH}-epoch centred rolling median as the chart,
    taken at epoch {e_end}, where the centred window is truncated to the epochs listed; &ldquo;val chains&rdquo;
    counts every validation chain in that window.
    <b>Read the one-step panel against its floor, never alone.</b> The floor is what you score by
    simply copying the <i>noised</i> input, so it rises with <i>t</i>: at t_high the corrupted map
    is already {hi["fl"]:.3f} correct and the model's {hi["md"]:.3f} is a {hi["md"] / hi["fl"]:.1f}&times; lift,
    while at t_low the input carries almost nothing ({lo["fl"]:.3f}) and the model's {lo["md"]:.3f} is a
    {lo["md"] / lo["fl"]:.1f}&times; lift. Quoting the headline {hi["md"]:.2f} as the model's skill would be
    crediting it with work the input already did.
    <b>Sampling P@L is not a held-out number:</b> {LEAK[0].lower() + LEAK[1:]}
    On that fixed set it is flat: the last eight rounds sit in {l8min:.3f}&ndash;{l8max:.3f}
    (spread {l8max - l8min:.3f}) across epochs {last8[0][0]}&ndash;{last8[-1][0]},
    against {sm[0][1]:.3f} at epoch {sm[0][0]}. The two metrics are not interchangeable: one scores a single
    denoising step on held-out chains, the other the full rollout on chains the model trained on.
  </p>
"""

# ---------------------------------------------------------------- c2c
c_val = _pts(C2C, "step", "val/loss")
c_tr = _pts(C2C, "step", "train/loss")
c_mae = _pts(C2C, "step", "val/dist_mae_sampled")
c_rms = _pts(C2C, "step", "val/rmsd_proper")
c_hel = _pts(C2C, "step", "val/helix_pos_frac")
c_mir = _pts(C2C, "step", "val/is_mirrored")
c_val_z = [p for p in c_val if p[0] >= CUT]
c_tr_z = [p for p in c_tr if p[0] >= CUT]
c_mae_z = [p for p in c_mae if p[0] >= CUT]
c_rms_z = [p for p in c_rms if p[0] >= CUT]

last3 = c_val[-3:]
c2c_best = min(y for _, y in c_val)
c2c_best_step = [x for x, y in c_val if y == c2c_best][0]
c2c_last_step, c2c_last = c_val[-1]
at_best = c2c_last == c2c_best
exc = [(x, y) for x, y in c_val if VAL_EXC[0] <= x <= VAL_EXC[1]]
exc_step, exc_max = max(exc, key=lambda p: p[1])
mae_last = c_mae[-1][1]
mae_min_step, mae_min = min(c_mae, key=lambda p: p[1])
mae_exc_step, mae_exc = max([p for p in c_mae if MAE_EXC[0] <= p[0] <= MAE_EXC[1]], key=lambda p: p[1])
hel_last = c_hel[-1][1]
hel_peak_step, hel_peak = max(c_hel, key=lambda p: p[1])
mir_nonzero = [x for x, y in c_mir if y > 0]
mir_last_nonzero = mir_nonzero[-1]
mir_at_last = mir_last_nonzero == c_mir[-1][0]
mir_txt = (f"It reads <b>non-zero at the latest round</b> (step {mir_last_nonzero:,}: {c_mir[-1][1]:.3f})"
           if mir_at_last else
           f"It last read non-zero at step {mir_last_nonzero:,} and has been 0 at every validation round since")
# discrete jumps in the sampled structure metrics between consecutive rounds (tbeta 25 Sep: 22,871 -> 23,515)
c_rms_all = _pts(C2C, "step", "val/rmsd_proper")
jumps = [(a[0], b[0], a[1], b[1]) for a, b in zip(c_rms_all, c_rms_all[1:]) if b[0] > CUT and b[1] > 1.8 * a[1]]
jump_txt = ("" if not jumps else
            " ⛔ <b>Structure quality jumped worse between rounds</b>: " + "; ".join(
                f"proper RMSD {r0:.2f} &rarr; {r1:.2f} &Aring; from step {s0:,} to {s1:,}" for s0, s1, r0, r1 in jumps)
            + ", and has not recovered since; the job ran continuously (no resume) across it, so the cause is open."
              " A 195-chain benchmark of the saved checkpoints on either side would separate a real regression from"
              " 16-structure validation noise.")

p99 = nearest_rank([y for _, y in c_tr_z], P99)
tr_top = y_axis(p99)[0]
above = [(x, y) for x, y in c_tr_z if y > tr_top]
above_txt = ", ".join(f"{y:.1f} at step {x:,}" for x, y in above)
tr_zero = [(x, y) for x, y in c_tr_z if y < 1e-3]
tr_zero_txt = " ".join(f"Step {x:,} logs train/loss = {y:.1e}, a logging anomaly plotted as recorded."
                       for x, y in tr_zero)
cap_caption = caption_p(
    f"Linear axis from a hard 0, capped: the top is the 99th percentile (nearest rank) of the "
    f"{len(c_tr_z)} plotted train/loss values, {p99:.3f}, rounded up to the 1/2/2.5/5&times;10<sup>k</sup> "
    f"tick ladder = {y_axis(p99)[1][-1][1]}. "
    + (f"{len(above)} value{'s' if len(above) != 1 else ''} above the cap "
       f"{'are' if len(above) != 1 else 'is'} drawn as &#9650; at the top edge with "
       f"{'their' if len(above) != 1 else 'its'} value: {above_txt}." if above else "No value exceeds the cap.")
    + (" " + tr_zero_txt if tr_zero else ""))

C = C2C_FILE = "c2c_steps.json"
charts_c2c = (
    chart(f"c2c — train loss (step {CUT}+)", [S(C, "train/loss", "train", ORANGE, c_tr_z)],
          xlab="global step", cap=p99, caption=cap_caption)
    + chart(f"c2c — validation loss (step {CUT}+)", [S(C, "val/loss", "validation", BLUE, c_val_z)],
            xlab="global step", notes=[(VAL_EXC[0], VAL_EXC[1], "excursion")])
    + chart("c2c — sampled distance MAE (Å, lower better)", [S(C, "val/dist_mae_sampled", "val", BLUE, c_mae_z)],
            xlab="global step", notes=[(MAE_EXC[0], MAE_EXC[1], "excursion")])
    + chart("c2c — proper-superposition RMSD (Å, lower better)", [S(C, "val/rmsd_proper", "val", BLUE, c_rms_z)],
            xlab="global step")
    + chart("c2c — helix_pos_frac (handedness; LOW = native)", [S(C, "val/helix_pos_frac", "val", BLUE, c_hel)],
            xlab="global step", band=HELIX_BAND, band_label=f"native ≈{HELIX_BAND[1]}")
)

# ---------------------------------------------------------------- CB-8 vs ConFind c2c at matched steps
GREEN = "var(--ok)"
CF = "c2c_steps_confind.json"
cf_max = max(r["step"] for r in C2C_CF)
cf_val_steps = [r["step"] for r in C2C_CF if "val/loss" in r]
tb_by = {r["step"]: r for r in C2C if "val/loss" in r}
cf_by = {r["step"]: r for r in C2C_CF if "val/loss" in r}
matched = [st for st in cf_val_steps if st in tb_by]
xdom_cf = (0, cf_max)
MKEYS = (("val/loss", "validation loss"), ("val/rmsd_proper", "proper-superposition RMSD (Å, lower better)"),
         ("val/dist_mae_sampled", "sampled distance MAE (Å, lower better)"))
charts_cmp = "".join(
    chart(f"c2c — {lab}, CB-8 vs ConFind (steps 0–{cf_max:,})",
          [S(C2C_FILE, k, "CB-8 (tbeta)", BLUE, [p for p in _pts(C2C, "step", k) if p[0] <= cf_max], opacity=1.0),
           S(CF, k, "ConFind (twin)", GREEN, _pts(C2C_CF, "step", k), opacity=1.0)],
          xlab="global step", xdom=xdom_cf)
    for k, lab in MKEYS)
cmp_rows = "".join(
    f'<tr><td class="num">{st:,}</td>'
    + "".join(f'<td class="num">{tb_by[st].get(k, float("nan")):.3f}</td><td class="num">{cf_by[st].get(k, float("nan")):.3f}</td>'
              for k, _l in MKEYS) + "</tr>" for st in matched)
ft = TRI_FT[-1]
CMP_BLOCK = f"""
  <h3 id="c2c-cb8-vs-confind">c2c &mdash; CB-8 vs ConFind at matched steps (the Stage B pair)</h3>
  <p class="note" style="margin:.2rem 0 .6rem">
    The ConFind twin trains with tbeta&rsquo;s exact recipe (48 diffusion samples, lr 3e-4, warmup 2,000, t_beta
    (1.3, 2.0), effective batch 8); only the input contact map differs. Both validate on the same steps, so the
    table compares identical steps. ⛔ <b>Too early to read a winner:</b> the twin is at step {cf_max:,} of the
    ~9,500 needed for the first matched comparison, and this window (before step {CUT:,}) is where tbeta itself
    was unstable &mdash; its step-2,143 RMSD spike was a transient the report&rsquo;s convergence view starts after.
    tbeta also crossed a mid-run target change (pseudo-CB, step 7,076) that the twin never sees.
  </p>
  <div class="grid2">{charts_cmp}</div>
  <div class="scroll"><table>
    <thead><tr><th class="num">step</th><th class="num">val loss CB-8</th><th class="num">ConFind</th>
      <th class="num">RMSD CB-8</th><th class="num">ConFind</th><th class="num">dist MAE CB-8</th><th class="num">ConFind</th></tr></thead>
    <tbody>{cmp_rows}</tbody></table></div>
"""
CF_CARDS = f"""
  <div class="cards" style="margin:1.1rem 0">
    <div class="card la">
      <h3 style="margin-top:0">c2c_confind_tbeta (ConFind twin) &mdash; early</h3>
      <p class="note">Step {cf_max:,} across {C2C_CF_SEGMENTS} chained segments, {len(cf_val_steps)} validation rounds.
      Training resumed cleanly after the 24 Sep scratch-quota incident. Needs step 9,500 (retention-ladder
      anchor) for the first matched structure-level comparison.</p>
    </div>
    <div class="card tri">
      <h3 style="margin-top:0">tri_confindsynth_ft (ConFind tri fine-tune) &mdash; just started</h3>
      <p class="note">Launched 25 Sep 13:57 on 2&times; RTX PRO 6000: the old ConFind tri (71,950 EMA) surgered into the
      current recipe (synthetic references, vocab 44, no CA features, align + MLM heads) on a pure-ConFind
      synthetic-reference index. Gates passed: total_training_steps 403,593 (= the CB-8 tri), only the 6 new head
      parameters cold-started. One validation point so far (step {int(ft["step"])}, epoch {ft["epoch"]}, during warmup,
      lr {ft["lr"]:.1e}): validation contact loss {ft["val_contact"]:.4f}. ⛔ Its numbers are on the ConFind
      definition and not comparable to the CB-8 tri&rsquo;s. ~200 steps/h &rArr; 10,000 steps ≈ 27 Sep.</p>
    </div>
  </div>
"""

# ---------------------------------------------------------------- prose that depends on the data
if tri_val_falling:
    tri_lede = (f"""<strong>tri: not yet.</strong> Over epochs {TAIL}&ndash;{int(tri_last_tail[0])} validation
    contact loss is still falling ({tri_val_slope:.1e} per epoch, t = {tri_val_t:.1f}), about as fast as
    training loss ({tri_tr_slope:.1e} per epoch; {tri_drop:.1f}% over {tri_span} epochs) &mdash; a slow
    late-training descent with no train/validation divergence.""")
    tri_title = "still descending slowly"
    tri_read = ("not overfitting (validation tracks training), and both curves are still improving, slowly; "
                "no plateau yet on this metric.")
else:
    tri_lede = (f"""<strong>tri: close, but not finished.</strong> Validation contact loss shows no
    downward trend over ~{tri_span} epochs ({tri_val_slope:.1e} per epoch, t = {tri_val_t:.1f}) while
    training loss still creeps down {tri_drop:.1f}% &mdash; no train/validation divergence.""")
    tri_title = "approaching convergence"
    tri_read = ("validation has plateaued; training has not yet. Not overfitting "
                "(validation tracks training), but there is little left to gain on this metric.")
if at_best:
    c2c_lede = (f"""<strong>c2c: no &mdash; still improving.</strong> Its validation loss is at the lowest value of the
    whole run right now ({c2c_best:.3f} at step {c2c_best_step:,}), and the curve has already shown one
    large non-monotone excursion, so a flat-looking tail here would not be evidence of convergence.""")
else:
    c2c_lede = (f"""<strong>c2c: no.</strong> Its lowest validation loss is {c2c_best:.3f} at step {c2c_best_step:,},
    but the latest round reads {c2c_last:.3f} at step {c2c_last_step:,}, and the curve had already shown one
    large non-monotone excursion before that, so convergence can only be judged over a window longer than it.""")
two_recent_lowest = sorted(y for _, y in c_val)[:2] == sorted(y for _, y in c_val[-2:])
trend_txt = (" &mdash; the two lowest values of the run are the\n        two most recent" if two_recent_lowest
             else f"; the run&rsquo;s lowest is {c2c_best:.3f} at step {c2c_best_step:,}")
mae_txt = (f"Sampled distance MAE is likewise at its best, {mae_last:.3f} &Aring;." if mae_last == mae_min
           else f"Sampled distance MAE reads {mae_last:.3f} &Aring; against its best, {mae_min:.3f} &Aring; at step {mae_min_step:,}.")
read_txt = ("still descending, and demonstrably capable of moving the wrong way for\n        thousands of steps."
            if at_best else
            f"not at its best (the latest round is {c2c_last / c2c_best:.1f}&times; the run&rsquo;s lowest), and\n"
            f"        demonstrably capable of moving the wrong way for thousands of steps.")
hel_txt = ("inside the native band" if hel_last <= HELIX_BAND[1]
           else f"above the {HELIX_BAND[0]:g}&ndash;{HELIX_BAND[1]} native band drawn on the chart")

html = f"""{SEC_OPEN}
  <div class="eyebrow" style="margin-bottom:.35rem">Current training status &middot; generated {SNAPSHOT} from wandb history</div>
  <h2 style="margin-top:0">Have the live models converged? CB-8 pair + the new ConFind pair</h2>
  <p class="lede" style="font-size:1.02rem">
    {tri_lede}
    {c2c_lede}
  </p>

  <div class="cards" style="margin:1.1rem 0">
    <div class="card tri">
      <h3 style="margin-top:0">tri_cb8synth_v5 &mdash; {tri_title}</h3>
      <p class="note">
        Epoch {int(tri_last_tail[0])}, step {tri_last_step:,}. Training contact loss
        {tri_first_tail[1]:.4f} (ep {int(tri_first_tail[0])}) &rarr; {tri_last_tail[1]:.4f}
        (ep {int(tri_last_tail[0])}), a {tri_drop:.1f}% decrease across {tri_span} epochs.
        Validation sits in a {tri_val_min:.4f}&ndash;{tri_val_max:.4f} band over the same span, with a
        least-squares slope of {tri_val_slope:.2e} per epoch (t = {tri_val_t:.1f}) against training&rsquo;s
        {tri_tr_slope:.2e} (t = {tri_tr_t:.1f}).
        <strong>Read:</strong> {tri_read}
      </p>
    </div>
    <div class="card la">
      <h3 style="margin-top:0">c2c_cb8_tbeta &mdash; NOT converged</h3>
      <p class="note">
        Step {int(c2c_last_step):,}. Validation loss {last3[0][1]:.3f} &rarr; {last3[1][1]:.3f} &rarr;
        {last3[2][1]:.3f} across the last three rounds{trend_txt}. {mae_txt}
        <strong>{'But' if at_best else 'Earlier,'}</strong> the run posted an excursion to {exc_max:.3f} at step {exc_step:,} and a
        distance-MAE excursion to {mae_exc:.2f} &Aring; at step {mae_exc_step:,} before recovering.
        <strong>Read:</strong> {read_txt} Judge convergence only over a window longer than that excursion.{jump_txt}
      </p>
    </div>
  </div>

{CF_CARDS}
  <h3>tri &mdash; the loss curves</h3>
  <div class="grid2">{charts_tri}</div>
  <p class="note">The full-run panel is dominated by the first five epochs (loss falls from
  {tri_tc[0][1]:.2f} to under 0.07); the epoch-{TAIL}+ panel is the one to judge convergence on.</p>
{BLOCK}
  <h3>c2c &mdash; loss, structure quality, and handedness</h3>
  <div class="grid2">{charts_c2c}</div>
{CMP_BLOCK}
  <div class="callout">
    <p><strong>How to read <code>helix_pos_frac</code> &mdash; low is correct.</strong>
    It is the fraction of helical-range CA pseudo-dihedrals that are positive, and it is a
    <em>handedness</em> indicator, not a quality score. Calibrated on <strong>254 native chains</strong>:
    median <strong>0.0815</strong>, mean <strong>0.1225</strong>, only 2.0% above 0.5. So
    <strong>~0.12 is native-handed and ~0.89 is mirrored</strong>. The model now reads
    <strong>{hel_last:.3f}</strong> &mdash; {hel_txt}. The fall from
    <strong>{hel_peak:.3f} at step {hel_peak_step:,}</strong> to under 0.13 is the fold flipping to the correct hand.</p>
    <p>⛔ Two traps this page deliberately does not fall into. <strong>(1)</strong>
    <code>chirality_agree</code> is <em>not</em> a mirror detector despite the name &mdash; it tests
    per-residue stereocentres, and mirrored generations keep their residues correctly L (measured
    0.999 across 122 mirrored chains), so it reports &ldquo;all good&rdquo; through a 48% mirror rate.
    <strong>(2)</strong> <code>helix_pos_frac</code> is a <em>local</em> statistic &mdash; the same
    quantity the chirality loss trains on &mdash; so it cannot independently confirm a chirality fix.
    The non-circular discriminator is the proper-vs-reflected superposition gap.</p>
  </div>

  <div class="callout warn">
    <p><strong>Why <code>is_mirrored</code> is not plotted as a rate.</strong> It fires only when
    <code>proper &gt; 2&times;reflected</code> and the gap exceeds 1 &Aring;, so it is
    fit-quality-biased: before ~step 3,000 the generated structures sit
    <strong>11.6&ndash;145 &Aring;</strong> from native, where the test cannot meaningfully fire and a
    reading of 0 means <em>no verdict</em>, not &ldquo;correct hand&rdquo;. {mir_txt}.
    ⛔ Per-round values are a liveness monitor over ~16 structures, <strong>not</strong> a rate:
    0 of 16 is entirely consistent with the ~1.6% rate measured by the 512-draw paired assay
    (P = 0.98<sup>16</sup> &asymp; 0.77).</p>
    <p><strong>tri&rsquo;s one-step <code>contact_precision_at_L</code> is only shown per noise level, beside
    its floor.</strong> Within one validation pass it swings with the noise level drawn, because it is a
    single denoising step from a corrupted map rather than a full rollout, so a single pooled curve
    would be meaningless. The full-rollout (sampling) number is the separate panel above, and a
    held-out version of it comes only from <code>tri_gen_eval.py</code> on a validation-split chain list.</p>
  </div>

  <p class="note">Sources: <code>tri_cb8synth_v5</code> {len(TRI)} epochs from one wandb run;
  <code>c2c_cb8_tbeta</code> {len({r['step'] for r in C2C})} distinct global steps concatenated across <strong>{C2C_SEGMENTS}</strong> chained
  segments (each segment gets a new run id) and de-duplicated on step. {len(c_mae)} validation rounds carry the
  sampled metrics. Every plotted point is measured; nothing is interpolated or estimated.</p>
{SEC_CLOSE}"""

# The archived 4 Sep narrative below quotes sampling P@L scored on the same fixed list; its bytes are
# kept, and this note (regenerated with the section) says what those numbers measure.
caveat = f"""{CAV_OPEN}
  <div style="background:var(--panel);border-left:3px solid var(--crit);border-radius:0 8px 8px 0;padding:12px 16px;max-width:86ch">
    <p style="margin:0;font-size:.9rem;line-height:1.55;color:var(--ink-2)"><strong>Correction, 22 Sep 2026, to the archived numbers below.</strong> Every sampling-P@L value
    in the 4 Sep narrative (the run cards, the <code>validation_sampling</code> table and the figure) was
    scored on <code>val_fixed32_max256.txt</code>. Under the max384 split that tri_full384 and
    localattn_full384 trained on, <b>{LEAK_TRAIN} of those {LEAK_TOTAL} chains are training-split chains</b>, so
    for those runs the numbers measure fit on seen chains, not generalisation. The list was drawn from
    max512 validation representatives, so the max512 runs are not affected.</p>
  </div>
{SEC_CLOSE}"""

# ---------------------------------------------------------------- splice in place
page = open(PAGE).read()
assert page.count(SEC_OPEN) == 1, f"section open tag count {page.count(SEC_OPEN)}"
a = page.index(SEC_OPEN)
b = page.index(SEC_CLOSE, a) + len(SEC_CLOSE)
assert "<section" not in page[a + len(SEC_OPEN):b], "nested <section> inside the convergence section"
rest = page[b:]
if rest.lstrip().startswith(CAV_OPEN):   # a previous run's note: replace it, never stack a second
    b = page.index(SEC_CLOSE, b + len(rest) - len(rest.lstrip())) + len(SEC_CLOSE)
assert page.count(CAV_OPEN) <= 1
out = page[:a] + html + "\n\n" + caveat + page[b:]
assert len(HEADER_DATE.findall(out)) == 1, "header date string not found exactly once"
out = HEADER_DATE.sub(f"convergence status updated {SNAPSHOT}", out)
open(PAGE, "w").write(out)

print(f"section: {b - a} -> {len(html)} chars; page {len(page)} -> {len(out)} chars")
print(f"tri: last train ep {int(tri_last_tail[0])}, last row ep {TRI[-1]['epoch']} step {int(TRI[-1]['step'])}, "
      f"drop {tri_drop:.1f}% over {tri_span} ep, val band {tri_val_min:.4f}-{tri_val_max:.4f}")
print(f"c2c: steps {len(C2C)}, val rounds {len(c_val)}, last3 {[round(y, 4) for _, y in last3]}, "
      f"best {c2c_best:.4f}@{c2c_best_step}, last {c2c_last:.4f}@{c2c_last_step}, at_best={at_best}")
print(f"c2c train cap: n={len(c_tr_z)} p99(nearest rank)={p99:.4f} -> top {tr_top:g}; above: {above}")
print(f"mae last {mae_last:.3f} min {mae_min:.3f}@{mae_min_step} exc {mae_exc:.3f}@{mae_exc_step}; "
      f"helix last {hel_last:.3f} peak {hel_peak:.3f}@{hel_peak_step}; last nonzero is_mirrored @{mir_last_nonzero}")
print(f"P@L raw chains/epoch/stratum {nmin}-{nmax}; sampling rounds {len(sm)}; last8 {l8min:.3f}-{l8max:.3f}")
for s in STRATA:
    t = table[s]
    print(f"  {s:7s} model {t['md']:.4f} floor {t['fl']:.4f} gap {t['md'] - t['fl']:+.4f} "
          f"lift {t['md'] / t['fl']:.2f}x window {t['e0']}-{t['e1']} ({t['k']} ep) chains {t['n']}")
