"""Deck example panels, step 2 (user 2026-10-08): render native | template | CB-8 output | ConFind output for each picked
query in ONE shared view (all were superposed onto the native frame), side by side, not overlaid. Cartoon, rainbow
N->C on every panel. One PNG per category (2 rows each), TM to native under each panel.
Usage: python t7_render_examples.py PICKS.json TEMPLATE_TM.tsv SUP_DIR OUT_DIR
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pymol import cmd

picks, tsv, sup, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
ttm = {l.split("\t")[0]: float(l.split("\t")[5]) for i, l in enumerate(open(tsv)) if i}
P = json.load(open(picks))
TITLES = {"both_succeed": "Both pipelines succeed", "confind_only": "ConFind succeeds, CB-8 fails", "both_fail": "Both fail"}


def render(stem):
    cmd.reinitialize()
    cmd.bg_color("white"); cmd.set("ray_opaque_background", 1); cmd.set("cartoon_fancy_helices", 1)
    cmd.set("antialias", 2); cmd.set("ray_shadows", 0); cmd.set("specular", 0.2)
    for tag in ("native", "template_sup", "CB8_sup", "CF_sup"):
        cmd.load(os.path.join(sup, f"{stem}_{tag}.pdb"), tag)
    cmd.orient("native")               # rotation from the native (all four already sit in its frame)
    # camera distance: fit the structures' BODIES -- all CA atoms except the 2% farthest from the native centroid, so a
    # single long disordered tail cannot shrink the whole row (such a tail may run off the panel edge)
    import numpy as np
    c = np.array(cmd.get_coords("native and name CA")).mean(0)
    xyz = np.array(cmd.get_coords("name CA")); cut = np.percentile(np.linalg.norm(xyz - c, axis=1), 98)
    cmd.pseudoatom("ctr", pos=list(map(float, c)))
    cmd.select("core", f"name CA within {cut:.1f} of ctr")
    cmd.zoom("core", 3, complete=1)
    view = cmd.get_view()
    pngs = {}
    for tag in ("native", "template_sup", "CB8_sup", "CF_sup"):
        cmd.delete("all")
        cmd.load(os.path.join(sup, f"{stem}_{tag}.pdb"), "m")
        cmd.dss("m")  # same secondary-structure assignment for every panel
        cmd.hide("everything"); cmd.show("cartoon", "m")
        cmd.spectrum("count", "rainbow", "m and name CA")
        cmd.set_view(view)  # identical camera for every panel: the shared native frame
        p = os.path.join(out, f"{stem}_{tag}.png"); cmd.png(p, width=900, height=900, dpi=300, ray=1); pngs[tag] = p
    return pngs


for cat, ents in P.items():
    fig, axes = plt.subplots(len(ents), 4, figsize=(13, 3.6 * len(ents)))
    for r, e in enumerate(ents):
        st = e["stem"]; pngs = render(st)
        labs = [("native", f"{st} native (L = {e['L']})"), ("template_sup", f"template  TM {ttm[st]:.3f}"),
                ("CB8_sup", f"CB-8 output  TM {e['CB8']['tm']:.3f}"), ("CF_sup", f"ConFind output  TM {e['CF']['tm']:.3f}")]
        for c, (tag, lab) in enumerate(labs):
            ax = axes[r, c]; ax.imshow(mpimg.imread(pngs[tag])); ax.set_axis_off(); ax.set_title(lab, fontsize=12)
    fig.suptitle(TITLES[cat], fontsize=15, y=0.995)
    fig.tight_layout(); f = os.path.join(out, f"H_examples_{cat}.png"); fig.savefig(f, dpi=150); plt.close(fig); print("WROTE", f)
