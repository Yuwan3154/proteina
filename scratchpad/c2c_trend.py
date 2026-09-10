"""Recent train-loss trend for c2c_nolddt, with the sampled noise level alongside it.

A single logged step of a diffusion loss is not comparable to another: the value depends on the
sigma drawn for that step. So a 0.65 -> 4.85 jump between two heartbeats is only evidence of
divergence if it is NOT explained by sigma. This prints both, plus a rolling median that is robust
to the per-step sigma draw.
"""

import argparse

import numpy as np
import wandb

ap = argparse.ArgumentParser()
ap.add_argument("--entity", default="DP_CO_AFdiffusion")   # c2c launcher sets no entity
ap.add_argument("--project", default="contact2coord")
ap.add_argument("--run", default="c2c_nolddt")
ap.add_argument("--last", type=int, default=40)
a = ap.parse_args()

api = wandb.Api()
runs = [r for r in api.runs(f"{a.entity}/{a.project}") if r.name == a.run]
assert runs, f"no run named {a.run} in {a.entity}/{a.project}"
r = sorted(runs, key=lambda x: str(x.created_at))[-1]
print(f"run {r.name} ({r.id}) state={r.state} created={str(r.created_at)[:16]}")

# NOT history(keys=...): if ANY requested key is absent wandb returns an EMPTY frame, which reads
# as "the run logged nothing" rather than "I asked for the wrong name".
df = r.history(samples=100000, pandas=True)
print("columns:", [c for c in df.columns if not c.startswith("_")])


def col(*cands):
    for c in cands:
        if c in df.columns and df[c].notna().any():
            return c
    return None


s, l = col("trainer/global_step"), col("train/loss_step", "train/loss")
rm, sg = col("train/rmsd_step", "train/rmsd"), col("train/sigma_step", "train/sigma")
assert s and l, f"missing step/loss columns; have {list(df.columns)}"
d = df[[c for c in (s, l, rm, sg) if c]].dropna(subset=[s, l]).sort_values(s)
assert len(d) >= 20, f"VACUOUS: only {len(d)} logged points"

print(f"logged points: {len(d)}   step range {int(d[s].min())}-{int(d[s].max())}")
tail = d.tail(a.last)
print(f"\nlast {len(tail)} logged steps:")
print(f"{'step':>7} {'loss':>9} {'rmsd':>9} {'sigma':>10}")
for _, row in tail.iterrows():
    print(f"{int(row[s]):>7} {row[l]:>9.4f} {row[rm]:>9.3f} " if rm else f"{int(row[s]):>7} {row[l]:>9.4f}",
          f"{row[sg]:>10.3f}" if sg else "")

for name, c in (("loss", l), ("rmsd", rm)):
    if c is None:
        continue
    v = d[c].to_numpy()
    for w in (50, 200):
        if len(v) >= 2 * w:
            print(f"{name}: median of last {w} = {np.median(v[-w:]):.4f}   "
                  f"vs the {w} before that = {np.median(v[-2 * w:-w]):.4f}")
if sg is not None:
    hi = d[d[sg] > d[sg].median()]
    lo = d[d[sg] <= d[sg].median()]
    print(f"\nsigma split: median loss at HIGH sigma = {hi[l].median():.4f} "
          f"(n={len(hi)}), at LOW sigma = {lo[l].median():.4f} (n={len(lo)})")
    # ⛔ Derive the verdict, never print a canned one: the first version of this script ended with
    # an unconditional "not divergence" line, which was FALSE for the very spike it was written to
    # explain (step 1149: loss 4.85 at sigma 19.0, while sigma 37.3 had produced loss 0.96).
    ratio = hi[l].median() / max(lo[l].median(), 1e-9)
    last, base = float(d[l].iloc[-1]), float(d[l].iloc[-11:-1].median())
    print(f"high/low sigma loss ratio = {ratio:.2f} "
          f"({'sigma explains the spread' if ratio > 2 else 'sigma does NOT explain the spread'})")
    print(f"last logged loss {last:.4f} vs median of the 10 before it {base:.4f} "
          f"=> {'SPIKE, not explained by sigma' if last > 3 * base and ratio <= 2 else 'within recent range'}")
