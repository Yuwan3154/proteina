"""Cross-run TopoRef comparison, INDEL-arm template choice + the chain set (user 2026-10-28: the same chains for every arm).

Reads the stage A inventory (templates_t2_synth_v2.tsv, survivors.txt). Indel candidates of a chain: templates of the
t2_synth_v2 final selection (what the T8b index serves these chains) with rewind >= 150 (user 2026-10-27 / 10-28),
>= 1 'ins' or 'del' op (user 2026-10-28 "indel rule kept"), and the T8 / T8b builder gates (TM to the native in
[0.5, 0.9], <= 4 non-bonded CA-CA pairs < 3.4 A). Chain set = survivors with >= 1 candidate.
Pick: candidates sorted by (round, draw); numpy default_rng(SEED) over the chains in sorted order draws one index each.
TM is gated on the pack's own float tm_native (the builder's value), not the inventory TSV's rounded print.
Chains on the T8b override (regen) list are refused: T8b serves them from the T8b-only pack, not t2_synth_v2.
Also writes control_candidates.txt: chains of the set whose t2_synth_v2 templates are all round 'base' (so T8's pack-mode
cap-64 pick is the one T8 made before the 10-16/18 top-up appends); xrun_cb8pool.py picks the controls from them.
Outputs: sel_indel.npz (final_selection format: chain, first, count, round, draw), pairs_indel.tsv, chains.txt,
control_candidates.txt.
"""

import argparse
import csv
import os

import numpy as np

from proteinfoundation.utils.t2_pack_reader import TemplatePack

TM_WINDOW = (0.5, 0.9)          # every run's tm_range / index eligibility
CLASH_MAX = 4                   # user 2026-10-13 "<=4" (pairs < 3.4 A, counted by the inventory)
MIN_REWIND = 150                # user 2026-10-27 "just use indels from rewind 150 up"
SEED = 0                        # = topology_nonself_seed of every stageA pass (tri_gen_eval default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True, help="stage A out dir")
    ap.add_argument("--t2-pack", default="/orcd/compute/so3/002/chenxi/t2_synth_v2/pack")
    ap.add_argument("--override-chains", default="/orcd/scratch/orcd/011/chenxiou/t8b_template_list/template_chains.txt")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    tp = TemplatePack(a.t2_pack)
    override = {l.strip() for l in open(a.override_chains) if l.strip()}
    surv = sorted(l.strip() for l in open(os.path.join(a.inventory, "survivors.txt")) if l.strip())
    assert not set(surv) & override, f"survivors on the T8b override list: {sorted(set(surv) & override)}"
    missing = [s for s in surv if s not in tp]
    assert not missing, f"survivors not in the t2 pack: {missing}"
    raw_tm = {}
    for s in surv:
        for r in tp.rows(s):
            raw_tm[(s, tp.round_of(r), int(tp.col["draw"][r]))] = float(tp.col["tm_native"][r])
    cand = {s: [] for s in surv}
    rounds = {}
    with open(os.path.join(a.inventory, "templates_t2_synth_v2.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rounds.setdefault(r["stem"], set()).add(r["round"])
            t = raw_tm[(r["stem"], r["round"], int(r["draw"]))]
            ok = TM_WINDOW[0] <= t <= TM_WINDOW[1] and int(r["clash"]) <= CLASH_MAX
            edits = int(r["n_ins_ops"]) + int(r["n_del_ops"])
            if ok and r["in_final_selection"] == "1" and int(r["rewind"]) >= MIN_REWIND and edits > 0:
                cand[r["stem"]].append((r["round"], int(r["draw"]), t, int(r["rewind"]),
                                        int(r["L"]), int(r["n_ins_ops"]), int(r["n_del_ops"])))
    rng = np.random.default_rng(SEED)
    picks = {}
    for s in surv:
        c = sorted(cand[s])
        if c:
            picks[s] = (c[int(rng.integers(len(c)))], len(c))
    chains = sorted(picks)
    ctl = [s for s in chains if rounds.get(s) == {"base"}]
    print(f"survivors {len(surv)}; chains with an indel candidate {len(chains)}; base-round-only {len(ctl)}")
    os.makedirs(a.out_dir, exist_ok=True)
    for name, lst in (("chains.txt", chains), ("control_candidates.txt", ctl)):
        with open(os.path.join(a.out_dir, name), "w") as fh:
            fh.write("\n".join(lst) + "\n")
    with open(os.path.join(a.out_dir, "pairs_indel.tsv"), "w") as fh:
        fh.write("stem\tround\tdraw\ttm_native\trewind\tL_template\tn_ins_ops\tn_del_ops\tn_candidates\n")
        for s in chains:
            p, n = picks[s]
            fh.write(f"{s}\t{p[0]}\t{p[1]}\t{p[2]:.6f}\t" + "\t".join(map(str, p[3:])) + f"\t{n}\n")
    np.savez(os.path.join(a.out_dir, "sel_indel.npz"), chain=np.asarray(chains),
             first=np.arange(len(chains), dtype=np.int64), count=np.ones(len(chains), dtype=np.int32),
             round=np.asarray([picks[s][0][0] for s in chains]),
             draw=np.asarray([picks[s][0][1] for s in chains], dtype=np.int32))
    print("XRUN_SELECT_DONE")


if __name__ == "__main__":
    main()
