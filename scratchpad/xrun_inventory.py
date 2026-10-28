"""Cross-run TopoRef comparison (CB-8 / T8 / T8b), stage A inventory (Engaging CPU). Read-only on every input.

Chain filter (user 2026-10-28): the P2 247-chain val list -> L <= 384 -> native + >= 1 in-window (TM 0.5-0.9) template
row in BOTH the T8 and the T8b index (and in both native packs) -> status 'unchanged' in the T8b re-process table.
Counts are printed at each step. Survivors are also checked for T8 == T8b == CB-8 native identity (torch.equal).

Templates of the survivors, from the t2_synth_v2 pack (every round), one TSV row each: edit ops (ins / del counted
from the ops plan AND from orig_idx, so a disagreement is visible), TM to the native, rewind, the CA clash count the
T8 / T8b builders gate on (> 4 non-bonded CA-CA pairs < 3.4 A, user 2026-10-13), and membership of the t2_synth_v2
final selection (what the T8b index serves these chains; override-listed chains are flagged).

Checkpoints: global_step / epoch of every path given with --ckpt.
Usage: python scratchpad/xrun_inventory.py --out-dir D --ckpt P [--ckpt P ...] (paths below default to the D10 layout)
"""

import argparse
import io
import json
import os
from collections import Counter

import numpy as np
import torch

from proteinfoundation.utils.precompute_synthetic_topology_index import CA_IDX, ca_clash_pairs
from proteinfoundation.utils.t2_pack_reader import FinalSet

S = "/orcd/scratch/orcd/011/chenxiou"
C = "/orcd/compute/so3/002/chenxi"
TM_WINDOW = (0.5, 0.9)          # every run's tm_range (data configs) and index eligibility
CLASH_MAX, CLASH_A = 4, 3.4     # user 2026-10-13 "gate at 3.4 A" + "<=4"
MAX_L = 384                     # user 2026-10-28; = CB-8's PaddingTransform max_size
MIN_REWIND = 150                # user 2026-10-27 "just use indels from rewind 150 up" (+ 10-28 indel arm "rewind >= 150")


def index_summary(path):
    idx = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    ids = list(idx["ids"])
    native = idx["row_is_native"].tolist()
    tm = idx["row_tm"].float().tolist()
    out = {}
    cur = None
    for i, (s, nat) in enumerate(zip(ids, native)):
        if nat:
            assert s not in out, f"{path}: duplicate native {s}"
            cur = s
            out[s] = [0, 0]
        else:
            assert s.startswith(cur + "@"), f"{path}: row {s} not under native {cur}"
            out[cur][0] += 1
            out[cur][1] += TM_WINDOW[0] <= tm[i] <= TM_WINDOW[1]
    return out


class NativePack:
    def __init__(self, path):
        z = np.load(path + ".idx.npz", allow_pickle=True)
        self.ent = {str(s): (int(o), int(l)) for s, o, l in zip(z["stems"], z["offsets"], z["lengths"])}
        self.fh = open(path, "rb")

    def get(self, stem):
        if stem not in self.ent:
            return None
        o, l = self.ent[stem]
        self.fh.seek(o)
        return torch.load(io.BytesIO(self.fh.read(l)), map_location="cpu", weights_only=False)


def same(a, b, key):
    x, y = getattr(a, key, None), getattr(b, key, None)
    if x is None or y is None:
        return f"{key}:absent({x is None},{y is None})"
    if isinstance(x, torch.Tensor):
        return "" if (x.shape == y.shape and torch.equal(x, y)) else f"{key}:differs"
    return "" if x == y else f"{key}:differs"


def ckpt_info(path):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    assert isinstance(ck, dict), path
    keys = sorted(ck.keys())
    return {"path": path, "global_step": ck.get("global_step"), "epoch": ck.get("epoch"), "keys": keys,
            "mtime": os.path.getmtime(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", default=f"{S}/t8_alignbias/val1pc.txt")
    ap.add_argument("--status", default=f"{S}/t8b_seqres/reproc_status_final_padfix.tsv")
    ap.add_argument("--t8-index", default=f"{C}/t8_index/topology_index_confind_synth_t8.pt")
    ap.add_argument("--t8b-index", default=f"{C}/t8b_index_v2/topology_index_confind_synth_t8b.pt")
    ap.add_argument("--cb8-index", default=f"{S}/synth_index_v4/topology_index_cb8_synth.pt")
    ap.add_argument("--t8-pack", default=f"{C}/proteina_pack/tri_t8_512.pack")
    ap.add_argument("--t8b-pack", default=f"{C}/proteina_pack/tri_t8b_512_v2.pack")
    ap.add_argument("--cb8-pack", default=f"{C}/proteina_pack/tri_v2.pack")
    ap.add_argument("--t2-pack", default=f"{C}/t2_synth_v2/pack")
    ap.add_argument("--override-chains", default=f"{S}/t8b_template_list/template_chains.txt")
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rep = {}

    chains = [l.strip() for l in open(a.chains) if l.strip()]
    assert len(chains) == len(set(chains)), "duplicate chains in the list"
    rep["n_list"] = len(chains)
    status = {}
    with open(a.status) as fh:
        head = fh.readline().rstrip("\n").split("\t")
        i_st, i_stem = head.index("status"), head.index("stem")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            assert f[i_stem] not in status, f"duplicate stem {f[i_stem]} in {a.status}"
            status[f[i_stem]] = f[i_st]
    t8i, t8bi, cb8i = index_summary(a.t8_index), index_summary(a.t8b_index), index_summary(a.cb8_index)
    p8, p8b, pcb = NativePack(a.t8_pack), NativePack(a.t8b_pack), NativePack(a.cb8_pack)
    override = {l.strip() for l in open(a.override_chains) if l.strip()}

    rows = []
    for s in chains:
        g8b, g8 = p8b.get(s), p8.get(s)
        L8b = None if g8b is None else int(g8b.coords.shape[0])
        L8 = None if g8 is None else int(g8.coords.shape[0])
        rows.append(dict(stem=s, L_t8b=L8b, L_t8=L8, in_t8_pack=g8 is not None, in_t8b_pack=g8b is not None,
                         t8_native=s in t8i, t8_tpl=t8i.get(s, [0, 0])[0], t8_tpl_win=t8i.get(s, [0, 0])[1],
                         t8b_native=s in t8bi, t8b_tpl=t8bi.get(s, [0, 0])[0], t8b_tpl_win=t8bi.get(s, [0, 0])[1],
                         cb8_native=s in cb8i, cb8_tpl_win=cb8i.get(s, [0, 0])[1],
                         status=status.get(s, "absent"), in_override=s in override))
    # L: the T8b native (the T8b data); T8 must agree on survivors (checked below)
    f1 = [r for r in rows if r["L_t8b"] is not None and r["L_t8b"] <= MAX_L]
    f2 = [r for r in f1 if r["in_t8_pack"] and r["in_t8b_pack"] and r["t8_native"] and r["t8b_native"]
          and r["t8_tpl_win"] > 0 and r["t8b_tpl_win"] > 0]
    f3 = [r for r in f2 if r["status"] == "unchanged"]
    rep["filter_counts"] = {"list": len(rows), "L<=384 (T8b native)": len(f1),
                            "native + in-window template in T8 AND T8b index, in both packs": len(f2),
                            "status unchanged": len(f3)}
    rep["dropped_by_presence"] = {r["stem"]: {k: r[k] for k in ("in_t8_pack", "in_t8b_pack", "t8_native", "t8b_native",
                                                                 "t8_tpl_win", "t8b_tpl_win")}
                                  for r in f1 if r not in f2}
    rep["dropped_by_status"] = Counter(r["status"] for r in f2 if r not in f3)
    rep["status_of_list"] = Counter(r["status"] for r in rows)
    surv = [r["stem"] for r in f3]
    rep["survivors_in_override_list"] = sorted(set(surv) & override)
    rep["survivors_cb8_native"] = sum(r["cb8_native"] for r in f3)
    rep["survivors_cb8_tpl_win>0"] = sum(r["cb8_tpl_win"] > 0 for r in f3)

    # native identity across the three runs' data (the 'unchanged' status, checked on the tensors themselves)
    ident = Counter()
    for s in surv:
        g8, g8b, gcb = p8.get(s), p8b.get(s), pcb.get(s)
        d = [same(g8, g8b, k) for k in ("coords", "coord_mask", "residue_type", "contact_map_confind")]
        ident["t8_vs_t8b:" + (",".join(x for x in d if x) or "equal")] += 1
        if gcb is None:
            ident["cb8_pack:absent"] += 1
        else:
            d = [same(g8, gcb, k) for k in ("coords", "coord_mask", "residue_type")]
            ident["t8_vs_cb8:" + (",".join(x for x in d if x) or "equal")] += 1
    rep["native_identity"] = ident
    with open(os.path.join(a.out_dir, "chains_filter.tsv"), "w") as fh:
        keys = list(rows[0].keys())
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[k]) for k in keys) + "\n")
    with open(os.path.join(a.out_dir, "survivors.txt"), "w") as fh:
        fh.write("\n".join(surv) + "\n")

    # templates of the survivors
    fs = FinalSet(a.t2_pack)
    tp = fs.tp
    col = tp.col
    op_types = Counter()
    per = {}
    L_nat = {r["stem"]: r["L_t8b"] for r in f3}
    with open(os.path.join(a.out_dir, "templates_t2_synth_v2.tsv"), "w") as fh:
        fh.write("stem\tround\tdraw\tL\tL_native\ttm_native\trewind\tarm\tsource\tn_ins_ops\tn_del_ops\t"
                 "n_ins_orig\tn_del_orig\tclash\tin_final_selection\n")
        for s in surv:
            if s not in tp:
                per[s] = None
                continue
            fin = set(fs.keys(s)) if s in fs else set()
            c = Counter()
            seen = set()
            for r in tp.rows(s):
                rec = tp._read(r)
                seen.add((rec["round"], rec["draw"]))
                ops = rec["ops"]
                for o in ops:
                    op_types[str(o[0])] += 1
                n_ins = sum(1 for o in ops if o[0] == "ins")
                n_del = sum(1 for o in ops if o[0] == "del")
                oi = rec["orig_idx"]
                assert rec["L_native"] == L_nat[s], (s, rec["round"], rec["draw"], rec["L_native"], L_nat[s])
                pos = oi[oi >= 0]
                assert (oi >= -1).all() and (pos < rec["L_native"]).all() and (np.diff(pos) > 0).all(), (s, rec["draw"])
                n_ins_o = int((oi < 0).sum())
                n_del_o = int(rec["L_native"] - (oi >= 0).sum())
                full = np.zeros((rec["atom_mask"].shape[0], 37, 3), dtype=np.float32)
                full[rec["atom_mask"]] = rec["coords"]
                clash = ca_clash_pairs(full[:, CA_IDX], rec["atom_mask"][:, CA_IDX], CLASH_A)
                infin = (rec["round"], rec["draw"]) in fin
                fh.write(f"{s}\t{rec['round']}\t{rec['draw']}\t{len(oi)}\t{rec['L_native']}\t{rec['tm_native']:.4f}\t"
                         f"{rec['rewind']}\t{rec['arm']}\t{rec['source']}\t{n_ins}\t{n_del}\t{n_ins_o}\t{n_del_o}\t"
                         f"{clash}\t{int(infin)}\n")
                ok = TM_WINDOW[0] <= rec["tm_native"] <= TM_WINDOW[1] and clash <= CLASH_MAX
                edit_free = n_ins == 0 and n_del == 0
                c["templates"] += 1
                c["edit_free"] += edit_free
                c["edit_free_ok"] += edit_free and ok
                c["ops_vs_orig_disagree"] += (n_ins > 0) != (n_ins_o > 0) or (n_del > 0) != (n_del_o > 0)
                c["final"] += infin
                c["final_rw150_ok"] += infin and rec["rewind"] >= MIN_REWIND and ok
                c["final_rw150_ok_indel"] += infin and rec["rewind"] >= MIN_REWIND and ok and not edit_free
            assert fin <= seen, (s, fin - seen)
            per[s] = c
    rep["op_types"] = op_types
    rep["survivors_not_in_final_selection"] = sorted(s for s in surv if s not in fs)
    rep["survivors_not_in_t2_pack"] = sorted(s for s in surv if per[s] is None)
    have = {k: sum(1 for s in surv if per[s] and per[s][k] > 0) for k in
            ("edit_free", "edit_free_ok", "final", "final_rw150_ok", "final_rw150_ok_indel")}
    rep["chains_with_at_least_one"] = have
    rep["chains_with_both_arms"] = sum(1 for s in surv if per[s] and per[s]["edit_free_ok"] > 0
                                       and per[s]["final_rw150_ok_indel"] > 0)
    tot = Counter()
    for s in surv:
        if per[s]:
            tot.update(per[s])
    rep["template_totals"] = tot
    rep["checkpoints"] = [ckpt_info(p) for p in a.ckpt]
    with open(os.path.join(a.out_dir, "inventory.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))
    print(json.dumps(rep, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)))
    print("XRUN_INVENTORY_DONE")


if __name__ == "__main__":
    main()
