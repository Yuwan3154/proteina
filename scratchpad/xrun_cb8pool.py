"""Cross-run TopoRef comparison, EQUAL-LENGTH arm (user 2026-10-28: "CB-8-era equal-length pool") from the Protpardelle
band templates CB-8 trained on.

How v4 read a template (synth_index_confind.sbatch "Same pool, band, alias ... as synth_index_v4"; template_geometry_audit.py
docstring): ONE root, the symlink pool $S/t2_pool_auth (link name = AUTH id, via --chain-alias synth_alias/auth_key.tsv)
with band index t2_pool_auth_band.npz. A live pool link -> os.readlink (t2_extra / t2_regen entries); a deleted link (the
T2 entries, whose of_run tree is gone) -> <of_run T2 tree>/shard{crc32(partner)%1000}/<partner>.npz of the alias row's
use_pool seq_partner (the rule template_geometry_audit.py refs validated against the recorded targets, 2026-10-09). The
of_run T2 tree survives byte-exact on SuperCloud ~/pp1c_work/templates_band (400/400 sha256 sample, 2026-10-08).

  resolve  per chain of --chains: auth id, resolved template path, source; asserts the pool band row of the auth carries
           every v4 template row of the chain (rewind, slot, fp16 TM); writes resolve.tsv and sc_relpaths.txt (the T2
           files to rsync from SuperCloud).
  pick     builds a private pool $XPOOL/shard{crc32(auth)}/<auth>.npz of symlinks to the resolved files (T2 ones to the
           rsynced copy), then candidates = v4 rows with TM in [0.5, 0.9] (v4 eligibility + CB-8 tm_range) that pass the
           T8 / T8b clash gate (<= 4 non-bonded CA-CA pairs < 3.4 A, user 2026-10-13) so all three builders keep the row;
           one per chain, candidates sorted by slot, numpy default_rng(SEED) over the chains in sorted order. Writes
           band_equal.npz (pool band rows of the picked auths, every other rung's slot -1), band_control.npz (full pool
           band rows of the control chains), pairs_equal.tsv, chains_equal.txt, control_chains.txt, report.tsv.
Control chains: the first N_CONTROL (sorted) of the equal-arm chains that xrun_select.py marked base-round-only.
"""

import argparse
import os
import zlib
from collections import Counter

import numpy as np
import torch

from proteinfoundation.utils.precompute_synthetic_topology_index import CA_IDX, ca_clash_pairs

TM_WINDOW = (0.5, 0.9)          # v4 eligibility + CB-8 tm_range (data_config_tri_cb8synth_v5.json)
CLASH_MAX, CLASH_A = 4, 3.4     # user 2026-10-13 "gate at 3.4 A" + "<=4" (T8 / T8b builders)
SEED = 0                        # = topology_nonself_seed of every stageA pass
N_CONTROL = 5                   # user 2026-10-28 "rebuild a few production rows per model"
T2_ORIG = "/orcd/compute/so3/002/chenxi/of_run/pp1c_work/templates_band"  # the path v4 read (geom_audit refs.tsv)


def read_alias(path):
    """label_id -> auth_id (verbatim template_geometry_audit.read_alias; that script cannot be imported: it parses argv
    at module level)."""
    alias = {}
    with open(path) as fh:
        head = fh.readline().rstrip("\n").split("\t")
        il, ia = head.index("label_id"), head.index("auth_id")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > max(il, ia) and f[ia]:
                alias[f[il]] = f[ia]
    return alias


def shard_rel(name):
    return f"shard{zlib.crc32(name.encode()) % 1000:04d}/{name}.npz"


def v4_rows(index_path, want):
    idx = torch.load(index_path, map_location="cpu", weights_only=False, mmap=True)
    rows = {s: [] for s in want}
    natives = set()
    for s, n, t in zip(idx["ids"], idx["row_is_native"].tolist(), idx["row_tm"].float().tolist()):
        if n:
            natives.add(s)
            continue
        stem, rest = s.split("@rw")
        if stem in rows:
            rw, slot = rest.split("#")
            rows[stem].append((t, int(rw), int(slot)))
    return rows, natives


def load_band(path):
    z = np.load(path, allow_pickle=True)
    b = {k: z[k] for k in z.files}
    return b, {str(c): i for i, c in enumerate(b["chains"])}


def cmd_resolve(a):
    chains = [l.strip() for l in open(a.chains) if l.strip()]
    alias = read_alias(a.alias)
    partner = {}
    with open(a.alias) as fh:
        head = fh.readline().rstrip("\n").split("\t")
        ia, ip, iac = head.index("auth_id"), head.index("seq_partner"), head.index("action")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[iac] == "use_pool" and f[ia]:
                assert partner.get(f[ia], f[ip]) == f[ip], f"auth {f[ia]}: two seq_partners"
                partner[f[ia]] = f[ip]
    v4, natives = v4_rows(a.v4_index, set(chains))
    band, brow = load_band(a.pool_band)
    rep = Counter()
    out = []
    for s in chains:
        if s not in natives or not v4[s]:
            rep["not_v4_native_or_no_v4_rows"] += 1
            out.append((s, "", "", "none"))
            continue
        au = alias[s]
        link = os.path.join(a.pool, shard_rel(au))
        if os.path.islink(link):
            tgt = os.path.normpath(os.path.join(os.path.dirname(link), os.readlink(link)))
            assert os.path.exists(tgt), f"{s}: live pool link to a missing file {tgt}"
            src = "live_link"
        else:
            assert not os.path.exists(link), f"{link} is a regular file, not a pool link"
            tgt = os.path.join(T2_ORIG, shard_rel(partner[au]))
            src = "t2_partner"
        i = brow[au]
        rung = {int(k): (int(r), float(t)) for k, r, t in zip(band["slot"][i], band["rewind"][i], band["tm"][i]) if k >= 0}
        for t, rw, k in v4[s]:
            assert k in rung and rung[k][0] == rw and float(np.float16(rung[k][1])) == t, \
                f"{s}: v4 row rw{rw}#{k} tm {t} not the pool band rung {rung.get(k)}"
        rep[src] += 1
        out.append((s, au, tgt, src))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "resolve.tsv"), "w") as fh:
        fh.write("stem\tauth\ttarget\tsource\n" + "".join("\t".join(r) + "\n" for r in out))
    with open(os.path.join(a.out_dir, "sc_relpaths.txt"), "w") as fh:
        fh.write("".join(os.path.relpath(t, T2_ORIG) + "\n" for _, _, t, src in out if src == "t2_partner"))
    print(dict(rep))
    print("XRUN_RESOLVE_DONE")


def write_band(path, band, brow, keep):
    """keep: {auth: set of slots to keep, or None = all} -> the pool band rows of those auths, other slots -1."""
    auths = sorted(keep)
    sl = np.stack([band["slot"][brow[x]] for x in auths]) if auths else np.zeros((0, band["slot"].shape[1]), np.int64)
    sl = sl.copy()
    for j, x in enumerate(auths):
        if keep[x] is not None:
            sl[j] = np.where(np.isin(sl[j], list(keep[x])), sl[j], -1)
    rows = [brow[x] for x in auths]
    np.savez(path, chains=np.asarray(auths), tm=band["tm"][rows], rewind=band["rewind"][rows], slot=sl,
             min_tm=band["min_tm"], max_tm=band["max_tm"])


def cmd_pick(a):
    res = {}
    with open(os.path.join(a.out_dir, "resolve.tsv")) as fh:
        fh.readline()
        for line in fh:
            s, au, tgt, src = line.rstrip("\n").split("\t")
            res[s] = (au, tgt, src)
    base_only = {l.strip() for l in open(a.control_candidates) if l.strip()}
    v4, _ = v4_rows(a.v4_index, set(res))
    band, brow = load_band(a.pool_band)
    rep = Counter()
    rng = np.random.default_rng(SEED)
    picks, cand_n = {}, {}
    for s in sorted(res):
        au, tgt, src = res[s]
        if src == "none":
            rep["chains_not_in_v4"] += 1
            continue
        f = os.path.join(a.sc_copy, os.path.relpath(tgt, T2_ORIG)) if src == "t2_partner" else tgt
        assert os.path.exists(f), f"{s}: resolved template {f} missing (rsync?)"
        dst = os.path.join(a.xpool, shard_rel(au))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.islink(dst):
            os.symlink(f, dst)
        assert os.path.realpath(dst) == os.path.realpath(f), f"{dst} points elsewhere"
        cand = sorted((k, t, r) for t, r, k in v4[s] if TM_WINDOW[0] <= t <= TM_WINDOW[1])
        rep["v4_rows_in_window"] += len(cand)
        z = np.load(f)
        amask = z["atom_mask"].astype(bool)
        keep = []
        for k, t, r in cand:
            full = np.zeros((amask.shape[0], 37, 3), dtype=np.float32)
            full[amask] = z["coords"][k]
            if ca_clash_pairs(full[:, CA_IDX], amask[:, CA_IDX], CLASH_A) <= CLASH_MAX:
                keep.append((k, t, r))
        rep["window_rows_failing_clash_gate"] += len(cand) - len(keep)
        cand_n[s] = len(keep)
        if not keep:
            rep["chains_no_window_row" if not cand else "chains_all_window_rows_clash"] += 1
            continue
        k, t, r = keep[int(rng.integers(len(keep)))]
        picks[s] = (k, t, r, int(amask.shape[0]), au, src)
    eq = sorted(picks)
    assert eq, "no chain has an equal-length template"
    assert len({picks[s][4] for s in eq}) == len(eq), "two chains share an auth id"
    ctl = [s for s in eq if s in base_only][:N_CONTROL]
    assert len(ctl) == N_CONTROL, f"only {len(ctl)} control chains: {ctl}"
    write_band(os.path.join(a.out_dir, "band_equal.npz"), band, brow, {picks[s][4]: {picks[s][0]} for s in eq})
    write_band(os.path.join(a.out_dir, "band_control.npz"), band, brow, {picks[s][4]: None for s in ctl})
    for name, lst in (("chains_equal.txt", eq), ("control_chains.txt", ctl)):
        with open(os.path.join(a.out_dir, name), "w") as fh:
            fh.write("\n".join(lst) + "\n")
    with open(os.path.join(a.out_dir, "pairs_equal.tsv"), "w") as fh:
        fh.write("stem\tslot\ttm_native\trewind\tL_template\tauth\tsource\tn_candidates\n")
        for s in eq:
            k, t, r, L, au, src = picks[s]
            fh.write(f"{s}\t{k}\t{t:.4f}\t{r}\t{L}\t{au}\t{src}\t{cand_n[s]}\n")
    rep["chains"] = len(res)
    rep["chains_with_equal_length_template"] = len(eq)
    lines = [f"{k}\t{v}" for k, v in sorted(rep.items())]
    lines += [f"missing\t{','.join(s for s in sorted(res) if s not in picks)}", f"control\t{','.join(ctl)}"]
    open(os.path.join(a.out_dir, "report.tsv"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("XRUN_PICK_DONE")


def main():
    S = "/orcd/scratch/orcd/011/chenxiou"
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("resolve", "pick"))
    ap.add_argument("--chains", default="")
    ap.add_argument("--control-candidates", default="")
    ap.add_argument("--v4-index", default=f"{S}/synth_index_v4/topology_index_cb8_synth.pt")
    ap.add_argument("--alias", default=f"{S}/synth_alias/auth_key.tsv")
    ap.add_argument("--pool", default=f"{S}/t2_pool_auth")
    ap.add_argument("--pool-band", default=f"{S}/t2_pool_auth_band.npz")
    ap.add_argument("--sc-copy", default="", help="local copy of SuperCloud ~/pp1c_work/templates_band (pick)")
    ap.add_argument("--xpool", default="", help="private symlink pool to write (pick)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    {"resolve": cmd_resolve, "pick": cmd_pick}[a.cmd](a)


if __name__ == "__main__":
    main()
