"""Score every generated synthetic template against its native and write the T2 template index.

The generation run stores `rewind_steps` per template, not TM. Training needs TM because the user's
sampling policy filters synthetic templates to **TM < 0.9** ("otherwise the task becomes trivial"),
and rewind is only a proxy for it: the rewind->TM curve was measured fold-INDEPENDENT on 5 chains,
which is not a basis for thresholding 82,734. TM is cheap to compute exactly, so compute it exactly.

Correspondence is fixed (the template was generated FROM this native, residue for residue), so this
is TM-score, not TM-align -- see openfold/utils/tm_score.py.

Output per shard: `<out>/index_<shard>.npz` with
    chains   (n,)      str
    rewind   (n, 64)   int16
    tm       (n, 64)   float32     TM to the native, normalized by the native
    length   (n,)      int32
Consolidate with --consolidate once every shard is done.

CPU-only by default so it can run on xeon-p8 while the 8 V100s finish generating.
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch

# ⛔ Loaded by PATH, not as `from openfold.utils.tm_score import ...`. `openfold/__init__.py` does
# `from . import resources`, and openfold/resources is UNTRACKED -- so a fresh clone or `git
# worktree add` cannot import the package at all, which is exactly how this script has to run on
# SuperCloud (where the templates live and where the only usable env is `protpardelle`).
# tm_score.py is pure torch with no openfold imports, so a path load is sufficient and portable.
_spec = importlib.util.spec_from_file_location(
    "tm_score", Path(__file__).resolve().parents[1] / "openfold/utils/tm_score.py")
_tms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tms)
REFERENCE_KWARGS, tm_score = _tms.REFERENCE_KWARGS, _tms.tm_score

CA = 1
# unchanged from the assert this replaced -- kept identical so only the FATALITY changes, not which
# chains are considered scoreable
MIN_NATIVE_CA = 5


def native_ca(pdb_path: Path, residue_index: np.ndarray):
    """Native CA coordinates ALIGNED to `residue_index`, plus a presence mask.

    ⛔ Not "CA lines in file order". A residue whose CA is unresolved in the deposited structure
    contributes no CA line, so the k-th CA line is not the k-th residue -- pairing by file order
    would shift the whole chain and silently score a misalignment. (Caught by the length assertion
    on 3ra5_A: 97 CA lines for L=98.) Keyed on the residue number instead, which
    `write_coords_to_pdb` wrote from this same `residue_index`.
    """
    by_resnum = {}
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            by_resnum[int(line[22:26])] = (
                float(line[30:38]), float(line[38:46]), float(line[46:54]))
    L = len(residue_index)
    xyz = np.zeros((L, 3), np.float32)
    present = np.zeros(L, bool)
    for i, r in enumerate(residue_index.astype(int)):
        if r in by_resnum:
            xyz[i] = by_resnum[r]
            present[i] = True
    return torch.from_numpy(xyz), torch.from_numpy(present.astype(np.float32))


def score_chain(npz_path: Path, pdb_path: Path, chunk: int, device: str):
    d = np.load(npz_path, allow_pickle=False)
    atom_mask = d["atom_mask"]                                   # (L, 37) bool
    L = atom_mask.shape[0]
    coords = d["coords"]                                         # (N, n_present, 3)
    N = coords.shape[0]

    full = np.zeros((N, L, 37, 3), np.float32)
    full[:, atom_mask] = coords
    tmpl = torch.from_numpy(full[:, :, CA, :])                   # (N, L, 3)
    tmpl_ca = torch.from_numpy(atom_mask[:, CA].astype(np.float32))

    ref, ref_ca = native_ca(pdb_path, d["residue_index"])
    # ⛔ Degenerate natives EXIST in the training set (4boh_M resolves 4 CA) and TM is undefined
    # there anyway -- d0 = 1.24*(L-15)^(1/3) - 1.8 is not real below L=15. Skip the chain and let
    # the caller record it; an assert here aborted a whole 1723-chain shard on one bad input.
    if int(ref_ca.sum()) < MIN_NATIVE_CA:
        return None
    both = ref_ca * tmpl_ca                                      # scored where BOTH have a CA

    tms = []
    for s in range(0, N, chunk):                                 # chunked to bound B*S memory
        b = tmpl[s:s + chunk].to(device)
        n = b.shape[0]
        tms.append(tm_score(
            b, ref[None].expand(n, L, 3).to(device),
            mask=both[None].expand(n, L).to(device),
            # normalized by the NATIVE's own coverage, so a template missing residues is penalized
            norm_mask=ref_ca[None].expand(n, L).to(device),
            **REFERENCE_KWARGS,
        ).cpu())
    return torch.cat(tms).numpy().astype(np.float32), d["rewind_steps"].astype(np.int16), L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates-root", required=True)
    ap.add_argument("--manifest", required=True, help="natives manifest.csv (chain -> pdb path)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--consolidate", action="store_true",
                    help="merge every index_*.npz in --out into index_all.npz and exit")
    ap.add_argument("--file-list", default=None,
                    help="read the chain npz paths from this file instead of globbing")
    ap.add_argument("--build-file-list", action="store_true",
                    help="glob once, write --file-list, and exit")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.consolidate:
        # ⛔ NOT glob("index_*.npz") -- that matches index_all.npz too, so a second consolidate
        # (e.g. after re-running one failed shard) would fold the previous merge back in and
        # silently DOUBLE every chain.
        parts = sorted(p for p in out.glob("index_[0-9]*.npz"))
        assert parts, f"no index_NNNN.npz under {out}"
        acc = {k: [] for k in ("chains", "rewind", "tm", "length")}
        skipped = []
        for p in parts:
            z = np.load(p, allow_pickle=False)
            for k in acc:
                acc[k].append(z[k])
            if "skipped" in z.files:
                skipped.extend(str(c) for c in z["skipped"])
        merged = {k: np.concatenate(v) for k, v in acc.items()}
        u, counts = np.unique(merged["chains"], return_counts=True)
        assert len(u) == len(merged["chains"]), \
            f"duplicate chains after merge: {u[counts > 1][:5].tolist()}"
        np.savez(out / "index_all.npz", **merged)
        if skipped:
            print(f"⚠️ {len(skipped)} chains SKIPPED as unscoreable "
                  f"(<{MIN_NATIVE_CA} native CA): {' '.join(skipped)}")
        tm = merged["tm"]
        elig = (tm < 0.9).sum(axis=1)
        print(f"consolidated {len(parts)} shards -> {len(merged['chains'])} chains")
        print(f"TM: min {tm.min():.3f}  median {np.median(tm):.3f}  max {tm.max():.3f}")
        print(f"templates with TM < 0.9: {100*(tm < 0.9).mean():.1f}% of all "
              f"({elig.min()}-{elig.max()} per chain, median {int(np.median(elig))})")
        print(f"chains with ZERO eligible template: {(elig == 0).sum()}")
        return

    # ⛔ Glob ONCE, into a file the shards read. Two reasons, and the second is a correctness bug,
    # not a performance nicety: (1) 48 processes each walking 1000 Lustre directories is 48x the
    # metadata load; (2) if the generation run is still WRITING, two processes globbing at
    # different moments see different file sets, so `files[shard::num_shards]` slices two different
    # lists and chains are silently missed or scored twice.
    if a.build_file_list:
        assert a.file_list, "--build-file-list needs --file-list"
        files = sorted(Path(a.templates_root).glob("shard*/*.npz"))
        Path(a.file_list).write_text("\n".join(str(f) for f in files) + "\n")
        print(f"wrote {a.file_list}  ({len(files)} chains)")
        return

    # ⛔ The manifest records ABSOLUTE paths from the host that produced it (the A6000), so on any
    # other host every `pdb` column is wrong. Same fallback the production driver uses: re-root the
    # shard dir + filename under the manifest's own directory. This already bit the production
    # launch once -- do not assume the recorded path resolves.
    manifest_dir = Path(a.manifest).parent
    pdb_of = {}
    with open(a.manifest) as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "ok":
                continue
            p = Path(r["pdb"])
            pdb_of[r["chain"]] = p if p.is_file() else manifest_dir / p.parent.name / p.name

    assert a.file_list, "pass --file-list (build it once with --build-file-list)"
    files = [Path(l) for l in Path(a.file_list).read_text().split() if l]
    mine = files[a.shard::a.num_shards]
    print(f"shard {a.shard}/{a.num_shards}: {len(mine)} of {len(files)} chains", flush=True)

    chains, rewinds, tms, lengths, skipped = [], [], [], [], []
    for i, f in enumerate(mine):
        chain = f.stem
        scored = score_chain(f, pdb_of[chain], a.chunk, a.device)
        if scored is None:
            skipped.append(chain)
            continue
        tm, rw, L = scored
        chains.append(chain)
        tms.append(tm)
        rewinds.append(rw)
        lengths.append(L)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(mine)}", flush=True)

    np.savez(
        out / f"index_{a.shard:04d}.npz",
        chains=np.array(chains), rewind=np.stack(rewinds),
        tm=np.stack(tms), length=np.array(lengths, np.int32),
        skipped=np.array(skipped),
    )
    print(f"wrote {out / f'index_{a.shard:04d}.npz'}  ({len(chains)} chains, "
          f"{len(skipped)} skipped: {' '.join(skipped) if skipped else '-'})")


if __name__ == "__main__":
    main()
