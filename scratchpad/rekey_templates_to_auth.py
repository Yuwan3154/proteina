"""Re-key a label-keyed template tree onto the AUTH id namespace (D23/D24).

The T2 pool is keyed `<pdbid>_<auth_asym_id>`; our B4 backfill wrote `<pdbid>_<label_asym_id>`,
because it generated from proteina's natives. A pool spanning BOTH conventions is the original bug in
a harder-to-see form: the same string names different chains depending on which tree answers first.

So the pool is made single-convention (auth), and the alias is applied HERE -- once, at write time --
rather than guessed at read time.

⛔ crc32 for the shard, never builtin hash(): Python randomises string hashing per process, so a
hash()-derived shard is not reproducible across runs.
⛔ Nothing is written unless the alias gives an auth id AND the npz residue count matches the alias's
recorded label length. Two names for one polymer must agree exactly; a disagreement means the alias
and the file disagree about what this chain IS, and guessing there is what caused the bug.
⛔ Never writes into the T2 trees -- round 2 is generating into them. --dest must be our own root.

usage: rekey_templates_to_auth.py --src <label-keyed tree> --dest <auth-keyed tree>
                                  --alias t2_chain_alias.npz [--link] [--dry-run]
"""

import argparse
import collections
import pathlib
import zlib

import numpy as np

T2_ROOTS = ("/orcd/compute/so3/002/chenxi/of_run/pp1c_work",)


def load_alias(path):
    """label_id -> (auth_id, n_res_label). Accepts the npz or the csv form."""
    p = pathlib.Path(path)
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=True)
        label = [str(x) for x in z["label_id"]]
        auth = [str(x) for x in z["t2_auth_id"]]
        nres = z["n_res_label"] if "n_res_label" in z else [None] * len(label)
        return {l: (a, int(n) if n is not None else None) for l, a, n in zip(label, auth, nres)}
    out = {}
    with open(p) as fh:
        header = fh.readline().rstrip("\n").split(",")
        il, ia = header.index("label_id"), header.index("t2_auth_id")
        inr = header.index("n_res_label") if "n_res_label" in header else None
        for line in fh:
            f = line.rstrip("\n").split(",")
            out[f[il]] = (f[ia], int(f[inr]) if inr is not None and f[inr] else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--link", action="store_true", help="hard-link instead of copying")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    dest = pathlib.Path(a.dest).resolve()
    for root in T2_ROOTS:
        if str(dest).startswith(str(pathlib.Path(root).resolve())):
            raise SystemExit(f"refusing to write into the T2 root {root}: round 2 is generating there")

    alias = load_alias(a.alias)
    src = pathlib.Path(a.src)
    files = sorted(src.rglob("*.npz"))
    print(f"alias entries: {len(alias)} | source npz: {len(files)} under {src}")

    counts = collections.Counter()
    unresolved = []
    for f in files:
        label_id = f.stem
        hit = alias.get(label_id)
        if hit is None or not hit[0]:
            counts["no_alias_entry"] += 1
            unresolved.append((label_id, "no_alias_entry"))
            continue
        auth_id, n_label = hit
        z = np.load(f)
        n_file = int(z["atom_mask"].shape[0]) if "atom_mask" in z else None
        if n_label is not None and n_file is not None and n_file != n_label:
            # the alias and the file disagree about what this chain is -- do not guess
            counts["length_disagreement"] += 1
            unresolved.append((label_id, f"alias says {n_label} res, npz has {n_file}"))
            continue
        out = dest / f"shard{zlib.crc32(auth_id.encode()) % 1000:04d}" / f"{auth_id}.npz"
        counts["rekeyed"] += 1
        if a.dry_run:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            counts["already_present"] += 1
            continue
        if a.link:
            out.hardlink_to(f)
        else:
            out.write_bytes(f.read_bytes())

    print(f"\n{'DRY RUN -- ' if a.dry_run else ''}results:")
    for k, v in sorted(counts.items()):
        print(f"  {k:22} {v}")
    if unresolved:
        report = dest.parent / "rekey_unresolved.tsv"
        if not a.dry_run:
            report.parent.mkdir(parents=True, exist_ok=True)
            with open(report, "w") as fh:
                fh.write("label_id\treason\n")
                for cid, why in unresolved:
                    fh.write(f"{cid}\t{why}\n")
        print(f"  {len(unresolved)} unresolved -> {report}"
              f"{' (not written, dry run)' if a.dry_run else ''}")
        for cid, why in unresolved[:5]:
            print(f"    {cid}: {why}")


if __name__ == "__main__":
    main()
