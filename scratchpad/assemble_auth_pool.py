"""Assemble ONE auth-keyed template pool from the three source trees (D23/D24).

Input is `auth_key.tsv` (label_id, auth_id, seq_status, seq_partner, source, action), the product of
cross-tabbing the mmCIF mapper against the sequence alias. This applies the translation ONCE, at
write time, so every consumer downstream sees a single convention and never has to guess.

⛔ The source tree decides when the mmCIF gave no auth id: a file under the T2 tree is auth-keyed by
construction, one under our label-keyed backfill is not and cannot be admitted without an auth id.
⛔ crc32 for the shard, never builtin hash().
⛔ Never writes into the T2 trees -- round 2 is generating there. Entries are SYMLINKS: the pool is an
intermediate (the index bakes its features in), so 47 GB of copies would buy nothing.
⛔ Every chain that cannot be placed is reported with a reason; nothing is silently skipped.

usage: assemble_auth_pool.py --auth-key auth_key.tsv --dest <pool> \
           --t2-tree <templates_band> --extra-tree <t2_extra/templates> --regen-tree <t2_regen/templates>
"""

import argparse
import collections
import pathlib
import zlib

T2_ROOTS = ("/orcd/compute/so3/002/chenxi/of_run/pp1c_work",)


def find_npz(chain, trees):
    """(path, tree_label) for the first tree holding this chain, or (None, None)."""
    shard = f"shard{zlib.crc32(chain.encode()) % 1000:04d}"
    for label, tree in trees:
        if tree is None:
            continue
        for p in (tree / shard / f"{chain}.npz", tree / f"{chain}.npz"):
            if p.exists():
                return p, label
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth-key", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--t2-tree", required=True)
    ap.add_argument("--extra-tree", default="")
    ap.add_argument("--regen-tree", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    dest = pathlib.Path(a.dest).resolve()
    for root in T2_ROOTS:
        if str(dest).startswith(str(pathlib.Path(root).resolve())):
            raise SystemExit(f"refusing to write inside the T2 root {root}")

    trees = [("t2", pathlib.Path(a.t2_tree))]
    if a.extra_tree:
        trees.append(("extra", pathlib.Path(a.extra_tree)))
    if a.regen_tree:
        trees.append(("regen", pathlib.Path(a.regen_tree)))

    counts = collections.Counter()
    unplaced = []
    manifest = []
    with open(a.auth_key) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        ix = {c: i for i, c in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            label_id = f[ix["label_id"]]
            auth_id = f[ix["auth_id"]]
            sst = f[ix["seq_status"]]
            partner = f[ix["seq_partner"]]
            action = f[ix["action"]]

            if action == "DROP":
                counts["drop_no_auth_no_match"] += 1
                unplaced.append((label_id, "no auth id and no matching npz"))
                continue

            src_name = partner if (sst == "remap" and partner) else label_id
            # ⛔ A `regenerate` chain STILL has an npz in the T2 tree under this stem -- the
            # mis-joined one. Searching the trees in order would return it and place a DIFFERENT
            # POLYMER under the correct auth name: worse than the original bug, because the result
            # looks right. Regenerated chains may come only from the regen tree.
            if action == "regenerate":
                search = [(l, t) for l, t in trees if l == "regen"]
                if not search:
                    counts["regen_tree_not_supplied"] += 1
                    unplaced.append((label_id, "action=regenerate but no --regen-tree given"))
                    continue
            else:
                search = trees
            src, tree_label = find_npz(src_name, search)
            if src is None:
                counts[f"missing_npz_{action}"] += 1
                unplaced.append((label_id, f"no npz for {src_name} in any tree ({action})"))
                continue

            key = auth_id
            if not key:
                # no mmCIF auth id: admissible ONLY if the file is already auth-keyed, i.e. from T2
                if tree_label == "t2":
                    key = src_name
                    counts["keyed_from_t2_name"] += 1
                else:
                    counts["drop_label_keyed_no_auth"] += 1
                    unplaced.append((label_id, f"npz from {tree_label} tree is label-keyed and no auth id available"))
                    continue

            out = dest / f"shard{zlib.crc32(key.encode()) % 1000:04d}" / f"{key}.npz"
            manifest.append((key, label_id, tree_label, str(src)))
            counts[f"placed_{tree_label}"] += 1
            if a.dry_run:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.is_symlink() or out.exists():
                counts["already_present"] += 1
                continue
            out.symlink_to(src)

    print(f"{'DRY RUN -- ' if a.dry_run else ''}pool: {dest}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28} {v:>7}")
    print(f"  {'TOTAL placed':28} {sum(v for k, v in counts.items() if k.startswith('placed')):>7}")

    if not a.dry_run:
        with open(dest.parent / "pool_manifest.tsv", "w") as fh:
            fh.write("auth_id\tlabel_id\tsource_tree\tsource_path\n")
            for row in manifest:
                fh.write("\t".join(row) + "\n")
        with open(dest.parent / "pool_unplaced.tsv", "w") as fh:
            fh.write("label_id\treason\n")
            for cid, why in unplaced:
                fh.write(f"{cid}\t{why}\n")
    print(f"  unplaced: {len(unplaced)} (see pool_unplaced.tsv)")
    for cid, why in unplaced[:5]:
        print(f"    {cid}: {why}")


if __name__ == "__main__":
    main()
