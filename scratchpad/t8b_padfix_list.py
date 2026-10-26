"""T8b June pad-fix (user 2026-10-26 "Yes" to recompute): native ConFind maps of UNCHANGED T8b chains that were padded
inside a mixed-length batch of the June winner run (Frame2ConFind _make_s_init: dihedrals over the padded batch without
the mask -> wrong last-residue psi/omega, artifact in row/col L-1). CPU only, no torch.

June batch rule (proteinfoundation/utils/precompute_frame2confind_maps.py run_precompute): pending sorted by L desc;
a batch = the next bs items that LOAD (a load failure -- status failed, infer_sec 0.0 -- is written immediately, before
that batch's rows, and does not take a batch slot); after inference every batch row is written (processed, or failed with
the batch's infer_sec if its save raised) in as_completed order, before the next batch is built. So: drop load-failure
rows, chunk the rest into consecutive groups of bs; only the final group may be shorter. Checked: one infer_sec per
group, adjacent groups differ, no group longer than the previous group's shortest chain. Padded = L < its group's max L.

Exact attribution (pdb_train held some stems twice -- a flat processed/<stem>.pt and a bucket copy -- and the CSV keys on
the basename): the June pending list is REPLAYED from the length cache winner_rerun.sh fed to the precompute (rows with
L <= --max-len, stable sort by L desc, load failures skip their slot, chunks of bs) and asserted equal to the CSV batches
(same (basename, L) multiset per batch). Each T8b stem is then mapped to the copy T8b reads, _processed_path_sharded(
<data-dir>/processed, stem, <data-dir>/shard_manifest.json), compared to the cache LEXICALLY as a path relative to
<data-dir>/processed (the cache spells <data-dir>/processed/...; processed has since become a symlink and the flat
copies are deleted, so nothing is resolved on disk); that path must be in the cache exactly once, and its batch decides
padding. n_june_rows (CSV rows with the stem's basename) is information only.

list:  python t8b_padfix_list.py list --june-csv band1_15356272.csv --length-cache full_pdb_train_band1.json \
           --data-dir DATA_PATH/pdb_train --max-len 512 --bs 8 --status-tsv reproc_status_final.tsv \
           --template-list template_chains.txt --eligible eligible_ids_tm0.5-0.9.txt --val-fixed val_fixed32.txt \
           --n-parts 60 --out-tsv padfix_list.tsv
table: python t8b_padfix_list.py table --status-tsv reproc_status_final.tsv --padfix-list padfix_list.tsv \
           --check-tsv check_padfix.tsv --dropped-tsv dropped_padfix.tsv --out reproc_status_final_padfix.tsv
       = the old table with the CHECKED pad-fix stems as status changed, category confind_pad, fields contact_map_confind;
       every other byte identical (round-trip asserted). check + dropped (t8b_confind_compare.py --padfix-check, written
       only on a full pass) must partition the list's stems; dropped stems (map identical) stay unchanged.
"""
import argparse
import collections
import csv
import io
import json
import os
import pathlib
import statistics as st

from proteinfoundation.datasets.pdb_data import _processed_path_sharded

LIST_COLS = ["stem", "L_june", "batch", "batch_n", "batch_max_L", "pad", "n_june_rows", "L_old", "L_new", "in_template_list",
             "index_part", "eligible", "val_fixed"]
PAD_CATEGORY, PAD_FIELDS = "confind_pad", "contact_map_confind"


def is_load_failure(r):
    return r["status"] == "failed" and float(r["infer_sec"]) == 0.0


def june_batches(csv_path, bs):
    """(batches = [[row, ...]] in CSV order, load-failure rows, save-failure rows)."""
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    # pdb_train holds some stems twice (flat + bucket copies): the CSV 'file' column is a basename, so a stem can have 2 rows
    mult = collections.Counter(collections.Counter(r["file"] for r in rows).values())
    print(f"[june] basename multiplicity {dict(sorted(mult.items()))}", flush=True)
    assert {r["status"] for r in rows} <= {"processed", "failed"}, {r["status"] for r in rows}
    load_failed = [r for r in rows if is_load_failure(r)]
    in_batch = [r for r in rows if not is_load_failure(r)]
    save_failed = [r for r in in_batch if r["status"] == "failed"]
    batches = [in_batch[i:i + bs] for i in range(0, len(in_batch), bs)]
    for k, b in enumerate(batches):
        assert len({r["infer_sec"] for r in b}) == 1, f"group {k} ({b[0]['file']}): infer_sec not uniform -> rule broken"
        if k:
            prev = batches[k - 1]
            assert prev[0]["infer_sec"] != b[0]["infer_sec"], \
                f"groups {k - 1}/{k} share infer_sec {b[0]['infer_sec']}: boundary not confirmable"
            assert max(int(r["length"]) for r in b) <= min(int(r["length"]) for r in prev), \
                f"group {k} has a chain longer than group {k - 1}'s shortest (pending is sorted by L desc)"
    return batches, load_failed, save_failed


def replay(cache_path, root, max_len, bs, batches, load_failed):
    """June pending replayed from the length cache -> {path relative to root: batch index}; asserted against the CSV."""
    with open(cache_path) as fh:
        cache = json.load(fh)
    root = os.path.normpath(root)
    rel = []
    for p, L, _ in cache:
        q = os.path.normpath(p)
        assert q.startswith(root + "/"), f"cache path {p} is not under {root}"
        rel.append((os.path.relpath(q, root), int(L)))
    assert len({r for r, _ in rel}) == len(rel), "cache paths are not unique after normalisation"
    pending = [x for x in rel if x[1] <= max_len]
    pending.sort(key=lambda x: x[1], reverse=True)  # stable, = the precompute's pending.sort(reverse=True)
    lf = collections.Counter(r["file"] for r in load_failed)
    n_by_name = collections.Counter(os.path.basename(r) for r, _ in pending)
    amb = [f for f in lf if n_by_name[f] != 1 or lf[f] != 1]
    assert not amb, f"load-failure basenames not unique in pending (which copy failed is unknown: ask): {amb[:5]}"
    slots = [x for x in pending if os.path.basename(x[0]) not in lf]
    groups = [slots[i:i + bs] for i in range(0, len(slots), bs)]
    print(f"[replay] {cache_path}: {len(rel)} paths, {len(pending)} with L <= {max_len}, {len(lf)} load failures skipped, "
          f"{len(groups)} batches (CSV {len(batches)})", flush=True)
    assert len(groups) == len(batches), "replayed batch count != CSV batch count"
    bad_name = [k for k, (g, b) in enumerate(zip(groups, batches))
                if collections.Counter(os.path.basename(r) for r, _ in g) != collections.Counter(x["file"] for x in b)]
    bad_len = [k for k, (g, b) in enumerate(zip(groups, batches))
               if collections.Counter((os.path.basename(r), L) for r, L in g)
               != collections.Counter((x["file"], int(x["length"])) for x in b)]
    print(f"[replay] batches whose basename multiset differs from the CSV: {len(bad_name)} {bad_name[:5]}; "
          f"(basename, L) multiset differs: {len(bad_len)} {bad_len[:5]}", flush=True)
    assert not bad_name and not bad_len, "replay does not reproduce the June CSV batches"
    return {r: k for k, g in enumerate(groups) for r, _ in g}, {r: L for r, L in rel}


def read_lines(path):
    with open(path) as fh:
        out = [l.strip() for l in fh if l.strip()]
    assert len(out) == len(set(out)), f"{path}: duplicate lines"
    return out


def read_table(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    t = {r["stem"]: r for r in rows}
    assert len(t) == len(rows), f"{path}: duplicate stems"
    return t


def qs(v):
    v = sorted(v)
    if not v:
        return "n=0"
    q = lambda p: v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]
    return f"n={len(v)} min {v[0]} p25 {q(0.25)} median {st.median(v)} p75 {q(0.75)} p95 {q(0.95)} max {v[-1]}"


def cmd_list(a):
    batches, load_failed, save_failed = june_batches(a.june_csv, a.bs)
    table = read_table(a.status_tsv)
    tmpl = read_lines(a.template_list)
    part_of = {s: i % a.n_parts for i, s in enumerate(tmpl)}  # precompute_synthetic_topology_index: ids[part::n_parts]
    elig, val = set(read_lines(a.eligible)), set(read_lines(a.val_fixed))
    n_rows = sum(len(b) for b in batches) + len(load_failed)
    mixed = [b for b in batches if len({r["length"] for r in b}) > 1]
    print(f"[june] {a.june_csv}: {n_rows} rows = {sum(len(b) for b in batches)} in {len(batches)} batches of bs {a.bs} "
          f"(final batch {len(batches[-1]) if batches else 0} rows) + {len(load_failed)} load failures; "
          f"{len(save_failed)} save failures inside batches; {len(mixed)} mixed-length batches", flush=True)

    pos, cache_L = replay(a.length_cache, os.path.join(a.data_dir, "processed"), a.max_len, a.bs, batches, load_failed)
    mx = [max(int(r["length"]) for r in b) for b in batches]
    sf_in = [{r["file"] for r in b if r["status"] == "failed"} for b in batches]
    n_rows_of = collections.Counter(r["file"][:-3] for b in batches for r in b)
    n_rows_of.update(r["file"][:-3] for r in load_failed)
    print(f"[june] paths padded (L < batch max L): {sum(cache_L[r] < mx[k] for r, k in pos.items())} of {len(pos)}",
          flush=True)
    man = json.load(open(os.path.join(a.data_dir, "shard_manifest.json")))
    root = pathlib.Path(os.path.normpath(os.path.join(a.data_dir, "processed")))
    info = {}  # stem -> (L, batch, batch size, batch max L, save failure in that batch, n_june_rows) of the READ copy
    for s in table:
        r = os.path.relpath(_processed_path_sharded(root, s, man), root)
        if r in pos:
            k = pos[r]
            info[s] = (cache_L[r], k, len(batches[k]), mx[k], f"{s}.pt" in sf_in[k], n_rows_of[s])
    dup = [s for s in info if info[s][5] > 1]
    print(f"[june] table stems with > 1 June row (duplicate copies): {len(dup)}; their read copy padded: "
          f"{sum(info[s][0] < info[s][3] for s in dup)}", flush=True)

    by_status = collections.defaultdict(list)
    for s, r in table.items():
        by_status[r["status"]].append(s)
    assert set(by_status) <= {"changed", "unchanged"}, sorted(by_status)
    for stt in sorted(by_status):
        ss = by_status[stt]
        in_j = [s for s in ss if s in info]
        loadf = {r["file"][:-3] for r in load_failed}
        print(f"[table] {stt}: {len(ss)} stems; read copy in a June batch {len(in_j)}, June load failure (basename) "
              f"{sum(s in loadf for s in ss)}, read copy in no June batch {sum(s not in info and s not in loadf for s in ss)} "
              f"{[s for s in ss if s not in info and s not in loadf][:5]}; padded {sum(info[s][0] < info[s][3] for s in in_j)}",
              flush=True)
    unch = by_status["unchanged"]
    absent = [s for s in unch if s not in info]
    sf = [s for s in unch if s in info and info[s][4]]
    l_bad = [(s, info[s][0], table[s]["L_old"], table[s]["L_new"]) for s in unch if s in info
             and not (info[s][0] == int(table[s]["L_old"]) == int(table[s]["L_new"]))]
    assert not absent, (f"{len(absent)} unchanged stems: the copy T8b reads is in no June batch (map provenance unknown: "
                        f"ask), e.g. {absent[:5]}")
    assert not sf, f"{len(sf)} unchanged stems had a June SAVE failure (map not from that batch: ask), e.g. {sf[:5]}"
    assert not l_bad, f"{len(l_bad)} unchanged stems: June L != L_old/L_new (file changed since June: ask), e.g. {l_bad[:5]}"

    hit = sorted(s for s in unch if info[s][0] < info[s][3])
    rows = []
    for s in hit:
        L, k, n, mx, _, nr = info[s]
        rows.append({"stem": s, "L_june": L, "batch": k, "batch_n": n, "batch_max_L": mx, "pad": mx - L, "n_june_rows": nr,
                     "L_old": table[s]["L_old"], "L_new": table[s]["L_new"], "in_template_list": int(s in part_of),
                     "index_part": part_of.get(s, ""), "eligible": int(s in elig), "val_fixed": int(s in val)})
    assert not os.path.exists(a.out_tsv), f"{a.out_tsv} exists"
    os.makedirs(os.path.dirname(os.path.abspath(a.out_tsv)), exist_ok=True)
    with open(a.out_tsv + ".partial", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LIST_COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(a.out_tsv + ".partial", a.out_tsv)

    pads = [r["pad"] for r in rows]
    in_t = [r for r in rows if r["in_template_list"]]
    parts = sorted({r["index_part"] for r in in_t})
    print(f"[affected] UNCHANGED T8b stems padded in June: {len(rows)} of {len(unch)} unchanged "
          f"({len(rows) / max(len(unch), 1):.4%}); changed stems padded in June (already recomputed at bs 1): "
          f"{sum(info[s][0] < info[s][3] for s in by_status['changed'] if s in info)}", flush=True)
    print(f"[affected] pad (batch max L - L): {qs(pads)}", flush=True)
    print(f"[affected] pad == 1: {sum(p == 1 for p in pads)}, 2-10: {sum(2 <= p <= 10 for p in pads)}, "
          f"11-50: {sum(11 <= p <= 50 for p in pads)}, > 50: {sum(p > 50 for p in pads)}", flush=True)
    print(f"[affected] L: {qs([r['L_june'] for r in rows])}", flush=True)
    print(f"[affected] in the T8 template list ({a.template_list}, {len(tmpl)} chains): {len(in_t)} in {len(parts)} of "
          f"{a.n_parts} index parts {parts}", flush=True)
    print(f"[affected] training-eligible ({a.eligible}, {len(elig)}): {sum(r['eligible'] for r in rows)}", flush=True)
    vf = [r["stem"] for r in rows if r["val_fixed"]]
    print(f"[affected] val_fixed ({a.val_fixed}, {len(val)}): {len(vf)} {vf}", flush=True)
    print(f"[done] {len(rows)} rows -> {a.out_tsv}", flush=True)


def cmd_table(a):
    assert not os.path.exists(a.out), f"{a.out} exists -- write a new table, never modify one"
    with open(a.padfix_list, newline="") as fh:
        pad = [r["stem"] for r in csv.DictReader(fh, delimiter="\t")]
    with open(a.check_tsv, newline="") as fh:
        chk = [r["stem"] for r in csv.DictReader(fh, delimiter="\t")]
    with open(a.dropped_tsv, newline="") as fh:
        drop = [r["stem"] for r in csv.DictReader(fh, delimiter="\t")]
    assert len(pad) == len(set(pad)) > 0 and len(chk) == len(set(chk)) > 0 and len(drop) == len(set(drop))
    assert not set(chk) & set(drop), f"stems both checked and dropped: {sorted(set(chk) & set(drop))[:5]}"
    assert set(chk) | set(drop) == set(pad), (f"check + dropped != pad-fix list: missing "
                                              f"{sorted(set(pad) - set(chk) - set(drop))[:5]}, "
                                              f"extra {sorted((set(chk) | set(drop)) - set(pad))[:5]}")
    print(f"[table] pad-fix list {len(pad)} = checked {len(chk)} + dropped (identical map) {len(drop)}", flush=True)
    text = open(a.status_tsv, newline="").read()
    rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    buf = io.StringIO()
    csv.writer(buf, delimiter="\t", lineterminator="\n").writerows(rows)
    assert buf.getvalue() == text, f"{a.status_tsv}: TSV round-trip is not byte-identical -- refusing to rewrite it"
    head = rows[0]
    i_stem, i_st, i_cat, i_f = (head.index(c) for c in ("stem", "status", "category", "fields"))
    want, seen, old_cat = set(chk), set(), collections.Counter()
    for r in rows[1:]:
        if r[i_stem] in want:
            assert r[i_st] == "unchanged" and r[i_f] == "", (r[i_stem], r[i_st], r[i_f])
            old_cat[r[i_cat]] += 1
            r[i_st], r[i_cat], r[i_f] = "changed", PAD_CATEGORY, PAD_FIELDS
            seen.add(r[i_stem])
    assert seen == want, f"{len(want - seen)} pad-fix stems not in the table, e.g. {sorted(want - seen)[:5]}"
    with open(a.out + ".partial", "w", newline="") as fh:
        csv.writer(fh, delimiter="\t", lineterminator="\n").writerows(rows)
    old_lines, new_lines = text.split("\n"), open(a.out + ".partial", newline="").read().split("\n")
    diff = [i for i, (x, y) in enumerate(zip(old_lines, new_lines)) if x != y]
    assert len(old_lines) == len(new_lines) and len(diff) == len(want), (len(old_lines), len(new_lines), len(diff))
    os.replace(a.out + ".partial", a.out)
    st_b = collections.Counter(r[i_st] for r in list(csv.reader(io.StringIO(text), delimiter="\t"))[1:])
    st_a = collections.Counter(r[i_st] for r in rows[1:])
    print(f"[table] {len(rows) - 1} rows; {len(want)} pad-fix stems unchanged -> changed / {PAD_CATEGORY} / "
          f"fields {PAD_FIELDS}; all other lines byte-identical", flush=True)
    print(f"[table] status before {dict(st_b)} -> after {dict(st_a)}", flush=True)
    print(f"[table] previous categories of the pad-fix stems: {dict(old_cat)}", flush=True)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("list")
    l.add_argument("--june-csv", required=True)
    l.add_argument("--bs", type=int, required=True, help="the June band's --batch-size (band1: 8, winner_rerun.sh)")
    l.add_argument("--length-cache", required=True, help="the June band's length cache (backfill/caches/full_<ds>_<band>.json)")
    l.add_argument("--data-dir", required=True, help="pdb_train: shard_manifest.json + the processed/ root the cache spells")
    l.add_argument("--max-len", type=int, required=True, help="the June band's --max-len (band1: 512, winner_rerun.sh)")
    l.add_argument("--status-tsv", required=True)
    l.add_argument("--template-list", required=True)
    l.add_argument("--eligible", required=True)
    l.add_argument("--val-fixed", required=True)
    l.add_argument("--n-parts", type=int, required=True, help="index build parts (t8b_index.sbatch N_PARTS)")
    l.add_argument("--out-tsv", required=True)
    t = sub.add_parser("table")
    t.add_argument("--status-tsv", required=True)
    t.add_argument("--padfix-list", required=True)
    t.add_argument("--check-tsv", required=True)
    t.add_argument("--dropped-tsv", required=True)
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_list(a) if a.cmd == "list" else cmd_table(a)
