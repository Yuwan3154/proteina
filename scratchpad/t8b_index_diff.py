"""T8b index v1 vs v2 (June pad-fix): the v2 build differs only in the native pack (tri_t8b_512_v2.pack), so every row must
be identical except the NATIVE rows of the CHECKED pad-fix stems (check_padfix.tsv; dropped stems kept their stored map and
must be identical). v2 must carry native_source == --expect-native-pack (+ md5 of its .idx.npz); the checked natives must
be in the index (> 0) and some must have changed (> 0). Asserts: identical row ids / order and
per-row metadata (cluster, members, seq_hash, TM, rewind, USalign TM, is_native); every template row and every other
native row byte-identical in runs / he (contact reference) / feat / align; pad-fix native rows identical runs (DSSP does not
read the map); identical non-row keys (feature stats come from template rows only); identical eligible-id files.
  python t8b_index_diff.py --old V1.pt --new V2.pt --old-eligible V1.txt --new-eligible V2.txt \
      --checked-tsv check_padfix.tsv --expect-native-pack tri_t8b_512_v2.pack
"""
import argparse
import csv
import hashlib
import json
import os

import torch

ROW_KEYS = ["cluster_of", "seq_hash", "row_tm", "row_is_native", "row_rewind", "row_usalign_tm", "he_size"]
FLATS = [("runs_flat", "runs_offset"), ("he_flat", "he_offset"), ("feat_flat", "feat_offset"), ("align_flat", "align_offset")]
MEMBER_KEYS = ["members_flat", "members_offset"]
STAMP = "native_source"  # differs between v1 and v2 by design; checked against --expect-native-pack


def as_bits(t):
    """Byte-exact comparison form (float16 NaN payloads included)."""
    return t.view(torch.int16) if t.dtype == torch.float16 else t


def flat_rows(d, flat, off, rows):
    o = d[off]
    return [as_bits(d[flat][int(o[r]):int(o[r + 1])]) for r in rows]


def element_mask(off, row_mask):
    """Row mask -> mask over the flat elements (first axis) of those rows."""
    lens = off[1:] - off[:-1]
    return torch.repeat_interleave(row_mask, lens)


def diff(old, new, allowed_stems):
    ids = list(old["ids"])
    assert ids == list(new["ids"]), "row ids / order differ"
    nat = old["row_is_native"].bool()
    allowed = torch.tensor([bool(n) and s in allowed_stems for s, n in zip(ids, nat.tolist())])
    print(f"[rows] {len(ids)} = {int(nat.sum())} natives + {int((~nat).sum())} template rows; pad-fix natives in the "
          f"index {int(allowed.sum())} of {len(allowed_stems)} listed stems", flush=True)
    per_row = set(ROW_KEYS + MEMBER_KEYS + [f for f, _ in FLATS] + [o for _, o in FLATS] + ["ids", STAMP])
    meta_old = {k: v for k, v in old.items() if k not in per_row}
    meta_new = {k: v for k, v in new.items() if k not in per_row}
    assert set(meta_old) == set(meta_new), f"keys differ: {set(meta_old) ^ set(meta_new)}"
    bad_meta = [k for k in meta_old if not (torch.equal(meta_old[k], meta_new[k]) if torch.is_tensor(meta_old[k])
                                            else meta_old[k] == meta_new[k])]
    assert not bad_meta, f"non-row keys differ: {bad_meta}"
    for k in MEMBER_KEYS + [k for k in ROW_KEYS if k != "he_size"]:
        assert torch.equal(as_bits(old[k]), as_bits(new[k])), f"{k} differs"
    keep = ~allowed
    assert torch.equal(old["he_size"][keep], new["he_size"][keep]), "he_size differs outside the pad-fix natives"
    for flat, off in FLATS:
        lo, ln = old[off][1:] - old[off][:-1], new[off][1:] - new[off][:-1]
        assert torch.equal(lo[keep], ln[keep]), f"{flat}: row lengths differ outside the pad-fix natives"
        a = as_bits(old[flat])[element_mask(old[off], keep)]
        b = as_bits(new[flat])[element_mask(new[off], keep)]
        assert torch.equal(a, b), f"{flat}: content differs outside the pad-fix natives"
    rows = torch.nonzero(allowed).flatten().tolist()
    for r, (x, y) in enumerate(zip(flat_rows(old, "runs_flat", "runs_offset", rows), flat_rows(new, "runs_flat", "runs_offset", rows))):
        assert torch.equal(x, y), f"{ids[rows[r]]}: native runs differ (DSSP should not depend on the map)"
    he_o, he_n = flat_rows(old, "he_flat", "he_offset", rows), flat_rows(new, "he_flat", "he_offset", rows)
    ft_o, ft_n = flat_rows(old, "feat_flat", "feat_offset", rows), flat_rows(new, "feat_flat", "feat_offset", rows)
    changed = sum(not (torch.equal(a, b) and torch.equal(c, d)) for a, b, c, d in zip(he_o, he_n, ft_o, ft_n))
    print(f"[rows] template rows + non-pad-fix natives byte-identical; pad-fix natives: {changed} changed (he/feat), "
          f"{len(rows) - changed} identical; runs identical for all", flush=True)
    assert len(rows) > 0, "no checked pad-fix stem has a native row in the index"
    assert changed > 0, "no checked pad-fix native row changed: was v2 built from the v2 native pack?"
    return len(rows), changed


def check_stamp(new, expect_pack):
    assert STAMP in new, f"v2 index has no {STAMP} stamp (built before the stamp existed?)"
    got = json.loads(new[STAMP])
    want = {"native_pack": os.path.abspath(expect_pack),
            "native_idx_md5": hashlib.md5(open(expect_pack + ".idx.npz", "rb").read()).hexdigest()}
    print(f"[stamp] v2 {STAMP} {got}", flush=True)
    assert got == want, f"v2 native source {got} != expected {want}"


def read_lines(p):
    return [l.rstrip("\n") for l in open(p)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old-eligible", required=True)
    ap.add_argument("--new-eligible", required=True)
    ap.add_argument("--checked-tsv", required=True, help="check_padfix.tsv (the kept pad-fix stems)")
    ap.add_argument("--expect-native-pack", required=True)
    a = ap.parse_args()
    with open(a.checked_tsv, newline="") as fh:
        pad = {r["stem"] for r in csv.DictReader(fh, delimiter="\t")}
    print(f"[checked] {len(pad)} pad-fix stems from {a.checked_tsv}", flush=True)
    e_old, e_new = read_lines(a.old_eligible), read_lines(a.new_eligible)
    print(f"[eligible] old {len(e_old)} lines, new {len(e_new)} lines", flush=True)
    assert e_old == e_new, "eligible-id files differ"
    old = torch.load(a.old, map_location="cpu", weights_only=False, mmap=True)
    new = torch.load(a.new, map_location="cpu", weights_only=False, mmap=True)
    check_stamp(new, a.expect_native_pack)
    diff(old, new, pad)
    print("[diff] PASS", flush=True)
