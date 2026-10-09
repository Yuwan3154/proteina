"""Backbone-geometry audit of the synthetic templates the tris use (user 2026-10-08), after the 6onw_A test template was
found with consecutive CA-CA 0.60-16.94 A at TM 0.799 to its native.

A template STEP (residue i -> i+1, both CA-present in the template) counts as a BREAK when its CA-CA distance exceeds
4.0 A -- the project's own chain-break definition (datasets/transforms.py ChainBreakPerResidueTransform default). Only
steps where the NATIVE is continuous (both CA present, native CA-CA <= 4.0 A) are scored, so a real native gap is never
charged to the template ("excess break"). Minimum CA-CA is reported as a distribution, with no threshold.

Which file is a ref's template. The v4 index was built from ONE root, the symlink pool t2_pool_auth (link name = auth id),
with its band index t2_pool_auth_band.npz (per auth chain, 64 rungs: npz slot, rewind, TM). refs RESOLVES the exact file
the builder read: a live pool link -> os.readlink (the t2_regen / t2_extra entries); a link that is gone (the T2 entries,
whose links were deleted once of_run was removed) -> the T2 tree path of the alias row's seq_partner
(<t2_orig>/shard{crc32(partner)%1000}/<partner>.npz), the rule the pool was built by. The rule is VALIDATED against the
targets recorded before the deletion (--control_targets) and refs stops unless every one matches. Full coverage: the pool
band row of EVERY ref's auth (64 TMs + rewinds + slots) must equal, exactly, the row of the resolved file's own name in that
source tree's band index (--band_sources PREFIX=BAND, ...), and that TM vector must be unique in the source band -- the
pool band was copied from those rows, and 64 independent partial-diffusion TMs are a fingerprint a swapped same-sequence
partner cannot share. (The identity guard below cannot do this: a seq_partner has the same aatype by definition, and
rewind ladders repeat across chains.) audit opens exactly
that path (after one --map ORIG=NEW prefix translation, e.g. to the SuperCloud copy of the T2 tree) and only CHECKS it:
aatype == native residue_type (the builder's identity gate) and rewind_steps[slot] == band rewind for EVERY used rung;
a failed check is 'guard_fail', a path missing on this host is 'absent'.

Modes
  refs    (Engaging) TRAIN = template rows of the synthetic index with row_tm inside the dataset's tm_range, of chains
          in BOTH the eligible list and the train split (what the training sampler can draw); rows with empty runs (the
          transform treats them as missing) are dropped and counted. TEST = the 195 pinned T7 ref_ids. Each ref's band
          rung (slot == #k) must carry the ref_id's rewind and the row's TM (float16). Writes REFS.tsv (with each ref's
          resolved original template path) and NATIVE.npz (per stem: native step continuity, CA presence, residue_type),
          natives read from the training pack.
  audit   per ref: open its resolved path (prefix-mapped), check identity, score the steps. --part/--n_parts split by stem.
  summary merge audit TSVs against REFS.tsv: every (set, ref_id) must resolve to exactly one row (precedence ok >
          guard_fail > absent; two 'ok' rows is an error); per set: counts, broken fraction per ref and per chain,
          percentiles; asserts the known-bad control (6onw_A, test).
"""

import argparse
import csv
import io
import mmap
import os
import zlib
from collections import defaultdict

import numpy as np
import torch

BREAK_A = 4.0  # ChainBreakPerResidueTransform(chain_break_cutoff=4.0)
CA = 1  # atom37 and PDB atom order both put CA at index 1
COLS = ["set", "stem", "ref_id", "tm", "status", "path", "L", "n_scored", "n_excess_break", "n_native_break", "max_ca", "min_ca", "median_ca"]


def steps(ca, present):
    """Consecutive CA-CA distances and a mask of steps with both residues present."""
    d = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
    both = present[1:] & present[:-1]
    return d, both


def read_alias(path):
    alias = {}
    with open(path) as fh:
        head = fh.readline().rstrip("\n").split("\t")
        il, ia = head.index("label_id"), head.index("auth_id")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > max(il, ia) and f[ia]:
                alias[f[il]] = f[ia]
    return alias


def cmd_refs(a):
    idx = torch.load(a.index, map_location="cpu", weights_only=False, mmap=True)
    ids, tm = idx["ids"], idx["row_tm"].numpy()
    roff = idx["runs_offset"].numpy()
    eligible = set(l.strip() for l in open(a.eligible) if l.strip())
    train = set(l.strip() for l in open(a.train_ids) if l.strip())
    alias = read_alias(a.alias)
    band = np.load(a.band, allow_pickle=True)
    brow = {str(c): i for i, c in enumerate(band["chains"])}
    lo, hi = np.float32(a.tm_lo), np.float32(a.tm_hi)  # the transform compares float32(row_tm) with float32 bounds
    rows, n_empty = [], 0
    for i, rid in enumerate(ids):
        rid = str(rid)
        if "@" not in rid:
            continue
        stem = rid.split("@")[0]
        if stem in eligible and stem in train and lo <= np.float32(tm[i]) <= hi:
            if roff[i + 1] == roff[i]:
                n_empty += 1
                continue
            rows.append(("train", stem, rid, tm[i]))
    n_train = len(rows)
    pos = {str(x): i for i, x in enumerate(ids)}
    with open(a.test_tsv) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            assert r["ref_id"] in pos, f"test ref {r['ref_id']} not in the index"
            rows.append(("test", r["stem"], r["ref_id"], tm[pos[r["ref_id"]]]))
    print(f"refs: train {n_train} template rows (tm {a.tm_lo}-{a.tm_hi}; eligible {len(eligible)} & train split {len(train)} "
          f"-> {len(eligible & train)} chains; {n_empty} rows with empty runs dropped); test {len(rows) - n_train}")
    no_alias = sorted({s for _, s, _, _ in rows if s not in alias})
    assert not no_alias, f"{len(no_alias)} stems without an alias entry, e.g. {no_alias[:10]}"
    partner = {}
    with open(a.alias) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["action"] == "use_pool" and r["auth_id"]:
                assert partner.get(r["auth_id"], r["seq_partner"]) == r["seq_partner"], f"auth {r['auth_id']}: two seq_partners"
                partner[r["auth_id"]] = r["seq_partner"]
    target, n_live, n_t2 = {}, 0, 0
    for auth in sorted({alias[s] for _, s, _, _ in rows}):
        link = f"{a.pool}/shard{zlib.crc32(auth.encode()) % 1000:04d}/{auth}.npz"
        if os.path.islink(link):
            target[auth] = os.path.normpath(os.path.join(os.path.dirname(link), os.readlink(link)))  # a relative target resolves against the link
            n_live += 1
        else:
            assert not os.path.exists(link), f"{link} is a regular file, not a pool link"
            assert auth in partner, f"auth {auth}: no live pool link and no use_pool alias row -> cannot resolve its template"
            p = partner[auth]
            target[auth] = f"{a.t2_orig}/shard{zlib.crc32(p.encode()) % 1000:04d}/{p}.npz"
            n_t2 += 1
    ctrl = {}
    with open(a.control_targets) as fh:
        for line in fh:
            rel, tgt = line.rstrip("\n").split("\t")
            ctrl[rel.split("/")[1][:-4]] = tgt
    pred = {au: target[au] for au in ctrl if au in target}
    miss = {au: (pred.get(au), ctrl[au]) for au in ctrl if pred.get(au) != ctrl[au]}
    assert not miss, f"target rule disagrees with {len(miss)} of {len(ctrl)} recorded targets, e.g. {list(miss.items())[:5]}"
    n_ct2 = sum(1 for au in pred if pred[au].startswith(a.t2_orig))
    n_crm = sum(1 for au in pred if pred[au].startswith(a.t2_orig) and partner.get(au, au) != au)
    print(f"targets: {n_live} live pool links (readlink), {n_t2} rebuilt T2 paths (seq_partner rule); control {len(pred)} of "
          f"{len(ctrl)} recorded targets reproduced exactly ({n_ct2} T2-sourced, {n_crm} of them with file name != auth id)")
    srcs = []
    for spec in a.band_sources.split(","):
        pre, bf = spec.split("=")
        z = np.load(bf, allow_pickle=True)
        rowof = {str(c): i for i, c in enumerate(z["chains"])}
        uniq = defaultdict(int)
        for v in z["tm"]:
            uniq[v.tobytes()] += 1
        srcs.append((pre, z, rowof, uniq))
    fp_bad, n_fp = [], defaultdict(int)
    for auth, tg in target.items():
        src = [s for s in srcs if tg.startswith(s[0].rstrip("/") + "/")]
        assert len(src) == 1, f"{auth}: target {tg} matches {len(src)} band sources"
        pre, z, rowof, uniq = src[0]
        key = os.path.basename(tg)[:-4]
        assert auth in brow, f"auth {auth} absent from the pool band index"
        b = brow[auth]
        if key not in rowof:
            fp_bad.append((auth, key, "not in source band"))
            continue
        i = rowof[key]
        same = all(np.array_equal(band[f][b], z[f][i]) for f in ("tm", "rewind", "slot"))
        if not same or uniq[z["tm"][i].tobytes()] != 1:
            fp_bad.append((auth, key, f"equal={same} copies={uniq[z['tm'][i].tobytes()]}"))
            continue
        n_fp[pre + (" (file name != auth id)" if key != auth else "")] += 1
    assert not fp_bad, f"band fingerprint failed for {len(fp_bad)} of {len(target)} auths, e.g. {fp_bad[:5]}"
    print("band fingerprint (64 TMs + rewinds + slots, unique in source): " + ", ".join(f"{k}: {v}" for k, v in sorted(n_fp.items())))
    bad, lines = [], []
    for st, stem, rid, t in rows:
        auth, k, rw = alias[stem], int(rid.split("#")[1]), int(rid.split("@rw")[1].split("#")[0])
        assert auth in brow, f"{stem} (auth {auth}) absent from the band index"
        b = brow[auth]
        sl, rws, tms = band["slot"][b], band["rewind"][b], band["tm"][b]
        r = [j for j in range(len(sl)) if sl[j] == k]
        if len(r) != 1 or int(rws[r[0]]) != rw or np.float16(tms[r[0]]) != t:
            bad.append((rid, r, [int(rws[j]) for j in r], [float(tms[j]) for j in r], float(t)))
            continue
        used = [j for j in range(len(sl)) if sl[j] >= 0]
        lines.append(f"{st}\t{stem}\t{auth}\t{rid}\t{k}\t{rw}\t{float(t):.4f}\t{','.join(str(int(sl[j])) for j in used)}\t"
                     f"{','.join(str(int(rws[j])) for j in used)}\t{target[auth]}\n")
    assert not bad, f"{len(bad)} refs disagree with the band index (rung, rewind, tm), e.g. {bad[:5]}"
    with open(a.out_refs, "w") as fh:  # written only after every ref passed the band check
        fh.write("set\tstem\tauth\tref_id\tslot\trewind\ttm\trung_slots\trung_rewinds\ttarget\n")
        fh.writelines(lines)
    z = np.load(a.pack + ".idx.npz", allow_pickle=True)
    ent = {str(s): (int(o), int(l)) for s, o, l in zip(z["stems"], z["offsets"], z["lengths"])}
    stems = sorted({s for _, s, _, _ in rows})
    missing = [s for s in stems if s not in ent]
    assert not missing, f"{len(missing)} stems absent from the pack, e.g. {missing[:10]}"
    cont, pres, aa, off = [], [], [], [0]
    with open(a.pack, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for k, s in enumerate(stems):
            o, l = ent[s]
            g = torch.load(io.BytesIO(mm[o:o + l]), map_location="cpu", weights_only=False)
            p = g.coord_mask[:, CA].numpy().astype(bool)
            d, both = steps(g.coords[:, CA].numpy(), p)
            cont.append(np.append(both & (d <= BREAK_A), False))  # padded to L so offsets are shared
            pres.append(p)
            aa.append(g.residue_type.numpy().astype(np.int16))
            off.append(off[-1] + len(p))
            if (k + 1) % 5000 == 0:
                print(f"  natives {k + 1}/{len(stems)}", flush=True)
    np.savez_compressed(a.out_native, stems=np.array(stems), offsets=np.array(off, np.int64),
                        present=np.concatenate(pres), cont=np.concatenate(cont), aatype=np.concatenate(aa))
    print(f"natives: {len(stems)} stems -> {a.out_native}")


def accept(path, n_aa, slots, rewinds):
    """The builder's identity gate + the band's slot map; returns the npz or None."""
    npz = np.load(path)
    t_aa = np.asarray(npz["aatype"]).astype(np.int16)
    if t_aa.shape != n_aa.shape or not (t_aa == n_aa).all():
        return None
    rs = npz["rewind_steps"]
    if max(slots) >= len(rs) or any(int(rs[s]) != r for s, r in zip(slots, rewinds)):
        return None
    return npz


def cmd_audit(a):
    z = np.load(a.native, allow_pickle=True)
    nat = {str(s): (int(z["offsets"][i]), int(z["offsets"][i + 1])) for i, s in enumerate(z["stems"])}
    present_all, cont_all, aa_all = z["present"], z["cont"], z["aatype"]
    orig, new = a.map.split("=") if a.map else ("", "")
    by_auth = defaultdict(list)
    with open(a.refs) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if a.sets and r["set"] not in a.sets.split(","):
                continue
            if zlib.crc32(r["stem"].encode()) % a.n_parts != a.part:
                continue
            by_auth[(r["auth"], r["stem"])].append(r)
    n = defaultdict(int)
    with open(a.out, "w") as out:
        out.write("\t".join(COLS) + "\n")
        for (auth, stem), refs in sorted(by_auth.items()):
            o0, o1 = nat[stem]
            slots = [int(x) for x in refs[0]["rung_slots"].split(",")]
            rewinds = [int(x) for x in refs[0]["rung_rewinds"].split(",")]
            path = refs[0]["target"]
            if orig and path.startswith(orig):
                path = new + path[len(orig):]
            npz = accept(path, aa_all[o0:o1], slots, rewinds) if os.path.exists(path) else None
            if npz is None:
                status = "guard_fail" if os.path.exists(path) else "absent"
                for r in refs:
                    out.write("\t".join([r["set"], stem, r["ref_id"], r["tm"], status] + [""] * (len(COLS) - 5)) + "\n")
                    n[status] += 1
                continue
            amask = npz["atom_mask"].astype(bool)
            ncont, npres = cont_all[o0:o1 - 1], present_all[o0:o1]
            n_native_break = int((npres[1:] & npres[:-1] & ~ncont).sum())
            for r in refs:
                full = np.zeros(amask.shape + (3,), np.float32)
                full[amask] = npz["coords"][int(r["slot"])]  # coords hold the PRESENT atoms only (builder scatter)
                d, both = steps(full[:, CA], amask[:, CA])
                ds = d[both & ncont]
                st = lambda f: f"{f(ds):.3f}" if len(ds) else "nan"
                out.write("\t".join([r["set"], stem, r["ref_id"], r["tm"], "ok", path, str(amask.shape[0]), str(len(ds)),
                                     str(int((ds > BREAK_A).sum())), str(n_native_break), st(np.max), st(np.min), st(np.median)]) + "\n")
                n["ok"] += 1
    print(f"part {a.part}/{a.n_parts}: " + ", ".join(f"{k} {v}" for k, v in sorted(n.items())))


def cmd_summary(a):
    rank = {"ok": 0, "guard_fail": 1, "absent": 2}
    best = {}
    for p in a.tsvs:
        with open(p) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                key = (r["set"], r["ref_id"])
                if key in best and best[key]["status"] == "ok" and r["status"] == "ok":
                    raise AssertionError(f"{key} audited 'ok' twice ({best[key]['path']} and {r['path']})")
                if key not in best or rank[r["status"]] < rank[best[key]["status"]]:
                    best[key] = r
    with open(a.refs) as fh:
        want = {(r["set"], r["ref_id"]) for r in csv.DictReader(fh, delimiter="\t")}
    lost, extra = want - set(best), set(best) - want
    assert not lost and not extra, f"audit rows vs REFS: {len(lost)} refs without a row (e.g. {sorted(lost)[:5]}), {len(extra)} rows not in REFS"
    print(f"{len(best)} refs resolved from {len(a.tsvs)} files (complete against {a.refs})")
    for st in ("test", "train"):
        R = [r for k, r in best.items() if k[0] == st]
        if not R:
            continue
        status = defaultdict(int)
        for r in R:
            status[r["status"]] += 1
        print(f"\n== {st.upper()}: {len(R)} refs; status " + ", ".join(f"{k} {v}" for k, v in sorted(status.items())))
        ok = [r for r in R if r["status"] == "ok" and int(r["n_scored"]) > 0]
        print(f"  scored: {len(ok)} (ok with 0 scored steps: {status['ok'] - len(ok)})")
        if not ok:
            continue
        nb = np.array([int(r["n_excess_break"]) for r in ok])
        frac = np.array([int(r["n_excess_break"]) / int(r["n_scored"]) for r in ok])
        mx = np.array([float(r["max_ca"]) for r in ok])
        mn = np.array([float(r["min_ca"]) for r in ok])
        chains = defaultdict(list)
        for r in ok:
            chains[r["stem"]].append(int(r["n_excess_break"]) > 0)
        q = lambda x: " / ".join(f"{v:.2f}" for v in np.percentile(x, [1, 5, 50, 95, 99]))
        print(f"  refs with >=1 excess break (CA-CA > {BREAK_A} A where the native is continuous): {(nb > 0).sum()} of {len(ok)} ({100 * (nb > 0).mean():.2f}%)")
        print(f"  chain-weighted (mean over chains of each chain's broken-ref fraction = a uniform draw per chain): "
              f"{100 * np.mean([np.mean(v) for v in chains.values()]):.2f}%")
        print(f"  excess breaks per broken ref: median {np.median(nb[nb > 0]) if (nb > 0).any() else 0:.0f}, max {nb.max()}")
        print(f"  broken-step fraction per ref p50/p95/p99/max: {np.percentile(frac, 50):.4f} / {np.percentile(frac, 95):.4f} / {np.percentile(frac, 99):.4f} / {frac.max():.4f}")
        print(f"  max CA-CA per ref   p1/p5/p50/p95/p99: {q(mx)} A")
        print(f"  min CA-CA per ref   p1/p5/p50/p95/p99: {q(mn)} A")
        print(f"  chains: {len(chains)}; with >=1 broken ref {sum(any(v) for v in chains.values())}; ALL refs broken {sum(all(v) for v in chains.values())}")
        worst = sorted(ok, key=lambda r: -int(r["n_excess_break"]))[:10]
        print("  worst: " + "; ".join(f"{r['ref_id']} {r['n_excess_break']}/{r['n_scored']} tm {float(r['tm']):.2f}" for r in worst))
    ctrl = [r for k, r in best.items() if k[0] == "test" and r["stem"] == "6onw_A"]
    if ctrl:
        assert ctrl[0]["status"] == "ok" and int(ctrl[0]["n_excess_break"]) > 0, f"known-bad control 6onw_A not flagged: {ctrl}"
        print(f"\ncontrol: 6onw_A flagged with {ctrl[0]['n_excess_break']} excess breaks of {ctrl[0]['n_scored']} scored steps (expected > 0)")
    else:
        print("\ncontrol: 6onw_A not in these refs (no test set) -- control NOT exercised")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="mode", required=True)
p = sub.add_parser("refs")
for k in ("index", "eligible", "train_ids", "alias", "band", "band_sources", "pool", "t2_orig", "control_targets", "test_tsv", "pack", "out_refs", "out_native"):
    p.add_argument(f"--{k}", required=True)
p.add_argument("--tm_lo", type=float, required=True)
p.add_argument("--tm_hi", type=float, required=True)
p = sub.add_parser("audit")
for k in ("refs", "native", "out"):
    p.add_argument(f"--{k}", required=True)
p.add_argument("--map", default="", help="ORIG_PREFIX=NEW_PREFIX applied to each ref's resolved path")
p.add_argument("--sets", default="")
p.add_argument("--part", type=int, default=0)
p.add_argument("--n_parts", type=int, default=1)
p = sub.add_parser("summary")
p.add_argument("--refs", required=True)
p.add_argument("tsvs", nargs="+")
a = ap.parse_args()
{"refs": cmd_refs, "audit": cmd_audit, "summary": cmd_summary}[a.mode](a)
