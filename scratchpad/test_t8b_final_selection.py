"""T8b final-selection mode of precompute_synthetic_topology_index on tiny synthetic packs: pack precedence (an
override-listed chain takes ONLY the override pack; zero templates there = none), TM window + clash gate still applied
and no cap, the final selection (not the whole pack) is what is used, pack-tagged keys (same (round, draw) in both packs
never collide), f2c T8-map reuse rules + f2c-plan counts == the real f2c run, parts audit (strict) incl. a precedence
violation caught, CLI guards, maps-file stamp checks (batch size / checkpoint / reuse), a count-0 selection entry ==
absent, the plan's missing-N/CA/C count, and the T8 pack-mode path bit-identical to HEAD (part + merged index; every
subprocess runs from a neutral cwd so the HEAD arm really imports HEAD).
Run (local: conda synthbench_cpu; needs git + USalign): python scratchpad/test_t8b_final_selection.py [USALIGN]"""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import proteinfoundation.utils.precompute_synthetic_topology_index as B  # noqa: E402
from proteinfoundation.utils.frame2confind_utils import _place_cb  # noqa: E402
from proteinfoundation.utils.t2_pack_reader import TemplatePack, layout  # noqa: E402

USALIGN = sys.argv[1] if len(sys.argv) > 1 else shutil.which("USalign")
assert USALIGN and os.path.exists(USALIGN), "pass the USalign binary"
MOD = "proteinfoundation.utils.precompute_synthetic_topology_index"
TAG0, TAG1 = "t2_synth_v2", "t2_synth_v2_t8b"
NEUTRAL = tempfile.mkdtemp(prefix="t8b_cwd_")  # `python -c` puts cwd first on sys.path: never run from a checkout
FAKE_F2C = """
import types, sys, torch
class P:
    def __init__(self, checkpoint, amp_dtype, compile_model):
        self.device = "cpu"
    def predict_batch(self, x, mask):
        ca = x[:, :, 1].float()
        pm = mask[:, :, None] & mask[:, None, :]
        return ((torch.cdist(ca, ca) < 8.0) & pm).float()
for n in ("Frame2ConFind", "Frame2ConFind.inference"):
    sys.modules[n] = types.ModuleType(n)
m = types.ModuleType("Frame2ConFind.inference.api")
m.Frame2ConFindPredictor = P
sys.modules["Frame2ConFind.inference.api"] = m
"""


# ─── geometry: ideal backbone by NeRF (helix then extended tail: no non-bonded CA-CA < 3.4 A) ───
def nerf(a, b, c, bond, angle, torsion):
    bc = (c - b) / np.linalg.norm(c - b)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    d = np.array([-bond * np.cos(angle), bond * np.sin(angle) * np.cos(torsion), bond * np.sin(angle) * np.sin(torsion)])
    return c + d[0] * bc + d[1] * np.cross(n, bc) + d[2] * n


def backbone(n_helix, n_tail):
    r = np.deg2rad
    phi = [-57.0] * n_helix + [-120.0] * n_tail
    psi = [-47.0] * n_helix + [130.0] * n_tail
    L = n_helix + n_tail
    N, CA, C = [np.zeros(3)], [np.array([1.458, 0, 0])], []
    C.append(CA[0] + 1.525 * np.array([-np.cos(r(111.2)), np.sin(r(111.2)), 0]))
    for i in range(L - 1):
        N.append(nerf(N[i], CA[i], C[i], 1.329, r(116.2), r(psi[i])))
        CA.append(nerf(CA[i], C[i], N[i + 1], 1.458, r(121.7), r(180.0)))
        C.append(nerf(C[i], N[i + 1], CA[i + 1], 1.525, r(111.2), r(phi[i + 1])))
    O = [nerf(N[i], CA[i], C[i], 1.231, r(120.5), r(psi[i] + 180.0)) for i in range(L)]
    N, CA, C, O = (torch.tensor(np.stack(v), dtype=torch.float32) for v in (N, CA, C, O))
    return N, CA, C, O, _place_cb(N, CA, C)


def native_graph(L_h, L_t):
    N, CA, C, O, CB = backbone(L_h, L_t)
    L = L_h + L_t
    coords = torch.zeros(L, 37, 3)
    for k, a in enumerate((N, CA, C, O, CB)):  # disk (ATOM_NUMBERING) order N, CA, C, O, CB
        coords[:, k] = a
    cmask = torch.zeros(L, 37, dtype=torch.bool)
    cmask[:, :5] = True
    cm = (torch.cdist(CA, CA) < 8.0).to(torch.float16)
    return SimpleNamespace(coords=coords, coord_mask=cmask, dssp_target=torch.zeros(L, dtype=torch.long),
                           residue_type=torch.zeros(L, dtype=torch.long), sequence="A" * L, contact_map_confind=cm)


def template(g, seed, scale=1.0, drop=None, no_n=None):
    """atom37 (N, CA, C, CB, O) copy of a native with 0.05 A noise; scale < 1 collapses it (clashes); drop = indel;
    no_n = a residue whose N is missing."""
    x = g.coords[:, :5].numpy()[:, [0, 1, 2, 4, 3]] * scale
    x = x + np.random.default_rng(seed).normal(0, 0.05, x.shape)
    if drop is not None:
        x = np.delete(x, drop, axis=0)
    full = np.zeros((x.shape[0], 37, 3), np.float32)
    full[:, :5] = x
    mask = np.zeros((x.shape[0], 37), bool)
    mask[:, :5] = True
    if no_n is not None:
        mask[no_n, 0] = False
        full[no_n, 0] = 0.0
    return full, mask


# ─── tiny pack writers in the t2 formats ───
def write_template_pack(root, chains):
    """chains: [(stem, [(round, draw, tm, full, mask)])] -> tmpl.000.bin + tmpl.idx.npz (TemplatePack layout)."""
    os.makedirs(root)
    buf, c = bytearray(), {k: [] for k in ("draw", "round", "shard", "off", "nbytes", "L", "A", "L_native", "tm_native",
                                           "tm_template", "rewind", "num_steps", "source", "model", "arm", "ops_len",
                                           "ops_off")}
    stems, first, count = [], [], []
    for stem, tmpls in chains:
        stems.append(stem)
        first.append(len(c["draw"]))
        count.append(len(tmpls))
        for rnd, draw, tm, full, mask in tmpls:
            L, xyz = mask.shape[0], full[mask].astype(np.float32)
            lay = layout(L, len(xyz))
            rec = bytearray(lay["nbytes"])
            rec[lay["residue_index"]:lay["residue_index"] + 4 * L] = np.arange(1, L + 1, dtype=np.int32).tobytes()
            rec[lay["orig_idx"]:lay["orig_idx"] + 4 * L] = np.arange(L, dtype=np.int32).tobytes()
            rec[lay["atom_mask"]:lay["atom_mask"] + 37 * L] = mask.astype(np.uint8).tobytes()
            rec[lay["coords"]:lay["coords"] + 12 * len(xyz)] = xyz.tobytes()
            vals = dict(draw=draw, round=rnd, shard=0, off=len(buf), nbytes=len(rec), L=L, A=len(xyz), L_native=L,
                        tm_native=tm, tm_template=tm, rewind=125, num_steps=250, source="production", model="cc89",
                        arm="mutfill", ops_len=2, ops_off=2 * len(c["draw"]))
            for k, v in vals.items():
                c[k].append(v)
            buf += rec
    open(os.path.join(root, "tmpl.000.bin"), "wb").write(bytes(buf))
    np.savez(os.path.join(root, "tmpl.idx.npz"), chain_stems=np.array(stems), chain_first=np.array(first, np.int64),
             chain_count=np.array(count, np.int32), ops_blob=np.frombuffer(b"[]" * len(c["draw"]), np.uint8),
             **{k: (np.array(v, np.float32) if k.startswith("tm_") else np.array(v)) for k, v in c.items()})


def write_selection(path, sel):
    """sel: {stem: [(round, draw)]} -> final_selection.npz (FinalSet layout)."""
    stems, first, count, rnd, draw = [], [], [], [], []
    for s, keys in sel.items():
        stems.append(s)
        first.append(len(draw))
        count.append(len(keys))
        rnd += [k[0] for k in keys]
        draw += [k[1] for k in keys]
    np.savez(path, chain=np.array(stems), first=np.array(first, np.int64), count=np.array(count, np.int32),
             round=np.array(rnd), draw=np.array(draw, np.int32))


def write_native_pack(path, graphs):
    stems, offs, lens, blob = [], [], [], bytearray()
    for s, g in graphs.items():
        b = io.BytesIO()
        torch.save(g, b)
        stems.append(s)
        offs.append(len(blob))
        lens.append(len(b.getvalue()))
        blob += b.getvalue()
    open(path, "wb").write(bytes(blob))
    np.savez(path + ".idx.npz", stems=np.array(stems, dtype=object), offsets=np.array(offs), lengths=np.array(lens))


def write_t8_maps(root, stem, entries, checkpoint):
    """Pack-mode (T8) maps file: entries [(draw, L)] all-ONES maps (so a copy is told apart from a computed map)."""
    p = B.pack_map_path(root, stem)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    bits = [np.packbits(np.ones(L * L, bool)) for _, L in entries]
    np.savez(p, draws=np.array([d for d, _ in entries], np.int32), L=np.array([L for _, L in entries], np.int32),
             byte_off=np.concatenate([[0], np.cumsum([b.size for b in bits])]).astype(np.int64),
             packed=np.concatenate(bits), threshold=np.float32(B.CONFIND_THRESHOLD), checkpoint=checkpoint)


def run(argv, pythonpath=REPO, pre="", expect_rc=0):
    code = (pre + "\nimport multiprocessing as mp\nmp.set_start_method('fork', force=True)\nimport sys\n"
            f"sys.argv = ['x'] + {argv!r}\nimport {MOD} as B\nB.main()\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=NEUTRAL,
                         env={**os.environ, "PYTHONPATH": str(pythonpath)})
    if (out.returncode == 0) != (expect_rc == 0):
        raise AssertionError(f"rc {out.returncode} for {argv[:1]}: {out.stdout[-2000:]}\n{out.stderr[-3000:]}")
    return out.stdout + out.stderr


def deep_eq(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(deep_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(deep_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    return a == b


def main():
    td = tempfile.mkdtemp(prefix="t8b_fs_")
    ckpt = os.path.join(td, "best.pt")
    open(ckpt, "wb").write(b"fake checkpoint")
    g = {s: native_graph(24 + i, 10) for i, s in enumerate(("a_A", "b_B", "c_C", "d_D", "e_E", "f_F"))}
    T = lambda s, rnd, d, tm, **kw: (rnd, d, tm, *template(g[s], seed=d + 1000 * ord(s[0]), **kw))  # noqa: E731
    old_chains = [
        ("a_A", [T("a_A", "base", 0, 0.60), T("a_A", "base", 1, 0.70), T("a_A", "base", 2, 0.80),
                 T("a_A", "base", 3, 0.45), T("a_A", "base", 4, 0.85), T("a_A", "P1", 100, 0.65, drop=30)]),
        ("b_B", [T("b_B", "base", 7, 0.70), T("b_B", "base", 8, 0.75)]),
        ("c_C", [T("c_C", "base", 0, 0.70)]),
        ("d_D", [T("d_D", "base", 0, 0.70)]),
        ("e_E", [T("e_E", "base", 0, 0.70, scale=0.3, no_n=3), T("e_E", "base", 1, 0.75, scale=0.3)]),
    ]
    new_chains = [("b_B", [T("b_B", "base", 7, 0.66), T("b_B", "base", 4050, 0.80), T("b_B", "base", 4051, 0.55)])]
    p0, p1 = os.path.join(td, "pack_old"), os.path.join(td, "pack_t8b")
    write_template_pack(p0, old_chains)
    write_template_pack(p1, new_chains)
    sel0 = {"a_A": [("base", 0), ("base", 1), ("base", 3), ("base", 4), ("P1", 100)], "b_B": [("base", 7), ("base", 8)],
            "c_C": [("base", 0)], "e_E": [("base", 0), ("base", 1)], "d_D": []}  # a_A base 2 NOT selected; d_D count 0
    sel1 = {"b_B": [("base", 7), ("base", 4050)], "f_F": []}  # b_B 4051 not selected; c_C absent, f_F count 0
    write_selection(os.path.join(p0, "final_selection.npz"), sel0)
    write_selection(os.path.join(p1, "final_selection.npz"), sel1)
    chains = os.path.join(td, "chains.txt")
    open(chains, "w").write("\n".join(["a_A", "b_B", "c_C", "d_D", "e_E", "f_F", "g_G"]) + "\n")
    override = os.path.join(td, "override.txt")
    open(override, "w").write("b_B\nc_C\nf_F\n")
    natp = os.path.join(td, "natives.pack")
    write_native_pack(natp, g)  # g_G has no native
    # pre-append index = the base-round rows only; a_A base 4 moved (a template the T8 map was NOT computed from)
    z = dict(np.load(os.path.join(p0, "tmpl.idx.npz")))
    base = np.flatnonzero(z["round"] == "base")
    pre = {k: z[k][base] for k in ("draw", "shard", "off", "L", "A")}
    st = z["chain_stems"]
    pre_first, pre_count, f = [], [], 0
    for c in range(len(st)):
        n = int(((base >= z["chain_first"][c]) & (base < z["chain_first"][c] + z["chain_count"][c])).sum())
        pre_first.append(f)
        pre_count.append(n)
        f += n
    stem_of = np.repeat(z["chain_stems"], z["chain_count"])
    a4 = int(np.flatnonzero((stem_of[base] == "a_A") & (pre["draw"] == 4))[0])
    pre["off"] = pre["off"].copy()
    pre["off"][a4] += 8
    pre_idx = os.path.join(td, "tmpl.idx.npz.pre-append")
    with open(pre_idx, "wb") as fh:  # the real file is named *.pre-append (np.savez would append .npz to a path)
        np.savez(fh, chain_stems=st, chain_first=np.array(pre_first), chain_count=np.array(pre_count), **pre)
    t8maps = os.path.join(td, "t8_f2c_maps")
    for s, ts in old_chains:
        write_t8_maps(t8maps, s, [(t[1], t[3].shape[0]) for t in ts if t[0] == "base"],
                      ckpt if s != "e_E" else ckpt + ".other")  # e_E: other checkpoint string -> never reused

    SEL = ["--chain-list", chains, "--processed-dir", os.path.join(td, "processed"), "--template-pack", p0,
           "--final-selection", os.path.join(p0, "final_selection.npz"), "--template-pack-tag", TAG0,
           "--override-pack", p1, "--override-selection", os.path.join(p1, "final_selection.npz"),
           "--override-chains", override, "--override-pack-tag", TAG1, "--tm-min", "0.5", "--tm-max", "0.9"]
    REUSE = ["--reuse-maps", t8maps, "--reuse-pack-idx", pre_idx, "--checkpoint", ckpt]
    BLD0 = ["--contact-source", "confind", "--element-types", "0,1,2", "--loop-axis", "zero", "--proline-donor-mask",
           "--native-pack", natp, "--workers", "2", "--usalign", USALIGN, "--max-ca-clashes", "4", "--ca-clash-a", "3.4"]
    BLD = [*BLD0, "--f2c-batch-size", "2", "--f2c-checkpoint", ckpt, "--f2c-allow-reuse"]
    maps = os.path.join(td, "maps")

    # --- f2c-plan (no GPU, no writes) must predict exactly what f2c then does
    run(["f2c-plan", *SEL, *REUSE, "--part", "0", "--n-parts", "1", "--maps-out", maps, "--plan-out", td + "/plan"])
    plan = dict(l.rstrip("\n").split("\t") for l in open(td + "/plan/f2c_plan_0000.tsv"))
    assert not os.path.exists(maps) or not os.listdir(maps), "f2c-plan wrote maps"
    exp_plan = {f"{TAG0}:chains": "2", f"{TAG0}:templates": "6", f"{TAG0}:maps_reused": "2",
                f"{TAG0}:maps_to_compute": "4", f"{TAG1}:chains": "1", f"{TAG1}:templates": "2",
                f"{TAG1}:maps_to_compute": "2", "not_reused:not_base_round": "1", "not_reused:pre_append_row_differs": "1",
                "not_reused:t8_checkpoint_or_threshold_differs": "2", f"{TAG0}:pack_templates": "12",
                f"{TAG0}:pack_templates_round_base": "11", f"{TAG1}:pack_templates": "3",
                f"{TAG0}:templates_missing_N_CA_C": "1", f"{TAG1}:templates_missing_N_CA_C": "0"}
    for k, v in exp_plan.items():
        assert plan.get(k) == v, (k, plan.get(k), v, plan)
    print("PASS f2c-plan: reuse only for base-round rows with matching pre-append (shard, off, L, A) + checkpoint; "
          f"counts {exp_plan}")

    # --- f2c (fake Frame2ConFind = CA-CA < 8 A) for 2 parts, then a rerun is a no-op
    for part in ("0", "1"):
        run(["f2c", *SEL, *REUSE, "--part", part, "--n-parts", "2", "--maps-out", maps, "--batch-size", "2"], pre=FAKE_F2C)
    origin = {}
    for tag, stem in ((TAG0, "a_A"), (TAG0, "e_E"), (TAG1, "b_B")):
        m = np.load(B.final_map_path(maps, tag, stem))
        assert str(m["pack_tag"]) == tag and int(m["batch_size"]) == 2 and str(m["checkpoint_md5"])
        for i, (rd, d, o) in enumerate(zip(m["rounds"], m["draws"], m["origin"])):
            origin[(tag, stem, str(rd), int(d))] = str(o)
            L = int(m["L"][i])
            bits = np.unpackbits(m["packed"][int(m["byte_off"][i]):int(m["byte_off"][i + 1])], count=L * L)
            if str(o) == "t8_reuse":
                assert bits.all(), "reused map is not the T8 file's"
            else:
                src = p0 if tag == TAG0 else p1
                rec = TemplatePack(src).get(stem, int(d), str(rd))
                full = np.zeros((L, 37, 3), np.float32)
                full[rec["atom_mask"]] = rec["coords"]
                ca = torch.from_numpy(full[:, 1])
                res = rec["atom_mask"][:, :3].all(1)  # f2c masks residues lacking N / CA / C
                exp = (torch.cdist(ca, ca) < 8.0).numpy() & res[:, None] & res[None, :]
                assert np.array_equal(bits.astype(bool), exp.reshape(-1)), (stem, d)
    exp_origin = {(TAG0, "a_A", "base", 0): "t8_reuse", (TAG0, "a_A", "base", 1): "t8_reuse",
                  (TAG0, "a_A", "P1", 100): "computed", (TAG0, "a_A", "base", 4): "computed",
                  (TAG0, "e_E", "base", 0): "computed", (TAG0, "e_E", "base", 1): "computed",
                  (TAG1, "b_B", "base", 7): "computed", (TAG1, "b_B", "base", 4050): "computed"}
    assert origin == exp_origin, origin
    assert not os.path.exists(B.final_map_path(maps, TAG0, "b_B")), "override chain got an original-pack maps file"
    assert not os.path.exists(B.final_map_path(maps, TAG0, "c_C")), "override chain c_C got original-pack maps"
    assert sum(v == "t8_reuse" for v in origin.values()) == int(exp_plan[f"{TAG0}:maps_reused"])
    out = run(["f2c-plan", *SEL, *REUSE, "--part", "0", "--n-parts", "1", "--maps-out", maps, "--plan-out", td + "/plan2"])
    plan2 = dict(l.rstrip("\n").split("\t") for l in open(td + "/plan2/f2c_plan_0000.tsv"))
    assert plan2[f"{TAG0}:chains_already_written"] == "2" and plan2[f"{TAG1}:chains_already_written"] == "1", plan2
    print("PASS f2c: maps per (pack tag, chain) keyed by (round, draw); T8 copies bit-exact, the rest computed; "
          "same key (base, 7) in both packs kept apart; plan == run; rerun sees all written")

    # --- maps-file stamps: an existing / loaded file that does not match the run is refused
    o = run(["f2c", *SEL, *REUSE, "--part", "0", "--n-parts", "1", "--maps-out", maps, "--batch-size", "1"],
            pre=FAKE_F2C, expect_rc=1)
    assert "does not match this run" in o and "batch_size" in o, o[-800:]
    o = run(["f2c", *SEL, "--checkpoint", ckpt, "--part", "0", "--n-parts", "1", "--maps-out", maps, "--batch-size", "2"],
            pre=FAKE_F2C, expect_rc=1)
    assert "does not match this run" in o and "origin" in o, o[-800:]
    stamp_bld = [*SEL, *BLD0, "--part", "0", "--n-parts", "1", "--f2c-maps", maps, "--out-dir", td + "/x"]
    o = run(["build", *stamp_bld, "--f2c-batch-size", "2", "--f2c-checkpoint", ckpt], expect_rc=1)
    assert "does not match this run" in o and "origin" in o, o[-800:]
    o = run(["build", *stamp_bld, "--f2c-batch-size", "1", "--f2c-checkpoint", ckpt, "--f2c-allow-reuse"], expect_rc=1)
    assert "does not match this run" in o and "batch_size" in o, o[-800:]
    o = run(["build", *stamp_bld, "--f2c-checkpoint", ckpt, "--f2c-allow-reuse"], expect_rc=1)
    assert "give --f2c-batch-size" in o, o[-800:]
    open(ckpt, "wb").write(b"another checkpoint")
    o = run(["build", *stamp_bld, "--f2c-batch-size", "2", "--f2c-checkpoint", ckpt, "--f2c-allow-reuse"], expect_rc=1)
    assert "does not match this run" in o and "checkpoint_md5" in o, o[-800:]
    open(ckpt, "wb").write(b"fake checkpoint")
    print("PASS stamps: f2c rerun at another batch size / with reuse off, and a build expecting another batch size, "
          "no reuse, or a changed checkpoint md5 are refused; a confind final-selection build needs --f2c-batch-size")

    # --- build (2 parts) + audit + merge
    parts = os.path.join(td, "parts")
    for part in ("0", "1"):
        run(["build", *SEL, *BLD, "--part", part, "--n-parts", "2", "--f2c-maps", maps, "--out-dir", parts])
    rows, skips, stamps = {}, [], set()
    for part in (0, 1):
        d = torch.load(os.path.join(parts, f"part_{part:04d}.pt"), weights_only=False)
        stamps.add(d["template_source"])
        assert d["template_gate"].endswith("no cap (templates = the final selection)"), d["template_gate"]
        for c in d["chains"]:
            rows[c["stem"]] = [r[3] for r in c["rows"]] if c["native"] is not None else None
        skips += [l.rstrip("\n").split("\t") for l in open(os.path.join(parts, f"skips_{part:04d}.tsv"))]
    src = json.loads(stamps.pop())
    assert not stamps and src["cap"] is None and [p["tag"] for p in src["packs"]] == [TAG0, TAG1], src
    exp_rows = {"a_A": [f"{TAG0}/base/0", f"{TAG0}/P1/100", f"{TAG0}/base/1", f"{TAG0}/base/4"],
                "b_B": [f"{TAG1}/base/7", f"{TAG1}/base/4050"], "c_C": [], "d_D": [], "e_E": [], "f_F": [], "g_G": None}
    assert rows == exp_rows, rows
    sk = {(s, w, r.split(":")[0]) for s, w, r in skips}
    for need in [("a_A", "templates", "final_selection_tm_out_of_window"),
                 ("c_C", "templates", "override_chain_without_override_templates"),
                 ("f_F", "templates", "override_chain_without_override_templates"),
                 ("d_D", "templates", "not_in_final_selection"), ("g_G", "native", "not_in_pack"),
                 ("e_E", f"tpl#{TAG0}/base/0", "ca_clashes_gt_4"), ("e_E", f"tpl#{TAG0}/base/1", "ca_clashes_gt_4"),
                 ("e_E", "templates", "no_template_after_clash_gate")]:
        assert need in sk, (need, sorted(sk))
    assert not {r for _, _, r in sk} & {"no_f2c_maps", "no_f2c_map_for_draw", "usalign_failed"}, sorted(sk)
    print("PASS build: precedence (b_B from the T8b pack only; c_C / f_F with zero T8b templates get NONE although c_C "
          "has original templates), final selection only (a_A base 2, b_B 4051, d_D unselected rows absent), TM window "
          "(a_A base 3 at 0.45 dropped), clash gate (e_E), no cap, pack-tagged keys, source stamp")

    aud = [sys.executable, str(REPO / "scratchpad/t8_audit_parts.py"), "--parts-dir", parts, "--n-parts", "2",
           "--chain-list", chains, "--override-chains", override, "--template-pack-tag", TAG0,
           "--override-pack-tag", TAG1, "--strict"]
    a = subprocess.run(aud, capture_output=True, text=True, cwd=NEUTRAL, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert a.returncode == 0 and "STRICT AUDIT PASS" in a.stdout, a.stdout + a.stderr
    bad = os.path.join(td, "parts_bad")
    shutil.copytree(parts, bad)
    for part in (0, 1):
        p = os.path.join(bad, f"part_{part:04d}.pt")
        d = torch.load(p, weights_only=False)
        for c in d["chains"]:
            if c["stem"] == "b_B":
                c["rows"][0] = (*c["rows"][0][:3], f"{TAG0}/base/7", *c["rows"][0][4:])
        torch.save(d, p)
    a = subprocess.run(aud[:3] + [bad] + aud[4:], capture_output=True, text=True, cwd=NEUTRAL,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    assert a.returncode == 1 and "precedence_violations" in a.stdout, a.stdout + a.stderr
    print("PASS audit --strict: clean parts pass; a b_B row re-keyed to the original pack is caught")

    run(["merge", "--parts-dir", parts, "--n-parts", "2", "--out", td + "/idx.pt", "--eligible-out", td + "/elig.txt"])
    idx = torch.load(td + "/idx.pt", weights_only=False)
    assert json.loads(idx["template_source"]) == src
    assert sorted(open(td + "/elig.txt").read().split()) == ["a_A", "b_B"]
    tpl_ids = [i for i, n in zip(idx["ids"], idx["row_is_native"].tolist()) if not n]
    assert sorted(tpl_ids) == sorted(f"{s}@rw125#{k}" for s in ("a_A", "b_B") for k in exp_rows[s]), tpl_ids
    assert len(set(idx["ids"])) == len(idx["ids"])
    print(f"PASS merge: {len(tpl_ids)} template rows, ids e.g. {tpl_ids[0]}, template_source carried, eligible a_A b_B")

    # --- CLI guards
    o = run(["build", *SEL, *BLD, "--max-templates", "64", "--part", "0", "--n-parts", "1", "--f2c-maps", maps,
             "--out-dir", td + "/x"], expect_rc=1)
    assert "--final-selection has no cap" in o, o[-500:]
    o = run(["build", *SEL[:-10], "--tm-min", "0.5", "--tm-max", "0.9", "--override-pack", p1, *BLD, "--part", "0",
             "--n-parts", "1", "--f2c-maps", maps, "--out-dir", td + "/x"], expect_rc=1)
    assert "give --override-pack, --override-selection and --override-chains together" in o, o[-500:]
    open(td + "/short_override.txt", "w").write("c_C\n")  # b_B is in the override selection but not in the list
    o = run(["build", *[td + "/short_override.txt" if a == override else a for a in SEL], *BLD, "--part", "0",
             "--n-parts", "1", "--f2c-maps", maps, "--out-dir", td + "/x"], expect_rc=1)
    assert "override-selection chains are not in --override-chains" in o, o[-500:]
    o = run(["f2c", "--chain-list", chains, "--processed-dir", td, "--template-pack", p0, "--tm-min", "0.5", "--tm-max",
             "0.9", *REUSE, "--part", "0", "--n-parts", "1", "--maps-out", td + "/x"], pre=FAKE_F2C, expect_rc=1)
    assert "--reuse-maps applies to --final-selection mode" in o, o[-500:]
    print("PASS CLI guards: --max-templates with --final-selection, partial --override-*, override selection outside "
          "the list, --reuse-maps outside final-selection mode -> refused")

    # --- T8 pack-mode path: this checkout == HEAD (same parts, same merged index)
    head = os.path.join(td, "head")
    os.makedirs(head)
    subprocess.run(f"git -C {REPO} archive HEAD proteinfoundation | tar -x -C {head}", shell=True, check=True)
    T8 = ["--chain-list", chains, "--processed-dir", os.path.join(td, "processed"), "--template-pack", p0,
          "--max-templates", "2", "--tm-min", "0.5", "--tm-max", "0.9"]
    for name, pp in (("new", REPO), ("head", head)):
        for part in ("0", "1"):
            run(["build", *T8, *BLD0, "--part", part, "--n-parts", "2", "--f2c-maps", t8maps,
                 "--out-dir", f"{td}/t8_{name}"], pythonpath=pp)
        run(["merge", "--parts-dir", f"{td}/t8_{name}", "--n-parts", "2", "--out", f"{td}/t8_{name}.pt"], pythonpath=pp)
    for part in (0, 1):
        a = torch.load(f"{td}/t8_new/part_{part:04d}.pt", weights_only=False)
        b = torch.load(f"{td}/t8_head/part_{part:04d}.pt", weights_only=False)
        assert deep_eq(a, b) and "template_source" not in a, f"T8 part {part} differs from HEAD"
        assert open(f"{td}/t8_new/skips_{part:04d}.tsv").read() == open(f"{td}/t8_head/skips_{part:04d}.tsv").read()
    a, b = torch.load(f"{td}/t8_new.pt", weights_only=False), torch.load(f"{td}/t8_head.pt", weights_only=False)
    assert deep_eq(a, b) and "template_source" not in a
    n_t8 = int((~a["row_is_native"]).sum())
    assert n_t8 > 0 and any("#" in i and "/" not in i for i in a["ids"])
    print(f"PASS T8 pack mode unchanged vs HEAD: parts, skip logs and merged index equal ({n_t8} template rows, "
          f"cap 2 + gate exercised)")
    shutil.rmtree(td)
    shutil.rmtree(NEUTRAL)
    print("ALL PASS")


if __name__ == "__main__":
    main()
