"""D36 W11: forensics of the AF2Rank cuEquivariance-vs-plain kernel difference (read-only w.r.t. production).

--mode e1     one process: 6UCK_A native template ('none' cell) through the production featurizer (OpenFoldAF2Rank
              ._featurize, seed 0) and forward, per model x kernel set; 3 in-process forwards; saves batch + outputs.
--mode e1cmp  bitwise comparison of the e1 runs (across processes and in-process).
--mode e2     same input batch (W4-diag D36ab batches) through arms {plain, plain_nochunk, plain_tf32, cueq, cueq_attn,
              cueq_mult, cueq_fix}; forward hooks record every listed module's output per recycle; per-module
              max/mean |diff| vs the plain arm; end-to-end pTM / pLDDT / TM. Also saves real op inputs for e3.
--mode e3     op unit tests (triangle attention start/end, triangle multiplication out/in; template + evoformer weights)
              cuEq vs plain vs a float64 reference, masks {ones, pad67, rand50}, chunking, TF32; the in-place return
              contract of the multiplication; TemplatePairStack block tests (N_templ 1/4, inplace_safe on/off).
cueq_fix applies the PROPOSED openfold fix only inside this process (z += update in place); production is untouched.
"""
import argparse
import copy
import csv
import glob
import os
import re
import shutil

import cuequivariance_ops_torch
import cuequivariance_torch
import openfold
import torch
from cuequivariance_ops_torch.triangle_attention import CUEQ_TRIATTN_FALLBACK_THRESHOLD
from cuequivariance_ops_torch.triangle_multiplicative_update import CUEQ_TRIMUL_FALLBACK_THRESHOLD
from openfold.model.triangular_attention import TriangleAttention
from openfold.model.triangular_multiplicative_update import BaseTriangleMultiplicativeUpdate

from proteinfoundation.prediction_pipeline.af2rank_openfold_scorer import (
    OpenFoldAF2Rank, _detect_chain_in_cif, get_ca_coords_from_structure, tmscore)
from proteinfoundation.utils.openfold_inference import OpenFoldTemplateInference

PARAMS = "/orcd/scratch/orcd/011/chenxiou/params"
BATCHES = "/orcd/scratch/orcd/011/chenxiou/d36_af2rank_mask/w4diag_20261121/out_29943699/batches"
NAT_W4 = "/orcd/scratch/orcd/011/chenxiou/d36u/natives"
TEMPL = "/orcd/scratch/orcd/011/chenxiou/d36_af2rank_mask/w4_20261121/templates"
E1_PID = "6UCK_A"
E2_CASES = (("6UCK_A", "none", False), ("8AUC_B", "both", True))  # (pid, cell, light recording)
PAD_VALID = 67  # 6UCK_A query length: the valid prefix in the padded-mask tests
CHUNK = 4  # openfold config globals.chunk_size
FIX_CALLS = [0]  # times the proposed-fix branch actually ran
W4DIAG = {  # forwards_w4.csv (job 29943699), features=d36ab, model_1_ptm: (pTM, TM)
    ("6UCK_A", "none", "plain"): (0.4629422724246979, 0.8615), ("6UCK_A", "none", "cueq"): (0.2753694951534271, 0.3948),
    ("8AUC_B", "both", "plain"): (0.6645792722702026, 0.9443), ("8AUC_B", "both", "cueq"): (0.37873244285583496, 0.4552)}
ARMS = {  # name: (cuEq attention, cuEq multiplicative update, chunking on, TF32 matmul, proposed fix)
    "plain": (False, False, True, False, False),
    "plain_nochunk": (False, False, False, False, False),
    "plain_tf32": (False, False, True, True, False),
    "cueq": (True, True, True, False, False),
    "cueq_attn": (True, False, True, False, False),
    "cueq_mult": (False, True, True, False, False),
    "cueq_fix": (True, True, True, False, True),
}
CHILD = r"(tri_att_start|tri_att_end|tri_mul_out|tri_mul_in|pair_transition)"
MODULE_RE = re.compile(
    r"^(input_embedder|recycling_embedder|extra_msa_embedder|structure_module|aux_heads"
    r"|template_embedder(\.template_pair_embedder|\.template_pointwise_att|\.template_single_embedder"
    rf"|\.template_pair_stack(\.blocks\.\d+(\.{CHILD})?)?)?"
    rf"|extra_msa_stack(\.blocks\.\d+(\.pair_stack\.{CHILD})?)?"
    rf"|evoformer(\.blocks\.\d+(\.pair_stack\.{CHILD})?)?)$")
CAPTURE = re.compile(rf"^(template_embedder\.template_pair_stack(\.blocks\.0\.{CHILD})?|evoformer\.blocks\.0\.pair_stack\.{CHILD})$")


def numerics_state():
    return {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuequivariance": cuequivariance_torch.__version__,
            "cuequivariance_ops_torch": (cuequivariance_ops_torch.__version__, cuequivariance_ops_torch.__git_commit__),
            "openfold": openfold.__file__,
            "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32, "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32, "cudnn.benchmark": torch.backends.cudnn.benchmark,
            "cudnn.deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "CUEQ_env": {k: v for k, v in os.environ.items() if k.startswith("CUEQ")},
            "CUEQ_TRIATTN_FALLBACK_THRESHOLD": CUEQ_TRIATTN_FALLBACK_THRESHOLD,
            "CUEQ_TRIMUL_FALLBACK_THRESHOLD": CUEQ_TRIMUL_FALLBACK_THRESHOLD}


def native_ca(pid):
    entry, chain = pid.rsplit("_", 1)
    cif = os.path.join(NAT_W4, f"{entry}.cif")
    return get_ca_coords_from_structure(cif, _detect_chain_in_cif(cif, chain))


def readout(out, ref_ca):
    pred_ca = out["final_atom_positions"][:, 1, :].detach().cpu().numpy()
    assert pred_ca.shape[0] == ref_ca.shape[0], (pred_ca.shape, ref_ca.shape)
    tm = tmscore(ref_ca, pred_ca, tmscore_exe=shutil.which("USalign"))["tms"]
    return {"ptm": float(out["ptm_score"].item()), "plddt": float(out["plddt"].mean().item()) / 100.0, "tm": tm}


def forward(model, batch):
    b = {k: v.clone().cuda() for k, v in batch.items()}
    with torch.no_grad():
        return model(b)


def tensors_of(out):
    if torch.is_tensor(out):
        return [("0", out)]
    if isinstance(out, (tuple, list)):
        return [(str(i), t) for i, t in enumerate(out) if torch.is_tensor(t)]
    if isinstance(out, dict):
        return [(k, v) for k, v in out.items() if torch.is_tensor(v)]
    return []


def diff_stats(x, ref):
    assert x.shape == ref.shape, (x.shape, ref.shape)
    xc, rc_ = x.detach().cpu(), ref.detach().cpu()
    x64, r64 = xc.double(), rc_.double()
    d = (x64 - r64).abs()
    ref_max = r64.abs().max().item()
    return {"equal": bool(torch.equal(xc, rc_)), "max_abs": d.max().item(),
            "mean_abs": d.mean().item(), "ref_absmax": ref_max,
            "rel_max": d.max().item() / ref_max if ref_max > 0 else float("nan"),
            "rel_fro": (d.norm() / r64.norm()).item() if r64.norm().item() > 0 else float("nan")}


def fix_trimul(module):
    """PROPOSED openfold fix, applied to one instance: in inference (inplace_safe and _add_with_inplace) the cuEq branch
    must add its update INTO z (callers such as TemplatePairStackBlock rely on z being updated in place)."""
    orig = module.forward

    def fixed(z, mask=None, inplace_safe=False, use_cuequivariance_multiplicative_update=False,
              _add_with_inplace=False, _inplace_chunk_size=256):
        if use_cuequivariance_multiplicative_update and inplace_safe and _add_with_inplace:
            FIX_CALLS[0] += 1
            z += orig(z, mask=mask, inplace_safe=False, use_cuequivariance_multiplicative_update=True,
                      _add_with_inplace=False)
            return z
        return orig(z, mask=mask, inplace_safe=inplace_safe,
                    use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
                    _add_with_inplace=_add_with_inplace, _inplace_chunk_size=_inplace_chunk_size)
    module.forward = fixed


def apply_fix(model):
    n = 0
    for m in model.modules():
        if isinstance(m, BaseTriangleMultiplicativeUpdate):
            fix_trimul(m)
            n += 1
    return n


def build(model_name, attn, mult, chunk, fix):
    inf = OpenFoldTemplateInference(model_name=model_name, jax_params_path=os.path.join(PARAMS, f"params_{model_name}.npz"),
                                    use_deepspeed_evoformer_attention=False, max_recycling_iters=6,
                                    use_cuequivariance_attention=attn, use_cuequivariance_multiplicative_update=mult)
    if not chunk:
        inf.model.globals.chunk_size = None
    if fix:
        print(f"[fix] wrapped {apply_fix(inf.model)} triangle-multiplication modules", flush=True)
    for p in inf.model.parameters():
        p.requires_grad_(False)
    return inf


class Recorder:
    """Forward hooks on the selected modules; recycle index = number of input_embedder calls so far - 1."""

    def __init__(self, model, light, ref=None, capture=False):
        self.light, self.ref, self.capture = light, ref, capture
        self.recycle, self.seq, self.counts = -1, 0, {}
        self.store, self.rows, self.captured, self.matched = {}, [], {}, set()
        self.handles = [model.input_embedder.register_forward_pre_hook(self._tick)]
        for name, mod in model.named_modules():
            if MODULE_RE.match(name):
                self.handles.append(mod.register_forward_hook(self._hook(name), with_kwargs=True))
            if capture and CAPTURE.match(name):
                self.handles.append(mod.register_forward_pre_hook(self._pre(name), with_kwargs=True))

    def _tick(self, mod, args):
        self.recycle += 1
        self.counts = {}

    def _keep(self, name):
        if not self.light:
            return True
        if ".pair_stack." in name:
            return False
        if self.recycle == 0:
            return True
        return not (re.search(r"\.blocks\.\d+", name) and not re.match(r"^template_embedder\.template_pair_stack\.blocks\.\d+$", name))

    def _pre(self, name):
        def f(mod, args, kwargs):
            if self.recycle != 0:
                return
            k = self.counts.get("pre:" + name, 0)
            self.counts["pre:" + name] = k + 1
            if k != 0 or name.endswith("pair_transition"):
                return
            x = args[0] if args else kwargs.get("z", kwargs.get("x", kwargs.get("t")))
            mask = args[1] if len(args) > 1 else kwargs.get("mask", kwargs.get("pair_mask"))
            self.captured[f"{name}#{k}"] = {"x": x.detach().cpu().clone(),
                                            "mask": None if mask is None else mask.detach().cpu().clone()}
        return f

    def _hook(self, name):
        def f(mod, args, kwargs, out):
            k = self.counts.get(name, 0)
            self.counts[name] = k + 1
            self.seq += 1
            if not self._keep(name):
                return
            inp = args[0] if args else kwargs.get("z", kwargs.get("x"))
            alias = ""
            if isinstance(mod, BaseTriangleMultiplicativeUpdate) and torch.is_tensor(inp) and torch.is_tensor(out):
                alias = out.data_ptr() == inp.data_ptr()
            for idx, t in tensors_of(out):
                key = (self.recycle, name, k, idx)
                if self.ref is None:
                    self.store[key] = t.detach().cpu().clone()
                    self.rows.append({"seq": self.seq, "recycle": self.recycle, "module": name, "call": k, "out": idx,
                                      "shape": "x".join(map(str, t.shape)), "aliases_input": alias})
                elif key in self.ref:
                    self.matched.add(key)
                    self.rows.append({"seq": self.seq, "recycle": self.recycle, "module": name, "call": k, "out": idx,
                                      "shape": "x".join(map(str, t.shape)), "aliases_input": alias,
                                      **diff_stats(t, self.ref[key])})
        return f

    def close(self):
        for h in self.handles:
            h.remove()
        if self.ref is None:
            return len(self.store), len(self.store), []
        missing = sorted(set(self.ref) - self.matched)
        return len(self.ref), len(self.matched), missing[:5]


def run_e1(args):
    ref_ca = native_ca(E1_PID)
    entry, chain = E1_PID.rsplit("_", 1)
    state = numerics_state()
    print(f"[numerics] {state}", flush=True)
    for model_name in ("model_1_ptm", "model_2_ptm"):
        for kern in ("plain", "cueq"):
            c = kern == "cueq"
            sc = OpenFoldAF2Rank(os.path.join(NAT_W4, f"{entry}.cif"), chain=chain, model_name=model_name, recycles=6,
                                 use_deepspeed_evoformer_attention=False, use_cuequivariance_attention=c,
                                 use_cuequivariance_multiplicative_update=c, mask_sidechains=False,
                                 usalign_path=shutil.which("USalign"), template_seq_mask="none")
            sc.set_segments(None)
            batch, _, _ = sc._featurize(os.path.join(TEMPL, E1_PID, "native.pdb"), decoy_chain="A", seed=0)
            batch = {k: v.detach().cpu() for k, v in batch.items()}
            outs = [{k: v.detach().cpu() for k, v in forward(sc.model.model, batch).items() if torch.is_tensor(v)}
                    for _ in range(3)]
            tuned = {n: getattr(getattr(sc.model.model, n, None), "chunk_size_tuner", None)
                     for n in ("evoformer", "extra_msa_stack")}
            tuned["template_pair_stack"] = sc.model.model.template_embedder.template_pair_stack.chunk_size_tuner
            tuned = {n: (None if t is None else t.cached_chunk_size) for n, t in tuned.items()}
            r = readout(outs[0], ref_ca)
            print(f"[e1 {args.tag}] {model_name} {kern}: {r} tuned_chunk {tuned}", flush=True)
            torch.save({"batch": batch, "outs": outs, "readout": r, "numerics": state, "tuned_chunk": tuned},
                       os.path.join(args.out, f"e1_{args.tag}_{model_name}_{kern}.pt"))
            del sc
            torch.cuda.empty_cache()
    print(f"W11_E1_{args.tag}_DONE", flush=True)


def run_e1cmp(args):
    files = sorted(glob.glob(os.path.join(args.out, "e1_run*_*.pt")))
    groups = {}
    for f in files:
        tag, rest = os.path.basename(f)[3:-3].split("_", 1)
        groups.setdefault(rest, {})[tag] = torch.load(f, map_location="cpu", weights_only=False)
    rows = []
    for rest, runs in sorted(groups.items()):
        tags = sorted(runs)
        base = runs[tags[0]]
        for t in tags:
            for i in (1, 2):
                for k, v in base["outs"][0].items():
                    rows.append({"case": rest, "comparison": f"{t} fwd0 vs fwd{i} (in-process)", "item": f"out:{k}",
                                 **diff_stats(runs[t]["outs"][i][k], runs[t]["outs"][0][k])})
        for t in tags[1:]:
            for k, v in base["batch"].items():
                rows.append({"case": rest, "comparison": f"{tags[0]} vs {t} (process)", "item": f"batch:{k}",
                             **diff_stats(runs[t]["batch"][k].float(), v.float())})
            for k, v in base["outs"][0].items():
                rows.append({"case": rest, "comparison": f"{tags[0]} vs {t} (process)", "item": f"out:{k}",
                             **diff_stats(runs[t]["outs"][0][k], v)})
        if rest.endswith("_cueq") and rest[:-5] + "_plain" in groups:
            for t in tags:
                for k, v in groups[rest[:-5] + "_plain"][t]["batch"].items():
                    rows.append({"case": rest[:-5], "comparison": f"{t} plain-scorer vs cueq-scorer featurizer",
                                 "item": f"batch:{k}", **diff_stats(runs[t]["batch"][k].float(), v.float())})
        print(f"[e1cmp] {rest}: runs {tags} readouts {[runs[t]['readout'] for t in tags]} "
              f"tuned {[runs[t]['tuned_chunk'] for t in tags]}", flush=True)
    write_csv(os.path.join(args.out, "e1_bitwise.csv"), rows)
    n_bad = sum(not r["equal"] for r in rows)
    print(f"W11_E1CMP rows {len(rows)} not_bitwise_equal {n_bad} groups {len(groups)}", flush=True)


def write_csv(path, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_e2(args):
    models = {a: build("model_1_ptm", at, mu, ch, fx) for a, (at, mu, ch, _, fx) in ARMS.items() if a != "plain_tf32"}
    models["plain_tf32"] = models["plain"]
    print(f"[numerics] {numerics_state()}", flush=True)
    assert not torch.backends.cuda.matmul.allow_tf32
    e2e, mods = [], []
    for pid, cell, light in E2_CASES:
        if pid not in args.e2_pids.split(","):
            continue
        batch = torch.load(os.path.join(BATCHES, f"{pid}_{cell}_model_1_ptm_d36ab.pt"), map_location="cpu", weights_only=False)
        print(f"[e2] {pid} {cell}: template_aatype {tuple(batch['template_aatype'].shape)} "
              f"template_mask {batch['template_mask'][..., 0].tolist()} aatype {tuple(batch['aatype'].shape)}", flush=True)
        ref_ca = native_ca(pid)
        ref = None
        for arm, (_, _, _, tf32, _) in ARMS.items():
            model = models[arm].model
            torch.backends.cuda.matmul.allow_tf32 = tf32
            # untracked warm-up so ChunkSizeTuner trial calls never enter the recorded call indices
            warm = readout(forward(model, batch), ref_ca)
            rec = Recorder(model, light, ref=ref, capture=ref is None)
            FIX_CALLS[0] = 0
            out = forward(model, batch)
            torch.backends.cuda.matmul.allow_tf32 = False
            n_ref, n_matched, missing = rec.close()
            r = readout(out, ref_ca)
            tuned = {n: (None if s.chunk_size_tuner is None else s.chunk_size_tuner.cached_chunk_size)
                     for n, s in (("template_pair_stack", model.template_embedder.template_pair_stack),
                                  ("extra_msa_stack", model.extra_msa_stack), ("evoformer", model.evoformer))}
            e2e.append({"pid": pid, "cell": cell, "model": "model_1_ptm", "arm": arm, **r,
                        "warmup_equal": warm == r, "ref_keys": n_ref, "matched_keys": n_matched,
                        "fix_calls": FIX_CALLS[0], "tuned_chunk": tuned, "w4diag_anchor": W4DIAG.get((pid, cell, arm), "")})
            print(f"[e2] {pid} {cell} {arm}: {r} warmup {warm} keys {n_matched}/{n_ref} missing {missing} "
                  f"fix_calls {FIX_CALLS[0]} tuned {tuned} W4-diag {W4DIAG.get((pid, cell, arm), '-')}", flush=True)
            for row in rec.rows:
                mods.append({"pid": pid, "arm": arm, **row})
            if ref is None:
                ref = rec.store
                torch.save(rec.captured, os.path.join(args.out, f"e3_inputs_{pid}.pt"))
            del out, rec
            torch.cuda.empty_cache()
            write_csv(os.path.join(args.out, "e2_endtoend.csv"), e2e)
            write_csv(os.path.join(args.out, "e2_modules.csv"), mods)
            assert n_matched == n_ref, (arm, n_matched, n_ref, missing)
            assert (FIX_CALLS[0] > 0) == (arm == "cueq_fix"), (arm, FIX_CALLS[0])
        del ref
    print(f"W11_E2_DONE e2e {len(e2e)} module_rows {len(mods)}", flush=True)


def op_call(mod, kind, x, mask, cueq, chunk):
    with torch.no_grad():
        if kind == "att":
            return mod(x, mask=mask, chunk_size=chunk, use_cuequivariance_attention=cueq, inplace_safe=False)
        return mod(x, mask=mask, inplace_safe=False, use_cuequivariance_multiplicative_update=cueq, _add_with_inplace=False)


def op_path(kind, mod, n):
    if kind == "mul":
        return "cueq_triton_kernel" if n > CUEQ_TRIMUL_FALLBACK_THRESHOLD else "cueq_torch_fallback"
    if n <= CUEQ_TRIATTN_FALLBACK_THRESHOLD:
        return "openfold_torch_fallback"
    thr = max(CUEQ_TRIATTN_FALLBACK_THRESHOLD, 200) if mod.c_hidden < 32 else CUEQ_TRIATTN_FALLBACK_THRESHOLD
    return "cueq_cuda_kernel" if n > thr else "cueq_torch_fallback"


def make_mask(kind, prefix, n, g):
    if kind == "ones":
        return torch.ones(prefix + (n, n), device="cuda")
    if kind == "pad67":
        s = torch.zeros(n, device="cuda")
        s[:PAD_VALID] = 1
        return (s[:, None] * s[None, :]).expand(prefix + (n, n)).contiguous()
    return (torch.rand(prefix + (n, n), device="cuda", generator=g) < 0.5).float()  # cuEq docstring example mask


def op_rows(label, mod, kind, prefix, x, mask, mtag, src):
    n = x.shape[-2]
    region = (slice(None),) * len(prefix) + ((slice(0, PAD_VALID), slice(0, PAD_VALID)) if mtag == "pad67" else (slice(None), slice(None)))
    mod64 = copy.deepcopy(mod).double()
    ref = op_call(mod64, kind, x.double(), mask.double(), False, None)
    del mod64
    rows, outs = [], {}
    chunks = (None, CHUNK) if kind == "att" else (None,)
    for impl in ("plain", "cueq"):
        for tf32 in (False, True):
            for ch in chunks:
                torch.backends.cuda.matmul.allow_tf32 = tf32
                o = op_call(mod, kind, x, mask, impl == "cueq", ch)
                torch.backends.cuda.matmul.allow_tf32 = False
                outs[(impl, tf32, ch)] = o
                rows.append({"op": label, "src": src, "N": n, "prefix": "x".join(map(str, prefix)) or "-", "mask": mtag,
                             "impl": impl, "tf32": tf32, "chunk": ch, "expected_path": op_path(kind, mod, n) if impl == "cueq" else "openfold_torch",
                             "vs": "ref64", **diff_stats(o[region], ref[region])})
    o_p, o_c = outs[("plain", False, None)], outs[("cueq", False, None)]
    rows.append({"op": label, "src": src, "N": n, "prefix": "x".join(map(str, prefix)) or "-", "mask": mtag, "impl": "cueq",
                 "tf32": False, "chunk": None, "expected_path": op_path(kind, mod, n), "vs": "plain32_valid", **diff_stats(o_c[region], o_p[region])})
    if mtag == "pad67":
        rows.append({"op": label, "src": src, "N": n, "prefix": "x".join(map(str, prefix)) or "-", "mask": mtag, "impl": "cueq",
                     "tf32": False, "chunk": None, "expected_path": op_path(kind, mod, n), "vs": "plain32_all_incl_padding",
                     **diff_stats(o_c, o_p)})
    if kind == "mul":
        upd = o_p
        for impl in ("plain", "cueq"):
            z = x.clone()
            z0 = z.clone()
            with torch.no_grad():
                o = mod(z, mask=mask, inplace_safe=True, use_cuequivariance_multiplicative_update=impl == "cueq",
                        _add_with_inplace=True)
            rows.append({"op": label, "src": src, "N": n, "prefix": "x".join(map(str, prefix)) or "-", "mask": mtag,
                         "impl": impl, "tf32": False, "chunk": None, "expected_path": "inplace_contract",
                         "vs": "z0+update(plain32)", "aliases_input": o.data_ptr() == z.data_ptr(),
                         "input_modified": not torch.equal(z, z0), **diff_stats(o[region], (z0 + upd)[region])})
    return rows


def stack_rows(stack, stack_fix, mask_trans, t, mask, src, mtag):
    stack64 = copy.deepcopy(stack).double()
    with torch.no_grad():
        ref = stack64(t.double().clone(), mask.double(), chunk_size=None, inplace_safe=False, _mask_trans=mask_trans)
    del stack64
    n = t.shape[-2]
    region = (slice(None), slice(0, PAD_VALID), slice(0, PAD_VALID)) if mtag == "pad67" else (slice(None),) * 3
    rows = []
    variants = (("plain", stack, False, False), ("cueq", stack, True, True), ("cueq_attn", stack, True, False),
                ("cueq_mult", stack, False, True), ("cueq_fix", stack_fix, True, True))
    for impl, s, at, mu in variants:
        for ip in (False, True):
            for ch in (None, CHUNK):
                FIX_CALLS[0] = 0
                with torch.no_grad():
                    o = s(t.clone(), mask, chunk_size=ch, use_cuequivariance_attention=at,
                          use_cuequivariance_multiplicative_update=mu, inplace_safe=ip, _mask_trans=mask_trans)
                rows.append({"test": "template_pair_stack", "src": src, "T": t.shape[0], "N": n, "mask": mtag, "impl": impl,
                             "inplace_safe": ip, "chunk": ch, "fix_calls": FIX_CALLS[0], "vs": "ref64",
                             **diff_stats(o[region], ref[region])})
    return rows


def run_e3(args):
    inf = build("model_1_ptm", False, False, True, False)
    model = inf.model
    print(f"[numerics] {numerics_state()}", flush=True)
    assert not torch.backends.cuda.matmul.allow_tf32
    tps = model.template_embedder.template_pair_stack
    ops = (("tmpl_tri_att_start", tps.blocks[0].tri_att_start, "att", (1,)),
           ("tmpl_tri_att_end", tps.blocks[0].tri_att_end, "att", (1,)),
           ("tmpl_tri_mul_out", tps.blocks[0].tri_mul_out, "mul", (1,)),
           ("tmpl_tri_mul_in", tps.blocks[0].tri_mul_in, "mul", (1,)),
           ("evo_tri_att_start", model.evoformer.blocks[0].pair_stack.tri_att_start, "att", ()),
           ("evo_tri_att_end", model.evoformer.blocks[0].pair_stack.tri_att_end, "att", ()),
           ("evo_tri_mul_out", model.evoformer.blocks[0].pair_stack.tri_mul_out, "mul", ()),
           ("evo_tri_mul_in", model.evoformer.blocks[0].pair_stack.tri_mul_in, "mul", ()))
    for label, mod, kind, _ in ops:
        assert isinstance(mod, TriangleAttention if kind == "att" else BaseTriangleMultiplicativeUpdate), label
        print(f"[e3] {label}: {type(mod).__name__} starting={getattr(mod, 'starting', '-')} c_in={getattr(mod, 'c_in', getattr(mod, 'c_z', '-'))} "
              f"c_hidden={mod.c_hidden}", flush=True)
    g = torch.Generator(device="cuda")
    rows = []
    sizes = args.sizes
    for label, mod, kind, prefix in ops:
        c = mod.c_in if kind == "att" else mod.c_z
        for n in sizes:
            for mtag in ("ones", "pad67", "rand50"):
                if mtag == "pad67" and n <= PAD_VALID:
                    continue
                g.manual_seed(0)
                x = torch.randn(prefix + (n, n, c), device="cuda", generator=g)
                mask = make_mask(mtag, prefix, n, g)
                rows += op_rows(label, mod, kind, prefix, x, mask, mtag, "randn")
                write_csv(os.path.join(args.out, "e3_ops.csv"), rows)
        print(f"[e3] {label} random done ({len(rows)} rows)", flush=True)
    capt_files = sorted(glob.glob(os.path.join(args.out, "e3_inputs_*.pt")))
    cap_map = {"tmpl": "template_embedder.template_pair_stack.blocks.0.", "evo": "evoformer.blocks.0.pair_stack."}
    for f in capt_files:
        pid = os.path.basename(f)[len("e3_inputs_"):-3]
        cap = torch.load(f, map_location="cpu", weights_only=False)
        for label, mod, kind, prefix in ops:
            key = cap_map[label.split("_")[0]] + label.split("_", 1)[1] + "#0"
            assert key in cap, (pid, key, sorted(cap))
            x, mask = cap[key]["x"].cuda(), cap[key]["mask"].cuda()
            rows += op_rows(label, mod, kind, prefix, x, mask, "real", f"captured_{pid}_recycle0")
            write_csv(os.path.join(args.out, "e3_ops.csv"), rows)
        print(f"[e3] {pid} captured-input ops done", flush=True)
    mask_trans = model.config._mask_trans
    tps = copy.deepcopy(tps)
    tps.chunk_size_tuner = None  # explicit chunk 4 in the stack test (the tuned production chunk is covered by e2)
    tps_fix = copy.deepcopy(tps)
    print(f"[e3] fix wrapped {apply_fix(tps_fix)} modules; _mask_trans={mask_trans}", flush=True)
    srows = []
    for T in (1, 4):
        for n in args.stack_sizes:
            for mtag in ("ones", "pad67"):
                if mtag == "pad67" and n <= PAD_VALID:
                    continue
                g.manual_seed(0)
                t = torch.randn((T, n, n, tps.blocks[0].c_t), device="cuda", generator=g)
                mask = make_mask(mtag, (1,), n, g)
                srows += stack_rows(tps, tps_fix, mask_trans, t, mask, "randn", mtag)
                write_csv(os.path.join(args.out, "e3_stack.csv"), srows)
    for f in capt_files:
        pid = os.path.basename(f)[len("e3_inputs_"):-3]
        cap = torch.load(f, map_location="cpu", weights_only=False)
        c0 = cap["template_embedder.template_pair_stack#0"]
        srows += stack_rows(tps, tps_fix, mask_trans, c0["x"].cuda(), c0["mask"].cuda(), f"captured_{pid}_recycle0", "real")
        write_csv(os.path.join(args.out, "e3_stack.csv"), srows)
    print(f"W11_E3_DONE op_rows {len(rows)} stack_rows {len(srows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["e1", "e1cmp", "e2", "e3"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--sizes", type=int, nargs="+", default=[67, 101, 201])  # below / just above the cuEq fallback thresholds 100 and 200
    ap.add_argument("--stack-sizes", type=int, nargs="+", default=[67, 101])
    ap.add_argument("--e2-pids", default=",".join(p for p, _, _ in E2_CASES))
    args = ap.parse_args()
    assert os.path.realpath(os.environ["AF2RANK_PARAMS_DIR"]) == os.path.realpath(PARAMS)
    assert not torch.backends.cuda.matmul.allow_tf32
    os.makedirs(args.out, exist_ok=True)
    {"e1": run_e1, "e1cmp": run_e1cmp, "e2": run_e2, "e3": run_e3}[args.mode](args)


if __name__ == "__main__":
    main()
