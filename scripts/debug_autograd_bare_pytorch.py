"""Bare-PyTorch reproducer for the no_grad-then-grad autograd disconnect bug.

Usage:
  python scripts/debug_autograd_bare_pytorch.py                # single-iter A vs B (fp32)
  python scripts/debug_autograd_bare_pytorch.py --bf16         # single-iter with bf16 autocast
  python scripts/debug_autograd_bare_pytorch.py --multi-iter N [--bf16]
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.utils.checkpoint as _ckpt_mod


def _noop_checkpoint(fn, *args, **kwargs):
    for k in ("use_reentrant", "preserve_rng_state", "context_fn",
              "determinism_check", "debug"):
        kwargs.pop(k, None)
    return fn(*args, **kwargs)


_ckpt_mod.checkpoint = _noop_checkpoint
torch.utils.checkpoint.checkpoint = _noop_checkpoint

from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3
import proteinfoundation.nn.protein_transformer as _pt_mod
_pt_mod.checkpoint = _noop_checkpoint


NN_KWARGS_PATH = "/home/ubuntu/proteina/scripts/_debug_nn_kwargs.json"
BATCH_PATH = "/tmp/sample_batch.pt"

NINE_DISCONNECTED = [
    "cond_factory._individual_factory.feat_creators.2.linear_embed.weight",
    "cond_factory._individual_factory.projections.dssp_sc.weight",
    "cond_factory._individual_factory.projections.fold_emb.weight",
    "cond_factory._individual_factory.projections.time_emb.weight",
    "init_repr_factory._individual_factory.projections.chain_break_per_res.weight",
    "pair_repr_builder.init_repr_factory._individual_factory.feat_creators.1.linear_embed.weight",
    "pair_repr_builder.init_repr_factory._individual_factory.projections.contact_map_sc.weight",
    "pair_repr_builder.init_repr_factory._individual_factory.projections.rel_seq_sep.weight",
    "pair_repr_builder.cond_factory._individual_factory.projections.time_emb.weight",
]


def _to_cuda(v):
    if torch.is_tensor(v):
        return v.cuda()
    if isinstance(v, dict):
        return {k: _to_cuda(x) for k, x in v.items()}
    if isinstance(v, list) and v and torch.is_tensor(v[0]):
        return [_to_cuda(x) for x in v]
    return v


def load_model_and_batch():
    with open(NN_KWARGS_PATH) as f:
        nn_kwargs = json.load(f)
    nn_kwargs["use_torch_compile"] = False
    nn_kwargs["use_torch_compile_sc"] = False
    print(f"[setup] instantiating ProteinTransformerAF3 with {len(nn_kwargs)} kwargs", flush=True)
    nn = ProteinTransformerAF3(**nn_kwargs).cuda()
    nn.train()
    n_params = sum(p.numel() for p in nn.parameters() if p.requires_grad)
    print(f"[setup] {n_params/1e6:.2f}M trainable params", flush=True)

    data = torch.load(BATCH_PATH, weights_only=False)
    print(f"[setup] loaded batch from {BATCH_PATH}", flush=True)
    batch = _to_cuda(data["batch"])
    return nn, batch, data


def compute_loss(out):
    parts = []
    for key in ("contact_map_logits", "dssp_logits", "pair_logits"):
        if key in out and torch.is_tensor(out[key]):
            parts.append(out[key].float().pow(2).mean())
    if not parts:
        raise RuntimeError(f"No usable head in nn_out: keys={list(out.keys())}")
    return sum(parts)


def _autocast_ctx(use_bf16):
    if use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    import contextlib
    return contextlib.nullcontext()


def run_scenario(nn, batch, with_sc_first, update_sc=True, sc_key="contact_map_sc", use_bf16=False):
    nn.zero_grad(set_to_none=True)
    batch_for_grad = batch
    if with_sc_first:
        with torch.no_grad(), _autocast_ctx(use_bf16):
            sc_out = nn(batch)
        if update_sc:
            batch_for_grad = dict(batch)
            if "contact_map_pred" in sc_out:
                batch_for_grad[sc_key] = sc_out["contact_map_pred"].detach()
            if "dssp_logits" in sc_out:
                batch_for_grad["dssp_sc"] = torch.softmax(sc_out["dssp_logits"], dim=-1).detach()
    with _autocast_ctx(use_bf16):
        out = nn(batch_for_grad)
        loss = compute_loss(out)
    loss.backward()
    grads = {name: (p.grad.detach().clone() if p.grad is not None else None)
             for name, p in nn.named_parameters()}
    return grads, float(loss.item())


def multi_iter(nn, batch, n_iters, sc_every=2, update_sc=True, use_bf16=False):
    opt = torch.optim.AdamW(nn.parameters(), lr=1e-4)
    any_disconnect = False
    for i in range(n_iters):
        with_sc = (i % sc_every) == 0
        grads, loss = run_scenario(nn, batch, with_sc_first=with_sc, update_sc=update_sc, use_bf16=use_bf16)
        none_set = [n for n in NINE_DISCONNECTED if grads.get(n) is None]
        zero_set = [n for n in NINE_DISCONNECTED
                    if (grads.get(n) is not None and grads[n].abs().sum().item() == 0.0)]
        marker = "SC " if with_sc else "   "
        print(f"  iter {i:2d} {marker} loss={loss:.4f}  None={len(none_set)}/9  zero={len(zero_set)}/9", flush=True)
        if none_set:
            any_disconnect = True
            for n in none_set:
                print(f"        DISCONNECT: {n}", flush=True)
        opt.step()
    return any_disconnect


def main():
    torch.manual_seed(0)
    use_bf16 = "--bf16" in sys.argv
    if use_bf16:
        print("[setup] using bf16 mixed-precision autocast", flush=True)
    nn, batch, _meta = load_model_and_batch()

    if "--multi-iter" in sys.argv:
        i = sys.argv.index("--multi-iter")
        n = int(sys.argv[i + 1])
        print(f"\n[multi-iter] {n} iterations, optimizer step each, SC every other iter ...", flush=True)
        any_disc = multi_iter(nn, batch, n_iters=n, sc_every=2, use_bf16=use_bf16)
        if any_disc:
            print("\nBUG REPRODUCES in bare PyTorch (multi-iter).")
            sys.exit(0)
        else:
            print("\nNo disconnect observed across multi-iter run.")
            sys.exit(1)

    print("\n[scenario A] single grad forward (baseline) ...", flush=True)
    g_A, loss_A = run_scenario(nn, batch, with_sc_first=False, use_bf16=use_bf16)
    print(f"[scenario A] loss = {loss_A:.6f}", flush=True)
    print("\n[scenario B] no_grad forward, then grad forward ...", flush=True)
    g_B, loss_B = run_scenario(nn, batch, with_sc_first=True, use_bf16=use_bf16)
    print(f"[scenario B] loss = {loss_B:.6f}", flush=True)

    print("\n" + "=" * 78)
    print("Per-param comparison for the 9 weights from summary section 4c")
    print("=" * 78)
    disconnect_b_only = []
    for name in NINE_DISCONNECTED:
        if name not in g_A:
            print(f"  MISSING from named_parameters: {name}")
            continue
        a = g_A[name]
        b = g_B[name]
        a_state = "None" if a is None else f"norm={a.norm().item():.4e}"
        b_state = "None" if b is None else f"norm={b.norm().item():.4e}"
        flag = ""
        if (a is not None) and (b is None):
            flag = "  <-- DISCONNECT in B only"
            disconnect_b_only.append(name)
        elif (a is None) and (b is not None):
            flag = "  <-- DISCONNECT in A only (unexpected)"
        print(f"  {name}\n      A: {a_state}\n      B: {b_state}{flag}")

    print("\n" + "=" * 78)
    print(f"Summary: {len(disconnect_b_only)} / {len(NINE_DISCONNECTED)} params disconnect ONLY in scenario B")
    print("=" * 78)
    if disconnect_b_only:
        print("\nBUG REPRODUCES in bare PyTorch:")
        for n in disconnect_b_only:
            print(f"  - {n}")
        sys.exit(0)
    else:
        print("\nNo disconnect observed in bare PyTorch.")
        sys.exit(1)


if __name__ == "__main__":
    main()
