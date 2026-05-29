"""Test: wrap BOTH SC forward (no_grad) and grad forward in a SINGLE outer
torch.autocast(bf16) context. This mirrors how Lightning's bf16-mixed
PrecisionPlugin wraps the entire training_step.

If this reproduces the disconnect, the bug is autocast-cache-related: a
cached bf16 tensor from the SC pass (created under no_grad, so it has
requires_grad=False) is reused in the grad pass, breaking the autograd
graph.

Compare:
  debug_autograd_bare_pytorch.py --bf16   (autocast around EACH forward) - no repro
  this script                              (single outer autocast)         - ?
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


def main():
    torch.manual_seed(0)
    with open(NN_KWARGS_PATH) as f:
        nn_kwargs = json.load(f)
    nn_kwargs["use_torch_compile"] = False
    nn_kwargs["use_torch_compile_sc"] = False
    nn = ProteinTransformerAF3(**nn_kwargs).cuda()
    nn.train()
    data = torch.load(BATCH_PATH, weights_only=False)
    batch = _to_cuda(data["batch"])
    sc_key = "contact_map_sc"

    def compute_loss(out):
        parts = []
        for key in ("contact_map_logits", "dssp_logits", "pair_logits"):
            if key in out and torch.is_tensor(out[key]):
                parts.append(out[key].float().pow(2).mean())
        return sum(parts)

    print("[scenario X] single OUTER autocast wrapping both SC + grad forwards", flush=True)
    nn.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            sc_out = nn(batch)
        new_batch = dict(batch)
        if "contact_map_pred" in sc_out:
            new_batch[sc_key] = sc_out["contact_map_pred"].detach()
        if "dssp_logits" in sc_out:
            new_batch["dssp_sc"] = torch.softmax(sc_out["dssp_logits"], dim=-1).detach()
        out = nn(new_batch)
        loss = compute_loss(out)
    loss.backward()
    grads = {name: (p.grad.detach().clone() if p.grad is not None else None)
             for name, p in nn.named_parameters()}

    print()
    print("Per-param result for the 9 §4c params:")
    n_disc = 0
    for name in NINE_DISCONNECTED:
        g = grads.get(name)
        if g is None:
            print(f"  DISCONNECT: {name}")
            n_disc += 1
        else:
            print(f"  OK ({g.norm().item():.4e}): {name}")
    print(f"\nTotal None: {sum(1 for g in grads.values() if g is None)} / {len(grads)}  (9-disc: {n_disc}/9)")


if __name__ == "__main__":
    main()
