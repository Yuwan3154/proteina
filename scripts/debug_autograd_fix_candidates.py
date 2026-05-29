"""Verify the fix candidates for the autocast-cache-leak bug.

Hypothesis: torch.autocast caches casts of weights to bf16. If the cache is
populated under no_grad (during the SC pass) and then the same weight is
accessed in a grad-enabled forward without clearing the cache, the cached
bf16 tensor (requires_grad=False) is reused, disconnecting the original
fp32 weight from the autograd graph.

Three candidates tested here:
  baseline   : single outer autocast wrapping both forwards            (should DISCONNECT)
  fix-cache  : torch.clear_autocast_cache() after the SC no_grad block (should be CLEAN)
  fix-disable: SC body wrapped in autocast(cache_enabled=False)        (should be CLEAN)
  fix-nested : SC body wrapped in a fresh autocast(bf16) -- separate ctx (should be CLEAN)
"""
from __future__ import annotations

import json
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

NINE = [
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


def compute_loss(out):
    parts = []
    for key in ("contact_map_logits", "dssp_logits", "pair_logits"):
        if key in out and torch.is_tensor(out[key]):
            parts.append(out[key].float().pow(2).mean())
    return sum(parts)


def run(mode, nn, batch):
    sc_key = "contact_map_sc"
    nn.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if mode == "fix-disable":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False), torch.no_grad():
                sc_out = nn(batch)
        elif mode == "fix-nested":
            # Re-enter a fresh autocast context (new cache scope)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
                sc_out = nn(batch)
        else:
            with torch.no_grad():
                sc_out = nn(batch)
        if mode == "fix-cache":
            torch.clear_autocast_cache()

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
    n_none_total = sum(1 for g in grads.values() if g is None)
    n_disc_9 = sum(1 for n in NINE if grads.get(n) is None)
    return n_none_total, n_disc_9


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

    print(f"{'mode':<14}  {'total_None':>10}  {'9-disc':>8}")
    print("-" * 40)
    for mode in ("baseline", "fix-cache", "fix-disable", "fix-nested"):
        n_none, n9 = run(mode, nn, batch)
        flag = " <-- CLEAN" if n9 == 0 else " <-- BUG"
        print(f"{mode:<14}  {n_none:>10}  {n9:>8}{flag}")


if __name__ == "__main__":
    main()
