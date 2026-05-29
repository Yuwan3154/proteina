"""Bisect step: wrap bare ProteinTransformerAF3 in minimal Lightning + pl.Trainer
(no DDP, no callbacks) and see if SC double-forward triggers the autograd
disconnect.

Order of bisect:
  bare PyTorch multi-iter        --> NO REPRO  (debug_autograd_bare_pytorch.py)
  + LightningModule + pl.Trainer --> THIS SCRIPT
  + DDP                          --> debug_autograd_lightning.py --ddp
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

import lightning as L
from torch.utils.data import Dataset, DataLoader

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


class OneBatchDataset(Dataset):
    """Yields the SAME captured batch every iteration. Length determines epoch size."""
    def __init__(self, batch, n=64):
        self.batch = batch
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, _idx):
        # Lightning's DataLoader will collate; pass through identity.
        return self.batch


def identity_collate(items):
    # Items is a list of dicts. All identical. Just return the first.
    return items[0]


def load_batch():
    data = torch.load(BATCH_PATH, weights_only=False)
    batch = data["batch"]
    return batch


def compute_loss(out):
    parts = []
    for key in ("contact_map_logits", "dssp_logits", "pair_logits"):
        if key in out and torch.is_tensor(out[key]):
            parts.append(out[key].float().pow(2).mean())
    return sum(parts)


class MiniProteina(L.LightningModule):
    """Minimal LightningModule that mimics _set_self_cond + predict_clean."""

    def __init__(self):
        super().__init__()
        with open(NN_KWARGS_PATH) as f:
            nn_kwargs = json.load(f)
        nn_kwargs["use_torch_compile"] = False
        nn_kwargs["use_torch_compile_sc"] = False
        self.nn = ProteinTransformerAF3(**nn_kwargs)
        self.sc_every = 2  # SC active on even iters

    def training_step(self, batch, batch_idx):
        # Mirror _set_self_cond: ensure SC keys present (zeros to start)
        sc_key = "contact_map_sc"
        # batch already has these keys (captured); just ensure zeros for SC inactive case
        with_sc = (int(batch_idx) % self.sc_every) == 0
        if with_sc:
            with torch.no_grad():
                sc_out = self.nn(batch)
            # Mirror the fix: clear autocast cache (PyTorch #105211)
            torch.clear_autocast_cache()
            new_batch = dict(batch)
            if "contact_map_pred" in sc_out:
                new_batch[sc_key] = sc_out["contact_map_pred"].detach()
            if "dssp_logits" in sc_out:
                new_batch["dssp_sc"] = torch.softmax(sc_out["dssp_logits"], dim=-1).detach()
            batch = new_batch
        out = self.nn(batch)
        loss = compute_loss(out)
        # Audit grads after backward via on_after_backward
        self._dbg_with_sc = with_sc
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

    def on_after_backward(self):
        step = int(getattr(self, "global_step", -1))
        with_sc = getattr(self, "_dbg_with_sc", None)
        none_set, zero_set = [], []
        for name, p in self.nn.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                none_set.append(name)
            elif p.grad.detach().abs().sum().item() == 0.0:
                zero_set.append(name)
        marker = "SC " if with_sc else "   "
        nine_none = [n for n in NINE_DISCONNECTED if n in none_set]
        print(f"  step={step:2d} {marker}  None={len(none_set):3d} (9-disc={len(nine_none)})  zero={len(zero_set):3d}", flush=True)
        if nine_none:
            for n in nine_none:
                print(f"        DISCONNECT (9c): {n}", flush=True)


def main():
    use_ddp = "--ddp" in sys.argv
    max_steps = 16
    print(f"[setup] max_steps={max_steps}  use_ddp={use_ddp}", flush=True)

    L.seed_everything(0, workers=True)
    batch = load_batch()
    ds = OneBatchDataset(batch, n=64)
    loader = DataLoader(ds, batch_size=1, collate_fn=identity_collate, num_workers=0)

    model = MiniProteina()

    trainer_kwargs = dict(
        max_steps=max_steps,
        max_epochs=-1,
        accumulate_grad_batches=2,
        devices=1,
        accelerator="gpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        precision="bf16-mixed",  # matches train.py:265
    )
    if use_ddp:
        trainer_kwargs["strategy"] = "ddp_find_unused_parameters_true"
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(model, loader)


if __name__ == "__main__":
    main()
