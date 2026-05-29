"""Wrapper around `proteinfoundation.train` that monkey-patches `L.Trainer`
to inject diagnostic callbacks based on `$PROTEINA_DDP_DEBUG`.

Usage: invoked by `scripts/debug_ddp_unused_params.py`. Not meant to be run
directly. argv is forwarded to `proteinfoundation.train.main`.
"""
from __future__ import annotations

import os
import sys
import runpy

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import Callback


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class GradAudit(Callback):
    """D1: after backward, list params with `grad is None` or all-zero grad."""

    def on_after_backward(self, trainer, pl_module):
        step = int(getattr(trainer, "global_step", -1))
        # Only audit every few steps to avoid spam
        if step % 1 != 0:
            return
        none_grads, zero_grads = [], []
        for name, p in pl_module.nn.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                none_grads.append(name)
            else:
                # Use a cheap check: just first element. For zero detection
                # across the whole tensor, sum the abs.
                if p.grad.detach().abs().sum().item() == 0.0:
                    zero_grads.append(name)
        if none_grads:
            print(f"[grad-audit step={step}] None grad ({len(none_grads)}): "
                  f"{none_grads[:10]}{'...' if len(none_grads)>10 else ''}", flush=True)
        if zero_grads:
            print(f"[grad-audit step={step}] zero grad ({len(zero_grads)}): "
                  f"{zero_grads[:10]}{'...' if len(zero_grads)>10 else ''}", flush=True)
        if not none_grads and not zero_grads:
            print(f"[grad-audit step={step}] all {sum(1 for _ in pl_module.nn.parameters() if _.requires_grad)} params have nonzero grad", flush=True)


class ForwardCount(Callback):
    """D2: count `nn.forward` invocations per minibatch."""

    def setup(self, trainer, pl_module, stage):
        # Avoid double-wrapping if setup is called multiple times.
        if getattr(pl_module.nn, "_dbg_wrapped", False):
            return
        pl_module.nn._dbg_fwd_count = 0
        pl_module.nn._dbg_wrapped = True
        orig = pl_module.nn.forward

        def counted(*a, **kw):
            pl_module.nn._dbg_fwd_count += 1
            return orig(*a, **kw)

        pl_module.nn.forward = counted

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pl_module.nn._dbg_fwd_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        print(f"[fwd-count batch={batch_idx}] nn.forward called {pl_module.nn._dbg_fwd_count}×", flush=True)


class DDPStateAudit(Callback):
    """D3: dump DistributedDataParallel state at iteration 8 (where the
    no-deepcopy mode normally fails)."""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Print at batch 7 (right before the failing batch 8 if it would fire)
        if batch_idx not in (7, 8, 9):
            return
        model = trainer.model
        from torch.nn.parallel import DistributedDataParallel
        if not isinstance(model, DistributedDataParallel):
            print(f"[ddp-state batch={batch_idx}] trainer.model is not DDP "
                  f"(type={type(model).__name__})", flush=True)
            return
        print(f"[ddp-state batch={batch_idx}] "
              f"has_rebuilt_buckets={getattr(model, '_has_rebuilt_buckets', '?')} "
              f"static_graph={getattr(model, 'static_graph', '?')} "
              f"find_unused={getattr(model, 'find_unused_parameters', '?')} "
              f"require_backward_sync={getattr(model, 'require_backward_grad_sync', '?')}",
              flush=True)


class ModuleTreeAudit(Callback):
    """D4: print the LightningModule's submodule tree (with trainable-param
    counts) once at setup."""

    def setup(self, trainer, pl_module, stage):
        if getattr(self, "_dumped", False):
            return
        self._dumped = True
        for name, mod in pl_module.named_children():
            n_train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in mod.parameters())
            print(f"[mod-tree] {name}: type={type(mod).__name__} "
                  f"trainable={n_train:,} total={n_total:,}", flush=True)


class DummySibling(Callback):
    """H2: inject a tiny frozen sibling module onto the LightningModule.

    Used to test whether the *shape* of the module tree (presence of a frozen
    sibling, not necessarily a full nn deepcopy) is what makes DDP happy with
    no-deepcopy mode.
    """

    def setup(self, trainer, pl_module, stage):
        if hasattr(pl_module, "_dbg_sibling"):
            return
        sibling = nn.Linear(1, 1)
        for p in sibling.parameters():
            p.requires_grad = False
        pl_module._dbg_sibling = sibling
        print(f"[mod-tree] injected _dbg_sibling (frozen Linear(1,1)) onto Proteina", flush=True)


# ---------------------------------------------------------------------------
# Build callback list from $PROTEINA_DDP_DEBUG
# ---------------------------------------------------------------------------

def build_debug_callbacks(spec: str) -> list[Callback]:
    spec = spec.strip().upper()
    parts = set(spec.split(",")) if spec != "ALL" else {"D1", "D2", "D3", "D4"}
    cbs: list[Callback] = []
    if "D1" in parts:  cbs.append(GradAudit())
    if "D2" in parts:  cbs.append(ForwardCount())
    if "D3" in parts:  cbs.append(DDPStateAudit())
    if "D4" in parts:  cbs.append(ModuleTreeAudit())
    if "H2" in parts:  cbs.append(DummySibling())
    return cbs


# ---------------------------------------------------------------------------
# Monkey-patch L.Trainer to inject our callbacks, then run train.main
# ---------------------------------------------------------------------------

_spec = os.environ.get("PROTEINA_DDP_DEBUG", "")
if _spec:
    _extra_cbs = build_debug_callbacks(_spec)
    print(f"[ddp-debug] injecting {len(_extra_cbs)} callbacks (spec={_spec!r}): "
          f"{[type(c).__name__ for c in _extra_cbs]}", flush=True)
    _orig_Trainer = L.Trainer

    def _Trainer_patched(*args, **kwargs):
        cbs = list(kwargs.get("callbacks") or [])
        cbs.extend(_extra_cbs)
        kwargs["callbacks"] = cbs
        return _orig_Trainer(*args, **kwargs)

    L.Trainer = _Trainer_patched


# Independent of the callback spec: when $PROTEINA_DDP_FIND_UNUSED=1, force
# DDPStrategy(find_unused_parameters=True) so training survives past the
# iter-8 failure point and the grad-audit callback can collect data.
# Avoids changing the strategy string (which would also change launcher
# behavior and break the single-process cluster sampler all_gather).
if os.environ.get("PROTEINA_DDP_FIND_UNUSED", "") == "1":
    from lightning.pytorch.strategies import DDPStrategy as _Orig_DDP
    _orig_init = _Orig_DDP.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs["find_unused_parameters"] = True
        _orig_init(self, *args, **kwargs)

    _Orig_DDP.__init__ = _patched_init
    print("[ddp-debug] forced find_unused_parameters=True on DDPStrategy", flush=True)


# Replace torch.utils.checkpoint with a no-op passthrough when $PROTEINA_DISABLE_CKPT=1.
# Used to test the hypothesis that gradient checkpointing's saved-tensor hook
# mechanism is the proximate contributor to the no-deepcopy autograd
# corruption. If the run completes cleanly, the checkpoint mechanism is
# implicated; if it still fails, look elsewhere.
if os.environ.get("PROTEINA_DISABLE_CKPT", "") == "1":
    import torch.utils.checkpoint as _ckpt_mod

    def _noop_checkpoint(fn, *args, **kwargs):
        # Strip non-call kwargs that torch.utils.checkpoint understands but
        # the underlying fn does not.
        kwargs.pop("use_reentrant", None)
        kwargs.pop("preserve_rng_state", None)
        kwargs.pop("context_fn", None)
        kwargs.pop("determinism_check", None)
        kwargs.pop("debug", None)
        return fn(*args, **kwargs)

    _ckpt_mod.checkpoint = _noop_checkpoint
    # Also patch the bare top-level import used by some files
    import torch
    torch.utils.checkpoint.checkpoint = _noop_checkpoint
    print("[ddp-debug] replaced torch.utils.checkpoint.checkpoint with no-op passthrough",
          flush=True)


# Now run train.main as if invoked via `python -m proteinfoundation.train ...`
# argv[0] is this script's path; everything after gets forwarded.
runpy.run_module("proteinfoundation.train", run_name="__main__")
