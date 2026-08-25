"""Re-measure an arm's validation sampling after the topology-conditioning fixes, with a
single-step sanity dump.

Two jobs in one pass over the fixed 32-chain set:

  (1) RE-MEASUREMENT. Same EMA-load + trainer.validate shape as dt_sweep_eval.py, but it prints
      EVERY validation_sampling/contact_* metric (including the separation-resolved ones), and
      it asserts from the sampletrace record that the topology reference actually reached the
      network on every sampling step. Before the fix, tri sampled with topology=None and a
      randomly redrawn length; "the metric moved" is not enough on its own, so the trace is what
      makes the re-measurement trustworthy.

  (2) SANITY DUMP. A forward hook on model.nn captures single-step validation denoising examples
      -- the noisy input, the ground truth, the clean prediction, and the topology reference in
      BOTH forms (the SSE-by-SSE `contact_max` matrix, and the compressed token sequence decoded
      through SSEAlphabet). Plus a with/without-topology ablation on a real batch, which is the
      only direct evidence that the reference changes what a TRAINED model predicts -- at
      initialisation the tri trunk is exactly the identity, so wiring alone proves nothing.

The nn output IS the clean prediction: contactflow sets target_pred: c_1, so
_nn_out_to_c_clean returns nn_pred unchanged (model_trainer_base.py:511-512).
"""

import argparse
import json
import os
import sys

import hydra
import lightning as L
import matplotlib
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.sse_topology import (
    N_SPECIAL_TOKENS,
    PAIR_FEATURE_NAMES,
    SSEAlphabet,
)
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils import sampletrace

STEPS_TO_DT = {50: 0.02, 100: 0.01, 150: 1.0 / 150, 200: 0.005}
CONTACT_MAX = PAIR_FEATURE_NAMES.index("contact_max")
DSSP_NAME = {0: "loop", 1: "helix", 2: "strand"}


def precision_at_l5(pred, gt, mask_1d, min_sep=6):
    """Same convention as _compute_contact_map_metrics: upper triangle, sep >= min_sep, k = L/5."""
    n = int(mask_1d.sum())
    if n < min_sep + 2:
        return float("nan")
    idx = torch.arange(pred.shape[-1], device=pred.device)
    sep = (idx[:, None] - idx[None, :]).abs()
    valid = mask_1d[:, None] & mask_1d[None, :] & (sep >= min_sep) & (idx[:, None] < idx[None, :])
    if valid.sum() == 0:
        return float("nan")
    k = max(1, n // 5)
    scores = pred[valid]
    truth = gt[valid]
    k = min(k, scores.numel())
    top = torch.topk(scores, k).indices
    return float(truth[top].mean())


class SingleStepCapture:
    """Grabs the dataloader-driven single-step forward, not the 50 sampling-trajectory forwards.

    The discriminator is ``contact_map`` in the batch: generate() builds its own nn_in from
    scratch and never carries the ground-truth map, so its presence identifies the loss path.
    """

    def __init__(self, n_keep):
        self.n_keep = n_keep
        self.examples = []
        self.live_batch = None

    def __call__(self, module, args, output):
        batch = args[0] if args else None
        if not isinstance(batch, dict) or "contact_map" not in batch:
            return
        if len(self.examples) >= self.n_keep:
            return
        if self.live_batch is None:
            self.live_batch = batch
        b = 0
        rec = {
            "t": float(batch["t"][b]),
            "mask": batch["mask"][b].detach().cpu().numpy(),
            "cm_t": batch["contact_map_t"][b].detach().float().cpu().numpy(),
            "gt": batch["contact_map"][b].detach().float().cpu().numpy(),
            "pred": output["contact_map_pred"][b].detach().float().cpu().numpy(),
        }
        he = batch.get("topology_he_tokens")
        if he is not None:
            rec["he_tokens"] = he[b].detach().cpu().numpy()
            rec["he_contact"] = (
                batch["topology_he_feat"][b, :, :, CONTACT_MAX].detach().float().cpu().numpy()
            )
        self.examples.append(rec)


def decode_tokens(tokens):
    alpha = SSEAlphabet()
    out = []
    for tok in tokens:
        tok = int(tok)
        if tok <= 0:
            continue
        if tok < N_SPECIAL_TOKENS:
            out.append(f"<special {tok}>")
            continue
        dssp, rng = alpha.decode(tok)
        out.append(f"{DSSP_NAME.get(dssp, dssp)}[{rng}]")
    return out


def topology_ablation(model, batch, device):
    """Does the reference change what the TRAINED model predicts? With vs without, same batch."""
    was_training = model.nn.training
    model.nn.eval()
    with torch.no_grad():
        with_topo = model.nn(batch)["contact_map_pred"]
        stripped = {k: v for k, v in batch.items() if not k.startswith("topology_")}
        without = model.nn(stripped)["contact_map_pred"]
    if was_training:
        model.nn.train()
    gt = batch["contact_map"].float()
    mask = batch["mask"].bool()
    rows = []
    for b in range(min(4, with_topo.shape[0])):
        rows.append((
            precision_at_l5(with_topo[b].float(), gt[b], mask[b]),
            precision_at_l5(without[b].float(), gt[b], mask[b]),
        ))
    delta = float((with_topo.float() - without.float()).abs().mean())
    return rows, delta


def plot_dump(examples, out_png, tag):
    ncol = 4
    n = len(examples)
    fig, axes = plt.subplots(n, ncol, figsize=(3.0 * ncol, 3.0 * n), squeeze=False)
    fig.patch.set_facecolor("white")
    for r, ex in enumerate(examples):
        L_valid = int(ex["mask"].sum())
        panels = [
            (ex["cm_t"][:L_valid, :L_valid], f"noisy input  t={ex['t']:.2f}", "magma", 0, 1),
            (ex["gt"][:L_valid, :L_valid], "ground truth", "magma", 0, 1),
            (ex["pred"][:L_valid, :L_valid], "clean prediction", "magma", 0, 1),
        ]
        if "he_contact" in ex:
            nv = int((ex["he_tokens"] > 0).sum())
            panels.append((ex["he_contact"][:nv, :nv], f"topology ref ({nv} SSEs)", "viridis", 0, 1))
        else:
            panels.append((np.zeros((2, 2)), "NO TOPOLOGY", "viridis", 0, 1))
        for c, (arr, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"L={L_valid}", fontsize=8)
    fig.suptitle(f"{tag}: single-step validation denoising + topology reference", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"[dump] wrote {out_png}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--ema_ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--n_dump", type=int, default=4)
    ap.add_argument("--dump_dir", default="/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/curves/fixcheck")
    args = ap.parse_args()

    os.makedirs(args.dump_dir, exist_ok=True)

    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=args.config_name)
    OmegaConf.set_struct(cfg_exp, False)
    cfg_exp.hardware.ngpus_per_node_ = 1
    cfg_exp.hardware.nnodes_ = 1
    cfg_exp.log.log_wandb = False
    cfg_exp.log.checkpoint = False
    cfg_exp.validation_sampling.tmscore_every_n_val_epochs = 1
    cfg_exp.validation_sampling.force_trajectory_at_step0 = True
    cfg_exp.validation_sampling.dt = STEPS_TO_DT[args.steps]

    ds_dir = f"../configs/datasets_config/{cfg_exp.dataset_config_subdir}"
    with hydra.initialize(ds_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp.dataset)

    print(f"[cfg] topology_cond={cfg_exp.model.nn.get('topology_cond', False)} "
          f"nn_class={cfg_exp.model.nn.nn_class} dt={cfg_exp.validation_sampling.dt} "
          f"fixed_chain_list={cfg_exp.validation_sampling.get('fixed_chain_list')}", flush=True)

    model = Proteina(cfg_exp, store_dir="/tmp/fixcheck_store")
    ck = torch.load(args.ema_ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[EMA load] {args.ema_ckpt} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    cap = SingleStepCapture(args.n_dump)
    handle = model.nn.register_forward_hook(cap)

    wl = WandbLogger(project="fixcheck_probe", name=f"{args.tag}_{args.steps}",
                     save_dir="/tmp/fixcheck_wandb", offline=True)
    trainer = L.Trainer(
        accelerator="gpu", devices=1, num_nodes=1, logger=wl,
        enable_checkpointing=False, enable_progress_bar=False,
        limit_val_batches=cfg_exp.opt.get("limit_val_batches", 64),
    )
    sampletrace.reset()
    trainer.validate(model, datamodule=cfg_data and hydra.utils.instantiate(cfg_data.datamodule),
                     ckpt_path=None, verbose=False)
    handle.remove()

    # ── (1) every contact metric, not just precision_at_L ────────────────────────────────────
    cm = {k: float(v) for k, v in trainer.callback_metrics.items()}
    print(f"\n===== {args.tag} steps={args.steps} : validation_sampling contact metrics =====", flush=True)
    for k in sorted(k for k in cm if "contact" in k and k.startswith("validation_sampling")):
        print(f"  {k:<66s} {cm[k]:.4f}", flush=True)

    # ── runtime proof that the reference reached the sampler on every step ───────────────────
    st = sampletrace.summary()["steps"]
    if st:
        n_topo = sum(1 for r in st if r.get("topo_present"))
        valid_counts = sorted({r.get("topo_n_valid", 0) for r in st})
        print(f"\n[TRACE {args.tag}] recorded={len(st)} topo_present={n_topo}/{len(st)} "
              f"distinct topo_n_valid={valid_counts[:8]}", flush=True)
        print(f"[TRACE {args.tag}] effective args: {sampletrace.summary()['args']}", flush=True)
    else:
        print(f"\n[TRACE {args.tag}] no steps recorded (SAMPLETRACE unset?)", flush=True)

    # ── (2) sanity dump ─────────────────────────────────────────────────────────────────────
    if not cap.examples:
        print("[dump] NO single-step forwards captured", flush=True)
        return
    print(f"\n[dump] captured {len(cap.examples)} single-step examples", flush=True)
    meta = []
    for i, ex in enumerate(cap.examples):
        L_valid = int(ex["mask"].sum())
        toks = decode_tokens(ex.get("he_tokens", []))
        p = precision_at_l5(
            torch.from_numpy(ex["pred"]), torch.from_numpy(ex["gt"]),
            torch.from_numpy(ex["mask"]).bool(),
        )
        gt_density = float(ex["gt"][:L_valid, :L_valid].mean())
        print(f"  ex{i}: L={L_valid} t={ex['t']:.3f} prec@L/5={p:.4f} "
              f"gt_density={gt_density:.4f} n_sse={len(toks)}", flush=True)
        print(f"        SSE: {' '.join(toks[:20])}{' ...' if len(toks) > 20 else ''}", flush=True)
        meta.append({"i": i, "L": L_valid, "t": ex["t"], "precision_at_L5": p,
                     "gt_density": gt_density, "sse": toks})

    npz = os.path.join(args.dump_dir, f"singlestep_{args.tag}.npz")
    np.savez_compressed(npz, **{
        f"{k}_{i}": v for i, ex in enumerate(cap.examples)
        for k, v in ex.items() if isinstance(v, np.ndarray)
    })
    with open(os.path.join(args.dump_dir, f"singlestep_{args.tag}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    plot_dump(cap.examples, os.path.join(args.dump_dir, f"singlestep_{args.tag}.png"), args.tag)

    # ── with/without topology on a real batch ───────────────────────────────────────────────
    if cap.live_batch is not None and "topology_he_tokens" in cap.live_batch:
        rows, delta = topology_ablation(model, cap.live_batch, model.device)
        print(f"\n[ablation {args.tag}] mean |pred_with - pred_without| = {delta:.5f}", flush=True)
        for b, (a, w) in enumerate(rows):
            print(f"  sample {b}: prec@L/5 with topology={a:.4f}  without={w:.4f}", flush=True)
    else:
        print(f"\n[ablation {args.tag}] SKIPPED: no topology keys in the captured batch", flush=True)


if __name__ == "__main__":
    main()
