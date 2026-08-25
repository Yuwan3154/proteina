"""Inference throughput vs query length L and topology-reference length T, at the MAXIMUM batch.

Two things this measures that the validation wall-clock does not:

  * Inference batch size. The ~23 min per 32-chain validation pass comes from `trainer.validate`
    inheriting the TRAINING dataset config -- `batch_size: 1`, a limit set by tri's N^3 einsum
    UNDER BACKPROP. With no optimizer state, no gradients and no activation graph, inference has
    far more VRAM, so the batch is sized here from measured free memory instead of inherited.
  * Real lengths. Training pads every chain to `max_size` (384); inference does not have to, and
    the fixed-32 val chains are all <= 256 residues.

Batch sizing is measured, not guessed: memory after the model loads gives the parameter
footprint, one B=1 generate gives the per-sample activation cost, and `torch.cuda.mem_get_info`
gives what is actually free. No safety fraction is invented -- if fragmentation makes the
prediction too big the process dies, and the driver retries that point at half.

One process per (arm, L) so an OOM costs one row, not the run, and every point is appended to the
TSV as soon as it is measured.
"""

import argparse
import os
import sys
import time

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.cath_utils import load_cath_mapping
from proteinfoundation.datasets.sse_topology import (
    DSSP_HELIX,
    N_PAIR_FEATURES,
    SSEAlphabet,
)
from proteinfoundation.proteinflow.proteina import Proteina

PROBE_STEPS = 5      # dt = 0.2; per-step cost is what scales, so a short trajectory suffices
TRAJ_STEPS = 50      # the production setting, for the extrapolated column
WARMUP = 1


def synth_topology(B, t_he, t_full, device):
    """A structurally valid reference of the requested size, matching assemble_reference's shapes."""
    alpha = SSEAlphabet()
    tok = alpha.token(DSSP_HELIX, 8)
    return {
        "topology_tokens": torch.full((B, t_full), tok, dtype=torch.long, device=device),
        "topology_pos": torch.linspace(0, 1, t_full, device=device)[None].expand(B, t_full).contiguous(),
        "topology_pos_raw": torch.arange(t_full, dtype=torch.float32, device=device)[None].expand(B, t_full).contiguous(),
        "topology_he_tokens": torch.full((B, t_he), tok, dtype=torch.long, device=device),
        "topology_he_pos": torch.linspace(0, 1, t_he, device=device)[None].expand(B, t_he).contiguous(),
        "topology_he_pos_raw": torch.arange(t_he, dtype=torch.float32, device=device)[None].expand(B, t_he).contiguous(),
        "topology_he_contact": (torch.rand(B, t_he, t_he, device=device) > 0.8).float(),
        "topology_he_feat": torch.rand(B, t_he, t_he, N_PAIR_FEATURES, device=device),
    }


def run_generate(model, cfg_exp, B, L, t_he, t_full, steps, device):
    mask = torch.ones(B, L, dtype=torch.bool, device=device)
    residue_type = torch.randint(0, 20, (B, L), device=device)
    cath_idx = cath_mask = None
    if cfg_exp.training.get("fold_cond", False):
        cath_code_dir = cfg_exp.model.nn.get("cath_code_dir")
        if cath_code_dir is not None:
            _, _, _, nC, nA, nT = load_cath_mapping(cath_code_dir)
            cath_idx = torch.zeros((B, 1, 3), device=device, dtype=torch.long)
            cath_idx[:, 0, 0], cath_idx[:, 0, 1], cath_idx[:, 0, 2] = nC, nA, nT
            cath_mask = torch.zeros((B, 1), device=device, dtype=torch.bool)
    with torch.no_grad():
        return model.generate(
            nsamples=B, n=L, dt=1.0 / steps,
            self_cond=cfg_exp.training.self_cond,
            cath_code=None, cath_code_indices=cath_idx, cath_code_indices_mask=cath_mask,
            residue_type=residue_type, guidance_weight=1.0, autoguidance_ratio=0.0,
            dtype=torch.float32,
            schedule_mode=cfg_exp.validation_sampling.get("schedule_mode", "uniform"),
            schedule_p=float(cfg_exp.validation_sampling.get("schedule_p", 1.0)),
            sampling_mode=cfg_exp.validation_sampling.get("sampling_mode", "sc"),
            sc_scale_noise=float(cfg_exp.validation_sampling.get("sc_scale_noise", 0.45)),
            sc_scale_score=1.0, gt_mode="us", gt_p=1.0, gt_clamp_val=None,
            mask=mask, topology=synth_topology(B, t_he, t_full, device),
            zero_sin_pos_emb=bool(cfg_exp.training.get("zero_sin_pos_emb", False)),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--topo_lens", type=int, nargs="+", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--max_batch_cap", type=int, default=256)
    ap.add_argument("--force_batch", type=int, default=0, help="driver retry after an OOM")
    args = ap.parse_args()

    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=args.config_name)
    OmegaConf.set_struct(cfg_exp, False)
    cfg_exp.log.log_wandb = False
    cfg_exp.log.checkpoint = False

    device = torch.device("cuda")
    model = Proteina(cfg_exp, store_dir="/tmp/speed_store").to(device).eval()
    torch.cuda.synchronize()
    mem_params = torch.cuda.memory_allocated()
    t_full_cap = int(cfg_exp.model.nn.get("max_topology_len", 128))

    if not os.path.exists(args.out_tsv):
        with open(args.out_tsv, "a") as fh:
            fh.write("arm\tL\tT\tbatch\tms_per_step\tms_per_sample_step\t"
                     f"s_traj{TRAJ_STEPS}_batch\ts_traj{TRAJ_STEPS}_per_sample\tpeak_GB\n")

    for t_he in args.topo_lens:
        t_full = min(t_he, t_full_cap)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        run_generate(model, cfg_exp, 1, args.length, t_he, t_full, PROBE_STEPS, device)
        torch.cuda.synchronize()
        per_sample = max(1, torch.cuda.max_memory_allocated() - mem_params)
        free_bytes, _ = torch.cuda.mem_get_info()
        predicted = int(free_bytes // per_sample)
        B = args.force_batch or max(1, min(predicted, args.max_batch_cap))
        print(f"[{args.tag} L={args.length} T={t_he}] per-sample={per_sample/2**30:.3f} GB  "
              f"free={free_bytes/2**30:.1f} GB  -> batch {B}"
              f"{' (forced)' if args.force_batch else f' (predicted {predicted})'}", flush=True)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(WARMUP):
            run_generate(model, cfg_exp, B, args.length, t_he, t_full, PROBE_STEPS, device)
        torch.cuda.synchronize()
        t0 = time.time()
        run_generate(model, cfg_exp, B, args.length, t_he, t_full, PROBE_STEPS, device)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        ms_step = elapsed / PROBE_STEPS * 1e3
        ms_sample_step = ms_step / B
        traj_batch = ms_step * TRAJ_STEPS / 1e3
        traj_sample = ms_sample_step * TRAJ_STEPS / 1e3
        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        with open(args.out_tsv, "a") as fh:
            fh.write(f"{args.tag}\t{args.length}\t{t_he}\t{B}\t{ms_step:.2f}\t{ms_sample_step:.3f}\t"
                     f"{traj_batch:.2f}\t{traj_sample:.4f}\t{peak_gb:.2f}\n")
        print(f"[{args.tag} L={args.length} T={t_he}] batch={B} {ms_step:.1f} ms/step "
              f"({ms_sample_step:.2f} ms/sample/step)  {TRAJ_STEPS}-step traj: "
              f"{traj_batch:.1f} s/batch = {traj_sample:.3f} s/sample  peak {peak_gb:.1f} GB",
              flush=True)


if __name__ == "__main__":
    main()
