"""Dump ModelCheckpoint's recorded val/loss per saved checkpoint, plus optimizer LR.

wandb's summary only carries the LATEST value, which cannot distinguish "noisy" from "diverging".
The checkpoint callback state holds the monitored score for every top-k save, so it reconstructs
the trajectory at the save points without needing the wandb history API.
"""

import argparse
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    args = ap.parse_args()
    d = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    print(f"global_step = {d.get('global_step')}   epoch = {d.get('epoch')}")
    for k, cb in d.get("callbacks", {}).items():
        if "ModelCheckpoint" not in str(k):
            continue
        best = cb.get("best_k_models", {})
        print(f"\n{len(best)} saved checkpoints, by monitored score:")
        for path, score in sorted(best.items(), key=lambda kv: str(kv[0])):
            print(f"  {str(path).split('/')[-1]:<28s} {float(score):.4f}")
        print(f"  best      = {str(cb.get('best_model_path','')).split('/')[-1]} "
              f"({float(cb.get('best_model_score', float('nan'))):.4f})")
        print(f"  last seen = {float(cb.get('current_score', float('nan'))):.4f}")

    for i, og in enumerate(d.get("optimizer_states", [{}])):
        lrs = {g.get("lr") for g in og.get("param_groups", [])}
        print(f"\noptimizer[{i}] lr = {lrs}")
    for i, sc in enumerate(d.get("lr_schedulers", [])):
        print(f"scheduler[{i}] last_lr = {sc.get('_last_lr')}  last_epoch = {sc.get('last_epoch')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
