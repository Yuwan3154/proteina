import argparse
import os

import torch
from hydra import compose, initialize_config_dir

from proteinfoundation.proteinflow.proteina import Proteina


def _n(mod) -> int:
    return sum(p.numel() for p in mod.parameters())


def main():
    parser = argparse.ArgumentParser(description="Count model params for an experiment config (no datamodule/training).")
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--config_dir", default=None, help="defaults to <repo>/configs/experiment_config")
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = args.config_dir or os.path.join(repo, "configs", "experiment_config")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.config_name)

    model = Proteina(cfg, store_dir="/tmp/count_params")

    total = _n(model)
    nn_total = _n(model.nn)
    ipa = _n(model.nn.coors_3d_decoder) if getattr(model.nn, "coors_3d_decoder", None) is not None else 0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print(f"config            : {args.config_name}")
    print(f"TOTAL params      : {total:,}")
    print(f"  nn (trunk+heads+ipa): {nn_total:,}")
    print(f"  - IPA coors_3d_decoder: {ipa:,}")
    print(f"  - nn minus IPA        : {nn_total - ipa:,}")
    print(f"TRAINABLE (post-freeze): {trainable:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
