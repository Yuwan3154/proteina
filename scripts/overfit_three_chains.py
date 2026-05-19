"""Tiny FAPE/IPA overfit harness for debugging stuck-FAPE training.

Runs ``proteinfoundation.train`` against the Stage-1 v2 contact-DSSP-UDLM-pb
config with Hydra overrides that:

- restrict training to **3 specific chains** via ``opt.overfit_pdb_chains``
  (the propagation lives in train.py — it pushes the list down to
  ``cfg_data.datamodule.overfit_pdb_chains``, which PDBLightningDataModule
  consumes pre-split),
- run on a **single GPU** with **batch_size=3 / accum=1** so one minibatch is
  exactly one pass over all 3 overfit chains,
- enable **gradient instrumentation** (``opt.debug_log_grads=True``) so the
  trainer prints `ipa_linear_s/z.weight |grad|` every 100 steps,
- cap at **2000 optim steps** (~few minutes on a single A100).

Decision rule from the resulting WandB run:

- If ``train/fape_loss`` drops well below 0.97 (the 1 nm clamp) within 2000
  steps, the IPA + projector + frame-scale wiring is **OK** — full Stage-1
  can launch.
- If FAPE stays around 0.97, the wiring is still broken. Inspect the printed
  projector grad norms:
    * non-zero, non-shrinking → optimization landscape problem, try
      ``model.nn.init_ipa_proj=identity_pad`` or ``xavier_small`` (next
      experiment in this same script — pass the override on the CLI).
    * zero or NaN → silent gradient disconnect upstream; needs deeper debug.

The IPA-mask sign bug fixed on 2026-05-18 (BUG #0 in
project_proteina_bugs.md) was the root cause of all previous "stuck at 0.97"
runs. With that fix in place, this overfit should drive FAPE meaningfully
below the clamp limit.

Usage (run on the A100 node, NOT in the sandbox):
    CUTLASS_PATH=/home/ubuntu/openfold/cutlass \\
    /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \\
        scripts/overfit_three_chains.py

To try an alternative projector init:
    CUTLASS_PATH=/home/ubuntu/openfold/cutlass \\
    /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \\
        scripts/overfit_three_chains.py model.nn.init_ipa_proj=identity_pad

### Default chain selection

The defaults are three SHORT (51 res) chains from DIFFERENT PDB entries that
have been **verified** to have valid `contact_map_confind` data in the
processed .pt files:

- ``5c39_A`` (51 res, ProTα N-terminal domain peptide)
- ``1r4a_E``  (51 res, complement C1 C2 light chain fragment)
- ``1hcg_B``  (51 res, coagulation factor Xa light chain peptide)

Shorter chains = faster training + tighter overfit. To override the
selection, set ``OVERFIT_CHAIN_IDS=ID1,ID2,ID3`` in the environment
before running. The dataset filter is enforced via the
``overfit_pdb_chains`` hook in PDBLightningDataModule.
"""

from __future__ import annotations

import os
import sys
import subprocess

# Default: 8 short chains (all processed_length=50) from 8 different PDB
# entries, all confirmed to have valid contact_map_confind data. Selected so
# every chain fits comfortably under PaddingTransform max_size=64 — keeps
# IPA triangle compute small + lets a larger batch_size fit in GPU memory.
DEFAULT_CHAIN_IDS = (
    "1c9p_B", "1fka_R", "1fse_F", "1g2c_A",
    "1guu_A", "1ixs_A", "1jcd_A", "1kfm_A",
)
# Pad to this max length. With chains all len=50, max_size=64 is comfortable
# headroom (round-up to a nice number) and ~16× less compute than 256.
PADDING_MAX_SIZE = 64
chain_ids = tuple(
    s.strip()
    for s in os.environ.get(
        "OVERFIT_CHAIN_IDS", ",".join(DEFAULT_CHAIN_IDS)
    ).split(",")
    if s.strip()
)
if len(chain_ids) < 1:
    print("ERROR: need at least one chain id in OVERFIT_CHAIN_IDS or the default.")
    sys.exit(2)

# Recognized argparse flags for proteinfoundation.train (single-dash with underscore
# per its parser). With chains padded to only 64 residues (vs 256), IPA
# triangle-multiplication compute drops ~64× (O(L^3) → O((L/4)^3)) so we can
# fit a full-batch (= all 8 chains) into one minibatch and still have ample
# GPU memory headroom. One gradient update per epoch sees all 8 overfit chains.
batch_size_per_gpu = len(chain_ids)  # = 8
argparse_args = [
    "--config_name=training_dssp_contact_20M_udlm_pb_v2_stage1",
    "--single",  # forces ngpus_per_node_=1, nnodes_=1 regardless of yaml
    "--show_prog_bar",
    "--nolog",  # disables wandb + checkpointing for a debug run
    "--ngpus_per_node=1",
    f"--batch_size={batch_size_per_gpu}",
    "--accumulate_grad_batches=1",
]

# Hydra-style overrides passed via parse_known_args → cfg_exp compose. The
# overfit_pdb_chains override below propagates through train.py:343 down to
# the dataset config (cfg_data.datamodule), where PDBLightningDataModule's
# pre-split filter (pdb_data.py:1214) restricts to just those IDs and copies
# train→val/test when those splits would be empty.
hydra_overrides = [
    "opt.max_steps=2000",  # full overfit run per the FAPE/IPA debug plan
    "opt.max_epochs=2000",
    f"+opt.overfit_pdb_chains=[{','.join(chain_ids)}]",  # enforced dataset filter
    f"+opt.padding_max_size={PADDING_MAX_SIZE}",         # caps PaddingTransform max_size
    "log.log_every_n_steps=1",
    "log.log_structure_every_n_steps=10000",  # no viz during this short run
    "opt.debug_log_grads=True",
    "+opt.debug_log_grads_every=100",  # cadence — print every 100 steps
    "validation_sampling.tmscore_n_samples=0",  # skip val trajectory
    # Disable torch.compile for fast debug: avoids the ~10-min first-step
    # tracing of the IPA+trunk graph (we want grad-flow signal in seconds,
    # not after compile finishes). Production runs leave this True.
    "+model.nn.use_torch_compile=False",
]

print("Launching FAPE/IPA overfit smoke test:")
print(f"  argparse: {argparse_args}")
print(f"  hydra:    {hydra_overrides}")
print(f"  chains (enforced via overfit_pdb_chains): {list(chain_ids)}")
print()

cmd = [sys.executable, "-m", "proteinfoundation.train", *argparse_args, *hydra_overrides]
env = dict(os.environ)
env.setdefault("CUTLASS_PATH", "/home/ubuntu/openfold/cutlass")
# Skip the per-file confind precompute iteration: contact_map_confind is already
# stored in the .pt files for the max256 dataset, and the existence-check sweep
# over 340 K files burns 20+ minutes of wall time before training even starts.
# Set this env var explicitly to False if .pt files are missing confind maps.
env.setdefault("SKIP_CONFIND_PRECOMPUTE", "1")
sys.exit(subprocess.call(cmd, cwd="/home/ubuntu/proteina", env=env))
