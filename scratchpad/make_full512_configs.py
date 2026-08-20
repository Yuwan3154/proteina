import pathlib
import re

CFG = pathlib.Path("configs/experiment_config")
DATASET = "pdb_train_contact-confind-topology_S25_max512_purge-test_cutoff-190828"

# (source overfit config, destination, run name)
JOBS = [
    ("training_contact_tri_overfit_v1.yaml", "training_contact_tri_full512_v1.yaml", "tri_full512"),
    ("training_contact_hier_topology_overfit_v1.yaml",
     "training_contact_hier_topology_full512_v1.yaml", "localattn_full512"),
]

HEADER = """# FULL dataset at max length 512, replacing the overfit-2000 regime.
# Both overfit arms peaked on validation at epoch 63 (local_attn) / 119 (conv_next) and then
# degraded 2.4x while training loss stayed flat -- the subset is too small to learn a general
# contact map, so the regime moved to the full split.
# conv_next is ABANDONED: local_attn matched or beat it on every validation metric.
"""

for src_name, dst_name, run in JOBS:
    src = (CFG / src_name).read_text()

    src = HEADER + src

    src = re.sub(r"^run_name_: .*$", f"run_name_: {run}", src, flags=re.M)
    src = re.sub(r"^dataset: .*$", f"dataset: {DATASET}", src, flags=re.M)

    # 500 -> 1000 per directive; validation peaked far earlier than 500, so this exists to give the
    # FULL dataset room, not because 500 was reached productively on the subset.
    src = re.sub(r"^(\s*)max_epochs: 500.*$",
                 r"\1max_epochs: 1000   # full dataset; the overfit runs peaked by epoch ~120",
                 src, flags=re.M)

    # Effective batch 32 is the invariant shared by all three arms. Both runs are 2-GPU DDP, so
    # accumulate = 32 / (2 * batch_size). Written for batch_size 1 and MUST be recomputed if the
    # VRAM probe shows a larger batch fits.
    src = re.sub(r"^(\s*)accumulate_grad_batches: \d+.*$",
                 r"\1accumulate_grad_batches: 16   # 2 GPUs x batch 1 x 16 = effective batch 32,"
                 "\n                                #  matching the overfit arms. Recompute as"
                 "\n                                #  32 / (2 * batch_size) if a bigger batch fits.",
                 src, flags=re.M)

    src = re.sub(r"^(\s*)ngpus_per_node_: \d+.*$", r"\1ngpus_per_node_: 2", src, flags=re.M)

    (CFG / dst_name).write_text(src)
    print("wrote", dst_name)
    for key in ("run_name_", "dataset", "max_epochs", "accumulate_grad_batches", "ngpus_per_node_"):
        for line in src.splitlines():
            if line.strip().startswith(key + ":"):
                print("   ", line.strip()[:96])
                break
