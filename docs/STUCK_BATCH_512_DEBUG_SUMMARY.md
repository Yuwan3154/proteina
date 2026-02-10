# Stuck at batch 512 – debug summary

## What we are trying to solve

**Symptom:** DDP training (4 GPUs) runs until step/batch 512, then hangs. After ~30 minutes, NCCL reports timeouts: Rank 3 first on a **BROADCAST** (SeqNum 54), then Ranks 0,1,2 on **ALLREDUCE** (SeqNum 55). Job is killed or abandoned.

**Goal:** Determine whether the hang is caused by:
1. **Corrupted or bad .pt data** in the batches used at step 512 (any of the four ranks), or  
2. **Training step** (forward/backward) or **collective** (e.g. gradient sync) hanging for a specific batch, or  
3. Something else (e.g. infra, driver, NCCL).

**Current conclusion from logs:** Data loading and collation **finish** for all ranks. The last activity on Rank 3 is `collate done` at 21:06:23 for batch `['4noo_B','6hd8_B','2zhx_L','5ywr_B']`. There are no further [R3] lines before the NCCL timeout. So the hang occurs **after** the DataLoader returns the batch—i.e. in the **training step** (forward/backward) or in the **next collective**. That could still be triggered by bad/corrupt data in that batch (e.g. NaN, weird shape, or a value that causes a CUDA kernel to hang).

---

## Data and environment paths

- **`DATA_PATH`** is set in **`proteina/.env`**:  
  `DATA_PATH=/home/ubuntu/proteina/data`
- **Processed PDB .pt files (training data):**  
  `proteina/data/pdb_train/processed/`  
  i.e. `$DATA_PATH/pdb_train/processed` or  
  `/home/ubuntu/proteina/data/pdb_train/processed`
- **Slurm logs (referred to in this debug):**  
  `slurm-9027755.err`, `slurm-9027755.out` (in workspace root, e.g. `/home/ubuntu/`)

---

## Last stuck batches (protein IDs) – all four ranks

From `slurm-9027755.err` at **21:06:23** (last collate start/done before hang):

| Rank | protein_ids |
|------|-------------|
| R0   | `4eyc_B`, `2z5b_B`, `6kfn_A`, `4zjm_A` |
| R1   | `1fl7_B`, `5j0f_A`, `4d50_B`, `3ch4_B` |
| R2   | `3g7n_B`, `1vkf_C`, `6j0k_B`, `6cfz_J` |
| R3   | `4noo_B`, `6hd8_B`, `2zhx_L`, `5ywr_B` |

**16 unique IDs.** Each corresponds to a file:  
`proteina/data/pdb_train/processed/<id>.pt`  
e.g. `4noo_B.pt`, `6hd8_B.pt`, etc.

---

## Checking those 16 .pt files for corruption

- **Validator script (stuck-batch only):**  
  `proteina/proteinfoundation/utils/validate_stuck_batch_pt.py`  
  - Validates only the 16 IDs above.  
  - **Use the `cue_openfold` conda environment** (same as training; has torch_geometric).  
  - **Run:**  
    `conda run -n cue_openfold python proteina/proteinfoundation/utils/validate_stuck_batch_pt.py /home/ubuntu/proteina/data/pdb_train/processed`  
  - Or from proteina dir with DATA_PATH from `.env`:  
    `conda activate cue_openfold` then  
    `python proteinfoundation/utils/validate_stuck_batch_pt.py` (uses `$DATA_PATH/pdb_train/processed`).  
  - **Result (run in cue_openfold):** All 16 files load successfully; no corruption detected in these .pt files.

- **Generic .pt validator (all or sampled files in a directory):**  
  `proteina/proteinfoundation/utils/validate_pt_files.py`  
  - Run in **cue_openfold** conda env.  
  - Usage:  
    `conda run -n cue_openfold python proteina/proteinfoundation/utils/validate_pt_files.py /home/ubuntu/proteina/data/pdb_train/processed [--sample N] [--workers W]`

---

## Dataset and training config (relevant to this run)

- **Dataset config (contact, CB 10Å, S25, max320, purge-test, cutoff-190828):**  
  `proteina/configs/datasets_config/pdb/pdb_train_contact-CB-10A_S25_max320_purge-test_cutoff-190828.yaml`  
  - Defines `data_dir: ${oc.env:DATA_PATH}/pdb_train/`, dataselector, datasplitter, transforms, batch_size, etc.  
  - **Exclude list:** We added the R3 batch IDs (`4noo_B`, `6hd8_B`, `2zhx_L`, `5ywr_B`) to `exclude_ids` in both dataselector and datasplitter to test if excluding that batch avoids the hang. You can remove them if you want to re-test with that batch included.

- **Confind dataset config (if you use confind maps):**  
  `proteina/configs/datasets_config/pdb/pdb_train_contact-confind_S25_max320_purge-test_cutoff-190828.yaml`

---

## Code paths involved in loading and training

- **Data loading (PDB dataset, getitem, retries, debug logging):**  
  `proteina/proteinfoundation/datasets/pdb_data.py`  
  - PDBDataset: `__getitem__` loads from `data_dir / "processed" / f"{pdb_code}_{chain}.pt"` (or equivalent), with retries and optional per-rank/per-worker debug logs (`[R{rank} W{worker_id}]`).

- **Base datamodule / dataloader creation (rank, debug flags passed to loader):**  
  `proteina/proteinfoundation/datasets/base_data.py`  
  - `_get_dataloader` sets `_rank_for_logging`, `_debug_data_loading`, and passes them into the dense padding dataloader.

- **Collation and collate logging:**  
  `proteina/proteinfoundation/utils/dense_padding_data_loader.py`  
  - DensePaddingCollater / DensePaddingDataLoader: collate start/done and `protein_ids` logged when debug is on.

- **Transforms (e.g. contact map, confind):**  
  `proteina/proteinfoundation/datasets/transforms.py`  
  - ContactMapTransform and others; relevant if you switch between contact_method distance vs confind.

- **Confind precompute (if you precompute confind maps into .pt):**  
  `proteina/proteinfoundation/utils/precompute_confind_maps.py`  
  - Writes/updates processed .pt files; atomic saves and indentation were fixed earlier.

---

## Log files

- **Stderr (NCCL, data-load debug, collate lines):**  
  `slurm-9027755.err`  
  - Search for `[R0]`, `[R1]`, `[R2]`, `[R3]` and `collate start/done` / `protein_ids=` to see last batch per rank.  
  - NCCL timeout messages at the end.

- **Stdout (training config, progress, etc.):**  
  `slurm-9027755.out`

---

## Quick reference – file paths

| Purpose | Path |
|--------|------|
| DATA_PATH | `proteina/.env` (single line: `DATA_PATH=/home/ubuntu/proteina/data`) |
| Processed .pt data | `proteina/data/pdb_train/processed/` |
| Validate 16 stuck-batch IDs | `proteina/proteinfoundation/utils/validate_stuck_batch_pt.py` |
| Validate all/sampled .pt | `proteina/proteinfoundation/utils/validate_pt_files.py` |
| Dataset config (contact CB) | `proteina/configs/datasets_config/pdb/pdb_train_contact-CB-10A_S25_max320_purge-test_cutoff-190828.yaml` |
| Dataset config (confind) | `proteina/configs/datasets_config/pdb/pdb_train_contact-confind_S25_max320_purge-test_cutoff-190828.yaml` |
| PDB dataset / getitem | `proteina/proteinfoundation/datasets/pdb_data.py` |
| Base datamodule / dataloader | `proteina/proteinfoundation/datasets/base_data.py` |
| Dense padding collater/loader | `proteina/proteinfoundation/utils/dense_padding_data_loader.py` |
| Transforms | `proteina/proteinfoundation/datasets/transforms.py` |
| Confind precompute | `proteina/proteinfoundation/utils/precompute_confind_maps.py` |
| Slurm stderr | `slurm-9027755.err` |
| Slurm stdout | `slurm-9027755.out` |
| Conda env for training / validation | `cue_openfold` |

---

## Suggested next steps (for you to take from here)

1. **Stuck-batch validator already run in `cue_openfold`:** All 16 files load OK; no corruption in those .pt files. To re-run:  
   `conda run -n cue_openfold python proteina/proteinfoundation/utils/validate_stuck_batch_pt.py /home/ubuntu/proteina/data/pdb_train/processed`

2. **Re-run training** with the current config (with the four R3 IDs excluded). If it passes step 512, the hang is likely tied to that batch or one of those samples.

3. Optionally add **step-level logging** (e.g. “rank N starting forward/backward for batch K”) to see how far Rank 3 gets before the BROADCAST.

4. On the cluster, ensure **DATA_PATH** is set the same way (e.g. from `proteina/.env`) so that `data_dir` resolves to the same `pdb_train` and `processed` location.
