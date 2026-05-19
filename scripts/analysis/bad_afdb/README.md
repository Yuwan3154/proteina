# bad_afdb evaluation pipelines

Scripts live under `proteina/scripts/analysis/bad_afdb/`; data still lives under `~/data/bad_afdb/`.

This folder contains **two major baselines** plus utilities to **filter to monomers** and **compare** against Proteina sampling.

## 1) Template search + structure comparison baseline

### Script
- `template_search_pipeline.py`

### Goal
For each target protein chain in the dataset, run an **MSA-based template search** (hhsearch against `pdb70`), download the top templates, **extract the template chains**, align template-vs-native using **USalign**, then summarize TM-scores.

### Inputs (defaults; all overridable via CLI)
- `--input_csv` (default `~/data/bad_afdb/pdb_70_cluster_reps_aligned_confidence_aggregate_plddt_50_length_50-768_tms_in-train.csv`)
- `--pdb70_db`  (default `~/data/pdb70_220313/pdb70`)
- `--conda_env` / `--conda_hook` — the env in which `hhsearch` and `USalign` are installed (defaults: `openfold` env; conda hook from `$CONDA_EXE` or `/opt/tljh/user/bin/conda` fallback)

### Workflow
1. **Download AFDB MSA** for each UniProt ID
   - URL pattern: `https://alphafold.ebi.ac.uk/files/msa/AF-{uniprot}-F1-msa_v6.a3m`
   - Saved under `--msa_dir` (default `~/data/bad_afdb/msa/`).
2. **Run `hhsearch`** on the MSA against `pdb70` — outputs `.hhr` under `--template_msa_dir` (default `~/data/bad_afdb/template_msa/`).
3. **Parse hhsearch hits** and take top-k (`-k/--top_k`, default 4).
4. **Download PDB mmCIF** for each hit and extract chain
   - Native PDB mmCIF cached under `--pdb_dir / <mid2> / <pdb>.cif`
   - Extracted template chains saved (flat) under `--template_pdb_dir / {PDB}_{CHAIN}.pdb`
5. **Run USalign** to align the native chain vs each template chain → TSVs under `--template_aln_dir / {PDB}_{CHAIN}.tsv`.
6. **Aggregate TM-scores**
   - `tm_scores_results.tsv` (per-target `top1_tm_score`, `max_tm_score`)
   - `tm_score_distribution.png` — histograms of top-1 and top-k(max) distributions

### Primary outputs
- Per-target alignment TSVs: `~/data/bad_afdb/template_aln/*.tsv`
- Summary table:             `~/data/bad_afdb/template_aln/tm_scores_results.tsv`
- Histogram figure:          `~/data/bad_afdb/template_aln/tm_score_distribution.png`


## 2) OpenFold multi-model multi-seed baseline (AF2 weights + original MSA)

### Script
- `run_openfold_baseline.py` (uses `aligned_msa_utils.py` from the same directory)

### Goal
For each target protein chain, run **OpenFold inference** using **all five AF2 pTM models** (`model_1_ptm` … `model_5_ptm`) and **k random seeds**. For each predicted structure, compute **TM-score vs the native PDB chain** using USalign, and record `ptm` from OpenFold outputs.

### Key design (efficiency)
Outer loop is **model → protein → seed**, so **weights are loaded once per model**, reused across all proteins/seeds.

### Inputs
- Input CSV passed by CLI (example): `~/data/bad_afdb/pdb_70_cluster_reps_aligned_confidence_aggregate_monomer_length_50-640_tm-05.csv`
- AFDB MSAs directory (CLI default): `~/data/bad_afdb/msa`
- Native PDB mmCIF directory (CLI default): `~/data/bad_afdb/pdb`
- OpenFold repo directory (CLI default): `~/openfold` (locates AF2 JAX params)

### Workflow
1. For each `pdb_chain`: load native mmCIF, write FASTA, link AFDB MSA into per-tag alignment dir, link native mmCIF into template dir for OpenFold's template featurizer.
2. For each model in `{model_1_ptm..model_5_ptm}`: load model weights once.
3. For each protein × seed: run inference, write predicted PDB and scores JSON, then run USalign against the native chain (`-chain2 {chain}`) and record `tm_pred_norm` and `tm_native_norm`.

### Primary outputs
- `output_dir/{pdb_chain}/predictions/*`
- Aggregate CSV: `output_dir/summary_results.csv`
- Per-protein CSVs: `output_dir/per_protein/{pdb_chain}_tm_scores.csv`


## 3) Stoichiometry (monomer filtering utility)

### Script
- `get_stoichiometry.py` (consolidates the original + `_enhanced` versions; supports obsolete-PDB replacement and GraphQL fallback)

### Goal
Fetch assembly/stoichiometry information from RCSB (with caching + obsolete-PDB handling + GraphQL fallback) to support filtering to monomers.

### CLI
- `--input_csv` / `--cache_dir` / `--output_csv` / `--id_column` / `--save_interval` (batch mode).
- `--test` runs a 3-case obsolete-PDB harness instead of batch processing.

### Output
- `<output_csv>` with columns `pdb, stoichiometry, oligomeric_state, full_composition, final_pdb, is_obsolete, replacement`
- `<output_csv with _simple suffix>` — just `pdb, stoichiometry` for downstream filtering (monomer ↔ `"A1"` / `"Monomer"`).


## 4) Comparison: template search vs Proteina

### Script
- `compare_template_vs_proteina.py`

### Goal
Compare TM-score outcomes between:
- **Template search baseline** (from `template_aln/*.tsv`, with failures filled as 0)
- **Proteina** results (from a provided `cross_protein_summary_data.csv`)

### CLI
- `--proteina_results_csv` (required) — `cross_protein_summary_data.csv` produced by `generate_cross_protein_plots.py`.
- `--af2rank_topk_csv` — required only in ProteinEBM scorer mode (when `top_*_tm_ref_pred` is absent from the main CSV).
- `--list_csv` / `--list_id_column` — optional subset filter (e.g. monomers); when omitted, processes every protein in the main CSV.
- `--template_aln_dir` — default `~/data/bad_afdb/template_aln`.
- `--output_dir` (required) — where to write plots.
- `--label` — appends `" (<label>)"` to titles and `_<slug>` to output filenames (e.g. `--label Monomers` → `… (Monomers)` / `…_monomer.png`).

### Key behaviours
- Builds `template_top_1` (first TM in USalign TSV) and `template_top_5` (max TM in TSV).
- Includes proteins missing template TSVs as `template_failed=True` with TM=0.0.
- Generates scatter plots + stats on the full dataset and on the subset where at least one value ≥ 0.5.


## 5) Monitoring helpers

- `monitor_pipeline.sh` — template-search pipeline progress. Override `TOTAL`, `RESULTS_FILE`, `LOG_FILE`, `PROCESS_PATTERN` via env vars.
- `monitor_stoichiometry.sh` — stoichiometry batch progress. Override `TOTAL`, `RESULTS_FILE`, `LOG_FILE`, `CACHE_DIR`, `PROCESS_PATTERN`.
