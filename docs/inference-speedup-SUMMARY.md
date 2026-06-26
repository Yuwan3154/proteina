# Proteina Inference Speedup — SUMMARY

Distilled do/don't + key-file map. Not a log. Motivations/understanding live in `inference-speedup-PLAN.md`.

## DECISIONS (locked this session)
- Training-free FIRST, then distillation; **combine** = distill on the chosen smart schedule's step boundaries.
- Quality gates in priority: **(1) AF2Rank quality distribution [PRIMARY]**, (2) fold recovery (GearNet+USalign), (3) FID/fJSD. scRMSD + self-consistency = diagnostics only.
- Working hypothesis = **global fold/topology only** (AF2Rank repairs local detail) — UNTESTED, must validate by ablating low-noise steps and checking AF2Rank holds.
- Evaluate BOTH SDE ("sc") and ODE ("vf") baselines.
- Eval set = bad_afdb **26 HELD-OUT** seqs (`bad_afdb_ref/pdb_70_..._with_cath.csv`; SuperCloud `~/data/bad_afdb/`), BOTH seq_cond + seq_cath_cond, 16 samples/seq. 2 seqs (`7F7N_A`,`8AP5_A`) lack CATH → seq_cond only. Gate = AF2Rank `composite` (ptm·plddt) vs 200-step ref, report ptm/plddt.

## DO
- Run all compute on **SuperCloud** (sbatch arrays, `xeon-g6-volta`); chain steps with `sbatch --dependency`, never a login-node poller.
- Treat the **schedule knobs that already exist** as approach-#2 levers: `schedule.schedule_mode ∈ {uniform,log,power,cos_sch_v_snr,loglinear,edm}`, `schedule.schedule_p`, `dt` (NFE=ceil(1/dt)).
- Establish a fixed reference set at 200 steps + a quality-gate script BEFORE tuning anything.
- Use **both** GearNet CA classifier and USalign/Foldseek-to-CATH for fold labels (default until user says otherwise).
- Ground every numeric/schedule/threshold choice in the lit search, repo, or our own runs.
- Cheapest commitment probe = **x0-prediction readout** (TM(x0_pred(t), final) vs t) — run BEFORE the K=8 fork sweep.
- Treat **SDE-native fast solvers** (SEEDS / SDE-DPM++2M-SDE) as a first-class workstream (default sampler IS the SDE).
- First training-free tries: torchdiffeq RK4/Heun on velocity → OSS/GITS DP schedules → EDM churn for stochasticity. Aggressive regime: SiD-Protein (arXiv:2510.03095).

## DON'T
- ⚠️⚠️ Don't trust ANY af2rank gate number (ref_pred_tm / composite) produced before the 2026-06-26 AF2Rank protocol fix — see PLAN top block + TaskCreate #1. CORRECT protocol: composite = pTM×pLDDT×**tm_io** (all three; tm_io dominant), **mask_sidechains_add_cb** (N/CA/C/O/CB + virtual CB on Gly), **recycles=1** (not 3/6), gap-token template seq, full decoy set, per-target Spearman→mean. Core fixes on origin/main (`97ca393` rec1, `d6e3ccf` mask-sc, `67cc19d`/`aafa509` composite); final tweaks uncommitted → confirm the deployed git state before re-scoring. ALL major analyses must be re-scored (reuse tars, `--rerun_af2rank_on_top_k`, rec1; do NOT regenerate). UNAFFECTED: ProteinEBM `tm_ref_template` (raw-selection / Q4 relaxation).
- Don't invent step counts, schedules, cutoffs, or hyperparameters — ask if unspecified.
- Don't assume pure-ODE sampling: Proteina's **default is the stochastic SDE** (`sampling_mode="sc"`). Switching to ODE ("vf") is a decision (Q4), not a default.
- Don't expect the coordinate flow loop to save/resume intermediates — it doesn't; a small hook in `r3n_fm.py` is required for approach #3.
- Don't score CA-only output directly where all-atom is needed — reconstruct atom37 first.
- Don't `scancel`/touch jobs you didn't submit (shared `chenxiou` account).
- Don't quote image-domain NFE (UniPC@10, SiD 16-step) as a Proteina guarantee — validate on our 3 gates (~20–50 NFE realistic training-free).
- Don't port a schedule before confirming the t=0 vs t=1 (noise vs data) direction in `r3n_fm.py`.
- Don't assume SiD-Protein's public ckpt fits us — it distills the UNCONDITIONAL model; seq-cond needs re-running SiD on our teacher.
- Don't look for a CLI/env knob for dt/NFE/schedule — there is NONE. Sweep via COPIED untracked config yamls (change only the `inference:` block, keep `run_name_`). nsamples override = `--nsamples_per_protein`/`PROTEINA_NSAMPLES_PER_PROTEIN`.
- Don't run on SuperCloud without `source activate cathfold` + `--direct_python` (cue_openfold/colabdesign envs are ABSENT). Always set `PROTEINA_CONDITIONING_MODE`.
- Don't pass `--nsamples_per_protein` OR `--skip_existing` to `run_prediction_pipeline.py` (both are only on `parallel_proteina_inference.py` → "unrecognized argument", exit 2). Set sample count via `export PROTEINA_NSAMPLES_PER_PROTEIN=N` (inference.py:323-327). For af2rank scoring, also add `--skip_cross_protein_plots` (the cross-protein plotter only supports the proteinebm-top-k summary format, else exit 1).
- In sbatch wrappers, capture python's exit (`RC=$?; ... ; exit $RC`) — a trailing `echo` makes SLURM report COMPLETED 0:0 even when python failed (masks errors).
- Don't run a SLURM **array** over DIFFERENT configs/datasets — `parallel_proteina_inference` auto-detects the array as shards of ONE run → deadlocks "Waiting for N inference step shards (0/N done)" forever. Use SEPARATE jobs + `--num_shards 1 --shard_index 0 --no-dynamic_resharding`. (Array is ONLY for sharding a single dataset, like prod wc_badafdb.)

## SUPERCLOUD RUN (cou@SuperCloud, repo /home/gridsan/cou/proteina)
⭐⭐ READ FIRST, do NOT guess flags: canonical inference doc `~/.claude/plans/users-chenxi-downloads-s41467-025-67127-rosy-fox.md` (+ -SUMMARY).
- Env: `module load conda/Python-ML-2025b-pytorch cuda/12.6 && source activate cathfold`; `export DATA_PATH=...proteina/data`; `--direct_python` + V100 cuEq flags; USalign on PATH (`$HOME/.local/bin`).
- Env-var overrides (orchestrator has NO CLI flags for these): `PROTEINA_NSAMPLES_PER_PROTEIN=16`, `PROTEINA_INFERENCE_PRECISION=fp16` (~2× faster, quality-equiv; ⛔ fp16 af2rank BROKEN on V100 → score fp32), `PROTEINA_CHECKPOINT_MODE` (default best), `PROTEINA_CONDITIONING_MODE={seq|seq_cath}`.
- PRODUCTION scoring (FAST, what the user wants): `run_prediction_pipeline.py --scorer proteinebm --proteinebm_checkpoint ~/ProteinEBM/weights/pae.ckpt --proteinebm_config ~/ProteinEBM/protein_ebm/config/pae_config.yaml --proteinebm_t 0.05 --proteinebm_batch_size 16 --af2rank_top_k 5 --recycles 6 --af2rank_backend openfold --segment_mode off --skip_diversity --skip_cross_protein_plots`. ProteinEBM ranks ALL samples fast → af2rank only top-5 (af2rank @1 recycle ≈ @6, 3× cheaper if needed).
- GATE file: `out/ref_<tag>/prediction_summary.csv` → best_ref_pred_tm (user 'ours y') / best_ptm / best_proteinebm_ptm per protein. Per-sample composite in `inference/<cfg>/<cond>/<pid>` af2rank_on_proteinebm_top_k csvs.
- Baseline model = 2-seq unified, dt=0.005/200 NFE, log/2.0, sc/0.45, chk_best epoch=180. Isolate every new mechanism via a COPIED config name (keep run_name_).

## KEY FILES (what each is for)
- `proteinfoundation/flow_matching/r3n_fm.py` — flow-matching sampler. `full_simulation()` :418-576 (Euler loop, NFE=ceil(1/dt) :484); `step_euler()` :269-351 (ODE/SDE step); `get_schedule()` :639-693 (the schedule modes); `get_gt()` :578-637 (SDE g(t)). **Approach #3 hook goes in the :519-575 loop.**
- `configs/experiment_config/inference_base.yaml` — defaults: `dt`(:21), `sampling_mode="sc"`(:26), `sc_scale_noise`(:27), `gt_mode`(:29), `schedule_mode="log"`/`schedule_p=2.0`(:34-35), `guidance_weight`(:52).
- `configs/.../inference_seq_cond_sampling_21-seq.yaml` — the seq-conditioned override (dt=0.005→200 steps, self_cond=True).
- `proteinfoundation/inference.py` — inference entry; `compose_inference_cfg()` :281-358; `--conditioning_mode` (seq / seq_cath) :838-847.
- `proteinfoundation/metrics/gearnet_utils.py` — GearNet CA classifier (`NoTrainCAGearNet` :480-512, forward :406-442); outputs pred_C/A/T.
- `script_utils/cat_recovery_usalign.py` + `proteinfoundation/prediction_pipeline/usalign_tabular.py` — USalign CAT recovery + parsing. `scripts/foldseek_cath_batch.py` — Foldseek CATH50.
- `proteinfoundation/metrics/designability.py` — scRMSD (ProteinMPNN→ESMFold) :301-352; `rmsd_metric` :246-298.
- `proteinfoundation/metrics/metric_factory.py` — FID/IS/fJSD via GearNet :34-194.
- `proteinfoundation/proteinflow/proteina.py` — CA→atom37 reconstruction :794-842.
- `docs/inference-speedup-PLAN.md` — the live plan (protocol, status, all motivations).
- `docs/inference-speedup-litsearch-raw.json` — full lit-search synthesis (final recommendation, gaps, verified-paper verdicts).

## DATA PATHS
- `$DATA_PATH/metric_factory/model_weights/gearnet_ca.pth`; `$DATA_PATH/cath_shared/{cath_label_mapping.pt, pdb_chain_to_cat.pkl, cat_recovery/reps}`.
