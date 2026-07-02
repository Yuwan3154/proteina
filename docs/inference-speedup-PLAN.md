# Proteina Inference Speedup — PLAN

Status legend: [ ] todo · [~] in progress · [x] done · [?] blocked on user decision
Dated blow-by-blow, full numeric tables, and incident narratives live in `inference-speedup-PLAN-RAW-ARCHIVE.md` — this file is distilled current state only. Do/don't gotchas live in `inference-speedup-SUMMARY.md`.

Compute targets: **SuperCloud** (xeon-g6-volta, 2×V100/node, sbatch arrays) and **A6000** (`A6000_offsite`/`A6000_MIT`, 4× RTX A6000/49GB, direct nohup-detached launches, no SLURM — check `nvidia-smi` for free GPUs every time, GPU3 is often another job's). Canonical env on every host: `~/cue-openfold-env` (uv-managed sibling repo, `source .venv/bin/activate && source ./activate.sh` — torch 2.7.1+cu126, openfold+proteina+ProteinEBM+Frame2ConFind all editable in one venv). Edit code locally, git-sync, pull on node.

---

## 0. GOAL & MOTIVATION (the WHY)

**Objective:** cut the number of denoising/integration steps (NFE) — hence wall-clock per sample — of Proteina's **sequence-conditioned** structure sampling, **without degrading fold/topology quality** that the downstream template-search pipeline depends on. Default ≈200 steps; naive step reduction degrades quality.

Three candidate directions (user-proposed):
1. **Flow-map / few-step sampling** — reduce NFE via a principled method (flow maps, consistency, shortcut, distillation, fast ODE solvers), not naive step-count reduction.
2. **Smarter noise/time schedule** — spend steps where the model does fold-determining work. *Working hypothesis:* template search needs global FOLD/topology only; local detail is repaired by downstream AF2Rank refinement — UNTESTED at the time this hypothesis was written, but the hybrid-schedule result below is consistent with it (aggressive low-noise step-cutting works).
3. **Fold-commitment / critical-window measurement** — find the noise level beyond which the fold is locked in, to place schedule cutoffs empirically rather than by guess.
4. **(#1+#2 combined) schedule-aware distillation** — distill a few-step model targeting the smart schedule's step boundaries.

## 1. GROUNDED REPO FACTS (file:line references, authoritative)

- Interpolant: linear/OT, `x_t = (1-t)x0 + t x1`. Integrator: explicit Euler. `r3n_fm.py:124-211,269-351,418-576`.
- **NFE = ceil(1/dt)**, config key `dt` (`inference_base.yaml:21`).
- **Convention: t=0=NOISE, t=1=DATA** (opposite standard diffusion) — always confirm before porting any schedule idea, a flipped direction silently misplaces steps.
- **Default sampler is the STOCHASTIC SDE** (`sampling_mode="sc"`), not ODE. ODE mode is `"vf"`. Training-free fast solvers target the ODE — accelerating the production SDE-as-is vs switching baseline to ODE is a real fork (resolved below: ODE ≈ SDE in final refined quality at 200 steps, but SDE's raw-sampling edge at low NFE comes from stochastic exploration — see the hybrid-schedule finding).
- Schedule modes (`get_schedule()`, `r3n_fm.py:639-693`): `uniform`, `log` (default, dense near t=1/low-noise), `power`, `cos_sch_v_snr`, `loglinear`, `edm`.
- Coordinate flow loop does NOT expose intermediate x_t or resume-from-σ by default (`r3n_fm.py:519-575`) — the trajectory-dump/fork hook (`PROTEINA_TRAJ_DUMP`/`PROTEINA_FORK_FROM`/`PROTEINA_FORK_T_IDX`) is an additive, env-gated, default-off feature added for the commitment experiment (committed `2f336bc`/`70aab4e`).
- Hybrid SDE→ODE schedule hook: `PROTEINA_SDE_UNTIL_T=<t>` (env-gated, default-off, committed `2f336bc`) — per-step mode = sc if t<thresh else vf.
- Eval metrics: designability scRMSD (`designability.py`), FID/fJSD via GearNet (`metric_factory.py`), no built-in TM-score (use USalign). Output is CA-only → reconstruct atom37 (`proteina.py:794-842`) before any all-atom scoring.
- Model in use: 2-seq finetune config `..._unified` family, checkpoint `chk_best epoch=180`. **Trained EXCLUSIVELY on chains L∈[50,256]** — proteins beyond that (6M5Y 270, 8XHT 287, 8TVL 338, 8AUC 493) are out-of-training-distribution for this checkpoint.
- Training time-distribution: `mix_up02_beta` = 98% Beta(1.3,2.0) + 2% Uniform — mode t≈0.231, CDF(0.6)=0.78. Model trained overwhelmingly on t∈[0,0.6]; little signal past t≈0.8.

## 1.5 LITERATURE SHORTLIST (from lit-search workflow `wid8pfxlf`; full synthesis → `inference-speedup-litsearch-raw.json`)

**Framing risk:** nearly all fast solvers are ODE-derived; Proteina's load-bearing default is the SDE. Test both samplers; pair any ODE solver with a stochasticity source. Image-domain NFE numbers do NOT transfer — realistic training-free landing zone ≈20-50 NFE.

Training-free (try first): torchdiffeq RK4/Heun on velocity → OSS/GITS DP schedules → EDM Heun+churn → SDE-native 2nd-order solver (SEEDS/SDE-DPM++2M-SDE) → UniPC/DPM-Solver++(2M) → SD3 time-shift → AYS → ParaDiGMS.
Training-required (aggressive regime, later): SiD-Protein (arXiv:2510.03095 — public ckpt distills the UNCONDITIONAL model, our seq-cond win needs re-running SiD on our teacher) → LD3/DSS → Reflow.
De-prioritized: MeanFlow/Shortcut/Consistency-FM/FlowSolver (zero protein results); arXiv:2510.24732 (SE(3)-frame, not CA-only).

## 1.6 SUPERCLOUD PIPELINE — reference

- User=cou, repo `/home/gridsan/cou/proteina`, `DATA_PATH=.../proteina/data`. Eval cifs+csvs in `~/data/bad_afdb/`. Shared account → read-only on others' work, never cancel others' jobs.
- Env: `module load conda/Python-ML-2025b-pytorch cuda/12.6 && source activate cathfold`; always `--direct_python`; V100 flags `--no-use_deepspeed_evoformer_attention --use_cuequivariance_attention --use_cuequivariance_multiplicative_update`; `export PROTEINA_CONDITIONING_MODE={seq|seq_cath}` mandatory.
- Two entry points: `parallel_proteina_inference.py` (sharded SAMPLING only) and `run_prediction_pipeline.py` (orchestrator: sample→reconstruct→score, `--scorer {af2rank|proteinebm}`).
- NFE/schedule control = `dt` + nested `schedule.*`/`sampling_caflow.*` **in the config only**, no CLI/env override. Sweep by copying the config to a new untracked name, editing only the `inference:` block, keeping `run_name_` unchanged (ckpt auto-resolves). `nsamples_per_len` DOES have an override: `--nsamples_per_protein`/`PROTEINA_NSAMPLES_PER_PROTEIN`.
- Baseline config: `dt=0.005` (200 NFE), `schedule=log/2.0`, `sampling_mode=sc`, `sc_scale_noise=0.45`, `self_cond=True`.

## 2. DESIGN DECISIONS — RESOLVED (locked)

- Training-free FIRST, then distillation; combine = distill on the chosen smart schedule's step boundaries.
- Quality gates in priority: (1) AF2Rank prediction-quality distribution [PRIMARY — AF2Rank refinement fixes local imperfections, so good sampled topology → good final AF2Rank prediction], (2) fold/topology recovery (GearNet + USalign-to-CATH), (3) FID/fJSD. scRMSD/self-consistency = diagnostics only.
- Evaluate BOTH SDE (production default) and ODE baselines every experiment.
- Eval sets: `scaling8` (8 hard targets: 8XHT/7AD5/6ZUS/8QXI/6ZTG/7KW9/8RJX/6ZYG — used for every SDE-200-vs-fast apples-to-apples comparison) and the full `bad_afdb26` held-out set (26 seqs, `bad_afdb_ref/pdb_70_..._with_cath.csv`, all `in_train=False`, len 62-492). 2 seqs (`7F7N_A`,`8AP5_A`) lack CATH → seq_cond only.
- **CORRECTED AF2RANK PROTOCOL (locked, external-audit-verified against Roney & Ovchinnikov 2022):** `composite = pTM × pLDDT × tm_io` (all three — tm_io is the dominant signal); `mask_sidechains_add_cb` (strip decoy template to N/CA/C/O/CB + virtual CB on Gly before templating, else AF2 leaks native sequence via sidechain geometry); `recycles=1` (not 3/6); template seq = gap tokens, single-sequence, model_1_ptm; use the FULL decoy set, not a subsample; metric = per-target Spearman then mean over targets. Sanity targets (stock, masked, rec1, full decoys): 1agy 0.89, 1acf 0.89, 1a32 0.92, 1cc8 0.83. Any af2rank-gated number scored before this protocol landed is provisional — re-score, don't trust. **UNAFFECTED by this correction:** ProteinEBM's `tm_ref_template` (raw USalign-to-native, no af2rank involved) — this is the metric used for all raw-sampling-quality comparisons below.

---

## 3. DISTILLED FINDINGS (current understanding — see RAW-ARCHIVE for the full numeric progression behind each)

1. **ODE ≈ SDE in final refined quality at 200 steps** → ODE (far more amenable to fast solvers) can be accelerated without losing final quality.
2. **High-noise-emphasis schedule (`power p3`) beats the default `log` schedule on pure-ODE**, ~8× speedup (NFE25) at ~97% of the 200-step baseline. This schedule is ODE-SPECIFIC — does not transfer to SDE.
3. **SDE-200's edge over fast-ODE at the deployed (8192-sample) operating point is mostly EBM SELECTION, not sampling quality** — oracle-ceiling gap is only ~-0.010 @8192 vs ~-0.026 deployed-energy gap.
4. **⭐⭐⭐ WINNING RESULT — hybrid SDE(t<0.6)→ODE schedule, power-p3 shape: NFE40 or NFE50 BEATS full SDE-200 (200 steps) on raw oracle TM, at 40-50 total steps (4-5× speedup with HIGHER quality, not just comparable).** SDE_UNTIL_T=0.6 was chosen because it empirically won a 2-point sweep AND covers ~78% of the training time-distribution's mass. Established on 7AD5 alone (2048 samples); currently being validated across `scaling8`→`bad_afdb26` at N=8192 (see §4 Active Work).
5. **ProteinEBM/energy selection is a genuinely weak, noisy selector on hard proteins (7AD5, 6ZYG, 8XHT)** — but a real, non-zero correlation exists in every tested config; the weakness is mostly a SAMPLE-COUNT effect (smaller pools give the same noisy signal fewer chances to land the argmin on a good structure), not a qualitative blind spot.
6. **Relaxation-before-scoring does NOT rescue weak EBM selection** — tested on 2 windows, all 8 scaling8 proteins, and again on the NFE30/40/50 pools: no meaningful, consistent Spearman improvement anywhere. Not the fix.
7. **⭐ ACTIONABLE FIX for weak selection: modest top-k rescues it cheaply.** k≤50 (often k≤5) reaches within 0.05 of the pool oracle for nearly every target/config tested, including the hard cases — deploying `af2rank_top_k~32-50` (vs the current default 5-8) should recover oracle-level quality without any sampling/scoring-model change. Per current user directive (2026-07-02), the ranking/scoring/refinement problem itself is deferred — push sampling-side (oracle) performance for now, revisit selection later.
8. **AF2Rank compile is worth enabling broadly, not just for long proteins**, once `torch.compile`'s persistent on-disk TorchInductor cache is warm (one-time-ever cost per length-bucket on a persistent host) — breakeven N is 8-134 decoys depending on length, well within typical `af2rank_top_k` usage.
9. **Length-bucketing (multiples of 32) lets `compile_inference_path=True` share one compiled kernel across many protein lengths** instead of recompiling per exact length (≤16 buckets cover the full 50-512 range). **Caveat:** padding applies ONLY to the compiled side (eager stays unpadded), so a length landing just under a bucket ceiling pays a real compute tax on the compiled path that can meaningfully dilute measured net speedup at that specific length (worst case ~31/L extra residues). **Open question, not decided:** keep bucket=32 (as directed) and accept this bounded tax, or go finer (16) to roughly halve it at ~2× more compiled shapes?
10. **bf16 (not fp16) is the right precision for OpenFold AF2Rank on bf16-capable hardware (A6000+).** bf16 EAGER — no compile — is the single best config found: ~44% faster than tf32 eager, and even beats tf32-compiled steady-state by 17-29%, with zero compile warmup. fp16 is NOT usable — OpenFold's custom `attention_core` CUDA kernel only supports `{float32, bfloat16}`, fp16 raises immediately (a genuine kernel limitation, not a bug — matches the existing V100 memory note). **Compiling bf16 makes it SLOWER, not faster** — do not combine `compile_inference_path=True` with `precision="bf16"` expecting a further win. Even precision-matched (both bf16), ColabDesign/JAX remains 3.6-5.5× faster than OpenFold — a genuine framework/kernel gap, not fully closeable by precision alone.
11. **AF2Rank re-score TODO (open):** all major analyses before the 2026-06-26 protocol fix need re-scoring under the corrected protocol (reuse tars, `--rerun_af2rank_on_top_k`, recycles=1, do NOT regenerate samples). Not yet done — status unclear, check deployed git state on SuperCloud before assuming it's current.

---

## 4. ACTIVE WORK

**Autonomous scaling8→bad_afdb26 hybrid-schedule campaign (A6000, all 3 free GPUs 0/1/2, GPU3 untouched — other job's).** Directive (user, 2026-07-02): expand the hybrid NFE40/NFE50 SDE(→0.6)→ODE result (finding #4 above) from 7AD5-only to `scaling8` then `bad_afdb26`, N=8192, **oracle ranking only** (`af2rank_top_k=0`, raw `tm_ref_template`) — selection/scoring/refinement fix deferred to a later session.

- Driver: `~/proteina_speedup/scripts/a6000_hybrid_campaign_driver.sh`, launched detached on A6000, **lock-file protected** (see SUMMARY DON'T). Sequential stages, each a 3-GPU parallel run via `run_prediction_pipeline.py --num_shards 3` (protein-level LPT-bin-packed sharding, not SLURM array): scaling8@NFE40 → scaling8@NFE50 (cross-validates the NFE40≈NFE50 tie beyond 7AD5) → bad_afdb26@NFE40 → bad_afdb26@NFE50, each gated on remaining wall-clock budget between stages (an in-flight stage is never killed mid-generation).
- Relaunched clean (after the incident in RAW-ARCHIVE) at **12:07 PM EDT 2026-07-02**; treat this as the fresh start of the ~12h budget (deadline ~00:07 AM EDT 2026-07-03).
- **Check status:** `ssh -n A6000_offsite 'cat ~/proteina_speedup/logs/campaign/driver.lock'` then verify that PID is alive and the ONLY process matching the driver script name. Results land in `~/proteina_speedup/logs/campaign/oracle_summary.{txt,csv}` when the campaign completes (or per-stage in `logs/campaign/{stage}_gpu{0,1,2}.log` for interim progress).
- `create_trigger`/`send_later` were 404ing at launch time — monitoring fell back to `ScheduleWakeup`; re-attempt the reliable tools on the next check-in. The on-box driver is fully self-advancing regardless.

## 5. LONGER-TERM STEPS (original provisional plan, mostly superseded by the hybrid-schedule finding above — kept for the parked workstreams)

- **Step D — training-free few-step solvers** `[ ]` not started: torchdiffeq RK4/Heun → OSS/GITS → SDE-native 2nd-order solver → UniPC/DPM-Solver++(2M). Needs a user design decision before starting.
- **Step E — distillation (aggressive regime)** `[ ]` not started: SiD-Protein re-run on our seq-conditioned teacher, targeting the winning hybrid schedule's step boundaries. Needs a user design decision before starting.
- **Fork+multi-completion true-commitment experiment** `[?]` PARKED (user 2026-07-01, "commitment time is highly stochastic — record and move on"): fork a reference SDE trajectory at candidate t, regenerate N=8 stochastic completions, measure pairwise/vs-original TM. Not launched. The d0(L)-artifact-corrected length↔commitment refit is also parked.

## 6. CONSTRAINTS / REMINDERS

- No invented numbers/schedules/thresholds/eval-sets — ground every choice in lit search, repo, or our own runs; ask when unspecified.
- `conditioning_mode` in inference MUST match the analysis conditioning mode.
- Proteina outputs CA-only → reconstruct atom37 before any all-atom scoring.
- SuperCloud: sbatch arrays (not salloc) over `xeon-g6-volta`; login nodes REAP nohup pollers → chain steps with `sbatch --dependency=afterany:JOBID`; <2000 files/dir.

## 7. TOP RISKS

1. **ODE-vs-SDE mismatch** — most training-free solvers are ODE-derived; the production default is SDE. Run both every experiment; pair any deterministic solver with a stochasticity source before declaring success. (Substantially de-risked by finding #4 — the hybrid schedule already gets SDE's exploration benefit at low total NFE.)
2. **Image-domain NFE ≠ fold quality** — validate every candidate on our own gates, not quoted image-diffusion numbers.
3. **Seq-conditioned-checkpoint gap for distillation** — SiD-Protein's public ckpt distills the unconditional model; confirm teacher match before trusting any borrowed distillation result.
