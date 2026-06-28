# Proteina Inference Speedup — PLAN

Status legend: [ ] todo · [~] in progress · [x] done · [?] blocked on user decision
Compute target: **SuperCloud** (xeon-g6-volta, 2×V100/node, sbatch arrays). Edit code locally, git-sync, pull on node.
Lit-search workflow: task `wid8pfxlf` — DONE (30 agents). Raw synthesis → `docs/inference-speedup-litsearch-raw.json`; condensed shortlist in §1.5.

## ⚠️⚠️⚠️ AF2RANK PROTOCOL CORRECTION (2026-06-26, external agent audit) — TODO: RE-RUN ALL MAJOR ANALYSES
An independent agent audited + reproduced AF2Rank (Roney & Ovchinnikov 2022, PRL 129 238101; 133 Rosetta-decoy targets) and found OUR af2rank protocol was wrong on several points. **Their edits are made but NOT yet committed → locate + deploy before any re-run.** Corrected STOCK AF2 (model_1_ptm) = per-target Spearman **0.84** (133 tgt, 25 decoys) / **0.882** (4 SI tgt, full ~1000 decoys); paper 0.925. A result well below ~0.8 ⇒ one of the bugs below.
**CORRECTED PROTOCOL (vs what we ran):**
1. **composite = pTM × pLDDT × tm_io — ALL THREE.** tm_io = TM(input decoy/template CA, AF2 output CA) = the DOMINANT ranking signal (output≈native ⇒ tm_io≈decoy-TM-to-native). Omitting tm_io is the most common bug (~0.69 vs ~0.84). ⚠️ our RAW scorer CSVs may carry provisional composite = pTM×pLDDT WITHOUT tm_io; `proteina_analysis.py` recomputes composite = pTM×pLDDT×tm_template_pred after enriching tm_io → TRUST the analysis output or confirm tm_io is multiplied in.
2. **mask_sidechains_add_cb — THE ROOT-CAUSE FIX (stock 0.69→0.84).** Strip the decoy BEFORE using it as the template: keep N, CA, C, O, CB only + project a virtual CB onto glycines. Full-atom decoys leak the native sequence via sidechain geometry → AF2 overconfident → confidence calibration collapses. (For us, templates are cg2all reconstructions of CA-only samples — not native sidechains — so the leak may be milder, but masking is still the correct protocol; VERIFY our OpenFold backend masks.)
3. **recycles = 1** (total forward iters = recycles+1 = 2), NOT 3. `run_prediction_pipeline.py` DEFAULTED `--recycles 3` (now fixed to 1) → pass `--recycles 1` if checkout predates the fix. ⚠️ our apples-to-apples + scaling8 baseline used **recycles 6** → re-score at rec1. (Memory said "af2rank rec1≈rec6"; the audit says rec1 is the correct protocol — re-validate, do not assume.)
4. Template seq = gap tokens (rm_seq / seq_replacement "-"); single-sequence (no real-seq MSA); model_1_ptm.
5. Use the FULL decoy set per target (~800–1000); a 25-decoy subsample DEFLATES per-target Spearman (1agy 0.62→0.89 going 25→816 decoys; 4-tgt mean 0.782→0.882).
6. Metric = per-target Spearman(composite, decoy-TM-to-native), then MEAN over targets (NOT one pooled Spearman).
**MECHANISM (so we don't misdiagnose):** AF2Rank's AF2 COPIES the decoy (output≈input) and ranks by CALIBRATED CONFIDENCE (pTM×pLDDT) with tm_io re-anchoring; it does NOT reject bad decoys (no-template AF2 avg TM~0.41 — can't fold native from scratch). Output≈input is EXPECTED + CORRECT, not a bug. Discrimination comes from well-calibrated confidence heads → exactly why sidechain masking (no cheating via leaked seq) matters.
**SANITY TARGETS (stock model_1_ptm, masked, rec1, FULL decoys):** 1agy 0.89, 1acf 0.89, 1a32 0.92, 1cc8 0.83.
**⇒ IMPLICATION — every gate number (ref_pred_tm / composite) in this plan is PROVISIONAL.** Step A 200-step baselines, Step B schedule-sweep ceiling, Step C, the 5-prot scaling + oracle, and the APPLES-TO-APPLES seq_cath energy/oracle convs (fast 0.365 vs SDE-200 0.404) were ALL scored under the OLD af2rank → must be re-scored with the corrected af2rank for accurate final quality calling; the fast-vs-SDE conclusion MAY shift. **UNAFFECTED:** ProteinEBM raw-selection (`tm_ref_template`) and the Q4 relaxation baseline (ρ on tm_ref_template) — those do not use af2rank.
**RE-RUN TODO (after the current campaign; ⛔ do NOT cancel running jobs):** (1) CONFIRM deployed git state on SuperCloud — per runbook, core fixes are ON origin/main (`97ca393` rec1 default, `d6e3ccf` mask-sidechains+virtual-CB, `67cc19d`/`aafa509` composite=pTM×pLDDT×tm_io); the audit's FINAL tweaks are NOT yet committed (user) → `git log`/`git diff` the SuperCloud checkout + verify recycles=1 default, tm_io multiplied, sidechain-mask in `af2rank_openfold_scorer.py` before re-scoring; (2) re-score EXISTING samples (reuse tars, `--rerun_af2rank_on_top_k`, rec1) — do NOT regenerate; (3) re-tabulate apples-to-apples + scaling; (4) confirm the fast-vs-SDE sign + magnitude. (Tracked: TaskCreate #1.)

## ⚠️⚠️ CRITICAL CORRECTION (2026-06-25 PM, user QC) — the speedup conclusion is under re-evaluation
**My earlier "fast p3@NFE25 = 8× at 97%, scaling saturates by ~256, fast ≫ baseline 0.348" was built on the WRONG comparison.** Three errors, all caught by the user:
1. **Wrong baseline.** I compared the fast 8192-sample run to SDE-200 at **16 samples** (a `ref_sc_seq` dir) and to the **26-protein aggregate 0.348**. The correct baseline is the previous agent's **SDE-200 8192-sample scaling8** run (8 hard targets, seq_cath), under the base config `..._unified` (confirmed SDE: `inference.sampling_caflow.sampling_mode: sc`, `dt 0.005` = NFE200).
2. **Wrong subset + conditioning.** My fast run = **seq_cond**, 5 length-diverse proteins (only 7AD5, 8QXI overlap scaling8). The established 8192 baseline = **seq_cath**, 8 hard targets (8XHT, 7AD5, 6ZUS, 8QXI, 6ZTG, 7KW9, 8RJX, 6ZYG).
3. **The fast config KILLS inference-time scaling on hard targets — this is the real finding.** SDE-200 seq_cath **7AD5 scales 0.425→0.589 (jump @2048)→0.684 @8192** (energy/EBM; oracle 0.569→0.708) — keeps improving well past 1024, exactly as the user remembered. My fast p3@NFE25 seq_cond 7AD5 is **flat at 0.398** (EBM stuck on one sample; oracle ceiling only 0.59). So the fast config both lowers the achievable pool AND breaks EBM selection on hard proteins.
**→ The "8× at 97%" held only at 16 samples on the broad 26-set. At the real operating point (8192 samples, hard targets) the fast config likely loses substantial scaling headroom. CONCLUSION PENDING the proper apples-to-apples.**

### ⭐⭐⭐ APPLES-TO-APPLES RESULT (seq_cath, 8 scaling8 targets, 8192, recycles 6 — CLEAN, same conditioning) — single-cutoff @8192 EBM-selected ref_pred_tm:
| protein | L | FAST p3@NFE25 (8×) | SDE-200 NFE200 | Δ |
|---|---|---|---|---|
| 8QXI | 77 | 0.466 | 0.467 | −0.001 |
| 8XHT | 100 | 0.200 | 0.207 | −0.007 |
| 6ZYG | 112 | 0.285 | **0.437** | **−0.152** |
| 7AD5 | 124 | 0.588 | **0.684** | **−0.096** |
| 6ZUS | 143 | 0.249 | 0.274 | −0.025 |
| 8RJX | 173 | 0.417 | 0.424 | −0.007 |
| 7KW9 | 178 | 0.484 | 0.494 | −0.010 |
| 6ZTG | 192 | 0.231 | 0.246 | −0.015 |
| **mean** | | **0.365** | **0.404** | **−0.039** |
⭐ **THE FAST CONFIG IS WORSE THAN SDE-200 AT THE REAL 8192 OPERATING POINT** — mean −0.039 (~10% relative), driven by 7AD5 (−0.096) and 6ZYG (−0.152). **In seq_cath (same conditioning) ⇒ the loss is CONFIG-driven, not conditioning.** Confirms the "8× at 97%" was a 16-sample illusion. NOTE 7AD5 fast seq_cath=0.588 ≫ fast seq_cond=0.398 (seq_cath conditioning helps the fast config a lot) but still < SDE-200 seq_cath 0.684. FULL convergence curve (all cutoffs) + ORACLE line from job 4986865 (energy launched) → will show whether the fast config's POOL is worse (achievable ceiling) or just EBM selection.
### ⭐⭐ FAST-vs-SDE200 seq_cath ENERGY CONVERGENCE (job 4986865 CV8_RC=0; 8 prot, EBM-selected mean):
| #samp | FAST p3@NFE25 | SDE-200 NFE200 | Δ |
|---|---|---|---|
| 256 | 0.315 | 0.331 | −0.016 |
| 512 | 0.340 | 0.332 | +0.008 |
| 1024 | 0.345 | 0.345 | 0.000 |
| 2048 | 0.345 | 0.367 | −0.022 |
| 4096 | 0.348 | 0.371 | −0.023 |
| 8192 | 0.378 | 0.404 | −0.026 |
⭐ **TIED at low N (256-1024); SDE-200 pulls ahead at high N (Δ−0.026 @8192, ~6.5% rel).** SDE-200 has more inference-time-scaling headroom (FAST flattens 1024-4096 at ~0.345-0.348 then ticks to 0.378; SDE-200 climbs steadily to 0.404). → the fast schedule's quality cost shows up specifically at the HIGH-sample operating point the downstream pipeline uses. **ORACLE conv = job 4987469 (launched)** → achievable-ceiling (pool) comparison: if FAST oracle ≈ SDE oracle, the gap is EBM-selection; if FAST oracle < SDE oracle, the fast POOL is genuinely worse. SDE seq_cond gen resumed: released 49-6ZYG, 50-8XHT; re-ran 6ZTG SDE = job 4987470.
### ⭐⭐⭐ FAST-vs-SDE200 seq_cath ORACLE CONVERGENCE (job 4987469 CV8_RC=0, 17:29; 8 prot, true-TM-selected ceiling = max_min_tm_ref_pred):
| #samp | FAST oracle | SDE-200 oracle | Δ | | FAST energy | SDE-200 energy |
|---|---|---|---|---|---|---|
| 256 | 0.379 | 0.419 | −0.040 | | 0.315 | 0.331 |
| 1024 | 0.404 | ~0.44 | ~−0.04 | | 0.345 | 0.345 |
| 8192 | **0.449** | **0.459** | **−0.010** | | 0.378 | 0.404 |
⭐⭐ **ANSWER — the fast config's POOL is NEARLY as good as SDE-200; the deficit is mostly EBM SELECTION.** Oracle ceiling @8192: FAST 0.449 vs SDE 0.459 = **−0.010** (pool barely worse) vs energy @8192 −0.026 (deployed gap). **EBM-selector gap (oracle−energy) @8192: FAST 0.071 > SDE 0.055** → EBM leaves MORE on the table for fast samples. Oracle scales to 8192 (flat ~0.40 then JUMPS to 0.449 @8192, driven by 7AD5 0.567→**0.724** appearing in the big pool) — confirms the user's memory that 7AD5 keeps improving past 1024. **Per-protein @8192 (energy → oracle, selector gap):** 7AD5 0.567→0.724 (0.157), 6ZYG 0.293→0.456 (0.163), 8XHT 0.247→0.420 (0.173), 6ZUS 0.256→0.328, 6ZTG 0.234→0.250, 8QXI 0.485→0.473, 8RJX 0.426→0.421, 7KW9 0.519→0.518. The big selector gaps are exactly the hard proteins (7AD5/6ZYG/8XHT) where EBM mis-ranks. **⇒ STRATEGIC: the fast config is viable IF we fix EBM selection on its samples — directly motivates the Q4 relaxation experiment (recover up to ~0.07 headroom → close most of the −0.026 deployed gap → 8× speedup at ~SDE-200 quality).** ⚠️ PROVISIONAL: scored under OLD af2rank (recycles 6, pre-tm_io/mask fix) per the §AF2RANK TODO; the RELATIVE fast-vs-SDE comparison (same protocol both sides) is robust, absolutes will shift on re-score.
**APPLES-TO-APPLES NOW RUNNING:** fast p3@NFE25 in **seq_cath, 8192, on the 8 scaling8 targets** (reusing scaling8.csv + the SDE-200 seq_cath scaling8 baseline), matched settings (recycles 6, af2rank_top_k 8, cutoffs 256-8192). Launcher `scripts/scaling_run_sc8.sh <cfg> <pid>`; smoke = 8QXI job **4980637**; fan out 7 more on validation → convergence (energy+oracle) vs SDE-200 scaling8 (energy 0.331→0.404, oracle 0.419→0.459). seq_cond gap: SDE-200 seq_cond only has 1039 samples (no 8192) → a seq_cond apples-to-apples would need fresh SDE-200 8192 generation (slow) — deferred/ask.
**OLD HEADLINE BELOW IS SUPERSEDED on the baseline-comparison points; Step B/C schedule + commitment findings still stand.**

### CAMPAIGN STATUS (dated blow-by-blow → `PLAN-RAW-ARCHIVE.md`)
Targets = 8 scaling8 (8XHT,7AD5,6ZUS,8QXI,6ZTG,7KW9,8RJX,6ZYG), 8192, recycles 6, cutoffs 256-8192, cif natives. Configs FAST=`..._spdref_vf_powerp3_nfe025_scale`, SDE=`..._unified_sderef8k`. Launchers `scaling_run_sc8.sh`/`scaling_run_seqcond8.sh`/`conv_sc8.sh`; parse `conv_summary.py`.
**DONE:** seq_cath FAST + SDE-200 energy & oracle convs (results above); SDE seq_cond energy & oracle convs (below); fast seq_cond gen 8/8.
**REMAINING (priority order):** (1) FAST seq_cond energy conv 4991367 → oracle → fast-vs-SDE seq_cond head-to-head; (2) Q4 relaxation compare (below); (3) AF2Rank re-score ALL analyses (§AF2RANK CORRECTION); (4) Step D/E (parked, §3); (5) docs cleanup [DONE 2026-06-27].

### ⭐ Q4 — ProteinEBM RELAX-THEN-SCORE: readiness + un-relaxed BASELINE (2026-06-26; FANNED OUT 2026-06-27, see end of section)
**Motivation (user):** low-NFE fast samples have imperfect LOCAL geometry (cut low-noise steps) → ProteinEBM scores a structure AS-IS (no relax) → it penalizes local imperfection instead of judging the FOLD → weak selection. Test: run a short ProteinEBM reverse-diffusion relaxation (t_max→t_min) on each sample BEFORE scoring; does Spearman(EBM score, tm_ref_template) improve? Decided metric (user): rank by BOTH energy + pTM, metric = Spearman vs tm_ref_template.
**ENGINE READY (no code change):** `ProteinEBM/protein_ebm/scripts/relax_decoys.py` (noise→t_max, reverse_steps×steps Langevin →t_min, emit model t0 pred; best-of-N by energy via `--batches_per_protein`; defaults t_max0.025/t_min0.01/reverse5/steps5/dt0.001) + `score_decoys.py` (scores a decoy-.pt dir → per-decoy energy, or pTM with `--use_pae`, over a t-grid; reads optional `tmscore` key → internal spearman). Decoy .pt = {atom37[B,N,37,3] Å, aatype[B,N], idx[B,N], atom37_mask[B,N,37]}; CA at atom index 1; the model reads CA from atom37 and PREDICTS sidechains itself (so a CA-only input is sufficient).
**GAPS (glue to write — why it is not one command):** (1) CA-only proteina sample (PDBs in `<pid>.tar`) → decoy `.pt` converter — DOES NOT EXIST; but the featurization already exists inside proteina `proteinebm_scorer.py` (it already scores these exact CA-only samples → reuse, low risk). (2) rescore-relaxed-dir + join to `tm_ref_template` + Spearman-compare harness (a few lines; `proteinebm_scores_<pid>.csv` already holds the join key).
**DIFFUSER WINDOW (grounded; r3_diffuser VP-SDE min_b0.1/max_b20/coord_scaling0.1):** t∈[0,1]; **t=0 is SINGULAR in compute_score (÷conditional_var(0)=0) → t_min MUST be >0** (default 0.01). Noise std: t=0.025→0.93 Å (gentle, fix local geom only), t=0.05→1.72 Å (1.84×, risks perturbing the fold). User's "0.05→0" → faithful interpretation = t_max0.05→t_min0.01.
**⭐ UN-RELAXED BASELINE (FREE — from existing `proteinebm_scores_<pid>.csv`; FAST seq_cath p3@NFE25, ~8192 samples/protein) — current EBM selector quality on the sped-up samples:**
| pid | L | ρ(E,TM) | ρ(pTM,TM) | tm@bestE | tm@bestpTM | tm_oracle | tm_mean |
|---|---|---|---|---|---|---|---|
| 8XHT | 100 | −0.049 | 0.208 | 0.181 | 0.196 | 0.381 | 0.192 |
| 7AD5 | 124 | −0.206 | 0.188 | 0.172 | 0.178 | 0.552 | 0.240 |
| 6ZUS | 143 | −0.163 | 0.178 | 0.200 | 0.192 | 0.331 | 0.179 |
| 8QXI | 77  | −0.720 | 0.728 | 0.468 | 0.457 | 0.535 | 0.381 |
| 6ZTG | 192 | −0.037 | 0.203 | 0.212 | 0.236 | 0.284 | 0.224 |
| 7KW9 | 178 | −0.330 | 0.424 | 0.406 | 0.419 | 0.476 | 0.231 |
| 8RJX | 173 | −0.566 | 0.574 | 0.362 | 0.326 | 0.399 | 0.280 |
| 6ZYG | 112 | +0.032 | 0.041 | 0.215 | 0.242 | 0.374 | 0.227 |
| **MEAN** | | **−0.255** | **0.318** | **0.277** | **0.281** | **0.416** | **0.244** |
**READ:** EBM energy is a WEAK selector on fast samples (mean ρ=−0.26; a good selector would be strongly negative); **BIMODAL** — strong on 8QXI(−0.72)/8RJX(−0.57), ~zero or WRONG-signed on 6ZYG(+0.03), 6ZTG(−0.04), 8XHT(−0.05), 7AD5(−0.21). Large selection gap: tm@bestE 0.277 vs tm_oracle 0.416 (7AD5: EBM picks a 0.172-raw-TM sample while a 0.552 exists). This QUANTIFIES the exact problem the relaxation targets. ⚠️ These are RAW sample TM (`tm_ref_template`, pre-AF2Rank); the §APPLES headline 0.365 is POST-AF2Rank `tm_ref_pred` (refinement lifts 7AD5 0.172→0.588). Relaxation acts on the RAW-selection layer → if it lifts ρ, better raw folds feed AF2Rank top-k → higher final. (Local compute: scratchpad ebm_baseline_fast_seqcath.csv + baseline_spearman.py.)
**DECIDED (user 2026-06-26):** windows = BOTH t_max0.05→0.01 AND t_max0.025→0.01 (reverse5/steps5/dt0.001); scope = ALL 8 proteins, full pool (~65k structs × 2 windows). **CHAIN (grounded, no repo edits):** `make_decoys.py` (reuses `residues_to_features`, joins tm by structure_file → decoy `.pt` w/ tmscore) → `relax_decoys.py` (×2 windows) → `score_decoys.py` @t=0.05 (energy; +`--use_pae` for pTM; prints internal Spearman(score,−tm)) → compare ρ raw-vs-relaxed. Scripts in `~/proteina_speedup/scripts/{make_decoys.py,smoke_relax.sh}` (deployed). **SMOKE FIRST (job 4987936, 8QXI×64, t_max0.025):** validates (a) featurization, (b) relax_decoys runs on this CA model (the [N,37]-vs-[N] atom_mask risk), (c) score_decoys loads pae.ckpt for energy & RAW ρ≈baseline +0.72. Build full-scale launchers (all 8 × both windows + compare) only AFTER smoke passes. SDE-200 seq_cath per-sample EBM scores NOT on disk (only FAST config has `proteinebm_scores` CSVs) → SDE-vs-fast selector comparison would need re-scoring; convergence already shows SDE selects better at scale (energy 0.404 vs fast 0.378 @8192).
**RELAX INFRA STATUS (2026-06-27): loader VALIDATED, 3 bugs fixed, smoke re-validating.** Loader fix DONE (`*_fix.py` use ProteinRegressionTrainer→.ebm; RAW ρ=+0.685≈baseline +0.72). make_decoys now reads from `<pid>.tar` (the protein dirs were tarred by the convs). atom37_mask stored as ones[B,N] (CA model wants per-residue mask). ⛔ **GOTCHA: relax `model.compute_score` = ∇ₓE → builds the backward graph (~3× a forward) → OOM at batch 64/N=78** while `score_decoys`'s `compute_energy` (forward-only) fit; fix = tiny relax batch (`--base_batch_size 6 --max_batch_size 12 --min_batch_size 1`; N=78→12, N=192→2). Full launcher `relax_full.sh <pid> <t_max> <tag>` deployed (convert→raw-score→relax→relaxed-score, energy+pTM, idempotent shared steps); fan out 8×2 windows once smoke (job 4989971) passes.
### ⭐⭐ SDE seq_cond ENERGY CONVERGENCE (job 4989011 CV8_RC=0; the primary-conditioning 8192 baseline that never existed) — EBM-selected mean max_min_tm_ref_pred:
| #samp | 256 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| SDE seq_cond energy | 0.329 | 0.339 | 0.404 | 0.404 | **0.410** |
| SDE seq_cond oracle | 0.426 | 0.428 | — | — | **0.460** |
SDE seq_cond @8192 (energy 0.410, oracle 0.460) ≈ SDE seq_cath (energy 0.404, oracle 0.459) → conditioning mode barely changes the SDE ceiling. **SDE seq_cond conv DONE (energy 4989011, oracle 4989964).** FAST seq_cond gen 8/8 done → FAST seq_cond ENERGY conv = job 4991367 (launched); on CV8_RC=0 → oracle → the fast-vs-SDE seq_cond head-to-head completes the campaign. ⚠️ PROVISIONAL (old af2rank, rec6) per §AF2RANK TODO.
### ⭐ Q4 RELAXATION — FANNED OUT (2026-06-27, smoke validated end-to-end)
Smoke `4989971` ALL GREEN; 8QXI (control, already-good selector) raw ρ 0.720 → relaxed ρ 0.725 (neutral, as expected — real signal is on broken-selector proteins). Full run LAUNCHED: `relax_full.sh <pid> {0.025,0.05}` for all 8, per-pid serialized (t0025 jobs 4991368/70/72/74/76/78/80/82 do convert+raw-score+relax; t005 dependents reuse). Ranks by energy AND pTM @ scoring t=0.05. ⛔ COST: `compute_score`=∇ₓE (backward graph) → tiny batch (N=78→12, N=192→2) → big proteins ~3-4h/window. Compare = `scripts/compare_relax.py --pids ...` → ρ(score,tm) + tm@best raw-vs-t0025-vs-t005, per protein + mean (does relaxation lift the broken selectors 7AD5/6ZYG/8XHT?).

## KEY RESULTS — autonomous window (2026-06-25; full directive + step-by-step decisions log → `PLAN-RAW-ARCHIVE.md`)
### ⭐ HEADLINE RESULTS (as of ~07:15 2026-06-25, for QC) — bad_afdb held-out, gate = af2rank refined-pred-TM (ref_pred_tm), vs SDE-200 baseline 0.348
- **Step A baseline:** SDE-200 ≈ ODE-200 (ref_pred_tm 0.348) → ODE is the right sampler to accelerate (and ODE holds up far better than SDE at low NFE).
- **Step B (approach #2) = THE WIN:** a HIGH-noise-emphasis `power` schedule on the ODE gives **8× speedup at 97% of baseline** (power p3 @ NFE25 = 0.338 vs 0.348). Full ceiling: p3@NFE25(8×)=0.338, p4@NFE15(13×)=0.325, p5@NFE10(20×)=0.303 — optimal exponent RISES as NFE drops. Schedule is **ODE-SPECIFIC** (same schedule on SDE = 0.265, doesn't transfer). Generalizes to seq_cath (0.334).
- **Step C (approach #3):** the model commits its fold at HIGH noise (t≲0.275), **LENGTH-DEPENDENT** (longer proteins commit earlier/higher-noise) — mechanistically explains why high-noise emphasis works + why optimal p rises with speed.
- **Item 3 (scaling + ProteinEBM-vs-Oracle, 4-prot subset):** inference-time scaling **SATURATES by ~256–512 samples**; EBM-vs-oracle gap **~0.05** (ProteinEBM is a decent selector); fast config + a few-hundred samples (~0.47–0.49) ≈ achievable ceiling (~0.53) ≫ baseline (0.348).
- **PENDING:** 8AUC (492, job 4974418) — sstat-confirmed ALIVE (~100% CPU); 8192×492-res at NFE25 ≈ ~5h SAMPLING alone (per L²-scaled rosy-fox timing), ~done ~08:00 + ~1h enrichment ~09:00. Plan: record its energy@8192 from prediction_summary (cheap); the **4-protein convergence is the main result** — will only re-run the full 5-protein convergence if time/nodes permit (low value-add: conclusion robust). p4-aggressive convergence = optional bonus. Detailed numbers + job IDs in the decisions log below.

### Step-by-step decisions log → `PLAN-RAW-ARCHIVE.md`. Distilled conclusions are in HEADLINE RESULTS above + the §APPLES-TO-APPLES / §SDE seq_cond / §Q4 sections below.

---

## 0. GOAL & MOTIVATION (the WHY — do not lose this)

**Objective:** Cut the number of denoising/integration steps (NFE) — hence wall-clock per sample — of Proteina's **sequence-conditioned** structure sampling, **without degrading the fold/topology quality** that our downstream **template-search pipeline** depends on. Current default ≈200 steps; naively dropping to 100 already degrades quality.

Three candidate directions (user-proposed; we critically evaluate + may add better ones from the lit search):

1. **Flow-map / few-step sampling.** Reduce denoising steps via a principled few-step method rather than naive step-count reduction (which degrades quality). Search literature deeply (flow maps, consistency, shortcut, distillation, fast ODE solvers).
   - *Why:* naive 200→100 already loses quality, so we need a method that preserves the learned transport with fewer function evals.

2. **Smarter sigma (noise/time) schedule — non-uniform step allocation.** Spend steps where the model does fold-determining work; take big steps where it doesn't.
   - *User's reasoning:* at **low noise** the model mostly refines **local detail**, not global fold; at **very high noise** the model has no clear idea of the fold yet. So the fold-determining work is concentrated at **mid-to-high noise** → put most steps there, and **skip/cut** (big steps) at low and extreme-high noise.
   - *Resolution (Q2+Q3):* working hypothesis = **template search needs the global FOLD/topology only**; local detail (H-bonding, 2'-structure, sidechain packing) is expected to be repaired by the downstream **AF2Rank refinement** step. UNTESTED → validate empirically (cut low-noise steps, then check AF2Rank quality + fold recovery hold). If confirmed, aggressive low-noise step-cutting is safe.

3. **Measure our model's denoising dynamics — the "fold commitment" / critical-window experiment.** Borrowed from an image-diffusion analysis: save partially-denoised samples along the trajectory, then from a given noise level re-run many completions under different seeds and see when the samples all converge to the same **category**. In images the category was e.g. "cat"; **for us the category is the FOLD**. Goal: find the noise level σ* beyond which the fold is locked in but local details are not → tells us where steps actually matter and validates approach #2's schedule design empirically.
   - *Fold determination options (repo ships both):* GearNet CA classifier (`gearnet_ca.pth`, C/A/T = 5/43/1336 classes) for fast scan; and/or USalign / Foldseek to CATH cluster representatives for ground truth. (User Q on which — default: both.)
   - *Why:* converts the schedule intuition (#2) into a measured, model-specific cutoff instead of a guessed one (aligns with the no-invented-values rule).

4. **(COMBINED #1+#2) Schedule-aware distillation (user idea).** Don't distill a generic few-step model; first find the best non-uniform smart schedule (training-free, #2), informed by #3's fold-commitment cutoff, then **distill the few-step / flow-map model targeting THAT schedule's step boundaries**. Distillation concentrates capacity exactly where the smart schedule places its (few) steps.
   - *Why:* generic consistency/flow-map distillation spreads effort uniformly; distilling on the chosen schedule should yield more quality-per-step than #1 or #2 alone.

---

## 1. GROUNDED REPO FACTS (from code exploration — authoritative, file:line)

### Sampler / flow-matching formulation
- Interpolant: **linear / optimal-transport**, `x_t = (1−t)·x0 + t·x1`, vector field `v = (x1−x_t)/(1−t)`. `r3n_fm.py:124-211`.
- Integrator: **explicit Euler**. Main loop `_CoordinateFlowMatcher.full_simulation()` `r3n_fm.py:418-576`; step `step_euler()` `r3n_fm.py:269-351`.
- **NFE = ceil(1/dt)**. `dt=0.005`→200 steps; `dt=0.0025`→400. `r3n_fm.py:484-486`. Config key `dt` (`inference_base.yaml:21`).
- **Default sampler is the STOCHASTIC SDE** (`sampling_mode="sc"`, `inference_base.yaml:26`), not pure ODE. SDE adds noise: `sc_scale_noise` (≈0.4–0.45), `g(t)` via `gt_mode` (default `1/t`). ODE mode is `"vf"` (deterministic). `r3n_fm.py:269-351,578-637`.
  - *Implication:* training-free fast solvers (DPM-Solver, Heun, etc.) target the **ODE**. Whether we accelerate the SDE as-is or switch baseline to the ODE is a real fork → Q4.

### Schedule knobs ALREADY in code (approach #2 has levers)
- `get_schedule()` `r3n_fm.py:639-693`. Modes: `uniform`, **`log` (default, `schedule_p=2.0`)**, `power`, `cos_sch_v_snr`, `loglinear`, `edm` (rho-param). Config: `schedule.schedule_mode`, `schedule.schedule_p` (`inference_base.yaml:34-35`).
- Time↔noise: log-SNR `= 2·log(t/(1−t))` `r3n_fm.py:155-179`. (Lets us translate sigma-schedule ideas to t.)

### Fold/topology classification (approach #3 tooling)
- **GearNet CA**: `NoTrainCAGearNet` `gearnet_utils.py:480-512`; outputs `pred_C/A/T` (5/43/1336) + 512-d feature; input PyG batch of CA coords (atom index 1). Weights `$DATA_PATH/metric_factory/model_weights/gearnet_ca.pth`. Forward `gearnet_utils.py:406-442`.
- **USalign CAT-recovery**: `script_utils/cat_recovery_usalign.py` (align query → CAT rep library, TM1). Parser `usalign_tabular.py`. **Foldseek CATH50**: `scripts/foldseek_cath_batch.py`.
- CATH label mapping `cath_utils.py:21-174`; reps under `$DATA_PATH/cath_shared/...`.

### Trajectory capture (approach #3 needs a hook)
- **Coordinate flow loop does NOT expose/save intermediate x_t**, and has **no resume-from-σ** path. Only the discrete contact-map branch saves a trajectory (`save_trajectory`, `trajectory_stride`). `r3n_fm.py:519-575` is where x_t lives; capturing/re-injecting x_t needs a small code hook here. `model_trainer_base.py:3191-3274` (discrete only).

### Eval metrics available
- Designability scRMSD (ProteinMPNN→ESMFold) `designability.py:301-352`, `rmsd_metric` `:246-298`. FID/IS_C/A/T/fJSD_C/A/T via GearNet `metric_factory.py:34-194`. **No built-in TM-score** (use USalign). Output = CA-only → atom37 reconstruct `proteina.py:794-842`.

---

## 1.5 LITERATURE SHORTLIST (from workflow `wid8pfxlf`; raw → docs/inference-speedup-litsearch-raw.json)

**Framing risk (#1):** Proteina's load-bearing default is the STOCHASTIC SDE; nearly all fast solvers are derived for the deterministic ODE. Test BOTH samplers every time; pair any ODE solver with a stochasticity source. Image-domain NFE numbers do NOT transfer to fold quality — validate on our gates; realistic training-free landing zone ≈ 20–50 NFE, not single digits.

### Training-free (try first)
- **torchdiffeq RK4/Heun on the velocity field** — most FM-native, no x0 reparam; cheapest first win.
- **OSS — optimal-stepsize DP schedule** (Pei 2025, arXiv:2503.21774, bebebe666/OptimalSteps) — native FM update; ~2–4× (200→~50–100). TF-1.
- **GITS — geometry-inspired step schedule** (Chen ICML2024, arXiv:2405.11326, zju-pi/diff-sampler) — A/B vs OSS on same dumped trajectories. TF-2.
- **EDM Heun(2nd-order)+churn** (Karras 2022, arXiv:2206.00364, NVlabs/edm) — only training-free option that natively keeps tunable noise; `schedule_mode='edm'` already wired. TF-3.
- **SDE-native 2nd-order solver** (SEEDS / SDE-DPM-Solver++(2M-SDE)) — keeps `sc` noise at ~20–50 NFE; own workstream, directly de-risks ODE-vs-SDE mismatch.
- **UniPC / DPM-Solver++(2M)** (arXiv:2302.04867 / 2211.01095) — diffusers FM path; ODE-only, disable thresholding, exact FM→x0 reparam. TF-4.
- **SD3 time-shift** `t'=a·t/(1+(a−1)·t)` (Esser 2024, arXiv:2403.03206) — free 1-param warp; fit `a(length)`, do NOT port a=3. Schedule knob for #2.
- **AYS** (arXiv:2404.14507) — stackable schedule refinement (~1.3–2×); apply after a solver is chosen.
- **ParaDiGMS** (Shih NeurIPS2023) — parallel-in-time; trades VRAM for wall-clock (our metric). Optional.

### Training-required (later, aggressive regime)
- **SiD-Protein — Score-identity Distillation of Proteina ITSELF** (Xie Oct2025, arXiv:2510.03095, LY-Xie/SiD_Protein) — 16-step ≈ 400-step teacher on designability, ~24× wall-clock; usable 8–20 steps, 1-step collapses. **CAVEAT:** public ckpt distills the UNCONDITIONAL FS-no-tri model; our seq-conditioned ckpt needs re-running SiD (real engineering). Slightly WORSE novelty (AFDB-TM ~0.80 vs 0.84). TR-1 / headline prize.
- **LD3 / Differentiable Solver Search** (arXiv:2405.15506 / 2505.21114) — learn ~N timestep scalars, net frozen; DSS validated on rectified-flow. ~2–4× safe. TR-2.
- **Reflow** (rectify-then-low-NFE; SO(3) conformer reflow arXiv:2507.09785) — fallback distillation backbone; aim few-step not 1-step (diversity-collapse risk).
- **(#4 combined) schedule-aware distillation** — distill targeting the chosen smart schedule's step boundaries + #3 cutoff.

### De-prioritized
- MeanFlow / Shortcut / Consistency-FM / FlowSolver — FM-native but zero protein results; SiD-Protein already covers our model class.
- "Flows straight but not so fast" (arXiv:2510.24732) — SE(3)-frame not CA-only; its t≈0 step-concentration is rotation-curvature-driven → re-measure direction, don't port.

---

## 1.6 SUPERCLOUD PIPELINE — GROUNDED (read before launching)

- **User=cou, repo `/home/gridsan/cou/proteina`, `DATA_PATH=/home/gridsan/cou/proteina/data`.** Eval cifs+csvs in `~/data/bad_afdb/`. Repo/account SHARED → read-only on others' work; never cancel others' jobs.
- **Env (only `cathfold` + `proteina` exist; cue_openfold/colabdesign ABSENT):** `module load conda/Python-ML-2025b-pytorch cuda/12.6 && source activate cathfold`; ALWAYS pass `--direct_python` (skips missing wrapper envs); V100 flags `--no-use_deepspeed_evoformer_attention --use_cuequivariance_attention --use_cuequivariance_multiplicative_update`; `export PROTEINA_CONDITIONING_MODE={seq|seq_cath}` (mandatory, no fallback); USalign on PATH (`$HOME/.local/bin`).
- **Two entry points:** `parallel_proteina_inference.py` (sharded SAMPLING only; `--conditioning_mode`, `--nsamples_per_protein`, length-sharded across GPUs, OOM auto-halve, `--skip_existing`) and `run_prediction_pipeline.py` (ORCHESTRATOR: sample→reconstruct→score; `--scorer {af2rank|proteinebm}`, `--af2rank_backend openfold`, `--segment_mode {off|joint}`, `--direct_python`).
- **AF2Rank gate output:** `--scorer af2rank` writes per-protein `inference/<config>/<cond_subdir>/<protein>/af2rank_analysis/af2rank_scores_<protein>.csv` with cols `ptm, plddt, pae_mean, composite(=ptm·plddt), tm_ref_*` (`af2rank_openfold_scorer.py:_extract_scores:1053`, composite:1074). proteinebm+top-k path only scores top-5 and its summary lacks composite.
- **⛔ NFE/schedule control = `dt` + nested `schedule.*`/`sampling_caflow.*` IN THE CONFIG ONLY. NO CLI/env override (`hydra.compose` has no `overrides=`).** Sweep without editing tracked files: COPY the unified config to a new untracked name, change ONLY the `inference:` block (`dt`, `schedule.schedule_mode/p`, `sampling_caflow.sampling_mode/sc_scale_noise`), KEEP `run_name_` identical (ckpt auto-resolves from `./store/<run_name_>/checkpoints/`). Output dir keyed by config filename → per-NFE configs auto-isolate. `nsamples_per_len` DOES have an override: `--nsamples_per_protein` / `PROTEINA_NSAMPLES_PER_PROTEIN`.
- **Baseline config (2-seq unified, current bad_afdb):** `..._default-fold-sum_2-seq-S25_16-eff-bs_200-epoch_..._maxlen-256_cutoff-190828_unified.yaml`: `dt=0.005`(200 NFE), `schedule=log/2.0`, `sampling_mode=sc`, `sc_scale_noise=0.45`, `self_cond=True`, `nsamples_per_len=1024`. **21-seq unified** = identical but `nsamples_per_len=512`. NFE=ceil(1/dt) `r3n_fm.py:484`; schedule modes: uniform/power/log/inv_smooth_step/cos_sch_v_snr/edm.

## 2. DESIGN DECISIONS — RESOLVED (user, this session)

- **Q1 → Training-free FIRST, then distillation.** Plus user idea: **combine** — distill specifically on the chosen smart schedule (approach #4).
- **Q2 → Quality gates (priority order):**
  1. **AF2Rank prediction-quality distribution** [PRIMARY]. Rationale (user): AF2Rank refinement fixes local imperfections (residue-interaction detail, 2'-structure H-bonding), so if sampled TOPOLOGY is good the final AF2Rank prediction is high-quality.
  2. **Fold/topology recovery** (GearNet C/A/T + USalign-to-CATH).
  3. **Distributional FID/fJSD.**
  (Self-consistency-vs-200-step and scRMSD = optional diagnostics, not gates.)
- **Q3 → Working hypothesis: global FOLD/topology only** (local detail repaired by AF2Rank). **UNTESTED — must validate** via the Q2 gates while ablating low-noise steps.
- **Q4 → Evaluate BOTH** SDE ("sc", production default) and ODE ("vf") baselines.
- Fold classifier for #3: **BOTH** GearNet (fast scan) + USalign-to-CATH (ground truth).

### RESOLVED (final decisions)
- **Eval set = the difficult HELD-OUT set** `bad_afdb_ref/pdb_70_cluster_reps_aligned_confidence_aggregate_monomer_tm-05_coverage-08_identity-07_length_50-512_with_cath.csv` (SuperCloud deployment path `~/data/bad_afdb/`; local copy in `SOLab/bad_afdb_ref/`). **26 sequences**, all `in_train=False`, len 62–492, diverse CATH. Columns: `pdb` (chain id = the condition), `cath_code` (CAT-level, H→x, e.g. `3.30.160.x`), `plddt`, `length`, `tm_score`, … **2 seqs have NO CATH (`x.x.x.x`): `7F7N_A`, `8AP5_A`** → seq_cond only (exclude from seq_cath_cond). Benchmark **BOTH** `seq_cond` and `seq_cath_cond`.
- **Budget:** all 26 seqs × **16 samples**/config (= 416 structures/config). Sweep over NFE × schedule × sampler × conditioning-mode.
- **Gate metric:** AF2Rank **`composite = ptm·plddt`** distribution vs the 200-step reference; also report `ptm`, `plddt` (+ `pae_mean`). Scorer `af2rank_scorer.py:833-845` (ColabDesign `ModernAF2Rank` / OpenFold `OpenFoldAF2Rank`; CA→all-atom via cg2all).

---

## 3. PROVISIONAL PLAN (locks after Q1–Q4) — do not execute science-affecting steps before decisions

### Step A — Establish baselines & harness on SuperCloud  [x] DONE
⭐⭐ **CANONICAL INFERENCE DOC (READ FIRST, do NOT guess flags):** `~/.claude/plans/users-chenxi-downloads-s41467-025-67127-rosy-fox.md` (+`-SUMMARY`) = the user's proteina inference project. Proper mechanism: env overrides `PROTEINA_{NSAMPLES_PER_PROTEIN,INFERENCE_PRECISION,CHECKPOINT_MODE}` (orchestrator has NO CLI flags for these, nor `--skip_existing`); isolation = cp config keep `run_name_`; model = 2-seq `chk_best epoch=180` (default `best`); **fp16 sampling validated ~2× faster + quality-equiv** (⛔ fp16 af2rank BROKEN on V100 → score fp32); **af2rank @1 recycle ≈ @6** (3× cheaper); production scorer = `--scorer proteinebm` (ProteinEBM ranks ALL samples fast → af2rank only on top-k). Pipeline gotchas (tar corruption from concurrent convs; dynamic_resharding drops proteins on late backfill → num_shards=1 for partial runs) are in that doc.
STATUS (2026-06-24): smoke PASSED. **REFERENCE = 4 SEPARATE single-shard jobs** (NOT a SLURM array): `4970369` sc_seq, `4970370` sc_seqcath, `4970371` vf_seq, `4970372` vf_seqcath; **PRODUCTION `--scorer proteinebm` (rank → af2rank top-5, recycles 6) + fp16 sampling**, isolated cfgs `..._spdref_{sc,vf}`, `--num_shards 1 --shard_index 0 --no-dynamic_resharding`. ⚠️ Earlier array `4970331_[0-3]` DEADLOCKED ("Waiting for N inference step shards (0/N done)") — a SLURM array over DIFFERENT configs/datasets is mis-detected by parallel_proteina_inference as shards of ONE run → siblings never align → use SEPARATE jobs. (Also superseded: af2rank-on-ALL grids 4970314/4970318 = wrong/slow scorer.) Launcher `~/proteina_speedup/scripts/ref_grid.sh <combo 0-3>`.
GATE: per-protein top-1 in `~/proteina_speedup/out/ref_<tag>/prediction_summary.csv` (best_ref_pred_tm = refined pred-TM = user's 'ours y'; best_ptm; best_proteinebm_ptm; best_energy; best_plddt). Distribution over 26 proteins = the speed-vs-quality gate. Per-sample composite/ptm in `inference/<cfg>/<cond>/<pid>` af2rank_on_proteinebm_top_k csvs. Collector: `~/proteina_speedup/scripts/collect_gate.py`.

⭐ **BASELINE RESULT (200-step, fp16, ProteinEBM→af2rank top-5; all 4 jobs RC=0):** gate = best_ref_pred_tm (refined pred-TM to native), `out/gate_summary.csv`:
| config | n | ref_pred_tm mean/med | af2rank pTM mean | proteinEBM pTM mean |
|---|---|---|---|---|
| SDE seq      | 26 | 0.348 / 0.297 | 0.492 | 0.536 |
| SDE seq_cath | 24 | 0.355 / 0.289 | 0.497 | 0.550 |
| ODE seq      | 26 | 0.348 / 0.296 | 0.442 | 0.350 |
| ODE seq_cath | 24 | 0.367 / 0.301 | 0.453 | 0.359 |

⭐ **KEY FINDING — ODE (vf) ≈ SDE (sc) in FINAL refined quality at 200 steps** (ref_pred_tm ~0.348 both) → the deterministic ODE clears the same downstream bar as the production SDE, so we CAN accelerate the ODE (far more amenable to fast solvers / flow maps) without losing final quality — **directly de-risks risk #1 (ODE-vs-SDE)**. Caveat: ODE samples have LOWER confidence signals (proteinEBM pTM 0.35 vs SDE 0.54; af2rank pTM 0.44 vs 0.49) — confidence ≠ true TM; EBM→af2rank selection still lands the same final quality. seq_cath ≈ seq (small +). Absolute TM ~0.30 (bad_afdb is a HARD set → limited dynamic range; the canonical doc's "pTM ranks well only with TM dynamic range" caveat may matter when comparing sweeps). NEXT (await user): Step B schedule sweep + Step C commitment, using ODE as the primary acceleration target.
- Eval set = bad_afdb 26 held-out seqs (above). Generate 200-step references for the 2×2 grid {seq_cond, seq_cath_cond} × {SDE "sc", ODE "vf"} (16 samples/seq) → 4 reference sets. → verify: reproducible; all 4 sets generated + scored.
- **ISOLATION (mandatory):** generate under COPIED config names (`..._unified_spdref_{sc,vf}.yaml`; `run_name_` UNCHANGED → same 2-seq ckpt `chk_best_tmscore...epoch=180`) so outputs land in `inference/<copy>/...` and never touch the SHARED production tree. Target model FORCED to **2-seq unified** (only unified ckpt present; 21-seq unified untrained). cif dir `~/data/bad_afdb/pdb` is sharded by PDB[1:3] (`6UF2_A`→`pdb/UF/6UF2.cif`); seq loads from cif. Working dir `~/proteina_speedup/`.
- ⚠️ Existing prod `seq_cath_cond` bad_afdb run = 1024 samples + proteinebm-top-k AF2Rank (NOT our gate) → do NOT reuse/pollute; `seq_cond/` currently holds a different (foldbench) set. Our gate uses `--scorer proteinebm` (ProteinEBM ranks all samples fast → af2rank top-5; matches production, far faster than af2rank-on-all).
- Check Proteina's TRAINING-time t-sampling distribution (uniform? logit-normal?) — it sets the prior on where inference steps should be densest. (gap-critic #6)
- Stand up the **3 quality gates** as scripts: (1) AF2Rank quality (reconstruct atom37 → AF2Rank → read quality metric); (2) fold recovery (GearNet + USalign); (3) FID/fJSD. → verify: each runs end-to-end on 1 sample, then the set.
- Wall-clock baseline per sample at 200 steps (both samplers). → verify: timing logged.

### Step B — Approach #2 first (cheapest, levers already exist)  [~]
⭐ **t-DIRECTION CONFIRMED (r3n_fm.py:512,520,555,684-690):** t=0 = NOISE (sample_reference prior), t=1 = DATA (x_1_pred clean); loop integrates 0→1. The DEFAULT `log`/p2 schedule is **DENSE near t=1 (LOW noise), SPARSE near t=0 (HIGH noise)** — gaps [0.69,0.22,0.068,0.022] for 4 steps — i.e. it already spends most steps on low-noise/local refinement, the OPPOSITE of the user's mid-to-high-noise hypothesis. So testing the hypothesis = a schedule DENSE near t=0: **`power` p1>1** gives increasing gaps (dense near t=0 = high-noise emphasis). `edm`/`cos_sch_v_snr` are SNR-parameterized alternatives. Model returns `x_1_pred` (clean pred) EVERY step → free for Step C x0-readout. Near t>0.99 the sampler force-switches to vf. SWEEP CONFIGS via copied yamls keeping run_name_; NFE=ceil(1/dt); sweep af2rank at **recycles=1** (≈recycles=6, 3× cheaper, runbook-validated). ⭐ **PHASE 1 RESULT — ODE (vf) NFE-degradation @ default log schedule, recycles=1, vf-seq 26 prot** (`out/gate_summary.csv`, jobs 4971663-66):
| NFE | speedup | ref_pred_tm mean/med | af2rank pTM | proteinEBM pTM |
|---|---|---|---|---|
| 200 (baseline, rec6) | 1× | 0.348 / 0.296 | 0.442 | 0.350 |
| 100 | 2× | 0.326 / 0.317 | 0.379 | 0.344 |
| 50  | 4× | 0.298 / 0.263 | 0.346 | 0.304 |
| 25  | 8× | 0.277 / 0.257 | 0.299 | 0.274 |
| 15  | 13× | 0.278 / 0.268 | 0.311 | 0.277 |

READ: smooth monotonic degradation. **NFE100 (2×) ~94% of baseline (−6%); NFE50 (4×) ~86% (−14%); NFE25/15 ~80% (−20%, floor, 25≈15).** Confidence (pTM/EBM) drops FASTER than refined quality. CAVEAT: baseline rec6, sweep rec1 (runbook 1≈6; rec1-200 cross-check not yet run). This DEFAULT log schedule (dense at LOW noise) = the curve to BEAT.
⭐ **BASELINE = SDE-200 (user 2026-06-24), reuse existing samples, NEVER overwrite.** SDE-200 dist already exists (`out/ref_sc_seq`, rec6) = 0.348/0.297 ≡ ODE-200 → used read-only as reference (rec1≈rec6 runbook). ⚠️ Pipeline OVERWRITES if re-run on an existing config dir / with fewer samples → all new experiments use FRESH config names (fresh inference dirs); to re-score existing samples, COPY (not symlink — re-tar writes through) into a dummy config dir. rec1-exact SDE-200 baseline available on request via dummy+copy + `--rerun_af2rank_on_top_k`.
⭐ **PHASE 2 LAUNCHED (schedule hypothesis test, vf-seq, rec1):** {power p2, power p3, edm rho7} × {NFE25, NFE50} = 6 fresh configs `..._spdref_vf_{powerp2,powerp3,edm}_nfe0NN`, vs the phase-1 default-log points (nfe025=0.277, nfe050=0.298) and SDE-200 baseline 0.348. power p>1 / edm = HIGH-noise emphasis (the hypothesis). Jobs spd_swp.
⭐⭐ **PHASE 2 RESULT — HYPOTHESIS CONFIRMED:** ref_pred_tm by schedule (vf-seq, rec1):
| NFE | log (default) | power p2 | power p3 | edm rho7 |
|---|---|---|---|---|
| 25 (8×) | 0.277 | 0.325 | **0.338** | 0.320 |
| 50 (4×) | 0.298 | 0.312 | **0.330** | 0.323 |
PHASE 2 COMPLETE (all 6 done). NFE25 ranking: **power p3 (0.338) > power p2 (0.325) > edm (0.320) ≫ log (0.277)**; NFE50: power p3 (0.330) > edm (0.323) > power p2 (0.312) > log (0.298). ALL high-noise-emphasis schedules beat default log; **power p3 is the consistent winner**, +0.06 at NFE25. **power p3 @ NFE25 = 0.338 ≈ SDE-200 baseline 0.348 → ~8× speedup at ~97% quality.**
NEXT (await user go): (1) push power p3/p4/p5 @ NFE15/10 (find ceiling); (2) Step C commitment → measured-optimal schedule; (3) confirm winner on seq_cath + SDE. ⭐ **power p3 @ NFE25 (8×) = 0.338 ≈ SDE-200 baseline 0.348 (−3%)**, and BEATS default-log @ NFE100 (0.326) with 4× FEWER steps. Monotonic in p (p3≥p2); confidence (pTM 0.37 vs 0.30) + plddt (0.51 vs 0.43) also much higher. → user's approach-#2 hypothesis (steps belong at mid-to-HIGH noise, NOT low-noise refinement) strongly VALIDATED; **~8× speedup at ~97% of baseline quality** with power p3. NEXT (await user): push further (power p3/p4 @ NFE15; even higher p); Step C t*→optimal schedule; confirm on seq_cath + SDE.

### Step B (original steps) — Approach #2  [ ]
- Sweep NFE × schedule_mode × schedule_p × (SDE vs ODE), training-free. → verify: quality-vs-NFE curve per schedule; find the non-uniform schedule holding quality at lowest NFE.
- Encode mid-to-high-noise emphasis as a schedule (via `edm`/`power`/custom). → verify: beats `log` default at matched NFE.
- **Q3-validation ablation:** specifically cut LOW-noise steps and check AF2Rank quality + fold recovery hold (tests global-fold-only hypothesis). → verify: AF2Rank distribution unshifted despite degraded local detail ⇒ hypothesis confirmed.

### Step C — Approach #3 commitment experiment  [~]
⭐ **IMPL DESIGN (code-grounded 2026-06-24):** the coordinate `full_simulation` (r3n_fm.py:519-576) computes `x_1_pred` (clean prediction) EVERY step (line 555) but returns only final `x`; the existing `save_trajectory` flag is discrete-branch only → need an ADDITIVE, ENV-GATED, default-OFF capture in r3n_fm.py:
  - `PROTEINA_TRAJ_DUMP=<path>`: collect (t, x_1_pred, x_t) per step → torch.save at loop end. Gives C0 x0-readout for free.
  - Fork: `PROTEINA_FORK_FROM=<x_t.pt> PROTEINA_FORK_T_IDX=<k>`: init x from saved x_t at schedule index k (instead of sample_reference) + fresh seed → resume. Verify: resume reproduces a full run when seed/t match.
  - full_simulation reads env directly (lowest level); caller sets a unique dump path per protein/run.
  - Driver = standalone script reusing `load_model_for_worker` (inference.py:408) + `run_one_protein_in_process` (inference.py:479) on a few diverse-fold bad_afdb proteins. Fold classify: GearNet CA + USalign-to-CATH.
  - ⚠️ r3n_fm.py is TRACKED + shared → edit LOCALLY, gate behind env (default off = ZERO behavior change, safe for other agent), push, pull on SuperCloud. NOT a monkeypatch (proper minimal feature).
- **C0 (free probe first):** plot TM(x0_pred(t), final) vs t using the model's x0/velocity prediction already computed each step — NO extra integration. Cheapest commitment signal; brackets the window before any fork sweep. (gap-critic #12)
- Add a minimal hook in `r3n_fm.py` (loop :519-575) to (i) dump x_t at chosen t and (ii) resume from a given x_t with a fresh seed. → verify: resumed run reproduces a full run when seed/t match.
- **Fork sweep:** for each t* on a grid, take one shared x_{t*}, run K=8 completions (different seeds), classify fold (GearNet + USalign-TM); measure mean-TM + cross-seed topology variance vs t*. → verify: t* = where TM crosses 0.5 / variance collapses.
- Also track designability (scTM/scRMSD) + novelty (max-TM to PDB/AFDB) per t — retention ≠ quality. (gap-critic #11)
- Hierarchical: GearNet C/A/T entropy vs t → commitment times t*_C ≥ t*_A ≥ t*_T (may differ per fold/length → length/fold-aware schedule).
- Feed t* back into Step B. → verify: schedule that coarsens steps outside [t* window] keeps fold recovery + AF2Rank quality.

### Step D — Approach #1 few-step (training-free first)  [ ]
- FM-native first: torchdiffeq RK4/Heun on the velocity field; then OSS/GITS DP schedules (warm-start from Step-B sweep). → verify: quality-vs-NFE beats Euler at matched NFE on the gates.
- Restore stochasticity for the SDE path: EDM churn OR an SDE-native 2nd-order solver (SEEDS / SDE-DPM++2M-SDE) — own workstream (de-risks ODE-vs-SDE). → verify: matches `sc` baseline quality at reduced NFE.
- ODE path: UniPC / DPM-Solver++(2M) via diffusers FM (thresholding off, exact reparam). → verify: ODE floor NFE.

### Step E — Distillation (aggressive regime) + #4 schedule-aware (combined)  [ ]
- TR-1: reproduce **SiD-Protein** (arXiv:2510.03095) — first confirm its public distilled ckpt's teacher; then re-run SiD against OUR seq-conditioned teacher targeting 16 NFE. Reflow = fallback. → verify: 16-NFE ≈ teacher on the 3 gates.
- #4 combined: distill targeting the Step-B smart schedule's step boundaries + Step-C cutoff. → verify: quality-per-step beats generic distillation.

### Step F — Combine & report  [ ]
- Best schedule (#2) + best solver/flow-map (#1/#4), validated by #3's cutoff; final speed vs quality table across all 3 gates.

---

## 4. CONSTRAINTS / REMINDERS
- SuperCloud: sbatch arrays (not salloc) over `xeon-g6-volta`; env `module load conda/Python-ML-2025b-pytorch cuda/12.6 && source activate cathfold`; V100 flags `--no-use_deepspeed_evoformer_attention --use_cuequivariance_attention --use_cuequivariance_multiplicative_update`. Login nodes REAP nohup pollers → chain steps with `sbatch --dependency=afterany:JOBID`, not a login poller. `<2000 files/dir`.
- No invented numbers/schedules/thresholds — ground every choice in lit search, repo, or our own runs; ask when unspecified.
- Conditioning_mode in inference MUST match analysis conditioning mode.
- Proteina outputs CA-only → reconstruct atom37 before any all-atom scoring.

## 5. TOP RISKS (from lit synthesis)
1. **ODE-vs-SDE mismatch (dominant).** Production default is the stochastic `sc` SDE; OSS/GITS/UniPC/DPM++/LD3/DSS are ODE-derived. Tuning on ODE and assuming SDE transfer wastes effort. Control: run both samplers every experiment; pair any deterministic solver with a stochasticity source (EDM churn / SDE-native solver / Restart) before declaring success.
2. **Image-domain NFE ≠ fold quality.** Every quoted NFE (UniPC@10, OSS 10×, SiD 16-step) is images/designability, not our AF2Rank/fold gate. Control: validate every candidate on our 3 gates; expect ~20–50 NFE training-free, not single digits.
3. **Seq-conditioned-checkpoint gap for distillation.** SiD-Protein's public ckpt distills the UNCONDITIONAL model; our seq-cond win needs re-running SiD on our conditioned teacher (unverified engineering). Control: confirm teacher match first; keep Reflow as fallback.
4. **Time-direction convention.** Confirm in `r3n_fm.py` whether t=0 is noise or data before porting any schedule — a flipped direction places steps at the wrong end and falsely fails schedule search.
