# Proteina Inference Speedup — RAW ARCHIVE (dated narrative / blow-by-blow)

Historical, dated logs moved out of the PLAN to keep it concise. Distilled conclusions live in the PLAN (§KEY RESULTS); this file is the audit trail (job IDs, mistakes, node-contention fixes, step-by-step decisions). Not authoritative for current state — the PLAN is.

---

## APPLES-TO-APPLES CAMPAIGN — dated progress (2026-06-25/26)
Setup: 8 scaling8 targets (8XHT,7AD5,6ZUS,8QXI,6ZTG,7KW9,8RJX,6ZYG), all 8192, settings matched to SDE-200 seq_cath baseline (recycles 6, af2rank_top_k 8, cutoffs 256-8192, cif natives). Launchers `scaling_run_sc8.sh`/`scaling_run_seqcond8.sh`/`conv_sc8.sh`. Configs FAST=`..._spdref_vf_powerp3_nfe025_scale`, SDE clone=`..._unified_sderef8k`. Monitor cron 1165ba6b.
- PROGRESS (21:00): 8QXI seq_cath FAST done — ref_pred_tm@8192 = 0.466 ≈ SDE-200 0.467 (easy/saturated; pipeline validated). 7/8 still running.
- ⚠️ PROGRESS (21:37): af2rank HANG recurring on 7AD5 (job 4981037) — 1/8 top-k written (CLEAN, 0 NaN), frozen 25min, CPU pinned ⇒ non-NaN trigger of the USalign-no-timeout hang (geometry-specific).
- ✅ RESOLVED (21:55): hang ALREADY FIXED in codebase (other agent's nanrepro: `USALIGN_TIMEOUT_S=120` on both USalign wrappers + NaN guard in `_save_openfold_prediction_pdb`). 7AD5 hung only because job 4981037 loaded pre-fix code. Scancel'd → cleared partial → re-ran = job 4982377. Done: 8QXI, 6ZUS clean.
- ⭐ PROGRESS (~23:10): 7/8 seq_cath FAST done — @8192 single-cutoff: 8QXI=0.466, 8XHT=0.200, 6ZUS=0.249, 6ZTG=0.231, 7KW9=0.484, 8RJX=0.417, 6ZYG=0.285 (mean7≈0.333). Only 7AD5 left.
- ⚠️ NODE-CONTENTION FIX (~23:10): held pending SDE seq_cond 4981047-50 + scancel'd 4981046 (6ZTG-SDE) to free a node for 7AD5-rerun.
- ⚠️ CORRECTION (~23:15): scancel did NOT unblock 7AD5-rerun — still AssocGrpNodeLimit with a node FREE ⇒ the cou node cap is SHARED with the other agent; freeing one of MY nodes doesn't help. LESSON: don't scancel to "free a node" under a shared-account cap. Reverted to holding pending SDE so 7AD5-rerun is top-eligible.
- ✅ STATE (~02:50): 8/8 seq_cath FAST done; ENERGY conv 4986865 running. SDE seq_cond submit order 43-50 = 8QXI,7AD5,6ZUS,6ZTG,7KW9,8RJX,6ZYG,8XHT; 6ZTG (46) cancelled → re-run 4987470.

## AUTONOMOUS DIRECTIVE (user away ~12h: 2026-06-25 early-AM → ~midday)
Order: KEEP PUSHING, never idle GPUs, record decisions in DETAIL, QC at end. Items: (1) push the ceiling power p3/p4/p5 @ NFE15/10; (2) implement Step C commitment (env-gated r3n_fm.py hook); (3) scaling experiment on the winner (up to 8192, power-of-2 cutoffs, EBM-vs-Oracle ranking) + confirm on seq_cath + SDE.
- AUTONOMOUS-WINDOW STATUS (~10:15, QC): items 1-3 ALL DONE. Step D (training-free solvers) + Step E (distillation) NOT started autonomously — each needs a user design decision (no-random-design rule). Driver cron ec848464.

## Decisions log (autonomous, 2026-06-25) — step-by-step
- Phase 2b configs = power{3,4,5}×NFE{10,15} + log NFE10 (7 jobs spd_swp). dt: NFE15=0.0667, NFE10=0.1.
- STEP C HOOK DONE: env-gated capture in r3n_fm.py (read PROTEINA_TRAJ_DUMP/FORK_FROM/FORK_T_IDX, capture x0_pred+x_t/step, fork loop, save dump). DEFAULT-OFF. Edited locally → scp'd to SC (later committed to git 2026-06-27, commit 70aab4e). Driver commitment_run.sh + commitment_analyze.py; uniform-200 config for even t-coverage.
- ✅ STEP C HOOK VALIDATED (02:22): 8QXI smoke 4974316; traj loads. 8QXI t* (x0_pred TM-to-final): TM≥0.5 @ t=0.275, ≥0.7 @ 0.395, ≥0.9 @ 0.75 → commitment window ≈ t∈[0.27,0.5] (mid noise).
- CEILING IDs→tags: 4974288=p3_nfe015, 89=p3_nfe010, 90=p4_nfe015, 91=p4_nfe010, 92=p5_nfe015, 93=p5_nfe010, 94=nfe010(log).
- ⭐⭐ STEP C C0 RESULT (x0_pred TM-to-final commitment t*, t=0 noise→t=1 data):
  | protein | L | t*(TM0.5) | t*(TM0.7) | t*(TM0.9) |
  |---|---|---|---|---|
  | 8QXI_A | 78 | 0.275 | 0.395 | 0.75 |
  | 6M5Y_A | 270 | 0.175 | 0.290 | 0.54 |
  | 8TVL_A | 338 | 0.105 | 0.195 | 0.52 |
  | 8AUC_B | 493 | 0.075 | 0.150 | 0.45 |
  KEY: commitment is LENGTH-DEPENDENT — longer proteins lock the fold EARLIER (higher noise/lower t). Files commitment/commitment_curve{,_tstar}.csv.
- ⭐⭐ STEP B CEILING CURVE (ref_pred_tm mean, vf-seq rec1):
  | NFE (speedup) | log | power p3 | power p4 | power p5 |
  |---|---|---|---|---|
  | 25 (8×) | 0.277 | **0.338** | — | — |
  | 15 (13×) | 0.278 | 0.298 | **0.325** | 0.302 |
  | 10 (20×) | ~0.27 | 0.297 | 0.288 | **0.303** |
  (SDE-200 baseline 0.348.) OPTIMAL p INCREASES as NFE drops. p3@NFE25=0.338 (97%, 8×) WINNER.
- ✅ WINNER CONFIRMS on seq_cath: p3@NFE25 seq_cath = 0.334 ≈ seq 0.338 (94% of seq_cath-200 0.355).
- ⭐ SDE CONFIRM — winner schedule is ODE-SPECIFIC: p3@NFE25 on SDE = 0.265 ≪ ODE 0.338. High-noise-emphasis does NOT transfer to the stochastic sampler.
- ITEM 3 SCALING (8192 gen): winner p3@NFE25, config `..._spdref_vf_powerp3_nfe025_scale`, 8192 samples, rec1 fp16. Subset 8QXI/7AD5/8TVL/8AUC/6M5Y. conv_scl.sh.
- ⭐ SCALING 8192 PREVIEW (single-cutoff EBM-selected): 8QXI p3=0.471/p4=0.477; 7AD5 p3=0.179 (EBM selected poor sample — selector-limited).
- ⭐⭐⭐ ORACLE CONVERGENCE (4-prot, job 4974655):
  | #samp | Energy(EBM) | Oracle(true-TM) | gap |
  |---|---|---|---|
  | 256 | 0.467 | 0.533 | +0.066 |
  | 1024 | 0.475 | 0.535 | +0.060 |
  | 8192 | 0.488 | 0.537 | +0.049 |
  Both lines SATURATE EARLY (~256-512). EBM-vs-oracle gap ~0.05. Both ≫ baseline 0.348.
- ⭐⭐ p4 ENERGY CONVERGENCE (job 4975143): 256=0.316 → 8192=0.434. p4 (13×) SAMPLE-HUNGRY; p3 stays ahead at every cutoff → p3@NFE25 is the better operating point.
- ⭐ p4 ORACLE (job 4975168): ~flat ~0.48-0.49. p4 ceiling (0.49) < p3 ceiling (0.54).
- ⭐⭐⭐ 5-PROTEIN p3 ENERGY-vs-ORACLE (oracle job 4975504; all 5 lengths 78-492 incl hard 8AUC):
  | #samp | Energy(EBM) | Oracle(true-TM) | gap |
  |---|---|---|---|
  | 256 | 0.410 | 0.506 | +0.096 |
  | 1024 | 0.418 | 0.508 | +0.090 |
  | 8192 | 0.448 | 0.510 | +0.062 |
  Oracle ceiling ~0.51 (flat, saturates ~256); energy ~0.45. CONFIRMS the 4-prot conclusion on the full length-diverse set. ITEM 3 COMPLETE.
- ⚠️ 8AUC-p4 (job 4975144) HUNG in af2rank → scancelled. Likely `.cache/cuequivariance` kernel-cache miss → pathological cuEq-attention path for 492-res (8AUC-p3 did NOT hang). GOTCHA: af2rank freeze + CPU pinned on long proteins → suspect cuEq cache miss; pre-warm `~/.cache/cuequivariance` or drop `--use_cuequivariance_attention` for the af2rank stage.
