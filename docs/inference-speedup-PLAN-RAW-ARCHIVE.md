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

---

## AF2Rank protocol audit + apples-to-apples campaign (2026-06-25/26) — dated narrative

**AF2RANK PROTOCOL CORRECTION (2026-06-26, external agent audit):** independent agent audited + reproduced AF2Rank (Roney & Ovchinnikov 2022, PRL 129 238101; 133 Rosetta-decoy targets), found our protocol wrong on several points. Corrected STOCK AF2 (model_1_ptm) = per-target Spearman 0.84 (133 tgt, 25 decoys) / 0.882 (4 SI tgt, full ~1000 decoys); paper 0.925. Bugs found: (1) composite must be pTM×pLDDT×tm_io (tm_io was often omitted, the DOMINANT signal — output≈native so tm_io≈decoy-TM-to-native); (2) mask_sidechains_add_cb is the root-cause fix (0.69→0.84) — strip decoy to N/CA/C/O/CB + virtual CB on Gly before templating, else AF2 leaks native sequence via sidechain geometry; (3) recycles=1 (not 3/6) is correct protocol; (4) template seq = gap tokens, single-sequence, model_1_ptm; (5) use full decoy set (~800-1000), 25-decoy subsample deflates Spearman; (6) metric = per-target Spearman then MEAN over targets. Mechanism: AF2Rank's AF2 COPIES the decoy (output≈input) and ranks by CALIBRATED CONFIDENCE — does not reject bad decoys (no-template AF2 avg TM~0.41). Sanity targets (stock, masked, rec1, full decoys): 1agy 0.89, 1acf 0.89, 1a32 0.92, 1cc8 0.83. Core fixes landed on origin/main (97ca393 rec1, d6e3ccf mask-sc, 67cc19d/aafa509 composite); implication — every af2rank-gated number scored before this fix is provisional. UNAFFECTED: ProteinEBM tm_ref_template (raw-selection metric, no af2rank involved).

**CRITICAL CORRECTION (2026-06-25 PM, user QC):** an earlier "fast p3@NFE25 = 8x at 97%" conclusion was built on a WRONG comparison — compared fast 8192-sample run to SDE-200 at only 16 samples, wrong subset (5 length-diverse vs the established scaling8 8 hard targets), wrong conditioning (seq_cond vs seq_cath). Real comparison showed the fast config KILLS inference-time scaling on hard targets (7AD5 SDE-200 keeps climbing 0.425→0.684 past 1024 samples; fast config flat at 0.398).

**APPLES-TO-APPLES RESULT (seq_cath, 8 scaling8 targets, 8192, recycles 6, single-cutoff EBM-selected ref_pred_tm):**
| protein | L | FAST p3@NFE25 (8x) | SDE-200 NFE200 | delta |
|---|---|---|---|---|
| 8QXI | 77 | 0.466 | 0.467 | -0.001 |
| 8XHT | 100 | 0.200 | 0.207 | -0.007 |
| 6ZYG | 112 | 0.285 | 0.437 | -0.152 |
| 7AD5 | 124 | 0.588 | 0.684 | -0.096 |
| 6ZUS | 143 | 0.249 | 0.274 | -0.025 |
| 8RJX | 173 | 0.417 | 0.424 | -0.007 |
| 7KW9 | 178 | 0.484 | 0.494 | -0.010 |
| 6ZTG | 192 | 0.231 | 0.246 | -0.015 |
| mean | | 0.365 | 0.404 | -0.039 |
The fast config is worse than SDE-200 at the real 8192 operating point (mean -0.039, ~10% relative), driven by 7AD5/6ZYG.

**FAST-vs-SDE200 seq_cath ENERGY convergence (job 4986865, EBM-selected mean):** 256:0.315/0.331, 512:0.340/0.332, 1024:0.345/0.345, 2048:0.345/0.367, 4096:0.348/0.371, 8192:0.378/0.404 (FAST/SDE-200). Tied at low N, SDE-200 pulls ahead at high N.

**FAST-vs-SDE200 seq_cath ORACLE convergence (job 4987469, true-TM-selected ceiling):** 256:0.379/0.419, 1024:0.404/~0.44, 8192:0.449/0.459 (FAST/SDE-200). ANSWER — the fast config's POOL is nearly as good as SDE-200 (oracle gap only -0.010 @8192); the DEPLOYED gap (-0.026 @8192 energy) is mostly EBM SELECTION, not sampling quality. Per-protein @8192 (energy->oracle, selector gap): 7AD5 0.567->0.724 (0.157), 6ZYG 0.293->0.456 (0.163), 8XHT 0.247->0.420 (0.173) — hard proteins have the biggest selector gaps. This motivated the Q4 relaxation experiment.

**SDE seq_cond ENERGY convergence (job 4989011):** 256:0.329, 1024:0.339, 2048:0.404, 4096:0.404, 8192:0.410; oracle 256:0.426, 8192:0.460. SDE seq_cond @8192 ~= SDE seq_cath @8192 -> conditioning mode barely changes the SDE ceiling.

**FAST-vs-SDE seq_cond ENERGY (FAST conv 4991367; SDE 4989011):** 256: 0.323/0.329, 8192: 0.373/0.410 (FAST/SDE). Same pattern as seq_cath.

## Q4 ProteinEBM relax-then-score experiment (2026-06-26/27) — dated narrative

**Motivation:** low-NFE fast samples have imperfect local geometry (cut low-noise steps) -> ProteinEBM scores AS-IS, penalizes local imperfection instead of judging fold -> weak selection. Test: relax (reverse-diffusion, t_max->t_min) before scoring, does Spearman(EBM,tm_ref_template) improve?

**UN-RELAXED BASELINE (FAST seq_cath p3@NFE25, ~8192 samples/protein):**
| pid | L | rho(E,TM) | rho(pTM,TM) | tm@bestE | tm@bestpTM | tm_oracle | tm_mean |
|---|---|---|---|---|---|---|---|
| 8XHT | 100 | -0.049 | 0.208 | 0.181 | 0.196 | 0.381 | 0.192 |
| 7AD5 | 124 | -0.206 | 0.188 | 0.172 | 0.178 | 0.552 | 0.240 |
| 6ZUS | 143 | -0.163 | 0.178 | 0.200 | 0.192 | 0.331 | 0.179 |
| 8QXI | 77  | -0.720 | 0.728 | 0.468 | 0.457 | 0.535 | 0.381 |
| 6ZTG | 192 | -0.037 | 0.203 | 0.212 | 0.236 | 0.284 | 0.224 |
| 7KW9 | 178 | -0.330 | 0.424 | 0.406 | 0.419 | 0.476 | 0.231 |
| 8RJX | 173 | -0.566 | 0.574 | 0.362 | 0.326 | 0.399 | 0.280 |
| 6ZYG | 112 | +0.032 | 0.041 | 0.215 | 0.242 | 0.374 | 0.227 |
| MEAN |  | -0.255 | 0.318 | 0.277 | 0.281 | 0.416 | 0.244 |
EBM is a weak selector on fast samples, bimodal (strong on 8QXI/8RJX, near-zero/wrong-signed on 6ZYG/6ZTG/8XHT/7AD5). Large gap: tm@bestE 0.277 vs tm_oracle 0.416.

**Infra bugs found+fixed:** loader must use ProteinRegressionTrainer.load_from_checkpoint(...).ebm (bare ProteinEBM keys unprefixed loads nothing); compute_score (relax) builds backward graph (~3x forward) -> OOM at batch 64/N=78, fixed with tiny relax batch (N=78->12, N=192->2); score_decoys default --bsize 256 OOMs for N>=110 (triangle attention ~B*N^3), fixed with --bsize 32; pTM unpack bug (ProteinRegressionTrainer.forward returns 5 values not 2).

**Q4 RESULT (compare_relax.py, 8 protein mean, raw->t0025->t005):** energy rho 0.280->0.284->0.284 (neutral); pTM rho 0.331->0.317->0.307 (slightly worse); energy tm@best 0.386->0.386->0.381 (neutral); pTM tm@best 0.300->0.341->0.339 (+0.04). Per-protein: relaxation helps 7AD5 most (pTM tm@best 0.172->0.492) but is mostly neutral/mixed elsewhere. CONCLUSION: relaxation does NOT broadly fix weak selection — a weak, mixed, protein-dependent lever, not the decisive fix. tm_ref_template is af2rank-independent so this conclusion survives the af2rank re-score.

**Side-finding:** the ProteinEBM energy signal is noise-unstable — two independent scoring passes over IDENTICAL structures gave meaningfully different per-sample energy (nfe30 TM@bestE 0.203 vs 0.241 originally) — when Spearman rho is this close to zero, argmin selection is dominated by scoring noise not genuine discrimination.

## Hybrid SDE->ODE schedule discovery (2026-06-28 to 07-01) — dated narrative, full progression

**Step-count-in-reasoning-region analysis (2026-06-28):** our r3n_fm convention is t=0=noise, t=1=data (opposite standard diffusion). User's "high noise, fold not yet committed" region = our LOW-t region t in [0, ~0.3]. SDE-200 (log schedule) spends only 16 of its 200 steps there (184 on low-noise refinement); fast p3@NFE25 already puts 17 steps there with 8x fewer total — the fast schedule was NOT under-resourced in the reasoning region, yet still lost to SDE-200 on 7AD5 energy (0.567 vs 0.684), though fast ORACLE (0.724) > SDE oracle (0.708) — the fast pool already beats SDE, the gap is EBM-selection.

**Overshoot-reasoning-steps experiment (7AD5 only, 2026-06-28/29):** tested power p3@NFE50/100, p5@NFE100 (34/67/79 reasoning steps) as pure-ODE. RESULT: oracle (RAW tm_ref_template) @2048: p3@NFE25=0.492, p3@NFE50=0.494, p3@NFE100=0.507, p5@NFE100=0.498 — modest +3% improvement, NFE100 sweet spot. vs SDE-200 RAW oracle 0.599 @2048 — STEP-COUNT HYPOTHESIS REFUTED: SDE-200 has the FEWEST reasoning steps (16) yet samples 7AD5 best. Its edge is STOCHASTICITY (exploration), not step allocation.

**Hybrid sampler test (2026-06-29):** put SDE (stochastic exploration) ONLY in high-noise/reasoning region via env-gated PROTEINA_SDE_UNTIL_T=<t>, then few-step ODE refine. HYBRID @2048 RESULT (verified activated): SDE->0.4 = 0.604, SDE->0.6 = 0.610, vs pure-ODE p3@NFE100 (100 steps) 0.507 and SDE-200 (200 steps) 0.599. THE HYBRID WINS: 100-step hybrid matches/beats full 200-step SDE-200 on raw sampling. HYBRID @8192 RESULT: SDE->0.4 = 0.634, SDE->0.6 = 0.631, vs SDE-200 @8192 = 0.650 — recovers ~97% of full SDE-200 raw sampling at HALF the NFE. EBM-selection bonus: SDE->0.6 @8192 TM@minE=0.539 (near-oracle) vs SDE->0.4 0.191 (selection still broken) — SDE->0.6 is the better hybrid.

**Commitment-experiment methodology gap (2026-06-30, user audit):** the original 11-protein t* fit used K=8 INDEPENDENT deterministic-ODE trajectories per protein (not true forking into stochastic completions from a shared partial state) — it measures ODE-trajectory self-consistency, not true inter-replica fold commitment. Reframed (user, valid for a different question): x0_pred(t)-vs-own-final under deterministic ODE tells you when it's SAFE TO SWITCH from SDE exploration to ODE refinement — exactly the hybrid sampler's SDE_UNTIL_T threshold. SDE-repeat of the same experiment COLLAPSED (R^2 0.678->0.128, sign flip) confirming the metric doesn't transfer to SDE (noise re-injected every step never lets x0_pred stabilize).

**d0(L) length-normalization artifact (2026-06-30, user audit):** the original strong length-trend in the ODE commitment fit was MOSTLY a TM-score scoring artifact — d0(L) leniency radius grows 2.5x across the length range (3.13A at L=78 -> 7.90A at L=493), so bigger proteins cross any fixed TM threshold earlier with zero real commitment difference. Tested directly: r(L,t*) collapses -0.808 (length-dependent d0) -> -0.527 (fixed d0) -> -0.246 (raw RMSD, not significant at n=11). Most of the "backward" trend was the artifact.

**USER DECISION (2026-07-01): commitment time is highly stochastic — record and move on.** Two concerns: (1) 4 of 11 commitment proteins exceed the fine-tune checkpoint's training length range L in [50,256] (confounds both fits ~36%); (2) training time-distribution is Beta(1.3,2.0), mode t~0.231, CDF(0.6)=0.78 -> model trained overwhelmingly on t in [0,0.6], supporting (post-hoc) the empirically-best SDE_UNTIL=0.6 threshold over 0.4.

**Targeted NFE30-50 sweep (2026-07-01, on A6000, FIXED SDE_UNTIL_T=0.6, power-p3, 7AD5 only, RAW oracle tm_ref_template @2048):**
| config | total steps | SDE+ODE split | oracle TM | EBM-selected (TM@minE) |
|---|---|---|---|---|
| pure-ODE p3@NFE100 | 100 | 100+0 | 0.507 | -- |
| hybrid NFE30, SDE->0.6 | 30 | 26+4 | 0.552 | 0.241 |
| hybrid NFE40, SDE->0.6 | 40 | 34+6 | 0.622 | 0.179 |
| hybrid NFE50, SDE->0.6 | 50 | 43+7 | 0.624 | 0.162 |
| hybrid NFE100, SDE->0.6 | 100 | 85+15 | 0.610 | -- |
| SDE-200 (full SDE) | 200 | 0+200(sc) | 0.599 | -- |
NFE40 AND NFE50 BOTH BEAT SDE-200 at 40-50 total steps (4-5x speedup with HIGHER raw quality). NFE30 (0.552) already beats pure-ODE NFE100 (0.507) at 3.3x fewer steps. Visualization (PyMOL, exact USalign matrix): NFE40 (best oracle) cleanest match on both beta-sheet and alpha-helix.

**Relaxation-rescues-scoring test on NFE30/40/50 pools (2026-07-01):** does relax-then-score fix EBM selection on these already-winning-on-oracle pools? RESULT: no meaningful, consistent rho improvement in any config (raw vs t0025): nfe30 energy -0.107->-0.133, nfe40 -0.024->-0.037, nfe50 -0.043->-0.077; pTM similarly flat/mixed. CONCLUSION unchanged from the original Q4 finding.

**Why ProteinEBM fails to rank 7AD5 — corrected round-2 analysis (2026-07-01, same-protein-across-configs, not cross-protein):**
| config | N | rho(E,TM) | rho(pTM,TM) | TM@argmin(E) | TM@argmax(pTM) |
|---|---|---|---|---|---|
| SDE-200 | 8192 | -0.137 | 0.201 | 0.635 | 0.221 |
| FAST-NFE25 | 8087 | -0.206 | 0.188 | 0.172 | 0.178 |
| NFE30 | 2048 | -0.104 | 0.110 | 0.241 | 0.241 |
| NFE40 | 2048 | -0.021 | 0.043 | 0.179 | 0.535 |
| NFE50 | 2048 | -0.057 | 0.109 | 0.162 | 0.216 |
7AD5 does NOT have zero energy signal — every config shows a real, consistent (funnel-shaped, noisy) negative energy-vs-TM trend. Likely explanation is SAMPLE COUNT not a qualitative EBM blind spot: SDE-200 has 524 samples at TM>=0.4 (8192 pool) vs NFE30/40/50's 61/110/140 (2048 pool, 1/4 the size) -- far fewer chances for the same noisy-but-real correlation to land the argmin on a good structure.

**Top-k dry analysis (2026-07-01) — k needed to reach oracle-0.05, pure CPU analysis on existing CSVs:**
Section 1 (all 8 scaling8 targets, FAST-NFE25 8192 samples): 8QXI k=5, 8RJX k=1, 7AD5 k=5, 6ZTG k=5, 7KW9 k=5, 6ZUS k=50, 6ZYG (wrong-signed rho) k=200, 8XHT k=400. Most "bad-rho" targets need only k=5 (0.06% of pool) despite weak overall Spearman -- only 6ZYG/8XHT genuinely need large k.
Section 2 (7AD5 across configs): SDE-200 k=1 (0.01%), FAST-NFE25 k=5 (0.06%), NFE30 k=50 (2.4%), NFE40 k=32 (1.6%), NFE50 k=16 (0.8%). All 5 configs reach near-oracle at a SMALL absolute k (<=50). Practical implication: deploying NFE40/50 with af2rank_top_k~32-50 should already recover ORACLE-level quality.

**Length/dataset gotchas found along the way:** true commitment lengths from dumps (not csv): 8QXI 78, 7AD5 124 (measured t*=0.280), 6M5Y 270 (cif chain A=277 CA, model used 270), 8TVL 338, 8AUC 493. Dataset length bug: csv `length` col = AFDB/UniProt monomer length, NOT the experimental cif chain actually sampled — 3 proteins diverge badly: 8XHT csv100->sampled 287, 6M5Y csv132->277, 7E7T csv90->260 (+7F7N 117->126 marginal); other 22 of 26 match within ~3. Length<->commitment bounded fit: t*(TM0.5) = sigmoid(-0.559 - 0.00409*L), R^2=0.955 (superseded by the d0-artifact finding above — take with a grain of salt).

## Step A/B/C schedule-sweep raw results (2026-06-24/25) — dated narrative

**Step A baseline (200-step, fp16, ProteinEBM->af2rank top-5, all 4 jobs RC=0):**
| config | n | ref_pred_tm mean/med | af2rank pTM mean | proteinEBM pTM mean |
|---|---|---|---|---|
| SDE seq      | 26 | 0.348 / 0.297 | 0.492 | 0.536 |
| SDE seq_cath | 24 | 0.355 / 0.289 | 0.497 | 0.550 |
| ODE seq      | 26 | 0.348 / 0.296 | 0.442 | 0.350 |
| ODE seq_cath | 24 | 0.367 / 0.301 | 0.453 | 0.359 |
KEY FINDING: ODE (vf) ~= SDE (sc) in FINAL refined quality at 200 steps -> can accelerate the ODE without losing final quality (de-risks ODE-vs-SDE). ODE confidence signals lower (proteinEBM pTM 0.35 vs SDE 0.54) but selection still lands the same final quality.

**Step B phase 1 — ODE NFE-degradation @ default log schedule, recycles=1:**
| NFE | speedup | ref_pred_tm mean/med | af2rank pTM | proteinEBM pTM |
|---|---|---|---|---|
| 200 (baseline, rec6) | 1x | 0.348 / 0.296 | 0.442 | 0.350 |
| 100 | 2x | 0.326 / 0.317 | 0.379 | 0.344 |
| 50  | 4x | 0.298 / 0.263 | 0.346 | 0.304 |
| 25  | 8x | 0.277 / 0.257 | 0.299 | 0.274 |
| 15  | 13x | 0.278 / 0.268 | 0.311 | 0.277 |
Smooth monotonic degradation with the DEFAULT log schedule (dense at low noise) -- this is the curve subsequently BEATEN by high-noise-emphasis schedules below.

**Step B phase 2 — schedule hypothesis test (power p>1 = high-noise emphasis) CONFIRMED:**
| NFE | log (default) | power p2 | power p3 | edm rho7 |
|---|---|---|---|---|
| 25 (8x) | 0.277 | 0.325 | 0.338 | 0.320 |
| 50 (4x) | 0.298 | 0.312 | 0.330 | 0.323 |
power p3 is the consistent winner: NFE25 = 0.338 ~= SDE-200 baseline 0.348 (8x speedup at ~97% quality), beats default-log @NFE100 (0.326) with 4x fewer steps. SDE confirm: same schedule on SDE = 0.265 -- the winning schedule is ODE-SPECIFIC, does NOT transfer to the stochastic sampler.

**Step C commitment (first pass, ODE, 4 proteins, before the training-length-mismatch + d0-artifact corrections above):**
| protein | L | t*(TM0.5) | t*(TM0.7) | t*(TM0.9) |
|---|---|---|---|---|
| 8QXI_A | 78 | 0.275 | 0.395 | 0.75 |
| 6M5Y_A | 270 | 0.175 | 0.290 | 0.54 |
| 8TVL_A | 338 | 0.105 | 0.195 | 0.52 |
| 8AUC_B | 493 | 0.075 | 0.150 | 0.45 |
Commitment appeared length-dependent (longer proteins lock the fold earlier) -- see the d0-artifact correction above, most of this trend was a scoring artifact.

**Item 3 — inference-time scaling + EBM-vs-oracle gap (4-5 protein subset, ~2026-06-24):** both energy and oracle lines SATURATE EARLY (~256-512 samples); EBM-vs-oracle gap ~0.05-0.06 across the range; fast config + a few hundred samples already approaches the achievable ceiling, far above the 200-step baseline 0.348.

## A6000 host onboarding (2026-07-01) — dated gotchas, now resolved

Multi-GPU dispatch clobbered CUDA_VISIBLE_DEVICES: run_prediction_pipeline.py/parallel_proteina_inference.py/parallel_proteinebm_scoring.py unconditionally set env["CUDA_VISIBLE_DEVICES"]=str(gpu_id) assuming the job owns physical GPUs 0..N-1 (true on SLURM, false on this shared host) — clobbered any pre-existing restriction, all 3 workers silently landed on the same physical GPU. Fixed at all 5 call sites (commit c17fc6a): index into an inherited CUDA_VISIBLE_DEVICES list when present. First env attempt wrongly cloned a bare `proteina` conda env into `proteina_ebm` and papered over torch-version gaps with workarounds instead of using the already-correct `cue-openfold-env` — reverted, `proteina_ebm` removed.

## AF2Rank compile-vs-eager breakeven — full benchmark tables (2026-07-01/02)

**Harness bug (first attempt, discarded, not real findings):** `bench_af2rank_3way.py` called score_structure() directly on raw CA-only PDBs in a naive loop, hitting the per-call cg2all reconstruction fallback (7-36s per call) and unneeded per-call USalign-vs-GT calls -- both constant overhead unrelated to compile status but large enough to produce nonsensical "compiled slower than eager" numbers.

**Corrected real-workload result (bench_af2rank_3way_v2.py):**
| pid | L | eager (s) | compile warmup (s) | compiled steady (s) | speedup | breakeven N |
|---|---|---|---|---|---|---|
| 8QXI_A | 78 | 3.19 | 49.77 | 2.84 | 11.0% | 134 |
| 7AD5_A | 124 | 5.50 | 43.64 | 4.40 | 20.0% | 36 |
| 6ZTG_A | 193 | 9.89 | 36.80 | 7.45 | 24.7% | 12 |
| 8XHT_A | 287 | 17.48 | 48.98 | 12.90 | 26.2% | 8 |
Featurization ruled out as a cause of the smaller real-vs-synthetic speedup (only 2.5-3.1% of total time, already hidden by the existing prefetch thread). Real-workload warmup (37-50s) is 8-10x smaller than synthetic cold-start warmup (298-501s) because of a persistent on-disk TorchInductor cache (/tmp/torchinductor_jupyter-chenxi) -- compile warmup is a one-time-ever cost per (length,config) on a persistent host, not paid again per job.

**Valid synthetic result (bench_compile_breakeven_4len.py, pure model-forward, no featurization):**
| L | eager (s) | compile warmup (s) | compiled steady (s) | speedup | breakeven N |
|---|---|---|---|---|---|
| 78 | 3.077 | 402.054 | 1.984 | 35.5% | 366 |
| 124 | 5.343 | 382.517 | 3.280 | 38.6% | 184 |
| 193 | 10.034 | 298.203 | 6.032 | 39.9% | 73 |
| 493 | 62.192 | 501.482 | 31.188 | 49.9% | 15 |
Compiled steady-state is always faster than eager -- confirms there's no reason compile should ever be slower (any "compiled slower" result is a harness bug). Breakeven N shrinks sharply with L.

**L=338 (8TVL) 5th point, filling the 193-493 gap:** synthetic (unpadded) eager=26.910s, warmup=489.276s, steady=14.884s, speedup=44.7%, breakeven N=39.4 -- fits the trend cleanly. Real-workload point showed an anomaly (breakeven N=236) traced to the bucketing feature padding the compiled path to 352 (bucket ceiling) while eager stayed at 338 -- see the bucketing section for the mechanism; this made the L=338 real-workload point NOT comparable to the pre-bucketing 4-point table.

**ColabDesign/JAX benchmark (bench_colabdesign_only.py, fixed for the same per-call-cg2all bug, 8QXI/7AD5/6ZTG/8XHT):**
| pid | L | load (s) | 1st/warmup (s) | steady (s) | ptm |
|---|---|---|---|---|---|
| 8QXI_A | 78 | 14.46 | 40.37 (incl. 15.12s cg2all) | 0.56 | 0.651 |
| 7AD5_A | 124 | 1.56 | 28.07 (incl. 7.85s cg2all) | 0.67 | 0.459 |
| 6ZTG_A | 193 | 10.12 | 32.18 (incl. 8.54s cg2all) | 1.22 | 0.346 |
| 8XHT_A | 287 | 4.35 | 31.78 (incl. 7.78s cg2all) | 2.27 | 0.275 |

## Length-bucketing verification + bf16/fp16 benchmark tables (2026-07-01/02)

**Bucketing correctness verification (verify_bucketing_correctness.py) — exact match:**
| pid | L | bucket (pad) | eager (ptm/plddt/pae) | compiled steady (ptm/plddt/pae) | delta |
|---|---|---|---|---|---|
| 8QXI_A | 78 | 96 (+18) | 0.6619 / 0.7824 / 7.997 | 0.6619 / 0.7824 / 7.997 | 0.0000 / 0.0000 |
| 7AD5_A | 124 | 128 (+4) | 0.4625 / 0.5388 / 14.018 | 0.4625 / 0.5388 / 14.018 | 0.0000 / 0.0000 |
Bucket-math sanity confirmed correct at every 32-boundary (L=32->32, 33->64, ..., 512->512, no off-by-one). Two real bugs found+fixed en route: (1) unmasked pTM/pLDDT/PAE corruption when padding was applied without a true-length mask (fixed in _extract_scores, af2rank_openfold_scorer.py); (2) skip_alignment identity-mapping index-out-of-range on a padded query in openfold's templates.py (separate repo, commit a71ae27).

**bf16 vs tf32 benchmark (bench_af2rank_precision.py):**
| pid | L | tf32 eager | bf16 eager | bf16 speedup | bf16+compiled warmup | bf16+compiled steady | ptm (tf32/bf16/bf16+compiled) |
|---|---|---|---|---|---|---|---|
| 8QXI_A | 78 | 3.66 | 2.02 | 44.9% | 86.7 | 3.08 | 0.662/0.659/0.661 |
| 7AD5_A | 124 | 6.29 | 3.66 | 41.7% | 84.3 | 4.36 | 0.462/0.463/0.464 |
| 6ZTG_A | 193 | 10.84 | 6.14 | 43.4% | 169.8 | 7.69 | 0.351/0.350/0.351 |
| 8XHT_A | 287 | 19.07 | 10.31 | 46.0% | 250.7 | 12.33 | 0.335/0.340/0.336 |
bf16+compiled is SLOWER than bf16 eager at every length -- torch.compile's per-block-graph Python/dynamo-guard overhead becomes relatively more significant once bf16 tensor cores already make each op fast, exceeding the fusion benefit. bf16 eager beats even the previously-best tf32-compiled steady-state (2.84/4.40/7.45/12.90s) by 17-29%. JAX-vs-OpenFold precision-matched (both bf16): JAX steady 0.56/0.67/1.22/2.27s vs OF-bf16-eager 2.02/3.66/6.14/10.31s -- JAX remains 3.6-5.5x faster even at matched precision, a genuine framework/kernel gap.

## Duplicate-driver incident, full narrative (2026-07-02, ~01:07 AM -> ~12:00 PM EDT, ~11h of GPU wasted)

A launch attempt of the 12h autonomous campaign driver appeared to fail (a mkdir/pid-file race printed an error), so the agent retried -- but the `nohup ... &` in the first attempt had already backgrounded successfully before the error, so the retry launched a second full copy. Both ran concurrently on GPUs 0/1/2 for ~11h, racing each other on the same protein tars; run_prediction_pipeline.py's own retry-on-transient-failure logic multiplied the duplication further (up to 2 parallel_proteina_inference.py children per shard, some with their own retried inference.py leaves) -- 30 stray PIDs total by the time it was caught via nvidia-smi + ps command-line inspection.

Damage (verified via tar member counts + proteinebm_scores CSV row counts, not assumed): zero raw sample data lost -- the existing tar-corruption guard (protein_tar_utils.py's "refusing to re-tar, would drop N members" check) protected every protein; 7/8 scaling8 proteins ended up with full correct 8192-sample tars. Collateral: a directory race crashed the ProteinEBM central-analysis step for some proteins mid-write (OSError: cannot save into a non-existent directory -- the loose dir got packed into the tar by the OTHER duplicate instance mid-write); 3 proteins' (8QXI/6ZYG/6ZTG) oracle scores were stuck reflecting only an earlier 128-sample smoke test, since the resume-check hadn't yet re-run after the real 8192 samples landed; 7AD5 stuck at 2048/8192 in both NFE40 and NFE50 dirs; 1 empty/corrupted PDB file for 8QXI (partial write, killed mid-flush, negligible 1/8192).

Claude Code's auto-mode safety classifier refused kill -9 on the duplicate PIDs even with verified command-line evidence (exact --pt <id> --config_name <cfg> matches) -- process termination on this shared host requires explicit user authorization. The agent stopped and reported to the user rather than working around the block.

Recovery (user authorized "Go"): killed the 6 duplicate GPU-consuming inference.py leaves (verified PIDs), then discovered + killed the FULL orphaned process tree (24 more PIDs -- run_prediction_pipeline.py/parallel_proteina_inference.py layers reparented to PID 1 after their bash-wrapper parents died; killing only the leaves is insufficient once a driver has been duplicated, the whole tree must be enumerated via `ps -eo pid,ppid,cmd`). Added a PID lock file (kill -0 liveness check + trap EXIT cleanup) directly into the driver script so a second launch attempt refuses to start. Verified (before trusting it) that the pipeline's own resume logic compares analysis-CSV row count against the CURRENT tar member count (find_proteins_needing_proteinebm / _proteinebm_csv_text_complete), so a clean relaunch self-heals the 3 stale-analysis proteins and the 7AD5 top-up without any manual deletion -- confirmed live: "[8XHT_A] Sample count changed (128 -> 8192), re-running analysis". Relaunched one clean instance ~12:07 PM EDT, verified exactly one live PID via the lock file. create_trigger/send_later were both 404ing at relaunch time -- fell back to ScheduleWakeup for monitoring (the on-box driver is fully self-advancing regardless of whether the agent's monitor fires).

## Stage 3/4 dataset-passthrough bug, discovery + additive fix (2026-07-02, ~15:55-16:08 PM EDT)

User instruction ("don't let the GPUs sit idle, start it on the next job when applicable") triggered a check on whether Stage 3 (bad_afdb26_nfe40) could be pre-started on GPU0/GPU2 while GPU1 was still finishing Stage 2's 8XHT_A. Re-reading `a6000_hybrid_campaign_driver.sh` end-to-end (not just the log tail) surfaced a real bug: `run_3shard_stage()` accepts a `csv_file` parameter and even has a comment claiming it "points the launcher at the right dataset," but the function body never actually passes `csv_file` to `a6000_scaling8_hybrid_run.sh` -- that launcher unconditionally hardcodes `--dataset_file .../scaling8.csv` and `--output_dir .../scaling8_...`. So Stage 3/4 as coded would have silently re-run the already-8-protein-complete scaling8 set at nfe40/nfe50 respectively instead of expanding to the 26-protein bad_afdb26 set -- harmless in terms of wasted GPU time (skip_existing short-circuits every protein almost instantly, matching Stage 1's ~10min completion), but it would have silently defeated the entire point of the campaign (the 18 bad_afdb26-only proteins would never be sampled, and the final oracle_summary.txt would show None for all of them).

Considered and rejected: live-editing the running driver script. Confirmed via first-principles reasoning about bash's execution model that this is unsafe/ineffective -- bash parses a function's body into memory when it reaches the `function_name() { ... }` definition (near the top of the script, already executed at process start), so editing that function body on disk afterward has NO effect on the already-running process for any FUTURE calls to that function within the same execution. Top-level sequential statements (the STAGE 1/2/3/4 blocks) are similarly unsafe to edit while a process may be mid-buffered-read past that point -- well-documented as undefined/version-dependent behavior for live script editing. Also rejected: killing the live driver to restart clean -- would lose ~2h+ of GPU1's in-flight 8XHT_A sampling (not checkpointed mid-protein), and the classifier/user-authorization bar for killing anything on this shared host is intentionally high after the previous incident.

Fix: a separate, additive script (`a6000_bad_afdb26_manual_driver.sh`), own PID lock file, immediately launches bad_afdb26_nfe40 shard 0 (GPU0) and shard 2 (GPU2) -- both idle at the time, waiting on the live driver's internal cross-shard barrier for GPU1. Shard 1 (GPU1) is deferred: the script polls the LIVE driver's lock file every 60s and only launches once that PID is confirmed dead, guaranteeing it never competes with GPU1's in-flight sampling. After all 3 nfe40 shards finish, it launches all 3 nfe50 shards, then re-runs `campaign_oracle_summary.py` (already correct -- keyed by `(inference_config, protein_id)` under `~/proteina/inference/{cfg}/seq_cath_cond/{pid}/`, not by `--output_dir` or which script launched the run) into `oracle_summary_full.txt`. This design guarantees no collision with the live driver's own (buggy-but-harmless) eventual stage 3/4 attempt on the same GPUs, since raw sampling output is keyed purely by (cfg, pid).

First launch attempt used `ssh -n ... < localfile` to transfer the script (avoiding fragile inline heredoc quoting) -- `-n` redirects the ssh session's stdin to /dev/null, which silently defeated the stdin-pipe transfer (0-byte file written). Fixed by dropping `-n` for that one transfer command (still fine since it's a one-shot non-interactive `cat >`, no hang risk). Verified via `md5sum` on both ends before executing. Launch itself printed `disown: not found` (remote non-interactive shell doesn't have the bash builtin available in that context) but the backgrounded process still detached correctly (`nohup` alone was sufficient) -- verified via `ps -p <pid> -o ppid` showing PPID=1. Multiple `parallel_proteina_inference.py` processes appeared per shard (3 each) which initially looked like a possible duplicate-launch repeat; cross-checked against the live driver's own long-running, healthy shard-1 process tree and found the identical 3-process-per-shard pattern there too (stable for 3.5h+) -- confirmed this is the pipeline's normal manager/worker/watchdog process shape, not a new incident.

The first attempt to write this same script via a single `ssh 'cat > file << HEREDOC ... HEREDOC'` one-liner with nested quoting was blocked by Claude Code's auto-mode safety classifier ("preparing to launch a second, concurrent GPU driver... exact kind of unilateral, self-justified concurrent-launch decision that caused the earlier incident"). Correctly deferred to the user rather than working around it; user chose "fix the bug first [verify correctness], then authorize the second driver" -- script was rewritten locally, syntax-checked (`bash -n`) both locally and after transfer, checksummed for exact match, then launched only after that verification passed.

## Campaign completion: full narrative + final results (2026-07-02 4:08 PM -> 2026-07-03 7:45 PM EDT, ~27.6h wall-clock)

**Timeline.** Live driver `a6000_hybrid_campaign_driver.sh` (buggy stage 3/4, see above) finished cleanly at 5:50 PM 2026-07-02 -- its own stage 3 attempt hit a real but inconsequential GPU0 OOM (tried loading ProteinEBM while the manual driver's 8AUC_B sampling held 47.13/47.43 GiB of GPU0 -- the underlying data was already fully scored, so nothing was lost) and stage 4 succeeded; its own `oracle_summary.txt` failed with `python: command not found` (no venv activation in the top-level script, a pre-existing bug, same class as below). Manual driver `a6000_bad_afdb26_manual_driver.sh` detected the live driver's lock-PID exit and launched GPU1's shard at 5:51 PM as designed.

nfe40 stage: GPU0/GPU2 shards (started 4:08 PM) ran the full 6-protein/7-protein lists; GPU1's shard (started 5:51 PM) ran 3 proteins. 8AUC_B (L=492, the campaign's known long-pole outlier) finished sampling at 12:30 AM 2026-07-03 in 8.36h -- faster than the 11-14h worst-case projection from the L²·log(L) cost-scaling extrapolation. `NFE40_ALL_SHARDS_DONE` printed 4:33 AM (12h25min total). One graceful mid-stage recovery: GPU0's ProteinEBM scoring of 8AUC_B hit a CUDA OOM at batch_size=16, auto-retried at batch_size=8, succeeded -- not a failure.

User authorized a ~12h autonomous push (1:57 AM, deadline ~1:57 PM) to keep monitoring/pushing regardless of whether it finished in time. A sanity check at ~2:15 AM (see PLAN finding #12) verified pipeline/data integrity (correct 8192-sample counts, no NaN, no missing files for completed proteins) and separately found real chain-break degradation in raw structures for large/OOD proteins -- non-monotonic with length (8TVL_A at L=338 worse than 8AUC_B at L=492).

nfe50 stage auto-started 4:33 AM on all 3 GPUs with the SAME shard assignment (LPT bin-packing is deterministic given identical cost inputs): GPU0=8AUC_B again, GPU1=8TVL_A again, GPU2=7DME_A first. GPU2's and GPU1's shards fully finished (TGT_EXIT=0) by 5:41 PM and 6:33 PM respectively. GPU0's shard finished sampling all 6 proteins by 6:50 PM (8AUC_B took 10.43h this time, matching the 50/40-scaled projection of ~10.5h almost exactly) then hit the SAME OOM-at-batch_size=16-retry-at-8 pattern during 8AUC_B's scoring, which again succeeded. The final step -- a USalign batch TM-score computation for all 8192 samples of 8AUC_B against the native structure -- ran at 99.9% CPU for ~19 minutes (the single most expensive step in the whole campaign, confirmed actively working via `ps`, not hung, before being reported as such). `NFE50_ALL_SHARDS_DONE` and `MANUAL_DRIVER_DONE` both printed 7:45:35 PM 2026-07-03.

**The venv-activation bug (predicted, confirmed) recurred exactly as documented:** the manual driver's own final `python3 scripts/campaign_oracle_summary.py` call produced `oracle_summary_full.txt` containing only the one-line error `python: command not found` (79 bytes). Regenerated manually with the venv properly activated (`bash -c "cd $HOME/cue-openfold-env && source .venv/bin/activate && source ./activate.sh && cd $HOME/proteina_speedup && python3 scripts/campaign_oracle_summary.py"`, passed as ONE single-quoted ssh argument) to get the real table below.

**Full final oracle table (`oracle_tm_ref_template`, N=8192 except 7AD5_A which is N=2048 from an earlier smoke test; 8QXI_A shows NaN at nfe40 in both scaling8 and bad_afdb26 stages -- a pre-existing 1-corrupted-sample artifact from the 2026-07-02 duplicate-driver incident, n_samples=8191 not 8192, unrelated to this campaign's own work):**

```
           stage    pdb  n_samples  oracle_tm_ref_template
  scaling8_nfe40 8QXI_A       8191                     NaN
  scaling8_nfe40 6ZYG_A       8192                  0.3744
  scaling8_nfe40 7AD5_A       2048                  0.6223
  scaling8_nfe40 6ZUS_A       8192                  0.3300
  scaling8_nfe40 8RJX_A       8192                  0.3925
  scaling8_nfe40 7KW9_A       8192                  0.4668
  scaling8_nfe40 6ZTG_A       8192                  0.2743
  scaling8_nfe40 8XHT_A       8192                  0.4115
  scaling8_nfe50 8QXI_A       8192                  0.5252
  scaling8_nfe50 6ZYG_A       8192                  0.4169
  scaling8_nfe50 7AD5_A       2048                  0.6235
  scaling8_nfe50 6ZUS_A       8192                  0.3232
  scaling8_nfe50 8RJX_A       8192                  0.3766
  scaling8_nfe50 7KW9_A       8192                  0.4776
  scaling8_nfe50 6ZTG_A       8192                  0.2884
  scaling8_nfe50 8XHT_A       8192                  0.4134
bad_afdb26_nfe40 8AP5_A       8192                  0.3021
bad_afdb26_nfe40 8QXI_A       8191                     NaN
bad_afdb26_nfe40 7ZK0_A       8192                  0.2246
bad_afdb26_nfe40 6ZYG_A       8192                  0.3744
bad_afdb26_nfe40 7MQQ_A       8192                  0.3038
bad_afdb26_nfe40 8F3K_A       8192                  0.3853
bad_afdb26_nfe40 7VZM_A       8192                  0.3603
bad_afdb26_nfe40 7AD5_A       2048                  0.6223
bad_afdb26_nfe40 6UF2_A       8192                  0.3302
bad_afdb26_nfe40 7F7N_A       8192                  0.2680
bad_afdb26_nfe40 8JB9_A       8192                  0.2642
bad_afdb26_nfe40 6ZUS_A       8192                  0.3300
bad_afdb26_nfe40 7OIO_A       8192                  0.2839
bad_afdb26_nfe40 7TXX_A       8192                  0.2649
bad_afdb26_nfe40 7A2D_A       8192                  0.3825
bad_afdb26_nfe40 8RJX_A       8192                  0.3925
bad_afdb26_nfe40 7KW9_A       8192                  0.4668
bad_afdb26_nfe40 6ZTG_A       8192                  0.2743
bad_afdb26_nfe40 8IN4_A       8192                  0.2703
bad_afdb26_nfe40 6U1O_A       8192                  0.3405
bad_afdb26_nfe40 7E7T_B       8192                  0.7308
bad_afdb26_nfe40 6M5Y_A       8192                  0.3483
bad_afdb26_nfe40 8XHT_A       8192                  0.4115
bad_afdb26_nfe40 7DME_A       8192                  0.4202
bad_afdb26_nfe40 8TVL_A       8192                  0.1901
bad_afdb26_nfe40 8AUC_B       8192                  0.4021
bad_afdb26_nfe50 8AP5_A       8192                  0.3055
bad_afdb26_nfe50 8QXI_A       8192                  0.5252
bad_afdb26_nfe50 7ZK0_A       8192                  0.2522
bad_afdb26_nfe50 6ZYG_A       8192                  0.4169
bad_afdb26_nfe50 7MQQ_A       8192                  0.3092
bad_afdb26_nfe50 8F3K_A       8192                  0.3607
bad_afdb26_nfe50 7VZM_A       8192                  0.3165
bad_afdb26_nfe50 7AD5_A       2048                  0.6235
bad_afdb26_nfe50 6UF2_A       8192                  0.3259
bad_afdb26_nfe50 7F7N_A       8192                  0.2672
bad_afdb26_nfe50 8JB9_A       8192                  0.3979
bad_afdb26_nfe50 6ZUS_A       8192                  0.3232
bad_afdb26_nfe50 7OIO_A       8192                  0.3406
bad_afdb26_nfe50 7TXX_A       8192                  0.3242
bad_afdb26_nfe50 7A2D_A       8192                  0.2719
bad_afdb26_nfe50 8RJX_A       8192                  0.3766
bad_afdb26_nfe50 7KW9_A       8192                  0.4776
bad_afdb26_nfe50 6ZTG_A       8192                  0.2884
bad_afdb26_nfe50 8IN4_A       8192                  0.2626
bad_afdb26_nfe50 6U1O_A       8192                  0.2858
bad_afdb26_nfe50 7E7T_B       8192                  0.8888
bad_afdb26_nfe50 6M5Y_A       8192                  0.2661
bad_afdb26_nfe50 8XHT_A       8192                  0.4134
bad_afdb26_nfe50 7DME_A       8192                  0.2975
bad_afdb26_nfe50 8TVL_A       8192                  0.1835
bad_afdb26_nfe50 8AUC_B       8192                  0.3588

=== per-stage mean oracle (proteins with data only) ===
                      mean  count
stage
bad_afdb26_nfe40  0.357756     25
bad_afdb26_nfe50  0.363835     26
scaling8_nfe40    0.410257      7
scaling8_nfe50    0.430600      8
```

See PLAN finding #13 for the interpretation (NFE40≈NFE50 tie generalizes; 8TVL_A/8AUC_B are the weak points, matching the chain-break sanity check).

## The missing SDE-200 comparison, found after user pushback (2026-07-03)

User asked "How do they compare to SDE200, the baseline" after the campaign-complete report above -- the report had only compared NFE40 vs NFE50 to each other, never against the actual SDE-200 baseline. First search (grepping doc files + A6000 filesystem for "SDE-200"/"sde_200") came up short beyond the already-known 7AD5-only result. User: "They definitely exist. Check more carefully under more config names; this has happened before. CHECK CAREFULLY!"

Re-grepped RAW-ARCHIVE's own older sections and found the lead: `..._unified_sderef8k` (line 8, "APPLES-TO-APPLES CAMPAIGN" section) -- checked on SuperCloud (`ssh SuperCloud`, not A6000 -- this data was from the original scaling8 apples-to-apples work which ran on SuperCloud/`cou` per §1.6), found the directory exists but contains ONLY 7AD5_A. This looked like it might be the full answer but was a decoy -- checking `ls ~/proteina/inference/` more broadly on SuperCloud revealed the BARE `..._unified` config (no suffix at all) actually contains raw samples + `proteinebm_v2_cathmd_analysis/proteinebm_scores_{pid}.csv` for essentially all 26 bad_afdb26 proteins. This is the true SDE-200 baseline (dt=0.005, schedule=log/2.0, sampling_mode=sc -- the actual production default) -- it was missed on the first pass because searching for "SDE"/"200" in directory names doesn't match a bare/unsuffixed config name.

Sample counts are NOT matched: the 8 scaling8 proteins have N=8192 (an early apples-to-apples investment); the 18 bad_afdb26-only proteins only have N=512 (7ZK0_A at 1024) -- SDE-200 was never run to the full 8192 for those, likely because 200-step sampling is 4-5x the compute of the fast configs and the original scope only covered the 8 hard scaling8 targets at that depth.

**Full comparison table** (best-of-NFE40/NFE50 vs SDE-200 oracle `tm_ref_template`, all NFE-side N=8192 except 7AD5 at N=2048 -- see campaign table above):

```
protein       nfe40    nfe50  best_nfe   sde200   sde_N  delta(nfe-sde)   winner
8QXI_A          nan   0.5252    0.5252   0.5187    8192         +0.0065      NFE
6ZYG_A       0.3744   0.4169    0.4169   0.4281    8192         -0.0112      SDE
7AD5_A       0.6223   0.6235    0.6235   0.6499    8192         -0.0264      SDE
6ZUS_A       0.3300   0.3232    0.3300   0.3283    8192         +0.0017      NFE
8RJX_A       0.3925   0.3766    0.3925   0.4211    8192         -0.0286      SDE
7KW9_A       0.4668   0.4776    0.4776   0.4894    8192         -0.0118      SDE
6ZTG_A       0.2743   0.2884    0.2884   0.2857    8192         +0.0027      NFE
8XHT_A       0.4115   0.4134    0.4134   0.4156    8192         -0.0022      SDE
8AP5_A       0.3021   0.3055    0.3055   0.2534     512         +0.0521      NFE
7ZK0_A       0.2246   0.2522    0.2522   0.5742    1024         -0.3220      SDE
7MQQ_A       0.3038   0.3092    0.3092   0.4196     512         -0.1104      SDE
8F3K_A       0.3853   0.3607    0.3853   0.3512     512         +0.0341      NFE
7VZM_A       0.3603   0.3165    0.3603   0.3536     512         +0.0067      NFE
6UF2_A       0.3302   0.3259    0.3302   0.4658     512         -0.1356      SDE
7F7N_A       0.2680   0.2672    0.2680   0.2647     512         +0.0033      NFE
8JB9_A       0.2642   0.3979    0.3979   0.2934     512         +0.1045      NFE
7OIO_A       0.2839   0.3406    0.3406   0.3496     512         -0.0090      SDE
7TXX_A       0.2649   0.3242    0.3242   0.3111     512         +0.0131      NFE
7A2D_A       0.3825   0.2719    0.3825   0.3692     512         +0.0133      NFE
8IN4_A       0.2703   0.2626    0.2703   0.2548     512         +0.0155      NFE
6U1O_A       0.3405   0.2858    0.3405   0.3370     512         +0.0035      NFE
7E7T_B       0.7308   0.8888    0.8888   0.8893     512         -0.0005      TIE
6M5Y_A       0.3483   0.2661    0.3483   0.6543     512         -0.3060      SDE
7DME_A       0.4202   0.2975    0.4202   0.4032     512         +0.0170      NFE
8TVL_A       0.1901   0.1835    0.1901   0.1863     512         +0.0038      NFE
8AUC_B       0.4021   0.3588    0.4021   0.3693     512         +0.0328      NFE

scaling8 (N=8192 both sides, fair):     NFE-best mean=0.4334  SDE200 mean=0.4421  wins: NFE=3 SDE=5 tie=0
new18 (SDE200 disadvantaged, N=512-1024 vs NFE's 8192): NFE-best mean=0.3620  SDE200 mean=0.3944  wins: NFE=12 SDE=5 tie=1
```

See PLAN finding #14 for the interpretation. Retrieval script: `~/scratchpad/sde200_oracle_summary.py` (stdlib-only, reads `proteinebm_scores_{pid}.csv` directly out of each protein's tar via Python's `tarfile` module, no venv/dependencies needed) run on SuperCloud against the bare `_unified` config's `seq_cath_cond` directory.

## N=1024/top-k=16 comparison + NFE30-50 @ SDE_UNTIL_T=0.8 launch (2026-07-16)

Session resumed after a 13-day gap (last A6000 action was 2026-07-03). User: "Try compare everything at N=1024 with top-k=16. And also try NFE30-50 with SDE->0.8. Run on scaling 8."

**N=1024/top-k=16 re-evaluation of existing data (no new compute, pure analysis):** for each config, took the FIRST 1024 rows of each protein's `proteinebm_scores_{pid}.csv` (generation order), sorted ascending by `energy` (lower = model-preferred, matching established `TM@minE` convention elsewhere in this doc), took the best (max `tm_ref_template`) among the 16 lowest-energy rows. Cross-checked the SDE-200 source: A6000 ALSO has a bare `_unified` config directory, but with only N=512 samples/protein -- a stale/different partial copy, inconsistent with the verified N=8192 SuperCloud baseline used for finding #14. Discarded the A6000 copy, used SuperCloud's N=8192 SDE-200 data throughout for consistency.

Full N=1024/top16 table (protein, SDE200_pool@1024, SDE200_top16, NFE40_pool@1024, NFE40_top16, NFE50_pool@1024, NFE50_top16, winner-by-top16):
```
protein     SDE200_pool  SDE200_k16  NFE40_pool  NFE40_k16  NFE50_pool  NFE50_k16  best_k16_winner
8QXI_A           0.5068      0.4814         nan        nan      0.5084     0.4818            nfe50
6ZYG_A           0.4281      0.4281      0.3370     0.2698      0.4169     0.2665              sde
7AD5_A           0.5738      0.5738      0.6223     0.6223      0.6141     0.6141            nfe40
6ZUS_A           0.3029      0.2353      0.3289     0.2303      0.3232     0.2548            nfe50
8RJX_A           0.3963      0.3963      0.3485     0.3464      0.3652     0.3652              sde
7KW9_A           0.4584      0.4584      0.4668     0.4428      0.4689     0.4689            nfe50
6ZTG_A           0.2823      0.2456      0.2580     0.2342      0.2699     0.2353              sde
8XHT_A           0.3724      0.2434      0.3253     0.2420      0.3325     0.2161              sde

mean top16_best_tm across 8 proteins: SDE200=0.3828  NFE40=0.3411 (n=7, 8QXI excluded -- known corrupted analysis)  NFE50=0.3628
win counts: sde=4, nfe40=1, nfe50=3
```
8QXI_A's NFE40 entry is NaN because its `energy` column (not just `tm_ref_template`) is all-unparseable in that specific analysis CSV -- the same pre-existing corruption from the 2026-07-02 duplicate-driver incident, not new. Conclusion: even at the cheaper, more realistic N=1024/top-16 deployment scenario, SDE-200 remains ahead on both mean and per-protein win count -- reinforces finding #14, does not overturn it.

**NFE30-50 @ SDE_UNTIL_T=0.8 launch.** First checked A6000 (where all prior nfe30/40/50 work ran): `date` showed 13 real days had passed; GPU0/GPU1 were now occupied by a legitimate, unrelated `train_openfold.py` job (prune_singleseq_ws5_continued_pda_eval, PID 709600, started 2026-07-14, 99.8% CPU, DDP across GPU0+GPU1 -- confirmed via `ps aux` + `nvidia-smi --query-compute-apps`, same `jupyter-chenxi` account but a different workstream, not touched). Only GPU2 was free; GPU3 remains the standing "never touch, another job's" exclusion regardless of instantaneous idle reading. Asked the user how to proceed given reduced A6000 capacity; user chose to run on SuperCloud instead.

SuperCloud's `~/proteina/configs/experiment_config/` did NOT have the nfe30/40/50_sde06 YAML files (only the OUTPUT directories exist there, likely pushed as data via the SuperCloud-as-hub convention while the actual earlier sampling ran on A6000) -- so the sde08 configs were built fresh from SuperCloud's own bare `_unified` config (the verified SDE-200 baseline file) by changing only `inference.dt` (0.005 -> 0.0333/0.025/0.02 for nfe30/40/50) and `inference.schedule.{schedule_mode,schedule_p}` (log/2.0 -> power/3.0, matching the "targeted p3" shape) -- diff-verified against the A6000 nfe40_sde06 content to confirm no other fields drifted. `PROTEINA_SDE_UNTIL_T` is a pure runtime env var (confirmed by reading the nfe40_sde06 YAML in full -- no SDE-related key anywhere in it), so no other config changes were needed for the new threshold.

Launched as 3 SLURM sbatch arrays (`scripts/scaling8_sde08_array.sh`, modeled on the existing `scaling_run_sc8.sh` but with `af2rank_top_k=0` instead of 8, `PROTEINA_NSAMPLES_PER_PROTEIN=1024`, and `PROTEINA_SDE_UNTIL_T=0.8` added), `--array=0-7` mapping array index to the 8 scaling8 proteins via a bash array (mirrors `~/data/bad_afdb/scaling8.csv` row order), one job per NFE config: 5156188=nfe30, 5156189=nfe40, 5156190=nfe50. Verified dependency paths (pae.ckpt, pretrain ckpt, scaling8.csv) exist before submitting. `xeon-g6-volta`'s per-account node cap throttled concurrent execution to ~4 nodes at a time (`AssocGrpNodeLimit` in squeue) -- expected/self-resolving, not an error, matches the memory note that this partition is capped.
