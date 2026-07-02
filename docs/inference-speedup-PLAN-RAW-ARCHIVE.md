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
