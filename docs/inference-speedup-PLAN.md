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
**APPLES-TO-APPLES NOW RUNNING:** fast p3@NFE25 in **seq_cath, 8192, on the 8 scaling8 targets** (reusing scaling8.csv + the SDE-200 seq_cath scaling8 baseline), matched settings (recycles 6, af2rank_top_k 8, cutoffs 256-8192). Launcher `scripts/scaling_run_sc8.sh <cfg> <pid>`; smoke = 8QXI job **4980637**; fan out 7 more on validation → convergence (energy+oracle) vs SDE-200 scaling8 (energy 0.331→0.404, oracle 0.419→0.459). seq_cond gap: SDE-200 seq_cond only has 1039 samples (no 8192) → a seq_cond apples-to-apples would need fresh SDE-200 8192 generation (slow) — deferred/ask.
**OLD HEADLINE BELOW IS SUPERSEDED on the baseline-comparison points; Step B/C schedule + commitment findings still stand.**

### APPLES-TO-APPLES CAMPAIGN — jobs + cron TODO (launched 2026-06-25 ~20:40)
Targets = 8 scaling8 (8XHT,7AD5,6ZUS,8QXI,6ZTG,7KW9,8RJX,6ZYG). All 8192. Settings match SDE-200 seq_cath baseline: **recycles 6, af2rank_top_k 8, cutoffs 256-8192, cif natives**. Launchers: `scripts/scaling_run_sc8.sh <cfg> <pid>` (seq_cath), `scaling_run_seqcond8.sh <cfg> <pid>` (seq_cond, recycles 6), `conv_sc8.sh <energy|oracle> <cfg> <cond_subdir>`.
Configs: FAST=`..._spdref_vf_powerp3_nfe025_scale`; SDE clone=`..._unified_sderef8k` (cp of base, sc/NFE200).
LAUNCHED: seq_cath FAST 8 = `spd_sc8` 4980637(8QXI RUNNING)+4981036-42 → `inference/<FAST>/seq_cath_cond/`. seq_cond SDE-200 8 = `spd_sq8` 4981043-50 → `inference/<SDE>/seq_cond/` (SLOW ~13.5h 8XHT).
**CRON TODO (priority order):**
1. ⭐ When all 8 seq_cath FAST `SC8_EXIT=0` → `sbatch conv_sc8.sh energy <FAST> seq_cath_cond`; on `CV8_RC=0` → `conv_sc8.sh oracle <FAST> seq_cath_cond`. Parse with `conv_summary.py`. **Compare to SDE-200 seq_cath scaling8 (energy 0.331→0.404, oracle 0.419→0.459)** + per-protein (esp 7AD5 0.43→0.68 SDE vs fast). THIS is the conclusion-deciding deliverable.
2. When 8 SDE seq_cond `SQ8_EXIT=0` → `conv_sc8.sh energy/oracle <SDE> seq_cond` → SDE-200 seq_cond 8192 curve (the primary-conditioning baseline that never existed).
3. seq_cond FAST side: existing fast seq_cond is recycles-1 (7AD5/8QXI/6M5Y/8TVL/8AUC). For the seq_cond comparison generate the 6 missing scaling8 fast seq_cond + run conv; cross-recycles (fast rec1 vs SDE rec6) acceptable per runbook (rec1≈rec6) — note caveat, or regen consistent.
4. Record all curves; keep GPUs busy; never idle. Monitor cron = **1165ba6b** (every 18 min).
PROGRESS (21:00): 8QXI seq_cath FAST done — best_ref_pred_tm@8192 = **0.466 ≈ SDE-200 seq_cath 0.467** (easy/saturated protein; pipeline validated). 7/8 seq_cath fast still running (sharing nodes with the other agent's sc_af2_full → ~2-3 nodes for us).
⚠️ PROGRESS (21:37): **af2rank HANG likely RECURRING on 7AD5 (job 4981037)** — 1/8 top-k structures written (CLEAN, 0 NaN), then frozen 25min at "Tuning chunk size" with CPU ~90% pinned. Clean structure #1 ⇒ a **non-NaN trigger** of the same USalign-no-timeout hang ⇒ the bug is broader than NaN (8QXI completed fine ⇒ geometry-specific). Holding scancel until next-fire confirmation (still 1 struct + frozen ⇒ definitive). **IMPLICATION: the af2rank timeout fix (task #9) is now a campaign BLOCKER — some targets will hang at af2rank; needs the timeout+guard fix before re-runs.** 8XHT still sampling (~63min, long pole), 6ZUS scoring (fresh log).
✅ RESOLVED (21:55): the af2rank hang is **ALREADY FIXED in the codebase** (the other agent's `nanrepro` work; md5 identical local↔remote): `USALIGN_TIMEOUT_S = 120` on BOTH USalign wrappers (`tmscore` :87, `usalign_files` :133) + NaN guard in `_save_openfold_prediction_pdb` (:150 raises on NaN). 7AD5 hung only because job 4981037 loaded the PRE-fix code at launch (~20:30). Action: scancel'd hung 7AD5 → cleared its partial `af2rank_on_proteinebm_top_k` → **re-ran = job 4982377** (reuses tar+EBM, redoes only af2rank with the 120s timeout). Campaign UNBLOCKED. **Task #9 (hang fix) = DONE by other agent; verify on 4982377 completion.** Done so far: 8QXI, 6ZUS (both completed cleanly — fix works for them).
⭐ PROGRESS (~23:10): **7/8 seq_cath FAST done — best_ref_pred_tm@8192 single-cutoff:** 8QXI=0.466, 8XHT=0.200, 6ZUS=0.249, 6ZTG=0.231, 7KW9=0.484, 8RJX=0.417, 6ZYG=0.285 (mean of 7 ≈ 0.333). Only 7AD5 left (re-run 4982377). PREVIEW: fast seq_cath mean (~0.33) looks BELOW SDE-200 seq_cath energy mean 0.404 — consistent with the fast config giving up quality; rigorous per-protein head-to-head awaits the convergence.
⚠️ NODE-CONTENTION FIX (~23:10): SDE seq_cond jobs (long) were hogging nodes, blocking the fast 7AD5 seq_cath re-run → seq_cath conv (PRIORITY) stuck. Action: **scontrol hold 4981047-50** (pending SDE seq_cond) + **scancel 4981046** (6ZTG-SDE, 59min, least progress) to free a node for 7AD5-rerun. (scontrol top = permission-denied; holding achieves it.) SDE seq_cond STILL RUNNING: 4981044(7AD5 ~2h), 4981045(6ZUS ~1.6h). **CRON TODO (updated):** (1) when 7AD5-rerun 4982377 SC8_EXIT=0 → all 8 seq_cath done → `conv_sc8.sh energy <FAST> seq_cath_cond` then oracle → record head-to-head vs SDE-200 seq_cath. (2) AFTER seq_cath conv launched → `scontrol release 4981047 4981048 4981049 4981050` + re-submit 6ZTG SDE (`scaling_run_seqcond8.sh <SDE> 6ZTG_A`) to resume the seq_cond baseline gen. (3) NB 8QXI SDE seq_cond (4981043) status unclear — verify it ran.
⚠️ CORRECTION (~23:15): the scancel of 6ZTG-SDE did NOT unblock 7AD5-rerun — it's still `AssocGrpNodeLimit` with node d-7-7-1 FREE ⇒ the **cou account node cap is SHARED with the other agent** and we're at it (my 2 + their 2). Freeing one of MY nodes doesn't help (the cap, not node availability, is the limit). LESSON: don't scancel to "free a node" under a shared-account cap. Reverting to: keep 7AD5-rerun (4982377) as top eligible (pending SDE 47-50 held) → it grabs the next slot when a running SDE (44=7AD5-SDE 2h, 45=6ZUS-SDE 1.6h) finishes (~2-3h). Nodes stay productive on seq_cond gen meanwhile; seq_cath conv comes when 7AD5-rerun completes. (6ZTG-SDE 59min wasted — re-run post-conv with 47-50 release.)
✅ STATE (~02:50): **8/8 seq_cath FAST done; ENERGY conv 4986865 RUNNING** (→ on CV8_RC=0, cron launches `conv_sc8.sh oracle <FAST> seq_cath_cond`). **@8192 head-to-head recorded above (fast 0.365 < SDE-200 0.404).** SDE seq_cond map (submit order 43-50 = 8QXI,7AD5,6ZUS,6ZTG,7KW9,8RJX,6ZYG,8XHT): **DONE 3/8** = 43-8QXI,44-7AD5,45-6ZUS; **RELEASED+running** = 47-7KW9, 48-8RJX; **HELD** = 49-6ZYG, 50-8XHT; **CANCELLED (re-run needed)** = 46-6ZTG. CRON: after seq_cath oracle conv launched → release 49,50 + re-run 6ZTG SDE; when SDE seq_cond 8/8 → `conv_sc8.sh energy/oracle <SDE> seq_cond`. STILL TODO for the seq_cond head-to-head: **fast seq_cond on ALL 8 scaling8 targets via `scaling_run_seqcond8.sh <FAST_cfg> <pid>` (recycles 6 — do NOT reuse the earlier recycles-1 7AD5/8QXI; the SDE seq_cond is recycles 6, must match)** + fast seq_cond convergence vs SDE seq_cond convergence. Launch when nodes free (after oracle conv + SDE seq_cond gen).
PARKED behind this priority check (resume after): Q4 EBM relax-then-score test (rank by BOTH pTM+energy, metric=Spearman vs tm_ref_template; relax_decoys.py exists on SuperCloud); Step D = OSS/GITS DP-searched schedule; Step E = LD3/DSS pilot → schedule-aware SiD (score-identity loss); fix af2rank NaN/USalign hang (timeout+NaN guard); add native-anchored commitment metric; full plan→summary cleanup.

### ⭐ Q4 — ProteinEBM RELAX-THEN-SCORE: readiness map + un-relaxed BASELINE (2026-06-26, NOT launched)
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

## ⭐⭐ AUTONOMOUS DIRECTIVE (user away ~12h: 2026-06-25 early-AM → ~midday 2026-06-25)
User left ~12h with explicit order: KEEP PUSHING, never let GPUs idle, use best judgment, record decisions in DETAIL, QC at end, DO NOT STOP unless truly blocked. ORDER:
1. **Push the ceiling** — power p3/p4/p5 @ NFE15 & NFE10 (13–20×), vf-seq, rec1. [launching]
2. **Implement Step C (commitment)** — env-gated r3n_fm.py hook (user: "exactly what I envisioned it for").
3. **AFTER tuning** (best schedule+NFE on the 16-sample eval): SCALING experiment on the winner — sample up to **8192**, check best_ref_pred_tm at each power-of-2 cutoff (256/512/1024/2048/4096/8192) + **ProteinEBM-vs-Oracle ranking** (mirror rosy-fox scaling8: `sampling_convergence_analysis.py --ranking_mode {energy,oracle}`; read `~/.claude/plans/users-chenxi-downloads-s41467-025-67127-rosy-fox.md` §scaling8 + runbook §convergence). Also confirm winner on seq_cath + SDE.
### ⭐ AUTONOMOUS-WINDOW STATUS (~10:15 EDT 2026-06-25, for QC)
**Directive items 1-3: ALL DONE + recorded below.** Completeness jobs still running: 5-prot p3 oracle (job 4975504, full oracle curve over all 5 lengths) + 8AUC-p4 (job 4975144, 5th p4 point) — both tabulated in next fires. **Step D (training-free few-step / higher-order ODE solvers, approach #1) and Step E (distillation on the smart schedule) were NOT started autonomously** — each needs a USER design decision (which solver: Heun/DPM-Solver/midpoint; distillation target+setup) per the no-random-design-decisions rule → flagged for your QC. 2 nodes are free but no in-scope item-1-3 work remains, so I did not fabricate make-work to fill them.

### ⭐ HEADLINE RESULTS (as of ~07:15 2026-06-25, for QC) — bad_afdb held-out, gate = af2rank refined-pred-TM (ref_pred_tm), vs SDE-200 baseline 0.348
- **Step A baseline:** SDE-200 ≈ ODE-200 (ref_pred_tm 0.348) → ODE is the right sampler to accelerate (and ODE holds up far better than SDE at low NFE).
- **Step B (approach #2) = THE WIN:** a HIGH-noise-emphasis `power` schedule on the ODE gives **8× speedup at 97% of baseline** (power p3 @ NFE25 = 0.338 vs 0.348). Full ceiling: p3@NFE25(8×)=0.338, p4@NFE15(13×)=0.325, p5@NFE10(20×)=0.303 — optimal exponent RISES as NFE drops. Schedule is **ODE-SPECIFIC** (same schedule on SDE = 0.265, doesn't transfer). Generalizes to seq_cath (0.334).
- **Step C (approach #3):** the model commits its fold at HIGH noise (t≲0.275), **LENGTH-DEPENDENT** (longer proteins commit earlier/higher-noise) — mechanistically explains why high-noise emphasis works + why optimal p rises with speed.
- **Item 3 (scaling + ProteinEBM-vs-Oracle, 4-prot subset):** inference-time scaling **SATURATES by ~256–512 samples**; EBM-vs-oracle gap **~0.05** (ProteinEBM is a decent selector); fast config + a few-hundred samples (~0.47–0.49) ≈ achievable ceiling (~0.53) ≫ baseline (0.348).
- **PENDING:** 8AUC (492, job 4974418) — sstat-confirmed ALIVE (~100% CPU); 8192×492-res at NFE25 ≈ ~5h SAMPLING alone (per L²-scaled rosy-fox timing), ~done ~08:00 + ~1h enrichment ~09:00. Plan: record its energy@8192 from prediction_summary (cheap); the **4-protein convergence is the main result** — will only re-run the full 5-protein convergence if time/nodes permit (low value-add: conclusion robust). p4-aggressive convergence = optional bonus. Detailed numbers + job IDs in the decisions log below.

### Decisions log (autonomous)
- (start) Phase 2b configs = power{3,4,5}×NFE{10,15} + log NFE10 baseline (7 jobs). dt: NFE15=0.0667, NFE10=0.1. Jobs spd_swp, tags swp_vf_seq_{powerp3,powerp4,powerp5}_nfe0{15,10} + swp_vf_seq_nfe010.
- Autonomous-driver cron = **ec848464** (drives the whole ordered list).
- **STEP C HOOK DONE + DEPLOYED:** env-gated capture in `r3n_fm.py` (added `import os`; in `full_simulation`: read `PROTEINA_TRAJ_DUMP`/`PROTEINA_FORK_FROM`/`PROTEINA_FORK_T_IDX`, capture x0_pred+x_t per step, loop `range(_sc_start,nsteps)` for fork, save dump before return). DEFAULT-OFF = zero behavior change. Decision: local+SC r3n_fm.py were md5-identical → edited LOCALLY (5 precise edits, py_compile OK) → scp'd ONLY r3n_fm.py to SC (NOT git: local/SC drifted 93bfbbb vs 62855c4 + shared repo; env-gated = safe). SC syntax-verified in cathfold. ⚠️ If another agent `git checkout`s r3n_fm.py the hook reverts → re-scp from /Users/Chenxi/SOLab/proteina/...
- **STEP C DRIVER READY:** `scripts/commitment_run.sh <pid>` = 1 protein, 8 samples, **uniform-200** config `..._spdref_vf_uniform200` (even t-coverage to resolve t*), `--skip_scoring`, dumps `commitment/traj_<pid>.pt` via the hook (fp32). SMOKE = **8QXI_A job 4974316** (queued behind phase 2b). Analysis `scripts/commitment_analyze.py --dumps traj_*.pt` → TM(x0_pred(t), final) vs t → t* (commitment noise-time; t=0 noise→t=1 data).
- **CRON TODO for Step C:** when 4974316 done → verify `traj_8QXI_A.pt` loads with keys t/x0_pred/x_t/x_final (lists of [K,L,3]); run commitment_analyze on it; if dump good → submit commitment_run for diverse proteins 6M5Y_A(132) 8TVL_A(338) 8AUC_B(492); re-analyze all → record t* in plan → use t* to design/justify the schedule (power emphasis should bracket t*).
- ✅ **STEP C HOOK VALIDATED (02:22):** 8QXI_A smoke 4974316 CMT_EXIT=0; `traj_8QXI_A.pt` (3.1MB, nsteps=200, K=8, L=78, coords in nm→×10) loads + analyzes. ⭐ **8QXI t*** (x0_pred TM-to-final): TM≥0.5 @ **t=0.275**, TM≥0.7 @ **0.395**, TM≥0.8 @ 0.56, TM≥0.9 @ 0.75. → **commitment window ≈ t∈[0.27, 0.5] (MID noise)**; fine detail locks late (t≈0.75). VALIDATES high-noise emphasis AND refines it: dense around t≈0.3–0.5 (NOT extreme t→0) → predicts power p3/p4 > p5 (p5 too extreme, wastes steps at t→0 before commitment). Diverse commitment runs LAUNCHED: 6M5Y_A 8TVL_A 8AUC_B = jobs **4974376-78**.
- **CEILING IDs→tags:** 4974288=powerp3_nfe015, 89=powerp3_nfe010, 90=powerp4_nfe015, 91=powerp4_nfe010, 92=powerp5_nfe015, 93=powerp5_nfe010, 94=nfe010(log). First wave (88-91) DONE; 92-94 running. Full ceiling tabulation when all 7 done.
- ⭐⭐ **STEP C C0 RESULT (4 proteins, x0_pred TM-to-final commitment t*, t=0 noise→t=1 data):**
| protein | L | t*(TM0.5) | t*(TM0.7) | t*(TM0.9) |
|---|---|---|---|---|
| 8QXI_A | 78 | 0.275 | 0.395 | 0.75 |
| 6M5Y_A | 270 | 0.175 | 0.290 | 0.54 |
| 8TVL_A | 338 | 0.105 | 0.195 | 0.52 |
| 8AUC_B | 493 | 0.075 | 0.150 | 0.45 |
  **⭐ KEY: commitment is LENGTH-DEPENDENT — longer proteins lock the fold EARLIER (higher noise / lower t):** TM0.5 at t=0.275(L78)→0.075(L493). Fold-determining work is at HIGH noise (t≲0.275, more so for long chains) → STRONGLY justifies power(p>1) high-noise emphasis, and predicts a **length-dependent optimal p** (more extreme for longer). Fine detail (TM0.9) locks at t≈0.45–0.75 → low-noise steps are refinement (coarse OK). Files: commitment/commitment_curve{,_tstar}.csv. **STEP C C0 DONE** (fork-sweep = optional later confirmation, needs SDE). seq_cath confirm of p3@NFE25 launched (tag swp_vf_seqcath_powerp3_nfe025) to fill the free node.
- ⭐⭐ **STEP B CEILING CURVE DONE (ref_pred_tm mean, vf-seq rec1):**
| NFE (speedup) | log | power p3 | power p4 | power p5 |
|---|---|---|---|---|
| 25 (8×) | 0.277 | **0.338** | — | — |
| 15 (13×) | 0.278 | 0.298 | **0.325** | 0.302 |
| 10 (20×) | ~0.27 | 0.297 | 0.288 | **0.303** |
  (SDE-200 baseline 0.348.) ⭐ **OPTIMAL p INCREASES as NFE DROPS: p3@25 (0.338) → p4@15 (0.325) → p5@10 (0.303)** — matches Step C (lower NFE needs more high-noise concentration). Speed-quality frontier: **p3@NFE25 = 0.338 (97% of baseline, 8×)** [WINNER for quality]; p4@NFE15 = 0.325 (94%, 13×); p5@NFE10 = 0.303 (87%, 20×). **STEP B DONE.** → Item 3 SCALING on WINNER = **power p3 @ NFE25** (also test p4@NFE15 as the aggressive point).
- ⭐ **ITEM 3 SCALING LAUNCHED (8192 gen):** winner power p3 @ NFE25, FRESH config `..._spdref_vf_powerp3_nfe025_scale` (no overwrite of the 16-sample dir), 8192 samples, ProteinEBM + af2rank top-8 + enrichment (tm_ref_template on ALL samples → Oracle data), recycles 1, fp16. Subset (length-diverse) = 8QXI_A(78) 7AD5_A(124) 8TVL_A(338) 8AUC_B(492) = jobs **spd_scl** (`scaling_run.sh <pid>`). Out: `out/scl_<pid>`, samples `inference/<...scale>/seq_cond/<pid>.tar`. ⏳ hours (8192 + scoring).
- **CONVERGENCE PREPPED (proven-script-based):** `~/proteina_speedup/scripts/conv_scl.sh <energy|oracle>` (1 GPU; mirrors rosy-fox conv_*_scaling8.sbatch). My `_scale` config, `--inference_dir inference/<_scale>/seq_cond`, `--csv_file scl_subset.csv` (4 prot: 8QXI/7AD5/8TVL/8AUC), `--cif_dir ~/af2ctrl_badafdb/natives` (natives confirmed for all 4), cutoffs 256-8192, af2rank_top_k 8, recycles 1, cuEq + tar_protein_dirs. **⭐ CRON ACTION when ALL 5 p3-scaling jobs SCL_EXIT=0** — IDs **4974415=8QXI, 4974416=7AD5, 4974480=6M5Y, 4974417=8TVL, 4974418=8AUC** (check `grep SCL_EXIT=0 logs/scl_<id>.out` for these 5; IGNORE other spd_scl = p4 bonus). Readiness = all 5 done sampling+scoring (8AUC is the long pole). NB the 8192 SCORING tail is SLOW for long proteins — enrichment does cg2all-reconstruct + USalign on ALL 8192 samples (~1h for L~338); so a job at 3h with a real tar is likely in enrichment, not hung (verify via sstat AveCPU). ALSO: a non-spd_ job `sc_af2_smo` (another agent, shared cou acct) is queued behind mine — DO NOT touch; minor node contention only. THEN: `sbatch conv_scl.sh energy`; when energy `CONV_RC=0` → `sbatch conv_scl.sh oracle` (⛔ NEVER concurrent = tar corruption; use afterok or wait). Outputs `inference/<_scale>/seq_cond/scl_convergence_{energy,oracle}/sampling_convergence_summary.json`. Tabulate energy-line (EBM-selected ref_pred_tm vs #samples), oracle-line (true-TM-selected), + EBM-vs-oracle GAP per cutoff → record (this is the inference-time-scaling + selector-bottleneck result).
- ✅ **WINNER CONFIRMS on seq_cath:** power p3 @ NFE25 seq_cath ref_pred_tm = **0.334** ≈ seq 0.338 (94% of seq_cath-200 baseline 0.355) → schedule GENERALIZES across conditioning. **SDE confirm LAUNCHED:** config `..._spdref_sc_powerp3_nfe025` (sampling_mode=sc + power p3 + NFE25), tag swp_sc_seq_powerp3_nfe025 (queued behind scaling) → tests if the winning SCHEDULE holds on the stochastic SDE sampler (not ODE-only). Cron: tabulate vs SDE-200 baseline 0.348 when done.
- ⭐ **SDE CONFIRM RESULT — winner schedule is ODE-SPECIFIC:** power p3 @ NFE25 on **SDE (sc) = 0.265** ≪ **ODE (vf) = 0.338** (SDE also EBM-ptm 0.04 vs 0.38, energy 15k vs 7.4k — much worse by all measures). The high-noise-emphasis schedule does NOT transfer to the stochastic sampler (SDE noise injection at the big low-noise steps disrupts refinement). → **For the speedup, use the deterministic ODE + power schedule.** SDE confirm DONE. Reinforces lit-review risk #1 (fast solvers/schedules target the ODE).
- SCALING PROGRESS (03:41): 8QXI_A SCL_EXIT=0 (76MB tar done); 7AD5_A finished sampling (8216 pdb)→scoring; 8TVL_A(338)/8AUC_B(492) still SAMPLING (slow long proteins, ~L²×8192). ⚠️ NORMAL during a scaling job's inference: orchestrator log mtime STALE (subprocess output hidden) + `find <pid> -name '*.pdb'`=0 until trainer.predict finishes (then ALL 8192 appear → score → tar). Do NOT false-alarm on stale-log/pdb=0 while SLURM=RUNNING — verify liveness via `sstat -j <jobid>.batch --format=AveCPU,MaxRSS` (CPU time accumulating ≈ elapsed = alive; confirmed 8TVL/8AUC ~100% CPU at 1:25/1:08 elapsed, just slow 8192 long-protein sampling). SDE confirm 4974443 running on 8QXI's freed node. UPDATE: 8QXI_A+7AD5_A SCL_EXIT=0 done; freed a node → launched scl 6M5Y_A + added it to scl_subset.csv (convergence subset now 5 proteins: 8QXI/7AD5/8TVL/8AUC/6M5Y; 6M5Y finishes before the 8AUC long-pole, no critical-path cost). NB scl_subset.csv "length" col is unreliable (6M5Y shows 132 but is ~270 FASTA — rosy-fox known issue); pipeline uses the true seq length.
- AGGRESSIVE-POINT (bonus): p4@NFE15 scaling launched on 8QXI via generalized `scaling_run_g.sh <cfg> <pid>` (config `..._spdref_vf_powerp4_nfe015_scale`, out `out/scl_powerp4_nfe015_scale_8QXI_A`). Short protein → finishes fast, won't delay the p3@NFE25 convergence. Gives p4@NFE15 best_ref_pred_tm @8192 (from prediction_summary) to compare vs p3@NFE25 @8192. ⚠️ PRIORITY: do NOT queue a full p4 batch that would steal nodes from the p3 convergence (the core item-3 deliverable); p4 is a light bonus only.
- ⭐ **SCALING 8192 PREVIEW (single-cutoff best_ref_pred_tm, EBM-selected from 8192 + af2rank top-8):** 8QXI: p3@NFE25=**0.471**, p4@NFE15=**0.477** (both up from ~0.34 @16 samples → big inference-time-scaling gain; the AGGRESSIVE 13× p4 matches the 8× p3 at 8192). 7AD5: p3@NFE25=**0.179** (EBM selected a poor sample — selector-limited; the oracle line will show the achievable ceiling). → preview confirms (a) inference-time scaling strongly helps + (b) ProteinEBM-selector bottleneck on hard proteins. Full energy-vs-oracle convergence curve runs after all 5 p3 proteins finish. p4 scaling: 8QXI=0.477 (≈p3 0.471), 7AD5=0.189 (≈p3 0.179) → both schedules EBM-selector-limited on the hard 7AD5 (schedule doesn't fix the selector; oracle line will show the ceiling). p4 extended to 6M5Y (fills node). 8TVL p3 finished sampling (228MB tar)→scoring; 8AUC p3 still sampling (long pole).
- ⭐ **CONVERGENCE LAUNCHED (energy, 4-protein):** 4/5 p3 done (8QXI/7AD5/6M5Y/8TVL); 8AUC EXCLUDED (its long sampling+enrichment would push past midday) → ran on the 4 done length-diverse proteins (78-338) via `conv_scl.sh energy ~/proteina_speedup/csv/scl_subset4.csv` = job **4974630** (spd_conv). conv_scl.sh now takes an optional CSV 2nd arg. **⭐ CRON: when 4974630 CONV_RC=0 → `sbatch scripts/conv_scl.sh oracle ~/proteina_speedup/csv/scl_subset4.csv`** (⛔ NOT concurrent with energy — tar corruption). Outputs `inference/<...powerp3_nfe025_scale>/seq_cond/scl_convergence_{energy,oracle}/sampling_convergence_summary.json` → tabulate energy(EBM-selected) vs oracle(true-TM) ref_pred_tm per cutoff + the gap. 8AUC's 8192 single-cutoff result recorded separately when it finishes. p4-8TVL scaling launched (job 4974631; extends p4 coverage to 4 lengths).
- ⭐⭐ **ENERGY CONVERGENCE RESULT (job 4974630 CONV_RC=0; 4 prot 8QXI/7AD5/6M5Y/8TVL; EBM-selected max_min_tm_ref_pred vs #samples):** 256=**0.467**, 512=0.475, 1024=0.475, 2048=0.476, 4096=0.487, 8192=**0.488** (median 0.416→0.434). → ProteinEBM-selected quality **SATURATES by ~512 samples** (~0.48; only +0.02 total 256→8192 = modest inference-time scaling for the DEPLOYED EBM pipeline). Per-protein **max=0.92** (a near-perfect sample EXISTS in the 8192 pool) → big EBM-selector bottleneck expected. ⭐ Even **256 fast samples (0.467, 8×) ≫ SDE-200 16-sample baseline (0.348)**. ORACLE line (true-TM-selected = achievable ceiling, job **4974655**) running → quantifies the selector gap. Parser `scripts/conv_summary.py <summary.json> [metric]`.
- ⭐⭐⭐ **ORACLE CONVERGENCE RESULT (job 4974655 CONV_RC=0; 4 prot; true-TM-selected = achievable ceiling). FULL energy-vs-oracle:**
| #samp | Energy(EBM) | Oracle(true-TM) | gap |
|---|---|---|---|
| 256 | 0.467 | 0.533 | +0.066 |
| 512 | 0.475 | 0.533 | +0.058 |
| 1024 | 0.475 | 0.535 | +0.060 |
| 2048 | 0.476 | 0.534 | +0.058 |
| 4096 | 0.487 | 0.545 | +0.058 |
| 8192 | 0.488 | 0.537 | +0.049 |
  ⭐ **KEY: BOTH lines SATURATE EARLY (~256–512 samples)** — oracle 0.533→0.537 (essentially flat), energy 0.467→0.488. **EBM-vs-oracle gap modest (~0.05)** — much smaller than rosy-fox scaling8's ~0.12 (different setting: fast config + bad_afdb winner here). Both ≫ SDE-200 baseline 0.348. → **PRACTICAL: power p3 @ NFE25 (8×) needs only ~256–512 samples + ProteinEBM to reach ~0.47–0.49 (near the ~0.53 achievable ceiling); 8192 doesn't help (scaling saturates), and ProteinEBM is a decent selector (~0.05 below perfect).** **ITEM 3 SCALING+ORACLE DONE** (4-prot subset; 8AUC 8192 point pending for completeness; p4-aggressive convergence optional bonus).
- **p4@NFE15 (13×) @8192 single-cutoff (EBM-selected, 4 prot):** 8QXI=0.477, 7AD5=0.189, 6M5Y=**0.837**(!), 8TVL=0.127 → mean **0.408** (HIGH protein-variance: EBM selector excellent for 6M5Y/8QXI, fails for 7AD5/8TVL). p4 mean@8192 (0.408) < p3@8192 (0.488) — the more-aggressive 13× config is slightly worse EBM-selected (driven by 8TVL). **p4 CONVERGENCE bonus LAUNCHED:** energy job **4975143** (`conv_scl.sh energy scl_subset4.csv <p4_cfg>`; conv_scl.sh now takes config as 3rd arg) → CRON: on energy CONV_RC=0 → `sbatch conv_scl.sh oracle scl_subset4.csv <p4_cfg>` (p4_cfg=..._spdref_vf_powerp4_nfe015_scale). 8AUC-p4 (job 4975144) for the 5th-point completeness. Free nodes appeared (other agent's jobs finished).
- ⭐⭐ **p4 ENERGY CONVERGENCE (job 4975143 CONV_RC=0; 4 prot; EBM-selected):** 256=0.316, 512=0.407, 1024=0.408, 2048=0.410, 4096=0.425, 8192=**0.434**. vs p3 energy (0.467→0.488). ⭐ **KEY: the more-aggressive p4 (13×) is SAMPLE-HUNGRY** — rises strongly with samples (+0.12 from 256→8192) whereas p3 (8×) saturated by ~256. p3 stays ahead at every cutoff (8192: 0.488 vs 0.434), but p4 closes the gap with more samples (256-gap −0.15 → 8192-gap −0.054). → **p3@NFE25 is the better operating point** (higher quality + saturates fast); p4's extra speed is partly offset by needing ≥512 samples. p4 ORACLE launched (after energy, sequential).
- 8AUC-p3 (492, longest/hardest) DONE; @8192 EBM-selected = **0.288** (scales modestly — will pull the 5-protein mean below the 4-protein 0.488). **5-PROTEIN p3 ENERGY convergence LAUNCHED** (incl 8AUC, csv scl_subset.csv) → output `scl_convergence_energy_5p` (conv_scl.sh now takes OUT_SUFFIX 4th arg; 4-prot result `scl_convergence_energy` preserved). Completes the main inference-time-scaling curve over all 5 lengths (78-492). Skipping the 5-prot oracle (won't finish in-window; the 4-prot oracle + the robust conclusion are recorded).
- ⭐ **p4 ORACLE (job 4975168 CONV_RC=0; 4 prot, true-TM ceiling):** 256=0.478, 512=0.471, 1024=0.462, 2048=0.482, 4096=0.468, 8192=**0.493** (~flat ~0.48). p4 EBM-vs-oracle gap @8192 = 0.059 (≈ p3's ~0.05). **p4 ceiling (0.49) < p3 ceiling (0.54)** → the aggressive 13× config's achievable quality is slightly lower, AND its EBM curve is sample-hungry → p3@NFE25 remains the better operating point.
- ⭐ **5-PROTEIN p3 ENERGY (job 4975388, complete incl 8AUC L=492):** 256=0.410, 512=0.416, 1024=0.418, 2048=0.433, 4096=0.446, 8192=**0.448** (vs 4-prot 0.488 — 8AUC@8192=0.288 drags the mean; median flat 0.398 as 8AUC+7AD5 are EBM-hard). Still ≫ SDE-200 baseline 0.348. **5-PROTEIN p3 ORACLE launched (job 4975504 → scl_convergence_oracle_5p)** for the complete oracle curve (may finish ~12:30, around return). 8AUC-p4 still running.
- ⭐⭐⭐ **5-PROTEIN p3 ENERGY-vs-ORACLE DONE (oracle job 4975504 CONV_RC=0; complete curve over ALL 5 lengths 78-492 incl hard 8AUC):**
| #samp | Energy(EBM) | Oracle(true-TM ceiling) | gap |
|---|---|---|---|
| 256 | 0.410 | 0.506 | +0.096 |
| 512 | 0.416 | 0.506 | +0.090 |
| 1024 | 0.418 | 0.508 | +0.090 |
| 2048 | 0.433 | 0.508 | +0.075 |
| 4096 | 0.446 | 0.516 | +0.070 |
| 8192 | 0.448 | 0.510 | +0.062 |
  ⭐ Oracle ceiling **~0.51 (flat, saturates ~256)**; energy **~0.45 (rises slightly with samples)**; gap @8192 **~0.06** (slightly > 4-prot's 0.05 — 8AUC drags energy more than oracle). Both ≫ SDE-200 baseline 0.348. **CONFIRMS the 4-prot conclusion on the full length-diverse set.** → **ITEM 3 FULLY COMPLETE** (5-protein energy + oracle). Only 8AUC-p4 (5th p4 point) still running; will NOT run a 5-prot p4 convergence (the 4-prot p3-vs-p4 comparison already stands; marginal + runs past return).
- ⚠️ **8AUC-p4 (job 4975144) HUNG in af2rank → SCANCELLED (my own job, verified spd_scl/cou).** Diagnosis: sampling+ProteinEBM finished, but af2rank top-k prediction FROZE — both logs + all output files stuck at 12:02–12:03 while CPU pinned ~100% for ~4.75h (16:47 wall), only **1 of 8** top-k structures written (`predicted_structures/8AUC_B_2812.pdb`). A CPU-spin hang, not slowness. Likely trigger: the `[Errno 2] .cache/cuequivariance` kernel-cache miss → pathological cuEq-attention path for the 492-res protein (8AUC-**p3** did NOT hang → transient/structure-specific). Cancelled at 8:06 to recover the node (was wasting a GPU for ~5h). **NOT resubmitted** — non-critical 5th p4 completeness point, re-hang risk, user back. **Only casualty = 8AUC's p4@8192 single point; the 4-prot p4 convergence + full 5-prot p3 convergence are the main results, intact.** GOTCHA for future scaling jobs: if af2rank logs freeze with CPU pinned, suspect the cuEq cache miss — pre-warm `~/.cache/cuequivariance` or drop `--use_cuequivariance_attention` for the af2rank stage on long proteins.

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
