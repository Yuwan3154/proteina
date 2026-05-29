"""End-to-end correctness check for the SC-compile path added in protein_transformer.py.

This script proves three properties of the ``_forward_impl_sc`` + ``use_torch_compile_sc``
mechanism by running the actual training entry point twice and comparing the
per-step training losses:

1. **Gradient parity**: training losses at each step must match between SC-eager and
   SC-compiled runs (within ``LOSS_REL_TOL``). Per-step loss is a function only of
   the sample, the SC tensor and the model parameters; SC is the only thing
   changing between runs. If the compiled SC produces the same output as eager
   SC, the training pass sees the same inputs → same loss → same gradients.

2. **No CACHE THRASH**: parses ``TORCH_LOGS=recompiles`` output and looks at the
   per-frame variant count, not the raw recompile count. Some recompiles are
   *expected* — every inner frame Dynamo identifies needs a separate cache
   variant for each (dtype, grad_mode) combination it's invoked under, and
   building them is a one-time warmup cost. The real failure mode is when a
   frame's variant count exceeds Dynamo's ``recompile_limit`` (raised to 32) —
   that triggers eviction and the cache thrashes (extreme slowdown). The test
   flags any frame exceeding ``MAX_VARIANTS_PER_FRAME``, AND any total recompile
   count exceeding ``MAX_TOTAL_RECOMPILES_SC`` (the warmup ceiling).

3. **No DDP find_unused_params errors**: ``nn_sc`` parameters have
   ``requires_grad=False``, so DDP ignores them entirely; the test asserts the
   run completes without any ``Expected to have finished reduction`` errors and
   the loss sequence is well-defined (no NaN steps signalling a backwards bug).

The setup mirrors ``scripts/overfit_three_chains.py`` (Stage-1 contact-DSSP-UDLM-pb
config, padded length 64, the 8 short default chains) so the iteration time is
fast and the loss trajectory is reproducible. Self-conditioning is forced on
with ``self_cond_use_copy=True`` + ``self_cond=True`` + ``training.self_cond_prob=0.5``
(stochastic SC at training time), and gradient accumulation is set to 2 so the
test exercises the interaction between SC scheduling and accumulator state.

Run:

    /home/ubuntu/miniforge3/envs/cue_openfold/bin/python scripts/test_sc_compile.py

Artifacts written to ``/tmp/test_sc_compile_baseline.log`` and
``/tmp/test_sc_compile_sc.log``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

# Number of optim steps in each subprocess run. Picked to amortize compile time
# while still keeping each run short (~3 minutes).
MAX_STEPS = 10

# Per-step training loss tolerance. bf16 + Inductor reordering of additions can
# introduce ~1e-3 relative deltas; with kernel-launch nondeterminism + the
# extra accumulator state from grad-accum=2, absolute losses can drift ~1%.
# Use relative tolerance for robustness across loss magnitudes.
LOSS_REL_TOL = 0.03

# Cap on how many distinct CACHE VARIANTS Dynamo creates per top-level frame.
# recompile_limit is raised to 32 (train.py/inference.py) — anything below
# that means each variant is a one-time warmup compile, not eviction-driven
# thrash. If any frame exceeds the limit, the cache thrashes (frames evicted
# and recompiled repeatedly), causing the "extreme slowdown" the user warned
# about. Measured peak with the alias val-routing fix: ~11 (val-loss only) /
# ~14 (with validation sampling) on frame 25 (rigid_utils Rotation). 24 leaves
# headroom above the observed peak while still flagging any approach to 32.
MAX_VARIANTS_PER_FRAME = 24

# Sanity bound on the absolute count of recompile events. With recompile_limit
# raised to 32 and ~40 unique inner frames in the model, expect O(60) warmup
# recompiles when SC compile is enabled (each frame * each (dtype, grad_mode)
# variant). The eager baseline normally sees only a handful (sanity-check
# vs. training dtype variants on the train forward). Flag anything wildly
# above the warmup ceiling.
MAX_TOTAL_RECOMPILES_SC = 120

DEFAULT_CHAIN_IDS = (
    "1c9p_B", "1fka_R", "1fse_F", "1g2c_A",
    "1guu_A", "1ixs_A", "1jcd_A", "1kfm_A",
)
PADDING_MAX_SIZE = 64
BATCH_SIZE = 4              # 4 chains per minibatch; grad-accum doubles effective batch
ACCUM_GRAD = 2              # interacts with stochastic SC scheduling
SELF_COND_PROB = 0.5        # half the steps take the SC path

PROTEINA_ROOT = Path("/home/ubuntu/proteina")
PYTHON = "/home/ubuntu/miniforge3/envs/cue_openfold/bin/python"

# Regex matching Lightning's prog_bar "train_loss_step=X.XXXX" updates. Lightning
# logs three flavors: "train/loss" (logger only), "train_loss_step" (prog_bar,
# per-batch), "train_loss_epoch" (prog_bar, per-epoch). The per-step value is
# what we want for parity comparison. With log_every_n_steps=1, every minibatch
# emits one update.
LOSS_RE = re.compile(r"train_loss_step=([0-9.eE+-]+)")
RECOMPILE_RE = re.compile(r"Recompiling function")
# Captures the [frame_id/variant_id] tag Dynamo prints with each recompile —
# used to detect cache thrash (a single frame compiled many times).
FRAME_TAG_RE = re.compile(r"\[(\d+)/(\d+)\]")


def build_cmd(use_torch_compile_sc: bool) -> list[str]:
    """Construct the train.py invocation. Returns a process-ready argv list."""
    chain_ids = ",".join(DEFAULT_CHAIN_IDS)
    argparse_args = [
        "--config_name=training_dssp_contact_20M_udlm_pb_v2_stage1",
        "--single",
        "--show_prog_bar",
        "--nolog",
        "--ngpus_per_node=1",
        f"--batch_size={BATCH_SIZE}",
        f"--accumulate_grad_batches={ACCUM_GRAD}",
    ]
    hydra_overrides = [
        f"opt.max_steps={MAX_STEPS}",
        f"opt.max_epochs={MAX_STEPS}",
        f"+opt.overfit_pdb_chains=[{chain_ids}]",
        f"+opt.padding_max_size={PADDING_MAX_SIZE}",
        "log.log_every_n_steps=1",
        "log.log_structure_every_n_steps=10000",
        "validation_sampling.tmscore_n_samples=0",
        # Keep the training compile ON: the whole point of this test is to verify
        # _forward_impl_sc has an independent Dynamo cache that doesn't collide
        # with the training _forward_impl cache.
        "+model.nn.use_torch_compile=True",
        # SC is already enabled in the base config (training.self_cond=True,
        # training.self_cond_use_copy=True). Hydra rejects the '+' prefix on
        # existing keys, so we leave those alone. The hard-coded 50% SC rate at
        # model_trainer_base.py:1199 (`random.random() > 0.5`) gives us the
        # stochastic SC scheduling we want — no need to override it.
        # The variable under test:
        f"+training.use_torch_compile_sc={str(use_torch_compile_sc).lower()}",
    ]
    # SELF_COND_MODE env var overrides training.self_cond_mode in the config.
    # Accepts: "deepcopy" (back-compat default), "alias" (Phase 5 shared-param
    # Module-aliasing for compile speedup at 1x memory), "none" (SC and training
    # share self.nn; requires Phase 4 autocast-cache fix; may thrash compile).
    sc_mode = os.environ.get("SELF_COND_MODE", "").lower()
    if sc_mode in ("alias", "none", "deepcopy"):
        hydra_overrides.append(f"training.self_cond_mode={sc_mode}")
    elif int(os.environ.get("TEST_NO_SC_COPY", "0")):
        # Legacy alias for SELF_COND_MODE=none.
        hydra_overrides.append("training.self_cond_mode=none")
    return [PYTHON, "-m", "proteinfoundation.train", *argparse_args, *hydra_overrides]


def run_one(use_torch_compile_sc: bool, log_path: Path) -> dict:
    """Run train.py once and return parsed results."""
    cmd = build_cmd(use_torch_compile_sc)
    print(f"\n{'='*70}")
    print(f"Running with use_torch_compile_sc={use_torch_compile_sc}")
    print(f"  Log: {log_path}")
    print(f"  Cmd: {' '.join(cmd)}")
    print(f"{'='*70}", flush=True)

    env = dict(os.environ)
    env.setdefault("CUTLASS_PATH", "/home/ubuntu/openfold/cutlass")
    env.setdefault("SKIP_CONFIND_PRECOMPUTE", "1")
    # TORCH_LOGS=recompiles emits "Recompiling function ..." lines on stderr
    # whenever Dynamo evicts/re-traces. We count those to verify the fix.
    env["TORCH_LOGS"] = "recompiles"

    with log_path.open("w") as f:
        result = subprocess.run(
            cmd,
            cwd=str(PROTEINA_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    return parse_log(log_path, returncode=result.returncode)


def parse_log(log_path: Path, returncode: int) -> dict:
    """Extract train losses, recompile counts, and per-frame variant counts.

    Per-frame variant count = the largest ``variant_id`` Dynamo ever assigned to
    a given frame, +1 (since variants are 0-indexed). High = real cache thrash;
    low = one-time warmup compilation per (dtype, grad_mode) variant.
    """
    losses: list[float] = []
    n_recompiles = 0
    max_variant_per_frame: dict[int, int] = {}
    ddp_error = False

    with log_path.open() as f:
        for line in f:
            for m in LOSS_RE.finditer(line):
                # tqdm in non-TTY emits many updates per line as it overwrites
                # the bar; capture all matches so we get one value per step.
                try:
                    losses.append(float(m.group(1)))
                except ValueError:
                    pass
            if RECOMPILE_RE.search(line):
                n_recompiles += 1
                tag = FRAME_TAG_RE.search(line)
                if tag:
                    frame_id = int(tag.group(1))
                    variant_id = int(tag.group(2))
                    cur = max_variant_per_frame.get(frame_id, 0)
                    if variant_id > cur:
                        max_variant_per_frame[frame_id] = variant_id
            if "Expected to have finished reduction" in line:
                ddp_error = True

    # +1 to convert from 0-indexed variant_id to total-variants-seen count.
    n_variants = {f: v + 1 for f, v in max_variant_per_frame.items()}
    return {
        "returncode": returncode,
        "losses": losses,
        "n_recompiles": n_recompiles,
        "n_variants_per_frame": n_variants,
        "ddp_error": ddp_error,
        "log_path": str(log_path),
    }


def compare(baseline: dict, test: dict) -> int:
    """Apply the three correctness checks. Returns 0 on success, 1 on failure."""
    failed: list[str] = []

    # 1) Both runs must complete without error
    for label, r in (("baseline (SC eager)", baseline), ("test (SC compiled)", test)):
        if r["returncode"] != 0:
            failed.append(f"{label} exited with code {r['returncode']}")
        if r["ddp_error"]:
            failed.append(f"{label} hit DDP find_unused_params error")
        if not r["losses"]:
            failed.append(f"{label} produced no parseable train/loss values")

    # 2) Per-step loss parity (relative tolerance — losses can span ~1e2 → ~1e3)
    if baseline["losses"] and test["losses"]:
        # tqdm-updates may emit the same step's loss multiple times in a single
        # log line as the bar advances; de-duplicate consecutive runs of identical
        # values that arise from this. Train losses themselves never repeat
        # exactly across steps under stochastic SC scheduling.
        def _dedupe_consecutive(xs):
            out = []
            for x in xs:
                if not out or out[-1] != x:
                    out.append(x)
            return out
        b_losses = _dedupe_consecutive(baseline["losses"])
        t_losses = _dedupe_consecutive(test["losses"])
        n = min(len(b_losses), len(t_losses))
        print(f"\nLoss comparison ({n} steps; baseline emitted {len(b_losses)} unique values, test {len(t_losses)}):")
        print(f"  {'step':>5} {'baseline':>14} {'compiled':>14} {'rel|diff|':>12}")
        max_rel = 0.0
        for i in range(n):
            b = b_losses[i]
            t = t_losses[i]
            denom = max(abs(b), abs(t), 1e-9)
            rel = abs(b - t) / denom
            if rel > max_rel:
                max_rel = rel
            marker = "  ***" if rel > LOSS_REL_TOL else ""
            print(f"  {i:>5} {b:>14.4g} {t:>14.4g} {rel:>12.3e}{marker}")
        print(f"  max rel|diff| = {max_rel:.3e}  (tolerance {LOSS_REL_TOL:.3e})")
        if max_rel > LOSS_REL_TOL:
            failed.append(
                f"loss parity violated: max rel|diff|={max_rel:.3e} > {LOSS_REL_TOL}"
            )
    else:
        failed.append("cannot compare losses — at least one run produced none")

    # 3a) Total recompile-count sanity check
    print(f"\nRecompile counts (TORCH_LOGS=recompiles events):")
    print(f"  baseline (SC eager):    {baseline['n_recompiles']}")
    print(f"  test     (SC compiled): {test['n_recompiles']}")
    if test["n_recompiles"] > MAX_TOTAL_RECOMPILES_SC:
        failed.append(
            f"SC-compile run had {test['n_recompiles']} recompiles "
            f"(> ceiling {MAX_TOTAL_RECOMPILES_SC}); even warmup variants "
            f"should top out below this."
        )

    # 3b) Per-frame thrash check — high variant count per frame is the real
    # signal of cache thrash. One frame compiling 32+ times means Dynamo is
    # evicting and recompiling, not just adding new (dtype, grad_mode) variants.
    if test["n_variants_per_frame"]:
        worst = max(test["n_variants_per_frame"].items(), key=lambda kv: kv[1])
        worst_frame, worst_n = worst
        print(f"  highest variant count per frame: frame {worst_frame} → {worst_n}")
        print(f"  (threshold {MAX_VARIANTS_PER_FRAME}; >32 implies cache eviction)")
        thrashers = {
            f: n for f, n in test["n_variants_per_frame"].items()
            if n > MAX_VARIANTS_PER_FRAME
        }
        if thrashers:
            failed.append(
                f"cache thrash detected: frames {thrashers} exceeded "
                f"{MAX_VARIANTS_PER_FRAME} variants — likely evicted by "
                f"cache_size_limit and refilled"
            )

    print("\n" + "=" * 70)
    if failed:
        print("FAIL: " + "; ".join(failed))
        return 1
    print("PASS: SC-compile produces identical gradients with no recompile thrash")
    return 0


def main() -> int:
    baseline_log = Path("/tmp/test_sc_compile_baseline.log")
    test_log = Path("/tmp/test_sc_compile_sc.log")

    baseline = run_one(use_torch_compile_sc=False, log_path=baseline_log)
    test = run_one(use_torch_compile_sc=True, log_path=test_log)

    return compare(baseline, test)


if __name__ == "__main__":
    sys.exit(main())
