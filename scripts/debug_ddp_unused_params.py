"""DDP find_unused_params investigation harness.

Runs `proteinfoundation.train` with diagnostic Lightning callbacks injected via
a monkey-patch on ``lightning.Trainer``. No source edits to train.py or the
trainer code are required.

The diagnostics are:

D1 — GradAudit         : after each backward, print which `nn.named_parameters()`
                          have `grad is None` or `grad.abs().sum() == 0`. Reveals
                          which conditionally-unused param the DDP check is
                          flagging.
D2 — ForwardCount      : wrap `nn.forward` and count invocations per minibatch.
                          Pinpoints whether nn is called once or twice per iter.
D3 — DDPStateAudit     : at the failing minibatch boundary, dump
                          `DistributedDataParallel`'s public state (find_unused,
                          static_graph, _has_rebuilt_buckets, etc.).
D4 — ModuleTreeAudit   : at trainer setup, print the LightningModule's named
                          submodules and which ones have requires_grad=True
                          params. Used by H2 (module-tree shape) tests.

Run modes (selected via $PROTEINA_DDP_DEBUG):

  D1     : grad audit
  D2     : forward counter
  D3     : DDP state dump
  D4     : module tree audit
  ALL    : all four

This harness invokes `python -m proteinfoundation.train` as a subprocess after
exporting the env var so the in-process monkey-patch is performed by a tiny
wrapper module (`scripts/_ddp_debug_runner.py`) that runs train.main() with
the patched Trainer.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

PROTEINA_ROOT = Path("/home/ubuntu/proteina")
PY = "/home/ubuntu/miniforge3/envs/cue_openfold/bin/python"

CHAINS = "1c9p_B,1fka_R,1fse_F,1g2c_A,1guu_A,1ixs_A,1jcd_A,1kfm_A"

LOSS_RE = re.compile(r"train_loss_step=([0-9.eE+-]+)")
FRAME_RE = re.compile(r"\[(\d+)/(\d+)\]")


def base_overrides(extra: list[str], compile_train: bool = True) -> list[str]:
    """Standard overfit setup used by every hypothesis test.

    Use ``compile_train=False`` to disable training-pass torch.compile (useful
    for testing whether compile is implicated in a failure mode).
    """
    return [
        f"+opt.overfit_pdb_chains=[{CHAINS}]",
        "+opt.padding_max_size=64",
        "opt.max_steps=12", "opt.max_epochs=12",
        "log.log_every_n_steps=1",
        "log.log_structure_every_n_steps=10000",
        "validation_sampling.tmscore_n_samples=0",
        f"+model.nn.use_torch_compile={str(compile_train).lower()}",
    ] + extra


def run(label: str, overrides: list[str], debug_modes: str = "") -> dict:
    """Run train.py via the debug runner; return a dict of parsed results."""
    log_path = Path(f"/tmp/ddp_debug_{label}.log")
    runner = str(PROTEINA_ROOT / "scripts" / "_ddp_debug_runner.py")
    cmd = [
        PY, runner,
        "--config_name=training_dssp_contact_20M_udlm_pb_v2_stage1",
        "--single", "--show_prog_bar", "--nolog",
        "--ngpus_per_node=1",
        "--batch_size=4",
        "--accumulate_grad_batches=2",
    ] + overrides

    env = dict(os.environ)
    env.setdefault("CUTLASS_PATH", "/home/ubuntu/openfold/cutlass")
    env.setdefault("SKIP_CONFIND_PRECOMPUTE", "1")
    env["TORCH_LOGS"] = "recompiles"
    if debug_modes:
        env["PROTEINA_DDP_DEBUG"] = debug_modes
    # If a flag is set, also force find_unused_parameters=True on the trainer's
    # DDPStrategy so the run survives past iter 8 and the audit collects data.
    if any(x in (debug_modes or "") for x in ("D1",)):
        env["PROTEINA_DDP_FIND_UNUSED"] = "1"
    # H1c: disable torch.utils.checkpoint.checkpoint by replacing with no-op.
    if "DISABLE_CKPT" in (debug_modes or ""):
        env["PROTEINA_DISABLE_CKPT"] = "1"

    print(f"\n{'='*70}\n[{label}] {' '.join(cmd)}\n{'='*70}", flush=True)
    with log_path.open("w") as f:
        r = subprocess.run(cmd, cwd=str(PROTEINA_ROOT), env=env,
                           stdout=f, stderr=subprocess.STDOUT)
    return parse(log_path, r.returncode)


def parse(log_path: Path, returncode: int) -> dict:
    losses = []
    n_recompiles = 0
    max_variant = {}
    ddp_unused_err = False
    grad_audit = []
    fwd_counts = []
    ddp_state = {}
    module_tree = []

    text = log_path.read_text()
    for line in text.splitlines():
        for m in LOSS_RE.finditer(line):
            try: losses.append(float(m.group(1)))
            except ValueError: pass
        if "Recompiling function" in line:
            n_recompiles += 1
            tag = FRAME_RE.search(line)
            if tag:
                fid = int(tag.group(1)); vid = int(tag.group(2))
                max_variant[fid] = max(max_variant.get(fid, 0), vid)
        if "parameters that were not used in producing the loss" in line:
            ddp_unused_err = True
        if "[grad-audit" in line:
            grad_audit.append(line.strip())
        if "[fwd-count" in line:
            fwd_counts.append(line.strip())
        if "[ddp-state" in line:
            ddp_state.setdefault("lines", []).append(line.strip())
        if "[mod-tree" in line:
            module_tree.append(line.strip())

    # dedupe consecutive losses (tqdm repeats them)
    deduped = []
    for x in losses:
        if not deduped or deduped[-1] != x:
            deduped.append(x)

    return {
        "returncode": returncode,
        "losses": deduped,
        "n_recompiles": n_recompiles,
        "max_variant_per_frame": max_variant,
        "ddp_unused_err": ddp_unused_err,
        "grad_audit": grad_audit,
        "fwd_counts": fwd_counts,
        "ddp_state": ddp_state,
        "module_tree": module_tree,
        "log_path": str(log_path),
    }


def summarize(label: str, r: dict) -> None:
    print(f"\n----- {label} -----")
    print(f"  returncode={r['returncode']}, DDP unused error={r['ddp_unused_err']}, "
          f"losses={len(r['losses'])}, recompiles={r['n_recompiles']}, "
          f"max-variants/frame={max(r['max_variant_per_frame'].values()) + 1 if r['max_variant_per_frame'] else 0}")
    if r["fwd_counts"]:
        print(f"  Forward counts (last 5): {r['fwd_counts'][-5:]}")
    if r["grad_audit"]:
        # unique param names that ever appeared
        names = set()
        for line in r["grad_audit"]:
            m = re.search(r"(None|zero) grad: (\S+)", line)
            if m: names.add(m.group(2))
        print(f"  Audited unused params ({len(names)} unique): {sorted(names)[:10]}{'...' if len(names) > 10 else ''}")
    if r["ddp_state"].get("lines"):
        for l in r["ddp_state"]["lines"]:
            print(f"  {l}")
    print(f"  log: {r['log_path']}")


# =====================================================================
# Hypothesis sweep
# =====================================================================

def main():
    if len(sys.argv) > 1:
        only = set(sys.argv[1:])
    else:
        only = None

    def should_run(name):
        return only is None or name in only

    # H1: no-deepcopy + no SC. Decisive: if DDP error vanishes, double-forward is the cause.
    if should_run("H1"):
        r = run("H1_no-sc_no-deepcopy",
                base_overrides(["training.self_cond_use_copy=False",
                                "training.self_cond=False"]))
        summarize("H1: no-deepcopy + no SC", r)

    # H1b: no-deepcopy + WITH SC + WITHOUT torch.compile. If the DDP error
    # PERSISTS, then compile is not the cause — the autograd-graph disconnection
    # is something inherent in calling nn() twice with grad-mode changes.
    # If the error GOES AWAY, compile is involved (e.g., Inductor pruning the
    # projection layers after first trace with zero-init dssp_sc inputs).
    if should_run("H1b"):
        r = run("H1b_no-compile_no-deepcopy",
                base_overrides(["training.self_cond_use_copy=False"],
                               compile_train=False),
                debug_modes="D1")
        summarize("H1b: no-deepcopy + SC + NO COMPILE", r)

    # H1c: no-deepcopy + SC, but disable torch.utils.checkpoint (replace it
    # with a no-op passthrough). The runner reads $PROTEINA_DISABLE_CKPT=1.
    # If this run completes cleanly, the checkpoint mechanism is the proximate
    # cause of the autograd state corruption (likely interacting with the
    # double-forward of nn via use_reentrant=False's saved-tensor hooks).
    if should_run("H1c"):
        r = run("H1c_no-checkpoint_no-deepcopy",
                base_overrides(["training.self_cond_use_copy=False"]),
                debug_modes="DISABLE_CKPT")
        summarize("H1c: no-deepcopy + SC + checkpoint replaced with no-op", r)

    # H1d: same as H1b (eager mode) but with checkpoint also disabled.
    # Isolates whether the eager-mode CheckpointError is the ONLY issue
    # when compile is off. If H1d passes, the eager-mode root fix is clear:
    # switch checkpoint(use_reentrant=False) → use_reentrant=True, or
    # disable checkpoint where the model is small enough to fit.
    if should_run("H1d"):
        r = run("H1d_eager_no-ckpt_no-deepcopy",
                base_overrides(["training.self_cond_use_copy=False"],
                               compile_train=False),
                debug_modes="DISABLE_CKPT")
        summarize("H1d: eager + no-deepcopy + SC + no checkpoint", r)

    # D2: forward-count instrumentation in BOTH deepcopy modes
    if should_run("D2"):
        for label, override in (("D2_deepcopy", "training.self_cond_use_copy=True"),
                                 ("D2_no-deepcopy", "training.self_cond_use_copy=False")):
            r = run(label, base_overrides([override]), debug_modes="D2")
            summarize(label, r)

    # D1: grad audit — needs find_unused_parameters=True so training survives
    # past iter 8. The "D1" debug mode auto-sets PROTEINA_DDP_FIND_UNUSED=1
    # which monkey-patches DDPStrategy(find_unused_parameters=True) without
    # changing the strategy string (and therefore without changing launcher
    # behavior which would break the single-process cluster sampler).
    # Run BOTH modes so we can directly compare which params are unused in
    # each. If both have the same None-grad set, then no-deepcopy doesn't
    # introduce new unused params — deepcopy was just masking the DDP check.
    # If they differ, no-deepcopy causes additional params to be skipped.
    if should_run("D1"):
        for label, override in (("D1_deepcopy", "training.self_cond_use_copy=True"),
                                 ("D1_no-deepcopy", "training.self_cond_use_copy=False")):
            r = run(label, base_overrides([override]), debug_modes="D1")
            summarize(label, r)

    # H4: disable EMA
    if should_run("H4"):
        r = run("H4_no-ema_no-deepcopy",
                base_overrides(["training.self_cond_use_copy=False",
                                "+ema.every_n_steps=10000"]))
        summarize("H4: no-deepcopy + EMA effectively disabled", r)

    # H5: switch optimizer to AdamW
    if should_run("H5"):
        r = run("H5_adamw_no-deepcopy",
                base_overrides(["training.self_cond_use_copy=False",
                                "opt.optimizer=adamw"]))
        summarize("H5: no-deepcopy + AdamW", r)

    # H2: dummy frozen sibling module — handled by debug_modes="H2"
    if should_run("H2"):
        r = run("H2_dummy-sibling_no-deepcopy",
                base_overrides(["training.self_cond_use_copy=False"]),
                debug_modes="H2")
        summarize("H2: no-deepcopy + dummy frozen sibling on Proteina", r)


if __name__ == "__main__":
    main()
