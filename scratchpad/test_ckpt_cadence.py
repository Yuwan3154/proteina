"""The checkpoint-cadence cap must bound resume loss WITHOUT reviving the overfit quota kill.

⛔⛔ Why this test exists. The cap shortens the checkpoint interval for production runs. In OVERFIT
mode the keep policy is save_top_k=-1 (keep EVERY checkpoint, deliberately, so the mirror-rate
trajectory survives), so applying the same cap there would write ~100 files / ~320 GB over a
2000-step run. A 55 GB version of exactly that already killed the first overfit run. The overfit
exclusion is a default-OFF branch: nothing else in the suite exercises it, and its failure mode is a
dead run hours later, not an exception here.

⛔ The cap VALUE is read out of train_c2c.py rather than retyped, so this test cannot silently pass
against a stale constant after someone edits the source.

Run: python scratchpad/test_ckpt_cadence.py
"""

import ast
import os
import sys

from lightning.pytorch.callbacks import ModelCheckpoint

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "train_c2c.py")

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# ── A. pin the constant and the guard to the real source ─────────────────────────────────────
src = open(SRC).read()
tree = ast.parse(src)
cap = None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CKPT_INTERVAL_CAP" for t in node.targets):
        cap = ast.literal_eval(node.value)
print(f"== A. source constants ==\n  CKPT_INTERVAL_CAP = {cap}")
check("CKPT_INTERVAL_CAP is defined and positive", isinstance(cap, int) and cap > 0)
check("the cap is guarded by `not args.overfit`", "not args.overfit and ckpt_every >" in src)
# the overfit keep-all policy now lives in c2c_ckpt_callbacks.build_ckpt_callbacks (2026-09-22)
cb_src = open(os.path.join(HERE, "c2c_ckpt_callbacks.py")).read()
check("overfit still keeps every checkpoint (save_top_k=-1)", "save_top_k=-1" in cb_src)


def cadence(ckpt_every, val_every, overfit):
    """Mirror of the decision in train_c2c.py, driven by the constant lifted from it."""
    every = ckpt_every or val_every
    if not overfit and every > cap:
        every = cap
    return every


# ── B. the decision table ────────────────────────────────────────────────────────────────────
print("\n== B. interval selection ==")
check("production, ckpt_every=0 -> falls back to val_every, then capped",
      cadence(0, 200, False) == cap, f"{cadence(0, 200, False)}")
check("production, explicit 200 -> capped", cadence(200, 500, False) == cap)
check("production, already tighter than the cap is LEFT ALONE",
      cadence(5, 500, False) == 5, "a caller asking for 5 must not be widened to 20")
check("OVERFIT, ckpt_every=0 -> val_every, NOT capped",
      cadence(0, 200, True) == 200, f"{cadence(0, 200, True)} (capping here = the 320 GB quota kill)")
check("OVERFIT, explicit 500 -> NOT capped", cadence(500, 200, True) == 500)

# ── C. through the real consumer ─────────────────────────────────────────────────────────────
# ⛔ Asserting on our own arithmetic proves nothing about what Lightning does with it. Build the
# actual callback both ways and read back what it will act on.
print("\n== C. through the real ModelCheckpoint ==")
prod = ModelCheckpoint(dirpath="/tmp/_cadence_prod", monitor="val/loss", mode="min", save_top_k=3,
                       save_last=True, every_n_train_steps=cadence(0, 200, False),
                       enable_version_counter=False)
over = ModelCheckpoint(dirpath="/tmp/_cadence_over", monitor=None, save_top_k=-1,
                       filename="step{step:07d}", auto_insert_metric_name=False,
                       save_last=True, every_n_train_steps=cadence(0, 200, True),
                       enable_version_counter=False)
check("production callback fires on the capped interval", prod._every_n_train_steps == cap,
      f"{prod._every_n_train_steps}")
check("overfit callback keeps the wide interval", over._every_n_train_steps == 200,
      f"{over._every_n_train_steps}")
check("production keeps a bounded number of files", prod.save_top_k == 3)
check("overfit still keeps everything", over.save_top_k == -1)
check("both write last.ckpt", bool(prod.save_last) and bool(over.save_last))

# ⭐ The whole point of the change: last.ckpt is the resume anchor, and it must be written on every
# interval hit even when val/loss is stale between validations. Lightning's _save_last_checkpoint
# returns early ONLY on `not self.save_last` and never reads self.monitor -- verify that in the
# installed source rather than trusting the docs.
import inspect
body = inspect.getsource(ModelCheckpoint._save_last_checkpoint)
check("Lightning's _save_last_checkpoint ignores `monitor`", "self.monitor" not in body,
      "so last.ckpt is saved between validations too")
check("_save_last_checkpoint early-returns only on save_last", "if not self.save_last:" in body)

print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
sys.exit(0 if ok else 1)
