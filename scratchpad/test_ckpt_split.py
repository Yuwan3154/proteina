"""Verify the two-callback checkpoint split in train_c2c.py is configured as intended.

Why this test exists: a single ModelCheckpoint carrying BOTH `every_n_train_steps` and `monitor`
ranks top-k on a STALE `val/loss` (Lightning's `_monitor_candidates` is a deepcopy of
`trainer.callback_metrics`, which retains the last logged value between validations). On
c2c_cb8_tbeta that turned "best 3" into "most recent 3" -- measured as 14,660 / 14,670 / 14,680,
three consecutive 10-step saves -- and it is how the step-9,750 reference was evicted mid-series.

⛔ These assertions are about CONFIGURATION, not about a training run. They confirm Lightning
accepts the split and that the trigger flags resolve so that top-k fires at validation end (fresh
metric) while last.ckpt keeps its every-N-steps cadence. Run on CPU; no GPU, no data.
"""

import sys

from lightning.pytorch.callbacks import ModelCheckpoint

CKPT_EVERY = 10
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


print("=== 1. production config (non-overfit): the split Lightning must accept ===")
keep = dict(monitor="val/loss", mode="min", save_top_k=3)
last_cb = ModelCheckpoint(dirpath="/tmp/x", monitor=None, save_top_k=0,
                          save_last=True, every_n_train_steps=CKPT_EVERY,
                          enable_version_counter=False)
best_cb = ModelCheckpoint(dirpath="/tmp/x", **keep, save_last=False,
                          enable_version_counter=False)

check("last_cb writes ONLY last.ckpt (save_top_k=0 short-circuits _save_topk_checkpoint)",
      last_cb.save_top_k == 0 and last_cb.save_last is True)
check("last_cb keeps the every-N-steps resume cadence",
      last_cb._every_n_train_steps == CKPT_EVERY, f"={last_cb._every_n_train_steps}")
check("last_cb does NOT also trigger per-epoch (mutual exclusion)",
      last_cb._every_n_epochs == 0, f"_every_n_epochs={last_cb._every_n_epochs}")

check("best_cb ranks on val/loss, min, top-3",
      best_cb.monitor == "val/loss" and best_cb.mode == "min" and best_cb.save_top_k == 3)
check("best_cb has NO step trigger -> it cannot rank on a stale metric",
      best_cb._every_n_train_steps == 0, f"={best_cb._every_n_train_steps}")
check("best_cb fires once per validation (_every_n_epochs==1 satisfies the on_validation_end gate)",
      best_cb._every_n_epochs == 1, f"={best_cb._every_n_epochs}")
check("best_cb does not write last.ckpt (only last_cb owns it)", best_cb.save_last is False)

print("\n=== 2. _should_save_on_train_epoch_end must be False so top-k runs at VALIDATION end ===")
# Lightning returns `trainer.val_check_interval == 1.0`; the real trainer uses val_every*accum.


class _FakeTrainer:
    def __init__(self, val_check_interval, num_val_batches=64):
        self.check_val_every_n_epoch = 1
        self.val_check_interval = val_check_interval
        self.num_val_batches = num_val_batches


real_interval = 500 * 8  # args.val_every * args.accum, as train_c2c.py sets it
check("with the real val_check_interval, saving happens at validation end, not train-epoch end",
      best_cb._should_save_on_train_epoch_end(_FakeTrainer(real_interval)) is False,
      f"val_check_interval={real_interval}")
check("control: the ONE case that would move it to train-epoch end is val_check_interval==1.0",
      best_cb._should_save_on_train_epoch_end(_FakeTrainer(1.0)) is True)

print("\n=== 3. overfit config must still keep EVERY validation checkpoint ===")
keep_of = dict(monitor=None, save_top_k=-1, filename="step{step:07d}",
               auto_insert_metric_name=False)
of_last = ModelCheckpoint(dirpath="/tmp/x", monitor=None, save_top_k=0, save_last=True,
                          every_n_train_steps=250, enable_version_counter=False)
of_best = ModelCheckpoint(dirpath="/tmp/x", **keep_of, save_last=False,
                          enable_version_counter=False)
check("overfit keeps every checkpoint (save_top_k=-1), so no trajectory is deleted",
      of_best.save_top_k == -1 and of_best.monitor is None)
check("overfit resume anchor still bounded", of_last.save_top_k == 0 and of_last.save_last is True)

print("\n=== 4. KNOWN-BAD CONTROL: the OLD single-callback config is the thing we removed ===")
# It is NOT rejected by Lightning -- that is precisely why the bug was silent for days.
old_cb = ModelCheckpoint(dirpath="/tmp/x", monitor="val/loss", mode="min", save_top_k=3,
                         save_last=True, every_n_train_steps=CKPT_EVERY,
                         enable_version_counter=False)
check("old config sets BOTH monitor and a step trigger -> ranks top-k on a stale metric",
      old_cb.monitor is not None and old_cb._every_n_train_steps == CKPT_EVERY)
check("and Lightning accepts it silently (no exception) -- why this needed a code fix, not a config error",
      True, "constructed without raising")

print("\n=== RESULT ===")
if FAILS:
    for f in FAILS:
        print("  FAILED:", f)
    sys.exit(1)
print("  ALL CHECKS PASSED")
