"""Measure what fraction of tri wall-clock goes to validation, from the diag log's own timestamps.

The cadence (val_check_interval_optim_steps 25, limit_val_batches 64) is shared with the ctrl/ot_fgw
arms, so changing it is the user's call -- this only supplies the number the decision needs, measured
rather than estimated. Each cycle is bracketed by a val_end marker; within a cycle, val time is the
span from the first validation_step enter to val_end, and train time is the remainder.
"""

import re
import sys

PAT = re.compile(r"\[diag rank=0 step=(\d+) t=([0-9.]+)\] (.+)")
events = []
with open(sys.argv[1]) as fh:
    for line in fh:
        m = PAT.search(line)
        if m:
            events.append((int(m.group(1)), float(m.group(2)), m.group(3)))

cycles, cur_val_start, last_end = [], None, None
for step, t, msg in events:
    if msg.startswith("validation_step enter batch_idx=0"):
        cur_val_start = t
    elif msg.startswith("val_end:"):
        if cur_val_start is not None:
            cycles.append((step, cur_val_start, t, last_end))
        last_end = t
        cur_val_start = None

print(f"{len(events)} diag events, {len(cycles)} completed validation rounds")
print(f"{'step':>6} {'val_s':>8} {'train_s':>9} {'cycle_s':>9} {'val_%':>7}")
tv = tt = 0.0
for step, vs, ve, prev_end in cycles:
    val_s = ve - vs
    if prev_end is None:
        print(f"{step:>6} {val_s:8.1f} {'-':>9} {'-':>9} {'-':>7}")
        continue
    cycle_s = ve - prev_end
    train_s = cycle_s - val_s
    tv += val_s
    tt += train_s
    print(f"{step:>6} {val_s:8.1f} {train_s:9.1f} {cycle_s:9.1f} {100*val_s/cycle_s:6.1f}%")
if tv + tt:
    print(f"\nTOTAL over {len(cycles)-1} full cycles: val {tv:.0f}s  train {tt:.0f}s  -> validation is {100*tv/(tv+tt):.1f}% of wall-clock")
    print(f"train throughput inside training windows: {25/(tt/(len(cycles)-1))*60:.1f} optim steps/min")
    print(f"observed end-to-end: {25/((tv+tt)/(len(cycles)-1))*60:.2f} optim steps/min")
