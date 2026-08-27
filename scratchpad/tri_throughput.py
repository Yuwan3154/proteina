import datetime
import re
import sys

PAT = re.compile(r"^([0-9-]{10} [0-9:]{8}).*batch_idx=(\d+)\s*$")

rows = []
for line in open(sys.argv[1]):
    m = PAT.match(line)
    if m:
        rows.append((datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), int(m.group(2))))

print("%6s %6s %7s %8s %8s" % ("from", "to", "dbatch", "secs", "s/mb"))
clean = []
for (t0, b0), (t1, b1) in zip(rows, rows[1:]):
    d = (t1 - t0).total_seconds()
    db = b1 - b0
    if db <= 0:
        continue
    rate = d / db
    print("%6d %6d %7d %8.0f %8.2f" % (b0, b1, db, d, rate))
    clean.append(rate)

clean.sort()
med = clean[len(clean) // 2]
print("\nwindows: %d   median s/minibatch: %.2f   min: %.2f   max: %.2f" % (len(clean), med, clean[0], clean[-1]))
print("total span: %.0f s over %d minibatches -> %.2f s/mb (incl. validation)"
      % ((rows[-1][0] - rows[0][0]).total_seconds(), rows[-1][1] - rows[0][1],
         (rows[-1][0] - rows[0][0]).total_seconds() / (rows[-1][1] - rows[0][1])))

STEPS_PER_EPOCH = 1802
MAX_EPOCHS = 500
for label, rate in (("median window (train only)", med), ("observed incl. validation",
                    (rows[-1][0] - rows[0][0]).total_seconds() / (rows[-1][1] - rows[0][1]))):
    ep_h = rate * STEPS_PER_EPOCH / 3600.0
    print("%-28s %6.2f s/mb -> %6.2f h/epoch -> %8.1f h = %6.1f days for %d epochs"
          % (label, rate, ep_h, ep_h * MAX_EPOCHS, ep_h * MAX_EPOCHS / 24.0, MAX_EPOCHS))
