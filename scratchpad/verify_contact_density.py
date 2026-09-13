"""DOUBLE-CHECK the contact density that every "N x random" claim about tri rests on.

A random predictor's precision@L equals the contact DENSITY, so that number sets the multiplier on
every generative-quality statement I have made. It was originally derived one way only: at round 1
recall was ~0, so accuracy ~ the true-negative fraction, giving density ~ 1 - 0.9747 = 0.0253. That
assumes the thresholded prediction was ENTIRELY negative, which is an approximation.

Derive it independently from the algebra instead, at rounds where recall is clearly non-zero:

    F1 = 2PR/(P+R)                      -> P from the logged F1 and recall
    TP = R * d * N                       (d = density, N = total pairs)
    FP = TP * (1-P)/P
    TN = N - dN - FP
    accuracy = (TP + TN)/N = 1 - d*(1 + (1-P)/P * R - R)   -> solve for d

Two rounds, far apart in training, should agree if the derivation is sound.
"""

# (label, accuracy, recall, f1) straight from validation_sampling, true optimizer steps
ROUNDS = [
    ("step 3002", 0.9579, 0.0444, 0.0461),
    ("step 10009", 0.9659, 0.1896, 0.2085),
]
ORIGINAL = 0.0253   # the round-1 approximation in use so far


def precision_from_f1(f1, r):
    # F1 = 2PR/(P+R)  ->  P (F1 - 2R) = -F1 R  ->  P = F1*R / (2R - F1)
    return f1 * r / (2 * r - f1)


print(f"{'round':>12} {'precision':>10} {'density':>9}")
ds = []
for label, acc, rec, f1 in ROUNDS:
    p = precision_from_f1(f1, rec)
    # accuracy = 1 - d * (1 + (1-P)/P * R - R)
    coeff = 1 + (1 - p) / p * rec - rec
    d = (1 - acc) / coeff
    ds.append(d)
    print(f"{label:>12} {p:10.4f} {d:9.4f}")

lo, hi = min(ds), max(ds)
print(f"\nindependent estimates: {lo:.4f} .. {hi:.4f}")
print(f"originally used:       {ORIGINAL:.4f}")
if ORIGINAL > hi:
    print(f"=> the value in use is HIGH by {100*(ORIGINAL/hi - 1):.0f}%, so every 'N x random'")
    print("   multiplier I have quoted is CONSERVATIVE (understated), not inflated.")
elif ORIGINAL < lo:
    print(f"=> the value in use is LOW by {100*(1 - ORIGINAL/lo):.0f}%, so the multipliers are OVERSTATED.")
else:
    print("=> the value in use sits inside the independent range; multipliers stand.")

print(f"\nprecision@L 0.2707 at step 10009 is "
      f"{0.2707/ORIGINAL:.1f}x random at the value in use, "
      f"{0.2707/hi:.1f}-{0.2707/lo:.1f}x at the independent range.")
