"""The t_beta verdict at n=32, under both mirror definitions.

At step 3788 the native-free detector calls 7 of 16 chains mirrored while the GT ratio test calls 1.
At 3144 it was 3 vs 1. The two definitions agreed within 1 chain in 48 on the CONTROL, so this
divergence is about the arm, not the instruments in general -- and the per-chain confound test
already showed the GT test loses sensitivity exactly where t_beta's structures sit.

Run both comparisons and let them speak.
"""

import math

CTRL = {"GT": (19, 48), "detector": (18, 48)}
TBETA = {"GT": (2, 32), "detector": (10, 32)}     # rounds 3144 + 3788


def two_prop(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return p1, p2, (p1 - p2) / se if se else float("nan")


print(f"{'definition':>10} {'control':>14} {'t_beta':>14} {'z':>7}  verdict")
for name in ("GT", "detector"):
    p1, p2, z = two_prop(*CTRL[name], *TBETA[name])
    v = "significant" if abs(z) > 1.96 else "NOT significant"
    print(f"{name:>10} {f'{CTRL[name][0]}/{CTRL[name][1]}={100*p1:.1f}%':>14} "
          f"{f'{TBETA[name][0]}/{TBETA[name][1]}={100*p2:.1f}%':>14} {z:7.2f}  {v}")

print("\n⛔ The divergence has WIDENED with more data, not closed:")
print("   at n=16  GT 6.2% vs detector 18.8%")
print("   at n=32  GT 6.2% vs detector 31.2%")
print("\n   On the CONTROL the two agreed within 1 chain in 48. Here they differ by 8 chains in 32.")
print("   The per-chain test already showed WHY: the GT ratio test misses mirrored chains as")
print("   structure quality degrades, and t_beta's structures sit squarely in that range.")
print("\n⭐ READING: by the instrument that does NOT depend on structure quality, t_beta sits at")
print("   31.2% against the control's 37.5% -- a 6.3 pp difference, z = 0.57, NOTHING.")
print("   ⛔ High-noise training has NOT reduced the mirror rate. The apparent effect is an")
print("      artifact of the GT test's quality sensitivity.")
