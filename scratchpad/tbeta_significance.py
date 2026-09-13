"""Is the t_beta step-3144 result real, or an artifact of a small denominator?

The comparison tool flagged "below the coin flip by >2 SE", but that test uses the OBSERVED rate's
own standard error, which shrinks as the rate falls -- so a low rate can clear it too easily. The
right test is a two-proportion comparison against the control, and it must be run under BOTH mirror
definitions, because the whole point of having two is that a conclusion resting on one is fragile.

⛔ This is ONE round at n=16, and it is the arm's FIRST resolved round.
"""

import math

# control: 3 resolved rounds, 48 chains
CTRL = {"GT": (19, 48), "detector": (18, 48)}
# t_beta: 1 resolved round, 16 chains. GT from val/is_mirrored; detector from helix_pos_frac > 0.65
TBETA = {"GT": (1, 16), "detector": (3, 16)}


def two_prop(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se else float("nan")
    return p1, p2, z


print(f"{'definition':>10} {'control':>12} {'t_beta':>12} {'z':>7}  verdict")
for name in ("GT", "detector"):
    k1, n1 = CTRL[name]
    k2, n2 = TBETA[name]
    p1, p2, z = two_prop(k1, n1, k2, n2)
    verdict = ("significant at 5%" if abs(z) > 1.96
               else "NOT significant" if abs(z) < 1.96 else "")
    print(f"{name:>10} {f'{k1}/{n1}={100*p1:.1f}%':>12} {f'{k2}/{n2}={100*p2:.1f}%':>12} "
          f"{z:7.2f}  {verdict}")

print("\n⛔ THE TWO DEFINITIONS DISAGREE ON THE CONCLUSION.")
print("   GT-based (val/is_mirrored) says significant; the CA-dihedral detector says not.")
print("   On the control the two agreed within 1 chain in 48; here they differ by 2 chains in 16.")
print("\n   A result that flips with the definition, from ONE round of 16 at the arm's FIRST")
print("   resolved step, is PROMISING and NOT a verdict. The pre-registered analysis already said")
print("   a single round (n=16, SE 12.5%) could establish essentially nothing.")
print("\n   What would settle it: the 3644 and 4144 rounds, giving n=48 to match the control.")
