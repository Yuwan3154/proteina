"""What effect size will the t_beta mirror verdict actually be able to detect?

Stating this BEFORE the data arrives, so a null is not later reported as "no effect" when the design
could never have seen a modest one. All inputs are measured, none assumed:
  control  19/48 = 39.6% over 3 resolved rounds
  t_beta   2 resolved rounds expected inside the window = n=32 (16 chains/round, set by --n_dump 16)
"""

import math

P_CTRL, N_CTRL = 19 / 48, 48
N_TB = 32
REF_COIN, REF_RECIPE = 0.518, 0.016    # measured reference flip; rejection-resampling residual

se_ctrl = math.sqrt(P_CTRL * (1 - P_CTRL) / N_CTRL)
print(f"control: {P_CTRL*100:.1f}% +/- {se_ctrl*100:.1f}% (n={N_CTRL})")

print(f"\nt_beta at n={N_TB}, detectable difference vs the control at 2 SE:")
for p_tb in (0.50, 0.40, 0.30, 0.20, 0.10, REF_RECIPE):
    se_tb = math.sqrt(max(p_tb * (1 - p_tb), 1e-9) / N_TB)
    se_diff = math.sqrt(se_ctrl**2 + se_tb**2)
    diff = P_CTRL - p_tb
    verdict = "DETECTABLE" if diff > 2 * se_diff else "not distinguishable"
    print(f"  if t_beta = {p_tb*100:5.1f}%  diff {diff*100:+6.1f} pp, 2SE = {2*se_diff*100:5.1f} pp"
          f"  -> {verdict}")

se_tb50 = math.sqrt(0.25 / N_TB)
thresh = P_CTRL - 2 * math.sqrt(se_ctrl**2 + se_tb50**2)
print(f"\n=> Inside this window the verdict can only resolve t_beta below ~{thresh*100:.0f}%.")
print(f"   A modest improvement (e.g. 40% -> 30%) would NOT be distinguishable.")
print(f"   The rejection-resampling recipe's {REF_RECIPE*100:.1f}% residual WOULD be.")
print(f"   ⛔ A null at this n means UNDERPOWERED, not 'no effect'. Say so.")
print(f"   More power needs a 3rd resolved round (~03:15 EDT, just past the window) or more chains.")
