"""What sigma does t_beta 1.3,2.0 actually train at?

The recorded prediction was a median training sigma of ~184, but `train/sigma` on the live
c2c_cb8_tbeta run reads a median of 476. Before that gap gets explained away, measure it: the two
numbers may simply be different statistics of the same distribution, since a per-step MEAN over a
heavy-tailed draw distribution sits well above the median DRAW.

Also reports the fraction of draws in the regime that matters for the chirality argument: the model
only has to GENERATE a hand (rather than copy one out of the noised input) where c_skip is small,
i.e. sigma >> SIGMA_DATA.
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_tri")

from proteinfoundation.nn.af3_diffusion import sample_noise_level, sample_noise_level_beta

B, N, REPS = 8, 48, 4000        # B x N matches the live run: 8 structures x 48 diffusion samples
SIGMA_DATA = 16.0

torch.manual_seed(0)


def summarise(label, draw_fn):
    per_step_means, all_draws = [], []
    for _ in range(REPS):
        s = draw_fn((B, N)).double().numpy()
        all_draws.append(s.ravel())
        per_step_means.append(s.mean())
    d = np.concatenate(all_draws)
    m = np.array(per_step_means)
    # c_skip = sigma_data^2 / (sigma^2 + sigma_data^2); small c_skip = must GENERATE, not copy
    c_skip = SIGMA_DATA**2 / (d**2 + SIGMA_DATA**2)
    print(f"\n{label}")
    print(f"  median DRAW            {np.median(d):10.2f}")
    print(f"  median per-step MEAN   {np.median(m):10.2f}   <- what train/sigma logs")
    print(f"  mean DRAW              {np.mean(d):10.2f}")
    print(f"  draws > 100            {100*(d > 100).mean():9.2f}%")
    print(f"  c_skip < 0.01 (GENERATE regime) {100*(c_skip < 0.01).mean():8.2f}%")


summarise("AF3 lognormal (t_beta OFF -- the c2c_cb8 control)",
          lambda sh: sample_noise_level(sh, torch.device("cpu")))
summarise("t_beta 1.3,2.0 (c2c_cb8_tbeta)",
          lambda sh: sample_noise_level_beta(sh, torch.device("cpu"), p1=1.3, p2=2.0))
print(f"\nlive c2c_cb8_tbeta logs train/sigma median 476.10; c2c_cb8 logged ~13-15")
