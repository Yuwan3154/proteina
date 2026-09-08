"""Is the TRAINED denoiser reflection-equivariant as a function?

This settles the question the mirror-rate sweeps cannot. A 48% mirror rate is consistent with two
very different situations:
  (a) the model learned a chiral prior but expresses it weakly / noisily, or
  (b) the model is O(3)-equivariant, i.e. f(Mx) = M f(x) IDENTICALLY, in which case its output hand
      is slaved to the input noise's hand and NO amount of right-handed training data can ever fix
      it, because the model never learns a prior at all -- it learns a MAP.

⭐ THE CONTROL IS THE POINT. Random SO(3) augmentation should have taught the model
f(Rx) = R f(x) for PROPER rotations. Measuring that residual calibrates the scale: it is the
floor set by numerics and by whatever equivariance the model genuinely failed to learn. The
reflection residual is only interpretable RELATIVE to it:
    reflection ~= rotation  -> O(3)-equivariant -> the mirror is architecturally FREE
    reflection >> rotation  -> the model DOES break reflection symmetry, look elsewhere

⛔ ref_pos is NOT transformed in either arm. That is deliberate and it matters: during training,
CentreRandomAugmentation rotates x_gt but leaves the reference conformer in its own fixed local
frame. So the equivariance the model was taught is exactly f(Rx; ref_pos) = R f(x; ref_pos), and
the honest reflection question is whether the FIXED chiral anchor in ref_pos/ref_feats is enough
to break the symmetry. If it were being used, the reflection residual would be large.
"""

import argparse
import sys

import torch

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh")

from proteinfoundation.datasets.atom_features import atom14_features
from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4, n_diffusion_samples=8,
)
SIGMAS = [2.0, 16.0, 40.0, 160.0]


def proper_rotation(seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))[None, :]
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    assert torch.linalg.det(q) > 0.999, "control transform must be a PROPER rotation"
    return q.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = ContactToCoordTrainer(model_cfg=MODEL_CFG)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    src = "ema" if "ema" in ck else "state_dict"
    if src == "ema":
        model.model.load_state_dict(ck["ema"]["params"], strict=True)
    else:
        model.load_state_dict(ck["state_dict"], strict=True)
    print(f"checkpoint : {args.ckpt}")
    print(f"  global_step = {ck.get('global_step')}   weights = {src}")
    net = model.model.to(dev).eval()

    B, L = 1, args.length
    aatype = torch.randint(0, 20, (B, L), device=dev)
    mask = torch.ones(B, L, device=dev)
    ref_feats, ref_pos, a2t, amask, uid = atom14_features(aatype, mask)
    # A banded contact map, the shape a real one has. Content is irrelevant to an equivariance
    # test -- the map is achiral either way -- but a degenerate all-zero input could sit in a dead
    # region of the network and understate every residual.
    idx = torch.arange(L, device=dev)
    contacts = ((idx[None, :] - idx[:, None]).abs() <= 3).float()[None].expand(B, -1, -1)

    R = proper_rotation(args.seed + 1).to(dev)
    M = torch.diag(torch.tensor([1.0, 1.0, -1.0], device=dev))
    assert torch.linalg.det(M) < 0, "test transform must be IMPROPER"

    with torch.no_grad():
        s, z, _ = net.encode(contacts, aatype, mask)

        def f(x, sig):
            sigma = torch.full((B,), sig, device=dev)
            return net.denoise(x, sigma, s, z, mask, ref_feats, ref_pos, a2t, amask, uid)

        print(f"\n{'sigma':>7} | {'rotation (control)':>19} | {'reflection':>11} | verdict")
        print("-" * 68)
        for sig in SIGMAS:
            # Pure noise at this level -- the state the SAMPLER actually starts from.
            x = torch.randn(B, L * 14, 3, device=dev) * sig * amask[..., None]
            y = f(x, sig)
            nrm = y.norm().clamp_min(1e-8)

            y_rot = f(x @ R.T, sig)
            res_rot = ((y_rot - y @ R.T).norm() / nrm).item()

            y_ref = f(x @ M.T, sig)
            res_ref = ((y_ref - y @ M.T).norm() / nrm).item()

            ratio = res_ref / max(res_rot, 1e-12)
            verdict = "O(3)-EQUIVARIANT" if ratio < 3 else ("breaks reflection" if ratio > 10 else "partial")
            print(f"{sig:7.1f} | {res_rot:19.6f} | {res_ref:11.6f} | {verdict} ({ratio:.1f}x)")

    print(
        "\nreflection ~ rotation  => the mirror is FREE: output hand is slaved to the input noise,\n"
        "                          and right-handed training data cannot teach a prior.\n"
        "reflection >> rotation => the model does use its chiral anchor; the 50/50 is elsewhere."
    )


if __name__ == "__main__":
    main()
