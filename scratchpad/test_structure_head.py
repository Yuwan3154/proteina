"""Gate for the AF3 structure head on the contact-map trunk.

Checks the things that would fail silently in a long run: that the head is inert when disabled, that
the detach actually isolates the trunk's gradients, that the EDM preconditioning matches AF3's
algebra at the limits, and that the rigid alignment is a rotation rather than a reflection.
"""

import sys

import torch

from proteinfoundation.datasets.sse_topology import N_PAIR_FEATURES
from proteinfoundation.nn.af3_diffusion import (
    FULL_INFERENCE_STEPS,
    MINI_ROLLOUT_STEPS,
    SIGMA_DATA,
    diffusion_loss,
    noise_schedule,
    sample_noise_level,
    smooth_lddt,
    weighted_rigid_align,
)
from proteinfoundation.nn.contact_map_tri import ContactMapTriSiT

B, L, T, DIM = 2, 16, 5, 64

BASE = dict(
    pair_dim=DIM, tri_hidden=DIM, n_blocks=2, transition_n=2, dim_cond=32, max_rel_pos=8,
    topology_cond=True, max_topology_he_len=T, topology_vocab_size=T + 1, n_residue_types=22,
    pair_ref_features="both", contact_map_mode=True, contact_map_input_dim=1, non_contact_value=0,
)
SH = dict(enabled=True, mode="diffusion", c_s=32, c_z=16,
          diffusion=dict(c_token=32, n_blocks=2, n_heads=4, c_noise_embedding=16))


def make_batch(seed=0, with_gt=True):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(B, L)
    mask[1, L - 3:] = 0.0
    he = torch.randint(1, T + 1, (B, T), generator=g)
    he[1, T - 2:] = 0
    b = {
        "contact_map_t": torch.rand(B, L, L, generator=g),
        "contact_map_sc": torch.rand(B, L, L, generator=g),
        "residue_type": torch.randint(0, 21, (B, L), generator=g),
        "mask": mask,
        "t": torch.rand(B, generator=g),
        "topology_he_tokens": he,
        "topology_he_pos_raw": torch.arange(T).float()[None].repeat(B, 1),
        "topology_he_feat": torch.rand(B, T, T, N_PAIR_FEATURES, generator=g),
    }
    if with_gt:
        b["x_1_ca"] = torch.randn(B, L, 3, generator=g) * 5.0
    return b


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return bool(ok)


def main():
    torch.manual_seed(0)
    r = []
    batch = make_batch()

    print("1. head disabled => bit-identical trunk")
    torch.manual_seed(0)
    base = ContactMapTriSiT(**BASE).eval()
    with torch.no_grad():
        out_b = base(dict(batch))
    r.append(check("structure_head is None", base.structure_head is None))
    r.append(check("no structure keys emitted",
                   not any(k.startswith(("x_denoised", "coords_rollout", "structure_")) for k in out_b)))

    print("2. head enabled: shapes")
    torch.manual_seed(0)
    m = ContactMapTriSiT(**BASE, structure_head=dict(SH)).eval()
    with torch.no_grad():
        out = m(dict(batch))
    r.append(check("x_denoised [B,L,3]", tuple(out["x_denoised"].shape) == (B, L, 3)))
    r.append(check("structure_pair_logits [B,L,L,39]",
                   tuple(out["structure_pair_logits"].shape) == (B, L, L, 39)))
    r.append(check("sigma [B]", tuple(out["sigma"].shape) == (B,)))
    r.append(check("denoised finite", torch.isfinite(out["x_denoised"]).all().item()))
    r.append(check("padded residues zeroed",
                   (out["x_denoised"][1, L - 3:].abs().sum() == 0).item()))
    r.append(check("distogram symmetric",
                   torch.allclose(out["structure_pair_logits"],
                                  out["structure_pair_logits"].transpose(1, 2), atol=1e-6)))

    print("3. ⛔ the detach: structure loss must NOT reach the trunk")
    m2 = ContactMapTriSiT(**BASE, structure_head=dict(SH)).train()
    o = m2(dict(batch))
    loss, _ = diffusion_loss(o["x_denoised"], batch["x_1_ca"], o["sigma"], batch["mask"])
    loss.sum().backward()
    trunk_grad = sum(
        p.grad.abs().sum().item()
        for n, p in m2.named_parameters()
        if not n.startswith("structure_head") and p.grad is not None
    )
    head_grad = sum(
        p.grad.abs().sum().item()
        for n, p in m2.named_parameters()
        if n.startswith("structure_head") and p.grad is not None
    )
    r.append(check(f"trunk grad EXACTLY zero ({trunk_grad:.3e})", trunk_grad == 0.0))
    r.append(check(f"head grad non-zero ({head_grad:.3e})", head_grad > 0))

    print("4. EDM preconditioning limits (AF3 / Karras)")
    sh = m.structure_head.structure
    s = torch.randn(B, L, 32)
    z = torch.randn(B, L, L, 16)
    x = torch.randn(B, L, 3)
    with torch.no_grad():
        tiny = sh.denoise(x, torch.full((B,), 1e-6), s, z, batch["mask"])
        huge = sh.denoise(x, torch.full((B,), 1e6), s, z, batch["mask"])
    # sigma -> 0: c_skip -> 1, c_out -> 0, so D(x) -> x exactly.
    r.append(check("sigma->0 gives identity", torch.allclose(tiny, x, atol=1e-4)))
    # sigma -> inf: c_skip -> 0, so the skip term vanishes and D is pure network output.
    r.append(check("sigma->inf kills the skip term",
                   (huge - x).abs().mean() > (tiny - x).abs().mean()))

    print("5. noise schedule + sampling distribution")
    ts = torch.linspace(0, 1, 5)
    sig = noise_schedule(ts)
    r.append(check(f"schedule descends {sig[0]:.1f} -> {sig[-1]:.2e}",
                   bool((sig[1:] < sig[:-1]).all())))
    r.append(check("schedule starts at sigma_data*s_max",
                   abs(sig[0].item() - SIGMA_DATA * 160.0) < 1.0))
    draw = sample_noise_level((20000,), torch.device("cpu"))
    med = draw.median().item()
    # median of sigma_data*exp(-1.2 + 1.5*N) = sigma_data*exp(-1.2) = 16*0.3012 = 4.82
    r.append(check(f"median sigma ~ 16*exp(-1.2)=4.82 (got {med:.2f})", abs(med - 4.82) < 0.35))

    print("6. weighted_rigid_align is a ROTATION, and recovers a known one")
    ang = torch.tensor(0.7)
    rot = torch.tensor([[torch.cos(ang), -torch.sin(ang), 0.0],
                        [torch.sin(ang), torch.cos(ang), 0.0],
                        [0.0, 0.0, 1.0]])
    xg = torch.randn(1, 12, 3)
    xp = xg @ rot.T + torch.tensor([3.0, -1.0, 2.0])
    mk = torch.ones(1, 12)
    aligned = weighted_rigid_align(xp, xg, mk, mk)
    r.append(check("recovers the rigid transform", torch.allclose(aligned, xp, atol=1e-4)))
    # A reflected target must NOT be fit by a mirror; error should stay large.
    refl = xg * torch.tensor([1.0, 1.0, -1.0])
    bad = weighted_rigid_align(xp, refl, mk, mk)
    r.append(check("reflection is not fitted", (bad - xp).abs().mean() > 1e-2))

    print("7. smooth_lddt bounds")
    perfect = smooth_lddt(xg, xg, mk)
    far = smooth_lddt(xg, xg + 50.0 * torch.randn_like(xg), mk)
    # ⛔ The floor is NOT zero and that is not a bug. AF3's smooth LDDT (SI Alg. 27) sums four
    # sigmoids that do not saturate at delta=0:
    #   0.25*(sigmoid(0.5)+sigmoid(1)+sigmoid(2)+sigmoid(4)) = 0.804082
    # so a perfect structure scores 1 - 0.804082 = 0.195918. Protenix computes the identical
    # quantity (model/loss.py SmoothLDDTLoss) and returns 1 - lddt as the loss (:159). Asserting
    # "~0" here would be asserting against AF3's own definition.
    floor = 1.0 - 0.25 * sum(
        torch.sigmoid(torch.tensor(t)).item() for t in (0.5, 1.0, 2.0, 4.0)
    )
    r.append(check(f"identical -> the analytic floor {floor:.4f} (got {perfect.item():.4f})",
                   abs(perfect.item() - floor) < 0.01))
    r.append(check(f"scrambled -> near 1 ({far.item():.4f})", far.item() > 0.8))
    r.append(check("bounded [0,1]", bool((perfect >= 0).all() and (far <= 1.0).all())))

    print("8. mini-rollout: 20 steps, no grad")
    r.append(check(f"MINI_ROLLOUT_STEPS == 20 (SI 4.1)", MINI_ROLLOUT_STEPS == 20))
    r.append(check(f"FULL_INFERENCE_STEPS == 200", FULL_INFERENCE_STEPS == 200))
    b2 = make_batch(with_gt=False)
    with torch.no_grad():
        o2 = m(dict(b2))
    r.append(check("rollout emitted when no GT", "coords_rollout" in o2))
    r.append(check("rollout detached", o2["coords_rollout"].grad_fn is None))
    r.append(check("rollout finite", torch.isfinite(o2["coords_rollout"]).all().item()))

    print("9. EDM loss weight matches (sigma^2+sd^2)/(sigma*sd)^2, not the SI's printed form")
    sg = torch.tensor([4.0])
    _, aux = diffusion_loss(torch.zeros(1, 4, 3), torch.zeros(1, 4, 3), sg, torch.ones(1, 4),
                            use_smooth_lddt=False)
    want = (4.0 ** 2 + SIGMA_DATA ** 2) / (4.0 * SIGMA_DATA) ** 2
    r.append(check(f"weight {aux['edm_weight'].item():.6f} == {want:.6f}",
                   abs(aux["edm_weight"].item() - want) < 1e-6))

    print("10. bf16 autocast (the regime training actually runs in)")
    # ⛔ This section exists because the CPU-only checks above ALL passed while the code was broken
    # under autocast: torch.linalg.det has no BFloat16 kernel, and autocast re-downcasts fp32 tensors
    # handed to einsum, so a .float() cast on the covariance was silently undone. Only a benchmark on
    # a real GPU caught it. Exercise the same path here, on whatever device is available.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16
    xg2 = torch.randn(2, 10, 3, device=dev)
    mk2 = torch.ones(2, 10, device=dev)
    with torch.autocast(device_type=dev.type, dtype=amp_dtype):
        a2 = weighted_rigid_align(xg2.to(amp_dtype), xg2.to(amp_dtype), mk2, mk2)
        l2, _ = diffusion_loss(xg2.to(amp_dtype), xg2, torch.full((2,), 4.0, device=dev), mk2)
    r.append(check(f"weighted_rigid_align survives autocast on {dev.type}",
                   torch.isfinite(a2).all().item()))
    r.append(check("diffusion_loss survives autocast", torch.isfinite(l2).all().item()))

    print()
    print(f"{sum(r)}/{len(r)} checks pass")
    return 0 if all(r) else 3


if __name__ == "__main__":
    sys.exit(main())
