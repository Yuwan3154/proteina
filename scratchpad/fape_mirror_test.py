import math

import torch

torch.set_default_dtype(torch.float64)

# ---------------- NeRF backbone builder (ideal Engh&Huber geometry) ----------------
B_N_CA, B_CA_C, B_C_N = 1.458, 1.525, 1.329
A_N_CA_C, A_CA_C_N, A_C_N_CA = 111.0, 116.2, 121.7


def place(a, b, c, bond, angle_deg, tors_deg):
    ang, tor = math.radians(angle_deg), math.radians(tors_deg)
    bc = c - b
    bc = bc / bc.norm()
    n = torch.cross(b - a, bc, dim=-1)
    n = n / n.norm()
    m = torch.stack([bc, torch.cross(n, bc, dim=-1), n])
    d = torch.tensor([-bond * math.cos(ang),
                      bond * math.sin(ang) * math.cos(tor),
                      bond * math.sin(ang) * math.sin(tor)])
    return c + d @ m


def build(phis, psis, omegas):
    """Return [L,3,3] of N,CA,C."""
    N = [torch.tensor([0., 0., 0.])]
    CA = [torch.tensor([B_N_CA, 0., 0.])]
    C = [place(torch.tensor([0., 1., 0.]), N[0], CA[0], B_CA_C, A_N_CA_C, phis[0])]
    for i in range(1, len(phis)):
        N.append(place(N[i - 1], CA[i - 1], C[i - 1], B_C_N, A_CA_C_N, psis[i - 1]))
        CA.append(place(CA[i - 1], C[i - 1], N[i], B_N_CA, A_C_N_CA, omegas[i - 1]))
        C.append(place(C[i - 1], N[i], CA[i], B_CA_C, A_N_CA_C, phis[i]))
    return torch.stack([torch.stack(N), torch.stack(CA), torch.stack(C)], dim=1)


# ---------------- openfold Rigid.from_3_points, verbatim logic ----------------
def from_3_points(p_neg_x_axis, origin, p_xy_plane, eps=1e-8):
    e0 = origin - p_neg_x_axis
    e1 = p_xy_plane - origin
    e0 = e0 / torch.sqrt((e0 * e0).sum(-1, keepdim=True) + eps)
    dot = (e0 * e1).sum(-1, keepdim=True)
    e1 = e1 - e0 * dot
    e1 = e1 / torch.sqrt((e1 * e1).sum(-1, keepdim=True) + eps)
    e2 = torch.cross(e0, e1, dim=-1)          # <-- cross product forces det=+1
    R = torch.stack([e0, e1, e2], dim=-1)     # columns = local basis in global coords
    return R, origin


def fape(Rp, tp, Rt, tt, xp, xt, clamp=None, length_scale=10.0):
    lp = torch.einsum('fji,fpj->fpi', Rp, xp[None] - tp[:, None])   # R^T (x - t)
    lt = torch.einsum('fji,fpj->fpi', Rt, xt[None] - tt[:, None])
    e = torch.sqrt(((lp - lt) ** 2).sum(-1) + 1e-8)
    if clamp is not None:
        e = e.clamp(0, clamp)
    return (e / length_scale).mean(), lp, lt


def kabsch_align_gt_to_pred(x, x_gt):
    """c2c weighted_rigid_align, det=+1 forced (af3_diffusion.py:286-289)."""
    xc = x - x.mean(0, keepdim=True)
    gc = x_gt - x_gt.mean(0, keepdim=True)
    u, _, vt = torch.linalg.svd(gc.T @ xc)
    d = torch.sign(torch.linalg.det(u @ vt))
    R = u @ torch.diag(torch.tensor([1., 1., d])) @ vt
    return gc @ R


L = 30
RH = build([-57.] * L, [-47.] * L, [180.] * (L - 1))   # right-handed alpha helix
LH = build([57.] * L, [47.] * L, [180.] * (L - 1))     # left-handed alpha helix (L-aa)
M = torch.diag(torch.tensor([1., 1., -1.]))
MIR = RH @ M                                            # exact global reflection


def frames(bb):
    return from_3_points(bb[:, 0], bb[:, 1], bb[:, 2])   # N, CA, C


print("=" * 78)
print("Q0. Is a left-handed alpha helix's BACKBONE the mirror of the right-handed one?")
Rr, tr = frames(RH)
Rl, tl = frames(LH)
Rm, tm = frames(MIR)
a = kabsch_align_gt_to_pred(LH.reshape(-1, 3), MIR.reshape(-1, 3))
lc = LH.reshape(-1, 3) - LH.reshape(-1, 3).mean(0)
print(f"   RMSD(LH backbone , reflected-RH backbone) proper-rotation align = "
      f"{torch.sqrt(((lc - a) ** 2).sum(-1).mean()):.4f} A")

print("=" * 78)
print("Q1/Q3. det of Gram-Schmidt frames built by openfold's from_3_points")
for nm, R in [("native  ", Rr), ("mirrored", Rm), ("LH helix", Rl)]:
    print(f"   {nm}: det min={torch.linalg.det(R).min():.9f}  max={torch.linalg.det(R).max():.9f}")

S = torch.diag(torch.tensor([1., 1., -1.]))
pred = torch.einsum('ij,fjk,kl->fil', M, Rr, S)
print("   analytic claim   R_mirror == M @ R_native @ diag(1,1,-1):")
print(f"     max |R_mirror - M R_native S| = {(Rm - pred).abs().max():.3e}   <- 0 => claim holds")
print(f"     max |R_mirror - M R_native  | = "
      f"{(Rm - torch.einsum('ij,fjk->fik', M, Rr)).abs().max():.3e}   "
      "<- would be 0 if frames merely reflected")

print("=" * 78)
print("Q2. Does FAPE cancel the reflection?")
xr, xm, xl = RH.reshape(-1, 3), MIR.reshape(-1, 3), LH.reshape(-1, 3)
for nm, (Rp, tp, xp) in [("native  vs native(RH)", (Rr, tr, xr)),
                         ("MIRROR  vs native(RH)", (Rm, tm, xm)),
                         ("LH-helix vs native(RH)", (Rl, tl, xl))]:
    f_un, lp, lt = fape(Rp, tp, Rr, tr, xp, xr, clamp=None)
    f_cl, _, _ = fape(Rp, tp, Rr, tr, xp, xr, clamp=10.0)
    print(f"   {nm:24s} FAPE_unclamped={f_un:9.5f}   FAPE_clamp10A={f_cl:9.5f}")

print("=" * 78)
print("   local-coordinate structure of the mirror (claim: lp == diag(1,1,-1) lt)")
_, lp, lt = fape(Rm, tm, Rr, tr, xm, xr, clamp=None)
print(f"   max |lp_x - lt_x| = {(lp[..., 0] - lt[..., 0]).abs().max():.3e}")
print(f"   max |lp_y - lt_y| = {(lp[..., 1] - lt[..., 1]).abs().max():.3e}")
print(f"   max |lp_z + lt_z| = {(lp[..., 2] + lt[..., 2]).abs().max():.3e}  <- z NEGATED, not equal")
print(f"   mean 2|z_local| (= per-pair FAPE error, A) = {(2 * lt[..., 2].abs()).mean():.4f}")
frac = ((2 * lt[..., 2].abs()) > 10.0).double().mean()
print(f"   fraction of (frame,atom) pairs with error > 10 A clamp = {frac:.4f}")

print("=" * 78)
print("Controls. What the ACHIRAL losses see (must be exactly 0):")
d_r, d_m, d_l = torch.cdist(xr, xr), torch.cdist(xm, xm), torch.cdist(xl, xl)
print(f"   max |D(mirror)  - D(native)| = {(d_m - d_r).abs().max():.3e}  <- distogram / smooth_lDDT BLIND")
print(f"   max |D(LHhelix) - D(native)| = {(d_l - d_r).abs().max():.3e}")
print("   What the EXISTING c2c loss (det=+1 Kabsch MSE, af3_diffusion.py:286) sees:")
for nm, xp in [("native ", xr), ("MIRROR ", xm), ("LHhelix", xl)]:
    g = kabsch_align_gt_to_pred(xp, xr)
    pc = xp - xp.mean(0, keepdim=True)
    print(f"     {nm}: aligned RMSD vs native = {torch.sqrt(((pc - g) ** 2).sum(-1).mean()):9.5f} A")
