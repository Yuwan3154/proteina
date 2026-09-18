"""Follow-ups: (1) is the windowed-FAPE mirror signal beyond per-atom stereochemistry?
(2) matched-RMSD retention with more reps, (3) does the det=+1 Kabsch gap really invert sign?"""
import torch

torch.set_default_dtype(torch.float64)
SRC = ('/Users/Chenxi/SOLab/proteina/.claude/worktrees/distogram-head/scratchpad/'
       'fape_mirror_test.py')
exec(open(SRC).read().split('L = 30')[0])

seg = [('H', 14), ('L', 4), ('H', 14), ('L', 4), ('E', 10), ('L', 4), ('H', 14)]
TOR = {'H': (-57., -47.), 'E': (-135., 135.), 'L': (-70., 140.)}
phis, psis = [], []
for k, n in seg:
    p, q = TOR[k]
    phis += [p] * n
    psis += [q] * n
L = len(phis)
PHI, PSI, OM = torch.tensor(phis), torch.tensor(psis), [180.] * (L - 1)
xn = build(phis, psis, OM).reshape(-1, 3)
xn = xn - xn.mean(0)
Rn, tn = from_3_points(xn.reshape(L, 3, 3)[:, 0], xn.reshape(L, 3, 3)[:, 1],
                       xn.reshape(L, 3, 3)[:, 2])
res_of_atom = torch.arange(L).repeat_interleave(3)
SEP = (torch.arange(L)[:, None] - res_of_atom[None, :]).abs()


def fape_sel(xp, sel, clamp):
    bb = xp.reshape(L, 3, 3)
    Rp, tp = from_3_points(bb[:, 0], bb[:, 1], bb[:, 2])
    lp = torch.einsum('fji,fpj->fpi', Rp, xp[None] - tp[:, None])
    lt = torch.einsum('fji,fpj->fpi', Rn, xn[None] - tn[:, None])
    e = torch.sqrt(((lp - lt) ** 2).sum(-1) + 1e-8)
    if clamp is not None:
        e = e.clamp(0, clamp)
    e = e / 10.0
    m = sel.to(e.dtype)
    return (e * m).sum() / m.sum()


def rms(xp):
    return torch.sqrt(((xp - xp.mean(0) - kabsch_align_gt_to_pred(xp, xn)) ** 2).sum(-1).mean())


SEL_0 = SEP == 0
SEL_13 = (SEP >= 1) & (SEP <= 3)
SEL_ALL = SEP >= 0
MIRR = xn @ torch.diag(torch.tensor([1., 1., -1.]))
MIRR = MIRR - MIRR.mean(0)

print("A. Perfect mirror, decomposed by sequence separation (is W3 beyond per-atom stereo?)")
for nm, sel in [("own-residue |i-j|=0  (per-atom stereo)", SEL_0),
                ("|i-j| in [1,3]       (CA-dihedral scale)", SEL_13),
                ("all |i-j|", SEL_ALL)]:
    print(f"   {nm:42s} FAPEc10 mirror={fape_sel(MIRR, sel, 10.0):.4f}  "
          f"unclamped={fape_sel(MIRR, sel, None):.4f}")

print()
print("B. Matched-RMSD retention, 40 reps. RET = GAP(degraded)/GAP(perfect).")
g = torch.Generator().manual_seed(7)
G0 = {'c10': fape_sel(MIRR, SEL_ALL, 10.0), 'unc': fape_sel(MIRR, SEL_ALL, None),
      'w13': fape_sel(MIRR, SEL_13, 10.0), 'kab': rms(MIRR)}
print(f"   perfect-mirror gaps: FAPEc10={G0['c10']:.3f}  unclamped={G0['unc']:.3f}  "
      f"W[1,3]={G0['w13']:.3f}  Kabsch={G0['kab']:.3f} A")
print(f"   {'mode':>8s} {'knob':>6s} {'RMSDcorr':>9s} | {'RET c10':>8s} {'RET unc':>8s} "
      f"{'RET W[1,3]':>11s} {'RET Kabsch':>11s}")
NR = 40


def run(mode, knob):
    a = torch.zeros(5)
    for _ in range(NR):
        if mode == 'IID':
            e = knob * torch.randn(xn.shape, generator=g)
            xr, xm = xn + e, MIRR + e
        else:
            ep = knob * torch.randn(L, generator=g)
            eq = knob * torch.randn(L, generator=g)
            xr = build((PHI + ep).tolist(), (PSI + eq).tolist(), OM).reshape(-1, 3)
            xm = build((-PHI + ep).tolist(), (-PSI + eq).tolist(), OM).reshape(-1, 3)
            xr, xm = xr - xr.mean(0), xm - xm.mean(0)
        a += torch.tensor([rms(xr),
                           fape_sel(xm, SEL_ALL, 10.0) - fape_sel(xr, SEL_ALL, 10.0),
                           fape_sel(xm, SEL_ALL, None) - fape_sel(xr, SEL_ALL, None),
                           fape_sel(xm, SEL_13, 10.0) - fape_sel(xr, SEL_13, 10.0),
                           rms(xm) - rms(xr)]) / NR
    print(f"   {mode:>8s} {knob:6.2f} {a[0]:9.2f} | {a[1]/G0['c10']:8.1%} "
          f"{a[2]/G0['unc']:8.1%} {a[3]/G0['w13']:11.1%} {a[4]/G0['kab']:11.1%}")


for k in [0.1, 0.5, 1.0, 2.0]:
    run('IID', k)
for k in [2.0, 5.0, 10.0, 20.0, 30.0, 45.0]:
    run('TORSION', k)
