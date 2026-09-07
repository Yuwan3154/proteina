"""Gate for the AF3 reference-offset block (fix A).

The feature is only worth anything if the frame gate is real. Two ways it could be silently wrong:
  - the uid collapses to the residue index, so cross-rigid-group offsets stay in (the bug we are
    fixing, just relabelled);
  - the uid is so fine-grained that N/CA/C/CB fall into different groups, which would DESTROY the
    chirality tetrahedron the feature exists to encode.
Both are checked numerically here, plus the usual masking/gradient hygiene.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.atom_features import N_REF_FEATS, N_RIGID_GROUPS, atom14_features
from proteinfoundation.nn.atom_attention import AtomAttentionEncoder
from proteinfoundation.openfold_stub.np import residue_constants as rc

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


torch.manual_seed(0)
B, L = 2, 6
aatype = torch.tensor([[rc.restype_order[rc.restype_3to1[a]] for a in
                        ["ALA", "TRP", "GLY", "ARG", "SER", "LEU"]]] * B)
mask = torch.ones(B, L)
mask[1, -2:] = 0.0

feats, pos, a2t, amask, uid = atom14_features(aatype, mask)
A = L * 14

print("\n=== 1. uid construction ===")
check("atom14_features returns 5 tensors", uid.shape == (B, A), str(tuple(uid.shape)))
check("uid is constant within (residue, group)",
      int(uid.view(B, L, 14)[0, 0, 0]) == int(uid.view(B, L, 14)[0, 0, 1]),
      "ALA N and CA share a uid")

u = uid.view(B, L, 14)
grp = torch.as_tensor(rc.restype_atom14_to_rigid_group)[aatype.clamp(0, 20)]
check("uid separates residues", int(u[0, 0, 0]) != int(u[0, 1, 0]))
check("uid separates rigid groups WITHIN a residue",
      int(u[0, 1, 4]) != int(u[0, 1, 5]),
      f"TRP CB(g{int(grp[0,1,4])})={int(u[0,1,4])} vs CG(g{int(grp[0,1,5])})={int(u[0,1,5])}")

# ⭐ The whole point of fix A: this tetrahedron must stay in ONE reference space.
tet = [int(u[0, r, i]) for r in range(L) for i in (0, 1, 2, 4)
       if amask.view(B, L, 14)[0, r, 4] > 0]
per_res = [tet[i:i + 4] for i in range(0, len(tet), 4)]
check("N/CA/C/CB share one uid in every residue",
      all(len(set(q)) == 1 for q in per_res), str(per_res))

print("\n=== 2. the gate actually gates ===")
enc = AtomAttentionEncoder(c_atom=32, c_atompair=8, c_token=48, c_s=24, c_z=16,
                           n_blocks=1, n_heads=2, n_ref_feats=N_REF_FEATS)
s = torch.randn(B, L, 24)
z = torch.randn(B, L, L, 16)
noisy = torch.randn(B, A, 3)

with torch.no_grad():
    a_tok, q = enc(feats, pos, a2t, s, z, amask, noisy_pos=noisy, ref_space_uid=uid)
check("encoder runs and is finite", torch.isfinite(a_tok).all().item() and
      tuple(a_tok.shape) == (B, L, 48), str(tuple(a_tok.shape)))

# Perturbing ref_pos of an atom must NOT change the offset term for atoms in other frames.
# Compare the pair track directly by monkey-free means: run twice, once with a shifted ref_pos on
# a single atom, and confirm the change is confined.
pos_b = pos.clone()
pos_b[0, 1 * 14 + 5] += 10.0                      # TRP CG, rigid group 4
with torch.no_grad():
    a2, _ = enc(feats, pos_b, a2t, s, z, amask, noisy_pos=noisy, ref_space_uid=uid)
    a3, _ = enc(feats, pos_b, a2t, s, z, amask, noisy_pos=noisy,
                ref_space_uid=torch.zeros_like(uid))     # everything in ONE space = no gate
d_gated = (a2 - a_tok).abs().max().item()
with torch.no_grad():
    a0_ungated, _ = enc(feats, pos, a2t, s, z, amask, noisy_pos=noisy,
                        ref_space_uid=torch.zeros_like(uid))
d_ungated = (a3 - a0_ungated).abs().max().item()
check("an ungated uid propagates the perturbation FURTHER than the real gate",
      d_ungated > d_gated, f"gated={d_gated:.4f} ungated={d_ungated:.4f}")

print("\n=== 3. masking and gradients ===")
enc.zero_grad()
a_tok, q = enc(feats, pos, a2t, s, z, amask, noisy_pos=noisy, ref_space_uid=uid)
a_tok.sum().backward()
for nm in ["pair_proj", "dist_proj", "valid_proj"]:
    g = getattr(enc, nm).weight.grad
    check(f"{nm} receives gradient", g is not None and g.abs().sum().item() > 0)
gm = sum(p.grad.abs().sum().item() for p in enc.pair_mlp.parameters() if p.grad is not None)
check("pair_mlp receives gradient", gm > 0, f"|grad|={gm:.3e}")

check("padded residues produce zero token output",
      a_tok[1, -2:].abs().max().item() < 1e-5 or mask[1, -1] == 1,
      f"max={a_tok[1, -2:].abs().max().item():.3e}")
check("uid stays inside the int range implied by N_RIGID_GROUPS",
      int(uid.max()) < L * N_RIGID_GROUPS, f"max uid={int(uid.max())} < {L * N_RIGID_GROUPS}")

# _pad_atoms zero-fills, so a raw uid of 0 (residue 0, group 0) would collide with PADDING and make
# padded slots look like they share residue 0's backbone frame. The encoder shifts by +1 to avoid it.
from proteinfoundation.nn.atom_attention import _pad_atoms, blocked_indices  # noqa: E402
_q, _k, _kv, _ap = blocked_indices(A, uid.device)
_up = _pad_atoms((uid + 1)[..., None], _ap)[..., 0]
check("padded atom slots share a frame with NO real atom",
      _ap == A or int((_up[:, A:] == 0).all()) == 1,
      f"padded slots {A}..{_ap} all sentinel-0")
check("no real atom carries the padding sentinel", int((_up[:, :A] != 0).all()) == 1)

print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
