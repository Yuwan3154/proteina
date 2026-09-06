import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/Users/Chenxi/SOLab/proteina/.claude/worktrees/distogram-head/scratchpad")
import _shim
_shim.load("proteinfoundation.nn.af3_diffusion", "proteinfoundation/nn/af3_diffusion.py")
_aa = _shim.load("proteinfoundation.nn.atom_attention", "proteinfoundation/nn/atom_attention.py")
torch.manual_seed(0)
N_QUERIES, N_KEYS = _aa.N_QUERIES, _aa.N_KEYS
blocked_indices = _aa.blocked_indices
gather_blocked_pair = _aa.gather_blocked_pair
_pad_atoms = _aa._pad_atoms
LocalAtomAttention = _aa.LocalAtomAttention
AtomAttentionEncoder = _aa.AtomAttentionEncoder
AtomAttentionDecoder = _aa.AtomAttentionDecoder
AtomTransformerBlock = _aa.AtomTransformerBlock

dev = "cpu"

# ---------- 1. blocked_indices arithmetic ----------
for A in [70, 128, 5376, 224 * 14]:
    qidx, kidx, kvalid, ap = blocked_indices(A, dev)
    nb = qidx.shape[0]
    assert (qidx.reshape(-1) == torch.arange(ap)).all()
    # raw (unclamped) key index
    centre = torch.arange(nb) * N_QUERIES + N_QUERIES // 2
    raw = centre[:, None] + (torch.arange(N_KEYS) - N_KEYS // 2)[None, :]
    # queries subset of valid keys?
    sub_ok = True
    dup_bad = 0
    for b in range(nb):
        vk = kidx[b][kvalid[b]]
        qb = qidx[b]
        if not set(qb.tolist()).issubset(set(vk.tolist())):
            sub_ok = False
        # duplicates among VALID keys?
        if len(set(vk.tolist())) != len(vk.tolist()):
            dup_bad += 1
    off_lo = (raw[0][kvalid[0]].min() - qidx[0].min()).item()
    print(f"A={A} nb={nb} ap={ap} queries_subset_of_valid_keys={sub_ok} "
          f"blocks_with_dup_valid_keys={dup_bad} "
          f"win_block0=[{raw[0].min().item()},{raw[0].max().item()}] "
          f"win_last=[{raw[-1].min().item()},{raw[-1].max().item()}] ap-1={ap-1}")

# per-query reach
qidx, kidx, kvalid, ap = blocked_indices(320, dev)
b = 4
vk = kidx[b][kvalid[b]]
for qi in [qidx[b][0].item(), qidx[b][-1].item()]:
    print("  query", qi, "reach", (vk.min() - qi).item(), "..", (vk.max() - qi).item())

# ---------- 2. _pad_atoms axis check ----------
t = torch.arange(2 * 3 * 4 * 5).float().reshape(2, 3, 4, 5)
p = _pad_atoms(t, 7)
print("pad shape", tuple(p.shape), "zeros_at_tail", bool((p[:, 3:] == 0).all()),
      "orig_preserved", bool(torch.equal(p[:, :3], t)))
t2 = torch.arange(2 * 3).reshape(2, 3)  # dim==2
try:
    p2 = _pad_atoms(t2, 5)
    print("pad dim2 shape", tuple(p2.shape))
except Exception as e:
    print("pad dim2 FAILED:", type(e).__name__, e)

# ---------- 3. gather_blocked_pair correctness ----------
B, L, c = 2, 9, 3
A = 40
z = torch.randn(B, L, L, c)
tok = torch.randint(0, L, (B, A))
qidx, kidx, kvalid, ap = blocked_indices(A, dev)
tokp = _pad_atoms(tok[..., None], ap)[..., 0]
g = gather_blocked_pair(z, tokp, qidx, kidx)
ok = True
for bb in range(B):
    for blk in range(qidx.shape[0]):
        for i in range(0, N_QUERIES, 7):
            for j in range(0, N_KEYS, 31):
                a1 = g[bb, blk, i, j]
                a2 = z[bb, tokp[bb, qidx[blk, i]], tokp[bb, kidx[blk, j]]]
                if not torch.allclose(a1, a2):
                    ok = False
print("gather_blocked_pair correct:", ok, "shape", tuple(g.shape))

# ---------- 4. LOCALITY via jacobian ----------
torch.manual_seed(1)
c_a, c_s, c_z, H = 8, 8, 4, 2
attn = LocalAtomAttention(c_a, c_s, c_z, H).double()
Ap_ = 128
qidx, kidx, kvalid, ap = blocked_indices(Ap_, dev)
NB = qidx.shape[0]
a = torch.randn(1, ap, c_a, dtype=torch.float64, requires_grad=True)
pair = torch.randn(1, NB, N_QUERIES, N_KEYS, c_z, dtype=torch.float64)
km = torch.ones(1, NB, N_KEYS, dtype=torch.bool) & kvalid[None]
out = attn(a, a.detach(), pair, km, qidx, kidx)
qsel = 40
g = torch.autograd.grad(out[0, qsel].sum(), a, retain_graph=True)[0][0].abs().sum(-1)
nz = (g > 1e-12).nonzero().flatten().tolist()
blk = qsel // N_QUERIES
vk = sorted(kidx[blk][kvalid[blk]].tolist())
print(f"locality: query {qsel} nonzero-grad atoms {min(nz)}..{max(nz)} count={len(nz)}; "
      f"expected valid key set {min(vk)}..{max(vk)} count={len(vk)}; "
      f"exact_match={set(nz) == set(vk) | {qsel}}")
print("  set(nz)==set(valid_keys):", set(nz) == set(vk))

# ---------- 5. all-padding-block guard: does it leak into a REAL query? ----------
torch.manual_seed(2)
Ap_ = 256
qidx, kidx, kvalid, ap = blocked_indices(Ap_, dev)
NB = qidx.shape[0]
# make atoms >=128 padding
mp = torch.zeros(1, ap)
mp[0, :128] = 1
key_mask = mp.bool()[:, kidx] & kvalid[None]
km = key_mask | (~key_mask.any(-1, keepdim=True))
# which blocks got wholesale-unmasked?
fired = (~key_mask.any(-1)).nonzero().tolist()
print("guard fired on blocks:", fired)
for blkid in [b[1] for b in fired]:
    qs = qidx[blkid]
    print("   block", blkid, "queries", qs.min().item(), "..", qs.max().item(),
          "any_real_query:", bool(mp[0, qs].any()))
# also: does any REAL query's block contain an unmasked padded key beyond kvalid?
leak = 0
for blkid in range(NB):
    qs = qidx[blkid]
    if not mp[0, qs].any():
        continue
    bad = km[0, blkid] & ~key_mask[0, blkid]
    leak += int(bad.sum())
print("padded keys unmasked in blocks containing a real query:", leak)

# ---------- 6. encoder: q_atom masked? token aggregation correctness ----------
torch.manual_seed(3)
B, L = 2, 5
c_s, c_z, c_token, c_atom, c_ap = 6, 4, 7, 8, 3
enc = AtomAttentionEncoder(c_atom=c_atom, c_atompair=c_ap, c_token=c_token, c_s=c_s,
                           c_z=c_z, n_blocks=2, n_heads=2, n_ref_feats=8).double()
A = L * 14
ref_feats = torch.randn(B, A, 8, dtype=torch.float64)
ref_pos = torch.randn(B, A, 3, dtype=torch.float64)
a2t = torch.arange(L)[None, :, None].expand(B, L, 14).reshape(B, A).contiguous()
amask = torch.zeros(B, A, dtype=torch.float64)
amask.reshape(B, L, 14)[:, :, :5] = 1.0
amask.reshape(B, L, 14)[:, 4, :] = 0.0            # residue 4 = padding residue
s = torch.randn(B, L, c_s, dtype=torch.float64)
z = torch.randn(B, L, L, c_z, dtype=torch.float64)
npos = torch.randn(B, A, 3, dtype=torch.float64)
a_tok, q_atom = enc(ref_feats, ref_pos, a2t, s, z, amask, noisy_pos=npos)
inv = (amask == 0)
print("q_atom masked at invalid atoms:", bool(q_atom[inv].abs().max() < 1e-12),
      "max|q_atom[invalid]|=%.3e" % q_atom[inv].abs().max().item())
print("a_token at padded residue 4 is zero:", bool(a_tok[:, 4].abs().max() < 1e-12))
# manual mean over valid atoms of to_token(q)
aa = enc.to_token(q_atom) * amask[..., None]
man = aa.reshape(B, L, 14, -1).sum(2) / amask.reshape(B, L, 14).sum(2).clamp_min(1.0)[..., None]
print("token mean-pool matches manual:", bool(torch.allclose(a_tok, man, atol=1e-10)))

# ---------- 7. does a padded atom's INPUT influence a real atom's output? ----------
ref_feats2 = ref_feats.clone()
ref_feats2[0, 4 * 14 + 3] += 100.0       # invalid slot of a REAL residue 3? -> residue 0..3 valid
# pick an invalid slot inside a real residue: residue 1, slot 9
ref_feats2 = ref_feats.clone()
ref_feats2[0, 1 * 14 + 9] += 100.0
a_tok2, q_atom2 = enc(ref_feats2, ref_pos, a2t, s, z, amask, noisy_pos=npos)
d_tok = (a_tok2 - a_tok).abs().max().item()
dq = (q_atom2 - q_atom)[amask > 0].abs().max().item()
print("perturb invalid slot -> max|d a_token|=%.3e  max|d q_atom[valid]|=%.3e" % (d_tok, dq))

# padded residue's atoms
ref_feats3 = ref_feats.clone()
ref_feats3[0, 4 * 14 + 2] += 100.0
a_tok3, q_atom3 = enc(ref_feats3, ref_pos, a2t, s, z, amask, noisy_pos=npos)
print("perturb padded-residue atom -> max|d a_token|=%.3e max|d q_atom[valid]|=%.3e"
      % ((a_tok3 - a_tok).abs().max().item(),
         (q_atom3 - q_atom)[amask > 0].abs().max().item()))

# ---------- 8. decoder ----------
dec = AtomAttentionDecoder(c_atom=c_atom, c_atompair=c_ap, c_token=c_token,
                           n_blocks=2, n_heads=2).double()
zap = torch.randn(B, L, L, c_ap, dtype=torch.float64)
o = dec(a_tok, q_atom, a2t, amask, zap)
print("decoder out masked:", bool(o[inv].abs().max() < 1e-12), "shape", tuple(o.shape))
print("decoder NaN:", bool(torch.isnan(o).any()))

# ---------- 9. conditioning identity: is `s` the evolving q? ----------
import inspect
from proteinfoundation.nn.atom_attention import AtomTransformerBlock
print("encoder block call site:",
      [l.strip() for l in inspect.getsource(AtomAttentionEncoder.forward).splitlines()
       if "blk(" in l])
print("decoder block call site:",
      [l.strip() for l in inspect.getsource(AtomAttentionDecoder.forward).splitlines()
       if "blk(" in l])

# ---------- 10. ref_pos pair feature sanity ----------
from proteinfoundation.datasets.atom_features import atom14_features
aat = torch.tensor([[0, 0, 1, 2, 3]])
mk = torch.ones(1, 5)
rf, rp, tk, am = atom14_features(aat, mk)
d_same = (rp[0, 1] - rp[0, 0]).norm().item()     # N of res0 vs CA of res0
d_cross = (rp[0, 14 + 1] - rp[0, 1]).norm().item()   # CA res1 vs CA res0
print("ref_pos: |CA_res0 - N_res0| = %.3f  |CA_res1 - CA_res0| = %.3f (should be ~3.8 if global)"
      % (d_same, d_cross))
print("ref_pos identical across identical residues:",
      bool(torch.equal(rp[0, :14], rp[0, 14:28])))
