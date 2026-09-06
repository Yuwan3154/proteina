import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from proteinfoundation.datasets.contact_augment import augment_contacts

torch.manual_seed(0)
B, L, Lreal = 2, 32, 20
mask = torch.zeros(B, L); mask[:, :Lreal] = 1.0

# synthetic confind-like map: band |i-j|<=3 plus a few long range, symmetric, zero diag
c = torch.zeros(B, L, L)
idx = torch.arange(L)
band = (idx[:, None] - idx[None, :]).abs()
c[:, :, :] = ((band <= 3) & (band > 0)).float()
c[:, 5, 15] = 1.0; c[:, 15, 5] = 1.0
pm = mask[:, :, None] * mask[:, None, :]
c = c * pm

def stats(m, name):
    up = torch.triu(torch.ones(L, L), 1)[None] * pm
    print(f"{name}: upper-pos per sample {(m*up).sum((1,2)).tolist()}  "
          f"sym_err {(m - m.transpose(1,2)).abs().max().item():.3g}  "
          f"diag_max {m.diagonal(dim1=1,dim2=2).max().item():.3g}  "
          f"pad_max {(m*(1-pm)).max().item():.3g}  "
          f"vals {sorted(set(m.unique().tolist()))}")

stats(c, "clean   ")
for rate in (0.1, 0.5):
    for mode in ("balanced", "uniform"):
        a = augment_contacts(c.clone(), mask, rate, mode)
        stats(a, f"{mode[:4]} r={rate}")
        up = torch.triu(torch.ones(L, L), 1)[None] * pm
        kept = ((a > 0.5) & (c > 0.5) & (up > 0)).sum((1, 2))
        dropped = ((a <= 0.5) & (c > 0.5) & (up > 0)).sum((1, 2))
        added = ((a > 0.5) & (c <= 0.5) & (up > 0)).sum((1, 2))
        print(f"    kept {kept.tolist()} dropped {dropped.tolist()} added {added.tolist()}")

# sequence-separation profile of ADDED false contacts, balanced, larger L
L2, R2 = 384, 384
mask2 = torch.ones(1, L2)
i2 = torch.arange(L2)
b2 = (i2[:, None] - i2[None, :]).abs()
c2 = (((b2 <= 4) & (b2 > 0)) | (torch.rand(L2, L2) < 0.02)).float()[None]
c2 = torch.maximum(c2, c2.transpose(1, 2))
c2 = c2 * (1 - torch.eye(L2))[None]
up2 = torch.triu(torch.ones(L2, L2), 1)[None]
P = ((c2 > 0.5) & (up2 > 0)).sum().item()
print(f"\nL=384 synthetic: upper positives {P} ({100*P/(L2*(L2-1)/2):.2f}% of upper pairs)")
a2 = augment_contacts(c2, mask2, 0.1, "balanced")
addmask = (a2 > 0.5) & (c2 <= 0.5) & (up2 > 0)
dropmask = (a2 <= 0.5) & (c2 > 0.5) & (up2 > 0)
sep_add = b2[None].expand_as(a2)[addmask].float()
sep_drop = b2[None].expand_as(a2)[dropmask].float()
print(f"  added {addmask.sum().item()} dropped {dropmask.sum().item()}")
print(f"  |i-j| of ADDED   : median {sep_add.median():.0f}  frac<=4 {(sep_add<=4).float().mean():.4f}")
print(f"  |i-j| of DROPPED : median {sep_drop.median():.0f}  frac<=4 {(sep_drop<=4).float().mean():.4f}")
# how many dropped are i,i+1 (physically impossible to be non-contact)
print(f"  dropped with |i-j|==1: {(sep_drop==1).sum().item()} of {dropmask.sum().item()}")

# uniform mode positive-rate blowup
a3 = augment_contacts(c2, mask2, 0.1, "uniform")
print(f"  uniform r=0.1 upper positives {((a3>0.5)&(up2>0)).sum().item()} vs clean {P}")
