"""Gate for contact-map augmentation.

The property that matters is the one that distinguishes the two modes: "balanced" must PRESERVE the
positive rate, and "uniform" must not. Getting that backwards would silently change the difficulty
of the whole training task by ~4x, and nothing downstream would complain.
"""

import sys

import torch

from proteinfoundation.datasets.contact_augment import augment_contacts

B, L = 4, 96
RATE = 0.1


def make_map(seed=0, density=0.025):
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand(B, L, L, generator=g) < density).float()
    m = torch.triu(m, diagonal=1)
    m = m + m.transpose(1, 2)
    mask = torch.ones(B, L)
    mask[1, L - 20:] = 0.0
    m = m * (mask[:, :, None] * mask[:, None, :])
    return m, mask


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return bool(ok)


def npos(m, mask):
    pair = (mask[:, :, None] * mask[:, None, :])
    return (m * pair).sum(dim=(1, 2))


def main():
    torch.manual_seed(0)
    r = []
    c, mask = make_map()
    base = npos(c, mask)
    print(f"input: {int(base.sum())} contacts over {B} samples, "
          f"density {float((c.sum() / (mask.sum(1) ** 2).sum())):.4f}")

    print("1. balanced (default): the positive COUNT is preserved exactly")
    a = augment_contacts(c, mask, rate=RATE, mode="balanced")
    ab = npos(a, mask)
    r.append(check(f"per-sample counts unchanged {base.tolist()} -> {ab.tolist()}",
                   torch.equal(base, ab)))
    r.append(check("symmetric", torch.allclose(a, a.transpose(1, 2))))
    r.append(check("diagonal zero", float(torch.diagonal(a, dim1=1, dim2=2).abs().sum()) == 0.0))
    r.append(check("binary", bool(((a == 0) | (a == 1)).all())))
    r.append(check("padded rows zero", float(a[1, L - 20:].abs().sum()) == 0.0))

    # How much actually changed? ~2*rate of the contacts (rate dropped + rate added), symmetrised.
    diff = (a != c).float() * (mask[:, :, None] * mask[:, None, :])
    changed_pairs = diff.sum(dim=(1, 2)) / 2.0     # /2 for symmetry
    expect = 2.0 * RATE * (base / 2.0)             # base counts both triangles
    r.append(check(f"changed ~2*rate*n_contacts: {changed_pairs.tolist()} vs expected ~{expect.tolist()}",
                   bool((changed_pairs - expect).abs().max() <= 1.5)))

    print("2. uniform: the positive rate EXPLODES -- this is the 4x-harder mode")
    u = augment_contacts(c, mask, rate=RATE, mode="uniform")
    ub = npos(u, mask)
    ratio = float(ub.sum() / base.sum())
    r.append(check(f"positive count grows sharply ({int(base.sum())} -> {int(ub.sum())}, {ratio:.1f}x)",
                   ratio > 2.0))
    r.append(check("symmetric", torch.allclose(u, u.transpose(1, 2))))
    r.append(check("binary", bool(((u == 0) | (u == 1)).all())))
    r.append(check("padded rows zero", float(u[1, L - 20:].abs().sum()) == 0.0))

    print("3. the two modes are genuinely different")
    r.append(check("balanced preserves rate, uniform does not",
                   torch.equal(base, ab) and ratio > 2.0))

    print("4. rate=0 is the identity")
    z = augment_contacts(c, mask, rate=0.0, mode="balanced")
    r.append(check("balanced rate=0 unchanged", torch.equal(z, c)))

    print("5. determinism under a seeded generator")
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    r.append(check("same seed -> same corruption",
                   torch.equal(augment_contacts(c, mask, RATE, "balanced", g1),
                               augment_contacts(c, mask, RATE, "balanced", g2))))

    print()
    print(f"{sum(r)}/{len(r)} checks pass")
    return 0 if all(r) else 3


if __name__ == "__main__":
    sys.exit(main())
