"""apply_random_rotation keeps coordinates fp32 under bf16 autocast (bf16-mixed training).

Before the fix the matmul ran in bf16 inside autocast: on coordinates up to ~60 A that is up to ~0.24 A of error. A rigid
rotation preserves pairwise distances, so the check is the max |D_rot - D_in| over a 4gsl_C-sized chain. CONTROL: the bare
bf16 matmul on the same input must show the error, so a pass is not vacuous.
Usage: python scratchpad/test_rotation_fp32.py   (CPU; no data needed)
"""

import types

import torch

from proteinfoundation.proteinflow.proteina import Proteina, sample_uniform_rotation


def pdist(x):
    return torch.cdist(x.double(), x.double())


def main():
    torch.manual_seed(0)
    n = 186
    x = (torch.rand(1, n, 3) * 120.0 - 60.0).float()
    mask = torch.ones(1, n, dtype=torch.bool)
    stub = types.SimpleNamespace(fm=types.SimpleNamespace(_mask_and_zero_com=lambda v, m: v * m[..., None]))

    rots = sample_uniform_rotation(shape=(1,), dtype=torch.float32, device=x.device)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        bare = torch.matmul(x, rots)
    err_bare = (pdist(bare[0].float()) - pdist(x[0])).abs().max().item()
    assert bare.dtype == torch.bfloat16, f"control: expected the bare matmul to be bf16 under autocast, got {bare.dtype}"
    assert err_bare > 0.05, f"control: bf16 matmul error {err_bare:.4f} A is too small to discriminate"

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out, out_mask = Proteina.apply_random_rotation(stub, x, mask, naug=1)
    err = (pdist(out[0]) - pdist(x[0])).abs().max().item()
    assert out.dtype == torch.float32, out.dtype
    assert torch.equal(out_mask, mask)
    assert err < 1e-3, f"rotation distorted distances by {err:.5f} A"
    print(f"PASS rotation fp32 under bf16 autocast: max |dD| {err:.2e} A (control bare bf16 matmul: {err_bare:.3f} A)")


if __name__ == "__main__":
    main()
