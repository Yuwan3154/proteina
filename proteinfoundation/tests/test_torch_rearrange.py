"""Unit tests for torch_rearrange.

Every test checks numerical equality against einops.rearrange (not just shape).
Categories mirror the approved plan: permutation, merge, split, squeeze,
unsqueeze, ellipsis variants, combinations, torch.compile compatibility,
historical Proteina usage patterns.
"""

import pytest
import torch
import einops

from proteinfoundation.utils.torch_rearrange import torch_rearrange


# ─────────────────────────── helper ───────────────────────────────────────────

def check(tensor: torch.Tensor, pattern: str, **kw):
    expected = einops.rearrange(tensor, pattern, **kw)
    actual   = torch_rearrange(tensor, pattern, **kw)
    assert actual.shape == expected.shape, (
        f"Shape mismatch for '{pattern}': got {actual.shape}, expected {expected.shape}"
    )
    assert torch.allclose(actual.float(), expected.float()), (
        f"Value mismatch for '{pattern}': max diff = {(actual.float() - expected.float()).abs().max()}"
    )


# ─────────────────────────── 1. Permutation ───────────────────────────────────

class TestPermutation:
    def test_4d_transpose(self):
        check(torch.randn(2, 3, 4, 5), "b h w c -> b c h w")

    def test_3d_cycle(self):
        check(torch.randn(2, 3, 4), "a b c -> c a b")

    def test_identity_1d(self):
        check(torch.randn(5,), "a -> a")

    def test_identity_4d(self):
        check(torch.randn(2, 3, 4, 5), "a b c d -> a b c d")

    def test_reverse_4d(self):
        check(torch.randn(2, 3, 4, 5), "a b c d -> d c b a")

    def test_2d_swap(self):
        check(torch.randn(7, 11), "a b -> b a")


# ─────────────────────────── 2. Merge axes ────────────────────────────────────

class TestMerge:
    def test_merge_last_two(self):
        check(torch.randn(2, 3, 4), "a b c -> a (b c)")

    def test_merge_middle_two(self):
        check(torch.randn(2, 3, 4, 5), "b h w c -> b (h w) c")

    def test_flatten_all(self):
        check(torch.randn(2, 3, 4, 5), "b h w c -> (b h w c)")

    def test_merge_with_reorder(self):
        # merge non-adjacent axes into output group, with reorder
        check(torch.randn(2, 3, 4, 5), "b h w c -> b (c h) w")

    def test_merge_first_two(self):
        check(torch.randn(2, 3, 5), "a b c -> (a b) c")

    def test_merge_3_axes(self):
        check(torch.randn(2, 3, 4, 5), "a b c d -> a (b c d)")


# ─────────────────────────── 3. Split axes ────────────────────────────────────

class TestSplit:
    def test_split_middle_h_given(self):
        check(torch.randn(2, 12, 5), "b (h w) c -> b h w c", h=3)

    def test_split_middle_w_given(self):
        check(torch.randn(2, 12, 5), "b (h w) c -> b h w c", w=4)

    def test_split_first(self):
        check(torch.randn(12, 5), "(h w) c -> h w c", h=3)

    def test_split_then_merge(self):
        check(torch.randn(32, 30, 40, 3), "b (h1 h) (w1 w) c -> (b h1 w1) h w c", h1=2, w1=2)

    def test_split_into_3(self):
        check(torch.randn(2, 24, 5), "b (a b2 c) d -> b a b2 c d", a=2, b2=3)

    def test_split_last(self):
        check(torch.randn(2, 5, 12), "b c (h d) -> b c h d", h=4)


# ─────────────────────────── 4. Squeeze ───────────────────────────────────────

class TestSqueeze:
    def test_squeeze_parens_middle(self):
        check(torch.randn(2, 1, 5), "b () c -> b c")

    def test_squeeze_1_middle(self):
        check(torch.randn(2, 1, 5), "b 1 c -> b c")

    def test_squeeze_both_ends(self):
        check(torch.randn(1, 3, 1, 5), "() b () c -> b c")

    def test_squeeze_first(self):
        check(torch.randn(1, 3, 5), "() b c -> b c")

    def test_squeeze_last(self):
        check(torch.randn(3, 5, 1), "b c () -> b c")

    def test_1_inside_parens_ignored(self):
        check(torch.randn(6, 5), "(h 1 w) c -> h w c", h=2)


# ─────────────────────────── 5. Unsqueeze ─────────────────────────────────────

class TestUnsqueeze:
    def test_unsqueeze_parens_middle(self):
        check(torch.randn(2, 5), "b c -> b () c")

    def test_unsqueeze_parens_first(self):
        check(torch.randn(2, 5), "b c -> () b c")

    def test_unsqueeze_1_middle(self):
        check(torch.randn(2, 5), "b c -> b 1 c")

    def test_unsqueeze_last(self):
        check(torch.randn(2, 5), "b c -> b c ()")

    def test_unsqueeze_both_ends(self):
        check(torch.randn(5,), "c -> () c ()")


# ─────────────────────────── 6. Ellipsis — permutation ───────────────────────

class TestEllipsisPermutation:
    def test_4d_move_last_to_second(self):
        check(torch.randn(2, 3, 4, 5), "b ... h -> b h ...")

    def test_3d_move_last_to_second(self):
        check(torch.randn(2, 3, 5), "b ... h -> b h ...")

    def test_2d_no_middle(self):
        check(torch.randn(2, 5), "b ... h -> b h ...")

    def test_merge_ellipsis_trailing(self):
        check(torch.randn(2, 3, 4, 5), "... h w -> ... (h w)")

    def test_merge_all_trailing(self):
        check(torch.randn(2, 3, 4, 5), "b h ... -> b h (...)")

    def test_ellipsis_identity(self):
        check(torch.randn(2, 3, 4), "... -> ...")

    def test_ellipsis_leading(self):
        check(torch.randn(2, 3, 4, 5), "... c -> c ...")

    def test_ellipsis_sandwiched(self):
        check(torch.randn(2, 3, 4, 5, 6), "b ... c d -> b d ... c")


# ─────────────────────────── 7. Ellipsis — with split/merge ──────────────────

class TestEllipsisSplitMerge:
    def test_split_last_with_ellipsis(self):
        check(torch.randn(2, 3, 20), "b ... (h d) -> b h ... d", h=4)

    def test_split_second_with_ellipsis(self):
        check(torch.randn(2, 20, 3), "b (h d) ... -> b h d ...", h=4)

    def test_merge_ellipsis_dims(self):
        # merge middle batch dims
        check(torch.randn(2, 3, 4, 5), "b h w c -> b (h w) c")

    def test_split_and_move_with_ellipsis(self):
        check(torch.randn(4, 3, 16), "b ... (h d) -> (b h) ... d", h=4)


# ─────────────────────────── 8. Combinations ─────────────────────────────────

class TestCombinations:
    def test_split_merge_permute(self):
        check(torch.randn(32, 30, 40, 3), "b (h h1) (w w1) c -> b h w (c h1 w1)", h1=2, w1=2)

    def test_squeeze_plus_split(self):
        check(torch.randn(2, 1, 12), "b () (h w) -> b h w", h=3)

    def test_merge_plus_unsqueeze(self):
        check(torch.randn(2, 3, 4), "b h w -> b () (h w)")

    def test_split_plus_permute_plus_merge(self):
        # AF3-style head split + merge
        check(torch.randn(2, 7, 32), "b n (h d) -> b h n d", h=4)

    def test_split_plus_merge_back(self):
        check(torch.randn(2, 7, 32), "b n (h d) -> b n (d h)", h=4)

    def test_squeeze_permute_merge(self):
        check(torch.randn(2, 1, 3, 4), "b () h w -> b (h w)")


# ─────────────────────────── 9. Numerical correctness ─────────────────────────

class TestNumericalCorrectness:
    def test_permutation_element_positions(self):
        x = torch.arange(24).reshape(2, 3, 4).float()
        r = torch_rearrange(x, "a b c -> c a b")
        assert r[0, 0, 0] == x[0, 0, 0]   # c=0, a=0, b=0
        assert r[1, 0, 0] == x[0, 0, 1]   # c=1, a=0, b=0
        assert r[2, 1, 1] == x[1, 1, 2]   # c=2, a=1, b=1

    def test_merge_element_order(self):
        x = torch.arange(12).reshape(3, 4).float()
        r = torch_rearrange(x, "a b -> (a b)")
        assert r.tolist() == list(range(12))

    def test_merge_column_major_would_differ(self):
        # C-order: last axis varies fastest — einops matches numpy default
        x = torch.arange(6).reshape(2, 3).float()
        r = torch_rearrange(x, "a b -> (a b)")
        expected = einops.rearrange(x, "a b -> (a b)")
        assert torch.allclose(r, expected)

    def test_split_element_values(self):
        x = torch.arange(12).reshape(1, 12, 1).float()
        r = torch_rearrange(x, "b (h w) c -> b h w c", h=3)
        assert r.shape == (1, 3, 4, 1)
        assert r[0, 0, 0, 0] == 0
        assert r[0, 0, 3, 0] == 3
        assert r[0, 2, 3, 0] == 11

    def test_unsqueeze_inserts_1_dim(self):
        x = torch.arange(6).reshape(2, 3).float()
        r = torch_rearrange(x, "a b -> a () b")
        assert r.shape == (2, 1, 3)
        assert torch.allclose(r.squeeze(1), x)

    def test_squeeze_removes_1_dim(self):
        x = torch.arange(6).reshape(2, 1, 3).float()
        r = torch_rearrange(x, "a () b -> a b")
        assert r.shape == (2, 3)
        assert torch.allclose(r, x.squeeze(1))


# ─────────────────────────── 10. torch.compile compatibility ──────────────────

class TestTorchCompile:
    def test_dynamic_shapes_no_recompile(self):
        """10 unique sequence lengths must not crash past recompile_limit=8."""
        compiled = torch.compile(
            lambda t: torch_rearrange(t, "b ... h -> b h ..."),
            dynamic=True,
        )
        lengths = [47, 53, 67, 71, 83, 89, 97, 101, 113, 127]
        for L in lengths:
            t = torch.randn(2, L, 5)
            out = compiled(t)
            assert out.shape == (2, 5, L), f"Shape wrong for L={L}"

    def test_compile_split_pattern(self):
        compiled = torch.compile(
            lambda t: torch_rearrange(t, "b n (h d) -> b h n d", h=4),
            dynamic=True,
        )
        for n in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]:
            t = torch.randn(2, n, 32)
            out = compiled(t)
            assert out.shape == (2, 4, n, 8)

    def test_compile_merge_pattern(self):
        compiled = torch.compile(
            lambda t: torch_rearrange(t, "b h w c -> b (h w) c"),
            dynamic=True,
        )
        for h in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
            t = torch.randn(2, h, h, 8)
            out = compiled(t)
            assert out.shape == (2, h * h, 8)

    def test_compile_pair_bias_attn_pattern(self):
        """Historical pattern from pair_bias_attn.py that was crashing."""
        compiled = torch.compile(
            lambda t: torch_rearrange(t, "b ... h -> b h ..."),
            dynamic=True,
        )
        for n in [37, 41, 43, 47, 53, 59, 61, 67, 71, 73]:
            t = torch.randn(2, n, n, 8)
            out = compiled(t)
            assert out.shape == (2, 8, n, n)

    def test_compile_head_split_then_merge(self):
        """Simulate b ... (h d) -> b h ... d with dynamic seq len."""
        compiled = torch.compile(
            lambda t: torch_rearrange(t, "b n (h d) -> b h n d", h=8),
            dynamic=True,
        )
        for n in [16, 32, 64, 96, 128, 160, 192, 224, 256, 288]:
            t = torch.randn(2, n, 64)
            out = compiled(t)
            assert out.shape == (2, 8, n, 8)


# ─────────────────────────── 11. Historical Proteina usage patterns ───────────

class TestHistoricalPatterns:
    """Validate every real-world pattern from the Proteina codebase."""

    def test_pair_bias_attn_to_bias(self):
        # pair_bias_attn.py: [b,n,n,h] → [b,h,n,n]
        check(torch.randn(2, 7, 7, 4), "b ... h -> b h ...")

    def test_pair_bias_attn_qkv_split(self):
        # pair_bias_attn.py: [b,n,h*d] → [b,h,n,d]
        h = 4
        check(torch.randn(2, 7, h * 8), "b ... (h d) -> b h ... d", h=h)

    def test_pair_bias_attn_merge_heads(self):
        # pair_bias_attn.py: [b,h,n,d] → [b,n,h*d]
        check(torch.randn(2, 4, 7, 8), "b h n d -> b n (h d)")

    def test_pair_bias_attn_mask_unsqueeze(self):
        # pair_bias_attn.py: [b,n,n] → [b,1,n,n]
        check(torch.randn(2, 7, 7), "b i j -> b () i j")

    def test_protein_transformer_seq_unsqueeze(self):
        # protein_transformer.py: [b,n,d] → [b,1,n,d]
        check(torch.randn(2, 7, 64), "b n d -> b () n d")

    def test_protein_transformer_mask_unsqueeze(self):
        # protein_transformer.py: [b,n] → [b,1,n]
        check(torch.randn(2, 7), "b n -> b () n")

    def test_protein_transformer_seq_squeeze(self):
        # protein_transformer.py: [b,1,n,c] → [b,n,c]
        check(torch.randn(2, 1, 7, 64), "b () n c -> b n c")

    def test_align_utils_squeeze_d(self):
        # align_utils.py: [... 1 d] → [... d]
        check(torch.randn(2, 3, 1, 64), "... () d -> ... d")

    def test_designability_merge_batch(self):
        # designability.py: [n,s,t] → [(n*s), t]
        check(torch.randn(4, 8, 64), "n s t -> (n s) t")


# ─────────────────────────── 12. Error cases ──────────────────────────────────

class TestErrorCases:
    def test_missing_arrow(self):
        with pytest.raises(ValueError, match="->"):
            torch_rearrange(torch.randn(2, 3), "a b")

    def test_axis_only_on_left(self):
        with pytest.raises(ValueError, match="only one side"):
            torch_rearrange(torch.randn(2, 3), "a b -> a")

    def test_axis_only_on_right(self):
        with pytest.raises(ValueError, match="only one side"):
            torch_rearrange(torch.randn(2,), "a -> a b")

    def test_non_unitary_anonymous_axis(self):
        with pytest.raises(ValueError, match="Non-unitary anonymous axis"):
            torch_rearrange(torch.randn(6, 5), "(3 h) c -> h c")

    def test_multiple_unknowns_in_group(self):
        with pytest.raises(ValueError, match="Cannot infer multiple unknown axes"):
            torch_rearrange(torch.randn(2, 12, 5), "b (h w) c -> b h w c")  # no h= or w= given

    def test_nested_parens(self):
        with pytest.raises(ValueError, match="Nested parentheses"):
            torch_rearrange(torch.randn(2, 3), "((a b)) -> a b")

    def test_ellipsis_only_on_right(self):
        with pytest.raises(ValueError, match="Ellipsis found in right side but not left"):
            torch_rearrange(torch.randn(2, 3), "a b -> a ... b")

    def test_too_few_dims(self):
        with pytest.raises(Exception):
            torch_rearrange(torch.randn(2, 3), "a b c -> a b c")

    def test_unclosed_paren(self):
        with pytest.raises(ValueError, match="Unclosed"):
            torch_rearrange(torch.randn(2, 3), "(a b -> a b")

    def test_unmatched_close_paren(self):
        with pytest.raises(ValueError, match="Unmatched"):
            torch_rearrange(torch.randn(2, 3), "a b) -> a b")
