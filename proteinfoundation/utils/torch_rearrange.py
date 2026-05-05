"""
torch_rearrange: A torch.compile-compatible reimplementation of einops.rearrange.

Unlike einops, this function caches only the pure-Python pattern-parsing step and
executes tensor operations with native torch primitives.  This avoids the SymInt
hashing failure that occurs when einops tries to use tensor shapes as LRU-cache
keys inside torch.compile's symbolic-shape tracing.

Algorithm (mirrors einops internals):
    tensor.reshape(init_shapes)  →  .permute(perm)  →  .reshape(final_shapes)

  init_shapes   — splits composite input groups, drops size-1 "() / 1" dims
  perm          — reorders elementary axes to match output order
  final_shapes  — merges composite output groups, inserts size-1 "() / 1" dims

Supported features (exact parity with einops.rearrange):
  • Axis permutation           "b h w c -> b c h w"
  • Merge axes                 "b h w -> b (h w)"
  • Split axis (with kwarg)    "b (h w) c -> b h w c", h=3
  • Ellipsis                   "b ... h -> b h ..."
  • Squeeze  ()  or  1         "b () c -> b c",  "b 1 c -> b c"
  • Unsqueeze  ()  or  1       "b c -> b () c",  "b c -> b 1 c"
  • 1 inside parentheses       "(h 1 w) -> h w"  (silently ignored, per einops)
  • All of the above combined  "b (h h1) (w w1) c -> b h w (c h1 w1)", h1=2, w1=2

Not supported (einops.rearrange also rejects these):
  • Non-unitary anonymous axes (e.g. "(3 h)")
  • Ellipsis inside parentheses on the left side
  • Axis names that appear on only one side
"""

import functools
from typing import Dict, List, Optional, Tuple

import torch

# ──────────────────────────── sentinel ────────────────────────────────────────
_ELLIPSIS = "__ELLIPSIS__"   # string sentinel used inside composition lists


# ──────────────────────────── parser ──────────────────────────────────────────

def _parse_expr(expr: str) -> list:
    """Parse one side of an einops pattern into a composition list.

    Each element of the returned list is one of:
        []                  — empty group: () or standalone '1'  (size-1 slot)
        ['name', ...]       — group of one or more axis names
        _ELLIPSIS           — ellipsis sentinel

    Examples
    --------
    "b h w c"    → [['b'], ['h'], ['w'], ['c']]
    "b (h w) c"  → [['b'], ['h', 'w'], ['c']]
    "b () c"     → [['b'], [], ['c']]
    "b 1 c"      → [['b'], [], ['c']]
    "b ... c"    → [['b'], _ELLIPSIS, ['c']]
    "(h 1 w)"    → [['h', 'w']]          # '1' inside () is silently ignored
    """
    expr = expr.strip()
    result: list = []
    bracket_group: Optional[list] = None
    i = 0

    while i < len(expr):
        ch = expr[i]

        if ch == ' ':
            i += 1
            continue

        if ch == '(':
            if bracket_group is not None:
                raise ValueError("Nested parentheses are not allowed in einops patterns")
            bracket_group = []
            i += 1
            continue

        if ch == ')':
            if bracket_group is None:
                raise ValueError("Unmatched ')' in einops pattern")
            result.append(bracket_group)
            bracket_group = None
            i += 1
            continue

        if ch == '.' and expr[i : i + 3] == '...':
            if bracket_group is not None:
                bracket_group.append(_ELLIPSIS)   # allowed on right side: (...) merges ellipsis dims
            else:
                result.append(_ELLIPSIS)
            i += 3
            continue

        # read identifier or numeric literal
        j = i
        while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
            j += 1
        token = expr[i:j]
        i = j

        if not token:
            raise ValueError(f"Unexpected character {ch!r} in einops pattern")

        if token.isdecimal():
            n = int(token)
            if n != 1:
                raise ValueError(
                    f"Non-unitary anonymous axis '{token}' is not supported in rearrange. "
                    "Only '1' or '()' may be used to denote size-1 dimensions."
                )
            # '1' → size-1 slot
            if bracket_group is None:
                result.append([])   # standalone: empty group
            # else inside (): silently ignored per einops spec
        elif token.isidentifier():
            if bracket_group is None:
                result.append([token])
            else:
                bracket_group.append(token)
        else:
            raise ValueError(f"Invalid axis token {token!r} in einops pattern")

    if bracket_group is not None:
        raise ValueError("Unclosed '(' in einops pattern")

    return result


# ──────────────────────────── static recipe ───────────────────────────────────

class _StaticRecipe:
    """Immutable, hashable description of a rearrange operation.

    Computed once per (pattern, axes_lengths) combination and cached.
    Contains only Python primitives — no tensors, no SymInts.
    """
    __slots__ = (
        "left_groups", "right_groups",
        "known_sizes",
        "has_ellipsis", "ellipsis_pos_left",
        "n_concrete_left",
        "left_named_axes",  # ordered list of named axes from left (excl. empty groups)
    )

    def __init__(self, left_groups, right_groups, known_sizes,
                 has_ellipsis, ellipsis_pos_left, n_concrete_left, left_named_axes):
        self.left_groups = left_groups
        self.right_groups = right_groups
        self.known_sizes = known_sizes
        self.has_ellipsis = has_ellipsis
        self.ellipsis_pos_left = ellipsis_pos_left
        self.n_concrete_left = n_concrete_left
        self.left_named_axes = left_named_axes


@functools.lru_cache(maxsize=256)
def _build_static_recipe(pattern: str, axes_lengths_items: tuple) -> _StaticRecipe:
    """Parse pattern and validate.  Cached per (pattern, sorted axes_lengths)."""
    if '->' not in pattern:
        raise ValueError(f"einops pattern must contain '->': {pattern!r}")
    left_str, right_str = pattern.split('->', 1)
    left_groups = _parse_expr(left_str)
    right_groups = _parse_expr(right_str)

    known_sizes: Dict[str, int] = dict(axes_lengths_items)

    # ── basic structural validation ─────────────────────────────────────────
    has_ellipsis_left  = _ELLIPSIS in left_groups
    # ellipsis on right: either as standalone sentinel or inside a group: (...)
    has_ellipsis_right = _ELLIPSIS in right_groups or any(
        isinstance(g, list) and _ELLIPSIS in g for g in right_groups
    )

    # Left side must not have ellipsis inside parentheses
    for group in left_groups:
        if isinstance(group, list) and _ELLIPSIS in group:
            raise ValueError(
                f"Ellipsis is not allowed inside parentheses on the left side of pattern: {pattern!r}"
            )

    if has_ellipsis_right and not has_ellipsis_left:
        raise ValueError(
            f"Ellipsis found in right side but not left side of pattern: {pattern!r}"
        )

    # Collect named axis sets (skip _ELLIPSIS sentinels wherever they appear)
    def _names(groups) -> set:
        s = set()
        for g in groups:
            if g is _ELLIPSIS or g == _ELLIPSIS:
                continue
            for name in g:
                if name != _ELLIPSIS:
                    s.add(name)
        return s

    left_names  = _names(left_groups)
    right_names = _names(right_groups)
    diff = left_names.symmetric_difference(right_names)
    if diff:
        raise ValueError(
            f"Axis names appear on only one side of '->': {diff}  (pattern: {pattern!r})"
        )

    # Validate: at most one unknown per input group (only one can be inferred)
    for group in left_groups:
        if group == _ELLIPSIS or len(group) <= 1:
            continue
        unknowns = [name for name in group if name not in known_sizes]
        if len(unknowns) > 1:
            raise ValueError(
                f"Cannot infer multiple unknown axes {unknowns} in group {group}. "
                "Provide sizes for all but one via keyword arguments."
            )

    # Ordered list of all named axes from left (for elementary axis ordering)
    left_named_axes: List[str] = []
    for group in left_groups:
        if group == _ELLIPSIS:
            continue
        for name in group:
            left_named_axes.append(name)

    ellipsis_pos_left = (
        left_groups.index(_ELLIPSIS) if has_ellipsis_left else -1
    )
    n_concrete_left = sum(1 for g in left_groups if g != _ELLIPSIS)

    return _StaticRecipe(
        left_groups=left_groups,
        right_groups=right_groups,
        known_sizes=known_sizes,
        has_ellipsis=has_ellipsis_left,
        ellipsis_pos_left=ellipsis_pos_left,
        n_concrete_left=n_concrete_left,
        left_named_axes=left_named_axes,
    )


# ──────────────────────────── execution ───────────────────────────────────────

def _expand_ellipsis(groups: list, ellipsis_names: List[str]) -> list:
    """Replace the _ELLIPSIS sentinel with one singleton group per ellipsis dim."""
    result = []
    for g in groups:
        if g == _ELLIPSIS:
            # standalone ellipsis → one singleton group per ellipsis dim
            for name in ellipsis_names:
                result.append([name])
        elif isinstance(g, list) and _ELLIPSIS in g:
            # (...) on right side → expand _ELLIPSIS in-place within the group
            new_group = []
            for item in g:
                if item == _ELLIPSIS:
                    new_group.extend(ellipsis_names)
                else:
                    new_group.append(item)
            result.append(new_group)
        else:
            result.append(g)
    return result


def _execute_recipe(
    tensor: torch.Tensor,
    recipe: _StaticRecipe,
) -> torch.Tensor:
    """Apply the rearrange recipe to a concrete tensor."""

    # ── 1. Expand ellipsis ────────────────────────────────────────────────────
    if recipe.has_ellipsis:
        ellipsis_ndim = tensor.ndim - recipe.n_concrete_left
        if ellipsis_ndim < 0:
            raise ValueError(
                f"Tensor has {tensor.ndim} dimensions but pattern requires "
                f"at least {recipe.n_concrete_left} concrete dimensions"
            )
        ellipsis_names = [f"__e{i}" for i in range(ellipsis_ndim)]
        left_groups  = _expand_ellipsis(recipe.left_groups,  ellipsis_names)
        right_groups = _expand_ellipsis(recipe.right_groups, ellipsis_names)
        # elementary axes: recipe's named axes + ellipsis axes inserted at the
        # correct position (matching left_groups order after expansion)
        elementary_axes: List[str] = []
        for g in left_groups:
            for name in g:   # empty groups contribute nothing
                elementary_axes.append(name)
    else:
        left_groups  = recipe.left_groups
        right_groups = recipe.right_groups
        elementary_axes = list(recipe.left_named_axes)

    # ── 2. Infer axis sizes from tensor.shape ─────────────────────────────────
    axis_sizes: Dict[str, object] = dict(recipe.known_sizes)   # may hold SymInts
    tensor_dim = 0
    for group in left_groups:
        dim_size = tensor.shape[tensor_dim]
        if len(group) == 0:
            # size-1 slot: consume the dim, add nothing to elementary axes
            tensor_dim += 1
            continue
        if len(group) == 1:
            # single axis: size comes directly from tensor shape (if not in kwargs)
            name = group[0]
            if name not in axis_sizes:
                axis_sizes[name] = dim_size
        else:
            # composite group: infer the one unknown axis by division
            known_product = 1
            unknown_name: Optional[str] = None
            for name in group:
                if name in axis_sizes:
                    known_product = known_product * axis_sizes[name]
                else:
                    unknown_name = name
            if unknown_name is not None:
                axis_sizes[unknown_name] = dim_size // known_product
        tensor_dim += 1

    # ── 3. Build init_shapes and reshape to elementary axes ───────────────────
    init_shapes = [axis_sizes[name] for name in elementary_axes]
    t = tensor.reshape(init_shapes)

    # ── 4. Compute permutation ────────────────────────────────────────────────
    # For each non-empty right group, collect its constituent axis names in order
    output_elem_order: List[str] = []
    for group in right_groups:
        for name in group:     # empty groups → nothing
            output_elem_order.append(name)

    perm: List[int] = [elementary_axes.index(name) for name in output_elem_order]

    # ── 5. Permute (skip if already in order) ─────────────────────────────────
    if perm != list(range(len(perm))):
        t = t.permute(*perm)

    # ── 6. Build final_shapes and reshape to output ───────────────────────────
    final_shapes: list = []
    for group in right_groups:
        if len(group) == 0:
            final_shapes.append(1)
        else:
            size: object = 1
            for name in group:
                size = size * axis_sizes[name]
            final_shapes.append(size)

    return t.reshape(final_shapes)


# ──────────────────────────── public API ──────────────────────────────────────

def torch_rearrange(tensor: torch.Tensor, pattern: str, **axes_lengths) -> torch.Tensor:
    """Native torch reimplementation of ``einops.rearrange``.

    Signature is identical to ``einops.rearrange`` for tensor inputs.
    Safe to use inside ``torch.compile``-compiled functions with dynamic shapes.

    Parameters
    ----------
    tensor : torch.Tensor
    pattern : str
        Rearrangement pattern, e.g. ``"b h w c -> b c h w"``.
    **axes_lengths : int
        Sizes for axes that cannot be inferred from the tensor shape alone
        (e.g. when splitting a composite dimension).

    Examples
    --------
    >>> torch_rearrange(x, "b h w c -> b c h w")              # permute
    >>> torch_rearrange(x, "b h w -> b (h w)")                # merge
    >>> torch_rearrange(x, "b (h w) c -> b h w c", h=4)      # split
    >>> torch_rearrange(x, "b ... h -> b h ...")              # ellipsis
    >>> torch_rearrange(x, "b () c -> b c")                   # squeeze
    >>> torch_rearrange(x, "b c -> b () c")                   # unsqueeze
    """
    recipe = _build_static_recipe(pattern, tuple(sorted(axes_lengths.items())))
    return _execute_recipe(tensor, recipe)
