"""Phase 5 Step 5.1 — Empirical pre-validation for Module aliasing.

Test the central claim: two distinct torch.nn.Module instances that share
their parameter and buffer tensors will get isolated Dynamo compile caches
(including resume-frame continuations after graph breaks), so alternating
grad-enabled / no_grad calls on them never trigger grad_mode recompiles.

Outline:
  1. Build mod_a (Linear -> graph_break() -> LayerNorm -> Linear)
  2. Build mod_b = alias of mod_a (same params, separate Module)
  3. Compile both via torch.compile
  4. Run 10 alternating iters:
        mod_a(x) under grad   (training-like)
        mod_b(x) under no_grad (SC-like)
  5. Count Dynamo recompile events
  6. PASS: total recompiles fixed and small (≤ ~6 = a few cold compiles per side)
     FAIL: recompile count scales with iter count (thrash)
"""
from __future__ import annotations

import copy
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


CHILD_SCRIPT = textwrap.dedent('''
    import copy, sys, torch
    torch.manual_seed(0)


    def _make_nn_sc_alias(nn):
        sc = copy.deepcopy(nn)
        src_params = dict(nn.named_parameters(recurse=True))
        src_buffers = dict(nn.named_buffers(recurse=True))
        for full_name in list(src_params.keys()):
            parent_name, _, leaf = full_name.rpartition(".")
            parent = sc.get_submodule(parent_name) if parent_name else sc
            parent._parameters[leaf] = src_params[full_name]
        for full_name in list(src_buffers.keys()):
            parent_name, _, leaf = full_name.rpartition(".")
            parent = sc.get_submodule(parent_name) if parent_name else sc
            parent._buffers[leaf] = src_buffers[full_name]
        return sc


    class M(torch.nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.lin1 = torch.nn.Linear(d, d)
            self.ln = torch.nn.LayerNorm(d)
            self.lin2 = torch.nn.Linear(d, d)

        def forward(self, x):
            x = self.lin1(x)
            torch._dynamo.graph_break()
            x = self.ln(x)
            return self.lin2(x)


    mod_a = M().cuda()
    mod_b = _make_nn_sc_alias(mod_a).cuda()

    # Sanity: confirm params are SHARED tensors.
    for (na, pa), (nb, pb) in zip(mod_a.named_parameters(), mod_b.named_parameters()):
        assert pa is pb, f"alias broken at {na}: {pa.data_ptr()} != {pb.data_ptr()}"
    print("[alias] confirmed shared parameter tensors", flush=True)

    compiled_a = torch.compile(mod_a)
    compiled_b = torch.compile(mod_b)

    x = torch.randn(2, 64, device="cuda")
    for i in range(10):
        # Training-like: grad enabled on mod_a
        y = compiled_a(x)
        loss = y.pow(2).mean()
        loss.backward()
        mod_a.zero_grad(set_to_none=True)

        # SC-like: no_grad on mod_b (which shares params with mod_a)
        with torch.no_grad():
            _ = compiled_b(x)
        if i in (0, 1, 2, 9):
            print(f"[iter {i}] ok", flush=True)
    print("[done] 10 iterations complete", flush=True)
''').lstrip()


def main():
    script_path = Path(tempfile.mkstemp(suffix=".py")[1])
    script_path.write_text(CHILD_SCRIPT)
    env = dict(os.environ)
    env["TORCH_LOGS"] = "recompiles"
    env.setdefault("CUTLASS_PATH", "/home/ubuntu/openfold/cutlass")

    print(f"Running child with TORCH_LOGS=recompiles ...", flush=True)
    r = subprocess.run(
        ["/home/ubuntu/miniforge3/envs/cue_openfold/bin/python", str(script_path)],
        env=env, capture_output=True, text=True, check=False,
    )
    print("--- child stdout ---")
    print(r.stdout)
    print("--- child stderr (filtered: recompile/Recompil/done/alias/iter/Error/Trace) ---")
    interesting = [
        line for line in r.stderr.splitlines()
        if re.search(r"Recompil|recompil|done|alias|iter |Error|Traceback", line)
    ]
    for line in interesting:
        print(line)

    # Count recompiles
    n_recompiles = sum(1 for line in r.stderr.splitlines() if "Recompiling function" in line)
    # Find max variant count per frame
    variants = {}
    for line in r.stderr.splitlines():
        m = re.search(r"\[(\d+)/(\d+)\]", line)
        if m and "Recompiling function" in line:
            fid, vid = int(m.group(1)), int(m.group(2))
            variants[fid] = max(variants.get(fid, 0), vid + 1)
    max_var = max(variants.values()) if variants else 0

    print()
    print("=" * 60)
    print(f"Recompile events:      {n_recompiles}")
    print(f"Max variants per frame: {max_var}")
    print(f"Per-frame variants:    {variants}")
    print("=" * 60)
    # PASS: recompiles fixed, modest count. Scaling-with-iters would be > 20.
    PASS_THRESH = 12
    if n_recompiles <= PASS_THRESH:
        print(f"PASS: recompile count {n_recompiles} <= {PASS_THRESH} threshold")
        sys.exit(0)
    else:
        print(f"FAIL: recompile count {n_recompiles} > {PASS_THRESH} threshold (likely thrash)")
        sys.exit(1)


if __name__ == "__main__":
    main()
