# Proteina Attention & Self-Conditioning Optimization — Summary

**Status:** Phase 1 (attention backends) and Phase 2 (eager-path / SC-compile) complete and verified. Phase 3 (no-deepcopy investigation) systematically narrowed the root cause to "double-forward of `nn` disconnects 9 specific feature-projection layers from the autograd graph"; the underlying PyTorch internals mechanism remains unidentified and is the recommended follow-up.

This document is the canonical reference for what was tried, what works, and what remains. Updated as work progresses.

---

## 1. Context

The hot path of training is `ProteinTransformerAF3._forward_impl`, dominated by `PairBiasAttention` calls inside `MultiheadAttnAndTransition` layers. Original implementation was hand-coded `einsum + masked_fill + softmax`. Two related performance issues motivated this work:

1. **Attention backend choice.** SDPA, Flash, and cuEquivariance were installed in the `cue_openfold` env but unused. The vanilla einsum path was leaving substantial GPU time on the table.
2. **Self-conditioning (SC) and validation ran eager** while training was `torch.compile`d, because compiling both together caused Dynamo cache thrash on `torch.is_grad_enabled()`. SC compile was disabled by setting `nn_sc.use_torch_compile = False` (see the historical comment in `proteina.py:153-160`). This left a real performance gap on the eager paths.

## 2. Phase 1 — Attention backends (complete)

**Files modified:**
- `proteinfoundation/nn/pair_bias_attn/pair_bias_attn.py`
- `proteinfoundation/nn/protein_transformer.py`

**What was added:**
- `attn_impl` parameter on `PairBiasAttention` selecting one of: `vanilla`, `sdpa`, `flash`, `cuequivariance`.
- `_attn_vanilla` (existing logic preserved), `_attn_sdpa`, `_attn_flash`, `_forward_cueq` dispatchers.
- Propagation through `MultiHeadBiasedAttentionADALN_MM` → `MultiheadAttnAndTransition` → `ProteinTransformerAF3`.
- Validation of constraints at construction (flash + pair_dim is rejected; cuEq + dim_head%32!=0 is rejected; cuEq import errors are surfaced clearly).

**Benchmark:** `scripts/benchmark_attention.py`
- Parity (fp32, B=2, N=256, all valid tokens): max |diff| ≤ 1e-5 for SDPA and cuEq; flash uses bf16 internally so its tolerance is 2e-3.
- Speed/VRAM (bf16, B=4, L=128/256/384, 16 replicas, with and without `torch.compile`): empirical curves fit `T ≈ a·L² + b` and `VRAM ≈ a·L² + b`. Flash gives O(L) VRAM scaling, the others O(L²).

**Headline results** (A100-SXM4-40GB, B=4):
- `vanilla+compile` is the speed winner for pair-biased attention (~0.80 ms at L=384). Inductor fuses QKV proj + attn + gate + out into one Triton kernel.
- `cuequivariance+eager` outperforms `vanilla+eager` by ~1.67× at L=384 (2.39 → 1.43 ms). Under compile, cuEq only gains 1.1-1.4× because the Triton kernel is opaque to Inductor.
- Flash gives a ~4.5× speedup and ~23× less VRAM at L=384, but lacks pair-bias support, so it's only viable where pair bias isn't required.

**Recommended config combination:**
```yaml
model.nn.attn_impl: "vanilla"               # for compiled training path
model.nn.attn_impl_eager: "cuequivariance"  # for SC + validation eager paths
```
The second key is the Phase 2 mechanism (see below).

## 3. Phase 2 — Eager-path override (`attn_impl_eager`) + SC compile (`use_torch_compile_sc`)

### 3a. `attn_impl_eager` (complete and verified)

**Problem:** Dynamo specializes on every `if self.attn_impl == "..."` branch under `torch.compile`. Changing `attn_impl` at runtime invalidates the compile cache. We need a *second* attribute that Dynamo never sees.

**Mechanism** (`pair_bias_attn.py`):
```python
def _get_impl(self) -> str:
    if not torch.compiler.is_compiling() and self.attn_impl_eager is not None:
        return self.attn_impl_eager
    return self.attn_impl
```

Under `torch.compile`, `is_compiling()==True`; Dynamo takes the `else` branch, specializes on `self.attn_impl`, and never observes `attn_impl_eager`. Outside compile, when the override is set, the eager backend is returned.

Propagated through `MultiHeadBiasedAttentionADALN_MM` → `MultiheadAttnAndTransition` → `ProteinTransformerAF3` kwargs. Also: module-level `set_attn_impl_eager(module, name)` helper for post-load swap.

**Tests** (`scripts/test_sc_compile_dispatcher.py` — fast unit test, ~5 s; `scripts/test_sc_compile_q1_q2.py` — extended):
- Eager call correctly routed to override backend
- `set_attn_impl_eager()` flips backend across all `PairBiasAttention` instances
- Toggling override mid-run produces **0 recompiles** (`unique_graphs` stays at 1)
- Validation rejects invalid override values
- All four backends still pass the existing fp32 parity test

### 3b. `use_torch_compile_sc` (complete and verified)

**Problem:** Compiling `_forward_impl` for both grad=True (training) and grad=False (SC) thrashes Dynamo's cache via `GLOBAL_STATE changed: grad_mode` guard failures.

**Mechanism** (`protein_transformer.py`):
```python
def _forward_impl_sc(self, batch_nn):
    """Thin wrapper — distinct __code__ gives Dynamo a separate cache from _forward_impl.
    Only ever called under torch.no_grad(), so its cache specializes on grad=False only."""
    return self._forward_impl(batch_nn)
```

Plus a third lazy-compile branch in `ProteinTransformerAF3.forward()`:
```python
if self.use_torch_compile_sc and not torch.is_grad_enabled():
    if getattr(self, "_forward_compiled_sc", None) is None:
        self._forward_compiled_sc = torch.compile(
            self._forward_impl_sc, dynamic=True
        )
    return self._forward_compiled_sc(batch_nn)
```

`proteina.py` reads `training.use_torch_compile_sc` and sets it on `nn_sc` (when deepcopy exists) or on `nn` (when it doesn't).

**Tests** (`scripts/test_sc_compile.py` — full e2e on overfit data):
- Compiled SC outputs are byte-identical to eager SC outputs (20/20 train_loss_step matches)
- Train cache and SC cache are SEPARATE Dynamo cache slots (`unique_graphs` = 2)
- 37 one-time warmup recompiles (dtype/grad-mode variants); max-variants-per-frame = 4 < cache_size_limit=8 → no thrash
- No DDP `find_unused_params` errors

### 3c. Q1 & Q2 outcomes (clarification questions from the user)

- **Q1: Intermittent SC invocation.** Verified safe (`test_sc_compile_q1_q2.py` Q1 section). The compile object is lazy and persists; non-SC minibatches don't touch it; a gap of 5 non-SC minibatches followed by an SC call reuses the cached graph (0 recompiles).
- **Q2: Can the `nn_sc` deepcopy be removed?** **No, not in production.** The deepcopy serves two independent roles:
  - *Compile-cache isolation* — now also covered by `_forward_impl_sc`.
  - *Autograd graph integrity* — **still required**; investigated in Phase 3 below.

## 4. Phase 3 — No-deepcopy investigation (complete, mechanism unknown)

### 4a. The empirical contradiction

When `self_cond_use_copy=False`, training fails at minibatch 8 with DDP `find_unused_params`. When `self_cond_use_copy=True` (production default), training proceeds cleanly. The only behavioral difference between the two configurations:

| Mode | Per-minibatch `nn.forward()` calls |
|---|---|
| deepcopy | **1×** — training only; SC goes through separate `nn_sc` |
| no-deepcopy | **1×** (non-SC iters) or **2×** (SC iters: no_grad SC + grad training) |

Both modes produce *identical losses* for the first 8 minibatches before no-deepcopy mode crashes, so the gradient math is provably the same.

### 4b. Decisive hypothesis sweep

| Test | Compile | SC | Deepcopy | Checkpoint | Outcome |
|---|---|---|---|---|---|
| baseline | ON | ON | **ON** | ON | ✅ 20 steps |
| **H1** | ON | **OFF** | OFF | ON | ✅ 12 steps → **double-forward is the proximate cause** |
| Q2 ctrl | ON or OFF | ON | OFF | ON | ❌ DDP find_unused @ iter 8 |
| **H1b** | OFF | ON | OFF | ON | ❌ CheckpointError @ iter 1 |
| **H1c** | ON | ON | OFF | **OFF** | ❌ DDP find_unused @ iter 8 |
| **H1d** | OFF | ON | OFF | **OFF** | ❌ DDP find_unused @ iter 8 |

The CheckpointError in H1b *masks* the same underlying DDP failure that fires in H1d once checkpointing is removed — confirmed by H1c (compile + no-checkpoint still fails) and H1d (eager + no-checkpoint fails the same way).

### 4c. The 9 disconnected parameters (D1 grad audit)

When `nn.forward` runs twice per minibatch (no-deepcopy + SC active), the following parameters report `grad is None` at SC-active iterations — meaning they are **not in the autograd graph** of the second (training) forward:

```
cond_factory._individual_factory.feat_creators.2.linear_embed.weight
cond_factory._individual_factory.projections.dssp_sc.weight
cond_factory._individual_factory.projections.fold_emb.weight
cond_factory._individual_factory.projections.time_emb.weight
init_repr_factory._individual_factory.projections.chain_break_per_res.weight
pair_repr_builder.init_repr_factory._individual_factory.feat_creators.1.linear_embed.weight
pair_repr_builder.init_repr_factory._individual_factory.projections.contact_map_sc.weight
pair_repr_builder.init_repr_factory._individual_factory.projections.rel_seq_sep.weight
pair_repr_builder.cond_factory._individual_factory.projections.time_emb.weight
```

All 9 are `nn.Linear` weights inside `IndividualFeatureFactory.projections` (or `feat_creators[*].linear_embed`). The owning code is `feature_factory.py:914-933`:

```python
for name, creator in zip(self.feat_names, self.feat_creators):
    raw = creator(batch)
    proj = self.projections[name](raw)
    output = output + proj
```

This is *pure deterministic PyTorch* — no `self.training`, no dropout, no buffer mutation, no conditional execution. Yet exactly these 9 projections are pruned from the autograd graph at SC-active iterations specifically when nn is invoked twice on the same instance.

In **deepcopy mode**, D1 grad audit prints "all 637 params have nonzero grad" at several steps. In **no-deepcopy mode**, it never does — the 9 specific projections are always missing at SC-active iterations.

### 4d. Hypotheses ruled out

| Hypothesis | How ruled out |
|---|---|
| Cross-rank RNG desync | Single-GPU runs (`[rank0]` only in logs) |
| `update_pair_repr_every_n > 1` sparse layer init | Config has `=1`; all layers present |
| `freeze_ipa` masking | `0 Non-trainable params` in log (coors_3d_decoder is None) |
| `torch.compile` | H1c (DDP error fires with compile + no-checkpoint) |
| Gradient checkpointing | H1d (DDP error fires with eager + no-checkpoint) |
| cuequivariance custom ops | `use_cueq=False` in this config — uses OpenFold tri_mult |

### 4e. Conclusion and recommendation

**Keep `self_cond_use_copy=True` (the production default).** The deepcopy is necessary for autograd-graph integrity in the presence of intermittent SC scheduling, independent of the compile-cache fix from Phase 2.

`proteina.py` was updated to warn users who set `self_cond_use_copy=False`, pointing them at `strategy='ddp_find_unused_parameters_true'` if they really want to disable the deepcopy.

## 5. Files modified and created

### Source
- `proteinfoundation/nn/pair_bias_attn/pair_bias_attn.py` — Phase 1 backends + Phase 2 `attn_impl_eager`
- `proteinfoundation/nn/protein_transformer.py` — Propagation + Phase 2 SC compile branch
- `proteinfoundation/proteinflow/proteina.py` — Phase 2 `use_torch_compile_sc` wiring + no-deepcopy DDP caveat comment

### Tests and benchmarks
- `scripts/benchmark_attention.py` — Phase 1 parity + speed/VRAM benchmark across all 4 backends
- `scripts/test_sc_compile_dispatcher.py` — Phase 2 fast mechanism unit test (~5 s)
- `scripts/test_sc_compile.py` — Phase 2 e2e correctness test on overfit data (~15 min, 2 compile cycles)
- `scripts/test_sc_compile_q1_q2.py` — Q1 (intermittent SC) + Q2 (deepcopy isolation) unit tests
- `scripts/debug_ddp_unused_params.py` — Phase 3 hypothesis sweep harness (H1, H1b, H1c, H1d, D1, D2, etc.)
- `scripts/_ddp_debug_runner.py` — Phase 3 monkey-patch runner for `L.Trainer`, `DDPStrategy`, and `torch.utils.checkpoint`

### Plan / investigation log
- `/home/ubuntu/.claude/plans/this-is-an-optimization-floofy-stardust.md` — full plan + Phase 3 investigation results

## 6. Reproducible commands

```bash
# Phase 1 — attention parity (5 s):
cd /home/ubuntu/proteina && \
  /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \
    scripts/benchmark_attention.py --parity-only

# Phase 1 — attention speed/VRAM benchmark (~3 min):
cd /home/ubuntu/proteina && \
  /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \
    scripts/benchmark_attention.py

# Phase 2 — fast SC-compile mechanism unit test (5 s):
/home/ubuntu/miniforge3/envs/cue_openfold/bin/python \
  /home/ubuntu/proteina/scripts/test_sc_compile_dispatcher.py

# Phase 2 — Q1 + Q2 unit tests (5 s):
/home/ubuntu/miniforge3/envs/cue_openfold/bin/python \
  /home/ubuntu/proteina/scripts/test_sc_compile_q1_q2.py

# Phase 2 — full e2e (~15 min):
cd /home/ubuntu/proteina && \
  /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \
    scripts/test_sc_compile.py

# Phase 3 — hypothesis sweep (~25 min for all six):
cd /home/ubuntu/proteina && \
  /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \
    scripts/debug_ddp_unused_params.py H1 H1b H1c H1d D1 D2

# Phase 3 — single hypothesis:
cd /home/ubuntu/proteina && \
  /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \
    scripts/debug_ddp_unused_params.py H1d
```

## 7. Follow-up worth doing

The 9-parameter autograd disconnect under double-forward is empirically reproducible but mechanistically unexplained. To root-cause it:

### 7a. Bare-PyTorch reproduction

Strip Lightning, Hydra, torch.compile, gradient checkpointing — build a *minimal* harness that constructs `ProteinTransformerAF3` from kwargs directly, builds one batch of fake data, and runs:

```python
# baseline (matches deepcopy mode behavior)
loss1 = nn(batch).backward()
g1 = {name: p.grad for name, p in nn.named_parameters()}

# reset grads
for p in nn.parameters(): p.grad = None

# repro (matches no-deepcopy + SC behavior)
with torch.no_grad():
    _ = nn(batch)   # SC pass
loss2 = nn(batch).backward()
g2 = {name: p.grad for name, p in nn.named_parameters()}

# diff
for name in g1:
    if (g1[name] is None) != (g2[name] is None):
        print(name, "g1=", g1[name], "g2=", g2[name])
```

If `g2[some_proj.weight]` is `None` while `g1[some_proj.weight]` is not, the disconnect reproduces outside the Lightning/DDP stack. That confirms the bug is in the model + torch interaction, and the next step is to bisect.

### 7b. Autograd graph dump

For both scenarios in 7a, print the autograd graph reaching `loss` via:

```python
def walk(grad_fn, depth=0, seen=None):
    seen = seen or set()
    if grad_fn is None or grad_fn in seen: return
    seen.add(grad_fn)
    print("  " * depth, type(grad_fn).__name__)
    for next_fn, _ in grad_fn.next_functions:
        walk(next_fn, depth+1, seen)

walk(loss.grad_fn)
```

Compare the trees. If the second scenario is missing certain `AddmmBackward` / `LinearBackward` nodes, we can identify exactly which projection chains are absent and trace upward to where they were elided.

### 7c. PyTorch issue search

Search the PyTorch issue tracker / forum for "double forward no_grad then grad parameter disconnect" or similar. This may be a known quirk involving:
- `nn.Linear`'s autograd specialization
- `torch.no_grad` context leaving stale autograd state
- The interaction between `requires_grad` propagation and `nn.Module` parameter caching

### 7d. If the disconnect reproduces in bare PyTorch (7a)

Bisect by progressively simplifying the model:
1. Remove most layers — keep only one `MultiheadAttnAndTransition` block + the conditioning/init factories
2. If it still reproduces: narrow further to just the factories
3. If only the factories reproduce: try replacing `IndividualFeatureFactory` with `FeatureFactory` (concatenation-mode instead of individual-projection-mode) — see `feature_factory.py:685` for the dispatcher

The goal is a 20-line reproduction that disconnects a single `nn.Linear.weight` from autograd via the "no_grad call, then grad call" pattern.

### 7e. Successful fix (if found)

If the underlying mechanism is identified and addressable, the practical benefit:
- ~50% reduction in model parameter memory footprint (no nn_sc duplicate)
- Live-weights SC predictions (currently lagged by `training.self_cond_copy_update_every` = 10 steps)

The fix would likely be small (clearing some stale autograd state at the boundary, or a context manager pattern around the SC call). Once it lands, update `proteina.py` to remove the DDP caveat comment and consider changing the default `self_cond_use_copy` to `False`.

---

## 8. Root cause identified (Phase 4 — fix landed)

The autograd disconnect from §4c is PyTorch issue [#105211](https://github.com/pytorch/pytorch/issues/105211): under an outer `torch.autocast(bf16)` context, weights cast inside a `torch.no_grad()` block are written into the autocast cache with `requires_grad=False`; on the next grad-enabled forward those cached no-grad weights are reused for the same `nn.Linear`, so the original `weight` parameter never enters the new autograd graph and `param.grad is None` after backward. Lightning's `bf16-mixed` `PrecisionPlugin` wraps the entire `training_step` in an autocast context, and `_set_self_cond` performs the SC forward inside that context. The deepcopy mode masked the bug because `nn_sc`'s weights are distinct tensor objects from `self.nn`'s, so the cache pollution affected only `nn_sc` and never collided with the training forward's weights.

### 8a. Reproduction outside Proteina

`scripts/debug_autograd_outer_autocast.py` (~110 lines, no Lightning, no Hydra, no DDP, no checkpointing) instantiates `ProteinTransformerAF3` directly, loads a captured batch, runs one SC no_grad forward and one grad forward both under a single outer `torch.autocast(bf16)`, and reports `param.grad is None` for the same 9 §4c parameters. Without the outer autocast (or with autocast applied separately to each forward — see `debug_autograd_bare_pytorch.py --bf16`), the disconnect does not appear. This confirms the bug is autocast-cache-driven, not a property of the SC pattern itself.

The diff between scenario A (single grad forward) and scenario B (no_grad then grad) under the single outer autocast matches the D1 grad-audit results from §4c bit-for-bit (315 total None grads on SC iterations, exactly 9 of them in the §4c canonical list; remaining 306 are coord-only / dssp-only paths skipped by the simplified test loss).

### 8b. Fix

`proteinfoundation/proteinflow/model_trainer_base.py:_set_self_cond` (around line 967), immediately after the `with torch.no_grad():` block exits:

```python
# PyTorch #105211: under an outer autocast context (Lightning bf16-mixed),
# weights cast inside the no_grad SC pass get cached with requires_grad=False
# and reuse in the next grad forward disconnects params from autograd. Only
# bites when sc_model is self.nn (no-deepcopy); harmless otherwise.
torch.clear_autocast_cache()
```

`torch.clear_autocast_cache()` is a no-op when no autocast cache exists (fp32 training, no SC fired, deepcopy mode with separate `nn_sc` weights), so the fix is safe to leave unconditional. An alternative — wrap the SC body in `torch.autocast(..., cache_enabled=False)` — also works but hardcodes device and dtype; the unconditional cache clear is one line, robust to any future precision choice, and points reviewers to the upstream issue.

### 8c. Verification

`scripts/debug_ddp_unused_params.py H1 H1b H1c H1d D1 D2` after the fix (all pass; counts are `(returncode, DDP unused error, losses)`):

| Test | Before fix | After fix |
|---|---|---|
| H1 (no-SC + no-deepcopy) | passing | (0, False, 24) |
| H1b (no-compile + no-deepcopy + SC) | (1, True, 8) | (0, False, 24) |
| H1c (no-checkpoint + no-deepcopy + SC + compile) | (1, True, 8) | (0, False, 24) |
| H1d (eager + no-deepcopy + SC + no-checkpoint) | (1, True, 8) | (0, False, 24) |
| D1_deepcopy | passing | (0, False, 24) |
| D1_no-deepcopy | (D1 forces find_unused=True; ran past iter 8 with 9 None params per audit) | (0, False, 24, 0 audited unused) |
| D2_deepcopy | passing | (0, False, 24) |
| D2_no-deepcopy | (1, True, 8) | (0, False, 24) |

`scripts/test_sc_compile.py` with the new `TEST_NO_SC_COPY=1` env var also passes (verified separately).

Removed the empirical DDP caveat block in `proteinfoundation/proteinflow/proteina.py` since `self_cond_use_copy=False` now runs cleanly with the default DDP strategy. The default has not been flipped to `False` — that is a separate decision (memory savings vs. live-weights SC vs. existing deepcopy behavior in tests/checkpoints) and is left to a follow-up.

### 8d. Files

Source (modified):
- `proteinfoundation/proteinflow/model_trainer_base.py` — added `torch.clear_autocast_cache()` + comment in `_set_self_cond` (one line + four-line comment).
- `proteinfoundation/proteinflow/proteina.py` — removed the 8-line DDP caveat comment block in the `elif sc_compile:` branch.
- `scripts/test_sc_compile.py` — added env-gated `TEST_NO_SC_COPY=1` toggle that appends `training.self_cond_use_copy=false` to the Hydra overrides.

Investigation scripts (new, in `scripts/`):
- `_dump_nn_kwargs.py` — composes the H1d Hydra config and dumps `cfg.model.nn` (with runtime-injected discrete dims) to `_debug_nn_kwargs.json`.
- `_debug_nn_kwargs.json` — captured kwargs.
- `debug_autograd_bare_pytorch.py` — bare-PyTorch repro harness with `--bf16` and `--multi-iter N` modes. Cleanly *did not* reproduce the bug, isolating the trigger to Lightning's outer autocast.
- `debug_autograd_lightning.py` — minimal Lightning + `pl.Trainer` wrapper, no DDP. Reproduces the bug when SC is inlined the same way `_set_self_cond` does it (confirms the autocast-cache mechanism inside Lightning), passes once the fix mirror is added.
- `debug_autograd_outer_autocast.py` — single-iteration smoking-gun: outer `torch.autocast(bf16)` wrapping both SC no_grad and grad forwards. The minimum-stack reproduction.
- `debug_autograd_fix_candidates.py` — side-by-side `baseline` / `fix-cache` / `fix-disable` / `fix-nested` comparison that selected the chosen fix.

Doc: this file (this section).

### 8e. No upstream issue filed

PyTorch #105211 already documents the bug (open as of this writing, triaged). No new upstream issue created. If a future PyTorch release fixes the cache-under-no_grad behavior at the engine level, the `torch.clear_autocast_cache()` call here will simply be a small redundancy; leave the comment in place so reviewers know why it exists.

### 8f. Caveat: `test_sc_compile.py` recompile thrash in `self_cond_use_copy=False` + `use_torch_compile_sc=True`

Running `TEST_NO_SC_COPY=1 python scripts/test_sc_compile.py` (with the fix in place):
- Both invocations complete to `max_steps=10`, no DDP errors, no crashes.
- Loss parity: max relative diff across 20 micro-batches = `0.000e+00` (well below the 3% tolerance).
- Recompile counts: baseline (SC eager) = 2, test (SC compiled) = 53; one resume-frame inside `_forward_impl` reaches 9 variants and trips the `>6` cache-thrash threshold the test enforces.

This is a known Phase 2 limitation, not a Phase 4 regression. `use_torch_compile_sc=True` gives `_forward_impl_sc` a distinct `__code__` for the top-level Dynamo cache, but Dynamo inlines the delegating call to `_forward_impl` into both the SC and training caches, and graph breaks inside `_forward_impl` produce resume continuations whose `__code__` IS shared between the two paths. With `self_cond_use_copy=False` the same `nn` serves both forwards, so the shared continuation cache sees `grad_mode` flip every iter and thrashes.

Mitigation choices (none required for correctness):
- Keep `self_cond_use_copy=True` (production default) when running SC compile — `nn_sc` and `self.nn` are separate Module instances so the continuation cache is also separate. No thrash. ~50% extra param memory.
- Or set `use_torch_compile_sc=False` when `self_cond_use_copy=False` — SC runs eager, training runs compiled, no cache collision possible. Slightly slower SC pass.
- Or accept the recompile cost (eventually stabilizes once the cache fills).

Phase 4's contribution is correctness (no more disconnected params, no more DDP unused-params errors). Phase 2's compile-thrash mitigation continues to be incomplete for the inlined-continuation case; that is a separate refactor (give every inlined helper a distinct copy keyed on the grad-mode call site) and out of scope here.

## 9. Phase 5 — Compile-friendly SC via parameter aliasing (`self_cond_mode`)

Resolves the §8f caveat. Goal: keep `use_torch_compile_sc=True` AND avoid the ~50% parameter-memory cost of deepcopy AND keep self-conditioning reading the live training weights — all at once, with no recompile thrash.

### 9a. Why deepcopy avoided thrash and `none` did not

From §8f: Dynamo gives `_forward_impl_sc` a distinct `__code__` so the *top-level* SC and training frames cache separately, but the graph break inside the shared `_forward_impl` helper produces resume continuations whose `__code__` is shared between the two paths. The continuation cache is keyed by that code object plus guards; the dominant differentiating guard is the `nn.Module` instance (`self`) the frame runs on.

- `self_cond_mode=deepcopy`: SC runs on `nn_sc`, a *separate* Module instance, so the continuation frames guard on a different `self` and get their own cache slot. `grad_mode` is constant within each slot → no thrash. Cost: `nn_sc` owns a full duplicate set of weights (~50% extra param memory) and must be periodically resynced (`_maybe_update_self_cond_copy`) so SC does not drift from training.
- `self_cond_mode=none`: SC and training both run on `self.nn` — same `self`, same continuation cache slot, `grad_mode` flips every step → the slot recompiles back and forth, trips `cache_size_limit`, and evicts/refills (53 recompiles; one frame reaches 9 variants).

### 9b. The aliasing fix

`self_cond_mode=alias` (new) builds `nn_sc` as a *distinct Module instance whose parameter and buffer tensors are the same objects as* `self.nn`'s:

```python
def _make_nn_sc_alias(nn: torch.nn.Module) -> torch.nn.Module:
    sc = copy.deepcopy(nn)
    src_params = dict(nn.named_parameters(recurse=True))
    src_buffers = dict(nn.named_buffers(recurse=True))
    for full_name in list(src_params):
        parent_name, _, leaf = full_name.rpartition(".")
        parent = sc.get_submodule(parent_name) if parent_name else sc
        parent._parameters[leaf] = src_params[full_name]
    for full_name in list(src_buffers):
        parent_name, _, leaf = full_name.rpartition(".")
        parent = sc.get_submodule(parent_name) if parent_name else sc
        parent._buffers[leaf] = src_buffers[full_name]
    return sc
```

`copy.deepcopy` yields a new module object graph (new `id(self)` → new Dynamo continuation-cache slot, exactly like deepcopy mode), then every leaf parameter/buffer is rebound to the original tensor (shared storage, no duplication, SC always reads live weights). Best of both: deepcopy's cache isolation with `none`'s zero memory cost and live weights.

Safe under `torch._dynamo.config.skip_no_tensor_aliasing_guards_on_parameters = True` (the default in torch 2.7.1): Dynamo does not emit cross-parameter aliasing guards, so `nn_sc` and `self.nn` pointing at the same storage does not by itself trigger guard failures or recompiles.

### 9c. Empirical pre-validation (isolated)

`scripts/debug_dynamo_alias_isolation.py` reduces the question to its core: build `mod_a`, `mod_b = _make_nn_sc_alias(mod_a)`, each with a `torch._dynamo.graph_break()` in forward, `torch.compile` both, then alternate `mod_b` under `no_grad` and `mod_a` under grad for 10 iterations while counting `TORCH_LOGS=recompiles` events. Result: **2 recompiles total** (one per module for its first grad/no_grad specialization), no thrash — confirming aliased modules occupy independent Dynamo cache slots despite shared storage. PASS.

### 9d. End-to-end verification

`SELF_COND_MODE=alias python scripts/test_sc_compile.py` (full Proteina stack, SC compiled, no deepcopy):

| Metric | `none` (§8f, FAIL) | `alias` (PASS) |
|---|---|---|
| max rel\|grad diff\| vs SC-eager baseline | 0.000e+00 | 0.000e+00 |
| recompiles (baseline SC-eager) | 2 | 2 |
| recompiles (SC compiled) | 53 | 37 |
| highest variant count per frame | frame 25 → **9** | frame 27 → **4** |
| cache-thrash check (>6 variants = fail) | tripped | clear |

Gradients are bit-identical to the SC-eager baseline; recompiles drop from 53 to 37 and the worst frame from 9 variants (eviction territory) to 4 (under the 6-variant threshold). A direct `scripts/_ddp_debug_runner.py` alias run also reached `max_steps=12` with zero DDP unused-parameter errors.

### 9e. Memory

In alias mode `nn_sc` shares all 637 parameter tensors with `self.nn` (verified by `data_ptr` identity), saving ≈76 MB — the full fp32 parameter footprint of the 20M model — relative to deepcopy mode, which duplicates every parameter.

### 9f. Config knob and behavior

`training.self_cond_mode` (stage-1 and stage-2 YAMLs):
- `deepcopy` — separate Module, duplicated weights, periodic resync. Original behavior.
- `alias` — separate Module, shared weights, live SC, no resync. **Recommended with `use_torch_compile_sc=True`.**
- `none` — single Module for both passes. Lowest memory but recompile thrash under SC compile (§8f).

Back-compat: when `self_cond_mode` is absent it derives from the legacy `self_cond_use_copy` flag (`True`→`deepcopy`, `False`→`none`). `_maybe_update_self_cond_copy` early-returns in alias mode (detected by first-parameter tensor identity), since a state_dict resync would be a redundant `copy_(self)`. `state_dict`/`load_state_dict` strip `nn_sc.` keys in all three modes, so checkpoints are mode-independent.

### 9g. Related upstream issues

- [#141589](https://github.com/pytorch/pytorch/issues/141589) — multiple inlined `nn.Module` instances and per-instance recompilation; closest match to the continuation-cache behavior exploited here.
- [#100977](https://github.com/pytorch/pytorch/issues/100977), [#111528](https://github.com/pytorch/pytorch/issues/111528) — `no_grad`/inference interplay with `torch.compile` graph reuse.
- [#105211](https://github.com/pytorch/pytorch/issues/105211) — the Phase 4 autocast-cache bug (§8); orthogonal but in the same SC code path.

### 9h. Files

- `proteinfoundation/proteinflow/proteina.py` — added `_make_nn_sc_alias` helper and the three-way `self_cond_mode` switch (deepcopy | alias | none) replacing the old `self_cond_use_copy`/`sc_compile` branch.
- `proteinfoundation/proteinflow/model_trainer_base.py` — `_maybe_update_self_cond_copy` early-returns when params are aliased.
- `configs/experiment_config/training_dssp_contact_20M_udlm_pb_v2_stage1.yaml`, `…_v2.yaml` — added `self_cond_mode` (default flipped to `alias`; the legacy `self_cond_use_copy: True` line is retained but ignored when `self_cond_mode` is set).
- `scripts/test_sc_compile.py` — `SELF_COND_MODE={alias,none,deepcopy}` env override.
- `scripts/debug_dynamo_alias_isolation.py` (new) — isolated alias cache-isolation pre-validation.

The production default is now `alias` (flipped after the verification in 9d passed); `deepcopy` and `none` remain selectable via `self_cond_mode`.

## 10. Phase 6 — Eval-pass compile + recompile-limit headroom

Phase 5 made the self-conditioning (SC) prior pass compile-friendly via parameter aliasing (`self_cond_mode: alias`, now the production default). Phase 6 closes two remaining gaps: training-time **validation** still ran eager, and the Dynamo recompile limit (default 8) left no headroom for the extra caches plus validation-sampling shape variants.

### 10a. Two eager validation paths (before fix)

Dedicated inference (`predict_step` → `generate(force_compile=True)`) already compiled via the `force_compile` eval branch on `self.nn`. But training-time validation ran eager on both of its forwards:

1. **Validation loss** — `validation_step_data` runs `training_step` under `torch.no_grad()` → `predict_clean` → `self.nn(batch)`. In alias mode `self.nn.use_torch_compile_sc=False`, so this fell to eager.
2. **Validation sampling** — `_run_validation_trajectory` → `generate(...)` **without** `force_compile` → `predict_clean_n_v_w_guidance` with `force_compile=False` → eager on both the main forward and the unconditional CFG forward.

### 10b. Fix: route no-grad eval forwards through alias `nn_sc`

`model_trainer_base.py` gains a helper, `_eval_forward_module(force_compile=False)`, used by `predict_clean` and `predict_clean_n_v_w_guidance`:

```python
def _eval_forward_module(self, force_compile: bool = False):
    if (
        not force_compile
        and not torch.is_grad_enabled()
        and getattr(self, "_nn_sc_aliased", False)
        and self.nn_sc is not None
    ):
        return self.nn_sc
    return self.nn
```

In **alias** mode (`_nn_sc_aliased`, set in `proteina.__init__`) a no-grad eval forward returns `nn_sc`, which:
- **shares live weights** with `self.nn` (alias rebinds leaves to the same storage; the EMA callback swaps weights in place via `swap_tensors`/`copy_`), so validation numerics are identical to running `self.nn`;
- owns a **separate Dynamo cache** (`_forward_impl_sc`'s distinct `__code__`), so adding validation's no-grad shape contexts never lands grad/no-grad variants on the **training** graph's cache → no train-path thrash.

Fall-backs to `self.nn`: `force_compile=True` (dedicated inference keeps its own eval path), grad enabled (training), and `deepcopy`/`none` modes (where `nn_sc` is stale or `None`). The autoguidance forward (`self.nn_ag`) is a separate checkpoint and is left untouched.

### 10c. Recompile-limit headroom: 8 → 32

`torch._dynamo.config.recompile_limit = 32` is set in both `train.py` and `inference.py` (canonical name in torch 2.7.1; `cache_size_limit` is its alias). Stock default is 8; once a single frame exceeds the limit Dynamo evicts and the cache thrashes (extreme slowdown). The accumulated limit (256) is untouched. Headroom is needed because there are now up to three distinct compiled graphs — train (`_forward_impl`, grad), SC/eval (`_forward_impl_sc`, no-grad), inference (`_forward_impl`, no-grad) — plus a new dynamic-shape variant per validation-sampling length.

### 10d. Verification

**Numeric parity, no thrash** — `SELF_COND_MODE=alias scripts/test_sc_compile.py` after the routing edits and threshold update:

| Metric | Value |
|---|---|
| max rel\|grad diff\| vs SC-eager baseline | 0.000e+00 |
| recompiles (baseline SC-eager) | 2 |
| recompiles (SC compiled) | 55 |
| highest variant count per frame | frame 25 → 11 |
| eviction / `recompile_limit reached` messages | none |

Gradients are bit-identical; the worst frame (25 = `rigid_utils.Rotation.__init__`, recompiling on rotation-tensor rank/size) holds 11 variants — under the new test threshold of 24 and far under the 32 limit. Because validation now calls the **same** `_forward_impl_sc` this test proves is bit-identical to eager, and `nn_sc` shares `self.nn`'s storage, **validation loss is numerically unchanged** by the routing. No separate SC-off val-loss control is run: validation loss is sampled at random diffusion timesteps and is not comparable across runs without seed control, so the deterministic per-step parity above is the authoritative evidence.

**Eval-pass compile + cold compile-time** — a 12-step training run with validation firing every 3 optim steps, SC compile on, fresh Inductor cache, `TORCH_LOGS=recompiles` (20M model, padded length 64, batch 4, A100):

| Phase | Cold wall-time |
|---|---|
| First train batch — train graph `_forward_impl` (grad), cold | 203 s |
| Sanity validation — SC/eval graph `_forward_impl_sc` (no-grad), cold | 231 s |
| First validation **sampling** (step 3) — new lengths in the `generate` trajectory | 242 s (one-time) |
| Warm validations (steps 6, 9, 12) | 58–59 s each |
| New-length train recompile (epoch 1) | 19 s |
| Steady-state train step | 0.3–0.5 s |
| Peak variants/frame **with** validation sampling | frame 25 → **14** (no eviction) |

Validation forwards are confirmed **compiled, not eager** (recompiles during the val window tag the `_forward_impl_sc` frames; zero eager-fallback / `recompile_limit reached` messages), and the peak per-frame variant count even with sampling (14) stays well under 32.

### 10e. Compile-time expectation (answering "it used to be ~1-2 min")

On a **cold** Inductor cache each graph now traces + Inductor-compiles in ≈ 3.4 min (train) / ≈ 3.8 min (SC), so a cold run pays ≈ 7 min before steady state, plus a one-time ≈ 4 min the first time validation sampling hits new lengths. The earlier "1-2 min" reflected a single compiled graph; the increase is (a) a second/third graph (SC + eval) and (b) cold-cache cost. On a **warm** Inductor cache (the normal case for repeat runs / resumed training, when `TORCHINDUCTOR_CACHE_DIR` persists) these collapse to ~19 s incremental recompiles and near-instant cache hits; steady-state training stays at ~0.3–0.5 s/step.

### 10f. Files

- `proteinfoundation/train.py`, `proteinfoundation/inference.py` — `torch._dynamo.config.recompile_limit = 32`.
- `proteinfoundation/proteinflow/proteina.py` — `self._nn_sc_aliased = (sc_mode == "alias")` in `__init__`.
- `proteinfoundation/proteinflow/model_trainer_base.py` — `_eval_forward_module` helper; `predict_clean` and `predict_clean_n_v_w_guidance` route no-grad eval forwards through it.
- `scripts/test_sc_compile.py` — `MAX_VARIANTS_PER_FRAME` 6 → 24 and threshold comments updated for `recompile_limit=32`.
