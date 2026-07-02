import math
import os
import logging
import random as _random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from openfold.model.template import TemplatePairStack, TemplatePointwiseAttention
from openfold.model.embedders import TemplatePairEmbedder
from openfold.model.structure_module import StructureModule
import openfold.np.residue_constants as rc
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.precision_utils import wrap_for_precision, PrecisionWrapper
from openfold.data.feature_pipeline import FeaturePipeline
from openfold.data.data_pipeline import (
    make_dummy_msa_feats,
    make_sequence_features_with_custom_template,
    make_sequence_features,
)
from openfold.utils.tensor_utils import tensor_tree_map

logger = logging.getLogger(__name__)


def _unwrap_precision(module: nn.Module) -> nn.Module:
    """Return the underlying module if wrapped by wrap_for_precision(), else module
    unchanged. Needed because per-block compile patches .forward directly on the
    blocks inside evoformer/extra_msa_stack, which PrecisionWrapper hides behind
    a `.model` attribute rather than exposing them at the top level."""
    return module.model if isinstance(module, PrecisionWrapper) else module


class OpenFoldTemplateInference(nn.Module):
    """
    Full OpenFold AlphaFold model inference using a distogram-only template.

    This is intended for Proteina contact-map diffusion logging/inference:
    given a predicted distogram probability tensor [B, L, L, 39] and the
    target sequence, produce atom37 coordinates.

    Supports:
      - torch.compile on compute-intensive submodules (evoformer, structure module)
        via enable_compilation() / disable_compilation()
      - Batched inference via forward_batched() for multiple samples
    """

    def __init__(
        self,
        *,
        model_name: str = "model_1_ptm",
        jax_params_path: str = None,
        slim_ckpt_path: Optional[str] = None,
        evoformer_keep_block_indices: Optional[str] = None,
        use_ema: bool = True,
        device: Optional[torch.device] = None,
        skip_template_alignment: bool = False,
        max_recycling_iters: Optional[int] = None,
        compile_model: bool = False,
        compile_interval: int = 64,
        use_mlm: bool = False,
        use_deepspeed_evoformer_attention: bool = True,
        use_cuequivariance_attention: bool = False,
        use_cuequivariance_triangle_attention: bool = False,
        use_cuequivariance_multiplicative_update: bool = False,
        compile_inference_path: bool = False,
        inference_attn_kernel: str = "sdpa",
        compile_strategy: str = "per_block",
        use_cueq_triangle_mul: bool = False,
        precision: str = "tf32",
    ):
        super().__init__()

        self.model_name = model_name
        if jax_params_path is None:
            jax_params_path = os.path.join(
                os.path.expanduser("~/openfold/openfold/resources/params"),
                f"params_{model_name}.npz",
            )
        self.jax_params_path = jax_params_path
        # use_cuequivariance_triangle_attention routes cuequivariance only to
        # triangle attention in PairStack (via monkey-patching), while MSA attention
        # continues to use deepspeed.  Model config must have cue_attn=False so
        # AlphaFold.forward passes False to EvoformerBlock MSA attention.
        self.cfg = model_config(
            model_name,
            use_deepspeed_evoformer_attention=use_deepspeed_evoformer_attention,
            use_cuequivariance_attention=use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
            precision=precision,
        )
        self.cfg.data.common.use_templates = True
        if max_recycling_iters is not None:
            self.cfg.data.common.max_recycling_iters = int(max_recycling_iters)

        # When use_mlm=False, disable the masked-language-model style random
        # corruption of MSA positions (masked_msa_replace_fraction).  This is
        # the main source of non-determinism in the feature pipeline and is
        # unnecessary for structure prediction from distograms.
        if not use_mlm:
            self.cfg.data.predict.masked_msa_replace_fraction = 0.0
            self.cfg.data.eval.masked_msa_replace_fraction = 0.0
        self.use_mlm = use_mlm

        self.model = AlphaFold(self.cfg)
        self.model.eval()
        if slim_ckpt_path is not None:
            if evoformer_keep_block_indices is None:
                raise ValueError("slim_ckpt_path set but evoformer_keep_block_indices is None")
            keep = [int(x) for x in str(evoformer_keep_block_indices).split(",")]
            self.model.evoformer.blocks = nn.ModuleList(
                [self.model.evoformer.blocks[i] for i in keep]
            )
            ck = torch.load(slim_ckpt_path, map_location="cpu", weights_only=False)
            if use_ema and "ema" in ck:
                sd = ck["ema"]["params"]
            else:
                sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
            if len(set(sd.keys()) & set(self.model.state_dict().keys())) == 0:
                raise RuntimeError(
                    f"slim ckpt {slim_ckpt_path}: 0 keys matched the sliced model "
                    f"(use_ema={use_ema}, keep={len(keep)} blocks) -- wrong format or KEEP indices"
                )
            miss, unexp = self.model.load_state_dict(sd, strict=False)
            logger.info(
                f"[slim] loaded {slim_ckpt_path} (ema={use_ema}, keep={len(keep)} blocks) "
                f"| missing={len(miss)} unexpected={len(unexp)}"
            )
        else:
            import_jax_weights_(self.model, jax_params_path, version=model_name.replace("model_", ""))

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise ValueError(
                "OpenFoldTemplateInference requires CUDA (OpenFold attention CUDA kernels are enabled). "
                f"Got device={device}."
            )
        self.device = device
        self.model = self.model.to(self.device)
        # Mirrors openfold/utils/script_utils.py's _accelerate(): only evoformer +
        # extra_msa_stack are cast; heads/structure module stay fp32 for numerical stability.
        if self.cfg.precision in ("bf16", "fp16"):
            self.model.evoformer = wrap_for_precision(self.model.evoformer, self.cfg.precision)
            self.model.extra_msa_stack = wrap_for_precision(self.model.extra_msa_stack, self.cfg.precision)
        self.feature_pipeline = FeaturePipeline(self.cfg.data)
        self.skip_template_alignment = skip_template_alignment
        self._template_use_unit_vector_default = None
        if hasattr(self.model, "template_embedder") and hasattr(self.model.template_embedder, "config"):
            self._template_use_unit_vector_default = self.model.template_embedder.config.use_unit_vector

        if use_cuequivariance_triangle_attention:
            if use_cuequivariance_attention:
                logger.warning(
                    "use_cuequivariance_triangle_attention=True has no effect when "
                    "use_cuequivariance_attention=True (already applied globally)."
                )
            else:
                self._wrap_pair_stacks_for_cue_triangle_attn()

        # Compilation state (torch.compile approach)
        self._compiled = False
        self._compile_dynamic = True  # default: dynamic shapes
        self._original_forwards = {}
        self._compile_interval = compile_interval

        # New full-graph inference path (torch.compile-compatible).
        self._compile_inference_path = False
        self._inference_attn_kernel = inference_attn_kernel

        if compile_inference_path:
            self.enable_compile_inference_path(
                attn_kernel=inference_attn_kernel,
                compile_strategy=compile_strategy,
                use_cueq_triangle_mul=use_cueq_triangle_mul,
            )
        elif compile_model:
            if use_deepspeed_evoformer_attention:
                # deepspeed attention uses a custom CUDA kernel whose fake/meta
                # kernel is incompatible with torch.compile (stride mismatch).
                # The two optimisations are mutually exclusive; deepspeed gives
                # a larger speedup (1.4-1.6x vs 1.17-1.19x), so we skip compile.
                logger.warning(
                    "compile_model=True is incompatible with "
                    "use_deepspeed_evoformer_attention=True (stride mismatch in "
                    "deepspeed's custom CUDA kernel).  Compilation will be skipped. "
                    "Disable deepspeed attention to use torch.compile."
                )
            else:
                # Use dynamic=False with interval-based padding for better per-call
                # speedup (~1.19x) vs dynamic=True (~1.17x).  Inputs are padded to
                # the nearest multiple of compile_interval so torch.compile reuses
                # shape-specialized kernels.  Each new bucket triggers ~12-17s of
                # one-time compilation; with interval=64, typical proteins (50-512
                # residues) need up to 8 buckets.  We raise the dynamo recompile
                # limit to 32 to give each sub-function (evoformer, msa, structure
                # module) enough headroom for all bucket shapes.
                self.enable_compilation(dynamic=False)
                self._compile_dynamic = False


    # ------------------------------------------------------------------
    # torch.compile support
    # ------------------------------------------------------------------

    @property
    def compiled(self) -> bool:
        return self._compiled

    def enable_compile_inference_path(
        self,
        attn_kernel: str = "sdpa",
        compile_strategy: str = "per_block",
        use_cueq_triangle_mul: bool = False,
    ) -> None:
        """Enable the compile-friendly inference path.

        Routes attention through the SDPA, vanilla matmul, or cuEquivariance
        branch added to openfold's Attention/IPA modules so the model can be
        compiled by torch.compile.

        Two compile strategies:
          * ``per_block`` (default) — compile each EvoformerBlock /
            ExtraMSABlock / TemplatePairStackBlock / structure-module piece
            separately. Bounded per-compile latency (~30-60 s per block
            type, one-shot); the Inductor cache shares across instances
            with the same shape.
          * ``whole_graph`` — compile the whole ``AlphaFold.iteration_inference``
            as one graph. Larger trace time (5-10 min) but slightly more
            fusion across block boundaries. Only worthwhile for very large
            decoy batches per protein.

        Three attention kernels:
          * ``sdpa`` — ``F.scaled_dot_product_attention``; broad support, no
            extra deps.
          * ``vanilla`` — explicit matmul + softmax via the existing
            ``_attention`` helper. Compile-friendly numerics-fidelity
            fallback if SDPA drifts.
          * ``cuequivariance`` — cuEquivariance triangle_attention kernel.
            Compile-traceable (registered as torch.library custom op with
            register_fake). Faster on triangle attention; AF2Rank model
            (c_hidden=32, no_heads=4 → 8/head) fits the kernel's
            hidden-dim constraints (fp32 ≤32 ÷4 = 0; fp16/bf16 ≤128 ÷8 = 0).

        Always-on in this mode:
          * cuEquivariance triangle multiplicative update — the kernel has
            ``torch.compiler.is_compiling()`` branches that explicitly
            disable its eager fallback during compile, so it traces cleanly
            via its registered fake meta implementation.
          * ChunkSizeTuner is removed from evoformer / extra_msa_stack /
            template_pair_stack so chunk sizes never depend on input
            shape (which would otherwise trigger per-L recompiles).

        Incompatible with ``use_deepspeed_evoformer_attention`` (skipped
        automatically by the inference path).
        """
        if self._compile_inference_path:
            return
        if attn_kernel not in ("sdpa", "vanilla", "cuequivariance"):
            raise ValueError(
                f"attn_kernel must be 'sdpa', 'vanilla', or 'cuequivariance'; "
                f"got {attn_kernel!r}"
            )
        if compile_strategy not in ("per_block", "whole_graph"):
            raise ValueError(
                f"compile_strategy must be 'per_block' or 'whole_graph'; "
                f"got {compile_strategy!r}"
            )

        # Defensive: the full-graph path skips chunking by construction, but
        # if a caller has set chunk_size on the config we make that explicit.
        if getattr(self.cfg.globals, "chunk_size", None) is not None:
            logger.info(
                "compile_inference_path overrides cfg.globals.chunk_size=%s "
                "(only the main evoformer trunk uses chunk_size=None; "
                "auxiliary chunked stacks still chunk via _aux_chunk).",
                self.cfg.globals.chunk_size,
            )
            self.cfg.globals.chunk_size = None

        # Step A — drop ChunkSizeTuner. The tuner runs OOM-binary-search
        # forward passes per input shape and caches by shape, so a different
        # L produces a different chunk_size and a fresh traced graph. Killing
        # it here is the simplest way to keep the compiled graph stable.
        for stack_attr, sub_attr in (
            ("evoformer", None),
            ("extra_msa_stack", None),
            ("template_embedder", "template_pair_stack"),
        ):
            stack = getattr(self.model, stack_attr, None)
            if stack is None:
                continue
            target = stack if sub_attr is None else getattr(stack, sub_attr, None)
            if target is None:
                continue
            if getattr(target, "chunk_size_tuner", None) is not None:
                logger.info(
                    "compile_inference_path: disabling chunk_size_tuner on %s%s",
                    stack_attr, f".{sub_attr}" if sub_attr else "",
                )
                target.chunk_size_tuner = None
                target.tune_chunk_size = False

        # Step F — warm the triton cache used by the cuEquivariance triangle
        # multiplicative update before installing torch.compile. The kernel
        # has compile-aware branches but its first-call autotuning is slow
        # if the cache isn't initialized.
        try:
            from cuequivariance_ops_torch import init_triton_cache  # type: ignore
            init_triton_cache()
            logger.info("compile_inference_path: cuEquivariance triton cache initialized")
        except Exception as exc:
            logger.info(
                "compile_inference_path: skipping init_triton_cache (%s); "
                "cueq triangle-mul autotuning will happen lazily on first call.",
                exc,
            )

        torch._dynamo.config.recompile_limit = max(
            64, getattr(torch._dynamo.config, "recompile_limit", 8),
        )

        # Attach attention-kernel selection to the AlphaFold instance so that
        # forward_inference() can read it without re-threading flags.
        self.model._inference_use_torch_sdpa = (attn_kernel == "sdpa")
        self.model._inference_use_torch_vanilla = (attn_kernel == "vanilla")
        self.model._inference_use_torch_cueq = (attn_kernel == "cuequivariance")
        # cuEq triangle multiplicative update — Triton-based, re-autotunes
        # per shape. Default OFF to preserve true dynamic-shape behaviour;
        # opt in with use_cueq_triangle_mul=True for the ~50% steady-state
        # speedup at the cost of per-shape compile time.
        self.model._inference_use_cueq_mul_update = use_cueq_triangle_mul

        if compile_strategy == "whole_graph":
            self._enable_whole_graph_compile()
        else:
            self._enable_per_block_compile()

        self._compile_inference_path = True
        self._compile_strategy = compile_strategy
        self._inference_attn_kernel = attn_kernel
        logger.info(
            "OpenFold inference-path compile enabled "
            "(strategy=%s; attn_kernel=%s)",
            compile_strategy, attn_kernel,
        )

    def _enable_per_block_compile(self) -> None:
        """Per-block compile strategy: compile each transformer block.

        Each block gets its own compile context. Without intervention,
        Dynamo specialises on the L-axis of every tensor entering each
        block, which means a new L triggers a recompile per block (~50
        blocks × per-shape recompile = catastrophic).

        Fix: wrap each compiled block.forward with a thin pre-call hook
        that calls torch._dynamo.mark_dynamic on the per-residue axis of
        each tensor argument. mark_dynamic on the FIRST call propagates
        to Dynamo's symbolic-shape engine, so the compiled kernel is
        produced with symbolic L. On subsequent calls (different L),
        mark_dynamic is a no-op and the cached kernel is reused.

        The hook is per-block-class because each block class has different
        L-axis positions:
          * EvoformerBlock / ExtraMSABlock: m at axis -2 (= L), z at -3, -2
          * TemplatePairStackBlock: z at axis -2 (and -3)
          * StructureModule transition/angle_resnet: single arg at axis -2
        """
        compile_kwargs = dict(dynamic=True, fullgraph=False, mode="default")

        # Track originals so we can restore on disable.
        self._original_forwards = {}

        def _make_l_axis_marker(spec):
            """Return a wrapper that marks per-residue axes dynamic on the
            FIRST call before invoking the compiled callable.  `spec` is a
            tuple of (arg_name_or_position, [negative-axis indices]) pairs.
            Once Dynamo has produced a graph with symbolic shapes, the hint
            on subsequent calls is a cheap no-op.

            Uses `maybe_mark_dynamic` (soft hint), NOT `mark_dynamic`
            (hard constraint).  The hard form raises ConstraintViolationError
            when Dynamo's symbolic engine infers the marked dim must be
            constant during tracing — which IS the case here: each per-block
            compile boundary creates a fresh tracing context where the
            block's body contains ops like `m + extra` and `m * pair_bias`
            whose other operands derive their L from non-marked sources,
            forcing specialisation.  The soft hint lets Dynamo specialise
            silently when needed; the worst case is the same as without
            the hint (recompile on a new L), but the compile no longer
            CRASHES the way it does with the strict version.
            """
            mark_fn = getattr(
                torch._dynamo, "maybe_mark_dynamic", torch._dynamo.mark_dynamic,
            )

            def wrap(compiled_fn):
                marked = [False]

                def wrapper(*args, **kwargs):
                    if not marked[0]:
                        for slot, axes in spec:
                            if isinstance(slot, int):
                                if slot < len(args) and isinstance(args[slot], torch.Tensor):
                                    t = args[slot]
                                    for ax in axes:
                                        if t.dim() >= abs(ax):
                                            try:
                                                mark_fn(t, t.dim() + ax)
                                            except Exception:
                                                pass
                            else:
                                t = kwargs.get(slot)
                                if isinstance(t, torch.Tensor):
                                    for ax in axes:
                                        if t.dim() >= abs(ax):
                                            try:
                                                mark_fn(t, t.dim() + ax)
                                            except Exception:
                                                pass
                        marked[0] = True
                    return compiled_fn(*args, **kwargs)
                return wrapper
            return wrap

        # IMPORTANT: only mark ONE axis per tensor.  All L axes across a
        # block's inputs derive from the SAME underlying L; Dynamo's
        # symbolic engine will broadcast the symbolic dim across other
        # axes via the ops that constructed them.  Marking multiple axes
        # of the same tensor creates independent symbols (s0, s1, …),
        # which the body code then asserts must be equal — triggering a
        # ConstraintViolationError.
        #
        # We mark m's L axis explicitly.  z's two L dims share a symbol
        # via the broadcasts that built it from m's pair embedding.
        # Same for msa_mask and pair_mask.
        evo_spec = [(0, [-2])]        # m's L
        extra_spec = evo_spec
        # TemplatePairStackBlock.forward(z, mask, ...): mark only z's
        # second-to-last L axis; the first L is the same symbol via
        # broadcasts.
        tps_spec = [(0, [-2])]
        # StructureModule transition / angle_resnet: shape (..., L, C).
        sm_spec = [(0, [-2])]

        # Evoformer blocks (the main bottleneck: 48 blocks).
        wrap_evo = _make_l_axis_marker(evo_spec)
        for i, block in enumerate(_unwrap_precision(self.model.evoformer).blocks):
            key = f"ip_evoformer_block_{i}"
            self._original_forwards[key] = block.forward
            block.forward = wrap_evo(torch.compile(block.forward, **compile_kwargs))

        # Extra-MSA blocks.
        if hasattr(self.model, "extra_msa_stack"):
            wrap_extra = _make_l_axis_marker(extra_spec)
            for i, block in enumerate(_unwrap_precision(self.model.extra_msa_stack).blocks):
                key = f"ip_extra_msa_block_{i}"
                self._original_forwards[key] = block.forward
                block.forward = wrap_extra(torch.compile(block.forward, **compile_kwargs))

        # Template pair stack blocks.
        if (
            hasattr(self.model, "template_embedder")
            and hasattr(self.model.template_embedder, "template_pair_stack")
        ):
            tps = self.model.template_embedder.template_pair_stack
            wrap_tps = _make_l_axis_marker(tps_spec)
            for i, block in enumerate(tps.blocks):
                key = f"ip_tps_block_{i}"
                self._original_forwards[key] = block.forward
                block.forward = wrap_tps(torch.compile(block.forward, **compile_kwargs))

        # Structure-module pieces that don't depend on the inplace CUDA
        # extension (we leave IPA itself eager — its non-inplace branch is
        # already small).
        sm = self.model.structure_module
        wrap_sm = _make_l_axis_marker(sm_spec)
        for name in ("transition", "angle_resnet"):
            submod = getattr(sm, name, None)
            if submod is not None:
                key = f"ip_sm_{name}"
                self._original_forwards[key] = submod.forward
                submod.forward = wrap_sm(torch.compile(submod.forward, **compile_kwargs))

    def _enable_whole_graph_compile(self) -> None:
        """Whole-graph compile strategy: compile iteration_inference as one graph.

        Note: trace time is large (5-10 min on AlphaFold-2 scale). Only
        worthwhile when amortised across many decoys at the same shape.
        """
        compile_kwargs = dict(dynamic=True, fullgraph=False, mode="default")
        self._original_forwards = {
            "iteration_inference": self.model.iteration_inference,
        }
        self.model.iteration_inference = torch.compile(
            self.model.iteration_inference, **compile_kwargs,
        )

    def forward_inference(self, batch: dict) -> dict:
        """Run the compile-friendly inference forward on a prepared batch.

        This is what AF2Rank should call when compile_inference_path=True:
            batch = wrapper.build_batch(...)
            out   = wrapper.forward_inference(batch)
        """
        with torch.no_grad():
            return self.model.forward_inference(batch)

    def enable_compilation(self, dynamic: bool = False) -> None:
        """Compile compute-intensive submodules with torch.compile.

        Targets the evoformer blocks, extra-MSA blocks, template pair stack,
        and structure-module components (IPA, transition, angle resnet).
        The outer recycling loop and auxiliary heads are left uncompiled to
        avoid graph-break issues with .item(), .cpu(), and custom CUDA kernels.

        Args:
            dynamic: If True, use dynamic shapes to avoid recompilation when
                sequence lengths change between calls.  Slightly slower per
                call but avoids repeated compilation overhead.
        """
        if self._compiled:
            return

        if not dynamic:
            # Raise dynamo's per-function recompile limit so that interval-based
            # bucket padding (up to 8 distinct shapes for interval=64, max 512) does
            # not exhaust the default limit of 8 guards per frame.
            # NOTE: access via attribute (not `import torch._dynamo`) to avoid
            # Python treating `torch` as a local variable and breaking torch.compile
            # calls later in this function.
            torch._dynamo.config.recompile_limit = 32

        compile_kwargs = dict(fullgraph=False, dynamic=dynamic)

        # --- Evoformer blocks (48 blocks – main compute bottleneck) ---
        for i, block in enumerate(_unwrap_precision(self.model.evoformer).blocks):
            key = f"evoformer_block_{i}"
            self._original_forwards[key] = block.forward
            block.forward = torch.compile(block.forward, **compile_kwargs)

        # --- Extra-MSA stack blocks ---
        if hasattr(self.model, "extra_msa_stack"):
            for i, block in enumerate(_unwrap_precision(self.model.extra_msa_stack).blocks):
                key = f"extra_msa_block_{i}"
                self._original_forwards[key] = block.forward
                block.forward = torch.compile(block.forward, **compile_kwargs)

        # --- Template pair stack blocks ---
        if (
            hasattr(self.model, "template_embedder")
            and hasattr(self.model.template_embedder, "template_pair_stack")
        ):
            tps = self.model.template_embedder.template_pair_stack
            for i, block in enumerate(tps.blocks):
                key = f"tps_block_{i}"
                self._original_forwards[key] = block.forward
                block.forward = torch.compile(block.forward, **compile_kwargs)

        # --- Structure module components ---
        # NOTE: The IPA is excluded because it uses attn_core_inplace_cuda, a
        # custom C++/CUDA extension that torch.compile cannot trace.  Graph
        # breaks around the kernel cause unacceptable numerical divergence.
        sm = self.model.structure_module
        for name in ("transition", "angle_resnet"):
            submod = getattr(sm, name, None)
            if submod is not None:
                key = f"sm_{name}"
                self._original_forwards[key] = submod.forward
                submod.forward = torch.compile(submod.forward, **compile_kwargs)

        self._compiled = True
        logger.info("OpenFold submodule compilation enabled (dynamic=%s)", dynamic)

    def disable_compilation(self) -> None:
        """Restore original (uncompiled) forward methods."""
        if not self._compiled:
            return

        for i, block in enumerate(_unwrap_precision(self.model.evoformer).blocks):
            key = f"evoformer_block_{i}"
            if key in self._original_forwards:
                block.forward = self._original_forwards[key]

        if hasattr(self.model, "extra_msa_stack"):
            for i, block in enumerate(_unwrap_precision(self.model.extra_msa_stack).blocks):
                key = f"extra_msa_block_{i}"
                if key in self._original_forwards:
                    block.forward = self._original_forwards[key]

        if (
            hasattr(self.model, "template_embedder")
            and hasattr(self.model.template_embedder, "template_pair_stack")
        ):
            tps = self.model.template_embedder.template_pair_stack
            for i, block in enumerate(tps.blocks):
                key = f"tps_block_{i}"
                if key in self._original_forwards:
                    block.forward = self._original_forwards[key]

        sm = self.model.structure_module
        for name in ("transition", "angle_resnet"):
            key = f"sm_{name}"
            if key in self._original_forwards:
                getattr(sm, name).forward = self._original_forwards[key]

        self._original_forwards.clear()
        self._compiled = False
        logger.info("OpenFold submodule compilation disabled")

    def _wrap_pair_stacks_for_cue_triangle_attn(self) -> None:
        """Route cuequivariance attention to triangle ops only, keeping deepspeed for MSA.

        EvoformerBlock.forward() passes use_cuequivariance_attention from the global
        config to both MSA attention (row/col) and PairStack (triangle attention).
        When the global config has use_cuequivariance_attention=False (to let deepspeed
        handle MSA attention), this method monkey-patches each PairStack.forward to
        inject use_cuequivariance_attention=True, so triangle attention gets the
        cuequivariance kernel while MSA attention keeps deepspeed.

        This achieves:
          MSA row/col attention → deepspeed kernel  (1.4-1.6x speedup)
          Triangle start/end    → cuequivariance kernel
          Triangle mul update   → cuequivariance (if use_cuequivariance_multiplicative_update)
        """
        count = 0
        for block in _unwrap_precision(self.model.evoformer).blocks:
            if not hasattr(block, "pair_stack"):
                continue
            orig = block.pair_stack.forward
            # Use default arg to capture `orig` in the closure correctly
            def _patched(orig=orig, *args, **kwargs):
                kwargs["use_cuequivariance_attention"] = True
                kwargs["use_deepspeed_evo_attention"] = False  # must clear; deepspeed takes unconditional priority in primitives.py
                return orig(*args, **kwargs)
            block.pair_stack.forward = _patched
            count += 1
        logger.info(
            "Patched %d PairStack.forward methods to use cuequivariance triangle attention "
            "while MSA attention uses deepspeed.", count
        )

    def _compute_pad_length(self, true_length: int) -> Optional[int]:
        """Return the padded length for compiled model, or None if not compiling.

        When compiled with dynamic=False, inputs are padded to the nearest
        multiple of compile_interval so that torch.compile reuses
        shape-specialized kernels instead of recompiling for every new length.
        """
        if not self._compiled or self._compile_dynamic:
            return None
        return math.ceil(true_length / self._compile_interval) * self._compile_interval

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _restype_idx_to_str(restype_idx: torch.Tensor) -> str:
        seq = []
        for x in restype_idx.detach().cpu().tolist():
            if x < 0:
                x = rc.restype_num
            if x > rc.restype_num:
                x = rc.restype_num
            seq.append(rc.restypes_with_x[x])
        return "".join(seq)

    def _set_template_use_unit_vector(self, zero_template_unit_vector: bool) -> None:
        if self._template_use_unit_vector_default is None:
            return
        if zero_template_unit_vector:
            self.model.template_embedder.config.use_unit_vector = False
        else:
            self.model.template_embedder.config.use_unit_vector = self._template_use_unit_vector_default

    @staticmethod
    def _apply_template_overrides(
        batch: dict,
        *,
        mask_template_aatype: bool,
        zero_template_torsion_angles: bool,
    ) -> None:
        if mask_template_aatype and "template_aatype" in batch:
            batch["template_aatype"] = torch.full_like(batch["template_aatype"], rc.restype_num)
        if zero_template_torsion_angles:
            for key in ("template_torsion_angles_sin_cos", "template_alt_torsion_angles_sin_cos"):
                if key in batch:
                    batch[key] = torch.zeros_like(batch[key])

    @staticmethod
    def _make_template_stub_features(
        sequence: str,
        mask: np.ndarray,
    ) -> dict:
        num_res = len(sequence)
        sequence_features = make_sequence_features(
            sequence=sequence,
            description="distogram_template",
            num_res=num_res,
        )
        msa_features = make_dummy_msa_feats(sequence)
        template_seq = sequence
        template_aatype_one_hot = rc.sequence_to_onehot(
            sequence=template_seq,
            mapping=rc.restype_order_with_x,
            map_unknown_to_x=True,
        ).astype(np.float32)
        template_aatype = np.zeros(
            (1, num_res, len(rc.restypes_with_x_and_gap)), dtype=np.float32
        )
        template_aatype[..., : template_aatype_one_hot.shape[-1]] = template_aatype_one_hot[None, ...]
        mask_f = mask.astype(np.float32)
        template_features = {
            "template_aatype": template_aatype,
            "template_all_atom_mask": np.ones(
                (1, num_res, rc.atom_type_num), dtype=np.float32
            ),
            "template_all_atom_positions": np.zeros(
                (1, num_res, rc.atom_type_num, 3), dtype=np.float32
            ),
            "template_pseudo_beta_mask": mask_f[None, ...],
            "template_pseudo_beta": np.zeros((1, num_res, 3), dtype=np.float32),
            "template_domain_names": np.array([b"distogram_template"], dtype=object),
            "template_sequence": np.array([template_seq.encode("utf-8")], dtype=object),
            "template_sum_probs": np.array([[1.0]], dtype=np.float32),
        }
        return {**sequence_features, **msa_features, **template_features}

    @staticmethod
    def _inject_template_dgram_probs(
        batch: dict,
        distogram_probs: torch.Tensor,
    ) -> None:
        if distogram_probs is None or "template_aatype" not in batch:
            return
        dgram = distogram_probs
        if dgram.dim() == 4:
            dgram = dgram[0]
        if dgram.dim() != 3:
            raise ValueError(f"Expected distogram_probs [L, L, B], got shape {tuple(dgram.shape)}")
        num_res = batch["template_aatype"].shape[1]
        if dgram.shape[0] != num_res or dgram.shape[1] != num_res:
            raise ValueError(
                f"Distogram size {tuple(dgram.shape)} does not match template length {num_res}"
            )
        num_templates = batch["template_aatype"].shape[0]
        dgram = dgram.unsqueeze(0).expand(num_templates, -1, -1, -1)
        num_ensembles = batch["template_aatype"].shape[-1] if batch["template_aatype"].dim() > 2 else 1
        dgram = dgram.unsqueeze(-1).expand(-1, -1, -1, -1, num_ensembles)
        device = batch["template_aatype"].device
        dtype = (
            batch["template_all_atom_positions"].dtype
            if "template_all_atom_positions" in batch
            else distogram_probs.dtype
        )
        batch["template_dgram_probs"] = dgram.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Single-sample batch construction
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_template_noncovered(batch: dict, segment_mask) -> None:
        """Zero template coord masks on uncovered (gap) residues so the composite
        template's placeholder CAs contribute no geometry. Applied in BOTH the
        mask and nomask cells (independent of the cross-segment toggle)."""
        if segment_mask is None:
            return
        ref = batch.get("template_all_atom_mask", batch.get("template_pseudo_beta_mask"))
        if ref is None:
            return
        cov = torch.as_tensor(segment_mask, device=ref.device, dtype=ref.dtype).reshape(-1)
        if "template_all_atom_mask" in batch:
            m = batch["template_all_atom_mask"]
            batch["template_all_atom_mask"] = m * cov.reshape(*([1] * (m.dim() - 3)), -1, 1, 1)
        if "template_pseudo_beta_mask" in batch:
            m = batch["template_pseudo_beta_mask"]
            batch["template_pseudo_beta_mask"] = m * cov.reshape(*([1] * (m.dim() - 2)), -1, 1)

    @staticmethod
    def _inject_template_segment_compat(batch: dict, segment_ids) -> None:
        """Inject a block-diagonal cross-segment compatibility mask under key
        'template_segment_compat_2d' (shape [T, N, N, E]). TemplateEmbedder.forward
        multiplies it into pair_mask, removing ordered-to-ordered cross-segment
        template pairs. Mirrors _inject_template_dgram_probs (trailing ensemble dim
        is required because the outer forward slices the recycling dim). Only called
        when the --mask_inter_segment toggle is ON."""
        if segment_ids is None or "template_aatype" not in batch:
            return
        ta = batch["template_aatype"]
        device = ta.device
        seg = torch.as_tensor(segment_ids, device=device, dtype=torch.long).reshape(-1)
        num_res = ta.shape[1]
        if seg.shape[0] != num_res:
            raise ValueError(f"segment_ids length {seg.shape[0]} != template length {num_res}")
        compat = (seg[:, None] == seg[None, :]) & (seg[:, None] >= 0)  # [N, N]
        num_templates = ta.shape[0]
        compat = compat.unsqueeze(0).expand(num_templates, -1, -1)  # [T, N, N]
        num_ensembles = ta.shape[-1] if ta.dim() > 2 else 1
        compat = compat.unsqueeze(-1).expand(-1, -1, -1, num_ensembles)  # [T, N, N, E]
        dtype = (
            batch["template_all_atom_positions"].dtype
            if "template_all_atom_positions" in batch else torch.float32
        )
        batch["template_segment_compat_2d"] = compat.to(device=device, dtype=dtype)

    def build_batch(
        self,
        distogram_probs: torch.Tensor,
        residue_type: torch.Tensor,
        mask: torch.Tensor,
        *,
        template_mode: str = "distogram_only",
        template_mmcif_path: Optional[str] = None,
        template_chain_id: Optional[str] = None,
        kalign_binary_path: Optional[str] = None,
        mask_template_aatype: bool = False,
        zero_template_unit_vector: bool = False,
        zero_template_torsion_angles: bool = False,
        _pad_to_length: Optional[int] = None,
        segment_ids=None,
        segment_mask=None,
        seed: Optional[int] = None,
        skip_template_alignment: Optional[bool] = None,
    ):
        """
        Build a feature dict for a single sample.

        Args:
            distogram_probs: [1, L, L, 39] float probabilities
            residue_type: [1, L] ints in [0..20] (20=unknown)
            mask: [1, L] bool/float
            _pad_to_length: (internal) Pad features to this length for
                batching with variable-length sequences.  When None the
                features are truncated to the true (unmasked) length.
            seed: (optional) If set, seed all RNG sources before running
                the feature pipeline so that processing is deterministic.
                The feature pipeline otherwise introduces randomness via
                make_masked_msa (15% random MSA masking) even with a
                single-sequence MSA.
        Returns:
            dict of tensors (no batch dim) ready for the AlphaFold model.
        """
        b, n = residue_type.shape[:2]
        if b != 1:
            raise ValueError(
                f"build_batch expects a single sample (batch dim 1), got {b}. "
                "Use build_batch_multi / forward_batched for multiple samples."
            )

        # Compute true (unmasked) length
        if mask.dtype != torch.float32:
            mask_f = mask.float()
        else:
            mask_f = mask
        l = int(mask_f[0].sum().item())
        if l <= 0:
            raise ValueError("Mask has no valid residues")

        # Target length: true length, or padded for batching
        target_l = l if _pad_to_length is None else max(l, _pad_to_length)

        # Truncate residue_type to true length, optionally pad with X (20)
        residue_type_trunc = residue_type[:, :l]
        if target_l > l:
            pad_rt = torch.full(
                (1, target_l - l), rc.restype_num,
                dtype=residue_type.dtype, device=residue_type.device,
            )
            residue_type = torch.cat([residue_type_trunc, pad_rt], dim=1)
        else:
            residue_type = residue_type_trunc

        # Mask: 1.0 for real residues, 0.0 for padding
        mask_f = torch.zeros((1, target_l), dtype=torch.float32, device=residue_type.device)
        mask_f[0, :l] = 1.0

        # Sequence string (pad with X)
        seq = self._restype_idx_to_str(residue_type[0][:l])
        if target_l > l:
            seq += "X" * (target_l - l)

        template_mode = template_mode.lower()
        if template_mode == "full_template_zero_coords":
            zero_template_unit_vector = True
            zero_template_torsion_angles = True

        # Truncate (and pad) distogram to target_l
        if distogram_probs is not None:
            distogram_probs = distogram_probs[:, :l, :l, :]
            if target_l > l:
                pad = target_l - l
                distogram_probs = F.pad(distogram_probs, (0, 0, 0, pad, 0, pad))

        if template_mode == "distogram_only":
            if distogram_probs is None:
                raise ValueError("distogram_probs is required for template_mode=distogram_only")
            mask_np = mask_f[0].detach().cpu().numpy()
            if template_mmcif_path is not None:
                if kalign_binary_path is None:
                    raise ValueError("kalign_binary_path is required for custom-template distogram mode")
                if template_chain_id is None:
                    raise ValueError("template_chain_id is required for custom-template distogram mode")
                pdb_id = os.path.splitext(os.path.basename(template_mmcif_path))[0]
                raw = make_sequence_features_with_custom_template(
                    sequence=seq,
                    mmcif_path=template_mmcif_path,
                    pdb_id=pdb_id,
                    chain_id=template_chain_id,
                    kalign_binary_path=kalign_binary_path,
                    rm_template_sequence=False,
                    skip_alignment=(self.skip_template_alignment if skip_template_alignment is None else skip_template_alignment),
                )
            else:
                raw = self._make_template_stub_features(
                    sequence=seq,
                    mask=mask_np,
                )
            zero_template_unit_vector = True
            zero_template_torsion_angles = True
        elif template_mode in ("full_template", "full_template_zero_coords"):
            if template_mmcif_path is None:
                raise ValueError("template_mmcif_path is required for full template modes")
            if kalign_binary_path is None:
                raise ValueError("kalign_binary_path is required for full template modes")
            if template_chain_id is None:
                raise ValueError("template_chain_id is required for full template modes")
            pdb_id = os.path.splitext(os.path.basename(template_mmcif_path))[0]
            raw = make_sequence_features_with_custom_template(
                sequence=seq,
                mmcif_path=template_mmcif_path,
                pdb_id=pdb_id,
                chain_id=template_chain_id,
                kalign_binary_path=kalign_binary_path,
                rm_template_sequence=False,
                skip_alignment=(self.skip_template_alignment if skip_template_alignment is None else skip_template_alignment),
            )
        else:
            raise ValueError(
                "Unknown template_mode. Expected one of: distogram_only, full_template, full_template_zero_coords"
            )
        # Seed all RNG sources so the feature pipeline is deterministic.
        # The primary source of randomness is make_masked_msa (randomly
        # replaces 15% of MSA positions) and ensemble_seed in
        # input_pipeline.process_tensors_from_config, both of which use
        # unseeded global RNG by default.
        if seed is not None:
            _random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        feats = self.feature_pipeline.process_features(raw, mode="predict", is_multimer=False)

        batch = tensor_tree_map(lambda x: x.to(self.device), feats)

        # When padding was applied, the feature pipeline sets seq_mask /
        # msa_mask to all-ones for the padded length.  Override them so the
        # model correctly ignores padding positions.
        if target_l > l:
            self._apply_padding_masks(batch, true_length=l, padded_length=target_l)

        if template_mode == "distogram_only":
            self._inject_template_dgram_probs(batch, distogram_probs)
        self._set_template_use_unit_vector(zero_template_unit_vector)
        self._apply_template_overrides(
            batch,
            mask_template_aatype=mask_template_aatype,
            zero_template_torsion_angles=zero_template_torsion_angles,
        )
        if segment_mask is not None:
            self._zero_template_noncovered(batch, segment_mask)
        if segment_ids is not None:
            self._inject_template_segment_compat(batch, segment_ids)
        return batch

    @staticmethod
    def _apply_padding_masks(
        batch: dict,
        true_length: int,
        padded_length: int,
    ) -> None:
        """Zero out masks for padding positions (indices >= true_length).

        The feature pipeline creates masks assuming the full sequence is valid.
        After padding for batching, we override seq_mask, msa_mask, and
        extra_msa_mask so the model ignores padded positions.
        """
        # Map from feature key -> dimension index where N_res appears.
        # The last dimension is always the recycling dim for processed features.
        mask_keys = {
            "seq_mask": 0,           # [N_res, R]
            "msa_mask": 1,           # [N_seq, N_res, R]
            "extra_msa_mask": 1,     # [N_extra, N_res, R]
        }
        for key, res_dim in mask_keys.items():
            if key not in batch:
                continue
            t = batch[key]
            # Build a slice that zeros positions [true_length:] along res_dim
            slices = [slice(None)] * t.dim()
            slices[res_dim] = slice(true_length, None)
            t[tuple(slices)] = 0.0

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------

    def build_batch_multi(
        self,
        distogram_probs_list: List[torch.Tensor],
        residue_type_list: List[torch.Tensor],
        mask_list: List[torch.Tensor],
        _pad_to_length: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """Build a batched feature dict from multiple samples.

        Each input element should have shape [1, L_i, ...] (single-sample
        with potentially different lengths L_i).  Samples are padded to the
        maximum true length (or _pad_to_length if larger) and stacked along
        a new leading batch dimension.

        Args:
            distogram_probs_list: list of [1, L_i, L_i, 39] tensors
            residue_type_list:    list of [1, L_i] tensors
            mask_list:            list of [1, L_i] tensors
            _pad_to_length: (internal) Override the padding target length.
                Useful when tracing requires a fixed length.
            **kwargs: forwarded to build_batch (template_mode, etc.)

        Returns:
            dict of tensors with a leading batch dimension [B, ...].
        """
        n_samples = len(distogram_probs_list)
        if n_samples == 0:
            raise ValueError("Empty sample list")
        if not (n_samples == len(residue_type_list) == len(mask_list)):
            raise ValueError("All input lists must have the same length")

        # Compute true (unmasked) lengths to determine padding target
        lengths = []
        for mask in mask_list:
            mask_f = mask.float() if mask.dtype != torch.float32 else mask
            lengths.append(int(mask_f[0].sum().item()) if mask_f.dim() >= 2 else int(mask_f.sum().item()))
        max_l = _pad_to_length if _pad_to_length is not None else max(lengths)

        # Build each single-sample batch, padded to max_l
        batches = []
        for i in range(n_samples):
            single = self.build_batch(
                distogram_probs_list[i],
                residue_type_list[i],
                mask_list[i],
                _pad_to_length=max_l,
                **kwargs,
            )
            batches.append(single)

        # Stack along a new leading batch dimension
        collated = {}
        for key in batches[0].keys():
            collated[key] = torch.stack([b[key] for b in batches], dim=0)
        return collated

    def forward_batched(
        self,
        distogram_probs_list: List[torch.Tensor],
        residue_type_list: List[torch.Tensor],
        mask_list: List[torch.Tensor],
        **kwargs,
    ) -> dict:
        """Run batched inference on multiple samples.

        Accepts lists of per-sample tensors (each with batch dim 1), pads to
        the longest sequence, stacks into a true batch, and runs a single
        forward pass through the model.

        Args:
            distogram_probs_list: list of [1, L_i, L_i, 39] tensors
            residue_type_list:    list of [1, L_i] tensors
            mask_list:            list of [1, L_i] tensors
            **kwargs: forwarded to build_batch (template_mode, etc.)

        Returns:
            dict of model outputs.  Coordinate tensors have shape
            [B, L_max, ...]; use the per-sample masks to extract valid
            residues.
        """
        # When compiled with dynamic=False, pad to interval bucket
        lengths = []
        for m in mask_list:
            mf = m.float() if m.dtype != torch.float32 else m
            lengths.append(int(mf[0].sum().item()) if mf.dim() >= 2 else int(mf.sum().item()))
        max_true_l = max(lengths)
        _pad_to = self._compute_pad_length(max_true_l)

        batch = self.build_batch_multi(
            distogram_probs_list,
            residue_type_list,
            mask_list,
            _pad_to_length=_pad_to,
            **kwargs,
        )

        with torch.no_grad():
            out = self.model(batch)
        return out

    # ------------------------------------------------------------------
    # Original single-sample forward (backward-compatible)
    # ------------------------------------------------------------------

    def forward(
        self,
        distogram_probs: torch.Tensor,
        residue_type: torch.Tensor,
        mask: torch.Tensor,
        *,
        template_mode: str = "distogram_only",
        template_mmcif_path: Optional[str] = None,
        template_chain_id: Optional[str] = None,
        kalign_binary_path: Optional[str] = None,
        mask_template_aatype: bool = False,
        zero_template_unit_vector: bool = False,
        zero_template_torsion_angles: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            distogram_probs: [B, L, L, 39] float probabilities
            residue_type: [B, L] ints in [0..20] (20=unknown)
            mask: [B, L] bool/float
            seed: (optional) Seed for deterministic feature pipeline processing.
        Returns:
            dict with atom37 coordinates [B, L, 37, 3]
        """
        # When compiled with dynamic=False, pad to interval bucket
        true_l = int(mask.float().reshape(-1).sum().item()) if mask.dim() >= 2 else int(mask.float().sum().item())
        _pad_to = self._compute_pad_length(true_l)

        batch = self.build_batch(
            distogram_probs,
            residue_type,
            mask,
            template_mode=template_mode,
            template_mmcif_path=template_mmcif_path,
            template_chain_id=template_chain_id,
            kalign_binary_path=kalign_binary_path,
            mask_template_aatype=mask_template_aatype,
            zero_template_unit_vector=zero_template_unit_vector,
            zero_template_torsion_angles=zero_template_torsion_angles,
            seed=seed,
            _pad_to_length=_pad_to,
        )

        with torch.no_grad():
            out = self.model(batch)
        return out
