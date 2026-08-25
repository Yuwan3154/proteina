"""Runtime instrumentation for the contact-map sampling loop.

Exists because self-conditioning correctness was asserted from static reading and had to be
withdrawn. Claims about what the sampler does must rest on RUNTIME TENSOR VALUES, so this
records, per step, what was actually fed to the network and what came back:

  * ``sc_present``     -- was the ``contact_map_sc`` key actually in nn_in?
  * ``sc_norm``        -- its L2 norm (0.0 would mean self-cond is fed but empty)
  * ``sc_is_prev``     -- does it EQUAL the previous step's prediction (bit-exact)?
  * ``c_norm``/``pred_norm`` -- so a collapsing or saturating trajectory is visible
  * the effective sampler args as seen INSIDE the loop, to confirm YAML values arrive

Optionally snapshots the contact map for trajectory figures.

Dependency-light on purpose (numpy only, imported lazily) so it can be unit tested standalone.
"""

import os
from typing import Any, Dict, List, Optional

_STEPS: List[Dict[str, Any]] = []
_MAPS: List[Any] = []
_ARGS: Dict[str, Any] = {}
_PREV_PRED = None


def enabled() -> bool:
    return os.environ.get("SAMPLETRACE") == "1"


def snapshot_every() -> int:
    """0 disables map snapshots; N stores every Nth step."""
    try:
        return int(os.environ.get("SAMPLETRACE_EVERY", "0"))
    except ValueError:
        return 0


def record_args(**kwargs) -> None:
    """Effective sampler arguments as seen inside the loop."""
    if not enabled():
        return
    _ARGS.update({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in kwargs.items()})


def record_step(step: int, t: float, nn_in: Dict[str, Any], c: Any, pred: Any) -> None:
    global _PREV_PRED
    if not enabled():
        return
    import torch

    sc = nn_in.get("contact_map_sc")
    sc_present = sc is not None
    sc_norm = float(sc.float().norm().item()) if sc_present else 0.0
    if sc_present and _PREV_PRED is not None:
        sc_is_prev = bool(torch.equal(sc.float(), _PREV_PRED.float()))
    else:
        sc_is_prev = False
    _STEPS.append({
        "step": int(step),
        "t": float(t),
        "sc_present": bool(sc_present),
        "sc_norm": sc_norm,
        "sc_is_prev_pred": sc_is_prev,
        "c_norm": float(c.float().norm().item()) if c is not None else float("nan"),
        "pred_norm": float(pred.float().norm().item()) if pred is not None else float("nan"),
        "pred_mean": float(pred.float().mean().item()) if pred is not None else float("nan"),
        "pred_frac_gt_half": float((pred.float() > 0.5).float().mean().item()) if pred is not None else float("nan"),
    })
    every = snapshot_every()
    if every > 0 and (step % every == 0) and pred is not None:
        _MAPS.append((int(step), float(t), pred.detach().float().cpu().numpy()))
    _PREV_PRED = pred.detach() if pred is not None else None


def reset() -> None:
    global _PREV_PRED
    _STEPS.clear()
    _MAPS.clear()
    _ARGS.clear()
    _PREV_PRED = None


def summary() -> Dict[str, Any]:
    return {"args": dict(_ARGS), "n_steps": len(_STEPS), "steps": list(_STEPS)}


def dump(path: str) -> Optional[str]:
    if not enabled() or not _STEPS:
        return None
    import json

    import numpy as np

    with open(path + ".json", "w") as fh:
        json.dump(summary(), fh, indent=2)
    if _MAPS:
        np.savez_compressed(
            path + "_maps.npz",
            steps=np.array([m[0] for m in _MAPS]),
            ts=np.array([m[1] for m in _MAPS]),
            maps=np.stack([m[2] for m in _MAPS]),
        )
    return path
