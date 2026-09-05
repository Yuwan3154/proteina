"""Gate the weight EMA before it costs a training run.

Four things that would each silently corrupt training or evaluation:
  1. EMA must advance ONLY on real optimizer steps -- updating per micro-batch would run the
     average accumulate_grad_batches times too fast and quietly shorten its horizon.
  2. Validation must run on the EMA weights AND restore the raw ones afterwards -- a failed
     restore would have training continue from averaged weights.
  3. The checkpoint must carry ema/params, and a warm start must prefer it over state_dict.
  4. Integer buffers must be copied, not averaged (0.999*int is not an int).
"""

import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


TINY = dict(c_s=32, c_z=16, c_token=32, c_atom=16, c_atompair=8,
            n_blocks=1, n_heads=2, n_tri_blocks=1, tri_hidden=16, transition_n=1,
            atom_blocks=1, atom_heads=2)


def fake_trainer(accum, last=False):
    return types.SimpleNamespace(accumulate_grad_batches=accum, is_last_batch=last)


def main():
    torch.manual_seed(0)
    m = ContactToCoordTrainer(model_cfg=dict(TINY, n_diffusion_samples=2), ema_decay=0.9)
    m.trainer = fake_trainer(accum=4)
    name = "model.seq_emb.weight"
    key = "seq_emb.weight"

    check("no EMA before any step", m._ema is None)

    # ---- 1. updates only on optimizer-step boundaries ----
    for b in range(3):                       # batch_idx 0,1,2 -> not a boundary at accum=4
        m.on_train_batch_end(None, None, b)
    check("no EMA after 3 of 4 accumulation micro-batches", m._ema is None)
    m.on_train_batch_end(None, None, 3)      # (3+1) % 4 == 0 -> boundary
    check("EMA created on the optimizer-step boundary", m._ema is not None)

    # ---- 4. int buffers copied, not scaled ----
    ints = [k for k, v in m.model.state_dict().items() if not v.is_floating_point()]
    if ints:
        ok = all(torch.equal(m._ema[k], m.model.state_dict()[k]) for k in ints)
        check(f"integer buffers copied verbatim ({len(ints)} found)", ok)
    else:
        check("integer buffers copied verbatim (none present)", True)

    # ---- EMA math: copy = d*copy + (1-d)*param ----
    before = m._ema[key].clone()
    with torch.no_grad():
        m.model.seq_emb.weight.add_(1.0)          # move the live weight by exactly +1
    live = m.model.state_dict()[key].clone()
    m.on_train_batch_end(None, None, 7)           # another boundary
    want = 0.9 * before + 0.1 * live
    check("EMA update matches d*copy + (1-d)*param",
          torch.allclose(m._ema[key], want, atol=1e-6),
          f"max|diff|={float((m._ema[key]-want).abs().max()):.2e}")
    check("EMA lags the live weight (it is actually averaging)",
          not torch.allclose(m._ema[key], live, atol=1e-4))

    # ---- 2. validation swaps to EMA, then restores ----
    raw = m.model.state_dict()[key].clone()
    m.on_validation_start()
    during = m.model.state_dict()[key].clone()
    check("validation runs on the EMA weights",
          torch.allclose(during, m._ema[key].to(during.dtype), atol=1e-6))
    check("...which differ from the raw weights", not torch.allclose(during, raw, atol=1e-4))
    m.on_validation_end()
    after = m.model.state_dict()[key].clone()
    check("raw weights restored exactly after validation", torch.equal(after, raw))
    check("cache released", m._cached is None)

    # ---- 3. checkpoint round-trip ----
    ck = {}
    m.on_save_checkpoint(ck)
    check("checkpoint carries ema/params", "ema" in ck and "params" in ck["ema"])
    check("ema decay recorded", ck["ema"].get("decay") == 0.9)
    check("ema keys are the INNER model's (no 'model.' prefix)",
          key in ck["ema"]["params"] and name not in ck["ema"]["params"])

    m2 = ContactToCoordTrainer(model_cfg=dict(TINY, n_diffusion_samples=2), ema_decay=0.9)
    m2.on_load_checkpoint(ck)
    check("EMA restored from checkpoint",
          m2._ema is not None and torch.allclose(m2._ema[key], m._ema[key], atol=1e-6))
    # the warm-start path loads ema params into the INNER model
    m2.model.load_state_dict({k: v for k, v in ck["ema"]["params"].items()}, strict=True)
    check("ema params load into the inner model with strict=True", True)

    print(f"\n{len(PASS)}/{len(PASS)+len(FAIL)} passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
