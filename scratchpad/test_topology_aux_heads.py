"""Gate for the Q x T alignment head and the masked-token (MLM) head on ContactMapTriSiT, and for the
trainer's `_topology_aux_losses` that consumes them.

Must hold:
  1. heads off => no extra parameters, no extra outputs (the baseline model is untouched);
  2. heads on => align_logits [B, L, T] (zero outside real cells), align_none_logits [B, L],
     mlm_logits [B, T, vocab];
  3. weights 0 => the helper returns exactly 0 and adds nothing;
  4. softmax and bce alignment losses are finite, gradients reach the TRUNK (end to end, not a
     frozen probe), samples without a single aligned residue contribute nothing;
  5. precision@Q matches a hand computation;
  6. the MLM loss uses only masked positions (target > 1), ignoring PAD (0) and collate padding (-1).
"""

import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.sse_topology import N_PAIR_FEATURES
from proteinfoundation.nn.contact_map_tri import ContactMapTriSiT
from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase

PASS, FAIL = [], []
B, L, T, V = 2, 12, 3, 44


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def make_model(align=False, mlm=False):
    torch.manual_seed(0)
    return ContactMapTriSiT(
        pair_dim=32, tri_hidden=32, n_blocks=1, transition_n=2, dim_cond=16, max_topology_he_len=8,
        max_rel_pos=16, topology_vocab_size=V, n_residue_types=22, pair_ref_features="both",
        align_head={"enabled": align}, mlm_head={"enabled": mlm},
    )


def make_batch():
    g = torch.Generator().manual_seed(1)
    mask = torch.ones(B, L)
    mask[1, 9:] = 0.0                               # sample 1 has 9 real residues
    he_tokens = torch.randint(2, V, (B, T), generator=g)
    he_tokens[1, 2] = 0                             # sample 1 has 2 real elements
    he_tokens[0, 1] = 1                             # a MASKed element (still real)
    batch = {
        "contact_map_t": (torch.rand(B, L, L, generator=g) < 0.1).float(),
        "mask": mask,
        "t": torch.rand(B, generator=g),
        "residue_type": torch.randint(0, 20, (B, L), generator=g),
        "topology_he_tokens": he_tokens,
        "topology_he_pos_raw": torch.tensor([[3.0, 8.0, 11.0], [2.0, 6.0, 0.0]]),
        "topology_he_feat": torch.randn(B, T, T, N_PAIR_FEATURES, generator=g),
    }
    # alignment ground truth: residues 0-3 -> element 0, 5-7 -> element 2 (sample 0); sample 1 none
    tgt = torch.full((B, L), -1, dtype=torch.long)
    tgt[0, 0:4] = 0
    tgt[0, 5:8] = 2
    batch["ref_align_target"] = tgt
    # masked-token targets: sample 0 element 1 was masked (target = original token 17); padding -1
    mt = torch.zeros(B, T, dtype=torch.long)
    mt[0, 1] = 17
    mt[1, 2] = -1
    batch["topology_he_tokens_target"] = mt
    batch["topology_missing_ref"] = torch.zeros(B, 1, dtype=torch.long)
    return batch


class Stub:
    """Just enough of the trainer for the unbound helper: a loss config and a recording log()."""

    def __init__(self, **loss):
        self.cfg_exp = types.SimpleNamespace(loss=loss)
        self.logged = {}

    def log(self, name, value, **kw):
        self.logged[name] = float(value)


def trunk_grad_norm(model):
    """Total gradient mass on the trunk (blocks + embeddings), excluding the heads themselves."""
    tot = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None and not name.startswith(("align_head", "align_none", "mlm_head", "out.", "out_norm")):
            tot += float(p.grad.abs().sum())
    return tot


def run_helper(nn_out, batch, **loss):
    stub = Stub(**loss)
    total = ModelTrainerBase._topology_aux_losses(stub, nn_out, batch, batch["mask"], "train")
    return total, stub.logged


def main():
    plain = make_model()
    both = make_model(align=True, mlm=True)
    n_plain = sum(p.numel() for p in plain.parameters())
    n_both = sum(p.numel() for p in both.parameters())
    check("heads off: no extra parameters vs heads on", n_both - n_plain == 2 * (32 + 1) + (32 * V + V),
          f"delta={n_both - n_plain}")
    batch = make_batch()
    out_plain = plain(dict(batch))
    check("heads off: no align/mlm outputs", not any(k in out_plain for k in ("align_logits", "align_none_logits", "mlm_logits")))

    out = both(dict(batch))
    check("align_logits shape [B, L, T]", tuple(out["align_logits"].shape) == (B, L, T))
    check("align_none_logits shape [B, L]", tuple(out["align_none_logits"].shape) == (B, L))
    check("mlm_logits shape [B, T, vocab]", tuple(out["mlm_logits"].shape) == (B, T, V))
    check("align_logits zero on padded residues / absent elements",
          float(out["align_logits"][1, 9:].abs().sum()) == 0.0 and float(out["align_logits"][1, :, 2].abs().sum()) == 0.0)
    check("contact logits unchanged by the heads (same trunk, same seed)",
          torch.allclose(out["contact_map_logits"], out_plain["contact_map_logits"], atol=1e-6))

    # 3. weights 0 -> exactly zero
    total, logged = run_helper(out, batch, align_loss_weight=0.0, mlm_loss_weight=0.0)
    check("weights 0: total is exactly 0", float(total) == 0.0)
    check("weights 0: only the missing-ref rate is logged", set(logged) == {"train/topology_missing_ref_frac"}, str(set(logged)))

    # 4. softmax form: finite, gradient reaches the trunk, sample without GT contributes nothing
    for form in ("softmax", "bce"):
        both.zero_grad(set_to_none=True)
        out = both(dict(batch))
        total, logged = run_helper(out, batch, align_loss_weight=1.0, mlm_loss_weight=0.0, align_loss=form)
        check(f"[{form}] alignment loss finite and positive", torch.isfinite(total) and float(total) > 0, f"{float(total):.4f}")
        total.backward()
        # at init OpenFold's zero-init output projections make the FIRST parameters' gradients zero, so
        # test the block as a whole (its output projections and the embeddings do receive gradient)
        check(f"[{form}] gradient reaches the trunk (end to end)", trunk_grad_norm(both) > 0)
        check(f"[{form}] align_frac_samples = 0.5 (one of two samples has ground truth)", abs(logged["train/align_frac_samples"] - 0.5) < 1e-6)
        check(f"[{form}] precision@Q logged in [0, 1]", 0.0 <= logged["train/align_precision_at_q"] <= 1.0)

    # 5. precision@Q by hand: logits that put element 0 on residues 0-3 and element 2 on 5-7 => 1.0
    perfect = dict(out)
    a = torch.full((B, L, T), -5.0)
    a[0, 0:4, 0] = 5.0
    a[0, 5:8, 2] = 5.0
    perfect["align_logits"] = a
    perfect["align_none_logits"] = torch.zeros(B, L)
    _, logged = run_helper(perfect, batch, align_loss_weight=1.0, align_loss="softmax")
    check("precision@Q = 1.0 for a perfect ranking", abs(logged["train/align_precision_at_q"] - 1.0) < 1e-6)
    a2 = a.clone()
    a2[0, 0:4, 0] = -5.0
    a2[0, 0:4, 1] = 5.0          # wrong element for residues 0-3 => 3 of 7 top cells correct
    perfect["align_logits"] = a2
    _, logged = run_helper(perfect, batch, align_loss_weight=1.0, align_loss="softmax")
    check("precision@Q = 3/7 when 4 of 7 top cells are wrong", abs(logged["train/align_precision_at_q"] - 3 / 7) < 1e-6,
          f"{logged['train/align_precision_at_q']:.4f}")

    # 6. MLM: only the masked position counts
    both.zero_grad(set_to_none=True)
    out = both(dict(batch))
    total, logged = run_helper(out, batch, align_loss_weight=0.0, mlm_loss_weight=1.0)
    check("mlm: one masked element counted", logged["train/mlm_n_masked"] == 1.0)
    check("mlm loss finite and positive", torch.isfinite(total) and float(total) > 0)
    ce = torch.nn.functional.cross_entropy(out["mlm_logits"][0, 1:2], torch.tensor([17]))
    check("mlm loss equals CE at the masked position", abs(float(total) - float(ce)) < 1e-5)
    total.backward()
    check("mlm gradient reaches the trunk", trunk_grad_norm(both) > 0)
    b2 = dict(batch)
    b2["topology_he_tokens_target"] = torch.zeros(B, T, dtype=torch.long)
    both.zero_grad(set_to_none=True)
    out2 = both(dict(batch))
    total, logged = run_helper(out2, b2, align_loss_weight=0.0, mlm_loss_weight=1.0)
    check("mlm: no masked positions -> loss 0, n_masked 0", float(total) == 0.0 and logged["train/mlm_n_masked"] == 0.0)
    # ⛔ DDP aborts on a parameter that produced no gradient, so an empty step must still put the
    # head in the graph (measured: the 1-GPU smoke died with "parameters that were not used").
    total.backward()
    check("mlm: empty step still gives mlm_head a gradient (DDP)",
          both.mlm_head.weight.grad is not None and float(both.mlm_head.weight.grad.abs().sum()) == 0.0)
    both.zero_grad(set_to_none=True)
    out3 = both(dict(batch))
    tb, _ = run_helper(out3, batch, align_loss_weight=1.0, mlm_loss_weight=0.0, align_loss="bce")
    tb.backward()
    check("bce: align_none still gets a (zero) gradient (DDP)",
          both.align_none.weight.grad is not None and float(both.align_none.weight.grad.abs().sum()) == 0.0)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
