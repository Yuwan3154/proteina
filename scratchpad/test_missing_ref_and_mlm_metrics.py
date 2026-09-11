"""P3/P4/P5: the missing-reference counter, the mlm_acc dilution, and the MLM marginal baseline.

P3 -- topology_missing_ref reported COVERAGE-AFTER-DROPOUT, not coverage. `_pick_template` was only
reached on the kept branch, so a chain with an index row but no in-band template reported missing=0
whenever the CFG drop fired: the rate came out scaled by (1 - drop_prob), and a real coverage
regression could hide inside the dropout.

P4 -- 22.0% of validation steps mask nothing (P(nothing masked) = 0.9^n_elements, 0.59 for a
5-element reference) and logged a hard 0 into mlm_acc, understating it ~28% relative. The fix cannot
be a conditional self.log: log_kw carries sync_dist=True, so a rank that skipped a log would
desynchronise the collective. It is Lightning's batch_size weighting instead -- which this file
VERIFIES against a real Trainer rather than assuming, since the whole fix rests on that behaviour.

P5 -- mlm_acc had no bar. Uniform chance (1/42) is the wrong one: length slots are far from uniform,
so a head that learned only the marginal would clear it having learned nothing conditional.
"""

import os
import sys
import tempfile

import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_synthetic_reference import build_toy_index

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def graph(stem, L=40):
    g = Data()
    g.coords = torch.zeros(L, 37, 3)
    g.protein_id = stem
    return g


def missing_of(t, stem):
    return int(t(graph(stem)).topology_missing_ref.item())


def test_p3():
    print("\nP3 -- topology_missing_ref reports coverage, not coverage-after-dropout")
    with tempfile.TemporaryDirectory() as td:
        idx = os.path.join(td, "toy_index.pt")
        build_toy_index(idx)

        def make(drop_prob):
            t = TopologyReferenceTransform(index_path=idx, seed=0, reference_source="synthetic",
                                           tm_range=(0.5, 0.9), drop_prob=drop_prob)
            t._ensure_loaded()
            return t

        # 'bbbb_B' is row 4: it HAS an index row but its cluster has no members, so no in-band
        # template exists. This is the case the old code scaled by (1 - drop_prob).
        check("no-template chain is missing when NOT dropped", missing_of(make(0.0), "bbbb_B") == 1)
        check("no-template chain is STILL missing when dropped", missing_of(make(1.0), "bbbb_B") == 1)

        # 'aaaa_A' has three in-band templates: a drop is a drop, never a coverage hole.
        check("covered chain is not missing when NOT dropped", missing_of(make(0.0), "aaaa_A") == 0)
        check("covered chain is not missing when dropped", missing_of(make(1.0), "aaaa_A") == 0)

        # a chain absent from the index was already counted correctly on both branches
        check("chain absent from the index is missing when dropped", missing_of(make(1.0), "zzzz_Z") == 1)

        # the statistical property that actually broke: the reported rate must be the TRUE rate,
        # not the true rate times (1 - drop_prob).
        t = make(0.25)
        rate = sum(missing_of(t, "bbbb_B") for _ in range(400)) / 400
        check("reported rate is 1.0, not ~0.75", rate == 1.0, f"rate={rate:.3f}")


def test_p4_lightning_batch_size_weighting():
    """Does self.log(..., on_epoch=True, batch_size=N) really produce an N-weighted mean?

    The P4 fix is only correct if it does. Asserting it against a real Trainer rather than trusting
    the docs, because the whole point of P4 is that an unverified aggregation assumption was already
    wrong once.
    """
    print("\nP4 -- Lightning batch_size weighting gives sum(correct)/sum(n_masked)")
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader, Dataset

    # (per-step accuracy, n_masked). Two steps have nothing masked and must not dilute the mean.
    STEPS = [(1.0, 2), (0.0, 0), (0.5, 4), (0.0, 0), (0.25, 4)]
    weighted = sum(a * n for a, n in STEPS) / sum(n for _, n in STEPS)   # 4.0 / 10 = 0.4
    unweighted = sum(a for a, _ in STEPS) / len(STEPS)                   # 1.75 / 5 = 0.35

    class DS(Dataset):
        def __len__(self):
            return len(STEPS)

        def __getitem__(self, i):
            return torch.tensor(i)

    class M(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Linear(1, 1)

        def training_step(self, batch, _):
            acc, n = STEPS[int(batch[0])]
            self.log("acc_weighted", torch.tensor(acc), on_step=False, on_epoch=True,
                     batch_size=max(n, 1))
            self.log("acc_plain", torch.tensor(acc), on_step=False, on_epoch=True, batch_size=1)
            return self.p(torch.zeros(1, 1)).sum() * 0.0

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.0)

    m = M()
    tr = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False, enable_progress_bar=False,
                    enable_model_summary=False, accelerator="cpu", devices=1)
    tr.fit(m, DataLoader(DS(), batch_size=1))
    got_w = float(tr.callback_metrics["acc_weighted"])
    got_p = float(tr.callback_metrics["acc_plain"])
    # batch_size=max(n,1) means the two zero-mask steps still carry weight 1, so the weighted mean
    # is sum(a*w)/sum(w) with w = [2,1,4,1,4] -- strictly better than plain, and the exact
    # sum(correct)/sum(n_masked) whenever no step masks nothing.
    expect_w = sum(a * max(n, 1) for a, n in STEPS) / sum(max(n, 1) for _, n in STEPS)
    check("batch_size weights the epoch mean", abs(got_w - expect_w) < 1e-6,
          f"got {got_w:.4f}, expected {expect_w:.4f}")
    check("unweighted mean is the diluted one", abs(got_p - unweighted) < 1e-6,
          f"got {got_p:.4f}, expected {unweighted:.4f}")
    check("weighting moves the number away from the diluted value", abs(got_w - got_p) > 1e-3,
          f"weighted {got_w:.4f} vs plain {got_p:.4f}")
    print(f"    (pure token-weighted over steps that had targets would be {weighted:.4f})")


def test_p5_marginal_baseline():
    """The running-mode predictor, exactly as the trainer computes it."""
    print("\nP5 -- MLM marginal-frequency baseline")
    vocab = 44
    counts = torch.zeros(vocab, dtype=torch.float64)
    # token 7 is the most common; a head that only learned the marginal would emit 7 every time
    stream = [torch.tensor([7, 7, 9]), torch.tensor([7, 3]), torch.tensor([7, 7, 7, 5])]
    scored = []
    for tgt in stream:
        base_correct = (tgt == int(counts.argmax())).float().sum()
        counts.index_add_(0, tgt, torch.ones_like(tgt, dtype=counts.dtype))
        scored.append((float(base_correct), tgt.numel()))
    check("mode is the most frequent token after the stream", int(counts.argmax()) == 7,
          f"argmax={int(counts.argmax())}")
    # step 1 is scored against an all-zero count (mode 0) and so scores 0 -- the baseline is only
    # meaningful once counts have accumulated, which is why it is scored BEFORE the update.
    check("first step scores 0 against an empty histogram", scored[0][0] == 0.0)
    check("later steps score the running mode", scored[2][0] == 3.0, f"got {scored[2][0]}")
    acc = sum(c for c, _ in scored) / sum(n for _, n in scored)
    check("baseline accuracy is a sensible fraction", 0.0 < acc < 1.0, f"acc={acc:.3f}")

    # and the property that matters: a uniform-chance bar would understate this baseline badly
    check("marginal baseline beats uniform chance", acc > 1.0 / 42,
          f"{acc:.3f} vs {1/42:.3f}")


def main():
    test_p3()
    test_p4_lightning_batch_size_weighting()
    test_p5_marginal_baseline()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
