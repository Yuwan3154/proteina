"""masked_reference() must be EXACTLY this model's training-time dropped reference (_set_empty).

Directive B, user 2026-09-22: the new tri model's mask arm must use its own no-reference regime
(a variable-length fully masked reference), not the single MASK token the sampler falls back to.
The transform is built from the REAL CB8-synthtopo dataset config, so drop_ref_len_range comes from
the yaml the model trained with rather than a retyped number.

Run (CPU): python scratchpad/test_masked_reference.py
"""

import os
import sys
import zlib

import hydra
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from proteinfoundation.datasets.sse_topology import MASK_TOKEN  # noqa: E402
from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase  # noqa: E402

DATASET = "pdb_train_contact-CB8-synthtopo_S25_max384_purge-test_cutoff-190828"
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


with hydra.initialize("../configs/datasets_config/pdb", version_base=hydra.__version__):
    cfg = hydra.compose(config_name=DATASET)
OmegaConf.set_struct(cfg, False)
spec = [t for t in cfg.datamodule.transforms if t._target_.endswith("TopologyReferenceTransform")]
assert len(spec) == 1
tr = hydra.utils.instantiate(spec[0])
lo, hi = tr.drop_ref_len_range
print(f"[config] drop_ref_len_range={tr.drop_ref_len_range} max_topology_he_len={tr.max_topology_he_len}")
check("config range is the one the run trained with ([5, 39])", (lo, hi) == (5, 39), f"{(lo, hi)}")

stems = [f"{i:04d}_A" for i in range(300)]
lengths = [50 + (i * 7) % 330 for i in range(300)]
saved = tr._generator
refs = [tr.masked_reference(s, L, seed=0) for s, L in zip(stems, lengths)]
check("the transform's own generator is restored", tr._generator is saved)

check("keys == ModelTrainerBase.TOPOLOGY_KEYS", all(set(r) == set(ModelTrainerBase.TOPOLOGY_KEYS) for r in refs),
      f"{sorted(refs[0])}")
n_el = [int(r["topology_he_tokens"].numel()) for r in refs]
check("element count within drop_ref_len_range", all(lo <= n <= min(hi, tr.max_topology_he_len) for n in n_el),
      f"min {min(n_el)} max {max(n_el)}")
check("the range is actually explored (not a constant)", len(set(n_el)) > 10, f"{len(set(n_el))} distinct")
check("every token is MASK", all(bool((r["topology_he_tokens"] == MASK_TOKEN).all()) for r in refs))
check("pair features all zero", all(float(r["topology_he_feat"].abs().max()) == 0.0 for r in refs))
mid_ok = all(torch.allclose(r["topology_he_pos"],
                            (torch.arange(n, dtype=torch.float32) + 0.5) * (L / n))
             for r, n, L in zip(refs, n_el, lengths))
check("element midpoints spread evenly over the query", mid_ok)

again = [tr.masked_reference(s, L, seed=0) for s, L in zip(stems[:20], lengths[:20])]
check("deterministic for a given (seed, stem)",
      all(torch.equal(a["topology_he_pos"], b["topology_he_pos"]) for a, b in zip(again, refs[:20])))

# identity with _set_empty itself under the same generator state
same = True
for s, L, r in zip(stems[:50], lengths[:50], refs[:50]):
    g = Data()
    tr._generator = torch.Generator().manual_seed((0 + zlib.crc32(s.encode())) % (2**63))
    tr._set_empty(g, L=L)
    same = same and all(torch.equal(getattr(g, k), r[k]) for k in ModelTrainerBase.TOPOLOGY_KEYS)
tr._generator = saved
check("bit-identical to _set_empty (the training path)", same)

# through the consumer's stacking: mixed lengths pad to one batch with invalid (<=0) padded tokens
batch = {k: ModelTrainerBase._stack_topology([r[k] for r in refs[:8]]) for k in ModelTrainerBase.TOPOLOGY_KEYS}
valid = (batch["topology_he_tokens"] > 0).sum(1).tolist()
check("stacked batch keeps each chain's own element count as its valid tokens", valid == n_el[:8],
      f"{valid} vs {n_el[:8]}")

print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
sys.exit(0 if ok else 1)
