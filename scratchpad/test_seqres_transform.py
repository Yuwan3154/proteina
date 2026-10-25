"""Gate for the T8b SEQRES plumbing (user 2026-10-20/21). CPU, toy files in a temp dir.

1. SeqresIndexTransform on a toy side table in the precompute_seqres_index.py merge format: a gapped chain gets its
   label_seq_id rebased so the first modelled residue = 0 (R2); a length mismatch, a non-increasing chain, a chain
   without positions (merged) and a missing stem each raise SeqresIndexError; the load-time coverage check passes on a
   covered id list and raises on one with an uncovered stem; a graph whose sequence differs from the table's seq_sha1
   (e.g. two residues swapped) raises, and a table without seq_sha1 raises at load.
2. PDBDataset: a SeqresIndexError is fatal (no silent swap to another chain); a different transform error still swaps
   (the known-good control: the swap path itself is unchanged).
3. Through the T8b dataset yaml (hydra): SeqresIndexTransform precedes TopologyReferenceTransform; placeholder_span is
   seqres; the dropped-reference midpoints are (i + 0.5) * span / n_el with span = last - first + 1 of the SEQRES index
   (not the modelled length L); masked_reference == _set_empty for the same seed and positions and raises without
   them; the T8 yaml keeps the length spread.
4. Template frame C (own index, rebased): the reference midpoints are the template's own residue index counted from
   its first residue = 0, independent of the query's SEQRES positions.
Run: python scratchpad/test_seqres_transform.py
"""

import hashlib
import os
import pathlib
import sys
import tempfile
import zlib

import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from proteinfoundation.datasets.pdb_data import PDBDataset  # noqa: E402
from proteinfoundation.openfold_stub.np.residue_constants import restypes  # noqa: E402
from proteinfoundation.datasets.sse_topology import STRUCTURAL_PAIR_FEATURES  # noqa: E402
from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform  # noqa: E402
from proteinfoundation.datasets.transforms import SeqresIndexError, SeqresIndexTransform  # noqa: E402

TMP = pathlib.Path(tempfile.mkdtemp(prefix="t8b_seqres_test_"))
# stem -> (status, label_seq_id positions or None, modelled length L)
ROWS = {
    "1abc_A": ("ok", [3, 4, 5, 9, 10, 20], 6),          # gapped: 3 unmodelled after 5, 9 after 10
    "1abc_B": ("ok", [1, 2, 3, 4, 5], 5),
    "2xyz_C": ("nonmonotone", [1, 3, 2, 4], 4),
    "3mrg_D": ("merged", None, 4),
}


def write_table(path):
    """The merge output of utils/precompute_seqres_index.py (cmd_merge): rows sorted by stem."""
    stems = sorted(ROWS)
    offset = np.zeros(len(stems) + 1, dtype=np.int64)
    flat = []
    for i, s in enumerate(stems):
        pos = ROWS[s][1]
        offset[i + 1] = offset[i] + (len(pos) if pos else 0)
        if pos:
            flat.append(np.asarray(pos, dtype=np.int32))
    table = {"def": "toy", "stems": stems, "status": [ROWS[s][0] for s in stems],
             "offset": torch.from_numpy(offset), "label_seq_flat": torch.from_numpy(np.concatenate(flat)),
             "length": torch.tensor([ROWS[s][2] for s in stems], dtype=torch.int32), "audit": {},
             "seq_sha1": [hashlib.sha1(toy_seq(ROWS[s][2]).encode()).hexdigest() for s in stems]}
    torch.save(table, path)
    return table


def toy_seq(L):
    """The toy chains' one-letter sequence: restypes in order, cycled (written independently of the transform)."""
    return "".join(restypes[i % 20] for i in range(L))


def graph(stem, L, residue_type=None):
    rt = torch.arange(L) % 20 if residue_type is None else residue_type
    return Data(protein_id=stem, coords=torch.zeros(L, 37, 3), coord_mask=torch.ones(L, 37, dtype=torch.bool),
                residue_type=rt)


def write_ids(name, ids):
    p = TMP / name
    p.write_text("".join(f"{i}\n" for i in ids))
    return str(p)


TABLE = str(TMP / "seqres_index_toy.pt")
TABLE_DICT = write_table(TABLE)


def test_transform():
    tr = SeqresIndexTransform(TABLE)
    g = tr(graph("1abc_A", 6))
    assert g.residue_seqres_idx.dtype == torch.long
    assert g.residue_seqres_idx.tolist() == [0, 1, 2, 6, 7, 17], g.residue_seqres_idx
    assert tr(graph("1abc_B", 5)).residue_seqres_idx.tolist() == [0, 1, 2, 3, 4]
    print("PASS gapped chain: label_seq [3,4,5,9,10,20] -> [0,1,2,6,7,17] (first modelled = 0); contiguous -> arange")
    with pytest.raises(SeqresIndexError, match="6 SEQRES positions"):
        tr(graph("1abc_A", 5))
    with pytest.raises(SeqresIndexError, match="not strictly increasing"):
        tr(graph("2xyz_C", 4))
    with pytest.raises(SeqresIndexError, match="0 SEQRES positions"):
        tr(graph("3mrg_D", 4))
    with pytest.raises(SeqresIndexError, match="not in the SEQRES side table"):
        tr(graph("9zzz_Z", 4))
    print("PASS raises SeqresIndexError: length mismatch, nonmonotone, merged (no positions), missing stem")
    SeqresIndexTransform(TABLE, required_ids_files=[write_ids("ok.txt", ["1abc_A", "1abc_B"])])
    for bad in ("3mrg_D", "2xyz_C", "9zzz_Z"):
        with pytest.raises(SeqresIndexError):
            SeqresIndexTransform(TABLE, required_ids_files=[write_ids("bad.txt", ["1abc_A", bad])])
    print("PASS load-time coverage: 2 of 2 covered ids pass; a list holding a merged / nonmonotone / missing stem raises")
    swapped = torch.arange(6) % 20
    swapped[[3, 4]] = swapped[[4, 3]]  # same residues, two swapped: positions and length still pass
    with pytest.raises(SeqresIndexError, match="residue sequence differs"):
        tr(graph("1abc_A", 6, residue_type=swapped))
    with pytest.raises(SeqresIndexError, match="residue sequence differs"):
        tr(graph("1abc_A", 6, residue_type=torch.full((6,), 7)))
    old = str(TMP / "seqres_index_no_sha1.pt")
    torch.save({k: v for k, v in TABLE_DICT.items() if k != "seq_sha1"}, old)
    with pytest.raises(SeqresIndexError, match="no seq_sha1"):
        SeqresIndexTransform(old)
    print("PASS seq_sha1: matching sequence passes; two swapped residues / a different sequence raise; a table "
          "without seq_sha1 raises at load")


class Boom:
    def __call__(self, g):
        if g.protein_id == "1abc_B":
            raise ValueError("ordinary transform failure")
        return g


def test_pdbdataset_no_swap():
    d = TMP / "data"
    (d / "processed").mkdir(parents=True)
    for stem, L in (("1abc_A", 6), ("1abc_B", 5), ("9zzz_Z", 4)):
        torch.save(graph(stem, L), d / "processed" / f"{stem}.pt")
    stems = ["1abc_A", "1abc_B", "9zzz_Z"]
    kw = dict(pdb_codes=[s.split("_")[0] for s in stems], chains=[s.split("_")[1] for s in stems], data_dir=str(d),
              file_names=stems, num_workers=0)
    ds = PDBDataset(transform=SeqresIndexTransform(TABLE), **kw)
    assert ds[0].protein_id == "1abc_A" and ds[0].residue_seqres_idx.tolist() == [0, 1, 2, 6, 7, 17]
    with pytest.raises(SeqresIndexError, match="9zzz_Z"):
        ds[2]
    ctrl = PDBDataset(transform=Boom(), **kw)
    assert ctrl[1].protein_id == "9zzz_Z", ctrl[1].protein_id
    print("PASS PDBDataset: a missing stem raises SeqresIndexError (no swap); control: an ordinary transform error "
          "still swaps 1abc_B -> 9zzz_Z")


def compose(name):
    with hydra.initialize("../configs/datasets_config/pdb", version_base=hydra.__version__):
        cfg = hydra.compose(config_name=name)
    OmegaConf.set_struct(cfg, False)
    return cfg.datamodule.transforms


def toy_topology_index(path):
    torch.save({"ids": [], "element_types": (0, 1, 2), "elem_feature_mean": 0.0, "elem_feature_std": 1.0}, path)


def test_placeholder_spread_and_frame_c():
    idx_dir = TMP / "t8b_index"
    idx_dir.mkdir()
    toy_topology_index(idx_dir / "topology_index_confind_synth_t8b.pt")
    toy_topology_index(idx_dir / "topology_index_confind_synth_t8.pt")
    os.environ.update(DATA_PATH=str(TMP), SYNTH_INDEX_DIR=str(idx_dir), SEQRES_INDEX_PATH=TABLE)
    specs = compose("pdb_train_contact-confind-synthtopo_S25_max512_t8b")
    names = [t._target_.rsplit(".", 1)[1] for t in specs]
    i_s, i_t = names.index("SeqresIndexTransform"), names.index("TopologyReferenceTransform")
    assert i_s == i_t - 1, names
    assert specs[i_s].index_path == TABLE and len(specs[i_s].required_ids_files) == 2
    sq = hydra.utils.instantiate(specs[i_s], required_ids_files=[write_ids("ok2.txt", ["1abc_A"])])
    tr = hydra.utils.instantiate(specs[i_t])
    assert isinstance(tr, TopologyReferenceTransform) and tr.placeholder_span == "seqres"
    assert tuple(tr.drop_ref_len_range) == (14, 80)
    print(f"PASS T8b yaml: {names[i_s]} directly before {names[i_t]}; placeholder_span seqres; drop_ref_len_range (14, 80)")

    g = sq(graph("1abc_A", 6))
    span = 17 + 1
    for seed in range(20):
        tr._generator = torch.Generator().manual_seed(seed)
        tr._set_empty(g, L=6)
        n = g.topology_he_pos_raw.numel()
        want = (torch.arange(n, dtype=torch.float32) + 0.5) * (span / n)
        assert 14 <= n <= 80 and torch.equal(g.topology_he_pos_raw, want) and torch.equal(g.topology_he_pos, want)
        assert float(g.topology_he_pos_raw[-1]) > 6, "spread over the SEQRES span 18, not the modelled length 6"
    print(f"PASS _set_empty midpoints = (i + 0.5) * 18 / n_el (SEQRES span 0..17), 20 of 20 draws, n_el in [14, 80]")

    stem, seed = "1abc_A", 3
    ref = tr.masked_reference(stem, 6, seed=seed, seqres_idx=g.residue_seqres_idx)
    h = Data(residue_seqres_idx=g.residue_seqres_idx)
    tr._generator = torch.Generator().manual_seed((seed + zlib.crc32(stem.encode())) % (2**63))
    tr._set_empty(h, L=6)
    assert all(torch.equal(getattr(h, k), v) for k, v in ref.items()), sorted(ref)
    with pytest.raises(ValueError, match="seqres_idx"):
        tr.masked_reference(stem, 6, seed=seed)
    with pytest.raises(SeqresIndexError, match="residue_seqres_idx"):
        tr._set_empty(Data(), L=6)
    print("PASS masked_reference == _set_empty (same seed + positions); raises without positions; _set_empty raises "
          "without residue_seqres_idx")

    t8 = [t for t in compose("pdb_train_contact-confind-synthtopo_S25_max512_t8") if
          t._target_.endswith("TopologyReferenceTransform")]
    tr8 = hydra.utils.instantiate(t8[0])
    assert tr8.placeholder_span == "length"
    tr8._generator = torch.Generator().manual_seed(0)
    g8 = Data()
    tr8._set_empty(g8, L=6)
    n = g8.topology_he_pos_raw.numel()
    assert torch.equal(g8.topology_he_pos_raw, (torch.arange(n, dtype=torch.float32) + 0.5) * (6 / n))
    print("PASS T8 yaml: placeholder_span length (default), midpoints (i + 0.5) * L / n_el as before")

    # frame C: runs (type, length) loop 3 | helix 5 | loop 2 | strand 4 -> spans from residue 0
    runs = [(0, 3), (1, 5), (0, 2), (2, 4)]
    feats = tr.assemble_reference(runs, torch.zeros(4, 4), torch.zeros(4, 4, len(STRUCTURAL_PAIR_FEATURES)), length=6, augment=False)
    assert feats["topology_he_pos_raw"].tolist() == [1.5, 5.5, 9.0, 12.0], feats["topology_he_pos_raw"]
    print("PASS frame C: template midpoints [1.5, 5.5, 9, 12] = own residue index from its first residue = 0, "
          "independent of the query's SEQRES positions")


if __name__ == "__main__":
    test_transform()
    test_pdbdataset_no_swap()
    test_placeholder_spread_and_frame_c()
    print("ALL PASS")
