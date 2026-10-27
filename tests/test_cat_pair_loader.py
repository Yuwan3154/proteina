"""CAT-regime loader tests. The logic tests run anywhere; the integration test needs the SuperCloud toy inputs
(T2CAT_TOY_DIR: units.tsv + a natives tree + fill inputs; T2CAT_USALIGN: the USalign binary; pydssp) and skips otherwise."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from proteinfoundation.datasets import cat_unit_store as cus
from proteinfoundation.utils.cluster_utils import CATUniformSampler


def test_seg_roundtrip():
    pos = np.array([3, 4, 5, 9, 10, 20])
    seg = cus.positions_to_seg(pos)
    assert seg == "3-6;9-11;20-21"
    assert cus.seg_positions(seg).tolist() == pos.tolist()
    assert cus.positions_to_seg(np.arange(7)) == "0-7"


def _toy_store(root, with_cbits):
    rng = np.random.default_rng(0)
    entries, ents = [], []
    for k, n in enumerate([5, 7, 4]):
        entries.append(dict(aatype=rng.integers(0, 20, n), resnum=np.arange(n) + 10, mask=rng.random((n, 37)) > 0.3, coords=rng.normal(size=(n, 37, 3))))
        ents.append(dict(entry_id=f"e{k}", kind="cath", src_unit=f"e{k}", n_res=n, shard=0, row=k))
    cus.write_shard(os.path.join(root, "shards", "s0000"), entries)
    pd.DataFrame(ents).to_csv(os.path.join(root, "entries.tsv"), sep="\t", index=False)
    pd.DataFrame(dict(member_id=["e0@1.1.1", "e1@1.1.1", "e2@1.1.1"], entry_id=["e0", "e1", "e2"], cat="1.1.1", split="train", role="query", seg=["0-5", "0-7", "0-4"])).to_csv(os.path.join(root, "members.tsv"), sep="\t", index=False)
    return entries


def test_store_roundtrip(tmp_path):
    entries = _toy_store(str(tmp_path), False)
    st = cus.UnitStore(str(tmp_path))
    for k, e in enumerate(entries):
        got = st.entry(f"e{k}")
        assert np.array_equal(got["aatype"], e["aatype"]) and np.array_equal(got["mask"], e["mask"])
        assert np.allclose(got["coords"], e["coords"].astype(np.float32))
    from proteinfoundation.utils.precompute_cat_pair_index import write_cbits
    maps = {f"e{k}": (np.random.default_rng(k).random((n, n)) > 0.5) for k, n in enumerate([5, 7, 4])}
    write_cbits(st, {k: np.packbits(v.reshape(-1)) for k, v in maps.items()})
    st = cus.UnitStore(str(tmp_path))
    for k, v in maps.items():
        assert np.array_equal(st.contact_map(k) > 0.5, v)


def test_sampler_one_per_cat_and_sharding():
    by_cat = [[0, 1, 2], [3], [4, 5], [6, 7, 8, 9]]
    s = CATUniformSampler(by_cat, seed=3, shard=False)
    e0 = list(s)
    assert len(e0) == 4 and sorted(next(c for c, m in enumerate(by_cat) if x in m) for x in e0) == [0, 1, 2, 3]
    assert list(CATUniformSampler(by_cat, seed=3, shard=False)) == e0
    s.set_epoch(1)
    e1 = list(s)
    assert len(e1) == 4
    assert sorted(next(c for c, m in enumerate(by_cat) if x in m) for x in e1) == [0, 1, 2, 3]
    seen = set()
    for ep in range(30):
        s.set_epoch(ep)
        seen.add(tuple(list(s)))
    assert len(seen) > 1, 'the draw never changes across epochs'
    fixed = CATUniformSampler(by_cat, seed=3, shard=False, fixed=True)
    first = list(fixed)
    for ep in range(1, 30):
        fixed.set_epoch(ep)
        assert list(fixed) == first, "a fixed (validation) sampler must ignore set_epoch"
    assert first == e0, "fixed=True must equal the epoch-0 draw of the rotating sampler"
    full = []
    for rank in range(3):
        s = CATUniformSampler(by_cat, seed=3, shard=True)
        s._world = lambda r=rank: (r, 3)
        out = list(s)
        assert len(out) == len(s) == 2
        full += out
    assert set(full) >= set(e0)  # padded by repeating the head: every drawn member is served


TOY = os.environ.get("T2CAT_TOY_DIR")
USALIGN = os.environ.get("T2CAT_USALIGN")


needs_toy = pytest.mark.skipif(not (TOY and USALIGN), reason="needs the SuperCloud toy inputs (T2CAT_TOY_DIR, T2CAT_USALIGN)")
TR_KW = dict(drop_prob=0.0, mutate_prob=0.0, type_mutate_prob=0.0, token_mask_prob=0.0, sse_types=[0, 1, 2], single_token_types=[0],
             element_types=[0, 1, 2], elem_features=True, max_topology_len=128, max_topology_he_len=96)


@pytest.fixture(scope="module")
def toy():
    """Store + index of the toy CATs, built with CB-8 contacts in place of Frame2ConFind (the GPU model is not needed to test the loader)."""
    from proteinfoundation.utils.precompute_cat_pair_index import entry_rows, merge_rows
    from proteinfoundation.utils.precompute_synthetic_topology_index import cb8_contacts

    meta = json.load(open(os.path.join(TOY, "meta.json")))
    store = cus.UnitStore(os.path.join(TOY, "store"))
    by_entry = {e: g for e, g in store.members.groupby("entry_id")}
    rows, cbits, skips = {}, {}, []
    for eid in store.entries.entry_id:
        e = store.entry(eid)
        f, m = torch.from_numpy(np.array(e["coords"])), torch.from_numpy(np.array(e["mask"]))
        cm = cb8_contacts(f, m, 3)
        got, sk = entry_rows(store, eid, by_entry[eid], cm, tuple(meta["element_types"]), meta["loop_axis"], int(meta["min_len"]))
        rows.update({r["member_id"]: r for r in got})
        cbits[eid] = np.packbits(cm.numpy().astype(bool).reshape(-1))
        skips += sk
    index_path = os.path.join(TOY, "index.pt")
    merge_rows(store, rows, cbits, skips, meta, index_path)
    return dict(store=cus.UnitStore(os.path.join(TOY, "store")), index_path=index_path, index=torch.load(index_path, weights_only=False))


def _transform(toy, **kw):
    from proteinfoundation.datasets.cat_pair_data import CATPairReferenceTransform
    return CATPairReferenceTransform(toy["index_path"], usalign=USALIGN, **{"seed": 1, **TR_KW, **kw})


def test_source_weights_and_sampler_mixture():
    from proteinfoundation.datasets.cat_pair_data import CATUnitDataset
    ds = CATUnitDataset.__new__(CATUnitDataset)
    ds.members = pd.DataFrame({"cat": ["a", "a", "a", "b", "b", "c", "d", "d"], "kind": ["cath", "ted", "ted", "ted", "ted", "cath", "cath", "cath"]})
    w = ds.member_weights_by_cat(0.3)
    assert [round(sum(x), 6) for x in w] == [1.0] * 4
    assert w[0] == pytest.approx([0.3, 0.35, 0.35]) and w[1] == pytest.approx([0.5, 0.5]) and w[2] == [1.0] and w[3] == pytest.approx([0.5, 0.5])
    by_cat = [[0, 1, 2], [3, 4], [5], [6, 7]]
    n_cath, n = 0, 4000
    for ep in range(n // 4):
        s = CATUniformSampler(by_cat, seed=5, shard=False, weights_by_cat=w)
        s.set_epoch(ep)
        got = list(s)
        n_cath += sum(g in (0, 5, 6, 7) for g in got)
    # CATs a (p 0.3), b (0), c (1), d (1): expected CATH draws per epoch 0.3 + 0 + 1 + 1
    assert n_cath / (n // 4) == pytest.approx(2.3, abs=0.1)
    from proteinfoundation.datasets.epoch_batch_plan import build_epoch_plan
    cath_plan = 0
    for ep in range(500):
        plan = build_epoch_plan(by_cat, [100] * 8, 2, [128], 3, ep, 1, cluster_weights=w)
        cath_plan += sum(i in (0, 5, 6, 7) for b in plan[0] for i in b)
    assert cath_plan / 500 == pytest.approx(2.3, abs=0.15)
    plan_uniform = build_epoch_plan(by_cat, [100] * 8, 2, [128], 3, 0, 1)
    assert plan_uniform == build_epoch_plan(by_cat, [100] * 8, 2, [128], 3, 0, 1, cluster_weights=None)


@needs_toy
def test_pairs_are_nonself_same_cat_and_aligned(toy):
    from proteinfoundation.datasets.cat_pair_data import CATUnitDataset
    ds = CATUnitDataset(os.path.join(TOY, "store"), "train", transform=_transform(toy))
    mem_tab = toy["store"].members.set_index("member_id")
    fracs = []
    for i in range(len(ds)):
        g, mem = ds[i], ds.members.iloc[i]
        assert int(g.coords.shape[0]) == ds.lengths[i] == int(g.ref_align_target.shape[0])
        assert g.topology_missing_ref.item() == 0
        t = mem_tab.loc[g.topology_ref_id]
        assert t["entry_id"] != mem["entry_id"] and t["cat"] == mem["cat"] and t["split"] == "train"
        n_he = int(g.topology_he_tokens.numel())
        assert int(g.ref_align_target.max()) < n_he and int(g.ref_align_target.min()) >= -1
        assert g.contact_map_confind.shape == (g.coords.shape[0],) * 2
        fracs.append(float((g.ref_align_target >= 0).float().mean()))
    print("aligned fraction of the query per pair: min %.2f median %.2f max %.2f over %d pairs" % (min(fracs), sorted(fracs)[len(fracs) // 2], max(fracs), len(fracs)))
    assert min(fracs) > 0.0


@needs_toy
def test_domain_cut_and_numbering(toy):
    from proteinfoundation.datasets.cat_pair_data import CATUnitDataset
    ds = CATUnitDataset(os.path.join(TOY, "store"), "train")
    two_seg = ds.members[ds.members.seg.str.contains(";")]
    assert len(two_seg) > 0, "the toy has no discontinuous domain"
    i = int(two_seg.index[0])
    g, mem = ds[i], ds.members.iloc[i]
    pos = cus.seg_positions(mem["seg"])
    entry = toy["store"].entry(mem["entry_id"])
    assert g.coords.shape[0] == len(pos) < toy["store"].entries.set_index("entry_id").loc[mem["entry_id"], "n_res"]
    assert np.array_equal(g.residue_pdb_idx.numpy(), np.array(entry["resnum"])[pos]), "residue_pdb_idx must stay un-renumbered"
    assert (np.diff(g.residue_pdb_idx.numpy()) > 1).any(), "the segment gap must stay visible"
    assert g.contact_map_confind.shape == (len(pos), len(pos))
    row = {m: k for k, m in enumerate(toy["index"]["ids"])}[mem["member_id"]]
    a, b = int(toy["index"]["ca_offset"][row]), int(toy["index"]["ca_offset"][row + 1])
    assert b - a == len(pos), "the stored template CA / element map must cover exactly the domain"


@needs_toy
def test_self_alignment_reproduces_element_map(toy):
    from proteinfoundation.datasets.cat_pair_data import CATUnitDataset
    tr = _transform(toy)
    tr._ensure_loaded()
    ds = CATUnitDataset(os.path.join(TOY, "store"), "train")
    for i in range(len(ds)):
        g, mem = ds[i], ds.members.iloc[i]
        row = tr._id_to_row[mem["member_id"]]
        a, b = int(tr._index["ca_offset"][row]), int(tr._index["ca_offset"][row + 1])
        elem = tr._index["elem_flat"][a:b].long()
        n_he = int(elem.max()) + 1
        got = tr._cat_align(g, row, n_he)
        assert torch.equal(got, elem), "USalign of a domain onto itself must give the identity element map"


@needs_toy
def test_determinism_exclusions_and_missing(toy):
    from proteinfoundation.datasets.cat_pair_data import CATUnitDataset
    from proteinfoundation.datasets.topology_reference import MASK_REF_ID
    ds_plain = CATUnitDataset(os.path.join(TOY, "store"), "train")

    def refs(**kw):
        torch.manual_seed(0)
        tr = _transform(toy, **kw)
        return [tr(ds_plain[i]).topology_ref_id for i in range(len(ds_plain))]

    assert refs() == refs(), "same seed must give the same template draws"
    src = toy["store"].entries.set_index("entry_id").src_unit
    mem_tab = toy["store"].members.set_index("member_id")
    for i, ref in enumerate(refs(exclude_own_synthetic=True)):
        assert src[mem_tab.loc[ref, "entry_id"]] != src[ds_plain.members.iloc[i]["entry_id"]]
    g = _transform(toy, drop_prob=1.0)(ds_plain[0])
    assert g.topology_ref_id == MASK_REF_ID and g.topology_missing_ref.item() == 0
    tr = _transform(toy)
    tr._ensure_loaded()
    tr._pick_cat_template = lambda row: -1
    g = tr(ds_plain[0])
    assert g.topology_ref_id == MASK_REF_ID and g.topology_missing_ref.item() == 1


@needs_toy
def test_fixed_validation_pairs(toy):
    from proteinfoundation.datasets.cat_pair_data import CATUnitDataset
    store = os.path.join(TOY, "store")

    def run(seed_torch, transform_seed, fixed):
        torch.manual_seed(seed_torch)
        ds = CATUnitDataset(store, "train", transform=_transform(toy, seed=transform_seed), fixed_pair_seed=fixed)
        out = [ds[i] for i in range(len(ds))]
        return [(g.topology_ref_id, g.ref_align_target.tolist(), g.topology_he_tokens.tolist(), hasattr(g, "pair_seed")) for g in out]

    a, b = run(0, 1, 0), run(5, 9, 0)
    assert a == b, "fixed validation: the same pair and reference for any torch / transform seed"
    assert not any(x[3] for x in a), "pair_seed must be consumed by the transform"
    assert run(0, 1, 0) != run(0, 1, 7), "a different fixed seed must change the draws"


@needs_toy
def test_datamodule_batches(toy):
    from proteinfoundation.datasets.cat_pair_data import CATPairLightningDataModule
    from proteinfoundation.datasets.transforms import ChainBreakPerResidueTransform, ContactMapTransform, DSSPTargetTransform, GlobalRotationTransform
    chain = [GlobalRotationTransform(), ChainBreakPerResidueTransform(), DSSPTargetTransform(),
             ContactMapTransform(contact_method="confind", confind_contact_threshold=0.01), _transform(toy)]
    dm = CATPairLightningDataModule(os.path.join(TOY, "store"), cath_draw_prob=0.3, transforms=chain, batch_size=2, num_workers=2, epoch_batch_plan=True,
                                    epoch_batch_plan_len_edges=[64, 128, 192, 256, 320, 384, 448, 512], cluster_sampler_seed=0)
    dm.setup("fit")
    progress = SimpleNamespace(current=SimpleNamespace(processed=0))
    dm.trainer = SimpleNamespace(current_epoch=0, fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(batch_progress=progress)), reload_dataloaders_every_n_epochs=0)
    dl = dm.train_dataloader()
    batches = list(dl)
    assert len(batches) == len(dl.batch_sampler) and len(batches) > 0
    b = batches[0]
    for key in ("contact_map", "ref_align_target", "topology_he_tokens", "topology_he_feat", "residue_type", "coord_mask", "dssp_target"):
        assert key in b, key
    assert b["ref_align_target"].shape == b["residue_type"].shape == b["dssp_target"].shape
    assert b["contact_map"].shape[1:] == b["residue_type"].shape[1:] * 2
    served = sum(int(x["residue_type"].shape[0]) for x in batches)
    assert served >= 2, "an epoch serves one member per CAT (2 CATs in the toy)"


@needs_toy
def test_topology_row_defaults_unchanged(toy):
    """The T8 builder calls topology_row with five arguments; the new keyword defaults must reproduce its module-global behaviour."""
    from proteinfoundation.utils import precompute_synthetic_topology_index as pre
    e = toy["store"].entry(toy["store"].entries.entry_id.iloc[0])
    f, m = torch.from_numpy(np.array(e["coords"])), torch.from_numpy(np.array(e["mask"]))
    cm = pre.cb8_contacts(f, m, 3)
    dssp = torch.randint(0, 3, (f.shape[0],))
    a = pre.topology_row(cm, f, m, dssp, 1)
    b = pre.topology_row(cm, f, m, dssp, 1, element_types=pre._ELEMENT_TYPES, loop_axis=pre._LOOP_AXIS)
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3] and np.array_equal(a[5], b[5])
