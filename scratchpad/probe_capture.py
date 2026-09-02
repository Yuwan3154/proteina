"""Capture tri's query-to-reference pair block, with the ground truth and the left-alignment floor.

The probe asks: has the trunk ALREADY learned to realign the query onto the reference, despite
being given only a clipped left-aligned relative position?

What is captured, per sample:
  feat  [L, T, 320]  -- z[:, :L, L:, :] taken AFTER the 12 TriBlocks and BEFORE `out`. That block
                        is literally the query-by-element pair representation, so the probe head
                        is a single Linear(320 -> 1) on frozen features.
  off   [L, T]       -- the CLIPPED relative offset the model itself uses,
                        clamp(i - he_pos_raw[e], +-max_rel_pos) + max_rel_pos. This is the FLOOR's
                        only input.
  gt    [L, T]       -- USalign ground truth from align_gt.build_alignment.

⛔ The floor is mandatory and is not "chance". Left-alignment is a decent prior, so a probe that
sees only `off` already scores well above chance; the trunk has to beat THAT, not zero. Giving the
floor a per-offset lookup (Embedding(2*max_rel+2, 1)) matches the capacity the model itself has for
representing left-alignment, so the comparison is fair rather than rigged.

Samples with no reference (drop_prob) and SELF-references are excluded: the former have nothing to
predict, the latter are trivially diagonal and would inflate every number.
"""

import argparse
import json
import os
import sys
import tempfile

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from align_gt import build_alignment, load_graph  # noqa: E402
from proteinfoundation.proteinflow.proteina import Proteina  # noqa: E402


def find_transform(obj, depth=0):
    if obj is None or depth > 3:
        return None
    if type(obj).__name__ == "TopologyReferenceTransform":
        return obj
    inner = getattr(obj, "transforms", None)
    if isinstance(inner, (list, tuple)):
        for t in inner:
            f = find_transform(t, depth + 1)
            if f is not None:
                return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", default="training_contact_tri_full384_v1")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--max_batches", type=int, default=4000)
    args = ap.parse_args()

    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=args.config_name)
    OmegaConf.set_struct(cfg_exp, False)
    cfg_exp.hardware.ngpus_per_node_ = 1
    cfg_exp.hardware.nnodes_ = 1
    cfg_exp.log.log_wandb = False
    cfg_exp.log.checkpoint = False
    ds_dir = f"../configs/datasets_config/{cfg_exp.dataset_config_subdir}"
    with hydra.initialize(ds_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp.dataset)

    model = Proteina(cfg_exp, store_dir="/tmp/probe_store")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"partial load {len(missing)}/{len(unexpected)}"
    print(f"[load] epoch={ck.get('epoch')} step={ck.get('global_step')} clean", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()

    nn = model.nn
    max_rel = int(nn.max_rel_pos)
    grab = {}
    stash = []

    def z_pre_hook(_mod, inp):
        # out_norm's input is the final z, [B, N, N, dim], after all 12 TriBlocks.
        grab["z"] = inp[0].detach()

    def nn_pre_hook(_mod, inp):
        # ⛔ `contact_map_t` is built by the TRAINER's noising step, not by the dataloader, so the
        # nn cannot be called on a raw batch. Hooking the real validate pass is what makes the
        # captured features the ones the model actually computes.
        grab["batch"] = inp[0]

    def nn_post_hook(_mod, _inp, _out):
        b = grab.get("batch")
        z = grab.get("z")
        if b is None or z is None:
            return
        if "keys_printed" not in grab:
            grab["keys_printed"] = True
            print(f"[nn batch keys] {sorted(k for k in b)}", flush=True)
        refs = b.get("topology_ref_id")
        pids = b.get("protein_id")
        if refs is None or pids is None:
            grab["missing_ids"] = True
            return
        mask = b["mask"]
        he_pos_raw = b["topology_he_pos_raw"]
        he_tokens = b["topology_he_tokens"]
        Lpad = mask.shape[1]
        for k in range(len(pids)):
            r, p = str(refs[k]), str(pids[k])
            if not r or r == p:
                continue
            Lv = int(mask[k].sum())
            Tv = int((he_tokens[k] > 0).sum())
            if Lv == 0 or Tv == 0:
                continue
            # Slice inside the hook: a whole z is 128 MB, the QT block is ~15 MB.
            feat = z[k, :Lv, Lpad:Lpad + Tv, :].to(torch.float16).cpu()
            i_idx = torch.arange(Lv, dtype=torch.float32)
            off = (i_idx[:, None] - he_pos_raw[k, :Tv].float().cpu()[None, :]).round().long()
            off = off.clamp(-max_rel, max_rel) + max_rel
            stash.append({"feat": feat, "off": off, "query": p, "ref": r, "L": Lv, "T": Tv})

    h1 = nn.out_norm.register_forward_pre_hook(z_pre_hook)
    h2 = nn.register_forward_pre_hook(nn_pre_hook)
    h3 = nn.register_forward_hook(nn_post_hook)

    dm = hydra.utils.instantiate(cfg_data.datamodule)
    # ⛔ Do NOT call dm.setup() here. trainer.validate() calls it again, and pdb_data.py:1584
    # does `if not self.df_data:` on an already-populated DataFrame -> "truth value is ambiguous".
    # Read the index FILE, not the live transform object: the object is reachable from neither
    # dm.transform nor dl.dataset.transform, and the file has everything needed.
    data_dir = os.environ["DATA_PATH"] + "/pdb_train"
    processed = os.path.join(data_dir, "processed")
    mpath = os.path.join(data_dir, "shard_manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else None
    idx = torch.load(os.path.join(data_dir, "topology_index.pt"), map_location="cpu",
                     weights_only=False, mmap=True)
    id_to_row = {str(v): i for i, v in enumerate(idx["ids"])}

    def runs_for(row):
        a, b = int(idx["runs_offset"][row]), int(idx["runs_offset"][row + 1])
        return [(int(t), int(n)) for t, n in idx["runs_flat"][a:b].tolist()]

    print(f"[index] {len(id_to_row)} chains", flush=True)

    os.makedirs(args.out, exist_ok=True)

    import lightning as L
    trainer = L.Trainer(
        accelerator="gpu" if dev == "cuda" else "cpu", devices=1, num_nodes=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        limit_val_batches=args.max_batches,
    )
    trainer.validate(model, datamodule=dm, ckpt_path=None, verbose=False)
    for h in (h1, h2, h3):
        h.remove()

    if grab.get("missing_ids"):
        print("FAIL: the batch reaching model.nn carries no protein_id/topology_ref_id", flush=True)
        return 5
    print(f"[stash] {len(stash)} cross-chain samples captured", flush=True)

    kept = 0
    skipped = {"no_runs": 0, "gt_error": 0, "no_alignment": 0}
    for item in stash:
        if kept >= args.n_samples:
            break
        q, r, Tv = item["query"], item["ref"], item["T"]
        rrow = id_to_row.get(r)
        if rrow is None:
            skipped["no_runs"] += 1
            continue
        runs = runs_for(rrow)
        if not runs:
            skipped["no_runs"] += 1
            continue
        try:
            gq = load_graph(processed, q, manifest)
            gr = load_graph(processed, r, manifest)
            with tempfile.TemporaryDirectory() as td:
                A, Q = build_alignment(gq, gr, runs, Tv, td)
        except Exception as e:  # noqa: BLE001
            skipped["gt_error"] += 1
            if skipped["gt_error"] <= 5:
                print(f"  [gt] {q}->{r}: {type(e).__name__}: {e}", flush=True)
            continue
        gt = A[: item["L"], : Tv].clone()
        if int(gt.sum()) == 0:
            skipped["no_alignment"] += 1
            continue
        torch.save(
            {"feat": item["feat"], "off": item["off"], "gt": gt, "query": q, "ref": r,
             "L": item["L"], "T": Tv, "Q": int((gt.sum(dim=1) > 0).sum())},
            os.path.join(args.out, f"s{kept:05d}.pt"),
        )
        kept += 1
        if kept % 20 == 0:
            print(f"  kept {kept}/{args.n_samples}", flush=True)

    h.remove()
    print(f"\nkept={kept}  skipped={skipped}", flush=True)
    if kept == 0:
        print("FAIL: captured nothing", flush=True)
        return 6
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
