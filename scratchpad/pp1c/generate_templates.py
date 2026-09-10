"""Production: generate N graded synthetic templates per training chain, one tiered pass each.

Design decisions, all measured rather than assumed (see ESMFOLD2_RECYCLE_SCALING.md):
  * TIERED scheduling -- one denoising pass per chain, each template injected at its own entry
    step. 2.16x faster than grouped at L=150, 1.24x at L=250, neutral at L=599, and it is the only
    way to get N DISTINCT noise levels without collapsing to batch 1.
  * MODEL BY SPAN: cc89 (all-atom, sequence-mask) has by far the best query-sequence compatibility
    (2.45 NLL at rewind 250 vs ~3.04 for cc91/cc94) but uses ROTARY position embeddings and
    degrades past a residue-index SPAN of ~500-600. Chains whose span exceeds the cutoff go to
    cc91 (relative encoding, no length cliff). ~8% of the training set is affected.
  * ONE npz PER CHAIN holding all N templates, not N files: 88k files instead of 5.6M, which also
    keeps every directory under the 1024-file limit once sharded.
  * Only atoms present in the (batch-shared) atom mask are stored -- the sequence is fixed under
    partial diffusion, so aatype/atom_mask/residue_index are identical across all N templates and
    are stored once.

Resumable by construction: a chain whose output npz already exists is skipped, so a requeued array
task picks up where it left off. No per-chain try/except -- inputs were validated during extraction
so a runtime failure is a real bug that should stop the task, not be swallowed.

⭐⭐ ROUND-2 ADDITIONS (2026-09-09). Defaults reproduce round 1 BYTE-FOR-BYTE; round 2 is opt-in via
`--n-rungs`, and the file is otherwise the same driver that produced the live pool.
  * `--ladder-starts` -- a per-chain ladder START from build_ladder_starts.py, replacing the global
    `--rmin`. User decision: each chain's own TM=0.9 crossing + 10, up to 375, which scored 55.0
    in-band templates per 64 rungs at skew 0.99 on the full range (n=81,011) against round 1's
    global 90-375 at 34.1 / 1.37.
  * `--n-rungs` / `--seeds-per-rung` -- split the per-chain budget between DISTINCT rewind values
    and repeated draws at the SAME rewind. Round 1 already ran both extremes: the tiered path used
    64 rungs x 1 seed, the grouped path 8 rungs x 8 seeds, which is what made the tradeoff
    measurable without generating anything (prune_work/t2_seed_vs_rung.py).
  ⛔ Cost is invariant to the split: runtime scales with rewind_steps, so a fixed budget over a
    fixed window costs the same however it divides into rungs vs seeds.
"""

import argparse
import csv
import time
import zlib
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from protpardelle.core.models import load_model
from protpardelle.data.pdb_io import load_feats_from_pdb
from protpardelle.env import (
    MINIMPNN_WEIGHTS,
    PROTPARDELLE_MODEL_CONFIGS,
    PROTPARDELLE_MODEL_WEIGHTS,
    PROTPARDELLE_RUNNING_CONFIGS,
)
from protpardelle.utils import apply_dotdict_recursively, seed_everything

CFG = "sampling_partial_diffusion_allatom"
MODEL_EPOCH = {"cc89": "415", "cc91": "383", "cc94": "3100"}


def sampling_kwargs(pdb, rewinds):
    with initialize_config_dir(config_dir=str(PROTPARDELLE_RUNNING_CONFIGS), version_base="1.3.2"):
        c = compose(config_name=CFG)
    c = OmegaConf.to_container(hydra.utils.call(c), resolve=True)
    s = c["sampling"]
    s["step_scale"], s["s_churn"] = 1.0, 0            # ODE, the repo's own recommendation
    s["conditional_cfg"]["crop_conditional_guidance"]["start"] = 0.0
    s["partial_diffusion"]["pdb_file_path"] = pdb
    s["partial_diffusion"]["num_steps"] = rewinds     # per-sample list -> tiered
    s["motif_file_path"] = "test_dir/empty.pdb"
    s.update(apply_dotdict_recursively(s.pop("allatom_cfg")))
    s.pop("stage2_cfg")
    return s


def rewind_ladder(n, rmin, rmax):
    """Descending, which the tiered scheduler requires (entry steps must ascend)."""
    if n == 1:
        return [rmax]
    return [int(round(rmax - i * (rmax - rmin) / (n - 1))) for i in range(n)]


def rewind_list(n_rungs, seeds_per_rung, rmin, rmax):
    """The per-chain sample list: each rung repeated `seeds_per_rung` times, still descending.

    Repeats are what give a rung multiple SEEDS -- the noise for each batch member is drawn
    independently inside model.sample, so duplicate rewind values produce different structures, not
    copies. Round 1's grouped path measured that spread directly: two samples at an identical rewind
    reach TM ~0.76 to each other, nowhere near the 1.0 a duplicate would give.
    Descending order is preserved because the tiered scheduler enters members as a contiguous
    prefix; equal entry steps simply enter together.
    """
    return [r for r in rewind_ladder(n_rungs, rmin, rmax) for _ in range(seeds_per_rung)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest.csv from extract_train_natives.py")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--n-templates", type=int, default=64)
    ap.add_argument("--rmin", type=int, default=90)
    ap.add_argument("--rmax", type=int, default=375)
    ap.add_argument("--span-cutoff", type=int, default=484,
                    help="chains with residue-index span above this use --long-model")
    ap.add_argument("--short-model", default="cc89")
    ap.add_argument("--long-model", default="cc91")
    ap.add_argument("--tiered-max-length", type=int, default=300,
                    help="chains longer than this use grouped scheduling; measured on a 32 GB "
                         "V100, tiered peaks at 16.4 GB by L=300, pins the card at L>=499, and is "
                         "SLOWER than grouped there, while grouped stays flat at <1.2 GB")
    ap.add_argument("--groups", type=int, default=8,
                    help="distinct rewind values when falling back to grouped scheduling")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chains", type=int, default=0)
    ap.add_argument("--ladder-starts", default=None,
                    help="npz from build_ladder_starts.py: per-chain ladder START, replacing "
                         "--rmin. A chain absent from the table is SKIPPED and named in the shard "
                         "summary -- never silently given --rmin, which would quietly regenerate "
                         "round 1's ladder for it. Absence is legitimate: the table comes from the "
                         "TM index, which drops chains with an unscoreable native.")
    ap.add_argument("--n-rungs", type=int, default=0,
                    help="ROUND 2: number of DISTINCT rewind values per chain. 0 (default) = "
                         "round-1 behaviour, where the tiered path uses --n-templates distinct "
                         "rungs and the grouped path uses --groups.")
    ap.add_argument("--seeds-per-rung", type=int, default=1,
                    help="ROUND 2: samples drawn at each rung. Requires --n-rungs. The per-chain "
                         "budget is n_rungs * seeds_per_rung.")
    ap.add_argument("--max-length", type=int, default=512,
                    help="skip chains longer than this. 512 is Protpardelle's own training "
                         "fixed_size, and it drops the expensive tail (grouped costs 58.7 s/chain "
                         "at L=599 vs ~19 s at L<=300).")
    args = ap.parse_args()

    # ⛔ Fail here, not 40 h in: seeds-per-rung without n-rungs would silently generate round 1.
    assert not (args.seeds_per_rung > 1 and args.n_rungs == 0), \
        "--seeds-per-rung requires --n-rungs (otherwise the ladder is round 1's)"
    round2 = args.n_rungs > 0
    n_per_chain = args.n_rungs * args.seeds_per_rung if round2 else args.n_templates

    starts = None
    if args.ladder_starts:
        _s = np.load(args.ladder_starts, allow_pickle=False)
        starts = {str(c): int(v) for c, v in zip(_s["chains"], _s["start"])}
        print(f"ladder starts: {len(starts)} chains, top={int(_s['top'])}, "
              f"start p50={int(np.median(_s['start']))}", flush=True)
    print(f"MODE: {'round 2' if round2 else 'round 1'}  "
          f"{n_per_chain} templates/chain"
          + (f" = {args.n_rungs} rungs x {args.seeds_per_rung} seeds" if round2 else "")
          + f"  ladder {'per-chain start' if starts else args.rmin}-{args.rmax}", flush=True)

    manifest_dir = Path(args.manifest).parent
    rows = [r for r in csv.DictReader(open(args.manifest)) if r["status"] == "ok"]
    n_all = len(rows)
    rows = [r for r in rows if int(r["length"]) <= args.max_length]
    print(f"manifest: {n_all} ok chains -> {len(rows)} with length <= {args.max_length} "
          f"({100*len(rows)/max(n_all,1):.1f}%)", flush=True)
    rows.sort(key=lambda r: r["chain"])                    # deterministic order
    # length-sorted round-robin keeps shards balanced: cost grows steeply with L, so a naive
    # contiguous split would leave one shard with all the long chains.
    rows.sort(key=lambda r: int(r["length"]))
    mine = rows[args.shard :: args.num_shards]
    if args.max_chains:
        mine = mine[: args.max_chains]

    out_root = Path(args.out_root)
    by_model = {}
    for r in mine:
        m = args.long_model if int(r["resid_span"]) > args.span_cutoff else args.short_model
        by_model.setdefault(m, []).append(r)

    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} chains  "
          + "  ".join(f"{k}={len(v)}" for k, v in sorted(by_model.items())), flush=True)

    # ⛔ Round 2's ladder is PER CHAIN, so it cannot be hoisted out of the loop the way round 1's
    # single global ladder was. Round 1 keeps the hoisted value so its behaviour is bit-identical.
    rewinds_global = None if (round2 or starts) else \
        rewind_ladder(args.n_templates, args.rmin, args.rmax)
    done = skipped = 0
    missing = []            # chains absent from --ladder-starts, recorded not swallowed
    t_start = time.perf_counter()

    for model_name, chains in sorted(by_model.items()):
        model = load_model(
            str(PROTPARDELLE_MODEL_CONFIGS / f"{model_name}.yaml"),
            str(PROTPARDELLE_MODEL_WEIGHTS / f"{model_name}_epoch{MODEL_EPOCH[model_name]}.pth"),
        )
        model.load_minimpnn(MINIMPNN_WEIGHTS)
        print(f"loaded {model_name} for {len(chains)} chains", flush=True)

        for r in chains:
            chain = r["chain"]
            # ⛔ NOT builtin hash(): Python randomizes string hashing per process, so every worker
            # (and every requeue) picked a DIFFERENT directory -- resumability silently failed and
            # outputs duplicated. crc32 is stable across processes and runs.
            shard_dir = out_root / f"shard{zlib.crc32(chain.encode()) % 1000:04d}"
            out_npz = shard_dir / f"{chain}.npz"
            if out_npz.is_file():
                skipped += 1
                continue
            shard_dir.mkdir(parents=True, exist_ok=True)

            # The manifest is written on the box that owns the mmCIF mirror, so its `pdb` column
            # holds that host's absolute paths. Fall back to resolving against the manifest's own
            # directory so the same manifest works unchanged after the rsync to SuperCloud.
            pdb = r["pdb"]
            if not Path(pdb).is_file():
                cand = manifest_dir / Path(pdb).parent.name / Path(pdb).name
                if not cand.is_file():
                    raise FileNotFoundError(f"{pdb} (and {cand}) missing for chain {chain}")
                pdb = str(cand)
            feats, _ = load_feats_from_pdb(pdb, include_pos_feats=True)
            L = int(feats["aatype"].shape[0])

            rmin = args.rmin
            if starts is not None:
                # ⛔⛔ SKIP-AND-RECORD, NEVER ASSERT, on per-item data. A chain can legitimately be
                # absent: ladder_starts.npz is built from the TM index, and the index drops chains
                # whose native is unscoreable (4boh_M resolves 4 CA). An assert here turned that one
                # chain into the total loss of a 10,342-chain shard, idling a GPU for the rest of
                # the run -- the exact failure mode round 1 already hit in build_template_index.py.
                if chain not in starts:
                    missing.append(chain)
                    skipped += 1
                    continue
                rmin = starts[chain]
            if rewinds_global is not None:
                rewinds = rewinds_global
            elif round2:
                rewinds = rewind_list(args.n_rungs, args.seeds_per_rung, rmin, args.rmax)
            else:
                rewinds = rewind_ladder(args.n_templates, rmin, args.rmax)
            assert len(rewinds) == n_per_chain, (len(rewinds), n_per_chain)

            ridx = torch.tile(feats["residue_index"][None], (n_per_chain, 1)).cuda()
            cidx = torch.tile(feats["chain_index"][None], (n_per_chain, 1)).cuda()
            mask = torch.ones_like(ridx).cuda()

            seed_everything(args.seed)
            t0 = time.perf_counter()
            if L <= args.tiered_max_length:
                # one pass, N distinct noise levels, batch ramps 0->N
                with torch.no_grad():
                    aux = model.sample(
                        seq_mask=mask, residue_index=ridx, chain_index=cidx, hotspots=None,
                        sse_cond=None, adj_cond=None, motif_placements_full=None,
                        dummy_fill_mode=model.config.data.dummy_fill_mode,
                        **sampling_kwargs(pdb, rewinds),
                    )
                schedule = "tiered"
                out_coords = aux["xt_traj"][-1].numpy().astype(np.float32)
                out_rewinds = rewinds
                atom_mask = aux["atom_mask"][0].cpu().numpy().astype(bool)
            else:
                # grouped: constant batch N/G. Tiered's ramping batch defeats the caching
                # allocator and pins a 32 GB V100 from L~499 while being slower there.
                g = args.n_rungs if round2 else args.groups
                per = args.seeds_per_rung if round2 else max(1, args.n_templates // g)
                ladder = rewind_ladder(g, rmin, args.rmax)
                chunks, out_rewinds = [], []
                sub_r = torch.tile(feats["residue_index"][None], (per, 1)).cuda()
                sub_c = torch.tile(feats["chain_index"][None], (per, 1)).cuda()
                sub_m = torch.ones_like(sub_r).cuda()
                for R in ladder:
                    with torch.no_grad():
                        a = model.sample(
                            seq_mask=sub_m, residue_index=sub_r, chain_index=sub_c, hotspots=None,
                            sse_cond=None, adj_cond=None, motif_placements_full=None,
                            dummy_fill_mode=model.config.data.dummy_fill_mode,
                            **sampling_kwargs(pdb, R),
                        )
                    chunks.append(a["xt_traj"][-1].numpy().astype(np.float32))
                    out_rewinds.extend([R] * per)
                    atom_mask = a["atom_mask"][0].cpu().numpy().astype(bool)
                    aux = a
                schedule = "grouped"
                out_coords = np.concatenate(chunks, axis=0)
            dt = time.perf_counter() - t0

            coords = out_coords                                           # (N, L, 37, 3)
            sel = atom_mask.reshape(-1)
            packed = coords.reshape(coords.shape[0], -1, 3)[:, sel, :]    # present atoms only
            np.savez(
                out_npz,
                coords=packed,                       # (N, n_present, 3)
                atom_mask=atom_mask,                 # (L, 37) -> reconstructs the layout
                aatype=aux["s"][0].cpu().numpy().astype(np.int8),
                residue_index=feats["residue_index_orig"].numpy().astype(np.int32),
                rewind_steps=np.asarray(out_rewinds, dtype=np.int16),
                model=model_name,
                schedule=schedule,
                seconds=np.float32(dt),
                # provenance, so a merged round-1 + round-2 tree stays self-describing and nobody
                # has to infer the ladder from the rewind values later
                n_rungs=np.int16(len(set(out_rewinds))),
                seeds_per_rung=np.int16(args.seeds_per_rung if round2 else
                                        len(out_rewinds) // max(len(set(out_rewinds)), 1)),
                ladder_start=np.int16(rmin),
                ladder_top=np.int16(args.rmax),
            )
            done += 1
            if done % 25 == 0:
                el = time.perf_counter() - t_start
                rate = done / el
                print(f"  {done} done ({skipped} skipped)  L={L}  {dt:.1f}s/chain  "
                      f"{rate*3600:.0f} chains/h  eta {(len(mine)-done-skipped)/rate/3600:.1f} h",
                      flush=True)

    el = time.perf_counter() - t_start
    print(f"\nshard {args.shard}: {done} generated, {skipped} already present/skipped, "
          f"{el/3600:.2f} h")
    # ⛔ A count is not evidence: name every skip so each can be re-checked individually.
    if missing:
        print(f"shard {args.shard}: {len(missing)} chains had NO ladder start and were skipped: "
              + " ".join(sorted(missing)))


if __name__ == "__main__":
    main()
