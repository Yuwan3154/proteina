import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import torch
from Bio.Data import IUPACData
from Bio.PDB import MMCIFParser, Polypeptide
from openfold.config import model_config
from openfold.data import data_pipeline, feature_pipeline, templates
from openfold.np import protein
from openfold.utils.script_utils import (
    load_models_from_command_line,
    parse_fasta,
    prep_output,
    run_model,
)
from openfold.utils.tensor_utils import tensor_tree_map

from aligned_msa_utils import load_indices_map, trim_a3m_to_uniprot_indices


MODEL_NAMES = [
    "model_1_ptm",
    "model_2_ptm",
    "model_3_ptm",
    "model_4_ptm",
    "model_5_ptm",
]


def build_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenFold with all AF2 pTM weights and multiple seeds, then "
            "align predictions to native PDB using USalign."
        )
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Input CSV with columns including pdb, uniprot, length, etc.",
    )
    parser.add_argument(
        "--output_dir",
        default="./openfold_baseline_output",
        help="Directory for predictions and results.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of random seeds per model.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Worker threads for coordinating jobs (GPU runs are serialized).",
    )
    parser.add_argument(
        "--openfold_dir",
        default="~/openfold",
        help="Path to OpenFold repository.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device string for OpenFold inference.",
    )
    parser.add_argument(
        "--msa_dir",
        default="~/data/bad_afdb/msa",
        help="Directory containing AFDB MSAs (.a3m).",
    )
    parser.add_argument(
        "--pdb_dir",
        default="~/data/bad_afdb/pdb",
        help="Directory containing native PDB mmCIF files (with mid-two-char subdirs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run proteins even if per-protein results exist.",
    )
    parser.add_argument(
        "--skip_alignment",
        action="store_true",
        help="Skip USalign step (for debugging).",
    )
    parser.add_argument(
        "--download_msa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled, download missing AFDB MSAs into --msa_dir using "
            "https://alphafold.ebi.ac.uk/files/msa/AF-{uniprot}-F1-msa_v6.a3m"
        ),
    )
    parser.add_argument(
        "--max_targets",
        type=int,
        default=None,
        help="Optional cap on number of targets to process (useful for debugging).",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only run dataset preparation (FASTA + MSA linking) and then exit.",
    )
    parser.add_argument(
        "--only_targets",
        default=None,
        help="Optional comma-separated list of pdb_chain targets to run (e.g. '6WPC_A,6U4K_A').",
    )
    parser.add_argument(
        "--dump_feature_stats",
        action="store_true",
        help="If set, write per-(target,model) feature stats JSON under output_dir/feature_stats/.",
    )
    parser.add_argument(
        "--indices_csv",
        default="~/data/bad_afdb/pdb_70_cluster_reps_aligned_confidence_aggregate_indices.csv",
        help="Path to UniProt->PDB residue index mapping CSV (two-line records).",
    )
    parser.add_argument(
        "--aligned_msa_dir",
        default="~/data/bad_afdb/aligned_msa",
        help="Directory to cache UniProt-index-trimmed MSAs.",
    )
    parser.add_argument(
        "--use_aligned_msa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, trim AFDB MSAs to the UniProt indices corresponding to the PDB chain and use the trimmed MSA.",
    )
    return parser.parse_args()


def aa_three_to_one_map():
    mapping = {}
    for k, v in IUPACData.protein_letters_3to1_extended.items():
        mapping[k.upper()] = v
    return mapping


def extract_chain_sequence(cif_path, chain_id, mapping):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("native", str(cif_path))
    model = structure[0]
    if chain_id not in model:
        return ""
    chain = model[chain_id]
    seq_letters = []
    for residue in chain:
        if not Polypeptide.is_aa(residue, standard=False):
            continue
        resname = residue.get_resname().upper()
        letter = mapping.get(resname, "X")
        seq_letters.append(letter)
    return "".join(seq_letters)


def write_fasta(seq, header, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        fp.write(f">{header}\n")
        fp.write(seq + "\n")


def symlink_msa(uniprot_id, msa_dir, align_dir, tag):
    msa_path = Path(msa_dir) / f"AF-{uniprot_id}-F1-msa_v6.a3m"
    if not msa_path.exists():
        alt_path = Path(msa_dir) / f"{uniprot_id}.a3m"
        msa_path = alt_path
    if not msa_path.exists():
        return None
    target_dir = Path(align_dir) / tag
    target_dir.mkdir(parents=True, exist_ok=True)
    link_path = target_dir / msa_path.name
    if link_path.exists():
        return link_path
    os.symlink(msa_path, link_path)
    return link_path


def symlink_msa_file(msa_path, align_dir, tag, force=False):
    msa_path = Path(msa_path)
    if not msa_path.exists():
        return None
    target_dir = Path(align_dir) / tag
    target_dir.mkdir(parents=True, exist_ok=True)
    link_path = target_dir / msa_path.name
    # NOTE: Path.exists() is False for broken symlinks; use lexists() to detect them.
    if os.path.lexists(link_path):
        if not force:
            return link_path
        Path(link_path).unlink()
    os.symlink(msa_path, link_path)
    return link_path


def download_afdb_msa(uniprot_id, msa_dir):
    """
    Download AFDB MSA for a UniProt ID into msa_dir.
    Mirrors the behavior in data/bad_afdb/template_search_pipeline.py:
      https://alphafold.ebi.ac.uk/files/msa/AF-{uniprot}-F1-msa_v6.a3m
    """
    msa_dir = Path(msa_dir)
    msa_dir.mkdir(parents=True, exist_ok=True)

    # Preserve characters like ':' by URL-encoding; most IDs won't need it.
    uniprot_safe = quote(str(uniprot_id))
    out_path = msa_dir / f"AF-{uniprot_id}-F1-msa_v6.a3m"
    if out_path.exists():
        return out_path

    url = f"https://alphafold.ebi.ac.uk/files/msa/AF-{uniprot_safe}-F1-msa_v6.a3m"
    cmd = ["wget", "-q", "-O", str(out_path), url]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        if out_path.exists():
            out_path.unlink()
        stderr = (res.stderr or "").strip()
        if len(stderr) > 500:
            stderr = stderr[:500] + "...(truncated)"
        print(f"[MSA][download_failed] uniprot={uniprot_id} url={url} rc={res.returncode} stderr={stderr}")
        return None
    return out_path


def get_msa_path(uniprot_id, msa_dir):
    msa_dir = Path(msa_dir)
    p1 = msa_dir / f"AF-{uniprot_id}-F1-msa_v6.a3m"
    if p1.exists():
        return p1
    p2 = msa_dir / f"{uniprot_id}.a3m"
    if p2.exists():
        return p2
    return None


def feature_stats(tag, model_name, feature_dict, processed_feature_dict):
    msa = feature_dict.get("msa", None)
    if msa is None:
        n_msa = None
        length = None
    else:
        n_msa = int(msa.shape[0])
        length = int(msa.shape[1])
    stats = {
        "tag": tag,
        "model_name": model_name,
        "n_msa": n_msa,
        "length": length,
        "feature_keys": sorted(list(feature_dict.keys())),
        "processed_feature_keys": sorted(list(processed_feature_dict.keys())),
    }
    return stats


def write_feature_stats(stats, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stats['tag']}_{stats['model_name']}_feature_stats.json"
    with open(out_path, "w") as fp:
        json.dump(stats, fp, indent=2, sort_keys=True)
    return out_path


def run_usalign(pred_path, native_cif, chain_id):
    cmd = [
        "USalign",
        str(pred_path),
        str(native_cif),
        "-chain2",
        chain_id,
        "-outfmt",
        "2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None, None, result.stderr
    lines = [ln for ln in result.stdout.strip().split("\n") if ln and not ln.startswith("#")]
    if not lines:
        return None, None, "USalign produced no data"
    parts = lines[0].split("\t")
    if len(parts) < 3:
        return None, None, "Unexpected USalign output"
    tm1 = float(parts[2])
    tm2 = float(parts[3]) if len(parts) > 3 else None
    return tm1, tm2, None


def load_openfold_components(config_preset, device, jax_param_path, template_mmcif_dir):
    config = model_config(
        config_preset,
        long_sequence_inference=False,
        use_deepspeed_evoformer_attention=False,
    )

    is_multimer = "multimer" in config_preset
    if is_multimer:
        template_featurizer = templates.HmmsearchHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date="9999-12-31",
            max_hits=config.data.predict.max_templates,
            kalign_binary_path=None,
            release_dates_path=None,
            obsolete_pdbs_path=None,
        )
    else:
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date="9999-12-31",
            max_hits=config.data.predict.max_templates,
            kalign_binary_path=None,
            release_dates_path=None,
            obsolete_pdbs_path=None,
        )

    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )
    if is_multimer:
        data_processor = data_pipeline.DataPipelineMultimer(
            monomer_data_pipeline=data_processor,
        )

    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    model_generator = load_models_from_command_line(
        config,
        device,
        openfold_checkpoint_path=None,
        jax_param_path=jax_param_path,
        output_dir=str(template_mmcif_dir),
    )
    model, _ = next(model_generator)
    return model, data_processor, feature_processor, is_multimer


def run_openfold_single(
    model,
    feature_processor,
    data_processor,
    is_multimer,
    fasta_path,
    alignment_dir,
    tag,
    model_name,
    seed,
    device,
    output_dir,
    cif_output=False,
    precomputed=None,
):
    start_wall = time.perf_counter()
    data_random_seed = seed if seed is not None else random.randrange(2**32)
    random.seed(data_random_seed)
    np.random.seed(data_random_seed)
    torch.manual_seed(data_random_seed + 1)

    if not Path(fasta_path).exists():
        raise SystemExit(f"Missing FASTA: {fasta_path}")

    if precomputed is None:
        with open(fasta_path, "r") as fp:
            data = fp.read()
        tags, seqs = parse_fasta(data)
        if len(tags) != 1 or len(seqs) != 1:
            raise SystemExit(f"Expected single FASTA entry in {fasta_path}, got {len(tags)}")

        msa_dir = Path(alignment_dir) / tag
        if not msa_dir.exists():
            raise SystemExit(f"Missing alignment dir for tag={tag}: {msa_dir}")
        msa_files = list(msa_dir.glob("*.a3m"))
        if len(msa_files) == 0:
            raise SystemExit(f"No .a3m found in alignment dir for tag={tag}: {msa_dir}")

        print(f"[{tag}][{model_name}][seed {seed}] building features fasta={fasta_path} msa_dir={msa_dir}")
        feature_dict = data_processor.process_fasta(
            fasta_path=str(fasta_path),
            alignment_dir=str(Path(alignment_dir) / tag),
            seqemb_mode=False,
        )

        print(f"[{tag}][{model_name}][seed {seed}] processing features")
        processed_feature_dict = feature_processor.process_features(
            feature_dict, mode="predict", is_multimer=is_multimer
        )
        processed_feature_dict = {
            k: torch.as_tensor(v, device=device)
            for k, v in processed_feature_dict.items()
        }
    else:
        feature_dict = precomputed["feature_dict"]
        processed_feature_dict = precomputed["processed_feature_dict"]

    t_model = time.perf_counter()
    print(f"[{tag}][{model_name}][seed {seed}] running model on device={device}")
    out = run_model(model, processed_feature_dict, tag, str(output_dir))
    model_elapsed = time.perf_counter() - t_model

    if precomputed is None:
        processed_feature_dict_np = tensor_tree_map(
            lambda x: x[..., -1].cpu().numpy(),
            processed_feature_dict,
        )
    else:
        processed_feature_dict_np = precomputed["processed_feature_dict_np"]
    out = tensor_tree_map(lambda x: x.cpu().numpy(), out)

    unrelaxed_protein = prep_output(
        out,
        processed_feature_dict_np,
        feature_dict,
        feature_processor,
        model_name,
        200,
        False,
    )

    predictions_dir = Path(output_dir) / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{tag}_{model_name}_seed_{seed}"
    suffix = "_unrelaxed.cif" if cif_output else "_unrelaxed.pdb"
    pred_path = predictions_dir / f"{output_name}{suffix}"

    with open(pred_path, "w") as fp:
        if cif_output:
            fp.write(protein.to_modelcif(unrelaxed_protein))
        else:
            fp.write(protein.to_pdb(unrelaxed_protein))

    scores = {}
    if "ptm" in out:
        scores["ptm"] = float(out["ptm"])
    elif "ptm_score" in out:
        scores["ptm"] = float(out["ptm_score"])
    if "iptm" in out:
        scores["iptm"] = float(out["iptm"])
    if "plddt" in out:
        scores["mean_plddt"] = float(np.mean(out["plddt"]))
    scores["keys"] = sorted(list(out.keys()))

    scores_path = predictions_dir / f"{output_name}_scores.json"
    with open(scores_path, "w") as fp:
        json.dump(scores, fp)

    wall_elapsed = time.perf_counter() - start_wall
    print(
        f"[{tag}][{model_name}][seed {seed}] wall_elapsed={wall_elapsed:.2f}s "
        f"model_elapsed={model_elapsed:.2f}s"
    )

    # Drop large intermediates as soon as we've serialized outputs.
    del out
    del unrelaxed_protein

    return {
        "pred_path": pred_path,
        "scores_path": scores_path,
        "ptm": scores.get("ptm"),
        "iptm": scores.get("iptm"),
        "mean_plddt": scores.get("mean_plddt"),
        "seed": seed,
        "elapsed_sec": wall_elapsed,
        "model_elapsed_sec": model_elapsed,
        "error": None,
    }


def prepare_protein(row, cfg, mapping):
    pdb_chain = row["pdb"]
    uniprot = row["uniprot"]
    pdb_id, chain_id = pdb_chain.split("_")

    native_cif = Path(cfg["pdb_dir"]) / pdb_id[1:3] / f"{pdb_id}.cif"
    if not native_cif.exists():
        print(f"[SKIP][missing_native_cif] {pdb_chain} expected={native_cif}")
        return None

    protein_dir = Path(cfg["output_dir"]) / pdb_chain
    align_dir = Path(cfg["output_dir"]) / "alignments"
    fasta_dir = Path(cfg["output_dir"]) / "fastas"
    protein_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)

    per_protein_csv = Path(cfg["output_dir"]) / "per_protein" / f"{pdb_chain}_tm_scores.csv"
    if per_protein_csv.exists() and not cfg["force"]:
        print(f"[SKIP][already_done] {pdb_chain} per_protein_csv={per_protein_csv}")
        return None

    seq = extract_chain_sequence(native_cif, chain_id, mapping)
    if len(seq) == 0:
        print(f"[SKIP][empty_chain_sequence] {pdb_chain} native_cif={native_cif}")
        return None
    fasta_path = fasta_dir / f"{pdb_chain}.fasta"
    write_fasta(seq, pdb_chain, fasta_path)

    msa_path = get_msa_path(uniprot, cfg["msa_dir"])
    if msa_path is None and cfg["download_msa"]:
        print(f"[MSA][missing] {pdb_chain} uniprot={uniprot} -> downloading")
        msa_path = download_afdb_msa(uniprot, cfg["msa_dir"])

    msa_to_link = None
    if cfg["use_aligned_msa"]:
        indices = cfg["indices_map"].get((pdb_chain, uniprot), None)
        if indices is None:
            print(f"[MSA][aligned_missing_indices] {pdb_chain} uniprot={uniprot} -> using raw MSA")
        else:
            aligned_path = Path(cfg["aligned_msa_dir"]) / f"{pdb_chain}_{uniprot}.a3m"
            if cfg["force"] or not aligned_path.exists():
                aligned_path.parent.mkdir(parents=True, exist_ok=True)
                trim_a3m_to_uniprot_indices(msa_path, indices, aligned_path)
            msa_to_link = aligned_path

    if msa_to_link is None:
        msa_to_link = msa_path

    msa_link = symlink_msa_file(msa_to_link, align_dir, pdb_chain, force=cfg["force"])
    if msa_link is None:
        print(f"[SKIP][missing_msa] {pdb_chain} uniprot={uniprot} msa_dir={cfg['msa_dir']}")
        return None

    template_root = cfg["template_root"]
    template_root.mkdir(parents=True, exist_ok=True)
    template_link = template_root / f"{pdb_id}.cif"
    if not template_link.exists():
        os.symlink(native_cif, template_link)

    return {
        "pdb_chain": pdb_chain,
        "uniprot": uniprot,
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "native_cif": native_cif,
        "protein_dir": protein_dir,
        "align_dir": align_dir,
        "fasta_path": fasta_path,
        "per_protein_csv": per_protein_csv,
    }


def main():
    args = build_args()

    cfg = {
        "input_csv": Path(os.path.expanduser(args.input_csv)),
        "output_dir": Path(os.path.expanduser(args.output_dir)),
        "num_seeds": args.num_seeds,
        "num_workers": args.num_workers,
        "openfold_dir": Path(os.path.expanduser(args.openfold_dir)),
        "device": args.device,
        "msa_dir": Path(os.path.expanduser(args.msa_dir)),
        "pdb_dir": Path(os.path.expanduser(args.pdb_dir)),
        "force": args.force,
        "skip_alignment": args.skip_alignment,
        "template_root": Path(os.path.expanduser(args.output_dir)) / "template_mmcif",
        "download_msa": args.download_msa,
        "use_aligned_msa": args.use_aligned_msa,
        "aligned_msa_dir": Path(os.path.expanduser(args.aligned_msa_dir)),
    }

    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    (cfg["output_dir"] / "alignments").mkdir(parents=True, exist_ok=True)
    (cfg["output_dir"] / "fastas").mkdir(parents=True, exist_ok=True)
    (cfg["output_dir"] / "per_protein").mkdir(parents=True, exist_ok=True)
    if cfg["use_aligned_msa"]:
        cfg["aligned_msa_dir"].mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg["input_csv"])
    indices_map = load_indices_map(os.path.expanduser(args.indices_csv))
    cfg["indices_map"] = indices_map
    mapping = aa_three_to_one_map()
    all_results = []
    summary_path = cfg["output_dir"] / "summary_results.csv"

    info_map = {}
    df_iter = df
    if args.max_targets is not None:
        df_iter = df.head(args.max_targets)
    only = None
    if args.only_targets is not None:
        only = {t.strip() for t in args.only_targets.split(",") if len(t.strip()) > 0}
    for _, row in df_iter.iterrows():
        if only is not None and row["pdb"] not in only:
            continue
        info = prepare_protein(row, cfg, mapping)
        if info is None:
            continue
        info_map[info["pdb_chain"]] = info

    print(f"[INFO] total_rows={len(df)} runnable_after_prepare={len(info_map)} download_msa={cfg['download_msa']}")

    if args.prepare_only:
        print("[INFO] prepare_only set; exiting before model runs")
        return

    for model_name in MODEL_NAMES:
        model_params = cfg["openfold_dir"] / "openfold" / "resources" / "params" / f"params_{model_name}.npz"
        if not model_params.exists():
            raise SystemExit(f"Missing OpenFold params: {model_params}")
        model, data_processor, feature_processor, is_multimer = load_openfold_components(
            model_name,
            cfg["device"],
            str(model_params),
            str(cfg["template_root"]),
        )

        for info in info_map.values():
            tag = info["pdb_chain"]
            # IMPORTANT: Don't keep a cache across all proteins; it will accumulate GPU tensors
            # and eventually OOM. Instead, compute features once per (protein, model), reuse
            # across seeds, then free.
            msa_dir = Path(info["align_dir"]) / tag
            print(f"[{tag}][{model_name}] caching features fasta={info['fasta_path']} msa_dir={msa_dir}")
            feature_dict = data_processor.process_fasta(
                fasta_path=str(info["fasta_path"]),
                alignment_dir=str(Path(info["align_dir"]) / tag),
                seqemb_mode=False,
            )
            processed_feature_dict = feature_processor.process_features(
                feature_dict, mode="predict", is_multimer=is_multimer
            )
            processed_feature_dict = {
                k: torch.as_tensor(v, device=cfg["device"])
                for k, v in processed_feature_dict.items()
            }
            processed_feature_dict_np = tensor_tree_map(
                lambda x: x[..., -1].cpu().numpy(),
                processed_feature_dict,
            )
            precomputed = {
                "feature_dict": feature_dict,
                "processed_feature_dict": processed_feature_dict,
                "processed_feature_dict_np": processed_feature_dict_np,
            }
            if args.dump_feature_stats:
                stats = feature_stats(tag, model_name, feature_dict, processed_feature_dict)
                stats["cuda"] = {
                    "is_available": bool(torch.cuda.is_available()),
                }
                if torch.cuda.is_available():
                    stats["cuda"]["memory_allocated"] = int(torch.cuda.memory_allocated())
                    stats["cuda"]["memory_reserved"] = int(torch.cuda.memory_reserved())
                out_path = write_feature_stats(
                    stats, Path(cfg["output_dir"]) / "feature_stats"
                )
                print(f"[{tag}][{model_name}] wrote feature stats: {out_path}")

            for seed in range(cfg["num_seeds"]):
                res = run_openfold_single(
                    model=model,
                    feature_processor=feature_processor,
                    data_processor=data_processor,
                    is_multimer=is_multimer,
                    fasta_path=info["fasta_path"],
                    alignment_dir=info["align_dir"],
                    tag=tag,
                    model_name=model_name,
                    seed=seed,
                    device=cfg["device"],
                    output_dir=info["protein_dir"],
                    cif_output=False,
                    precomputed=precomputed,
                )

                tm1 = None
                tm2 = None
                aln_error = res.get("error")
                if not cfg["skip_alignment"]:
                    tm1, tm2, aln_error = run_usalign(res["pred_path"], info["native_cif"], info["chain_id"])

                all_results.append(
                    {
                        "protein_id": tag,
                        "uniprot": info["uniprot"],
                        "model": model_name,
                        "seed": seed,
                        "tm_pred_norm": tm1,
                        "tm_native_norm": tm2,
                        "ptm": res.get("ptm"),
                        "pred_path": str(res["pred_path"]),
                        "elapsed_sec": res.get("elapsed_sec"),
                        "model_elapsed_sec": res.get("model_elapsed_sec"),
                        "error": aln_error,
                    }
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Free per-protein cached features after all seeds.
            del precomputed
            del processed_feature_dict_np
            del processed_feature_dict
            del feature_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(all_results) > 0:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(summary_path, index=False)
        per_dir = cfg["output_dir"] / "per_protein"
        per_dir.mkdir(parents=True, exist_ok=True)
        for protein_id, df_group in df_all.groupby("protein_id"):
            per_path = per_dir / f"{protein_id}_tm_scores.csv"
            df_group.to_csv(per_path, index=False)


if __name__ == "__main__":
    main()

