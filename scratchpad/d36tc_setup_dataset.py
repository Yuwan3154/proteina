"""D36tc: write the rows CSV under the name the dataset config's dataselector implies (D36m recipe).

prepare_data only takes the chain-aware branch when the dataset CSV already exists under the identifier
derived from the dataselector's own parameters, so the name is derived here, never guessed.
"""
import argparse
import os
import sys

import hydra
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--rows", required=True, help="csv with pdb,chain,id,role")
args = ap.parse_args()

sys.path.insert(0, args.repo)
from proteinfoundation.datasets.pdb_data import PDBLightningDataModule  # noqa: E402,F401

with hydra.initialize_config_dir(os.path.join(args.repo, "configs/datasets_config/pdb"), version_base=hydra.__version__):
    cfg = hydra.compose(config_name=args.dataset)
dm = hydra.utils.instantiate(cfg.datamodule)
ident = dm._get_file_identifier(dm.dataselector)
csv_path = os.path.join(str(dm.data_dir), f"{ident}.csv")
assert not os.path.exists(csv_path), f"{csv_path} exists -- fresh data dir expected"

df = pd.read_csv(args.rows)
assert df["id"].is_unique, "id column must be unique"
missing = [r.pdb for r in df.itertuples() if not os.path.exists(os.path.join(str(dm.data_dir), "raw", f"{r.pdb}.cif"))]
assert not missing, f"raw cif missing for {len(missing)} entries: {missing[:8]}"
df[["pdb", "chain", "id"]].to_csv(csv_path, index=False)
print(f"data_dir  {dm.data_dir}\ncsv       {csv_path}\nrows      {len(df)} " + str(df["role"].value_counts().to_dict()))
