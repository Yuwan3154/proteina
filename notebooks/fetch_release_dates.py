import sys
import time
import requests
import pandas as pd


def fetch_release_dates(pdb_ids):
    """Map 4-char PDB id -> RCSB initial_release_date (YYYY-MM-DD); '' if not in RCSB."""
    out = {}
    for pid in pdb_ids:
        r = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pid}", timeout=30)
        if r.status_code == 200:
            d = r.json().get("rcsb_accession_info", {}).get("initial_release_date", "")
            out[pid] = d[:10] if d else ""
        else:
            out[pid] = ""
        time.sleep(0.1)
    return out


if __name__ == "__main__":
    in_csv, out_csv = sys.argv[1], sys.argv[2]
    id_col = sys.argv[3] if len(sys.argv) > 3 else "pdb"
    df = pd.read_csv(in_csv)
    pdb4 = df[id_col].astype(str).str.split("_").str[0].str.upper()
    dmap = fetch_release_dates(sorted(set(pdb4)))
    df["release_date"] = pdb4.map(dmap)
    df.to_csv(out_csv, index=False)
    print(df[[id_col, "release_date"]].sort_values("release_date").to_string(index=False))
