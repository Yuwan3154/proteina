import csv
import os
from pathlib import Path


def load_indices_map(indices_csv_path):
    """
    Parse data/bad_afdb/pdb_70_cluster_reps_aligned_confidence_aggregate_indices.csv.

    Format is two-line records:
      line1: pdb_chain,uniprot,identity,plddt,coverage,length
      line2: comma-separated 0-based UniProt residue indices (ungapped) that map to the PDB chain

    Returns:
      dict with key (pdb_chain, uniprot) -> list[int]
    """
    indices_csv_path = Path(indices_csv_path)
    mapping = {}

    with open(indices_csv_path, "r") as fp:
        while True:
            meta = fp.readline()
            if meta == "":
                break
            meta = meta.strip()
            if meta == "":
                continue

            idx_line = fp.readline()
            if idx_line == "":
                break
            idx_line = idx_line.strip()

            meta_parts = [p.strip() for p in meta.split(",")]
            if len(meta_parts) < 2:
                continue
            pdb_chain = meta_parts[0]
            uniprot = meta_parts[1]

            if idx_line == "":
                continue

            idx_parts = [p.strip() for p in idx_line.split(",") if p.strip() != ""]
            indices = [int(p) for p in idx_parts]
            mapping[(pdb_chain, uniprot)] = indices

    return mapping


def read_a3m(a3m_path):
    """
    Read an A3M file as (header, sequence) records. Sequences may span multiple lines.
    Header includes the leading '>' line (without trailing newline).
    """
    a3m_path = Path(a3m_path)
    records = []
    header = None
    seq_chunks = []

    with open(a3m_path, "r") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line
                seq_chunks = []
            else:
                if header is None:
                    continue
                seq_chunks.append(line.strip())

    if header is not None:
        records.append((header, "".join(seq_chunks)))

    return records


def write_a3m(records, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        for header, seq in records:
            fp.write(f"{header}\n")
            fp.write(f"{seq}\n")


def _strip_lowercase_insertions(seq):
    return "".join([c for c in seq if not c.islower()])


def _uniprot_index_to_alignment_col(query_aligned_seq):
    """
    Build mapping UniProt residue index -> alignment column index,
    treating '-' in the query as not a residue (doesn't increment residue index).
    """
    m = {}
    res_i = 0
    for col_i, c in enumerate(query_aligned_seq):
        if c == "-":
            continue
        m[res_i] = col_i
        res_i += 1
    return m


def trim_a3m_to_uniprot_indices(a3m_in, indices, a3m_out):
    """
    Trim an AFDB A3M to the subset of UniProt residue indices that map to the PDB chain.

    Semantics:
      - indices are 0-based residue positions on the ungapped UniProt sequence.
      - discontinuities are handled by concatenating only selected indices.
      - lowercase insertions in A3M are removed before slicing.
    """
    records = read_a3m(a3m_in)
    if len(records) == 0:
        raise SystemExit(f"Empty A3M: {a3m_in}")

    # Find query record.
    query_idx = None
    for i, (h, _) in enumerate(records):
        if h == ">query":
            query_idx = i
            break
    if query_idx is None:
        query_idx = 0

    query_aligned = _strip_lowercase_insertions(records[query_idx][1])
    idx_to_col = _uniprot_index_to_alignment_col(query_aligned)

    max_idx = max(indices) if len(indices) > 0 else -1
    if max_idx >= len(idx_to_col):
        raise SystemExit(
            f"Indices out of range for {a3m_in}: max_idx={max_idx} uniprot_len={len(idx_to_col)}"
        )

    cols = [idx_to_col[i] for i in indices]

    out_records = []
    for header, seq in records:
        aligned = _strip_lowercase_insertions(seq)
        trimmed = "".join([aligned[c] for c in cols])
        out_records.append((header, trimmed))

    # Validate query length.
    out_query_seq = out_records[query_idx][1]
    if len(out_query_seq) != len(indices):
        raise SystemExit(
            f"Trimmed query length mismatch for {a3m_in}: got={len(out_query_seq)} expected={len(indices)}"
        )

    write_a3m(out_records, a3m_out)
    return Path(a3m_out)


