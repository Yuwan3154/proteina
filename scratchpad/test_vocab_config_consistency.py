"""The dataset's SSE alphabet and the model's vocabulary size must agree.

`sse_types` lives in the DATASET yaml and `topology_vocab_size` in the NN yaml, with nothing linking
them. `topology_vocab_size` also defaults to 65 in code. So editing `sse_types` without touching the
nn yaml would silently mis-size BOTH nn.Embedding(topology_vocab_size, ...) and the MLM head's
output layer -- too small gives an out-of-bounds index (a CUDA assert, hours in), too large gives
the MLM head classes it can never be trained on.

This reads the REAL yamls and checks them against the alphabet the code builds.
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.sse_topology import (
    DSSP_LOOP,
    N_SPECIAL_TOKENS,
    SSEAlphabet,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_YAML = os.path.join(
    REPO, "configs/datasets_config/pdb",
    "pdb_train_contact-CB8-synthtopo_S25_max384_purge-test_cutoff-190828.yaml")
NN_YAML = os.path.join(REPO, "configs/experiment_config/model/nn/contact_map_tri_30M_cb8.yaml")


def find_key(node, key):
    """First value for `key` anywhere in the nested structure."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            got = find_key(v, key)
            if got is not None:
                return got
    elif isinstance(node, list):
        for v in node:
            got = find_key(v, key)
            if got is not None:
                return got
    return None


def main():
    ds = yaml.safe_load(open(DATASET_YAML))
    nn = yaml.safe_load(open(NN_YAML))
    sse_types = find_key(ds, "sse_types")
    vocab = find_key(nn, "topology_vocab_size")
    assert sse_types is not None, "sse_types not found in the dataset yaml"
    assert vocab is not None, "topology_vocab_size not found in the nn yaml"

    alphabet = SSEAlphabet(types=tuple(int(t) for t in sse_types))
    print(f"dataset sse_types      : {sse_types}")
    print(f"nn topology_vocab_size : {vocab}")
    print(f"alphabet vocab_size    : {alphabet.vocab_size} "
          f"({N_SPECIAL_TOKENS} special + {len(alphabet.types)} types x {alphabet.slots_per_type})")

    ok = True
    if alphabet.vocab_size != vocab:
        ok = False
        print(f"FAIL: nn topology_vocab_size {vocab} != alphabet {alphabet.vocab_size} for "
              f"sse_types {sse_types}; the embedding and the MLM head would be mis-sized")
    else:
        print("PASS: embedding / MLM head size matches the dataset's alphabet")

    if DSSP_LOOP in [int(t) for t in sse_types]:
        ok = False
        print(f"FAIL: loops (DSSP {DSSP_LOOP}) are in sse_types; tri is meant to be helix+strand only")
    else:
        print("PASS: loops are excluded from the alphabet")

    hi = max(alphabet.token(t, n) for t in alphabet.types for n in range(alphabet.min_len, 60))
    if hi >= vocab:
        ok = False
        print(f"FAIL: highest token id {hi} does not fit in a {vocab}-row embedding")
    else:
        print(f"PASS: highest token id {hi} < {vocab}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
