"""T8 alignment-bias study, step 1 (Engaging CPU): join each dumped sample's alignment-head outputs (first and last
network call of the sampling run, from stageA_pass DUMP_ALIGN=1) with its ground truth: the stored USalign
(-TMscore 0, i.e. TM-align) residue -> reference-element target of the template row it was conditioned on, read
through TopologyReferenceTransform._align_for exactly as training reads it. Writes one compact npz for plotting.
Usage: python t8_align_bias_collect.py DUMP_DIR [DUMP_DIR ...] INDEX_PT OUT.npz
"""

import json
import os
import sys

import numpy as np

from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform


def main(dump_dirs, index_pt, out):
    tf = TopologyReferenceTransform(index_path=index_pt, reference_source="synthetic", sse_types=(1, 2))
    tf._ensure_loaded()
    recs = []
    for d in dump_dirs:
        for line in open(os.path.join(d, "samples.jsonl")):
            r = json.loads(line)
            z = np.load(os.path.join(d, r["file"]))
            assert "align_logits_last" in z.files, f"{r['file']}: no align dump (run with DUMP_ALIGN=1)"
            L, T = int(z["L"]), int(z["he_tokens"].shape[0])
            row = tf._id_to_row[r["ref_id"]]
            assert not bool(tf._index["row_is_native"][row]), f"{r['ref_id']}: a native row, not a template"
            a0, a1 = int(tf._index["align_offset"][row]), int(tf._index["align_offset"][row + 1])
            assert a1 - a0 == L, f"{r['ref_id']}: stored alignment length {a1 - a0} != L {L} (_align_for would go all-NONE)"
            tgt = tf._align_for(row, L, T).numpy()
            assert tgt.shape == (L,) and int(tgt.max()) < T
            recs.append(dict(stem=r["stem"], ref_id=r["ref_id"], L=L, T=T, target=tgt,
                             template_tm=float(tf._index["row_tm"][row]),
                             he_tokens=z["he_tokens"], he_pos_raw=z["he_pos_raw"],
                             a_first=z["align_logits_first"], n_first=z["align_none_first"],
                             a_last=z["align_logits_last"], n_last=z["align_none_last"],
                             n_calls=int(z["n_nn_calls"])))
    calls = sorted({r["n_calls"] for r in recs})
    assert len(calls) == 1, f"network calls per run differ across samples {calls}: first/last would mean different steps"
    print(f"[collect] {len(recs)} samples from {len(dump_dirs)} dump dirs; network calls per run {calls[0]}")
    np.savez_compressed(out, recs=np.array(recs, dtype=object))
    print(f"[collect] -> {out}")


if __name__ == "__main__":
    assert len(sys.argv) >= 4, __doc__
    main(sys.argv[1:-2], sys.argv[-2], sys.argv[-1])
