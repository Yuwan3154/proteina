"""Weight surgery: old ConFind tri (tri_full384, vocab 65, 10 pair features) -> current tri recipe (vocab 44,
8 pair features, align + mlm heads). User 2026-09-23: fine-tune the ConFind tri with all the new features.

Exact, no approximation:
  * nn.cell_in.weight [320, 2 + 10] -> [320, 2 + 8]: drop the min_ca_dist / mean_ca_dist columns (old input cols
    4, 5 -- the removal Stage A validated as equivalent to mean-filling them);
  * nn.topo_emb.weight [65, d] -> [44, d]: old alphabet types (loop, helix, strand), new (helix, strand), 21 slots per
    type, 2 special tokens => new = old[0:2] + old[23:44] + old[44:65];
  * everything else copied unchanged; the new align/mlm heads are absent and cold-start from the model's own init
    (train.py's pretrain loader logs them).
The pretrain loader COLD-STARTS any shape-mismatched tensor, so every surgered shape is asserted here.

Usage: python scratchpad/surgery_confind_tri.py OLD_EMA_CKPT OUT_CKPT
"""

import sys

import torch

old_path, out_path = sys.argv[1], sys.argv[2]
ck = torch.load(old_path, map_location="cpu", weights_only=False)
sd = ck["state_dict"]
N_SLOTS, N_SPECIAL = 21, 2

w = sd["nn.cell_in.weight"]
assert w.shape[1] == 12, f"cell_in expected 12 inputs, got {tuple(w.shape)}"
keep = [c for c in range(12) if c not in (4, 5)]
emb = sd["nn.topo_emb.weight"]
assert emb.shape[0] == N_SPECIAL + 3 * N_SLOTS == 65, f"topo_emb expected 65 rows, got {tuple(emb.shape)}"
rows = list(range(N_SPECIAL)) + list(range(N_SPECIAL + N_SLOTS, N_SPECIAL + 3 * N_SLOTS))

new = dict(sd)
new["nn.cell_in.weight"] = w[:, keep].clone()
new["nn.topo_emb.weight"] = emb[rows].clone()
assert new["nn.cell_in.weight"].shape[1] == 10 and new["nn.topo_emb.weight"].shape[0] == 44
assert not any(k.startswith("nn.align") or k.startswith("nn.mlm") for k in new), "old ckpt unexpectedly has new heads"

meta = {"source": old_path, "source_global_step": ck.get("global_step"),
        "surgery": "cell_in drop cols 4,5 (min/mean_ca_dist); topo_emb rows [0:2]+[23:65] (65->44)"}
torch.save({"state_dict": new, "surgery": meta}, out_path)
changed = [k for k in new if not torch.equal(new[k], sd[k]) if new[k].shape == sd[k].shape] + \
          [k for k in new if new[k].shape != sd[k].shape]
print(f"[surgery] {len(new)} tensors; reshaped: {sorted(set(changed))}; source step {meta['source_global_step']}")
print(f"[surgery] wrote {out_path}")
