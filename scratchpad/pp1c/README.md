# Vendored T2 synthetic-template tools

Verbatim copies of the tools that produced the T2 synthetic templates, brought into this repo so
they reach Engaging by `git pull` instead of an rsync of code. Checksums verified against the
copies actually used in production on SuperCloud:

| file | md5 | production copy |
|---|---|---|
| `generate_templates.py` | `fdd0c20542617041ea4a15f17a931307` | `SuperCloud:~/pp1c_work/scripts/` |
| `build_template_index.py` | `ff9c7c60fa9d7bb4debd10ea55ceaf8c` | `SuperCloud:~/openfold_t2/prune_work/` |
| `prune_templates_to_band.py` | `5a8bab97da4fd321695728991efd2053` | `SuperCloud:~/openfold_t2/prune_work/` |
| `../openfold/utils/tm_score.py` | `e2932930262808f397898817f0d35a95` | `SuperCloud:~/openfold_t2/openfold/utils/` |

⛔ `tm_score.py` sits at `scratchpad/openfold/utils/` and NOT next to the other three on purpose:
`build_template_index.py` path-loads it as `parents[1]/openfold/utils/tm_score.py` (it cannot import
the openfold package, whose `__init__` needs an untracked `resources/`). Keeping that layout lets the
tool stay byte-identical to the version whose output the existing 82,730-chain index was built from.

`generate_templates.py` additionally needs a **patched** protpardelle-1c: stock upstream does not
accept a per-sample rewind list (tiered staggered injection). On Engaging the patched checkout is
`/orcd/scratch/orcd/011/chenxiou/pp1c/protpardelle-1c` (branch `t2-batch-opt`, the 10 patches from
`SuperCloud:~/pp1c_work/scripts/0*.patch` applied with `git am`), env `/orcd/scratch/orcd/011/chenxiou/pp1c/env`.

Launcher: `gen_templates_engaging.sbatch` (round-1 recipe: 64 rungs 90-375, tiered <=300 / grouped
8x8, cc89 for residue-index span <=484 else cc91, seed 0, max length 512).
