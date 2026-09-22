Stage A chain lists (Directive B, 2026-09-22). Built from valset_analysis/leakage_23501005/ref_{old,new}.tsv
(job 23501005) and the max384 cluster TSV.
stageA_val_1percluster.txt  PRIMARY. max384 VAL chains with: an old non-self reference (tri_full384 index),
                            NO max384-train chain among its old-index cluster-mates, a reference that is not
                            a train chain, and a new synthetic template (synth_index_v4). One chain per max384
                            cluster: the representative if eligible, else the lexicographically first eligible.
stageA_novel.txt            the novel-fold chains (novel_fold_val_chains.txt) passing the same filter;
                            reported SEPARATELY as a fold-level check.
stageA_run_union.txt        what the passes actually sample (primary + novel); analyses subset from it.
stageA_smoke4.txt           4-chain smoke list.
