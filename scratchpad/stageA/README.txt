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
stageA_primary_seqclean.txt 144 of the 191 primary chains with NO max384-train chain at >= 25% identity AND
                            >= 80% query or target coverage (mmseqs easy-search val/test vs all 254,946 train
                            chains, job 23498227 -> valset_analysis/leakage_23498227/hits_all.m8; thresholds = the
                            S25 clustering's own --min-seq-id 0.25 -c 0.8). The cluster split is not transitive:
                            47 of the 191 have such a hit (best identity median 0.32, max 0.60).
stageA_novel_seqclean.txt   8 of the 9 novel-fold chains under the same test.
Pre-registered (2026-09-23, before any Stage A result): report the 191-chain set AND the 144-chain
sequence-clean subset side by side; flag any conclusion that differs between them.
