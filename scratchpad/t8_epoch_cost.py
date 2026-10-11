"""Amortised torch.compile speedup over one T8 epoch, from measured per-bucket step times.

An epoch draws one member per train cluster (epoch batch plan), so the expected epoch compute is
sum over clusters of mean over its members of t(bucket(L_member)), with t the measured median steady time of one
micro-step (forward + backward, incl. its self-conditioning pass when drawn) per L bucket, at batch 1, from
scratchpad/t8_compile_ddp_stress.py runs. The T bucket is averaged inside each L bucket's median.
Compiled epoch = that sum with compiled times + the one-off compile cost: per rank, the time ABOVE the steady median
summed over its train_step rows (a cold SC compile lands inside some train_step row, first call or not), then the MAX
over ranks (ranks compile in parallel and wait for each other). Stated assumptions: batch 1; T drawn uniformly in 1..96
by the stress runs (the real element-count distribution is not known until the T8 index exists); validation compute
and compiles left out of both arms; all train members (not yet restricted to chains with templates).
Usage: python scratchpad/t8_epoch_cost.py --clusters TSV --csv CSV --train_ids TXT --timing_dir DIR [--world 4]
"""

import argparse
import collections
import glob
import json
import statistics

import pandas as pd

LEN_EDGES = [64, 128, 192, 256, 320, 384, 448, 512]


def bucket(x):
    return next(e for e in LEN_EDGES if e >= x)


def medians(rows):
    by = collections.defaultdict(list)
    for r in rows:
        if r["kind"] == "train_step" and not r["first_call"] and r["B"] == 1:
            by[r["Lb"]].append(r["s"])
    return {k: statistics.median(v) for k, v in by.items()}


def compile_excess(rows, t):
    """max over ranks of sum, over that rank's train_step rows that COMPILED (a first call, or its nested sc row --
    written just before it in file order -- was a first call), of max(0, s - steady median of its L bucket)."""
    per_rank = collections.defaultdict(float)
    prev_sc_first = collections.defaultdict(bool)
    for r in rows:
        if r["kind"] == "sc":
            prev_sc_first[r["rank"]] = r["first_call"]
        elif r["kind"] == "train_step" and r["B"] == 1:
            if r["first_call"] or prev_sc_first[r["rank"]]:
                per_rank[r["rank"]] += max(0.0, r["s"] - t[r["Lb"]])
            prev_sc_first[r["rank"]] = False
    return max(per_rank.values()), len(per_rank)


def load(d, tag, run):
    return [json.loads(x) for f in glob.glob(f"{d}/{tag}_run{run}_rank*.jsonl") for x in open(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--train_ids", required=True)
    ap.add_argument("--timing_dir", required=True)
    ap.add_argument("--world", type=int, default=4)
    args = ap.parse_args()

    train = set(open(args.train_ids).read().split())
    df = pd.read_csv(args.csv, usecols=["id", "processed_length"])
    plen = dict(zip(df["id"].astype(str), df["processed_length"]))
    members = collections.defaultdict(list)
    for line in open(args.clusters):
        rep, mem = line.split()[:2]
        if mem in train and mem in plen and plen[mem] == plen[mem]:
            assert plen[mem] <= LEN_EDGES[-1], (mem, plen[mem])
            members[rep].append(int(plen[mem]))
    n_cl = len(members)
    n_mem = sum(map(len, members.values()))
    print(f"train clusters with lengths: {n_cl}; members matched {n_mem} of {len(train)} train ids")

    cold, warm, eager = load(args.timing_dir, "compiled", 0), load(args.timing_dir, "compiled", 1), load(args.timing_dir, "eager", 2)
    t_e = medians(eager)
    t_warm = medians(warm)
    t_c = {b: t_warm.get(b, v) for b, v in medians(cold).items()}  # warm-run steady rows carry no nested cold SC compile
    missing = [b for b in LEN_EDGES if b not in t_c or b not in t_e]
    assert not missing, f"no steady timing for L buckets {missing}"
    share = collections.Counter()
    epoch = {"eager": 0.0, "compiled": 0.0}
    for lens in members.values():
        for L in lens:
            w = 1.0 / len(lens)
            share[bucket(L)] += w
            epoch["eager"] += w * t_e[bucket(L)]
            epoch["compiled"] += w * t_c[bucket(L)]
    compile_cold, n_r = compile_excess(cold, t_c)
    compile_warm, _ = compile_excess(warm, t_c)
    seen_warm = {(r["Lb"], r["Tb"]) for r in warm if r["kind"] == "train_step"}
    seen_cold = {(r["Lb"], r["Tb"]) for r in cold if r["kind"] == "train_step"}
    print(f"ranks in the timing runs: {n_r}; (L, T) train buckets seen: cold {len(seen_cold)} of 48, warm {len(seen_warm)}"
          + ("" if seen_warm >= seen_cold else " -> warm compile cost UNDERCOUNTED (warm run missed buckets)"))
    print("expected samples per epoch by L bucket:", {b: round(share[b]) for b in LEN_EDGES})
    print("per-micro-step median (s) eager / compiled / speedup by L bucket:")
    for b in LEN_EDGES:
        print(f"  L<={b}: {t_e[b]:.3f} / {t_c[b]:.3f} / x{t_e[b] / t_c[b]:.2f}")
    W = args.world
    e, c = epoch["eager"] / W, epoch["compiled"] / W
    print(f"TRAINING epoch compute per GPU at world {W} (batch 1, validation excluded): eager {e / 60:.1f} min, compiled {c / 60:.1f} min "
          f"(steady speedup x{e / c:.2f})")
    print(f"one-off compile (wall clock, slowest rank, above steady time): cold {compile_cold / 60:.1f} min, warm "
          f"{compile_warm / 60:.1f} min -- measured with the CPUs of the stress step (see its log: 'cpus N'); more CPUs"
          f" compile faster")
    for name, comp in (("cold", compile_cold), ("warm", compile_warm)):
        for n_ep in (1, 5, 30):
            print(f"  amortised over {n_ep:2d} epoch(s), {name} cache: x{n_ep * e / (n_ep * c + comp):.2f}")


if __name__ == "__main__":
    main()
