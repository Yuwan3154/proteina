# T7-FULL: the 96 held-out queries not yet scored -> 8 length-balanced chunks of 12, each stem listed x8 (8 samples).
import json
import pathlib

S = pathlib.Path("/orcd/scratch/orcd/011/chenxiou")
REPO = S / "proteina_t7"
O = S / "t7/full"
O.mkdir(parents=True, exist_ok=False)
union = [l.split()[0] for l in open(REPO / "scratchpad/stageA/stageA_run_union.txt") if l.strip()]
done = {l.split()[0] for l in open(S / "t7/quick/subset.txt") if l.strip()}
assert len(union) == len(set(union)) == 195 and done <= set(union) and len(done) == 99
rest = [s for s in union if s not in done]
L = {}
for l in open(S / "stageA/new_nonself/samples.jsonl"):
    r = json.loads(l); L[r["stem"]] = int(r["L"])
assert all(s in L for s in rest), [s for s in rest if s not in L]
rest.sort(key=lambda s: -L[s])
NCH = 8
chunks = [rest[i::NCH] for i in range(NCH)]  # round-robin on descending length -> balanced, chunk 0 has the longest
(O / "rest_q.txt").write_text("".join(s + "\n" for s in rest))
for i, c in enumerate(chunks):
    (O / f"chunk{i:02d}_q.txt").write_text("".join(s + "\n" for s in c))
    (O / f"chunk{i:02d}.txt").write_text("".join(s + "\n" for s in c for _ in range(8)))
    print(f"chunk{i:02d}: {len(c)} queries, L max {max(L[s] for s in c)}, sum L^2 {sum(L[s]**2 for s in c)/1e3:.0f}k")
print(f"rest {len(rest)} queries, L range {L[rest[-1]]}-{L[rest[0]]}; done subset {len(done)}")
