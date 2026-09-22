#!/bin/bash
# Export tri_cb8synth_v5's validation P@L history (CPU only, wandb API -- no GPU, no local .wandb
# parsing). A real #!/bin/bash file, NOT `sbatch --wrap`: --wrap runs under /bin/sh where `source`
# silently no-ops and the venv would never activate.
set -uo pipefail
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
OUT=/orcd/scratch/orcd/011/chenxiou/.tmp/tri_val_pal.json
python -c "import wandb, sys; print('[precheck] wandb', wandb.__version__, file=sys.stderr)"
echo "PRECHECK_EXIT=$?"
python /orcd/scratch/orcd/011/chenxiou/.tmp/tri_val_pal_history.py > "$OUT"
echo "EXPORT_EXIT=$?"
# ⛔ report the ARTIFACT, not just the exit status: an empty/short file means the keys were wrong
wc -c < "$OUT" | sed 's/^/[out] bytes: /'
python -c "
import json
d = json.load(open('$OUT'))
print('[out] run', d['run'], 'state', d['state'], 'step', d['summary_global_step'])
for k, v in d['series'].items():
    print('[out] %-66s n=%d' % (k, len(v)))
"
echo "VERIFY_EXIT=$?"
