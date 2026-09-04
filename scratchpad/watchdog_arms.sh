#!/bin/bash
# Hourly failsafe for the two warm-started arms (2026-09-04 user directive).
#
# The self-chain already handles ordinary segment turnover; this covers what it cannot:
#   - the chain running out (CHAIN reaches 0) and the arm silently ending
#   - a segment dying without queueing a successor
#   - a fast-fail loop, which is what burned all four tri segments in 26 minutes on 2026-09-03
#
# It relaunches ONLY when an arm has nothing running and nothing pending, and a circuit breaker
# refuses to relaunch when the recent segments have been dying fast -- relaunching into a broken
# state is how that outage turned one failure into four.
#
# Deliberately does NOT touch tri_sm120 (the pi_so3 baseline) or any other project's jobs.
set -uo pipefail

REPO=/orcd/scratch/orcd/011/chenxiou/proteina_ot
STATE=$REPO/.watchdog
LOG=$STATE/watchdog.log
mkdir -p "$STATE"

MIN_HEALTHY_MIN=20   # a segment shorter than this counts as a fast-fail
MAX_FASTFAIL=2       # this many consecutive fast-fails trips the breaker
MAX_RELAUNCH=6       # absolute cap on watchdog relaunches per arm

say() { echo "[$(date -Is)] $*" >> "$LOG"; }

for spec in "tri_ot_fgw:run_tri_ot_h200.sbatch" "tri_ctrl:run_tri_ctrl_h200.sbatch"; do
    NAME=${spec%%:*}
    SCRIPT=${spec##*:}

    # Count this arm's live jobs by NAME. Safe now that the chain passes --job-name through;
    # before that fix successors came back as "tri_sm120" and would have been invisible here.
    # ⛔⛔ Gate on squeue's EXIT CODE, not on its output. Piping a FAILED squeue into `wc -l` yields
    # 0, which is indistinguishable from "this arm has no jobs" -- so a transient controller outage
    # would make the watchdog conclude the arm is dead and relaunch it ALONGSIDE the one still
    # running, two processes writing one last.ckpt. Observed for real on 2026-09-04, when
    # `scontrol ping` reported "Slurmctld(primary) at slurm001 is DOWN" and squeue began timing out.
    # A query we could not answer is NOT evidence of absence.
    if ! SQ=$(squeue -u "$USER" -h -o '%j %t' 2>/dev/null); then
        say "$NAME: SKIP -- squeue failed (controller unreachable); making no decision this tick"
        continue
    fi
    ALIVE=$(printf '%s\n' "$SQ" | awk -v n="$NAME" '$1 == n && ($2 == "R" || $2 == "PD")' | wc -l)
    if [ "$ALIVE" -gt 0 ]; then
        say "$NAME: OK ($ALIVE alive)"
        continue
    fi

    BREAK_F=$STATE/$NAME.breaker
    COUNT_F=$STATE/$NAME.relaunches
    [ -f "$BREAK_F" ] && { say "$NAME: DOWN but breaker tripped -- $(cat "$BREAK_F")"; continue; }

    RELAUNCHES=$(cat "$COUNT_F" 2>/dev/null || echo 0)
    if [ "$RELAUNCHES" -ge "$MAX_RELAUNCH" ]; then
        echo "relaunch cap $MAX_RELAUNCH reached" > "$BREAK_F"
        say "$NAME: DOWN, relaunch cap reached -- breaker tripped"
        continue
    fi

    # How did the last few segments end? Elapsed under MIN_HEALTHY_MIN means it never really ran.
    FAST=$(sacct -u "$USER" -n -X --name="$NAME" -S "$(date -d '1 day ago' +%F)" \
              --format=Elapsed,State 2>/dev/null | tail -"$MAX_FASTFAIL" \
           | awk -v lim="$MIN_HEALTHY_MIN" '
               { split($1, t, ":"); n = split($1, d, "-");
                 mins = (n > 1 ? d[1] * 1440 : 0) + t[length(t)-2] * 60 + t[length(t)-1];
                 if (mins < lim) c++ }
               END { print c + 0 }')
    if [ "${FAST:-0}" -ge "$MAX_FASTFAIL" ]; then
        echo "$FAST consecutive segments ended in under ${MIN_HEALTHY_MIN}m" > "$BREAK_F"
        say "$NAME: DOWN with a fast-fail loop ($FAST short segments) -- breaker tripped, NOT relaunching"
        continue
    fi

    say "$NAME: DOWN with no live job and no fast-fail loop -- relaunching (attempt $((RELAUNCHES + 1)))"
    if OUT=$(cd "$REPO" && CHAIN=4 sbatch "$SCRIPT" 2>&1); then
        echo $((RELAUNCHES + 1)) > "$COUNT_F"
        say "$NAME: relaunched -> $OUT"
    else
        say "$NAME: RELAUNCH FAILED -> $OUT"
    fi
done
