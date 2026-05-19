#!/bin/bash
# Monitor get_stoichiometry.py batch run.
#
# Override defaults via env vars:
#   TOTAL=25863  RESULTS_FILE=stoichiometry_results_full.csv
#   LOG_FILE=stoichiometry_full_run.log  CACHE_DIR=pdb_info
#   PROCESS_PATTERN="python get_stoichiometry.py"

TOTAL=${TOTAL:-25863}
RESULTS_FILE=${RESULTS_FILE:-stoichiometry_results_full.csv}
LOG_FILE=${LOG_FILE:-stoichiometry_full_run.log}
CACHE_DIR=${CACHE_DIR:-pdb_info}
PROCESS_PATTERN=${PROCESS_PATTERN:-python get_stoichiometry.py}

echo "=== Stoichiometry Pipeline Monitor ==="
echo ""

if pgrep -f "$PROCESS_PATTERN" > /dev/null; then
    STATUS="🟢 RUNNING"
else
    STATUS="🔴 STOPPED"
fi
echo "Status: $STATUS"
echo ""

echo "Progress:"
if [ -f "$RESULTS_FILE" ]; then
    PROCESSED=$(wc -l < "$RESULTS_FILE")
    PROCESSED=$((PROCESSED - 1))  # Subtract header
    echo "  Processed: $PROCESSED / $TOTAL"

    if [ "$PROCESSED" -eq "$TOTAL" ]; then
        echo "  ✅ COMPLETE!"
    else
        PERCENT=$(echo "scale=2; $PROCESSED * 100 / $TOTAL" | bc)
        echo "  Progress: ${PERCENT}%"

        if [ "$STATUS" = "🟢 RUNNING" ] && [ -f "$LOG_FILE" ]; then
            START_TIME=$(stat -c %Y "$LOG_FILE")
            CURRENT_TIME=$(date +%s)
            ELAPSED=$((CURRENT_TIME - START_TIME))
            if [ "$ELAPSED" -gt 0 ] && [ "$PROCESSED" -gt 0 ]; then
                RATE=$(echo "scale=2; $PROCESSED / $ELAPSED" | bc)
                REMAINING=$((TOTAL - PROCESSED))
                if [ "$(echo "$RATE > 0" | bc)" = "1" ]; then
                    ETA_SECONDS=$(echo "scale=0; $REMAINING / $RATE" | bc)
                    ETA_MINUTES=$((ETA_SECONDS / 60))
                    echo "  Rate: ${RATE} entries/sec"
                    echo "  ETA: ~${ETA_MINUTES} minutes"
                fi
            fi
        fi
    fi
else
    echo "  No output file yet ($RESULTS_FILE)..."
fi

echo ""
echo "Cache directory:"
CACHE_COUNT=$(ls "$CACHE_DIR" 2>/dev/null | wc -l)
echo "  Cached files: $CACHE_COUNT ($CACHE_DIR)"

echo ""
if [ -f "$LOG_FILE" ]; then
    echo "Last 15 lines of log ($LOG_FILE):"
    tail -15 "$LOG_FILE"
else
    echo "No log file found ($LOG_FILE)"
fi
