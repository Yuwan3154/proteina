#!/bin/bash
# Monitor template-search-pipeline progress.
#
# Override defaults via env vars:
#   TOTAL=143  RESULTS_FILE=...  LOG_FILE=...  PROCESS_PATTERN=template_search_pipeline.py
#
# Example:
#   watch -n 10 TOTAL=200 RESULTS_FILE=/tmp/run/template_aln/tm_scores_results.tsv \
#     ~/proteina/scripts/analysis/bad_afdb/monitor_pipeline.sh

TOTAL=${TOTAL:-143}
RESULTS_FILE=${RESULTS_FILE:-$HOME/data/bad_afdb/template_aln/tm_scores_results.tsv}
LOG_FILE=${LOG_FILE:-$HOME/data/bad_afdb/pipeline_output.log}
PROCESS_PATTERN=${PROCESS_PATTERN:-template_search_pipeline.py}

echo "=== Template Search Pipeline Monitor ==="
echo ""
echo "Total proteins in dataset: $TOTAL"
echo "Results file:              $RESULTS_FILE"
echo ""

if [ -f "$RESULTS_FILE" ]; then
    PROCESSED=$(( $(wc -l < "$RESULTS_FILE") - 1 ))  # Subtract header
    echo "Proteins processed: $PROCESSED / $TOTAL"
    echo "Progress: $(awk "BEGIN {printf \"%.1f\", ($PROCESSED/$TOTAL)*100}")%"
    echo ""
    echo "Last 5 results:"
    tail -5 "$RESULTS_FILE"
    echo ""
else
    echo "Results file not found. Pipeline may not have started yet."
fi

if pgrep -f "$PROCESS_PATTERN" > /dev/null; then
    echo "Status: ✓ Pipeline is RUNNING"
    PID=$(pgrep -f "$PROCESS_PATTERN")
    echo "PID: $PID"
else
    echo "Status: ✗ Pipeline is NOT running"
fi

echo ""
echo "Commands:"
echo "  Watch progress: watch -n 10 $0"
echo "  View log:       tail -f $LOG_FILE"
echo "  Stop pipeline:  pkill -f $PROCESS_PATTERN"
