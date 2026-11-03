#!/usr/bin/env bash
# Runs groups B, C, E, F and G back to back with one shared timestamp, then checks that every group has
# its summary table (summary_all.csv) and plots (curves_group.png, timelines/), redoing them where missing.
# The whole console output is also kept in results/run_BCEFG-<TS>.log. Long on CPU: run inside tmux.
#   bash script/run_BCEFG.sh
#   TS=20261029-1200 bash script/run_BCEFG.sh     # resume: finished experiments are skipped
#   TEST=1 bash script/run_BCEFG.sh               # fake LLM, no server: checks the whole pipeline quickly
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export TS

STEPS=(B_workload C_context E_retained F_kv_capacity G_queueing)
DIRS=(B-workload C-context E-retained F-kv-capacity G-queueing)     # group_init names of the steps above
LOG="$RESULTS/run_BCEFG-$TS.log"
mkdir -p "$RESULTS"

{
  echo "== run_BCEFG TS=$TS start $(date '+%F %T')"
  for s in "${STEPS[@]}"; do
    echo "== $s start $(date '+%F %T')"
    bash "$REPO/script/$s.sh"
  done

  [[ $DRY_RUN == 1 ]] && exit 0
  echo "== tables and plots"
  for name in "${DIRS[@]}"; do
    GROUP_DIR="$RESULTS/$name-$TS"
    FAILED="$GROUP_DIR/failed.txt"
    [[ -d $GROUP_DIR ]] || { echo "$name: no results"; continue; }
    if [[ ! -f $GROUP_DIR/summary_all.csv || ! -f $GROUP_DIR/curves_group.png ]]; then
      echo "$name: table or plot missing, redoing"
      group_summary
    fi
    runs=$(( $(wc -l < "$GROUP_DIR/summary_all.csv" 2>/dev/null || echo 1) - 1 ))
    plot=$([[ -f $GROUP_DIR/curves_group.png ]] && echo yes || echo MISSING)
    timelines=$(ls "$GROUP_DIR/timelines" 2>/dev/null | wc -l | tr -d ' ')
    failed=$([[ -s $FAILED ]] && paste -sd, "$FAILED" || echo none)
    echo "$name: $runs runs in summary_all.csv, curves_group.png $plot, $timelines timelines, failed: $failed"
  done
  echo "== run_BCEFG TS=$TS done $(date '+%F %T')"
} 2>&1 | tee -a "$LOG"
