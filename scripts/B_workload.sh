#!/usr/bin/env bash
# B. Workload mixture: short/long context x short/long response (B1), and mixed arrivals (B2).
#    SS vs SL and LS vs LL use the same documents, so they isolate response length (also covers C2).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
group_init B-workload

for p in SS SL LS LL; do
  run_exp B1-$p --combination $p -n 6
done
for p in SS-SL SS-LS SS-LL; do
  run_exp B2-$p --combination $p -n 6
done

group_summary
