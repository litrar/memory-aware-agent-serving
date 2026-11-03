#!/usr/bin/env bash
# D. Idle periods between agent steps (simulated tool delay).
#    D1: delay sweep on LL. D2: same total idle placed early vs late (late = larger retained context).
#    D3: the delay effect on light SS tasks; the LL counterparts are D1-delay0 / D1-delay5.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
group_init D-idle

for d in 0 1 2 5; do
  run_exp D1-delay$d --combination LL -n 6 --tool-delay $d
done
run_exp D2-short-then-long --combination LL -n 6 --tool-delay 0.5 5
run_exp D2-long-then-short --combination LL -n 6 --tool-delay 5 0.5
for d in 0 5; do
  run_exp D3-SS-delay$d --combination SS -n 6 --tool-delay $d
done

group_summary
