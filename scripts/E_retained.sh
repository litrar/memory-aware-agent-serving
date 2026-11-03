#!/usr/bin/env bash
# E. Retained contexts: server slots = C (no spare slots keep finished tasks' caches) vs 8 slots
#    (idle slots retain caches inside the shared KV pool), with and without idle time between steps.
#    Prefix reuse (cache_hit_ratio first vs later steps) is read from every run's metrics.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
group_init E-retained

for slots in "$C" 8; do
  for d in 0 5; do
    run_exp E1-slots$slots-delay$d --combination LL -n 6 --slots $slots --tool-delay $d
  done
done

group_summary
