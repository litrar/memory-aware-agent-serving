#!/usr/bin/env bash
# F. Memory pressure: total shared KV capacity (--ctx-size) for long-context tasks at fixed C.
#    Shows where context-capacity failures start and how RSS / KV buffers scale with capacity.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
group_init F-kv-capacity

for ctx in 4096 6144 8192 16384; do
  run_exp F1-ctx$ctx --combination LL -n 6 --ctx-size $ctx
done

group_summary
