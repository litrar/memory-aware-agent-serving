#!/usr/bin/env bash
# C. Context growth from input length: long-context tasks from the near (~2,000-token) and
#    overload (~2,600-token) document pools. Output-length growth is covered by B1 (SS/SL, LS/LL).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
group_init C-context

for tier in near overload; do
  run_exp C1-LL-$tier --combination LL -n 6 --long-tier $tier
done

group_summary
