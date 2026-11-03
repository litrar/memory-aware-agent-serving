#!/usr/bin/env bash
# G. Queueing under FIFO at fixed C.
#    G1: arrival order with the same tasks (head-of-line blocking of short tasks behind long ones).
#    G2: offered load, more tasks queued behind the same C.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
group_init G-queueing

for p in SS-LL LL-first SS-first; do
  run_exp G1-$p --combination $p -n 6
done
for n in 4 8 12 16; do
  run_exp G2-n$n --combination SS-LL -n $n
done

group_summary
