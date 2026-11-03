#!/usr/bin/env bash
# Shared settings and helpers for fixed-concurrency experiment groups.
# Sourced by each group script; not meant to be run on its own.
#
# Every setting can be overridden from the environment, e.g.
#   THREADS=8 bash scripts/H_conditions.sh
#   TS=20261029-1200 bash scripts/H_conditions.sh   # resume a group: finished experiments are skipped
#   TEST=1 N=1 C=1 SHORT_DELAY=0 LONG_DELAY=0.001 bash scripts/H_conditions.sh
#   DRY_RUN=1 bash scripts/H_conditions.sh   # print the experiment.py commands only
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"    # scripts live in <repo>/scripts/
cd "$REPO"

LLAMA_SERVER="${LLAMA_SERVER:-/home/ec2-user/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/home/ec2-user/model/Qwen3-4B-Q8_0.gguf}"
PYTHON="${PYTHON:-python}"
C="${C:-2}"
SEED="${SEED:-0}"
TIMEOUT="${TIMEOUT:-1200}"
MEM_INTERVAL="${MEM_INTERVAL:-0.5}"
CTX_SIZE="${CTX_SIZE:-8192}"
SLOTS="${SLOTS:-8}"
THREADS="${THREADS:-4}"
PORT="${PORT:-8080}"
RESULTS="${RESULTS:-results}"
TS="${TS:-$(date +%Y%m%d-%H%M%S)}"
TEST="${TEST:-0}"
DRY_RUN="${DRY_RUN:-0}"
NPROC="$(getconf _NPROCESSORS_ONLN)"
# Verbosity 4 makes llama-server log its model/KV/compute buffer sizes, which metrics.py reads.
export LLAMA_ARG_LOG_VERBOSITY="${LLAMA_ARG_LOG_VERBOSITY:-4}"

# Baseline condition. Group scripts append the one flag they vary; argparse keeps the last value.
COMMON=(-C "$C" --seed "$SEED" --timeout "$TIMEOUT" --mem-interval "$MEM_INTERVAL"
        --ctx-size "$CTX_SIZE" --slots "$SLOTS" --threads "$THREADS" --port "$PORT"
        --server-binary "$LLAMA_SERVER" --model "$MODEL")
[[ $TEST == 1 ]] && COMMON+=(--test)


# group_init NAME: every experiment of the group goes under results/NAME-<TS>/
group_init() {
  GROUP_DIR="$RESULTS/$1-$TS"
  FAILED="$GROUP_DIR/failed.txt"
  [[ $DRY_RUN == 1 ]] || mkdir -p "$GROUP_DIR"
  echo "== $1 -> $GROUP_DIR"
  preflight
}


preflight() {
  [[ $TEST == 1 || $DRY_RUN == 1 ]] && return 0
  local ok=1
  [[ -x $LLAMA_SERVER ]] || { echo "missing llama-server: $LLAMA_SERVER (set LLAMA_SERVER=...)"; ok=0; }
  [[ -f $MODEL ]] || { echo "missing model: $MODEL (set MODEL=...)"; ok=0; }
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    echo "port $PORT is in use; stop the other llama-server first"; ok=0
  fi
  (( THREADS <= NPROC )) || echo "warning: THREADS=$THREADS > $NPROC online CPUs"
  (( ok )) || exit 1
}


# run_exp NAME [experiment.py flags...]: one experiment.py call, logged to NAME.terminal.log.
# A finished experiment (summary.csv present) is skipped, so rerunning with the same TS resumes the
# group; an interrupted one is moved aside. A failure is recorded and the group continues.
run_exp() {
  local name=$1; shift
  local out="$GROUP_DIR/$name"
  if [[ $DRY_RUN == 1 ]]; then
    printf '%q ' "$PYTHON" -B experiment.py "${COMMON[@]}" "$@" --no-plot --out-dir "$out"; echo
    return 0
  fi
  if [[ -f $out/summary.csv ]]; then echo "-- skip $name (already done)"; return 0; fi
  if [[ -e $out ]]; then
    local aside="$out.partial-$(date +%H%M%S)"
    mv "$out" "$aside"
    [[ -f $out.terminal.log ]] && mv "$out.terminal.log" "$aside.terminal.log"
  fi
  echo "-- $name: $*"
  if ! "$PYTHON" -B experiment.py "${COMMON[@]}" "$@" --no-plot --out-dir "$out" 2>&1 | tee "$out.terminal.log"; then
    echo "$name" >> "$FAILED"
    echo "!! $name failed; see $out.terminal.log and $out/c*.server.log"
  fi
  return 0
}


# group_summary: one CSV row per finished run of the group, labelled with its experiment name and
# server settings (every run folder is named c<C>_n<N>, so the run name alone is ambiguous).
group_summary() {
  [[ $DRY_RUN == 1 ]] && return 0
  "$PYTHON" -B - "$GROUP_DIR" <<'EOF' || echo "!! summary failed"
import json, sys
from pathlib import Path
import metrics

group = Path(sys.argv[1])
results = []
for run in sorted(group.glob('*/c*_n*/config.json')):
    run = run.parent
    exp = run.parent
    if '.partial-' in exp.name or not (exp / 'summary.csv').exists() or not (run / 'events.jsonl').exists():
        continue
    m = metrics.compute(str(run))
    cmd = (json.loads((exp / 'experiment.json').read_text()).get('server_command') or [])
    flag = lambda f: cmd[cmd.index(f) + 1] if f in cmd else None
    m.update(run=f'{exp.name}/{run.name}', experiment=exp.name,
             ctx_size=flag('--ctx-size'), slots=flag('-np'), threads=flag('-t'))
    results.append(m)
if results:
    metrics.write_summary(results, group / 'summary_all.csv')
else:
    print(f'no finished runs in {group}')
EOF
  "$PYTHON" -B scripts/plot_group.py "$GROUP_DIR" || echo "!! plot failed"
  if [[ -s $FAILED ]]; then echo "failed experiments:"; cat "$FAILED"; fi
  return 0
}
