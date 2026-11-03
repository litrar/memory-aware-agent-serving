#!/usr/bin/env bash
# H. SS/SL/LS/LL x two tool delays x prompt-prefix reuse on/off: 16 runs at fixed C.
# Reuses common.sh's execution/resume/logging and plot_group.py's charts.
#   bash script/H_conditions.sh
#   N=4 SHORT_DELAY=0 LONG_DELAY=5 C=2 bash script/H_conditions.sh
#   DRY_RUN=1 bash script/H_conditions.sh
#   TEST=1 N=1 C=1 SHORT_DELAY=0 LONG_DELAY=0.001 bash script/H_conditions.sh
# Reuse TS to resume with identical settings; changed settings require a new TS.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
N="${N:-6}"
SHORT_DELAY="${SHORT_DELAY:-0}"
LONG_DELAY="${LONG_DELAY:-5}"
LONG_TIER="${LONG_TIER:-near}"
group_init H-conditions

# Validate the whole matrix up front and prevent resuming a different configuration.
"$PYTHON" -B - "$GROUP_DIR" "$N" "$SHORT_DELAY" "$LONG_DELAY" "$LONG_TIER" "$DRY_RUN" "${COMMON[@]}" <<'PY' || exit 1
import hashlib, itertools, json, math, sys
from pathlib import Path
import agent, experiment

group, n, short, long, tier, dry = sys.argv[1:7]
n, delays, common = int(n), [float(short), float(long)], sys.argv[7:]
value = lambda flag: common[common.index(flag) + 1]
c, slots, threads = [int(value(f)) for f in ('-C', '--slots', '--threads')]
if not 0 < c <= n or slots < c or threads <= 0 or int(value('--ctx-size')) <= 0:
    raise ValueError('require 0 < C <= N, slots >= C, and positive threads/context capacity')
if not all(math.isfinite(d) for d in delays) or not 0 <= delays[0] < delays[1]:
    raise ValueError('require finite delays with 0 <= SHORT_DELAY < LONG_DELAY')
source = Path('task-generator/source/phase1-articles.toml')
for p in experiment.PROFILES:
    agent.load_workload(source, p, tier, n, int(value('--seed')))
conditions = [dict(name=f'H1-{p}-delay{d}-cache{cache}', combination=p,
                   tool_delay_s=float(d), cache_prompt=cache == 'on')
              for p, d, cache in itertools.product(experiment.PROFILES, (short, long), ('on', 'off'))]
plan = dict(n_tasks=n, common_args=common, long_tier=tier, conditions=conditions,
            workload_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), git=experiment.git_state())
print(f'16 conditions at C={c}; {n} total tasks per run', flush=True)
if dry != '1':
    path = Path(group) / 'matrix.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('matrix settings/source/code changed; use a new TS')
    elif any(Path(group).iterdir()):
        raise ValueError('existing results without a matrix manifest; use a new TS')
    else:
        path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
PY

for p in SS SL LS LL; do
  for d in "$SHORT_DELAY" "$LONG_DELAY"; do
    for cache in on off; do
      run_exp "H1-$p-delay$d-cache$cache" --combination "$p" -n "$N" \
        --long-tier "$LONG_TIER" --tool-delay "$d" --cache-prompt "$cache"
    done
  done
done
group_summary
[[ $DRY_RUN == 1 ]] && exit 0

# common.sh continues after individual failures. Do not claim a complete matrix if outputs are missing.
"$PYTHON" -B - "$GROUP_DIR" <<'PY'
import csv, json, sys
from pathlib import Path
group = Path(sys.argv[1])
names = {c['name'] for c in json.loads((group / 'matrix.json').read_text())['conditions']}
missing = [name for name in sorted(names) if not (group / name / 'summary.csv').is_file()]
if missing:
    raise SystemExit(f'Incomplete matrix: {missing}; inspect terminal logs/failed.txt, then resume with the same TS')
with (group / 'summary_all.csv').open() as f:
    rows = list(csv.DictReader(f))
if len(rows) != 16 or {r['experiment'] for r in rows} != names:
    raise SystemExit('summary_all.csv does not contain exactly the 16 conditions')
if not (group / 'curves_group.png').is_file() or any(
        not (group / 'timelines' / f'{name}.png').is_file() for name in names):
    raise SystemExit('plots missing; rerun the group with the same TS to regenerate them')
print(f'Matrix complete: {group}/summary_all.csv, curves_group.png and 16 timelines')
PY
