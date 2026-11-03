"""Run a fixed capacity experiment using the pilot pair or a selected custom workload.

  python task-generator/agent-pressure.py --scale 2 --check-only
  python task-generator/agent-pressure.py --scale 2 --profile LS -c 1
  python task-generator/agent-pressure.py --scale 2 --profile LS -c 2
  python task-generator/agent-pressure.py --scale 2 --profile LL -c 1
  python task-generator/agent-pressure.py --report results/pressure-demo/<run>

SS/SL use original short documents with scale=1. For LS/LL, scale=1 uses
the original long documents; scale=2/3/4 selects complete expanded articles. Scale is document tokens, not total context.
The first profile letter selects input length; the second selects response length.
SS/LS request short responses, and SL/LL request long responses.
Restart llama-server before each inference run, keeping its command unchanged.
Prompt reuse is enabled within a run. This is development calibration with two
fictional reports, not an independent evaluation dataset or exact KV-byte meter.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import tomllib

import httpx

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT))

import agent


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report(folder, show_answers=False):
    config = json.loads((folder / 'config.json').read_text())
    events = read_jsonl(folder / 'events.jsonl')
    samples = read_jsonl(folder / 'memory.jsonl')
    ends = [e for e in events if e['event'] == 'task_end']
    requests = [e for e in events if e['event'] == 'llm_end']
    duration = max((e['t'] for e in ends), default=0)
    success = sum(e['status'] == 'ok' for e in ends)
    print(f"\n--- {folder.name}: C={config['concurrency']}, n_ctx={config['server']['n_ctx']} ---")
    for e in ends:
        print(f"{e['task_id']}: {e['status']} {e.get('error') or ''}")
    print(f"Tasks ended: {len(ends)}/{config['n_tasks']}; successful: {success}; elapsed: {duration:.2f}s")
    print(f"Successful tasks/min: {success * 60 / duration if duration else 0:.3f}")
    for e in requests:
        print(f"  {e['task_id']} step {e['step']}: prompt={e.get('prompt_tokens')}, "
              f"output={e.get('completion_tokens')}, cached={e.get('cached_tokens', e.get('cache_n'))}, "
              f"finish={e.get('finish_reason')}")
        if show_answers:
            print(e.get('text', '(intermediate text was not recorded in this older run)'))
    # older memory.jsonl files call it kv_tokens_busy
    busy = [s.get('slot_context_tokens_busy', s.get('kv_tokens_busy')) for s in samples if 'error' not in s]
    busy = [x for x in busy if x is not None]
    rss = [s['rss_mb'] for s in samples if s.get('rss_mb') is not None]
    print(f"Peak busy slot context tokens: {max(busy, default=None)}")
    print(f"Peak server RSS MB: {max(rss, default=None)}")
    print('Slot tokens are not unique KV cells; RSS includes more than KV memory.')
    print('finish=length means the generation cap was reached; HTTP success alone does not establish quality.')
    print(f'Inspect every intermediate answer (text) and final summary in {folder / "events.jsonl"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scale', type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument('-c', type=int, default=1, help='task concurrency; must not exceed selected task count')
    parser.add_argument('--workload', type=Path, help='optional fixed TOML/JSONL workload')
    parser.add_argument('--target-tokens', type=int, help='select doc_target_tokens in a custom workload')
    parser.add_argument('--limit', type=int, default=0, help='take the first N selected tasks; 0 means all')
    parser.add_argument('--profile', choices=['SS', 'SL', 'LS', 'LL'], default='LS')
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--timeout', type=float, default=300, help='timeout per HTTP request, not entire task')
    parser.add_argument('--server-pid', type=int, default=0, help='override automatic server PID detection for RSS')
    parser.add_argument('--reverse', action='store_true', help='reverse the selected task arrivals for a repeated trial')
    parser.add_argument('--check-only', action='store_true', help='count tokens only; no inference or result files')
    parser.add_argument('--show-answers', action='store_true', help='print saved answers for manual factual review')
    parser.add_argument('--report', type=Path, help='summarize saved results without contacting the server')
    args = parser.parse_args()
    if args.report:
        report(args.report, args.show_answers)
        return
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    if args.workload:
        tasks = [t for t in agent.load_workload(args.workload) if t['profile'] == args.profile]
        if args.target_tokens:
            tasks = [t for t in tasks if t.get('doc_target_tokens') == args.target_tokens]
        if args.limit > 0:
            tasks = tasks[:args.limit]
        originals = {t['doc_id']: t['doc'] for t in tasks}
    else:
        if args.profile in ['SS', 'SL'] and args.scale != 1:
            parser.error('SS/SL use the original short documents; use --scale 1')
        tasks = [t for t in agent.load_workload(BASE / 'calibration/pilot-v1.toml') if t['profile'] == args.profile]
        documents = tomllib.loads((BASE / 'calibration/context-calibration.toml').read_text())['documents']
        originals = {t['doc_id']: t['doc'] for t in tasks}
        if len(tasks) != 2:
            parser.error('expected two pilot tasks for this profile')
        for task in tasks:
            if args.scale > 1:
                task['doc'] = next(d['doc'] for d in documents
                                   if d['doc_id'] == task['doc_id'] and d['scale'] == args.scale)
            task['task_id'] += f'-{args.scale}x'
    if not tasks or args.c < 1 or args.c > len(tasks):
        parser.error('require at least C tasks; concurrency must be positive')
    if args.reverse:
        tasks.reverse()

    # Tokenizer calls do not run inference. Template counting matches the actual chat payload.
    measurements = []
    with httpx.Client(base_url=args.url.rstrip('/'), timeout=30) as client:
        def token_count(text):
            response = client.post('/tokenize', json={'content': text, 'add_special': False})
            response.raise_for_status()
            return len(response.json()['tokens'])
        response = client.get('/props')
        response.raise_for_status()
        props = response.json()
        response = client.get('/slots')
        response.raise_for_status()
        initial_slots = response.json()
        print(f"Model: {props.get('model_path')}; build={props.get('build_info')}; "
              f"slots={props.get('total_slots')}; n_ctx={props.get('default_generation_settings', {}).get('n_ctx')}")
        print('Initial retained slot tokens:', sum(s.get('n_prompt_tokens', 0) for s in initial_slots))
        for task in tasks:
            base_tokens = token_count(originals[task['doc_id']])
            doc_tokens = token_count(task['doc'])
            ratio = doc_tokens / base_tokens
            if abs((doc_tokens / task.get('doc_target_tokens', doc_tokens) if args.workload else ratio / args.scale) - 1) > .05 + 1e-9:
                parser.error(f"{task['task_id']}: document length is outside the target +/-5%")
            response = client.post('/apply-template', json={
                'messages': agent.initial_messages(task), 'add_generation_prompt': True,
                'chat_template_kwargs': {'enable_thinking': False}})
            response.raise_for_status()
            prompt_tokens = token_count(response.json()['prompt'])
            measurements.append(dict(task_id=task['task_id'], base_doc_tokens=base_tokens,
                                     doc_tokens=doc_tokens, ratio=ratio, initial_prompt_tokens=prompt_tokens))
            length_label = f"target={task.get('doc_target_tokens')}" if args.workload else f"{ratio:.3f}x"
            print(f"{task['task_id']}: doc={doc_tokens} ({length_label}); initial prompt={prompt_tokens}; "
                  f"output caps={task['step_tokens']}/{task['step_tokens']}/{task['final_tokens']}")
    if args.check_only:
        return
    if any(s.get('is_processing') for s in initial_slots):
        parser.error('server has active requests; finish them before running the experiment')
    if any(s.get('n_prompt_tokens', 0) for s in initial_slots):
        parser.error('restart llama-server for a fresh KV state before this calibration run')

    label = f't{args.target_tokens}_n{len(tasks)}' if args.workload else f'{args.scale}x'
    run = f'{args.profile}_{label}_c{args.c}_{time.time_ns()}'
    base = ROOT / 'results/pressure-demo'
    base.mkdir(parents=True, exist_ok=True)
    workload = base / f'{run}.jsonl'
    workload.write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in tasks))
    # Save preflight before launching the runner so interruption cannot erase it.
    preflight = dict(target_tokens=args.target_tokens, workload=str(args.workload) if args.workload else None, scale=None if args.workload else args.scale, profile=args.profile, reverse=args.reverse, concurrency=args.c,
                     cache_prompt=True, measurements=measurements, props=props, initial_slots=initial_slots,
                     source=str(args.workload) if args.workload else ('task-generator/calibration/context-calibration.toml' if args.scale > 1 else 'task-generator/calibration/pilot-v1.toml'))
    (base / f'{run}.preflight.json').write_text(json.dumps(preflight, indent=2, ensure_ascii=False))
    command = [sys.executable, '-B', str(ROOT / 'agent.py'), '--workload', str(workload),
               '-c', str(args.c), '--url', args.url, '--timeout', str(args.timeout),
               '--out-dir', str(base), '--run-name', run, '--server-pid', str(args.server_pid)]
    print('Starting:', ' '.join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if (base / run / 'events.jsonl').exists():
        report(base / run, args.show_answers)
    print(f'Replay summary: python task-generator/agent-pressure.py --report {base / run}')
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    try:
        main()
    except httpx.HTTPError as error:
        raise SystemExit(f'llama-server request failed: {error}. Check the server URL and readiness.') from error
