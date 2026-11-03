import argparse
import hashlib
import asyncio
import json
import time
import os
import random
import math
import tomllib
from functools import partial
from pathlib import Path

import httpx

import metrics
import monitor


URL = "http://127.0.0.1:8080"

T0 = time.perf_counter()
T0_EPOCH = time.time()
EVENTS = []
EVENT_FILE = None


def log(event, **kw):
    rec = {"t": round(time.perf_counter() - T0, 3), "event": event, **kw}
    EVENTS.append(rec)
    print(json.dumps(rec), flush=True)
    if EVENT_FILE:
        EVENT_FILE.write(json.dumps(rec) + "\n")
        EVENT_FILE.flush()


async def chat(client, messages, max_tokens, seed=None, cache_prompt=True):
    payload = {
        "model": "local", "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "ignore_eos": False, "cache_prompt": cache_prompt,
        "chat_template_kwargs": {"enable_thinking": False},
        "verbose": True,                      # adds __verbose.id_slot: the slot that served the request
    }
    if seed is not None:
        payload['seed'] = seed
    t0 = time.perf_counter()
    r = await client.post(f"{URL}/v1/chat/completions", json=payload)    # await: let other tasks run while waiting for the response
    if r.status_code >= 400:                  # keep the server's message (e.g. context size exceeded)
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    usage, timings = data.get("usage", {}), data.get("timings", {})
    requested = usage.get("prompt_tokens")
    if requested is None and timings.get("prompt_n") is not None:
        requested = timings["prompt_n"] + (timings.get("cache_n") or 0)
    cache_n, prompt_n = timings.get("cache_n"), timings.get("prompt_n")
    cache_reuse_ratio = round(cache_n / (cache_n + prompt_n), 3) if cache_n is not None and prompt_n is not None \
        and cache_n + prompt_n else None
    return data["choices"][0]["message"]["content"], dict(
        latency_s=round(time.perf_counter() - t0, 3),
        finish_reason=data["choices"][0].get("finish_reason"),
        prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
        cached_tokens=usage.get("prompt_tokens_details", {}).get("cached_tokens"),
        prompt_n=timings.get("prompt_n"), cache_n=timings.get("cache_n"),
        prompt_ms=timings.get("prompt_ms"), predicted_n=timings.get("predicted_n"), predicted_ms=timings.get("predicted_ms"),
        prompt_tokens_requested=requested, cache_reuse_ratio=cache_reuse_ratio,
        slot_id=data.get("__verbose", {}).get("id_slot"))


TEST_LATENCY = 0.5

async def test_llm(client, messages, max_tokens, seed=None):
    await asyncio.sleep(TEST_LATENCY)          # Simulate the LLM taking 0.5 seconds
    return "(fake reply)", dict(latency_s=TEST_LATENCY, completion_tokens=max_tokens)


async def tool(delay_s, result):
    await asyncio.sleep(delay_s)               # all tasks will block together if use time.sleep
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def article_task(corpus, article, response):
    settings = corpus['task_settings']
    profile = settings['response_profiles'][response]
    task = dict(
        schema_version=corpus['schema_version'],
        task_id=f"{article['article_id']}-{response}",
        doc_id=article['article_id'],
        doc=article['doc'],
        system_prompt=settings['system_prompt'],
        step_instructions=profile['step_instructions'],
        tool_calls=settings['tool_calls'],
        tool_delay_s=settings['tool_delay_s'],
        tool_results=settings['tool_results'],
        step_tokens=profile['step_tokens'],
        final_tokens=profile['final_tokens'],
        llm_seed=settings['llm_seed'],
        target_words=profile['target_words'],
        response_class=response,
        context_class='short' if article['tier'] == 'short' else 'long',
        profile=('S' if article['tier'] == 'short' else 'L') + ('S' if response == 'short' else 'L'),
        input_tier=article['tier'],
        doc_target_tokens=article['doc_target_tokens'],
        document_tokens=article['doc_tokens'],
        dataset_id=settings['dataset_id'],
        required_facts=article['required_facts'],
        forbidden_claims=article['forbidden_claims'],
        validation_rules={**settings['validation_rules'], 'target_words': profile['target_words']},
    )
    return task


def build_article_tasks(corpus, combination=None, long_tier='near', n=6, seed=0):
    """Expand the Phase 1 article source and its shared settings into runner tasks."""
    settings = corpus['task_settings']
    tasks = []
    combination = combination or settings['default_combination']
    require(combination in settings['combinations'], f'unknown combination: {combination}')
    preset = settings['combinations'][combination]
    pattern = preset['pattern']
    require(n > 0 and n % len(pattern) == 0, 'task count must be positive and contain complete combination cycles')
    sequence = [pattern[i % len(pattern)] for i in range(n)]
    if preset.get('grouped', False):
        sequence.sort(key=pattern.index)
    pools = {}
    for tier, letter in [('short', 'S'), (long_tier, 'L')]:
        pool = sorted((a for a in corpus['articles'] if a['tier'] == tier), key=lambda a: a['article_id'])
        # Independent pools keep the selected documents identical across ordering presets.
        random.Random(f'{seed}:{tier}').shuffle(pool)
        needed = sum(p[0] == letter for p in sequence)
        require(needed <= len(pool), f'{combination} needs {needed} unique {tier} articles; only {len(pool)} available')
        pools[tier] = iter(pool)
    for label in sequence:
        tier = 'short' if label[0] == 'S' else long_tier
        article = next(pools[tier])
        response = 'short' if label[1] == 'S' else 'long'
        task = article_task(corpus, article, response)
        tasks.append(task)
    return tasks


def reference_source(manifest_path, source):
    candidate = Path(manifest_path).resolve().parent / source['path']
    if not candidate.exists() and source.get('repo_path'):
        candidate = Path(__file__).resolve().parent / source['repo_path']
    return candidate.resolve()


def make_workload_spec(path, tasks, manifest_path):
    """Record selections and overrides; source data remains in its original file."""
    source = Path(path).resolve()
    if source.suffix == '.json':
        old = json.loads(source.read_text())
        source = reference_source(source, old['workload_spec']['source'])
    raw = source.read_bytes()
    corpus = tomllib.loads(raw.decode('utf-8')) if source.suffix == '.toml' else {}
    source_info = dict(path=os.path.relpath(source, Path(manifest_path).resolve().parent),
                       sha256=hashlib.sha256(raw).hexdigest(),
                       dataset_id=corpus.get('dataset_id'), schema_version=corpus.get('schema_version'))
    try:
        source_info['repo_path'] = str(source.relative_to(Path(__file__).resolve().parent))
    except ValueError:
        pass
    refs = []
    if 'articles' in corpus:
        articles = {a['article_id']: a for a in corpus['articles']}
        for task in tasks:
            base = article_task(corpus, articles[task['doc_id']], task['response_class'])
            ref = dict(task_id=task['task_id'], doc_id=task['doc_id'], response_class=task['response_class'])
            overrides = {k: v for k, v in task.items() if k not in base or v != base[k]}
            if overrides:
                ref['overrides'] = overrides
            refs.append(ref)
    else:
        base_tasks = {t['task_id']: t for t in load_workload(source)}
        for task in tasks:
            base = base_tasks[task['task_id']]
            ref = dict(task_id=task['task_id'])
            overrides = {k: v for k, v in task.items() if k not in base or v != base[k]}
            if overrides:
                ref['overrides'] = overrides
            refs.append(ref)
    return dict(schema_version=1, source=source_info, tasks=refs)


def manifest_tasks(path):
    manifest = json.loads(Path(path).read_text())
    return load_workload_from_spec(manifest['workload_spec'], path)


def load_workload_from_spec(spec, path):
    require(spec['schema_version'] == 1, 'unsupported workload reference schema')
    source = reference_source(path, spec['source'])
    raw = source.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == spec['source']['sha256'],
            f'workload source SHA-256 mismatch: {source}; restore the original source before replay/evaluation')
    corpus = tomllib.loads(raw.decode('utf-8')) if source.suffix == '.toml' else {}
    articles = {a['article_id']: a for a in corpus.get('articles', [])}
    fixed = None if articles else {t['task_id']: t for t in load_workload(source)}
    tasks = []
    for ref in spec['tasks']:
        if articles:
            task = article_task(corpus, articles[ref['doc_id']], ref['response_class'])
        else:
            task = dict(fixed[ref['task_id']])
        task.update(ref.get('overrides', {}))
        require(task['task_id'] == ref['task_id'], 'manifest task_id does not match resolved task')
        tasks.append(task)
    return tasks


def load_workload(path, combination=None, long_tier='near', n=6, seed=0):
    tasks, ids = [], set()
    source = Path(path)
    text = source.read_text()
    if source.suffix == '.json':
        require(combination is None and long_tier == 'near', 'manifest selection is fixed')
        entries = manifest_tasks(source)
    elif source.suffix == '.toml':
        data = tomllib.loads(text)
        if 'articles' in data:
            entries = build_article_tasks(data, combination, long_tier, n, seed)
        else:
            require(combination is None and long_tier == 'near', 'combination options require an article-source TOML')
            entries = data['tasks']
    else:
        require(combination is None and long_tier == 'near', 'combination options require an article-source TOML')
        entries = text.splitlines()
    for line_no, entry in enumerate(entries, 1):
        if isinstance(entry, str) and not entry.strip():
            continue
        try:
            t = json.loads(entry) if isinstance(entry, str) else entry
            require(t['schema_version'] == 1, 'unsupported schema_version')
            for name in ['task_id', 'doc_id', 'doc', 'system_prompt']:
                require(isinstance(t[name], str) and t[name].strip(), name)
            require(t['task_id'] not in ids, 'duplicate task_id')
            require(type(t['tool_calls']) is int and t['tool_calls'] >= 0, 'tool_calls')
            # one delay for every tool, or a list with one delay per tool call
            delays = t['tool_delay_s'] if isinstance(t['tool_delay_s'], list) else [t['tool_delay_s']] * t['tool_calls']
            require(len(delays) == t['tool_calls'], 'tool_delay_s list must have one value per tool call')
            require(all(isinstance(d, (int, float)) and math.isfinite(d) and d >= 0 for d in delays), 'tool_delay_s')
            for name in ['step_tokens', 'final_tokens']:
                require(type(t[name]) is int and t[name] > 0, name)
            require(type(t['llm_seed']) is int, 'llm_seed')
            for name, n in [('step_instructions', t['tool_calls'] + 1), ('tool_results', t['tool_calls'])]:
                require(isinstance(t[name], list) and len(t[name]) == n, name)
                require(all(isinstance(s, str) and s.strip() for s in t[name]), name)
        except (ValueError, KeyError, AssertionError, TypeError, AttributeError) as e:
            raise ValueError(f'{path}:{line_no}: invalid task: {e}') from e
        tasks.append(t)
        ids.add(t['task_id'])
    if not tasks:
        raise ValueError('workload must not be empty')
    return tasks


def apply_tool_delay(tasks, delays):
    """--tool-delay override: one value for every tool call, or one value per tool call.

    The changed tool_delay_s is recorded as a task override in the manifest, so replay keeps it.
    """
    require(all(math.isfinite(d) and d >= 0 for d in delays), 'tool-delay must be finite and nonnegative')
    for t in tasks:
        require(len(delays) in (1, t['tool_calls']),
                f"--tool-delay needs 1 or {t['tool_calls']} values for task {t['task_id']}")
        t['tool_delay_s'] = delays[0] if len(delays) == 1 else list(delays)
    return tasks


def initial_messages(task):
    # required_facts stay outside the model input.
    return [
        {'role': 'system', 'content': task['system_prompt']},
        {'role': 'user', 'content': task['step_instructions'][0] + '\n\nDocument:\n' + task['doc']},
    ]


async def run_agent(llm, client, task):
    tid = task["task_id"]
    messages = initial_messages(task)
    n_steps = task["tool_calls"] + 1
    for step in range(n_steps):
        last = step == n_steps - 1
        if step:
            messages.append({'role': 'user', 'content': task['step_instructions'][step]})
        log("llm_start", task_id=tid, step=step)
        max_tokens = task["final_tokens"] if last else task["step_tokens"]
        text, info = await llm(client, messages, max_tokens, seed=task['llm_seed'])
        log("llm_end", task_id=tid, step=step, text=text, **info)
        messages.append({"role": "assistant", "content": text})
        if not last:
            call_id = f"call_{tid}_{step}"
            messages[-1]["tool_calls"] = [{"id": call_id, "type": "function", "function": {
                "name": "load_checklist", "arguments": json.dumps({"step": step})}}]
            delay = task["tool_delay_s"]
            delay = delay[step] if isinstance(delay, list) else delay
            log("tool_start", task_id=tid, step=step, delay_s=delay)
            result = await tool(delay, task['tool_results'][step])
            log("tool_end", task_id=tid, step=step)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
    return text


async def producer(queue, tasks, n_workers):
    for task in tasks:
        log("task_arrive", task_id=task["task_id"])
        await queue.put(task)
    for _ in range(n_workers):
        await queue.put(None)  # signal to workers to exit

async def worker(worker_id, queue, llm, client):
    while True:
        task = await queue.get()
        if task is None:
            return
        log("task_start", task_id=task["task_id"], worker=worker_id, doc_id=task["doc_id"])
        summary = None
        try:
            summary = await run_agent(llm, client, task)
            status, err = "ok", None
        except httpx.TimeoutException:
            status, err = "timeout", "request timeout"
        except Exception as e:                  # server error, connection dropped, etc.
            status, err = "error", repr(e)[:300]
        log("task_end", task_id=task["task_id"], worker=worker_id, status=status, error=err, summary=summary)


async def main(args):
    global EVENT_FILE, URL, T0, T0_EPOCH
    T0, T0_EPOCH = time.perf_counter(), time.time()
    EVENTS.clear()
    URL = args.url.rstrip('/')
    if args.c < 1 or args.n < 1 or args.timeout <= 0:
        raise ValueError('concurrency, task count and timeout must be positive')
    require(math.isfinite(args.mem_interval) and args.mem_interval >= 0,
            'mem-interval must be finite and nonnegative')
    tasks = load_workload(args.workload, args.combination, args.long_tier, args.n, args.seed)
    if args.tool_delay is not None:
        apply_tool_delay(tasks, args.tool_delay)

    llm = test_llm if args.test else partial(chat, cache_prompt=args.cache_prompt == 'on')
    run = args.run_name or f"c{args.c}_n{len(tasks)}_{time.time_ns()}" + ("_test" if args.test else "")
    out_dir = os.path.join(args.out_dir, run)
    # Refuse to silently overwrite failed runs or earlier pilot outputs.
    os.makedirs(out_dir, exist_ok=False)
    workload_manifest = Path(args.workload).resolve()
    # A manifest input is reused as-is unless --tool-delay changed its tasks; then record a local one.
    if workload_manifest.suffix != '.json' or args.tool_delay is not None:
        workload_manifest = Path(out_dir).resolve() / 'experiment.json'
        spec = make_workload_spec(args.workload, tasks, workload_manifest)
        require(load_workload_from_spec(spec, workload_manifest) == tasks, 'source changed during selection')
        workload_manifest.write_text(json.dumps(dict(workload_spec=spec), indent=2), encoding='utf-8')
    queue = asyncio.Queue()
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        props = {}
        if not args.test:
            try:
                props = (await client.get(f"{URL}/props", timeout=5)).json()
            except Exception:
                pass
        props.pop('chat_template', None)
        proc = None if args.test else monitor.find_server_process(URL, args.server_pid)
        pid = proc.pid if proc else None
        if not args.test and proc is None:
            print('warning: server PID unavailable; process memory will be missing')
        if props.get('total_slots', args.c) < args.c:
            print('warning: server slots < C; requests may queue inside the server')
        server = dict(total_slots=props.get("total_slots"), build=props.get("build_info"), pid=pid,
                      n_ctx=props.get("default_generation_settings", {}).get("n_ctx"))
        config = dict(run=run, concurrency=args.c, n_tasks=len(tasks), test=args.test, url=URL, model=props.get("model_path"),
                      server=server, mem_interval=args.mem_interval, t0_epoch=T0_EPOCH,
                      task_ids=[t['task_id'] for t in tasks],
                      workload_manifest=os.path.relpath(workload_manifest, Path(out_dir).resolve()))
        config.update(schema_version=3, runner='agent.py', scheduler='task_fifo',
                      host=None if args.test else monitor.host_info(),
                      server_process=monitor.describe_process(proc), server_props=props)
        config.update(workload=args.workload, combination=args.combination, long_tier=args.long_tier,
                      selection_seed=args.seed, tool_delay_override=args.tool_delay,
                      arrival='all_at_once_file_order', timeout_s=args.timeout,
                      cache_prompt=args.cache_prompt == 'on', temperature=0, ignore_eos=False, enable_thinking=False)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        mon = monitor.MemoryMonitor(os.path.join(out_dir, 'memory.jsonl'),
                                    lambda: time.perf_counter() - T0, args.mem_interval,
                                    proc, None if args.test else client, URL,
                                    scrape_every_s=args.mem_interval)
        with open(os.path.join(out_dir, "events.jsonl"), "w", encoding="utf-8") as EVENT_FILE:
            try:
                if not args.test and args.mem_interval > 0:
                    await mon.start()
                else:
                    Path(out_dir, 'memory.jsonl').touch()
                workers = [asyncio.create_task(worker(w, queue, llm, client)) for w in range(args.c)]
                await producer(queue, tasks, args.c)
                await asyncio.gather(*workers)
            finally:
                if mon._task is not None:
                    await mon.stop()
                EVENT_FILE = None
    check(args.c)
    metrics.report(out_dir)
    print(f"\nResults written to {out_dir}/events.jsonl, memory.jsonl, metrics.json and config.json")

def check(c):
    """Print per-task times, and check FIFO order and the concurrency limit."""
    arrive = {e["task_id"]: e["t"] for e in EVENTS if e["event"] == "task_arrive"}
    start = {e["task_id"]: e["t"] for e in EVENTS if e["event"] == "task_start"}
    end = {e["task_id"]: e["t"] for e in EVENTS if e["event"] == "task_end"}
    print("\ntask  queue_s  service_s  completion_s")
    for tid in arrive:                          # arrival order (== start order under FIFO)
        print(f"{tid:>4}  {start[tid] - arrive[tid]:7.2f}  {end[tid] - start[tid]:9.2f}  {end[tid] - arrive[tid]:12.2f}")
    order_start = sorted(start, key=start.get)
    print("\nFIFO (start order == arrival order):", order_start == sorted(arrive, key=arrive.get))
    running, peak = 0, 0
    for e in EVENTS:                            # events are already sorted by time
        if e["event"] == "task_start":
            running += 1
        elif e["event"] == "task_end":
            running -= 1
        peak = max(peak, running)
    print(f"Peak concurrent tasks {peak} (limit C={c}):", peak <= c)
    print(f"Total time {max(end.values()):.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=int, default=2, help="concurrency workers")
    parser.add_argument("-n", type=int, default=6, help="number of article-source tasks (fixed task files keep their own count)")
    parser.add_argument("--test", action="store_true", help="test")
    parser.add_argument('--workload', required=True,
                        help='experiment manifest JSON, fixed TOML/JSONL tasks, or article-source TOML')
    parser.add_argument('--combination', help='article-source preset (default: source task_settings.default_combination)')
    parser.add_argument('--long-tier', choices=['near', 'overload'], default='near',
                        help='article pool for L context: near (~2000 tokens) or overload (~2600)')
    parser.add_argument('--url', default=URL, help='llama.cpp server URL')
    parser.add_argument('--timeout', type=float, default=120, help='HTTP timeout in seconds')
    parser.add_argument('--mem-interval', type=float, default=0.5,
                        help='memory sampling interval in seconds, 0 = off')
    parser.add_argument('--server-pid', type=int, default=0,
                        help='llama-server PID (default: detect the process listening on --url)')
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--run-name", default="", help="default: c<C>_n<N>_<time_ns>")
    parser.add_argument("--seed", type=int, default=0, help="random seed for choosing articles")
    parser.add_argument('--cache-prompt', choices=['on', 'off'], default='on',
                        help='reuse matching prompt prefixes (off recomputes prompts; KV allocation is unchanged)')
    parser.add_argument("--tool-delay", type=float, nargs="+", metavar="SEC",
                        help="override tool delay: one value for every tool, or one value per tool call "
                             "(e.g. --tool-delay 0.5 3); default: tool_delay_s from the workload")
    asyncio.run(main(parser.parse_args()))
