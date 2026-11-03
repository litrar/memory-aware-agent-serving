"""Phase 1 FIFO concurrency sweep: the same workload at each C, with a fixed shared KV capacity.

Each C starts a fresh llama-server; model loading is excluded from task makespan.
No warm-up request is issued, preserving empty slot state at the start of each run.
Each run keeps agent.py's events.jsonl, monitor.py's memory.jsonl, the server's log and metrics.json. The sweep
adds summary.csv (one row per C), curves.png, and in experiment.json the best fixed-concurrency baseline that
Phase 2 policies are compared against.
Use --test for a fake-LLM orchestration check without a server.
"""
import argparse
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import agent
import metrics

HERE = Path(__file__).resolve().parent
BIN = HERE / 'third_party/llama.cpp/build/bin'
LLAMA_SERVER = BIN / 'llama-server'
PROFILES = ['SS', 'SL', 'LS', 'LL']             # (context, response) short/long, in a fixed color order


def client_host(args):
    return args.client_host or ('127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host)


def server_cmd(args):
    return [str(args.server_binary), '-m', str(args.model), '--ctx-size', str(args.ctx_size),
            '-np', str(args.slots), '--kv-unified', '-ngl', '0', '-t', str(args.threads),
            '-tb', str(args.threads), '-ctk', 'f16', '-ctv', 'f16', '--fit', 'off',
            '--jinja', '--reasoning', 'off', '--no-context-shift', '--metrics', '--no-webui',
            '--host', args.host, '--port', str(args.port)]


def stop_server(proc, log):
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    finally:
        log.close()


def start_server(args, log_path):
    host = client_host(args)
    with socket.socket() as sock:
        if sock.connect_ex((host, args.port)) == 0:
            raise RuntimeError(f'port {args.port} is already in use')
    log = log_path.open('w', encoding='utf-8')
    env = dict(os.environ)
    binary_dir = str(args.server_binary.resolve().parent)    # shared llama.cpp builds keep libllama/libggml here
    library_path = 'DYLD_LIBRARY_PATH' if platform.system() == 'Darwin' else 'LD_LIBRARY_PATH'
    env[library_path] = binary_dir + os.pathsep + env.get(library_path, '')
    try:
        proc = subprocess.Popen(server_cmd(args), stdout=log, stderr=subprocess.STDOUT, env=env)
    except BaseException:
        log.close()
        raise
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f'server exited: see {log_path}')
            try:
                with urllib.request.urlopen(f'http://{host}:{args.port}/health', timeout=2) as response:
                    if json.load(response).get('status') == 'ok':
                        return proc, log
            except OSError:
                pass
            time.sleep(0.5)
        raise RuntimeError(f'server readiness timeout: see {log_path}')
    except BaseException:
        stop_server(proc, log)
        raise


def git_state():
    """The commit the sweep ran from, and whether tracked files had uncommitted changes."""
    def git(*a):
        return subprocess.run(['git', *a], cwd=HERE, capture_output=True, text=True, timeout=60).stdout.strip()
    try:
        return dict(commit=git('rev-parse', 'HEAD') or None, dirty=bool(git('status', '--porcelain', '--untracked-files=no')))
    except (OSError, subprocess.SubprocessError):
        return None


def best_fixed(results):
    """The tuned fixed-concurrency baseline: fewest failed tasks first, then the most completed tasks per minute,
    then the lower p95 completion time."""
    def key(m):
        h = m['headline']
        return (h['failure_rate'] or 0, -(h['tasks_per_min'] or 0), h['p95_completion_s'] or math.inf)
    m = min(results, key=key)
    h = m['headline']
    return dict(concurrency=m['concurrency'], run=m['run'], tasks_per_min=h['tasks_per_min'],
                failure_rate=h['failure_rate'], p95_completion_s=h['p95_completion_s'],
                rule='min failure rate, then max tasks/min, then min p95 completion time')


def print_sweep(results, best):
    """One line per C with the numbers the project report asks for, plus the checks that a faster run did not
    drop tasks or shorten outputs."""
    def f(x, fmt='.1f'):
        return 'n/a' if x is None else format(x, fmt)

    head = (f"{'C':>3} {'done':>7} {'fail':>5} {'tasks/min':>9} {'p50 s':>7} {'p95 s':>7} {'queue p50':>9} "
            f"{'RSS MiB':>8} {'anon MiB':>8} {'slot ctx':>8} {'busy':>5} {'KV full':>7} {'out/task':>8} {'cut':>5}")
    print('\n== sweep\n' + head)
    for m in sorted(results, key=lambda m: m['concurrency']):
        h, t, fl = m['headline'], m['tasks'], m.get('failures') or {}
        kv_full = (None if fl.get('server_kv_full_retries') is None else
                   fl['server_kv_full_purges'] + fl['server_kv_full_retries'] + fl['server_kv_full_fatal'])
        print(f"{m['concurrency']:>3} {t['ok']:>3}/{t['n']:<3} {metrics.fmt_pct(h['failure_rate']):>5} "
              f"{f(h['tasks_per_min'], '.3f'):>9} {f(h['median_completion_s']):>7} {f(h['p95_completion_s']):>7} "
              f"{f((m['queue_s'] or {}).get('p50')):>9} {f(h['peak_server_rss_mib'], '.0f'):>8} "
              f"{f(h['peak_server_anon_mib'], '.0f'):>8} {f(h['slot_context_tokens_total_peak'], '.0f'):>8} "
              f"{f(h['effective_concurrency'], '.2f'):>5} {f(kv_full, 'd'):>7} {f(h['output_tokens_per_task'], '.0f'):>8} "
              f"{metrics.fmt_pct(h['length_cut_frac']):>5}"
              + ('   <- best fixed C' if m['run'] == best['run'] else ''))
    print('done = completed/offered tasks; slot ctx = peak /slots context tokens (a token count, not KV use); busy = mean '
          'server busy slots; KV full = server.log purges + retries + errors; out/task = output tokens per completed '
          'task; cut = requests stopped at max_tokens')


def summarize(results, out, no_plot):
    metrics.write_summary(results, out / 'summary.csv')
    best = best_fixed(results)
    print_sweep(results, best)
    manifest = out / 'experiment.json'
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        data['best_fixed_concurrency'] = best
        manifest.write_text(json.dumps(data, indent=2), encoding='utf-8')
    if not no_plot:
        plot(results, out / 'curves.png', best)


SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# categorical slots 1-4 in their fixed order (validated as a set for adjacent pairs; slot 4 is under 3:1
# contrast on SURFACE, so lines() labels every line directly). A 5th series needs a new chart, not a new color.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def plot(results, path, best=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = sorted(results, key=lambda m: m["concurrency"])
    cs = [m["concurrency"] for m in results]

    def get(m, *keys, scale=1.0):
        for k in keys:
            m = (m or {}).get(k)
        return None if m is None else m * scale

    def each(*keys, scale=1.0):
        return [get(m, *keys, scale=scale) for m in results]

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2,
                         "ytick.color": INK_2, "axes.edgecolor": GRID, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.titlelocation": "left"})
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for ax in axes.flat:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xticks(cs)
    for ax in axes[-1]:
        ax.set_xlabel("concurrency C (tasks admitted at once)")

    def legend_above(ax):
        """Legend in one row between the title and the plot, so it never covers data. Left out of tight_layout,
        which would otherwise narrow every panel to make room for the widest legend."""
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
                  ncol=max(1, len(ax.get_legend_handles_labels()[1])), fontsize=8,
                  borderaxespad=0.2, handlelength=1.5).set_in_layout(False)

    def lines(ax, title, series, unit_fmt, ymax=None, ref=None):
        """series: [(label, values)]; for >= 2 series a legend plus a direct label at each line's right end.
        ref: (label, y) drawn as a dashed neutral reference line, e.g. a capacity."""
        if len(series) > len(SERIES):
            raise ValueError(f"{title}: {len(series)} series but only {len(SERIES)} colors; split the panel")
        ends, peak = [], 0
        for i, (label, ys) in enumerate(series):
            pts = [(c, y) for c, y in zip(cs, ys) if y is not None]
            if pts:
                ax.plot(*zip(*pts), color=SERIES[i], linewidth=2, marker="o", markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
                ends.append((pts[-1][1], label, pts[-1]))
                peak = max([peak] + [y for _, y in pts])
        if ref and ref[1] is not None:
            peak = max(peak, ref[1])
        top = ymax or (peak * 1.18 if peak else 1)
        ax.set_ylim(0, top)
        ax.set_xlim(min(cs) - 0.3, max(cs) + 1.1)
        ax.set_title(title, pad=22)          # same pad everywhere, so titles line up across panels
        if ref and ref[1] is not None:
            ax.axhline(ref[1], color=INK_2, linestyle="--", linewidth=1)
            ax.annotate(f"{ref[0]} {unit_fmt(ref[1])}", (min(cs) - 0.3, ref[1]), xytext=(2, 3),
                        textcoords="offset points", color=INK_2, fontsize=8).set_in_layout(False)
        if len(series) > 1:
            legend_above(ax)
            placed = []                      # nudge end labels apart when the values nearly coincide
            for y, label, (x, y_real) in sorted(ends):
                y_text = max([y] + [p + 0.07 * top for p in placed])
                placed.append(y_text)
                ax.annotate(f"{label} {unit_fmt(y_real)}", (x, y_real), xytext=(x + 0.12, y_text),
                            va="center", color=INK_2, fontsize=8).set_in_layout(False)

    # --- row 1: completed work and task latency (the numbers the report asks for)
    lines(axes[0, 0], "Tasks completed and failed",
          [("completed", each("tasks", "ok")), ("timeout", each("tasks", "timeout")), ("error", each("tasks", "error"))],
          "{:.0f}".format, ref=("offered", results[0]["n_tasks"]))

    lines(axes[0, 1], "Throughput: completed tasks per minute",
          [("tasks/min", each("throughput", "tasks_per_min"))], "{:.2f}".format)
    if best:
        axes[0, 1].axvline(best["concurrency"], color=INK_2, linestyle=":", linewidth=1)
        axes[0, 1].annotate("best fixed C", (best["concurrency"], 0), xytext=(3, 3), textcoords="offset points",
                            color=INK_2, fontsize=8)

    lines(axes[0, 2], "Task completion time (arrive → end), s",
          [("p50", each("completion_s", "p50")), ("p95", each("completion_s", "p95")),
           ("p99", each("completion_s", "p99"))], "{:.0f} s".format)

    lines(axes[0, 3], "Queueing time (arrive → start), s",
          [("p50", each("queue_s", "p50")), ("p95", each("queue_s", "p95")), ("p99", each("queue_s", "p99"))],
          "{:.0f} s".format)

    # --- row 2: memory pressure
    lines(axes[1, 0], "Server memory (peak), system free (min), MiB",
          [("RSS", each("memory", "server_rss_mib", "peak")), ("anon", each("memory", "server_anon_mib", "peak")),
           ("USS", each("memory", "server_uss_mib", "peak")), ("system free", each("memory", "sys_avail_min_mib"))],
          "{:.0f}".format)

    lines(axes[1, 1], "Context tokens (peak): slots vs running tasks",
          [("slots (/slots)", each("memory", "slot_context", "slot_context_tokens_total_peak")),
           ("running tasks", each("memory", "ctx_tokens_held", "peak"))],
          "{:.0f}".format, ref=("n_ctx", get(results[0], "kv_cache", "n_ctx")))

    lines(axes[1, 2], "Mean concurrency: server slots, requests, agents",
          [("busy slots", each("effective_concurrency", "server_busy_slots", "mean")),
           ("in flight", each("effective_concurrency", "client_in_flight", "mean")),
           ("runnable", each("concurrency_timeline", "runnable", "mean")),
           ("in tools", each("concurrency_timeline", "in_tool", "mean"))], "{:.2f}".format)

    lines(axes[1, 3], "KV cache full (server.log), out-of-memory errors",
          [("purges", each("failures", "server_kv_full_purges")),
           ("retries", each("failures", "server_kv_full_retries")),
           ("KV errors", each("failures", "server_kv_full_fatal")),
           ("OOM errors", each("failures", "memory_errors"))], "{:.0f}".format)

    # --- row 3: where request time goes, context reuse, fairness
    ax = axes[2, 0]
    parts = [("prefill", each("requests", "prefill_ms", "mean", scale=1e-3)),
             ("decode", each("requests", "decode_ms", "mean", scale=1e-3)),
             ("waiting in server", each("requests", "non_compute_s", "mean"))]
    bottom = [0.0] * len(cs)
    for i, (label, ys) in enumerate(parts):
        ys = [max(y or 0, 0) for y in ys]        # server and client clocks can disagree by a little
        ax.bar(cs, ys, bottom=bottom, width=0.55, color=SERIES[i], edgecolor=SURFACE, linewidth=2, label=label)
        bottom = [b + y for b, y in zip(bottom, ys)]
    ax.set_ylim(0, max(max(bottom) * 1.12, 1e-6))
    ax.set_title("Mean time per LLM request, s", pad=22)
    legend_above(ax)

    lines(axes[2, 1], "Decode speed, output tokens/s",
          [("all requests together", each("throughput", "output_tok_per_s")),
           ("one request (p50)", each("requests", "decode_tok_per_s", "p50"))], "{:.1f}".format)

    lines(axes[2, 2], "Prompt tokens served from KV cache, %",
          [("later steps", each("requests", "cache_hit_ratio", "later_steps", scale=100)),
           ("first step", each("requests", "cache_hit_ratio", "first_step", scale=100)),
           ("per request p50", each("requests", "cache_reuse_ratio", "p50", scale=100))],
          "{:.0f}%".format, ymax=110)

    present = [p for p in PROFILES
               if any(get(m, "head_of_line", "task_p50_by_profile", p, "completion_s") is not None for m in results)]
    lines(axes[2, 3], "Completion p50 by profile (ctx, response), s",
          [(p, each("head_of_line", "task_p50_by_profile", p, "completion_s")) for p in present], "{:.0f} s".format)

    r0 = results[0]
    model = os.path.basename(r0.get("model") or "?")
    backend = "fake LLM" if r0.get("test") else "llama.cpp llama-server (fixed shared KV)"
    cond = r0.get("condition") or {}
    delay = cond.get("tool_delay_override")
    fig.suptitle(f"FIFO concurrency sweep — {model}, {backend}, {r0['n_tasks']} tasks per run, "
                 f"combination {cond.get('combination') or 'default'}, long tier {cond.get('long_tier') or 'near'}, "
                 f"tool delay {delay if delay is not None else 'from workload'}",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    print(f"curves -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-C', '--concurrency', type=int, nargs='+', default=[1, 2])
    ap.add_argument('-n', type=int, default=6)
    ap.add_argument('--workload', type=Path, default=HERE / 'task-generator/source/phase1-articles.toml')
    ap.add_argument('--combination')
    ap.add_argument('--long-tier', choices=['near', 'overload'], default='near')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cache-prompt', choices=['on', 'off'], default='on',
                    help='reuse matching prompt prefixes; off does not disable KV allocation')
    ap.add_argument('--tool-delay', type=float, nargs='+', metavar='SEC',
                    help='idle time between agent steps: one value for every tool call, or one per tool call '
                         '(default: tool_delay_s from the workload)')
    ap.add_argument('--timeout', type=float, default=1200)
    ap.add_argument('--mem-interval', type=float, default=0.5)
    ap.add_argument('--ctx-size', type=int, default=8192, help='total shared KV capacity; fixed across C')
    ap.add_argument('--slots', type=int, default=8, help='fixed server slot count; must cover max C')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--server-binary', type=Path, default=LLAMA_SERVER)
    ap.add_argument('--model', type=Path, default=HERE / 'models/Qwen3-4B-Q8_0.gguf')
    ap.add_argument('--host', default='127.0.0.1', help='address llama-server listens on')
    ap.add_argument('--client-host', help='address the runner connects to (default: --host, or 127.0.0.1 for 0.0.0.0)')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--out-dir', type=Path)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--plot-only', action='store_true', help='recompute metrics, summary and curves of --out-dir')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()
    if args.plot_only:
        if args.out_dir is None:
            ap.error('--plot-only requires --out-dir')
        results = [metrics.report(str(p.parent)) for p in sorted(args.out_dir.glob('*/config.json'))]
        if results:
            summarize(results, args.out_dir, args.no_plot)
        return
    if (not args.concurrency or min(args.concurrency) < 1 or len(set(args.concurrency)) != len(args.concurrency)
            or args.slots < max(args.concurrency) or args.ctx_size <= 0 or args.threads <= 0
            or args.timeout <= 0 or args.mem_interval < 0):
        ap.error('require distinct positive C, slots >= max C, positive capacity/threads/timeout and nonnegative sampling interval')
    tasks = agent.load_workload(args.workload, args.combination, args.long_tier, args.n, args.seed)
    if args.tool_delay is not None:             # recorded in the manifest overrides, so every C replays it
        agent.apply_tool_delay(tasks, args.tool_delay)
    if len(tasks) < max(args.concurrency):
        ap.error('selected workload must contain at least max C tasks')
    if args.tool_delay is not None:
        if not all(math.isfinite(d) and d >= 0 for d in args.tool_delay):
            ap.error('--tool-delay must be finite and nonnegative')
        if any(len(args.tool_delay) not in (1, t['tool_calls']) for t in tasks):
            ap.error('--tool-delay needs 1 value or one per tool call')
    out = args.out_dir or HERE / 'results' / f'experiment_{time.time_ns()}'
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out.resolve() / 'workload.jsonl'
    snapshot.write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in tasks), encoding='utf-8')
    url = f'http://{client_host(args)}:{args.port}'
    manifest = dict(workload=str(args.workload), combination=args.combination, long_tier=args.long_tier,
                    selection_seed=args.seed, tool_delay_override=args.tool_delay,
                    cache_prompt=args.cache_prompt == 'on', n_tasks=len(tasks),
                    concurrency=args.concurrency, runner='agent.py', scheduler='task_fifo', arrival='all_at_once_file_order',
                    test=args.test, server_command=None if args.test else server_cmd(args), url=url,
                    warmup=False,
                    timeout_s=args.timeout, mem_interval_s=args.mem_interval, snapshot=str(snapshot),
                    git=git_state(), python=sys.version.split()[0], platform=platform.platform())
    (out / 'experiment.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    results = []
    for c in args.concurrency:
        run = f'c{c}_n{len(tasks)}'
        server_log = out / f'{run}.server.log'
        proc = log = None
        if not args.test:
            proc, log = start_server(args, server_log)
        try:
            cmd = [sys.executable, '-B', str(HERE / 'agent.py'), '--workload', str(snapshot),
                   '-c', str(c), '--timeout', str(args.timeout), '--mem-interval', str(args.mem_interval),
                   '--url', url, '--out-dir', str(out.resolve()), '--run-name', run,
                   '--cache-prompt', args.cache_prompt]
            if args.tool_delay is not None:
                cmd.extend(['--tool-delay', *map(str, args.tool_delay)])
            if args.test:
                cmd.append('--test')
            else:
                cmd.extend(['--server-pid', str(proc.pid)])
            with (out / f'{run}.agent.log').open('w', encoding='utf-8') as agent_log:
                rc = subprocess.run(cmd, cwd=HERE, stdout=agent_log, stderr=subprocess.STDOUT).returncode
        finally:
            if proc is not None:
                stop_server(proc, log)
        run_dir = out / run
        if server_log.exists() and run_dir.exists():
            shutil.copyfile(server_log, run_dir / 'server.log')     # metrics reads KV buffers and KV full events
        if rc:
            raise RuntimeError(f'runner exited {rc}; retained logs: {out / (run + ".agent.log")}')
        config_path = run_dir / 'config.json'
        config = json.loads(config_path.read_text())
        config.update(combination=args.combination, long_tier=args.long_tier, selection_seed=args.seed,
                      workload_source=str(args.workload))
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
        results.append(metrics.report(str(run_dir)))
    summarize(results, out, args.no_plot)
    print(f'Experiment saved to {out}')


if __name__ == '__main__':
    main()
