"""Phase 1 metrics for agent.py runs.

    python metrics.py results/c2_n6                    # print the metrics, write results/c2_n6/metrics.json
    python metrics.py results/c1_n12 results/c4_n12    # several runs: also write results/summary.csv
    python metrics.py "results/c*_n12"                 # globs are expanded here, so this works in PowerShell too

Reads config.json, events.jsonl, memory.jsonl (optional) and server.log (optional, llama-server's
log as experiment.py saves it) from each run directory.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import Counter


def r(x, nd=3):
    return None if x is None else round(x, nd)


def pct(xs, p):
    """Percentile with linear interpolation (same as numpy's default)."""
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def dist(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return dict(n=len(xs), mean=r(sum(xs) / len(xs)), p50=r(pct(xs, 50)), p90=r(pct(xs, 90)),
                p95=r(pct(xs, 95)), p99=r(pct(xs, 99)), max=r(max(xs)))


def tail(d):
    """The tail of a distribution: p95, p99, max, and how far p99 sits above the median (1.0 = no tail;
    left out when the median is ~0, e.g. queueing at low load). With few samples (n < 100) p99 is
    interpolated between the largest values, so it is close to max."""
    if not d:
        return None
    return dict(n=d["n"], p50=d["p50"], p95=d["p95"], p99=d["p99"], max=d["max"],
                p99_over_p50=r(d["p99"] / d["p50"], 2) if d["p50"] >= 0.01 else None)


def timeline(changes, t0, t1):
    """Step function given as (t, delta) changes, in log order -> peak, time-weighted mean, share of time at 0."""
    level = peak = 0
    area = zero = 0.0
    last = t0
    for t, d in changes:
        t = min(max(t, t0), t1)
        area += level * (t - last)
        zero += (t - last) if level == 0 else 0
        last, level = t, level + d
        peak = max(peak, level)
    area += level * (t1 - last)
    zero += (t1 - last) if level == 0 else 0
    span = (t1 - t0) or 1e-9
    return dict(peak=peak, mean=r(area / span), zero_frac=r(zero / span))


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gguf_metadata(path):
    """The key/value header of a GGUF file (stops before the tensor table, so it reads a few KB, not the model)."""
    import struct
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    with open(path, "rb") as f:
        def read(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

        def string():
            return f.read(read("Q")).decode("utf-8", errors="replace")

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                item_t, n = read("I"), read("Q")
                if item_t in scalar:                 # skip big numeric arrays (e.g. token scores) without decoding
                    f.seek(n * struct.calcsize(scalar[item_t]), 1)
                    return None
                return [value(item_t) for _ in range(n)]
            return read(scalar[t])

        if f.read(4) != b"GGUF":
            return None
        read("I"), read("Q")                         # version, tensor count
        return {string(): value(read("I")) for _ in range(read("Q"))}


def kv_bytes_per_token(model_path):
    """KV cache bytes per token for an f16 cache (llama-server's default): layers x KV heads x (K + V dim) x 2."""
    try:
        meta = gguf_metadata(model_path)
    except (OSError, KeyError, ValueError, TypeError):
        return None
    if not meta:
        return None
    a = meta.get("general.architecture")
    n_layer, n_kv = meta.get(f"{a}.block_count"), meta.get(f"{a}.attention.head_count_kv")
    head = (meta.get(f"{a}.embedding_length") or 0) // (meta.get(f"{a}.attention.head_count") or 1)
    k, v = meta.get(f"{a}.attention.key_length", head), meta.get(f"{a}.attention.value_length", head)
    if not (n_layer and n_kv and k and v):
        return None
    return n_layer * n_kv * (k + v) * 2


def server_buffers(path, model_path=None):
    """Read allocated buffers from server logs; never infer physical use from slot sums.

    Total context is taken from the context log, not slots * per-slot limit:
    unified KV can expose the full shared capacity as each slot's limit.
    Missing buffer sizes remain missing rather than assuming f16 precision.
    """
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    def total(kind):
        xs = [float(x) for x in re.findall(rf"{kind} buffer size =\s*([\d.]+) MiB", text)]
        return r(sum(xs), 1) if xs else None

    def last_int(pattern):
        xs = re.findall(pattern, text)
        return int(xs[-1]) if xs else None

    slots, ctx_slot = last_int(r"n_slots = (\d+)"), last_int(r"n_ctx_slot = (\d+)")
    n_ctx = last_int(r"llama_context: n_ctx\s*=\s*(\d+)")
    out = dict(model_mib=total("model"), kv_mib=total("KV"), compute_mib=total("compute"),
               n_slots=slots, n_ctx_slot=ctx_slot, n_ctx=n_ctx, kv_source="log")
    if out["kv_mib"] and n_ctx:
        out["kv_kib_per_token"] = r(out["kv_mib"] * 1024 / n_ctx, 1)
    else:
        # Unknown cache precision/layout: keep missing measurements explicit.
        out['kv_kib_per_token'] = None
        if out['kv_mib'] is None:
            out['kv_source'] = None
    return out if any(v is not None for k, v in out.items() if k != "kv_source") else None


KV_LIMIT = 0.95          # a KV cache this full counts as "at the memory limit"
LOW_MEMORY = 0.10        # ... and so does system memory with less than this share of RAM available
PAGE_KIB = 4
MEMORY_ERROR = re.compile(r"context size has been exceeded|exceed_context_size|failed to find free space|"
                          r"out of memory|failed to allocate|bad_alloc", re.I)


def server_log_counts(path):
    """Memory events in server.log: decode batches that found no free KV cells (llama-server then purges an
    idle slot, or retries with a smaller batch), requests failed because the KV cache was full, allocation
    failures."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return dict(kv_full_purges=len(re.findall(r"purging slot \d+ with \d+ tokens", text)),
                kv_full_retries=text.count("failed to find free space in the KV cache"),
                kv_full_fatal=text.count("Context size has been exceeded"),
                alloc_failures=len(re.findall(r"failed to allocate|out of memory|bad_alloc", text, re.I)))


def context_reuse(llm):
    """Prompt tokens requested vs. prefilled vs. reused from the KV cache, and which slots served the requests."""
    def total(key, old):                         # runs before these fields existed log only the old names
        xs = [e[key] if e.get(key) is not None else e.get(old) for e in llm]
        xs = [x for x in xs if x is not None]
        return sum(xs) if xs else None

    requested = total("prompt_tokens_requested", "prompt_tokens")
    processed = total("prompt_tokens_processed", "prompt_n")
    reused = total("cached_tokens_reused", "cache_n")
    slots = Counter(e["slot_id"] for e in llm if e.get("slot_id") is not None)
    return dict(
        prompt_tokens_requested=requested, prompt_tokens_processed=processed, cached_tokens_reused=reused,
        reused_frac=r(reused / requested) if requested and reused is not None else None,
        requests_per_slot={str(k): v for k, v in sorted(slots.items())} or None)


def frac(samples, cond):
    """Share of the samples where cond holds (None = not measured); samples are evenly spaced, so of the time."""
    xs = [x for x in map(cond, samples) if x is not None]
    return r(sum(xs) / len(xs)) if xs else None


def stat(xs):
    xs = [x for x in xs if x is not None]
    return dict(peak=max(xs), mean=r(sum(xs) / len(xs))) if xs else None


def gauge(m, name):
    return (m.get("srv_metrics") or {}).get(f"llamacpp:{name}")


def level_at(changes, t):
    return sum(d for ti, d in changes if ti <= t)


def head_of_line(ev, llm):
    """Head-of-line blocking: the time a request waited for its first token while another request with a
    larger prefill was being prefilled. A request's prefill is placed just before its decode, which ends at
    llm_end; its wait for the first token runs from llm_start to the end of its own prefill."""
    started = {(e["task_id"], e["step"]): e["t"] for e in ev if e["event"] == "llm_start"}
    reqs = []
    for e in llm:
        if e.get("prompt_ms") is None or e.get("predicted_ms") is None:
            continue
        first = e["t"] - e["predicted_ms"] / 1000
        reqs.append(dict(start=started.get((e["task_id"], e["step"]), first), first=first,
                         prefill=(first - e["prompt_ms"] / 1000, first),
                         size=e.get("prompt_tokens_processed", e.get("prompt_n")) or 0))
    if not reqs:
        return None
    delays = []
    for q in reqs:
        spans = sorted((max(o["prefill"][0], q["start"]), min(o["prefill"][1], q["first"]))
                       for o in reqs if o is not q and o["size"] > q["size"])
        blocked, end = 0.0, q["start"]
        for a, b in spans:                       # length of the union of the overlaps
            a = max(a, end)
            if b > a:
                blocked, end = blocked + b - a, b
        delays.append(blocked)
    waits = sum(max(q["first"] - q["start"], 0) for q in reqs)
    hit = [d for d in delays if d > 0.001]
    return dict(n_requests=len(reqs), blocked=len(hit), blocked_frac=r(len(hit) / len(reqs)),
                blocked_s=dist(hit), share_of_ttft=r(sum(delays) / waits) if waits else None)


def estimation_error(cfg, llm, end):
    """Each task's pre-run estimate of its peak context (est_context_tokens, what an admission policy would
    budget) against the measured peak, prompt + output of its largest request; completed tasks only.
    error = (estimate - actual) / actual: > 0 over-reserves memory, < 0 under-reserves it."""
    est = {t["task_id"]: t.get("est_context_tokens") for t in cfg.get("tasks") or []}
    actual = {}
    for e in llm:
        if e.get("prompt_tokens") is not None:
            actual[e["task_id"]] = max(actual.get(e["task_id"], 0), e["prompt_tokens"] + (e.get("completion_tokens") or 0))
    pairs = [(est[t], a) for t, a in actual.items() if est.get(t) and a and end.get(t, {}).get("status") == "ok"]
    if not pairs:
        return None
    errs = [(x - a) / a for x, a in pairs]
    return dict(n=len(pairs), est_tokens=sum(x for x, _ in pairs), actual_tokens=sum(a for _, a in pairs),
                mean_error=r(sum(errs) / len(errs)), mean_abs_error=r(sum(map(abs, errs)) / len(errs)),
                max_over=r(max(errs)), max_under=r(min(errs)), underestimated=sum(x < 0 for x in errs))


def counter_delta(mem, key):
    xs = [(m["t"], m[key]) for m in mem if m.get(key) is not None]
    if len(xs) < 2:
        return None
    d, span = xs[-1][1] - xs[0][1], xs[-1][0] - xs[0][0]
    return dict(total=d, per_s=r(d / span if span else None))


def compute(run_dir):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    ev = load_jsonl(os.path.join(run_dir, "events.jsonl"))
    mem = load_jsonl(os.path.join(run_dir, "memory.jsonl"))

    arrive = {e["task_id"]: e["t"] for e in ev if e["event"] == "task_arrive"}
    start = {e["task_id"]: e["t"] for e in ev if e["event"] == "task_start"}
    end = {e["task_id"]: e for e in ev if e["event"] == "task_end"}
    llm = [e for e in ev if e["event"] == "llm_end"]
    status = Counter(e["status"] for e in end.values())
    ok = [tid for tid, e in end.items() if e["status"] == "ok"]
    t0 = min(arrive.values(), default=0)
    t1 = max([e["t"] for e in ev] + [t0])
    makespan = t1 - t0

    # --- tasks: completion = arrive -> end, queueing = arrive -> start, service = start -> end
    n = len(arrive)
    tasks = dict(n=n, ok=len(ok), timeout=status["timeout"], error=status["error"],
                 unfinished=n - len(end), failure_rate=r((n - len(ok)) / n if n else 0))

    # --- requests: llama-server reports prefill (prompt_ms) and decode (predicted_ms) time per request;
    #     the rest of the latency is spent waiting inside the server (no free slot) or in HTTP.
    def non_compute(e):
        if e.get("prompt_ms") is None or e.get("predicted_ms") is None:
            return None
        return e["latency_s"] - (e["prompt_ms"] + e["predicted_ms"]) / 1000

    def decode_rate(e):
        n = e.get("predicted_n") or e.get("completion_tokens")
        if not e.get("predicted_ms") or not n:
            return None
        return n / (e["predicted_ms"] / 1000)

    def ttft(e):                                 # time to first token: waiting in the server + prefill
        return None if e.get("predicted_ms") is None else e["latency_s"] - e["predicted_ms"] / 1000

    def tpot(e):                                 # time per output token while decoding (grows with batching)
        n = e.get("predicted_n") or e.get("completion_tokens")
        return e["predicted_ms"] / n if e.get("predicted_ms") and n else None

    def hit_ratio(es):
        cached = sum(e.get("cache_n") or 0 for e in es)
        total = cached + sum(e.get("prompt_n") or 0 for e in es)
        return r(cached / total) if total else None

    def reuse_ratio(e):                          # cache_n / (cache_n + prompt_n) of one request
        if e.get("cache_reuse_ratio") is not None:
            return e["cache_reuse_ratio"]
        if e.get("cache_n") is None or e.get("prompt_n") is None or not e["cache_n"] + e["prompt_n"]:
            return None
        return e["cache_n"] / (e["cache_n"] + e["prompt_n"])

    prefill_s = sum(e.get("prompt_ms") or 0 for e in llm) / 1000
    decode_s = sum(e.get("predicted_ms") or 0 for e in llm) / 1000
    out_tok = sum(e.get("completion_tokens") or 0 for e in llm)
    prefill_tok = sum(e.get("prompt_n") or 0 for e in llm)
    requests = dict(
        n=len(llm), latency_s=dist([e["latency_s"] for e in llm]),
        prefill_ms=dist([e.get("prompt_ms") for e in llm]), decode_ms=dist([e.get("predicted_ms") for e in llm]),
        decode_tok_per_s=dist([decode_rate(e) for e in llm]), non_compute_s=dist([non_compute(e) for e in llm]),
        ttft_s=dist([ttft(e) for e in llm]), tpot_ms=dist([tpot(e) for e in llm]),
        prompt_tokens=dist([e.get("prompt_tokens") for e in llm]),
        prefill_s_total=r(prefill_s, 1), decode_s_total=r(decode_s, 1),
        prefill_share=r(prefill_s / (prefill_s + decode_s)) if prefill_s + decode_s else None,
        cache_hit_ratio=dict(all=hit_ratio(llm), first_step=hit_ratio([e for e in llm if e["step"] == 0]),
                             later_steps=hit_ratio([e for e in llm if e["step"] > 0])),
        cache_reuse_ratio=dist([reuse_ratio(e) for e in llm]))

    log = server_log_counts(os.path.join(run_dir, "server.log"))
    context = context_reuse(llm)

    throughput = dict(makespan_s=r(makespan), tasks_per_min=r(len(ok) / makespan * 60 if makespan else None),
                      requests_per_s=r(len(llm) / makespan if makespan else None),
                      output_tok_per_s=r(out_tok / makespan if makespan else None),
                      prefill_tok_per_s=r(prefill_tok / makespan if makespan else None))

    # --- concurrency over time, rebuilt from the event log. A task that fails mid-request or mid-tool
    #     has no llm_end / tool_end, so task_end closes whatever it still had open.
    ch = {k: [] for k in ("tasks", "llm", "tool", "ctx", "runnable")}   # runnable = running, not in a tool
    open_llm, open_tool, ctx = set(), set(), {}
    tool_s = 0.0
    tool_start = {}
    for e in ev:
        tid, t = e.get("task_id"), e["t"]
        if e["event"] == "task_start":
            ch["tasks"].append((t, 1)); ch["runnable"].append((t, 1))
        elif e["event"] == "llm_start":
            ch["llm"].append((t, 1)); open_llm.add(tid)
        elif e["event"] == "llm_end":
            ch["llm"].append((t, -1)); open_llm.discard(tid)
            if e.get("prompt_tokens") is not None:          # tokens this task now holds in the KV cache
                new = e["prompt_tokens"] + (e.get("completion_tokens") or 0)
                ch["ctx"].append((t, new - ctx.get(tid, 0))); ctx[tid] = new
        elif e["event"] == "tool_start":
            ch["tool"].append((t, 1)); open_tool.add(tid); tool_start[tid] = t; ch["runnable"].append((t, -1))
        elif e["event"] == "tool_end":
            ch["tool"].append((t, -1)); open_tool.discard(tid); tool_s += t - tool_start.pop(tid)
            ch["runnable"].append((t, 1))
        elif e["event"] == "task_end":
            ch["tasks"].append((t, -1))
            if tid not in open_tool:
                ch["runnable"].append((t, -1))
            if tid in open_llm:
                ch["llm"].append((t, -1)); open_llm.discard(tid)
            if tid in open_tool:
                ch["tool"].append((t, -1)); open_tool.discard(tid); tool_s += t - tool_start.pop(tid)
            if tid in ctx:
                ch["ctx"].append((t, -ctx.pop(tid)))
    busy_s = sum(end[tid]["t"] - start[tid] for tid in end if tid in start)
    llm_tl, tasks_tl, tool_tl = timeline(ch["llm"], t0, t1), timeline(ch["tasks"], t0, t1), timeline(ch["tool"], t0, t1)
    concurrency = dict(tasks=tasks_tl, llm_in_flight=llm_tl, in_tool=tool_tl, runnable=timeline(ch["runnable"], t0, t1),
                       tool_waiting_frac=r(tool_tl["mean"] / tasks_tl["mean"]) if tasks_tl["mean"] else None,
                       client_no_request_frac=llm_tl["zero_frac"],
                       tool_share_of_task_time=r(tool_s / busy_s if busy_s else None))

    # Normalize main's canonical memory schema and accept historical PR samples.
    # Runs before the renames stored RssAnon as srv_private_mib, and PR samples' srv_slots as kv_tokens.
    def srv_slots(m):
        if 'slots' in m:
            return dict(busy=m['busy_slots'], slot_context_tokens=[s['tokens'] for s in m['slots']])
        s = m.get('srv_slots')
        return s and dict(busy=s['busy'], slot_context_tokens=s.get('slot_context_tokens', s.get('kv_tokens')))

    mem = [dict(m, srv_rss_mib=m.get('rss_mb', m.get('srv_rss_mib')),
                srv_anon_mib=m.get('srv_anon_mib', m.get('srv_private_mib')), srv_slots=srv_slots(m)) for m in mem]
    # --- memory: samples from monitor.py; the first one is taken before any task starts
    def series(key):
        return [m[key] for m in mem if m.get(key) is not None]

    def summary(key):
        xs = series(key)
        if not xs:
            return None
        return dict(baseline=xs[0], peak=max(xs), mean=r(sum(xs) / len(xs), 1), delta=r(max(xs) - xs[0], 1))

    gauges = {}                                  # llama-server /metrics: peak of every non-counter value
    for m in mem:
        for k, v in (m.get("srv_metrics") or {}).items():
            if not k.endswith("_total"):
                gauges[k] = max(gauges.get(k, v), v)
    ctx_tl = timeline(ch["ctx"], t0, t1)
    buffers = server_buffers(os.path.join(run_dir, "server.log"), cfg.get("model"))

    # --- llama-server slots, sampled from /slots during the makespan: how many are decoding at once, and the
    #     length of the token sequences they keep (idle slots keep their last conversation, so this is >= the
    #     context of running tasks). Token counts, not converted to KV cache use.
    slots = [m["srv_slots"] for m in mem if m.get("srv_slots") and t0 <= m["t"] <= t1]
    slot_context = None
    if slots:
        busy = [s["busy"] for s in slots]
        tot = [sum(s["slot_context_tokens"]) for s in slots]
        slot_context = dict(n_samples=len(slots), busy_peak=max(busy), busy_mean=r(sum(busy) / len(busy)),
                            slot_context_tokens_total_peak=max(tot), slot_context_tokens_total_mean=r(sum(tot) / len(tot), 1),
                            slot_context_tokens_peak_one_slot=max(max(s["slot_context_tokens"], default=0) for s in slots),
                            interpretation='/slots n_prompt_tokens (prompt.tokens.size()); not KV cache allocation or use')

    memory = dict(
        n_samples=len(mem),
        server_rss_mib=summary("srv_rss_mib"), server_anon_mib=summary("srv_anon_mib"),
        server_uss_mib=summary("srv_uss_mib"), server_private_dirty_mib=summary("srv_private_dirty_mib"),
        server_swap_mib=summary("srv_swap_mib"),
        server_os_peak_rss_mib=max(series("srv_peak_rss_mib"), default=None),
        sys_used_mib=summary("sys_used_mib"), sys_avail_min_mib=min(series("sys_avail_mib"), default=None),
        swap_used_mib=summary("swap_used_mib"), gpu_used_mib=summary("gpu_used_mib"),
        ctx_tokens_held=dict(peak=ctx_tl["peak"], mean=ctx_tl["mean"]),
        server_buffers=buffers, slot_context=slot_context,
        server_metrics_peak=gauges or None)

    # --- the samples taken while tasks ran
    win = [m for m in mem if t0 <= m["t"] <= t1]

    # --- KV cache. allocated_mib: the KV buffers llama-server allocates once at startup (server.log), n_ctx:
    #     their size in tokens. usage_ratio / tokens: llama-server's own llamacpp:kv_cache_usage_ratio and
    #     llamacpp:kv_cache_tokens gauges, removed in b5441 (#13660), so null on newer builds.
    #     slot_context_tokens_total: the sum of /slots n_prompt_tokens, the token sequences the slots keep
    #     (prompt.tokens.size()); a token count, not the KV cache's allocation or memory use.
    n_ctx, n_ctx_src = (buffers or {}).get("n_ctx"), "server.log"
    if n_ctx is None and (cfg.get("server") or {}).get("n_ctx"):
        n_ctx, n_ctx_src = cfg["server"]["n_ctx"], "/props n_ctx (shared capacity with --kv-unified)"
    slot_ctx = [sum(m["srv_slots"]["slot_context_tokens"]) for m in win if m.get("srv_slots")]
    usage = [gauge(m, "kv_cache_usage_ratio") for m in win]
    scraped = any(m.get("srv_metrics") for m in win)
    kv_cache = None
    if slot_ctx or buffers or scraped:
        kv_cache = dict(allocated_mib=(buffers or {}).get("kv_mib"), n_ctx=n_ctx, n_ctx_source=n_ctx_src if n_ctx else None,
                        usage_ratio=stat(usage), tokens=stat([gauge(m, "kv_cache_tokens") for m in win]),
                        server_kv_metrics=(any(x is not None for x in usage) if scraped else None),
                        slot_context_tokens_total=stat(slot_ctx),
                        slot_context_frac_of_n_ctx=dict(peak=r(max(slot_ctx) / n_ctx), mean=r(sum(slot_ctx) / len(slot_ctx) / n_ctx))
                        if slot_ctx and n_ctx else None)

    ram = (cfg.get("host") or {}).get("ram_total_mib")
    time_at_limit = dict(
        kv_full=frac(win, lambda m: None if gauge(m, "kv_cache_usage_ratio") is None
                     else gauge(m, "kv_cache_usage_ratio") >= KV_LIMIT),
        server_deferring=frac(win, lambda m: None if gauge(m, "requests_deferred") is None else gauge(m, "requests_deferred") > 0),
        low_system_memory=frac(win, lambda m: None if m.get("sys_avail_mib") is None or not ram
                               else m["sys_avail_mib"] < LOW_MEMORY * ram),
        thresholds=dict(kv_cache_usage_ratio=KV_LIMIT, sys_avail_frac=LOW_MEMORY))

    # --- failures: requests that failed (llm_error; older runs: failed tasks), which of them for lack of memory,
    #     and the load when the first one happened
    errors = [e for e in ev if e["event"] == "llm_error"] or [e for e in end.values() if e["status"] != "ok"]
    mem_errors = [e for e in errors if MEMORY_ERROR.search(e.get("error") or "")]
    first = None
    if mem_errors:
        t = mem_errors[0]["t"]
        near = min(win or mem, key=lambda m: abs(m["t"] - t), default=None)
        first = dict(t=r(t), tasks_running=level_at(ch["tasks"], t), llm_in_flight=level_at(ch["llm"], t),
                     kv_cache_usage_ratio=gauge(near, "kv_cache_usage_ratio") if near else None,
                     slot_context_tokens_total=sum(near["srv_slots"]["slot_context_tokens"]) if near and near.get("srv_slots") else None)
    failures = dict(request_errors=len(errors), timeouts=sum(1 for e in errors if e.get("timeout") or e.get("status") == "timeout"),
                    memory_errors=len(mem_errors), first_memory_error=first,
                    server_kv_full_purges=(log or {}).get("kv_full_purges"),
                    server_kv_full_retries=(log or {}).get("kv_full_retries"),
                    server_kv_full_fatal=(log or {}).get("kv_full_fatal"),
                    server_alloc_failures=(log or {}).get("alloc_failures"),
                    server_exited=any(m.get("srv_exited") for m in mem))

    # --- effective concurrency: requests the server actually works on at once (/slots, /metrics), next to
    #     the requests the client has in flight
    per_decode = [gauge(m, "n_busy_slots_per_decode") for m in win if gauge(m, "n_busy_slots_per_decode") is not None]
    effective = dict(server_busy_slots=stat([m["srv_slots"]["busy"] for m in win if m.get("srv_slots")]),
                     server_processing=stat([gauge(m, "requests_processing") for m in win]),
                     server_deferred=stat([gauge(m, "requests_deferred") for m in win]),
                     client_in_flight=dict(peak=llm_tl["peak"], mean=llm_tl["mean"]),
                     busy_slots_per_decode=per_decode[-1] if per_decode else None)

    # --- duplicate work: retried and cancelled requests, and tokens spent on tasks that did not complete
    starts = Counter((e["task_id"], e["step"]) for e in ev if e["event"] == "llm_start")
    done = {(e["task_id"], e["step"]) for e in llm}
    failed = {tid for tid, e in end.items() if e["status"] != "ok"}
    retries = dict(retries=sum(c - 1 for c in starts.values()), cancelled=sum(k not in done for k in starts),
                   failed_task_tokens=sum((e.get("prompt_n") or 0) + (e.get("completion_tokens") or 0)
                                          for e in llm if e["task_id"] in failed))

    # --- CPU / GPU utilization (CPU: 100% = all cores; the server's only if its process is visible) and paging
    utilization = dict(sys_cpu_pct=stat([m.get("sys_cpu_pct") for m in win]),
                       server_cpu_pct=stat([m.get("srv_cpu_pct") for m in win]),
                       gpu_util_pct=stat([m.get("gpu_util_pct") for m in win]))
    swap = {k: counter_delta(mem, f"vm_{k}") for k in ("pswpin", "pswpout")}
    paging = dict(major_faults=counter_delta(mem, "vm_pgmajfault"), server_major_faults=counter_delta(mem, "srv_majflt"),
                  swap_in_mib=r(swap["pswpin"]["total"] * PAGE_KIB / 1024, 1) if swap["pswpin"] else None,
                  swap_out_mib=r(swap["pswpout"]["total"] * PAGE_KIB / 1024, 1) if swap["pswpout"] else None,
                  swap_used_mib=memory["swap_used_mib"])

    props = cfg.get("server_props") or {}
    main_server = cfg.get("server") or {}
    server = dict(total_slots=main_server.get("total_slots", props.get("total_slots")),
                  n_ctx_per_slot=main_server.get("n_ctx", (props.get("default_generation_settings") or {}).get("n_ctx")),
                  build=main_server.get("build", props.get("build_info")), cmdline=(cfg.get("server_process") or {}).get("cmdline"))

    completion = dist([end[t]["t"] - arrive[t] for t in ok])
    task_definitions = cfg.get("tasks") or []
    if cfg.get("workload_manifest"):
        from agent import load_workload
        task_definitions = load_workload(os.path.join(run_dir, cfg["workload_manifest"]))
    profile = {t["task_id"]: t.get("profile") for t in task_definitions}
    by_profile = {p: dict(n=sum(profile.get(t) == p for t in arrive),
                          queue_s=(dist([start[t] - arrive[t] for t in start if profile.get(t) == p]) or {}).get("p50"),
                          completion_s=(dist([end[t]["t"] - arrive[t] for t in ok if profile.get(t) == p]) or {}).get("p50"))
                  for p in sorted({p for p in profile.values() if p})}
    hol = head_of_line(ev, llm)
    if hol is not None or by_profile:
        hol = dict(hol or {}, task_p50_by_profile=by_profile or None)

    # --- outputs: a faster run must not come from shorter answers. Output tokens of each completed task, and
    #     requests cut off at max_tokens (finish_reason "length") rather than ending on their own ("stop")
    out_by_task = Counter()
    for e in llm:
        out_by_task[e["task_id"]] += e.get("completion_tokens") or 0
    finish = Counter(e["finish_reason"] for e in llm if e.get("finish_reason"))
    outputs = dict(tokens_per_completed_task=dist([out_by_task[t] for t in ok]), finish_reasons=dict(finish) or None,
                   length_cut_frac=r(finish["length"] / sum(finish.values())) if finish else None)
    queue = dist([start[t] - arrive[t] for t in start])
    tail_latency = dict(completion_s=tail(completion), queue_s=tail(queue), request_latency_s=tail(requests["latency_s"]),
                        ttft_s=tail(requests["ttft_s"]), tpot_ms=tail(requests["tpot_ms"]))

    # --- the numbers the project report asks for: completed tasks, median and tail completion latency,
    #     throughput, peak memory, failure rate. Completion latency covers ok tasks only, so read it with
    #     the failure rate next to it.
    def peak(s):
        return (memory.get(s) or {}).get("peak")

    headline = dict(completed_tasks=len(ok), n_tasks=n, failure_rate=tasks["failure_rate"],
                    median_completion_s=(completion or {}).get("p50"), p95_completion_s=(completion or {}).get("p95"),
                    p99_completion_s=(completion or {}).get("p99"), max_completion_s=(completion or {}).get("max"),
                    tasks_per_min=throughput["tasks_per_min"],
                    output_tokens_per_task=(outputs["tokens_per_completed_task"] or {}).get("mean"),
                    length_cut_frac=outputs["length_cut_frac"],
                    kv_cache_usage_ratio_peak=((kv_cache or {}).get("usage_ratio") or {}).get("peak"),
                    time_at_kv_limit_frac=time_at_limit["kv_full"],
                    slot_context_tokens_total_peak=((kv_cache or {}).get("slot_context_tokens_total") or {}).get("peak"),
                    memory_errors=failures["memory_errors"],
                    effective_concurrency=(effective["server_busy_slots"] or {}).get("mean"),
                    tool_waiting_frac=concurrency["tool_waiting_frac"],
                    peak_server_rss_mib=peak("server_rss_mib"), peak_server_anon_mib=peak("server_anon_mib"),
                    peak_server_uss_mib=peak("server_uss_mib"),
                    peak_gpu_used_mib=peak("gpu_used_mib"), peak_sys_used_mib=peak("sys_used_mib"))

    return dict(run=cfg.get("run"), concurrency=cfg.get("concurrency"), n_tasks=cfg.get("n_tasks"),
                headline=headline,
                condition={k: cfg.get(k) for k in ('scheduler', 'workload', 'combination', 'long_tier', 'selection_seed', 'timeout_s', 'arrival',
                                          'tool_delay_override', 'cache_prompt')},
                test=cfg.get("test"), model=cfg.get("model"), host=cfg.get("host"), server=server, tasks=tasks,
                completion_s=completion, queue_s=queue, service_s=dist([end[t]["t"] - start[t] for t in ok]),
                tail_latency=tail_latency,
                throughput=throughput, requests=requests, context=context, concurrency_timeline=concurrency, memory=memory,
                kv_cache=kv_cache, time_at_limit=time_at_limit, failures=failures, effective_concurrency=effective,
                outputs=outputs, head_of_line=hol, estimation=estimation_error(cfg, llm, end), retries=retries,
                utilization=utilization, paging=paging)


# ---------------------------------------------------------------- printing
def fmt_dist(d, unit="s", nd=2):
    if not d:
        return "n/a"
    return "  ".join(f"{k} {d[k]:.{nd}f}" for k in ("p50", "p95", "p99", "max", "mean")) + f" {unit}"


def fmt_tail(d, unit="s", nd=2):
    if not d:
        return "n/a"
    ratio = "" if d["p99_over_p50"] is None else f"  (p99/p50 {d['p99_over_p50']:.2f}x)"
    return "  ".join(f"{k} {d[k]:.{nd}f}" for k in ("p50", "p95", "p99", "max")) + f" {unit}{ratio}"


def fmt_pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def fmt_mem(s, label):
    if not s:
        return None
    return f"{label} {s['baseline']:.0f} -> peak {s['peak']:.0f} MiB ({s['delta']:+.0f})"


def print_report(m):
    t, th, rq, cc, mem, sv = (m["tasks"], m["throughput"], m["requests"], m["concurrency_timeline"],
                              m["memory"], m["server"])
    print(f"\n== {m['run']}  C={m['concurrency']}  tasks={m['n_tasks']}  model={m['model']}")
    hl = m["headline"]

    def num(x, fmt):
        return "n/a" if x is None else format(x, fmt)

    mem_peaks = [f"{label} {hl[k]:.0f}" for k, label in (("peak_server_rss_mib", "server RSS"),
                 ("peak_server_anon_mib", "server anon"), ("peak_server_uss_mib", "server USS"), ("peak_gpu_used_mib", "GPU"),
                 ("peak_sys_used_mib", "system")) if hl[k] is not None]
    print(f"HEADLINE     completed {hl['completed_tasks']}/{hl['n_tasks']}  failure rate {fmt_pct(hl['failure_rate'])}  |  "
          f"completion median {num(hl['median_completion_s'], '.1f')} s, tail p95 {num(hl['p95_completion_s'], '.1f')} s "
          f"p99 {num(hl['p99_completion_s'], '.1f')} s  |  {num(hl['tasks_per_min'], '.3f')} tasks/min  |  "
          f"peak MiB: {', '.join(mem_peaks) or 'n/a'}")
    if sv["total_slots"] is not None:
        print(f"server       {sv['total_slots']} slots; reported context limit {sv['n_ctx_per_slot']} tokens")
    print(f"tasks        ok {t['ok']}/{t['n']}  timeout {t['timeout']}  error {t['error']}  "
          f"unfinished {t['unfinished']}  failure rate {fmt_pct(t['failure_rate'])}")
    print(f"completion   {fmt_dist(m['completion_s'])}   (ok tasks, arrive -> end)")
    print(f"queueing     {fmt_dist(m['queue_s'])}")
    print(f"service      {fmt_dist(m['service_s'])}")
    print(f"throughput   {th['tasks_per_min']} tasks/min  {th['requests_per_s']} req/s  "
          f"{th['output_tok_per_s']} output tok/s  {th['prefill_tok_per_s']} prefill tok/s  "
          f"(makespan {th['makespan_s']} s)")
    print(f"llm request  latency {fmt_dist(rq['latency_s'])}")
    if rq["prefill_ms"]:
        print(f"             prefill p50 {rq['prefill_ms']['p50']:.0f} ms  decode p50 {rq['decode_ms']['p50']:.0f} ms  "
              f"decode p50 {rq['decode_tok_per_s']['p50']:.1f} tok/s  "
              f"wait outside prefill+decode p50 {rq['non_compute_s']['p50']:.3f} s, max {rq['non_compute_s']['max']:.3f} s")
        h = rq["cache_hit_ratio"]
        print(f"             prompt cache hit {fmt_pct(h['all'])}  "
              f"(first step {fmt_pct(h['first_step'])}, later steps {fmt_pct(h['later_steps'])})")
        if rq.get("cache_reuse_ratio"):
            c = rq["cache_reuse_ratio"]
            print(f"             cache reuse ratio per request p50 {fmt_pct(c['p50'])}  mean {fmt_pct(c['mean'])}  "
                  f"(cache_n / (cache_n + prompt_n))")
    cx = m.get("context")
    if cx and cx["prompt_tokens_requested"] is not None:
        print(f"context      prompt tokens requested {cx['prompt_tokens_requested']}, prefilled {cx['prompt_tokens_processed']}, "
              f"reused {cx['cached_tokens_reused']} ({fmt_pct(cx['reused_frac'])})")
        if cx["requests_per_slot"]:
            print("             requests per slot " + "  ".join(f"#{k}: {v}" for k, v in cx["requests_per_slot"].items()))
    if rq.get("prefill_share") is not None:
        print(f"             prefill {rq['prefill_s_total']} s vs decode {rq['decode_s_total']} s in total "
              f"(prefill {fmt_pct(rq['prefill_share'])} of compute time)")
    tl = m["tail_latency"]
    print(f"tail latency completion {fmt_tail(tl['completion_s'])}   (n={(tl['completion_s'] or {}).get('n')} tasks)")
    print(f"             queueing   {fmt_tail(tl['queue_s'])}")
    print(f"             request    {fmt_tail(tl['request_latency_s'])}   (n={rq['n']} requests)")
    if tl["ttft_s"]:
        print(f"             TTFT estimate {fmt_tail(tl['ttft_s'])}   (server wait + prefill)")
        print(f"             TPOT       {fmt_tail(tl['tpot_ms'], 'ms', 0)}   (per output token)")
    print(f"concurrency  tasks peak {cc['tasks']['peak']} mean {cc['tasks']['mean']}  |  "
          f"llm in flight peak {cc['llm_in_flight']['peak']} mean {cc['llm_in_flight']['mean']}  |  "
          f"client no request {fmt_pct(cc['client_no_request_frac'])}  |  tool share of task time {fmt_pct(cc['tool_share_of_task_time'])}")
    lines = [fmt_mem(mem["server_rss_mib"], "server RSS"), fmt_mem(mem["server_anon_mib"], "server anon (RssAnon)"),
             fmt_mem(mem["server_uss_mib"], "server USS (private)"),
             fmt_mem(mem["server_private_dirty_mib"], "server private dirty"),
             fmt_mem(mem["server_swap_mib"], "server swapped out"),
             fmt_mem(mem.get("gpu_used_mib"), "GPU used"),
             fmt_mem(mem["sys_used_mib"], "system used"), fmt_mem(mem["swap_used_mib"], "swap used")]
    lines = [x for x in lines if x]
    if mem["sys_avail_min_mib"] is not None:
        lines.append(f"system available min {mem['sys_avail_min_mib']:.0f} MiB")
    if mem["ctx_tokens_held"]["peak"]:
        lines.append(f"context held by running tasks: peak {mem['ctx_tokens_held']['peak']} tok, "
                     f"mean {mem['ctx_tokens_held']['mean']:.0f} tok")
    b, sc = mem.get("server_buffers"), mem.get("slot_context")
    if b:
        sizes = [f"weights {b['model_mib']} MiB" if b["model_mib"] else None,
                 f"KV cache {b['kv_mib']} MiB (context {b['n_ctx']} tok; slots {b['n_slots']}, reported slot limit {b['n_ctx_slot']}; "
                 f"{b['kv_kib_per_token']} KiB/tok, from {b['kv_source']})" if b["kv_mib"] else None,
                 f"compute {b['compute_mib']} MiB" if b["compute_mib"] else None]
        lines.append("server allocated: " + ", ".join(s for s in sizes if s))
    if sc:                                       # the slot context totals are printed with the KV cache
        lines.append(f"slots decoding peak {sc['busy_peak']} mean {sc['busy_mean']}; "
                     f"largest slot context {sc['slot_context_tokens_peak_one_slot']} tok")
    for i, line in enumerate(lines):
        print(("memory       " if i == 0 else "             ") + line)
    for k, v in (mem["server_metrics_peak"] or {}).items():
        print(f"             peak {k} = {v:g}")
    print_pressure(m)


def print_pressure(m):
    """KV cache, memory limits, failures, effective concurrency, blocking, estimates, duplicate work, utilization."""
    def num(x, fmt=""):
        return "n/a" if x is None else format(x, fmt)

    kv, lim, fl, ec = m.get("kv_cache"), m.get("time_at_limit"), m.get("failures"), m.get("effective_concurrency")
    if kv:
        if kv["usage_ratio"]:
            usage = (f"usage ratio peak {fmt_pct(kv['usage_ratio']['peak'])} mean {fmt_pct(kv['usage_ratio']['mean'])}, "
                     f"tokens peak {num((kv['tokens'] or {}).get('peak'))}")
        elif kv["server_kv_metrics"] is False:
            usage = "llamacpp:kv_cache_usage_ratio / kv_cache_tokens not exported by this llama-server (removed in b5441)"
        else:
            usage = "usage n/a (/metrics not scraped)"
        print(f"KV cache     allocated {num(kv['allocated_mib'])} MiB for n_ctx {num(kv['n_ctx'])} tok  |  {usage}")
        sc, sf = kv["slot_context_tokens_total"], kv["slot_context_frac_of_n_ctx"]
        if sc:
            print(f"             slot context tokens total (/slots n_prompt_tokens; a token count, not KV use) peak {sc['peak']} "
                  f"mean {sc['mean']:.0f}" + (f" ({fmt_pct(sf['peak'])} / {fmt_pct(sf['mean'])} of n_ctx)" if sf else ""))
    if lim:
        print(f"at limit     KV cache usage ratio >= {lim['thresholds']['kv_cache_usage_ratio']:.0%} {fmt_pct(lim['kv_full'])} of the time, "
              f"server deferring requests {fmt_pct(lim['server_deferring'])}, "
              f"system memory < {lim['thresholds']['sys_avail_frac']:.0%} free {fmt_pct(lim['low_system_memory'])}")
    if fl:
        log = "" if fl["server_kv_full_retries"] is None else (
            f"  |  server.log: KV full purges {fl['server_kv_full_purges']}, retries {fl['server_kv_full_retries']}, "
            f"KV full errors {fl['server_kv_full_fatal']}, "
            f"allocation failures {fl['server_alloc_failures']}")
        print(f"failures     request errors {fl['request_errors']} (timeouts {fl['timeouts']}), out of memory {fl['memory_errors']}"
              + (", server exited" if fl["server_exited"] else "") + log)
        if fl["first_memory_error"]:
            f = fl["first_memory_error"]
            print(f"             first out-of-memory error at {f['t']} s with {f['tasks_running']} tasks running, "
                  f"{f['llm_in_flight']} requests in flight, KV cache usage ratio {fmt_pct(f['kv_cache_usage_ratio'])}, "
                  f"slot context tokens total {num(f['slot_context_tokens_total'])}")
    if ec:
        sb, dq = ec["server_busy_slots"], ec["server_deferred"]
        print(f"effective    server busy slots mean {num((sb or {}).get('mean'))} peak {num((sb or {}).get('peak'))}  |  "
              f"client in flight mean {ec['client_in_flight']['mean']} peak {ec['client_in_flight']['peak']}  |  "
              f"deferred in server peak {num((dq or {}).get('peak'))}  |  busy slots per decode {num(ec['busy_slots_per_decode'])}")
    cc = m["concurrency_timeline"]
    if cc.get("runnable"):
        print(f"agents       runnable mean {cc['runnable']['mean']} peak {cc['runnable']['peak']}  |  waiting on tools mean "
              f"{cc['in_tool']['mean']} peak {cc['in_tool']['peak']} ({fmt_pct(cc['tool_waiting_frac'])} of running agents)")
    hol = m.get("head_of_line")
    if hol:
        if hol.get("n_requests"):
            print(f"head of line {hol['blocked']}/{hol['n_requests']} requests waited behind a larger prefill, "
                  f"p50 {num((hol['blocked_s'] or {}).get('p50'), '.2f')} s max {num((hol['blocked_s'] or {}).get('max'), '.2f')} s "
                  f"({fmt_pct(hol['share_of_ttft'])} of all time to first token)")
        for p, d in (hol.get("task_p50_by_profile") or {}).items():
            print(f"             {p}: {d['n']} tasks, queueing p50 {num(d['queue_s'], '.1f')} s, completion p50 {num(d['completion_s'], '.1f')} s")
    est = m.get("estimation")
    if est:
        print(f"estimate     task peak context estimated {est['est_tokens']} vs actual {est['actual_tokens']} tok: "
              f"mean error {est['mean_error']:+.0%} (|error| {est['mean_abs_error']:.0%}), range {est['max_under']:+.0%} .. "
              f"{est['max_over']:+.0%}, underestimated {est['underestimated']}/{est['n']} tasks")
    out = m.get("outputs")
    if out and out["tokens_per_completed_task"]:
        print(f"outputs      {out['tokens_per_completed_task']['mean']:.0f} output tokens per completed task (p50 "
              f"{out['tokens_per_completed_task']['p50']:.0f})  |  cut off at max_tokens {fmt_pct(out['length_cut_frac'])} "
              f"of requests {out['finish_reasons'] or ''}")
    rt = m.get("retries")
    if rt:
        print(f"duplicates   retries {rt['retries']}, cancelled requests {rt['cancelled']}, "
              f"tokens spent on failed tasks {rt['failed_task_tokens']}")
    ut, pg = m.get("utilization") or {}, m.get("paging") or {}
    use = [f"{label} mean {d['mean']}% peak {d['peak']}%" for label, d in
           (("system CPU", ut.get("sys_cpu_pct")), ("server CPU", ut.get("server_cpu_pct")), ("GPU", ut.get("gpu_util_pct"))) if d]
    if use:
        print("utilization  " + "  |  ".join(use))
    if pg.get("major_faults") or pg.get("swap_in_mib") is not None:
        print(f"paging       major faults {num((pg['major_faults'] or {}).get('total'))} "
              f"({num((pg['major_faults'] or {}).get('per_s'))}/s; server {num((pg['server_major_faults'] or {}).get('total'))})  |  "
              f"swap in {num(pg['swap_in_mib'])} MiB, out {num(pg['swap_out_mib'])} MiB")


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = " ".join(map(str, v))
        else:
            out[key] = v
    return out


def report(run_dir):
    """Compute, print and save metrics.json for one run (agent.py calls this at the end of a run)."""
    m = compute(run_dir)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print_report(m)
    return m


def write_summary(results, path):
    """One CSV row per run (flattened metrics), sorted by task count, then concurrency."""
    rows = sorted((flatten(m) for m in results),
                  key=lambda x: (x.get("n_tasks") or 0, x.get("concurrency") or 0, x.get("run") or ""))
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nsummary of {len(rows)} runs -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories (glob patterns allowed)")
    ap.add_argument("--csv", help="where to write the per-run summary (default: <common parent>/summary.csv)")
    args = ap.parse_args()

    dirs = sorted({d for pat in args.runs for d in (glob.glob(pat) or [pat])
                   if os.path.exists(os.path.join(d, "config.json"))})
    if not dirs:
        ap.error("no run directories with a config.json found")
    results = [report(d) for d in dirs]
    if len(results) > 1:
        write_summary(results, args.csv or os.path.join(os.path.commonpath([os.path.abspath(d) for d in dirs]),
                                                        "summary.csv"))


if __name__ == "__main__":
    main()
