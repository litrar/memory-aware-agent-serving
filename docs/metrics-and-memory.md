# Metrics and memory

Run all commands from the repository root.

## Integrated metrics and experiment runner

Phase 1 uses task-level FIFO with fixed C; a worker owns an entire multi-step task, including tool waits. Request-level scheduling is reserved as a Phase 2 optimization candidate.

`agent.py` uses `monitor.py` for memory collection and calls `metrics.py` after the tasks finish. Each run saves `config.json`, `events.jsonl`, `memory.jsonl`, and `metrics.json`. Config schema version 3 references the per-run `experiment.json`, which resolves the shared workload snapshot or original source. It retains server, memory interval, host and execution settings.

Memory samples preserve main's `rss_mb` (MiB), `slots` (`id`, `busy`, `tokens`), and `busy_slots`, and record `slot_context_tokens_busy` / `slot_context_tokens_total` (formerly `kv_tokens_busy` / `kv_tokens_total`): sums of `/slots` `n_prompt_tokens`, which is `prompt.tokens.size()`, a token count, not KV cache allocation or memory use. The monitor adds the server's `srv_anon_mib` (RssAnon: resident anonymous memory such as KV cache and compute buffers, excluding mmap'ed weights), `srv_uss_mib` (USS, Private_Clean + Private_Dirty from `smaps_rollup`: pages only this process maps, which includes mmap'ed weights when no other process maps the GGUF), `srv_private_dirty_mib`, `srv_swap_mib`, OS peak RSS when available, system/swap memory, NVIDIA memory when available, and `srv_metrics`, with llama-server's `llamacpp:kv_cache_usage_ratio` and `llamacpp:kv_cache_tokens` copied to `kv_cache_usage_ratio` / `kv_cache_tokens` (null on builds from b5441 on, which no longer export them). The first sample records endpoint availability. On macOS, process RSS and CPU are collected through psutil; Linux-only `/proc` fields remain unavailable. RSS can be missing when process discovery or access fails; use `--server-pid` for a known local server. Sampling starts before task arrivals. Test mode and `--mem-interval 0` produce an empty memory file. Metrics accepts both this schema and historical PR samples (`srv_rss_mib`, `srv_slots` with `kv_tokens`, and `srv_private_mib`, which was RssAnon and is read as `srv_anon_mib`). The calibration report helper reads `slot_context_tokens_busy`, falling back to `kv_tokens_busy` for older files.

`metrics.json` includes successful-task completion/service distributions, queueing distributions, successful tasks per minute, task failures/timeouts/unfinished counts, request timings and cache reuse, concurrency timelines and sampled memory peaks. All offered tasks remain in the completion-rate denominator. Request distributions include returned requests; they do not measure unreturned partial computation. Completion latency covers successful tasks and must be read with failure rate. Small-run p95/p99 values are interpolated descriptive statistics, not stable population estimates. Semantic response quality is reviewed separately under the corpus rules.

`memory.slot_context` and `kv_cache.slot_context_tokens_*` report slot sequence lengths, including idle retained slot state. Shared prefixes can be counted multiple times, so the sums are not converted to physical KV bytes or occupancy percentages. `ctx_tokens_held` is an estimate from each active task's latest completed request, not measured residency or in-flight growth. TTFT is estimated from non-streaming latency minus decode time. `client_no_request_frac` describes client activity; after a timeout it does not prove backend idleness. RSS, anon, USS and system memory measure different things; the field definitions are Linux's and are saved as `host.process_memory_sources` in config.json.

Recompute a report or aggregate runs:

```bash
python metrics.py results/ACTUAL_RUN
python metrics.py 'results/ACTUAL_EXPERIMENT/c*_n*'
```

`experiment.py` selects the workload once and saves `workload.jsonl`. Every C reads that same snapshot and creates a per-run manifest recording its SHA-256 and ordered task IDs. It starts a fresh server per run, keeps total shared KV capacity and server slots fixed (defaults 8192 and 8), and uses the reference CPU/f16 settings (`-ngl 0`, so inference stays on the CPU even on a GPU host). It sends no warm-up inference request, so every run begins with fresh slot state; model loading is outside task makespan. Record this convention when comparing other experiments. Model and server paths can be overridden with `--model` and `--server-binary`.

```bash
# Small orchestration check: fake LLM, no inference server.
python experiment.py --test --combination SS -n 4 -C 1 2 --no-plot

# Real inference: explicit low-load sweep, after checking the small run.
LLAMA_ARG_LOG_VERBOSITY=4 python experiment.py --combination SS -n 4 -C 1 2 --timeout 1200

# Plot an existing experiment.
python experiment.py --plot-only --out-dir results/ACTUAL_EXPERIMENT
```

An experiment saves its manifest, per-C run folders, client/server logs, summary CSV and optional curves. Existing experiment directories are rejected to preserve results. Stop any manually started server on the chosen port before running orchestration. The previous experiment's `--shared-doc` and `--backend llama-py` modes are replaced by explicit workload references and the reference llama-server configuration. The workload is never modified to add artificial document headers during a sweep.


Integration checks (no model inference):

```bash
python -B -m unittest discover -s tests -v
```

The tests cover main/PR memory schema compatibility, timeout/error accounting, partial runs and mocked monitor endpoints. A fake-LLM four-task sweep at C=1/2 also checked identical task selections, FIFO start order, concurrency limits and 12 returned LLM calls / eight tool calls per run. These checks establish software integration; real server telemetry, output quality and capacity behavior still require an inference pilot.
