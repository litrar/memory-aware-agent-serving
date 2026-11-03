# CSE-291-Memory-Aware-Agent-Edge-Serving
CSE 291 Project 1

# experiment.py: Experiment orchestrator; agent.py: task-level FIFO runner

Use `experiment.py` as the entry point for a complete experiment. It saves the selected workload once in `workload.jsonl`, records experiment settings in `experiment.json`, starts a fresh local llama.cpp server for each requested concurrency, invokes `agent.py`, collects memory/server metrics, and writes per-run reports plus a comparison summary. `agent.py` owns the task-level FIFO queue and records the simulated tool calls as assistant tool calls and tool replies.

## Repository layout

| Path | Purpose |
|---|---|
| `experiment.py` | Experiment orchestrator: selects the workload, starts llama-server per C, runs `agent.py`, writes summaries/plots |
| `agent.py` | Task-level FIFO runner (one worker owns a whole multi-step task) |
| `monitor.py` / `metrics.py` | Linux memory/slot sampler and derived metrics |
| `quality.py` | Offline summary similarity scoring |
| `task-generator/` | Article source, calibration workloads, `agent-pressure.py` helper and calibration report |
| `tests/` | Integration tests (no model inference) |
| `scripts/` | Experiment sweep, plotting, and setup scripts |
| `infra/` | Optional Terraform for a single-GPU AWS EC2 host |

`models/`, `results/` and `third_party/llama.cpp/` are git-ignored and created locally.

## Setup

Linux is required for real runs: `monitor.py` reads `/proc` (`status`, `smaps_rollup`, `vmstat`). On Windows use WSL2; on macOS use a Linux VM or the EC2 host below. Use Python 3.12.

```bash
python -m pip install -r requirements.txt

# llama.cpp at the pinned calibration commit -> third_party/llama.cpp/build/bin/llama-server.
# CUDA is enabled when nvcc is found; --cpu forces a CPU build, --install-deps apt-installs the toolchain.
scripts/build-llama-cpp.sh

# Qwen3-4B Q8_0 weights -> models/Qwen3-4B-Q8_0.gguf (resumable, SHA-256 verified).
scripts/download-model.sh
```

`LLAMA_CPP_REF` builds a different llama.cpp commit/tag; keep the pinned one when comparing with the calibration results.

## Quick start

```bash
# Fake-LLM orchestration check; no server is started.
python -B experiment.py --test --combination SS -n 4 -C 1 2 --no-plot

# Real local inference; experiment.py starts and stops llama-server itself.
RUN_DIR="results/ss-real-$(date +%Y%m%d-%H%M%S)-$$"
LLAMA_ARG_LOG_VERBOSITY=4 python -B experiment.py \
  --combination SS -n 4 -C 2 --seed 0 --timeout 1200 \
  --mem-interval 0.5 --ctx-size 8192 --slots 8 --threads 4 \
  --out-dir "$RUN_DIR"
```

Run from the repository root with the project Python environment activated. The default model is `models/Qwen3-4B-Q8_0.gguf` and the default server binary is `third_party/llama.cpp/build/bin/llama-server`; override them with `--model` and `--server-binary`. The orchestrator sets the shared-library search path for Linux and macOS. Do not start a separate server for the `experiment.py` command.

The real-run command sets llama.cpp verbosity to 4 so its startup log includes allocated model, KV and compute buffer sizes for `metrics.py`. This pinned llama.cpp build supports `LLAMA_ARG_LOG_VERBOSITY`; the equivalent server CLI option is `-lv 4`. The normal default is 3, which omits those trace-level allocation lines.

The Phase 1 source and shared task settings come from `task-generator/source/phase1-articles.toml`. `-n` sets the sampled task count, `-C` sets one or more task concurrency values, and `--combination` selects a profile pattern such as `SS` or `SS-LL`. A sweep reuses one workload snapshot across C values and starts a fresh server for each C. `--long-tier` only selects the document pool for `L` input tasks.

For a one-off run against a server you started yourself, invoke `agent.py` directly with `--workload` and lowercase `-c`. The experiment sweep uses uppercase `-C` and manages server lifecycle for you.

> The server should have at least `-c` parallel slots. Otherwise requests queue up inside the server, and we can no longer measure the real queueing time (the project requires queueing to happen in *our* queue).

## All 16 conditions

`scripts/H_conditions.sh` uses `common.sh` helpers adapted from the teammate's experiment scripts to call `experiment.py` sequentially for SS/SL/LS/LL × two tool delays × prompt cache on/off. The default is **16 runs at C=2**, six total tasks per run, seed 0, near-tier long articles, and 0 s / 5 s tool delays. Choose delays appropriate for the measured inference speed. Every run starts a fresh server with the same CPU/f16 KV configuration. For another C, run the group again with a new timestamp; two C values give 32 measurements.

From the repository root with its Python environment activated, preview the commands before running:

```bash
DRY_RUN=1 bash scripts/H_conditions.sh
```

On the shared EC2 host with the existing model and llama.cpp paths, after the current experiment finishes:

```bash
bash scripts/H_conditions.sh

# Optional overrides; defaults use the existing EC2 model/server paths in common.sh.
N=4 C=2 SHORT_DELAY=0 LONG_DELAY=5 TS=pilot-1 bash scripts/H_conditions.sh
```

Use `TEST=1 N=1 C=1 SHORT_DELAY=0 LONG_DELAY=0.001 bash scripts/H_conditions.sh` for a short fake-LLM check, including charts; it does not validate inference, real memory telemetry or cache behavior. `LLAMA_SERVER`, `MODEL`, `PYTHON`, `C`, `SEED`, `CTX_SIZE`, `SLOTS`, `THREADS`, `PORT`, `RESULTS` and `TS` are configured through environment variables. `N`, `SHORT_DELAY`, `LONG_DELAY` and `LONG_TIER` configure H. Run serving experiments alone on the host, even when ports differ, to avoid resource contention. Run inside `tmux` if the sweep should continue after disconnecting SSH.

The shared helper defaults server log verbosity to 4. Outputs go to `results/H-conditions-<TS>/`:

- `matrix.json`: the 16 conditions, shared command arguments, workload hash and code revision. Resume with the same `TS` and settings to skip finished conditions and move interrupted attempts aside. Changed settings/source/code require a new `TS`.
- `<condition>.terminal.log`: execution output, also printed to the terminal.
- `<condition>/`: the existing experiment workload snapshot, per-C server/agent logs, events, sampled memory, configs, `metrics.json` and `summary.csv`.
- `summary_all.csv`: one row per finished condition, with combination, actual delay, cache setting and concurrency. Incomplete experiments are excluded. Individual failures are logged in `failed.txt` and the remaining conditions continue; H returns an error if the final matrix or charts are incomplete. Task failures/timeouts inside a finished experiment remain measurements.
- `curves_group.png` and `timelines/<condition>.png`: generated by `scripts/plot_group.py`, adapted from the teammate's plotter and preserving its metrics and visual style. The comparison figure widens for 16 conditions. To redraw without inference: `python -B scripts/plot_group.py results/H-conditions-<TS>`.

Within each SS/SL/LS/LL profile, task content and order are identical across delay/cache settings at the same seed/count. SS and SL share documents; LS and LL share documents. Short-context and long-context profiles use different articles, so their comparison also includes topic differences. Repeat across seeds for an evaluation; a small pilot does not establish stable tail latency.

Cache on/off means the request's `cache_prompt` setting: on reuses matching prompt prefixes when available; off recomputes them. It does **not** disable attention's KV cache, release idle contexts, or reduce the preallocated KV buffer. Check actual reuse in the recorded token timings. The default remains on for existing commands. Final responses remain in `events.jsonl`; run `quality.py` separately after serving finishes.

## Direct runner arguments (`agent.py`)

The experiment-only flags (`-C`, `--ctx-size`, `--slots`, `--threads`, `--model`, `--server-binary`, `--host`, `--client-host`, `--port`, `--plot-only` and `--no-plot`) are documented by `python experiment.py --help`.

| Flag | Default | Description |
|---|---|---|
| `-c` | 2 | Concurrency: number of workers (max tasks running at the same time) |
| `-n` | 6 | Number of tasks sampled from the article source (or legacy docs); ignored for manifests or fixed JSONL tasks |
| `--workload` | required | Experiment manifest JSON, fixed task TOML/JSONL, or article-source TOML |
| `--combination` | `SS-LL` in source | Profile pattern defined and explained in source `task_settings.combinations` |
| `--long-tier` | `near` | Pool for L context: `near` (~2,000 tokens) or `overload` (~2,600) |
| `--seed` | 0 | Seed for reproducible article sampling without replacement (or legacy document selection) |
| `--timeout` | 120 | HTTP request deadline in seconds; baseline command explicitly uses 1200 |
| `--test` | off | Use the fake LLM (`test_llm`); no server needed |
| `--out-dir` | `results` | Output directory |
| `--run-name` | `c<C>_n<N>_<timestamp>` (plus `_test` in test mode) | Unique run folder; existing names are rejected. |
| `--url` | `http://127.0.0.1:8080` | llama-server endpoint |
| `--mem-interval` | 0.5 | Memory sampling interval in seconds; 0 disables sampling |
| `--server-pid` | auto | Server PID; otherwise detect the listener on `--url` |
| `--tool-delay` | workload `tool_delay_s` | Override the simulated tool wait (seconds). One value applies to every tool call; several values set each tool call separately and must match `tool_calls`, e.g. `--tool-delay 0.5 3` = tool 1 waits 0.5 s, tool 2 waits 3 s. Recorded as a task override in `experiment.json`, so replay keeps it. `experiment.py` accepts the same flag |
| `--cache-prompt` | `on` | `on`/`off`: reuse matching prompt prefixes or recompute them. Does not change KV buffer allocation. Also accepted by `experiment.py`; recorded in config/metrics and the experiment manifest |

## How it works

```
producer ──put──▶ asyncio.Queue (FIFO) ──get──▶ worker 0..C-1 ──▶ run_agent ──▶ llama.cpp
```

- **producer**: puts all tasks into the queue at t≈0 (file order), then puts C `None` values to tell the workers to stop.
- **worker**: takes a task from the queue, runs `run_agent`, and only takes the next task once that one finishes. So C is the concurrency limit, and a worker stays occupied during tool waits (task-level scheduling).
- **run_agent** (one task): `tool_calls + 1` LLM calls on one growing conversation. With the Phase 1 settings (`tool_calls = 2`):

  ```
  [system] system_prompt
  [user]   step_instructions[0] + document
  [assistant] reply 0  + tool_calls(load_checklist, step 0)   ← LLM call 0
  [tool]   tool_results[0]                                     ← 1 s asyncio.sleep
  [user]   step_instructions[1]
  [assistant] reply 1  + tool_calls(load_checklist, step 1)   ← LLM call 1
  [tool]   tool_results[1]                                     ← 1 s asyncio.sleep
  [user]   step_instructions[2]
  [assistant] final answer                                     ← LLM call 2 (final_tokens)
  ```

  The tool is just `asyncio.sleep` returning a fixed checklist from the workload. It models a tool wait while the client keeps the conversation; every request resends the full history. With the default `cache_prompt: True`, later calls can reuse the prefix already in the slot's KV cache.

### Task settings

The Phase 1 article source sets three LLM calls, two one-second simulated tools, short responses capped at 192 tokens and long responses capped at 768 tokens per call. Request-level scheduling is a Phase 2 candidate, not part of this baseline.

### Request settings to be aware of (`chat()`)

- `temperature: 0` and `seed: llm_seed`: reduce variation; batching and cache state may still change generated output.
- `ignore_eos: False`: responses stop naturally; caps are upper bounds, so record actual output lengths and finish reasons.
- `enable_thinking: False`: disables Qwen3's `<think>` block.
- `cache_prompt: True` by default (`--cache-prompt on`): reuse the matching prompt prefix in the slot. `--cache-prompt off` sends `false` to recompute prompts.

## Output

Each runner execution creates `results/<run>/`; `experiment.py` adds the shared experiment manifest, server log, summary CSV, and optional plot in its output folder:

- **`experiment.json`** (experiment root): experiment settings, runner, server command, and snapshot path.
- **`workload.jsonl`** (experiment root): the complete selected tasks, shared across concurrency runs.
- **`experiment.json`** (per-run folder): `workload_spec` references the shared snapshot with SHA-256 and ordered task IDs. Direct TOML runs reference their source instead.
- **`config.json`**: schema version 3 records execution settings, ordered `task_ids`, and a relative `workload_manifest` path to `experiment.json`, plus model, host and server information. The runner creates a local `experiment.json` when its input is TOML/JSONL.
- **`events.jsonl`**: one event per line. `t` is seconds since the program started.
- **`memory.jsonl`**: sampled server/system memory plus llama-server `/metrics` and `/slots` data when available.
- **`metrics.json`**: derived task, request, throughput, concurrency, and memory metrics.

| event | Extra fields |
|---|---|
| `task_arrive` | `task_id` |
| `task_start` | `task_id`, `worker`, `doc_id` |
| `llm_start` | `task_id`, `step` |
| `llm_end` | `latency_s`, `prompt_tokens`, `completion_tokens`, `prompt_n` (prompt tokens actually recomputed), `cache_n` (tokens served from cache), `prompt_ms`, `predicted_ms` |
| `tool_start` / `tool_end` | `task_id`, `step`, `delay_s` |
| `task_end` | `status` (`ok` / `timeout` / `error`), `error`, `summary` (final summary text) |

**Model answers**: the final summary is in the `summary` field of `task_end`. Intermediate replies are logged in `llm_end.text`, with usage and `finish_reason`.

Keep the referenced workload snapshot (or source for direct TOML runs) with the results: its SHA-256 must match for replay or quality evaluation. Configs do not duplicate task content. Source paths are relative to the manifest, with a repo-relative fallback for results moved to another machine. A hash mismatch stops loading; restore the original corpus version rather than updating the recorded hash. Existing results and legacy JSONL inputs remain readable.

Replay the recorded selection without resampling:

```bash
python -B agent.py --workload results/ACTUAL_EXPERIMENT/c2_n4/experiment.json -c 2 --timeout 1200
```

For offline quality scoring, resolve the manifest and join the reconstructed tasks to `events.jsonl` by `task_id`:

```python
import json
from pathlib import Path
import agent

run = Path("results/ACTUAL_EXPERIMENT/c2_n4")
config = json.loads((run / "config.json").read_text())
tasks = agent.load_workload(run / config["workload_manifest"])
by_id = {task["task_id"]: task for task in tasks}
# by_id[task_id] provides doc, required_facts, forbidden_claims, and validation_rules.
```

After the run, the terminal prints a summary:

```
task  queue_s  service_s  completion_s     # queueing time / service time / total completion time
FIFO(開始順序 = 到達順序）: True              # start order == arrival order
同時在跑的任務最多 2 個（上限 C=2）: True      # peak concurrent tasks <= C
總時間 ...s                                   # total time
```

## Example: reading the results

```python
import json
ev = [json.loads(l) for l in open("results/ACTUAL_RUN/events.jsonl")]
llm = [e for e in ev if e["event"] == "llm_end"]
print("mean LLM latency:", sum(e["latency_s"] for e in llm) / len(llm))
print("cached tokens:", [e["cache_n"] for e in llm])
for e in ev:
    if e["event"] == "task_end":
        print(e["task_id"], e["status"], e["summary"])
```

## Troubleshooting

- **`ConnectError` / every task is `error`**: the server isn't running, or `--url` points to the wrong host/port.
- **`timeout`**: a single request took longer than `--timeout`. Usually C is too large or the context is too long; CPU runs need ~1200.
- **`HTTP 400` mentioning context size**: the shared KV pool (`--ctx-size`) is full for the current C and document length.
- **Small queue_s but large latency_s**: the server probably has fewer slots than C, so requests are queueing inside the server.
- **Tool messages rendered as plain text / template error**: start llama-server with `--jinja`.
- **If you add a simulated tool, use `await asyncio.sleep`**, not `time.sleep`. Otherwise every worker blocks together.

## Offline summary similarity score

`quality.py` scores saved responses locally with Sentence Transformers and cosine similarity. It loads a small embedding encoder itself; no judge server, API key, or manual submission is needed. Install the project dependencies once:

```bash
python -m pip install -r requirements.txt
python -B quality.py results/PHASE1/c2_n4 results/PHASE2/c2_n4 \
  --out-dir results/quality-comparison-01
```

The first run downloads `sentence-transformers/all-MiniLM-L6-v2` at the pinned revision in `quality.py`; later runs reuse cached weights. After downloading, `HF_HUB_OFFLINE=1` can enforce offline operation. Evaluation runs on CPU after serving experiments finish. Keep the encoder revision, method and dependency versions fixed across phase comparisons.

For each required fact, find the highest cosine similarity among the answer's single-sentence and adjacent two-sentence spans. Clip each similarity to [0,1], average across required facts, then multiply by 100. This is `semantic_score` (0–100), **a similarity measure, not a factual correctness percentage**. Each task's final score is zero if execution failed or the final answer is missing. `summary.csv` reports `mean_task_score` over ALL selected tasks, `returned_final_score_mean` over returned final answers, and length/normal-stop compliance separately. No arbitrary factual-pass threshold is applied.

`responses.jsonl` stores per-fact similarities, matching response spans and numeric/negation diagnostics; `tasks.jsonl` stores final task scores. The first input run is the baseline. `summary.csv` includes score differences, and `comparison.csv` pairs task IDs and reports individual score changes. Comparing multiple runs requires identical resolved task inputs/settings/order; also keep server configuration and concurrency fixed when comparing phases. Differences are descriptive, not a statistical non-inferiority verdict.

Cosine similarity can remain high after a wrong number, changed unit, reversed negation or added unsupported claim. Literal missing/unmatched numeric tokens (including currency/percent symbols) and differing negation markers are warnings, not proven errors or automatic score penalties; spelled-out numbers and paraphrased negation may trigger false alarms. Review these alongside scores. Full factual acceptance still follows the corpus's semantic-review rules. This scorer does not certify absence of hallucinations.

`evaluation.json` records method, pinned encoder revision, package versions and input hashes. Only `complete=true` indicates finished evaluation. Source hashes are verified for manifests; older `workload.jsonl` inputs are supported. Existing output directories and fake-LLM runs are rejected.

## Task generator and workload calibration

This section describes the task-generator contribution and its calibration helper. For these fixed workloads, current runner behavior is natural EOS (`ignore_eos=false`), temperature=0, prompt reuse enabled, and unique run names that refuse overwrites. Fixed task parameters come from the workload file. Temperature=0 and a seed reduce variability but do not guarantee identical outputs across batching/cache configurations.

Files:

- `task-generator/calibration/pilot-v1.toml`: eight readable fixed tasks (two cases × short/long input × short/long output). `doc` and instructions use multiline strings. `required_facts` stay outside model input.
- `task-generator/calibration/context-calibration.toml`: six saved expanded articles, with unique paragraphs within each article. Scales target document tokens, not words or total prompt length.
- `task-generator/calibration/boundary-v1.toml`: controlled four-case variants at approximately 1,600/1,800/2,000 document tokens. They retain two factual scenarios with different headers/supporting sections; they are not independent real-world sources.
- `task-generator/agent-pressure.py`: selects fixed tasks and calls the existing FIFO runner. It does not generate documents during timed inference.
- [task-generator/context-calibration-report.md](task-generator/context-calibration-report.md): measurements, recommendations, quality issues and limitations. Raw logs stay in ignored `results/` and are not included in Git.

Use Python 3.12 with the project's Python environment activated, then install the dependency:

```bash
python -m pip install -r requirements.txt
```

### Reference server (terminal A)

The calibration uses Qwen3-4B Q8_0 weights (`models/Qwen3-4B-Q8_0.gguf`), llama.cpp commit `797de63b2fa7396dcc8caca76834c699ac84e1c3`, CPU inference, four threads and f16 KV. Run from the repo root:

```bash
mkdir -p results/server-logs
./third_party/llama.cpp/build/bin/llama-server \
  -m models/Qwen3-4B-Q8_0.gguf \
  --ctx-size 8192 -np 8 --kv-unified \
  -ngl 0 -t 4 -tb 4 -ctk f16 -ctv f16 \
  --fit off --jinja --reasoning off --no-context-shift -lv 4 \
  --metrics --no-webui --port 8080 \
  2>&1 | tee "results/server-logs/$(date +%Y%m%d-%H%M%S).log"
```

Build llama.cpp on the execution host with `scripts/build-llama-cpp.sh` and report hardware separately. `-ngl 0` keeps inference on the CPU to match the calibration; on a GPU host, `-ngl 99` offloads all layers, but that is a different configuration from these results. Restart this **server process** before every independent run (Ctrl+C and repeat the command). This resets retained slot/KV state between runs; keep normal prompt reuse within a task/run. It does not reset OS page caches or thermal state.

### Select and run a fixed workload (terminal B)

```bash
curl http://127.0.0.1:8080/health
python task-generator/agent-pressure.py --scale 2 --profile LS --check-only
python task-generator/agent-pressure.py --scale 2 --profile LS -c 2 --timeout 600
```

`--check-only` uses the actual tokenizer and chat template without inference. For SS/SL use `--scale 1` (original short documents). For LS/LL, scale=1 means the original long document, and scale=2/3/4 selects an expanded article. SS/LS request short responses with a 192-token cap; SL/LL request long responses with a 768-token cap. Each task has three LLM steps and two one-second tool delays.

The original helper selects two tasks, so C>2 would not create extra concurrency. For a fair C=1/2/3/4 sweep, use the same four selected boundary tasks for every run. For example, after inspecting the report choose a tested token target:

```bash
python task-generator/agent-pressure.py --workload task-generator/calibration/boundary-v1.toml --target-tokens 2000 --profile LL --check-only
python task-generator/agent-pressure.py --workload task-generator/calibration/boundary-v1.toml --target-tokens 2000 --profile LL -c 1 --timeout 600
# Restart terminal A's server, then change only -c:
python task-generator/agent-pressure.py --workload task-generator/calibration/boundary-v1.toml --target-tokens 2000 --profile LL -c 2 --timeout 600
# Repeat with -c 3 and -c 4, restarting the server before each run.
```

For two-task input-length bracketing add `--limit 2` and compare `--target-tokens 1600`, `1800`, `2000` at C=2. Do not compare completion counts between different task counts as if only concurrency had changed. The helper rejects C larger than the selected task count. `--reverse` is available for repeated trials with reversed arrivals; `--server-pid PID` overrides RSS PID discovery; `--url URL` changes the endpoint.

### Save, inspect and interpret results

A unique run is saved under `results/pressure-demo/`, with a workload snapshot and `.preflight.json`, plus `config.json`, `events.jsonl`, `memory.jsonl` in the run directory. Intermediate answers are saved in each `llm_end.text`; final answers are in `task_end.summary`. The helper prints the actual run path and a replay command:

```bash
python task-generator/agent-pressure.py --report "results/pressure-demo/ACTUAL_RUN_NAME" --show-answers
```

Record **completed tasks / offered tasks**, elapsed time, successful-task throughput, every request's prompt/output/cached tokens and finish reason, and server memory. A shell exit code of zero does not imply that all tasks succeeded. Retain failed tasks, their intermediate answers and short/truncated outputs. `stop` is a normal end; `length` means the output cap was reached; `timeout` is per HTTP request. Actual output length and factual accuracy need separate review.

Document tokens are not full prompt context. The initial prompt adds system/instructions/template; later prompts add previous replies and tool/instruction content. Cached tokens are part of the prompt, not extra tokens. `/slots` sums are logical sequence counts, may double-count shared prefixes, and busy-only sums omit idle retained caches. RSS includes weights and other allocations; neither is exact unique KV bytes. Context-capacity failure is not automatically physical RAM OOM. This pinned build allocates and clears the configured KV buffer during initialization, so longer prompts can consume more of the existing pool without proportional RSS growth. Track allocated KV capacity, logical sequence occupancy, process RSS and task outcomes separately.

For new documents, use token targets from the report rather than accepting word count alone. Keep a low-load short-input control, a normal long-input case, and an explicitly labeled near-capacity/over-capacity case. A failure observed at ~2,200 document tokens with long responses and C=2 is a sampled deployment boundary, not a universal model limit. It depends on multi-step generated output, concurrency and the KV pool. Freeze task count/content/order, arrival schedule, cache policy, output instructions/caps and server settings when comparing schedulers. Use separate evaluation documents after tuning, and preserve quality errors instead of filtering them out.


### Measured starting points and deadline confirmation

For this CPU / f16 KV / 8,192-token configuration, use short documents near **200 tokens** (about 130–180 English words). A **2,000-token** long document (about 1,600–1,750 English words) plus three long responses puts C=2 near the shared KV boundary in this controlled four-task set. Keep approximately 1,100-token long documents as a more conservative control. Tokenize new documents; word counts alone are insufficient.

| C | Request timeout s | Completed | Context errors | Timeouts | Peak logical busy tokens |
|---:|---:|---|---:|---:|---:|
| 1 | 600 | 4/4 | 0 | 0 | 4018 |
| 2 | 600 | 4/4 | 0 | 0 | 7877 |
| 3 | 600 | 0/4 | 2 | 2 | 10063 |
| 3 | 1200 | 1/4 | 3 | 0 | 8244 |
| 4 | 600 | 0/4 | 4 | 0 | 8192 |

The table records the measured 600-second sweep plus the separate 1,200-second C=3 confirmation. To reproduce the confirmation, restart the server and run:

```bash
python task-generator/agent-pressure.py --workload task-generator/calibration/boundary-v1.toml --target-tokens 2000 --profile LL -c 3 --timeout 1200
```

For a new capacity-only sweep on a slow CPU, use the same generous request deadline (e.g. 1200) for every C. Keep the original deadline-constrained results as well; do not silently replace failures. Client timeout may free a worker before backend inference has stopped: the original C=3 trial briefly had four server busy slots, so client C is not always the backend request count during failure handling.

These are calibration settings, not quality or SLA guarantees. C=1 is a conservative control; C=2 is the tested near-capacity LL condition; higher C are overload cases for this corpus. Actual generation varies, slot sums are not physical KV utilization, and observed timing varied substantially. Preserve all FIFO C settings, including the best successful baseline, for later scheduler comparisons. See the report for exact outcomes, quality defects and raw run paths.

### Phase 1 candidate articles and quality rules

Task-generator files are grouped under `task-generator/`. The formal article source and its shared task settings are in `source/phase1-articles.toml`; the older pilot and capacity-calibration inputs used to reproduce the report are in `calibration/`. `agent.py` reads the article source and saves task references in a manifest; it reconstructs task content only in memory. `agent-pressure.py` and the calibration report document earlier capacity experiments. The experiment entry point uses `agent.py` as its task-level FIFO runner.

`task-generator/source/phase1-articles.toml` contains **30 different synthetic articles**, ten per input tier: short (~200 document tokens), near-capacity candidate (~2,000), and overload candidate (~2,600). Each article is saved once with its measured token count, four required facts, exact supporting quotes and forbidden-claim examples. Its `[task_settings]` section holds the shared system prompt, tool behavior, short/long response instructions and caps, and quality-review rules. `agent.py` reads this TOML directly and creates the runner tasks in memory. They share a controlled report style; they are not natural-source benchmark documents. Capacity labels refer to the reference 8,192-token shared-KV deployment with multi-step long responses, not guaranteed success/failure or the model's context-window definition. This corpus has passed input-token checks; full inference and response-quality acceptance are still pending.

The article token counts were measured once with the pinned Qwen3-4B Q8_0 tokenizer and are stored in the source file. Running the fixed corpus does not repeat token counting. If an article's text changes, recount that changed document as part of validating the edit. Full multi-step capacity and response quality still require inference.

Run the article source directly (from the repo root):

```bash
python agent.py --workload task-generator/source/phase1-articles.toml -c 1 --timeout 1200
```

The default samples six distinct articles in `SS,LL,SS,LL,SS,LL` order, using near-pool long documents and C=2 unless overridden. Each task has three LLM calls and two one-second simulated tools. All six arrive together; workers start only C tasks, then take the next FIFO task on completion. `-n` changes the sampled count; mixed presets require an even count to preserve equal class sizes. Insufficient unique articles is an error rather than silently repeating articles. The experiment manifest records the exact article/profile selection and order, source hash, and settings overrides.

Preset definitions and experiment-design comments live together in `task_settings.combinations` in the source TOML:

| `--combination` | Six-task arrival order | Main comparison |
|---|---|---|
| `SS`, `SL`, `LS`, `LL` | Six tasks of the named profile | Uniform controls; SS vs SL and LS vs LL hold documents fixed at the same seed |
| `SS-SL` | SS, SL, SS, SL, SS, SL | Short inputs, mixed response budgets |
| `SS-LS` | SS, LS, SS, LS, SS, LS | Short-response settings, mixed input categories |
| `SS-LL` | SS, LL, SS, LL, SS, LL | Default mixed light/heavy workload |
| `LL-first` | LL, LL, LL, SS, SS, SS | Heavy tasks first; inspect FIFO waiting for short tasks |
| `SS-first` | SS, SS, SS, LL, LL, LL | Paired order control for LL-first |

For a manual smoke test, start the reference server in terminal A, then run in terminal B:

```bash
curl http://127.0.0.1:8080/health
python agent.py --workload task-generator/source/phase1-articles.toml --combination SS -c 2 --timeout 300
```

This now samples six short-context/short-response tasks (18 LLM calls); the earlier ten-task smoke test took about five minutes on CPU. This checks source loading and inference; assess response quality separately.

Change only the selection arguments to test other cases:

```bash
python agent.py --workload task-generator/source/phase1-articles.toml --combination LL -c 2 --timeout 1200
python agent.py --workload task-generator/source/phase1-articles.toml --combination LL --long-tier overload -c 2 --timeout 1200
python agent.py --workload task-generator/source/phase1-articles.toml --combination LL-first --seed 0 -c 2 --timeout 1200
python agent.py --workload task-generator/source/phase1-articles.toml --combination SS-first --seed 0 -c 2 --timeout 1200
```

At the same seed/count/long-tier, `LL-first`, `SS-first` and `SS-LL` use the same selected tasks, changing only their order. SS vs SL and LS vs LL use the same documents with different response settings. SS vs LS uses different source articles, so topic differences prevent a pure length-only causal claim. Keep the source, selection arguments, model settings and deadline fixed when varying concurrency/schedulers; repeat across seeds rather than relying on a single draw. Restart the server before every independent run. Manifest replay is available: `python agent.py --workload results/ACTUAL_EXPERIMENT/c2_n4/experiment.json -c 2 --timeout 1200`. Older `workload.jsonl` files can still be used as inputs.

Short response instructions request 60–90 English words with a 192-token cap; long response instructions request 300–400 words with a 768-token cap. Natural EOS and disabled thinking remain unchanged. Caps are upper limits, not target-length guarantees. Required facts, evidence quotes and review rules are excluded from model input; subsequent tools provide procedural checklists, not reference answers.

Review each returned step against its article. Record required-fact recall, contradictory claims, unsupported factual claims, word-length compliance and truncation separately. Factual pass requires all four facts and zero contradictory/unsupported factual claims; strict pass additionally requires length compliance and `finish_reason=stop`. Preserve every timeout, context error and short/truncated answer; execution failures cannot pass. These semantic checks require human review (or a separately validated judge); exact keyword matching alone is insufficient. See the report for the scoring protocol and corpus limitations.


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

## Optional: AWS GPU host (`infra/`)

`infra/main.tf` creates one EC2 instance (default `g4dn.xlarge`, one NVIDIA T4) from the AWS Deep Learning Base GPU AMI (Ubuntu 22.04, NVIDIA driver and CUDA preinstalled) in the default VPC. The security group allows SSH only, from `ssh_cidr` and from the region's EC2 Instance Connect range; llama-server stays on localhost and is reached through an SSH tunnel.

Prerequisites: Terraform >= 1.5, AWS credentials (`aws configure`), an EC2 key pair in the target region, and a vCPU quota of at least 4 for "Running On-Demand G and VT instances" (spot runs need the separate spot quota). AWS Free-plan accounts can only launch Free Tier instance types; upgrade the account plan before launching a GPU instance.

```bash
cd infra
terraform init
terraform apply -var="key_name=<your-key-pair>" -var="ssh_cidr=$(curl -s https://checkip.amazonaws.com)/32"
terraform output ssh          # ssh command with -L 8080:localhost:8080
```

| Variable | Default | Notes |
|---|---|---|
| `key_name` | required | Existing EC2 key pair name |
| `region` | `us-east-2` | Must match where the key pair exists and where your account may launch EC2 |
| `instance_type` | `g4dn.xlarge` | |
| `ssh_cidr` | `0.0.0.0/0` | Narrow to your own IP (`x.x.x.x/32`) |
| `use_spot` | `false` | Cheaper but interruptible; needs spot quota |

On the instance, clone the repository and follow [Setup](#setup) (`scripts/build-llama-cpp.sh --install-deps` builds with CUDA). Copy results back before tearing down; `terraform destroy` deletes the root volume:

```bash
scp -r -i <path-to-key.pem> ubuntu@<instance-ip>:CSE-291-Memory-Aware-Agent-Edge-Serving/results ./results-ec2
terraform destroy -var="key_name=<your-key-pair>"   # pass the same -var values used for apply
```

GPU instances bill while running. `terraform.tfstate*`, `.terraform/` and `*.tfvars` hold account-specific data and are git-ignored; do not commit them, key files or AWS credentials.
