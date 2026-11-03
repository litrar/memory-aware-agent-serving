# Memory-Aware Agent Serving

A system for studying how admission and scheduling affect multi-step agent workloads under a limited memory and context budget.

The project measures task completion, latency, throughput, memory usage, and output quality while varying workload characteristics and serving conditions.

## Current baseline

The Phase 1 baseline uses a **task-level FIFO queue** with fixed concurrency. Each task contains three dependent LLM requests and two simulated tool calls. A worker remains assigned to its task during tool waits.

Phase 1 experiments use llama.cpp with Qwen3-4B Q8_0. Request-level scheduling is a possible Phase 2 direction; it is not part of the current baseline.

## Workloads and measurements

The workload combines short and long inputs with short and long response settings:

| Profile | Input | Response |
|---|---|---|
| `SS` | Short | Short |
| `SL` | Short | Long |
| `LS` | Long | Short |
| `LL` | Long | Long |

Experiments can vary task concurrency, simulated tool delay, and prompt-cache settings. Workload snapshots preserve the selected tasks and their order for reproducible comparisons.

The runner records task and request events, completion and queueing times, throughput, failures, and available memory measurements. Output quality can be evaluated separately using `quality.py`; its fact-similarity score does not establish factual correctness on its own.

## Repository layout

| Path | Purpose |
|---|---|
| `experiment.py` | Runs experiment sweeps and manages the llama.cpp server |
| `agent.py` | Runs multi-step tasks using task-level FIFO |
| `monitor.py` / `metrics.py` | Collect memory samples and compute experiment metrics |
| `quality.py` | Scores saved summaries against required facts |
| `task-generator/` | Contains the article workload and calibration materials |
| `scripts/` | Contains experiment sweep, plotting, and setup scripts |
| `tests/` | Contains integration tests |
| `infra/` | Contains optional AWS infrastructure configuration |
| `docs/` | Contains experiment, workload, metrics, and setup guides |

`models/`, `results/`, and `third_party/llama.cpp/` are ignored by Git and created locally.

## Setup

Use Python 3.12. Real inference and Linux-specific memory measurements require a Linux environment. From the repository root:

```bash
python -m pip install -r requirements.txt
```

Build llama.cpp and download the model using the setup scripts:

```bash
scripts/build-llama-cpp.sh
scripts/download-model.sh
```

## Quick start

Run a small orchestration check with a fake LLM. This does not start a model server:

```bash
python -B experiment.py --test --combination SS -n 4 -C 1 2 --no-plot
```

Run the integration tests:

```bash
python -B -m unittest discover -s tests -v
```

For real inference, `experiment.py` starts and stops a local llama.cpp server for each concurrency setting. See the [experiment guide](docs/experiments.md) for configuration, output files, and replay.

## Runner arguments

These arguments apply to `agent.py`. The experiment-only flags (`-C`, `--ctx-size`, `--slots`, `--threads`, `--model`, `--server-binary`, `--host`, `--client-host`, `--port`, `--plot-only` and `--no-plot`) are documented by `python experiment.py --help`.

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

## Documentation

- [Experiment guide](docs/experiments.md) — real inference, H1 condition matrix, outputs, and replay.
- [Workloads and quality](docs/workloads-and-quality.md) — article corpus, calibration, and quality scoring.
- [Metrics and memory](docs/metrics-and-memory.md) — collected metrics and interpretation.
- [AWS setup](docs/aws-setup.md) — optional infrastructure and host setup.
