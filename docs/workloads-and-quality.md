# Workloads and quality

Run all commands from the repository root.

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
- [task-generator/context-calibration-report.md](../task-generator/context-calibration-report.md): measurements, recommendations, quality issues and limitations. Raw logs stay in ignored `results/` and are not included in Git.

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

`task-generator/source/phase1-articles.toml` contains **30 different synthetic articles**, ten per input tier: short (~200 document tokens), near-capacity candidate (~2,000), and overload candidate (~2,600). Each article is saved once with its measured token count, four required facts, exact supporting quotes and forbidden-claim examples. Its `[task_settings]` section holds the shared system prompt, tool behavior, short/long response instructions and caps, and quality-review rules. `agent.py` reads this TOML directly and creates the runner tasks in memory. They share a controlled report style; they are not natural-source benchmark documents. Capacity labels refer to the reference 8,192-token shared-KV deployment with multi-step long responses, not guaranteed success/failure or the model's context-window definition. The corpus has passed input-token checks. H1 ran selected short and near-tier articles; the remaining articles and overload tier have not all received full inference and quality acceptance.

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
