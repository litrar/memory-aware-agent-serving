"""Offline fact-coverage similarity scores (0–100); no LLM judge or API required."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
from decimal import Decimal
from pathlib import Path

import agent

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
METHOD = 'fact-coverage-cosine-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def spans(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])|\n+', text) if s.strip()]
    # Bound sentence lengths; never embed an entire long summary as one vector.
    pieces = []
    for sentence in sentences:
        words = sentence.split()
        pieces.extend(' '.join(words[i:i+60]) for i in range(0, len(words), 60))
    return pieces + [' '.join(pieces[i:i+2]) for i in range(len(pieces)-1)]


def numbers(text):
    # Literal numeric/currency/percentage tokens only; spelled-out numbers are not parsed.
    return {prefix + str(Decimal(value.replace(',', '')).normalize()) + suffix
            for prefix, value, suffix in re.findall(r'([$£€]?)(\d[\d,]*(?:\.\d+)?)(%?)', text)}


def negative(text):
    return bool(re.search(r"\b(no|not|cannot|never|without|absence|unable)\b|n't\b", text.lower()))


class CosineScorer:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL, revision=REVISION, device='cpu')
        self.model.eval()

    def __call__(self, facts, candidates):
        texts = facts + candidates
        lengths = self.model.tokenizer(texts, truncation=False)['input_ids']
        if any(len(ids) > self.model.max_seq_length for ids in lengths):
            raise ValueError('text span exceeds encoder context; refusing silent truncation')
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return (vectors[:len(facts)] @ vectors[len(facts):].T).tolist()


def score_response(task, event, scorer):
    candidates = spans(event['text'])
    facts = task['required_facts']
    if not facts:
        raise ValueError('required_facts must not be empty')
    matrix = scorer([f['claim'] for f in facts], candidates) if candidates else [[] for _ in facts]
    matches = []
    for fact, row in zip(facts, matrix):
        best = max(range(len(row)), key=row.__getitem__) if row else None
        evidence = candidates[best] if best is not None else ''
        similarity = max(0., min(1., float(row[best]))) if best is not None else 0.
        matches.append(dict(id=fact['id'], similarity=similarity, response_span=evidence,
            missing_numeric_tokens=sorted(numbers(fact['claim']) - numbers(evidence)),
            negation_marker_differs=negative(fact['claim']) != negative(evidence)))
    words = len(event['text'].split())
    low, high = task['target_words']
    return dict(task_id=task['task_id'], step=event['step'],
        semantic_score=100 * sum(m['similarity'] for m in matches) / len(matches), facts=matches,
        words=words, length_pass=low <= words <= high, finish_reason=event.get('finish_reason'),
        unmatched_numeric_tokens=sorted(numbers(event['text']) - numbers(task['doc'])))


def evaluate_run(run, scorer):
    cfg = json.loads((run / 'config.json').read_text())
    if cfg.get('test'):
        raise ValueError(f'{run}: fake LLM replies cannot establish summary quality')
    source = run / cfg['workload_manifest'] if cfg.get('workload_manifest') else run / 'workload.jsonl'
    tasks = agent.load_workload(source)
    by_id = {t['task_id']: t for t in tasks}
    events = read_jsonl(run / 'events.jsonl')
    responses, ends = {}, {}
    for event in events:
        tid = event.get('task_id')
        if tid is not None and tid not in by_id:
            raise ValueError(f'event task absent from workload: {tid}')
        if event['event'] == 'llm_end':
            step = event['step']
            key = (tid, step)
            if key in responses or type(step) is not int or not 0 <= step <= by_id[tid]['tool_calls']:
                raise ValueError(f'duplicate or invalid response step: {key}')
            responses[key] = score_response(by_id[tid], event, scorer)
        elif event['event'] == 'task_end':
            if tid in ends:
                raise ValueError('duplicate task_end')
            ends[tid] = event
    task_scores = []
    for task in tasks:
        tid = task['task_id']
        final = responses.get((tid, task['tool_calls']))
        ok = ends.get(tid, {}).get('status') == 'ok'
        task_scores.append(dict(task_id=tid, profile=task.get('profile'),
            status=ends.get(tid, {}).get('status', 'unfinished'),
            final_semantic_score=None if final is None else final['semantic_score'],
            task_score=final['semantic_score'] if ok and final else 0.,
            length_and_stop=bool(ok and final and final['length_pass'] and final['finish_reason'] == 'stop')))
    finals = [t['final_semantic_score'] for t in task_scores if t['final_semantic_score'] is not None]
    n = len(tasks)
    summary = dict(run=str(run), concurrency=cfg.get('concurrency'), n_tasks=n,
        completed_tasks=sum(t['status'] == 'ok' for t in task_scores), final_returned=len(finals),
        mean_task_score=sum(t['task_score'] for t in task_scores) / n,
        returned_final_score_mean=sum(finals) / len(finals) if finals else None,
        length_and_stop_rate=sum(t['length_and_stop'] for t in task_scores) / n)
    provenance = dict(run=str(run), tasks_sha256=digest(tasks),
        events_sha256=hashlib.sha256((run / 'events.jsonl').read_bytes()).hexdigest())
    return list(responses.values()), task_scores, summary, provenance


def compare_tasks(baseline, candidate):
    left = {t['task_id']: t for t in baseline}
    right = {t['task_id']: t for t in candidate}
    if left.keys() != right.keys():
        raise ValueError('comparison requires the same task IDs')
    return [dict(task_id=tid, baseline_score=left[tid]['task_score'],
                 candidate_score=right[tid]['task_score'],
                 delta=right[tid]['task_score'] - left[tid]['task_score']) for tid in left]


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path, help='first run is the comparison baseline')
    parser.add_argument('--out-dir', type=Path, required=True, help='new evaluation directory')
    args = parser.parse_args()
    runs = [p.resolve() for p in args.runs]
    if len(set(runs)) != len(runs):
        parser.error('duplicate run directories')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(method=METHOD, model=MODEL, revision=REVISION, device='cpu',
                    complete=False, inputs=[])
    meta_path = args.out_dir / 'evaluation.json'
    meta_path.write_text(json.dumps(metadata, indent=2))
    scorer = CosineScorer()
    metadata['packages'] = {name: importlib.metadata.version(name)
                            for name in ('sentence-transformers', 'torch', 'transformers', 'numpy')}
    summaries, comparisons, baseline = [], [], None
    with (args.out_dir / 'responses.jsonl').open('w') as rf, (args.out_dir / 'tasks.jsonl').open('w') as tf:
        for run in runs:
            responses, tasks, summary, provenance = evaluate_run(run, scorer)
            if baseline is None:
                baseline = (tasks, summary, provenance)
            if provenance['tasks_sha256'] != baseline[2]['tasks_sha256']:
                raise ValueError('comparison requires identical task inputs/settings/order')
            summary['delta_vs_first'] = summary['mean_task_score'] - baseline[1]['mean_task_score']
            if run != runs[0]:
                comparisons.extend(dict(baseline_run=str(runs[0]), candidate_run=str(run), **row)
                                   for row in compare_tasks(baseline[0], tasks))
            for rows, file in ((responses, rf), (tasks, tf)):
                for row in rows:
                    file.write(json.dumps(dict(run=str(run), **row), ensure_ascii=False) + '\n')
            summaries.append(summary)
            metadata['inputs'].append(provenance)
            meta_path.write_text(json.dumps(metadata, indent=2))
            print(f'{run}: score={summary["mean_task_score"]:.2f}/100, '
                  f'delta={summary["delta_vs_first"]:+.2f}')
    write_csv(args.out_dir / 'summary.csv', summaries)
    if comparisons:
        write_csv(args.out_dir / 'comparison.csv', comparisons)
    metadata['complete'] = True
    meta_path.write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
