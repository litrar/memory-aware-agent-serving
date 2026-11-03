"""Scoring arithmetic and result accounting; no model download needed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import agent
import quality

SOURCE = Path(__file__).resolve().parents[1] / 'task-generator/source/phase1-articles.toml'
TEXT = 'Recorded 246 loans.'


def similarities(facts, candidates):
    return [[.8] * len(candidates) for _ in facts]


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.tasks = agent.load_workload(SOURCE, 'SS', n=3)
        self.event = dict(event='llm_end', task_id=self.tasks[0]['task_id'], step=0,
                          text=TEXT, finish_reason='stop', t=1)

    def test_score_and_diagnostics(self):
        event = dict(self.event, text='There were 999 loans and no outcomes.')
        scorer = lambda facts, candidates: [[x] for x in (.9, .8, .7, .6)]
        score = quality.score_response(self.tasks[0], event, scorer)
        self.assertAlmostEqual(score['semantic_score'], 75)
        self.assertIn('999', score['unmatched_numeric_tokens'])
        self.assertIn('246', score['facts'][0]['missing_numeric_tokens'])
        self.assertTrue(score['facts'][0]['negation_marker_differs'])
        # Similarity is intentionally not relabeled as a factual pass.
        self.assertNotIn('factual_pass', score)
        self.assertEqual(quality.numbers('$1,440'), quality.numbers('$1440'))

    def test_empty_answer_is_zero(self):
        score = quality.score_response(self.tasks[0], dict(self.event, text=''), None)
        self.assertEqual(score['semantic_score'], 0)
        self.assertFalse(score['length_pass'])

    def make_run(self, path, fail=False):
        path.mkdir()
        (path / 'config.json').write_text(json.dumps(dict(concurrency=2)))
        (path / 'workload.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in self.tasks))
        ids = [t['task_id'] for t in self.tasks]
        events = [dict(event='task_arrive', task_id=tid, t=0) for tid in ids]
        if not fail:
            events.append(dict(self.event, task_id=ids[0], step=2))
            events.append(dict(event='task_end', task_id=ids[0], status='ok', t=3))
        events.append(dict(event='task_end', task_id=ids[1], status='timeout', t=4))
        (path / 'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))

    def test_failure_denominator_and_manifest_compatibility(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / 'run'
            self.make_run(run)
            responses, tasks, summary, provenance = quality.evaluate_run(run, similarities)
            self.assertEqual(summary['n_tasks'], 3)
            self.assertAlmostEqual(summary['mean_task_score'], 80/3)
            self.assertEqual(summary['returned_final_score_mean'], 80)
            self.assertIsNone(tasks[1]['final_semantic_score'])
            self.assertEqual(tasks[2]['status'], 'unfinished')
            self.assertEqual(tasks[1]['task_score'], 0)
            manifest = run / 'experiment.json'
            spec = agent.make_workload_spec(run / 'workload.jsonl', self.tasks, manifest)
            manifest.write_text(json.dumps(dict(workload_spec=spec)))
            (run / 'config.json').write_text(json.dumps(dict(concurrency=2, workload_manifest='experiment.json')))
            self.assertEqual(quality.evaluate_run(run, similarities)[-1], provenance)

    def test_cli_comparison_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            left, right, out = folder/'phase1', folder/'phase2', folder/'scores'
            self.make_run(left)
            self.make_run(right, fail=True)
            with patch('sys.argv', ['quality.py', str(left), str(right), '--out-dir', str(out)]), \
                    patch.object(quality, 'CosineScorer', return_value=similarities), \
                    patch.object(quality.importlib.metadata, 'version', return_value='fixture'):
                quality.main()
            self.assertTrue(json.loads((out/'evaluation.json').read_text())['complete'])
            self.assertTrue((out/'summary.csv').exists())
            self.assertTrue((out/'comparison.csv').exists())
            scores = quality.read_jsonl(out/'tasks.jsonl')
            delta = quality.compare_tasks(scores[:3], scores[3:])
            self.assertEqual(delta[0]['delta'], -80)
            with self.assertRaisesRegex(ValueError, 'same task IDs'):
                quality.compare_tasks(scores[:3], scores[4:])


if __name__ == '__main__':
    unittest.main()
