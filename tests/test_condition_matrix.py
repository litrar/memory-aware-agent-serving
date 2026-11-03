"""Check the full condition matrix without a model, and the actual HTTP cache flag."""
import asyncio
import csv
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import agent

SCRIPT = ROOT / 'scripts/H_conditions.sh'


class ConditionMatrix(unittest.TestCase):
    def environment(self, folder):
        return dict(os.environ, PYTHON=sys.executable, RESULTS=str(folder), TS='test', TEST='1',
                    N='1', C='1', SHORT_DELAY='0', LONG_DELAY='0.001', DRY_RUN='0',
                    MPLCONFIGDIR=str(Path(folder) / 'mpl-cache'))

    def test_cache_flag_in_http_payload(self):
        payloads = []

        def respond(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'summary'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 1},
                'timings': {'prompt_n': 10, 'cache_n': 0},
            })

        async def check():
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                await agent.chat(client, [{'role': 'user', 'content': 'same prompt'}], 1)
                await agent.chat(client, [{'role': 'user', 'content': 'same prompt'}], 1, cache_prompt=False)

        asyncio.run(check())
        self.assertEqual([p['cache_prompt'] for p in payloads], [True, False])
        self.assertEqual(payloads[0]['messages'], payloads[1]['messages'])

    def test_all_16_conditions_save_metrics_and_same_selections(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'H-conditions-test'
            command = ['bash', str(SCRIPT)]
            env = self.environment(folder)
            subprocess.run(command, env=env, cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
            plan = json.loads((out / 'matrix.json').read_text())
            expected = set(itertools.product(('SS', 'SL', 'LS', 'LL'), (0, 0.001), (True, False)))
            self.assertEqual({(c['combination'], c['tool_delay_s'], c['cache_prompt'])
                              for c in plan['conditions']}, expected)
            selections = {}
            for case in plan['conditions']:
                run = out / case['name'] / 'c1_n1'
                config = json.loads((run / 'config.json').read_text())
                result = json.loads((run / 'metrics.json').read_text())
                manifest = json.loads((run.parent / 'experiment.json').read_text())
                self.assertEqual(config['cache_prompt'], case['cache_prompt'])
                self.assertEqual(manifest['cache_prompt'], case['cache_prompt'])
                self.assertEqual(result['condition']['cache_prompt'], case['cache_prompt'])
                self.assertEqual(result['condition']['tool_delay_override'], [case['tool_delay_s']])
                self.assertEqual(result['condition']['combination'], case['combination'])
                self.assertTrue(result['test'])
                self.assertEqual(result['tasks']['ok'], 1)
                events = [json.loads(s) for s in (run / 'events.jsonl').read_text().splitlines()]
                self.assertEqual(sum(e['event'] == 'llm_end' for e in events), 3)
                self.assertEqual([e['delay_s'] for e in events if e['event'] == 'tool_start'],
                                 [case['tool_delay_s']] * 2)
                tasks = agent.load_workload(run / config['workload_manifest'])
                for task in tasks:
                    task.pop('tool_delay_s')
                previous = selections.setdefault(case['combination'], tasks)
                self.assertEqual(tasks, previous)
                self.assertTrue((run / 'memory.jsonl').exists())
            with (out / 'summary_all.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 16)
            self.assertEqual(len({row['run'] for row in rows}), 16)
            self.assertTrue((out / 'curves_group.png').is_file())
            self.assertEqual(len(list((out / 'timelines').glob('*.png'))), 16)
            # Same-TS resume keeps successful measurements; an incomplete run is excluded from the CSV/plots.
            shutil.copytree(out / plan['conditions'][0]['name'] / 'c1_n1', out / 'unfinished' / 'c1_n1')
            before = {p: p.stat().st_mtime_ns for p in out.glob('H1-*/c1_n1/metrics.json')}
            rerun = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(rerun.returncode, 0, rerun.stdout + rerun.stderr)
            self.assertEqual(rerun.stdout.count('-- skip '), 16)
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})
            with (out / 'summary_all.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 16)
            changed = subprocess.run(command, env=dict(env, LONG_DELAY='0.002'), cwd=ROOT,
                                     capture_output=True, text=True, timeout=10)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn('settings/source/code changed', changed.stderr)

    def test_dry_run_does_not_create_output_and_invalid_delays_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'H-conditions-test'
            command = ['bash', str(SCRIPT)]
            env = dict(self.environment(folder), DRY_RUN='1')
            result = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True, check=True, timeout=10)
            self.assertIn('16 conditions at C=1', result.stdout)
            self.assertEqual(result.stdout.count('--cache-prompt'), 16)
            self.assertFalse(out.exists())
            for delays in (('10', '1'), ('nan', '10')):
                result = subprocess.run(command, env=dict(env, SHORT_DELAY=delays[0], LONG_DELAY=delays[1]), cwd=ROOT,
                                        capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out.exists())
            # A dry run must not move aside existing partial results.
            partial = out / 'H1-SS-delay0-cacheon'
            partial.mkdir(parents=True)
            marker = partial / 'keep.txt'
            marker.write_text('partial measurements')
            subprocess.run(command, env=env, cwd=ROOT, check=True, capture_output=True, timeout=10)
            self.assertEqual(marker.read_text(), 'partial measurements')
            self.assertFalse(list(out.glob('*.partial-*')))


if __name__ == '__main__':
    unittest.main()
