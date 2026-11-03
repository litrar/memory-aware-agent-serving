"""Small schema and monitor checks; no inference server required."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent
import metrics
import monitor


class MetricsIntegration(unittest.TestCase):
    def test_experiment_runs_agent_and_writes_metrics(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'experiment'
            subprocess.run([
                sys.executable, '-B', str(root / 'experiment.py'), '--test',
                '--workload', str(root / 'task-generator/source/phase1-articles.toml'),
                '--combination', 'SS', '-n', '4', '-C', '1', '2', '--timeout', '30',
                '--cache-prompt', 'off',
                '--out-dir', str(out), '--no-plot',
            ], cwd=root, check=True, capture_output=True, text=True, timeout=60)

            run_dir = out / 'c1_n4'
            manifest = json.loads((out / 'experiment.json').read_text())
            config = json.loads((run_dir / 'config.json').read_text())
            result = json.loads((run_dir / 'metrics.json').read_text())
            events = [json.loads(line) for line in (run_dir / 'events.jsonl').read_text().splitlines()]

            self.assertEqual(manifest['runner'], 'agent.py')
            self.assertFalse(manifest['cache_prompt'])
            self.assertEqual(config['runner'], 'agent.py')
            self.assertEqual(result['headline']['completed_tasks'], 4)
            self.assertEqual(result['requests']['n'], 12)
            self.assertEqual(sum(e['event'] == 'task_end' for e in events), 4)
            self.assertTrue((run_dir / 'memory.jsonl').exists())

            snapshot = Path(manifest['snapshot'])
            expected = agent.load_workload(root / 'task-generator/source/phase1-articles.toml', 'SS', n=4)
            self.assertTrue(snapshot.is_file())
            self.assertEqual(agent.load_workload(snapshot), expected)
            for c in (1, 2):
                folder = out / f'c{c}_n4'
                cfg = json.loads((folder / 'config.json').read_text())
                self.assertFalse(cfg['cache_prompt'])
                self.assertFalse(json.loads((folder / 'metrics.json').read_text())['condition']['cache_prompt'])
                ev = [json.loads(x) for x in (folder / 'events.jsonl').read_text().splitlines()]
                local_manifest = folder / cfg['workload_manifest']
                spec = json.loads(local_manifest.read_text())['workload_spec']
                ids = [task['task_id'] for task in expected]
                self.assertEqual(cfg['task_ids'], ids)
                self.assertNotIn('tasks', cfg)
                self.assertEqual([t['task_id'] for t in agent.load_workload(local_manifest)], ids)
                self.assertEqual(len(spec['tasks']), 4)
                self.assertEqual([e['task_id'] for e in ev if e['event'] == 'task_start'], ids)
                active = peak = 0
                for e in ev:
                    if e['event'] == 'task_start':
                        active += 1
                        peak = max(peak, active)
                    elif e['event'] == 'task_end':
                        active -= 1
                self.assertEqual(peak, c)
                self.assertEqual(active, 0)
                self.assertEqual(sum(e['event'] == 'llm_end' for e in ev), 12)
                self.assertEqual(sum(e['event'] == 'tool_end' for e in ev), 8)
                grouped = json.loads((folder / 'metrics.json').read_text())['head_of_line']['task_p50_by_profile']
                self.assertEqual(grouped['SS']['n'], 4)
                self.assertGreater(grouped['SS']['completion_s'], 0)

    def test_monitor_keeps_rss_without_linux_proc(self):
        proc = Mock()
        proc.memory_info.return_value.rss = 123 * monitor.MiB
        proc.cpu_percent.return_value = 50
        mon = monitor.MemoryMonitor('unused.jsonl', lambda: 0, proc=proc)
        with patch.object(monitor.platform, 'system', return_value='Darwin'), \
             patch.object(monitor, '_proc_status_kib', side_effect=AssertionError('no /proc on Darwin')), \
             patch.object(monitor, 'major_faults', side_effect=AssertionError('no /proc on Darwin')):
            for _ in range(2):
                sample = mon.sample()
                self.assertEqual(sample['rss_mb'], 123)
                self.assertIn('srv_cpu_pct', sample)
                self.assertNotIn('srv_exited', sample)
                self.assertNotIn('srv_anon_mib', sample)
                self.assertIs(mon.proc, proc)

    def test_monitor_missing_proc_file_does_not_mean_exit(self):
        proc = Mock()
        mon = monitor.MemoryMonitor('unused.jsonl', lambda: 0, proc=proc)
        with patch.object(monitor, 'process_memory', side_effect=FileNotFoundError('/proc')):
            sample = mon.sample()
        self.assertIn('srv_memory_error', sample)
        self.assertNotIn('srv_exited', sample)
        self.assertIs(mon.proc, proc)
        with patch.object(monitor, 'process_memory', side_effect=monitor.psutil.NoSuchProcess(123)):
            self.assertTrue(mon.sample()['srv_exited'])
        self.assertIsNone(mon.proc)

    def test_manifest_roundtrip_hash_overrides_and_portability(self):
        source = Path(__file__).resolve().parents[1] / 'task-generator/source/phase1-articles.toml'
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            copy = folder / 'source.toml'
            copy.write_bytes(source.read_bytes())
            tasks = agent.load_workload(copy, 'SS-LL', n=4)
            tasks[0]['final_tokens'] = 99
            manifest = folder / 'experiment.json'
            spec = agent.make_workload_spec(copy, tasks, manifest)
            manifest.write_text(json.dumps(dict(workload_spec=spec)))
            self.assertEqual(agent.load_workload(manifest), tasks)
            self.assertEqual(spec['tasks'][0]['overrides'], {'final_tokens': 99})
            self.assertIn('required_facts', agent.load_workload(manifest)[0])
            # Relative source paths survive moving an entire result/source directory.
            moved = folder / 'moved'
            moved.mkdir()
            copy.rename(moved / copy.name)
            manifest.rename(moved / manifest.name)
            self.assertEqual(agent.load_workload(moved / manifest.name), tasks)
            (moved / copy.name).write_text((moved / copy.name).read_text() + '\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                agent.load_workload(moved / manifest.name)
            # Results moved away from the repo can use the recorded repo-relative path.
            portable = agent.make_workload_spec(source, agent.load_workload(source, 'SS', n=1), manifest)
            portable['source']['path'] = 'missing/source.toml'
            manifest.write_text(json.dumps(dict(workload_spec=portable)))
            self.assertEqual(agent.load_workload(manifest), agent.load_workload(source, 'SS', n=1))

    def test_legacy_jsonl_reference_roundtrip(self):
        source = Path(__file__).resolve().parents[1] / 'task-generator/source/phase1-articles.toml'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old-workload.jsonl'
            tasks = agent.load_workload(source, 'SS', n=2)
            path.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
            self.assertEqual(agent.load_workload(path), tasks)
            manifest = Path(folder) / 'experiment.json'
            spec = agent.make_workload_spec(path, tasks, manifest)
            manifest.write_text(json.dumps(dict(workload_spec=spec)))
            self.assertEqual(agent.load_workload(manifest), tasks)
            self.assertEqual(spec['tasks'], [{'task_id': t['task_id']} for t in tasks])

    def test_main_and_pr_memory_schemas(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = dict(run='fixture', concurrency=1, n_tasks=3,
                          server=dict(total_slots=8, n_ctx=8192, build='fixture'))
            (root / 'config.json').write_text(json.dumps(config))
            events = [dict(t=0, event='task_arrive', task_id=i) for i in range(3)]
            for i, status in enumerate(['ok', 'timeout', 'error']):
                events.extend([dict(t=i*2, event='task_start', task_id=i),
                               dict(t=i*2+2, event='task_end', task_id=i, status=status)])
            (root / 'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            main = dict(t=1, rss_mb=123, busy_slots=1, slots=[dict(id=0, busy=True, tokens=9000)])
            old = dict(t=1, srv_rss_mib=123, srv_slots=dict(busy=1, kv_tokens=[9000]))
            results = []
            for sample in (main, old):
                (root / 'memory.jsonl').write_text(json.dumps(sample)+'\n')
                results.append(metrics.compute(folder))
            self.assertEqual(results[0]['memory'], results[1]['memory'])
            self.assertEqual(results[0]['headline']['completed_tasks'], 1)
            self.assertEqual(results[0]['tasks']['timeout'], 1)
            self.assertEqual(results[0]['tasks']['error'], 1)
            self.assertEqual(results[0]['throughput']['tasks_per_min'], 10)
            self.assertEqual(results[0]['server']['total_slots'], 8)
            self.assertEqual(results[0]['memory']['slot_context']['slot_context_tokens_total_peak'], 9000)
            self.assertNotIn('kv_used_frac_peak', results[0]['memory']['slot_context'])

    def test_monitor_preserves_main_fields(self):
        async def run():
            def handler(request):
                if request.url.path == '/slots':
                    return httpx.Response(200, json=[
                        dict(id=0, is_processing=True, n_prompt_tokens=100),
                        dict(id=1, is_processing=False, n_prompt_tokens=200)])
                return httpx.Response(200, text='llamacpp:requests_processing 1\n')
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'memory.jsonl'
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    start = time.perf_counter()
                    mon = monitor.MemoryMonitor(path, lambda: time.perf_counter()-start,
                                                interval=.01, client=client, url='http://fixture', scrape_every_s=0)
                    mon.nvidia = False
                    await mon.start()
                    await asyncio.sleep(.025)
                    await mon.stop()
                rows = [json.loads(s) for s in path.read_text().splitlines()]
                self.assertGreaterEqual(len(rows), 2)
                for row in rows:
                    self.assertEqual(row['busy_slots'], 1)
                    self.assertEqual(row['slot_context_tokens_busy'], 100)
                    self.assertEqual(row['slot_context_tokens_total'], 300)
                    self.assertIn('rss_mb', row)
                    self.assertEqual(row['srv_metrics']['llamacpp:requests_processing'], 1)
                self.assertTrue(rows[0]['endpoint_status']['slots'])
        asyncio.run(run())

    def test_shared_capacity_is_not_multiplied_by_slot_count(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'server.log'
            path.write_text('n_slots = 8\nn_ctx_slot = 8192\nllama_context: n_ctx = 8192\nCPU KV buffer size = 896.00 MiB\n')
            buffers = metrics.server_buffers(path)
            self.assertEqual(buffers['n_ctx'], 8192)
            self.assertEqual(buffers['kv_mib'], 896)
            path.write_text('n_slots = 8\nn_ctx_slot = 8192\n')
            buffers = metrics.server_buffers(path)
            self.assertIsNone(buffers['n_ctx'])
            self.assertIsNone(buffers['kv_mib'])

    def test_partial_run_keeps_offered_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text(json.dumps(dict(n_tasks=2)))
            events = [dict(t=0, event='task_arrive', task_id=i) for i in range(2)]
            events += [dict(t=0, event='task_start', task_id=0),
                       dict(t=5, event='llm_start', task_id=0, step=0)]
            (root / 'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            result = metrics.compute(folder)
            self.assertEqual(result['tasks']['unfinished'], 2)
            self.assertEqual(result['tasks']['failure_rate'], 1)
            self.assertEqual(result['throughput']['makespan_s'], 5)


if __name__ == '__main__':
    unittest.main()
