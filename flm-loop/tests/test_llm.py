import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flmloop.graph import Graph  # noqa: E402

_STACK = {}


def tiny_backbone():
    """Build (once per test process) a fully offline tiny chat backbone from local text."""
    if 'dir' not in _STACK:
        preset = os.environ.get('FLMLOOP_TINY_BACKBONE')
        if preset and (Path(preset) / 'model.safetensors').exists():
            _STACK['dir'] = Path(preset)
        else:
            sys.path.insert(0, str(ROOT / 'scripts'))
            from make_tiny_backbone import build
            folder = Path(tempfile.mkdtemp(prefix='tiny-backbone-'))
            build(folder, [str(ROOT.parent / '*.md'), str(ROOT / 'data' / '*.json')], vocab_size=1024, hidden=64, layers=2, heads=4, steps=0, verbose=False)
            _STACK['dir'] = folder
    return _STACK['dir']


class LoopedFLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flmloop.llm import LoopedFLM
        cls.graph = Graph.random(400, 8, seed=41)
        cls.model = LoopedFLM(tiny_backbone(), cls.graph, device='cpu', interface={'max_iterations': 3, 'feedback': 0.6, 'step_size': 0.8},
                              fast_weights={'learning_rate': 0.5, 'chunk': 4, 'max_norm': 2.0}, threads=1)

    def test_untrained_adapter_leaves_base_logits_untouched(self):
        m = self.model
        hidden = torch.randn(2, m.hidden_size); features = torch.randn(2, m.interface['dimensions'])
        logits, base, delta = m.scores(hidden, features)
        self.assertEqual(torch.count_nonzero(delta).item(), 0)
        torch.testing.assert_close(logits, base)

    def test_no_edges_features_are_zero_so_the_correction_vanishes(self):
        m = self.model
        with torch.no_grad():
            m.adapter.output.weight.normal_(std=0.1)
        try:
            reservoir = m.reservoir()
            feats = np.stack([reservoir.step(m.embeddings[t], 'no_edges') for t in m.prompt_ids([{'role': 'user', 'content': 'hello'}])])
            self.assertEqual(np.count_nonzero(feats), 0)
            hidden = torch.randn(len(feats), m.hidden_size)
            logits, base, delta = m.scores(hidden, torch.as_tensor(feats), 'no_edges')
            torch.testing.assert_close(logits, base)
            intact = np.stack([m.reservoir().step(m.embeddings[t]) for t in [5, 6]])
            _, _, delta = m.scores(hidden[:2], torch.as_tensor(intact), 'intact')
            self.assertGreater(float(delta.abs().max()), 0)
        finally:
            with torch.no_grad():
                m.adapter.output.weight.zero_()

    def test_base_generation_streams_events_and_bounded_delta(self):
        events = list(self.model.generate([{'role': 'user', 'content': 'Say hi.'}], mode='base', max_tokens=6, seed=3))
        self.assertEqual(events[-1]['type'], 'done')
        self.assertLessEqual(events[-1]['tokens'], 6)
        with self.assertRaisesRegex(ValueError, 'Train'):
            list(self.model.generate([{'role': 'user', 'content': 'x'}], mode='intact'))

    def test_fast_weights_learn_the_prefix_and_reset(self):
        m = self.model
        with torch.no_grad():
            m.adapter.output.weight.normal_(std=0.05)
        try:
            messages = [{'role': 'user', 'content': 'The cactus is named Pickle. It lives on my desk. ' * 3},
                        {'role': 'assistant', 'content': 'Pickle lives on the desk.'}]
            plain = m.prompt_nll(messages, 'intact', ttt=False)
            with_ttt = m.prompt_nll(messages, 'intact', ttt=True)
            self.assertTrue(np.isfinite(plain) and np.isfinite(with_ttt))
            losses = m.fast.losses
            self.assertGreater(len(losses), 2)
            self.assertLess(losses[-1], losses[0])
            self.assertLessEqual(m.fast.norm(), 2.0 + 1e-6)
            m.fast.reset()
            self.assertEqual(m.fast.norm(), 0.0)
            self.assertEqual(m.prompt_nll(messages, 'intact', ttt=False), plain)
        finally:
            with torch.no_grad():
                m.adapter.output.weight.zero_()

    def test_assistant_mask_marks_answers_only(self):
        m = self.model
        messages = [{'role': 'user', 'content': 'UNIQUEUSER'}, {'role': 'assistant', 'content': 'prism colors'},
                    {'role': 'user', 'content': 'SECONDUSER'}, {'role': 'assistant', 'content': 'rain light'}]
        ids = m.prompt_ids(messages, add_generation_prompt=False)
        mask = m.assistant_mask(messages, ids)
        rendered = m.tokenizer.decode(np.asarray(ids)[mask])
        self.assertIn('prism colors', rendered); self.assertIn('rain light', rendered)
        self.assertNotIn('UNIQUEUSER', rendered); self.assertNotIn('SECONDUSER', rendered)


class TrainAdapterScriptTests(unittest.TestCase):
    def test_train_adapter_end_to_end_and_checkpoint_round_trip(self):
        backbone = tiny_backbone()
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / 'run'
            cmd = [sys.executable, str(ROOT / 'scripts' / 'train_adapter.py'), '--backbone', str(backbone), '--graph', 'random:500',
                   '--output', str(run), '--device', 'cpu', '--epochs', '2', '--max-conversations', '12', '--max-iterations', '3',
                   '--lanes', '4', '--threads', '1', '--ttt-chunk', '8']
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env={**os.environ, 'OMP_NUM_THREADS': '1'})
            self.assertEqual(result.returncode, 0, result.stderr[-4000:])
            report = json.loads((run / 'report.json').read_text())
            for key in ['base', 'fly_adapter', 'direct_input_adapter', 'relabeled_wiring', 'no_edges', 'fly_adapter_ttt']:
                self.assertIn(key, report)
                self.assertTrue(np.isfinite(report[key]['nll']))
            manifest = json.loads((run / 'run.json').read_text())
            hidden = json.loads((backbone / 'config.json').read_text())['hidden_size']
            self.assertEqual(manifest['trainable_parameters'], 128 * 128 + 128 * hidden)
            from flmloop.llm import LoopedFLM
            graph = Graph.from_spec('random:500', seed=manifest['training']['seed'])
            model = LoopedFLM(backbone, graph, checkpoint=run / 'adapter.safetensors', device='cpu', threads=1)
            self.assertTrue(model.trained)
            self.assertEqual(model.interface['max_iterations'], 3)
            events = list(model.generate([{'role': 'user', 'content': 'Name a tiny museum exhibit.'}], mode='intact', max_tokens=8, seed=1, ttt=True))
            self.assertEqual(events[-1]['type'], 'done')
            self.assertGreaterEqual(events[-1]['ttt']['steps'], 1)
            with self.assertRaisesRegex(ValueError, 'different graph'):
                LoopedFLM(backbone, Graph.random(500, 12, seed=999), checkpoint=run / 'adapter.safetensors', device='cpu', threads=1)


if __name__ == '__main__':
    unittest.main()
