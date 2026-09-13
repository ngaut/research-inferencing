import unittest

import numpy as np
import torch

from flmloop.graph import Graph, random_signs
from flmloop.kernel import load_kernel
from flmloop.reservoir import LoopedReservoir
from flmloop.torchloop import TorchLoopedReservoir, propagate


class TorchLoopTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.graph = Graph.random(240, 6, seed=31)
        self.kernel = load_kernel()

    def test_sparse_propagate_gradient_matches_transpose(self):
        x = torch.randn(self.graph.n, 3, dtype=torch.float32, requires_grad=True)
        y = propagate(x, self.graph, self.kernel)
        np.testing.assert_allclose(y.detach().numpy(), self.graph.matrix @ x.detach().numpy(), rtol=1e-5, atol=1e-6)
        g = torch.randn_like(y)
        (y * g).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), self.graph.matrix.T @ g.numpy(), rtol=1e-5, atol=1e-6)

    def test_matches_numpy_reservoir_for_fixed_iterations(self):
        efficacy = random_signs(self.graph.n, 0.3, seed=2)
        gain = np.full(self.graph.n, 0.9, np.float32)
        for feedback, step, K in [(0.0, 1.0, 1), (0.7, 0.6, 5)]:
            ref = LoopedReservoir(self.graph, 12, dimensions=8, lanes=2, feedback=feedback, step_size=step, max_iterations=K,
                                  gain=gain, efficacy=efficacy, kernel=self.kernel)
            module = TorchLoopedReservoir(self.graph, 12, dimensions=8, feedback=feedback, step_size=step, max_iterations=K,
                                          gain=gain, efficacy=efficacy, kernel=self.kernel)
            x = np.random.default_rng(3).normal(size=(2, 7, 12)).astype(np.float32)
            expected = ref.sequence(x)
            with torch.no_grad():
                got, state, _ = module(torch.as_tensor(x))
            np.testing.assert_allclose(got.numpy(), expected, rtol=2e-4, atol=2e-5)
            np.testing.assert_allclose(state.numpy(), ref.state, rtol=2e-4, atol=2e-5)

    def test_gradients_reach_gain_efficacy_and_pooling_and_loss_decreases(self):
        module = TorchLoopedReservoir(self.graph, 12, dimensions=8, feedback=0.6, step_size=0.7, max_iterations=4,
                                      efficacy=random_signs(self.graph.n, 0.3, seed=5), kernel=self.kernel, gain_limit=3.0)
        head = torch.nn.Linear(8, 5)
        rng = np.random.default_rng(6)
        x = torch.as_tensor(rng.normal(size=(3, 10, 12)).astype(np.float32))
        y = torch.as_tensor(rng.integers(0, 5, (3, 10)))
        opt = torch.optim.Adam(list(module.parameters()) + list(head.parameters()), lr=0.05)
        losses = []
        for _ in range(12):
            opt.zero_grad()
            features, _, extra = module(x, supervise=(2, 4), detach_every=5, loop_backprop=2)
            loss = sum(torch.nn.functional.cross_entropy(head(f).reshape(-1, 5), y.reshape(-1)) for f in [features, extra[2]])
            loss.backward()
            for name in ['gain', 'efficacy', 'output_weight']:
                self.assertIsNotNone(getattr(module, name).grad)
                self.assertTrue(torch.isfinite(getattr(module, name).grad).all())
            opt.step()
            losses.append(float(loss))
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(np.all(np.abs(module.export()['gain']) <= 3.0))

    def test_export_round_trips_into_numpy_reservoir(self):
        module = TorchLoopedReservoir(self.graph, 12, dimensions=8, feedback=0.5, step_size=0.8, max_iterations=3, kernel=self.kernel)
        with torch.no_grad():
            module.gain.mul_(0.7); module.output_weight.mul_(1.3); module.efficacy[:50] = -1
        arrays = module.export()
        ref = LoopedReservoir(self.graph, 12, dimensions=8, lanes=1, feedback=0.5, step_size=0.8, max_iterations=3,
                              gain=arrays['gain'], efficacy=arrays['efficacy'], output_weight=arrays['output_weight'], kernel=self.kernel)
        x = np.random.default_rng(8).normal(size=(1, 6, 12)).astype(np.float32)
        with torch.no_grad():
            got, _, _ = module(torch.as_tensor(x))
        np.testing.assert_allclose(got.numpy(), ref.sequence(x), rtol=2e-4, atol=2e-5)


class TrainGraphScriptTests(unittest.TestCase):
    def test_train_graph_script_exports_a_block_that_the_numpy_reservoir_loads(self):
        import json, os, subprocess, sys, tempfile
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'block.npz'
            cmd = [sys.executable, str(root / 'scripts' / 'train_graph.py'), '--graph', 'random:600', '--steps', '4', '--lanes', '2',
                   '--window', '8', '--max-iterations', '3', '--supervise', '2', '--random-k', '--eval-tokens', '16', '--output', str(out)]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root), env={**os.environ, 'OMP_NUM_THREADS': '1'})
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            arrays = np.load(out)
            for key in ['gain', 'efficacy', 'output_weight', 'input_projection']:
                self.assertIn(key, arrays.files)
            summary = json.loads(out.with_suffix('.json').read_text())
            self.assertEqual(len(summary['train_log']), 4)
            graph = Graph.from_spec('random:600', seed=0)
            LoopedReservoir(graph, 64, gain=arrays['gain'], efficacy=arrays['efficacy'], output_weight=arrays['output_weight'])


if __name__ == '__main__':
    unittest.main()
