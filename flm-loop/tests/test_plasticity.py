import unittest

import numpy as np

from flmloop.graph import Graph
from flmloop.plasticity import IntrinsicPlasticity, OnlineSoftmaxReadout, RLSReadout
from flmloop.probes import memory_capacity, effective_rank, logistic_probe, spectral_radius, ridge_fit, ridge_predict, calibrate_gain
from flmloop.reservoir import LoopedReservoir


class PlasticityTests(unittest.TestCase):
    def test_intrinsic_plasticity_moves_activity_toward_target_within_bounds(self):
        graph = Graph.random(300, 8, seed=1)
        r = LoopedReservoir(graph, 8, dimensions=8, lanes=4, feedback=0.6, max_iterations=4)
        x = np.random.default_rng(2).normal(size=(80, 4, 8)).astype(np.float32)
        for e in x[:20]:
            r.step(e)
        before = float(np.mean(r.state ** 2))
        target = before / 4  # gains start at the upper bound, so the rule must shrink them
        rule = IntrinsicPlasticity(graph.n, target=target, rate=0.2, smoothing=0.5, bounds=(0.05, 1.0))
        for e in x[20:]:
            r.step(e)
            rule.update(r)
        after = float(np.mean(r.state ** 2))
        self.assertLess(abs(after - target), abs(before - target))
        self.assertLess(float(r.gain.mean()), 1.0)
        self.assertTrue(np.all(r.gain >= 0.05) and np.all(r.gain <= 1.0))
        self.assertLess(r.contraction_bound(), 1)

    def test_online_softmax_readout_learns_a_separable_stream(self):
        rng = np.random.default_rng(3)
        centers = rng.normal(size=(5, 6))
        readout = OnlineSoftmaxReadout(6, 5, learning_rate=0.5)
        for _ in range(400):
            y = rng.integers(0, 5, 8)
            readout.observe(centers[y] + 0.1 * rng.normal(size=(8, 6)), y)
        summary = readout.summary()
        self.assertGreater(summary['tail_accuracy'], 0.95)
        self.assertLess(summary['tail_nll'], summary['online_nll'])
        readout.reset()
        self.assertEqual(readout.summary()['observations'] if readout.losses else 0, 0)

    def test_rls_recovers_a_linear_map(self):
        rng = np.random.default_rng(4)
        true = rng.normal(size=(7, 2))
        rls = RLSReadout(6, 2, delta=1e-4)
        for _ in range(200):
            f = rng.normal(size=6)
            rls.update(f, np.concatenate([f, [1.0]]) @ true)
        np.testing.assert_allclose(rls.weights, true, atol=1e-4)


class ProbeTests(unittest.TestCase):
    def test_memory_capacity_of_a_delay_line(self):
        rng = np.random.default_rng(5)
        signal = rng.normal(size=(3, 400))
        # A perfect 6-tap delay line: features at t are the last 6 inputs.
        features = np.stack([np.stack([np.concatenate([np.zeros(d), lane[:len(lane) - d]]) for d in range(6)], axis=1) for lane in signal])
        result = memory_capacity(features, signal, max_delay=10, alpha=1e-6)
        self.assertGreater(result['capacity'], 5.5)
        self.assertLess(result['per_delay'][8], 0.2)

    def test_effective_rank_bounds(self):
        rng = np.random.default_rng(6)
        full = rng.normal(size=(500, 16))
        low = np.outer(rng.normal(size=500), rng.normal(size=16)) + 1e-3 * rng.normal(size=(500, 16))
        self.assertGreater(effective_rank(full), 14)
        self.assertLess(effective_rank(low), 2)

    def test_logistic_probe_separates_clusters(self):
        rng = np.random.default_rng(7)
        centers = rng.normal(size=(4, 5)) * 3
        y = rng.integers(0, 4, 600); x = centers[y] + rng.normal(size=(600, 5)) * 0.3
        result = logistic_probe(x[:400], y[:400], x[400:], y[400:], classes=4, steps=200)
        self.assertGreater(result['test_accuracy'], 0.95)
        self.assertLess(result['test_nll'], 0.3)

    def test_ridge_fit_predicts(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(100, 3)); w = rng.normal(size=(4, 1))
        y = np.concatenate([x, np.ones((100, 1))], axis=1) @ w
        np.testing.assert_allclose(ridge_predict(ridge_fit(x, y, 1e-9), x), y, atol=1e-5)

    def test_spectral_radius_of_row_stochastic_graph_is_one_and_uniform(self):
        graph = Graph.random(400, 10, seed=9)
        result = spectral_radius(graph, iterations=100)
        self.assertAlmostEqual(result['radius'], 1.0, places=2)
        self.assertGreater(result['uniform_cosine'], 0.9)
        halved = spectral_radius(graph, gain=np.full(graph.n, 0.5, np.float32), iterations=100)
        self.assertAlmostEqual(halved['radius'], 0.5, places=2)
        gain, _ = calibrate_gain(graph, target_radius=0.8, iterations=100)
        self.assertAlmostEqual(spectral_radius(graph, gain=np.full(graph.n, gain, np.float32), iterations=100)['radius'], 0.8, places=2)


if __name__ == '__main__':
    unittest.main()
