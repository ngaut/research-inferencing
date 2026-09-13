import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from flmloop.graph import Graph, neurotransmitter_signs, random_signs, NEUROTRANSMITTER_SIGNS
from flmloop.kernel import load_kernel, spmm
from flmloop.reservoir import LoopedReservoir


class FLMReference:
    """FLM's Reservoir (nftechie/flm, MIT), verbatim dynamics, as the equivalence oracle."""

    def __init__(self, graph, embedding_dim, dimensions=128, seed=7301):
        self.graph = graph
        self.n = graph.matrix.shape[0]
        self.dimensions = dimensions
        rng = np.random.default_rng(seed)
        self.input_projection = (rng.standard_normal((embedding_dim, dimensions)) / np.sqrt(embedding_dim)).astype(np.float32)
        self.input_bins = rng.integers(0, dimensions, self.n)
        self.input_sign = rng.choice(np.array([-1, 1], np.float32), self.n)
        self.output_bins = rng.integers(0, dimensions, self.n)
        self.output_sign = rng.choice(np.array([-1, 1], np.float32), self.n)
        self.output_scale = np.sqrt(np.maximum(1, np.bincount(self.output_bins, minlength=dimensions))).astype(np.float32)
        self.permutation = rng.permutation(self.n)
        self.inverse = np.argsort(self.permutation)
        self.state = np.zeros(self.n, np.float32)

    def project_input(self, embedding):
        code = np.asarray(embedding, np.float32) @ self.input_projection
        return code / np.sqrt(np.mean(code * code) + 1e-6)

    def step(self, embedding, mode='intact'):
        code = self.project_input(embedding)
        drive = 0.6 * self.state + 0.4 * code[self.input_bins] * self.input_sign
        if mode == 'no_edges':
            self.state.fill(0)
        elif mode == 'shuffled':
            self.state = np.tanh((self.graph.matrix @ drive[self.permutation])[self.inverse])
        else:
            self.state = np.tanh(self.graph.matrix @ drive)
        f = np.bincount(self.output_bins, weights=self.state * self.output_sign, minlength=self.dimensions).astype(np.float32) / self.output_scale
        return f / np.sqrt(np.mean(f * f) + 1e-6)


def inputs(steps, lanes, dim, seed):
    return np.random.default_rng(seed).normal(size=(steps, lanes, dim)).astype(np.float32)


class FLMEquivalenceTests(unittest.TestCase):
    def test_k1_reproduces_flm_bit_for_bit_on_all_controls(self):
        for graph in (Graph.toy(), Graph.random(300, 6, seed=3)):
            for mode in ['intact', 'shuffled', 'no_edges']:
                reference = FLMReference(graph, 16, dimensions=8)
                ours = LoopedReservoir(graph, 16, dimensions=8, max_iterations=1)
                for x in inputs(12, 1, 16, 5):
                    np.testing.assert_array_equal(ours.step(x[0], mode), reference.step(x[0], mode))
                    np.testing.assert_array_equal(ours.state[:, 0], reference.state)

    def test_seeded_interfaces_match_flm_draw_order(self):
        graph = Graph.random(64, 4, seed=1)
        reference, ours = FLMReference(graph, 32), LoopedReservoir(graph, 32)
        for name in ['input_projection', 'input_bins', 'input_sign', 'output_bins', 'output_sign', 'output_scale', 'permutation', 'inverse']:
            np.testing.assert_array_equal(getattr(ours, name), getattr(reference, name))

    def test_zero_feedback_ignores_extra_iterations(self):
        graph = Graph.random(200, 5, seed=2)
        one = LoopedReservoir(graph, 8, dimensions=8, max_iterations=1)
        many = LoopedReservoir(graph, 8, dimensions=8, max_iterations=6, feedback=0.0, tolerance=1e-7)
        for x in inputs(6, 1, 8, 9):
            np.testing.assert_array_equal(one.step(x[0]), many.step(x[0]))
        self.assertLessEqual(int(many.last['iterations'][0]), 2)


class FeedbackSemanticsTests(unittest.TestCase):
    def test_k1_with_feedback_is_not_flm_but_c0_is_for_any_k(self):
        graph = Graph.random(150, 5, seed=61)
        x = inputs(5, 1, 8, 62)
        reference = FLMReference(graph, 8, dimensions=8)
        expected = np.stack([reference.step(e[0]) for e in x])
        many = LoopedReservoir(graph, 8, dimensions=8, feedback=0.0, max_iterations=7, tolerance=None)
        np.testing.assert_array_equal(np.stack([many.step(e[0]) for e in x]), expected)
        self.assertEqual(int(many.last['iterations'][0]), 1)
        one = LoopedReservoir(graph, 8, dimensions=8, feedback=0.8, max_iterations=1)
        got = np.stack([one.step(e[0]) for e in x])
        self.assertGreater(np.abs(got - expected).max(), 1e-3)


class LoopTests(unittest.TestCase):
    def test_loop_contracts_geometrically_and_exits_early(self):
        graph = Graph.random(400, 8, seed=4)
        r = LoopedReservoir(graph, 8, dimensions=16, feedback=0.8, max_iterations=40, tolerance=1e-5)
        self.assertLess(r.contraction_bound(), 1)
        x = inputs(1, 1, 8, 11)[0]
        r.step(x)
        used = int(r.last['iterations'][0])
        self.assertGreater(used, 2)
        self.assertLess(used, 40)
        # Residuals recorded per iteration decay at least as fast as the contraction bound predicts.
        r2 = LoopedReservoir(graph, 8, dimensions=16, feedback=0.8, max_iterations=1)
        r2.state = r.state * 0  # fresh
        residuals = []
        z = np.zeros((graph.n, 1), np.float32)
        code = r2.project_input(x)
        u = code[:, r2.input_bins].T * r2.input_sign[:, None]
        m = 0.4 * u
        for k in range(12):
            candidate = np.tanh(r2._propagate(m + 0.8 * z, 'intact'))
            residuals.append(float(np.abs(candidate - z).max()))
            z = candidate
        for a, b in zip(residuals[:-1], residuals[1:]):
            self.assertLessEqual(b, 0.8 * a + 1e-6)

    def test_more_iterations_change_the_fixed_point_and_reach_deeper_neurons(self):
        graph = Graph.toy()  # chain 0 -> 1 -> 2: with one step only neuron 1 can respond to input at 0
        r = LoopedReservoir(graph, 2, dimensions=2, feedback=0.9, max_iterations=1)
        r.input_projection = np.eye(2, dtype=np.float32)
        r.input_bins = np.array([0, 1, 1]); r.input_sign = np.ones(3, np.float32)
        r.step(np.array([1, 0], np.float32))
        self.assertEqual(r.state[2, 0], 0)
        r.reset()
        r.max_iterations = 3
        r.step(np.array([1, 0], np.float32))
        self.assertGreater(r.state[2, 0], 0)

    def test_lanes_are_independent_even_with_adaptive_exit(self):
        graph = Graph.random(300, 7, seed=6)
        batch = LoopedReservoir(graph, 8, dimensions=8, lanes=3, feedback=0.7, max_iterations=30, tolerance=1e-4)
        singles = [LoopedReservoir(graph, 8, dimensions=8, feedback=0.7, max_iterations=30, tolerance=1e-4) for _ in range(3)]
        for x in inputs(5, 3, 8, 12):
            expected = np.stack([s.step(x[i] * (i + 1)) for i, s in enumerate(singles)])
            got = batch.step(x * np.arange(1, 4)[:, None])
            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(batch.last['iterations'], [int(s.last['iterations'][0]) for s in singles])

    def test_no_edges_is_exactly_zero_and_graph_is_not_mutated(self):
        graph = Graph.random(100, 4, seed=7)
        before = graph.matrix.toarray().copy()
        r = LoopedReservoir(graph, 8, dimensions=8, feedback=0.5, max_iterations=4)
        for mode in ['intact', 'shuffled', 'no_edges']:
            values = r.sequence(np.ones((6, 8), np.float32), mode)
            self.assertTrue(np.isfinite(values).all())
            if mode == 'no_edges':
                self.assertEqual(np.count_nonzero(values), 0)
        np.testing.assert_array_equal(before, graph.matrix.toarray())

    def test_gain_and_efficacy_enter_the_dynamics_and_the_bound(self):
        graph = Graph.random(120, 5, seed=8)
        base = LoopedReservoir(graph, 8, dimensions=8, feedback=0.5, max_iterations=3)
        signed = LoopedReservoir(graph, 8, dimensions=8, feedback=0.5, max_iterations=3,
                                 efficacy=random_signs(graph.n, 0.3, seed=1), gain=np.full(graph.n, 0.5, np.float32))
        self.assertAlmostEqual(signed.contraction_bound(), 0.25)
        x = inputs(3, 1, 8, 13)
        a = np.stack([base.step(e[0]) for e in x]); b = np.stack([signed.step(e[0]) for e in x])
        self.assertGreater(np.abs(a - b).max(), 1e-3)

    def test_sequence_matches_stepwise_and_batch_layout(self):
        graph = Graph.random(150, 5, seed=9)
        r = LoopedReservoir(graph, 8, dimensions=8, feedback=0.6, max_iterations=4)
        x = inputs(7, 2, 8, 14)
        full = r.sequence(np.transpose(x, (1, 0, 2)))
        self.assertEqual(full.shape, (2, 7, 8))
        single = r.sequence(x[:, 0])
        np.testing.assert_allclose(full[0], single, rtol=1e-5, atol=1e-6)


class DampedLoopTests(unittest.TestCase):
    def test_drive_scale_and_step_size_defaults_keep_flm_exact(self):
        graph = Graph.random(120, 5, seed=21)
        r = LoopedReservoir(graph, 8, dimensions=8)
        self.assertEqual(r.drive_scale, 1.0)
        self.assertEqual(r.step_size, 1.0)
        looped = LoopedReservoir(graph, 8, dimensions=8, feedback=0.8, step_size=0.5, max_iterations=6)
        self.assertAlmostEqual(looped.drive_scale, 0.2)
        self.assertEqual(LoopedReservoir(graph, 8, dimensions=8, feedback=0.8, normalize_feedback=False).drive_scale, 1.0)

    def test_damped_iteration_reaches_the_same_fixed_point(self):
        graph = Graph.random(300, 8, seed=22)
        x = inputs(1, 1, 8, 23)[0][0]
        undamped = LoopedReservoir(graph, 8, dimensions=8, feedback=0.7, step_size=1.0, max_iterations=200, tolerance=1e-7)
        damped = LoopedReservoir(graph, 8, dimensions=8, feedback=0.7, step_size=0.5, max_iterations=400, tolerance=1e-7)
        undamped.step(x); damped.step(x)
        np.testing.assert_allclose(undamped.state, damped.state, atol=2e-5)
        self.assertGreater(int(damped.last['iterations'][0]), int(undamped.last['iterations'][0]))

    def test_signed_calibrated_loop_stays_bounded_and_converges(self):
        from flmloop.probes import calibrate_gain
        graph = Graph.random(500, 12, seed=24)
        efficacy = random_signs(graph.n, 0.35, seed=4)
        gain, estimate = calibrate_gain(graph, efficacy, target_radius=0.9, iterations=80)
        r = LoopedReservoir(graph, 8, dimensions=16, lanes=2, feedback=0.8, step_size=0.7, max_iterations=60,
                            tolerance=1e-4, gain=np.full(graph.n, gain, np.float32), efficacy=efficacy)
        r.spectral_radius = estimate['radius'] * gain
        for e in inputs(20, 2, 8, 25):
            r.step(e)
            self.assertTrue(np.isfinite(r.state).all())
            self.assertLess(int(r.last['iterations'].max()), 60)
        self.assertGreater(r.describe()['negative_efficacy_fraction'], 0.3)


class GraphTests(unittest.TestCase):
    def test_rewired_control_preserves_degrees_and_weights(self):
        graph = Graph.random(500, 9, seed=10)
        control = graph.rewired(seed=3)
        np.testing.assert_array_equal(graph.in_degree(), control.in_degree())
        np.testing.assert_array_equal(np.sort(graph.out_degree()), np.sort(control.out_degree()))
        np.testing.assert_array_equal(graph.matrix.data, control.matrix.data)
        self.assertGreater(np.count_nonzero(graph.matrix.indices != control.matrix.indices), graph.matrix.nnz // 2)

    def test_kernel_matches_scipy_path(self):
        kernel = load_kernel()
        if kernel is None:
            self.skipTest('no C compiler available')
        graph = Graph.random(700, 11, seed=11)
        x = np.random.default_rng(15).normal(size=(graph.n, 5)).astype(np.float32)
        np.testing.assert_allclose(spmm(graph.matrix, x, kernel), np.asarray(graph.matrix @ x, np.float32), rtol=3e-5, atol=3e-6)
        r1 = LoopedReservoir(graph, 8, dimensions=8, lanes=5, feedback=0.7, max_iterations=5, kernel=kernel)
        r2 = LoopedReservoir(graph, 8, dimensions=8, lanes=5, feedback=0.7, max_iterations=5, kernel=None)
        for e in inputs(4, 5, 8, 16):
            np.testing.assert_allclose(r1.step(e), r2.step(e), rtol=3e-5, atol=3e-6)

    def test_neurotransmitter_signs_from_a_feather_table(self):
        import pyarrow as pa
        import pyarrow.feather as feather
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'nt.feather'
            feather.write_feather(pa.table({'body': [5, 3, 9, 7], 'consensus_nt': ['gaba', 'acetylcholine', 'unclear', None]}), str(path))
            values, counts = neurotransmitter_signs(np.array([3, 5, 7, 9, 11]), path)
            np.testing.assert_array_equal(values, [1, -1, 1, 1, 1])
            self.assertEqual(counts['gaba'], 1)
            self.assertEqual(counts['unknown'], 2)
        self.assertEqual(NEUROTRANSMITTER_SIGNS['glutamate'], -1.0)

    def test_random_signs_share(self):
        values = random_signs(1000, 0.25, seed=2)
        self.assertEqual(int((values < 0).sum()), 250)

    def test_flm_cache_loader_verifies_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.save(root / 'data.npy', np.zeros(1, np.float32))
            (root / 'manifest.json').write_text('{"arrays": {"data.npy": "0000"}, "directed_edges": 0}')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                Graph.from_flm_cache(root)


if __name__ == '__main__':
    unittest.main()
