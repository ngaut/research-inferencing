"""The looped connectome block: FLM's reservoir generalized to K inner iterations per token.

FLM, one propagation per token:      x_t = tanh( W (a x_{t-1} + b u_t) )

Looped block (looped-transformer principle: iterate one shared block on a latent, re-inject
the input every pass, stop when the latent stops changing):

    m_t     = d * (a x_{t-1} + b u_t)                 memory + input drive, fixed inside the loop
    z_0     = x_{t-1}                                 warm start
    z_{k+1} = (1-h) z_k + h tanh( g * W (s * (m_t + c z_k)) )    k = 0 .. K-1, early exit on residual
    x_t     = z_K,   f_t = rms_normalize( pool(x_t) )

g is a per-neuron postsynaptic gain, s a per-neuron presynaptic efficacy (signed: -1 for
inhibitory transmitters), c the loop feedback, h the step size, d = 1 - c the drive scale that
keeps the loop's small-signal DC gain equal to FLM's. With c = 0 and g = s = 1, h = 1 this
reproduces FLM exactly (bit for bit on the SciPy path, to ~3e-5 relative through the C kernel),
including its seeded interfaces, so FLM adapters transfer. Note that K = 1 with c != 0 is NOT
FLM: the warm start enters the drive as c * x_{t-1}, giving (1-c)(a x + b u) + c x.

Two readings of the same loop. Looped transformer: a weight-tied block applied K times with the
input injected every pass. Rate model: h < 1 is an Euler step of  dz/dt = -z + tanh(g W s (m + c z)),
the textbook dynamics for running a connectome; FLM's single step is one Euler step of size 1.

Convergence. Rows of W sum to at most 1 and tanh is 1-Lipschitz, so |c| * max|g| * max|s| < 1
guarantees a unique fixed point per token (contraction in the max norm). Signed, gain-calibrated
graphs typically exceed that loose bound; then convergence is local, governed by the spectral
radius of c * g W s, and the residual telemetry plus K is the safety net.
Iteration k propagates the input k synapses deep within one token; FLM reaches one synapse per token.
"""
import numpy as np

from .kernel import spmm

MODES = ('intact', 'no_edges', 'shuffled')


def rms_normalize(values, axis=-1):
    values = np.asarray(values, np.float32)
    return values / np.sqrt(np.mean(values * values, axis=axis, keepdims=True) + 1e-6)


class LoopedReservoir:
    def __init__(self, graph, embedding_dim, dimensions=128, seed=7301, lanes=1,
                 memory=0.6, drive=0.4, feedback=0.0, step_size=1.0, normalize_feedback=True,
                 max_iterations=1, tolerance=None, gain=None, efficacy=None, output_weight=None, kernel=None):
        self.graph = graph
        self.matrix = graph.matrix
        self.n = graph.n
        self.dimensions = int(dimensions)
        self.seed = int(seed)
        self.lanes = int(lanes)
        self.memory, self.drive, self.feedback = float(memory), float(drive), float(feedback)
        self.step_size = float(step_size)
        self.normalize_feedback = bool(normalize_feedback)
        self.max_iterations = int(max_iterations)
        self.tolerance = None if tolerance is None else float(tolerance)
        self.kernel = kernel
        rng = np.random.default_rng(self.seed)
        # Same draw order as FLM's Reservoir, so seed 7301 reproduces its interfaces bit for bit.
        self.input_projection = (rng.standard_normal((embedding_dim, self.dimensions)) /
                                 np.sqrt(embedding_dim)).astype(np.float32)
        self.input_bins = rng.integers(0, self.dimensions, self.n)
        self.input_sign = rng.choice(np.array([-1, 1], np.float32), self.n)
        self.output_bins = rng.integers(0, self.dimensions, self.n)
        self.output_sign = rng.choice(np.array([-1, 1], np.float32), self.n)
        self.output_scale = np.sqrt(np.maximum(1, np.bincount(self.output_bins, minlength=self.dimensions))).astype(np.float32)
        self.permutation = rng.permutation(self.n)
        self.inverse = np.argsort(self.permutation)
        self.gain = np.ones(self.n, np.float32) if gain is None else np.array(gain, np.float32, copy=True)
        self.efficacy = np.ones(self.n, np.float32) if efficacy is None else np.array(efficacy, np.float32, copy=True)
        if self.gain.shape != (self.n,) or self.efficacy.shape != (self.n,):
            raise ValueError('gain and efficacy must have one value per neuron.')
        # Learned pooling (from TorchLoopedReservoir.export); None keeps FLM's signed binning exactly.
        self.output_weight = None if output_weight is None else np.array(output_weight, np.float32, copy=True)
        if self.output_weight is not None and self.output_weight.shape != (self.n,):
            raise ValueError('output_weight must have one value per neuron.')
        self.spectral_radius = None  # set by calibration; informational
        self.state = np.zeros((self.n, self.lanes), np.float32)
        self.tokens = 0
        self.last = {'iterations': np.zeros(self.lanes, np.int32), 'residual': np.zeros(self.lanes, np.float32)}
        self.history = []

    # ----- bookkeeping -------------------------------------------------------------------
    @property
    def drive_scale(self):
        return (1.0 - self.feedback) if (self.normalize_feedback and self.feedback != 0.0) else 1.0

    def contraction_bound(self):
        """Guaranteed Lipschitz bound of the loop in the max norm; < 1 proves global convergence."""
        return abs(self.feedback) * float(np.abs(self.gain).max()) * float(np.abs(self.efficacy).max())

    def describe(self):
        return {'graph': self.graph.name, 'learned_output': self.output_weight is not None, 'neurons': int(self.n), 'dimensions': self.dimensions, 'seed': self.seed,
                'memory': self.memory, 'drive': self.drive, 'feedback': self.feedback, 'step_size': self.step_size,
                'drive_scale': self.drive_scale, 'max_iterations': self.max_iterations, 'tolerance': self.tolerance,
                'gain_mean': float(self.gain.mean()), 'gain_max': float(np.abs(self.gain).max()),
                'negative_efficacy_fraction': float(np.mean(self.efficacy < 0)),
                'contraction_bound': self.contraction_bound(), 'spectral_radius': self.spectral_radius}

    def reset(self, lanes=None):
        if lanes is not None:
            self.lanes = int(lanes)
        self.state = np.zeros((self.n, self.lanes), np.float32)
        self.tokens = 0
        self.last = {'iterations': np.zeros(self.lanes, np.int32), 'residual': np.zeros(self.lanes, np.float32)}
        self.history = []

    def project_input(self, embeddings):
        """Matched direct-input baseline (FLM's): a seeded random projection, no graph computation."""
        embeddings = np.asarray(embeddings, np.float32)
        code = embeddings @ self.input_projection
        return rms_normalize(code, axis=-1)

    # ----- dynamics ----------------------------------------------------------------------
    def _propagate(self, v, mode):
        """g * W (s * v), with FLM's relabeling control applied in graph-native indexing."""
        if mode == 'shuffled':
            v = v[self.permutation]
        y = self.gain[:, None] * spmm(self.matrix, self.efficacy[:, None] * v, self.kernel)
        return y[self.inverse] if mode == 'shuffled' else y

    def step(self, embeddings, mode='intact', iterations=None):
        """One token for every lane. embeddings: (E,) for a single lane or (lanes, E)."""
        if mode not in MODES:
            raise ValueError('Unknown graph control.')
        embeddings = np.asarray(embeddings, np.float32)
        single = embeddings.ndim == 1
        if single:
            embeddings = embeddings[None]
        if embeddings.shape[0] != self.lanes:
            raise ValueError(f'Expected {self.lanes} lane(s) of embeddings, got {embeddings.shape[0]}.')
        code = self.project_input(embeddings)                              # (lanes, D)
        u = code[:, self.input_bins].T * self.input_sign[:, None]          # (n, lanes)
        m = self.memory * self.state + self.drive * u
        if self.drive_scale != 1.0:
            m = m * np.float32(self.drive_scale)
        limit = self.max_iterations if iterations is None else max(1, int(iterations))
        used = np.zeros(self.lanes, np.int32)
        residual = np.full(self.lanes, np.inf, np.float32)
        if mode == 'no_edges':
            self.state = np.zeros_like(self.state)
            residual[:] = 0
        else:
            if self.feedback == 0.0 and self.step_size == 1.0:
                limit = 1  # every further iteration would recompute the same state
            z = np.array(self.state, dtype=np.float32, copy=True)
            active = np.ones(self.lanes, bool)
            for k in range(limit):
                # Only lanes that have not converged are propagated; lanes are independent in the
                # kernel and in SciPy, so a lane's numbers never depend on its batch companions.
                idx = np.flatnonzero(active)
                z_active = z[:, idx]
                v = m[:, idx] + np.float32(self.feedback) * z_active if self.feedback != 0.0 else m[:, idx]
                target = np.tanh(self._propagate(np.ascontiguousarray(v), mode))
                candidate = target if self.step_size == 1.0 else \
                    np.float32(1.0 - self.step_size) * z_active + np.float32(self.step_size) * target
                change = np.sqrt(np.mean((candidate - z_active) ** 2, axis=0))
                z[:, idx] = candidate
                used[idx] = k + 1
                residual[idx] = change
                if self.tolerance is not None:
                    active[idx[change < self.tolerance]] = False
                    if not active.any():
                        break
            self.state = np.ascontiguousarray(z, dtype=np.float32)
        if self.output_weight is None:
            features = np.stack([np.bincount(self.output_bins, weights=self.state[:, l] * self.output_sign,
                                             minlength=self.dimensions).astype(np.float32) / self.output_scale
                                 for l in range(self.lanes)])
        else:
            features = np.stack([np.bincount(self.output_bins, weights=self.state[:, l] * self.output_weight,
                                             minlength=self.dimensions).astype(np.float32)
                                 for l in range(self.lanes)])
        # Bias-free scaling: a disconnected graph cannot generate an adapter signal.
        features = rms_normalize(features, axis=-1)
        self.tokens += 1
        self.last = {'iterations': used, 'residual': residual}
        self.history.append((float(used.mean()), float(np.where(np.isfinite(residual), residual, 0).mean()),
                             float(np.sqrt(np.mean(self.state ** 2)))))
        return features[0] if single else features

    def sequence(self, embeddings, mode='intact'):
        """Causal features for a whole sequence: (T, E) -> (T, D) or (lanes, T, E) -> (lanes, T, D). Resets first."""
        embeddings = np.asarray(embeddings, np.float32)
        if embeddings.ndim == 2:
            self.reset(1)
            return np.stack([self.step(e, mode) for e in embeddings])
        self.reset(embeddings.shape[0])
        return np.stack([self.step(embeddings[:, t], mode) for t in range(embeddings.shape[1])], axis=1)

    def telemetry(self):
        indices = np.linspace(0, self.n - 1, 96).astype(int)
        finite = np.isfinite(self.last['residual'])
        return {'updates': self.tokens, 'state_rms': float(np.sqrt(np.mean(self.state ** 2))),
                'iterations': self.last['iterations'].tolist(),
                'residual': float(self.last['residual'][finite].mean()) if finite.any() else None,
                'saturation': float(np.mean(np.abs(self.state) > 0.99)),
                'contraction_bound': self.contraction_bound(), 'spectral_radius': self.spectral_radius,
                'sampled_neuron_ids': [str(int(self.graph.ids[i])) for i in indices],
                'sampled_states': self.state[indices, 0].tolist(), 'units': 'abstract rate-model state'}
