"""Differentiable looped reservoir: the same dynamics as LoopedReservoir with the per-neuron
gains, efficacies and pooling weights as parameters, so the block itself can be trained
through the loop (random iteration counts, deep supervision, truncated backprop).

The sparse propagation runs through the numpy kernel in both directions (W forward, W^T
backward), which keeps the 25.6M-edge product fast and deterministic on CPU.
"""
import numpy as np
import torch
from torch import nn

from .kernel import spmm
from .reservoir import LoopedReservoir


class SparsePropagate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, graph, kernel):
        ctx.graph, ctx.kernel = graph, kernel
        y = spmm(graph.matrix, x.detach().cpu().numpy(), kernel)
        return torch.from_numpy(y).to(x.device, x.dtype)

    @staticmethod
    def backward(ctx, grad):
        gx = spmm(ctx.graph.transpose(), grad.detach().cpu().contiguous().numpy(), ctx.kernel)
        return torch.from_numpy(gx).to(grad.device, grad.dtype), None, None


def propagate(x, graph, kernel=None):
    return SparsePropagate.apply(x, graph, kernel)


class TorchLoopedReservoir(nn.Module):
    """Trainable twin of LoopedReservoir. Feature layout: (lanes, T, D); state layout: (n, lanes)."""

    def __init__(self, graph, embedding_dim, dimensions=128, seed=7301, memory=0.6, drive=0.4, feedback=0.8,
                 step_size=0.7, normalize_feedback=True, max_iterations=8, gain=None, efficacy=None, kernel=None,
                 learn_gain=True, learn_efficacy=True, learn_output=True, learn_input=False, gain_limit=None):
        super().__init__()
        self.graph, self.kernel = graph, kernel
        self.n, self.dimensions = graph.n, int(dimensions)
        self.memory, self.drive, self.feedback = float(memory), float(drive), float(feedback)
        self.step_size, self.normalize_feedback = float(step_size), bool(normalize_feedback)
        self.max_iterations = int(max_iterations)
        reference = LoopedReservoir(graph, embedding_dim, dimensions, seed, gain=gain, efficacy=efficacy)
        f32 = lambda a: torch.as_tensor(np.asarray(a, np.float32))
        i64 = lambda a: torch.as_tensor(np.asarray(a, np.int64))
        projection = f32(reference.input_projection)
        self.input_projection = nn.Parameter(projection) if learn_input else projection
        if not learn_input:
            self.register_buffer('input_projection_buffer', projection)
        self.register_buffer('input_bins', i64(reference.input_bins))
        self.register_buffer('input_sign', f32(reference.input_sign))
        self.register_buffer('output_bins', i64(reference.output_bins))
        initial_output = f32(reference.output_sign / reference.output_scale[reference.output_bins])
        self.output_weight = nn.Parameter(initial_output) if learn_output else initial_output
        self.gain = nn.Parameter(f32(reference.gain)) if learn_gain else f32(reference.gain)
        self.efficacy = nn.Parameter(f32(reference.efficacy)) if learn_efficacy else f32(reference.efficacy)
        self.gain_limit = None if gain_limit is None else float(gain_limit)

    @property
    def drive_scale(self):
        return (1.0 - self.feedback) if (self.normalize_feedback and self.feedback != 0.0) else 1.0

    def bounded_gain(self):
        return self.gain if self.gain_limit is None else self.gain_limit * torch.tanh(self.gain / self.gain_limit)

    def project_input(self, embeddings):
        code = embeddings @ self.input_projection
        return code / torch.sqrt((code * code).mean(-1, keepdim=True) + 1e-6)

    def pool(self, state):
        """(n, lanes) -> (lanes, D): FLM's signed binning with learnable per-neuron weights."""
        pooled = torch.zeros(self.dimensions, state.shape[1], dtype=state.dtype, device=state.device)
        pooled = pooled.index_add(0, self.output_bins, self.output_weight[:, None] * state).T
        return pooled / torch.sqrt((pooled * pooled).mean(-1, keepdim=True) + 1e-6)

    def token(self, embeddings, state, iterations=None, keep_graph_from=0):
        """One token for all lanes. Returns the new state and the list of states after every iteration.
        keep_graph_from: iterations before this index run without autograd (TRM-style truncated loop backprop)."""
        code = self.project_input(embeddings)                                   # (lanes, D)
        u = code[:, self.input_bins].T * self.input_sign[:, None]               # (n, lanes)
        m = (self.memory * state + self.drive * u) * self.drive_scale
        gain, efficacy = self.bounded_gain()[:, None], self.efficacy[:, None]
        z = state
        trajectory = []
        K = self.max_iterations if iterations is None else int(iterations)
        for k in range(K):
            if k < keep_graph_from:
                with torch.no_grad():
                    v = m + self.feedback * z
                    target = torch.tanh(gain * propagate(efficacy * v, self.graph, self.kernel))
                    z = (1 - self.step_size) * z + self.step_size * target
                z = z.detach()
            else:
                v = m + self.feedback * z
                target = torch.tanh(gain * propagate(efficacy * v, self.graph, self.kernel))
                z = (1 - self.step_size) * z + self.step_size * target
            trajectory.append(z)
        return z, trajectory

    def forward(self, embeddings, state=None, iterations=None, supervise=(), detach_every=None, loop_backprop=None):
        """embeddings: (lanes, T, E). Returns features (lanes, T, D), final state, and a dict
        {k: features after k iterations} for every k in `supervise` (deep supervision targets).
        detach_every: truncate backprop through tokens every so many tokens.
        loop_backprop: keep autograd only for the last this-many inner iterations."""
        lanes, steps, _ = embeddings.shape
        if state is None:
            state = torch.zeros(self.n, lanes, dtype=embeddings.dtype, device=embeddings.device)
        K = self.max_iterations if iterations is None else int(iterations)
        keep_from = 0 if loop_backprop is None else max(0, K - int(loop_backprop))
        outputs, extra = [], {k: [] for k in supervise if k <= K}
        for t in range(steps):
            if detach_every and t and t % detach_every == 0:
                state = state.detach()
            state, trajectory = self.token(embeddings[:, t], state, K, keep_from)
            outputs.append(self.pool(state))
            for k in extra:
                extra[k].append(self.pool(trajectory[k - 1]))
        features = torch.stack(outputs, dim=1)
        return features, state, {k: torch.stack(v, dim=1) for k, v in extra.items()}

    def export(self):
        """Arrays for the numpy LoopedReservoir: gain, efficacy, output_weight (and input projection)."""
        with torch.no_grad():
            return {'gain': self.bounded_gain().cpu().numpy().astype(np.float32),
                    'efficacy': self.efficacy.detach().cpu().numpy().astype(np.float32),
                    'output_weight': self.output_weight.detach().cpu().numpy().astype(np.float32),
                    'input_projection': self.input_projection.detach().cpu().numpy().astype(np.float32)}
