"""Test-time learning rules that need no backpropagation.

IntrinsicPlasticity  — local homeostatic gain rule on the looped block (per-neuron, online, bounded).
OnlineSoftmaxReadout — prequential (predict-then-update) multinomial readout: the test-time-training
                       readout for backbone-free experiments; its running log loss is the metric.
RLSReadout           — recursive least squares (FORCE-style) online linear readout for regression targets.
"""
import numpy as np


class IntrinsicPlasticity:
    """Each neuron tracks its mean squared activity and nudges its own gain toward a target level.

    The fly's readout plasticity lives at the synapse; this rule is the cheaper, older idea of
    intrinsic plasticity: keep every unit in its informative range. Gains stay inside `bounds`,
    so the loop's contraction bound (feedback * max|gain| * max|efficacy|) is never violated.
    """

    def __init__(self, n, target=0.1, rate=0.02, smoothing=0.05, bounds=(0.05, 1.0)):
        self.target = float(target)
        self.rate = float(rate)
        self.smoothing = float(smoothing)
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.activity = np.full(n, self.target, np.float32)

    def update(self, reservoir):
        squared = np.mean(reservoir.state ** 2, axis=1)
        self.activity += self.smoothing * (squared - self.activity)
        reservoir.gain *= np.exp(self.rate * (self.target - self.activity)).astype(np.float32)
        np.clip(reservoir.gain, self.bounds[0], self.bounds[1], out=reservoir.gain)
        return reservoir.gain


class OnlineSoftmaxReadout:
    """Linear softmax readout trained prequentially with Adagrad; reports running log loss and accuracy."""

    def __init__(self, dim, classes, learning_rate=0.05, l2=1e-5, bias=True, seed=0):
        self.dim, self.classes, self.bias = int(dim), int(classes), bool(bias)
        self.learning_rate, self.l2 = float(learning_rate), float(l2)
        self.width = self.dim + (1 if self.bias else 0)
        self.weights = np.zeros((self.width, self.classes), np.float64)
        self.accumulator = np.full((self.width, self.classes), 1e-8, np.float64)
        self.losses, self.hits = [], []

    def _phi(self, features):
        features = np.asarray(features, np.float64)
        return np.concatenate([features, np.ones((features.shape[0], 1))], axis=1) if self.bias else features

    def probabilities(self, features):
        logits = self._phi(features) @ self.weights
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def observe(self, features, targets):
        """Score a batch of (features, targets), then learn from it. Returns per-example log loss."""
        features = np.atleast_2d(np.asarray(features, np.float64))
        targets = np.atleast_1d(np.asarray(targets, np.int64))
        phi = self._phi(features)
        p = self.probabilities(features)
        loss = -np.log(p[np.arange(len(targets)), targets] + 1e-12)
        self.losses.extend(loss.tolist())
        self.hits.extend((p.argmax(axis=1) == targets).tolist())
        grad = p.copy()
        grad[np.arange(len(targets)), targets] -= 1
        grad = phi.T @ grad / len(targets) + self.l2 * self.weights
        self.accumulator += grad ** 2
        self.weights -= self.learning_rate * grad / np.sqrt(self.accumulator)
        return loss

    def summary(self, tail_fraction=0.25):
        losses, hits = np.asarray(self.losses), np.asarray(self.hits, np.float64)
        tail = max(1, int(len(losses) * tail_fraction))
        return {'online_nll': float(losses.mean()), 'online_accuracy': float(hits.mean()),
                'tail_nll': float(losses[-tail:].mean()), 'tail_accuracy': float(hits[-tail:].mean()),
                'observations': int(len(losses))}

    def reset(self):
        self.weights[:] = 0
        self.accumulator[:] = 1e-8
        self.losses, self.hits = [], []


class RLSReadout:
    """Recursive least squares: exact online ridge regression from features to real-valued targets."""

    def __init__(self, dim, outputs, forgetting=1.0, delta=1.0, bias=True):
        self.bias = bool(bias)
        self.width = int(dim) + (1 if self.bias else 0)
        self.outputs = int(outputs)
        self.forgetting, self.delta = float(forgetting), float(delta)
        self.reset()

    def reset(self):
        self.P = np.eye(self.width, dtype=np.float64) / self.delta
        self.weights = np.zeros((self.width, self.outputs), np.float64)

    def _phi(self, features):
        features = np.asarray(features, np.float64).ravel()
        return np.concatenate([features, [1.0]]) if self.bias else features

    def predict(self, features):
        return self._phi(features) @ self.weights

    def update(self, features, target):
        phi = self._phi(features)
        target = np.asarray(target, np.float64).ravel()
        error = target - phi @ self.weights
        p_phi = self.P @ phi
        gain = p_phi / (self.forgetting + phi @ p_phi)
        self.weights += np.outer(gain, error)
        self.P = (self.P - np.outer(gain, p_phi)) / self.forgetting
        return error
