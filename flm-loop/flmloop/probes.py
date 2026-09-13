"""Backbone-free measurements of what a reservoir's features carry.

memory_capacity — Jaeger's linear memory capacity: how well past inputs can be read out linearly.
logistic_probe  — offline multinomial logistic regression (next-token prediction from features).
effective_rank  — exp(entropy of the normalized singular-value spectrum): how many directions the
                  128 pooled features really span.
spectral_radius — power-iteration growth rate of diag(gain) W diag(efficacy): where the loop sits
                  relative to criticality.
"""
import numpy as np

from .kernel import spmm


def ridge_fit(x, y, alpha=1e-3):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    phi = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    gram = phi.T @ phi + alpha * np.eye(phi.shape[1])
    return np.linalg.solve(gram, phi.T @ y)


def ridge_predict(weights, x):
    x = np.asarray(x, np.float64)
    return np.concatenate([x, np.ones((len(x), 1))], axis=1) @ weights


def r_squared(prediction, target):
    target = np.asarray(target, np.float64); prediction = np.asarray(prediction, np.float64)
    residual = np.sum((target - prediction) ** 2)
    total = np.sum((target - target.mean()) ** 2) + 1e-12
    return float(1 - residual / total)


def memory_capacity(features, signal, max_delay=32, alpha=1e-2, train_fraction=0.7):
    """features: (lanes, T, D); signal: (lanes, T) scalar input history. Fits one ridge readout per
    delay on the first part of every lane, scores R^2 on the rest, sums the positive parts."""
    features = np.asarray(features, np.float64); signal = np.asarray(signal, np.float64)
    lanes, steps, _ = features.shape
    split = int(steps * train_fraction)
    scores = []
    for delay in range(max_delay + 1):
        x = features[:, delay:, :]; y = signal[:, :steps - delay]
        train_x = x[:, :split - delay].reshape(-1, x.shape[-1]); train_y = y[:, :split - delay].reshape(-1)
        test_x = x[:, split - delay:].reshape(-1, x.shape[-1]); test_y = y[:, split - delay:].reshape(-1)
        if len(train_y) < 8 or len(test_y) < 4:
            break
        weights = ridge_fit(train_x, train_y, alpha)
        scores.append(max(0.0, r_squared(ridge_predict(weights, test_x), test_y)))
    return {'capacity': float(sum(scores)), 'per_delay': scores}


def effective_rank(features):
    x = np.asarray(features, np.float64).reshape(-1, np.shape(features)[-1])
    x = x - x.mean(axis=0)
    s = np.linalg.svd(x, compute_uv=False)
    p = s ** 2 / max(np.sum(s ** 2), 1e-12)
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def logistic_probe(train_x, train_y, test_x, test_y, classes, steps=400, learning_rate=0.05, l2=1e-4, seed=0):
    """Full-batch Adam multinomial logistic regression with a bias; returns train/test log loss and accuracy."""
    train_x = np.asarray(train_x, np.float64); test_x = np.asarray(test_x, np.float64)
    train_y = np.asarray(train_y, np.int64); test_y = np.asarray(test_y, np.int64)
    phi = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
    phi_test = np.concatenate([test_x, np.ones((len(test_x), 1))], axis=1)
    weights = np.zeros((phi.shape[1], classes))
    m = np.zeros_like(weights); v = np.zeros_like(weights)
    b1, b2, eps = 0.9, 0.999, 1e-8
    onehot = np.zeros((len(train_y), classes)); onehot[np.arange(len(train_y)), train_y] = 1

    def evaluate(p, y):
        return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean()), float((p.argmax(axis=1) == y).mean())

    def softmax(logits):
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    for step in range(1, steps + 1):
        p = softmax(phi @ weights)
        grad = phi.T @ (p - onehot) / len(train_y) + l2 * weights
        m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad ** 2
        weights -= learning_rate * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
    train_nll, train_acc = evaluate(softmax(phi @ weights), train_y)
    test_nll, test_acc = evaluate(softmax(phi_test @ weights), test_y)
    return {'train_nll': train_nll, 'train_accuracy': train_acc, 'test_nll': test_nll, 'test_accuracy': test_acc}


def spectral_radius(graph, gain=None, efficacy=None, iterations=60, seed=0, kernel=None):
    """Power-iteration estimate of the growth rate of A = diag(gain) W diag(efficacy) on a random vector.
    Also returns the cosine between the final iterate and the uniform vector (the Perron mode of unsigned W)."""
    n = graph.n
    gain = np.ones(n, np.float32) if gain is None else np.asarray(gain, np.float32)
    efficacy = np.ones(n, np.float32) if efficacy is None else np.asarray(efficacy, np.float32)
    v = np.random.default_rng(seed).normal(size=n).astype(np.float32)
    v /= np.linalg.norm(v)
    growth = []
    for _ in range(iterations):
        w = gain * spmm(graph.matrix, efficacy * v, kernel)
        norm = float(np.linalg.norm(w))
        if norm == 0:
            return {'radius': 0.0, 'uniform_cosine': 0.0}
        growth.append(norm)
        v = w / norm
    uniform = np.ones(n, np.float32) / np.sqrt(n)
    return {'radius': float(np.exp(np.mean(np.log(growth[-10:])))), 'uniform_cosine': float(abs(v @ uniform))}


def calibrate_gain(graph, efficacy=None, target_radius=0.95, iterations=60, kernel=None, seed=0):
    """Uniform gain that puts the spectral radius of gain * W * diag(efficacy) at `target_radius`
    (echo-state-network practice). Returns (gain, power-iteration estimate for gain = 1)."""
    estimate = spectral_radius(graph, None, efficacy, iterations=iterations, seed=seed, kernel=kernel)
    return float(target_radius / max(estimate['radius'], 1e-9)), estimate
