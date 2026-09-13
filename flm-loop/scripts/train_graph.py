#!/usr/bin/env python
"""Train the looped block itself — per-neuron gains, signed efficacies and pooling weights on the
fixed connectome topology — through the loop, with looped-transformer training tricks: a random
iteration count per step, deep supervision at intermediate iteration counts, and backprop through
only the last few inner iterations. Objective: next-byte prediction from the features on local text.
Exports block.npz for LoopedReservoir / train_adapter.py and reports before/after online NLL.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flmloop import Graph, LoopedReservoir, load_kernel, neurotransmitter_signs  # noqa: E402
from flmloop.plasticity import OnlineSoftmaxReadout  # noqa: E402
from flmloop.probes import calibrate_gain  # noqa: E402
from flmloop.torchloop import TorchLoopedReservoir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--graph', default='random:2000')
    p.add_argument('--nt', type=Path)
    p.add_argument('--text', nargs='+', default=[str(ROOT.parent / '*.md')])
    p.add_argument('--lanes', type=int, default=8)
    p.add_argument('--window', type=int, default=32)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--learning-rate', type=float, default=0.02)
    p.add_argument('--feedback', type=float, default=0.8)
    p.add_argument('--step-size', type=float, default=0.7)
    p.add_argument('--max-iterations', type=int, default=8)
    p.add_argument('--random-k', action='store_true', help='sample the iteration count per step (Huginn-style)')
    p.add_argument('--supervise', default='', help='comma list of iteration counts that also receive the loss (deep supervision)')
    p.add_argument('--loop-backprop', type=int, default=3, help='autograd only through the last N inner iterations')
    p.add_argument('--gain-limit', type=float, default=3.0)
    p.add_argument('--target-radius', type=float, default=0.95)
    p.add_argument('--embed', type=int, default=64)
    p.add_argument('--dims', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-tokens', type=int, default=256)
    p.add_argument('--output', type=Path, default=ROOT / 'runs' / 'block' / 'block.npz')
    a = p.parse_args()

    torch.manual_seed(a.seed)
    kernel = load_kernel()
    graph = Graph.from_spec(a.graph, seed=a.seed)
    efficacy, counts = (neurotransmitter_signs(graph.ids, a.nt) if a.nt else (None, None))
    scale, estimate = calibrate_gain(graph, efficacy, target_radius=a.target_radius, kernel=kernel, seed=a.seed)
    gain = np.full(graph.n, scale, np.float32)
    module = TorchLoopedReservoir(graph, a.embed, dimensions=a.dims, feedback=a.feedback, step_size=a.step_size,
                                  max_iterations=a.max_iterations, gain=gain, efficacy=efficacy, kernel=kernel, gain_limit=a.gain_limit)
    head = torch.nn.Linear(a.dims, 256)
    optimizer = torch.optim.Adam(list(module.parameters()) + list(head.parameters()), lr=a.learning_rate)
    data = b'\n\n'.join(Path(f).read_bytes() for pattern in a.text for f in sorted(glob.glob(pattern)))
    split = int(len(data) * 0.8)
    train_bytes, eval_bytes = np.frombuffer(data[:split], np.uint8), np.frombuffer(data[split:], np.uint8)
    table = torch.as_tensor(np.random.default_rng(a.seed + 7).normal(size=(256, a.embed)).astype(np.float32))
    supervise = tuple(int(x) for x in a.supervise.split(',') if x.strip())
    rng = np.random.default_rng(a.seed)
    initial = module.export()
    log = []
    started = time.time()
    for step in range(a.steps):
        starts = rng.integers(0, len(train_bytes) - a.window - 1, a.lanes)
        ids = np.stack([train_bytes[s:s + a.window + 1] for s in starts]).astype(np.int64)
        x = table[torch.as_tensor(ids[:, :-1])]; y = torch.as_tensor(ids[:, 1:])
        K = int(rng.integers(1, a.max_iterations + 1)) if a.random_k else a.max_iterations
        features, _, extra = module(x, iterations=K, supervise=[k for k in supervise if k < K], loop_backprop=a.loop_backprop)
        losses = [torch.nn.functional.cross_entropy(head(f).reshape(-1, 256), y.reshape(-1)) for f in [features, *extra.values()]]
        loss = sum(losses) / len(losses)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0); optimizer.step()
        entry = {'step': step + 1, 'K': K, 'loss': float(loss), 'gain_mean': float(module.bounded_gain().mean()),
                 'negative_efficacy': float((module.efficacy < 0).float().mean()), 'elapsed': round(time.time() - started, 1)}
        log.append(entry)
        if (step + 1) % 10 == 0 or step == 0:
            print(json.dumps(entry), flush=True)
    trained = module.export()

    def online_nll(arrays):
        r = LoopedReservoir(graph, a.embed, dimensions=a.dims, lanes=a.lanes, feedback=a.feedback, step_size=a.step_size,
                            max_iterations=a.max_iterations, tolerance=1e-3, gain=arrays['gain'], efficacy=arrays['efficacy'],
                            output_weight=arrays['output_weight'], kernel=kernel)
        r.input_projection = arrays['input_projection']
        stride = (len(eval_bytes) - a.eval_tokens - 1) // a.lanes
        ids = np.stack([eval_bytes[i * stride:i * stride + a.eval_tokens + 1] for i in range(a.lanes)]).astype(np.int64)
        readout = OnlineSoftmaxReadout(a.dims, 256)
        emb = table.numpy()
        for t in range(a.eval_tokens):
            readout.observe(r.step(emb[ids[:, t]]), ids[:, t + 1])
        return readout.summary()

    before, after = online_nll(initial), online_nll(trained)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, **trained)
    summary = {'graph': graph.summary(), 'neurotransmitters': None if counts is None else dict(counts), 'calibrated_gain': scale,
               'radius_estimate': estimate, 'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
               'train_log': log, 'eval_before': before, 'eval_after': after}
    a.output.with_suffix('.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({'stage': 'complete', 'eval_before': before, 'eval_after': after, 'output': str(a.output)}), flush=True)


if __name__ == '__main__':
    main()
