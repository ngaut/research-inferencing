#!/usr/bin/env python
"""Robustness of the benchmark conclusions to the readout: re-scores cached features
(bench_reservoir.py --save-features) under several online learning rates and probe
regularizations, and gives paired bootstrap confidence intervals over lanes."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flmloop.plasticity import OnlineSoftmaxReadout  # noqa: E402
from flmloop.probes import logistic_probe  # noqa: E402


def online(features, targets, lr):
    readout = OnlineSoftmaxReadout(features.shape[-1], 256, learning_rate=lr)
    for t in range(features.shape[0]):
        readout.observe(features[t], targets[:, t])
    losses = np.asarray(readout.losses).reshape(features.shape[0], features.shape[1])  # (T, lanes)
    return losses


def probe(features, targets, l2, steps):
    steps_total, lanes, dims = features.shape
    split = int(steps_total * 0.75)
    fit = logistic_probe(features[:split].reshape(-1, dims), targets[:, :split].T.reshape(-1),
                         features[split:].reshape(-1, dims), targets[:, split:].T.reshape(-1), classes=256, steps=steps, l2=l2)
    return fit


def lane_bootstrap(per_lane, resamples, seed=0):
    rng = np.random.default_rng(seed)
    lanes = per_lane.shape[0]
    draws = [per_lane[rng.integers(0, lanes, lanes)].mean() for _ in range(resamples)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True, help='folder written by --save-features')
    p.add_argument('--variants', default='flm,flm-rewired,loop-unsigned,loop-signed')
    p.add_argument('--online-lrs', default='0.02,0.05,0.1')
    p.add_argument('--probe-l2s', default='0.001,0.003,0.01')
    p.add_argument('--probe-steps', type=int, default=300)
    p.add_argument('--resamples', type=int, default=1000)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    names = a.variants.split(',')
    data = {n: np.load(a.features / f'{n}.npz') for n in names}
    out = {'online': {}, 'probe': {}, 'paired': {}}
    tail_losses = {}
    for n in names:
        f = data[n]['features'].astype(np.float32); y = data[n]['targets']
        out['online'][n] = {}
        for lr in map(float, a.online_lrs.split(',')):
            losses = online(f, y, lr)
            tail = losses[int(losses.shape[0] * 0.75):]
            per_lane = tail.mean(axis=0)
            lo, hi = lane_bootstrap(per_lane, a.resamples)
            out['online'][n][str(lr)] = {'online_nll': float(losses.mean()), 'tail_nll': float(tail.mean()), 'tail_ci95_lanes': [lo, hi]}
            if lr == 0.05:
                tail_losses[n] = tail
        out['probe'][n] = {str(l2): probe(f, y, l2, a.probe_steps) for l2 in map(float, a.probe_l2s.split(','))}
        print(json.dumps({'variant': n, 'online': out['online'][n], 'probe': {k: round(v['test_nll'], 3) for k, v in out['probe'][n].items()}}), flush=True)
    base = names[0]
    for n in names[1:]:
        diff = (tail_losses[n] - tail_losses[base]).mean(axis=0)  # per lane, paired
        lo, hi = lane_bootstrap(diff, a.resamples)
        out['paired'][f'{n} - {base}'] = {'tail_nll_difference': float(diff.mean()), 'ci95_lanes': [lo, hi],
                                          'per_token_se': float((tail_losses[n] - tail_losses[base]).std() / np.sqrt(tail_losses[n].size))}
    print(json.dumps(out['paired'], indent=1))
    if a.output:
        a.output.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
