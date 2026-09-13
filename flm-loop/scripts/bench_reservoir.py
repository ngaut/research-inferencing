#!/usr/bin/env python
"""Backbone-free benchmark on the fly connectome: does a looped, signed block carry more
next-token information than FLM's one-step unsigned reservoir, and does the real wiring matter?

Input: English text as bytes, one fixed random embedding per byte (the stand-in for the
backbone's token embedding). Each variant turns the byte stream into 128-d features; the
metrics ask what a linear readout can get out of them:
  online_nll   prequential next-byte log loss of a readout trained as it goes (the test-time-
               training readout; lower is better; the `direct` row is the bigram baseline)
  probe_nll    offline logistic probe, fit on the first 75% of every lane, scored on the rest
  memory       Jaeger linear memory capacity for the input code's first component
  rank         effective rank of the features
plus loop telemetry (iterations, residual, state RMS, saturation) and seconds per token.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flmloop import Graph, LoopedReservoir, load_kernel, neurotransmitter_signs, random_signs  # noqa: E402
from flmloop.plasticity import IntrinsicPlasticity, OnlineSoftmaxReadout  # noqa: E402
from flmloop.probes import memory_capacity, effective_rank, logistic_probe, calibrate_gain, spectral_radius  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_text(patterns):
    chunks = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            chunks.append(Path(path).read_bytes())
    data = b'\n\n'.join(chunks)
    if not data:
        raise SystemExit('No text found for --text patterns.')
    return data


def make_lanes(data, lanes, tokens, seed):
    """Contiguous byte windows of length tokens+1, evenly spread over the corpus from a seeded start."""
    span = tokens + 1
    if len(data) < span * lanes:
        raise SystemExit(f'Corpus too small: need {span * lanes} bytes, have {len(data)}.')
    rng = np.random.default_rng(seed)
    stride = (len(data) - span) // lanes
    start = int(rng.integers(0, max(1, stride)))
    return np.stack([np.frombuffer(data[start + i * stride:start + i * stride + span], np.uint8) for i in range(lanes)]).astype(np.int64)


def build_variants(args, graph, kernel, nt_efficacy, nt_negative):
    loop = dict(feedback=args.feedback, step_size=args.step_size, max_iterations=args.max_iterations, tolerance=args.tolerance)
    one = dict(feedback=0.0, step_size=1.0, max_iterations=1, tolerance=None)
    rewired = graph.rewired(seed=args.seed)
    random_efficacy = random_signs(graph.n, nt_negative, seed=args.seed + 1)
    table = {
        'direct': dict(graph=None),
        'flm': dict(graph=graph, efficacy=None, calibrate=False, **one),
        'flm-rewired': dict(graph=rewired, efficacy=None, calibrate=False, **one),
        'flm-signed': dict(graph=graph, efficacy=nt_efficacy, calibrate=True, **one),
        'loop-unsigned': dict(graph=graph, efficacy=None, calibrate=False, **loop),
        'loop-signed': dict(graph=graph, efficacy=nt_efficacy, calibrate=True, **loop),
        'loop-signed-rewired': dict(graph=rewired, efficacy=nt_efficacy, calibrate=True,
                                    calibrate_on=(graph if args.control_gain == 'same' else rewired), **loop),
        'loop-random-signs': dict(graph=graph, efficacy=random_efficacy, calibrate=True, **loop),
        'loop-signed-ip': dict(graph=graph, efficacy=nt_efficacy, calibrate=True, ip=True, **loop),
    }
    if nt_efficacy is None:
        needs_prior = {'flm-signed', 'loop-signed', 'loop-signed-rewired', 'loop-signed-ip', 'loop-random-signs'}
        table = {k: v for k, v in table.items() if k not in needs_prior}
    wanted = [v.strip() for v in args.variants.split(',')] if args.variants else list(table)
    return [(name, table[name]) for name in wanted]


def run_variant(name, cfg, ids, table, args, kernel):
    lanes, steps = ids.shape[0], ids.shape[1] - 1
    embeddings = table[ids[:, :-1]]                                       # (lanes, T, E)
    targets = ids[:, 1:]                                                  # next byte
    started = time.time()
    if cfg['graph'] is None:
        probe = LoopedReservoir(Graph.toy(), args.embed, dimensions=args.dims, seed=args.interface_seed)
        features = np.stack([probe.project_input(embeddings[:, t]) for t in range(steps)])   # (T, lanes, D)
        signal = features[..., 0].T.copy()
        telemetry = {'iterations': 0.0, 'residual': None, 'state_rms': None, 'saturation': None, 'gain': None,
                     'spectral_radius': None, 'contraction_bound': None}
    else:
        graph = cfg['graph']
        efficacy = cfg['efficacy']
        gain = None
        estimate = None
        if cfg.get('calibrate'):
            source = cfg.get('calibrate_on', graph)   # 'same' control: calibrate on the real graph, apply to the rewired one
            scale, estimate = calibrate_gain(source, efficacy, target_radius=args.target_radius, iterations=60, kernel=kernel, seed=args.seed)
            gain = np.full(graph.n, scale, np.float32)
        reservoir = LoopedReservoir(graph, args.embed, dimensions=args.dims, seed=args.interface_seed, lanes=lanes,
                                    drive=0.4 * args.input_scale, feedback=cfg['feedback'], step_size=cfg['step_size'],
                                    max_iterations=cfg['max_iterations'], tolerance=cfg['tolerance'], gain=gain, efficacy=efficacy, kernel=kernel)
        if estimate is not None:
            own = estimate if source is graph else spectral_radius(graph, None, efficacy, iterations=60, kernel=kernel, seed=args.seed)
            reservoir.spectral_radius = float(own['radius'] * gain[0])  # the radius this reservoir actually has
        if cfg.get('ip'):
            top = float(np.abs(reservoir.gain).max())
            rule = IntrinsicPlasticity(graph.n, target=args.ip_target, rate=args.ip_rate, smoothing=0.05, bounds=(0.25 * top, 1.5 * top))
            for t in range(min(args.warmup, steps)):
                reservoir.step(embeddings[:, t])
                rule.update(reservoir)
            reservoir.reset()
        features = np.empty((steps, lanes, args.dims), np.float32)
        signal = reservoir.project_input(embeddings.reshape(-1, args.embed))[:, 0].reshape(lanes, steps)
        history = []
        for t in range(steps):
            features[t] = reservoir.step(embeddings[:, t])
            history.append(reservoir.history[-1])
            if (t + 1) % 64 == 0:
                print(json.dumps({'variant': name, 'token': t + 1, 'elapsed': round(time.time() - started, 1),
                                  'iterations': round(float(np.mean([h[0] for h in history[-64:]])), 2),
                                  'residual': float(np.mean([h[1] for h in history[-64:]])),
                                  'state_rms': round(float(np.mean([h[2] for h in history[-64:]])), 4)}), flush=True)
        telemetry = {'iterations': float(np.mean([h[0] for h in history])), 'residual': float(np.mean([h[1] for h in history[-64:]])),
                     'state_rms': float(np.mean([h[2] for h in history])), 'saturation': reservoir.telemetry()['saturation'],
                     'gain': float(reservoir.gain.mean()), 'spectral_radius': reservoir.spectral_radius,
                     'contraction_bound': reservoir.contraction_bound(), 'negative_efficacy_fraction': float(np.mean(reservoir.efficacy < 0))}
    seconds_per_token = (time.time() - started) / steps

    readout = OnlineSoftmaxReadout(args.dims, 256, learning_rate=args.online_lr)
    for t in range(steps):
        readout.observe(features[t], targets[:, t])
    online = readout.summary()
    if args.save_features:
        folder = args.output.parent / (args.output.stem + '-features')
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f'{name}.npz', features=features.astype(np.float16), targets=targets,
                            signal=signal.astype(np.float32), online_losses=np.asarray(readout.losses, np.float32))
    split = int(steps * 0.75)
    train_x = features[:split].reshape(-1, args.dims); train_y = targets[:, :split].T.reshape(-1)
    test_x = features[split:].reshape(-1, args.dims); test_y = targets[:, split:].T.reshape(-1)
    probe = logistic_probe(train_x, train_y, test_x, test_y, classes=256, steps=args.probe_steps, l2=args.probe_l2)
    memory = memory_capacity(np.transpose(features, (1, 0, 2)), signal, max_delay=args.max_delay)
    return {'variant': name, 'seconds_per_token': seconds_per_token, 'online': online, 'probe': probe,
            'memory_capacity': memory['capacity'], 'memory_per_delay': memory['per_delay'][:12],
            'effective_rank': effective_rank(features), 'telemetry': telemetry,
            'graph': None if cfg['graph'] is None else cfg['graph'].name}


def markdown_table(results):
    rows = ['| variant | graph | online NLL | tail NLL | probe NLL | probe acc | memory cap. | eff. rank | iters | residual | state RMS | s/token |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        t = r['telemetry']
        fmt = lambda v, d=3: '—' if v is None else f'{v:.{d}f}'
        rows.append(f"| {r['variant']} | {r['graph'] or '—'} | {r['online']['online_nll']:.3f} | {r['online']['tail_nll']:.3f} | "
                    f"{r['probe']['test_nll']:.3f} | {r['probe']['test_accuracy']:.3f} | {r['memory_capacity']:.2f} | {r['effective_rank']:.1f} | "
                    f"{fmt(t['iterations'], 2)} | {fmt(t['residual'], 4) if t['residual'] is not None else '—'} | {fmt(t['state_rms'], 3)} | {r['seconds_per_token']:.3f} |")
    return '\n'.join(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--graph', type=Path, help='FLM prepared graph folder (cache/malecns_v1); omit for a random test graph')
    p.add_argument('--random-graph', type=int, default=2000, help='nodes of the random graph used when --graph is omitted')
    p.add_argument('--nt', type=Path, help='MaleCNS body-neurotransmitters feather for the sign prior')
    p.add_argument('--text', nargs='+', default=[str(ROOT.parent / '*.md')])
    p.add_argument('--lanes', type=int, default=16)
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--dims', type=int, default=128)
    p.add_argument('--embed', type=int, default=64)
    p.add_argument('--interface-seed', type=int, default=7301)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--feedback', type=float, default=0.8)
    p.add_argument('--step-size', type=float, default=0.7)
    p.add_argument('--max-iterations', type=int, default=8)
    p.add_argument('--tolerance', type=float, default=1e-3)
    p.add_argument('--target-radius', type=float, default=0.95)
    p.add_argument('--input-scale', type=float, default=1.0, help='multiplies FLM\'s input drive 0.4 (ESN input scaling; 1 = FLM)')
    p.add_argument('--control-gain', choices=['recalibrate', 'same'], default='recalibrate',
                   help="rewired signed control: recalibrate its own gain to --target-radius, or reuse the real graph's gain")
    p.add_argument('--warmup', type=int, default=128)
    p.add_argument('--ip-target', type=float, default=0.05)
    p.add_argument('--ip-rate', type=float, default=0.02)
    p.add_argument('--online-lr', type=float, default=0.05)
    p.add_argument('--probe-steps', type=int, default=300)
    p.add_argument('--probe-l2', type=float, default=3e-3)
    p.add_argument('--max-delay', type=int, default=24)
    p.add_argument('--variants', default='')
    p.add_argument('--save-features', action='store_true', help='store features, targets and per-observation online losses next to the output')
    p.add_argument('--output', type=Path, default=ROOT / 'results' / 'bench_reservoir.json')
    args = p.parse_args()

    kernel = load_kernel()
    graph = Graph.from_flm_cache(args.graph) if args.graph else Graph.random(args.random_graph, 12, seed=args.seed, name='random')
    nt_efficacy, nt_negative, nt_counts = None, 0.0, None
    if args.nt:
        nt_efficacy, nt_counts = neurotransmitter_signs(graph.ids, args.nt)
        nt_negative = float(np.mean(nt_efficacy < 0))
    data = load_text(args.text)
    ids = make_lanes(data, args.lanes, args.tokens, args.seed)
    table = np.random.default_rng(args.seed + 7).normal(size=(256, args.embed)).astype(np.float32)
    header = {'graph': graph.summary(), 'kernel': None if kernel is None else kernel.flags, 'corpus_bytes': len(data),
              'lanes': args.lanes, 'tokens': args.tokens, 'neurotransmitter_counts': dict(nt_counts) if nt_counts else None,
              'negative_efficacy_fraction': nt_negative, 'input_scale': args.input_scale, 'control_gain': args.control_gain, 'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    print(json.dumps({'stage': 'setup', **header}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for name, cfg in build_variants(args, graph, kernel, nt_efficacy, nt_negative):
        result = run_variant(name, cfg, ids, table, args, kernel)
        results.append(result)
        print(json.dumps({'stage': 'result', **result}), flush=True)
        args.output.write_text(json.dumps({'header': header, 'results': results}, indent=1))
        args.output.with_suffix('.md').write_text(markdown_table(results) + '\n')
    print(markdown_table(results))


if __name__ == '__main__':
    main()
