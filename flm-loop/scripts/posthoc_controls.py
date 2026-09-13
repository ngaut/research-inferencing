#!/usr/bin/env python
"""Post-hoc controls for a train_adapter.py run: (1) a constant-feature adapter — the same
architecture fed an all-ones feature, so it can only learn a fixed logit offset (a corpus prior);
if it matches the fly and direct adapters, their gain over the base model is that prior, not the
graph; (2) a sweep of test-time-training learning rates and norm bounds for the run's trained fly
adapter, scoring each held-out window's second half after fast weights learned on its first half."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flmloop import Graph, load_kernel  # noqa: E402
from flmloop.llm import LoopedFLM, FlyAdapter, FastWeights  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_adapter import text_records  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--backbone', type=Path)
    p.add_argument('--graph')
    p.add_argument('--text-corpus', type=Path, required=True)
    p.add_argument('--text-holdout', type=Path)
    p.add_argument('--chunk-tokens', type=int, default=256)
    p.add_argument('--train-chunks', type=int, default=96)
    p.add_argument('--val-chunks', type=int, default=16)
    p.add_argument('--test-chunks', type=int, default=24)
    p.add_argument('--ttt-lrs', default='0.05,0.5,5,50')
    p.add_argument('--ttt-max-norms', default='0.5,5')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    manifest = json.loads((a.run / 'run.json').read_text())
    a.seed = manifest['training']['seed']
    backbone = a.backbone or Path(manifest['backbone'])
    graph = Graph.from_spec(a.graph or manifest['graph'], seed=a.seed)
    block = {}
    if manifest.get('block_file'):
        arrays = np.load(a.run / manifest['block_file']); block = {k: arrays[k] for k in arrays.files}
    m = LoopedFLM(backbone, graph, checkpoint=a.run / 'adapter.safetensors', device=a.device, kernel=load_kernel(), threads=a.threads, **block)
    torch.manual_seed(a.seed)
    splits = text_records(m, a)
    lanes = 8

    @torch.no_grad()
    def hidden_and_features(records, need_features):
        hidden, feats = [], []
        for offset in range(0, len(records), lanes):
            group = records[offset:offset + lanes]
            for item in group:
                hidden.append(m.inner(torch.as_tensor(item['ids'][None, :], device=m.device), use_cache=False).last_hidden_state[0].float().cpu().numpy())
            if need_features:
                r = m.reservoir(lanes=len(group))
                steps = max(len(x['ids']) for x in group)
                out = [[] for _ in group]
                for t in range(steps):
                    emb = np.stack([m.embeddings[int(x['ids'][t])] if t < len(x['ids']) else np.zeros(m.hidden_size, np.float32) for x in group])
                    f = r.step(emb)
                    for j, x in enumerate(group):
                        if t < len(x['ids']):
                            out[j].append(f[j].copy())
                feats.extend(np.asarray(o, np.float32) for o in out)
        return hidden, feats

    train_h, _ = hidden_and_features(splits['train'], False)
    val_h, _ = hidden_and_features(splits['validation'], False)
    test_h, test_f = hidden_and_features(splits['test'], True)
    D = m.interface['dimensions']

    def stack(hs, records):
        return (torch.as_tensor(np.concatenate([h[x['mask']] for h, x in zip(hs, records)]), device=m.device),
                torch.as_tensor(np.concatenate([x['labels'][x['mask']] for x in records]), device=m.device))
    train_x, train_y = stack(train_h, splits['train']); val_x, val_y = stack(val_h, splits['validation']); test_x, test_y = stack(test_h, splits['test'])

    @torch.no_grad()
    def nll(adapter, hidden, labels, features):
        total = 0.0
        for i in range(0, len(labels), 32):
            logits, _, _ = m.scores(hidden[i:i + 32].to(m.base.dtype), features[i:i + 32], 'intact', adapter)
            total += float(torch.nn.functional.cross_entropy(logits, labels[i:i + 32], reduction='sum'))
        return total / len(labels)

    # ---- constant-feature adapter: FLM's recipe, features = ones ----
    ones = lambda n: torch.ones(n, D, device=m.device)
    adapter = FlyAdapter(m.hidden_size, features=D, scale=m.interface['adapter_scale']).to(m.device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=3e-4, weight_decay=0.01)
    best, best_state = float('inf'), None
    for epoch in range(3):
        adapter.train()
        for ix in np.array_split(np.random.default_rng(a.seed + 100 + epoch).permutation(len(train_y)), max(1, len(train_y) // 32)):
            ix = torch.as_tensor(ix, device=m.device)
            opt.zero_grad(set_to_none=True)
            logits, base, _ = m.scores(train_x[ix].to(m.base.dtype), ones(len(ix)), adapter=adapter)
            loss = torch.nn.functional.cross_entropy(logits, train_y[ix]) + 0.5 * torch.nn.functional.kl_div(torch.log_softmax(logits, -1), torch.softmax(base, -1), reduction='batchmean')
            loss.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1); opt.step()
        adapter.eval()
        dev = nll(adapter, val_x, val_y, ones(len(val_y)))
        if dev < best:
            best, best_state = dev, {k: v.clone() for k, v in adapter.state_dict().items()}
    adapter.load_state_dict(best_state); adapter.eval()
    report = {'constant_feature_adapter': {'nll': nll(adapter, test_x, test_y, ones(len(test_y))), 'validation_nll': best}}
    report['constant_feature_adapter']['perplexity'] = math.exp(report['constant_feature_adapter']['nll'])

    # ---- TTT sweep on the trained fly adapter ----
    feats_test = [torch.as_tensor(f, device=m.device) for f in test_f]
    report['ttt_sweep'] = {}
    for lr in map(float, a.ttt_lrs.split(',')):
        for max_norm in map(float, a.ttt_max_norms.split(',')):
            fast = FastWeights(m.adapter, learning_rate=lr, decay=0.0, max_norm=max_norm, chunk=32)
            total = count = 0; norms = []
            for h, f, x in zip(test_h, feats_test, splits['test']):
                fast.reset()
                hidden = torch.as_tensor(h, device=m.device).to(m.base.dtype); y = torch.as_tensor(x['labels'], device=m.device)
                prefix = np.where(~x['mask'])[0]; answer = np.where(x['mask'])[0]
                fast.learn(lambda hh, ff, d: m.scores(hh, ff, 'intact', fast=d)[0], hidden[prefix], f[prefix], y[prefix])
                with torch.no_grad():
                    logits, _, _ = m.scores(hidden[answer], f[answer], 'intact', fast=fast.delta)
                    total += float(torch.nn.functional.cross_entropy(logits, y[answer], reduction='sum')); count += len(answer)
                norms.append(fast.norm())
            report['ttt_sweep'][f'lr={lr},max_norm={max_norm}'] = {'nll': total / count, 'mean_fast_norm': float(np.mean(norms))}
            print(json.dumps({'ttt': {'lr': lr, 'max_norm': max_norm, 'nll': total / count, 'fast_norm': float(np.mean(norms))}}), flush=True)
    report['fly_adapter_no_ttt'] = {'nll': nll(m.adapter, test_x, test_y, torch.cat([f[x['mask']] for f, x in zip(feats_test, splits['test'])]))}
    (a.run / 'posthoc.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
