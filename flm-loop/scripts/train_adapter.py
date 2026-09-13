#!/usr/bin/env python
"""Train FLM-style readouts on looped-connectome features.

Port of nftechie/flm scripts/train_conversation.py (MIT) to any local backbone, any Graph, and the
looped block. Trains the fly adapter (intact graph) and the parameter-matched direct-input control,
selects each on validation NLL, and evaluates held-out conversations in base / intact / shuffled /
no_edges modes — plus the intact adapter with test-time-trained fast weights (learned on each
conversation's non-answer prefix only, never on the answer being scored).

Outputs: adapter.safetensors, direct-control.safetensors, block.npz (gains/efficacies/pooling if
non-default), run.json (pinned hashes of backbone, graph, block, adapter, interface), report.json,
training.json, selection.json. A completed run is never overwritten.
"""
import argparse
import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flmloop import Graph, load_kernel, neurotransmitter_signs  # noqa: E402
from flmloop.llm import (LoopedFLM, FlyAdapter, FastWeights, backbone_signature, graph_signature,  # noqa: E402
                         block_signature, sha256_file)

ROOT = Path(__file__).resolve().parents[1]


def text_records(m, a):
    """Plain-text mode. Training and validation windows of chunk_tokens+1 tokens are drawn without
    overlap from --text-corpus; test windows from --text-holdout (or the corpus tail). Train and
    validation supervise every position; test windows score their second half only, so the
    fly_adapter_ttt row learns fast weights on a window's first half and is scored on the second."""
    def tokens(path):
        return np.asarray(m.encode(path.read_text(errors='replace')), np.int64)
    corpus = tokens(a.text_corpus)
    if a.text_holdout:
        holdout = tokens(a.text_holdout)
    else:
        cut = int(len(corpus) * 0.9)
        corpus, holdout = corpus[:cut], corpus[cut:]
    span = a.chunk_tokens + 1
    rng = np.random.default_rng(a.seed)

    def windows(source, count, prefix):
        starts = rng.permutation((len(source) - span) // span)[:count] * span
        if len(starts) < count:
            raise SystemExit(f'{prefix}: text too short for {count} windows of {span} tokens.')
        return [source[s:s + span] for s in starts], starts

    train_windows, _ = windows(corpus, a.train_chunks + a.val_chunks, 'corpus')
    test_windows, _ = windows(holdout, a.test_chunks, 'holdout')

    def make(ids, key, score_from):
        ids = np.asarray(ids, np.int64)
        return {'key': key, 'ids': ids[:-1], 'labels': ids[1:], 'mask': np.arange(len(ids) - 1) >= score_from,
                'conversation_sha256': hashlib.sha256(ids.tobytes()).hexdigest()}
    train = [make(w, f'text:train:{i}', 0) for i, w in enumerate(train_windows[:a.train_chunks])]
    validation = [make(w, f'text:validation:{i}', 0) for i, w in enumerate(train_windows[a.train_chunks:])]
    test = [make(w, f'text:test:{i}', a.chunk_tokens // 2) for i, w in enumerate(test_windows)]
    return {'train': train, 'validation': validation, 'test': test}


def shared_prefix(records):
    """Length of the token prefix common to every record, capped so it never reaches any record's
    answer tokens (positions < n get prefix-state features; answers must be pooled per record)."""
    all_ids = [x['ids'] for x in records]
    n = 0
    while n < min(map(len, all_ids)) and all(ids[n] == all_ids[0][n] for ids in all_ids):
        n += 1
    first_answer = min(int(np.argmax(x['mask'])) if x['mask'].any() else len(x['mask']) for x in records)
    return min(n, first_answer)


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backbone', type=Path, required=True, help='local Hugging Face causal LM directory with a chat template')
    p.add_argument('--graph', default='random:2000', help="prepared FLM graph folder, or 'random:N[:degree]'")
    p.add_argument('--nt', type=Path, help='MaleCNS body-neurotransmitters feather (signed efficacies)')
    p.add_argument('--block', type=Path, help='block.npz from train_graph.py (gain, efficacy, output_weight)')
    p.add_argument('--conversations', type=Path, default=ROOT / 'data' / 'conversations.json')
    p.add_argument('--text-corpus', type=Path, help='plain-text mode: training/validation windows are drawn from this file (no chat template)')
    p.add_argument('--text-holdout', type=Path, help='plain-text mode: held-out text for the test windows (defaults to the tail of --text-corpus)')
    p.add_argument('--chunk-tokens', type=int, default=256)
    p.add_argument('--train-chunks', type=int, default=96)
    p.add_argument('--val-chunks', type=int, default=16)
    p.add_argument('--test-chunks', type=int, default=24)
    p.add_argument('--output', type=Path, default=ROOT / 'runs' / 'looped-v1')
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--kl-weight', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=27)
    p.add_argument('--max-conversations', type=int, default=0)
    p.add_argument('--lanes', type=int, default=8)
    p.add_argument('--prefix-limit', type=int, default=256)
    p.add_argument('--min-answer', type=int, default=5)
    p.add_argument('--max-answer', type=int, default=144)
    p.add_argument('--feedback', type=float)
    p.add_argument('--step-size', type=float)
    p.add_argument('--max-iterations', type=int)
    p.add_argument('--tolerance', type=float)
    p.add_argument('--dimensions', type=int)
    p.add_argument('--adapter-scale', type=float)
    p.add_argument('--max-logit-rms', type=float)
    p.add_argument('--target-radius', type=float, default=0.95, help='gain calibration for signed graphs without --block')
    p.add_argument('--system-text')
    p.add_argument('--ttt-lr', type=float, default=0.05)
    p.add_argument('--ttt-chunk', type=int, default=32)
    p.add_argument('--ttt-max-norm', type=float, default=0.5)
    return p.parse_args()


def main():
    a = parse()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / 'run.json').exists():
        raise SystemExit('Completed run exists; use a new output.')
    started = time.monotonic()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    kernel = load_kernel()
    graph = Graph.from_spec(a.graph, seed=a.seed)
    interface = {k: v for k, v in {'feedback': a.feedback, 'step_size': a.step_size, 'max_iterations': a.max_iterations,
                                   'tolerance': a.tolerance, 'dimensions': a.dimensions, 'adapter_scale': a.adapter_scale,
                                   'max_logit_rms': a.max_logit_rms}.items() if v is not None}
    block = {'gain': None, 'efficacy': None, 'output_weight': None}
    block_source = None
    if a.block:
        arrays = np.load(a.block)
        block = {k: (arrays[k] if k in arrays else None) for k in block}
        block_source = {'file': str(a.block), 'sha256': sha256_file(a.block)}
    elif a.nt:
        from flmloop.probes import calibrate_gain
        efficacy, counts = neurotransmitter_signs(graph.ids, a.nt)
        scale, estimate = calibrate_gain(graph, efficacy, target_radius=a.target_radius, kernel=kernel, seed=a.seed)
        block = {'gain': np.full(graph.n, scale, np.float32), 'efficacy': efficacy, 'output_weight': None}
        block_source = {'neurotransmitters': str(a.nt), 'counts': dict(counts), 'gain': scale, 'radius_estimate': estimate}
    m = LoopedFLM(a.backbone, graph, checkpoint=None, device=a.device, interface=interface, system_text=a.system_text,
                  kernel=kernel, threads=a.threads, fast_weights={'learning_rate': a.ttt_lr, 'chunk': a.ttt_chunk, 'max_norm': a.ttt_max_norm},
                  **block)
    print(json.dumps({'stage': 'loaded', 'device': m.device, 'backbone_parameters': sum(x.numel() for x in m.base.parameters()),
                      'graph': graph.summary(), 'interface': m.interface}), flush=True)

    # ----- records: one supervised answer per assistant turn, exactly the prefix used at inference -----
    def record(messages, key):
        context = [dict(x) for x in messages[:-1]]; target = messages[-1]['content']
        while True:
            prefix = m.prompt_ids(context, max_context=10 ** 9)
            if len(prefix) <= a.prefix_limit or len(context) <= 1:
                break
            context = context[2:]
        if len(prefix) > a.prefix_limit + 64:
            return None
        complete = m.encode(m.render(context + [{'role': 'assistant', 'content': target}], add_generation_prompt=False))
        if complete[:len(prefix)] != prefix:
            raise ValueError('Training/inference prefix mismatch.')
        answer = complete[len(prefix):]
        if m.tokenizer.eos_token_id in answer:
            answer = answer[:answer.index(m.tokenizer.eos_token_id) + 1]
        if not a.min_answer <= len(answer) <= a.max_answer:
            return None
        ids = np.array(prefix + answer, dtype=np.int64)
        mask = np.arange(len(ids) - 1) >= len(prefix) - 1
        sha = hashlib.sha256(json.dumps(context + [{'role': 'assistant', 'content': target}], sort_keys=True).encode()).hexdigest()
        return {'key': key, 'ids': ids[:-1], 'labels': ids[1:], 'mask': mask, 'conversation_sha256': sha}

    if a.text_corpus:
        splits = text_records(m, a)
    else:
        conversations = json.loads(a.conversations.read_text())['conversations']
        if a.max_conversations:
            conversations = conversations[:a.max_conversations]
        order = np.random.default_rng(a.seed).permutation(len(conversations))
        n_val = max(1, int(round(len(order) * 0.15))); n_test = max(1, int(round(len(order) * 0.15)))
        groups = {'test': order[:n_test], 'validation': order[n_test:n_test + n_val], 'train': order[n_test + n_val:]}
        splits = {}
        for split, indices in groups.items():
            items, seen = [], set()
            for i in indices:
                lines = conversations[int(i)]
                messages = [{'role': 'user' if j % 2 == 0 else 'assistant', 'content': s} for j, s in enumerate(lines)]
                for j in range(1, len(messages), 2):
                    item = record(messages[:j + 1], f'{split}:{int(i)}:{j}')
                    if item is not None and item['conversation_sha256'] not in seen:
                        seen.add(item['conversation_sha256']); items.append(item)
            splits[split] = items
    if not all(splits.values()):
        raise SystemExit(f'Every split needs records: {{k: len(v) for k, v in splits.items()}}')
    selection = {split: [{k: v for k, v in x.items() if k in ('key', 'conversation_sha256')} for x in records] for split, records in splits.items()}
    plan = {**{k: len(v) for k, v in splits.items()}, 'epochs': a.epochs, 'seed': a.seed, 'learning_rate': a.learning_rate,
            'batch_size': a.batch_size, 'KL_weight': a.kl_weight, 'lanes': a.lanes,
            'selection': 'lowest validation NLL, separately for fly and matched direct-input control', 'test_policy': 'only after selection',
            'ttt': m.fast_config, 'conversations_sha256': None if a.text_corpus else sha256_file(a.conversations),
            'text_mode': None if not a.text_corpus else {'corpus': str(a.text_corpus), 'corpus_sha256': sha256_file(a.text_corpus),
                                                        'holdout': None if not a.text_holdout else str(a.text_holdout),
                                                        'holdout_sha256': None if not a.text_holdout else sha256_file(a.text_holdout),
                                                        'chunk_tokens': a.chunk_tokens, 'test_scores_second_half': True}}
    (a.output / 'selection.json').write_text(json.dumps(selection, indent=2)); (a.output / 'plan.json').write_text(json.dumps(plan, indent=2))
    all_ids = [x['ids'] for records in splits.values() for x in records]
    n = shared_prefix([x for records in splits.values() for x in records])
    prefix_states, prefix_features = {}, {}
    for mode in ['intact', 'shuffled']:
        r = m.reservoir()
        prefix_features[mode] = np.stack([r.step(m.embeddings[int(token)], mode) for token in all_ids[0][:n]]) if n else np.zeros((0, m.interface['dimensions']), np.float32)
        prefix_states[mode] = r.state[:, 0].copy()
    prefix_direct = m.reservoir().project_input(m.embeddings[np.asarray(all_ids[0][:n], np.int64)]) if n else np.zeros((0, m.interface['dimensions']), np.float32)
    print(json.dumps({'stage': 'plan', 'counts': {k: len(v) for k, v in splits.items()}, 'shared_prefix_tokens': n}), flush=True)

    # ----- feature extraction: hidden states from the frozen backbone, features from the looped block -----
    @torch.no_grad()
    def extract(split, records, keep_all=False):
        modes = ['intact'] + (['shuffled'] if split == 'test' else [])
        columns = {key: [] for key in ['hidden', 'labels', 'direct', 'mask'] + modes}
        token_ids = np.unique(np.concatenate([item['ids'] for item in records]))
        embedding_values = m.embeddings[token_ids]
        lookup = {int(token): i for i, token in enumerate(token_ids)}
        embedded = [embedding_values[[lookup[int(token)] for token in item['ids'][n:]]] for item in records]
        for item in records:
            keep = np.ones(len(item['ids']), bool) if keep_all else item['mask']
            h = m.inner(torch.as_tensor(item['ids'][None, :], device=m.device), use_cache=False).last_hidden_state[0]
            columns['hidden'].append(h[torch.as_tensor(keep, device=m.device)].float().cpu().numpy().astype(np.float16))
            columns['labels'].append(item['labels'][keep]); columns['mask'].append(item['mask'][keep])
        for mode in modes:
            for offset in range(0, len(records), a.lanes):
                group = records[offset:offset + a.lanes]; vectors = embedded[offset:offset + a.lanes]
                r = m.reservoir(lanes=len(group))
                r.state = np.repeat(prefix_states[mode][:, None], len(group), axis=1).astype(np.float32); r.tokens = n
                pooled = [[] for _ in group]; direct = [[] for _ in group]
                for t in range(max(map(len, vectors))):
                    emb = np.stack([v[t] if t < len(v) else np.zeros(m.hidden_size, np.float32) for v in vectors])
                    f = r.step(emb, mode)
                    for j, item in enumerate(group):
                        if t < len(vectors[j]) and (keep_all or item['mask'][n + t]):
                            pooled[j].append(f[j].copy())
                            if mode == 'intact':
                                direct[j].append(r.project_input(emb[j]))
                if keep_all:  # the shared prefix positions get the features computed once for every record
                    for j in range(len(group)):
                        pooled[j] = list(prefix_features[mode]) + pooled[j]
                        if mode == 'intact':
                            direct[j] = list(prefix_direct) + direct[j]
                columns[mode].extend(np.asarray(row, np.float32) for row in pooled)
                if mode == 'intact':
                    columns['direct'].extend(np.asarray(row, np.float32) for row in direct)
                print(json.dumps({'stage': 'graph-batch', 'split': split, 'mode': mode, 'rows': min(offset + a.lanes, len(records)),
                                  'total': len(records), 'elapsed': round(time.monotonic() - started, 1)}), flush=True)
        return columns

    per_record = {split: extract(split, records, keep_all=(split == 'test')) for split, records in splits.items()}
    features = {split: {k: np.concatenate(v) for k, v in cols.items()} for split, cols in per_record.items()}
    for split in ('train', 'validation'):
        features[split] = {k: v[features[split]['mask']] if k != 'mask' else v for k, v in features[split].items()}
    test_all = per_record['test']

    def batches(data, kind, order):
        for offset in range(0, len(order), a.batch_size):
            ix = order[offset:offset + a.batch_size]
            yield (torch.as_tensor(data['hidden'][ix], device=m.device), torch.as_tensor(data[kind][ix], device=m.device),
                   torch.as_tensor(data['labels'][ix], device=m.device))

    @torch.no_grad()
    def evaluate(adapter, data, kind, positions=None):
        total = kl_sum = flips = rms_sum = count = 0
        order = np.arange(len(data['labels'])) if positions is None else positions
        for h, f, y in batches(data, kind, order):
            logits, base, _ = m.scores(h.to(m.base.dtype), f, 'base' if adapter is None else 'intact', adapter)
            total += float(torch.nn.functional.cross_entropy(logits, y, reduction='sum'))
            logp = torch.log_softmax(base, -1); logq = torch.log_softmax(logits, -1)
            kl_sum += float((logp.exp() * (logp - logq)).sum()); flips += int((base.argmax(-1) != logits.argmax(-1)).sum())
            rms_sum += float((logits - base).square().mean(-1).sum()); count += len(y)
        return {'nll': total / count, 'perplexity': math.exp(total / count), 'mean_KL_base_to_mode': max(0, kl_sum / count),
                'top_token_change_fraction': flips / count, 'logit_delta_rms': math.sqrt(rms_sum / count), 'targets': count}

    # ----- adapters: fly (intact features) and matched direct-input control -----
    curves, selected = [], {}
    for kind in ['intact', 'direct']:
        torch.manual_seed(a.seed)
        adapter = FlyAdapter(m.hidden_size, features=m.interface['dimensions'], scale=m.interface['adapter_scale']).to(m.device)
        opt = torch.optim.AdamW(adapter.parameters(), lr=a.learning_rate, weight_decay=0.01)
        best, best_weights = float('inf'), None
        for epoch in range(a.epochs):
            adapter.train(); total = count = 0
            for h, f, y in batches(features['train'], kind, np.random.default_rng(a.seed + 100 + epoch).permutation(len(features['train']['labels']))):
                opt.zero_grad(set_to_none=True)
                logits, base, _ = m.scores(h.to(m.base.dtype), f, adapter=adapter)
                ce = torch.nn.functional.cross_entropy(logits, y)
                divergence = torch.nn.functional.kl_div(torch.log_softmax(logits, -1), torch.softmax(base, -1), reduction='batchmean')
                loss = ce + a.kl_weight * divergence
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite objective')
                loss.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1); opt.step()
                total += float(ce.detach()) * len(y); count += len(y)
            adapter.eval(); dev = evaluate(adapter, features['validation'], kind)
            item = {'stage': 'epoch', 'model': kind, 'epoch': epoch + 1, 'train_nll': total / count, 'validation': dev,
                    'elapsed': round(time.monotonic() - started, 1)}
            curves.append(item); print(json.dumps(item), flush=True)
            if dev['nll'] < best:
                best = dev['nll']; best_weights = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        adapter.load_state_dict(best_weights); adapter.eval(); selected[kind] = adapter
        save_file(best_weights, str(a.output / ('adapter.safetensors' if kind == 'intact' else 'direct-control.safetensors')))

    # ----- held-out evaluation, answer positions only -----
    test_mask = features['test']['mask']
    test = {k: v[test_mask] for k, v in features['test'].items() if k != 'mask'}
    report = {'base': evaluate(None, test, 'intact'), 'fly_adapter': evaluate(selected['intact'], test, 'intact'),
              'direct_input_adapter': evaluate(selected['direct'], test, 'direct'),
              'relabeled_wiring': evaluate(selected['intact'], test, 'shuffled')}
    report['no_edges'] = {**report['base'], 'exact_base_identity': True}
    # Test-time training: fast weights learned on each record's non-answer prefix, scored on its answer.
    fast = FastWeights(selected['intact'], **m.fast_config)
    ttt_total = ttt_count = 0; ttt_curves = []
    for hidden, feats, labels, mask in zip(test_all['hidden'], test_all['intact'], test_all['labels'], test_all['mask']):
        fast.reset()
        prefix = np.where(~mask)[0]; answer = np.where(mask)[0]
        h = torch.as_tensor(hidden, device=m.device).to(m.base.dtype); f = torch.as_tensor(feats, device=m.device); y = torch.as_tensor(labels, device=m.device)
        if len(prefix) >= 2:
            fast.learn(lambda hh, ff, d: m.scores(hh, ff, 'intact', selected['intact'], fast=d)[0], h[prefix], f[prefix], y[prefix])
        with torch.no_grad():
            logits, _, _ = m.scores(h[answer], f[answer], 'intact', selected['intact'], fast=fast.delta)
            ttt_total += float(torch.nn.functional.cross_entropy(logits, y[answer], reduction='sum')); ttt_count += len(answer)
        ttt_curves.append({'prefix_losses': fast.losses, 'fast_norm': fast.norm()})
    report['fly_adapter_ttt'] = {'nll': ttt_total / ttt_count, 'perplexity': math.exp(ttt_total / ttt_count), 'targets': ttt_count,
                                 'mean_fast_norm': float(np.mean([c['fast_norm'] for c in ttt_curves])),
                                 'mean_prefix_loss_drop': float(np.mean([c['prefix_losses'][0] - c['prefix_losses'][-1] for c in ttt_curves if len(c['prefix_losses']) >= 2] or [0.0]))}
    assert torch.count_nonzero(selected['intact'](torch.zeros(2, m.interface['dimensions'], device=m.device))).item() == 0
    assert all(not x.requires_grad and x.grad is None for x in m.base.parameters())
    reservoir = m.reservoir()
    block_file = None
    if any(v is not None for v in block.values()):
        block_file = a.output / 'block.npz'
        np.savez(block_file, **{k: v for k, v in block.items() if v is not None})
    manifest = {'schema': 'flm-loop-1', 'backbone': str(a.backbone), 'backbone_signature': backbone_signature(a.backbone),
                'graph': a.graph, 'graph_name': graph.name, 'graph_signature': graph_signature(graph),
                'block_signature': block_signature(reservoir), 'block_file': None if block_file is None else block_file.name,
                'block_source': block_source, 'adapter_sha256': sha256_file(a.output / 'adapter.safetensors'),
                'interface': m.interface, 'system_text': m.system['content'], 'context_tokens': m.max_context, 'sampling': m.sampling,
                'training': {**plan, 'shared_prefix_tokens': n, 'train_targets': int(len(features['train']['labels'])), 'test_targets': int(len(test['labels']))},
                'trainable_parameters': sum(x.numel() for x in selected['intact'].parameters()),
                'base_parameters': sum(x.numel() for x in m.base.parameters()), 'device': m.device,
                'claim': 'A frozen language backbone with a bounded, trained readout of a looped fly-connectome block, plus per-conversation fast weights learned at test time. Not biological language.'}
    report['notes'] = ['Held-out conversations are disjoint from fitting and validation; this is a small local split, not a benchmark.',
                       'Wiring specificity needs matched topology controls (see bench_reservoir.py); a direct-input readout is included here.',
                       'fly_adapter_ttt learns fast weights on the non-answer prefix of each test record and scores the answer; the slow adapter is unchanged.']
    for name, data in [('run', manifest), ('report', report), ('training', curves)]:
        (a.output / f'{name}.json').write_text(json.dumps(data, indent=2))
    print(json.dumps({'stage': 'complete', 'report': report, 'elapsed': round(time.monotonic() - started, 1)}), flush=True)


if __name__ == '__main__':
    main()
