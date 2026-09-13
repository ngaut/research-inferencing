#!/usr/bin/env python
"""Chat with a trained LoopedFLM run locally, optionally with test-time-trained fast weights (--ttt).
No server, no logging; conversations stay in memory."""
import argparse
import json
import secrets
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flmloop import Graph, load_kernel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs' / 'looped-v1')
    parser.add_argument('--backbone', type=Path, help='defaults to the backbone recorded in run.json')
    parser.add_argument('--graph', help='defaults to the graph recorded in run.json')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--mode', choices=['intact', 'no_edges', 'shuffled', 'base'], default='intact')
    parser.add_argument('--ttt', action='store_true', help='learn per-conversation fast weights from the prompt before answering')
    parser.add_argument('--prompt', help='answer once and exit; omit for interactive chat')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max-tokens', type=int, default=160)
    parser.add_argument('--telemetry', action='store_true', help='print loop/TTT telemetry after each reply')
    args = parser.parse_args()
    checkpoint = args.run / 'adapter.safetensors'
    if not checkpoint.is_file() or not (args.run / 'run.json').is_file():
        parser.error('No completed training run. Run scripts/train_adapter.py first, or select --run.')
    manifest = json.loads((args.run / 'run.json').read_text())
    backbone = args.backbone or Path(manifest['backbone'])
    graph = Graph.from_spec(args.graph or manifest['graph'], seed=manifest['training']['seed'])
    block = {}
    if manifest.get('block_file'):
        arrays = np.load(args.run / manifest['block_file'])
        block = {k: arrays[k] for k in arrays.files}
    from flmloop.llm import LoopedFLM
    print('Loading the backbone and the looped fly graph…', file=sys.stderr, flush=True)
    model = LoopedFLM(backbone, graph, checkpoint=checkpoint, device=args.device, kernel=load_kernel(), **block)
    history = []
    if args.prompt is None:
        print('Type /new to clear this conversation, /quit to exit. Chats are not saved.')
    while True:
        try:
            prompt = args.prompt if args.prompt is not None else input('\nYou: ').strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if not prompt:
            if args.prompt is not None:
                parser.error('--prompt cannot be empty')
            continue
        if args.prompt is None and prompt == '/quit':
            return
        if args.prompt is None and prompt == '/new':
            history.clear(); print('New conversation.'); continue
        pending = history + [{'role': 'user', 'content': prompt}]
        while True:
            try:
                model.prompt_ids(pending); break
            except ValueError:
                if len(pending) <= 1:
                    print('That message is too long. Try a shorter message.'); pending = None; break
                pending = pending[2:]
        if pending is None:
            continue
        print('FLM-Loop: ', end='', flush=True)
        text, last = '', None
        for event in model.generate(pending, mode=args.mode, max_tokens=args.max_tokens, ttt=args.ttt,
                                    seed=args.seed if args.seed is not None else secrets.randbits(31), telemetry=args.telemetry):
            if event['type'] == 'token':
                if event['text'].startswith(text):
                    print(event['text'][len(text):], end='', flush=True)
                text = event['text']; last = event
            elif event['type'] == 'done':
                text = event['text']
                if args.telemetry:
                    print('\n' + json.dumps({'graph': last['graph'] if last else None, 'ttt': event['ttt'],
                                             'logit_delta_rms': last['logit_delta_rms'] if last else None}, indent=None), file=sys.stderr)
        print()
        if text.strip():
            history = pending + [{'role': 'assistant', 'content': text}]
        if args.prompt is not None:
            return


if __name__ == '__main__':
    main()
