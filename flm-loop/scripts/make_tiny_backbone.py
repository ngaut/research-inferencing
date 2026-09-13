#!/usr/bin/env python
"""Build a tiny, fully offline chat backbone for tests and demos: a byte-level BPE tokenizer trained
on local text, a ChatML chat template, and a two-layer Llama-style model optionally trained for a
few hundred steps on the same text. Everything is saved in Hugging Face format so LoopedFLM can load
it with local_files_only=True (no downloads).
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHATML = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
          "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")


def build(output, text_patterns, vocab_size=2048, hidden=128, layers=2, heads=4, steps=300, seed=0, verbose=True):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    torch.manual_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    texts = []
    for pattern in text_patterns:
        for path in sorted(glob.glob(pattern)):
            texts.append(Path(path).read_text(errors='replace'))
    corpus = '\n\n'.join(texts)
    if len(corpus) < 2000:
        raise SystemExit('Need more local text to build the tiny backbone.')
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=['<|endoftext|>', '<|im_start|>', '<|im_end|>'],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator([corpus[i:i + 4096] for i in range(0, len(corpus), 4096)], trainer)
    hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token='<|endoftext|>', eos_token='<|im_end|>',
                                 pad_token='<|endoftext|>', additional_special_tokens=['<|im_start|>'])
    hf.chat_template = CHATML
    hf.save_pretrained(str(output))
    config = LlamaConfig(vocab_size=len(hf), hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=layers,
                         num_attention_heads=heads, num_key_value_heads=heads, max_position_embeddings=4096,
                         bos_token_id=hf.bos_token_id, eos_token_id=hf.eos_token_id, pad_token_id=hf.pad_token_id,
                         tie_word_embeddings=False)
    model = LlamaForCausalLM(config)
    losses = []
    if steps > 0:
        # A few hundred steps of next-token training on chat-formatted local text: enough for non-flat logits.
        chunks = [corpus[i:i + 600] for i in range(0, len(corpus) - 600, 600)]
        rendered = [hf.apply_chat_template([{'role': 'user', 'content': c[:200]}, {'role': 'assistant', 'content': c[200:]}], tokenize=False) for c in chunks]
        ids = [hf(r, add_special_tokens=False)['input_ids'][:256] for r in rendered]
        ids = [x for x in ids if len(x) >= 32]
        rng = np.random.default_rng(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
        model.train()
        started = time.time()
        for step in range(steps):
            batch = [ids[i] for i in rng.integers(0, len(ids), 16)]
            length = min(len(x) for x in batch)
            x = torch.tensor([b[:length] for b in batch])
            loss = model(input_ids=x, labels=x).loss
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss))
            if verbose and (step + 1) % 50 == 0:
                print(json.dumps({'step': step + 1, 'loss': round(float(np.mean(losses[-50:])), 3), 'elapsed': round(time.time() - started, 1)}), flush=True)
        model.eval()
    model.save_pretrained(str(output), safe_serialization=True)
    (output / 'tiny_backbone.json').write_text(json.dumps({'vocab': len(hf), 'hidden': hidden, 'layers': layers, 'heads': heads,
                                                           'steps': steps, 'final_loss': losses[-1] if losses else None,
                                                           'corpus_chars': len(corpus)}, indent=2))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, default=ROOT / 'cache' / 'tiny-backbone')
    p.add_argument('--text', nargs='+', default=[str(ROOT.parent / '*.md'), str(ROOT / 'data' / '*.json')])
    p.add_argument('--vocab', type=int, default=2048)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    print(build(args.output, args.text, args.vocab, args.hidden, args.layers, args.heads, args.steps, args.seed))


if __name__ == '__main__':
    main()
