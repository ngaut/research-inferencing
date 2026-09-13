#!/usr/bin/env python
"""Convert a KerasNLP GPT-2 preset (model.h5 + vocab.json + merges.txt, Keras 2 H5 layout, as
served from the public keras-nlp GCS bucket) into a Hugging Face GPT2LMHeadModel directory with
a GPT-2 tokenizer and a minimal prefix-stable chat template, then sanity-check the conversion:
perplexity on a text file (a wrong layer mapping gives perplexities in the thousands) and a
greedy continuation. Gives the sandbox a real pretrained backbone when Hugging Face is unreachable.
"""
import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

TEMPLATE = ("{% for message in messages %}{{ message['role'] + ': ' + message['content'] }}"
            "{% if message['role'] == 'assistant' %}{{ '<|endoftext|>' }}{% endif %}{{ '\\n' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ 'assistant:' }}{% endif %}")


def convert(source, output, swap_layernorms=False):
    from transformers import GPT2Config, GPT2LMHeadModel
    source, output = Path(source), Path(output)
    f = h5py.File(source / 'model.h5', 'r')
    get = lambda name: torch.from_numpy(np.asarray(f[name], dtype=np.float32))
    wte = get('token_embedding/token_embedding/embeddings:0')
    wpe = get('position_embedding/position_embedding/embeddings:0')
    vocab, hidden = wte.shape
    layers = sorted({n for n in f.keys() if n.startswith('transformer_layer_')}, key=lambda s: int(s.split('_')[-1]))
    heads = f[f'{layers[0]}/{layers[0]}/multi_head_attention/query/kernel:0'].shape[1]
    config = GPT2Config(vocab_size=vocab, n_positions=wpe.shape[0], n_embd=hidden, n_layer=len(layers), n_head=heads,
                        activation_function='gelu_new', layer_norm_epsilon=1e-5, tie_word_embeddings=True,
                        bos_token_id=vocab - 1, eos_token_id=vocab - 1)
    model = GPT2LMHeadModel(config)
    state = {'transformer.wte.weight': wte, 'transformer.wpe.weight': wpe,
             'transformer.ln_f.weight': get('layer_norm/layer_norm/gamma:0'), 'transformer.ln_f.bias': get('layer_norm/layer_norm/beta:0'),
             'lm_head.weight': wte}
    first, second = ('layer_normalization_1', 'layer_normalization') if swap_layernorms else ('layer_normalization', 'layer_normalization_1')
    for i, layer in enumerate(layers):
        p = f'{layer}/{layer}/'
        q, k, v = (get(p + f'multi_head_attention/{name}/kernel:0').reshape(hidden, hidden) for name in ('query', 'key', 'value'))
        qb, kb, vb = (get(p + f'multi_head_attention/{name}/bias:0').reshape(-1) for name in ('query', 'key', 'value'))
        h = f'transformer.h.{i}.'
        state[h + 'attn.c_attn.weight'] = torch.cat([q, k, v], dim=1)          # Conv1D: (in, 3*out), heads major
        state[h + 'attn.c_attn.bias'] = torch.cat([qb, kb, vb])
        state[h + 'attn.c_proj.weight'] = get(p + 'multi_head_attention/attention_output/kernel:0').reshape(hidden, hidden)
        state[h + 'attn.c_proj.bias'] = get(p + 'multi_head_attention/attention_output/bias:0')
        state[h + 'ln_1.weight'] = get(p + f'{first}/gamma:0'); state[h + 'ln_1.bias'] = get(p + f'{first}/beta:0')
        state[h + 'ln_2.weight'] = get(p + f'{second}/gamma:0'); state[h + 'ln_2.bias'] = get(p + f'{second}/beta:0')
        state[h + 'mlp.c_fc.weight'] = get(p + 'dense/kernel:0'); state[h + 'mlp.c_fc.bias'] = get(p + 'dense/bias:0')
        state[h + 'mlp.c_proj.weight'] = get(p + 'dense_1/kernel:0'); state[h + 'mlp.c_proj.bias'] = get(p + 'dense_1/bias:0')
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if not m.endswith(('attn.bias', 'attn.masked_bias'))]
    if missing or unexpected:
        raise ValueError(f'Mapping incomplete: missing {missing[:5]} unexpected {unexpected[:5]}')
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output), safe_serialization=True)
    # Build the byte-level BPE tokenizer directly with `tokenizers` (transformers 5's
    # GPT2TokenizerFast(vocab_file=...) constructor yields an empty vocabulary).
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    bpe = Tokenizer(models.BPE.from_file(str(source / 'vocab.json'), str(source / 'merges.txt')))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe, bos_token='<|endoftext|>', eos_token='<|endoftext|>',
                                        unk_token='<|endoftext|>', pad_token='<|endoftext|>')
    if len(tokenizer) != vocab:
        raise ValueError(f'Tokenizer vocabulary {len(tokenizer)} does not match the embedding table {vocab}.')
    tokenizer.chat_template = TEMPLATE
    tokenizer.save_pretrained(str(output))
    (output / 'conversion.json').write_text(json.dumps({'source': str(source), 'layers': len(layers), 'hidden': hidden, 'heads': heads,
                                                        'vocab': vocab, 'swap_layernorms': swap_layernorms}, indent=2))
    return model, tokenizer


@torch.no_grad()
def perplexity(model, tokenizer, text, tokens=2048, window=512):
    ids = tokenizer(text, add_special_tokens=False)['input_ids'][:tokens]
    total, count = 0.0, 0
    for start in range(0, len(ids) - 1, window):
        chunk = torch.tensor([ids[start:start + window + 1]])
        if chunk.shape[1] < 2:
            break
        logits = model(chunk[:, :-1]).logits[0]
        total += float(torch.nn.functional.cross_entropy(logits, chunk[0, 1:], reduction='sum')); count += chunk.shape[1] - 1
    return math.exp(total / count), total / count


@torch.no_grad()
def continuation(model, tokenizer, prompt, tokens=12):
    ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    out = model.generate(ids, max_new_tokens=tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-text', type=Path, help='text file for the perplexity sanity check')
    p.add_argument('--swap-layernorms', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    model, tokenizer = convert(a.source, a.output, a.swap_layernorms)
    report = {}
    if a.check_text:
        ppl, nll = perplexity(model, tokenizer, a.check_text.read_text(errors='replace'))
        report['perplexity'] = ppl; report['nll'] = nll
    for prompt in ['The capital of France is', 'Alice was beginning to get very tired of sitting by her sister on the bank, and of having']:
        report[prompt] = continuation(model, tokenizer, prompt)
    print(json.dumps(report, indent=2))
    (a.output / 'sanity.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
