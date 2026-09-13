"""FLM's chat model with a looped connectome block and a test-time-training readout.

LoopedFLM mirrors the FLM class of nftechie/flm (MIT): a frozen causal language model whose
next-token scores receive a bounded correction from a small adapter that reads the reservoir's
features. Two changes:

  * the reservoir is LoopedReservoir (FLM's dynamics at K = 1, a looped block otherwise);
  * the adapter can carry per-conversation fast weights, learned during prefill from the next
    tokens the prompt already reveals (dynamic-evaluation-style test-time training), bounded in
    norm, and discarded on reset. The slow adapter and the backbone never change.

Any local Hugging Face causal LM directory with a chat template works as the backbone.
"""
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file

from .graph import Graph
from .kernel import sha256 as sha256_file
from .reservoir import LoopedReservoir

DEFAULT_INTERFACE = {'seed': 7301, 'dimensions': 128, 'input_gain': 0.4, 'recurrence_gain': 0.6,
                     'adapter_scale': 0.03, 'max_logit_rms': 0.25,
                     'feedback': 0.8, 'step_size': 0.7, 'max_iterations': 8, 'tolerance': 1e-3}
DEFAULT_SAMPLING = {'temperature': 0.4, 'top_k': 50, 'top_p': 0.9, 'repetition_penalty': 1.05}
DEFAULT_SYSTEM = ('You are FLM-Loop, an experimental assistant: a pretrained language model with a trained '
                  'readout of a fly connectome run as a looped block. Answer directly and concisely. You are '
                  'software, not a biological fly. Say when you do not know.')
FAST_WEIGHTS = {'learning_rate': 0.05, 'decay': 0.0, 'max_norm': 0.5, 'chunk': 32}


def backbone_signature(folder):
    """Hash of every weight file in the backbone directory (sharded or not)."""
    folder = Path(folder)
    h = hashlib.sha256()
    for path in sorted(folder.glob('*.safetensors')) + sorted(folder.glob('*.bin')):
        h.update(path.name.encode()); h.update(sha256_file(path).encode())
    return h.hexdigest()


def graph_signature(graph):
    """Hash of the graph's identity (FLM's manifest arrays when present, else its CSR arrays)."""
    if graph.manifest.get('arrays'):
        payload = json.dumps({'release': graph.manifest.get('release'), 'arrays': graph.manifest['arrays'],
                              'control': graph.manifest.get('control')}, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()
    h = hashlib.sha256()
    for array in (graph.matrix.indptr, graph.matrix.indices, graph.matrix.data):
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def block_signature(reservoir):
    h = hashlib.sha256()
    for array in (reservoir.gain, reservoir.efficacy, np.zeros(0) if reservoir.output_weight is None else reservoir.output_weight):
        h.update(np.ascontiguousarray(array, np.float32).tobytes())
    return h.hexdigest()


def bound_logit_delta(delta, maximum):
    """Limit per-position RMS of the logit correction; zero input remains exactly zero (FLM)."""
    if maximum is None:
        return delta
    rms = (delta.float().square().mean(-1, keepdim=True) + 1e-12).sqrt()
    return delta * (maximum / rms).clamp(max=1)


def sample_token(logits, generator, temperature=0.7, top_k=20, top_p=0.8, previous_tokens=(), repetition_penalty=1.0):
    if previous_tokens and repetition_penalty != 1.0:
        logits = logits.clone()
        seen = torch.tensor(sorted(set(previous_tokens)), device=logits.device)
        values = logits[seen]
        logits[seen] = torch.where(values < 0, values * repetition_penalty, values / repetition_penalty)
    if temperature <= 0:
        return int(logits.argmax())
    values, choices = torch.topk(logits.float() / temperature, min(top_k, logits.numel()))
    probabilities = torch.softmax(values, -1)
    remove = probabilities.cumsum(-1) - probabilities > top_p
    probabilities = probabilities.masked_fill(remove, 0)
    selected = int(torch.multinomial(probabilities.cpu(), 1, generator=generator))
    return int(choices[selected])


class FlyAdapter(nn.Module):
    """FLM's bias-free readout: features -> width -> hidden, zero-initialized output, bounded by tanh."""

    def __init__(self, hidden_size, features=128, width=128, scale=0.03):
        super().__init__()
        self.scale = scale
        self.input = nn.Linear(features, width, bias=False)
        self.output = nn.Linear(width, hidden_size, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, features, fast=None):
        hidden = torch.nn.functional.gelu(self.input(features))
        weight = self.output.weight if fast is None else self.output.weight + fast
        return self.scale * torch.tanh(torch.nn.functional.linear(hidden, weight))


class FastWeights:
    """Per-conversation delta on the adapter's output projection.

    learn() takes gradient steps on the next-token loss of prompt positions whose targets are
    already known, in chunks (large-chunk TTT); the delta is norm-bounded and decays; reset()
    discards it. The slow adapter's weights are never touched.
    """

    def __init__(self, adapter, learning_rate=0.05, decay=0.0, max_norm=0.5, chunk=32):
        self.adapter = adapter
        self.learning_rate, self.decay, self.max_norm, self.chunk = float(learning_rate), float(decay), float(max_norm), int(chunk)
        self.delta = torch.zeros_like(adapter.output.weight)
        self.steps, self.losses = 0, []

    def reset(self):
        self.delta.zero_()
        self.steps, self.losses = 0, []

    def norm(self):
        return float(self.delta.norm())

    def learn(self, score, hidden, features, targets):
        """score(hidden, features, fast) -> logits. hidden (T,H), features (T,D), targets (T,)."""
        for start in range(0, len(targets), self.chunk):
            sl = slice(start, start + self.chunk)
            with torch.enable_grad():
                delta = self.delta.clone().requires_grad_(True)
                logits = score(hidden[sl], features[sl], delta)
                loss = torch.nn.functional.cross_entropy(logits.float(), targets[sl])
                grad, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                self.delta.mul_(1.0 - self.decay).sub_(self.learning_rate * grad)
                norm = self.delta.norm()
                if norm > self.max_norm:
                    self.delta.mul_(self.max_norm / norm)
            self.steps += 1
            self.losses.append(float(loss))
        return self.losses


class EmbeddingView:
    """Gather only requested rows; never duplicate the vocabulary matrix (FLM)."""

    def __init__(self, weight):
        self.weight = weight.detach()

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            index = torch.as_tensor(index, device=self.weight.device)
        return self.weight[index].float().cpu().numpy()


def device_name(requested='auto'):
    if requested != 'auto':
        return requested
    return 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


class LoopedFLM:
    def __init__(self, backbone, graph, checkpoint=None, device='auto', interface=None, system_text=None,
                 sampling=None, max_context=1536, efficacy=None, gain=None, output_weight=None, kernel=None,
                 dtype=None, fast_weights=None, threads=None):
        torch.set_num_threads(max(1, min(8, int(threads or os.environ.get('FLM_CPU_THREADS', '2')))))
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device_name(device)
        self.backbone_dir = Path(backbone)
        self.graph = graph if isinstance(graph, Graph) else Graph.from_flm_cache(graph)
        self.kernel = kernel
        manifest = None
        if checkpoint is not None:
            checkpoint = Path(checkpoint)
            manifest = json.loads(checkpoint.with_name('run.json').read_text())
        self.interface = {**DEFAULT_INTERFACE, **(manifest['interface'] if manifest else {}), **(interface or {})}
        self.system = {'role': 'system', 'content': (manifest['system_text'] if manifest else None) or system_text or DEFAULT_SYSTEM}
        self.sampling = {**DEFAULT_SAMPLING, **((manifest['sampling'] if manifest else None) or sampling or {})}
        self.max_context = int((manifest['context_tokens'] if manifest else None) or max_context)
        self.fast_config = {**FAST_WEIGHTS, **(fast_weights or {})}
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.backbone_dir), local_files_only=True, trust_remote_code=False)
        if dtype is None:
            dtype = torch.float16 if self.device in ('mps', 'cuda') else torch.float32
        self.base = AutoModelForCausalLM.from_pretrained(str(self.backbone_dir), local_files_only=True, trust_remote_code=False,
                                                         dtype=dtype, attn_implementation='eager').to(self.device).eval()
        self.base.requires_grad_(False)
        self.hidden_size = self.base.config.hidden_size
        self.embeddings = EmbeddingView(self.base.get_input_embeddings().weight)
        self.block = {'gain': gain, 'efficacy': efficacy, 'output_weight': output_weight}
        self.adapter = FlyAdapter(self.hidden_size, features=self.interface['dimensions'], scale=self.interface['adapter_scale']).to(self.device).eval()
        self.fast = FastWeights(self.adapter, **self.fast_config)
        self.trained = False
        self.run_manifest = None
        self.ttt_positions = np.zeros(0, np.int64)
        if manifest is not None:
            if manifest['graph_signature'] != graph_signature(self.graph):
                raise ValueError('Checkpoint belongs to a different graph.')
            if manifest['backbone_signature'] != backbone_signature(self.backbone_dir):
                raise ValueError('Checkpoint belongs to a different language backbone.')
            if manifest['interface'] != self.interface:
                raise ValueError('Checkpoint interface configuration differs.')
            if manifest['block_signature'] != block_signature(self.reservoir()):
                raise ValueError('Checkpoint belongs to different gains/efficacies/pooling weights.')
            if manifest['adapter_sha256'] != sha256_file(checkpoint):
                raise ValueError('Checkpoint checksum mismatch.')
            self.adapter.load_state_dict(load_file(str(checkpoint)))
            self.run_manifest = manifest
            self.trained = True
        self.prefix_cache = None
        if self.trained:
            prefix = self.prompt_ids([], add_generation_prompt=False, max_context=10 ** 9)
            reservoir = self.reservoir()
            for token in prefix:
                reservoir.step(self.embeddings[token])
            self.prefix_cache = (prefix, reservoir.state.copy())

    # ----- pieces -------------------------------------------------------------------------
    def reservoir(self, lanes=1):
        i = self.interface
        return LoopedReservoir(self.graph, self.hidden_size, dimensions=i['dimensions'], seed=i['seed'], lanes=lanes,
                               memory=i['recurrence_gain'], drive=i['input_gain'], feedback=i['feedback'],
                               step_size=i['step_size'], max_iterations=i['max_iterations'], tolerance=i['tolerance'],
                               gain=self.block['gain'], efficacy=self.block['efficacy'],
                               output_weight=self.block['output_weight'], kernel=self.kernel)

    def render(self, messages, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template([self.system] + list(messages), tokenize=False,
                                                  add_generation_prompt=add_generation_prompt)

    def encode(self, text):
        return list(self.tokenizer(text, add_special_tokens=False)['input_ids'])

    def prompt_ids(self, messages, max_context=None, add_generation_prompt=True):
        ids = self.encode(self.render(messages, add_generation_prompt))
        limit = max_context or self.max_context
        if len(ids) > limit:
            raise ValueError(f'Conversation exceeds the {limit}-token context. Start a new chat.')
        return ids

    def scores(self, hidden, features, mode='intact', adapter=None, fast=None):
        """Bounded additive model: logits = lm_head(hidden) + bound(lm_head(adapter(features)))."""
        adapter = adapter or self.adapter
        delta = adapter(features.float(), fast) if mode != 'base' else torch.zeros_like(hidden, dtype=torch.float32)
        projected = self.base.lm_head(torch.cat((hidden, delta.to(hidden.dtype)), dim=0)).float()
        baseline, change = projected.split(hidden.shape[0], dim=0)
        change = bound_logit_delta(change, self.interface['max_logit_rms'])
        return baseline + change, baseline, delta

    # ----- generation ---------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, messages, mode='intact', max_tokens=160, temperature=None, seed=42, ttt=False,
                 cancelled=None, telemetry=True):
        if mode not in ('intact', 'no_edges', 'shuffled', 'base'):
            raise ValueError('Unknown comparison mode.')
        if mode != 'base' and not self.trained:
            raise ValueError('Train the graph adapter before chatting in FLM mode.')
        ids = self.prompt_ids(messages)
        temperature = self.sampling['temperature'] if temperature is None else temperature
        reservoir = self.reservoir()
        started = time.perf_counter()
        features = np.zeros(self.interface['dimensions'], np.float32)
        start = 0
        if mode == 'intact' and self.prefix_cache:
            prefix, state = self.prefix_cache
            if ids[:len(prefix)] == prefix:
                start = len(prefix)
                reservoir.state = state.copy()
                reservoir.tokens = start
        prompt_features = []
        if mode != 'base':
            for i, token in enumerate(ids[start:], start):
                if cancelled is not None and cancelled.is_set():
                    return
                features = reservoir.step(self.embeddings[token], mode)
                prompt_features.append(features)
                if i % 24 == 0:
                    yield {'type': 'progress', 'processed': i + 1, 'total': len(ids)}
        tokens = torch.tensor([ids], device=self.device)
        result = self.base.model(tokens, use_cache=True)
        hidden_all, past = result.last_hidden_state[0], result.past_key_values
        hidden = hidden_all[-1:]
        self.fast.reset()
        fast = None
        if ttt and mode != 'base' and len(prompt_features) > 1:
            # Position i (which has consumed ids[:i+1]) predicts ids[i+1]; the prompt reveals those targets.
            count = len(prompt_features) - 1
            self.fast.learn(lambda h, f, d: self.scores(h, f, mode, fast=d)[0],
                            hidden_all[start:start + count], torch.as_tensor(np.stack(prompt_features[:count]), device=self.device),
                            torch.as_tensor(ids[start + 1:start + 1 + count], device=self.device))
            fast = self.fast.delta
        generated = []
        rng = torch.Generator(device='cpu').manual_seed(seed)
        for index in range(max_tokens):
            if cancelled is not None and cancelled.is_set():
                return
            logits, baseline_logits, delta = self.scores(hidden, torch.as_tensor(features, device=self.device).unsqueeze(0), mode, fast=fast)
            logits, baseline_logits = logits[0], baseline_logits[0]
            token = sample_token(logits, rng, temperature, self.sampling['top_k'], self.sampling['top_p'],
                                 ids + generated, self.sampling['repetition_penalty'])
            if token == self.tokenizer.eos_token_id:
                break
            generated.append(token)
            event = {'type': 'token', 'text': self.tokenizer.decode(generated, skip_special_tokens=True),
                     'tokens': len(generated), 'elapsed': round(time.perf_counter() - started, 3)}
            if telemetry:
                graph = reservoir.telemetry()
                event.update(logit_delta_rms=float(torch.sqrt(torch.mean((logits - baseline_logits) ** 2))),
                             adapter_rms=float(torch.sqrt(torch.mean(delta ** 2))),
                             graph={k: graph[k] for k in ('updates', 'state_rms', 'iterations', 'residual', 'saturation')},
                             fast_weight_norm=self.fast.norm(), ttt_steps=self.fast.steps,
                             ttt_loss=self.fast.losses[-1] if self.fast.losses else None)
            yield event
            if index + 1 >= max_tokens:
                break
            if mode != 'base':
                features = reservoir.step(self.embeddings[token], mode)
            result = self.base.model(torch.tensor([[token]], device=self.device), past_key_values=past, use_cache=True)
            hidden, past = result.last_hidden_state[:, -1], result.past_key_values
        yield {'type': 'done', 'tokens': len(generated), 'elapsed': round(time.perf_counter() - started, 3),
               'stop_reason': 'eos' if len(generated) < max_tokens else 'token_limit',
               'text': self.tokenizer.decode(generated, skip_special_tokens=True),
               'ttt': {'steps': self.fast.steps, 'losses': self.fast.losses, 'norm': self.fast.norm()}}

    # ----- offline evaluation helpers -----------------------------------------------------
    @torch.no_grad()
    def prompt_nll(self, messages, mode='intact', ttt=False):
        """Mean next-token NLL over the assistant turns of `messages`, optionally after test-time
        training the fast weights on the prompt positions that precede the first assistant token —
        never on positions whose hidden state has already seen a scored answer. Used for evaluation."""
        ids = self.prompt_ids(messages, add_generation_prompt=False, max_context=10 ** 9)
        reservoir = self.reservoir()
        feats = np.stack([reservoir.step(self.embeddings[token], mode if mode != 'base' else 'intact') for token in ids])
        hidden_all = self.base.model(torch.tensor([ids], device=self.device), use_cache=False).last_hidden_state[0]
        mask = self.assistant_mask(messages, ids)
        self.fast.reset()
        fast = None
        self.ttt_positions = np.zeros(0, np.int64)
        if ttt and mode != 'base':
            first = int(np.argmax(mask)) if mask.any() else len(ids)
            # Position i has consumed ids[:i+1] and predicts ids[i+1]: causal iff i + 1 < first.
            positions = np.arange(max(0, first - 1))
            self.ttt_positions = positions
            if len(positions):
                f = torch.as_tensor(feats[positions], device=self.device)
                self.fast.learn(lambda h, ff, d: self.scores(h, ff, mode, fast=d)[0], hidden_all[positions], f,
                                torch.as_tensor(np.asarray(ids)[positions + 1], device=self.device))
                fast = self.fast.delta
        positions = np.where(mask[1:])[0]
        logits, _, _ = self.scores(hidden_all[positions], torch.as_tensor(feats[positions], device=self.device), mode, fast=fast)
        targets = torch.as_tensor(np.asarray(ids)[positions + 1], device=self.device)
        return float(torch.nn.functional.cross_entropy(logits, targets))

    def assistant_mask(self, messages, ids):
        """Boolean mask over ids marking assistant-content tokens, from prefix-stable renderings (FLM's rule)."""
        mask = np.zeros(len(ids), bool)
        full = [self.system] + list(messages)
        for i, message in enumerate(full):
            if message['role'] != 'assistant':
                continue
            prefix = self.encode(self.tokenizer.apply_chat_template(full[:i], tokenize=False, add_generation_prompt=True))
            end = self.encode(self.tokenizer.apply_chat_template(full[:i + 1], tokenize=False, add_generation_prompt=False))
            if ids[:len(prefix)] != prefix or ids[:len(end)] != end:
                raise ValueError('Chat template is not prefix-stable; refusing an incorrect mask.')
            mask[len(prefix):len(end)] = True
        return mask
