# flm-loop — FLM's fly-connectome reservoir as a looped block that learns at test time

An implementation study on top of [nftechie/flm](https://github.com/nftechie/flm) (Fly Language
Model, MIT): a frozen 1.2B chat model whose next-token scores get a bounded correction from a
278K-parameter readout of the MaleCNS v1.0 fly connectome (166,700 neurons, 25.6M directed edges).
FLM's own controls show its wiring does not help. This package keeps FLM's contract — the exact
graph, the seeded interfaces, the bias-free adapter, the `intact / no_edges / shuffled / base`
controls — and changes what the graph *does*:

| | FLM | flm-loop |
|---|---|---|
| propagation per token | one step, `x = tanh(W(0.6x + 0.4u))` | K damped iterations of the same block with the input re-injected every pass, early exit on the residual (looped-transformer / rate-model dynamics); K = 1 reproduces FLM bit for bit |
| synapses | unsigned, incoming-normalized counts | the same, times per-neuron **efficacy signs from the connectome's own neurotransmitter predictions** (GABA / glutamate / histamine = inhibitory) and a per-neuron gain calibrated to a target spectral radius |
| what learns | the readout, offline | the readout offline **plus** per-conversation fast weights learned during prefill from the prompt's own next tokens (test-time training), a local intrinsic-plasticity gain rule, and optionally the gains / efficacies / pooling weights trained *through the loop* (random iteration counts, deep supervision, truncated loop backprop) |
| controls | no_edges, shuffled (relabeling), direct-input adapter | those, plus a degree-sequence-preserving **rewired** graph and random signs with the same inhibitory share |

Everything runs locally on CPU (NumPy + an optional OpenMP C kernel for the 25.6M-edge product),
Apple Silicon (MPS) or CUDA. The research note with the analysis and the measurements is
[../flm-loop.md](../flm-loop.md).

## Layout

```
flmloop/reservoir.py   LoopedReservoir — FLM's dynamics generalized to K iterations (numpy)
flmloop/graph.py       Graph (FLM's prepared arrays, toy/random graphs), rewired control, NT sign prior
flmloop/kernel.py      optional C kernel (native/graph_lanes.c): CSR x lanes, row-parallel, no FMA
flmloop/plasticity.py  IntrinsicPlasticity (gain homeostasis), OnlineSoftmaxReadout (prequential TTT readout), RLSReadout
flmloop/probes.py      memory capacity, logistic probe, effective rank, spectral radius, gain calibration
flmloop/torchloop.py   TorchLoopedReservoir — differentiable twin (autograd through W and W^T)
flmloop/llm.py         LoopedFLM — FLM's chat model on any local HF backbone + FastWeights TTT
scripts/bench_reservoir.py   backbone-free benchmark on the real graph (FLM vs looped vs controls)
scripts/train_graph.py       train gains / efficacies / pooling through the loop -> block.npz
scripts/train_adapter.py     FLM's adapter recipe on looped features + TTT evaluation -> runs/<name>/
scripts/chat.py              chat with a trained run, --ttt for fast weights
scripts/make_tiny_backbone.py fully offline tiny chat backbone for tests
tests/                       36 unit tests (toy graphs; tiny backbone; end-to-end scripts)
data/conversations.json      FLM's 32 synthetic style conversations (MIT), vendored
results/                     benchmark outputs from the MaleCNS graph
```

## Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pyarrow safetensors torch transformers tokenizers
python -m unittest discover -s tests -p 'test_*.py'        # ~30 s, no downloads
```

Real graph: run FLM's `scripts/download.py` (backbone + corpus, Hugging Face) and
`scripts/prepare_graph.py` (MaleCNS from the public bucket, SHA-256 pinned) in a checkout of
nftechie/flm; `flm/cache/malecns_v1` is the graph folder used below. The neurotransmitter table
is one more public file from the same bucket:

```sh
curl -O https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather
```

## Use

```python
from flmloop import Graph, LoopedReservoir, load_kernel, neurotransmitter_signs
from flmloop.probes import calibrate_gain

graph = Graph.from_flm_cache('path/to/flm/cache/malecns_v1')
kernel = load_kernel()                                   # None -> SciPy path, same numbers
signs, counts = neurotransmitter_signs(graph.ids, 'body-neurotransmitters-male-cns-v1.0.feather')
gain, _ = calibrate_gain(graph, signs, target_radius=0.95, kernel=kernel)

flm = LoopedReservoir(graph, embedding_dim=2048, kernel=kernel)                 # == FLM
loop = LoopedReservoir(graph, 2048, feedback=0.8, step_size=0.7, max_iterations=8, tolerance=1e-3,
                       gain=[gain] * graph.n, efficacy=signs, kernel=kernel)   # looped, signed
features = loop.step(token_embedding)        # (128,) per token; loop.telemetry() has iterations/residual
```

Benchmark without a backbone (bytes of local text as tokens; ~75 min on 4 CPU cores):

```sh
python scripts/bench_reservoir.py --graph .../malecns_v1 --nt body-neurotransmitters-male-cns-v1.0.feather
```

Train the block through the loop, then the adapter on a backbone, then chat with TTT:

```sh
python scripts/train_graph.py --graph .../malecns_v1 --nt ....feather --steps 200 --random-k --supervise 4 --output runs/block/block.npz
python scripts/train_adapter.py --backbone .../flm/cache/lfm25-1.2b --graph .../malecns_v1 --block runs/block/block.npz --output runs/looped-v1
python scripts/chat.py --run runs/looped-v1 --ttt --telemetry
```

`train_adapter.py` reports base / fly adapter / direct-input control / relabeled wiring / no_edges
exactly like FLM, plus `fly_adapter_ttt`: the same adapter with fast weights learned on each
held-out conversation's non-answer prefix and scored on the answer.

## What is and is not claimed

The code reproduces FLM at K = 1 (tested bit for bit against FLM's reservoir on the same
interfaces). The looped and signed variants are measured against degree-matched rewired graphs
and random signs, on the real connectome, with a linear readout and no language model — see the
research note for the numbers and their error bars. Nothing here is biological language, and a
better reservoir is not evidence that a fly understands text.

## License

MIT for the code here. FLM's code and its synthetic examples are MIT (Copyright (c) 2026 Alex
Wormuth). MaleCNS v1.0 data: CC BY 4.0 (FlyEM / HHMI Janelia, Cambridge, MRC LMB, Google Research).
LFM2.5-1.2B-Instruct carries the LFM Open License; it is never redistributed here.
