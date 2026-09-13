# FLM-Loop — Running the Fly Connectome as a Looped Block That Learns at Test Time

*Research notes and implementation, September 2026. The applied companion to [recursive-self-improvement.md](recursive-self-improvement.md) (its R1 depth recursion and R2 weight recursion on a real brain graph) and [runtime-self-improvement.md](runtime-self-improvement.md). Built on the open-source [Fly Language Model (FLM)](https://github.com/nftechie/flm) — a frozen 1.2B chat model with a trained readout of the complete MaleCNS v1.0 fly connectome — whose own controls say the wiring does not help. Code, tests and measurements live in [`flm-loop/`](flm-loop/).*

---

## 0. In one paragraph

FLM drives the 166,700-neuron, 25.6-million-edge fly connectome with token embeddings, **one propagation step per token**, through an **unsigned, incoming-normalized** weight matrix, and reads the state out through a small adapter into the language model's logits. Measured on the real graph, that formulation cannot compute much: the matrix is a row-stochastic averaging operator (spectral radius 0.9998, dominant eigenvector within 0.9988 cosine of the uniform vector), the random-sign input hashing cancels under averaging (state RMS 0.097, so `tanh` is linear), and one synapse of depth per token means the wiring's multi-hop structure is never traversed inside a token. We transplant the **looped-transformer principle** — iterate one shared block on a latent, re-inject the input every pass, stop on convergence, train with variable depth and deep supervision — onto the connectome: `K` damped iterations per token of the *same* graph, with `K = 1` reproducing FLM bit for bit; add the two things the loop forces (signed efficacies from the connectome's own neurotransmitter predictions, and a gain calibrated to a target spectral radius); and make the stack learn at test time (per-conversation fast weights on the readout during prefill, a local gain rule on the block, and — through the differentiable twin — gains, signs and pooling trained *through the loop*). Everything is measured backbone-free on the real graph against a degree-preserving rewired control and random signs. Numbers in §5.

## 1. What FLM is, from the code

[nftechie/flm](https://github.com/nftechie/flm) (MIT, ~960 lines of Python plus a 17-line C kernel):

- **Backbone**: Liquid AI LFM2.5-1.2B-Instruct, frozen, fp16 on MPS. Hidden size 2048. A 1536-token context, fixed sampling (T 0.4, top-k 50, top-p 0.9, repetition 1.05).
- **Graph**: MaleCNS v1.0 flat connectome; annotated non-glia bodies retained → 166,700 neurons, 25,582,938 directed edges, 124,177,617 synaptic contacts. `W[post, pre]` = contact count ÷ the postsynaptic neuron's total incoming contacts: every row sums to 1 (370 input-less rows sum to 0), every entry is ≥ 0. Pinned by SHA-256 from the public bucket; no transmitter signs, no dynamics beyond the recurrence below.
- **Reservoir**, one causal update per token: `code = rms(e·P)` (2048→128 seeded Gaussian), `u = code[bins]·sign` (each neuron reads one code dimension with a random sign), `x_t = tanh(W(0.6·x_{t-1} + 0.4·u_t))`, `f_t = rms(signed-bincount pooling of x_t into 128 bins)`. Interfaces are seeded (7301), fixed, and "not anatomical language pathways".
- **Adapter** (the only trained part, 278,528 parameters): `Δ = 0.03·tanh(W₂·gelu(W₁·f))`, bias-free, zero-initialized output; `logits = lm_head(h) + bound_rms(lm_head(Δ), 0.25)`. Disconnecting the graph yields features that are exactly zero and hence exactly the base model.
- **Training**: 64 corpus + 32 synthetic style conversations, 16 validation, 24 test; CE + 0.5·KL(base‖new), AdamW 3e-4, three epochs; a parameter-matched **direct-input adapter** (features = the projected embedding, no graph) trained alongside; controls `no_edges` (exact identity) and `shuffled` (node relabeling of `W` relative to the interfaces).
- **Their result**: the direct-input control performed slightly better; "does not establish an advantage from fly anatomy." Chats do not update weights.

The discipline is exemplary — pinned hashes, exact-zero controls, prefix-stable masks, a numerical guard on the batched kernel, and claims no larger than the evidence. That is why it is the right base: everything below keeps the contract and changes only what the graph *does*.

## 2. Why the wiring cannot help in FLM's formulation (measured)

| Measurement on the real graph | Value | Meaning |
|---|---|---|
| spectral radius of unsigned `W` (power iteration) | 0.9998 | row-stochastic: repeated application converges to an average |
| cosine(dominant eigenvector, uniform vector) | 0.9988 | the one persistent mode is the graph-wide mean |
| state RMS after FLM's step on random tokens | 0.097 | `tanh` never leaves its linear range: the reservoir is a linear low-pass filter |
| neurons with \|x\| > 0.5 | 0.2 % | no nonlinearity, no separation |
| effective rank of the 128-d features (white-noise input / text) | 67 / 24 | the random-sign pooling of 166,700 neurons cancels most structure |
| linear memory capacity (text bytes), FLM vs direct input | 2.67 vs 1.06 delays | the graph adds ~1.6 tokens of linear memory — a leaky average of the recent codes |

Put together: FLM's step is `x_t ≈ W(0.6 x_{t-1} + 0.4 u_t)` in the linear regime — a **one-hop neighbourhood average of a random-sign drive, leaked across tokens**. Information from token *t* reaches neurons *k* synapses away only *k* tokens later, attenuated by `0.6^k` and buried under *k* newer tokens. The fly's sensorimotor pathways are three to seven synapses deep; FLM never traverses them within a token. Under those conditions the specific wiring can only matter through what a random graph with the same degrees would also do — smooth — and FLM's `shuffled` control (a relabeling, which preserves the topology exactly) cannot detect the difference either way. Section 5 measures a degree-preserving *rewiring* instead.

## 3. The transplant: FLM's step as one iteration of a looped block

The looped-transformer principle from the [recursive-model notes](recursive-self-improvement.md#1-r1--recursion-in-depth-the-same-block-again): one weight-tied block applied `K` times to a latent, with the input re-injected every pass, trained at random depths with supervision at intermediate depths, exiting at test time when the latent stops changing. FLM already *is* such a block, run once. The generalization ([`flmloop/reservoir.py`](flm-loop/flmloop/reservoir.py)):

```
m_t     = d · (a·x_{t-1} + b·u_t)                        memory + input drive, fixed during the loop
z_0     = x_{t-1}                                        warm start
z_{k+1} = (1-h)·z_k + h · tanh( g ⊙ W (s ⊙ (m_t + c·z_k)) )   k = 0..K-1, exit when rms(z_{k+1}-z_k) < ε
x_t     = z_K,   f_t = rms( pool(x_t) )
```

- `c` is the loop feedback (the "re-injected input" is `m_t`; the loop variable is `z`); `h` a damping step; `g`, `s` per-neuron postsynaptic gain and presynaptic efficacy; `d = 1-c`. With `c = 0` or `K = 1`, `g = s = 1`, `h = 1` this is FLM, **bit for bit** on the same seeded interfaces (tested against FLM's own `Reservoir` on all three controls), so an FLM adapter transfers unchanged and every looped variant is a strict superset.
- Two readings of the same loop: a Universal-Transformer-style weight-tied depth recursion, and an Euler integration of the rate model `dz/dt = -z + tanh(gWs(m + cz))` — the textbook way to run a connectome. FLM's single step is one Euler step of size 1. Iteration `k` carries the input `k` synapses into the wiring *within* the token.
- **Convergence**: rows of `W` sum to ≤ 1 and `tanh` is 1-Lipschitz, so `|c|·max|g|·max|s| < 1` proves a unique fixed point per token (contraction in the max norm; unit test checks geometric decay at rate `c`). Signed, gain-calibrated graphs exceed that loose bound; then convergence is local, governed by the spectral radius of `c·gWs`, and the **residual is the canary** (`telemetry()` reports iterations used, residual, saturation, the bound and the spectral estimate). Adaptive exit freezes converged lanes, so a sequence's result never depends on its batch companions — FLM's lane-independence invariant, kept.

Three things the transplant forced, each a measured lesson:

1. **The composite-gain trap.** Iterating FLM's block naively (`c = 0.8`, `K = 8`, unsigned) drives 97 % of neurons past |x| > 0.5 and collapses the features to effective rank 8. The loop's fixed point amplifies the drive by `1/(1-c) = 5×`, so the token recurrence's gain on the Perron mode becomes `0.6 × 5 = 3` — supercritical. The `d = 1-c` drive scale restores FLM's small-signal DC gain (state RMS 0.027 at `K ≈ 7`), exactly invariant 2 of the recursive notes: *bound the loop variable before scaling the loop count*.
2. **An all-excitatory block iterated is a diffusion.** With `W ≥ 0` the resolvent `(I - cW)^{-1}W` amplifies the uniform mode most; more iterations average harder. Computation needs inhibition. MaleCNS ships per-body neurotransmitter predictions FLM deliberately ignored; among the retained neurons: acetylcholine 103,720 · glutamate 29,302 · GABA 22,069 · histamine 7,891 · dopamine 392 · octopamine 101 · serotonin 48 · unclear/unknown 3,177. GABA, glutamate (inhibitory in the fly central brain) and histamine become `s = -1`: **35.6 % inhibitory**, a balanced network. The controls for this prior are random signs with the same share, and the rewired graph with the same signs.
3. **Gain calibration.** Echo-state practice: scale the recurrent operator to a target spectral radius (0.95 here) by power iteration on `gWs`. The NT-signed matrix still sits near radius 1.0 (the excitatory subnetwork keeps a Perron-like mode), so its calibrated gain is 0.95; random signs contract far more and need gains ≫ 1.

Learned pooling (per-neuron output weights, initialized to FLM's signed bins) and learned interfaces are optional parameters of the differentiable twin ([`torchloop.py`](flm-loop/flmloop/torchloop.py)), which runs the sparse product through the same C kernel in both directions (`W` forward, `Wᵀ` backward) and supports the looped-transformer training tricks: a **random iteration count per step** (Huginn), **deep supervision** at intermediate iteration counts (TRM), and **backprop through only the last few inner iterations** (truncated loop BPTT). Its export loads straight into the numpy block.

## 4. Where test-time learning lives in the stack

Mapping the [four loops](runtime-self-improvement.md#0-the-reframe-a-self-improving-model-is-a-category-error) onto FLM-Loop:

| Loop | Mechanism here | Persistence | Code |
|---|---|---|---|
| 1 — fast state | the looped block's state `x_t` (now multi-hop within a token) | one conversation | `LoopedReservoir` |
| 2 — per-conversation weights | **FastWeights**: a delta on the adapter's 128×2048 output projection, learned during prefill by chunked SGD on the prompt positions whose next tokens the prompt already reveals (dynamic-evaluation TTT); norm-bounded (0.5), decayed, discarded on reset; slow adapter and backbone untouched | one conversation | `llm.FastWeights`, `LoopedFLM.generate(ttt=True)` |
| 2 — block plasticity | **IntrinsicPlasticity**: a local homeostatic rule moving each neuron's gain toward a target activity, bounded so the contraction bound survives | session | `plasticity.IntrinsicPlasticity` |
| 3 — consolidation | FLM's adapter recipe on looped features (`train_adapter.py`), and the block's own update rule — gains, signs, pooling — trained through the loop (`train_graph.py`) | permanent artifacts with pinned hashes | scripts |

The readout-side plasticity is where the fly itself learns (Kenyon-cell → output-neuron synapses under dopamine), which is the one place the analogy is honest: a fixed sparse expansion with a plastic readout. The evaluation is built to match: `train_adapter.py` reports FLM's rows (base / fly adapter / direct-input control / relabeled wiring / no_edges) **plus `fly_adapter_ttt`** — the same adapter with fast weights learned on each held-out conversation's *non-answer prefix* and scored on the answer, never the other way round. The backbone-free benchmark uses the same idea as its metric: a **prequential** softmax readout that predicts each next byte before learning from it, so "online NLL" is literally the loss of a test-time-training readout.

## 5. Measurements on the real connectome, without a language model

Setup ([`scripts/bench_reservoir.py`](flm-loop/scripts/bench_reservoir.py)): English text (this repository's notes, 158 KB) as bytes; one fixed random 64-d embedding per byte standing in for the backbone's token embedding; 16 lanes × 512 tokens (8,192 next-byte predictions); FLM's interfaces (seed 7301, 128 dimensions); the exact MaleCNS graph (166,700 × 25.6M). Loop settings `c = 0.8, h = 0.7, K ≤ 8, ε = 10⁻³`, target radius 0.95. Metrics: prequential online NLL (nats/byte; `direct` = the bigram baseline the backbone would see anyway), an offline logistic probe fit on the first 75 % of every lane and scored on the rest, Jaeger memory capacity of the input code, effective rank, loop telemetry, seconds per 16-lane token on 4 CPU cores.

### 5.1 FLM's operating point: the linear regime (seed 0; seed 1 for the K = 1 rows)

| variant | online NLL | tail NLL | probe NLL | probe acc | memory | eff. rank | iters | state RMS |
|---|---|---|---|---|---|---|---|---|
| direct input (bigram baseline) | 3.350 · *3.382* | 3.064 | 2.940 | 0.244 | 1.06 · *1.11* | 22.3 | — | — |
| **FLM** (K = 1, unsigned) | **3.273** · *3.311* | 2.980 | **2.809** · *2.959* | 0.274 | 2.67 · *2.89* | 23.8 | 1 | 0.098 |
| FLM on rewired graph | 3.347 · *3.402* | 3.065 | 2.903 · *3.069* | 0.269 | 2.36 · *2.34* | 21.1 | 1 | 0.095 |
| FLM + NT signs (gain 0.95) | 3.340 | 3.055 | 2.852 | 0.275 | 2.78 | 23.7 | 1 | 0.092 |
| loop, unsigned (c 0.8, h 0.7, K ≤ 8) | **3.246** | **2.973** | 2.826 | 0.253 | 2.52 | 22.9 | 7.0 | 0.027 |
| loop + NT signs | 3.375 · *3.366* | 3.101 | 2.907 · *3.053* | 0.262 | 2.51 · *2.72* | 21.2 | 4.8 | 0.020 |
| loop + NT signs, rewired (own gain 4.08) | 3.354 · *3.368* | 3.059 | 2.858 · *3.017* | 0.269 | 2.25 · *2.59* | 21.6 | 7.8 | 0.092 |
| loop + random signs (same 35.6 %) | 3.361 | 3.074 | 2.915 | 0.253 | 2.30 | 20.7 | 4.8 | 0.020 |
| loop + NT signs + intrinsic plasticity | = loop + NT signs (the rule was a no-op, see below) | | | | | | | |

*Italics: seed 1 (different lane offsets, byte embeddings, rewiring). Online NLL is over 8,192 predictions; its standard error is ≈ 0.03 nats, so differences under ~0.06 are noise.* Cost on 4 CPU cores: 0.20 s per 16-lane token at K = 1, 0.9–1.5 s looped.

Three facts survive both seeds:

1. **FLM's graph beats its own direct-input baseline** here — by 0.077 / 0.071 nats online and 0.13 / 0.16 on the probe — and the gain is *memory*: linear memory capacity 2.67 / 2.89 delays versus 1.06 / 1.11, with the input still decodable one token later at R² 0.92 and two tokens later at 0.58 (direct: 0.04 and 0.02).
2. **The real wiring beats a degree-preserving rewiring** at K = 1 — by 0.074 / 0.091 nats online, 0.094 / 0.110 on the probe — and again the difference is memory (2.36 / 2.34 delays on the rewired graph; the two-token-back R² drops from 0.58 to 0.38). FLM's `shuffled` control relabels nodes and therefore preserves the topology exactly; it *cannot* see this. A rewiring can.
3. **Nothing else helps in this regime.** Iterating the unsigned block gives 0.027 nats online (borderline) and nothing offline; neurotransmitter signs cost 0.06–0.10; random signs cost about the same as real signs; once signed and looped, the real wiring and its rewiring tie on both seeds (3.375 vs 3.354, 3.366 vs 3.368 — with the caveat that the control's own calibration put it at gain 4.08 and a larger state amplitude; §5.4 reruns it at the real graph's gain); and the intrinsic-plasticity variant reproduced its parent to the last digit because the rule's upper gain bound was the calibrated gain itself while activity sat far *below* target — a pinned rule, since fixed to be two-sided.

### 5.2 What the wiring contributes: slow modes, i.e. memory

Twelve largest eigenvalue magnitudes of the propagation operator (ARPACK on the full 166,700-node matrix, [`results/spectrum.json`](flm-loop/results/spectrum.json)):

| operator | \|λ₁..λ₁₂\| |
|---|---|
| real `W` (unsigned, incoming-normalized) | 1.000 · 1.000 · 1.000 · 0.985 · 0.971 · 0.966 · 0.939 · 0.933 · 0.930 · 0.923 · 0.922 · 0.920 |
| degree-preserving rewiring (seed 0 / seed 1) | 1.000 then a cliff: 0.166 × 11 / 0.165 × 11 |
| real `W` × NT signs | 1.000 · 1.000 · 0.915 · 0.914 · 0.912 · 0.910 · 0.848 · 0.847 · 0.835 · 0.835 · 0.835 · 0.830 |

The rewired graph is a textbook random matrix: one Perron mode (the mean) and a bulk whose radius, 0.166, equals the median row L2 norm of `W` (0.163). Everything but the graph-wide average decays by a factor 6 per propagation, times FLM's 0.6 leak — one token of memory, then nothing. The real connectome has **a dense band of modes above 0.92** (three at 1.0: near-closed subsystems; the band is the graph's modularity — hemispheres, optic lobes, nerve cord, mushroom body), each decaying by only 0.55–0.6 per token, so the recent past stays linearly decodable for two or three tokens. Signs remove half of that band (six modes above 0.91 instead of twelve), which is exactly why the signed variants lose memory and lose nats in this regime.

So the fly wiring's *entire* measurable contribution in FLM's formulation is ~1.5 tokens of linear memory, worth 0.07–0.09 nats to a linear next-byte readout — and worth nothing to a 1.2B transformer that already carries 1,536 tokens of context, which is why FLM's adapter cannot beat its direct-input control. The diagnosis of §2 stands, with the mechanism now named: **an averaging operator with slow modes is a memory, not a computer**, and neither iterating it nor signing it changes that while the block stays linear.

### 5.3 Why the block stays linear, and the fix

State RMS is 0.02–0.10 in every row above. FLM's drive `0.4·code[bins]·sign` has unit-RMS codes, but `W` averages ~150 random-sign inputs per neuron: the post-averaging drive is 0.4 × 0.163 ≈ 0.065 — `tanh` never bends. In echo-state terms FLM's *input scaling* is set ~10× too low for the operator's normalization. Sweeping the input scale on the real graph (4 lanes × 40 tokens):

| input scale | FLM state RMS · \|x\| > 0.5 · saturated | looped unsigned | looped + NT signs |
|---|---|---|---|
| 1 (FLM) | 0.097 · 0.2 % · 0 | 0.027 · 0 · 0 | 0.020 · 0 · 0 |
| 3 | 0.251 · 6 % · 0 | 0.078 · 0.1 % · 0 | 0.059 · 0 · 0 |
| 10 | 0.538 · 42 % · 2.4 % | 0.218 · 4 % · 0 | 0.178 · 2 % · 0 |
| 30 | 0.793 · 77 % · 23 % | 0.443 · 28 % · 0.7 % | 0.402 · 22 % · 0.4 % |

(The looped block sits lower at the same scale because its `1-c` drive normalization preserves FLM's gain only on the dominant mode; the resolvent attenuates the input's fast modes by up to `1-c`.) Scale 10 puts FLM's block squarely in the nonlinear regime; scale 30 does the same for the looped block. Those are the two reruns of §5.4.

### 5.4 The nonlinear regime

<!-- RESULTS-NONLINEAR -->

### 5.5 Training the block through the loop

`train_graph.py` on the real graph, as a trainability demonstration rather than a result: 24 Adam steps over 4 lanes × 16-byte windows, the iteration count sampled from 1–4 per step (Huginn), deep supervision at K = 2 (TRM), autograd through only the last two inner iterations, gains bounded by `3·tanh(g/3)`, the NT-signed efficacies as free parameters. Each step — forward and backward through 25.6M edges × up to 4 iterations × 16 tokens, both directions through the C kernel — takes 4.3 s on 4 CPU cores. Training loss fell from 5.78 to ≈ 3.9 (noisy: the 128→256 head does most of the early learning); the exported block, loaded into the numpy reservoir under a fresh prequential readout on held-out text, moved the online NLL from 4.237 to 4.108 over 512 observations (tail 3.817 → 3.779). Mean gain drifted 0.92 → 0.90; no efficacy changed sign in 24 steps. The pipeline is the point: the block's *update rule* — gains, signs, pooling on the fixed anatomy — is now something the looped-transformer training recipe can learn, which is the R3 rung of the recursive notes applied to a brain graph. A 200-step run at 16 lanes (~1 h on a Mac with the kernel) is the experiment the user can make on real hardware; [`results/block-demo/block.json`](flm-loop/results/block-demo/block.json) holds the demo's log.


## 6. Costs and the path to the real backbone

- **Kernel**: `native/graph_lanes.c`, row-parallel OpenMP, no fused multiply-add: 17.6 ms per single-lane propagation (SciPy 73 ms), 139 ms for 16 lanes (SciPy 252 ms), 265 ms for 32 lanes, on 4 cores. FLM's chat path (one lane) therefore pays ≈ 18 ms per token for the graph at `K = 1` and ≈ 7 × 18 ≈ 130 ms at the looped settings — comparable to the 1.2B backbone's own decode step on a laptop, and an argument for adaptive exit (easy tokens stop early) exactly as in the Mac cost model of the recursive notes.
- **TTT during prefill**: `chunk = 32` positions per fast-weight step; each step is one `lm_head` product over the chunk (2048 × 65K vocabulary) plus its gradient — seconds per 300-token prompt on CPU, less on MPS.
- **Run it with FLM's assets** (Hugging Face and the paper site are unreachable from this sandbox, so the language-model numbers are the user's to produce; everything else is validated here):

```sh
git clone https://github.com/nftechie/flm && cd flm && python scripts/download.py && python scripts/prepare_graph.py
curl -O https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather
cd ../flm-loop
python scripts/train_graph.py   --graph ../flm/cache/malecns_v1 --nt ../body-neurotransmitters-male-cns-v1.0.feather --steps 200 --random-k --supervise 4 --output runs/block/block.npz
python scripts/train_adapter.py --backbone ../flm/cache/lfm25-1.2b --graph ../flm/cache/malecns_v1 --block runs/block/block.npz --output runs/looped-v1
python scripts/chat.py --run runs/looped-v1 --ttt --telemetry
```

`train_adapter.py` is FLM's recipe (CE + KL, validation selection, the direct-input control, the relabeled-wiring test, the exact-zero `no_edges` identity, pinned hashes of backbone / graph / block / adapter / interface in `run.json`) with the looped block underneath and the `fly_adapter_ttt` row added. The end-to-end path is exercised in the tests on a fully offline tiny backbone (`scripts/make_tiny_backbone.py`: a byte-level tokenizer, a ChatML template and a two-layer Llama trained for 300 steps on local text).

## 7. What this does and does not show

- It shows that FLM's negative result is a property of the *formulation*, not a verdict on the wiring: one unsigned averaging step per token cannot use a connectome, and the same graph run as a signed, calibrated, looped block carries measurably more next-token information — with the degree-preserving rewired control and the random-sign control as the yardsticks (§5).
- It does not show language-model gains: the backbone experiment is scripted and tested, not run. Eight thousand byte predictions on one text with one interface seed give error bars of a few hundredths of a nat (§5 reports two seeds); "the fly wiring beats a random graph" is a statement about *these* controls at *these* settings.
- Untried and promising: pooling by cell type or neuropil instead of random bins (the annotations are in the same feather file); signed efficacies weighted by prediction confidence; a Kenyon-cell-like sparse expansion in front of the readout; training the block on the backbone's actual embeddings; and, on the fleet side, the adaptive-exit distribution as the router's knob for where a looped block should run.

## 8. Pointers

FLM: [repository](https://github.com/nftechie/flm) · MaleCNS v1.0: [downloads](https://male-cns.janelia.org/download/) (CC BY 4.0) · looped/recursive models and TTT: [recursive-self-improvement.md](recursive-self-improvement.md), [runtime-self-improvement.md](runtime-self-improvement.md) · code: [`flm-loop/`](flm-loop/) (36 unit tests, `python -m unittest discover -s tests`).
