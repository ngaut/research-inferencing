# FLM-Loop — Running the Fly Connectome as a Looped Block That Learns at Test Time

*Research notes and implementation, September 2026. The applied companion to [recursive-self-improvement.md](recursive-self-improvement.md) (its R1 depth recursion and R2 weight recursion on a real brain graph) and [runtime-self-improvement.md](runtime-self-improvement.md). Built on the open-source [Fly Language Model (FLM)](https://github.com/nftechie/flm) — a frozen 1.2B chat model with a trained readout of the complete MaleCNS v1.0 fly connectome — whose own controls say the wiring does not help. Code, tests and measurements live in [`flm-loop/`](flm-loop/).*

---

## 0. In one paragraph

FLM drives the 166,700-neuron, 25.6-million-edge fly connectome with token embeddings, **one propagation step per token**, through an **unsigned, incoming-normalized** weight matrix, and reads the state out through a small adapter into the language model's logits. Measured on the real graph, that formulation cannot compute much: the matrix is a row-stochastic averaging operator (spectral radius 0.9998, dominant eigenvector within 0.9988 cosine of the uniform vector), the random-sign input hashing cancels under averaging (state RMS 0.097, so `tanh` is linear), and one synapse of depth per token means the wiring's multi-hop structure is never traversed inside a token. We transplant the **looped-transformer principle** — iterate one shared block on a latent, re-inject the input every pass, stop on convergence, train with variable depth and deep supervision — onto the connectome: `K` damped iterations per token of the *same* graph, with `K = 1` reproducing FLM bit for bit; add the two things the loop forces (signed efficacies from the connectome's own neurotransmitter predictions, and a gain calibrated to a target spectral radius); and make the stack learn at test time (per-conversation fast weights on the readout during prefill, a local gain rule on the block, and — through the differentiable twin — gains, signs and pooling trained *through the loop*). Everything is measured backbone-free on the real graph against a degree-preserving rewired control and random signs — and the verdict (§5) is not the one the transplant hoped for. Under a linear readout with FLM's random pooling the connectome is a *memory*, not a *computer*, at every operating point: the real wiring beats a degree-matched rewiring by 0.04–0.09 nats on every run and both seeds, through a dense band of slow modes (three eigenvalues at 1.0 and twelve above 0.92, versus one and a bulk at 0.166 for the rewiring — modularity is memory), while looping, transmitter signs and driving the block into its nonlinear range all leave next-byte prediction within the noise of FLM's single step or make it worse. What remains is a tested, strict superset of FLM with test-time learning at three levels, a mechanism for FLM's null result, and a pointer to the knob nothing here turned: the readout interface.

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

- `c` is the loop feedback (the "re-injected input" is `m_t`; the loop variable is `z`); `h` a damping step; `g`, `s` per-neuron postsynaptic gain and presynaptic efficacy; `d = 1-c`. With `c = 0`, `g = s = 1`, `h = 1` this is FLM on the same seeded interfaces — bit-identical on the SciPy path and, on this x86-64 machine, through the C kernel as well (tested against a transcription of FLM's `Reservoir` on the toy graphs and on the real one; on FMA hardware the kernel forbids contraction while SciPy may fuse, so agreement there is to rounding) — so an FLM adapter transfers unchanged and every looped variant is a strict superset. `K = 1` with `c ≠ 0` is *not* FLM: the warm start enters the drive as `c·x_{t-1}`.
- Two readings of the same loop: a Universal-Transformer-style weight-tied depth recursion, and an Euler integration of the rate model `dz/dt = -z + tanh(gWs(m + cz))` — the textbook way to run a connectome. FLM's single step is one Euler step of size 1. Iteration `k` carries the input `k` synapses into the wiring *within* the token.
- **Convergence**: rows of `W` sum to ≤ 1 and `tanh` is 1-Lipschitz, so `|c|·max|g|·max|s| < 1` proves a unique fixed point per token (contraction in the max norm; unit test checks geometric decay at rate `c`). Signed, gain-calibrated graphs exceed that loose bound; then convergence is local, governed by the spectral radius of `c·gWs`, and the **residual is the canary** (`telemetry()` reports iterations used, residual, saturation, the bound and the spectral estimate). Adaptive exit freezes converged lanes, so a sequence's result never depends on its batch companions — FLM's lane-independence invariant, kept.

Three things the transplant forced, each a measured lesson:

1. **The composite-gain trap.** Iterating FLM's block naively (`c = 0.8`, `K = 8`, unsigned) drives 97 % of neurons past |x| > 0.5 and collapses the features to effective rank 8. The loop's fixed point amplifies the drive by `1/(1-c) = 5×`, so the token recurrence's gain on the Perron mode becomes `0.6 × 5 = 3` — supercritical. The `d = 1-c` drive scale restores FLM's small-signal DC gain (state RMS 0.027 at `K ≈ 7`), exactly invariant 2 of the recursive notes: *bound the loop variable before scaling the loop count*.
2. **An all-excitatory block iterated is a diffusion.** With `W ≥ 0` the resolvent `(I - cW)^{-1}W` amplifies the uniform mode most; more iterations average harder. Computation needs inhibition. MaleCNS ships per-body neurotransmitter predictions FLM deliberately ignored; among the retained neurons: acetylcholine 103,720 · glutamate 29,302 · GABA 22,069 · histamine 7,891 · dopamine 392 · octopamine 101 · serotonin 48 · unclear/unknown 3,177. GABA, glutamate (inhibitory in the fly central brain) and histamine become `s = -1`: **35.6 % inhibitory**, a balanced network. The controls for this prior are random signs with the same share, and the rewired graph with the same signs. (Measured outcome, §5: at these settings the signs remove half of the slow-mode band and cost 0.06–0.09 nats in every regime — the prior is anatomically right and, for a linear readout of this block, computationally wrong.)
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
| direct input (bigram baseline) | 3.350 · *3.382* | 3.064 | 2.940 · *3.120* | 0.244 | 1.06 · *1.11* | 22.3 | — | — |
| **FLM** (K = 1, unsigned) | **3.273** · *3.311* | 2.980 | **2.809** · *2.959* | 0.274 | 2.67 · *2.89* | 23.8 | 1 | 0.098 |
| FLM on rewired graph | 3.347 · *3.402* | 3.065 | 2.903 · *3.069* | 0.269 | 2.36 · *2.34* | 21.1 | 1 | 0.095 |
| FLM + NT signs (gain 0.95) | 3.340 | 3.055 | 2.852 | 0.275 | 2.78 | 23.7 | 1 | 0.092 |
| loop, unsigned (c 0.8, h 0.7, K ≤ 8) | **3.246** | **2.973** | 2.826 | 0.253 | 2.52 | 22.9 | 7.0 | 0.027 |
| loop + NT signs | 3.375 · *3.366* | 3.101 | 2.907 · *3.053* | 0.262 | 2.51 · *2.72* | 21.2 | 4.8 | 0.020 |
| loop + NT signs, rewired (own gain 4.08) | 3.354 · *3.368* | 3.059 | 2.858 · *3.017* | 0.269 | 2.25 · *2.59* | 21.6 | 7.8 | 0.092 |
| loop + random signs (same 35.6 %) | 3.361 | 3.074 | 2.915 | 0.253 | 2.30 | 20.7 | 4.8 | 0.020 |
| loop + NT signs + intrinsic plasticity | = loop + NT signs (the rule was a no-op, see below) | | | | | | | |

*Italics: seed 1 (different lane offsets, byte embeddings, rewiring). Online NLL is over 8,192 predictions; unpaired, its run-to-run wobble is ≈ 0.03 nats, so treat unpaired differences under ~0.06 as noise — the paired lane-bootstrap intervals in §8 resolve the actual effects to ±0.02.* Cost on 4 CPU cores: 0.20 s per 16-lane token at K = 1, 0.9–1.5 s looped.

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

The rewired graph is a textbook random matrix: one Perron mode (the mean) and a bulk whose radius, 0.166, coincides with the median row L2 norm of `W` (0.163) — the scale of a sum of ~150 random-signed weights. Everything but the graph-wide average decays by a factor 6 per propagation, times FLM's 0.6 leak — one token of memory, then nothing. The real connectome has **a dense band of modes above 0.92** (three at 1.0: near-closed subsystems; the band is the graph's modularity — hemispheres, optic lobes, nerve cord, mushroom body), each decaying by only 0.55–0.6 per token, so the recent past stays linearly decodable for two or three tokens. Signs remove half of that band (six modes above 0.91 instead of twelve), which is exactly why the signed variants lose memory and lose nats in this regime.

So the fly wiring's *entire* measurable contribution in FLM's formulation is ~1.5 tokens of linear memory, worth 0.07–0.09 nats to a linear next-byte readout — and worth nothing to a 1.2B transformer that already carries 1,536 tokens of context, which is why FLM's adapter cannot beat its direct-input control. The diagnosis of §2 stands, with the mechanism now named: **an averaging operator with slow modes is a memory, not a computer**, and neither iterating it nor signing it changes that while the block stays linear.

### 5.3 Why the block stays linear, and the fix

State RMS is 0.02–0.10 in every row above. FLM's drive `0.4·code[bins]·sign` has unit-RMS codes, but `W` averages ~150 random-sign inputs per neuron: the post-averaging drive is 0.4 × the RMS row L2 norm of `W` (0.244) ≈ 0.098 per neuron (measured 0.098) — `tanh` never bends. In echo-state terms FLM's *input scaling* is set ~10× too low for the operator's normalization. Sweeping the input scale on the real graph (4 lanes × 40 tokens):

| input scale | FLM state RMS · \|x\| > 0.5 · saturated | looped unsigned | looped + NT signs |
|---|---|---|---|
| 1 (FLM) | 0.097 · 0.2 % · 0 | 0.027 · 0 · 0 | 0.020 · 0 · 0 |
| 3 | 0.251 · 6 % · 0 | 0.078 · 0.1 % · 0 | 0.059 · 0 · 0 |
| 10 | 0.538 · 42 % · 2.4 % | 0.218 · 4 % · 0 | 0.178 · 2 % · 0 |
| 30 | 0.793 · 77 % · 23 % | 0.443 · 28 % · 0.7 % | 0.402 · 22 % · 0.4 % |

(The looped block sits lower at the same scale because its `1-c` drive normalization preserves FLM's gain only on the dominant mode; the resolvent attenuates the input's fast modes by up to `1-c`.) Scale 10 puts FLM's block squarely in the nonlinear regime; scale 30 does the same for the looped block. Those are the two reruns of §5.4.

### 5.4 The nonlinear regime

Reruns with FLM's input drive multiplied by 10 (FLM's block nonlinear: 42 % of neurons beyond |x| = 0.5, 2.5 % saturated) and by 30 (23 % saturated; the looped block at 28 % beyond 0.5); the ×30 run gives the rewired signed control the real graph's calibrated gain (0.95) instead of its own. Seed 0 throughout; the ×1 column repeats §5.1 for reference.

| variant | ×1 online / probe / memory | ×10 online / probe / memory | ×30 online / probe / memory |
|---|---|---|---|
| **FLM** (K = 1) | **3.273** / **2.809** / 2.67 | 3.297 / **2.792** / 2.00 | **3.281** / 2.867 / 1.66 |
| FLM on rewired graph | 3.347 / 2.903 / 2.36 | 3.334 / 2.848 / 1.83 | 3.335 / 2.918 / 1.40 |
| loop, unsigned | **3.246** / 2.826 / 2.52 | **3.271** / 2.802 / 2.19 | 3.315 / 2.905 / 1.89 |
| loop + NT signs | 3.375 / 2.907 / 2.51 | 3.361 / 2.902 / 2.02 | 3.357 / 2.963 / 1.44 |
| loop + NT signs, rewired, same gain | — | — | 3.373 / 2.986 / 1.00 |

Reading it:

- **Nonlinearity buys nothing.** FLM's block at ×10 matches itself at ×1 within the noise (probe −0.017, online +0.024); at ×30 saturation erodes the linear memory (2.67 → 1.66 delays) and the probe with it (2.809 → 2.867). The connectome's features are as good as they get in the linear regime FLM happened to pick.
- **The looped block never beats the single step by more than the noise floor** — its best showing is −0.027 nats online at ×1 and ×10, never on the offline probe — and at ×30 it is worse. Multi-hop propagation within a token, at the fixed point of a damped loop, produced nothing a linear readout could use beyond what one hop already gave.
- **Signs cost 0.06–0.09 nats in every regime.** At equal gain the signed real wiring still beats its rewiring (3.357 vs 3.373; memory 1.44 vs 1.00), so the wiring effect is real there too — but the signs' own cost dominates it.
- **The wiring effect at K = 1 is robust across operating points**: −0.037 (×10), −0.054 (×30), −0.074 / −0.091 (×1, two seeds), and in every case the real graph holds more linear memory than its rewiring.

**Verdict.** Under a linear readout with FLM's random pooling, the fly connectome is a *memory*, not a *computer*, at every operating point we could give it. Iterating the block, signing it from its own transmitter predictions, and driving it into its nonlinear range all leave next-byte prediction within ±0.03 nats of FLM's single step or make it worse; the one positive, exact result is that the real wiring beats degree-matched rewiring by 0.04–0.09 nats on every run and both seeds, entirely through its band of slow modes. This is the recursive notes' invariant read backwards: depth recursion buys nothing from a block that only averages — the fixed point of a contraction toward an averaging operator is a deeper average.

What the benchmark does *not* rule out, and where the remaining leverage is:

1. **The readout interface.** FLM pools 166,700 neurons into 128 sums of ~1,300 random-signed neurons each. By the central limit theorem such sums are nearly Gaussian and dominated by their linear components; whatever nonlinear structure the looped block computes in individual neurons is averaged away before the readout sees it. This is a hypothesis, not a measurement — but it is the one knob nothing above turned, and the differentiable twin already exposes it (learned per-neuron pooling weights; pooling by cell type or neuropil from the annotations; Kenyon-cell → MBON-style readouts).
2. **The task.** Byte-level English rewards linear memory of the last two symbols; a task needing nonlinear conjunctions of past inputs, or the backbone's actual 2048-d embeddings instead of random byte codes, could separate one hop from eight.
3. **Sample size.** 8,192 predictions per row give ±0.03 nats; effects that size are invisible here and irrelevant to a 1.2B backbone anyway.

### 5.5 Training the block through the loop

`train_graph.py` on the real graph, as a trainability demonstration rather than a result: 24 Adam steps over 4 lanes × 16-byte windows, the iteration count sampled from 1–4 per step (Huginn), deep supervision at K = 2 (TRM), autograd through only the last two inner iterations, gains bounded by `3·tanh(g/3)`, the NT-signed efficacies as free parameters. Each step — forward and backward through 25.6M edges × up to 4 iterations × 16 tokens, both directions through the C kernel — takes 4.3 s on 4 CPU cores. Training loss fell from 5.8 to ≈ 3.6–4.0 (noisy: the 128→256 head does most of the early learning); the exported block, loaded into the numpy reservoir under a fresh prequential readout on held-out text, moved the online NLL from 3.794 to 3.581 over 512 observations (tail 3.001 → 2.846). Mean gain drifted 0.950 → 0.929 from its calibrated start (rerun after the initialization fix in §8); no efficacy changed sign in 24 steps. The pipeline is the point: the block's *update rule* — gains, signs, pooling on the fixed anatomy — is now something the looped-transformer training recipe can learn, which is the R3 rung of the recursive notes applied to a brain graph. A 200-step run at 16 lanes (~1 h on a Mac with the kernel) is the experiment the user can make on real hardware; [`results/block-demo/block.json`](flm-loop/results/block-demo/block.json) holds the demo's log.


## 6. The real backbone, run inside the sandbox — and the path to FLM's own

### 6.1 Getting a pretrained language model past the egress policy

Hugging Face, its mirrors, ModelScope, Ollama and the PyTorch download host are all refused by the sandbox's egress proxy, so LFM2.5-1.2B and FLM's conversation corpus are unreachable. Two hosts are open: `storage.googleapis.com` (the connectome came from there) and GitHub raw content. The public **`keras-nlp` bucket** on GCS serves KerasNLP's GPT-2 presets as Keras-2 H5 files, and [`scripts/convert_keras_gpt2.py`](flm-loop/scripts/convert_keras_gpt2.py) maps them onto `GPT2LMHeadModel` (q|k|v concatenation, Conv1D layouts, `gelu_new`, tied head), rebuilds the byte-level BPE tokenizer from `vocab.json`/`merges.txt`, and adds a prefix-stable chat template. Sanity checks of the conversion: **GPT-2 base perplexity 38.3** on *Alice in Wonderland*, **GPT-2 medium 22.2**, and greedy continuations that behave ("The capital of France is → Paris, and the capital of France is Paris."). Text comes from the NLTK Gutenberg corpus on GitHub: 16 books for training (11.3 M characters), *Alice* and Chesterton's *Thursday* held out. `train_adapter.py --text-corpus` runs FLM's recipe on 256-token windows (96 train, 16 validation, 24 test; every training position supervised; test windows scored on their second half so the TTT row learns fast weights on the first half). Same graph, same seeded interfaces, same bias-free adapter (114,688 parameters at hidden size 768), same bounded logit correction, same CE + 0.5·KL objective and validation selection, same direct-input and relabeled controls — FLM with a different backbone and different text, plus our block variants and two extra controls.

### 6.2 Results (held-out NLL, nats per token; 3,072 targets per row)

| backbone · block | base | **fly adapter** | direct-input control | constant-feature control | relabeled wiring | fly + TTT (best lr) |
|---|---|---|---|---|---|---|
| GPT-2 base (124M) · FLM-equivalent (`c = 0`, K = 1) | 3.6668 | **3.5989** | 3.5985 | 3.6433 | 3.6837 | 3.5978 |
| GPT-2 base (124M) · looped, unsigned (`c = 0.8`, `h = 0.7`, K ≤ 8) | 3.6668 | **3.5987** | 3.5985 | *pending* | 3.6762 | *pending* |
| GPT-2 base (124M) · looped + NT signs (gain 0.95) | 3.6668 | **3.5982** | 3.5985 | *pending* | 3.6600 | *pending* |
<!-- GPT2-ROWS -->

*Perplexities for the first row: 39.1 → 36.6 / 36.5 / 38.2 / 39.8. The constant-feature control is the same adapter fed an all-ones feature, so it can only learn one fixed logit offset — a corpus prior. TTT: fast weights on the adapter's output projection, learned on each window's first 128 tokens, swept over learning rates 0.05–50 and norm bounds 0.5–5 ([`posthoc.json`](flm-loop/results/gpt2-base/flm/posthoc.json)).*

### 6.3 Reading

- **FLM's null result replicates on the first try**: the fly adapter and the parameter-matched direct-input control land 0.0004 nats apart (3.5989 vs 3.5985), exactly the paper's finding with LFM2.5. The relabeled-wiring control is *worse than the base model* (3.6837 vs 3.6668), also as in FLM: an adapter fitted to one node labeling misreads another.
- **What the adapters do learn is a token-conditional corpus prior, not connectome computation.** The constant-feature control captures 0.024 of the 0.068-nat gain (a fixed shift toward Gutenberg's token statistics); the direct-input control — a function of the current token only — captures all 0.068; the fly features add nothing on top. This is the backbone-free §5 result seen from the language model's side: the block's only asset was ~1.5 tokens of linear memory, and GPT-2 already carries 1,024 tokens of it.
- **Test-time training on FLM's readout has no leverage.** With the adapter capped at `0.03·tanh(·)` and the logit correction at RMS 0.25, fast weights learned on a window's first half move the second half's NLL by at most −0.001 nats (learning rate 5, fast-weight norm 0.8); at learning rate 50 they hurt (+0.004). Loop 2 of the [self-improvement notes](runtime-self-improvement.md) needs a substrate with more leverage than a bounded readout — a fast logit bias or a last-layer delta on the backbone itself — which is outside FLM's contract and left for the next study.
- Rows for the looped blocks and for GPT-2 medium are filled in as the runs complete; the mechanism predicts that the fly-versus-direct gap stays at zero for every block, and that the whole adapter gain shrinks as the backbone grows.

### 6.4 Costs, and FLM's own backbone on a laptop

- **Kernel**: `native/graph_lanes.c`, row-parallel OpenMP, no fused multiply-add: 17.6 ms per single-lane propagation (SciPy 73 ms), 139 ms for 16 lanes (SciPy 252 ms), 265 ms for 32 lanes, on 4 cores. The GPT-2 runs spend their time in the graph, not the language model: with 8 lanes, an FLM-equivalent extraction of 35 K tokens takes ~10 min, a looped one (~7 iterations) ~100 min; GPT-2 base's own forward over the same tokens is under a minute.
- **FLM's chat path** (one lane) pays ≈ 18 ms per token for the graph at `K = 1` and ≈ 7 × 18 ≈ 130 ms at the looped settings — comparable to the 1.2B backbone's decode step on a laptop, and an argument for adaptive exit exactly as in the Mac cost model of the recursive notes.
- **TTT during prefill**: one `lm_head` product over each 32-position chunk plus its gradient — seconds per 300-token prompt on CPU, less on MPS.
- **Run it with FLM's own assets** on a machine that can reach Hugging Face (the recipe is the one exercised above, with the backbone and data swapped back):

```sh
git clone https://github.com/nftechie/flm && cd flm && python scripts/download.py && python scripts/prepare_graph.py
curl -O https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather
cd ../flm-loop
python scripts/train_adapter.py --backbone ../flm/cache/lfm25-1.2b --graph ../flm/cache/malecns_v1 --feedback 0 --max-iterations 1 --output runs/flm-baseline
python scripts/train_adapter.py --backbone ../flm/cache/lfm25-1.2b --graph ../flm/cache/malecns_v1 --output runs/looped-v1
python scripts/posthoc_controls.py --run runs/looped-v1 --text-corpus <any text file>     # constant-feature control + TTT sweep
python scripts/chat.py --run runs/looped-v1 --ttt --telemetry
```

## 7. What this does and does not show

- **It explains FLM's null result mechanistically.** FLM runs the connectome as an unsigned averaging operator, once per token, in `tanh`'s linear range. Such an operator's only asset is its slow modes, and the fly's modular wiring has many of them: that is ~1.5 tokens of linear memory, worth 0.07–0.09 nats to a linear next-byte readout, robust across two seeds and three operating points against a degree-preserving rewiring — and worth nothing to a 1.2B transformer that already remembers 1,536 tokens, so the adapter cannot beat its direct-input control. FLM's own `shuffled` control relabels nodes and preserves the topology; it could never have seen the wiring effect that a rewiring shows.
- **It shows that the looped-transformer transplant, by itself, does not rescue the block.** Multi-hop iteration to a damped fixed point, signed efficacies from the connectome's transmitter predictions, and a nonlinear operating regime were each implemented, controlled, and measured; none moved a linear readout beyond ±0.03 nats of FLM's single step, and signs made it worse. The recursion principle of the companion notes holds in its negative form: iterating a block that only averages yields a deeper average.
- **On a real pretrained backbone it reproduces FLM's null result and explains it** (§6): with GPT-2 fetched from a public GCS bucket, the fly adapter and the direct-input control tie to 0.0004 nats, a constant-feature control shows the shared gain is a corpus prior, and test-time training on FLM's bounded readout has no leverage. FLM's own 1.2B backbone and conversation corpus remain the laptop experiment (§6.4).
- **What it leaves standing**: a bit-for-bit FLM superset with adaptive-exit telemetry, per-conversation fast weights, a differentiable block trained through the loop (demonstrated on the real graph), the rewired and random-sign controls, and the benchmark harness — plus the one untested hypothesis with real leverage: FLM's random pooling of ~1,300 neurons per feature averages away nonlinear structure before the readout sees it. Learned or anatomical pooling (cell types, neuropils, MBON/DN readouts), the backbone's real embeddings, and a task that needs nonlinear conjunctions are the next experiments; the code exposes all three.

## 8. Verification

*A second, adversarial pass over the code and the claims: what it found, what changed, and what checks out independently. Artifacts: [`results/verification.json`](flm-loop/results/verification.json), [`results/readout_sweep.json`](flm-loop/results/readout_sweep.json).*

**Code review** (all modules, single pass). Ten findings; four were correctness defects, each now fixed with a regression test (42 tests):

1. **Gain calibration was silently undone by the gain bound.** `TorchLoopedReservoir` stored the calibrated gain as the raw pre-`tanh` parameter, so a "calibrated to 0.95" block actually started at `3·tanh(0.95/3) = 0.919` — confirmed against the committed demo log, whose first `gain_mean` was exactly that number. Fixed by initializing the raw parameter at `limit·atanh(gain/limit)`; the §5.5 demonstration was rerun.
2. **"K = 1 reproduces FLM" was false for `c ≠ 0`**: the warm start enters the drive as `c·x_{t-1}`, so `K = 1, c = 0.8` computes `0.92x + 0.08u`, not FLM's `0.6x + 0.4u`. The correct statement — `c = 0`, any `K` — is now in the docstring, both READMEs and §3, with a test for both directions.
3. **`prompt_nll`'s test-time training was not causal on multi-turn input**: it learned on every non-assistant position, including later user turns whose hidden states had already attended to the answer being scored. Now restricted to positions before the first assistant token; tested on a two-turn conversation. `train_adapter.py`'s `fly_adapter_ttt` row builds one record per assistant turn and was already causal.
4. **Deep supervision below the truncated-backprop window trained only the head**: supervised iteration counts earlier than the last `loop_backprop` iterations were detached, so the block received no gradient from them. The window now extends to the earliest supervised count, and `supervise = 0` (which aliased the final state) is rejected. This affected the §5.5 demo at `K = 4` steps; rerun.

Also fixed: the same-gain rewired control recorded the *real* graph's spectral radius in its telemetry; `train_adapter.py`'s shared prefix could extend into answer tokens when every record opens identically (now capped before the first answer token); frozen arrays in the torch twin were plain attributes, invisible to `.to()` and `state_dict()` (now buffers); three verbatim copies of the hashing helper; and two efficiency defects — converged lanes were still propagated on every iteration (now only active lanes run, results bit-identical, kernel time ~1.6× lower at the benchmark's 4.8 mean iterations) and `feedback = 0` ran a second, identical iteration per token. Left as in FLM: `chat.py` drops a turn whose reply is empty.

**Independent checks on the real graph.**

- *The slow-mode mechanism, observed directly.* One random kick, then silence; state RMS per token (FLM dynamics, 0.6 leak):

  | operator | after kick | +1 | +2 | +3 | +4 | ratios |
  |---|---|---|---|---|---|---|
  | real `W` | 0.0947 | 0.0220 | 0.0088 | 0.0035 | 0.0017 | 0.23 · 0.40 · 0.40 · 0.49 → 0.56 |
  | rewired | 0.0943 | 0.0095 | 0.0010 | 0.0002 | 0.0001 | 0.10 · 0.10 · 0.20 → 0.6 (the mean mode alone) |
  | real `W` × NT signs, gain 0.95 | 0.0903 | 0.0177 | 0.0056 | 0.0016 | 0.0006 | 0.20 · 0.31 · 0.28 · 0.42 → 0.48 |

  Two tokens after an input the real wiring retains 9× the signal of its rewiring, three tokens after, 17×; the asymptotic per-token ratio 0.55–0.56 is the slow band (0.92–1.0) times the 0.6 leak. Signs cut the retained signal roughly in half, as §5.2 inferred from the spectrum.
- *FLM equivalence on the actual data*: a transcription of FLM's `Reservoir.step` and `LoopedReservoir(c = 0)` agree **bit for bit** on the 166,700-node graph — through SciPy and, on this x86-64 machine, through the C kernel as well.
- *Spectrum scale*: the rewired bulk radius (0.1655 / 0.1647 on two rewirings) coincides with the median row L2 norm 0.1635; the circular-law scale `sqrt(Σw²/n) = 0.244` is instead what sets the post-averaging drive — 0.4 × 0.244 = 0.098, measured 0.098 — so §2 and §5.3 now carry that number (the draft's 0.065 used the wrong norm).
- *Numbers*: a script compared 115 table entries in this note against the JSON results; one omission (seed-1 direct probe) was found and fixed.

**Readout robustness and error bars.** The four rows that carry the conclusions (FLM, its rewiring, the unsigned loop, the signed loop; seed-0 settings) were rerun after the fixes with their features cached ([`results/bench_reservoir_check.json`](flm-loop/results/bench_reservoir_check.json)), then re-scored under three online learning rates and three probe regularizations ([`results/readout_sweep.json`](flm-loop/results/readout_sweep.json)), with paired bootstrap intervals over the 16 lanes — the independent text samples. The default corpus (this repository's notes) had grown from 158,623 to 195,445 bytes since the first run (this note was added), so the rerun samples different passages: absolute levels move by ~0.2 nats, the comparisons do not. The benchmark header now records the corpus hash and file list.

| rerun, new text sample | online NLL | tail NLL | probe NLL at l2 = 10⁻³ / 3·10⁻³ / 10⁻² | memory |
|---|---|---|---|---|
| FLM (K = 1) | 3.101 | 2.771 | 2.566 / 2.619 / 2.775 | 2.92 |
| FLM on rewired graph | 3.195 | 2.849 | 2.626 / 2.691 / 2.857 | 2.66 |
| loop, unsigned | 3.080 | 2.767 | 2.580 / 2.630 / 2.769 | 2.78 |
| loop + NT signs | 3.196 | 2.859 | 2.644 / 2.715 / 2.880 | 2.79 |

Tail NLL under online learning rates 0.02 / 0.05 / 0.1: FLM 2.920 / 2.771 / 2.795 · rewired 2.996 / 2.849 / 2.868 · loop 2.874 / 2.767 / 2.806 · signed 3.028 / 2.859 / 2.878 — the ordering never changes, at any learning rate or regularization. Paired differences in tail NLL (learning rate 0.05) with 95 % lane-bootstrap intervals: **rewired − FLM = +0.078 [+0.058, +0.102]**, **loop − FLM = −0.003 [−0.014, +0.009]**, **signed − FLM = +0.088 [+0.062, +0.115]**; per-token standard errors ≈ 0.01. So the wiring effect is real and about 0.08 nats; the loop's effect is zero to within ±0.01; the signs' cost is as large as the wiring's benefit. The lane intervals of the *absolute* tail NLLs are wide (±0.2; FLM's is [2.55, 2.95]) because lanes are different passages — only paired, within-run comparisons mean anything, which is how every claim in §5 is made.

**Not covered by this verification**: the language-model path has been run on GPT-2 (§6) but not on FLM's LFM2.5 backbone or its conversation corpus; every benchmark row uses one interface seed (7301) and 8,192 predictions; bit-identity through the C kernel is a property of this machine (no FMA), not of the kernel on arm64.

## 9. Pointers

FLM: [repository](https://github.com/nftechie/flm) · MaleCNS v1.0: [downloads](https://male-cns.janelia.org/download/) (CC BY 4.0) · looped/recursive models and TTT: [recursive-self-improvement.md](recursive-self-improvement.md), [runtime-self-improvement.md](runtime-self-improvement.md) · code: [`flm-loop/`](flm-loop/) (36 unit tests, `python -m unittest discover -s tests`).
