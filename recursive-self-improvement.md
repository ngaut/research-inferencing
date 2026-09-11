# Recursive Models That Learn at Test Time — Three Recursions and the Fixed Point That Keeps Them Honest

*Research notes, September 2026. The recursive deepening of [runtime-self-improvement.md](runtime-self-improvement.md): that doc laid out four learning loops by timescale; this one asks what it means for the **model itself** to be recursive — the same block iterated on a latent state, fast weights iterated by their own gradient, and the update rule iterated by the model — and how test-time training composes with each. Companion to [fleet-architecture.md](fleet-architecture.md) and [omlx-code-study.md](omlx-code-study.md). Survey covers published work through mid-2026.*

---

## 0. One equation, three loop variables

Every "recursive model" in the literature is a fixed-point iteration:

```
s_{k+1} = F(s_k ; x, θ)        run K times, then read the answer out of s_K
```

What differs is **what `s` is**:

| Recursion | Loop variable `s` | One step is… | Representative work | Grain |
|---|---|---|---|---|
| **R1 — depth** | activations / latent state `z` | apply the shared block again | Universal Transformer, DEQ, looped transformers, Huginn, Mixture-of-Recursions, Ouro, HRM, TRM, Coconut, EBT | per token / per answer |
| **R2 — weights** | fast weights `W` | one (self-)supervised gradient step | linear attention/DeltaNet, TTT layers, Titans/ATLAS, LaCT, MesaNet — and per-task LoRA TTT, Cartridges | per token (fine) / per task (coarse) |
| **R3 — learner** | the update rule itself (its parameters, data policy, or code) | the model rewrites how it learns | self-referential weight matrices, HOPE's self-modifying memory, meta-trained TTT (TTT-E2E), SEAL, Darwin Gödel Machine, AlphaEvolve | per task / per night / per release |

Test-time training in the narrow sense is R2. "Test-time compute scaling in latent space" is R1. "Recursive self-improvement" is R3. They nest — an R2 layer is an R1-style loop whose loop variable happens to be a weight matrix; R3 is the loop that sets R1/R2's rules — and the engineering questions are identical at every level:

1. **What is the loop variable, and what bounds it?** (contraction, weight decay, normalization)
2. **How does the gradient cross the recursion?** (full BPTT · 1-step/Neumann approximation · implicit differentiation · deep supervision)
3. **Who decides `K` at test time?** (fixed · ACT halting · KL/entropy convergence · router · budget)
4. **What grounds the fixed point?** — the question that decides whether recursion converges to *correctness* or merely to *self-consistency*.

Question 4 is the whole doc. A recursion with no external grounding signal is an iteration on the model's own beliefs; its fixed point is agreement with itself. At R3 that failure has a name — **model collapse**, "the curse of recursion" (Shumailov et al., [arXiv 2305.17493](https://arxiv.org/abs/2305.17493)): train on your own outputs and the tails of the distribution vanish generation by generation. At R2 it is TTRL's ceiling (majority vote extracts only what the base can already reach). At R1 it is a looped block that oscillates or settles on a confident wrong latent. **The design rule is the same at every level: every recursion needs a grounding signal from *outside* the loop variable — training targets for R1, reconstruction or verified task data for R2, external verifiers for R3.**

## 1. R1 — recursion in depth: the same block, again

### 1.1 What the evidence says

- **Theory first.** Looped transformers are programmable computers (Giannou et al., [arXiv 2301.13196](https://arxiv.org/abs/2301.13196)); a `k`-layer block looped `L` times matches a `kL`-layer model on reasoning-heavy tasks while inducing a chain-of-thought-like inductive bias (Saunshi et al., [arXiv 2502.17416](https://arxiv.org/abs/2502.17416)). Depth recursion buys *reasoning* depth, not *knowledge* — knowledge scales with unique parameters, iteration scales with steps. Design consequence: recursive cores pair with something that holds knowledge (a large prelude/coda, retrieval, or R2 fast weights).
- **Huginn — recurrent depth at LM scale** (Geiping et al., [arXiv 2502.05171](https://arxiv.org/abs/2502.05171)): 3.5B params as prelude → recurrent core → coda, trained with a *random* recurrence count so any `K` works at test time. More loops = more test-time compute with no tokens emitted; on reasoning benchmarks it behaves like a far larger static model. Two operational gifts: per-token adaptive exit when successive latents stop changing (a KL threshold), and KV sharing across iterations — cache cost does not scale with `K`.
- **Mixture-of-Recursions** (Bae et al., [arXiv 2507.10524](https://arxiv.org/abs/2507.10524)): a lightweight router assigns each *token* its own recursion depth, with recursion-wise KV caching — hard tokens loop, easy tokens exit. **Ouro** (ByteDance, [arXiv 2510.25741](https://arxiv.org/abs/2510.25741)) pretrains looped 1.4B/2.6B LMs on 7.7T tokens with an entropy-regularized learned exit and reports parity with 4B–12B static models. Adaptive depth is what makes recursion *cheap on average*; without it a `K`-loop model is just a `K`× slower one.
- **HRM → TRM: the ARC surprise.** HRM (Sapient, [arXiv 2506.21734](https://arxiv.org/abs/2506.21734)) — 27M params, two coupled recurrent modules, 1-step gradient — reached ~40% on ARC-AGI-1. ARC Prize's [independent ablation](https://arcprize.org/blog/hrm-analysis) found the *hierarchy* barely mattered: the **outer refinement loop with deep supervision**, plus augmentation and voting, did the work. TRM (Jolicoeur-Martineau, [arXiv 2510.04871](https://arxiv.org/abs/2510.04871)) took the hint: a single 2-layer network, ~7M params, recursed over a latent `z` and a running answer `y` (`z ← f(x,y,z)` several times, then `y ← g(y,z)`, repeated under up to 16 deep-supervision steps, with full backprop through the recursion) — ~45% ARC-AGI-1, ~8% ARC-AGI-2, ~87% Sudoku-Extreme, beating most frontier LLMs on ARC with <0.01% of their parameters. The honest reading: HRM/TRM are trained *per benchmark, transductively* — the evaluation tasks' demonstration pairs are in the training set. That is per-task TTT in batch clothing; TRM is the strongest evidence yet that **tiny recursive specialists + task-local training** beat general giants on puzzle-shaped problems.
- **Latent-space variants**: Coconut ([arXiv 2412.06769](https://arxiv.org/abs/2412.06769)) feeds the last hidden state back as the next input — recursion through the model's own embedding space instead of tokens; Energy-Based Transformers ([arXiv 2507.02092](https://arxiv.org/abs/2507.02092)) refine the *output* by gradient descent on a learned energy inside the forward pass — R1 whose step is literally an optimizer step, the bridge to R2.

### 1.2 The mechanics that decide whether R1 works

| Concern | Options | What the results favor |
|---|---|---|
| Gradient through the loop | full BPTT · 1-step (HRM) · implicit/DEQ ([arXiv 1909.01377](https://arxiv.org/abs/1909.01377)) · deep supervision | **deep supervision + truncated BPTT** (TRM); DEQ's fixed-point elegance loses to stability in practice |
| Choosing `K` | fixed · ACT ([arXiv 1603.08983](https://arxiv.org/abs/1603.08983)) · KL/entropy convergence · router (MoR) | train with random `K`, exit on convergence at test time; router when tokens differ wildly in difficulty |
| Boundedness | LayerNorm/RMSNorm on `s` · residual injection of `x` every step · weight decay on fast weights | inject `x` each iteration (the loop must not forget the input); normalize the state |
| Convergence monitoring | ‖s_{k+1} − s_k‖ residual | expose it — it is the recursion's **canary** (§5) |
| Cost model | `K` × block time; weights re-streamed per loop unless on-chip; KV shared or per-recursion | on bandwidth-bound silicon, only *adaptive* `K` pays (§4) |

## 2. R2 — recursion in weights: test-time training proper

R2 is R1 with the loop variable moved from activations to parameters — and the family tree is the one [the companion doc](runtime-self-improvement.md#21-loop-1--fast-weights-the-state-is-a-model) drew: DeltaNet's delta rule is one gradient step per token; TTT layers ([arXiv 2407.04620](https://arxiv.org/abs/2407.04620)) make the inner learner a linear model or MLP; Titans ([arXiv 2501.00663](https://arxiv.org/abs/2501.00663)) add surprise-gated momentum and decay; ATLAS ([arXiv 2505.23735](https://arxiv.org/abs/2505.23735)) widens the objective to a window and upgrades the inner optimizer. Two 2025 results matter specifically for *recursive* design:

- **LaCT — large-chunk TTT** ([arXiv 2505.23884](https://arxiv.org/abs/2505.23884)): update fast weights in chunks of thousands to a million tokens instead of per token, which turns the inner loop into dense matmuls (hardware-friendly) and lets the fast weights be a large fraction of the model. The lesson: **the inner recursion's granularity is a free design variable** — per token (recurrent), per chunk (LaCT), per task (LoRA TTT), per corpus (Cartridges). Coarser grains are cheaper and more parallel; finer grains adapt faster. Pick per layer, not per model.
- **MesaNet** ([arXiv 2506.05233](https://arxiv.org/abs/2506.05233)): run the inner least-squares problem *to convergence* (conjugate gradient) inside the forward pass — R2 iterated to its fixed point rather than one step. It is the cleanest demonstration that "one gradient step per token" is a budget choice, and that more inner iterations buy accuracy at compute cost — the R2 analogue of Huginn's `K`.

The **per-task** end of R2 (ARC TTT, TTT-NN, SEAL, Cartridges — evidence in the companion doc) is the same recursion at coarse grain, with the grounding signal changed: per-token R2 is grounded by self-supervised reconstruction of the stream; per-task R2 is grounded by the task's own verified demonstrations or by retrieved neighbors. That is why per-task R2 is the one that produces *reasoning specialization* (ARC's ~6× lift, AlphaProof) rather than just recall.

### R1 × R2: the two recursions stack

A recursive core whose shared block contains R2 layers iterates twice per token: the latent refines over `K` depth loops **and** the fast weights refine as tokens stream. Nothing in the literature forbids this and the pieces already coexist — GDN hybrids (R2-per-token) sit inside models that could be looped (R1); TRM's recursion (R1) is trained with task-local data (coarse R2). The obvious combination for our workload is a looped hybrid core with GDN/TTT fast-weight layers plus a few full-attention layers for exact recall, trained with random recurrence depth and deep supervision — depth recursion for reasoning, weight recursion for in-context adaptation, unique parameters kept small so the model fits a Mac and its weights stay hot.

## 3. R3 — recursion in the learner: self-referential improvement

R3 is where "递归自我提升" lives: the process that improves the model is itself a product of the model.

- **Self-referential weight matrices** (Schmidhuber, 1993; Irie et al., [arXiv 2202.05780](https://arxiv.org/abs/2202.05780)): a weight matrix that generates the key/value/learning-rate signals used to modify *itself* — the update rule is not fixed but computed by the weights being updated. **HOPE / Nested Learning** (Behrouz et al., NeurIPS 2025) is the modern scaled form: a self-modifying Titans memory whose update targets, keys, and learning rates are its own outputs, wrapped in a continuum of slower memories.
- **Meta-training the inner loop.** MAML's lineage ([arXiv 1703.03400](https://arxiv.org/abs/1703.03400); learned optimizers, [arXiv 1606.04474](https://arxiv.org/abs/1606.04474); VSML, [arXiv 2012.14905](https://arxiv.org/abs/2012.14905)) trains the *outer* parameters so that the *inner* test-time updates work well — the outer loss is measured after the inner steps. TTT layers already do this for their inner learning rate; TTT-E2E (Sun et al., late 2025, reported) pushes it end-to-end so a small model with test-time weight updates matches full attention at long context by compressing context into weights. **This is the correct way to make R2 reliable: don't hope the base model happens to be fine-tunable at test time; train it to be.** For the fleet, it is a next-model-family selection criterion, not an engine feature.
- **Learned data policies.** SEAL ([arXiv 2506.10943](https://arxiv.org/abs/2506.10943)) trains, by RL, the policy that writes the model's own fine-tuning data and optimization directives — recursion at the data level, with the reported forgetting across sequential edits as the warning. RISE ([arXiv 2407.18219](https://arxiv.org/abs/2407.18219)) trains the model to recursively improve its *own prior answer* over turns — R3 at the response level.
- **Self-modifying scaffolds and pipelines.** The Darwin Gödel Machine ([arXiv 2505.22954](https://arxiv.org/abs/2505.22954)) keeps an *archive* of agent variants that rewrite their own code, selected by benchmark (SWE-bench roughly 20% → 50%); AlphaEvolve improved the kernels used to train the model that powers it (a reported ~23% kernel speedup, ~1% of total training time) — recursive self-improvement at the organization's timescale, and the clearest demonstration that R3 works when, and only when, a hard external evaluator (a benchmark, a kernel timer) is the referee. Self-play closes the loop from zero data — Absolute Zero ([arXiv 2505.03335](https://arxiv.org/abs/2505.03335)), R-Zero ([arXiv 2508.05004](https://arxiv.org/abs/2508.05004)) — again with an executor as ground.
- **Recursive inference over context.** Recursive Language Models (Zhang, Kraska & Khattab, MIT, late 2025) let the model treat an oversized prompt as a variable in an environment and *call itself* on slices — recursion in the call graph rather than the weights; a loop-0 recursion that composes with everything above and is the natural shape of an agent that decomposes a task and spawns TTT'd specialists for the pieces (§4.3).

The pattern across every R3 success: **an archive rather than replacement** (DGM keeps lineage; the companion doc's immutable artifact registry is the same idea), **an external referee** (tests, timers, executors, benchmarks), and **a rate limit** (nightly/weekly, not per request). The Gödel machine's original demand — prove the rewrite is an improvement before applying it — is replaced in practice by an empirical gate: the successor must beat the predecessor on held-out and anchor suites, or it is archived unused.

## 4. Design: a recursive, test-time-learning model on our fleet

### 4.1 The model recipe

```
              x (tokens)
                 │
   PRELUDE  ──── embed + a few full layers (knowledge, format)        [static]
                 │
   RECURSIVE ─┐  shared block, looped K times per token:
   CORE       │    · GDN / TTT fast-weight layers   ← R2, per-token/chunk
              │    · 1–2 full-attention layers      (exact recall over the KV)
              │    · x re-injected every iteration; RMSNorm on the state
              │    · exit when KL(s_k ‖ s_{k+1}) < ε  or  K = K_max
              └─ trained with random K (log-normal), deep supervision,
                 truncated BPTT through the last few loops              [R1]
                 │
   CODA     ──── few layers + LM head                                  [static]

   outer training: meta-objective measured AFTER inner R2 updates
                   (TTT-E2E / MAML style), so test-time steps are learned [R3]
   test-time knobs: K (depth loops) · T (inner R2 steps per chunk)
                    · per-task LoRA on demonstrations (coarse R2)
```

Why this shape: the prelude/coda hold the knowledge that recursion cannot create; the looped core buys reasoning depth with few unique parameters — which on a Mac means the hot weights fit in unified memory with room for KV; the fast-weight layers inside the loop give in-context adaptation without touching the KV budget; meta-training makes the R2 layers *reliably* trainable at test time instead of accidentally so; and the three knobs (`K`, `T`, per-task LoRA) are the levers a router can turn per request.

### 4.2 Recursion budgets are a routing decision — the P/D-threshold rule again

The [fleet doc](fleet-architecture.md) turned "should we disaggregate?" into a measured per-turn threshold. Recursion does the same: **`K`, `T`, and whether to spend per-task TTT are per-request knobs set from measured marginal-gain curves**, not architecture constants.

- On bandwidth-bound Macs each depth loop re-streams the shared block's weights, so a fixed `K` is a `K`× decode slowdown; only *adaptive* exit (Huginn's KL threshold, MoR's router, Ouro's learned exit) makes R1 affordable interactively. Measure the exit-depth distribution per workload; if the median token exits at `K ≤ 2` while hard tokens go to 8, recursion pays; if the distribution is flat at `K_max`, it doesn't — serve the static model.
- Sparks (compute-rich, decode-starved) are where high `K` and inner-loop `T` belong — the same comparative advantage that made them the prefill service: iteration-heavy, bandwidth-light work.
- Per-task R2 (minutes of LoRA on a task's demonstrations) is batch-tier, on the lab Spark — the ARC recipe as a service for puzzle-shaped subtasks.

A router rule in the fleet's style: `interactive → Mac, adaptive K, T=1; hard-flagged or verifier-failed-twice → Spark, K_max, T>1, per-task TTT allowed; batch → Spark P/D cell, everything on`.

### 4.3 The LLM as outer loop, tiny recursive specialists as inner tools

TRM's result suggests an agent pattern worth prototyping: the large model decomposes; for a subproblem with a small verified example set (grid puzzles, format transductions, schema mappings, regex/DSL induction, layout constraints), it **spawns a tiny recursive model trained on that task's examples**, runs it, verifies, and integrates. A 7M-parameter network trains in seconds to minutes on one Mac, needs no distributed anything, and is discarded or archived as a skill artifact. This is R2 at its coarsest (train a whole model per task) made cheap by R1 (the model is tiny because it recurses). It is a bet, not a result — TRM trained once per benchmark over days, not per task in seconds — but it is exactly the experiment the lab Spark and idle Macs exist for.

### 4.4 The nested recursion, as a runtime

```
  timescale     recursion            loop variable       grounded by            gate / canary
  ─────────     ─────────            ─────────────       ───────────            ─────────────
  per token     R1 depth loops       latent z            trained targets        residual ‖Δz‖, exit-depth dist.
  per token     R2 fast weights      W_fast              stream reconstruction  state-norm bound, token-ID probes
  per task      R2 coarse (LoRA/TTT) W_task              task demos / neighbors held-out demo accuracy
  per night     R3 consolidation     shared adapters     verifier-signed episodes anchor suite, auto-rollback
  per release   R3 meta / learner    update rule, model  external evals          archive, successor-beats-predecessor
```

Each level's grounding comes from *outside its own loop variable*, and each level runs at least an order of magnitude less often than the one above it. The runtime pieces are those of the companion doc — experience store, artifact registry, eval gate, learner plane — with one addition: **recursion telemetry** (exit depths, inner-step counts, residual norms, per-task TTT wins) feeding the router's marginal-gain curves.

## 5. Invariants specific to recursion

1. **Never recurse on an ungrounded fixed point.** Every level names the signal outside its loop variable that anchors it (§4.4). A level with no such signal is capped at loop 0 (it may write notes; it may not iterate on weights or rules).
2. **Bound the loop variable before you scale the loop count.** Normalize latents, decay fast weights, inject the input every iteration, KL-bound successive learners (R3's trust region). Convergence problems show up as quality drift, not errors.
3. **Instrument the residual.** ‖s_{k+1} − s_k‖ per level is the recursion's canary: a non-decreasing residual is a divergent loop (R1), a saturated fast-weight norm is a memory that stopped learning (R2), a successor that only ties the predecessor is a stalled R3. Alert on the distribution, not the mean.
4. **Archive, don't overwrite.** R3 produces successors into a lineage (DGM's archive; our artifact registry). Rollback is a pointer flip; branching from an earlier ancestor is always possible; the current head is never the only copy.
5. **The referee is never the loop variable.** Verifiers, anchor suites, and eval prompts are owned outside the learner and versioned independently; an R3 step may not modify the tests it is scored by (agent-written tests are provisional until CI runs them).
6. **`K` and `T` are measured, per workload, like every fleet threshold.** Ship the adaptive-exit distribution and the per-task-TTT win-rate as first-class dashboards; the architecture stays fixed while the constants move.
7. **Rate-limit upward.** Faster loops may run freely; each slower loop runs strictly less often and only on the accumulated, verified output of the faster ones. Recursion that skips levels (a per-request step that edits shared weights) is the bug class to make impossible by construction.

## 6. Napkin numbers

| Quantity | Value | Basis |
|---|---|---|
| TRM | ~7M params · ARC-AGI-1 ~45% · ARC-AGI-2 ~8% · Sudoku-Extreme ~87% | [2510.04871](https://arxiv.org/abs/2510.04871) |
| HRM | 27M params · ARC-AGI-1 ~40% · ARC-AGI-2 ~5% | [2506.21734](https://arxiv.org/abs/2506.21734), [ARC Prize ablation](https://arcprize.org/blog/hrm-analysis) |
| Huginn | 3.5B params, 800B tokens; reasoning ≈ much larger static models at high `K`; KV shared across loops | [2502.05171](https://arxiv.org/abs/2502.05171) |
| Ouro | looped 1.4B / 2.6B ≈ 4B–12B static on reasoning; 7.7T tokens | [2510.25741](https://arxiv.org/abs/2510.25741) |
| Looped-depth equivalence | `k` layers × `L` loops ≈ `kL` layers on reasoning-heavy tasks | [2502.17416](https://arxiv.org/abs/2502.17416) |
| LaCT chunk sizes | 2K–1M tokens per inner update; fast weights a large share of params | [2505.23884](https://arxiv.org/abs/2505.23884) |
| Per-task LoRA TTT (ARC) | minutes/task; ~6× lift for an 8B | [2411.07279](https://arxiv.org/abs/2411.07279) |
| Tiny-specialist training | 7M-param model: seconds–minutes on one Mac (estimate; TRM itself trained per benchmark over days) | scaling from TRM |
| RL-style consolidation capacity | rank-1 LoRA ≈ full fine-tuning for RL (≈1 bit/episode) | [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) |
| R1 decode cost on a Mac | `E[K]` × block bytes per token; pays only if exit-depth median ≪ `K_max` | roofline, [inference-engine-architecture.md](inference-engine-architecture.md) |
| DGM self-modification | SWE-bench ~20% → ~50% over archive generations | [2505.22954](https://arxiv.org/abs/2505.22954) |

## 7. Build order

1. **Recursion telemetry + budget router** on existing models: expose adaptive-exit/inner-step counters for any looped or hybrid model served, and route `K`/`T` by measured marginal gain. Zero model risk; establishes the curves everything else is judged by. *Small.*
2. **Per-task R2 as a batch service** (ARC recipe on the lab Spark; results archived as skill artifacts). Reuses the learner plane and eval gate from the companion doc. *Medium.*
3. **Tiny recursive specialists** (§4.3) as an agent tool: TRM-class network, task-local training on a Mac, verifier in the loop, archive on success. The cheapest experiment with the largest upside. *Medium.*
4. **Looped hybrid core for the next model family** — prelude/looped-core/coda with GDN/TTT layers, random-`K` training, deep supervision, meta-trained inner loop — selected or trained on the Spark island, served on Macs only after the exit-depth distribution says it pays. *Large; a model-selection decision.*
5. **R3 with an archive**: DGM-style lineage over playbooks → adapters → learner configs, verifier-refereed, nightly at most, successor-beats-predecessor gated. *Ongoing; the last thing to automate.*

## 8. Decision rules

- Residual keeps falling with `K` on held-out tasks → raise `K_max` on the Spark tier; flat after 2–3 loops → the workload doesn't need R1; serve static.
- A task class fails verification twice under the static model but passes after per-task TTT → make TTT the default for that class and queue a tiny-specialist experiment.
- Fast-weight norm saturates within a session → the R2 layer's decay is too weak or the chunk too large; retune before adding capacity.
- A successor ties the predecessor on anchors two nights running → the R3 loop has found its fixed point on the current data; stop spending until new verified surprise accumulates.
- Any proposal to let a per-request loop write to shared weights → no; that is the level-skipping bug (invariant 7).
- The one-line summary: **recurse on activations for reasoning, on weights for adaptation, on the learner for growth — and at every level, iterate only toward a signal the loop cannot forge.**

## 9. References

R1 depth recursion: [Universal Transformer](https://arxiv.org/abs/1807.03819) · [ACT](https://arxiv.org/abs/1603.08983) · [DEQ](https://arxiv.org/abs/1909.01377) · [looped transformers as computers](https://arxiv.org/abs/2301.13196) · [latent thoughts / looping](https://arxiv.org/abs/2502.17416) · [Huginn recurrent depth](https://arxiv.org/abs/2502.05171) · [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) · [Ouro](https://arxiv.org/abs/2510.25741) · [HRM](https://arxiv.org/abs/2506.21734) · [ARC Prize HRM analysis](https://arcprize.org/blog/hrm-analysis) · [TRM](https://arxiv.org/abs/2510.04871) · [Coconut](https://arxiv.org/abs/2412.06769) · [Energy-Based Transformers](https://arxiv.org/abs/2507.02092).
R2 weight recursion: [TTT layers](https://arxiv.org/abs/2407.04620) · [Titans](https://arxiv.org/abs/2501.00663) · [ATLAS](https://arxiv.org/abs/2505.23735) · [LaCT](https://arxiv.org/abs/2505.23884) · [MesaNet](https://arxiv.org/abs/2506.05233) · [ARC TTT](https://arxiv.org/abs/2411.07279) · [Cartridges](https://arxiv.org/abs/2506.06266) · [TTRL](https://arxiv.org/abs/2504.16084).
R3 learner recursion: Schmidhuber 1993 (self-referential weight matrix) · [Irie et al. 2022](https://arxiv.org/abs/2202.05780) · Nested Learning / HOPE (NeurIPS 2025) · [MAML](https://arxiv.org/abs/1703.03400) · [learned optimizers](https://arxiv.org/abs/1606.04474) · [VSML](https://arxiv.org/abs/2012.14905) · [SEAL](https://arxiv.org/abs/2506.10943) · [RISE](https://arxiv.org/abs/2407.18219) · [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) · [Absolute Zero](https://arxiv.org/abs/2505.03335) · [R-Zero](https://arxiv.org/abs/2508.05004) · Recursive Language Models (Zhang, Kraska & Khattab, 2025) · [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/).
Failure modes: [the curse of recursion / model collapse](https://arxiv.org/abs/2305.17493) · Dohare et al., *Loss of plasticity in deep continual learning*, Nature 2024 · companion docs in this repo.
