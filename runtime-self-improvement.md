# Designing a Runtime Self-Improving Model — Test-Time Training Across Four Loops

*Research notes, August 2026. Companion to [fleet-architecture.md](fleet-architecture.md), [omlx-code-study.md](omlx-code-study.md) and [sglang-lessons.md](sglang-lessons.md). The question: how do you design a model that keeps getting better **while it serves** — "test-time training" in the broad sense — and what would that look like on the fleet we already sketched (M5 Macs + CPU servers + DGX Sparks, agentic coding traffic)? Survey covers published work through early 2026.*

---

## 0. The reframe: "a self-improving model" is a category error

Nothing in the literature is *one* mechanism called test-time training. What exists is a stack of **learning loops at different timescales**, each updating a different kind of state, with different persistence and different blast radius when it goes wrong. Google's Nested Learning line (Behrouz et al., NeurIPS 2025) makes this explicit — every component, optimizer state included, is an associative memory updating at its own frequency, the way cortical rhythms nest — but the engineering version of the insight is older and simpler: **you don't design a self-improving model; you design a self-improving *system* whose model has plasticity at several timescales, each behind its own gate.**

| Loop | What updates | Cadence | Persists | Mechanisms | Risk / cost |
|---|---|---|---|---|---|
| **0 — context** | KV cache, prompt artifacts (playbooks, skills, retrieval, drafter corpora) | every turn | session; artifacts are durable | ICL, RAG, ACE/GEPA-style playbook evolution, Voyager skill libraries, NGRAM corpus | ~zero — revert = drop the context |
| **1 — fast weights** | a hidden state that *is* a small model, updated inside the forward pass | every token, by construction | one sequence | linear attention / DeltaNet / GDN, TTT layers, Titans, ATLAS | an architecture choice, not an ops choice |
| **2 — task weights** | LoRA / trained-KV "cartridge" for *this* task, repo, or user | per task or per session; seconds–minutes of GPU | task-scoped; durable only if promoted | dynamic evaluation, TTT-NN, ARC-style TTT, SEAL, Cartridges, TTRL | medium — gradient cost, forgetting; gateable |
| **3 — consolidation** | shared adapters, drafters, routers, eventually base weights | nightly / weekly ("sleep") | permanent | replay + distillation, STaR/ReSTEM/RLVR self-training, drafter retraining, adapter merging | highest — drift, reward hacking, forgetting |

Deliberately *outside* the table: **test-time search** (o1/R1 long CoT, best-of-N, MCTS + verifiers). Search improves the *answer*, not the model — but it is the premier *generator of training signal* for loop 3 (STaR's whole point is to close that loop: search at test time, keep what verified, train on it at night).

Two facts anchor the rest of the doc:

1. **We already run loop 1 in production.** The GDN/Mamba hybrid state that [oMLX's cache taxonomy](omlx-code-study.md) checkpoints per-block is, mathematically, a fast-weight memory updated by a delta rule every token (§2.1). "Should we do TTT?" is the wrong question; the question is how far up the ladder to climb.
2. **We already store the right artifact.** A Cartridge (§2.2) is a *trained KV prefix* — and the fleet's content-addressed KV block store speaks that format natively. The most systems-synergistic TTT result of 2025 drops into our architecture as "just another block chain."

## 1. TTT is three different things — keep them straight

The literature uses "test-time training" for three mechanisms that share nothing but the phrase:

- **TTT as architecture** (loop 1): the recurrent state is itself a learner, updated by gradient-like rules token-by-token *inside* the forward pass. Zero ops surface; you get it by choosing the model.
- **TTT as procedure** (loop 2): run actual optimizer steps at inference time, on data derived from the test input (its retrieved neighbors, its demonstrations, its own augmentations), producing throwaway or promotable weights.
- **TTT as search** (not training): spend more forward passes per answer. Included here only because marketing conflates it, and because its verified outputs feed loop 3.

The design consequence: loop 1 is a *model-selection* decision made rarely; loop 2 is a *serving-time* feature needing scheduler and cache support; loop 3 is an *offline pipeline* needing eval gates. Conflate them and you'll try to bolt loop-1 papers onto a serving stack (wrong layer) or run loop-3 self-training synchronously with requests (wrong cadence).

## 2. What the evidence actually says, loop by loop

### 2.1 Loop 1 — fast weights: the state *is* a model

The unifying result of 2024–25 sequence-model research is a duality: **linear-attention-style state updates are online gradient descent on an associative-recall loss.** Schlag et al. showed linear transformers are fast weight programmers ([arXiv 2102.11174](https://arxiv.org/abs/2102.11174)); DeltaNet's update is literally one GD step on `‖Sk−v‖²` per token, made parallelizable in 2024 and gated in Gated DeltaNet ([arXiv 2412.06464](https://arxiv.org/abs/2412.06464)) — which is what ships inside Qwen3-Next-class GDN hybrids, i.e. inside models our fleet already serves. RWKV-7 frames its state the same way ([arXiv 2503.14456](https://arxiv.org/abs/2503.14456)). In-context learning itself is implicit gradient descent in this sense (von Oswald et al., [arXiv 2212.07677](https://arxiv.org/abs/2212.07677)).

The research frontier climbs the same ladder by making the inner learner richer:

- **TTT-Linear / TTT-MLP** (Sun et al., [arXiv 2407.04620](https://arxiv.org/abs/2407.04620)): hidden state = a linear model or 2-layer MLP trained by self-supervised reconstruction on the sequence as it streams; matches/beats Mamba, keeps improving past 8K context where Mamba plateaus; mini-batched inner steps recover hardware efficiency. The same layers later stretched a 3-second video model to one-minute generations (Stanford/NVIDIA, 2025).
- **Titans** (Behrouz et al., [arXiv 2501.00663](https://arxiv.org/abs/2501.00663)): a deep memory module updated at test time by **surprise** (gradient magnitude of the associative loss) with momentum and weight-decay-as-forgetting; effective context reported past 2M tokens, strong BABILong results. **ATLAS** ([arXiv 2505.23735](https://arxiv.org/abs/2505.23735)) upgrades the inner rule again: sliding-window (Omega) objectives instead of per-token, Muon-style second-order inner steps.
- **Nested Learning / HOPE** (NeurIPS 2025): generalizes to a *continuum* of memories updating at different frequencies — the whole loop table above, internalized into one architecture.

Takeaway for us: this loop arrives **with the next model family**, not with engine work. The engine-side obligations are exactly the ones oMLX already discovered serving hybrids: recurrent-state checkpointing so cache hits don't replay the stream, state-aware block signatures, and (new, if TTT-layer models ship) inner-step determinism probes — the `tp_identity_probe` culture applied to fast weights. When evaluating next-generation models for the fleet, "how expressive is the test-time learner in the state" is now a first-class selection criterion alongside bandwidth-per-token.

### 2.2 Loop 2 — per-task gradients: the strongest results in the field

This is the loop people usually mean by TTT, and its results are startlingly consistent: **a small number of gradient steps on test-derived data beats much larger static models.**

- **Dynamic evaluation** (Krause et al., [arXiv 1709.07432](https://arxiv.org/abs/1709.07432)) is the ancestor: SGD on the test stream as you evaluate; large LM perplexity gains, especially under distribution shift. Sun et al.'s vision TTT ([arXiv 1909.13231](https://arxiv.org/abs/1909.13231)) made it self-supervised.
- **TTT-NN** (Hardt & Sun, [arXiv 2305.18466](https://arxiv.org/abs/2305.18466)): retrieve ~50 nearest neighbors of the prompt from a corpus index, one gradient step on each, then answer. Most of the gain lands in the first ~20 neighbors; largest wins exactly where the base model is weakest (code, off-distribution domains), where a TTT'd small model closes most of the gap to a model several sizes up. This is *local learning* (Bottou & Vapnik, 1992) reborn: don't be a good global function; become the best local function on demand.
- **ARC** (Akyürek et al., [arXiv 2411.07279](https://arxiv.org/abs/2411.07279)): per-task LoRA trained at test time on leave-one-out + geometric augmentations of the task's demonstrations lifts an 8B model ~6× over its non-TTT self, to 53% on the public validation set — 61.9% ensembled with program synthesis, human-average territory. The ARC Prize 2024 winners used the same recipe ([arXiv 2412.04604](https://arxiv.org/abs/2412.04604)); **AlphaProof** ran the loop at the IMO itself, fine-tuning on self-generated variations of each competition problem over days. TTT is how you buy *reasoning specialization* per problem, not just knowledge.
- **SEAL** (MIT, [arXiv 2506.10943](https://arxiv.org/abs/2506.10943)): the model *writes its own finetuning data* ("self-edits") and optimization directives, applies the LoRA update, and is RL-trained so its edits maximize downstream performance — self-modification as a learned policy. Small-scale but directionally important: the edit generator is itself improvable, and its reported failure mode (catastrophic forgetting across sequential edits) is a warning we design for in §4.
- **Cartridges** (Stanford Hazy, [arXiv 2506.06266](https://arxiv.org/abs/2506.06266)): instead of stuffing a whole corpus into context, *train a small KV cache* on it offline via "self-study" (self-generated QA + context distillation). Matches full-context ICL quality at **~39× less KV memory and ~26× higher decode throughput**, and cartridges compose by concatenation. This converts a repo, a doc set, or a long transcript from a per-request prefill tax into a one-time training cost plus a cheap, cacheable, *content-addressable* artifact.
- **TTRL** (Tsinghua, [arXiv 2504.16084](https://arxiv.org/abs/2504.16084)): RL at test time with **majority-vote pseudo-rewards** — no labels; Qwen2.5-Math-7B goes 16.7 → 43.3 pass@1 on AIME 2024 (~2.6×). The honest caveat is the ceiling: majority vote only extracts what the base model can already reach on a good day. Self-consistency is an *amplifier*, not an oracle — a theme that returns in §5.

The through-line: loop 2 pays off precisely when the test instance sits in a **thin slice of the distribution** — a specific repo, a specific formal domain, a specific user's conventions — which is a perfect description of agentic coding traffic.

### 2.3 Loop 0 — improvement without gradients: don't skip the free rungs

An embarrassing amount of "self-improvement" needs no optimizer at all, and current results say the ceiling is high:

- **Evolving playbooks**: ACE ([arXiv 2510.04618](https://arxiv.org/abs/2510.04618)) treats the context as a living document — generate, reflect, curate deltas — reporting ~+10% on agent benchmarks over strong baselines while avoiding the "context collapse" of naive rewrite-everything loops. **GEPA** ([arXiv 2507.19457](https://arxiv.org/abs/2507.19457)) evolves prompts by natural-language reflection on execution traces and beats GRPO by up to 20% with ~35× fewer rollouts — read that twice: *reflective text mutation currently out-performs policy-gradient RL per unit of compute* for agent adaptation. Tencent's "training-free GRPO" pushes the same move: keep the "policy improvement" as an editable token prior, not a weight delta.
- **Skill libraries**: Voyager ([arXiv 2305.16291](https://arxiv.org/abs/2305.16291)) accumulates verified, reusable code skills; the modern agent-memory systems (MemGPT-lineage, Reflexion [arXiv 2303.11366](https://arxiv.org/abs/2303.11366)) are the same idea for episodic memory. In coding agents, "a skill" is a function/snippet/runbook that passed its test — a naturally verifiable unit.
- **System-side adaptation**: the [SGLang study](sglang-lessons.md) already flagged NGRAM speculation with a runtime-fed corpus (≥2× acceptance when the corpus matches the output distribution) — the serving *system* self-improving within a session with zero gradients. Its gradient-ful sibling, online speculative decoding ([arXiv 2310.07177](https://arxiv.org/abs/2310.07177)), retrains the drafter on the live query distribution using spare decode-batch capacity, and belongs to loop 3.
- **Sleep-time compute** (Letta, [arXiv 2504.13171](https://arxiv.org/abs/2504.13171)): spend idle cycles pre-digesting context into conclusions before the user returns — loop 0's answer to "the fleet is idle at night," and the non-gradient on-ramp to the consolidation rhythm of §4.

Loop 0 artifacts have three properties gradients can't match: they're **auditable** (you can read the playbook), **portable across model upgrades** (swap the base model, keep the memory), and **instantly revocable**. That's why the ladder in §3 puts them first.

### 2.4 Loop 3 — consolidation: self-training with a verifier, or not at all

The nightly loop turns experience into permanent capability. Evidence for the pattern: **STaR** ([arXiv 2203.14465](https://arxiv.org/abs/2203.14465)) and **ReSTEM** ([arXiv 2312.06585](https://arxiv.org/abs/2312.06585)) — sample, filter by correctness, train on the survivors, repeat; **Self-Rewarding LMs** ([arXiv 2401.10020](https://arxiv.org/abs/2401.10020)) and **SPIN** ([arXiv 2401.01335](https://arxiv.org/abs/2401.01335)) iterate with the model as its own judge/opponent; **Absolute Zero** ([arXiv 2505.03335](https://arxiv.org/abs/2505.03335)) closes the loop entirely — the model *proposes* tasks, a code executor verifies solutions, zero external data — and works precisely because the executor, not the model, is the referee. The agent-level analogue is the Darwin Gödel Machine ([arXiv 2505.22954](https://arxiv.org/abs/2505.22954)): an archive of agent variants that rewrite their own scaffolding, kept honest by benchmark evaluation.

Two hard warnings from this literature:

1. **Self-judged reward degrades into an echo chamber.** Every loop that worked (STaR, ReSTEM, Absolute Zero, AlphaProof) had an *external* verifier — a unit test, a proof checker, an executor. Loops where the model grades itself drift, saturate, or hack the judge. Majority vote (TTRL) sits in between: an amplifier bounded by the base model's reachable set.
2. **Plasticity is a consumable.** Continual training erodes both old knowledge (catastrophic forgetting — EWC-era literature, SEAL's observed failure) and the *ability to keep learning* (loss of plasticity, Dohare et al., Nature 2024). Loops that touch shared weights need replay, anchoring, and rate limits as first-class design elements, not patches.

## 3. Design invariants

1. **Climb the plasticity ladder from the bottom.** Capture each improvement at the *cheapest substrate that holds it*: context → durable artifact (playbook/skill/corpus) → cartridge → task LoRA → shared adapter → base weights. Promote up the ladder only on measured evidence the cheaper rung leaks (the playbook that every session re-derives → cartridge; the cartridge every repo needs → adapter). Don't burn gradients on what a cache line can remember.
2. **Gradients eat only verified signal.** A trust hierarchy, descending: execution/test/CI verdicts → typed tool errors → human accept/reject → majority vote → model-as-judge. Loop 3 admits the top of the hierarchy; loop 0 may hold the bottom (a playbook can say "the judge liked X" — a weight update may not act on it). Never train on unverified self-output.
3. **Every learned thing is an immutable, content-addressed, signed artifact.** Adapters, cartridges, playbooks, drafter corpora — same discipline as KV blocks: hash-addressed, versioned, provenance-tagged (which episodes, which verifier, which tenant), promoted by an eval gate that *signs* them, revocable by pointer flip. No in-place mutation of anything the serving plane reads.
4. **The learner never sits on the serving path.** Loops 2–3 run async on learner hardware; a request composes *already-promoted* artifacts. The only in-band plasticity is loop 1, which is inside the forward pass by construction, and loop 0 context edits. (Per-task TTT for batch jobs — ARC-style — is a scheduled job with a deadline, not an interactive path.)
5. **Spend gradients on surprise.** Titans' gate generalizes: an episode earns learner compute in proportion to (verifier-confirmed) surprise — a failure the playbook predicted a success for, a repo the cartridge doesn't cover, a drafter acceptance collapse. Routine wins are cache hits, not lessons.
6. **Scope beats regularization.** The first defense against forgetting and cross-contamination is that updates live in *scoped artifacts* (per-repo, per-tenant, per-toolchain) composed at request time — not in shared weights guarded by cleverness. Shared consolidation happens rarely, with replay from the experience store and an anchor suite that must not regress.
7. **Every loop has a kill switch and a canary.** Per-artifact and per-loop: anchor-suite greedy **token-ID probes** before/after promotion (the `kv_canary` / `tp_identity_probe` culture applied to weights), automatic rollback on regression, and deliberate fault injection — poison an episode in staging and prove the pipeline rejects it.

## 4. The architecture on our fleet

The fleet doc's three planes map onto the loops with almost no new hardware roles: **Macs serve and hold loop 0–1; CPU servers hold the experience and artifact state; Sparks (plus idle Macs at night) are the learner.**

```
                              agent clients
                                    │
  ┌─ SERVING PLANE (M5 Macs, mixed-phase oMLX) ─────────────────────────────┐
  │  per-request composition:                                               │
  │    base model (GDN hybrid state = loop-1 fast weights, checkpointed)    │
  │    + repo cartridge        (trained-KV blocks, fetched by hash)         │
  │    + tenant/repo LoRA      (hot-swapped adapter, promoted only)         │
  │    + playbook & skills     (loop-0 context artifacts)                   │
  │    + session NGRAM corpus  (drafter fed with open files/prior output)   │
  └────────┬────────────────────────────────────────────────▲───────────────┘
           │ episodes: {context hash, actions,              │ artifacts by
           │ tool traces, verifier verdicts}                │ content address
           ▼                                                │
  ┌─ EXPERIENCE + ARTIFACT PLANE (CPU servers; extends the KV block store) ─┐
  │  EXPERIENCE STORE  append-only episodes, provenance-tagged,             │
  │                    verifier-signed rewards, per-tenant partitions       │
  │  ARTIFACT REGISTRY adapters · cartridges · playbooks · corpora —        │
  │                    immutable, content-addressed, eval-gate-signed       │
  │  REPLAY SAMPLER    anchor suites · surprise scoring · GC               │
  └────────┬───────────────────────────────────────────────▲────────────────┘
           │ training batches (surprise-gated)             │ promoted artifacts
           ▼                                               │
  ┌─ LEARNER PLANE (Spark lab bench + idle Macs overnight) ─────────────────┐
  │  cartridge self-study (repo → trained KV)   per-repo/tenant LoRA        │
  │  drafter + router retraining                per-task TTT batch jobs     │
  │  EVAL GATE: anchor suite + canary token-ID probes + fault injection     │
  │             → sign & promote, or discard; auto-rollback on regression   │
  └─────────────────────────────────────────────────────────────────────────┘
```

### A turn's lifecycle, with learning attached

```
turn N over repo R, tenant T:
 1  router resolves (T, R) → artifact set {LoRA_R@v3, cartridge_R@v7,
    playbook_toolchain@v12, corpus_session}; all content-addressed, all cached
    on the session's Mac from previous turns (miss = one store fetch)
 2  Mac serves: cartridge blocks splice in as prefix KV (no prefill for the
    repo's distilled knowledge); adapter applied to the batch; drafter uses
    the session corpus; GDN state adapts within the sequence (loop 1)
 3  agent acts; tests/CI/compiler run — the verifier verdicts are captured
    with the episode, not inferred later
 4  episode (hashes, verdicts, surprise score) appended to the experience
    store; playbook/skill deltas proposed by a reflection pass (loop 0) land
    immediately as new artifact versions — no gradient, no gate beyond lint
 5  nightly: replay sampler assembles surprise-weighted batches per scope;
    Sparks/idle Macs run cartridge self-study + LoRA consolidation; eval
    gate probes, signs, promotes; morning traffic composes the new versions
```

### Why each piece sits where it does

**Cartridges are the keystone, because the store already exists.** A cartridge is KV; our block store's whole contract — content addressing, signatures keyed on model+layout, portability across Metal/CUDA, splice-in as prefix — applies verbatim. The economics are the fleet doc's own numbers inverted: a 32K-token repo context costs ~2 GB of KV and ~1.6 s fetch *per session* (or ~70 s re-prefill); a cartridge distilling it is trained once on the lab Spark, is ~an order smaller (the ~39× compression from the paper), and decodes ~26× faster than full-context serving. For the prefix-heavy agentic workload, this is the single highest-leverage TTT result in the literature.

**Adapters ride the same rails.** A rank-16 LoRA for a 27B is tens of MB (attention-only) to a few hundred MB (all-linear) — small next to the KV traffic the store already moves. Serving-side, multi-adapter batching is solved technology in the CUDA world (S-LoRA, [arXiv 2311.03285](https://arxiv.org/abs/2311.03285); Punica's SGMV kernels) — the honest gap is that **oMLX/MLX needs the batched-LoRA kernel** to serve mixed-adapter batches without decode-throughput collapse; until it lands, adapter granularity is per-box (router affinity groups a tenant's sessions onto Macs holding their adapter — the affinity machinery exists).

**The learner is the Spark lab bench, scaled by sleeping Macs.** The fleet doc already reserves a Spark for "quantization runs, draft-head training, evals" — that *is* the learner plane; consolidation and cartridge self-study are new jobs on existing hardware. The overnight Mac fleet is the surprising capacity: MLX LoRA training runs at hundreds of tok/s on M-class boxes, so 8 idle hours × 30 Macs ≈ 10⁸ replay tokens per night without touching the Sparks — per-repo consolidation is embarrassingly parallel across boxes (one repo's LoRA per Mac, no distributed training needed).

**Verifiers are already in the traffic.** Coding-agent episodes come with compile results, test outcomes, CI verdicts, typed tool errors, and user accept/reject of diffs. This is the one workload where invariant 2's "verified signal only" is cheap: the reward model is `pytest`. (It still needs adversarial hygiene: a test the agent itself wrote is self-judged signal until CI runs it — tag provenance accordingly.)

**Per-task TTT (ARC-style) is a batch product, not an interactive one.** Minutes of per-task gradients don't fit interactive TTFT; they fit the "one-shot / batch job → Spark P/D cell" route that never touches Macs. Offer it as a tier: hard formal tasks (a gnarly migration, a proof, an optimization contest) get a TTT job — augment the task, fine-tune a throwaway LoRA, solve, discard or archive the LoRA as a skill artifact.

## 5. The reward problem, stated plainly

Everything above is plumbing around one question: **where does trustworthy improvement signal come from?** The literature's answer, compressed:

| Signal | Trust | Use it for |
|---|---|---|
| Execution / tests / CI / proof checkers | high — external, mechanical | loops 2–3 weight updates (RLVR/ReSTEM-style filtering) |
| Typed tool errors, crash/timeout | high but narrow | negative filtering; drafter/router training |
| Human accept / reject / edit-distance on diffs | medium; sparse, gameable via sycophancy | loop-3 preference data, diluted and audited |
| Majority vote / self-consistency | amplifier only (TTRL's ceiling) | loop-2 pseudo-labels where no verifier exists |
| Model-as-judge, "looks right" | low; echo-chamber fuel | loop 0 only (notes in a playbook), never gradients |

And the poisoning corollary, which is new attack surface unique to self-improving deployments: the experience store ingests text that flowed through sessions — including **prompt-injected tool output**. An adversary who can make a "successful" episode (inject instructions, let the agent's own test pass trivially) is running a supply-chain attack on tomorrow's weights. Defenses are the invariants: verifier-signed rewards only (and CI-run tests outrank agent-written tests), provenance tags on every episode, per-tenant partitions that never co-train, surprise-gated *human review* for episodes that would move shared weights, and staged fault injection to prove poisoned episodes get rejected. Treat the experience store as untrusted input to the learner, always.

## 6. Napkin numbers

| Quantity | Value | Basis |
|---|---|---|
| TTT-NN adaptation | ~20–50 neighbors × 1 step ≈ seconds on one box | [2305.18466](https://arxiv.org/abs/2305.18466) |
| ARC-style per-task TTT | minutes/task on one accelerator; ~6× accuracy lift (8B) | [2411.07279](https://arxiv.org/abs/2411.07279) |
| Cartridge vs full context | ≈39× smaller KV, ≈26× decode throughput, ICL-quality | [2506.06266](https://arxiv.org/abs/2506.06266) |
| TTRL, no labels | AIME24 16.7 → 43.3 pass@1 (Qwen2.5-Math-7B) | [2504.16084](https://arxiv.org/abs/2504.16084) |
| GEPA vs GRPO | up to +20% at ~35× fewer rollouts | [2507.19457](https://arxiv.org/abs/2507.19457) |
| NGRAM corpus drafting | ≥2× acceptance with matching corpus | [SGLang study](sglang-lessons.md) |
| 27B LoRA artifact | ~40 MB (attn-only, r16, bf16) – ~300 MB (all-linear) | param count |
| LoRA train rate, M-class Mac (MLX) | O(10²–10³) tok/s → ~10⁷ tok/box/night | mlx-lm reports; validate on M5 |
| Overnight fleet budget (30 Macs × 8 h) | ~10⁸ replay tokens/night, zero marginal hardware | above |
| Repo context, 32K tokens | ~2 GB KV, ~1.6 s fetch @10GbE vs ~70 s re-prefill | [fleet doc](fleet-architecture.md) |
| Anchor-suite gate | O(10³) fixed prompts, greedy token-ID compare ≈ minutes | sized like kv_canary sweeps |

## 7. What must be built, in order

The ladder (invariant 1) is also the build order — each stage ships value alone and de-risks the next:

1. **Loop 0 artifacts** — playbook/skill store with versioning + the session-fed NGRAM drafter corpus (already argued in the [SGLang study](sglang-lessons.md)). No gradients, no gate beyond review; immediate quality and decode-speed wins. *Small.*
2. **Experience store + verifier capture** — append-only episodes with CI/test verdicts and provenance, on the CPU plane next to the block store. Pure logging until a learner exists, but nothing later works without it, and it must predate the data you'll wish you had. *Small-medium.*
3. **Cartridge pipeline** — self-study jobs on the lab Spark producing trained-KV block chains; splice-in via the existing store path; per-repo cartridges for the hottest repos first. Needs a `cartridge` signature type and an eval gate comparing against full-context answers (token-ID probes). *Medium; highest leverage.*
4. **Sleep consolidation** — nightly per-repo/per-tenant LoRA on idle Macs + Sparks, replay-sampled, anchor-gated, registry-promoted; router learns artifact affinity. Includes the batched-LoRA serving kernel for MLX/oMLX, or the interim per-box adapter affinity. *Medium-large.*
5. **Online drafter/router adaptation** — spare-capacity drafter finetuning on live traffic ([2310.07177](https://arxiv.org/abs/2310.07177)); router thresholds re-fit from measurements (the fleet doc's `bucket`-policy spirit). System self-improvement, invisible to model quality. *Small-medium.*
6. **Per-task TTT tier + loop-1 model selection** — batch TTT jobs for hard tasks on Sparks; and when the next model family is chosen, weight test-time-learner expressiveness (TTT-layer/Titans-class state) as a first-class criterion. *Ongoing.*

Deliberately last: **any update to shared base weights.** By the time that's tempting, stages 1–5 have usually captured the win at a rung where rollback is a pointer flip.

## 8. Decision rules

- A lesson recurs across sessions but costs no gradient to state → it's a playbook line, not a LoRA.
- Same repo context prefilled by many sessions (store hit-rate says so) → fund a cartridge; cartridge answer-parity holds on probes → stop shipping the raw context.
- Per-repo verifier pass-rate flat while playbooks grow → context is saturated; fund sleep consolidation for that repo.
- Anchor suite regresses on promotion → auto-rollback, quarantine the training batch, review the episodes that moved it (the loop's incident report, in the constants-carry-incidents tradition).
- Verifier coverage of an episode class < ~90% → that class trains nothing but playbooks, whatever the throughput of plausible-looking wins.
- Drafter acceptance sags for a session/repo → feed the corpus first (loop 0); retrain the drafter only if corpus feeding stops paying (loop 3).
- The one-line summary: **serve on fast weights, remember in artifacts, learn at night — and let nothing into the weights that a verifier didn't sign.**

## 9. References

Fast weights / loop 1: [Schlag et al. 2021](https://arxiv.org/abs/2102.11174) · [von Oswald et al. 2022](https://arxiv.org/abs/2212.07677) · [Gated DeltaNet](https://arxiv.org/abs/2412.06464) · [RWKV-7](https://arxiv.org/abs/2503.14456) · [TTT layers](https://arxiv.org/abs/2407.04620) · [Titans](https://arxiv.org/abs/2501.00663) · [ATLAS](https://arxiv.org/abs/2505.23735) · Nested Learning/HOPE (Behrouz et al., NeurIPS 2025).
Per-task TTT / loop 2: Bottou & Vapnik 1992 · [dynamic evaluation](https://arxiv.org/abs/1709.07432) · [vision TTT](https://arxiv.org/abs/1909.13231) · [TTT-NN](https://arxiv.org/abs/2305.18466) · [ARC TTT](https://arxiv.org/abs/2411.07279) · [ARC Prize 2024](https://arxiv.org/abs/2412.04604) · [AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) · [SEAL](https://arxiv.org/abs/2506.10943) · [Cartridges](https://arxiv.org/abs/2506.06266) · [TTRL](https://arxiv.org/abs/2504.16084) · [s1](https://arxiv.org/abs/2501.19393).
Context/artifact evolution / loop 0: [Voyager](https://arxiv.org/abs/2305.16291) · [Reflexion](https://arxiv.org/abs/2303.11366) · [GEPA](https://arxiv.org/abs/2507.19457) · [ACE](https://arxiv.org/abs/2510.04618) · [sleep-time compute](https://arxiv.org/abs/2504.13171).
Consolidation / loop 3: [STaR](https://arxiv.org/abs/2203.14465) · [ReSTEM](https://arxiv.org/abs/2312.06585) · [Self-Rewarding](https://arxiv.org/abs/2401.10020) · [SPIN](https://arxiv.org/abs/2401.01335) · [Absolute Zero](https://arxiv.org/abs/2505.03335) · [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) · [online speculative decoding](https://arxiv.org/abs/2310.07177) · [EWC](https://arxiv.org/abs/1612.00796) · Dohare et al., *Loss of plasticity in deep continual learning*, Nature 2024.
Serving: [S-LoRA](https://arxiv.org/abs/2311.03285) · [Punica](https://arxiv.org/abs/2310.18859) · [Transformer²](https://arxiv.org/abs/2501.06252) · [LoraHub](https://arxiv.org/abs/2307.13269) · companion docs in this repo.
