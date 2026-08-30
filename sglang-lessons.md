# Lessons from SGLang — What Transfers to an Apple-Silicon Engine and a Heterogeneous Fleet

*Research notes, August 2026. Studied at [sgl-project/sglang](https://github.com/sgl-project/sglang) `78fa921` (main, ~Aug 29 2026) with the explicit question: what should [oMLX-class engines](omlx-code-study.md) and [our fleet design](fleet-architecture.md) steal? Companion to those two docs; numbers are SGLang's own in-repo measurements and CI-asserted floors.*

SGLang in 2026 is much more than the RadixAttention engine of the papers: a ~32K-line Rust **model gateway** (auth, MCP tool loops, WASM middleware, PD routing), eight speculative algorithms behind capability predicates, a hierarchical KV cache with pluggable L3 stores, a diffusion-model stack, elastic expert parallelism — and, the surprise most relevant to this repo, **an MLX hardware backend for Apple Silicon**.

---

## 1. The headline: SGLang already solved the Apple Silicon overlap problem

oMLX measured ~8% decode loss from per-token GIL/event-loop handoffs and mitigated it with burst budgets. SGLang's MLX backend (`srt/hardware_backend/mlx/scheduler_mixin.py`) solves it structurally: keep **two decode graphs in flight** — build step N+1's lazy graph directly against step N's *unmaterialized* output tokens and hand it to `mx.async_eval` before blocking on step N. MLX's dependency tracking makes the GPU run the steps back-to-back with no scheduling gap; the chain breaks only on composition changes (a waiting request, a finished request, grammar masks or custom logits processors, non-pure-decode batches). The MLX TP worker also shows the reference pattern for grammar on Metal: xgrammar's CPU bitmask path builds additive mask rows on the host and inserts them into the lazy graph.

The CUDA-side equivalent is instructive too: `FutureMap`, a pool-indexed on-device relay — the scheduler builds batch N+1 with `input_ids = None` and the forward gathers last-sampled tokens on-GPU from a `[req_pool_size]` buffer at entry. **The host never observes a sampled token before scheduling the next step.** And they hold themselves to it contractually: a CI gate asserts **≥95% GPU occupancy at batch size 1** (`fwd_occupancy`), the regime where CPU overhead shows first.

Process-topology lesson: what SGLang exiled from the GPU-loop process is **detokenization** (Python string work proportional to tokens/sec), with tokenization+HTTP in a second process — and the 2026 endgame (`SGLANG_RUST_SERVER`) collapses back to *one* process by rewriting that front-end as Rust threads inside the scheduler process. The arc is: identify the GIL work, exile it, then eliminate the IPC by leaving Python. Never "add Python threads."

## 2. Scheduling: honesty about cache-aware policies, plus a 52-line admission governor

- **FCFS is their default too.** Longest-prefix-match scheduling exists but *self-disables above 128 queued requests* because the per-pass tree walk and sort cost too much — and their own tuning guide says a healthy queue is 100–2000 deep. oMLX's FCFS-only queue is less of a gap than it looks.
- **In-batch dedup** is the cheap trick worth taking: a throwaway radix tree over the waiting queue; a request whose real-cache hit is ≤32 tokens but whose match *within the queue* is ≥32 gets deprioritized so **one member of a cold same-prefix cohort runs first and warms the cache for the rest**. Free hit-rate for concurrent subagents sharing a system prompt.
- **`new_token_ratio`** is an AIMD admission governor in 52 lines: start conservative (0.7× of worst-case decode reservation), decay toward 0.14× over 600 decode steps without incident, jump back up on a retraction. It self-tunes to the workload's actual output-length distribution — a principled replacement for hand-tuned decode-headroom constants.
- **Retraction ≠ recompute when a lower tier exists**: in PD-decode mode, retracted requests offload their KV to host and resume by reload. With a block store, memory-pressure eviction of running requests stops being a recompute event — directly applicable to the fleet.
- Their whole prefill/decode fairness feature is `--prefill-decode-interval` (~15 lines, default off) — chunked prefill bounds the stall well enough. oMLX's time-domain debt scheduler is *more* sophisticated than what SGLang ships; the difference is Metal's non-preemption forced oMLX's hand.

## 3. The KV-store playbook — our block store, pre-validated

SGLang's HiCache documents the exact contract our fleet design chose: **L2 (host RAM) is instance-private; only L3 is shared** — "cross-instance reuse is the job of L3." (And the trap: the `file` backend defaults to node-local `/tmp`, giving zero sharing while looking like L3.) The mechanisms to copy into the CPU-fleet block store, all platform-neutral policy:

1. **Content address = chained per-page SHA-256** (`SHA256(prior_digest ‖ page_tokens)`), computed in C++, stable under radix-node splits. oMLX independently converged on the same chain — two production systems arriving at one design is strong evidence it's correct.
2. **Contiguous-prefix write invariant**: never store a block whose parent block isn't stored. This is what makes "longest existing prefix" a single monotone query instead of a scatter.
3. **Probe-first, allocate-exactly-the-hit**: existence checks in 128-page batches that stop at the first partial batch; staging memory allocated lazily after the hit length is known; degrade to a shorter page-aligned prefix under pressure.
4. **Prefetch stop policy**: default `timeout` with a linear budget — `min(30 s, 2 s + 0.1 s × tokens/1024)` — so a slow store degrades TTFT gracefully instead of stalling admission.
5. **Advisory existence cache** semantics: a bounded LRU of "pages I believe the store holds," keyed by content hash so entries survive tree mutations; a stale positive costs a skipped write-back, a miss costs one idempotent rewrite, never correctness.
6. **Decide staged fetches against the live tree, not carried IDs** (their buffer mode): between "I decided to fetch" and "I can splice it," the local tree has split/grown/evicted — re-match by tokens, cancel what's now covered, free holds that can no longer splice. Directly applicable to our store-mediated Spark→Mac handoff, where that gap is a network round-trip wide.
7. **Storage page size need not equal local page size** — the chain just has to be computed at the storage granularity. Decouple oMLX's 256-token blocks from any engine-local page choice from day one.
8. **`kv_canary`**: 32 bytes per KV slot of chained fingerprint (token, position, prev-slot hash, optional 16-byte data hash), verified per-forward plus a periodic radix-walk sweep, with a liveness counter proving the checker itself ran, and deliberate fault injection to test the detector. For a fleet where KV blocks cross machines and vendors (CUDA-written, Metal-read), this is the cheapest correctness net available — build it into the block store client.

PD-disaggregation details that harden our tier-3/Spark path: two-tier addressing (a per-request `bootstrap_room` + a per-engine session id); decode-side pre-allocation with a *three-way* length split (device-resident prefix / store-covered prefix the sender must NOT transfer / the delta actually shipped) — which is precisely how a store-mediated handoff avoids resending what the receiver can fetch itself; transfer is **chunk-granular, not layer-wise** (one batched RDMA per prefill chunk, destination-contiguous coalescing, partial tail page withheld for the next send — matching our corrected streaming math); the abort ordering rule (mark failed → register ack → drain, so no in-flight write lands in freed pages); and **decode nodes writing their generated KV back to L3** so prefill nodes reuse it next turn — our "reply backfill," shipped.

## 4. The router playbook — including what *not* to copy

Their gateway's `cache_aware` policy is the productized version of our affinity router: an approximate radix tree over **raw request characters** (no tokenizer in the router), keyed `pool::model` (un-keyed trees made cache-aware "flip-flop between pools" under PD — a bug class our router avoids by construction), updated write-through on every decision, LRU'd by an atomic epoch counter. The load override is a **two-condition AND** — `(max−min) > 64` requests *and* `max > 1.5×min` — because single-threshold designs oscillate. Other keepers: retries **re-select the worker pair per attempt** (rerouting, not resending); streaming failures are recorded when the stream *drops*, not when headers return 200 ("200-then-broken" decode workers); `manual` sticky sessions keep 2 pre-warmed failover candidates and never remap on scale-*up*; and the `bucket` policy is our P/D length threshold, *learned* — boundaries re-fit to equalize observed character-load per worker, with 2× hysteresis.

The anti-lessons matter as much: the router's cache model is **open-loop** — no feedback from workers about actual KV state (no KV-event subscription is wired), so it models "what I told each worker," not "what each worker holds"; multi-router tree sync is unimplemented (the receive path has no callers); PD pair selection has zero pair affinity; and the stale-tenant fallback routes to `first()`, a hot-spot generator during worker flaps. **Our design is structurally better positioned on exactly this axis**: with a shared block store, the router doesn't need to *guess* cache state — the store's block index is ground truth, and SGLang's engine even emits the right feed (`take_events()` → `KVEventBatch` of stored/removed blocks) that their own router ignores. Also their honest complexity note: consistent-hash-with-bounded-load (`prefix_hash`, first 256 tokens, spill at 1.25× average load) is the right operating point when caches aren't disjoint — which, with a shared store, ours aren't. Start there; add the tree only if measurements demand it.

Load reporting done right: two channels — a shared-memory snapshot for local dispatch, and a network PUB gauge for routers (tiny queue, dedup, 1 s heartbeat, never blocks the loop; "shedding at a full pipe loses readings the next heartbeat supersedes").

## 5. Agent-workload specials

- **Jump-forward decoding is dead code.** The famous compressed-FSM optimization has implementations in all four grammar backends and *zero call sites*. Its inventor removed it from the hot path — the retokenization hazard and radix-cache interaction weren't worth it. The replacement for the same goal (deterministic JSON spans emitted cheaply) is **speculation + grammar-aware tree masks**: draft tokens by any means; the grammar prunes the verify tree. Strong prior for anyone (oMLX included) tempted to implement jump-forward.
- **The grammar overhead-hiding triple** (~90 lines to port): async-D2H the verify tree *before* the target verify launches; build the token bitmask on the host *after* the launch (all host work hides under the forward); an idempotent FSM barrier advances matchers over prior committed tokens so masks are never stale. Wrapped in a `supports_grammar_overlap()` capability so non-conforming algorithms serialize instead of corrupting. Plus a defensive gem: reject schemas containing NUL bytes — they segfault xgrammar's regex converter, and agent tool schemas are user-supplied.
- **NGRAM speculation with a runtime-loadable external corpus** — the highest-leverage cheap win for coding agents: a suffix-trie/SAM drafter with no draft model, plus HTTP endpoints to feed it documents at runtime. Their test asserts **≥2× acceptance** when the corpus matches the output distribution. Agentic coding is maximally repetitive (file contents echoed across turns, tool-call envelopes, diffs): seed the corpus with the session's open files and prior outputs for near-free decode speedup on any engine, Metal included.
- **Adaptive speculation turns itself off**: per-batch-size tiers with EMA'd acceptance — candidate depths `[1,3,7]` at BS 1 shrinking to `[0]` at BS ≥ 64, pre-built graph state per tier swapped by reference. Same conclusion oMLX reached from its row-wise-batch measurements (spec pays at low concurrency, batching wins at high), here as an automatic policy.
- **DSpark's global knapsack** is the principled version of per-request adaptivity: a trained confidence head → per-position survival = cumprod(confidence) → sort *every (request, position) candidate in the batch* and pick the budget maximizing `E[accepted tokens] × steps/s` against a profiled cost table — so one request gets 7 verify slots and another gets 1 in the same step. Acceptance on JSON tool calls is bimodal; a fixed draft depth is wrong for both modes.
- Session-aware radix eviction (v0.5.17: "agentic workloads track session references for smarter eviction") and heterogeneous CPU offload of vision encoding to Xeons (v0.5.13: ~1.3× P99 TTFT) — the latter validating our CPU-plane aux-model role verbatim.

## 6. Multi-rank divergence: enumerate the decision sites

`SGLANG_SPEC_TP_SYNC` names **16 individually-toggleable sites** where a speculative step makes a rank-local decision that could diverge across TP ranks — with the measured ground truth that *free-GPU-memory queries* are the one input that actually differs. The fix menu per site: reduce to group-min, or broadcast from rank 0; the toggles let live traffic isolate which site diverges. The general rule for our Spark↔Mac path and any cross-box control decisions: any control decision derived from a machine-local quantity (free memory, clocks, RNG) will diverge, and it shows up as a hang or a silent quality drop, not an error. Enumerate the sites; make each one switchable. (Their TP-identity probes — raw greedy token IDs, never text — are the matching validation tool, same as oMLX's.)

## 7. Numbers worth keeping

| Claim | Value | Source |
|---|---|---|
| Overlap-scheduler contract | ≥95% GPU occupancy at batch 1 (CI-gated) | `fwd_occupancy` kit |
| EAGLE-3 vs no spec (8B, H100, MT-bench) | 158 → 373 tok/s (**2.36×**) | docs |
| DFlash / DSpark CI acceptance floors | 2.8 @ acc 0.74 / 2.0 @ acc 0.80 | registered tests |
| NGRAM + matching external corpus | **≥2× acceptance** vs trie-only | registered test |
| Router cache-aware defaults | threshold 0.3 · abs 64 · rel 1.5 · evict 120 s | gateway CLI |
| GPU-assisted KV I/O kernels vs `cudaMemcpyAsync` | up to 3× | HiCache design doc |
| Heterogeneous-TP PD staging | 2–5× at high concurrency, ~5% off homogeneous | PD docs |
| Xeon offload of vision encoding | ~1.3× P99 TTFT | v0.5.13 notes |
| Prefetch timeout budget | min(30 s, 2 s + 0.1 s/Ki-token) | HiCache defaults |

## 8. Meta-lesson

SGLang and oMLX, at very different scales, share the culture that makes their code worth studying: constants carry their incident reports; policies self-disable outside their measured competence (LPM > 128 queue, spec at BS ≥ 64, TBO refusing unbalanced splits); old paths get deleted, not deprecated in place (spec V1 removed, jump-forward abandoned); correctness is verified end-to-end by construction (kv_canary, TP-identity probes, poison-value CI buffers, fault-injecting the detector). The fleet we designed should hold itself to the same bar — most cheaply by adopting their two verification nets (canary fingerprints in the block store, raw-token-ID cross-device probes) before the first heterogeneous handoff ships.
