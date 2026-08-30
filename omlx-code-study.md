# oMLX Code Study — Inside a Production LLM Inference Engine for Apple Silicon

*Research notes, August 2026. Studied at [jundot/omlx](https://github.com/jundot/omlx) commit `e008a66` (v0.6.4, 2026-08-30). A companion to [inference-engine-architecture.md](inference-engine-architecture.md) — oMLX turns out to be the most concrete real-world answer available to that document's question "can an engine use all of the hardware (CPU, GPU, ANE) for max throughput?"*

All performance numbers below are the project's own in-repo measurements (docs, code comments, benchmark scripts), mostly on M3 Ultra; they are unusually well-documented but not independently reproduced here.

---

## 1. What oMLX is

**oMLX** is an open-source (Apache 2.0) LLM inference *server* for Apple Silicon by Jun Kim (jundot), built on Apple's MLX. It began as a fork of vllm-mlx v0.1.0 and now goes far beyond it: a vLLM-class serving stack (continuous batching, paged KV cache, prefix caching, speculative decoding) fused with things no datacenter engine has (SSD-persistent KV that survives restarts, multi-model desktop pooling under a hard memory ceiling, a native SwiftUI menu-bar app, and an Apple Neural Engine offload). Its reason to exist, per the README: make local coding agents (Claude Code, OpenClaw, Cursor, Codex) actually usable — agents re-send huge prompts every turn, so warm-prefix TTFT drops from 30–90 s to seconds.

Vital statistics:

| | |
|---|---|
| Code | ~225K lines of Python (server) + SwiftUI app; `scheduler.py` alone is 13.3K lines, `oq.py` 9.1K, `server.py` 7.4K; 361 test files |
| Velocity | ~400 commits in the last month; near-daily releases (0.5.4 → 0.6.4 across August); new frontier models (DeepSeek V4 Flash, GLM-5.3, Qwen3.8-Flash-Next) supported within days of upstream release |
| Pins | `mlx==0.32.0` (exact, ABI-coupled to bundled Metal kernels via `nanobind==2.13.0`), `mlx-lm@ab1806e` (v0.31.3), `mlx-vlm`, `mlx-embeddings` |
| APIs | OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`), Anthropic (`/v1/messages`), embeddings, rerank; admin dashboard; MCP |
| Model types | LLM, VLM, OCR, embeddings, rerankers, STT/TTS |

Architecture at 10,000 ft (one OS process):

```
FastAPI / uvicorn (asyncio)  ── OpenAI + Anthropic + admin routes, SSE keep-alive
    │
    ├── EnginePool (asyncio.Lock) — every discovered model has an entry:
    │     LRU eviction · TTL · pinning · lease refcounts · pre-load admission
    │     BatchedEngine / VLMEngine / DFlashEngine / Embedding / Reranker / audio
    │
    ├── EngineCore per loaded model — ONE dedicated MLX worker thread
    │     with its own mx.Stream (so two models can decode concurrently)
    │
    ├── Scheduler per engine (13.3K lines):
    │     external chunked prefill · decode-fairness debt · admission guard
    │     └── mlx-lm BatchGenerator (continuous batching backend, monkey-patched)
    │
    ├── Cache stack: PagedCacheManager (metadata only) → hot RAM tier → SSD tier
    │     + boundary-snapshot store for recurrent/hybrid state
    │
    └── ProcessMemoryEnforcer — polls 1s/10s/30s, ceiling = min(static, dynamic, metal_cap)
```

---

## 2. The serving core: five design decisions worth knowing

### 2.1 Prefill runs *outside* the batching engine

vLLM schedules prefill and decode into one batched forward. oMLX inverts this: `Scheduler` runs prompt processing itself as a standalone chunked per-request loop (`model(chunk, cache)` directly), and hands mlx-lm's `BatchGenerator` only the **last prompt token plus a fully built cache**. The docstring is explicit that mlx-lm already does token-level continuous batching, "so we use it as the backend." Everything distinctive in oMLX — the prefix cache, boundary snapshots, the memory guard, per-chunk throttling, cross-engine fairness — hangs off that inversion. The cost is a set of careful monkey patches on `GenerationBatch` fixing row/processor misalignment bugs that batch merges expose (a stale `logits_processors` list surviving `filter()` meant a later request's grammar processor was silently never applied).

### 2.2 Scheduling in the time domain, because Metal cannot preempt

"Metal cannot preempt a running kernel, so bounding chunk duration IS the interleave mechanism." Under contention, prefill chunk size is derived from a **500 ms stall target divided by the measured prefill tok/s** — "stall tolerance is a human constant while tokens/second is a machine constant" — snapped down to a 64-token grid (DeepSeek-V4's native indexer only engages when chunk % 64 == 0; an arbitrary 297-token chunk would silently take a ~4× slower fallback). Each chunk accrues **decode debt** (fair share 0.5); repayment has two channels — measured decode wall time for the engine's own decodes, real-time deadlines for other engines' — with a *process-global* hold deadline so two engines prefilling concurrently pause together. Cross-engine visibility comes from a `DecodeActivityRegistry` with a 2.5 s TTL so a wedged engine can't throttle the pool forever. Documented tuning: 0.5 gives decode ~1/3 of wall time (measured M3 Ultra sweet spot: contended decode holds ~3× unthrottled rate at parity batch-completion time).

### 2.3 Admission control prices the *throttled* regime

Memory admission uses one formula (`estimated = current + kv_exact + transient`) at three rejection sites, but charges the **floor chunk (32 tokens), not the nominal 2048** — because under pressure the adaptive throttle will actually run at floor-size chunks. Measured justification in-line: charging the full step rejected an 80K-token prompt that in fact ran to completion at 32-token chunks. In `prefill_priority: "speed"` mode the throttle never shrinks, so admission charges the full step — the guard stays self-consistent with the execution regime. The admission limit is `min(hard_limit, 0.95 × ceiling)` rather than the ceiling itself, because the enforcer aborts at 95% — admitting into that band means a request burns a minute of prefill and dies anyway ("every residual mid-abort across three A/B regimes sat in this band"). On rejection, the guard logs the bound decomposed into its terms (current, predicted transient, observed max, ANE transient) "so a rejection is diagnosable from one log line."

Other admission machinery: FCFS only (a priority enum exists but nothing sorts the queue); head-of-line requests that stall 60 s at an admission gate are failed with a typed error instead of hanging; a new request whose prompt overlaps an in-flight cache store waits up to 4 s so its prefix lookup sees the fresh blocks.

### 2.4 The memory ceiling is three ceilings, and the enforcer fights its own runtime

`ceiling = min(static, dynamic, metal_cap)` where static = RAM − tier reserve (8/6/4/2 GiB for safe/balanced/aggressive/custom), dynamic = a live estimate from mach `vm_statistics64` of what macOS can actually give (free + inactive + a tier-dependent fraction of active), and metal_cap = the kernel's `iogpu.wired_limit_mb` or Apple's recommended working set (~75% RAM). Pressure levels ok/soft/hard/emergency escalate from admission pauses through hot-cache shrink, pooled-buffer reclaim, LRU model eviction, request aborts — with a 5-poll grace window before destructive action, and recovery hysteresis down to the soft watermark. The **abort** limit deliberately excludes the dynamic ceiling: other-app pressure jitters every poll and must not kill a near-complete prefill that fits the physical envelope.

A remarkable amount of this code exists because MLX/Metal doesn't return memory: `set_cache_limit` pins the buffer pool (a kernel-panic guard), so dropping array references doesn't move `phys_footprint` and the enforcer can livelock concluding "no evictable models." Hence GIL-atomic reclaim flags drained on the inference thread, an 8-step deferred `mx.clear_cache` (immediate clearing races IOKit's async callbacks → kernel panic), jetsam-aware wired limits (a jetsammed process strands its wired allocation *until reboot* — 488 GiB stable vs 510 GiB crash-looping on a 512 GiB box), and deliberately-leaked "immortal" MLX threads when a compile-cache-clear symbol is missing (a thread-local C++ destructor would free Python objects without the GIL).

### 2.5 The API layer is client-battle-scarred

The SSE keep-alive encodes specific client bugs: keepalive chunks carry the stream's own `response_id` (openai-go latches the first id and silently drops mismatched chunks — losing tool calls) and `"role":"assistant"` (LangChain.js otherwise discards `tool_call_chunks` when merging). Non-streaming responses keep connections alive with *space characters* (JSON parsers skip leading whitespace) after a 2 s grace so real HTTP error codes still win. The Claude Code launcher force-injects `--disallowedTools LSP` because Claude Code splices the full LSP tool schema into the *system* region the moment a language server connects — one insertion that re-prefills the whole conversation on a prefix-caching server. "Context scaling" turned out to be the refusal of scaling: it sets Claude Code's max-context and auto-compact env vars to the same operator value so both sides stay on one scale.

---

## 3. The tiered KV cache: vLLM's block pool, re-imagined for unified memory + SSD

The flagship subsystem (~19K lines). Explicitly "vLLM v1 block pool, adapted to MLX" — but three structural inversions make it a different animal:

**Blocks hold no tensors.** In vLLM a block *is* GPU memory. Here `PagedCacheManager` is pure metadata (refcounts, SHA-256 chain hashes, an O(1) intrusive LRU); `max_blocks` is a hard-coded 100,000 because blocks cost nothing. The actual bytes live in a **content-addressed two-tier store**: a hot RAM tier of raw serialized bytes (never live `mx.array`s — pinned Metal allocations caused "IOGPUMemory underflow" kernel panics) and an SSD tier of one safetensors file per block (`<first-hex-char>/<sha256>.safetensors`, 256-token blocks). Copy-on-write degenerates to refcount bookkeeping — payloads are immutable and content-addressed. Unified memory makes this coherent: with no HBM/DRAM boundary, the interesting boundary moved to RAM/SSD.

**The store self-populates from disk.** On a prefix-lookup miss, if the SSD index has the block, a metadata entry is materialized inline mid-walk (`ref_count=0`) — restart warmth requires no rebuild pass. Startup scans read only safetensors *metadata*; LRU order is rebuilt from file mtimes. Multiple models share one cache directory: blocks incompatible with the running model go to a separate index and are left on disk ("stale for this prefix cache may still be valid for another model"), and a `SharedHotCacheBudget` runs one global LRU across all loaded models' hot tiers.

**Compatibility is a proof obligation.** Every block carries a `cache_signature` — canonical JSON of `{model, num_layers, block_size, layer_cache_types}` plus conditionally-stamped TurboQuant bits, payload layout, sidecar dtype (unstamped when irrelevant so old signatures stay byte-identical). The recurring rule: a block must *prove* compatibility; absence of proof is a rejected hit. A TurboQuant block that can't resolve its `(bits, seed)` is refused rather than dequantized at a guessed width ("plausible-but-wrong tensors — silent output corruption after a cache hit").

Write path details that matter: write-back by default (SSD index entry created lazily at hot-tier eviction, with a strict buffer→index→queue ordering so readers never see an index hit without bytes); the background writer never touches a Metal API (`mx.eval` + byte extraction happen on the inference thread; the writer emits the safetensors format by hand); atomic rename + parent-directory fsync; the write queue is sized in *bytes*, not entries (10% of RAM ÷ block bytes, clamped), and saturation degrades to a 1 s wait then an inline write rather than dropping the block. Restores are deliberately *synchronous on the inference thread* — a worker-thread `mx.load` deadlocked against Metal, and ~2 ms per block at 5 GB/s isn't worth the risk.

### 3.1 TurboQuant KV: quantize after prefill, not during

KV-cache quantization (per-token vector quantization with rotation + codebook, from mlx-vlm; fractional bits give keys `floor(bits)` and values `ceil(bits)`) is applied **once, after the prefill loop completes and after boundary snapshots are captured**. Earlier attempts quantized on the fly and corrupted prefill hidden states; this way "prefill hidden states stay exact and quantization error only enters at decode-time reads." SSD restores of homogeneous TurboQuant chains concatenate packed states and rebuild codecs deterministically from `(head_dim, bits, seed)` — **zero dequantization on the common path** (tests assert exact equality). Mixed dense/TQ chains are "healed" to dense with one grouped dequantize.

### 3.2 The hybrid-model state taxonomy — the genuinely new part

2026 models are not uniform-KV transformers: sliding windows (Gemma), Gated DeltaNet recurrent state (Qwen3.5/3.6/3.8), pooling caches (DeepSeek V4), sparse-attention index state (Qwen4 QSA). vLLM-style paged KV assumes per-token slices; oMLX built a handler registry with per-element axis metadata and **three storage strategies coexisting in one block chain**: per-block slices (normal KV), per-member splits (mixed `CacheList`s), and *boundary snapshots* (positionless recurrent state captured only at block-aligned token counts).

The consequences ripple everywhere: rotating layers store placeholders in interior blocks with the full window state only in the newest "tip" block, plus **walk-back truncation** at restore (drop trailing blocks until every non-sliceable layer has real state) and window padding (re-prefill `window/block_size` blocks so the sliding window refills); a supersede-on-extend pass strips the tip-before-last so multi-turn chats keep two heavy blocks instead of hundreds of MB per turn (without it "multi-turn cache hit collapses to 0% after ~10–20 turns"). GDN recurrent state can split into **durable SSD sidecars** keyed by a signature hash, optionally quantized with a randomized-Hadamard-transform int8 codec whose sign diagonal derives from SHA-256 (never the MLX RNG — reproducible across processes), with validity checks batched into one `mx.eval` (per-layer checks cost 19 ms per 48-layer state, ~19× the codec). DeepSeek pooling caches store append-only deltas with absolute row ranges so out-of-order chains are rejected rather than silently corrupted. Exact-hit handling refuses to reuse state at N and re-feed the last token on stateful models (it would shift recurrent state) — it trims to N−1 or falls back to full prefill.

There's even a **prefix-divergence probe**: the scheduler keeps the last 4 stored token sequences and, on every lookup, reports `unreused_common_prefix_tokens` — separating "the client's prompt drifted" (tool-schema change, re-rendered system prompt) from "the cache should have hit and didn't" (eviction, signature sweep, lost async store). Cache observability treated as a product surface.

---

## 4. Speculative decoding: three different beasts

### 4.1 Lightning MTP — native multi-token-prediction heads

No draft model: the target checkpoint's own MTP heads (Qwen3.5/3.6, DeepSeek-V4 — whose "DSpark" variant is a 3-stage parallel block drafter —, Nemotron-H, GLM-5.2, Gemma4, Inkling, Step-3.7, Qwen4-Exp VLM) draft up to 8 tokens; **one backbone forward verifies the chain**, acceptance is computed in-graph (greedy `cumprod` or batched Leviathan/Chen rejection sampling with residual distributions), and the whole cycle costs exactly **one host sync**. The next draft chain dispatches via `mx.async_eval` and resolves inside the next cycle's sync. Verify-shape Metal kernels ported from MTPLX fix the M=1-tuned qmm penalty (lm_head: 1.2 ms at M=1 → 4.3 ms at M=4 stock; kernels win 2.6–3.3×), and a fused Gemma4 verify attention (head_dim 512, multi-row) cuts L=4 from 3.8 ms to 1.35 ms per layer at 16K.

The distinctive part is treating **speculation depth as a control problem**. An adaptive controller scores each depth by `(1 + p₁ + p₁p₂ + …)/t(d)` with a token-domain acceptance EMA and a *wall-clock* cost EMA (τ = 400 ms — context growth, thermals and contention tracked in real time), staleness-directed bidirectional probes (a depth measured during slow post-prefill cycles "looks expensive forever"), hysteresis, and — after 16 losing decisions — **parking the request out of MTP entirely**, because even a depth-0 MTP cycle pays a host-sync loop tax the standard decoder pipelines away. The park handoff measures the real standard-step rate and stores the loop tax to seed future controllers. Honest numbers motivated this: Gemma4 code hits 1.89×, but story-writing measured **0.67×** — speculation can lose, and the engine is built to notice. Rollback is exact even on hybrid models: rotating caches get a verify-block undo log (a rotated ring buffer isn't trimmable — "phantom tokens progressively corrupted output"), GDN gets state snapshots, Nemotron-H replays the accepted prefix through a pristine forward. Head-cache priming during prefill takes acceptance from 0.26 to 0.90 on depth-1 code.

MTP is per-request and single-row by default: the row-wise batch mode runs one backbone forward per row per cycle, and measured aggregate at batch 4 was 52.5 tok/s vs 86.5 standard *despite 83–93% acceptance* — so batched serving wins with plain decode, and MTP shines at concurrency 1–2. Late joins into an active MTP batch are locked out (a mid-cycle merge would re-prefill outside the guarded path) except via an exact handoff when the draft queue is drained. Measured E2E: Step-3.7-Flash 1.22×→1.30× (growing with context, 4K→32K); release notes claim 2.33–2.62× on 0.6.3-era models.

### 4.2 DFlash — block-diffusion drafting

An external draft model generates **16-token blocks in parallel by diffusion denoising**; the target verifies the block in one forward; greedy longest-prefix-match commit (lossless at temp 0). Paper claims 4.9× on H200; oMLX integrates the dflash-mlx fork for Qwen, Gemma4, Laguna, and Muse-Glimmer targets. It is single-request, bypasses the paged cache (own snapshot cache with SSD spill), and above a context threshold the engine *evicts its own draft weights* and delegates to a nested BatchedEngine. Half the integration code is lifecycle defense: dflash patches attention classes *at class level*, and those hooks outlive the engine — a resident DFlash engine once crashed a concurrent text engine's decode, and an MTP load once silently stripped DFlash's hooks ("generations degenerate into repetition loops"). The fix is a dynamic guard that routes per-call by cache type, with MTP installing itself as the guard's *fallback base*. Composition of monkey patches is a real architecture problem here.

### 4.3 SpecPrefill — speculative *prefill* (sparse prompt processing)

Not speculative decoding despite the name (arXiv:2502.02789): a small draft model prefills the prompt, 8 lookahead decode steps capture post-RoPE queries, importance = smoothed max-over-heads attention from lookahead queries to prompt keys; keep the top 20% of 32-token chunks **plus a mandatory 512-token tail window** (chat templates end in structural markers; losing them leaves the target mid-structure emitting a bare `</tool_response>`). The target then prefills only selected tokens with manual RoPE at their *original* positions, plus an offset-shim so decode continues at the true position. Activates above 8192 uncached tokens; claimed ~3× TTFT at keep 20%. Fails soft — any exception falls back to full prefill.

---

## 5. The heterogeneous centerpiece: GPU + both ANEs + CPU on one matmul

This is the answer to the companion doc's question, in running code. Experimental, default-off, Qwen3.5/3.6/3.8 dense family only — and remarkable.

### 5.1 What runs where

**Prefill only; decode is untouched** (bandwidth-bound — exactly the roofline logic from the companion doc). Two projection sites are split by *output channel*:

- **MLP gate+up**: default 53% of channels on ANE (26.5% per die on M3 Ultra, pinned to physical ANE instances 1 and 2), 47% on GPU; optionally more slices on CPU (fp16 via Accelerate/BNNS). A native kernel merges ANE0+ANE1+CPU+GPU partial outputs *and applies SwiGLU* in one pass, never materializing the full gate/up result.
- **GDN z+qkv input projection — z only.** The precision policy: the ANE path is approximate (per-output-channel INT8 requant), and putting *recurrent* qkv rows on it deterministically failed ordering/format tasks at 32K (error accumulates through the recurrence). Only the token-local z gate goes to ANE — capped at the model's z boundary (37.5%) no matter what fraction is requested. **Approximate compute units get only token-local math; recurrent state stays exact.** That placement rule generalizes.

### 5.2 How: private APIs, hand-written MIL, zero-copy surfaces

No Core ML, no MPSGraph. The code dlopens the **private `AppleNeuralEngine.framework`** and drives `_ANEInMemoryModel`/`_ANERequest` via `objc_msgSend`. The "model" is hand-emitted **MIL text** (Core ML's IR built as a string): `constexpr_blockwise_shift_scale(int8, fp16 scale) → conv` expressing the linear as a 1×1 conv over `[1, K, 1, S]`; weights ship as raw blobs with hand-rolled chunk headers. Die pinning via `kANEFAneInstanceHint`. I/O is zero-copy: each program owns IOSurfaces mapped simultaneously as ANE objects and as `MTLBuffer`s — a Metal kernel packs activations directly into the surface the ANE reads, and the merge kernel reads ANE output surfaces directly. No host round-trip.

**Procedure banks** beat a driver limit: the runtime accepts ~121 resident model handles, so per-layer programs capped at ~60 dual MLPs; packing every layer-slice as a *procedure* inside one program per die fits 64 MLP + 48 GDN = 112 procedures in two programs. Each bank maps its weight blob into the die's ~4 GiB device address window (~3.75 GiB for the 27B layout — fits per-die on M3 Ultra; fails with `0x20004` on single-die chips, then retries a split ladder: monolithic → two near-halves → progressively smaller → per-layer). The monolithic-first choice is a values statement: split banks measured ~1% *faster*, but occasionally diverged at greedy ties, while the monolithic bank was bit-stable across five reruns — **bit-stability outranked 1% throughput**. Eager compilation takes 37–40 s at startup; an opt-in compile cache reuses *Apple's own* compiled-program cache (keyed by descriptor hash) with a flock for cross-process safety, turning compile into load.

### 5.3 True concurrency, measured

Per accelerated layer: the GPU pack kernel signals a `MTLSharedEvent` ticket → the GPU quantized suffix is committed async → two detached threads fire one ANE evaluation per die → the CPU fp16 branch runs on the calling thread → join, merge. Profiling shows each ANE executing requests **38.8% of total body time** with 29–37 µs launch delay; the dominant downtime is waiting for the GPU to produce the next layer's input. The negative results are documented with the same care: the driver does *not* stripe one procedure across both dies (unpinned single procedure 39.5% slower than two pinned), persistent worker threads regressed, the async completion-handler path regressed, and moving the blocking input-pack wait anywhere else destroyed device overlap entirely (the Metal-callback version came out 5.6% *slower than GPU-only*).

Results on M3 Ultra, Qwen3.8-27B (oQ4e):

| Configuration | Prompt throughput | vs GPU-only |
|---|---:|---:|
| GPU only | 334.9 tok/s | 1.000× |
| Dual ANE (53% MLP / 50%→z GDN) + GPU | 454.3 tok/s | **1.356×** |
| Five-way tuned ANE+CPU+GPU (45/45/14/20/13) | 517.9 tok/s | **+45.8%** |

End-to-end: +17.8% prompt processing at 16K, +18.9% at 32K, TTFT −15%, **decode unchanged**. Fidelity: logit cosine 0.9985–0.99999, top-1 unchanged; 16K/32K output hashes matched the GPU path exactly. Costs: +4.15 GB peak memory, eager load 3.35 s → 27–40 s.

### 5.4 The tuner, and graceful retreat

Optimal splits are machine- and model-specific, so a built-in **five-way split tuner** (MLP→ANE, MLP→CPU, down→CPU, GDN→ANE, GDN-qkv→CPU) packs candidate widths from one real MLP and one real GDN layer into a temporary procedure bank, measures production paths, rebalances fractions once by measured branch *rates*, then compiles only the predicted winner and verifies end-to-end — avoiding a 5-D full-model grid while keeping end-to-end throughput as the criterion. It rejects any candidate whose profiler shows zero ANE operations (otherwise GPU-only throughput gets reported as an ANE result), computes the tail-padding crossover `floor(S·G/H)+1` from its own measurements, and on machines without the private runtime returns a *completed* "recommendation: disabled" — "on such a machine that IS the tuning answer, not an error." An isolated fixed-split sweep of the CPU-GDN fraction picked the wrong answer versus joint tuning — documented as the reason the tuner is joint.

Retreat paths are everywhere: every wait is bounded (30 s; a wedged driver latches a per-module failure flag that routes that layer back to GPU permanently); a memory-pressure ladder can **shed the ANE banks** (~13 GB on a 27B) to buy prefill headroom, with escalation logic that skips a reclaim rung proven to "succeed" marginally forever; a bank-compile headroom gate refuses to even attempt compilation above 70% system footprint (the driver's async release once climbed a retry ladder to jetsam). The whole feature is triple-gated and default-off, and the doc says plainly: private APIs "can stop working after a macOS update."

### 5.5 NAX: the *other* neural hardware (M5)

NAX = the M5-generation GPU's per-core neural accelerators (tensor units) — inside the GPU, unrelated to the ANE. oMLX ships NAX variants of its quantized-matmul kernels (needs macOS 26.2 SDK) and detects support by *byte-scanning the installed mlx.metallib* for the NAX kernel symbol (the bundled Sequoia wheel is built without NAX, so hardware detection alone would mis-route). On NAX machines the hybrid's GPU suffix runs on tensor units, and the tuner's ANE fraction grid drops from ~[0.40–0.60] to [0.15–0.53] — when the GPU gets its own matrix engines, the profitable ANE share shrinks. A separate patch works around two *defective* NAX gather-QMM kernels in mlx 0.32.0 via a self-arming canary: the first matching call compares against an fp32 reference and only intervenes if the corruption actually manifests on this machine.

---

## 6. Custom kernels and oQ quantization

**Kernels.** Five AOT Metal packages (built only with full Xcode; every one degrades silently to generic paths, with an ABI probe against the mlx wheel) plus ~25 JIT `mx.fast.metal_kernel` patches. Highlights: GLM-5.2/DeepSeek DSA fused sparse-attention prefill — **~30× (845 vs ~29 tok/s on M3 Ultra)** over the generic fallback, the single largest number in the repo; Qwen4 QSA sparse attention (radix top-512 block selection returning only indices, exact sparse GQA with in-kernel causal tail); a steel-template attention instantiated at head_dim 256 that stock MLX doesn't cover; MoE gather-QMM for mxfp4; a GDN prefill kernel ~2× over stock mlx-lm (rel. err 5e-8); a GDN "prework" fusion collapsing ~10 dispatches per layer to 1; and a port of a *closed, unmerged* upstream MLX PR for decode SDPA. A correctness ledger for one kernel-port campaign records a hard bit-exactness bar with four documented rejections — including a fusion that was bit-exact on CI but ULP-divergent on M3 Ultra *feeding argpartition expert selection*, where one ULP flips which experts run. Kept eager.

**oQ ("oMLX Universal Dynamic Quantization")** is not a new format — output is stock mlx-lm-loadable affine safetensors. The value is *allocation*: per-tensor bit widths driven by measured sensitivity (`MSE/mean(out²)` per layer, normalized so late layers aren't blamed for residual accumulation), tiered boosts under a bits-per-weight budget, and hard role protections (MoE routers fp16, shared experts ≥8-bit, SSM/GDN dynamics params unquantized, GLM DSA indexers pinned q8 as a format invariant). Routed experts stay at base bits "not by rule but because their byte cost loses in the budget optimization." oQ+ adds GPTQ with a MoE trick — all experts in a layer share one Hessian, so batched compensation runs **15× faster (90 min → 6 min for 256 experts × 40 layers) with identical results**. oQe adds an activation-energy importance matrix (2,679 calibration samples across 10 domains, with a required 5th-percentile expert-coverage floor) feeding a 15-candidate importance-weighted clipping search — still emitting the exact stock affine layout. Self-reported results, Qwen3.5-35B-A3B at 2-bit: **MMLU 14.0% → 64.0%, HumanEval 0.0% → 78.0%** vs naive mlx-lm quantization; 4-bit is near parity. The ANE path consumes oQ output directly (affine 4/5/6/8, group 64/128) — mixed q4/q5 checkpoints work *only because* the z-boundary policy leaves a homogeneous GPU suffix.

---

## 7. The cluster: pipeline across unequal Macs, and Macs + CUDA

Source-build experimental. Pipeline parallelism over MLX's distributed backends — TCP Ring, or **JACCL, MLX's Thunderbolt-RDMA collective backend** (the Mac NCCL). Measured on the same cable: **28.6 tok/s over RDMA vs 6.6 over TCP Ring — 4.3× from the transport alone**. Rank 0 owns the *late* layers plus the HTTP coordinator (activations flow high rank → 0). Tensor parallel exists but is gated to all-nodes-or-nothing (mlx-lm `shard()` implementations only operate on complete models), with strategy adapters derived from Exo and an AST inspection proving a model's native `shard()` is layer-local before trusting it.

The planner is a DP over contiguous layer cuts using **two size vectors** — weight bytes and *resident* bytes (weights + that stage's KV at the planned context) — with the invariant "a performance pass may move weights toward a faster Mac, but never so far that the stage cannot hold its KV." Capacity is the live `ProcessMemoryEnforcer` ceiling, not installed RAM (a 128 GiB MacBook addresses 107.5 GiB; RAM-based plans were "~20 GiB optimistic," and one 56 GiB placement took a MacBook down — hence a 0.50 workstation admission fraction and a load-time memory watchdog). Concrete capacity win: MiniMax-M3-4bit (225 GiB) across a 128 GiB MacBook + 256 GiB Studio reaches **971K tokens of context balanced — vs 644K when the split is pinned badly, vs ~1K on the Studio alone**. Measured per-rank compute/link profiles (all-or-nothing; partial measurements fall back to the memory-only planner) feed profile-weighted stage-time predictions (interactive weights decode 0.8; throughput weights prefill 0.75).

Operational care stands out: per-rank prompt caches are forced to size 1 because independent per-rank eviction desynchronizes token offsets and **"blocks forever in the first unmatched collective"** (long-prefix reuse is recovered via per-rank SSD snapshots keyed identically); an experimental token-only output path (rank 0 samples, an all-sum broadcasts only token IDs; workers skip the final all-gather *and* the vocab projection) is gated by literally `inspect.getsource`-ing the model's forward for exactly one all_gather and one send; and the TP correctness probe compares **raw greedy token IDs** across 1 vs N nodes — "sharding bugs do not crash… every one still emits fluent text; detokenization can hide a divergence."

The **heterogeneous design doc** (2026-08-10) extends the pipeline across *Apple Silicon + NVIDIA CUDA* machines — MLX 0.32 runs on CUDA, so a Mac Studio and DGX Sparks can join one Ring as ranks of one model (e.g., 256 GB M3 Ultra + 5 × 128 GB DGX Spark = 896 GB installed, honestly reported as much less usable). Its rules are the companion doc's economics restated: this is a *capacity* pool, not unified memory; every rank participates in both prefill and decode (no "CUDA does prefill only" — those ranks own layers needed for every token); prefill/decode disaggregation is a secondary optimization requiring two complete placements; and the planner "must not pretend the DGX-to-DGX 200 Gb/s fabric makes the 10 GbE Mac edge faster." A planned "supernode" gateway would collapse ConnectX-linked CUDA pairs into one logical rank (outer Ring, inner NCCL). All speculative/TurboQuant/grammar features are rejected — not ignored — on the cluster.

---

## 8. What oMLX settles about "using all the hardware"

Point by point against the companion doc's framework:

1. **Decode stays on the bandwidth-dominant unit — confirmed by construction.** Nobody who measured harder than anyone else chose to split decode. ANE offload, CPU branches, SpecPrefill — all prefill-side. Decode gains come from speculation (raising arithmetic intensity), not from more silicon.
2. **Prefill is where heterogeneous compute pays, and it genuinely pays.** +36% body throughput dual-ANE, +46% with CPU sharing, +19% end-to-end prompt processing at 32K — on top of an already kernel-optimized GPU baseline. The "extra TOPS help the compute-bound phase" prediction holds with real numbers.
3. **The winning partitions are communication-light with rare sync points** — output-channel slices merged once per layer via shared IOSurfaces; phase splits; pipeline stages. The one attempt at fine-grained async coordination (completion handlers, callback-launched ANE) *destroyed* the overlap.
4. **Precision-aware placement is the new rule the reference doc lacked**: approximate accelerators (INT8 ANE) may take token-local math but never recurrent state. This is a placement constraint that didn't exist in the dense-transformer era and will matter for every hybrid-architecture model.
5. **Static ratios don't transfer — measure on the machine.** The five-way tuner, the NAX-shifted fraction grid, the per-machine crossover threshold, and the cluster's measured rebalancing all say the same thing: heterogeneous splits are a *runtime calibration problem*, not a design-time constant. (AMD's hybrid flow ships pre-tuned ratios; oMLX 0.6.2's release note explicitly calls out eliminating "reliance on pre-tuned ratios from other machines.")
6. **"Use all the units" is bounded by the shared-bandwidth ceiling, as predicted** — the measured ANE duty cycle is 38.8% per die, with the gap being GPU-produced input readiness; the CPU branch helps only in joint-tuned small fractions and *hurts* when sized naively. The units multiply *compute*, not bandwidth, so they help exactly the phase that is compute-bound and exactly as much as the memory system allows.
7. **One correction to the reference doc**: "third-party stacks use GPU (Metal) and CPU only; the ANE is reached only via Core ML" is now outdated. oMLX drives the ANE through private APIs, in production code (default-off, but shipped, tuned, and validated), and Apple's own runtime cache is being reused for it. The ANE-for-LLMs door is open — at the price of undocumented-API risk that oMLX documents candidly.

## 9. Assessment

**Worth stealing** (for anyone building inference systems, on any platform):

- The *external prefill* inversion — keep the token loop simple and hang every value-add (paged cache, admission, fairness, offload) off a prefill you fully control.
- Time-domain fairness scheduling when preemption doesn't exist (500 ms stall target ÷ measured tok/s), and admission control priced at the throttled regime with one-line-diagnosable rejections.
- The cache-signature "prove compatibility or reject the hit" contract, and content-addressed metadata-only paging with lazy cold registration — the right shape for any persistent, multi-model, tiered KV store.
- Speculation depth as a measured control loop with a park state — versus the fixed-k speculation most engines ship.
- The recurrent-safe placement rule for approximate accelerators.
- Correctness culture: bit-stability chosen over 1% wins; raw-token-id cross-node parity probes; self-arming canaries for driver bugs; negative results documented next to the winning ones; constants carrying their incident report (a 13K-line scheduler stays navigable *because* every threshold cites the A/B that set it).

**Structural risks**, visible in the same reading: the ANE stack rests on undocumented private APIs (default-off and latched, but one macOS update from dead); the whole engine is a deep monkey-patch stack over exact-pinned mlx/mlx-lm/mlx-vlm commits (the DFlash-vs-MTP class-patch collisions show how sharp composition gets — patch isolation is a recurring bug source and now has its own guard architecture); everything runs in one process where vLLM V1 chose process isolation (GIL contention is managed with decode-burst budgets rather than removed); and there's visible drift (a dead priority policy, an unused adapter abstraction, doc/code divergence in oQ's 8-bit mode). None of these look fatal; all of them are the price of one person's stack moving at ~400 commits a month.

The bottom line for this research repo: oMLX demonstrates that a 2026 on-device engine can carry datacenter-grade serving machinery (continuous batching, paged+tiered KV, structured output, speculative decoding) *and* exploit Apple's full heterogeneous silicon — GPU, both ANE dies, CPU, and M5 tensor units — where the physics says it can pay: compute-bound prefill, coarse-grained partitions, measured per-machine splits, exact state kept on exact hardware. That is the state of the art for "use all of the hardware," and its boundaries are exactly where the roofline said they would be.

---

## 10. Pointers

- Repo: https://github.com/jundot/omlx (v0.6.4, commit `e008a66`) · https://omlx.ai
- Key in-repo reading: `docs/experimental/qwen35_ane_prefill.md` (the ANE split), `docs/oQ_Quantization.md`, `docs/distributed-cluster.md`, `docs/heterogeneous-cluster.md`, `docs/experimental/dflash_mlx_integration.md`, `docs/laguna-mlxfast-port-correctness.md`
- Code entry points: `omlx/scheduler.py` (batching/fairness/admission), `omlx/cache/prefix_cache.py` + `omlx/cache/paged_ssd_cache.py` (tiered cache), `omlx/patches/qwen35_ane_prefill.py` + `omlx/custom_kernels/qwen35_prefill/csrc/qwen35_ane.mm` (ANE), `omlx/patches/mlx_lm_mtp/` (Lightning MTP), `omlx/cluster/planner.py`
- Upstream credits it builds on: MLX/mlx-lm (Apple), mlx-vlm, vllm-mlx (origin), MTPLX (verify kernels), dflash-mlx, Exo (TP strategies)
- Companion: [inference-engine-architecture.md](inference-engine-architecture.md) — the general architecture/roofline framework this study tests against reality
