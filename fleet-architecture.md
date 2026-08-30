# A Heterogeneous Inference Fleet — M5 Macs + CPU Servers + DGX Sparks

*Research notes, August 2026. The scale-out companion to [omlx-code-study.md](omlx-code-study.md) and [inference-engine-architecture.md](inference-engine-architecture.md): a serving architecture for agentic workloads on the hardware actually on hand — tens of M5-class Macs, many commodity CPU servers, and a few NVIDIA DGX Sparks.*

---

## 1. Hardware truths the design is built on

| Silicon | Strong | Weak | Therefore |
|---|---|---|---|
| M5 Macs (NAX GPUs) | interactive decode per $ and per W; strong prefill (tensor units); quiet/cheap | ~150 GB/s (base) memory BW caps dense-model decode; 10 GbE NICs | **turn bandwidth into interactive tokens** |
| CPU servers (Xeon/EPYC) | huge cheap RAM, NVMe, 300–580 GB/s DDR, AMX ~20 TFLOPS, existing ops | wrong for interactive main-model decode | **turn RAM into cache hit rate** |
| DGX Spark (GB10) | massive FP4/FP8 compute; 128 GB unified; ConnectX-7 200 Gb/s (~25 GB/s) | only 273 GB/s → decode-starved; ~$4K each | **turn FP4 compute into prefill and capacity** |

Workload truth: agentic traffic is multi-turn, prefix-heavy (each turn re-sends the transcript plus 5–50K tokens of tool results), prefill-dominated once decode is batched, and extremely cache-sensitive.

## 2. Design invariants

1. **Only state crosses machines — never per-token computation.** KV blocks move; activations, per-layer expert calls, and logits never do (the serial decode chain dies on any per-token RPC; KTransformers-style expert offload works only inside one box).
2. **Locality is a preference, not a correctness condition.** The block store is the source of truth; any Mac can pick up any session in ~1 s. Router affinity is an optimization hint.
3. **Statically separate prefill/decode only across *asymmetric* silicon.** Identical boxes stay mixed-phase and pooled (statistical multiplexing beats fixed partitions at tens-of-boxes scale); asymmetric silicon (Spark↔Mac, Spark↔Spark) separates by comparative advantage.
4. **One engine family end-to-end where KV must be portable.** oMLX's content-addressed blocks carry a signature keyed on model + layout, not device; oMLX-on-Metal and oMLX-on-CUDA can interchange KV through the store. Mixed stacks (vLLM↔oMLX) share nothing but bytes at rest.
5. **Every threshold is measured, not assumed** — per machine and per model, in the spirit of oMLX's ANE split tuner and its tail-padding crossover `floor(S·G/H)+1`.

## 3. The architecture

```
                                     agent clients
                      (Claude Code · OpenClaw · Codex · CI bots)
                                           │ OpenAI/Anthropic API
                                           ▼
  ╔═ CONTROL PLANE (runs on CPU servers) ═════════════════════════════════════╗
  ║  ROUTER — per-turn placement:                                             ║
  ║    suffix < threshold (~4–8K) ──► session's Mac: local prefill + decode   ║
  ║    suffix ≥ threshold ──────────► Spark prefill → store → Mac decode      ║
  ║    one-shot / batch job ────────► Spark P/D cell (never touches Macs)     ║
  ║    frontier model ──────────────► Spark island                            ║
  ║  + block index · health · metrics · GC     (affinity = hint, not truth)   ║
  ╚════════╤═════════════════════════╤═══════════════════════╤════════════════╝
           │                         │                       │
           ▼                         ▼                       ▼
  ┌─ MAC PLANE ────────────┐ ┌─ SPARK PLANE ──────────┐ ┌─ CPU-SERVER PLANE ──────┐
  │ interactive decode +   │ │ compute & capacity     │ │ state & auxiliaries     │
  │ short-turn prefill     │ │                        │ │                         │
  │ ┌────┐ ┌────┐  ┌────┐  │ │ PREFILL SERVICE        │ │ KV BLOCK STORE          │
  │ │M5#1│ │M5#2│ …│M5#N│  │ │  oMLX-on-CUDA: long    │ │  RAM hot / NVMe cold    │
  │ │oMLX│ │oMLX│  │oMLX│  │ │  suffixes → blocks     │ │  content-addressed      │
  │ └─┬──┘ └─┬──┘  └─┬──┘  │ │  streamed into store   │ │  dedup'd · replicated   │
  │   │      │       │     │ │                        │ │  = SOURCE OF TRUTH      │
  │   local SSD = L1 cache │ │ P/D CELLS              │ │                         │
  │   mixed-phase boxes,   │ │  Spark ─200Gb─ Spark   │ │ AUX MODELS              │
  │   no static split      │ │  Dynamo/NIXL, ~80 ms   │ │  embeddings · rerank ·  │
  │   (chunked prefill +   │ │  KV handoff            │ │  draft/guard · batch    │
  │   decode-debt on-box)  │ │                        │ │  MoE on AMX             │
  └────┼──────┼───────┼────┘ │ ISLAND (2–4 Sparks)    │ │                         │
       │      │       │      │  frontier MoE, 512 GB, │ │ hosts the control plane │
       │  write-through /    │  671B@4bit ~10–15 t/s  │ └───────────▲─────────────┘
       │  fetch-on-miss      │                        │             │
       │  (10–25 GbE)        │ LAB: quantization ·    │             │
       ▼      ▼       ▼      │  EAGLE heads · evals   │             │
       └──────┴───────┴──────┴───────────┬────────────┘             │
                                         └────── blocks ────────────┘

           ONE INTERCHANGE: content-addressed KV blocks (oMLX format)
           same signature on Metal and CUDA → KV is portable across vendors

   NEVER on a wire: per-token activations · per-layer expert calls · logits
   ALWAYS on wires: requests/SSE (Ethernet) · packed KV blocks (streamed)
```

### A turn's lifecycle (the routing rule in action)

```
turn N+1 arrives: 20K tokens of tool results, session lives on Mac#7
 1  router: suffix 20K ≥ threshold and a Spark is free → assign Spark#2
 2  Spark#2: prefix already warm (turn N was written through to the store);
    prefills the suffix at FP4 rate, streaming each 2048-token block into
    the store as its chunk completes (~250 MB/s KV production — fits 10GbE)
 3  Mac#7 streams the blocks in parallel with Spark's compute
 4  Mac#7 decodes the reply inside its continuous batch; SSE to the client
 5  reply KV (small) writes through to the store — session complete everywhere
 TTFT ≈ max(Spark prefill, Mac fetch) ≈ 4–9 s   vs  ~10–20 s Mac-local
```

Short turns (suffix under the threshold) skip steps 1–3: the session's Mac hits its local SSD prefix cache and prefills the small suffix itself — the hop isn't worth it.

## 4. Why each piece is where it is

**Macs stay mixed-phase (no P/D split inside the pool).** oMLX already bounds on-box prefill/decode interference (chunked prefill sized by a 500 ms stall target, decode-debt fairness). Splitting identical boxes into fixed pools buys interference isolation the boxes already have, and pays for it with queueing rigidity — a prefill burst queues on the P subset while D boxes idle. Pooled identical servers absorb variance; partitioned ones don't. Revisit only on *measured* sustained >~70% utilization with ITL p99 violations — and then it's a reversible drain-and-relabel, not an architecture change.

**Sparks do the heavy prefill and the big models.** Spark's compute:bandwidth ratio (huge FP4 throughput vs 273 GB/s) is exactly prefill's profile, and its ConnectX-7 is the best NIC in the fleet. Spark↔Spark ConnectX pairs make textbook Dynamo-style P/D cells (~80 ms KV handoff for a 32K-token context — the fabric condition datacenter disaggregation assumes). 2–4 Sparks pooled (256–512 GB) serve frontier-class MoE as a slow-but-present capacity island. One Spark is always the lab bench (quantization runs, draft-head training, evals) — tooling that doesn't exist on Macs.

**CPU servers hold the state.** The block store is the Mooncake role on commodity hardware: RAM-hot/NVMe-cold, content-addressed, deduplicated fleet-wide (system prompts and tool schemas stored once, hit by every replica), replicated, restart-proof. It converts router stickiness from a correctness requirement into a latency preference: a 32K-token prefix is ~2 GB TurboQuant-packed → ~1.6 s at 10 GbE / ~0.65 s at 25 GbE versus ~70 s of re-prefill (40–100×). The same servers absorb the auxiliary model zoo (embeddings, rerankers, draft/guard models on AMX) so those stop stealing Mac bandwidth, plus the control plane. The blocks are immutable content-addressed objects with a small metadata index — a natural fit for existing distributed-KV/object-store infrastructure and operational practice.

**P/D separation is a per-turn routing decision, not a topology.** Once the store exists, "should we disaggregate?" reduces to a threshold: route a turn's prefill to a Spark when `Spark_prefill + fetch < Mac_local_prefill`, i.e. when the suffix is long enough to fund the hop. The threshold is measured per model/machine pair (same method as oMLX's tuner crossover). Statically partitioning happens only where silicon is asymmetric.

## 5. Napkin numbers (from the oMLX study + spec sheets; validate on real hardware)

| Quantity | Value |
|---|---|
| Packed KV, 32K tokens, 27B-class GQA, TurboQuant | ~2 GB (fp16 ~8 GB); GDN hybrids much less |
| Block fetch, 32K prefix | ~1.6 s @10 GbE · ~0.65 s @25 GbE · ~80 ms @ConnectX · vs ~70 s re-prefill |
| Streamed P→D transfer exposure (chunk-overlapped) | ≈ last chunk + final recurrent state ≈ 25–100 ms |
| Spark-assisted long-turn TTFT | ≈ 2× better than Mac-local (20K-token suffix) |
| Decode ceilings | base M5: A3B MoE ~85 t/s, dense 27B ~10 t/s · Spark: A3B ~160 t/s, dense 27B ~18 t/s |
| Reply-KV backfill (0.5–2K generated tokens) | 30–130 MB — off the critical path |
| Frontier island (4 Sparks, 671B @4bit) | pipeline-limited ~10–15 tok/s |

## 6. What must be built (oMLX + ecosystem gaps)

1. **The router** — stateless; rendezvous-hash affinity + admission-aware spill + the per-turn P/D threshold. Small.
2. **The block store service** — get/put by chain hash over RAM/NVMe with replication and GC; oMLX's safetensors block format and signature contract are the wire format, already defined. Medium.
3. **Write-through/fetch hooks in oMLX** — its SSD tier gains a remote tier (the cache code is already structured as pluggable tiers). Medium-small.
4. **oMLX-on-CUDA for the Spark prefill service** — MLX 0.32 supports CUDA; oMLX's heterogeneous-cluster work is partway there. Validation required: **CUDA→Metal KV token-identity probes** (ULP-level kernel divergence is almost certainly fine for cache reuse — smaller than TurboQuant's deliberate error — but prove it with raw-token-ID comparisons, the way oMLX's `tp_identity_probe` does).
5. **Dashboards for the three deciding metrics**: fraction of Mac wall-time in prefill; ITL p99 during prefill bursts; per-turn suffix-length distribution. These numbers, not taste, decide whether Spark-assist and any future Mac-pool split are worth it.

## 7. Decision rules going forward

- Mac prefill share < ~20–30% of wall time (cache doing its job) → skip Spark-assist for that model; Sparks focus on batch/frontier.
- Sustained hot fleet + long prompts + ITL p99 misses → carve a decode-only Mac group (reversible).
- Fleet gains a real fabric (25 GbE+ on Macs via TB adapters, or more ConnectX boxes) → thresholds drop; more traffic flows through the store; the architecture doesn't change, only the constants.
- The one-line summary: **Macs turn bandwidth into tokens, Sparks turn FP4 compute into prefill and capacity, CPU servers turn RAM into hit rate — and the content-addressed block store is the only place all three meet.**
