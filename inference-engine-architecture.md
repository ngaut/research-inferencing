# Inference Engine Architecture — and Whether Engines Can Use *All* the Hardware (CPU, GPU, NPU/ANE) for Max Throughput

*Research notes, August 2026.*

This document answers two questions:

1. **What is the architecture of modern inference engines?** (vLLM, SGLang, TensorRT-LLM, llama.cpp, MLX, Core ML, ONNX Runtime, OpenVINO, Ryzen AI, …)
2. **Can they use all of the hardware — CPU, GPU, ANE/NPU — simultaneously to get maximum throughput?**

**TL;DR for question 2:** For a single dense model's forward pass, engines deliberately do *not* spread compute across every unit, because token generation is bound by **memory bandwidth**, not compute, and heterogeneous units either share one memory system (SoCs) or are separated by a slow bus (PCIe). "Use everything" adds synchronization and contention, not tokens. Instead, state-of-the-art engines (a) saturate the *bottleneck resource* of the fastest unit and (b) give the other units **complementary roles** — scheduling, tokenization, KV-cache tiering, prefill, draft models, other pipeline stages. There is, however, a growing class of real heterogeneous-compute wins: CPU+GPU MoE hybrids (KTransformers), NPU-prefill + iGPU-decode (AMD Ryzen AI hybrid), stage splits (whisper.cpp's ANE encoder), and datacenter prefill/decode disaggregation (NVIDIA Dynamo). Details below.

---

## Contents

1. [The two regimes: datacenter serving vs. on-device](#1-the-two-regimes)
2. [The physics that shapes every engine](#2-the-physics-that-shapes-every-engine)
3. [Reference architecture of a datacenter serving engine](#3-reference-architecture-datacenter-serving-engines)
4. [Architecture of on-device engines](#4-architecture-of-on-device-engines)
5. [Can an engine use ALL the hardware at once?](#5-can-an-engine-use-all-the-hardware-at-once)
6. [Where heterogeneous execution genuinely wins today](#6-where-heterogeneous-execution-genuinely-wins-today)
7. [Practical guidance per platform](#7-practical-guidance-max-throughput-per-platform)
8. [Future directions](#8-future-directions)
9. [References](#9-references)

---

## 1. The two regimes

Inference engines split into two families with different goals, which drives different architectures:

| | Datacenter serving | On-device / edge |
|---|---|---|
| Examples | vLLM, SGLang, TensorRT-LLM, DeepSpeed-FastGen, TGI, NVIDIA Dynamo (orchestration) | llama.cpp/GGML, Ollama, MLX, Core ML, ONNX Runtime, OpenVINO, ExecuTorch, LiteRT, MLC-LLM, Ryzen AI SW, Qualcomm Genie/QNN |
| Objective | Max **throughput/$** under latency SLOs (TTFT, inter-token latency), thousands of concurrent requests | Min **latency and energy** for 1–few streams, fit in limited RAM, thermals |
| Hardware | Fleets of homogeneous GPUs (H100/B200/MI300X/TPU), NVLink/RDMA fabrics | One SoC with heterogeneous units: CPU + iGPU/dGPU + NPU (ANE, XDNA, Hexagon, Intel NPU) sharing DRAM |
| Key techniques | Continuous batching, paged KV cache, prefix caching, chunked prefill, disaggregation, tensor/expert parallelism, speculative decoding | Aggressive quantization (4-bit), layer/expert offloading, static-graph compilation for NPUs, unified memory |

The "can it use all hardware" question has a different answer in each regime — in the datacenter the units are homogeneous GPUs and the question becomes "is every GPU's bandwidth and compute saturated?"; on-device the units are heterogeneous and share one DRAM.

---

## 2. The physics that shapes every engine

### 2.1 Prefill vs. decode: two different workloads in one request

A transformer request has two phases with opposite hardware profiles:

- **Prefill** (process the prompt): all prompt tokens computed in parallel → big matrix–matrix multiplies → **compute-bound**. Scales with TFLOPS.
- **Decode** (generate tokens one at a time): each step is matrix–*vector* work that must re-read **every weight** (plus the growing KV cache) to produce one token → **memory-bandwidth-bound**. Scales with GB/s.

### 2.2 The roofline: why decode wastes compute

Arithmetic intensity (AI) = FLOPs per byte moved. A device's "ridge point" is FLOPS ÷ bandwidth; below it you're bandwidth-bound.

- H100 SXM: ~989 TFLOPS dense BF16 ÷ 3.35 TB/s ≈ **ridge at ~295 FLOPs/byte**.
- Batch-1 FP16 decode: ~2 FLOPs per parameter per token, 2 bytes per parameter → **AI ≈ 1 FLOP/byte**.

So single-stream decode uses **well under 1% of an H100's compute**. This single fact explains most of modern engine architecture:

- **Batching** raises AI: weights are read once per step and amortized over B sequences → throughput grows almost linearly with batch size until compute or KV bandwidth saturates. Hence *continuous batching* is the #1 throughput feature.
- **KV cache** is each sequence's private state; its reads scale with batch size and context length, so at high batch/long context KV bandwidth becomes the new wall → GQA/MLA attention, KV quantization (FP8), paged layouts, FlashAttention.
- **Quantization** (8/4-bit weights, FP8/FP4) shrinks the bytes per token → directly raises the decode ceiling.
- **Speculative decoding** converts bandwidth into parallelism: a cheap draft proposes k tokens, the big model verifies them in one (compute-cheap, bandwidth-amortized) pass.

### 2.3 Bandwidth ceilings, not TOPS, decide tokens/sec

Max decode speed ≈ usable bandwidth ÷ bytes touched per token. Rough single-stream ceilings:

| Hardware | Bandwidth | 8B @ 4-bit (~4.9 GB) | 70B @ 4-bit (~40 GB) |
|---|---|---|---|
| Dual-channel DDR5 desktop CPU | ~90 GB/s | ~18 tok/s | ~2 tok/s |
| Apple M4 Max (unified) | 546 GB/s | ~110 tok/s | ~13 tok/s |
| RTX 4090 | ~1.0 TB/s | ~200 tok/s | (doesn't fit) |
| H100 SXM | 3.35 TB/s | ~680 tok/s | ~84 tok/s |
| B200 | ~8 TB/s | — | ~200 tok/s |

(Real numbers land at 60–85% of ceiling.) **Marketing TOPS of an NPU are irrelevant to decode**: an ANE with 38 INT8 TOPS attached to the same LPDDR as the GPU adds zero bandwidth, so it cannot raise the decode ceiling — a fact central to Section 5. NPU TOPS *do* matter for prefill, which is compute-bound.

---

## 3. Reference architecture: datacenter serving engines

Modern serving engines (vLLM V1, SGLang, TensorRT-LLM) converge on the same layered design:

```
┌────────────────────────────────────────────────────────────────┐
│ 0. Cluster orchestration (NVIDIA Dynamo, llm-d, sglang-router) │
│    KV-aware routing · prefill/decode disaggregation · scaling  │
├────────────────────────────────────────────────────────────────┤
│ 1. API frontend (CPU): OpenAI-compat HTTP, tokenize, stream    │
├────────────────────────────────────────────────────────────────┤
│ 2. Scheduler (CPU): continuous batching, token budgets,        │
│    chunked prefill, priorities, preemption                     │
├────────────────────────────────────────────────────────────────┤
│ 3. KV-cache manager: PagedAttention block tables,              │
│    prefix cache (RadixAttention), FP8 KV, CPU/NVMe tiering     │
├────────────────────────────────────────────────────────────────┤
│ 4. Model executor (GPU): CUDA Graphs / torch.compile,          │
│    FlashAttention-3 / FlashInfer, fused MoE kernels,           │
│    quantized GEMMs (Marlin, Machete, CUTLASS FP8/FP4)          │
├────────────────────────────────────────────────────────────────┤
│ 5. Parallelism: tensor (NCCL all-reduce) · pipeline · expert   │
│    (DeepEP all-to-all) · data (for MLA) · context/sequence     │
├────────────────────────────────────────────────────────────────┤
│ 6. Speculative decoding: EAGLE-3, MTP, n-gram drafts           │
├────────────────────────────────────────────────────────────────┤
│ 7. Sampler + structured output (grammar FSMs on CPU)           │
└────────────────────────────────────────────────────────────────┘
```

Key mechanisms, and which engine pioneered them:

- **Continuous (in-flight) batching** (Orca, 2022; now universal): requests join/leave the running batch at every step instead of waiting for a batch to finish. Turns idle bubbles into throughput; typically the single biggest win (>10×) over static batching.
- **PagedAttention** (vLLM): KV cache stored in fixed-size blocks with a per-request block table, like virtual memory. Eliminates fragmentation, enables prefix sharing and preemption-by-swap.
- **RadixAttention / prefix caching** (SGLang): a radix tree over token prefixes lets concurrent requests share KV for common prefixes (system prompts, few-shot contexts, multi-turn chat) — huge for agentic workloads.
- **Chunked prefill** (Sarathi/DeepSpeed-FastGen; default in vLLM V1): long prompts are split into chunks and *mixed into the same step as decode tokens*. This is important for the hardware question: it deliberately packs compute-bound prefill work and bandwidth-bound decode work into one GPU step so both the SMs and the HBM are saturated simultaneously — "using all of the hardware" *within* one device.
- **vLLM V1 process architecture**: an isolated `EngineCore` busy-loop process (scheduler + executor) communicating with the API process over ZMQ, with async scheduling that overlaps the CPU-side scheduling/detokenization of step *n+1* with GPU execution of step *n*. SGLang's "zero-overhead scheduler" does the same. This is the mainstream answer to "use the CPU too": the CPU runs the control plane *concurrently and hidden under* GPU compute, not the forward pass.
- **Prefill/decode disaggregation** (DistServe, Splitwise, Mooncake → productized in NVIDIA Dynamo, GA since March 2026; supported in vLLM, SGLang, TensorRT-LLM): prefill and decode run on *separate GPU pools*, with KV caches transferred over RDMA/NVLink (NIXL, Mooncake transfer engine). Pools can use different GPU SKUs and different parallelism degrees tuned per phase (compute-heavy TP for prefill, bandwidth/capacity-oriented for decode). Reported 2–3× throughput at the same SLOs. This *is* heterogeneous "use the right hardware per phase" — at rack scale.
- **Speculative decoding** (EAGLE-3, Medusa, DeepSeek MTP): the target model verifies k drafted tokens per step, amortizing one weight-read over multiple tokens — effectively raising decode's arithmetic intensity.
- **Quantization as a first-class path**: FP8 (Hopper), NVFP4/MXFP4 (Blackwell), INT4 AWQ/GPTQ, plus FP8 KV cache.

---

## 4. Architecture of on-device engines

### 4.1 llama.cpp / GGML — the portable hybrid

- A C/C++ **compute-graph** runtime with a **backend registry**: CPU (AVX2/AVX-512/AMX, NEON), CUDA, Metal, Vulkan, HIP/ROCm, SYCL, OpenCL-Adreno. GGUF file format with k-quant / i-quant 2–8-bit weights.
- **Layer offloading** (`-ngl N`): put N transformer layers on the GPU, the rest on CPU; `--tensor-split` across multiple GPUs; `--override-tensor` to pin *specific tensors* (e.g. MoE experts) to CPU while attention stays on GPU.
- This is genuine CPU+GPU cooperative execution — but it exists for **capacity** (run models bigger than VRAM), not speed: layers execute sequentially, so the slow unit sits on the critical path (see §5.2). The community-observed "cliff" when a model doesn't fully fit in VRAM is this math in action.

### 4.2 Apple: Core ML, MLX, and the ANE

- **Core ML** is the only public route to the **Apple Neural Engine**. You declare `MLComputeUnits.all` and Core ML's black-box scheduler *partitions the graph* across CPU/GPU/ANE based on op support — the closest thing Apple ships to "use everything," but you don't control placement, and unsupported ops silently fall back.
- **ANE constraints**: static shapes, FP16-centric, conv-style memory layouts, no public ISA. Growing-KV autoregressive attention fits poorly (mitigations: enumerated shapes, fixed-window chunked attention). Apple's *own* on-device foundation models run on the ANE; third-party access is via Core ML conversion (the ANEMLL project) or reverse-engineered private APIs (`ane.cpp`, Orion). At WWDC 2026 Apple moved further with "Core AI" and a `LanguageModel` protocol opening the on-device stack to third parties.
- **Measured reality (2026)**: ANEMLL gets ~47–62 tok/s on 1B and ~9 tok/s on 8B models on the ANE at very low power, while **MLX on the same machine's GPU does ~93 tok/s on the same 8B model**. The GPU wins on speed because both share the same LPDDR bandwidth and the GPU's software stack (Metal) is mature and shape-flexible; the ANE wins on energy (~⅓–⅕ the power). That's the trade in one line: **ANE = perf/W, GPU = perf.**
- **MLX** (Apple's ML framework): unified-memory, lazy-evaluation array framework targeting the **GPU first**; it does not use the ANE. whisper.cpp is the classic heterogeneous counter-example: its *encoder* (static-shape, conv-friendly) runs on the ANE via Core ML while its decoder runs on CPU/GPU.
- **Update (Aug 2026)**: the "third-party stacks don't touch the ANE" claim is now outdated. **oMLX** ships an experimental (default-off) prefill path that drives the ANE through the *private* `AppleNeuralEngine.framework` — splitting MLP matmuls by output channel across the GPU, **both physical ANE dies** (M3 Ultra), and optionally the CPU, merged in one fused SwiGLU kernel via shared IOSurfaces. Measured: +36–46% prompt throughput, +18.9% end-to-end prompt processing at 32K, decode unchanged. See the full code study in [omlx-code-study.md](omlx-code-study.md).

### 4.3 Pluggable-backend runtimes: ONNX Runtime, OpenVINO, ExecuTorch, LiteRT

- **ONNX Runtime**: an **execution provider (EP)** architecture — CUDA, TensorRT, DirectML, CoreML, QNN, OpenVINO, ROCm, XNNPACK… The runtime *partitions the graph* by which EP claims which nodes, and falls back to CPU for the rest. Partitioning is static per session, not a dynamic load balancer.
- **OpenVINO** (Intel): device plugins (CPU, iGPU, NPU) plus meta-plugins: `AUTO` (pick the best single device), `HETERO` (static graph partitioning across devices), `MULTI` (run *different requests* on different devices — throughput via parallel streams, not by splitting one model).
- **ExecuTorch / LiteRT**: delegate-based (XNNPACK, Core ML, QNN, Vulkan) — same pattern: compile subgraphs to whichever accelerator supports them, CPU as fallback.

### 4.4 AMD Ryzen AI hybrid — production NPU+iGPU split

AMD's Ryzen AI SW (OnnxRuntime GenAI "hybrid" flow, Strix/Krackan Point) is the clearest shipping example of phase-level heterogeneous execution on one chip: **prefill runs on the XDNA2 NPU** (~50 TOPS, compute-bound phase → minimizes time-to-first-token) while **decode runs on the RDNA iGPU** (bandwidth-bound phase → maximizes tokens/sec). Intel's OpenVINO stack on AI PCs is moving the same direction. Qualcomm (Genie/QNN) and MediaTek ship similar NPU-centric LLM flows on phones, where the NPU is preferred for *thermals*, not peak speed.

---

## 5. Can an engine use ALL the hardware at once?

### 5.1 Reframe the question

"Are all units busy?" is the wrong metric; **"is the bottleneck resource saturated at every instant?"** is the right one. For decode the bottleneck is bandwidth. Lighting up a second compute unit only helps if it adds *usable* bandwidth or removes work from the critical path without adding synchronization that costs more than it saves.

### 5.2 Why naive "run the model on everything" loses

1. **Shared DRAM on SoCs.** On Apple Silicon / Strix / Snapdragon, CPU, GPU and NPU draw from *the same* LPDDR. Decode is bandwidth-bound and the GPU alone can already pull most of the fabric's bandwidth — adding the ANE/CPU to the same forward pass adds **contention, not bandwidth**. This is the fundamental reason "CPU+GPU+ANE all at once" does not stack on a Mac.
2. **PCIe is ~100× slower than HBM.** On discrete systems (GPU 1–8 TB/s vs PCIe 32–64 GB/s), any partition that ships activations or KV across the bus every token is dominated by transfer time. Only *communication-light* partitions survive: whole layers or whole experts resident on the other side.
3. **Autoregressive decode is a serial chain.** Split layers across units and per-token latency = *sum* of stage times — the slow unit sits on the critical path. If the CPU is 10× slower per layer, offloading just 20% of layers ~triples token time (the llama.cpp partial-offload cliff). Pipelining recovers throughput only with many concurrent streams, which rarely exist on-device.
4. **NPUs want static graphs.** ANE/Hexagon/XDNA compilers need fixed shapes; growing KV caches and dynamic continuous batching fight that. (Workarounds — enumerated shapes, fixed attention windows — cost efficiency.)
5. **Format mismatch.** Each unit wants different weight layouts/quantizations (Metal blocks vs. ANE conv layouts vs. AVX tiles) → either duplicate weights in RAM or pay conversion. On memory-limited devices duplication is a non-starter.
6. **Load-balancing granularity + Amdahl.** A 5-TFLOP NPU beside a 60-TFLOP GPU contributes ≤8% even with perfect, free partitioning — and partitioning is neither.
7. **Power/thermal envelope.** On phones/laptops, all-units-blazing throttles the package clock; the NPU exists for perf/W, not peak. Sustained throughput can be *higher* using the NPU alone than everything at once.

### 5.3 What engines do instead: complementary roles, all concurrent

A modern serving step *does* use CPU and GPU simultaneously — in different roles:

| Unit | Role during steady-state serving |
|---|---|
| GPU | The forward pass: prefill chunks + decode batch fused into each step (chunked prefill saturates SMs *and* HBM together) |
| CPU | Runs *concurrently, hidden under GPU time*: scheduling step n+1, tokenization/detokenization, sampling bookkeeping, grammar FSM compilation for structured output, prefix-cache/radix-tree management, KV tier orchestration |
| CPU RAM / NVMe | **Bandwidth/capacity tier, not compute**: KV-cache offload and prefix-cache store (LMCache, Mooncake); reusing a cached prefix beats recomputing it even over PCIe |
| NPU/ANE | Prefill (AMD hybrid), auxiliary models (embeddings, rerankers, ASR encoders, drafts), or deliberately idle on battery-insensitive setups |

vLLM V1's separated EngineCore and SGLang's overlap scheduler exist precisely so the CPU work costs *zero* GPU idle time. That's "using all the hardware" done right: no unit blocks another.

---

## 6. Where heterogeneous execution genuinely wins today

These are the real, shipping cases where multiple unit types do compute for the same workload — and why they work:

1. **CPU+GPU MoE hybrids — KTransformers (SOSP '25), now integrated into SGLang.** For sparse MoE models (DeepSeek-R1/V3, 671B) : attention + KV + dense trunk on a 24 GB GPU; the **experts live in CPU RAM and are computed with AMX/AVX-512 kernels** (up to ~21 TFLOPS sustained per Xeon socket), with an *expert-deferral* mechanism that overlaps CPU and GPU work to ~100% CPU utilization. Gains: 4.6–19.7× prefill and 1.25–4× decode vs. prior offloading. Why it works: only ~5% of experts activate per token, so DDR bandwidth is adequate for the sparse slice, and communication is expert-granular (tiny). PowerInfer (hot/cold neuron split) and Fiddler are the same idea for different sparsity structures. **This is the strongest true "CPU and GPU computing simultaneously" result — motivated by capacity, delivering real throughput.**
2. **Phase split on one SoC — AMD Ryzen AI hybrid.** NPU does compute-bound prefill (best TTFT), iGPU does bandwidth-bound decode (best tok/s). Works because the phases are *sequential per request*, so there's one handoff, not per-token sync.
3. **Phase split at rack scale — disaggregated serving (Dynamo, Mooncake, DistServe).** Same logic with GPUs as the units: prefill pool and decode pool scale independently, use different SKUs/parallelism, KV moves once over RDMA. 2–3× throughput at equal SLOs; standard across vLLM/SGLang/TRT-LLM in 2026.
4. **Stage split across models — whisper.cpp** (ANE encoder + CPU/GPU decoder), mobile VLM stacks (vision tower on NPU, LLM on GPU), agent pipelines (embedding/reranker models on CPU/NPU beside the GPU-resident LLM). Works because stages are separate static graphs with one boundary.
5. **Speculative decoding across units** (research → early product): draft model on NPU/CPU, target verification on GPU. The draft is small enough for the weak unit, and verification batches amortize the sync.
6. **Memory tiering** — CPU RAM and NVMe as KV/prefix-cache tiers behind the GPU (LMCache, Mooncake store, vLLM/SGLang hierarchical cache). The CPU side contributes *capacity and reuse*, which converts directly into throughput on agentic/multi-turn workloads.
7. **Operator-level splits across GPU + NPU + CPU — oMLX's ANE prefill** (studied in depth in [omlx-code-study.md](omlx-code-study.md)): the same MLP matmul sliced by output channel across the GPU, both ANE dies, and the CPU, merged once per layer through zero-copy shared surfaces, with a per-machine five-way tuner and a precision rule (the approximate INT8 ANE gets only token-local math; recurrent state stays on exact hardware). +36–46% prompt throughput on M3 Ultra; decode untouched.

Pattern in all of them: **coarse-grained partitions with rare synchronization points** (per phase, per expert, per stage — never per-op), each unit running the sub-workload whose roofline matches it.

---

## 7. Practical guidance: max throughput per platform

| Platform | Do this | Avoid |
|---|---|---|
| NVIDIA datacenter | vLLM / SGLang / TRT-LLM; FP8/FP4; continuous batching + chunked prefill (defaults); disaggregate P/D at scale (Dynamo); EAGLE-3/MTP spec-dec | CPU compute in the token path (except MoE expert spill); tiny batches on big GPUs |
| Mac (Apple Silicon) | MLX or llama.cpp with full Metal offload; buy bandwidth (Max/Ultra) not TOPS; ANE via Core ML/ANEMLL only for small models or battery-critical apps | Splitting a dense model across CPU+GPU+ANE expecting additive speed — they share one LPDDR |
| AI PC (Ryzen AI / Intel) | Vendor hybrid flows: OGA hybrid (NPU prefill + iGPU decode), OpenVINO AUTO/NPU | Hand-rolling per-op CPU/NPU splits |
| 24 GB GPU + big-RAM workstation | Dense: quantize to fit VRAM entirely. MoE: KTransformers or llama.cpp `--override-tensor` expert offload (AMX CPU if available) | Partial *layer* offload of dense models (the throughput cliff) |
| Phones (Snapdragon/MediaTek) | QNN/Genie or LiteRT NPU paths — for sustained thermals, NPU-only usually beats all-units | Burst benchmarks that ignore throttling |

Cross-cutting: quantize weights (and KV at scale), maximize batch/concurrency if serving, exploit prefix caching for agent/chat workloads, and measure **bandwidth utilization** — if it's near peak on the decode path, that hardware is doing all it physically can, regardless of how many units show 0%.

---

## 8. Future directions

- **NPUs are growing up**: dynamic-shape and paged-attention support is arriving in NPU compilers; expect NPUs to graduate from "prefill + small models" to fuller LLM roles.
- **Unified memory is spreading** (Apple M-series, AMD Strix Halo, NVIDIA GB10/DGX Spark): when CPU and GPU share one physical memory at full bandwidth, the offloading question dissolves into a pure scheduling question — heterogeneous placement gets cheap.
- **Compiler-driven heterogeneous placement** (IREE, TVM, Apple's Core AI direction): declare the graph, let the compiler split it across units with cost models — today's hand-built hybrids (KTransformers, OGA hybrid) becoming a compiler feature.
- **Disaggregation standardizing**: KV-transfer interfaces (NIXL, Mooncake TE) and KV-aware routers (Dynamo, llm-d) are becoming the "TCP/IP" of inference clusters.
- **Sparsity as the heterogeneity enabler**: MoE and activation sparsity are what make slow-unit participation profitable; as models get sparser, "use all the silicon" becomes more true, not less.

---

## 9. References

- vLLM V1 architecture: [vLLM V1 blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release), [docs: arch overview](https://docs.vllm.ai/en/latest/design/arch_overview/), [Inside vLLM (Gordić)](https://www.aleksagordic.com/blog/vllm)
- PagedAttention: Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*, SOSP 2023
- SGLang / RadixAttention: Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, 2024
- Continuous batching: Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022
- Chunked prefill: Agrawal et al., *Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*, 2023
- Disaggregation: Zhong et al., *DistServe*, OSDI 2024; Patel et al., *Splitwise*, ISCA 2024; Qin et al., *Mooncake*, FAST 2025; [NVIDIA Dynamo disaggregated serving docs](https://docs.dynamo.nvidia.com/dynamo/design-docs/disaggregated-serving), [Dynamo 1.0 deployment guide](https://www.spheron.network/blog/nvidia-dynamo-disaggregated-inference-guide/)
- KTransformers: [MADSys, SOSP 2025](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/), [SGLang integration blog](https://www.lmsys.org/blog/2025-10-22-KTransformers/), [GitHub](https://github.com/kvcache-ai/ktransformers)
- PowerInfer: Song et al., *PowerInfer: Fast LLM Serving with a Consumer-grade GPU*, SOSP 2024
- AMD Ryzen AI hybrid (NPU prefill + iGPU decode): [OGA hybrid flow docs](https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html), [DeepSeek distilled on Ryzen AI](https://amd.com/en/developer/resources/technical-articles/deepseek-distilled-models-on-ryzen-ai-processors.html)
- Apple ANE for LLMs: [Orion: Characterizing and Programming Apple's Neural Engine](https://arxiv.org/abs/2603.06728), [ANE LLM inference guide (InsiderLLM)](https://insiderllm.com/guides/apple-neural-engine-llm-inference/), [ANEMLL](https://github.com/anemll/anemll), [Apple Core AI / WWDC 2026](https://www.buildmvpfast.com/blog/apple-core-ai-on-device-llm-swift-wwdc-2026)
- llama.cpp backends & offloading: [ggml/llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- ONNX Runtime execution providers: [ORT EP docs](https://onnxruntime.ai/docs/execution-providers/)
- OpenVINO device plugins / AUTO / HETERO: [OpenVINO docs](https://docs.openvino.ai/)
