# research-inferencing

Research notes on LLM/ML inference systems.

## Documents

- **[oMLX Code Study — Inside a Production LLM Inference Engine for Apple Silicon](omlx-code-study.md)** *(2026-08)* — deep read of [jundot/omlx](https://github.com/jundot/omlx) v0.6.4 (~225K LOC): external chunked prefill over mlx-lm continuous batching, time-domain decode fairness, memory-admission control, the metadata-only paged KV cache with RAM/SSD tiers and hybrid-model state checkpointing, Lightning MTP / DFlash / SpecPrefill speculation, oQ quantization, custom Metal kernels, the multi-Mac (and Mac+CUDA) pipeline cluster — and the centerpiece: prefill split across GPU + both ANE dies + CPU via private APIs (+36–46% prompt throughput, decode unchanged).
- **[Inference Engine Architecture — and Whether Engines Can Use All the Hardware (CPU/GPU/NPU/ANE) for Max Throughput](inference-engine-architecture.md)** *(2026-08)* — anatomy of datacenter serving engines (vLLM, SGLang, TensorRT-LLM, Dynamo) and on-device engines (llama.cpp, MLX, Core ML/ANE, ONNX Runtime, OpenVINO, Ryzen AI); the prefill/decode roofline physics; why "run it on every unit" usually loses to saturating the bandwidth-dominant unit; and the real heterogeneous wins (KTransformers CPU+GPU MoE, NPU-prefill + iGPU-decode, disaggregated serving, stage splits).
