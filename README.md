# research-inferencing

Research notes on LLM/ML inference systems.

## Documents

- **[Inference Engine Architecture — and Whether Engines Can Use All the Hardware (CPU/GPU/NPU/ANE) for Max Throughput](inference-engine-architecture.md)** *(2026-08)* — anatomy of datacenter serving engines (vLLM, SGLang, TensorRT-LLM, Dynamo) and on-device engines (llama.cpp, MLX, Core ML/ANE, ONNX Runtime, OpenVINO, Ryzen AI); the prefill/decode roofline physics; why "run it on every unit" usually loses to saturating the bandwidth-dominant unit; and the real heterogeneous wins (KTransformers CPU+GPU MoE, NPU-prefill + iGPU-decode, disaggregated serving, stage splits).
