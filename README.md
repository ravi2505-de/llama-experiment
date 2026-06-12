---
title: Minimal Qwen Llama Cpp T4
sdk: gradio
python_version: 3.11
app_file: app.py
license: apache-2.0
suggested_hardware: t4-small
---

# Minimal HF Space: Qwen2.5 1.5B GGUF on Nvidia T4

This Space is intentionally minimal. It verifies:

- build success with `gradio` and `llama-cpp-python`
- loading `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
- Q4_K_M quantized model file loading
- CUDA inference with full GPU offload via `n_gpu_layers=-1`
- memory usage before load, after load, and after generation
- basic response generation through a Gradio chatbot

No ZeroGPU configuration is used.
