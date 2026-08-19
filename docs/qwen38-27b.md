# Target shape: Qwen3.8-27B

Weights: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Architecture is the Qwen3.5 **hybrid**, not Llama, not dense Qwen2.

| | |
|---|---|
| Params | 27B dense (+ vision encoder, MTP heads) |
| Hidden | 5120 |
| Layers | 64 = 16 × (3× Gated DeltaNet→FFN + 1× Gated Attention→FFN) |
| FFN | 17408 |
| Context | 262144 native |

**Gated attention (16/64, softmax):** 24 Q / 4 KV, **head_dim 256**, RoPE on ~64 dims, **gate after softmax**.

**Gated DeltaNet (48/64):** 16 QK / 48 V heads, head_dim 128, recurrent state (not S×S softmax).

On this **2080 Ti Mod 22GB**, Q4/IQ4 GGUF fits at `-ngl 99`. Q8/BF16 does not. Stock 11 GB does not hold 27B Q4. Distro `llama.cpp` packages often have **no CUDA** — build `GGML_CUDA` + `CMAKE_CUDA_ARCHITECTURES=75` yourself.

`attn_sm75.cu` is causal softmax FA at D=64/128, no GQA, no gate, no DeltaNet. Its TFLOP/s figure cannot be an LLM before/after for this model.
