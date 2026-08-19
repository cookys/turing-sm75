# Survey: Turing vs the 2026 LLM stack

## Raster vs LLM

Game raster of a 2080 Ti is still ~4060 Ti / 5060 class ([`raster.md`](raster.md)). LLM is not the same contest.

| Stage | Work | What it actually burns |
|---|---|---|
| Prefill | Big GEMM, attention | Tensor Core generation + dedicated kernels |
| Decode | Sweep weights / KV every token | Effective bandwidth (DRAM + L2), not brochure GB/s |

Four things stack:

1. **2nd-gen Tensor Cores.** No first-class BF16, no FP8/FP4, no TF32. 544 cores look many; each is weaker than Ada/Blackwell.
2. **Software left `sm_75`.** FA2/3 official paths are sm_80+. vLLM/HF default BF16. Marlin / fused AWQ / EXL2 / TRT-LLM / CUTLASS 3 prefer Ampere+. Turing gets generic CUDA fallbacks.
3. **L2 5.5 MB vs ~32 MB on 40/50 60-class.** Decode looks 616 GB/s on paper; tiles that miss L2 hit GDDR every time. A 288–448 GB/s 4060/5060 can match tok/s (Puget Systems, 2024-08, llama.cpp Phi-3 4-bit: 2080 Ti has more bandwidth and FP16, not more tokens).
4. **11 GB is awkward in 2026.** 22 GB boards (modded) hold 27B Q4; they do **not** get a new kernel ISA.

llama.cpp + GGUF Q4 on 7B/8B stays close. transformers / vLLM / long context / BF16 opens the gap.

## Engines on this card

| Engine | On Turing sm_75 |
|---|---|
| **llama.cpp CUDA** | The daily driver. GGUF, KV quant, kernels that still compile. |
| Ollama / LM Studio | Fine; fewer flags; usually llama.cpp underneath. |
| vLLM / SGLang | Fast paths are FA / FlashInfer on sm_80+. |
| ExLlamaV3 | Community focus is 30-series and up. |

## Who writes sm_75 kernels

X posts about 2080 Ti tok/s are almost all llama.cpp flags, not HMMA. GitHub FA2 is sm_80+. Closest public Turing FA: [JohnScheuer/flash-attention-sm75](https://github.com/JohnScheuer/flash-attention-sm75) (WMMA, D=64/128, ~1.4 TFLOPS on a 2070). No D=256, no native GQA, no DeltaNet — not a Qwen3.8-27B spec. More: [`survey-kernels.md`](survey-kernels.md).
