# 手寫 kernel：X／GitHub 有誰

錨：2026-08。結論：有人寫，但不是 2080 Ti 日常玩家在 X 發 tok/s。

## X

三類，幾乎都不是「Turing 生產級 FA」：

- 打卡／作業：softmax、transpose、tiled GEMM。
- 論文／教學 PTX：例如 Hand-Written PTX Tensor-Core GEMM on **L4**（Ada，[arxiv:2608.10103](https://arxiv.org/abs/2608.10103)）。4060 上手寫 PTX MMA 到 cuBLAS ~81%。
- Agent 改別人的 kernel：SageAttention 手寫 CUDA 在 3090 上輸同庫 Triton（tile 為 A100 寫死）。

2080 Ti 帖 = llama.cpp 調參，不是「我寫了 HMMA 進主線」。

## GitHub 出貨棧（本來就是手寫）

| 專案 | 給誰 |
|---|---|
| Dao-AILab FlashAttention | **sm_80+**（[Turing issue 實質 Wontfix](https://github.com/Dao-AILab/flash-attention/issues/1235)） |
| llama.cpp `ggml-cuda` | 含 Turing 的一套自己的 FA／GEMM，不是 FA2 |
| Marlin／AWQ／EXL2 | INT4 GEMM，Ampere 優先 |
| SageAttention、FlashInfer | 新卡 |
| [Tugbars/Flash-Attention-PTX-CUDA](https://github.com/Tugbars/Flash-Attention-PTX-CUDA) | 5080，對標 vLLM FA2 |

## 專門給 Turing 的（最接近本 repo）

**[JohnScheuer/flash-attention-sm75](https://github.com/JohnScheuer/flash-attention-sm75)**（2026）

- FA **v1** forward、WMMA、sm_75、d=64/128、causal
- 對 naive ~2.5×；作者承認對 **PyTorch SDPA efficient 慢 10–13×**
- RTX 2070：~**1.4 TFLOPS**，峰值 ~5%（中間 smem S/P/T + scalar softmax）
- 端到端 Qwen2-0.5B 仍 +27% tok/s（省 HBM、prefill）
- 另有 [fused 版號稱 5 TFLOPS](https://github.com/JohnScheuer/flash-attention-sm75-fused)、[sm75-tensorcore-microkernel](https://github.com/JohnScheuer/sm75-tensorcore-microkernel)

對標這份時只比 **標準 softmax FA、D=64/128**。數字是 **2070**，不是這張 2080 Ti。  
他沒有 D=256、沒有原生 GQA、沒有 gate、沒有 DeltaNet——**不是** Qwen3.8-27B 的說明書（見 `qwen38-27b.md`）。GQA 他用展開 KV。接的是 HF SDPA + Qwen2-0.5B，不是 llama.cpp。

對標這份，不要對標 X 上的 27B llama.cpp 帖。那些不是他們手寫的 MMA。

## 為什麼看起來像沒人寫

能發推的用 llama.cpp；能寫 MMA 的對準 sm_80／90／120（`cp.async`、BF16、大 L2、wgmma）。Turing 投資報酬差。2026 熱潮是 LLM **生** kernel（KernelBench），目標卡也是新卡。
