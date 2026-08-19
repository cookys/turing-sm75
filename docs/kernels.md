# Kernels in this repo

Both binaries are **`compute_75,code=sm_75` only**. Do not add `sm_120` “so it also runs”.

## Softmax FA — `src/attn_sm75.cu`

FlashAttention-style online softmax. QK and PV are WMMA 16×16×16 (`HMMA.1688.F32`). Softmax/rescale stay on CUDA cores. No `cp.async`, TMA, BF16 MMA, wgmma.

Same binary keeps `flash_attn_sm75_ffma` (CUDA-core dot) as the opponent.

Tiles `Br=Bc=16`. D instantiated at **64 and 128 only**. That is **not** Qwen3.8-27B (D=256 gated attn + DeltaNet).

Correctness vs naive two-pass softmax: `max |*-naive| < 0.02`.

Example, one RTX 2080 Ti, nvcc 13.1, same binary:

| kernel | shape | time | effective | vs ffma | vs naive |
|---|---|---:|---:|---:|---:|
| wmma (HMMA) | 2×8×512×128 | 0.603 ms | 3.6 TFLOP/s | 3.21× | err 0.00098 |
| ffma | 2×8×512×128 | 1.939 ms | 1.1 TFLOP/s | 1× | err 0.00024 |
| wmma | 1×4×256×128 | 0.094 ms | 1.4 TFLOP/s | 2.21× | err 0.00049 |

`cuobjdump -sass`: ELF is sm_75 only. WMMA kernel has `HMMA.1688.F32`. `LDGSTS` = 0.

3.6 TFLOP/s is ~7% of the ~54 TFLOP/s FP16-Tensor (FP32-acc) peak. It proves HMMA works. It is not a 27B tok/s result.

## Gated DeltaNet — `src/gdn_sm75.cu`

Shape matches Qwen3.8-27B: Sv=128, 16 QK heads, 48 V heads, KDA=false. Error vs `gdn_cpu` `< 0.02`. Chunked CPU path follows ggml `build_delta_net_chunking`.

Example, same 2080 Ti, T=512 one layer:

| kernel | T=512 | ×48 layers | notes |
|---|---:|---:|---|
| serial (128 cols/thread, spill) | 1.83 ms | 88 ms | |
| occ (1 col/warp, ggml-like) | 1.53 | 73 | |
| occ4 (4 cols/warp) | 1.02 | 49 | legal opponent |
| **occ8 (8 cols/warp)** | **0.91** | **44** | **landed in llama.cpp as `kCols=8`** |
| wmma chunk (HMMA×96) | 5.00 | 240 | has HMMA, slower |

occ8 ≈ 1.45× vs ggml-like occ, ≈ 1.12× vs occ4. 96 regs → ~62% occupancy. End-to-end 27B: pp512@4k 622 → 653. Decode almost unchanged (weight bandwidth).

## How to claim

```bash
make && make run && ./build/attn_sm75 2 8 512 128
make gdn
make sass    # HMMA on WMMA; FFMA on softmax / old kernel
```
