# Example measurements

One **RTX 2080 Ti** (TU102 cut-down: 68 SM, 352-bit, **616 GB/s**, 22 GB VRAM). Date 2026-08. Protocol: [`scripts/llama-bench-grid.sh`](../scripts/llama-bench-grid.sh). Model: Qwen3.8-27B ([`qwen38-27b.md`](qwen38-27b.md)). Engine: llama.cpp CUDA, `CMAKE_CUDA_ARCHITECTURES=75`.

These are a **worked example** of [`method.md`](method.md), not a fleet leaderboard.

## Lab kernels (this repo)

See [`kernels.md`](kernels.md). Headline: WMMA FA 3.21× vs FFMA at `2×8×512×128` (3.6 TFLOP/s). GDN occ8 0.91 ms/layer at T=512 vs ggml-like 1.53 ms.

## End-to-end 27B four-grid

Same card, `r=3`, GPU name and sm 7.5 printed.

| Setup | pp512 | tg128 | pp@4k | tg@4k |
|---|---:|---:|---:|---:|
| Q4_K_M, `fa on`, stock GDN | 669 | 27.5 | 622 | 27.1 |
| + GDN 8 cols/warp | 704 | 27.8 | **653** | 27.5 |
| IQ4_XS, still stock clocks | 787 | 29.1 | 741 | 28.9 |
| IQ4_XS + example clocks (+285 / +1250) | **886 ± 59** | **35.55 ± 0.07** | **837 ± 24** | **34.86 ± 0.26** |

Paper decode ceiling ~38 tok/s (616 GB/s / ~16 GB). 35.6 is ~90%.

## MTP (long prompt, n=512, not the four-grid)

llama.cpp `--spec-type draft-mtp`. Grid winner on this card: **`n_max=3`, `p_min=0.4`**. `n_draft=5` lost on long text. After clocks, IQ4 MTP also won vs Q4 (short-prompt MTP ranking is not the same).

## What did not win

- Raising TGP 250 → 280: prefill clocks up a little, decode still bandwidth-bound.
- DSpark, ngram, extra `-ub` on `pp512@d4096` (need a real 4k ingest run).
- Chunked WMMA GDN (HMMA present, slower than occ8).
- mmq MMA rewrite (25% occupancy from 255 registers).
- mmvq PTX (ncu DRAM 81% on the main ncols=1 kernel).
- Core +315 / +345 (CUDA abort).

## Agentic

A private 34-task SWE-style exam on this card (IQ4, vendor thinking): 26/34 without MTP, 27/34 with the MTP recipe above (McNemar p=1 — do not call MTP more accurate). Public scores live on [model-dyno](https://cookys.github.io/model-dyno/#/swe/comp) as machine `itx-5950x`. Task names from that exam are **not** in this repository.
