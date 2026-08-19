# Example measurements

One **RTX 2080 Ti** (TU102 cut-down: 68 SM, 352-bit, **616 GB/s**, 22 GB VRAM). Date 2026-08. Protocol: [`scripts/llama-bench-grid.sh`](../scripts/llama-bench-grid.sh). Model: Qwen3.8-27B ([`qwen38-27b.md`](qwen38-27b.md)). Engine: llama.cpp CUDA, `CMAKE_CUDA_ARCHITECTURES=75`.

These are a **worked example** of [`method.md`](method.md), not a fleet leaderboard.
**Stock 11 GB 2080 Ti cannot hold 27B Q4 at `-ngl 99`.** The table needs the 22 GB mod.

## Lab kernels (this repo)

See [`kernels.md`](kernels.md). Headline: WMMA FA 3.21× vs FFMA at `2×8×512×128` (3.6 TFLOP/s). GDN occ8 0.91 ms/layer at T=512 vs ggml-like 1.53 ms.

## End-to-end 27B four-grid

Same card, `r=3`, GPU name and sm 7.5 printed.

| Setup | pp512 | tg128 | pp@4k | tg@4k |
|---|---:|---:|---:|---:|
| Q4_K_M, `fa on`, stock GDN | 669 | 27.5 | 622 | 27.1 |
| + local GDN `kCols=8` (not upstream) | 704 | 27.8 | **653** | 27.5 |
| IQ4_XS, still those clocks / COLS=8 | 787 | 29.1 | 741 | 28.9 |
| IQ4_XS + example clocks (+285 / +1250) | **886 ± 59** | **35.55 ± 0.07** | **837 ± 24** | **34.86 ± 0.26** |

Paper decode ceiling ~38 tok/s (616 GB/s / ~16 GB). 35.6 is ~90%.

## MTP (long prompt, n=512, not the four-grid)

llama.cpp `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.4`. Compare only on the **same long prompt** (`n=512`); a different paragraph is a different cell (old 31.7 vs later 36.1 was that). `n_draft=5` lost on long text. After clocks, IQ4 MTP also won vs Q4 on that same prompt.

## What did not win

- Raising TGP 250 → 280: prefill clocks up a little, decode still bandwidth-bound.
- DSpark, ngram. Real 4k ingest (`-p 4096 -n 0`): `-b 2048 -ub 1024` was 719 vs 692 at `-ub 512`. `pp512@d4096` does not measure that.
- Chunked WMMA GDN (HMMA present, slower than occ8).
- mmq MMA rewrite (25% occupancy from 255 registers).
- mmvq PTX (ncu DRAM 81% on the main ncols=1 kernel).
- Core +315 / +345 (CUDA abort).

## Agentic

A private 34-task SWE-style exam on this card (IQ4, vendor thinking): 26/34 without MTP, 27/34 with the MTP recipe above (McNemar p=1 — do not call MTP more accurate). Public scores live on [model-dyno](https://cookys.github.io/model-dyno/#/swe/comp) as machine `itx-5950x`. Task names from that exam are **not** in this repository.
