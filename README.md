# turing-sm75 — RTX 2080 Ti Mod 22GB (`sm_75`)

Kernels, clocks, and a method for running LLMs on a **modded 22 GB RTX 2080 Ti** (Turing `sm_75`, still 352-bit / 616 GB/s). Not a stock 11 GB card.

Raster on this class is still close to a 4060 Ti / 5060. LLM stacks are not: FlashAttention 2/3, vLLM, and most fused INT4 kernels dropped pre-Ampere. This repo is the missing middle — what is still legal on Turing, what stock llama.cpp already does, and how to tell **clocks / quant / spec-decode** from **handwritten CUDA**.

中文：從一張 **2080 Ti 22GB 改件** 抽出的公開套件。不含內網、VBIOS、私有考卷。決策法在 [`docs/method.md`](docs/method.md)。Clone 之後對**你手上那張卡**走 skill `squeeze-gpu`（`/squeeze-gpu`）——2080 Ti 只是例子，滑桿不能抄。

## What's here

| Piece | Path |
|---|---|
| **Session playbook (any local GPU)** | [`.grok/skills/squeeze-gpu/SKILL.md`](.grok/skills/squeeze-gpu/SKILL.md) |
| Softmax FA lab (WMMA, D=64/128) | [`src/attn_sm75.cu`](src/attn_sm75.cu) |
| Gated DeltaNet lab (Qwen3.8-27B shape) | [`src/gdn_sm75.cu`](src/gdn_sm75.cu) |
| Decision method (clocks **and** kernels) | [`docs/method.md`](docs/method.md) |
| Occupancy gate before writing CUDA | [`docs/impl-gate.md`](docs/impl-gate.md) |
| TU102 layout / HMMA / what not to copy | [`docs/tu102.md`](docs/tu102.md) |
| Example numbers (this 22GB board, 2026-08) | [`docs/results.md`](docs/results.md) |
| Four-grid llama-bench protocol | [`scripts/llama-bench-grid.sh`](scripts/llama-bench-grid.sh) |
| Afterburner VF-curve writer | [`scripts/write-afterburner-uv.py`](scripts/write-afterburner-uv.py) |
| Survey (who writes sm_75 kernels, why LLM loses) | [`docs/survey.md`](docs/survey.md) |

Not in this tree (on purpose): hostnames, LAN IPs, GPU UUIDs, VBIOS dumps, private SWE task names, Afterburner profile paths, or a live fleet dashboard.

## Build (must be `sm_75`)

```bash
source scripts/env.sh          # WSL2: Windows driver at /usr/lib/wsl/lib
# bash scripts/setup-wsl-cuda.sh  # toolkit only — never apt install nvidia-driver*
make
make run                       # 1×4×256×128 vs naive
./build/attn_sm75 2 8 512 128
make gdn                       # 27B-shaped DeltaNet microbench
make sass                      # must see HMMA on the WMMA kernel
```

Claims need: printed `RTX 2080 Ti` (or your Turing card) + `sm_75` + error vs naive `< 0.02`. A fatbinary that loads `sm_120` on Blackwell is a different kernel.

## The result in one table

Same 2080 Ti Mod 22GB, Qwen3.8-27B, llama.cpp CUDA `sm_75`. Four-grid protocol in `scripts/llama-bench-grid.sh`.

| Change | tg128 | pp512 @ 4k |
|---|---:|---:|
| stock Q4, `fa on` | 27.5 | 622 |
| GDN 8 cols/warp (local llama.cpp patch, not upstream) | 27.8 | 653 |
| IQ4_XS | 29.1 | 741 |
| clocks (example: core +285 / mem +1250) | **35.6** | **837** |

Handwritten CUDA moved decode ~1%. Clocks + smaller weights moved the rest. Details: [`docs/results.md`](docs/results.md).

## License

MIT. Measurements are from one board; treat them as a protocol, not a leaderboard.
