---
name: squeeze-gpu
description: >
  Squeeze a local GPU for LLM tok/s: identify the card, pick one lever
  (clocks, quant, spec-decode, or kernels), run a four-grid, kill if
  unstable. Use when the user says 最佳化, 超頻, 榨, 跑分, Afterburner,
  llama-bench, tok/s, or names a card (2080 Ti, 4090, 5090, 7900, M4,
  Strix Halo). Works on any local GPU — this repo's 2080 Ti Mod 22GB
  is the worked example, not the only target. Slash: /squeeze-gpu
---

# Squeeze this machine's GPU

Decision law (read once, do not rewrite): [`docs/method.md`](../../../docs/method.md).
Worked example only: RTX 2080 Ti **Mod 22GB** — [`docs/results.md`](../../../docs/results.md), [`docs/clocks.md`](../../../docs/clocks.md).

**Do not copy that board's sliders, TGP, or `kCols=8` onto another card.**
The skill is portable. The numbers are not.

## 0. Identify (this session's card)

Run what exists. Record in the reply before changing anything:

| Fact | How |
|---|---|
| Name, VRAM, driver | `nvidia-smi` / `rocm-smi` / `system_profiler SPDisplaysDataType` |
| Arch / compute | NVIDIA: `sm_75` Turing, `sm_80/86` Ampere, `sm_89` Ada, `sm_90+` Hopper, `sm_120` Blackwell. AMD: `gfx11xx`. Apple: Metal / MLX. |
| Memory kind | Dedicated VRAM vs unified (Apple, many APUs). Fit math is different. |
| Paper decode ceiling | `tok/s ≈ DRAM_GB/s ÷ weight_GB`. If already ≥85% of that, stop writing kernels. |
| Engine that actually runs | llama.cpp / MLX / ROCm / vendor. Compile **this** arch only. |

If load clocks are stuck at idle / stock-boost after a profiler: fix power plan, TGP, or profiler clock-control **before** any “optimization.”

NVIDIA ncu: always `--clock-control none`. If already pinned, reboot the host OS.

## 1. Baseline (void without this)

Same weights, same binary family, `r≥3`, report σ:

- `pp512`, `tg128`, `pp512 @ d4096`, `tg128 @ d4096`

Helper in this repo: `scripts/llama-bench-grid.sh <gguf>` (assumes llama.cpp CUDA). On MLX/ROCm, same four cells, their harness.

Print the GPU name from the bench. A fatbinary that loaded a **different** cubin than you compiled for is a different result.

## 2. One lever, written kill line first

Order unless the user names a different question:

1. **Seconds/byte** — clocks / TGP / mem. Dedicated NVIDIA: mem first, core second. Unified: there is no GDDR slider; don't pretend.
2. **Bytes/token** — new quant = **new cell**. Don't subtract from Q4.
3. **Tokens/cycle** — MTP / draft. Long prompt, same paragraph, report accept. Short filler is not a cell.
4. **Work/byte** — handwritten CUDA/PTX **last**. Only if a profiler says this kernel is the hotspot **and** not already DRAM-bound.

Kill if: not monotonic vs the last step, σ explodes, CUDA/ROCm abort, driver reset. Instability wins over a lucky peak.

Never in the same run: mem+core, quant+clocks, MTP+kernel.

## 3. Architecture branch (constraints, not recipes)

**Turing `sm_75` (this repo's lab):** read [`docs/tu102.md`](../../../docs/tu102.md) and [`docs/impl-gate.md`](../../../docs/impl-gate.md). Illegal: `cp.async`, TMA, BF16 MMA, `wgmma`, `m16n8k16`, L2 persisting. Afterburner writer needs explicit `--mhz --mem --power`. Stock llama.cpp already has Turing MMA fattn/mmq — see [`docs/ggml-turing.md`](../../../docs/ggml-turing.md). Do not rewrite softmax FA to “replace FA2.”

**Ampere / Ada / Blackwell:** those ISA features may be legal. Do **not** apply the Turing ban list as if it were physics. Do **not** use this repo's 35.6 tg or +285/+1250 as a target. Same four-grid and kill lines.

**AMD / Apple:** no Afterburner path. Still: identify → ceiling → four-grid → one lever. Unified memory: usable pool is not the brochure RAM figure.

**VRAM smaller than the example:** a stock 11 GB 2080 Ti will not hold 27B Q4 at `-ngl 99`. Pick a model that fits. The title of this repo is the 22 GB **mod**.

## 4. Write back

- New measurement: append a dated row (card name, arch, weights, one variable, four-grid ±σ). Do not overwrite the 2080 Ti example table.
- New chip fact for Turing: [`docs/tu102.md`](../../../docs/tu102.md).
- Do not put another board's OC into `scripts/write-afterburner-uv.py` defaults.

## Refuse

- Flash VBIOS “to unlock TGP.”
- Install Linux `nvidia-driver*` inside WSL.
- Quote lab FA TFLOP/s as LLM tok/s.
- Add `sm_120` (or any second arch) to a Turing claim binary.
- Treat `docs/results.md` as a leaderboard to beat on a different GPU.
