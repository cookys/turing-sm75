# How to squeeze a Turing card (clocks and kernels, one method)

Numbers live in [`results.md`](results.md). Chip facts: [`tu102.md`](tu102.md). Occupancy arithmetic before any `.cu`: [`impl-gate.md`](impl-gate.md). This file is only the **decision law**.

## One sentence

Measure load clocks and the whole-model hotspot first. Then pick a lever. A kernel that does not match the 27B hotspot is not an optimization. Afterburner's on-disk F column, Nsight clocks, and a single peak run are not evidence.

## Four levers

| # | Lever | You change | Open it when |
|---|---|---|---|
| 1 | Seconds per byte | GDDR6 / core clocks | `nvidia-smi dmon` under load is below a stable `mclk`/`pclk` this board can hold |
| 2 | Bytes per token | Quant (e.g. Q4 → IQ4) | Decode is still far from the bandwidth ceiling, and you will publish a **new cell** |
| 3 | Tokens per cycle | Speculative decode (MTP n, p_min) | A **long** prompt grid, not a 4-token filler |
| 4 | Work per byte | Handwritten / PTX kernels | nsys points at that kernel **and** ncu says it is not already DRAM-bound |

Order: **confirm the card is boosting → clocks → quant as its own cell → long-prompt MTP → kernels last.**

On the example 2080 Ti (2026-08): the only handwritten change that landed in 27B was Gated DeltaNet **8 columns/warp** (~+1% tg, +5% pp@4k). IQ4 + clocks took tg 27.5 → 35.6. Extra FA / mmq-MMA / GDN-Tensor-Core kernels missed their kill lines.

## Shared protocol (clocks, flags, and CUDA)

A comparison is void unless all of these hold:

1. The binary prints the Turing GPU and **compute 7.5**. A Blackwell run of the same source is an `sm_120` cubin.
2. Same llama.cpp family, same weights. A new quant is a new cell — do not subtract from an old one.
3. The speed anchor is four cells: `pp512`, `tg128`, `pp512 @ d4096`, `tg128 @ d4096`, `r≥3`, with stddev. MTP is a **separate** long prompt (`n=512`) reporting tok/s **and** accept rate. Changing the paragraph makes a new cell — do not subtract from an old MTP number.
4. Trust **load** `pclk` / `mclk` / power / temp from `nvidia-smi dmon`. Do not trust Afterburner's rewritten F column. Do not trust clocks after ncu unless you passed `--clock-control none`.
5. One variable per run.
6. Write the kill line **before** the change.
7. Instability (huge σ, CUDA abort, driver reset) is a kill even if one run is faster.

Paper decode ceiling ≈ DRAM GB/s ÷ weight bytes. Q4 ~16 GB / 616 GB/s ≈ **38 tok/s** on a 2080 Ti. 35.6 is ~90% of that; more TGP will not invent the last 10%.

## Clocks

### 0. Is the GPU actually boosting?

Load `pclk` stuck near 1100 MHz: fix Windows power plan / TGP / WSL first. Do not write CUDA and do not declare a bandwidth wall.

After ncu, `pclk` pinned at the **stock Boost (1545 on reference 2080 Ti)** while memory OC remains: that is ncu clock-control, not a 215 W wall. `-rgc` / restarting Afterburner is not enough — **reboot Windows**. Always pass `--clock-control none`.

### 1. Read the walls. Do not wish.

- Many 2080 Ti VBIOS images cap at **280 W**. There is no 325 W slider to unlock in software.
- Filling the cap is not faster. Best decode on the example board was **under** the cap (230–275 W) with `pclk` locked 1800–1890. At the wall, clocks drop.
- Afterburner 4.6.x curve editor is **X = mV, Y = MHz**. Voltage is not an independent axis.

### 2. Memory first, core second

Decode streams weights. Sweep GDDR6 before core.

1. Step memory (+500, +750, …). Kill if not monotonic, if EDR/reset, or if the four-grid loses to the previous step.
2. Then core. A flat “from 800 mV hold 1800 MHz” curve is a valid intent. Afterburner will turn that into a **core slider offset** and rewrite F. Believe sliders + dmon.
3. Core kill line is **stability**, not score. On the example board, +315 / +345 aborted inside `l2_norm` / `rope` / `synchronize`.
4. Never step memory and core in the same run. Mixed runs depressed `pclk` and blew up σ.

MTP and the four-grid can disagree. After any clock or quant change, rerun **both**.

### 3. Apply

Close Afterburner, then `scripts/write-afterburner-uv.py --cfg … --mhz … --mem … --power …`. Writing while the UI is open gets overwritten. A no-args run must refuse. Publish a **range** (e.g. tg 34.6–35.6), not one peak.

Details: [`clocks.md`](clocks.md).

## Handwritten kernels

Occupancy math: [`impl-gate.md`](impl-gate.md). This section is **whether to open that gate**.

### Alignment

- `attn_sm75.cu` vs naive, vs FFMA in the same binary, SASS contains `HMMA`: proves Turing Tensor Cores work. **No D=256 / GQA / gate / DeltaNet ⇒ not Qwen3.8-27B.**
- 27B decode hotspot (nsys) is **mmvq** streaming weights, not mmq MMA and not softmax FA. Stock llama.cpp fattn/mmq already use `ldmatrix` + `mma.sync` on Turing. Another WMMA wrapper is a higher abstraction, not a lower one.
- The only 27B-aligned handwritten win here: GDN **8 columns/warp**. Chunked WMMA GDN has HMMA and is **slower**. HMMA in SASS ≠ faster.

### Amdahl

After COLS=8, GDN+FA are a few percent of 27B decode. A new D=256 FA or GDN Tensor-Core kernel cannot clear a 3% kill line on the whole model. mmq at 255 registers → 25% occupancy is geometry; a week-long rewrite only helps prefill.

### Example kill lines (measured)

| Knife | Open if | Measured | Outcome |
|---|---|---|---|
| mmvq inline PTX (LDG.128 + ILP, not HMMA) | ncu DRAM **< 85%** | ncols=1 main kernel **81%**, warp occ 87–92% | Do not ship. Maybe +2–5% research. |
| D=256 FA / GDN-TC / raw SASS | — | — | Closed |
| Flash 325 W | VBIOS has that limit | Typical 2080 Ti max 280 W | Closed |
| `n_draft=5`, DSpark, extra TGP | four-grid or long MTP beats current | Did not | Closed |

“We have not aligned to 27B / we should go lower-level”: the FA demo is not aligned; the hotspot measurement and GDN are. Legal floor is inline PTX — stock is already there. Below that is SASS. Don't.

## What you see → what you touch

| Symptom | Touch | Do not |
|---|---|---|
| Load `pclk` ~1100 | Power / TGP / WSL | CUDA, “bandwidth wall” |
| After ncu, `pclk` 1545, `mclk` still OC | Reboot; next ncu `--clock-control none` | More slider |
| `mclk` not at this OC step | Afterburner did not apply | Reuse old four-grid |
| Want 325 W | Stop if VBIOS max is 280 | Flash BIOS “for science” |
| At 280 W, `pclk` falling | Back off, lock clocks | More watts |
| CUDA abort at a core offset | Previous stable offset | “one more step” |
| FA lab 3.6 TFLOP/s, SASS has HMMA | Record in [`kernels.md`](kernels.md) | Quote as 27B tok/s |
| nsys decode = mmvq, DRAM 81% | Stop (or research only) | Rewrite FA / mmq MMA |
| Want tok/s | Quant + clocks | Another kernel |
| Want quality | Heavier quant; clocks can stay | Mix IQ4 SWE with Q4 |

## Reproduce

```bash
scripts/llama-bench-grid.sh /path/to/model.gguf
```
