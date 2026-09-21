# Closed: hand-written CUDA on this card

**2026-08-20. Status: closed, archived.**

This document is the verdict. It was written the day after this repository was first
published, which is why the rest of the docs still read as if the work were open. It is not.
Do not open "can we hand-write more of this?" as the next knife.

The experiment logs behind it stay as they are: [`kernels.md`](kernels.md),
[`results.md`](results.md), [`ggml-turing.md`](ggml-turing.md), [`method.md`](method.md).

## One line

27B decode was measured down to **`mul_mat_vec_q` sweeping weights**, bounded by
**616 GB/s**. What was worth writing was written and what was worth measuring was measured.
The remaining `mmvq` PTX was not written because the ceiling and the ncu profile had already
put the expected gain **below the delivery bar (≥3%)** — not because it is still open.

## Reproducible, and not overturned

Everything here is the same card, the same 27B, with the binary printing `RTX 2080 Ti` and
`sm_75`. Not another machine, not a 4B stand-in.

| Claim | Evidence | How to re-run |
|---|---|---|
| Decode is not FA, not GDN, not mmq | nsys tg128: the weight kernels are all `mul_mat_vec_q` ncols=1; GDN 1.1%, FA 0.5% | `nsys` on `llama-bench -p 0 -n 128` |
| Another softmax FA / D=256 buys nothing whole-model | pp512 FA **0.9%**; only a 4k prefill reaches 3.4% | same, `-p 512` and `-p 512 -d 4096` |
| GDN via WMMA/HMMA is *slower* | lab T=512: wmma chunk **5.0 ms** vs occ8 **0.91 ms** (SASS shows HMMA) | `make gdn && ./build/gdn_sm75 512` |
| `COLS=8` is the only hand-written kernel that reached 27B | pp@4k **622 → 653**; tg essentially unchanged | same binary family, four-cell grid |
| mmq at I=64 to raise occupancy | CUDA abort on this box; geometrically still 25% occupancy without rewriting `write_back_mma` | knife 1 in the nsys log |
| mmvq warps 2→4 | tg **26.74** vs 27.85 — slower, reverted | same |
| The paper decode wall | 616 GB/s ÷ ~16 GB ≈ **38 t/s**; IQ4+OC measures **35.55** (~93%) | four-cell grid + `dmon` under load |

## Why it closes without the PTX head-to-head

The `mmvq` inline-PTX idea (`LDG.128` + ILP) never got a four-cell comparison. The reason is
not that it lost:

1. ncu on the ncols=1 main kernel: DRAM **81%**, warp occupancy **87–92%** (IQ4 FFN).
   Decode's M=1 *is* ncols=1, so the ncols=4 path never being hit is not an argument for
   opening the case.
2. Even deleting **all** of that kernel's remaining 19% non-DRAM time caps the in-kernel gain
   at about `1/0.81 ≈ 1.23×`, and that does **not** convert 1:1 into tok/s — there is still
   the graph, the other kernels, and the host.
3. Whole-model performance is already at ~93% of the paper wall. The expectation on the table
   was **+2–5%**, under the **≥3% and passing the four-cell kill line** bar this project used
   throughout.

## What actually moved the number

For anyone arriving hoping hand-written kernels are the lever: on this card they were worth
about **1%**. Clocks and quantisation were worth **27.5 → 35.6 tok/s**. The ordering in
[`method.md`](method.md) is not a stylistic preference; it is the measured ordering.

## What would reopen it

A different bottleneck, not a better kernel. Concretely: a decode path that stops being
`ncols=1` weight-sweeping — a larger batch, a different speculative-decode shape that makes
the verify step wide, or a quantisation whose dequant cost (not bandwidth) dominates. Any of
those changes the profile this verdict rests on. Re-profiling is the entry condition, and
that needs ncu, which is blocked under WSL here (`ERR_NVGPUCTRPERM`).
