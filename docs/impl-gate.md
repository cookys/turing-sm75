# Gate before writing a kernel

Do not edit `.cu` until this page is filled in the reply, not just in your head.
Chip book: [`tu102.md`](tu102.md). Whether this knife should even be CUDA: [`method.md`](method.md).

The failure mode this catches: write WMMA first, measure second, discover occupancy was dead on paper.

## A. Read

- [ ] [`method.md`](method.md) — clocks vs CUDA
- [ ] [`tu102.md`](tu102.md) — SMSP registers, HMMA, forbidden Ampere copies
- [ ] [`kernels.md`](kernels.md) — existing kernels and numbers
- [ ] If the target is Qwen3.8-27B: [`qwen38-27b.md`](qwen38-27b.md) + [`ggml-turing.md`](ggml-turing.md)

## B. One sentence

Shape (B,H,T,D or Sv/Hq/Hv), current fastest opponent, success rule (same card, same shape, error `< 0.02`, faster than opponent).

Do not rewrite something stock llama.cpp already runs as Turing MMA (fattn D=256, mmq).

## C. Occupancy on paper (required)

```
threads / block      =
warps / block        = threads / 32
regs / thread (est.) =
smem / block         =

SMSP:  32 × regs × ceil(warps_per_block / 4)  =        ≤ 16384 ?
SM:    active warps                             =        / 32 =     % occupancy
ILP:   independent FMAs per warp                =        (guide: 4 to hide 4-cycle FMA with 4 warps)
```

If it fails, change the design. Do not wait for ptxas.

| Want | Need |
|---|---|
| 100% occupancy (32 warps) | 4-warp block and ≤64 regs, or equivalent |
| 4-way ILP | ≥4 independent columns/warp on a serial kernel |
| Tensor Cores worth it | still ≥8 warps/SMSP, GEMM N,K not a skinny 16 |
| 254–255 regs | 2 warps/SMSP → 25% occ. Keep only if measured faster than occ4 |

## D. Legal on Turing?

- [ ] No `cp.async` / `LDGSTS` / TMA / `wgmma` / BF16 MMA / `m16n8k16` / L2 persisting
- [ ] Tensor path is `ldmatrix` + `mma.sync.m16n8k8` (or WMMA, which is two `HMMA.1688`)
- [ ] After compile: `cuobjdump -sass` on **sm_75** cubin. `HMMA` required to call it a Tensor Core kernel. HMMA ≠ faster.
