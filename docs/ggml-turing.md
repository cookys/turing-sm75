# What stock llama.cpp already does on sm_75

Do not rewrite a path that is already Turing MMA.

On a `CMAKE_CUDA_ARCHITECTURES=75` build of `libggml-cuda.so`, `cuobjdump -sass` shows tens of thousands of `HMMA`/`IMMA` and **zero** `LDGSTS`. It is already a Tensor Core library.

## Gated attention (16/64 layers of Qwen3.8-27B, D=256)

`fattn.cu` takes `BEST_FATTN_KERNEL_MMA_F16` when `turing_mma_available(cc)`. There is a `DKQ=256, DV=256` MMA instance. Turing cap: `ncols1*ncols2 <= 32`.

A handwritten D=256 softmax FA is not the first 27B knife. On this card `fa on` vs `off` is ~+6% at 4k prefill and ~nothing on short pp / decode.

## Gated DeltaNet (48/64 layers, D=128)

Stock `gated_delta_net.cu` is fp32, token-serial, warp reduce. No HMMA. There is an upstream TODO for a chunked prefill kernel.

This repo's lab (`gdn_sm75.cu`) compared column occupancy vs chunked WMMA. **8 columns/warp** won. That is a one-line change (`kCols=8`) in ggml, not a new MMA kernel. Chunked WMMA was slower here.

## Q4 GEMM

`mmq.cu` also has `turing_mma_available`. Decode mostly lives here plus DRAM, not in another D=128 softmax.

| Idea | Worth it? |
|---|---|
| Another D=128 softmax like `attn_sm75.cu` | Almost no 27B effect |
| D=256 softmax “to replace FA2” | Stock fattn is already Turing MMA |
| GDN column occupancy | Yes — measured |
| Chunked GDN WMMA | Measured slower than occ8 on this card |
| mmq full remap | Prefill-only, 255-reg occupancy wall |
