// attn_sm75.cu
//
// Tiled online-softmax attention that is legal on Turing (sm_75 / 2080 Ti).
//
// Allowed on Turing:
//   fp16 loads, CUDA-core FMA, WMMA / mma.sync.m16n8k8 (HMMA), smem tiling,
//   online softmax.
//
// Deliberately NOT used (this is why shipping LLM kernels skip the 2080 Ti):
//   cp.async / TMA          — sm_80 / sm_90
//   native bf16 MMA         — sm_80+
//   wgmma / TMEM / TCGEN05  — Hopper / Blackwell
//   FA2 / FA3 CUDA paths    — compiled for sm_80+ only
//   FP8 / NVFP4             — Ada / Blackwell
//
// The algorithm is FlashAttention. The kernel is not the FA2/FA3 kernel.
//
//   nvcc -O3 -std=c++17 -gencode arch=compute_75,code=sm_75 --cudart=static \
//     src/attn_sm75.cu -o build/attn_sm75
//
//   ./build/attn_sm75
//   ./build/attn_sm75 2 8 512 128

#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int kBr = 16;
constexpr int kBc = 16;
constexpr int kWmma = 16;
constexpr int kNwarps = 4;
constexpr int kThreadsWmma = kNwarps * 32;

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err = (expr);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(err));                                  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Naive attention used only as a correctness oracle.
// It recomputes every QK dot inside the D loop — slower than a real O(S^2 D)
// framework fallback. Do not quote its milliseconds as "the CPU/CUDA fallback".
// ---------------------------------------------------------------------------
__global__ void naive_attn_kernel(const half* __restrict__ Q,
                                  const half* __restrict__ K,
                                  const half* __restrict__ V,
                                  half* __restrict__ O,
                                  int S, int D, float scale, int causal) {
    const int bh = blockIdx.x;
    const int q = threadIdx.x;
    if (q >= S) return;

    const size_t off = (size_t)bh * S * D;
    const half* q_row = Q + off + (size_t)q * D;

    float m = -INFINITY;
    for (int k = 0; k < S; ++k) {
        if (causal && k > q) break;
        const half* k_row = K + off + (size_t)k * D;
        float dot = 0.f;
        for (int d = 0; d < D; ++d) {
            dot += __half2float(q_row[d]) * __half2float(k_row[d]);
        }
        m = fmaxf(m, dot * scale);
    }

    for (int d = 0; d < D; ++d) {
        float acc = 0.f;
        float l = 0.f;
        for (int k = 0; k < S; ++k) {
            if (causal && k > q) break;
            const half* k_row = K + off + (size_t)k * D;
            float dot = 0.f;
            for (int dd = 0; dd < D; ++dd) {
                dot += __half2float(q_row[dd]) * __half2float(k_row[dd]);
            }
            const float p = expf(dot * scale - m);
            l += p;
            acc += p * __half2float(V[off + (size_t)k * D + d]);
        }
        O[off + (size_t)q * D + d] = __float2half(acc / l);
    }
}

// ---------------------------------------------------------------------------
// Tiled online softmax. FlashAttention math, Turing hardware.
//
// Grid : (ceil(S / Br), B*H)
// Block: Br * Bc threads. Thread (i, j) owns score / P of the current tile.
//
// Per Q-tile we keep running (m, l, O) in smem and stream K/V tiles through.
// The S matrix never lands in HBM. That is the whole point of the algorithm.
//
// Inner GEMM is a CUDA-core fp32 dot. Kept as the same-binary baseline.
// ---------------------------------------------------------------------------
template <int D>
__global__ void flash_attn_sm75_ffma(const half* __restrict__ Q,
                                const half* __restrict__ K,
                                const half* __restrict__ V,
                                half* __restrict__ O,
                                int S, float scale, int causal) {
    const int q_tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int tid = threadIdx.x;
    const int i = tid / kBc;
    const int j = tid % kBc;
    const int q0 = q_tile * kBr;
    const int qi = q0 + i;
    const size_t head_off = (size_t)bh * (size_t)S * D;

    extern __shared__ char smem_ffma[];
    half* Qs = reinterpret_cast<half*>(smem_ffma);
    half* Ks = Qs + kBr * D;
    half* Vs = Ks + kBc * D;
    float* Ss = reinterpret_cast<float*>(Vs + kBc * D);

    __shared__ float m_i[kBr];
    __shared__ float l_i[kBr];
    __shared__ float alpha_i[kBr];
    __shared__ float Oacc[kBr * D];

    for (int e = tid; e < kBr * D; e += blockDim.x) {
        const int r = e / D;
        const int c = e % D;
        const int gq = q0 + r;
        Qs[e] = (gq < S) ? Q[head_off + (size_t)gq * D + c] : __float2half(0.f);
        Oacc[e] = 0.f;
    }
    if (i < kBr && j == 0) {
        m_i[i] = -INFINITY;
        l_i[i] = 0.f;
        alpha_i[i] = 0.f;
    }
    __syncthreads();

    const int num_kv = (S + kBc - 1) / kBc;
    for (int kb = 0; kb < num_kv; ++kb) {
        const int k0 = kb * kBc;
        if (causal && k0 > q0 + (kBr - 1)) break;

        for (int e = tid; e < kBc * D; e += blockDim.x) {
            const int r = e / D;
            const int c = e % D;
            const int gk = k0 + r;
            const half z = __float2half(0.f);
            Ks[e] = (gk < S) ? K[head_off + (size_t)gk * D + c] : z;
            Vs[e] = (gk < S) ? V[head_off + (size_t)gk * D + c] : z;
        }
        __syncthreads();

        if (i < kBr && j < kBc) {
            const half* qrow = Qs + i * D;
            const half* krow = Ks + j * D;
            float dot = 0.f;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                dot += __half2float(qrow[d]) * __half2float(krow[d]);
            }
            dot *= scale;
            const int gk = k0 + j;
            if (qi >= S || gk >= S || (causal && gk > qi)) dot = -INFINITY;
            Ss[i * kBc + j] = dot;
        }
        __syncthreads();

        if (i < kBr && j == 0) {
            float tile_max = -INFINITY;
            for (int c = 0; c < kBc; ++c) {
                tile_max = fmaxf(tile_max, Ss[i * kBc + c]);
            }
            const float m_old = m_i[i];
            const float m_new = fmaxf(m_old, tile_max);
            // exp(-inf - x) is defined as 0; isfinite(-inf) is false.
            const float alpha = isfinite(m_old) ? expf(m_old - m_new) : 0.f;
            m_i[i] = m_new;
            alpha_i[i] = alpha;
        }
        __syncthreads();

        // Rescale the running output: O *= exp(m_old - m_new)
        if (i < kBr) {
            const float alpha = alpha_i[i];
            for (int d = j; d < D; d += kBc) {
                Oacc[i * D + d] *= alpha;
            }
        }
        __syncthreads();

        if (i < kBr && j < kBc) {
            const float s = Ss[i * kBc + j];
            Ss[i * kBc + j] = isfinite(s) ? expf(s - m_i[i]) : 0.f;
        }
        __syncthreads();

        if (i < kBr && j == 0) {
            float rowsum = 0.f;
            for (int c = 0; c < kBc; ++c) rowsum += Ss[i * kBc + c];
            l_i[i] = alpha_i[i] * l_i[i] + rowsum;
        }
        __syncthreads();

        // O[i, d] += sum_j P[i, j] * V[j, d]   — no atomics, j-stride over D
        if (i < kBr) {
            for (int d = j; d < D; d += kBc) {
                float acc = 0.f;
                for (int c = 0; c < kBc; ++c) {
                    acc += Ss[i * kBc + c] * __half2float(Vs[c * D + d]);
                }
                Oacc[i * D + d] += acc;
            }
        }
        __syncthreads();
    }

    if (i < kBr && qi < S) {
        const float inv_l = (l_i[i] > 0.f) ? (1.f / l_i[i]) : 0.f;
        for (int d = j; d < D; d += kBc) {
            O[head_off + (size_t)qi * D + d] =
                __float2half(Oacc[i * D + d] * inv_l);
        }
    }
}

// ---------------------------------------------------------------------------
// Same tiles / online softmax, QK^T and PV via WMMA 16x16x16 (HMMA on sm_75).
// 4 warps split the head dim. No cp.async / TMA / wgmma.
// ---------------------------------------------------------------------------
template <int D>
__global__ void flash_attn_sm75_wmma(const half* __restrict__ Q,
                                     const half* __restrict__ K,
                                     const half* __restrict__ V,
                                     half* __restrict__ O,
                                     int S, float scale, int causal) {
    using namespace nvcuda::wmma;
    static_assert(kBr == kWmma && kBc == kWmma, "one 16x16 WMMA per Q/K tile");
    static_assert(D % (kWmma * kNwarps) == 0, "each warp owns a 16-multiple of D");

    const int q_tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int q0 = q_tile * kBr;
    const size_t head_off = (size_t)bh * (size_t)S * D;
    const int d0 = warp * (D / kNwarps);
    const int d1 = d0 + D / kNwarps;

    extern __shared__ __align__(16) char smem_wmma[];
    half* Qs = reinterpret_cast<half*>(smem_wmma);
    half* Ks = Qs + kBr * D;
    half* Vs = Ks + kBc * D;
    half* Ph = Vs + kBc * D;
    float* Ss = reinterpret_cast<float*>(Ph + kBr * kBc);
    float* Sp = Ss + kBr * kBc;

    __shared__ float m_i[kBr];
    __shared__ float l_i[kBr];
    __shared__ float alpha_i[kBr];
    __shared__ float Oacc[kBr * D];

    for (int e = tid; e < kBr * D; e += blockDim.x) {
        const int r = e / D;
        const int c = e % D;
        const int gq = q0 + r;
        Qs[e] = (gq < S) ? Q[head_off + (size_t)gq * D + c] : __float2half(0.f);
        Oacc[e] = 0.f;
    }
    for (int r = tid; r < kBr; r += blockDim.x) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.f;
        alpha_i[r] = 0.f;
    }
    __syncthreads();

    const int num_kv = (S + kBc - 1) / kBc;
    for (int kb = 0; kb < num_kv; ++kb) {
        const int k0 = kb * kBc;
        if (causal && k0 > q0 + (kBr - 1)) break;

        for (int e = tid; e < kBc * D; e += blockDim.x) {
            const int r = e / D;
            const int c = e % D;
            const int gk = k0 + r;
            const half z = __float2half(0.f);
            Ks[e] = (gk < S) ? K[head_off + (size_t)gk * D + c] : z;
            Vs[e] = (gk < S) ? V[head_off + (size_t)gk * D + c] : z;
        }
        __syncthreads();

        fragment<matrix_a, kWmma, kWmma, kWmma, half, row_major> a_frag;
        fragment<matrix_b, kWmma, kWmma, kWmma, half, col_major> b_frag;
        fragment<accumulator, kWmma, kWmma, kWmma, float> s_frag;
        fill_fragment(s_frag, 0.f);
        for (int d = d0; d < d1; d += kWmma) {
            load_matrix_sync(a_frag, Qs + d, D);
            load_matrix_sync(b_frag, Ks + d, D);
            mma_sync(s_frag, a_frag, b_frag, s_frag);
        }
        store_matrix_sync(Sp + warp * kBr * kBc, s_frag, kBc, mem_row_major);
        __syncthreads();

        for (int e = tid; e < kBr * kBc; e += blockDim.x) {
            float s = 0.f;
            for (int w = 0; w < kNwarps; ++w) s += Sp[w * kBr * kBc + e];
            s *= scale;
            const int i = e / kBc;
            const int j = e % kBc;
            const int qi = q0 + i;
            const int gk = k0 + j;
            if (qi >= S || gk >= S || (causal && gk > qi)) s = -INFINITY;
            Ss[e] = s;
        }
        __syncthreads();

        for (int i = tid; i < kBr; i += blockDim.x) {
            float tile_max = -INFINITY;
            for (int c = 0; c < kBc; ++c) tile_max = fmaxf(tile_max, Ss[i * kBc + c]);
            const float m_old = m_i[i];
            const float m_new = fmaxf(m_old, tile_max);
            const float alpha = isfinite(m_old) ? expf(m_old - m_new) : 0.f;
            m_i[i] = m_new;
            alpha_i[i] = alpha;
        }
        __syncthreads();

        for (int e = tid; e < kBr * D; e += blockDim.x) {
            Oacc[e] *= alpha_i[e / D];
        }
        for (int e = tid; e < kBr * kBc; e += blockDim.x) {
            const float s = Ss[e];
            const float p = isfinite(s) ? expf(s - m_i[e / kBc]) : 0.f;
            Ss[e] = p;
            Ph[e] = __float2half(p);
        }
        __syncthreads();

        for (int i = tid; i < kBr; i += blockDim.x) {
            float rowsum = 0.f;
            for (int c = 0; c < kBc; ++c) rowsum += Ss[i * kBc + c];
            l_i[i] = alpha_i[i] * l_i[i] + rowsum;
        }
        __syncthreads();

        fragment<matrix_a, kWmma, kWmma, kWmma, half, row_major> p_frag;
        fragment<matrix_b, kWmma, kWmma, kWmma, half, row_major> v_frag;
        fragment<accumulator, kWmma, kWmma, kWmma, float> o_frag;
        load_matrix_sync(p_frag, Ph, kBc);
        for (int d = d0; d < d1; d += kWmma) {
            load_matrix_sync(o_frag, Oacc + d, D, mem_row_major);
            load_matrix_sync(v_frag, Vs + d, D);
            mma_sync(o_frag, p_frag, v_frag, o_frag);
            store_matrix_sync(Oacc + d, o_frag, D, mem_row_major);
        }
        __syncthreads();
    }

    for (int e = tid; e < kBr * D; e += blockDim.x) {
        const int i = e / D;
        const int d = e % D;
        const int qi = q0 + i;
        if (qi >= S) continue;
        const float inv_l = (l_i[i] > 0.f) ? (1.f / l_i[i]) : 0.f;
        O[head_off + (size_t)qi * D + d] = __float2half(Oacc[e] * inv_l);
    }
}

static size_t wmma_dyn_smem(int D) {
    return (size_t)(kBr + kBc + kBc) * D * sizeof(half) +
           (size_t)kBr * kBc * sizeof(half) +
           (size_t)kBr * kBc * sizeof(float) +
           (size_t)kNwarps * kBr * kBc * sizeof(float);
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------
struct Tensor {
    half* d = nullptr;
    std::vector<half> h;
    size_t n = 0;
};

static Tensor make_tensor(size_t n, bool randomize, std::mt19937& rng) {
    Tensor t;
    t.n = n;
    t.h.resize(n);
    if (randomize) {
        std::normal_distribution<float> dist(0.f, 0.5f);
        for (size_t i = 0; i < n; ++i) t.h[i] = __float2half(dist(rng));
    } else {
        std::fill(t.h.begin(), t.h.end(), __float2half(0.f));
    }
    CUDA_CHECK(cudaMalloc(&t.d, n * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(t.d, t.h.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    return t;
}

static void free_tensor(Tensor& t) {
    if (t.d) CUDA_CHECK(cudaFree(t.d));
    t.d = nullptr;
}

static float max_abs_err(const std::vector<half>& a, const std::vector<half>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, fabsf(__half2float(a[i]) - __half2float(b[i])));
    }
    return m;
}

template <int D>
static void launch_ffma(const half* Q, const half* K, const half* V, half* O,
                        int B, int H, int S, float scale, int causal) {
    dim3 grid((S + kBr - 1) / kBr, B * H);
    dim3 block(kBr * kBc);
    const size_t smem =
        (size_t)(kBr + kBc + kBc) * D * sizeof(half) + (size_t)kBr * kBc * sizeof(float);
    flash_attn_sm75_ffma<D><<<grid, block, smem>>>(Q, K, V, O, S, scale, causal);
    CUDA_CHECK(cudaGetLastError());
}

template <int D>
static void launch_wmma(const half* Q, const half* K, const half* V, half* O,
                        int B, int H, int S, float scale, int causal) {
    dim3 grid((S + kBr - 1) / kBr, B * H);
    dim3 block(kThreadsWmma);
    flash_attn_sm75_wmma<D><<<grid, block, wmma_dyn_smem(D)>>>(Q, K, V, O, S,
                                                               scale, causal);
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
    int B = 2, H = 8, S = 512, D = 128;
    if (argc == 5) {
        B = std::atoi(argv[1]);
        H = std::atoi(argv[2]);
        S = std::atoi(argv[3]);
        D = std::atoi(argv[4]);
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [B H S D]\n", argv[0]);
        return 2;
    }
    if (D != 64 && D != 128) {
        fprintf(stderr, "demo instantiates D=64 and D=128 only\n");
        return 2;
    }
    if (S > 1024) {
        fprintf(stderr, "naive check is O(S^2 D); keep S <= 1024\n");
        return 2;
    }

    const float scale = 1.f / std::sqrt(static_cast<float>(D));
    const int causal = 1;
    const size_t elems = (size_t)B * H * S * D;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU             : %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("shape           : B=%d H=%d S=%d D=%d  causal=1\n", B, H, S, D);
    printf("tile            : Br=%d Bc=%d  wmma %dx%dx%d  warps=%d  (no cp.async, no TMA)\n",
           kBr, kBc, kWmma, kWmma, kWmma, kNwarps);

    std::mt19937 rng(42);
    Tensor Q = make_tensor(elems, true, rng);
    Tensor K = make_tensor(elems, true, rng);
    Tensor V = make_tensor(elems, true, rng);
    Tensor Ow = make_tensor(elems, false, rng);
    Tensor Of = make_tensor(elems, false, rng);
    Tensor On = make_tensor(elems, false, rng);

    auto wmma = [&](half* dst) {
        if (D == 64) launch_wmma<64>(Q.d, K.d, V.d, dst, B, H, S, scale, causal);
        else launch_wmma<128>(Q.d, K.d, V.d, dst, B, H, S, scale, causal);
    };
    auto ffma = [&](half* dst) {
        if (D == 64) launch_ffma<64>(Q.d, K.d, V.d, dst, B, H, S, scale, causal);
        else launch_ffma<128>(Q.d, K.d, V.d, dst, B, H, S, scale, causal);
    };

    wmma(Ow.d);
    ffma(Of.d);
    naive_attn_kernel<<<B * H, S>>>(Q.d, K.d, V.d, On.d, S, D, scale, causal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(Ow.h.data(), Ow.d, elems * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Of.h.data(), Of.d, elems * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(On.h.data(), On.d, elems * sizeof(half), cudaMemcpyDeviceToHost));

    const float err_w = max_abs_err(Ow.h, On.h);
    const float err_f = max_abs_err(Of.h, On.h);
    printf("max |wmma-naive| : %.5f   %s\n", err_w, err_w < 2e-2f ? "OK" : "FAIL");
    printf("max |ffma-naive| : %.5f   %s\n", err_f, err_f < 2e-2f ? "OK" : "FAIL");

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    const int iters = 20;

    auto bench = [&](auto&& fn, half* dst) {
        CUDA_CHECK(cudaEventRecord(ev0));
        for (int t = 0; t < iters; ++t) fn(dst);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        return ms / iters;
    };

    const float wmma_ms = bench(wmma, Ow.d);
    const float ffma_ms = bench(ffma, Of.d);
    float naive_ms = 0.f;
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int t = 0; t < iters; ++t) {
        naive_attn_kernel<<<B * H, S>>>(Q.d, K.d, V.d, On.d, S, D, scale, causal);
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, ev0, ev1));
    naive_ms /= iters;

    // Dense-equivalent 4*B*H*S*S*D (what older notes quoted as 3.6 TFLOP/s).
    // Causal executed work is ~2*B*H*S*(S+1)*D — about half at these shapes.
    const double flops_dense = 4.0 * B * H * (double)S * S * D;
    const double flops_exec = 2.0 * B * H * (double)S * (S + 1.0) * D;
    auto tflops = [&](double flops, float ms) { return flops / (ms * 1e-3) / 1e12; };
    printf("wmma  (HMMA)     : %7.3f ms   %6.1f TFLOP/s dense-eq   %6.1f causal-exec\n",
           wmma_ms, tflops(flops_dense, wmma_ms), tflops(flops_exec, wmma_ms));
    printf("ffma  (CUDA core): %7.3f ms   %6.1f TFLOP/s dense-eq   %6.1f causal-exec\n",
           ffma_ms, tflops(flops_dense, ffma_ms), tflops(flops_exec, ffma_ms));
    printf("naive oracle     : %7.3f ms   (not a realistic fallback)\n", naive_ms);
    printf("wmma vs ffma     : %.2fx  (same FLOP convention both sides)\n", ffma_ms / wmma_ms);
    printf("\n"
           "Same recurrence FA2 uses. Not the FA2 kernel. Not a 27B tok/s number.\n"
           "WMMA 16x16x16 is the Turing-legal TC path. No cp.async / TMA / wgmma.\n"
           "Published 3.6 TFLOP/s is dense-equivalent; causal-exec is ~half.\n");

    free_tensor(Q);
    free_tensor(K);
    free_tensor(V);
    free_tensor(Ow);
    free_tensor(Of);
    free_tensor(On);
    return err_w < 2e-2f ? 0 : 1;
}
