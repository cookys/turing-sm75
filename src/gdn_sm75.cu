// gdn_sm75.cu
//
// Gated DeltaNet matching llama.cpp gated_delta_net.cu (KDA=false).
// Shape is Qwen3.8-27B: S=128, 16 QK heads, 48 V heads.
//
//   g = exp(g_log)
//   kv   = S^T k
//   d    = (v - g * kv) * beta
//   S    = g * S + k ⊗ d
//   out  = scale * S^T q
// V-head h reads Q/K head (h % 16).
//
// Chunked path follows ggml build_delta_net_chunking (non-KDA):
//   CS=16 here (WMMA tile). ggml graph uses CS=64; the algebra is the same.
//   decay / kb / kq are UPPER in ggml (ne0,ne1) layout (tri LOWER keeps i<=j).
//   solve_tri is the ggml CPU loop: X[r] uses A[t,r] for t<r.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int kSv = 128;
constexpr int kHq = 16;
constexpr int kHv = 48;
constexpr int kCs = 16;
constexpr int kWarp = 32;
constexpr int kWarps = 4;
constexpr int kRowsLane = kSv / kWarp; // 4

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err = (expr);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(err));                                  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        x += __shfl_xor_sync(0xffffffff, x, m);
    return x;
}

// ---------------------------------------------------------------------------
// Serial: one block = one V-head, 128 threads, thread c owns column c.
// Spills (~255 regs). Kept as the naive-on-device reference.
// ---------------------------------------------------------------------------
__global__ void gdn_serial_kernel(const float* __restrict__ q,
                                  const float* __restrict__ k,
                                  const float* __restrict__ v,
                                  const float* __restrict__ g_log,
                                  const float* __restrict__ beta,
                                  float* __restrict__ state,
                                  float* __restrict__ out,
                                  int T, float scale) {
    const int h = blockIdx.x;
    const int c = threadIdx.x;
    if (c >= kSv) return;
    const int hq = h % kHq;

    float s_col[kSv];
    float* Sh = state + (size_t)h * kSv * kSv;
    for (int i = 0; i < kSv; ++i) s_col[i] = Sh[c * kSv + i];

    for (int t = 0; t < T; ++t) {
        const float* qt = q + ((size_t)t * kHq + hq) * kSv;
        const float* kt = k + ((size_t)t * kHq + hq) * kSv;
        const float* vt = v + ((size_t)t * kHv + h) * kSv;
        const float g = expf(g_log[t * kHv + h]);
        const float b = beta[t * kHv + h];

        float kv = 0.f;
        for (int i = 0; i < kSv; ++i) kv += s_col[i] * kt[i];
        const float d = (vt[c] - g * kv) * b;
        float attn = 0.f;
        for (int i = 0; i < kSv; ++i) {
            s_col[i] = g * s_col[i] + kt[i] * d;
            attn += s_col[i] * qt[i];
        }
        out[((size_t)t * kHv + h) * kSv + c] = attn * scale;
    }
    for (int i = 0; i < kSv; ++i) Sh[c * kSv + i] = s_col[i];
}

// Same math, S in smem so we do not spill 128 floats per thread.
__global__ void gdn_smem_kernel(const float* __restrict__ q,
                                const float* __restrict__ k,
                                const float* __restrict__ v,
                                const float* __restrict__ g_log,
                                const float* __restrict__ beta,
                                float* __restrict__ state,
                                float* __restrict__ out,
                                int T, float scale) {
    const int h = blockIdx.x;
    const int c = threadIdx.x;
    if (c >= kSv) return;
    const int hq = h % kHq;

    __shared__ half Ss[kSv * kSv];
    float* Sh = state + (size_t)h * kSv * kSv;
    for (int i = 0; i < kSv; ++i) Ss[c * kSv + i] = __float2half(Sh[c * kSv + i]);
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        const float* qt = q + ((size_t)t * kHq + hq) * kSv;
        const float* kt = k + ((size_t)t * kHq + hq) * kSv;
        const float* vt = v + ((size_t)t * kHv + h) * kSv;
        const float g = expf(g_log[t * kHv + h]);
        const float b = beta[t * kHv + h];

        float kv = 0.f;
        for (int i = 0; i < kSv; ++i) kv += __half2float(Ss[c * kSv + i]) * kt[i];
        const float d = (vt[c] - g * kv) * b;
        float attn = 0.f;
        for (int i = 0; i < kSv; ++i) {
            const float s = g * __half2float(Ss[c * kSv + i]) + kt[i] * d;
            Ss[c * kSv + i] = __float2half(s);
            attn += s * qt[i];
        }
        out[((size_t)t * kHv + h) * kSv + c] = attn * scale;
        __syncthreads();
    }
    for (int i = 0; i < kSv; ++i) Sh[c * kSv + i] = __half2float(Ss[c * kSv + i]);
}

// ggml-style: 4 warps / block, 1 column per warp, 4 rows per lane. No spill.
// grid (Hv, 1, Sv/4)  block (32, 4)
__global__ void __launch_bounds__(128, 4)
gdn_occ_kernel(const float* __restrict__ q,
               const float* __restrict__ k,
               const float* __restrict__ v,
               const float* __restrict__ g_log,
               const float* __restrict__ beta,
               float* __restrict__ state,
               float* __restrict__ out,
               int T, float scale) {
    const int h = blockIdx.x;
    const int lane = threadIdx.x;
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= kSv) return;
    const int hq = h % kHq;

    float* Sh = state + ((size_t)h * kSv + col) * kSv;
    float s_shard[kRowsLane];
#pragma unroll
    for (int r = 0; r < kRowsLane; ++r) s_shard[r] = Sh[r * kWarp + lane];

    for (int t = 0; t < T; ++t) {
        const float* qt = q + ((size_t)t * kHq + hq) * kSv;
        const float* kt = k + ((size_t)t * kHq + hq) * kSv;
        const float* vt = v + ((size_t)t * kHv + h) * kSv;
        const float g = expf(g_log[t * kHv + h]);
        const float b = beta[t * kHv + h];

        float k_reg[kRowsLane];
        float q_reg[kRowsLane];
        float kv = 0.f;
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) {
            const int i = r * kWarp + lane;
            k_reg[r] = kt[i];
            q_reg[r] = qt[i];
            kv += s_shard[r] * k_reg[r];
        }
        kv = warp_sum(kv);
        const float d = (vt[col] - g * kv) * b;

        float attn = 0.f;
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) {
            s_shard[r] = g * s_shard[r] + k_reg[r] * d;
            attn += s_shard[r] * q_reg[r];
        }
        attn = warp_sum(attn);
        if (lane == 0) out[((size_t)t * kHv + h) * kSv + col] = attn * scale;
    }
#pragma unroll
    for (int r = 0; r < kRowsLane; ++r) Sh[r * kWarp + lane] = s_shard[r];
}

// Chunked: same launch as occ. Intra-chunk C×C is ggml's upper-tri form.
// Each column-block recomputes the cheap 16×16 (once per head would need extra sync).
__global__ void __launch_bounds__(128, 4)
gdn_chunked_kernel(const float* __restrict__ q,
                   const float* __restrict__ k,
                   const float* __restrict__ v,
                   const float* __restrict__ g_log,
                   const float* __restrict__ beta,
                   float* __restrict__ state,
                   float* __restrict__ out,
                   int T, float scale) {
    const int h = blockIdx.x;
    const int lane = threadIdx.x;
    const int wy = threadIdx.y;
    const int col = blockIdx.z * kWarps + wy;
    const int hq = h % kHq;
    const int tid = wy * kWarp + lane;

    float s_shard[kRowsLane];
    float* Sh = state + ((size_t)h * kSv + col) * kSv;
    if (col < kSv) {
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) s_shard[r] = Sh[r * kWarp + lane];
    }

    __shared__ float sq[kCs][kSv];
    __shared__ float sk[kCs][kSv];
    __shared__ float sv[kCs][kSv];
    __shared__ float sgl[kCs];
    __shared__ float sbt[kCs];
    __shared__ float gcs[kCs];
    __shared__ float kb[kCs * kCs];
    __shared__ float kq[kCs * kCs];
    __shared__ float attn[kCs * kCs];

    const int nch = (T + kCs - 1) / kCs;
    for (int ch = 0; ch < nch; ++ch) {
        const int t0 = ch * kCs;
        for (int t = 0; t < kCs; ++t) {
            const int tt = t0 + t;
            if (tid < kSv) {
                if (tt < T) {
                    sq[t][tid] = q[((size_t)tt * kHq + hq) * kSv + tid];
                    sk[t][tid] = k[((size_t)tt * kHq + hq) * kSv + tid];
                    sv[t][tid] = v[((size_t)tt * kHv + h) * kSv + tid];
                } else {
                    sq[t][tid] = 0.f;
                    sk[t][tid] = 0.f;
                    sv[t][tid] = 0.f;
                }
            }
        }
        if (tid < kCs) {
            const int tt = t0 + tid;
            sgl[tid] = (tt < T) ? g_log[tt * kHv + h] : 0.f;
            sbt[tid] = (tt < T) ? beta[tt * kHv + h] : 0.f;
        }
        __syncthreads();

        if (tid == 0) {
            gcs[0] = sgl[0];
            for (int t = 1; t < kCs; ++t) gcs[t] = gcs[t - 1] + sgl[t];
        }
        __syncthreads();

        // kb[i,j] = (k_i·k_j)*β_j*decay, kq same with q_j. ggml layout [i + j*CS].
        for (int e = tid; e < kCs * kCs; e += 128) {
            const int i = e % kCs;
            const int j = e / kCs;
            const float mask = (i <= j) ? expf(gcs[j] - gcs[i]) : 0.f;
            float dk = 0.f, dq = 0.f;
#pragma unroll
            for (int d = 0; d < kSv; ++d) {
                dk += sk[i][d] * sk[j][d];
                dq += sk[i][d] * sq[j][d];
            }
            kb[e] = dk * sbt[j] * mask;
            kq[e] = (i <= j) ? dq * mask : 0.f;
        }
        __syncthreads();

        if (tid == 0) {
            float lhs[kCs * kCs];
            float X[kCs * kCs];
            for (int j = 0; j < kCs; ++j) {
                for (int i = 0; i < kCs; ++i) {
                    const float a = (i < j) ? kb[i + j * kCs] : 0.f;
                    lhs[i + j * kCs] = a + (i == j ? 1.f : 0.f);
                }
            }
            for (int colj = 0; colj < kCs; ++colj) {
                for (int r = 0; r < kCs; ++r) {
                    float sum = 0.f;
                    for (int t = 0; t < r; ++t)
                        sum += lhs[t + r * kCs] * X[t + colj * kCs];
                    const float b = (r < colj) ? -kb[r + colj * kCs] : 0.f;
                    X[r + colj * kCs] = (b - sum) / lhs[r + r * kCs];
                }
            }
            for (int j = 0; j < kCs; ++j)
                for (int i = 0; i < kCs; ++i)
                    attn[i + j * kCs] = X[i + j * kCs] + (i == j ? 1.f : 0.f);
        }
        __syncthreads();

        if (col < kSv) {
            float k_dot[kCs];
            float q_dot[kCs];
#pragma unroll
            for (int t = 0; t < kCs; ++t) {
                float kd = 0.f, qd = 0.f;
#pragma unroll
                for (int r = 0; r < kRowsLane; ++r) {
                    const int i = r * kWarp + lane;
                    kd += s_shard[r] * sk[t][i];
                    qd += s_shard[r] * sq[t][i];
                }
                k_dot[t] = warp_sum(kd);
                q_dot[t] = warp_sum(qd);
            }

            float vnew[kCs];
#pragma unroll
            for (int t = 0; t < kCs; ++t) {
                float vm = 0.f, vp = 0.f;
#pragma unroll
                for (int i = 0; i < kCs; ++i) {
                    const float a = attn[i + t * kCs];
                    vm += sv[i][col] * sbt[i] * a;
                    vp += a * sbt[i] * expf(gcs[i]) * k_dot[i];
                }
                vnew[t] = vm - vp;
            }

            const float g_last = gcs[kCs - 1];
            const float g_last_e = expf(g_last);
#pragma unroll
            for (int t = 0; t < kCs; ++t) {
                const int tt = t0 + t;
                if (tt >= T) continue;
                float va = 0.f;
#pragma unroll
                for (int i = 0; i < kCs; ++i) va += vnew[i] * kq[i + t * kCs];
                const float oi = expf(gcs[t]) * q_dot[t] + va;
                if (lane == 0) out[((size_t)tt * kHv + h) * kSv + col] = oi * scale;
            }

#pragma unroll
            for (int r = 0; r < kRowsLane; ++r) {
                const int d = r * kWarp + lane;
                float add = 0.f;
#pragma unroll
                for (int t = 0; t < kCs; ++t)
                    add += sk[t][d] * expf(g_last - gcs[t]) * vnew[t];
                s_shard[r] = g_last_e * s_shard[r] + add;
            }
        }
        __syncthreads();
    }

    if (col < kSv) {
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) Sh[r * kWarp + lane] = s_shard[r];
    }
}

// One block per V-head, 8 warps. S in smem as fp16. QK/PV-style GEMMs are WMMA
// 16×16×16 (HMMA.1688 on sm_75). C×C is computed once per head per chunk.
constexpr int kWmmaWarps = 8;
constexpr int kWmmaSmem = 59520;

__global__ void __launch_bounds__(256, 1)
gdn_wmma_kernel(const float* __restrict__ q,
                const float* __restrict__ k,
                const float* __restrict__ v,
                const float* __restrict__ g_log,
                const float* __restrict__ beta,
                float* __restrict__ state,
                float* __restrict__ out,
                int T, float scale) {
    using namespace nvcuda;
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid / kWarp;
    const int lane = tid % kWarp;
    const int hq = h % kHq;

    extern __shared__ char smem[];
    half* Ss = reinterpret_cast<half*>(smem);
    half* Qg = Ss + kSv * kSv;
    half* Kcd = Qg + kSv * kCs;
    float* Cinter = reinterpret_cast<float*>(Kcd + kSv * kCs);
    float* Cvp = Cinter + kSv * kCs;
    float* attn = Cvp + kSv * kCs;
    float* kq = attn + kCs * kCs;
    float* gcs = kq + kCs * kCs;
    float* sbt = gcs + kCs;

    float* Sh = state + (size_t)h * kSv * kSv;
    for (int i = tid; i < kSv * kSv; i += 256) Ss[i] = __float2half(Sh[i]);
    __syncthreads();

    const int nch = (T + kCs - 1) / kCs;
    for (int ch = 0; ch < nch; ++ch) {
        const int t0 = ch * kCs;

        if (tid < kCs) {
            const int tt = t0 + tid;
            gcs[tid] = (tt < T) ? g_log[tt * kHv + h] : 0.f;
            sbt[tid] = (tt < T) ? beta[tt * kHv + h] : 0.f;
        }
        __syncthreads();
        if (tid == 0) {
            for (int t = 1; t < kCs; ++t) gcs[t] += gcs[t - 1];
        }
        __syncthreads();

        // 256 threads, 256 C×C entries. k/q loaded from HBM.
        {
            const int i = tid % kCs;
            const int j = tid / kCs;
            const int tti = t0 + i;
            const int ttj = t0 + j;
            const float mask = (i <= j) ? expf(gcs[j] - gcs[i]) : 0.f;
            float dk = 0.f, dq = 0.f;
            if (tti < T && ttj < T) {
                const float* ki = k + ((size_t)tti * kHq + hq) * kSv;
                const float* kj = k + ((size_t)ttj * kHq + hq) * kSv;
                const float* qj = q + ((size_t)ttj * kHq + hq) * kSv;
#pragma unroll
                for (int d = 0; d < kSv; ++d) {
                    dk += ki[d] * kj[d];
                    dq += ki[d] * qj[d];
                }
            }
            // stash kb in attn, kq in kq; solve next
            attn[i + j * kCs] = dk * sbt[j] * mask;
            kq[i + j * kCs] = (i <= j) ? dq * mask : 0.f;
        }
        __syncthreads();

        if (tid == 0) {
            float lhs[kCs * kCs];
            float X[kCs * kCs];
            for (int j = 0; j < kCs; ++j) {
                for (int i = 0; i < kCs; ++i) {
                    const float a = (i < j) ? attn[i + j * kCs] : 0.f;
                    lhs[i + j * kCs] = a + (i == j ? 1.f : 0.f);
                }
            }
            for (int colj = 0; colj < kCs; ++colj) {
                for (int r = 0; r < kCs; ++r) {
                    float sum = 0.f;
                    for (int p = 0; p < r; ++p)
                        sum += lhs[p + r * kCs] * X[p + colj * kCs];
                    const float b = (r < colj) ? -attn[r + colj * kCs] : 0.f;
                    X[r + colj * kCs] = (b - sum) / lhs[r + r * kCs];
                }
            }
            for (int j = 0; j < kCs; ++j)
                for (int i = 0; i < kCs; ++i)
                    attn[i + j * kCs] = X[i + j * kCs] + (i == j ? 1.f : 0.f);
        }
        __syncthreads();

        // Qg[i,t] = q[t][i] * exp(gcs[t]); Kcd[i,t] = sum_j k[j][i]*β[j]*e^{gcs[j]}*attn[j,t]
        for (int e = tid; e < kSv * kCs; e += 256) {
            const int i = e / kCs;
            const int t = e % kCs;
            const int tt = t0 + t;
            float qg = 0.f;
            if (tt < T) qg = q[((size_t)tt * kHq + hq) * kSv + i] * expf(gcs[t]);
            Qg[e] = __float2half(qg);

            float kc = 0.f;
#pragma unroll
            for (int j = 0; j < kCs; ++j) {
                const int tj = t0 + j;
                if (tj >= T) continue;
                kc += k[((size_t)tj * kHq + hq) * kSv + i] * sbt[j] * expf(gcs[j]) *
                      attn[j + t * kCs];
            }
            Kcd[e] = __float2half(kc);
        }
        __syncthreads();

        // Cinter = Ss @ Qg   and   Cvp = Ss @ Kcd    (128×128 @ 128×16)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_qg;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_kd;
            wmma::fill_fragment(c_qg, 0.f);
            wmma::fill_fragment(c_kd, 0.f);
            const int row0 = warp * 16;
#pragma unroll
            for (int kk = 0; kk < kSv; kk += 16) {
                wmma::load_matrix_sync(a_frag, Ss + row0 * kSv + kk, kSv);
                wmma::load_matrix_sync(b_frag, Qg + kk * kCs, kCs);
                wmma::mma_sync(c_qg, a_frag, b_frag, c_qg);
                wmma::load_matrix_sync(b_frag, Kcd + kk * kCs, kCs);
                wmma::mma_sync(c_kd, a_frag, b_frag, c_kd);
            }
            wmma::store_matrix_sync(Cinter + row0 * kCs, c_qg, kCs, wmma::mem_row_major);
            wmma::store_matrix_sync(Cvp + row0 * kCs, c_kd, kCs, wmma::mem_row_major);
        }
        __syncthreads();

        // vnew[t,col] = vmix[col,t] - Cvp[col,t];  out = Cinter + vnew^T @ kq
        // reuse Qg as half Vnew^T[col, t]
        for (int e = tid; e < kSv * kCs; e += 256) {
            const int col = e / kCs;
            const int t = e % kCs;
            float vm = 0.f;
#pragma unroll
            for (int j = 0; j < kCs; ++j) {
                const int tj = t0 + j;
                if (tj >= T) continue;
                vm += v[((size_t)tj * kHv + h) * kSv + col] * sbt[j] * attn[j + t * kCs];
            }
            const float vn = vm - Cvp[e];
            Qg[e] = __float2half(vn); // Vnew^T
            Cvp[e] = vn;              // keep fp32 for the output saxpy
        }
        __syncthreads();

        for (int e = tid; e < kSv * kCs; e += 256) {
            const int col = e / kCs;
            const int t = e % kCs;
            const int tt = t0 + t;
            if (tt >= T) continue;
            float va = 0.f;
#pragma unroll
            for (int j = 0; j < kCs; ++j) va += Cvp[col * kCs + j] * kq[j + t * kCs];
            out[((size_t)tt * kHv + h) * kSv + col] = (Cinter[e] + va) * scale;
        }

        // Kg[t, i] into Kcd as [t, i] row-major 16×128, then S = g_last S + Vnew^T @ Kg
        const float g_last = gcs[kCs - 1];
        const float g_last_e = expf(g_last);
        for (int e = tid; e < kCs * kSv; e += 256) {
            const int t = e / kSv;
            const int i = e % kSv;
            const int tt = t0 + t;
            float kg = 0.f;
            if (tt < T)
                kg = k[((size_t)tt * kHq + hq) * kSv + i] * expf(g_last - gcs[t]);
            Kcd[e] = __float2half(kg);
        }
        __syncthreads();

        // scale S by g_last in place (fp16)
        for (int i = tid; i < kSv * kSv; i += 256)
            Ss[i] = __float2half(g_last_e * __half2float(Ss[i]));
        __syncthreads();

        // Ss += Qg(Vnew^T 128×16) @ Kcd(Kg 16×128). Each warp owns 16 rows;
        // Cinter[warp] holds that warp's 16×16 tile (8*256 = 2048 floats).
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            const int row0 = warp * 16;
#pragma unroll
            for (int nt = 0; nt < kSv; nt += 16) {
                wmma::fill_fragment(c_frag, 0.f);
                wmma::load_matrix_sync(a_frag, Qg + row0 * kCs, kCs);
                wmma::load_matrix_sync(b_frag, Kcd + nt, kSv);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                wmma::store_matrix_sync(Cinter + warp * 256, c_frag, 16, wmma::mem_row_major);
                __syncthreads();
                for (int e = tid; e < kWmmaWarps * 256; e += 256) {
                    const int w = e / 256;
                    const int r = (e % 256) / 16;
                    const int c = (e % 256) % 16;
                    const int dst = (w * 16 + r) * kSv + nt + c;
                    Ss[dst] = __float2half(__half2float(Ss[dst]) + Cinter[e]);
                }
                __syncthreads();
            }
        }
    }

    for (int i = tid; i < kSv * kSv; i += 256) Sh[i] = __half2float(Ss[i]);
}

// Two columns per warp — same math as occ, more ILP. grid.z = Sv/8.
__global__ void __launch_bounds__(128, 8)
gdn_occ2_kernel(const float* __restrict__ q,
                const float* __restrict__ k,
                const float* __restrict__ v,
                const float* __restrict__ g_log,
                const float* __restrict__ beta,
                float* __restrict__ state,
                float* __restrict__ out,
                int T, float scale) {
    const int h = blockIdx.x;
    const int lane = threadIdx.x;
    const int col0 = (blockIdx.z * blockDim.y + threadIdx.y) * 2;
    if (col0 >= kSv) return;
    const int hq = h % kHq;

    float* Sh0 = state + ((size_t)h * kSv + col0) * kSv;
    float* Sh1 = Sh0 + kSv;
    float s0[kRowsLane], s1[kRowsLane];
#pragma unroll
    for (int r = 0; r < kRowsLane; ++r) {
        const int i = r * kWarp + lane;
        s0[r] = Sh0[i];
        s1[r] = Sh1[i];
    }

    for (int t = 0; t < T; ++t) {
        const float* qt = q + ((size_t)t * kHq + hq) * kSv;
        const float* kt = k + ((size_t)t * kHq + hq) * kSv;
        const float* vt = v + ((size_t)t * kHv + h) * kSv;
        const float g = expf(g_log[t * kHv + h]);
        const float b = beta[t * kHv + h];

        float k_reg[kRowsLane];
        float q_reg[kRowsLane];
        float kv0 = 0.f, kv1 = 0.f;
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) {
            const int i = r * kWarp + lane;
            k_reg[r] = kt[i];
            q_reg[r] = qt[i];
            kv0 += s0[r] * k_reg[r];
            kv1 += s1[r] * k_reg[r];
        }
        kv0 = warp_sum(kv0);
        kv1 = warp_sum(kv1);
        const float d0 = (vt[col0] - g * kv0) * b;
        const float d1 = (vt[col0 + 1] - g * kv1) * b;

        float a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) {
            s0[r] = g * s0[r] + k_reg[r] * d0;
            s1[r] = g * s1[r] + k_reg[r] * d1;
            a0 += s0[r] * q_reg[r];
            a1 += s1[r] * q_reg[r];
        }
        a0 = warp_sum(a0);
        a1 = warp_sum(a1);
        if (lane == 0) {
            const size_t base = ((size_t)t * kHv + h) * kSv + col0;
            out[base] = a0 * scale;
            out[base + 1] = a1 * scale;
        }
    }
#pragma unroll
    for (int r = 0; r < kRowsLane; ++r) {
        const int i = r * kWarp + lane;
        Sh0[i] = s0[r];
        Sh1[i] = s1[r];
    }
}

// Four columns per warp. grid.z = Sv/16.
__global__ void __launch_bounds__(128, 8)
gdn_occ4_kernel(const float* __restrict__ q,
                const float* __restrict__ k,
                const float* __restrict__ v,
                const float* __restrict__ g_log,
                const float* __restrict__ beta,
                float* __restrict__ state,
                float* __restrict__ out,
                int T, float scale) {
    const int h = blockIdx.x;
    const int lane = threadIdx.x;
    const int col0 = (blockIdx.z * blockDim.y + threadIdx.y) * 4;
    if (col0 >= kSv) return;
    const int hq = h % kHq;

    float* Sh = state + ((size_t)h * kSv + col0) * kSv;
    float s[4][kRowsLane];
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) s[c][r] = Sh[c * kSv + r * kWarp + lane];

    for (int t = 0; t < T; ++t) {
        const float* qt = q + ((size_t)t * kHq + hq) * kSv;
        const float* kt = k + ((size_t)t * kHq + hq) * kSv;
        const float* vt = v + ((size_t)t * kHv + h) * kSv;
        const float g = expf(g_log[t * kHv + h]);
        const float b = beta[t * kHv + h];

        float k_reg[kRowsLane];
        float q_reg[kRowsLane];
        float kv[4] = {0, 0, 0, 0};
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) {
            const int i = r * kWarp + lane;
            k_reg[r] = kt[i];
            q_reg[r] = qt[i];
#pragma unroll
            for (int c = 0; c < 4; ++c) kv[c] += s[c][r] * k_reg[r];
        }
        float d[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            kv[c] = warp_sum(kv[c]);
            d[c] = (vt[col0 + c] - g * kv[c]) * b;
        }
        float attn[4] = {0, 0, 0, 0};
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                s[c][r] = g * s[c][r] + k_reg[r] * d[c];
                attn[c] += s[c][r] * q_reg[r];
            }
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) attn[c] = warp_sum(attn[c]);
        if (lane == 0) {
            const size_t base = ((size_t)t * kHv + h) * kSv + col0;
#pragma unroll
            for (int c = 0; c < 4; ++c) out[base + c] = attn[c] * scale;
        }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) Sh[c * kSv + r * kWarp + lane] = s[c][r];
}

// Eight columns per warp. grid.z = Sv/32.
// Gate: 4-warp / 0 smem; regs ~90 → SMSP 32*90*1=2880≤16384; occ ~62% vs occ4 100%.
// 8-way ILP vs 4-cycle FMA. Beat occ4 ~1.02 ms @ T=512 (0.91). Ported as ggml COLS=8.
__global__ void __launch_bounds__(128, 4)
gdn_occ8_kernel(const float* __restrict__ q,
                const float* __restrict__ k,
                const float* __restrict__ v,
                const float* __restrict__ g_log,
                const float* __restrict__ beta,
                float* __restrict__ state,
                float* __restrict__ out,
                int T, float scale) {
    const int h = blockIdx.x;
    const int lane = threadIdx.x;
    const int col0 = (blockIdx.z * blockDim.y + threadIdx.y) * 8;
    if (col0 >= kSv) return;
    const int hq = h % kHq;

    float* Sh = state + ((size_t)h * kSv + col0) * kSv;
    float s[8][kRowsLane];
#pragma unroll
    for (int c = 0; c < 8; ++c)
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) s[c][r] = Sh[c * kSv + r * kWarp + lane];

    for (int t = 0; t < T; ++t) {
        const float* qt = q + ((size_t)t * kHq + hq) * kSv;
        const float* kt = k + ((size_t)t * kHq + hq) * kSv;
        const float* vt = v + ((size_t)t * kHv + h) * kSv;
        const float g = expf(g_log[t * kHv + h]);
        const float b = beta[t * kHv + h];

        float k_reg[kRowsLane];
        float q_reg[kRowsLane];
        float kv[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) {
            const int i = r * kWarp + lane;
            k_reg[r] = kt[i];
            q_reg[r] = qt[i];
#pragma unroll
            for (int c = 0; c < 8; ++c) kv[c] += s[c][r] * k_reg[r];
        }
        float d[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            kv[c] = warp_sum(kv[c]);
            d[c] = (vt[col0 + c] - g * kv[c]) * b;
        }
        float attn[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) {
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                s[c][r] = g * s[c][r] + k_reg[r] * d[c];
                attn[c] += s[c][r] * q_reg[r];
            }
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) attn[c] = warp_sum(attn[c]);
        if (lane == 0) {
            const size_t base = ((size_t)t * kHv + h) * kSv + col0;
#pragma unroll
            for (int c = 0; c < 8; ++c) out[base + c] = attn[c] * scale;
        }
    }
#pragma unroll
    for (int c = 0; c < 8; ++c)
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) Sh[c * kSv + r * kWarp + lane] = s[c][r];
}

// occ4 + software pipeline: issue next-token LDG while FFMA on current.
// Same launch. v is 4 consecutive columns → float4. q/k stay stride-32 (lane-owned).
__global__ void __launch_bounds__(128, 8)
gdn_occ4p_kernel(const float* __restrict__ q,
                 const float* __restrict__ k,
                 const float* __restrict__ v,
                 const float* __restrict__ g_log,
                 const float* __restrict__ beta,
                 float* __restrict__ state,
                 float* __restrict__ out,
                 int T, float scale) {
    const int h = blockIdx.x;
    const int lane = threadIdx.x;
    const int col0 = (blockIdx.z * blockDim.y + threadIdx.y) * 4;
    if (col0 >= kSv) return;
    const int hq = h % kHq;

    float* Sh = state + ((size_t)h * kSv + col0) * kSv;
    float s[4][kRowsLane];
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) s[c][r] = Sh[c * kSv + r * kWarp + lane];

    const float* qt = q + (size_t)hq * kSv;
    const float* kt = k + (size_t)hq * kSv;
    const float* vt = v + (size_t)h * kSv;
    float k_reg[kRowsLane];
    float q_reg[kRowsLane];
#pragma unroll
    for (int r = 0; r < kRowsLane; ++r) {
        const int i = r * kWarp + lane;
        k_reg[r] = kt[i];
        q_reg[r] = qt[i];
    }
    float g = expf(g_log[h]);
    float b = beta[h];
    float4 vv = *reinterpret_cast<const float4*>(vt + col0);

    for (int t = 0; t < T; ++t) {
        const bool more = (t + 1) < T;
        const float* qt_n = more ? qt + kHq * kSv : qt;
        const float* kt_n = more ? kt + kHq * kSv : kt;
        const float* vt_n = more ? vt + kHv * kSv : vt;
        float k_n[kRowsLane];
        float q_n[kRowsLane];
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) {
            const int i = r * kWarp + lane;
            k_n[r] = kt_n[i];
            q_n[r] = qt_n[i];
        }
        const float g_n = more ? expf(g_log[(t + 1) * kHv + h]) : g;
        const float b_n = more ? beta[(t + 1) * kHv + h] : b;
        const float4 vv_n = *reinterpret_cast<const float4*>(vt_n + col0);

        float kv[4] = {0, 0, 0, 0};
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) {
#pragma unroll
            for (int c = 0; c < 4; ++c) kv[c] += s[c][r] * k_reg[r];
        }
        float d[4];
        const float varr[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            kv[c] = warp_sum(kv[c]);
            d[c] = (varr[c] - g * kv[c]) * b;
        }
        float attn[4] = {0, 0, 0, 0};
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                s[c][r] = g * s[c][r] + k_reg[r] * d[c];
                attn[c] += s[c][r] * q_reg[r];
            }
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) attn[c] = warp_sum(attn[c]);
        if (lane == 0) {
            const size_t base = ((size_t)t * kHv + h) * kSv + col0;
#pragma unroll
            for (int c = 0; c < 4; ++c) out[base + c] = attn[c] * scale;
        }

#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) {
            k_reg[r] = k_n[r];
            q_reg[r] = q_n[r];
        }
        g = g_n;
        b = b_n;
        vv = vv_n;
        qt = qt_n;
        kt = kt_n;
        vt = vt_n;
    }
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) Sh[c * kSv + r * kWarp + lane] = s[c][r];
}

// Prep: C×C once per (head, chunk). grid (Hv, nch) block 128.
__global__ void gdn_prep_kernel(const float* __restrict__ q,
                                const float* __restrict__ k,
                                const float* __restrict__ g_log,
                                const float* __restrict__ beta,
                                float* __restrict__ attn_all,
                                float* __restrict__ kq_all,
                                float* __restrict__ gcs_all,
                                float* __restrict__ bt_all,
                                int T) {
    const int h = blockIdx.x;
    const int ch = blockIdx.y;
    const int tid = threadIdx.x;
    const int hq = h % kHq;
    const int t0 = ch * kCs;

    __shared__ float gcs[kCs];
    __shared__ float sbt[kCs];
    __shared__ float kb[kCs * kCs];

    if (tid < kCs) {
        const int tt = t0 + tid;
        gcs[tid] = (tt < T) ? g_log[tt * kHv + h] : 0.f;
        sbt[tid] = (tt < T) ? beta[tt * kHv + h] : 0.f;
    }
    __syncthreads();
    if (tid == 0) {
        for (int t = 1; t < kCs; ++t) gcs[t] += gcs[t - 1];
    }
    __syncthreads();

    if (tid < kCs) {
        gcs_all[((size_t)h * gridDim.y + ch) * kCs + tid] = gcs[tid];
        bt_all[((size_t)h * gridDim.y + ch) * kCs + tid] = sbt[tid];
    }

    for (int e = tid; e < kCs * kCs; e += 128) {
        const int i = e % kCs;
        const int j = e / kCs;
        const int tti = t0 + i;
        const int ttj = t0 + j;
        const float mask = (i <= j) ? expf(gcs[j] - gcs[i]) : 0.f;
        float dk = 0.f, dq = 0.f;
        if (tti < T && ttj < T) {
            const float* ki = k + ((size_t)tti * kHq + hq) * kSv;
            const float* kj = k + ((size_t)ttj * kHq + hq) * kSv;
            const float* qj = q + ((size_t)ttj * kHq + hq) * kSv;
#pragma unroll
            for (int d = 0; d < kSv; ++d) {
                dk += ki[d] * kj[d];
                dq += ki[d] * qj[d];
            }
        }
        kb[i + j * kCs] = dk * sbt[j] * mask;
        kq_all[(((size_t)h * gridDim.y + ch) * kCs + j) * kCs + i] =
            (i <= j) ? dq * mask : 0.f;
    }
    __syncthreads();

    if (tid == 0) {
        float lhs[kCs * kCs];
        float X[kCs * kCs];
        for (int colj = 0; colj < kCs; ++colj) {
            for (int r = 0; r < kCs; ++r) {
                const float a = (r < colj) ? kb[r + colj * kCs] : 0.f;
                lhs[r + colj * kCs] = a + (r == colj ? 1.f : 0.f);
            }
        }
        float* attn = attn_all + ((size_t)h * gridDim.y + ch) * kCs * kCs;
        for (int colj = 0; colj < kCs; ++colj) {
            for (int r = 0; r < kCs; ++r) {
                float sum = 0.f;
                for (int p = 0; p < r; ++p)
                    sum += lhs[p + r * kCs] * X[p + colj * kCs];
                const float b = (r < colj) ? -kb[r + colj * kCs] : 0.f;
                X[r + colj * kCs] = (b - sum) / lhs[r + r * kCs];
            }
        }
        for (int colj = 0; colj < kCs; ++colj)
            for (int r = 0; r < kCs; ++r)
                attn[r + colj * kCs] = X[r + colj * kCs] + (r == colj ? 1.f : 0.f);
    }
}

// Apply precomputed C×C. Same launch as occ.
__global__ void __launch_bounds__(128, 4)
gdn_apply_kernel(const float* __restrict__ q,
                 const float* __restrict__ k,
                 const float* __restrict__ v,
                 const float* __restrict__ attn_all,
                 const float* __restrict__ kq_all,
                 const float* __restrict__ gcs_all,
                 const float* __restrict__ bt_all,
                 float* __restrict__ state,
                 float* __restrict__ out,
                 int T, int nch, float scale) {
    const int h = blockIdx.x;
    const int lane = threadIdx.x;
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= kSv) return;
    const int hq = h % kHq;

    float* Sh = state + ((size_t)h * kSv + col) * kSv;
    float s_shard[kRowsLane];
#pragma unroll
    for (int r = 0; r < kRowsLane; ++r) s_shard[r] = Sh[r * kWarp + lane];

    for (int ch = 0; ch < nch; ++ch) {
        const int t0 = ch * kCs;
        const float* attn = attn_all + ((size_t)h * nch + ch) * kCs * kCs;
        const float* kq = kq_all + ((size_t)h * nch + ch) * kCs * kCs;
        const float* gcs = gcs_all + ((size_t)h * nch + ch) * kCs;
        const float* sbt = bt_all + ((size_t)h * nch + ch) * kCs;

        float k_dot[kCs];
        float q_dot[kCs];
#pragma unroll
        for (int t = 0; t < kCs; ++t) {
            const int tt = t0 + t;
            float kd = 0.f, qd = 0.f;
            if (tt < T) {
                const float* kt = k + ((size_t)tt * kHq + hq) * kSv;
                const float* qt = q + ((size_t)tt * kHq + hq) * kSv;
#pragma unroll
                for (int r = 0; r < kRowsLane; ++r) {
                    const int i = r * kWarp + lane;
                    kd += s_shard[r] * kt[i];
                    qd += s_shard[r] * qt[i];
                }
            }
            k_dot[t] = warp_sum(kd);
            q_dot[t] = warp_sum(qd);
        }

        float vnew[kCs];
#pragma unroll
        for (int t = 0; t < kCs; ++t) {
            float vm = 0.f, vp = 0.f;
#pragma unroll
            for (int i = 0; i < kCs; ++i) {
                const int ti = t0 + i;
                const float a = attn[i + t * kCs];
                const float bi = sbt[i];
                if (ti < T) vm += v[((size_t)ti * kHv + h) * kSv + col] * bi * a;
                vp += a * bi * expf(gcs[i]) * k_dot[i];
            }
            vnew[t] = vm - vp;
        }

        const float g_last = gcs[kCs - 1];
        const float g_last_e = expf(g_last);
#pragma unroll
        for (int t = 0; t < kCs; ++t) {
            const int tt = t0 + t;
            if (tt >= T) continue;
            float va = 0.f;
#pragma unroll
            for (int i = 0; i < kCs; ++i) va += vnew[i] * kq[i + t * kCs];
            if (lane == 0)
                out[((size_t)tt * kHv + h) * kSv + col] =
                    (expf(gcs[t]) * q_dot[t] + va) * scale;
        }

#pragma unroll
        for (int r = 0; r < kRowsLane; ++r) {
            const int d = r * kWarp + lane;
            float add = 0.f;
#pragma unroll
            for (int t = 0; t < kCs; ++t) {
                const int tt = t0 + t;
                const float kt = (tt < T) ? k[((size_t)tt * kHq + hq) * kSv + d] : 0.f;
                add += kt * expf(g_last - gcs[t]) * vnew[t];
            }
            s_shard[r] = g_last_e * s_shard[r] + add;
        }
    }
#pragma unroll
    for (int r = 0; r < kRowsLane; ++r) Sh[r * kWarp + lane] = s_shard[r];
}

// ---------------------------------------------------------------------------
// Host reference
// ---------------------------------------------------------------------------
static void gdn_cpu(const std::vector<float>& q, const std::vector<float>& k,
                    const std::vector<float>& v, const std::vector<float>& g_log,
                    const std::vector<float>& beta, std::vector<float>& state,
                    std::vector<float>& out, int T, float scale) {
    for (int h = 0; h < kHv; ++h) {
        const int hq = h % kHq;
        std::vector<float> S(kSv * kSv);
        for (int i = 0; i < kSv * kSv; ++i) S[i] = state[(size_t)h * kSv * kSv + i];
        for (int t = 0; t < T; ++t) {
            const float* qt = q.data() + ((size_t)t * kHq + hq) * kSv;
            const float* kt = k.data() + ((size_t)t * kHq + hq) * kSv;
            const float* vt = v.data() + ((size_t)t * kHv + h) * kSv;
            const float g = std::exp(g_log[t * kHv + h]);
            const float b = beta[t * kHv + h];
            for (int c = 0; c < kSv; ++c) {
                float kv = 0.f;
                for (int i = 0; i < kSv; ++i) kv += S[c * kSv + i] * kt[i];
                const float d = (vt[c] - g * kv) * b;
                float attn = 0.f;
                for (int i = 0; i < kSv; ++i) {
                    S[c * kSv + i] = g * S[c * kSv + i] + kt[i] * d;
                    attn += S[c * kSv + i] * qt[i];
                }
                out[((size_t)t * kHv + h) * kSv + c] = attn * scale;
            }
        }
        for (int i = 0; i < kSv * kSv; ++i) state[(size_t)h * kSv * kSv + i] = S[i];
    }
}

// ggml non-KDA chunk. Layout matches build_delta_net_chunking + solve_tri CPU.
static void gdn_chunked_cpu(const std::vector<float>& q, const std::vector<float>& k,
                            const std::vector<float>& v, const std::vector<float>& g_log,
                            const std::vector<float>& beta, std::vector<float>& state,
                            std::vector<float>& out, int T, float scale) {
    const int nch = (T + kCs - 1) / kCs;
    for (int h = 0; h < kHv; ++h) {
        const int hq = h % kHq;
        std::vector<float> S(kSv * kSv);
        for (int i = 0; i < kSv * kSv; ++i) S[i] = state[(size_t)h * kSv * kSv + i];
        for (int ch = 0; ch < nch; ++ch) {
            const int t0 = ch * kCs;
            float qdc[kSv * kCs]{};
            float kdc[kSv * kCs]{};
            float vdc[kSv * kCs]{};
            float gl[kCs]{};
            float bt[kCs]{};
            for (int t = 0; t < kCs; ++t) {
                const int tt = t0 + t;
                if (tt >= T) continue;
                const float* qt = q.data() + ((size_t)tt * kHq + hq) * kSv;
                const float* kt = k.data() + ((size_t)tt * kHq + hq) * kSv;
                const float* vt = v.data() + ((size_t)tt * kHv + h) * kSv;
                for (int d = 0; d < kSv; ++d) {
                    qdc[d + t * kSv] = qt[d];
                    kdc[d + t * kSv] = kt[d];
                    vdc[d + t * kSv] = vt[d];
                }
                gl[t] = g_log[tt * kHv + h];
                bt[t] = beta[tt * kHv + h];
            }

            float gcs[kCs];
            gcs[0] = gl[0];
            for (int t = 1; t < kCs; ++t) gcs[t] = gcs[t - 1] + gl[t];

            // ggml: A[i,j] at i + j*CS. tri LOWER keeps i < j (strict upper).
            float kb[kCs * kCs];
            float kq[kCs * kCs];
            for (int j = 0; j < kCs; ++j) {
                for (int i = 0; i < kCs; ++i) {
                    const float mask = (i <= j) ? std::exp(gcs[j] - gcs[i]) : 0.f;
                    float dk = 0.f, dq = 0.f;
                    for (int d = 0; d < kSv; ++d) {
                        dk += kdc[d + i * kSv] * kdc[d + j * kSv];
                        dq += kdc[d + i * kSv] * qdc[d + j * kSv];
                    }
                    kb[i + j * kCs] = dk * bt[j] * mask;
                    kq[i + j * kCs] = (i <= j) ? dq * mask : 0.f;
                }
            }

            float lhs[kCs * kCs];
            float X[kCs * kCs];
            for (int j = 0; j < kCs; ++j) {
                for (int i = 0; i < kCs; ++i) {
                    const float a = (i < j) ? kb[i + j * kCs] : 0.f;
                    lhs[i + j * kCs] = a + (i == j ? 1.f : 0.f);
                }
            }
            for (int colj = 0; colj < kCs; ++colj) {
                for (int r = 0; r < kCs; ++r) {
                    float sum = 0.f;
                    for (int p = 0; p < r; ++p)
                        sum += lhs[p + r * kCs] * X[p + colj * kCs];
                    const float b = (r < colj) ? -kb[r + colj * kCs] : 0.f;
                    X[r + colj * kCs] = (b - sum) / lhs[r + r * kCs];
                }
            }
            float attn[kCs * kCs];
            for (int j = 0; j < kCs; ++j)
                for (int i = 0; i < kCs; ++i)
                    attn[i + j * kCs] = X[i + j * kCs] + (i == j ? 1.f : 0.f);

            float vmix[kSv * kCs]{};
            float kcd[kSv * kCs]{};
            for (int t = 0; t < kCs; ++t) {
                for (int d = 0; d < kSv; ++d) {
                    float vm = 0.f, kc = 0.f;
                    for (int i = 0; i < kCs; ++i) {
                        const float a = attn[i + t * kCs];
                        vm += vdc[d + i * kSv] * bt[i] * a;
                        kc += kdc[d + i * kSv] * bt[i] * std::exp(gcs[i]) * a;
                    }
                    vmix[d + t * kSv] = vm;
                    kcd[d + t * kSv] = kc;
                }
            }

            float qg[kSv * kCs];
            float kg[kSv * kCs];
            const float g_last = gcs[kCs - 1];
            for (int t = 0; t < kCs; ++t) {
                const float eg = std::exp(gcs[t]);
                const float gd = std::exp(g_last - gcs[t]);
                for (int d = 0; d < kSv; ++d) {
                    qg[d + t * kSv] = qdc[d + t * kSv] * eg;
                    kg[d + t * kSv] = kdc[d + t * kSv] * gd;
                }
            }

            float vnew[kCs * kSv];
            for (int t = 0; t < kCs; ++t) {
                for (int d = 0; d < kSv; ++d) {
                    float vp = 0.f;
                    for (int i = 0; i < kSv; ++i) vp += kcd[i + t * kSv] * S[d * kSv + i];
                    vnew[t * kSv + d] = vmix[d + t * kSv] - vp;
                }
            }

            for (int t = 0; t < kCs; ++t) {
                const int tt = t0 + t;
                if (tt >= T) continue;
                for (int d = 0; d < kSv; ++d) {
                    float va = 0.f, ai = 0.f;
                    for (int i = 0; i < kCs; ++i)
                        va += vnew[i * kSv + d] * kq[i + t * kCs];
                    for (int i = 0; i < kSv; ++i) ai += S[d * kSv + i] * qg[i + t * kSv];
                    out[((size_t)tt * kHv + h) * kSv + d] = (ai + va) * scale;
                }
            }

            const float glast_e = std::exp(g_last);
            std::vector<float> Sn(kSv * kSv);
            for (int b = 0; b < kSv; ++b) {
                for (int a = 0; a < kSv; ++a) {
                    float add = 0.f;
                    for (int t = 0; t < kCs; ++t)
                        add += kg[a + t * kSv] * vnew[t * kSv + b];
                    Sn[b * kSv + a] = glast_e * S[b * kSv + a] + add;
                }
            }
            S.swap(Sn);
        }
        for (int i = 0; i < kSv * kSv; ++i) state[(size_t)h * kSv * kSv + i] = S[i];
    }
}

int main(int argc, char** argv) {
    int T = 64;
    if (argc == 2) T = std::atoi(argv[1]);
    if (T < 1 || T > 2048) {
        fprintf(stderr, "usage: %s [T]\n", argv[0]);
        return 2;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU             : %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("GDN shape       : Sv=%d  Hq=%d  Hv=%d  T=%d  CS=%d  (Qwen3.8-27B)\n",
           kSv, kHq, kHv, T, kCs);

    const float scale = 1.f / std::sqrt((float)kSv);
    std::vector<float> q((size_t)T * kHq * kSv);
    std::vector<float> k((size_t)T * kHq * kSv);
    std::vector<float> v((size_t)T * kHv * kSv);
    std::vector<float> g((size_t)T * kHv);
    std::vector<float> b((size_t)T * kHv);
    std::vector<float> st((size_t)kHv * kSv * kSv, 0.f);
    std::vector<float> o_cpu((size_t)T * kHv * kSv);
    std::vector<float> o_gpu((size_t)T * kHv * kSv);

    unsigned seed = 1;
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return ((seed >> 8) / 16777216.f) - 0.5f;
    };
    for (auto& x : q) x = rnd() * 0.2f;
    for (auto& x : k) x = rnd() * 0.2f;
    for (auto& x : v) x = rnd() * 0.2f;
    for (auto& x : g) x = -0.2f + rnd() * 0.05f;
    for (auto& x : b) x = 0.4f + (rnd() + 0.5f) * 0.4f;

    auto st_cpu = st;
    gdn_cpu(q, k, v, g, b, st_cpu, o_cpu, T, scale);

    auto st_ch = st;
    std::vector<float> o_ch((size_t)T * kHv * kSv);
    gdn_chunked_cpu(q, k, v, g, b, st_ch, o_ch, T, scale);
    float err_ch = 0.f;
    for (size_t i = 0; i < o_cpu.size(); ++i)
        err_ch = std::max(err_ch, fabsf(o_cpu[i] - o_ch[i]));
    printf("max |chunk-serial| : %.5f   %s\n", err_ch, err_ch < 2e-2f ? "OK" : "FAIL");

    float *dq, *dk, *dv, *dg, *db, *ds, *do_;
    CUDA_CHECK(cudaMalloc(&dq, q.size() * 4));
    CUDA_CHECK(cudaMalloc(&dk, k.size() * 4));
    CUDA_CHECK(cudaMalloc(&dv, v.size() * 4));
    CUDA_CHECK(cudaMalloc(&dg, g.size() * 4));
    CUDA_CHECK(cudaMalloc(&db, b.size() * 4));
    CUDA_CHECK(cudaMalloc(&ds, st.size() * 4));
    CUDA_CHECK(cudaMalloc(&do_, o_gpu.size() * 4));
    CUDA_CHECK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));

    auto check_out = [&](const char* name) {
        CUDA_CHECK(cudaMemcpy(o_gpu.data(), do_, o_gpu.size() * 4, cudaMemcpyDeviceToHost));
        float err = 0.f;
        for (size_t i = 0; i < o_cpu.size(); ++i)
            err = std::max(err, fabsf(o_cpu[i] - o_gpu[i]));
        printf("max |%s-cpu| : %.5f   %s\n", name, err, err < 2e-2f ? "OK" : "FAIL");
        return err;
    };

    auto bench = [&](auto launch, const char* name, int iters) {
        CUDA_CHECK(cudaMemcpy(ds, st.data(), st.size() * 4, cudaMemcpyHostToDevice));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const float err = check_out(name);
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) launch();
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        ms /= iters;
        printf("%-16s: %7.3f ms   x48 layers %7.3f ms\n", name, ms, ms * 48);
        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
        return err;
    };

    const dim3 occ_grid(kHv, 1, kSv / kWarps);
    const dim3 occ_block(kWarp, kWarps);
    const dim3 occ2_grid(kHv, 1, kSv / (kWarps * 2));

    const int nch = (T + kCs - 1) / kCs;
    float *dattn, *dkq, *dgcs, *dbt;
    CUDA_CHECK(cudaMalloc(&dattn, (size_t)kHv * nch * kCs * kCs * 4));
    CUDA_CHECK(cudaMalloc(&dkq, (size_t)kHv * nch * kCs * kCs * 4));
    CUDA_CHECK(cudaMalloc(&dgcs, (size_t)kHv * nch * kCs * 4));
    CUDA_CHECK(cudaMalloc(&dbt, (size_t)kHv * nch * kCs * 4));

    float err = 0.f;
    err = std::max(err, bench([&]() {
        gdn_serial_kernel<<<kHv, kSv>>>(dq, dk, dv, dg, db, ds, do_, T, scale);
    }, "serial", 20));
    err = std::max(err, bench([&]() {
        gdn_smem_kernel<<<kHv, kSv>>>(dq, dk, dv, dg, db, ds, do_, T, scale);
    }, "smem-f16", T >= 256 ? 5 : 20));
    err = std::max(err, bench([&]() {
        gdn_occ_kernel<<<occ_grid, occ_block>>>(dq, dk, dv, dg, db, ds, do_, T, scale);
    }, "occ", 20));
    err = std::max(err, bench([&]() {
        gdn_occ2_kernel<<<occ2_grid, occ_block>>>(dq, dk, dv, dg, db, ds, do_, T, scale);
    }, "occ2", 20));
    err = std::max(err, bench([&]() {
        gdn_occ4_kernel<<<dim3(kHv, 1, kSv / (kWarps * 4)), occ_block>>>(dq, dk, dv, dg, db, ds,
                                                                         do_, T, scale);
    }, "occ4", 20));
    {
        cudaFuncSetAttribute(gdn_occ4p_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxL1);
        cudaFuncSetAttribute(gdn_occ4_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxL1);
    }
    err = std::max(err, bench([&]() {
        gdn_occ4p_kernel<<<dim3(kHv, 1, kSv / (kWarps * 4)), occ_block>>>(dq, dk, dv, dg, db, ds,
                                                                          do_, T, scale);
    }, "occ4p", 20));
    err = std::max(err, bench([&]() {
        gdn_occ8_kernel<<<dim3(kHv, 1, kSv / (kWarps * 8)), occ_block>>>(dq, dk, dv, dg, db, ds,
                                                                         do_, T, scale);
    }, "occ8", 20));
    if (T <= 64) {
        err = std::max(err, bench([&]() {
            gdn_chunked_kernel<<<occ_grid, occ_block>>>(dq, dk, dv, dg, db, ds, do_, T, scale);
        }, "chunked", 5));
    }
    err = std::max(err, bench([&]() {
        gdn_prep_kernel<<<dim3(kHv, nch), 128>>>(dq, dk, dg, db, dattn, dkq, dgcs, dbt, T);
        gdn_apply_kernel<<<occ_grid, occ_block>>>(dq, dk, dv, dattn, dkq, dgcs, dbt, ds, do_, T,
                                                  nch, scale);
    }, "twopass", 20));

    CUDA_CHECK(cudaFuncSetAttribute(gdn_wmma_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kWmmaSmem));
    err = std::max(err, bench([&]() {
        gdn_wmma_kernel<<<kHv, kWmmaWarps * kWarp, kWmmaSmem>>>(dq, dk, dv, dg, db, ds, do_, T,
                                                                scale);
    }, "wmma", T >= 256 ? 5 : 20));
    err = std::max(err, err_ch);

    cudaFree(dattn);
    cudaFree(dkq);
    cudaFree(dgcs);
    cudaFree(dbt);

    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dg);
    cudaFree(db);
    cudaFree(ds);
    cudaFree(do_);
    return err < 2e-2f ? 0 : 1;
}
