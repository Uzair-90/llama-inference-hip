/*
 * hip_kernels.cpp
 * Implementation of all HIP GPU kernels for Llama2 transformer hot-paths.
 * Ported from karpathy/llama2.c (https://github.com/karpathy/llama2.c)
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>
#include "hip_kernels.h"

// ---------------------------------------------------------------------------
// matmul: W(d,n) @ x(n) -> xout(d)
// Each GPU thread computes one row of the output.
// This is the single hottest function in the transformer, accounting for
// the majority of FLOPs. With HBM-backed wavefronts the memory bandwidth
// utilization is dramatically higher than single-core DRAM.
// ---------------------------------------------------------------------------
__global__ void kernel_matmul(float* xout, const float* x, const float* w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;

    float val = 0.0f;
    const float* row = w + i * n;
    for (int j = 0; j < n; j++) {
        val += row[j] * x[j];
    }
    xout[i] = val;
}

// ---------------------------------------------------------------------------
// RMSNorm: o[j] = weight[j] * ( x[j] / sqrt(mean(x^2) + eps) )
// One block, threads cooperate to accumulate the sum-of-squares via shared mem.
// ---------------------------------------------------------------------------
__global__ void kernel_rmsnorm(float* o, const float* x, const float* weight, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    float partial = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        partial += x[i] * x[i];
    }
    sdata[tid] = partial;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    float ss = sdata[0] / size + 1e-5f;
    float inv_rms = 1.0f / sqrtf(ss);

    for (int i = tid; i < size; i += blockDim.x) {
        o[i] = weight[i] * (inv_rms * x[i]);
    }
}

// ---------------------------------------------------------------------------
// Softmax in-place
// Two-pass: find max (for numerical stability), then exp + normalize.
// ---------------------------------------------------------------------------
__global__ void kernel_softmax(float* x, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;

    // Pass 1: find max
    float local_max = -1e30f;
    for (int i = tid; i < size; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    float max_val = sdata[0];

    // Pass 2: exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        x[i] = expf(x[i] - max_val);
        local_sum += x[i];
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float sum = sdata[0];

    for (int i = tid; i < size; i += blockDim.x) {
        x[i] /= sum;
    }
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding (RoPE)
// Each thread rotates one (real, imag) pair in the query and/or key.
// ---------------------------------------------------------------------------
__global__ void kernel_rope(float* q, float* k, int dim, int kv_dim, int head_size, int pos) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= dim) return;

    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, (float)head_dim / (float)head_size);
    float val   = pos * freq;
    float fcr   = cosf(val);
    float fci   = sinf(val);

    // Rotate query
    float q0 = q[i], q1 = q[i + 1];
    q[i]     = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;

    // Rotate key only for positions within the kv dimension
    if (i < kv_dim) {
        float k0 = k[i], k1 = k[i + 1];
        k[i]     = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
    }
}

// ---------------------------------------------------------------------------
// SwiGLU non-linearity: hb[i] = silu(hb[i]) * hb2[i]
// Trivially parallel over hidden_dim.
// ---------------------------------------------------------------------------
__global__ void kernel_swiglu(float* hb, const float* hb2, int hidden_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_dim) return;
    float v = hb[i];
    v *= (1.0f / (1.0f + expf(-v)));  // silu
    hb[i] = v * hb2[i];
}

// ---------------------------------------------------------------------------
// Element-wise accumulation: x[i] += y[i]
// ---------------------------------------------------------------------------
__global__ void kernel_add(float* x, const float* y, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    x[i] += y[i];
}

// ---------------------------------------------------------------------------
// Multi-head attention
// Block layout: one block per attention head.
// Each block:
//   1. Computes all QK dot-products (scores) for the current head.
//   2. Applies softmax over the [0..pos] scores.
//   3. Accumulates the value-weighted output into xb.
// ---------------------------------------------------------------------------
__global__ void kernel_attention(
    float*       xb,
    const float* q,
    const float* key_cache,
    const float* value_cache,
    float*       att,
    int          head_size,
    int          kv_dim,
    int          kv_mul,
    int          seq_len,
    int          pos,
    int          loff)
{
    int h = blockIdx.x;            // each block handles one head
    const float* qh = q + h * head_size;
    float* atth = att + h * seq_len;

    // --- Step 1: compute attention scores ---
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        const float* kh = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
            score += qh[i] * kh[i];
        }
        atth[t] = score / sqrtf((float)head_size);
    }
    __syncthreads();

    // --- Step 2: in-block softmax over atth[0..pos] ---
    // Find max
    extern __shared__ float sdata[];
    float local_max = -1e30f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        local_max = fmaxf(local_max, atth[t]);
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    float max_val = sdata[0];

    float local_sum = 0.0f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        atth[t] = expf(atth[t] - max_val);
        local_sum += atth[t];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        atth[t] *= inv_sum;
    }
    __syncthreads();

    // --- Step 3: weighted sum of values ---
    float* xbh = xb + h * head_size;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t <= pos; t++) {
            const float* vh = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            acc += atth[t] * vh[i];
        }
        xbh[i] = acc;
    }
}
