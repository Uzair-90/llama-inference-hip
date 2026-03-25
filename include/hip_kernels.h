#pragma once
#include <hip/hip_runtime.h>

// ---------------------------------------------------------------------------
// GPU Kernel Declarations
// All kernels operate on device pointers only.
// ---------------------------------------------------------------------------

// matmul: xout[d] = sum_n( w[d*n + i] * x[i] )
//   - d output rows distributed one-per-thread across grid
__global__ void kernel_matmul(float* xout, const float* x, const float* w, int n, int d);

// RMSNorm: o[j] = weight[j] * (x[j] / rms(x))
//   - single block launch, shared memory accumulation
__global__ void kernel_rmsnorm(float* o, const float* x, const float* weight, int size);

// Softmax in-place over a 1-D slice of length `size`
//   - single block, shared memory reduction for max and sum
__global__ void kernel_softmax(float* x, int size);

// RoPE positional encoding applied to q and k vectors
//   - each thread handles one complex pair (i, i+1)
__global__ void kernel_rope(float* q, float* k, int dim, int kv_dim, int head_size, int pos);

// SwiGLU: hb[i] = silu(hb[i]) * hb2[i]
//   - element-wise, trivially parallel
__global__ void kernel_swiglu(float* hb, const float* hb2, int hidden_dim);

// Element-wise addition with accumulation: x[i] += y[i]
__global__ void kernel_add(float* x, const float* y, int size);

// Multi-head attention score accumulation
//   - one block per head; each thread iterates over timesteps
__global__ void kernel_attention(
    float*       xb,           // output: (n_heads, head_size)
    const float* q,            // (dim,)
    const float* key_cache,    // (n_layers, seq_len, kv_dim)
    const float* value_cache,  // (n_layers, seq_len, kv_dim)
    float*       att,          // (n_heads, seq_len)
    int          head_size,
    int          kv_dim,
    int          kv_mul,
    int          seq_len,
    int          pos,
    int          loff
);

// ---------------------------------------------------------------------------
// Helper: check HIP errors and abort on failure
// ---------------------------------------------------------------------------
#define HIP_CHECK(cmd) do {                                        \
    hipError_t e = (cmd);                                          \
    if (e != hipSuccess) {                                         \
        fprintf(stderr, "HIP error %s at %s:%d\n",                \
                hipGetErrorString(e), __FILE__, __LINE__);         \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while(0)
