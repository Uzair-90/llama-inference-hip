/*
 * transformer.cpp
 * GPU-backed transformer forward pass using HIP kernels.
 * Host code manages device memory allocation and kernel dispatching.
 * Replaces the CPU forward() in karpathy/llama2.c with GPU equivalents.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <hip/hip_runtime.h>

#include "model.h"
#include "hip_kernels.h"

// ---------------------------------------------------------------------------
// Device-side activation buffers (mirroring RunState but on GPU)
// ---------------------------------------------------------------------------
static struct {
    float* x;
    float* xb;
    float* xb2;
    float* hb;
    float* hb2;
    float* q;
    float* att;
    float* logits;
    float* key_cache;
    float* value_cache;
} dstate;

// Host-side copy of logits for sampling
static float* h_logits = nullptr;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
static void alloc_device_state(const Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    HIP_CHECK(hipMalloc(&dstate.x,           p->dim           * sizeof(float)));
    HIP_CHECK(hipMalloc(&dstate.xb,          p->dim           * sizeof(float)));
    HIP_CHECK(hipMalloc(&dstate.xb2,         p->dim           * sizeof(float)));
    HIP_CHECK(hipMalloc(&dstate.hb,          p->hidden_dim    * sizeof(float)));
    HIP_CHECK(hipMalloc(&dstate.hb2,         p->hidden_dim    * sizeof(float)));
    HIP_CHECK(hipMalloc(&dstate.q,           p->dim           * sizeof(float)));
    HIP_CHECK(hipMalloc(&dstate.att,         p->n_heads * p->seq_len * sizeof(float)));
    HIP_CHECK(hipMalloc(&dstate.logits,      p->vocab_size    * sizeof(float)));
    HIP_CHECK(hipMalloc(&dstate.key_cache,   (long)p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&dstate.value_cache, (long)p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    HIP_CHECK(hipMemset(dstate.key_cache,   0, (long)p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    HIP_CHECK(hipMemset(dstate.value_cache, 0, (long)p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    h_logits = (float*)malloc(p->vocab_size * sizeof(float));
}

// Upload all weight tensors to the device and store device pointers
// inside TransformerWeights (repurposing the struct pointer fields).
static void weights_to_device(TransformerWeights* w, const Config* p) {
    auto upload = [](float** d_ptr, float* h_ptr, size_t n) {
        HIP_CHECK(hipMalloc(d_ptr, n * sizeof(float)));
        HIP_CHECK(hipMemcpy(*d_ptr, h_ptr, n * sizeof(float), hipMemcpyHostToDevice));
    };
    int hs = p->dim / p->n_heads;
    long nl = p->n_layers;
    int kv_dim = p->dim * p->n_kv_heads / p->n_heads;

    float* h_emb = w->token_embedding_table;
    upload(&w->token_embedding_table, h_emb,            (long)p->vocab_size * p->dim);
    upload(&w->rms_att_weight,        w->rms_att_weight, nl * p->dim);
    upload(&w->rms_ffn_weight,        w->rms_ffn_weight, nl * p->dim);
    upload(&w->wq,                    w->wq,             nl * p->dim * (p->n_heads * hs));
    upload(&w->wk,                    w->wk,             nl * p->dim * (p->n_kv_heads * hs));
    upload(&w->wv,                    w->wv,             nl * p->dim * (p->n_kv_heads * hs));
    upload(&w->wo,                    w->wo,             nl * (p->n_heads * hs) * p->dim);
    upload(&w->w1,                    w->w1,             nl * p->dim * p->hidden_dim);
    upload(&w->w2,                    w->w2,             nl * p->hidden_dim * p->dim);
    upload(&w->w3,                    w->w3,             nl * p->dim * p->hidden_dim);
    upload(&w->rms_final_weight,      w->rms_final_weight, p->dim);
    // wcls is shared with token_embedding_table when shared_weights==1;
    // it has already been aliased in memory_map_weights so just re-point.
    if (w->wcls == h_emb) {
        w->wcls = w->token_embedding_table;
    } else {
        upload(&w->wcls, w->wcls, (long)p->vocab_size * p->dim);
    }
}

// Kernel launch helpers
static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static void gpu_matmul(float* xout, const float* x, const float* w, int n, int d) {
    int threads = 256;
    hipLaunchKernelGGL(kernel_matmul, dim3(ceil_div(d, threads)), dim3(threads), 0, 0,
                       xout, x, w, n, d);
}

static void gpu_rmsnorm(float* o, const float* x, const float* weight, int size) {
    int threads = 256;
    hipLaunchKernelGGL(kernel_rmsnorm, dim3(1), dim3(threads), threads * sizeof(float), 0,
                       o, x, weight, size);
}

static void gpu_add(float* x, const float* y, int size) {
    int threads = 256;
    hipLaunchKernelGGL(kernel_add, dim3(ceil_div(size, threads)), dim3(threads), 0, 0,
                       x, y, size);
}

// ---------------------------------------------------------------------------
// GPU-backed forward pass
// ---------------------------------------------------------------------------
float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    int dim      = p->dim;
    int kv_dim   = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul   = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size  = dim / p->n_heads;

    // Copy token embedding to device x
    HIP_CHECK(hipMemcpy(dstate.x,
                        w->token_embedding_table + token * dim,
                        dim * sizeof(float),
                        hipMemcpyDeviceToDevice));

    for (int l = 0; l < p->n_layers; l++) {
        long loff = (long)l * p->seq_len * kv_dim;
        float* dk = dstate.key_cache   + loff + pos * kv_dim;
        float* dv = dstate.value_cache + loff + pos * kv_dim;

        // Attention RMSNorm
        gpu_rmsnorm(dstate.xb, dstate.x, w->rms_att_weight + l * dim, dim);

        // QKV projections
        gpu_matmul(dstate.q,  dstate.xb, w->wq + l * dim * dim,         dim, dim);
        gpu_matmul(dk,        dstate.xb, w->wk + l * dim * kv_dim,      dim, kv_dim);
        gpu_matmul(dv,        dstate.xb, w->wv + l * dim * kv_dim,      dim, kv_dim);

        // RoPE
        {
            int threads = 128;
            int pairs = dim / 2;
            hipLaunchKernelGGL(kernel_rope,
                               dim3(ceil_div(pairs, threads)), dim3(threads), 0, 0,
                               dstate.q, dk, dim, kv_dim, head_size, pos);
        }

        // Multi-head attention (one block per head)
        {
            int attn_threads = 128;
            size_t shared = attn_threads * sizeof(float);
            hipLaunchKernelGGL(kernel_attention,
                               dim3(p->n_heads), dim3(attn_threads), shared, 0,
                               dstate.xb, dstate.q,
                               dstate.key_cache, dstate.value_cache,
                               dstate.att,
                               head_size, kv_dim, kv_mul,
                               p->seq_len, pos, (int)loff);
        }

        // Output projection
        gpu_matmul(dstate.xb2, dstate.xb, w->wo + l * dim * dim, dim, dim);

        // Residual add: x += xb2
        gpu_add(dstate.x, dstate.xb2, dim);

        // FFN RMSNorm
        gpu_rmsnorm(dstate.xb, dstate.x, w->rms_ffn_weight + l * dim, dim);

        // FFN projections
        gpu_matmul(dstate.hb,  dstate.xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        gpu_matmul(dstate.hb2, dstate.xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        // SwiGLU
        {
            int threads = 256;
            hipLaunchKernelGGL(kernel_swiglu,
                               dim3(ceil_div(hidden_dim, threads)), dim3(threads), 0, 0,
                               dstate.hb, dstate.hb2, hidden_dim);
        }

        // Down projection
        gpu_matmul(dstate.xb, dstate.hb, w->w2 + l * hidden_dim * dim, hidden_dim, dim);

        // Residual add: x += xb
        gpu_add(dstate.x, dstate.xb, dim);
    }

    // Final RMSNorm
    gpu_rmsnorm(dstate.x, dstate.x, w->rms_final_weight, dim);

    // Classifier logits
    gpu_matmul(dstate.logits, dstate.x, w->wcls, dim, p->vocab_size);

    // Sync and copy logits back to host for sampling
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_logits, dstate.logits, p->vocab_size * sizeof(float), hipMemcpyDeviceToHost));

    // Reuse host RunState logits pointer to avoid changing the API
    transformer->state.logits = h_logits;
    return h_logits;
}

// ---------------------------------------------------------------------------
// Checkpoint loading (plain mmap, identical to karpathy)
// ---------------------------------------------------------------------------
static void memory_map_weights(TransformerWeights* w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long nl = p->n_layers;
    w->token_embedding_table = ptr; ptr += (long)p->vocab_size * p->dim;
    w->rms_att_weight        = ptr; ptr += nl * p->dim;
    w->wq                    = ptr; ptr += nl * p->dim * (p->n_heads * head_size);
    w->wk                    = ptr; ptr += nl * p->dim * (p->n_kv_heads * head_size);
    w->wv                    = ptr; ptr += nl * p->dim * (p->n_kv_heads * head_size);
    w->wo                    = ptr; ptr += nl * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight        = ptr; ptr += nl * p->dim;
    w->w1                    = ptr; ptr += nl * p->dim * p->hidden_dim;
    w->w2                    = ptr; ptr += nl * p->hidden_dim * p->dim;
    w->w3                    = ptr; ptr += nl * p->dim * p->hidden_dim;
    w->rms_final_weight      = ptr; ptr += p->dim;
    ptr += p->seq_len * head_size / 2;  // skip RoPE sin/cos tables
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void build_transformer(Transformer* t, char* checkpoint_path) {
    FILE* f = fopen(checkpoint_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", checkpoint_path); exit(1); }
    if (fread(&t->config, sizeof(Config), 1, f) != 1) { exit(1); }

    int shared_weights = t->config.vocab_size > 0 ? 1 : 0;
    t->config.vocab_size = abs(t->config.vocab_size);

    fseek(f, 0, SEEK_END);
    t->file_size = ftell(f);
    fclose(f);

    t->fd = open(checkpoint_path, O_RDONLY);
    if (t->fd == -1) { fprintf(stderr, "open failed\n"); exit(1); }
    t->data = (float*)mmap(NULL, t->file_size, PROT_READ, MAP_PRIVATE, t->fd, 0);
    if (t->data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(1); }

    float* weights_ptr = t->data + sizeof(Config) / sizeof(float);
    memory_map_weights(&t->weights, &t->config, weights_ptr, shared_weights);

    // Move weights to GPU and allocate activation buffers
    weights_to_device(&t->weights, &t->config);
    alloc_device_state(&t->config);

    // Allocate host logits buffer through RunState for sampler compatibility
    t->state.logits = h_logits;
}

void free_transformer(Transformer* t) {
    munmap(t->data, t->file_size);
    close(t->fd);
    free(h_logits);
    // Device memory would be freed here in a production build
}
