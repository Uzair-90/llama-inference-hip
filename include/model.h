#pragma once
#include <stddef.h>
#include <sys/types.h>

// ---------------------------------------------------------------------------
// Transformer Architecture Config
// ---------------------------------------------------------------------------
typedef struct {
    int dim;         // transformer hidden dimension
    int hidden_dim;  // feedforward layer dimension (typically 4x dim)
    int n_layers;    // number of transformer blocks
    int n_heads;     // number of query attention heads
    int n_kv_heads;  // number of key/value heads (< n_heads in grouped-query)
    int vocab_size;  // vocabulary size
    int seq_len;     // maximum context length
} Config;

// ---------------------------------------------------------------------------
// Transformer Weight Tensors (memory-mapped from .bin checkpoint)
// ---------------------------------------------------------------------------
typedef struct {
    float* token_embedding_table; // (vocab_size, dim)
    float* rms_att_weight;        // (n_layers, dim)
    float* rms_ffn_weight;        // (n_layers, dim)
    float* wq;                    // (n_layers, dim, n_heads * head_size)
    float* wk;                    // (n_layers, dim, n_kv_heads * head_size)
    float* wv;                    // (n_layers, dim, n_kv_heads * head_size)
    float* wo;                    // (n_layers, n_heads * head_size, dim)
    float* w1;                    // (n_layers, hidden_dim, dim)
    float* w2;                    // (n_layers, dim, hidden_dim)
    float* w3;                    // (n_layers, hidden_dim, dim)
    float* rms_final_weight;      // (dim,)
    float* wcls;                  // classifier weight (vocab_size, dim)
} TransformerWeights;

// ---------------------------------------------------------------------------
// Activation Buffers for One Forward Pass
// ---------------------------------------------------------------------------
typedef struct {
    float* x;           // current token embedding (dim,)
    float* xb;          // residual buffer (dim,)
    float* xb2;         // second residual buffer (dim,)
    float* hb;          // ffn hidden state buffer (hidden_dim,)
    float* hb2;         // ffn gate buffer (hidden_dim,)
    float* q;           // query projection (dim,)
    float* k;           // key projection (kv_dim,)
    float* v;           // value projection (kv_dim,)
    float* att;         // attention scores (n_heads, seq_len)
    float* logits;      // output logits (vocab_size,)
    float* key_cache;   // KV-cache keys   (n_layers, seq_len, kv_dim)
    float* value_cache; // KV-cache values (n_layers, seq_len, kv_dim)
} RunState;

// ---------------------------------------------------------------------------
// Top-level Transformer object
// ---------------------------------------------------------------------------
typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------
typedef struct {
    char*  str;
    int    id;
} TokenIndex;

typedef struct {
    char**      vocab;
    float*      vocab_scores;
    TokenIndex* sorted_vocab;
    int         vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------
typedef struct {
    float   prob;
    int     index;
} ProbIndex;

typedef struct {
    int          vocab_size;
    ProbIndex*   probindex;
    float        temperature;
    float        topp;
    unsigned long long rng_state;
} Sampler;

// ---------------------------------------------------------------------------
// Function Declarations
// ---------------------------------------------------------------------------
void build_transformer(Transformer* t, char* checkpoint_path);
void free_transformer(Transformer* t);
float* forward(Transformer* t, int token, int pos);

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer* t);
char* decode(Tokenizer* t, int prev_token, int token);
void encode(Tokenizer* t, char* text, int8_t bos, int8_t eos, int* tokens, int* n_tokens);

void build_sampler(Sampler* s, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler* s);
int  sample(Sampler* s, float* logits);

void generate(Transformer* t, Tokenizer* tok, Sampler* s, char* prompt, int steps);

long time_in_ms();
