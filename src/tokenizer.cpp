/*
 * tokenizer.cpp
 * BPE Tokenizer implementation.
 * Adapted verbatim from karpathy/llama2.c - this component runs on CPU.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "model.h"

static int compare_tokens(const void* a, const void* b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size   = vocab_size;
    t->vocab        = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2]     = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE* file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "Could not load tokenizer %s\n", tokenizer_path); exit(1); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { exit(1); }
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { exit(1); }
        int len;
        if (fread(&len, sizeof(int), 1, file) != 1) { exit(1); }
        t->vocab[i] = (char*)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { exit(1); }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char* piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

static void safe_printf(char* piece) {
    if (!piece || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char b = piece[0];
        if (!(isprint(b) || isspace(b))) return;
    }
    printf("%s", piece);
}

static int str_lookup(char* str, TokenIndex* sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex* res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
}

void encode(Tokenizer* t, char* text, int8_t bos, int8_t eos, int* tokens, int* n_tokens) {
    if (!text) { fprintf(stderr, "NULL text\n"); exit(1); }
    if (!t->sorted_vocab) {
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id  = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    char* buf = (char*)malloc(t->max_token_length * 2 + 3);
    size_t len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dp = str_lookup((char*)" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dp;
    }
    for (char* c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) len = 0;
        buf[len++] = *c;
        buf[len]   = '\0';
        if ((*(c+1) & 0xC0) == 0x80 && len < 4) continue;
        int id = str_lookup(buf, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < len; i++)
                tokens[(*n_tokens)++] = (unsigned char)buf[i] + 3;
        }
        len = 0;
    }
    while (1) {
        float best_score = -1e10;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < *n_tokens - 1; i++) {
            sprintf(buf, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(buf, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id]; best_id = id; best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < *n_tokens - 1; i++) tokens[i] = tokens[i+1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(buf);
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------
static unsigned int random_u32(unsigned long long* s) {
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    return (unsigned int)((*s * 0x2545F4914F6CDD1Dull) >> 32);
}
static float random_f32(unsigned long long* s) {
    return (random_u32(s) >> 8) / 16777216.0f;
}
static int compare_prob(const void* a, const void* b) {
    float pa = ((ProbIndex*)a)->prob, pb = ((ProbIndex*)b)->prob;
    return (pa > pb) ? -1 : (pa < pb) ? 1 : 0;
}

void build_sampler(Sampler* s, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    s->vocab_size  = vocab_size;
    s->temperature = temperature;
    s->topp        = topp;
    s->rng_state   = rng_seed;
    s->probindex   = (ProbIndex*)malloc(vocab_size * sizeof(ProbIndex));
}
void free_sampler(Sampler* s) { free(s->probindex); }

int sample(Sampler* s, float* logits) {
    if (s->temperature == 0.0f) {
        int max_i = 0;
        for (int i = 1; i < s->vocab_size; i++)
            if (logits[i] > logits[max_i]) max_i = i;
        return max_i;
    }
    for (int i = 0; i < s->vocab_size; i++) logits[i] /= s->temperature;
    // softmax on CPU for sampling
    float mx = logits[0];
    for (int i = 1; i < s->vocab_size; i++) if (logits[i] > mx) mx = logits[i];
    float sm = 0.0f;
    for (int i = 0; i < s->vocab_size; i++) { logits[i] = expf(logits[i] - mx); sm += logits[i]; }
    for (int i = 0; i < s->vocab_size; i++) logits[i] /= sm;

    float coin = random_f32(&s->rng_state);
    if (s->topp <= 0.0f || s->topp >= 1.0f) {
        float cdf = 0.0f;
        for (int i = 0; i < s->vocab_size; i++) { cdf += logits[i]; if (coin < cdf) return i; }
        return s->vocab_size - 1;
    }
    float cutoff = (1.0f - s->topp) / (s->vocab_size - 1);
    int n0 = 0;
    for (int i = 0; i < s->vocab_size; i++) {
        if (logits[i] >= cutoff) { s->probindex[n0].index = i; s->probindex[n0].prob = logits[i]; n0++; }
    }
    qsort(s->probindex, n0, sizeof(ProbIndex), compare_prob);
    float cum = 0.0f; int last = n0 - 1;
    for (int i = 0; i < n0; i++) { cum += s->probindex[i].prob; if (cum > s->topp) { last = i; break; } }
    coin *= cum; cum = 0.0f;
    for (int i = 0; i <= last; i++) { cum += s->probindex[i].prob; if (coin < cum) return s->probindex[i].index; }
    return s->probindex[last].index;
}

// ---------------------------------------------------------------------------
// Generation Loop
// ---------------------------------------------------------------------------
#include <time.h>
long time_in_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

void generate(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, char* prompt, int steps) {
    char* empty = (char*)"";
    if (!prompt) prompt = empty;

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) { fprintf(stderr, "No prompt tokens\n"); exit(1); }

    long start = 0;
    int token = prompt_tokens[0], pos = 0;

    while (pos < steps) {
        float* logits = forward(transformer, token, pos);
        int next;
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;
        if (next == 1) break;
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;
        if (start == 0) start = time_in_ms();
    }
    printf("\n");
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %.2f\n", (pos - 1) / ((end - start) / 1000.0));
    }
    free(prompt_tokens);
}
