/*
 * main.cpp
 * Entry point for the llama-inference-hip program.
 * Parses CLI arguments, builds the transformer + tokenizer + sampler,
 * and runs the generation loop.
 *
 * Usage:
 *   ./llama_hip <model.bin> [options]
 *
 * Options:
 *   -t <temperature>   (default 1.0)
 *   -p <topp>          top-p nucleus sampling cutoff (default 0.9)
 *   -s <seed>          RNG seed (default: time-based)
 *   -n <steps>         number of tokens to generate (default 256)
 *   -i <prompt>        prompt string
 *   -z <tokenizer>     path to tokenizer.bin (default: tokenizer.bin)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "model.h"

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <model.bin> [options]\n"
        "  -t <temperature>   sampling temperature (default 1.0)\n"
        "  -p <topp>          top-p nucleus cutoff (default 0.9)\n"
        "  -s <seed>          RNG seed (default: time-based)\n"
        "  -n <steps>         tokens to generate (default 256)\n"
        "  -i <prompt>        input prompt\n"
        "  -z <tokenizer>     tokenizer.bin path (default: tokenizer.bin)\n",
        prog);
    exit(1);
}

int main(int argc, char* argv[]) {
    if (argc < 2) usage(argv[0]);

    char* checkpoint_path = argv[1];
    char* tokenizer_path  = (char*)"tokenizer.bin";
    float temperature     = 1.0f;
    float topp            = 0.9f;
    int   steps           = 256;
    char* prompt          = NULL;
    unsigned long long rng_seed = (unsigned long long)time(NULL);

    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) usage(argv[0]);
        if      (!strcmp(argv[i], "-t")) temperature     = atof(argv[i+1]);
        else if (!strcmp(argv[i], "-p")) topp            = atof(argv[i+1]);
        else if (!strcmp(argv[i], "-s")) rng_seed        = (unsigned long long)atoll(argv[i+1]);
        else if (!strcmp(argv[i], "-n")) steps           = atoi(argv[i+1]);
        else if (!strcmp(argv[i], "-i")) prompt          = argv[i+1];
        else if (!strcmp(argv[i], "-z")) tokenizer_path  = argv[i+1];
        else usage(argv[0]);
    }
    if (temperature < 0.0f) temperature = 0.0f;
    if (topp < 0.0f || topp > 1.0f) topp = 0.9f;
    if (steps < 1) steps = 1;

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    fprintf(stderr, "Model: %s | dim=%d layers=%d heads=%d vocab=%d seq=%d\n",
            checkpoint_path,
            transformer.config.dim,
            transformer.config.n_layers,
            transformer.config.n_heads,
            transformer.config.vocab_size,
            transformer.config.seq_len);

    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
