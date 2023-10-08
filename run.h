#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif


// Forward declarations of the structs
typedef struct Transformer Transformer;
typedef struct Tokenizer Tokenizer;
typedef struct Sampler Sampler;

// Function declarations
void build_transformer(Transformer* transformer, char* checkpoint_path);
void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size);
void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void generate(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, char* prompt, int steps);
void chat(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, char* prompt, char* system_prompt, int steps);
void free_sampler(Sampler* sampler);
void free_tokenizer(Tokenizer* t);
void free_transformer(Transformer* transformer);

int llama2c(int argc, char *argv[]);
