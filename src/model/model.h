#ifndef MLXD_MODEL_H
#define MLXD_MODEL_H

#include <stdbool.h>

typedef struct {
    char *model_type;
    char *architectures;
    int   vocab_size;
    int   hidden_size;
    int   num_hidden_layers;
    int   num_attention_heads;
    int   num_key_value_heads;
    int   max_position_embeddings;
} model_config_t;

/* Load model config from a directory (reads config.json).
 * Returns 0 on success, -1 on error. */
int model_config_load(model_config_t *cfg, const char *model_dir);

void model_config_free(model_config_t *cfg);

#endif
