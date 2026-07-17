#ifndef MLXD_WEIGHTS_H
#define MLXD_WEIGHTS_H

#include "model/model.h"

#include <mlx/c/mlx.h>
#include <stdbool.h>
#include <stddef.h>

/* CPU-testable, no mlx calls */
int weights_enumerate_shards(const char *model_dir, char ***paths,
                             size_t *count, bool *from_index);
void weights_free_shard_paths(char **paths, size_t count);

int weights_tensor_name(char *buf, size_t n, const model_config_t *cfg,
                        int layer, const char *suffix);

/* Engine-thread only (mlx calls) */
typedef struct {
    mlx_array weight;
    mlx_array scales;
    mlx_array biases;
    bool      quantized;
} weight_triplet_t;

typedef struct weights {
    mlx_map_string_to_array   params;
    mlx_map_string_to_string  meta;
    size_t                    count;
    size_t                    total_bytes;
} weights_t;

int    weights_load(weights_t *w, const char *model_dir,
                    const model_config_t *cfg, char *err, size_t errlen);
int    weights_get(mlx_array *out, const weights_t *w, const char *name);
int    weights_get_triplet(weight_triplet_t *out, const weights_t *w,
                           const char *base);
size_t weights_count(const weights_t *w);
size_t weights_total_bytes(const weights_t *w);
void   weights_free(weights_t *w);

#endif
