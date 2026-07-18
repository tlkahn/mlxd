#ifndef MLXD_ENGINE_EMODEL_H
#define MLXD_ENGINE_EMODEL_H

#include "mlxbridge/mlxbridge.h"
#include "model/model.h"
#include "model/weights.h"
#include "engine/kvcache.h"

#include <stdbool.h>

typedef struct {
    model_config_t cfg;
    weights_t w;
    mlx_stream stream;
    float attn_scale;
    bool stub;
} engine_model_t;

int engine_model_load(engine_model_t *em, const char *model_dir);
void engine_model_free(engine_model_t *em);

int model_forward(engine_model_t *em, mlx_array ids, kvcache_t *kv,
                  bool want_logits, mlx_array *logits_out);

#endif /* MLXD_ENGINE_EMODEL_H */
