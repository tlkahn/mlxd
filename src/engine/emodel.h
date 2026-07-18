#ifndef MLXD_ENGINE_EMODEL_H
#define MLXD_ENGINE_EMODEL_H

#include "mlxbridge/mlxbridge.h"
#include "model/model.h"
#include "model/weights.h"
#include "engine/kvcache.h"

#include <stdbool.h>

typedef struct engine_model {
    model_config_t cfg;
    weights_t w;
    mlx_stream stream;
    bool stub; /* true when loaded via MLXD_STUB_MODEL_PATH; no GPU state */
} engine_model_t;

int engine_model_check_supported(const model_config_t *cfg,
                                 char *err, size_t errlen);
int engine_model_load(engine_model_t *em, const char *model_dir,
                      char *err, size_t errlen);
void engine_model_free(engine_model_t *em);

/* *logits_out must be a live mlx_array (e.g. from mlx_array_new()); on success
   the previous value is freed and replaced. NULL when want_logits is false. */
int model_forward(engine_model_t *em, mlx_array ids, kvcache_t *kv,
                  bool want_logits, mlx_array *logits_out);

#endif /* MLXD_ENGINE_EMODEL_H */
