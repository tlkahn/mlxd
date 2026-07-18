#ifndef MLXD_ENGINE_FORWARD_H
#define MLXD_ENGINE_FORWARD_H

#include "mlxbridge/mlxbridge.h"
#include "model/model.h"
#include "model/weights.h"
#include "engine/kvcache.h"

int fwd_linear(mlx_array *out, mlx_array x, const weight_triplet_t *tri,
               const model_config_t *cfg, bool bias, mlx_stream s);

int fwd_embed(mlx_array *out, mlx_array ids, const weights_t *w,
              const model_config_t *cfg, mlx_stream s);

int fwd_rmsnorm(mlx_array *out, mlx_array x, mlx_array weight, float eps,
                mlx_stream s);

int fwd_attention(mlx_array *out, mlx_array x, int layer,
                  const weights_t *w, const model_config_t *cfg,
                  kvcache_t *kv, mlx_stream s);

int fwd_swiglu(mlx_array *out, mlx_array x, int layer,
               const weights_t *w, const model_config_t *cfg, mlx_stream s);

int fwd_decoder_layer(mlx_array *out, mlx_array x, int layer,
                      const weights_t *w, const model_config_t *cfg,
                      kvcache_t *kv, mlx_stream s);

#endif /* MLXD_ENGINE_FORWARD_H */
