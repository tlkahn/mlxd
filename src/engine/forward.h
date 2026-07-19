#ifndef MLXD_ENGINE_FORWARD_H
#define MLXD_ENGINE_FORWARD_H

#include "mlxbridge/mlxbridge.h"
#include "model/model.h"
#include "model/weights.h"
#include "engine/kvcache.h"

#include <stdbool.h>

/* *out must be a live mlx_array (e.g. from mlx_array_new()). On success the
   previous value is freed and replaced; unchanged on failure. */

int fwd_linear(mlx_array *out, mlx_array x, const weight_triplet_t *tri,
               const model_config_t *cfg, mlx_stream s);

int fwd_embed(mlx_array *out, mlx_array ids, const weights_t *w,
              const model_config_t *cfg, mlx_stream s);

/* add_unit_offset applies the gemma-style (1 + weight) variant: bf16 1.0 is
   added to the raw weight before the norm. */
int fwd_rmsnorm(mlx_array *out, mlx_array x, mlx_array weight, float eps,
                bool add_unit_offset, mlx_stream s);

int fwd_attention(mlx_array *out, mlx_array x, int layer,
                  const weights_t *w, const model_config_t *cfg,
                  kvcache_t *kv, mlx_array rope_freqs, mlx_stream s);

/* Sliding-window additive masks (bf16, 0 / -inf) for SDPA mask mode "array".
   Prefill mask is [1,1,q_len,kv_len]: causal + window over absolute
   positions, where query row i sits at absolute position kv_len - q_len + i.
   Decode mask is [1,1,1,kv_len]: positions before kv_len - window masked. */
int fwd_sliding_window_mask(mlx_array *out, int q_len, int kv_len, int window,
                            mlx_stream s);

int fwd_sliding_window_decode_mask(mlx_array *out, int kv_len, int window,
                                   mlx_stream s);

int fwd_swiglu(mlx_array *out, mlx_array x, int layer,
               const weights_t *w, const model_config_t *cfg, mlx_stream s);

int fwd_decoder_layer(mlx_array *out, mlx_array x, int layer,
                      const weights_t *w, const model_config_t *cfg,
                      kvcache_t *kv, mlx_array rope_freqs, mlx_stream s);

int fwd_rope_llama3_freqs(const model_config_t *cfg, float *out, int n);

/* Build precomputed freqs array for the model's rope scaling type.
   rc 0 + out->ctx == NULL: family needs no custom freqs.
   rc 0 + live array: [head_dim/2] f32.
   rc -1: validation or alloc failure. */
int fwd_rope_freqs_build(mlx_array *out, const model_config_t *cfg);

#endif /* MLXD_ENGINE_FORWARD_H */
