#ifndef MLXD_ENGINE_KVCACHE_H
#define MLXD_ENGINE_KVCACHE_H

/* Dense per-layer KV cache. Engine-thread only: all mlx calls including
 * frees must happen on the engine thread. */

#include "mlxbridge/mlxbridge.h"

#include <stdbool.h>

typedef struct {
    mlx_array keys;   /* [B, H, capacity, D], valid rows: [0, offset) */
    mlx_array values;
    int offset;
    int capacity;
    int n_kv_heads;   /* recorded on first update */
    int head_dim;     /* recorded on first update */
    bool initialized;
} kv_entry_t;

typedef struct {
    kv_entry_t *entries; /* one per layer */
    int num_layers;
} kvcache_t;

int kvcache_init(kvcache_t *kv, int num_layers);
void kvcache_free(kvcache_t *kv);

/* Pre-update offset for a layer == rope offset for the incoming tokens. */
int kvcache_layer_offset(const kvcache_t *kv, int layer);

/* *k_view and *v_view must be live mlx_array (e.g. from mlx_array_new()); on
   success the previous values are freed and replaced; unchanged on failure.
   max_kv (0 = uncapped) caps the returned views at the last max_kv rows on
   decode steps only (new_len == 1); prefill views stay full because the
   sliding-window prefill mask handles older positions. offset semantics are
   unchanged: it keeps growing (absolute rope positions).
   Note: max_kv trims the returned view only; backing storage and offset grow
   unbounded (D0 scope; ring-buffer compaction is a follow-up). */
int kvcache_update(kvcache_t *kv, int layer, mlx_array new_k, mlx_array new_v,
                   int max_kv, mlx_array *k_view, mlx_array *v_view,
                   mlx_stream s);

int kvcache_capacity(const kvcache_t *kv, int layer);

/* Read-only view into an already-updated layer's cache. Returns the
   [B,H,len,D] slice, optionally trimmed to the last max_kv rows on decode
   (q_len == 1). Fails if the layer has never been updated (not initialized).
   q_len==1 must stay aligned with kvcache_update's new_len==1 trim logic. */
int kvcache_view(const kvcache_t *kv, int layer, int max_kv, int q_len,
                 mlx_array *k_view, mlx_array *v_view, mlx_stream s);

#endif /* MLXD_ENGINE_KVCACHE_H */
