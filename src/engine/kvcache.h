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
    bool initialized;
} kv_entry_t;

typedef struct {
    kv_entry_t *entries; /* one per layer */
    int num_layers;
    int n_kv_heads;
    int head_dim;
} kvcache_t;

int kvcache_init(kvcache_t *kv, int num_layers, int n_kv_heads, int head_dim);
void kvcache_free(kvcache_t *kv);

/* Pre-update offset for a layer == rope offset for the incoming tokens. */
int kvcache_layer_offset(const kvcache_t *kv, int layer);

int kvcache_update(kvcache_t *kv, int layer, mlx_array new_k, mlx_array new_v,
                   mlx_array *k_view, mlx_array *v_view, mlx_stream s);

int kvcache_capacity(const kvcache_t *kv, int layer);

#endif /* MLXD_ENGINE_KVCACHE_H */
