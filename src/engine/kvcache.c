#include "engine/kvcache.h"

#include <stdlib.h>

int kvcache_init(kvcache_t *kv, int num_layers, int n_kv_heads, int head_dim) {
    if (!kv || num_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0) return -1;
    kv->entries = calloc((size_t)num_layers, sizeof(kv_entry_t));
    if (!kv->entries) return -1;
    kv->num_layers = num_layers;
    kv->n_kv_heads = n_kv_heads;
    kv->head_dim = head_dim;
    return 0;
}

void kvcache_free(kvcache_t *kv) {
    if (!kv || !kv->entries) return;
    for (int i = 0; i < kv->num_layers; i++) {
        kv_entry_t *e = &kv->entries[i];
        if (e->keys.ctx) mlx_array_free(e->keys);
        if (e->values.ctx) mlx_array_free(e->values);
    }
    free(kv->entries);
    kv->entries = NULL;
    kv->num_layers = 0;
}

int kvcache_layer_offset(const kvcache_t *kv, int layer) {
    if (!kv || !kv->entries || layer < 0 || layer >= kv->num_layers) return -1;
    return kv->entries[layer].offset;
}

int kvcache_capacity(const kvcache_t *kv, int layer) {
    if (!kv || !kv->entries || layer < 0 || layer >= kv->num_layers) return -1;
    return kv->entries[layer].capacity;
}

static int grow_entry(kv_entry_t *e, int B, int H, int D,
                      int needed, mlx_dtype dt, mlx_stream s) {
    int nc = e->capacity > 0 ? e->capacity : 1;
    while (nc < needed) nc *= 2;

    int new_shape[] = {B, H, nc, D};
    mlx_array new_keys = mlx_array_new();
    mlx_array new_vals = mlx_array_new();
    int rc = -1;

    if (!MLXB_CHECK(mlx_zeros(&new_keys, new_shape, 4, dt, s))) goto cleanup;
    if (!MLXB_CHECK(mlx_zeros(&new_vals, new_shape, 4, dt, s))) goto cleanup;

    if (e->initialized && e->offset > 0) {
        int start[] = {0, 0, 0, 0};
        int stop[]  = {B, H, e->offset, D};
        int str[]   = {1, 1, 1, 1};

        mlx_array old_k = mlx_array_new();
        mlx_array old_v = mlx_array_new();
        if (!MLXB_CHECK(mlx_slice(&old_k, e->keys, start, 4, stop, 4, str, 4, s))) {
            mlx_array_free(old_v);
            mlx_array_free(old_k);
            goto cleanup;
        }
        if (!MLXB_CHECK(mlx_slice(&old_v, e->values, start, 4, stop, 4, str, 4, s))) {
            mlx_array_free(old_v);
            mlx_array_free(old_k);
            goto cleanup;
        }

        mlx_array tmp_k = mlx_array_new();
        mlx_array tmp_v = mlx_array_new();
        if (!MLXB_CHECK(mlx_slice_update(&tmp_k, new_keys, old_k, start, 4, stop, 4, str, 4, s)) ||
            !MLXB_CHECK(mlx_slice_update(&tmp_v, new_vals, old_v, start, 4, stop, 4, str, 4, s))) {
            mlx_array_free(tmp_v);
            mlx_array_free(tmp_k);
            mlx_array_free(old_v);
            mlx_array_free(old_k);
            goto cleanup;
        }
        mlx_array_free(old_k);
        mlx_array_free(old_v);
        mlx_array_free(new_keys);
        mlx_array_free(new_vals);
        new_keys = tmp_k;
        new_vals = tmp_v;
    }

    if (e->keys.ctx) mlx_array_free(e->keys);
    if (e->values.ctx) mlx_array_free(e->values);
    e->keys = new_keys;
    e->values = new_vals;
    e->capacity = nc;
    e->initialized = true;
    return 0;

cleanup:
    mlx_array_free(new_vals);
    mlx_array_free(new_keys);
    return rc;
}

int kvcache_update(kvcache_t *kv, int layer, mlx_array new_k, mlx_array new_v,
                   mlx_array *k_view, mlx_array *v_view, mlx_stream s) {
    if (!kv || !kv->entries || layer < 0 || layer >= kv->num_layers) return -1;
    kv_entry_t *e = &kv->entries[layer];

    int B = mlx_array_dim(new_k, 0);
    int H = mlx_array_dim(new_k, 1);
    int new_len = mlx_array_dim(new_k, 2);
    int D = mlx_array_dim(new_k, 3);
    mlx_dtype dt = mlx_array_dtype(new_k);

    int needed = e->offset + new_len;
    if (!e->initialized || needed > e->capacity) {
        if (grow_entry(e, B, H, D, needed, dt, s) != 0) return -1;
    }

    int start[] = {0, 0, e->offset, 0};
    int stop[]  = {B, H, e->offset + new_len, D};
    int str[]   = {1, 1, 1, 1};

    mlx_array updated_k = mlx_array_new();
    mlx_array updated_v = mlx_array_new();
    int rc = -1;

    if (!MLXB_CHECK(mlx_slice_update(&updated_k, e->keys, new_k, start, 4, stop, 4, str, 4, s)))
        goto cleanup;
    if (!MLXB_CHECK(mlx_slice_update(&updated_v, e->values, new_v, start, 4, stop, 4, str, 4, s)))
        goto cleanup;

    mlx_array_free(e->keys);
    mlx_array_free(e->values);
    e->keys = updated_k;
    e->values = updated_v;
    e->offset += new_len;

    int vstart[] = {0, 0, 0, 0};
    int vstop[]  = {B, H, e->offset, D};

    if (e->offset == e->capacity) {
        if (!MLXB_CHECK(mlx_array_set(k_view, e->keys))) goto view_err;
        if (!MLXB_CHECK(mlx_array_set(v_view, e->values))) goto view_err;
    } else {
        if (!MLXB_CHECK(mlx_slice(k_view, e->keys, vstart, 4, vstop, 4, str, 4, s)))
            goto view_err;
        if (!MLXB_CHECK(mlx_slice(v_view, e->values, vstart, 4, vstop, 4, str, 4, s)))
            goto view_err;
    }

    return 0;

view_err:
    return -1;
cleanup:
    mlx_array_free(updated_v);
    mlx_array_free(updated_k);
    return rc;
}
