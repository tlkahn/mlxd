#include "engine/kvcache.h"

#include <stdlib.h>

int kvcache_init(kvcache_t *kv, int num_layers) {
    if (!kv || num_layers <= 0) return -1;
    kv->entries = calloc((size_t)num_layers, sizeof(kv_entry_t));
    if (!kv->entries) return -1;
    kv->num_layers = num_layers;
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
                   int max_kv, mlx_array *k_view, mlx_array *v_view,
                   mlx_stream s) {
    if (!kv || !kv->entries || layer < 0 || layer >= kv->num_layers) return -1;
    if (max_kv < 0) return -1;

    if ((int)mlx_array_ndim(new_k) != 4 || (int)mlx_array_ndim(new_v) != 4)
        return -1;

    int H = mlx_array_dim(new_k, 1);
    int D = mlx_array_dim(new_k, 3);

    kv_entry_t *e = &kv->entries[layer];
    if (e->initialized) {
        if (H != e->n_kv_heads || D != e->head_dim) return -1;
    }

    if (mlx_array_dim(new_v, 1) != H || mlx_array_dim(new_v, 3) != D)
        return -1;
    if (mlx_array_dim(new_k, 0) != mlx_array_dim(new_v, 0) ||
        mlx_array_dim(new_k, 2) != mlx_array_dim(new_v, 2))
        return -1;

    int B = mlx_array_dim(new_k, 0);
    int new_len = mlx_array_dim(new_k, 2);
    mlx_dtype dt = mlx_array_dtype(new_k);

    int needed = e->offset + new_len;
    if (!e->initialized || needed > e->capacity) {
        if (grow_entry(e, B, H, D, needed, dt, s) != 0) return -1;
        e->n_kv_heads = H;
        e->head_dim = D;
    }

    int start[] = {0, 0, e->offset, 0};
    int stop[]  = {B, H, e->offset + new_len, D};
    int str[]   = {1, 1, 1, 1};

    mlx_array updated_k = mlx_array_new();
    mlx_array updated_v = mlx_array_new();
    mlx_array local_kview = mlx_array_new();
    mlx_array local_vview = mlx_array_new();

    if (!MLXB_CHECK(mlx_slice_update(&updated_k, e->keys, new_k, start, 4, stop, 4, str, 4, s)))
        goto cleanup;
    if (!MLXB_CHECK(mlx_slice_update(&updated_v, e->values, new_v, start, 4, stop, 4, str, 4, s)))
        goto cleanup;

    int new_off = e->offset + new_len;
    int view_start = (max_kv > 0 && new_len == 1 && new_off > max_kv)
                         ? new_off - max_kv : 0;
    int vstart[] = {0, 0, view_start, 0};
    int vstop[]  = {B, H, new_off, D};

    if (new_off == e->capacity && view_start == 0) {
        if (!MLXB_CHECK(mlx_array_set(&local_kview, updated_k))) goto cleanup;
        if (!MLXB_CHECK(mlx_array_set(&local_vview, updated_v))) goto cleanup;
    } else {
        if (!MLXB_CHECK(mlx_slice(&local_kview, updated_k, vstart, 4, vstop, 4, str, 4, s)))
            goto cleanup;
        if (!MLXB_CHECK(mlx_slice(&local_vview, updated_v, vstart, 4, vstop, 4, str, 4, s)))
            goto cleanup;
    }

    mlx_array_free(e->keys);
    mlx_array_free(e->values);
    e->keys = updated_k;
    e->values = updated_v;
    e->offset = new_off;

    mlx_array_free(*k_view);
    *k_view = local_kview;
    mlx_array_free(*v_view);
    *v_view = local_vview;
    return 0;

cleanup:
    mlx_array_free(local_vview);
    mlx_array_free(local_kview);
    mlx_array_free(updated_v);
    mlx_array_free(updated_k);
    return -1;
}

int kvcache_view(const kvcache_t *kv, int layer, int max_kv, int q_len,
                 mlx_array *k_view, mlx_array *v_view, mlx_stream s) {
    if (!kv || !kv->entries || layer < 0 || layer >= kv->num_layers)
        return -1;
    if (!k_view || !v_view) return -1;

    const kv_entry_t *e = &kv->entries[layer];
    if (!e->initialized || e->offset <= 0) return -1;

    int B = mlx_array_dim(e->keys, 0);
    int H = e->n_kv_heads;
    int D = e->head_dim;
    int off = e->offset;

    int view_start = (max_kv > 0 && q_len == 1 && off > max_kv)
                         ? off - max_kv : 0;
    int str[] = {1, 1, 1, 1};
    int vstart[] = {0, 0, view_start, 0};
    int vstop[]  = {B, H, off, D};

    mlx_array local_kview = mlx_array_new();
    mlx_array local_vview = mlx_array_new();

    if (view_start == 0 && off == e->capacity) {
        if (!MLXB_CHECK(mlx_array_set(&local_kview, e->keys))) goto fail;
        if (!MLXB_CHECK(mlx_array_set(&local_vview, e->values))) goto fail;
    } else {
        if (!MLXB_CHECK(mlx_slice(&local_kview, e->keys, vstart, 4, vstop, 4, str, 4, s)))
            goto fail;
        if (!MLXB_CHECK(mlx_slice(&local_vview, e->values, vstart, 4, vstop, 4, str, 4, s)))
            goto fail;
    }

    mlx_array_free(*k_view);
    *k_view = local_kview;
    mlx_array_free(*v_view);
    *v_view = local_vview;
    return 0;

fail:
    mlx_array_free(local_vview);
    mlx_array_free(local_kview);
    return -1;
}
