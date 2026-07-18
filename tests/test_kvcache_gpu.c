#include "engine/kvcache.h"
#include "mlxbridge/mlxbridge.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

/* ---- B2.1: init/free ---- */

static void test_init_free(void) {
    kvcache_t kv;
    int rc = kvcache_init(&kv, 2, 2, 16);
    assert(rc == 0);
    assert(kv.num_layers == 2);
    assert(kv.n_kv_heads == 2);
    assert(kv.head_dim == 16);
    for (int layer = 0; layer < 2; layer++) {
        assert(kvcache_layer_offset(&kv, layer) == 0);
        assert(kv.entries[layer].initialized == false);
        assert(kv.entries[layer].capacity == 0);
    }
    kvcache_free(&kv);

    /* free is idempotent on a zeroed struct */
    kvcache_t kv2 = {0};
    kvcache_free(&kv2);
}

/* ---- B2.2: single-step update + view ---- */

static void test_single_step(void) {
    mlx_stream s = mlxbridge_gpu_stream();
    kvcache_t kv;
    int rc = kvcache_init(&kv, 2, 2, 16);
    assert(rc == 0);

    /* new_k = new_v = ones([1,2,1,16], bf16) */
    float ones[32];
    for (int i = 0; i < 32; i++) ones[i] = 1.0f;
    int shape4[] = {1, 2, 1, 16};
    mlx_array ones_f32 = mlx_array_new_data(ones, shape4, 4, MLX_FLOAT32);
    mlx_array new_k = mlx_array_new();
    mlx_array new_v = mlx_array_new();
    MLXB_CHECK(mlx_astype(&new_k, ones_f32, MLX_BFLOAT16, s));
    MLXB_CHECK(mlx_astype(&new_v, ones_f32, MLX_BFLOAT16, s));

    mlx_array kview = mlx_array_new();
    mlx_array vview = mlx_array_new();

    /* First update */
    rc = kvcache_update(&kv, 0, new_k, new_v, &kview, &vview, s);
    assert(rc == 0);
    assert(kvcache_layer_offset(&kv, 0) == 1);

    assert(MLXB_CHECK(mlx_array_eval(kview)));
    assert(mlx_array_ndim(kview) == 4);
    assert(mlx_array_dim(kview, 0) == 1);
    assert(mlx_array_dim(kview, 1) == 2);
    assert(mlx_array_dim(kview, 2) == 1);
    assert(mlx_array_dim(kview, 3) == 16);

    /* Check all elements are 1.0 */
    mlx_array kview_f32 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&kview_f32, kview, MLX_FLOAT32, s));
    assert(MLXB_CHECK(mlx_array_eval(kview_f32)));
    const float *kdata = mlx_array_data_float32(kview_f32);
    assert(kdata);
    for (int i = 0; i < 32; i++)
        assert(fabsf(kdata[i] - 1.0f) < 1e-3f);

    /* Second update: new values = 2.0 */
    float twos[32];
    for (int i = 0; i < 32; i++) twos[i] = 2.0f;
    mlx_array twos_f32 = mlx_array_new_data(twos, shape4, 4, MLX_FLOAT32);
    mlx_array new_k2 = mlx_array_new();
    mlx_array new_v2 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&new_k2, twos_f32, MLX_BFLOAT16, s));
    MLXB_CHECK(mlx_astype(&new_v2, twos_f32, MLX_BFLOAT16, s));

    mlx_array kview2 = mlx_array_new();
    mlx_array vview2 = mlx_array_new();

    rc = kvcache_update(&kv, 0, new_k2, new_v2, &kview2, &vview2, s);
    assert(rc == 0);
    assert(kvcache_layer_offset(&kv, 0) == 2);

    assert(MLXB_CHECK(mlx_array_eval(kview2)));
    assert(mlx_array_dim(kview2, 2) == 2);  /* [1,2,2,16] */

    /* Check row t=0 preserved (1.0) and row t=1 is 2.0 */
    mlx_array kview2_f32 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&kview2_f32, kview2, MLX_FLOAT32, s));
    assert(MLXB_CHECK(mlx_array_eval(kview2_f32)));
    const float *k2data = mlx_array_data_float32(kview2_f32);
    assert(k2data);
    /* head 0: row 0 = 1.0, row 1 = 2.0 */
    for (int i = 0; i < 16; i++) {
        assert(fabsf(k2data[i] - 1.0f) < 1e-3f);       /* t=0 */
        assert(fabsf(k2data[16 + i] - 2.0f) < 1e-3f);  /* t=1 */
    }

    mlx_array_free(kview2_f32);
    mlx_array_free(vview2);
    mlx_array_free(kview2);
    mlx_array_free(new_v2);
    mlx_array_free(new_k2);
    mlx_array_free(twos_f32);
    mlx_array_free(kview_f32);
    mlx_array_free(vview);
    mlx_array_free(kview);
    mlx_array_free(new_v);
    mlx_array_free(new_k);
    mlx_array_free(ones_f32);
    kvcache_free(&kv);
}

/* ---- B2.3: doubling growth ---- */

static void test_doubling(void) {
    mlx_stream s = mlxbridge_gpu_stream();
    kvcache_t kv;
    assert(kvcache_init(&kv, 1, 2, 16) == 0);

    int expected_cap[] = {1, 2, 4, 4, 8};
    int shape4[] = {1, 2, 1, 16};

    for (int step = 0; step < 5; step++) {
        float vals[32];
        float v = (float)(step + 1);
        for (int i = 0; i < 32; i++) vals[i] = v;
        mlx_array vf32 = mlx_array_new_data(vals, shape4, 4, MLX_FLOAT32);
        mlx_array nk = mlx_array_new();
        mlx_array nv = mlx_array_new();
        MLXB_CHECK(mlx_astype(&nk, vf32, MLX_BFLOAT16, s));
        MLXB_CHECK(mlx_astype(&nv, vf32, MLX_BFLOAT16, s));

        mlx_array kview = mlx_array_new();
        mlx_array vview = mlx_array_new();
        assert(kvcache_update(&kv, 0, nk, nv, &kview, &vview, s) == 0);

        assert(kvcache_capacity(&kv, 0) == expected_cap[step]);
        assert(kvcache_layer_offset(&kv, 0) == step + 1);

        /* Verify all previously written rows are intact */
        assert(MLXB_CHECK(mlx_array_eval(kview)));
        assert(mlx_array_dim(kview, 2) == step + 1);
        mlx_array kf32 = mlx_array_new();
        MLXB_CHECK(mlx_astype(&kf32, kview, MLX_FLOAT32, s));
        assert(MLXB_CHECK(mlx_array_eval(kf32)));
        const float *kdata = mlx_array_data_float32(kf32);
        for (int t = 0; t <= step; t++) {
            float expect = (float)(t + 1);
            for (int d = 0; d < 16; d++)
                assert(fabsf(kdata[t * 16 + d] - expect) < 1e-2f);
        }

        mlx_array_free(kf32);
        mlx_array_free(vview);
        mlx_array_free(kview);
        mlx_array_free(nv);
        mlx_array_free(nk);
        mlx_array_free(vf32);
    }
    kvcache_free(&kv);
}

/* ---- B2.4: bulk update ---- */

static void test_bulk(void) {
    mlx_stream s = mlxbridge_gpu_stream();
    kvcache_t kv;
    assert(kvcache_init(&kv, 1, 2, 16) == 0);

    /* Bulk update: 3 tokens at once, shape [1,2,3,16] row-major */
    int shape_bulk[] = {1, 2, 3, 16};
    float bulk_vals[96];
    for (int h = 0; h < 2; h++)
        for (int t = 0; t < 3; t++)
            for (int d = 0; d < 16; d++)
                bulk_vals[h * 48 + t * 16 + d] = (float)(t + 1);
    mlx_array bf32 = mlx_array_new_data(bulk_vals, shape_bulk, 4, MLX_FLOAT32);
    mlx_array bk = mlx_array_new();
    mlx_array bv = mlx_array_new();
    MLXB_CHECK(mlx_astype(&bk, bf32, MLX_BFLOAT16, s));
    MLXB_CHECK(mlx_astype(&bv, bf32, MLX_BFLOAT16, s));

    mlx_array kview = mlx_array_new();
    mlx_array vview = mlx_array_new();
    assert(kvcache_update(&kv, 0, bk, bv, &kview, &vview, s) == 0);
    assert(kvcache_layer_offset(&kv, 0) == 3);
    assert(kvcache_capacity(&kv, 0) == 4);

    assert(MLXB_CHECK(mlx_array_eval(kview)));
    assert(mlx_array_dim(kview, 2) == 3);

    /* Then a single decode step */
    int shape1[] = {1, 2, 1, 16};
    float dvals[32];
    for (int i = 0; i < 32; i++) dvals[i] = 9.0f;
    mlx_array df32 = mlx_array_new_data(dvals, shape1, 4, MLX_FLOAT32);
    mlx_array dk = mlx_array_new();
    mlx_array dv = mlx_array_new();
    MLXB_CHECK(mlx_astype(&dk, df32, MLX_BFLOAT16, s));
    MLXB_CHECK(mlx_astype(&dv, df32, MLX_BFLOAT16, s));

    mlx_array kview2 = mlx_array_new();
    mlx_array vview2 = mlx_array_new();
    assert(kvcache_update(&kv, 0, dk, dv, &kview2, &vview2, s) == 0);
    assert(kvcache_layer_offset(&kv, 0) == 4);
    assert(kvcache_capacity(&kv, 0) == 4);

    assert(MLXB_CHECK(mlx_array_eval(kview2)));
    assert(mlx_array_dim(kview2, 2) == 4);

    mlx_array kf32 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&kf32, kview2, MLX_FLOAT32, s));
    assert(MLXB_CHECK(mlx_array_eval(kf32)));
    const float *kdata = mlx_array_data_float32(kf32);
    /* Rows 0-2 preserved, row 3 == 9.0 */
    for (int t = 0; t < 3; t++)
        for (int d = 0; d < 16; d++)
            assert(fabsf(kdata[t * 16 + d] - (float)(t + 1)) < 1e-2f);
    for (int d = 0; d < 16; d++)
        assert(fabsf(kdata[3 * 16 + d] - 9.0f) < 1e-2f);

    mlx_array_free(kf32);
    mlx_array_free(vview2);
    mlx_array_free(kview2);
    mlx_array_free(dv);
    mlx_array_free(dk);
    mlx_array_free(df32);
    mlx_array_free(vview);
    mlx_array_free(kview);
    mlx_array_free(bv);
    mlx_array_free(bk);
    mlx_array_free(bf32);
    kvcache_free(&kv);
}

int main(void) {
    test_init_free();
    printf("  test_init_free: passed\n");

    test_single_step();
    printf("  test_single_step: passed\n");

    test_doubling();
    printf("  test_doubling: passed\n");

    test_bulk();
    printf("  test_bulk: passed\n");

    printf("test_kvcache_gpu: all passed\n");
    return 0;
}
