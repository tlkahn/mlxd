#include "engine/emodel.h"
#include "engine/forward.h"
#include "engine/kvcache.h"
#include "mlxbridge/mlxbridge.h"
#include "model/model.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef MLXD_FIXTURES_DIR
#error "MLXD_FIXTURES_DIR must be defined"
#endif

#define FIXTURES MLXD_FIXTURES_DIR

static mlx_stream gpu;

static int is_finite_f32(mlx_array a, mlx_stream s) {
    mlx_array f32 = mlx_array_new();
    if (!MLXB_CHECK(mlx_astype(&f32, a, MLX_FLOAT32, s))) {
        mlx_array_free(f32);
        return 0;
    }
    if (!MLXB_CHECK(mlx_array_eval(f32))) {
        mlx_array_free(f32);
        return 0;
    }
    size_t n = mlx_array_size(f32);
    const float *d = mlx_array_data_float32(f32);
    if (!d) { mlx_array_free(f32); return 0; }
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(d[i])) { mlx_array_free(f32); return 0; }
    }
    mlx_array_free(f32);
    return 1;
}

static float max_abs_diff_arr(mlx_array a, mlx_array b, mlx_stream s) {
    mlx_array af = mlx_array_new();
    mlx_array bf = mlx_array_new();
    MLXB_CHECK(mlx_astype(&af, a, MLX_FLOAT32, s));
    MLXB_CHECK(mlx_astype(&bf, b, MLX_FLOAT32, s));
    MLXB_CHECK(mlx_array_eval(af));
    MLXB_CHECK(mlx_array_eval(bf));
    size_t n = mlx_array_size(af);
    const float *da = mlx_array_data_float32(af);
    const float *db = mlx_array_data_float32(bf);
    float mx = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(da[i] - db[i]);
        if (d > mx) mx = d;
    }
    mlx_array_free(bf);
    mlx_array_free(af);
    return mx;
}

static int argmax_f32(mlx_array a, mlx_stream s) {
    mlx_array f32 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&f32, a, MLX_FLOAT32, s));
    MLXB_CHECK(mlx_array_eval(f32));
    size_t n = mlx_array_size(f32);
    const float *d = mlx_array_data_float32(f32);
    int best = 0;
    float best_v = d[0];
    for (size_t i = 1; i < n; i++) {
        if (d[i] > best_v) { best_v = d[i]; best = (int)i; }
    }
    mlx_array_free(f32);
    return best;
}


/* ---- config flags ---- */

static void test_gemma4_config_flags(engine_model_t *em) {
    assert(em->cfg.family == MODEL_GEMMA4);
    assert(em->cfg.attn_scale_one == true);
    assert(em->cfg.has_v_norm == true);
    assert(em->cfg.has_qk_norm == true);
    assert(em->cfg.has_pre_ff_norm == true);
    assert(em->cfg.hidden_act == HIDDEN_ACT_GELU_APPROX);
    assert(em->cfg.attention_k_eq_v == true);
    assert(em->cfg.rope_proportional == true);
    assert(em->cfg.num_kv_shared_layers == 2);
    assert(em->cfg.global_head_dim == 32);
    assert(em->cfg.num_global_key_value_heads == 1);
    assert(em->cfg.final_logit_softcapping == 30.0f);
    assert(em->cfg.hidden_size_per_layer_input == 8);
}

/* ---- forward shape: logits [1, vocab=256], all finite ---- */

static void test_gemma4_forward_shape(engine_model_t *em) {
    int32_t ids_data[] = {1, 2, 3, 4, 5};
    int ids_shape[] = {1, 5};
    mlx_array ids = mlx_array_new_data(ids_data, ids_shape, 2, MLX_INT32);

    kvcache_t kv;
    assert(kvcache_init(&kv, em->cfg.num_hidden_layers) == 0);

    mlx_array logits = mlx_array_new();
    assert(model_forward(em, ids, &kv, true, &logits) == 0);

    assert(MLXB_CHECK(mlx_array_eval(logits)));
    assert((int)mlx_array_ndim(logits) == 2);
    assert(mlx_array_dim(logits, 0) == 1);
    assert(mlx_array_dim(logits, 1) == 256);
    assert(is_finite_f32(logits, gpu));

    /* Softcap: all logits must satisfy |logit| <= cap */
    if (em->cfg.final_logit_softcapping > 0) {
        float cap = em->cfg.final_logit_softcapping;
        mlx_array lf = mlx_array_new();
        MLXB_CHECK(mlx_astype(&lf, logits, MLX_FLOAT32, gpu));
        MLXB_CHECK(mlx_array_eval(lf));
        size_t n = mlx_array_size(lf);
        const float *ld = mlx_array_data_float32(lf);
        for (size_t i = 0; i < n; i++) {
            assert(fabsf(ld[i]) <= cap + 0.1f);
        }
        mlx_array_free(lf);
    }

    /* KV sharing: only non-shared layers (0,1) should have advanced offset */
    assert(kvcache_layer_offset(&kv, 0) == 5);
    assert(kvcache_layer_offset(&kv, 1) == 5);
    assert(kvcache_layer_offset(&kv, 2) == 0);
    assert(kvcache_layer_offset(&kv, 3) == 0);

    /* Shared layers should not have capacity allocated */
    assert(kvcache_capacity(&kv, 2) == 0);
    assert(kvcache_capacity(&kv, 3) == 0);

    mlx_array_free(logits);
    kvcache_free(&kv);
    mlx_array_free(ids);
}

/* ---- incremental == full parity ---- */

static void test_gemma4_incremental_parity(engine_model_t *em) {
    int32_t ids_data[] = {10, 20, 30, 40, 50, 60};
    int full_shape[] = {1, 6};
    mlx_array ids_full = mlx_array_new_data(ids_data, full_shape, 2, MLX_INT32);

    /* Full forward: returns last-token logits [1, 256] */
    kvcache_t kv_full;
    assert(kvcache_init(&kv_full, em->cfg.num_hidden_layers) == 0);

    mlx_array logits_full = mlx_array_new();
    assert(model_forward(em, ids_full, &kv_full, true, &logits_full) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_full)));

    /* Incremental: prefill 5 + decode 1 */
    kvcache_t kv_inc;
    assert(kvcache_init(&kv_inc, em->cfg.num_hidden_layers) == 0);

    int pfx_shape[] = {1, 5};
    mlx_array ids_pfx = mlx_array_new_data(ids_data, pfx_shape, 2, MLX_INT32);
    assert(model_forward(em, ids_pfx, &kv_inc, false, NULL) == 0);

    int dec_shape[] = {1, 1};
    int32_t dec_data[] = {60};
    mlx_array ids_dec = mlx_array_new_data(dec_data, dec_shape, 2, MLX_INT32);
    mlx_array logits_dec = mlx_array_new();
    assert(model_forward(em, ids_dec, &kv_inc, true, &logits_dec) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_dec)));

    /* Full forward last-token logits should match decode logits */
    float diff = max_abs_diff_arr(logits_full, logits_dec, gpu);
    if (diff >= 5e-2f) {
        fprintf(stderr, "incremental parity FAIL: max_abs_diff = %f\n", (double)diff);
    }
    assert(diff < 5e-2f);

    assert(argmax_f32(logits_full, gpu) == argmax_f32(logits_dec, gpu));

    mlx_array_free(logits_dec);
    mlx_array_free(ids_dec);
    mlx_array_free(ids_pfx);
    kvcache_free(&kv_inc);
    mlx_array_free(logits_full);
    mlx_array_free(ids_full);
    kvcache_free(&kv_full);
}

/* ---- fwd_attention rejects unresolved KV source ---- */

static void test_gemma4_unresolved_kv_source(engine_model_t *em) {
    /* Mutate config: set num_kv_shared_layers=3 so boundary=1.
       Layer 1 (global) is shared but only layer 0 (local) is below boundary -
       no global source exists, so model_kv_source_layer returns -1. */
    model_config_t saved = em->cfg;

    em->cfg.num_kv_shared_layers = 3;
    assert(model_kv_source_layer(&em->cfg, 1) == -1);

    /* Build input [1,1,hidden_size] */
    int shape[] = {1, 1, em->cfg.hidden_size};
    mlx_array x = mlx_array_new();
    MLXB_CHECK(mlx_zeros(&x, shape, 3, MLX_BFLOAT16, gpu));

    kvcache_t kv;
    assert(kvcache_init(&kv, em->cfg.num_hidden_layers) == 0);

    mlx_array out = mlx_array_new();
    int rc = fwd_attention(&out, x, 1, &em->w, &em->cfg, &kv,
                           em->rope_freqs, gpu);
    assert(rc == -1);

    mlx_array_free(out);
    kvcache_free(&kv);
    mlx_array_free(x);
    em->cfg = saved;
}

int main(void) {
    gpu = mlxbridge_gpu_stream();

    engine_model_t em;
    char err[256] = {0};
    int rc = engine_model_load(&em, FIXTURES "/tiny_gemma4", err, sizeof(err));
    if (rc != 0) {
        fprintf(stderr, "SKIP: failed to load tiny_gemma4 fixture: %s\n", err);
        return 1;
    }

    test_gemma4_config_flags(&em);
    printf("  test_gemma4_config_flags: passed\n");

    test_gemma4_forward_shape(&em);
    printf("  test_gemma4_forward_shape: passed\n");

    test_gemma4_incremental_parity(&em);
    printf("  test_gemma4_incremental_parity: passed\n");

    test_gemma4_unresolved_kv_source(&em);
    printf("  test_gemma4_unresolved_kv_source: passed\n");

    engine_model_free(&em);
    printf("test_forward_gemma4_gpu: all passed\n");
    return 0;
}
