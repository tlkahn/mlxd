#include "engine/emodel.h"
#include "engine/forward.h"
#include "engine/kvcache.h"
#include "mlxbridge/mlxbridge.h"
#include "model/model.h"
#include "model/weights.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

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

/* Load engine_model_t bypassing the support gate (gemma3 not yet whitelisted) */
static int load_gemma3(engine_model_t *em, const char *path) {
    memset(em, 0, sizeof(*em));
    if (model_config_load(&em->cfg, path) != 0) return -1;

    char err[256] = {0};
    if (weights_load(&em->w, path, &em->cfg, err, sizeof(err)) != 0) {
        fprintf(stderr, "weights_load: %s\n", err);
        model_config_free(&em->cfg);
        memset(em, 0, sizeof(*em));
        return -1;
    }
    em->stream = mlxbridge_gpu_stream();
    return 0;
}

/* ---- config flags at point of use ---- */

static void test_gemma3_config_flags(engine_model_t *em) {
    assert(em->cfg.family == MODEL_GEMMA3);
    assert(em->cfg.norm_has_offset == true);
    assert(em->cfg.scale_embeddings == true);
    assert(em->cfg.has_pre_ff_norm == true);
    assert(em->cfg.has_qk_norm == true);
    assert(em->cfg.tie_word_embeddings == true);
    assert(em->cfg.has_sliding_window == true);
    assert(em->cfg.sliding_window == 4);
    assert(em->cfg.sliding_window_pattern == 2);
    assert(em->cfg.query_pre_attn_scalar == 64);
    assert(em->cfg.rope_local_base_freq == 10000.0f);
}

/* ---- forward shape test ---- */

static void test_gemma3_forward_shape(engine_model_t *em) {
    int32_t ids_data[] = {1, 2, 3, 4, 5, 6};
    int ids_shape[] = {1, 6};
    mlx_array ids = mlx_array_new_data(ids_data, ids_shape, 2, MLX_INT32);

    kvcache_t kv;
    assert(kvcache_init(&kv, em->cfg.num_hidden_layers,
                        em->cfg.num_key_value_heads, em->cfg.head_dim) == 0);

    mlx_array logits = mlx_array_new();
    assert(model_forward(em, ids, &kv, true, &logits) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits)));
    assert(mlx_array_ndim(logits) == 2);
    assert(mlx_array_dim(logits, 0) == 1);
    assert(mlx_array_dim(logits, 1) == 256);
    assert(is_finite_f32(logits, gpu));

    mlx_array_free(logits);
    kvcache_free(&kv);
    mlx_array_free(ids);
}

/* ---- incremental == full ---- */

static void test_gemma3_incremental_equals_full(engine_model_t *em) {
    int32_t prompt[] = {10, 20, 30, 40, 50, 60};
    int P = 6;

    /* Path A: full context */
    kvcache_t kv_a;
    assert(kvcache_init(&kv_a, em->cfg.num_hidden_layers,
                        em->cfg.num_key_value_heads, em->cfg.head_dim) == 0);

    int shape_all[] = {1, P};
    mlx_array ids_all = mlx_array_new_data(prompt, shape_all, 2, MLX_INT32);
    mlx_array logits_a = mlx_array_new();
    assert(model_forward(em, ids_all, &kv_a, true, &logits_a) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_a)));

    /* Path B: prefill 5, decode 1 */
    kvcache_t kv_b;
    assert(kvcache_init(&kv_b, em->cfg.num_hidden_layers,
                        em->cfg.num_key_value_heads, em->cfg.head_dim) == 0);

    int shape_pre[] = {1, P - 1};
    mlx_array ids_pre = mlx_array_new_data(prompt, shape_pre, 2, MLX_INT32);
    assert(model_forward(em, ids_pre, &kv_b, false, NULL) == 0);

    int shape_dec[] = {1, 1};
    mlx_array ids_dec = mlx_array_new_data(&prompt[P - 1], shape_dec, 2, MLX_INT32);
    mlx_array logits_b = mlx_array_new();
    assert(model_forward(em, ids_dec, &kv_b, true, &logits_b) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_b)));

    /* Compare: argmax match + max-abs-diff < 5e-2 */
    mlx_array la_f32 = mlx_array_new();
    mlx_array lb_f32 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&la_f32, logits_a, MLX_FLOAT32, gpu));
    MLXB_CHECK(mlx_astype(&lb_f32, logits_b, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(la_f32)));
    assert(MLXB_CHECK(mlx_array_eval(lb_f32)));

    const float *da = mlx_array_data_float32(la_f32);
    const float *db = mlx_array_data_float32(lb_f32);
    int vocab = em->cfg.vocab_size;

    int argmax_a = 0, argmax_b = 0;
    for (int i = 1; i < vocab; i++) {
        if (da[i] > da[argmax_a]) argmax_a = i;
        if (db[i] > db[argmax_b]) argmax_b = i;
    }
    assert(argmax_a == argmax_b);

    float maxdiff = 0.0f;
    for (int i = 0; i < vocab; i++) {
        float d = fabsf(da[i] - db[i]);
        if (d > maxdiff) maxdiff = d;
    }
    assert(maxdiff < 5e-2f);

    mlx_array_free(lb_f32);
    mlx_array_free(la_f32);
    mlx_array_free(logits_b);
    mlx_array_free(ids_dec);
    mlx_array_free(ids_pre);
    kvcache_free(&kv_b);
    mlx_array_free(logits_a);
    mlx_array_free(ids_all);
    kvcache_free(&kv_a);
}

int main(void) {
    gpu = mlxbridge_gpu_stream();

    char path[512];
    snprintf(path, sizeof(path), "%s/tiny_gemma3", FIXTURES);

    engine_model_t em;
    if (load_gemma3(&em, path) != 0) {
        fprintf(stderr, "failed to load tiny_gemma3 fixture\n");
        return 1;
    }

    test_gemma3_config_flags(&em);
    printf("  test_gemma3_config_flags: passed\n");

    test_gemma3_forward_shape(&em);
    printf("  test_gemma3_forward_shape: passed\n");

    test_gemma3_incremental_equals_full(&em);
    printf("  test_gemma3_incremental_equals_full: passed\n");

    engine_model_free(&em);

    printf("test_forward_gemma3_gpu: all passed\n");
    return 0;
}
