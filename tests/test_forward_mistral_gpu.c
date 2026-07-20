#include "engine/emodel.h"
#include "engine/forward.h"
#include "engine/kvcache.h"
#include "mlxbridge/mlxbridge.h"
#include "model/model.h"
#include "model/weights.h"
#include "gpu_test_util.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURES MLXD_FIXTURES_DIR

static mlx_stream gpu;

/* ---- helpers ---- */

static float max_abs_logit_diff(mlx_array a, mlx_array b, int vocab) {
    mlx_array af = mlx_array_new();
    mlx_array bf = mlx_array_new();
    assert(MLXB_CHECK(mlx_astype(&af, a, MLX_FLOAT32, gpu)));
    assert(MLXB_CHECK(mlx_astype(&bf, b, MLX_FLOAT32, gpu)));
    assert(MLXB_CHECK(mlx_array_eval(af)));
    assert(MLXB_CHECK(mlx_array_eval(bf)));

    const float *da = mlx_array_data_float32(af);
    const float *db = mlx_array_data_float32(bf);
    assert(da && db);

    float maxdiff = 0.0f;
    for (int i = 0; i < vocab; i++) {
        float d = fabsf(da[i] - db[i]);
        if (d > maxdiff) maxdiff = d;
    }

    mlx_array_free(bf);
    mlx_array_free(af);
    return maxdiff;
}

static int argmax_logits(mlx_array logits, int vocab) {
    mlx_array f32 = mlx_array_new();
    assert(MLXB_CHECK(mlx_astype(&f32, logits, MLX_FLOAT32, gpu)));
    assert(MLXB_CHECK(mlx_array_eval(f32)));
    const float *d = mlx_array_data_float32(f32);
    assert(d);
    int best = 0;
    for (int i = 1; i < vocab; i++) {
        if (d[i] > d[best]) best = i;
    }
    mlx_array_free(f32);
    return best;
}

static mlx_array make_ids(const int32_t *data, int n) {
    int shape[] = {1, n};
    return mlx_array_new_data(data, shape, 2, MLX_INT32);
}

/* ---- config flags ---- */

static void test_mistral_config_flags(engine_model_t *em) {
    assert(em->cfg.family == MODEL_MISTRAL);
    assert(em->cfg.has_sliding_window == true);
    assert(em->cfg.sliding_window == 8);
    assert(em->cfg.sliding_window_pattern == 0);
    assert(em->cfg.has_qk_norm == false);
    assert(em->cfg.attention_bias == false);
    assert(em->cfg.hidden_act == HIDDEN_ACT_SILU);
    assert(em->cfg.rope_theta == 10000.0f);
    assert(em->cfg.rope_scaling_type == NULL);
    assert(em->cfg.tie_word_embeddings == false);
    /* plain theta path: no custom freqs buffer */
    assert(em->rope_freqs.ctx == NULL);

    for (int i = 0; i < em->cfg.num_hidden_layers; i++)
        assert(model_layer_is_global(&em->cfg, i) == false);
}

static void test_mistral_tied_config_flags(engine_model_t *em) {
    assert(em->cfg.family == MODEL_MISTRAL);
    assert(em->cfg.tie_word_embeddings == true);
    assert(em->cfg.has_sliding_window == true);
    assert(em->cfg.sliding_window == 8);
    assert(em->cfg.sliding_window_pattern == 0);
    assert(em->cfg.has_qk_norm == false);
    assert(em->rope_freqs.ctx == NULL);
}

/* ---- weight mapping ---- */

static void test_mistral_weight_mapping(engine_model_t *em) {
    assert(em->w.count == 21);

    mlx_array probe = mlx_array_new();
    assert(weights_get(&probe, &em->w, "model.layers.0.self_attn.q_proj.weight") == 0);
    mlx_array_free(probe);

    probe = mlx_array_new();
    assert(weights_get(&probe, &em->w, "lm_head.weight") == 0);
    mlx_array_free(probe);

    probe = mlx_array_new();
    assert(weights_get(&probe, &em->w, "model.layers.0.self_attn.q_norm.weight") != 0);
    mlx_array_free(probe);
}

static void test_mistral_tied_weight_mapping(engine_model_t *em) {
    /* tied quantized: embed(3) + norm(1) + 2*(7*3 + 2) = 4 + 46 = 50 */
    assert(em->w.count == 50);

    mlx_array probe = mlx_array_new();
    assert(weights_get(&probe, &em->w, "lm_head.weight") != 0);
    mlx_array_free(probe);
}

/* ---- forward shape ---- */

static void test_mistral_forward_shape(engine_model_t *em) {
    int32_t ids_data[] = {1, 2, 3, 4, 5, 6};
    mlx_array ids = make_ids(ids_data, 6);

    kvcache_t kv;
    assert(kvcache_init(&kv, em->cfg.num_hidden_layers) == 0);

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

/* ---- short incremental (<= window) equals full ---- */

static void test_mistral_short_incremental_equals_full(engine_model_t *em) {
    /* total length 6 <= sliding_window=8 */
    int32_t prompt[] = {10, 20, 30, 40, 50, 60};
    int P = 6;

    kvcache_t kv_a;
    assert(kvcache_init(&kv_a, em->cfg.num_hidden_layers) == 0);

    mlx_array ids_all = make_ids(prompt, P);
    mlx_array logits_a = mlx_array_new();
    assert(model_forward(em, ids_all, &kv_a, true, &logits_a) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_a)));

    kvcache_t kv_b;
    assert(kvcache_init(&kv_b, em->cfg.num_hidden_layers) == 0);

    mlx_array ids_pre = make_ids(prompt, P - 1);
    assert(model_forward(em, ids_pre, &kv_b, false, NULL) == 0);

    mlx_array ids_dec = make_ids(&prompt[P - 1], 1);
    mlx_array logits_b = mlx_array_new();
    assert(model_forward(em, ids_dec, &kv_b, true, &logits_b) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_b)));

    int vocab = em->cfg.vocab_size;
    assert(argmax_logits(logits_a, vocab) == argmax_logits(logits_b, vocab));
    assert(max_abs_logit_diff(logits_a, logits_b, vocab) < 5e-2f);

    for (int L = 0; L < em->cfg.num_hidden_layers; L++) {
        assert(kvcache_layer_offset(&kv_a, L) == P);
        assert(kvcache_layer_offset(&kv_b, L) == P);
    }

    mlx_array_free(logits_b);
    mlx_array_free(ids_dec);
    mlx_array_free(ids_pre);
    kvcache_free(&kv_b);
    mlx_array_free(logits_a);
    mlx_array_free(ids_all);
    kvcache_free(&kv_a);
}

/* ---- Cycle 7: degenerate-window behavior ---- */

static void test_mistral_degenerate_window(engine_model_t *em) {
    const int W = em->cfg.sliding_window;
    assert(W == 8);
    assert(em->cfg.has_sliding_window == true);

    /* Inside / on boundary: S in {1, W} must match no-window attention. */
    static const int lens[] = {1, 8};
    for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
        int S = lens[li];
        int32_t ids_data[8];
        for (int i = 0; i < S; i++)
            ids_data[i] = (int32_t)(i + 1);

        mlx_array ids = make_ids(ids_data, S);

        kvcache_t kv_win;
        assert(kvcache_init(&kv_win, em->cfg.num_hidden_layers) == 0);
        mlx_array logits_win = mlx_array_new();
        assert(model_forward(em, ids, &kv_win, true, &logits_win) == 0);
        assert(MLXB_CHECK(mlx_array_eval(logits_win)));

        /* same weights, disable window */
        bool saved_sw = em->cfg.has_sliding_window;
        int saved_w = em->cfg.sliding_window;
        em->cfg.has_sliding_window = false;
        em->cfg.sliding_window = 0;

        kvcache_t kv_full;
        assert(kvcache_init(&kv_full, em->cfg.num_hidden_layers) == 0);
        mlx_array logits_full = mlx_array_new();
        assert(model_forward(em, ids, &kv_full, true, &logits_full) == 0);
        assert(MLXB_CHECK(mlx_array_eval(logits_full)));

        em->cfg.has_sliding_window = saved_sw;
        em->cfg.sliding_window = saved_w;

        float diff = max_abs_logit_diff(logits_win, logits_full, em->cfg.vocab_size);
        assert(diff < 2e-3f);

        mlx_array_free(logits_full);
        kvcache_free(&kv_full);
        mlx_array_free(logits_win);
        kvcache_free(&kv_win);
        mlx_array_free(ids);
    }

    /* Outside boundary: S = W+2 must diverge from no-window. */
    {
        int S = W + 2; /* 10 */
        int32_t ids_data[10];
        for (int i = 0; i < S; i++)
            ids_data[i] = (int32_t)(i + 3);

        mlx_array ids = make_ids(ids_data, S);

        kvcache_t kv_win;
        assert(kvcache_init(&kv_win, em->cfg.num_hidden_layers) == 0);
        mlx_array logits_win = mlx_array_new();
        assert(model_forward(em, ids, &kv_win, true, &logits_win) == 0);
        assert(MLXB_CHECK(mlx_array_eval(logits_win)));

        bool saved_sw = em->cfg.has_sliding_window;
        int saved_w = em->cfg.sliding_window;
        em->cfg.has_sliding_window = false;
        em->cfg.sliding_window = 0;

        kvcache_t kv_full;
        assert(kvcache_init(&kv_full, em->cfg.num_hidden_layers) == 0);
        mlx_array logits_full = mlx_array_new();
        assert(model_forward(em, ids, &kv_full, true, &logits_full) == 0);
        assert(MLXB_CHECK(mlx_array_eval(logits_full)));

        em->cfg.has_sliding_window = saved_sw;
        em->cfg.sliding_window = saved_w;

        float diff = max_abs_logit_diff(logits_win, logits_full, em->cfg.vocab_size);
        assert(diff > 1e-2f);

        mlx_array_free(logits_full);
        kvcache_free(&kv_full);
        mlx_array_free(logits_win);
        kvcache_free(&kv_win);
        mlx_array_free(ids);
    }
}

/* ---- Cycle 8: window-boundary incremental == full ---- */

static void test_mistral_window_boundary_incremental(engine_model_t *em) {
    const int W = em->cfg.sliding_window;
    assert(W == 8);
    /* P = W + 4 crosses the window */
    int P = W + 4; /* 12 */
    int32_t prompt[12];
    for (int i = 0; i < P; i++)
        prompt[i] = (int32_t)(7 + i * 3);

    /* Full path */
    kvcache_t kv_a;
    assert(kvcache_init(&kv_a, em->cfg.num_hidden_layers) == 0);
    mlx_array ids_all = make_ids(prompt, P);
    mlx_array logits_a = mlx_array_new();
    assert(model_forward(em, ids_all, &kv_a, true, &logits_a) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_a)));

    /* Incremental: prefill P-1, decode last */
    kvcache_t kv_b;
    assert(kvcache_init(&kv_b, em->cfg.num_hidden_layers) == 0);
    mlx_array ids_pre = make_ids(prompt, P - 1);
    assert(model_forward(em, ids_pre, &kv_b, false, NULL) == 0);

    mlx_array ids_dec = make_ids(&prompt[P - 1], 1);
    mlx_array logits_b = mlx_array_new();
    assert(model_forward(em, ids_dec, &kv_b, true, &logits_b) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_b)));

    int vocab = em->cfg.vocab_size;
    assert(argmax_logits(logits_a, vocab) == argmax_logits(logits_b, vocab));
    assert(max_abs_logit_diff(logits_a, logits_b, vocab) < 5e-2f);

    for (int L = 0; L < em->cfg.num_hidden_layers; L++) {
        assert(kvcache_layer_offset(&kv_a, L) == P);
        assert(kvcache_layer_offset(&kv_b, L) == P);
    }

    mlx_array_free(logits_b);
    mlx_array_free(ids_dec);
    mlx_array_free(ids_pre);
    kvcache_free(&kv_b);
    mlx_array_free(logits_a);
    mlx_array_free(ids_all);
    kvcache_free(&kv_a);
}

/* ---- suite runner ---- */

static void run_suite(const char *label, const char *path,
                      void (*config_test)(engine_model_t *),
                      void (*weight_test)(engine_model_t *),
                      int run_window_tests) {
    engine_model_t em;
    char err[256] = {0};
    if (engine_model_load(&em, path, err, sizeof(err)) != 0) {
        fprintf(stderr, "failed to load %s: %s\n", path, err);
        abort();
    }

    config_test(&em);
    printf("  %s config_flags: passed\n", label);

    weight_test(&em);
    printf("  %s weight_mapping: passed\n", label);

    test_mistral_forward_shape(&em);
    printf("  %s forward_shape: passed\n", label);

    test_mistral_short_incremental_equals_full(&em);
    printf("  %s short_incremental_equals_full: passed\n", label);

    if (run_window_tests) {
        test_mistral_degenerate_window(&em);
        printf("  %s degenerate_window: passed\n", label);

        test_mistral_window_boundary_incremental(&em);
        printf("  %s window_boundary_incremental: passed\n", label);
    }

    engine_model_free(&em);
}

int main(void) {
    gpu = mlxbridge_gpu_stream();

    char dense_path[512], tied_path[512];
    snprintf(dense_path, sizeof(dense_path), "%s/tiny_mistral", FIXTURES);
    snprintf(tied_path, sizeof(tied_path), "%s/tiny_mistral_tied", FIXTURES);

    run_suite("mistral_dense", dense_path,
              test_mistral_config_flags, test_mistral_weight_mapping,
              /*run_window_tests=*/1);

    run_suite("mistral_tied", tied_path,
              test_mistral_tied_config_flags, test_mistral_tied_weight_mapping,
              /*run_window_tests=*/0);

    printf("test_forward_mistral_gpu: all passed\n");
    return 0;
}
