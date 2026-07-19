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

/* ---- config flags ---- */

static void test_llama3_config_flags(engine_model_t *em) {
    assert(em->cfg.family == MODEL_LLAMA);
    assert(em->cfg.rope_theta == 500000.0f);
    assert(em->cfg.rope_scaling_type != NULL);
    assert(strcmp(em->cfg.rope_scaling_type, "llama3") == 0);
    assert(em->cfg.rope_scaling_factor == 8.0f);
    assert(em->cfg.tie_word_embeddings == false);
    assert(em->cfg.has_qk_norm == false);
    assert(em->cfg.attention_bias == false);
    assert(em->cfg.hidden_act == HIDDEN_ACT_SILU);
    assert(em->rope_freqs.ctx != NULL);
}

static void test_llama3_tied_config_flags(engine_model_t *em) {
    assert(em->cfg.family == MODEL_LLAMA);
    assert(em->cfg.tie_word_embeddings == true);
    assert(em->cfg.quant_bits == 4);
    assert(em->cfg.rope_scaling_type != NULL);
    assert(strcmp(em->cfg.rope_scaling_type, "llama3") == 0);
    assert(em->cfg.has_qk_norm == false);
    assert(em->rope_freqs.ctx != NULL);
}

/* ---- weight mapping ---- */

static void test_llama3_weight_mapping(engine_model_t *em) {
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

static void test_llama3_tied_weight_mapping(engine_model_t *em) {
    /* tied quantized: 3 top (embed w/s/b) + 2*(7 matmul * 3 q/s/b + 2 norms) = 3 + 2*23 = 49
       But embed_tokens quantized = weight+scales+biases = 3 tensors, norm.weight = 1 tensor
       So: embed(3) + norm(1) + 2*(7*3 + 2) = 4 + 46 = 50 */
    assert(em->w.count == 50);

    mlx_array probe = mlx_array_new();
    assert(weights_get(&probe, &em->w, "lm_head.weight") != 0);
    mlx_array_free(probe);
}

/* ---- forward shape ---- */

static void test_llama3_forward_shape(engine_model_t *em) {
    int32_t ids_data[] = {1, 2, 3, 4, 5, 6};
    int ids_shape[] = {1, 6};
    mlx_array ids = mlx_array_new_data(ids_data, ids_shape, 2, MLX_INT32);

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

/* ---- incremental == full ---- */

static void test_llama3_incremental_equals_full(engine_model_t *em) {
    int32_t prompt[] = {10, 20, 30, 40, 50, 60};
    int P = 6;

    kvcache_t kv_a;
    assert(kvcache_init(&kv_a, em->cfg.num_hidden_layers) == 0);

    int shape_all[] = {1, P};
    mlx_array ids_all = mlx_array_new_data(prompt, shape_all, 2, MLX_INT32);
    mlx_array logits_a = mlx_array_new();
    assert(model_forward(em, ids_all, &kv_a, true, &logits_a) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_a)));

    kvcache_t kv_b;
    assert(kvcache_init(&kv_b, em->cfg.num_hidden_layers) == 0);

    int shape_pre[] = {1, P - 1};
    mlx_array ids_pre = mlx_array_new_data(prompt, shape_pre, 2, MLX_INT32);
    assert(model_forward(em, ids_pre, &kv_b, false, NULL) == 0);

    int shape_dec[] = {1, 1};
    mlx_array ids_dec = mlx_array_new_data(&prompt[P - 1], shape_dec, 2, MLX_INT32);
    mlx_array logits_b = mlx_array_new();
    assert(model_forward(em, ids_dec, &kv_b, true, &logits_b) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_b)));

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

/* ---- freqs builder ---- */

static void test_freqs_build_plain_cfg(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.family = MODEL_LLAMA;
    cfg.head_dim = 128;
    cfg.hidden_act = HIDDEN_ACT_SILU;
    cfg.partial_rotary_factor = 1.0f;
    cfg.rope_theta = 10000.0f;

    mlx_array out = mlx_array_new();
    assert(fwd_rope_freqs_build(&out, &cfg) == 0);
    assert(out.ctx == NULL);
    mlx_array_free(out);
}

static void test_freqs_build_llama3(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.family = MODEL_LLAMA;
    cfg.head_dim = 128;
    cfg.hidden_act = HIDDEN_ACT_SILU;
    cfg.partial_rotary_factor = 1.0f;
    cfg.rope_theta = 500000.0f;
    cfg.rope_scaling_type = "llama3";
    cfg.rope_scaling_factor = 8.0f;
    cfg.rope_low_freq_factor = 1.0f;
    cfg.rope_high_freq_factor = 4.0f;
    cfg.rope_original_max_position_embeddings = 8192;

    mlx_array out = mlx_array_new();
    assert(fwd_rope_freqs_build(&out, &cfg) == 0);
    assert(out.ctx != NULL);
    assert(mlx_array_ndim(out) == 1);
    assert(mlx_array_dim(out, 0) == 64);
    assert(mlx_array_dtype(out) == MLX_FLOAT32);

    assert(MLXB_CHECK(mlx_array_eval(out)));
    const float *d = mlx_array_data_float32(out);
    assert(d != NULL);

    float ref[64];
    assert(fwd_rope_llama3_freqs(&cfg, ref, 64) == 0);
    for (int i = 0; i < 64; i++) {
        float diff = fabsf(d[i] - ref[i]);
        float rel = diff / (fabsf(ref[i]) + 1e-30f);
        assert(rel < 1e-5f);
    }

    mlx_array_free(out);
}

static void test_freqs_build_degenerate(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.family = MODEL_LLAMA;
    cfg.head_dim = 128;
    cfg.hidden_act = HIDDEN_ACT_SILU;
    cfg.partial_rotary_factor = 1.0f;
    cfg.rope_scaling_type = "llama3";
    cfg.rope_scaling_factor = 0.0f;
    cfg.rope_original_max_position_embeddings = 0;

    mlx_array out = mlx_array_new();
    assert(fwd_rope_freqs_build(&out, &cfg) == -1);
    mlx_array_free(out);
}

/* ---- main ---- */

static void run_suite(const char *label, const char *path,
                      void (*config_test)(engine_model_t *),
                      void (*weight_test)(engine_model_t *)) {
    engine_model_t em;
    char err[256] = {0};
    if (engine_model_load(&em, path, err, sizeof(err)) != 0) {
        fprintf(stderr, "failed to load %s: %s\n", path, err);
        assert(0);
    }

    config_test(&em);
    printf("  %s config_flags: passed\n", label);

    weight_test(&em);
    printf("  %s weight_mapping: passed\n", label);

    test_llama3_forward_shape(&em);
    printf("  %s forward_shape: passed\n", label);

    test_llama3_incremental_equals_full(&em);
    printf("  %s incremental_equals_full: passed\n", label);

    engine_model_free(&em);
}

int main(void) {
    gpu = mlxbridge_gpu_stream();

    test_freqs_build_plain_cfg();
    printf("  test_freqs_build_plain_cfg: passed\n");

    test_freqs_build_llama3();
    printf("  test_freqs_build_llama3: passed\n");

    test_freqs_build_degenerate();
    printf("  test_freqs_build_degenerate: passed\n");

    char dense_path[512], tied_path[512];
    snprintf(dense_path, sizeof(dense_path), "%s/tiny_llama3", FIXTURES);
    snprintf(tied_path, sizeof(tied_path), "%s/tiny_llama3_tied", FIXTURES);

    run_suite("llama3_dense", dense_path,
              test_llama3_config_flags, test_llama3_weight_mapping);

    run_suite("llama3_tied", tied_path,
              test_llama3_tied_config_flags, test_llama3_tied_weight_mapping);

    printf("test_forward_llama_gpu: all passed\n");
    return 0;
}
