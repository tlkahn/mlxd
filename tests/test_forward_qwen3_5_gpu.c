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

static void test_qwen3_5_config_flags(engine_model_t *em) {
    assert(em->cfg.family == MODEL_QWEN3_5);
    assert(em->cfg.attn_output_gate == true);
    assert(em->cfg.partial_rotary_factor == 0.25f);
    assert(em->cfg.has_qk_norm == true);
    assert(em->cfg.hidden_act == HIDDEN_ACT_SILU);
    assert(em->cfg.tie_word_embeddings == false);
    assert(em->cfg.head_dim == 16);
    assert(em->cfg.rope_theta == 10000000.0f);
    assert(em->cfg.linear_num_key_heads == 0);
    assert(em->cfg.num_experts == 0);
    /* plain theta rope - no custom freqs array */
    assert(em->rope_freqs.ctx == NULL);
}

static void test_qwen3_5_tied_config_flags(engine_model_t *em) {
    assert(em->cfg.family == MODEL_QWEN3_5);
    assert(em->cfg.tie_word_embeddings == true);
    assert(em->cfg.attn_output_gate == true);
    assert(em->cfg.partial_rotary_factor == 0.25f);
    assert(em->cfg.has_qk_norm == true);
    assert(em->cfg.quant_bits == 4);
}

static void test_qwen3_5_weight_mapping(engine_model_t *em) {
    /* Same tensor count as tiny_qwen3 dense (gate is shape, not extra name). */
    assert(em->w.count == 25);

    mlx_array probe = mlx_array_new();
    assert(weights_get(&probe, &em->w,
                       "model.layers.0.self_attn.q_proj.weight") == 0);
    /* Doubled q_proj: [2*H*D, hidden] = [128, 64] */
    assert(mlx_array_ndim(probe) == 2);
    assert(mlx_array_dim(probe, 0) == 2 * em->cfg.num_attention_heads *
                                          em->cfg.head_dim);
    assert(mlx_array_dim(probe, 1) == em->cfg.hidden_size);
    mlx_array_free(probe);

    probe = mlx_array_new();
    assert(weights_get(&probe, &em->w, "lm_head.weight") == 0);
    mlx_array_free(probe);

    probe = mlx_array_new();
    assert(weights_get(&probe, &em->w,
                       "model.layers.0.linear_attn.in_proj_qkv.weight") != 0);
    mlx_array_free(probe);
}

static void test_qwen3_5_tied_weight_mapping(engine_model_t *em) {
    /* tied quantized with qk_norm:
       embed(3) + norm(1) + 2*(7*3 matmul + 2 ln + 2 qk) = 4 + 2*25 = 54 */
    assert(em->w.count == 54);

    mlx_array probe = mlx_array_new();
    assert(weights_get(&probe, &em->w, "lm_head.weight") != 0);
    mlx_array_free(probe);
}

static void test_qwen3_5_forward_shape(engine_model_t *em) {
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

static void test_qwen3_5_incremental_equals_full(engine_model_t *em) {
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

    /* Compare: argmax match + max-abs-diff < 5e-2. bf16 prefill-vs-decode
       accumulation drift; same tolerance as sibling GPU tests. */
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

static void test_hybrid_config_load_rejected(void) {
    engine_model_t em;
    char err[256] = {0};
    int rc = engine_model_load(
        &em, FIXTURES "/model_config_qwen3_5_hybrid_dense", err, sizeof(err));
    assert(rc != 0);
    assert(strstr(err, "linear attention") != NULL);
}

static void run_suite(const char *label, const char *path,
                      void (*config_test)(engine_model_t *),
                      void (*weight_test)(engine_model_t *)) {
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

    test_qwen3_5_forward_shape(&em);
    printf("  %s forward_shape: passed\n", label);

    test_qwen3_5_incremental_equals_full(&em);
    printf("  %s incremental_equals_full: passed\n", label);

    engine_model_free(&em);
}

int main(void) {
    gpu = mlxbridge_gpu_stream();

    test_hybrid_config_load_rejected();
    printf("  test_hybrid_config_load_rejected: passed\n");

    char dense_path[512], tied_path[512];
    snprintf(dense_path, sizeof(dense_path), "%s/tiny_qwen3_5", FIXTURES);
    snprintf(tied_path, sizeof(tied_path), "%s/tiny_qwen3_5_tied", FIXTURES);

    run_suite("qwen3_5_dense", dense_path,
              test_qwen3_5_config_flags, test_qwen3_5_weight_mapping);

    run_suite("qwen3_5_tied", tied_path,
              test_qwen3_5_tied_config_flags, test_qwen3_5_tied_weight_mapping);

    printf("test_forward_qwen3_5_gpu: all passed\n");
    return 0;
}
