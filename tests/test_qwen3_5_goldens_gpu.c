#include "engine/emodel.h"
#include "engine/forward.h"
#include "engine/kvcache.h"
#include "mlxbridge/mlxbridge.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define FIXTURES MLXD_FIXTURES_DIR

/* mlx-lm 0.31.3 (mlx_lm.models.qwen3_5) last-token logits on ids [1,2,3,4,5].
   Generated against the tiny fixtures after full_attention_interval:1 was
   added. Pins the D6 deltas: sigmoid gate multiply + partial rope dims. */

/* tiny_qwen3_5 (bf16 dense, untied lm_head) */
static const float dense_first8[] = {
    -1.75781250e-01f, -1.33789062e-01f, 1.60156250e-01f, -5.55419922e-03f,
    -2.51953125e-01f, -4.02343750e-01f, 1.23046875e-01f, 2.69775391e-02f
};
static const int dense_argmax = 33;

/* tiny_qwen3_5_tied (4-bit affine quant, tied embeddings) */
static const float tied_first8[] = {
    7.41506517e-02f, -1.21642277e-01f, -1.48075938e-01f, 1.20444193e-01f,
    -6.43087775e-02f, 1.13397074e+00f, -5.17259240e-02f, 1.73729137e-01f
};
static const int tied_argmax = 5;

static mlx_stream gpu;

static void compare_last_logits(engine_model_t *em,
                                const float *expected_first8,
                                int expected_argmax,
                                float atol,
                                const char *label) {
    int32_t ids_data[] = {1, 2, 3, 4, 5};
    int ids_shape[] = {1, 5};
    mlx_array ids = mlx_array_new_data(ids_data, ids_shape, 2, MLX_INT32);

    kvcache_t kv;
    assert(kvcache_init(&kv, em->cfg.num_hidden_layers) == 0);

    mlx_array logits = mlx_array_new();
    assert(model_forward(em, ids, &kv, true, &logits) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits)));

    /* model_forward returns last-token logits [1, vocab] */
    assert(mlx_array_ndim(logits) == 2);
    assert(mlx_array_dim(logits, 0) == 1);
    assert(mlx_array_dim(logits, 1) == em->cfg.vocab_size);

    mlx_array f32 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&f32, logits, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(f32)));

    const float *d = mlx_array_data_float32(f32);
    assert(d != NULL);

    int vocab = em->cfg.vocab_size;
    int argmax = 0;
    for (int i = 1; i < vocab; i++) {
        if (d[i] > d[argmax]) argmax = i;
    }
    if (argmax != expected_argmax) {
        fprintf(stderr, "%s: argmax got %d expected %d (got_val=%f exp_val=%f)\n",
                label, argmax, expected_argmax,
                (double)d[argmax], (double)d[expected_argmax]);
    }
    assert(argmax == expected_argmax);

    float max_diff = 0.0f;
    for (int i = 0; i < 8; i++) {
        float diff = fabsf(d[i] - expected_first8[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > atol) {
            fprintf(stderr, "%s first8[%d]: got %f expected %f (diff %f)\n",
                    label, i, (double)d[i], (double)expected_first8[i],
                    (double)diff);
        }
        assert(diff <= atol);
    }
    printf("  %s: argmax=%d max_first8_diff=%g (atol=%g)\n",
           label, argmax, (double)max_diff, (double)atol);

    mlx_array_free(f32);
    mlx_array_free(logits);
    kvcache_free(&kv);
    mlx_array_free(ids);
}

static void test_fixture_golden(const char *relpath,
                                const float *expected_first8,
                                int expected_argmax,
                                float atol,
                                const char *label) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", FIXTURES, relpath);

    engine_model_t em;
    char err[256] = {0};
    if (engine_model_load(&em, path, err, sizeof(err)) != 0) {
        fprintf(stderr, "failed to load %s: %s\n", path, err);
        assert(0);
    }

    assert(em.cfg.family == MODEL_QWEN3_5);
    assert(em.cfg.attn_output_gate == true);
    assert(em.cfg.partial_rotary_factor == 0.25f);
    assert(em.cfg.full_attention_interval == 1);

    compare_last_logits(&em, expected_first8, expected_argmax, atol, label);
    engine_model_free(&em);
}

int main(void) {
    gpu = mlxbridge_gpu_stream();

    /* bf16 dense: tight tolerance */
    test_fixture_golden("tiny_qwen3_5", dense_first8, dense_argmax,
                        1e-2f, "dense_bf16");
    printf("  test_qwen3_5_dense_golden: passed\n");

    /* 4-bit tied: looser tolerance for quant noise */
    test_fixture_golden("tiny_qwen3_5_tied", tied_first8, tied_argmax,
                        5e-2f, "tied_4bit");
    printf("  test_qwen3_5_tied_golden: passed\n");

    printf("test_qwen3_5_goldens_gpu: all passed\n");
    return 0;
}
