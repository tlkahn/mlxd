/* GPU tests for the Stage C sampler pipeline (src/engine/sampler.{h,c}).
 * Crafted logits only - no model load. Run via `make test-gpu`. */

#include "engine/sampler.h"
#include "mlxbridge/mlxbridge.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#define VOCAB 8

static mlx_array logits_1xv(const float *vals) {
    int shape[] = {1, VOCAB};
    return mlx_array_new_data(vals, shape, 2, MLX_FLOAT32);
}

/* argmax lives at index 2 */
static const float kLogits[VOCAB] = {0.1f, -2.0f, 3.5f, 1.0f, 0.0f, 2.9f, -1.0f, 0.5f};

/* ---- Cycle 1: temperature 0 hits the greedy limit ---- */

static void test_greedy_limit(void) {
    mlx_stream s = mlxbridge_gpu_stream();
    mlx_array logits = logits_1xv(kLogits);
    mlx_array nokey = mlx_array_new();

    sampling_params_t sp = SAMPLING_PARAMS_DEFAULT;
    sp.temperature = 0.0f;

    mlx_array tok = mlx_array_new();
    assert(sampler_sample_lazy(logits, &sp, nokey, false, s, &tok, NULL) == 0);
    int32_t id = -1;
    assert(mlxbridge_item_int32(&id, tok) == 0);
    assert(id == 2);

    mlx_array_free(tok);
    mlx_array_free(nokey);
    mlx_array_free(logits);
    mlx_stream_free(s);
}

/* ---- Cycle 2: top_k filter masks all but the k largest with -inf ---- */

/* Eval `arr` and copy its VOCAB floats into out. */
static void readback_1xv(mlx_array arr, float *out) {
    assert(mlx_array_eval(arr) == 0);
    const float *data = mlx_array_data_float32(arr);
    assert(data != NULL);
    for (int i = 0; i < VOCAB; i++)
        out[i] = data[i];
}

static void test_top_k_filter(void) {
    mlx_stream s = mlxbridge_gpu_stream();
    mlx_array logits = logits_1xv(kLogits);
    float out[VOCAB];

    /* k=3 keeps indices 2 (3.5), 5 (2.9), 3 (1.0); rest -inf */
    mlx_array filtered = mlx_array_new();
    assert(sampler_apply_top_k(logits, 3, s, &filtered) == 0);
    readback_1xv(filtered, out);
    for (int i = 0; i < VOCAB; i++) {
        if (i == 2 || i == 5 || i == 3)
            assert(out[i] == kLogits[i]);
        else
            assert(isinf(out[i]) && out[i] < 0);
    }
    mlx_array_free(filtered);

    /* k=-1 (disabled) and k >= vocab leave logits unchanged */
    int passthrough_ks[] = {-1, VOCAB, VOCAB + 5};
    for (size_t t = 0; t < sizeof(passthrough_ks) / sizeof(*passthrough_ks); t++) {
        mlx_array same = mlx_array_new();
        assert(sampler_apply_top_k(logits, passthrough_ks[t], s, &same) == 0);
        readback_1xv(same, out);
        for (int i = 0; i < VOCAB; i++)
            assert(out[i] == kLogits[i]);
        mlx_array_free(same);
    }

    mlx_array_free(logits);
    mlx_stream_free(s);
}

/* ---- Cycle 3: top_k=1 at high temperature still equals greedy ---- */

static void test_top_k1_equals_greedy(void) {
    mlx_stream s = mlxbridge_gpu_stream();
    mlx_array logits = logits_1xv(kLogits);
    mlx_array nokey = mlx_array_new();

    sampling_params_t sp = SAMPLING_PARAMS_DEFAULT;
    sp.temperature = 5.0f;
    sp.top_k = 1;

    for (int i = 0; i < 16; i++) {
        mlx_array tok = mlx_array_new();
        assert(sampler_sample_lazy(logits, &sp, nokey, false, s, &tok, NULL) == 0);
        int32_t id = -1;
        assert(mlxbridge_item_int32(&id, tok) == 0);
        assert(id == 2);
        mlx_array_free(tok);
    }

    mlx_array_free(nokey);
    mlx_array_free(logits);
    mlx_stream_free(s);
}

/* ---- Cycle 4: top_p nucleus filter ---- */

static void test_top_p_filter(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    /* unsorted probs {.15, .5, .05, .3}; nucleus for p=0.7 is {.5, .3} = idx 1, 3 */
    const float probs[4] = {0.15f, 0.5f, 0.05f, 0.3f};
    float vals[4];
    for (int i = 0; i < 4; i++)
        vals[i] = logf(probs[i]);
    int shape[] = {1, 4};
    mlx_array logits = mlx_array_new_data(vals, shape, 2, MLX_FLOAT32);
    float out[4];

    mlx_array filtered = mlx_array_new();
    assert(sampler_apply_top_p(logits, 0.7f, s, &filtered) == 0);
    assert(mlx_array_eval(filtered) == 0);
    const float *data = mlx_array_data_float32(filtered);
    assert(data != NULL);
    for (int i = 0; i < 4; i++)
        out[i] = data[i];
    for (int i = 0; i < 4; i++) {
        if (i == 1 || i == 3)
            assert(fabsf(out[i] - vals[i]) < 1e-6f);
        else
            assert(isinf(out[i]) && out[i] < 0);
    }
    mlx_array_free(filtered);

    /* top_p = 1.0 (edge case from the issue): logits unchanged */
    mlx_array same = mlx_array_new();
    assert(sampler_apply_top_p(logits, 1.0f, s, &same) == 0);
    assert(mlx_array_eval(same) == 0);
    data = mlx_array_data_float32(same);
    assert(data != NULL);
    for (int i = 0; i < 4; i++)
        assert(fabsf(data[i] - vals[i]) < 1e-6f);
    mlx_array_free(same);

    mlx_array_free(logits);
    mlx_stream_free(s);
}

int main(void) {
    test_greedy_limit();
    test_top_k_filter();
    test_top_k1_equals_greedy();
    test_top_p_filter();
    printf("test_sampler_gpu: all tests passed\n");
    return 0;
}
