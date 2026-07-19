#include "engine/forward.h"
#include "mlxbridge/mlxbridge.h"
#include "model/model.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static mlx_stream gpu;

/* --- GELU approx golden (mlx.nn.gelu_approx) --- */
#define GELU_GOLDEN_N 16
static const float gelu_input[] = {
    -6.00000000e+00f, -5.18750000e+00f, -4.40625000e+00f, -3.59375000e+00f, -2.79687500e+00f,
    -2.00000000e+00f, -1.20312500e+00f, -4.00390625e-01f, 4.00390625e-01f, 1.20312500e+00f,
    2.00000000e+00f, 2.79687500e+00f, 3.59375000e+00f, 4.40625000e+00f, 5.18750000e+00f,
    6.00000000e+00f
};
static const float gelu_expected[] = {
    -0.00000000e+00f, -0.00000000e+00f, -0.00000000e+00f, -0.00000000e+00f, -5.46264648e-03f,
    -4.68750000e-02f, -1.38671875e-01f, -1.37695312e-01f, 2.63671875e-01f, 1.06250000e+00f,
    1.95312500e+00f, 2.79687500e+00f, 3.59375000e+00f, 4.40625000e+00f, 5.18750000e+00f,
    6.00000000e+00f
};

/* --- Proportional RoPE golden: fixture case --- */
#define ROPE_FIXTURE_H 2
#define ROPE_FIXTURE_S 3
#define ROPE_FIXTURE_HD 16
#define ROPE_FIXTURE_DIMS 16
#define ROPE_FIXTURE_OFFSET 2
#define ROPE_FIXTURE_N 96
#define ROPE_FIXTURE_NFREQS 8
static const float rope_fixture_freqs[] = {
    8.00000000e+00f, 4.49873047e+01f, 2.52982208e+02f, 1.42262354e+03f, INFINITY, INFINITY,
    INFINITY, INFINITY
};
static const float rope_fixture_input[] = {
    3.04687500e-01f, -1.03906250e+00f, 7.50000000e-01f, 9.41406250e-01f, -1.95312500e+00f,
    -1.30468750e+00f, 1.27929688e-01f, -3.16406250e-01f, -1.68457031e-02f, -8.51562500e-01f,
    8.78906250e-01f, 7.77343750e-01f, 6.59179688e-02f, 1.12500000e+00f, 4.66796875e-01f,
    -8.59375000e-01f, 3.69140625e-01f, -9.57031250e-01f, 8.78906250e-01f, -4.98046875e-02f,
    -1.84570312e-01f, -6.79687500e-01f, 1.21875000e+00f, -1.54296875e-01f, -4.27734375e-01f,
    -3.51562500e-01f, 5.31250000e-01f, 3.65234375e-01f, 4.12109375e-01f, 4.31640625e-01f,
    2.14062500e+00f, -4.06250000e-01f, -5.11718750e-01f, -8.12500000e-01f, 6.17187500e-01f,
    1.13281250e+00f, -1.13769531e-01f, -8.39843750e-01f, -8.24218750e-01f, 6.52343750e-01f,
    7.42187500e-01f, 5.42968750e-01f, -6.64062500e-01f, 2.32421875e-01f, 1.16699219e-01f,
    2.18750000e-01f, 8.71093750e-01f, 2.23632812e-01f, 6.79687500e-01f, 6.73828125e-02f,
    2.89062500e-01f, 6.32812500e-01f, -1.46093750e+00f, -3.20312500e-01f, -4.70703125e-01f,
    -6.40625000e-01f, -2.75390625e-01f, 1.49218750e+00f, -8.67187500e-01f, 9.68750000e-01f,
    -1.67968750e+00f, -3.33984375e-01f, 1.63085938e-01f, 5.85937500e-01f, 7.10937500e-01f,
    7.92968750e-01f, -3.49609375e-01f, -4.62890625e-01f, 8.59375000e-01f, -1.91406250e-01f,
    -1.27343750e+00f, -1.13281250e+00f, -9.17968750e-01f, 4.98046875e-01f, 1.42578125e-01f,
    6.91406250e-01f, -4.27734375e-01f, 1.58203125e-01f, 6.25000000e-01f, -3.08593750e-01f,
    4.57031250e-01f, -6.60156250e-01f, -3.63281250e-01f, -3.80859375e-01f, -1.19531250e+00f,
    4.86328125e-01f, -4.68750000e-01f, 1.25122070e-02f, 4.80468750e-01f, 4.47265625e-01f,
    6.64062500e-01f, -9.86328125e-02f, -4.23828125e-01f, -7.95898438e-02f, -1.68750000e+00f,
    -1.44531250e+00f
};
static const float rope_fixture_expected[] = {
    2.98828125e-01f, -1.00000000e+00f, 7.42187500e-01f, 9.41406250e-01f, -1.95312500e+00f,
    -1.30468750e+00f, 1.27929688e-01f, -3.16406250e-01f, 5.90820312e-02f, -8.98437500e-01f,
    8.86718750e-01f, 7.77343750e-01f, 6.59179688e-02f, 1.12500000e+00f, 4.66796875e-01f,
    -8.59375000e-01f, 5.00000000e-01f, -9.29687500e-01f, 8.71093750e-01f, -5.05371094e-02f,
    -1.84570312e-01f, -6.79687500e-01f, 1.21875000e+00f, -1.54296875e-01f, -2.63671875e-01f,
    -4.14062500e-01f, 5.42968750e-01f, 3.65234375e-01f, 4.12109375e-01f, 4.31640625e-01f,
    2.14062500e+00f, -4.06250000e-01f, -8.04687500e-01f, -8.59375000e-01f, 6.28906250e-01f,
    1.13281250e+00f, -1.13769531e-01f, -8.39843750e-01f, -8.24218750e-01f, 6.52343750e-01f,
    4.06250000e-01f, 4.68750000e-01f, -6.52343750e-01f, 2.35351562e-01f, 1.16699219e-01f,
    2.18750000e-01f, 8.71093750e-01f, 2.23632812e-01f, 7.26562500e-01f, 9.99450684e-04f,
    2.96875000e-01f, 6.32812500e-01f, -1.46093750e+00f, -3.20312500e-01f, -4.70703125e-01f,
    -6.40625000e-01f, -9.86328125e-02f, 1.49218750e+00f, -8.63281250e-01f, 9.68750000e-01f,
    -1.67968750e+00f, -3.33984375e-01f, 1.63085938e-01f, 5.85937500e-01f, 9.96093750e-01f,
    7.57812500e-01f, -3.51562500e-01f, -4.64843750e-01f, 8.59375000e-01f, -1.91406250e-01f,
    -1.27343750e+00f, -1.13281250e+00f, -5.93750000e-01f, 5.50781250e-01f, 1.38671875e-01f,
    6.91406250e-01f, -4.27734375e-01f, 1.58203125e-01f, 6.25000000e-01f, -3.08593750e-01f,
    1.70898438e-01f, -6.95312500e-01f, -3.73046875e-01f, -3.80859375e-01f, -1.19531250e+00f,
    4.86328125e-01f, -4.68750000e-01f, 1.25122070e-02f, 6.40625000e-01f, 3.86718750e-01f,
    6.60156250e-01f, -9.96093750e-02f, -4.23828125e-01f, -7.95898438e-02f, -1.68750000e+00f,
    -1.44531250e+00f
};

/* --- Proportional RoPE golden: e2b case --- */
#define ROPE_E2B_H 2
#define ROPE_E2B_S 3
#define ROPE_E2B_HD 16
#define ROPE_E2B_DIMS 16
#define ROPE_E2B_OFFSET 2
#define ROPE_E2B_N 96
#define ROPE_E2B_NFREQS 8
static const float rope_e2b_freqs[] = {
    1.00000000e+00f, 5.62341309e+00f, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY
};
static const float rope_e2b_input[] = {
    3.04687500e-01f, -1.03906250e+00f, 7.50000000e-01f, 9.41406250e-01f, -1.95312500e+00f,
    -1.30468750e+00f, 1.27929688e-01f, -3.16406250e-01f, -1.68457031e-02f, -8.51562500e-01f,
    8.78906250e-01f, 7.77343750e-01f, 6.59179688e-02f, 1.12500000e+00f, 4.66796875e-01f,
    -8.59375000e-01f, 3.69140625e-01f, -9.57031250e-01f, 8.78906250e-01f, -4.98046875e-02f,
    -1.84570312e-01f, -6.79687500e-01f, 1.21875000e+00f, -1.54296875e-01f, -4.27734375e-01f,
    -3.51562500e-01f, 5.31250000e-01f, 3.65234375e-01f, 4.12109375e-01f, 4.31640625e-01f,
    2.14062500e+00f, -4.06250000e-01f, -5.11718750e-01f, -8.12500000e-01f, 6.17187500e-01f,
    1.13281250e+00f, -1.13769531e-01f, -8.39843750e-01f, -8.24218750e-01f, 6.52343750e-01f,
    7.42187500e-01f, 5.42968750e-01f, -6.64062500e-01f, 2.32421875e-01f, 1.16699219e-01f,
    2.18750000e-01f, 8.71093750e-01f, 2.23632812e-01f, 6.79687500e-01f, 6.73828125e-02f,
    2.89062500e-01f, 6.32812500e-01f, -1.46093750e+00f, -3.20312500e-01f, -4.70703125e-01f,
    -6.40625000e-01f, -2.75390625e-01f, 1.49218750e+00f, -8.67187500e-01f, 9.68750000e-01f,
    -1.67968750e+00f, -3.33984375e-01f, 1.63085938e-01f, 5.85937500e-01f, 7.10937500e-01f,
    7.92968750e-01f, -3.49609375e-01f, -4.62890625e-01f, 8.59375000e-01f, -1.91406250e-01f,
    -1.27343750e+00f, -1.13281250e+00f, -9.17968750e-01f, 4.98046875e-01f, 1.42578125e-01f,
    6.91406250e-01f, -4.27734375e-01f, 1.58203125e-01f, 6.25000000e-01f, -3.08593750e-01f,
    4.57031250e-01f, -6.60156250e-01f, -3.63281250e-01f, -3.80859375e-01f, -1.19531250e+00f,
    4.86328125e-01f, -4.68750000e-01f, 1.25122070e-02f, 4.80468750e-01f, 4.47265625e-01f,
    6.64062500e-01f, -9.86328125e-02f, -4.23828125e-01f, -7.95898438e-02f, -1.68750000e+00f,
    -1.44531250e+00f
};
static const float rope_e2b_expected[] = {
    -1.11328125e-01f, -6.75781250e-01f, 7.50000000e-01f, 9.41406250e-01f, -1.95312500e+00f,
    -1.30468750e+00f, 1.27929688e-01f, -3.16406250e-01f, 2.83203125e-01f, -1.15625000e+00f,
    8.78906250e-01f, 7.77343750e-01f, 6.59179688e-02f, 1.12500000e+00f, 4.66796875e-01f,
    -8.59375000e-01f, -3.04687500e-01f, -6.44531250e-01f, 8.78906250e-01f, -4.98046875e-02f,
    -1.84570312e-01f, -6.79687500e-01f, 1.21875000e+00f, -1.54296875e-01f, 4.74609375e-01f,
    -7.89062500e-01f, 5.31250000e-01f, 3.65234375e-01f, 4.12109375e-01f, 4.31640625e-01f,
    2.14062500e+00f, -4.06250000e-01f, 8.94531250e-01f, -9.68750000e-01f, 6.17187500e-01f,
    1.13281250e+00f, -1.13769531e-01f, -8.39843750e-01f, -8.24218750e-01f, 6.52343750e-01f,
    -9.76562500e-02f, -1.19140625e-01f, -6.64062500e-01f, 2.32421875e-01f, 1.16699219e-01f,
    2.18750000e-01f, 8.71093750e-01f, 2.23632812e-01f, -3.24707031e-02f, -4.57031250e-01f,
    2.89062500e-01f, 6.32812500e-01f, -1.46093750e+00f, -3.20312500e-01f, -4.70703125e-01f,
    -6.40625000e-01f, 7.34375000e-01f, 1.42187500e+00f, -8.67187500e-01f, 9.68750000e-01f,
    -1.67968750e+00f, -3.33984375e-01f, 1.63085938e-01f, 5.85937500e-01f, -5.74218750e-01f,
    4.29687500e-01f, -3.49609375e-01f, -4.62890625e-01f, 8.59375000e-01f, -1.91406250e-01f,
    -1.27343750e+00f, -1.13281250e+00f, 1.00781250e+00f, 8.32031250e-01f, 1.42578125e-01f,
    6.91406250e-01f, -4.27734375e-01f, 1.58203125e-01f, 6.25000000e-01f, -3.08593750e-01f,
    6.49414062e-02f, -7.92968750e-01f, -3.63281250e-01f, -3.80859375e-01f, -1.19531250e+00f,
    4.86328125e-01f, -4.68750000e-01f, 1.25122070e-02f, -6.60156250e-01f, -9.22851562e-02f,
    6.64062500e-01f, -9.86328125e-02f, -4.23828125e-01f, -7.95898438e-02f, -1.68750000e+00f,
    -1.44531250e+00f
};

/* ---- GELU approx test ---- */

static void test_gelu_approx_golden(void) {
    int shape[] = {1, GELU_GOLDEN_N};
    mlx_array x = mlx_array_new_data(gelu_input, shape, 2, MLX_FLOAT32);
    mlx_array xbf = mlx_array_new();
    MLXB_CHECK(mlx_astype(&xbf, x, MLX_BFLOAT16, gpu));

    mlx_array out = mlx_array_new();
    assert(fwd_gelu_approx(&out, xbf, gpu) == 0);

    mlx_array outf = mlx_array_new();
    MLXB_CHECK(mlx_astype(&outf, out, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(outf)));

    const float *d = mlx_array_data_float32(outf);
    for (int i = 0; i < GELU_GOLDEN_N; i++) {
        float diff = fabsf(d[i] - gelu_expected[i]);
        if (diff > 5e-3f) {
            fprintf(stderr, "gelu[%d]: got %f, expected %f (diff %f)\n",
                    i, (double)d[i], (double)gelu_expected[i], (double)diff);
        }
        assert(diff <= 5e-3f);
    }

    mlx_array_free(outf);
    mlx_array_free(out);
    mlx_array_free(xbf);
    mlx_array_free(x);
}

/* ---- Proportional RoPE test helper ---- */

static void test_rope_case(const float *freqs_data, int n_freqs,
                           const float *input_data, const float *expected_data,
                           int H, int S, int head_dim, int dims, int offset,
                           float rope_theta, float partial_rotary,
                           float prop_factor, const char *label) {
    int shape[] = {1, H, S, head_dim};
    mlx_array x = mlx_array_new_data(input_data, shape, 4, MLX_FLOAT32);
    mlx_array xbf = mlx_array_new();
    MLXB_CHECK(mlx_astype(&xbf, x, MLX_BFLOAT16, gpu));

    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.head_dim = head_dim;
    cfg.global_head_dim = head_dim;
    cfg.rope_theta = rope_theta;
    cfg.rope_proportional = true;
    cfg.rope_proportional_factor = prop_factor;
    cfg.partial_rotary_factor_global = partial_rotary;
    cfg.num_hidden_layers = 1;
    cfg.layer_is_global[0] = true;
    cfg.has_explicit_layer_types = true;

    /* Verify fwd_rope_proportional_freqs matches the oracle freqs */
    float computed_freqs[16];
    assert(n_freqs == head_dim / 2);
    assert(fwd_rope_proportional_freqs(&cfg, computed_freqs, n_freqs) == 0);
    for (int i = 0; i < n_freqs; i++) {
        if (isinf(freqs_data[i])) {
            assert(isinf(computed_freqs[i]));
        } else {
            float diff = fabsf(computed_freqs[i] - freqs_data[i]);
            if (diff > 1e-3f) {
                fprintf(stderr, "rope %s freqs[%d]: C=%f oracle=%f (diff %f)\n",
                        label, i, (double)computed_freqs[i],
                        (double)freqs_data[i], (double)diff);
            }
            assert(diff <= 1e-3f);
        }
    }

    /* Build freqs array and apply rope */
    int fshape[] = {n_freqs};
    mlx_array freqs = mlx_array_new_data(freqs_data, fshape, 1, MLX_FLOAT32);

    mlx_array out = mlx_array_new();
    assert(fwd_rope_apply(&out, xbf, dims, &cfg, 0, offset, freqs, gpu) == 0);

    mlx_array outf = mlx_array_new();
    MLXB_CHECK(mlx_astype(&outf, out, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(outf)));

    const float *d = mlx_array_data_float32(outf);
    int n = 1 * H * S * head_dim;
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(d[i] - expected_data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    if (max_diff > 5e-3f) {
        fprintf(stderr, "rope %s: max_diff = %f\n", label, (double)max_diff);
    }
    assert(max_diff <= 5e-3f);

    mlx_array_free(outf);
    mlx_array_free(out);
    mlx_array_free(freqs);
    mlx_array_free(xbf);
    mlx_array_free(x);
}

static void test_rope_proportional_fixture(void) {
    test_rope_case(rope_fixture_freqs, ROPE_FIXTURE_NFREQS,
                   rope_fixture_input, rope_fixture_expected,
                   ROPE_FIXTURE_H, ROPE_FIXTURE_S, ROPE_FIXTURE_HD,
                   ROPE_FIXTURE_DIMS, ROPE_FIXTURE_OFFSET,
                   1000000.0f, 0.5f, 8.0f, "fixture");
}

static void test_rope_proportional_e2b(void) {
    test_rope_case(rope_e2b_freqs, ROPE_E2B_NFREQS,
                   rope_e2b_input, rope_e2b_expected,
                   ROPE_E2B_H, ROPE_E2B_S, ROPE_E2B_HD,
                   ROPE_E2B_DIMS, ROPE_E2B_OFFSET,
                   1000000.0f, 0.25f, 1.0f, "e2b");
}

int main(void) {
    gpu = mlxbridge_gpu_stream();

    test_gelu_approx_golden();
    printf("  test_gelu_approx_golden: passed\n");

    test_rope_proportional_fixture();
    printf("  test_rope_proportional_fixture: passed\n");

    test_rope_proportional_e2b();
    printf("  test_rope_proportional_e2b: passed\n");

    printf("test_gemma4_goldens_gpu: all passed\n");
    return 0;
}
