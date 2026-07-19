#include "engine/forward.h"
#include "model/model.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* --- 2a: arg validation -------------------------------------------------- */

static model_config_t make_valid_llama3_cfg(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.head_dim = 128;
    cfg.rope_scaling_type = "llama3";
    cfg.rope_theta = 500000.0f;
    cfg.rope_scaling_factor = 8.0f;
    cfg.rope_low_freq_factor = 1.0f;
    cfg.rope_high_freq_factor = 4.0f;
    cfg.rope_original_max_position_embeddings = 8192;
    return cfg;
}

static void test_arg_validation(void) {
    model_config_t cfg = make_valid_llama3_cfg();
    float out[64];

    assert(fwd_rope_llama3_freqs(NULL, out, 64) == -1);
    assert(fwd_rope_llama3_freqs(&cfg, NULL, 64) == -1);
    assert(fwd_rope_llama3_freqs(&cfg, out, 32) == -1);

    model_config_t cfg2 = cfg;
    cfg2.rope_scaling_type = "linear";
    assert(fwd_rope_llama3_freqs(&cfg2, out, 64) == -1);

    model_config_t cfg3 = cfg;
    cfg3.rope_scaling_type = NULL;
    assert(fwd_rope_llama3_freqs(&cfg3, out, 64) == -1);
}

/* --- 2a2: degenerate parameter validation -------------------------------- */

static void test_reject_factor_zero(void) {
    model_config_t cfg = make_valid_llama3_cfg();
    cfg.rope_scaling_factor = 0.0f;
    float out[64];
    assert(fwd_rope_llama3_freqs(&cfg, out, 64) == -1);
}

static void test_reject_factor_negative(void) {
    model_config_t cfg = make_valid_llama3_cfg();
    cfg.rope_scaling_factor = -1.0f;
    float out[64];
    assert(fwd_rope_llama3_freqs(&cfg, out, 64) == -1);
}

static void test_reject_original_max_zero(void) {
    model_config_t cfg = make_valid_llama3_cfg();
    cfg.rope_original_max_position_embeddings = 0;
    float out[64];
    assert(fwd_rope_llama3_freqs(&cfg, out, 64) == -1);
}

static void test_reject_low_freq_factor_zero(void) {
    model_config_t cfg = make_valid_llama3_cfg();
    cfg.rope_low_freq_factor = 0.0f;
    float out[64];
    assert(fwd_rope_llama3_freqs(&cfg, out, 64) == -1);
}

static void test_reject_low_freq_factor_negative(void) {
    model_config_t cfg = make_valid_llama3_cfg();
    cfg.rope_low_freq_factor = -1.0f;
    float out[64];
    assert(fwd_rope_llama3_freqs(&cfg, out, 64) == -1);
}

static void test_reject_high_leq_low(void) {
    model_config_t cfg = make_valid_llama3_cfg();
    cfg.rope_high_freq_factor = cfg.rope_low_freq_factor;
    float out[64];
    assert(fwd_rope_llama3_freqs(&cfg, out, 64) == -1);

    cfg.rope_high_freq_factor = cfg.rope_low_freq_factor - 0.1f;
    assert(fwd_rope_llama3_freqs(&cfg, out, 64) == -1);
}

static void test_reject_odd_head_dim(void) {
    model_config_t cfg = make_valid_llama3_cfg();
    cfg.head_dim = 127;
    float out[63];
    assert(fwd_rope_llama3_freqs(&cfg, out, 63) == -1);
}

static void test_reject_head_dim_lt_2(void) {
    model_config_t cfg = make_valid_llama3_cfg();
    cfg.head_dim = 1;
    float out[1];
    assert(fwd_rope_llama3_freqs(&cfg, out, 0) == -1);
}

/* --- 2b: high-frequency band passthrough --------------------------------- */

static void test_high_freq_passthrough(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.head_dim = 128;
    cfg.rope_scaling_type = "llama3";
    cfg.rope_theta = 500000.0f;
    cfg.rope_scaling_factor = 8.0f;
    cfg.rope_low_freq_factor = 1.0f;
    cfg.rope_high_freq_factor = 4.0f;
    cfg.rope_original_max_position_embeddings = 8192;

    float out[64];
    assert(fwd_rope_llama3_freqs(&cfg, out, 64) == 0);

    for (int i = 0; i < 29; i++) {
        double expected = pow(500000.0, 2.0 * i / 128.0);
        float diff = fabsf(out[i] - (float)expected);
        float rel = diff / (fabsf((float)expected) + 1e-30f);
        assert(rel < 1e-5f);
    }
}

/* --- 2c: low-frequency band ---------------------------------------------- */

static void test_low_freq_scaled(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.head_dim = 128;
    cfg.rope_scaling_type = "llama3";
    cfg.rope_theta = 500000.0f;
    cfg.rope_scaling_factor = 8.0f;
    cfg.rope_low_freq_factor = 1.0f;
    cfg.rope_high_freq_factor = 4.0f;
    cfg.rope_original_max_position_embeddings = 8192;

    float out[64];
    assert(fwd_rope_llama3_freqs(&cfg, out, 64) == 0);

    for (int i = 35; i < 64; i++) {
        double base_freq = pow(500000.0, 2.0 * i / 128.0);
        double expected = base_freq * 8.0;
        float diff = fabsf(out[i] - (float)expected);
        float rel = diff / (fabsf((float)expected) + 1e-30f);
        assert(rel < 1e-5f);
    }
}

/* --- 2d: medium band smooth interpolation -------------------------------- */

static void test_medium_band(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.head_dim = 128;
    cfg.rope_scaling_type = "llama3";
    cfg.rope_theta = 500000.0f;
    cfg.rope_scaling_factor = 8.0f;
    cfg.rope_low_freq_factor = 1.0f;
    cfg.rope_high_freq_factor = 4.0f;
    cfg.rope_original_max_position_embeddings = 8192;

    float out[64];
    assert(fwd_rope_llama3_freqs(&cfg, out, 64) == 0);

    /* dim 29: hand-computed smooth = 0.803621, expected = 4.61558892e+02 */
    float diff = fabsf(out[29] - 4.61558892e+02f);
    float rel = diff / 4.61558892e+02f;
    assert(rel < 1e-5f);
}

/* --- 2e: full golden vector ---------------------------------------------- */

/* Generated by: python3 replicating mlx-lm Llama3RoPE formula
   Parameters: dims=128, base=500000, factor=8.0, low=1.0, high=4.0, old_ctx=8192
   Cross-validated against mlx-lm _freqs (max rel diff < 3e-7). */
static const float golden[64] = {
    1.00000000e+00f, 1.22757040e+00f, 1.50692908e+00f, 1.84986152e+00f,
    2.27083524e+00f, 2.78761011e+00f, 3.42198765e+00f, 4.20073073e+00f,
    5.15669269e+00f, 6.33020328e+00f, 7.77077015e+00f, 9.53916739e+00f,
    1.17099995e+01f, 1.43748487e+01f, 1.76461387e+01f, 2.16618775e+01f,
    2.65914795e+01f, 3.26429130e+01f, 4.00714736e+01f, 4.91905547e+01f,
    6.03848687e+01f, 7.41266772e+01f, 9.09957144e+01f, 1.11703645e+02f,
    1.37124088e+02f, 1.68329471e+02f, 2.06636275e+02f, 2.53660574e+02f,
    3.11386211e+02f, 4.61558892e+02f, 7.28919519e+02f, 1.16719971e+03f,
    1.90532021e+03f, 3.19801723e+03f, 5.60199570e+03f, 1.04643970e+04f,
    1.28457840e+04f, 1.57691041e+04f, 1.93576854e+04f, 2.37629215e+04f,
    2.91706589e+04f, 3.58090373e+04f, 4.39581141e+04f, 5.39616796e+04f,
    6.62417603e+04f, 8.13164239e+04f, 9.98216347e+04f, 1.22538084e+05f,
    1.50424124e+05f, 1.84656201e+05f, 2.26678486e+05f, 2.78263798e+05f,
    3.41588401e+05f, 4.19323809e+05f, 5.14749494e+05f, 6.31891240e+05f,
    7.75690979e+05f, 9.52215282e+05f, 1.16891129e+06f, 1.43492090e+06f,
    1.76146641e+06f, 2.16232402e+06f, 2.65440495e+06f, 3.25846894e+06f,
};

static void test_golden_vector(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_llama3");
    assert(rc == 0);

    int n = cfg.head_dim / 2;
    assert(n == 64);

    float out[64];
    assert(fwd_rope_llama3_freqs(&cfg, out, n) == 0);

    for (int i = 0; i < n; i++) {
        float diff = fabsf(out[i] - golden[i]);
        float rel = diff / (fabsf(golden[i]) + 1e-30f);
        if (rel >= 1e-5f) {
            fprintf(stderr, "golden mismatch at dim %d: got %.8e, want %.8e (rel %.2e)\n",
                    i, out[i], golden[i], (double)rel);
        }
        assert(rel < 1e-5f);
    }

    model_config_free(&cfg);
}

int main(void) {
    test_arg_validation();
    printf("  test_arg_validation: passed\n");

    test_reject_factor_zero();
    printf("  test_reject_factor_zero: passed\n");

    test_reject_factor_negative();
    printf("  test_reject_factor_negative: passed\n");

    test_reject_original_max_zero();
    printf("  test_reject_original_max_zero: passed\n");

    test_reject_low_freq_factor_zero();
    printf("  test_reject_low_freq_factor_zero: passed\n");

    test_reject_low_freq_factor_negative();
    printf("  test_reject_low_freq_factor_negative: passed\n");

    test_reject_high_leq_low();
    printf("  test_reject_high_leq_low: passed\n");

    test_reject_odd_head_dim();
    printf("  test_reject_odd_head_dim: passed\n");

    test_reject_head_dim_lt_2();
    printf("  test_reject_head_dim_lt_2: passed\n");

    test_high_freq_passthrough();
    printf("  test_high_freq_passthrough: passed\n");

    test_low_freq_scaled();
    printf("  test_low_freq_scaled: passed\n");

    test_medium_band();
    printf("  test_medium_band: passed\n");

    test_golden_vector();
    printf("  test_golden_vector: passed\n");

    printf("test_rope_llama3: all passed\n");
    return 0;
}
