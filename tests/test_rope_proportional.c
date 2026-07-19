#include "engine/forward.h"
#include "model/model.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef MLXD_FIXTURES_DIR
#error "MLXD_FIXTURES_DIR must be defined"
#endif

static model_config_t make_proportional_cfg(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.family = MODEL_GEMMA4;
    cfg.rope_proportional = true;
    cfg.rope_proportional_factor = 8.0f;
    cfg.rope_theta = 1000000.0f;
    cfg.global_head_dim = 32;
    cfg.partial_rotary_factor_global = 0.5f;
    cfg.head_dim = 16;
    return cfg;
}

/* --- arg validation ------------------------------------------------------- */

static void test_arg_validation(void) {
    model_config_t cfg = make_proportional_cfg();
    float out[16];

    assert(fwd_rope_proportional_freqs(NULL, out, 16) == -1);
    assert(fwd_rope_proportional_freqs(&cfg, NULL, 16) == -1);

    /* n must equal global_head_dim / 2 */
    assert(fwd_rope_proportional_freqs(&cfg, out, 8) == -1);

    /* rope_proportional must be true */
    model_config_t cfg2 = cfg;
    cfg2.rope_proportional = false;
    assert(fwd_rope_proportional_freqs(&cfg2, out, 16) == -1);

    /* factor must be > 0 */
    model_config_t cfg3 = cfg;
    cfg3.rope_proportional_factor = 0.0f;
    assert(fwd_rope_proportional_freqs(&cfg3, out, 16) == -1);

    cfg3.rope_proportional_factor = -1.0f;
    assert(fwd_rope_proportional_freqs(&cfg3, out, 16) == -1);

    /* global_head_dim must be even and >= 2 */
    model_config_t cfg4 = cfg;
    cfg4.global_head_dim = 3;
    assert(fwd_rope_proportional_freqs(&cfg4, out, 16) == -1);

    model_config_t cfg5 = cfg;
    cfg5.global_head_dim = 1;
    assert(fwd_rope_proportional_freqs(&cfg5, out, 16) == -1);

    /* partial_rotary_factor_global out of (0,1] */
    model_config_t cfg6 = cfg;
    cfg6.partial_rotary_factor_global = 0.0f;
    assert(fwd_rope_proportional_freqs(&cfg6, out, 16) == -1);

    cfg6.partial_rotary_factor_global = 1.5f;
    assert(fwd_rope_proportional_freqs(&cfg6, out, 16) == -1);
}

/* --- correctness: freqs[i] = factor * base^(2i/ghd) for i < rotated,
       then INFINITY for the rest ------------------------------------------ */

static void test_proportional_values(void) {
    model_config_t cfg = make_proportional_cfg();
    /* ghd=32, partial_rotary=0.5 => rotated_dims=16, n=ghd/2=16
       rotated_dims/2 = 8 freq entries, rest = INFINITY */
    float out[16];
    assert(fwd_rope_proportional_freqs(&cfg, out, 16) == 0);

    double base = 1e6;
    double factor = 8.0;
    int ghd = 32;
    int rotated = 8; /* rotated_dims / 2 = (ghd * partial) / 2 = 16/2 = 8 */

    for (int i = 0; i < rotated; i++) {
        double expected = factor * pow(base, 2.0 * i / (double)ghd);
        float diff = fabsf(out[i] - (float)expected);
        float rel = diff / (fabsf((float)expected) + 1e-30f);
        if (rel >= 1e-5f) {
            fprintf(stderr, "mismatch at %d: got %.8e, want %.8e (rel %.2e)\n",
                    i, out[i], (float)expected, (double)rel);
        }
        assert(rel < 1e-5f);
    }

    for (int i = rotated; i < 16; i++) {
        assert(isinf(out[i]) && out[i] > 0);
    }
}

/* --- full partial_rotary_factor=1.0: no infinity padding ----------------- */

static void test_full_rotary(void) {
    model_config_t cfg = make_proportional_cfg();
    cfg.partial_rotary_factor_global = 1.0f;
    /* rotated_dims = ghd * 1.0 = 32, rotated_dims/2 = 16 = n, no padding */
    float out[16];
    assert(fwd_rope_proportional_freqs(&cfg, out, 16) == 0);

    for (int i = 0; i < 16; i++) {
        double expected = 8.0 * pow(1e6, 2.0 * i / 32.0);
        float diff = fabsf(out[i] - (float)expected);
        float rel = diff / (fabsf((float)expected) + 1e-30f);
        assert(rel < 1e-5f);
    }
}

/* --- fwd_rope_freqs_build dispatches on rope_proportional ---------------- */

static void test_freqs_build_proportional(void) {
    model_config_t cfg = make_proportional_cfg();
    mlx_array freqs = mlx_array_new();
    assert(fwd_rope_freqs_build(&freqs, &cfg) == 0);
    assert(freqs.ctx != NULL);
    assert((int)mlx_array_ndim(freqs) == 1);
    assert(mlx_array_dim(freqs, 0) == 16); /* ghd/2 */
    assert(mlx_array_dtype(freqs) == MLX_FLOAT32);
    mlx_array_free(freqs);
}

static void test_freqs_build_plain(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.family = MODEL_QWEN3;
    cfg.head_dim = 128;
    cfg.rope_theta = 5000000.0f;
    mlx_array freqs = mlx_array_new();
    assert(fwd_rope_freqs_build(&freqs, &cfg) == 0);
    assert(freqs.ctx == NULL);
    mlx_array_free(freqs);
}

int main(void) {
    test_arg_validation();
    printf("  test_arg_validation: passed\n");

    test_proportional_values();
    printf("  test_proportional_values: passed\n");

    test_full_rotary();
    printf("  test_full_rotary: passed\n");

    test_freqs_build_proportional();
    printf("  test_freqs_build_proportional: passed\n");

    test_freqs_build_plain();
    printf("  test_freqs_build_plain: passed\n");

    printf("test_rope_proportional: all passed\n");
    return 0;
}
