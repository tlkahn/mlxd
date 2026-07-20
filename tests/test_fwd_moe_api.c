#include "engine/forward.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

/* Cycle 1: CPU-only API / struct smoke. No mlx graph execution. */

static void test_structs_zero_init(void) {
    fwd_moe_weights_t mw;
    fwd_moe_params_t p;
    memset(&mw, 0, sizeof(mw));
    memset(&p, 0, sizeof(p));
    assert(sizeof(fwd_moe_weights_t) > 0);
    assert(sizeof(fwd_moe_params_t) > 0);
    assert(!mw.has_shared);
    assert(!mw.has_shared_expert_gate);
    assert(p.num_experts == 0);
    assert(p.top_k == 0);
    assert(!p.norm_topk_prob);
    assert(mw.router.weight.ctx == NULL);
    assert(mw.switch_gate.weight.ctx == NULL);
}

static void test_deepseek_params_zero_init(void) {
    fwd_moe_route_deepseek_params_t p;
    memset(&p, 0, sizeof(p));
    assert(sizeof(fwd_moe_route_deepseek_params_t) > 0);
    assert(p.top_k == 0);
    assert(p.n_group == 0);
    assert(p.topk_group == 0);
    assert(p.routed_scaling_factor == 0.f);
    assert(!p.norm_topk_prob);
}

static void test_null_args_return_error(void) {
    mlx_array dummy = mlx_array_new();
    mlx_stream s = {.ctx = NULL};

    assert(fwd_array_contiguous(NULL, dummy, s) == -1);
    assert(fwd_moe_route_softmax(NULL, &dummy, dummy, 1, false, s) == -1);
    assert(fwd_moe_route_softmax(&dummy, NULL, dummy, 1, false, s) == -1);
    assert(fwd_moe_route_softmax(&dummy, &dummy, (mlx_array){.ctx = NULL},
                                 1, false, s) == -1);
    assert(fwd_moe_route_softmax(&dummy, &dummy, dummy, 0, false, s) == -1);

    assert(fwd_gather_expert_linear(NULL, dummy, dummy, NULL,
                                    (mlx_array){.ctx = NULL}, false,
                                    NULL, s) == -1);
    assert(fwd_switch_glu(NULL, dummy, dummy, NULL, NULL, NULL, NULL, s) == -1);
    assert(fwd_moe_combine(NULL, dummy, dummy, dummy, NULL, NULL, s) == -1);
    assert(fwd_moe(NULL, dummy, NULL, NULL, NULL, s) == -1);

    mlx_array_free(dummy);
}

static void test_route_deepseek_null_args(void) {
    mlx_array dummy = mlx_array_new();
    mlx_stream s = {.ctx = NULL};
    fwd_moe_route_deepseek_params_t p = {
        .top_k = 1,
        .n_group = 1,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };

    assert(fwd_moe_route_deepseek(NULL, &dummy, dummy, dummy, &p, s) == -1);
    assert(fwd_moe_route_deepseek(&dummy, NULL, dummy, dummy, &p, s) == -1);
    assert(fwd_moe_route_deepseek(&dummy, &dummy, (mlx_array){.ctx = NULL},
                                  dummy, &p, s) == -1);
    assert(fwd_moe_route_deepseek(&dummy, &dummy, dummy,
                                  (mlx_array){.ctx = NULL}, &p, s) == -1);
    assert(fwd_moe_route_deepseek(&dummy, &dummy, dummy, dummy, NULL, s) == -1);

    p.top_k = 0;
    assert(fwd_moe_route_deepseek(&dummy, &dummy, dummy, dummy, &p, s) == -1);

    mlx_array_free(dummy);
}

/* Host-only f32 array builder for prologue validation (no stream ops). */
static mlx_array host_f32(const float *data, const int *shape, int ndim) {
    return mlx_array_new_data(data, shape, ndim, MLX_FLOAT32);
}

/* Characterization of pure-metadata rejects. Dummy stream is fine: every
   case must return -1 before any stream op. */
static void test_route_deepseek_cpu_validation_matrix(void) {
    mlx_stream s = {.ctx = NULL};
    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();

    /* top_k > E */
    {
        float logits_h[] = {0.f, 1.f, 2.f, 3.f};
        float bias_h[] = {0.f, 0.f, 0.f, 0.f};
        mlx_array logits = host_f32(logits_h, (int[]){1, 1, 4}, 3);
        mlx_array bias = host_f32(bias_h, (int[]){4}, 1);
        fwd_moe_route_deepseek_params_t p = {
            .top_k = 5,
            .n_group = 1,
            .topk_group = 1,
            .routed_scaling_factor = 1.f,
            .norm_topk_prob = false,
        };
        assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, s) ==
               -1);
        mlx_array_free(bias);
        mlx_array_free(logits);
    }

    /* E % n_group != 0 */
    {
        float logits_h[6];
        float bias_h[6];
        for (int i = 0; i < 6; i++) {
            logits_h[i] = (float)i;
            bias_h[i] = 0.f;
        }
        mlx_array logits = host_f32(logits_h, (int[]){1, 1, 6}, 3);
        mlx_array bias = host_f32(bias_h, (int[]){6}, 1);
        fwd_moe_route_deepseek_params_t p = {
            .top_k = 2,
            .n_group = 4,
            .topk_group = 1,
            .routed_scaling_factor = 1.f,
            .norm_topk_prob = false,
        };
        assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, s) ==
               -1);
        mlx_array_free(bias);
        mlx_array_free(logits);
    }

    /* topk_group == 0 */
    {
        float logits_h[] = {0.f, 1.f, 2.f, 3.f};
        float bias_h[] = {0.f, 0.f, 0.f, 0.f};
        mlx_array logits = host_f32(logits_h, (int[]){1, 1, 4}, 3);
        mlx_array bias = host_f32(bias_h, (int[]){4}, 1);
        fwd_moe_route_deepseek_params_t p = {
            .top_k = 2,
            .n_group = 2,
            .topk_group = 0,
            .routed_scaling_factor = 1.f,
            .norm_topk_prob = false,
        };
        assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, s) ==
               -1);
        mlx_array_free(bias);
        mlx_array_free(logits);
    }

    /* topk_group > n_group */
    {
        float logits_h[] = {0.f, 1.f, 2.f, 3.f};
        float bias_h[] = {0.f, 0.f, 0.f, 0.f};
        mlx_array logits = host_f32(logits_h, (int[]){1, 1, 4}, 3);
        mlx_array bias = host_f32(bias_h, (int[]){4}, 1);
        fwd_moe_route_deepseek_params_t p = {
            .top_k = 2,
            .n_group = 2,
            .topk_group = 3,
            .routed_scaling_factor = 1.f,
            .norm_topk_prob = false,
        };
        assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, s) ==
               -1);
        mlx_array_free(bias);
        mlx_array_free(logits);
    }

    /* n_group > 1 && Eg < 2 (E=2, n_group=2 => Eg=1) */
    {
        float logits_h[] = {1.f, 2.f};
        float bias_h[] = {0.f, 0.f};
        mlx_array logits = host_f32(logits_h, (int[]){1, 1, 2}, 3);
        mlx_array bias = host_f32(bias_h, (int[]){2}, 1);
        fwd_moe_route_deepseek_params_t p = {
            .top_k = 1,
            .n_group = 2,
            .topk_group = 1,
            .routed_scaling_factor = 1.f,
            .norm_topk_prob = false,
        };
        assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, s) ==
               -1);
        mlx_array_free(bias);
        mlx_array_free(logits);
    }

    /* bias last dim != E */
    {
        float logits_h[] = {0.f, 1.f, 2.f, 3.f};
        float bias_h[] = {0.f, 0.f, 0.f};
        mlx_array logits = host_f32(logits_h, (int[]){1, 1, 4}, 3);
        mlx_array bias = host_f32(bias_h, (int[]){3}, 1);
        fwd_moe_route_deepseek_params_t p = {
            .top_k = 2,
            .n_group = 1,
            .topk_group = 1,
            .routed_scaling_factor = 1.f,
            .norm_topk_prob = false,
        };
        assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, s) ==
               -1);
        mlx_array_free(bias);
        mlx_array_free(logits);
    }

    /* bias rank != 1 (broadcastable [1, E] still rejected) */
    {
        float logits_h[] = {0.f, 1.f, 2.f, 3.f};
        float bias_h[] = {0.f, 0.f, 0.f, 0.f};
        mlx_array logits = host_f32(logits_h, (int[]){1, 1, 4}, 3);
        mlx_array bias = host_f32(bias_h, (int[]){1, 4}, 2);
        fwd_moe_route_deepseek_params_t p = {
            .top_k = 2,
            .n_group = 1,
            .topk_group = 1,
            .routed_scaling_factor = 1.f,
            .norm_topk_prob = false,
        };
        assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, s) ==
               -1);
        mlx_array_free(bias);
        mlx_array_free(logits);
    }

    mlx_array_free(scores);
    mlx_array_free(inds);
}

int main(void) {
    test_structs_zero_init();
    printf("  test_structs_zero_init: passed\n");
    test_deepseek_params_zero_init();
    printf("  test_deepseek_params_zero_init: passed\n");
    test_null_args_return_error();
    printf("  test_null_args_return_error: passed\n");
    test_route_deepseek_null_args();
    printf("  test_route_deepseek_null_args: passed\n");
    test_route_deepseek_cpu_validation_matrix();
    printf("  test_route_deepseek_cpu_validation_matrix: passed\n");
    printf("all test_fwd_moe_api tests passed\n");
    return 0;
}
