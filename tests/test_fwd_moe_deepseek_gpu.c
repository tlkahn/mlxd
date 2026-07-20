#include "engine/forward.h"
#include "gpu_test_util.h"
#include "mlxbridge/mlxbridge.h"
#include "model/model.h"
#include "model/weights.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GPU suite for #108 DeepSeek-style MoE gate (fwd_moe_route_deepseek).
   Goldens cross-checked against mlx-lm 0.31.x group_expert_select. */

static mlx_stream gpu;
static model_config_t cfg;

/* ---- helpers ---- */

static void cfg_init_basic(int H) {
    memset(&cfg, 0, sizeof(cfg));
    cfg.hidden_act = HIDDEN_ACT_SILU;
    cfg.hidden_size = H;
}

static mlx_array make_f32(const float *data, const int *shape, int ndim) {
    return mlx_array_new_data(data, shape, ndim, MLX_FLOAT32);
}

static mlx_array make_bf16(const float *data, const int *shape, int ndim) {
    mlx_array f32 = mlx_array_new_data(data, shape, ndim, MLX_FLOAT32);
    mlx_array out = mlx_array_new();
    assert(MLXB_CHECK(mlx_astype(&out, f32, MLX_BFLOAT16, gpu)));
    mlx_array_free(f32);
    return out;
}

static void read_f32(mlx_array a, float *out, size_t n) {
    mlx_array f32 = mlx_array_new();
    assert(MLXB_CHECK(mlx_astype(&f32, a, MLX_FLOAT32, gpu)));
    assert(MLXB_CHECK(mlx_array_eval(f32)));
    assert(mlx_array_size(f32) == n);
    const float *d = mlx_array_data_float32(f32);
    assert(d);
    memcpy(out, d, n * sizeof(float));
    mlx_array_free(f32);
}

static void read_i32(mlx_array a, int32_t *out, size_t n) {
    mlx_array i32 = mlx_array_new();
    assert(MLXB_CHECK(mlx_astype(&i32, a, MLX_INT32, gpu)));
    assert(MLXB_CHECK(mlx_array_eval(i32)));
    assert(mlx_array_size(i32) == n);
    const int32_t *d = mlx_array_data_int32(i32);
    assert(d);
    memcpy(out, d, n * sizeof(int32_t));
    mlx_array_free(i32);
}

static float max_abs_diff(const float *a, const float *b, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static int has_nonzero(const float *a, size_t n, float thr) {
    for (size_t i = 0; i < n; i++)
        if (fabsf(a[i]) > thr) return 1;
    return 0;
}

static float host_sigmoid(float x) {
    if (x >= 0.f) {
        float z = expf(-x);
        return 1.f / (1.f + z);
    }
    float z = expf(x);
    return z / (1.f + z);
}

/* Sort (ind, score) pairs by expert id for partition-order-stable compare. */
static void sort_pairs_by_ind(int32_t *inds, float *scores, int k) {
    for (int i = 0; i < k - 1; i++) {
        for (int j = i + 1; j < k; j++) {
            if (inds[j] < inds[i]) {
                int32_t ti = inds[i];
                inds[i] = inds[j];
                inds[j] = ti;
                float ts = scores[i];
                scores[i] = scores[j];
                scores[j] = ts;
            }
        }
    }
}

static int set_eq_inds(const int32_t *got, const int *exp, int k) {
    int used[64];
    assert(k <= 64);
    memset(used, 0, sizeof(used));
    for (int i = 0; i < k; i++) {
        int found = 0;
        for (int j = 0; j < k; j++) {
            if (!used[j] && got[i] == exp[j]) {
                used[j] = 1;
                found = 1;
                break;
            }
        }
        if (!found) return 0;
    }
    return 1;
}

/* Host mirror of mlx-lm group_expert_select for small E (single row). */
static int host_group_expert_select(const float *gates, int E,
                                    const float *bias, int top_k, int n_group,
                                    int topk_group, float routed_scaling_factor,
                                    int norm_topk_prob, int *inds_out,
                                    float *scores_out) {
    if (E < 1 || top_k < 1 || top_k > E || n_group < 1 || E % n_group != 0)
        return -1;
    if (topk_group < 1 || topk_group > n_group) return -1;
    int eg = E / n_group;
    if (n_group > 1 && eg < 2) return -1;

    float *orig = (float *)malloc((size_t)E * sizeof(float));
    float *sel = (float *)malloc((size_t)E * sizeof(float));
    assert(orig && sel);
    for (int i = 0; i < E; i++) {
        orig[i] = host_sigmoid(gates[i]);
        sel[i] = orig[i] + bias[i];
    }

    if (n_group > 1) {
        int k_drop = n_group - topk_group;
        if (k_drop > 0) {
            float gscore[64];
            assert(n_group <= 64);
            for (int g = 0; g < n_group; g++) {
                float a = -INFINITY, b = -INFINITY;
                for (int j = 0; j < eg; j++) {
                    float v = sel[g * eg + j];
                    if (v > a) {
                        b = a;
                        a = v;
                    } else if (v > b) {
                        b = v;
                    }
                }
                gscore[g] = a + b;
            }
            /* Mark k_drop worst groups (stable via indices). */
            int order[64];
            for (int g = 0; g < n_group; g++) order[g] = g;
            for (int i = 0; i < n_group - 1; i++) {
                for (int j = i + 1; j < n_group; j++) {
                    if (gscore[order[j]] < gscore[order[i]]) {
                        int t = order[i];
                        order[i] = order[j];
                        order[j] = t;
                    }
                }
            }
            for (int i = 0; i < k_drop; i++) {
                int g = order[i];
                for (int j = 0; j < eg; j++) sel[g * eg + j] = 0.f;
            }
        }
    }

    /* Top-k on sel (argpartition-style: pick k largest; order undefined). */
    int *order = (int *)malloc((size_t)E * sizeof(int));
    assert(order);
    for (int i = 0; i < E; i++) order[i] = i;
    for (int i = 0; i < E - 1; i++) {
        for (int j = i + 1; j < E; j++) {
            if (sel[order[j]] > sel[order[i]]) {
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
        }
    }
    for (int i = 0; i < top_k; i++) {
        inds_out[i] = order[i];
        scores_out[i] = orig[order[i]];
    }
    if (top_k > 1 && norm_topk_prob) {
        float sum = 0.f;
        for (int i = 0; i < top_k; i++) sum += scores_out[i];
        if (sum != 0.f) {
            for (int i = 0; i < top_k; i++) scores_out[i] /= sum;
        }
    }
    for (int i = 0; i < top_k; i++)
        scores_out[i] *= routed_scaling_factor;

    free(order);
    free(sel);
    free(orig);
    return 0;
}

static mlx_array make_zero_bias(int E) {
    float *z = (float *)calloc((size_t)E, sizeof(float));
    assert(z);
    mlx_array b = make_f32(z, (int[]){E}, 1);
    free(z);
    return b;
}

static weight_triplet_t make_bf16_triplet(mlx_array w) {
    weight_triplet_t tri;
    memset(&tri, 0, sizeof(tri));
    tri.weight = w;
    tri.quantized = false;
    return tri;
}

static weight_triplet_t clone_bf16_weight(const float *data, const int *shape,
                                          int ndim) {
    return make_bf16_triplet(make_bf16(data, shape, ndim));
}

static void fill_linear(float *w, int out, int in, float scale, int seed) {
    for (int o = 0; o < out; o++)
        for (int i = 0; i < in; i++)
            w[o * in + i] =
                scale * (float)(((o * 17 + i * 3 + seed) % 13) - 6);
}

static void fill_expert_eoi(float *w, int E, int out, int in, int seed) {
    for (int e = 0; e < E; e++)
        for (int o = 0; o < out; o++)
            for (int i = 0; i < in; i++)
                w[((e * out) + o) * in + i] =
                    0.05f * (float)(((e * 11 + o * 5 + i + seed) % 13) - 6);
}

static void transpose_eoi_to_eio(const float *eoi, float *eio, int E, int out,
                                 int in) {
    for (int e = 0; e < E; e++)
        for (int o = 0; o < out; o++)
            for (int i = 0; i < in; i++)
                eio[((e * in) + i) * out + o] = eoi[((e * out) + o) * in + i];
}

static weight_triplet_t make_dense_linear(int out, int in, int seed) {
    float *w = (float *)malloc((size_t)out * in * sizeof(float));
    assert(w);
    fill_linear(w, out, in, 0.04f, seed);
    weight_triplet_t tri = clone_bf16_weight(w, (int[]){out, in}, 2);
    free(w);
    return tri;
}

/* bf16 expert stacks shaped [E, in, out] for gather_mm. */
static void build_bf16_switch(weight_triplet_t *gate, weight_triplet_t *up,
                              weight_triplet_t *down, int E, int H, int MI) {
    size_t n_gu = (size_t)E * MI * H;
    size_t n_d = (size_t)E * H * MI;
    float *eoi = (float *)malloc(n_gu * sizeof(float));
    float *eio = (float *)malloc(n_gu * sizeof(float));
    assert(eoi && eio);

    fill_expert_eoi(eoi, E, MI, H, 10);
    transpose_eoi_to_eio(eoi, eio, E, MI, H);
    *gate = clone_bf16_weight(eio, (int[]){E, H, MI}, 3);

    fill_expert_eoi(eoi, E, MI, H, 20);
    transpose_eoi_to_eio(eoi, eio, E, MI, H);
    *up = clone_bf16_weight(eio, (int[]){E, H, MI}, 3);

    free(eoi);
    free(eio);
    eoi = (float *)malloc(n_d * sizeof(float));
    eio = (float *)malloc(n_d * sizeof(float));
    assert(eoi && eio);
    fill_expert_eoi(eoi, E, H, MI, 30);
    transpose_eoi_to_eio(eoi, eio, E, H, MI);
    *down = clone_bf16_weight(eio, (int[]){E, MI, H}, 3);
    free(eoi);
    free(eio);
}

/* ---- Cycle 2: sigmoid + top-k, n_group=1 ---- */

static void test_route_deepseek_sigmoid_topk_n_group_1(void) {
    float logits_h[] = {0.f, 10.f, 1.f, 9.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 4}, 3);
    mlx_array bias = make_zero_bias(4);

    fwd_moe_route_deepseek_params_t p = {
        .top_k = 2,
        .n_group = 1,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(inds)));
    assert(MLXB_CHECK(mlx_array_eval(scores)));
    assert(mlx_array_ndim(inds) == 3);
    assert(mlx_array_dim(inds, 2) == 2);

    int32_t ih[2];
    float sh[2];
    read_i32(inds, ih, 2);
    read_f32(scores, sh, 2);

    int exp_inds[] = {1, 3};
    assert(set_eq_inds(ih, exp_inds, 2));
    assert(fabsf(sh[0] - host_sigmoid(logits_h[ih[0]])) < 1e-5f);
    assert(fabsf(sh[1] - host_sigmoid(logits_h[ih[1]])) < 1e-5f);

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

static void test_route_deepseek_scores_ignore_bias_for_values(void) {
    /* Bias pushes selection to experts 0 and 2; scores stay unbiased sigmoid. */
    float logits_h[] = {0.f, 10.f, 1.f, 9.f};
    float bias_h[] = {100.f, 0.f, 100.f, 0.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 4}, 3);
    mlx_array bias = make_f32(bias_h, (int[]){4}, 1);

    fwd_moe_route_deepseek_params_t p = {
        .top_k = 2,
        .n_group = 1,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == 0);

    int32_t ih[2];
    float sh[2];
    read_i32(inds, ih, 2);
    read_f32(scores, sh, 2);

    int exp_inds[] = {0, 2};
    assert(set_eq_inds(ih, exp_inds, 2));
    /* S1: scores = sigmoid(logits[inds]), NOT biased. */
    assert(fabsf(sh[0] - host_sigmoid(logits_h[ih[0]])) < 1e-5f);
    assert(fabsf(sh[1] - host_sigmoid(logits_h[ih[1]])) < 1e-5f);
    /* Explicitly not the biased values. */
    assert(fabsf(sh[0] - (host_sigmoid(logits_h[ih[0]]) + bias_h[ih[0]])) >
           1.f);

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

/* ---- Cycle 3: scale + norm ---- */

static void test_route_deepseek_routed_scaling_factor(void) {
    float logits_h[] = {0.f, 10.f, 1.f, 9.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 4}, 3);
    mlx_array bias = make_zero_bias(4);

    fwd_moe_route_deepseek_params_t p = {
        .top_k = 2,
        .n_group = 1,
        .topk_group = 1,
        .routed_scaling_factor = 2.5f,
        .norm_topk_prob = false,
    };

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == 0);

    int32_t ih[2];
    float sh[2];
    read_i32(inds, ih, 2);
    read_f32(scores, sh, 2);
    assert(fabsf(sh[0] - 2.5f * host_sigmoid(logits_h[ih[0]])) < 1e-5f);
    assert(fabsf(sh[1] - 2.5f * host_sigmoid(logits_h[ih[1]])) < 1e-5f);

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

static void test_route_deepseek_norm_topk_prob_top_k_gt_1(void) {
    float logits_h[] = {0.f, 2.f, -1.f, 1.5f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 4}, 3);
    mlx_array bias = make_zero_bias(4);

    fwd_moe_route_deepseek_params_t p = {
        .top_k = 2,
        .n_group = 1,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = true,
    };

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == 0);

    int32_t ih[2];
    float sh[2];
    read_i32(inds, ih, 2);
    read_f32(scores, sh, 2);
    float sum = sh[0] + sh[1];
    assert(fabsf(sum - 1.f) < 1e-5f);

    float s0 = host_sigmoid(logits_h[ih[0]]);
    float s1 = host_sigmoid(logits_h[ih[1]]);
    float n0 = s0 / (s0 + s1);
    float n1 = s1 / (s0 + s1);
    assert(fabsf(sh[0] - n0) < 1e-5f);
    assert(fabsf(sh[1] - n1) < 1e-5f);

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

static void test_route_deepseek_norm_topk_prob_ignored_when_top_k_1(void) {
    /* S2: top_k==1 => no renormalize even if flag set; score = scale*sigmoid. */
    float logits_h[] = {0.f, 10.f, 1.f, 9.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 4}, 3);
    mlx_array bias = make_zero_bias(4);

    fwd_moe_route_deepseek_params_t p = {
        .top_k = 1,
        .n_group = 1,
        .topk_group = 1,
        .routed_scaling_factor = 2.5f,
        .norm_topk_prob = true,
    };

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == 0);

    int32_t ih[1];
    float sh[1];
    read_i32(inds, ih, 1);
    read_f32(scores, sh, 1);
    assert(ih[0] == 1);
    float expect = 2.5f * host_sigmoid(10.f);
    assert(fabsf(sh[0] - expect) < 1e-4f);
    /* Not left at plain sigmoid or plain 1.0 (scale must apply; S2/S3). */
    assert(fabsf(sh[0] - 1.f) > 0.5f);
    assert(fabsf(sh[0] - host_sigmoid(10.f)) > 0.5f);

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

/* ---- Cycle 4: group restriction ---- */

static void test_route_deepseek_group_keeps_topk_group_only(void) {
    /* E=8, G=2, topk_group=1.
       Group0 has one huge expert; group1 has two moderate experts whose
       top-2 sum wins (mlx-lm golden E). Expect inds in {4,5}. */
    float logits_h[] = {5.f, -10.f, -10.f, -10.f, 2.f, 2.f, -10.f, -10.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 8}, 3);
    mlx_array bias = make_zero_bias(8);

    fwd_moe_route_deepseek_params_t p = {
        .top_k = 2,
        .n_group = 2,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == 0);

    int32_t ih[2];
    read_i32(inds, ih, 2);
    int exp[] = {4, 5};
    assert(set_eq_inds(ih, exp, 2));

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

static void test_route_deepseek_group_score_uses_top2_sum(void) {
    /* S5: max expert in group0, but top-2 sum larger in group1.
       Same geometry as above (mlx-lm golden E). */
    float logits_h[] = {5.f, -10.f, -10.f, -10.f, 2.f, 2.f, -10.f, -10.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 8}, 3);
    mlx_array bias = make_zero_bias(8);

    fwd_moe_route_deepseek_params_t p = {
        .top_k = 2,
        .n_group = 2,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == 0);

    int32_t ih[2];
    float sh[2];
    read_i32(inds, ih, 2);
    read_f32(scores, sh, 2);
    /* Not from group0 (would include expert 0 if max-pooling). */
    assert(ih[0] != 0 && ih[1] != 0);
    int exp[] = {4, 5};
    assert(set_eq_inds(ih, exp, 2));
    assert(fabsf(sh[0] - host_sigmoid(2.f)) < 1e-5f);
    assert(fabsf(sh[1] - host_sigmoid(2.f)) < 1e-5f);

    /* Opposite: group0 wins. */
    float logits_h2[] = {2.f, 2.f, -10.f, -10.f, 5.f, -10.f, -10.f, -10.f};
    mlx_array logits2 = make_f32(logits_h2, (int[]){1, 1, 8}, 3);
    mlx_array inds2 = mlx_array_new();
    mlx_array scores2 = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds2, &scores2, logits2, bias, &p, gpu) ==
           0);
    int32_t ih2[2];
    read_i32(inds2, ih2, 2);
    int exp2[] = {0, 1};
    assert(set_eq_inds(ih2, exp2, 2));

    mlx_array_free(scores2);
    mlx_array_free(inds2);
    mlx_array_free(logits2);
    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

static void test_route_deepseek_k_drop_zero_skips_mask(void) {
    /* D7: topk_group == n_group => no group drop; both groups eligible. */
    float logits_h[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 8}, 3);
    mlx_array bias = make_zero_bias(8);

    fwd_moe_route_deepseek_params_t p = {
        .top_k = 2,
        .n_group = 2,
        .topk_group = 2,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == 0);

    int32_t ih[2];
    read_i32(inds, ih, 2);
    int exp[] = {6, 7};
    assert(set_eq_inds(ih, exp, 2));

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

static void test_route_deepseek_bias_breaks_group_ties(void) {
    /* Equal group logits; bias tips group1. */
    float logits_h[] = {1.f, 1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f};
    float bias_h[] = {0.f, 0.f, 0.f, 0.f, 5.f, 5.f, 0.f, 0.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 8}, 3);
    mlx_array bias = make_f32(bias_h, (int[]){8}, 1);

    fwd_moe_route_deepseek_params_t p = {
        .top_k = 2,
        .n_group = 2,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == 0);

    int32_t ih[2];
    float sh[2];
    read_i32(inds, ih, 2);
    read_f32(scores, sh, 2);
    int exp[] = {4, 5};
    assert(set_eq_inds(ih, exp, 2));
    /* Scores still unbiased. */
    assert(fabsf(sh[0] - host_sigmoid(1.f)) < 1e-5f);
    assert(fabsf(sh[1] - host_sigmoid(1.f)) < 1e-5f);

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

/* ---- Cycle 5: batch/seq + bf16 ---- */

/* ndim==3 group path: shape + per-row value match vs host_group_expert_select.
   Catches wrong group axis (-2) bugs that still emit [B,S,K] shapes. */
static void test_route_deepseek_batch_seq_matches_host_rows(void) {
    const int B = 2, S = 3, E = 8, K = 2;
    float logits_h[2 * 3 * 8];
    for (int i = 0; i < B * S * E; i++)
        logits_h[i] = 0.1f * (float)((i * 7) % 11) - 0.4f;
    mlx_array logits = make_f32(logits_h, (int[]){B, S, E}, 3);
    mlx_array bias = make_zero_bias(E);

    fwd_moe_route_deepseek_params_t p = {
        .top_k = K,
        .n_group = 2,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(inds)));
    assert(MLXB_CHECK(mlx_array_eval(scores)));
    assert(mlx_array_ndim(inds) == 3);
    assert(mlx_array_dim(inds, 0) == B);
    assert(mlx_array_dim(inds, 1) == S);
    assert(mlx_array_dim(inds, 2) == K);
    assert(mlx_array_dim(scores, 0) == B);
    assert(mlx_array_dim(scores, 1) == S);
    assert(mlx_array_dim(scores, 2) == K);
    assert(is_finite_f32(scores, gpu));

    int32_t ih[2 * 3 * 2];
    float sh[2 * 3 * 2];
    read_i32(inds, ih, (size_t)(B * S * K));
    read_f32(scores, sh, (size_t)(B * S * K));

    float zb[8] = {0};
    for (int b = 0; b < B; b++) {
        for (int s = 0; s < S; s++) {
            int row = b * S + s;
            const float *row_logits = logits_h + row * E;
            int exp_i[2];
            float exp_s[2];
            assert(host_group_expert_select(row_logits, E, zb, K, 2, 1, 1.f, 0,
                                            exp_i, exp_s) == 0);

            int32_t got_i[2] = {ih[row * K + 0], ih[row * K + 1]};
            float got_s[2] = {sh[row * K + 0], sh[row * K + 1]};
            assert(set_eq_inds(got_i, exp_i, K));

            /* Align scores by expert id for order-stable compare. */
            for (int k = 0; k < K; k++) {
                float expect = -1.f;
                for (int j = 0; j < K; j++) {
                    if (exp_i[j] == got_i[k]) {
                        expect = exp_s[j];
                        break;
                    }
                }
                assert(fabsf(got_s[k] - expect) < 1e-5f);
            }
        }
    }

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

static void test_route_deepseek_bf16_logits_match_f32_host(void) {
    float logits_h[] = {0.f, 2.f, -1.f, 1.5f, 0.5f, -0.5f, 3.f, 1.f};
    mlx_array logits = make_bf16(logits_h, (int[]){1, 1, 8}, 3);
    mlx_array bias = make_zero_bias(8);

    fwd_moe_route_deepseek_params_t p = {
        .top_k = 2,
        .n_group = 1,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == 0);

    int32_t ih[2];
    float sh[2];
    read_i32(inds, ih, 2);
    read_f32(scores, sh, 2);

    float zb[8] = {0};
    int exp_i[2];
    float exp_s[2];
    assert(host_group_expert_select(logits_h, 8, zb, 2, 1, 1, 1.f, 0, exp_i,
                                    exp_s) == 0);
    assert(set_eq_inds(ih, exp_i, 2));
    /* Align scores by ind. */
    for (int i = 0; i < 2; i++) {
        float expect = host_sigmoid(logits_h[ih[i]]);
        assert(fabsf(sh[i] - expect) < 2e-2f);
    }

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

/* ---- Cycle 6: validation matrix ---- */

static void test_route_deepseek_rejects_top_k_gt_E(void) {
    float logits_h[] = {0.f, 1.f, 2.f, 3.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 4}, 3);
    mlx_array bias = make_zero_bias(4);
    fwd_moe_route_deepseek_params_t p = {
        .top_k = 5,
        .n_group = 1,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };
    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == -1);
    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

static void test_route_deepseek_rejects_bad_n_group_divisibility(void) {
    float logits_h[6];
    for (int i = 0; i < 6; i++) logits_h[i] = (float)i;
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 6}, 3);
    mlx_array bias = make_zero_bias(6);
    fwd_moe_route_deepseek_params_t p = {
        .top_k = 2,
        .n_group = 4, /* 6 % 4 != 0 */
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };
    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == -1);
    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

static void test_route_deepseek_rejects_bad_topk_group(void) {
    float logits_h[] = {0.f, 1.f, 2.f, 3.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 4}, 3);
    mlx_array bias = make_zero_bias(4);
    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();

    fwd_moe_route_deepseek_params_t p0 = {
        .top_k = 2,
        .n_group = 2,
        .topk_group = 0,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p0, gpu) ==
           -1);

    fwd_moe_route_deepseek_params_t p1 = {
        .top_k = 2,
        .n_group = 2,
        .topk_group = 3,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p1, gpu) ==
           -1);

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

static void test_route_deepseek_rejects_tiny_group(void) {
    /* E=2, n_group=2 => Eg=1; top-2 group score undefined. */
    float logits_h[] = {1.f, 2.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 2}, 3);
    mlx_array bias = make_zero_bias(2);
    fwd_moe_route_deepseek_params_t p = {
        .top_k = 1,
        .n_group = 2,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };
    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == -1);
    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

static void test_route_deepseek_rejects_null_bias(void) {
    float logits_h[] = {0.f, 1.f, 2.f, 3.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 4}, 3);
    fwd_moe_route_deepseek_params_t p = {
        .top_k = 2,
        .n_group = 1,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };
    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits,
                                  (mlx_array){.ctx = NULL}, &p, gpu) == -1);
    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(logits);
}

/* Bias contract is rank-1 [E]. Broadcastable [1, E] must still be rejected
   at the prologue (not deferred to mlx_add). Live GPU stream required so a
   currently-accepted rank-2 bias would succeed and fail this assert. */
static void test_route_deepseek_rejects_bias_rank_ne_1(void) {
    float logits_h[] = {0.f, 1.f, 2.f, 3.f};
    float bias_h[] = {0.f, 0.f, 0.f, 0.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 4}, 3);
    mlx_array bias = make_f32(bias_h, (int[]){1, 4}, 2);
    fwd_moe_route_deepseek_params_t p = {
        .top_k = 2,
        .n_group = 1,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };
    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == -1);
    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

static void test_route_deepseek_rejects_bias_last_dim_ne_E(void) {
    float logits_h[] = {0.f, 1.f, 2.f, 3.f};
    float bias_h[] = {0.f, 0.f, 0.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 4}, 3);
    mlx_array bias = make_f32(bias_h, (int[]){3}, 1);
    fwd_moe_route_deepseek_params_t p = {
        .top_k = 2,
        .n_group = 1,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };
    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == -1);
    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

/* ---- Cycle 7: mlx-lm frozen goldens ---- */

/* Generated with mlx-lm group_expert_select (mlx-lm 0.31.x). */
static void test_route_deepseek_matches_mlx_lm_golden_vectors(void) {
    /* Case A: n_group=1, top_k=2, norm=true, scale=2.5 */
    {
        float gates[] = {0.f, 2.f, -1.f, 1.5f};
        float bias_h[] = {0.f, 0.f, 0.f, 0.f};
        /* Sorted by expert id: inds {1,3}, scores from mlx-lm */
        int exp_i[] = {1, 3};
        float exp_s[] = {1.2965316772460938f, 1.2034682035446167f};

        mlx_array logits = make_f32(gates, (int[]){1, 1, 4}, 3);
        mlx_array bias = make_f32(bias_h, (int[]){4}, 1);
        fwd_moe_route_deepseek_params_t p = {
            .top_k = 2,
            .n_group = 1,
            .topk_group = 1,
            .routed_scaling_factor = 2.5f,
            .norm_topk_prob = true,
        };
        mlx_array inds = mlx_array_new();
        mlx_array scores = mlx_array_new();
        assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) ==
               0);

        int32_t ih[2];
        float sh[2];
        read_i32(inds, ih, 2);
        read_f32(scores, sh, 2);
        assert(set_eq_inds(ih, exp_i, 2));
        sort_pairs_by_ind(ih, sh, 2);
        assert(fabsf(sh[0] - exp_s[0]) < 1e-5f);
        assert(fabsf(sh[1] - exp_s[1]) < 1e-5f);

        mlx_array_free(scores);
        mlx_array_free(inds);
        mlx_array_free(bias);
        mlx_array_free(logits);
    }

    /* Case B: n_group=4, E=16, topk_group=2, top_k=4, norm=true, scale=2.5,
       non-zero bias.
       Generation recipe (numpy):
         rs = numpy.RandomState(42)
         gates = rs.randn(16)
         bias  = 0.5 * rs.randn(16)
       Frozen inds/scores are mlx-lm group_expert_select outputs on those
       arrays. */
    {
        float gates[] = {
            0.49671414494514465f,  -0.13826429843902588f, 0.6476885676383972f,
            1.5230298042297363f,   -0.2341533750295639f,  -0.23413695394992828f,
            1.5792127847671509f,   0.7674347162246704f,   -0.4694743752479553f,
            0.5425600409507751f,   -0.4634176790714264f,  -0.4657297432422638f,
            0.241962268948555f,    -1.9132802486419678f,  -1.7249178886413574f,
            -0.5622875094413757f};
        float bias_h[] = {
            -0.5064155459403992f,  0.15712366998195648f,  -0.45401203632354736f,
            -0.7061518430709839f,  0.7328243851661682f,   -0.1128881499171257f,
            0.033764101564884186f, -0.7123740911483765f,  -0.2721913754940033f,
            0.05546129494905472f,  -0.5754967927932739f,  0.1878490149974823f,
            -0.3003193438053131f,  -0.1458468735218048f,  -0.30085331201553345f,
            0.9261391162872314f};
        /* mlx-lm inds [[15, 4, 6, 5]]; scores paired then sorted by id */
        int exp_i[] = {4, 5, 6, 15};
        float exp_s_by_id[] = {
            0.5320556163787842f, /* 4 */
            0.5320605039596558f, /* 5 */
            0.9986324310302734f, /* 6 */
            0.4372512698173523f  /* 15 */
        };

        mlx_array logits = make_f32(gates, (int[]){1, 1, 16}, 3);
        mlx_array bias = make_f32(bias_h, (int[]){16}, 1);
        fwd_moe_route_deepseek_params_t p = {
            .top_k = 4,
            .n_group = 4,
            .topk_group = 2,
            .routed_scaling_factor = 2.5f,
            .norm_topk_prob = true,
        };
        mlx_array inds = mlx_array_new();
        mlx_array scores = mlx_array_new();
        assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) ==
               0);

        int32_t ih[4];
        float sh[4];
        read_i32(inds, ih, 4);
        read_f32(scores, sh, 4);
        assert(set_eq_inds(ih, exp_i, 4));
        sort_pairs_by_ind(ih, sh, 4);
        for (int i = 0; i < 4; i++) {
            assert(ih[i] == exp_i[i]);
            assert(fabsf(sh[i] - exp_s_by_id[i]) < 1e-5f);
        }

        mlx_array_free(scores);
        mlx_array_free(inds);
        mlx_array_free(bias);
        mlx_array_free(logits);
    }
}

/* ---- Cycle 8: compose with switch + combine ---- */

static void test_fwd_moe_deepseek_compose_routed_only(void) {
    const int E = 4, H = 8, MI = 8, K = 2;
    cfg_init_basic(H);

    weight_triplet_t gate, up, down;
    build_bf16_switch(&gate, &up, &down, E, H, MI);

    float x_h[H];
    for (int i = 0; i < H; i++) x_h[i] = 0.02f * (float)(i + 1);
    mlx_array x = make_bf16(x_h, (int[]){1, 1, H}, 3);

    float logits_h[] = {0.f, 2.f, -1.f, 1.5f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, E}, 3);
    mlx_array bias = make_zero_bias(E);

    fwd_moe_route_deepseek_params_t rp = {
        .top_k = K,
        .n_group = 1,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = true,
    };

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &rp, gpu) == 0);

    mlx_array y_exp = mlx_array_new();
    assert(fwd_switch_glu(&y_exp, x, inds, &gate, &up, &down, &cfg, gpu) == 0);

    fwd_moe_weights_t mw;
    memset(&mw, 0, sizeof(mw));
    mw.has_shared = false;

    mlx_array out = mlx_array_new();
    assert(fwd_moe_combine(&out, y_exp, scores, x, &mw, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out)));
    assert(mlx_array_ndim(out) == 3);
    assert(mlx_array_dim(out, 0) == 1);
    assert(mlx_array_dim(out, 1) == 1);
    assert(mlx_array_dim(out, 2) == H);
    assert(is_finite_f32(out, gpu));

    float oh[H];
    read_f32(out, oh, (size_t)H);
    assert(has_nonzero(oh, (size_t)H, 1e-6f));

    /* Manual: (y_exp * scores[..., None]).sum(-2) */
    mlx_array scores_exp = mlx_array_new(), weighted = mlx_array_new(),
              ref = mlx_array_new();
    assert(MLXB_CHECK(mlx_expand_dims(&scores_exp, scores, -1, gpu)));
    assert(MLXB_CHECK(mlx_multiply(&weighted, y_exp, scores_exp, gpu)));
    assert(MLXB_CHECK(mlx_sum_axis(&ref, weighted, -2, false, gpu)));
    assert(MLXB_CHECK(mlx_array_eval(ref)));
    float rh[H];
    read_f32(ref, rh, (size_t)H);
    assert(max_abs_diff(oh, rh, (size_t)H) < 0.05f);

    mlx_array_free(ref);
    mlx_array_free(weighted);
    mlx_array_free(scores_exp);
    mlx_array_free(out);
    mlx_array_free(y_exp);
    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
    mlx_array_free(x);
    weights_triplet_free(&down);
    weights_triplet_free(&up);
    weights_triplet_free(&gate);
}

static void test_fwd_moe_deepseek_compose_with_shared_always_active(void) {
    const int E = 4, H = 8, MI = 8, SI = 6, K = 2;
    cfg_init_basic(H);

    weight_triplet_t gate, up, down;
    build_bf16_switch(&gate, &up, &down, E, H, MI);

    float x_h[H];
    for (int i = 0; i < H; i++) x_h[i] = 0.015f * (float)(i + 1);
    mlx_array x = make_bf16(x_h, (int[]){1, 1, H}, 3);

    float logits_h[] = {0.5f, 1.5f, -0.5f, 1.0f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, E}, 3);
    mlx_array bias = make_zero_bias(E);

    fwd_moe_route_deepseek_params_t rp = {
        .top_k = K,
        .n_group = 1,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = true,
    };

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &rp, gpu) == 0);

    mlx_array y_exp = mlx_array_new();
    assert(fwd_switch_glu(&y_exp, x, inds, &gate, &up, &down, &cfg, gpu) == 0);

    fwd_moe_weights_t mw;
    memset(&mw, 0, sizeof(mw));
    mw.shared_gate = make_dense_linear(SI, H, 80);
    mw.shared_up = make_dense_linear(SI, H, 81);
    mw.shared_down = make_dense_linear(H, SI, 82);
    mw.has_shared = true;
    mw.has_shared_expert_gate = false;

    mlx_array out = mlx_array_new();
    assert(fwd_moe_combine(&out, y_exp, scores, x, &mw, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out)));
    assert(is_finite_f32(out, gpu));

    /* routed-only + dense swiglu residual */
    fwd_moe_weights_t mw_r = mw;
    mw_r.has_shared = false;
    mlx_array routed = mlx_array_new();
    assert(fwd_moe_combine(&routed, y_exp, scores, x, &mw_r, &cfg, gpu) == 0);

    mlx_array g = mlx_array_new(), u = mlx_array_new(),
              gate_s = mlx_array_new(), silu = mlx_array_new(),
              mid = mlx_array_new(), shared = mlx_array_new(),
              ref = mlx_array_new();
    assert(fwd_linear(&g, x, &mw.shared_gate, &cfg, gpu) == 0);
    assert(fwd_linear(&u, x, &mw.shared_up, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_sigmoid(&gate_s, g, gpu)));
    assert(MLXB_CHECK(mlx_multiply(&silu, g, gate_s, gpu)));
    assert(MLXB_CHECK(mlx_multiply(&mid, silu, u, gpu)));
    assert(fwd_linear(&shared, mid, &mw.shared_down, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_add(&ref, routed, shared, gpu)));
    assert(MLXB_CHECK(mlx_array_eval(ref)));

    float oh[H], rh[H];
    read_f32(out, oh, (size_t)H);
    read_f32(ref, rh, (size_t)H);
    assert(max_abs_diff(oh, rh, (size_t)H) < 0.08f);

    mlx_array_free(ref);
    mlx_array_free(shared);
    mlx_array_free(mid);
    mlx_array_free(silu);
    mlx_array_free(gate_s);
    mlx_array_free(u);
    mlx_array_free(g);
    mlx_array_free(routed);
    mlx_array_free(out);
    mlx_array_free(y_exp);
    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
    mlx_array_free(x);
    weights_triplet_free(&mw.shared_down);
    weights_triplet_free(&mw.shared_up);
    weights_triplet_free(&mw.shared_gate);
    weights_triplet_free(&down);
    weights_triplet_free(&up);
    weights_triplet_free(&gate);
}

/* ---- Cycle 9: ownership hygiene ---- */

static void test_route_deepseek_failure_leaves_out_untouched(void) {
    float logits_h[] = {0.f, 1.f, 2.f, 3.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 4}, 3);
    mlx_array bias = make_zero_bias(4);

    /* Preload outs with known live arrays. */
    float pre_i[] = {9.f};
    float pre_s[] = {3.14f};
    mlx_array inds = make_f32(pre_i, (int[]){1}, 1);
    mlx_array scores = make_f32(pre_s, (int[]){1}, 1);
    void *inds_ctx = inds.ctx;
    void *scores_ctx = scores.ctx;

    fwd_moe_route_deepseek_params_t p = {
        .top_k = 0, /* invalid */
        .n_group = 1,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == -1);
    assert(inds.ctx == inds_ctx);
    assert(scores.ctx == scores_ctx);

    /* Also top_k > E */
    p.top_k = 5;
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == -1);
    assert(inds.ctx == inds_ctx);
    assert(scores.ctx == scores_ctx);

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

static void test_route_deepseek_success_replaces_out(void) {
    float logits_h[] = {0.f, 10.f, 1.f, 9.f};
    mlx_array logits = make_f32(logits_h, (int[]){1, 1, 4}, 3);
    mlx_array bias = make_zero_bias(4);

    float pre_i[] = {9.f};
    float pre_s[] = {3.14f};
    mlx_array inds = make_f32(pre_i, (int[]){1}, 1);
    mlx_array scores = make_f32(pre_s, (int[]){1}, 1);
    void *old_inds = inds.ctx;
    void *old_scores = scores.ctx;

    fwd_moe_route_deepseek_params_t p = {
        .top_k = 2,
        .n_group = 1,
        .topk_group = 1,
        .routed_scaling_factor = 1.f,
        .norm_topk_prob = false,
    };
    assert(fwd_moe_route_deepseek(&inds, &scores, logits, bias, &p, gpu) == 0);
    assert(inds.ctx != old_inds);
    assert(scores.ctx != old_scores);
    assert(MLXB_CHECK(mlx_array_eval(inds)));
    assert(MLXB_CHECK(mlx_array_eval(scores)));
    assert(mlx_array_dim(inds, 2) == 2);
    assert(mlx_array_dim(scores, 2) == 2);

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(bias);
    mlx_array_free(logits);
}

int main(void) {
    gpu = mlxbridge_gpu_stream();
    assert(gpu.ctx);

    test_route_deepseek_sigmoid_topk_n_group_1();
    printf("  test_route_deepseek_sigmoid_topk_n_group_1: passed\n");
    test_route_deepseek_scores_ignore_bias_for_values();
    printf("  test_route_deepseek_scores_ignore_bias_for_values: passed\n");

    test_route_deepseek_routed_scaling_factor();
    printf("  test_route_deepseek_routed_scaling_factor: passed\n");
    test_route_deepseek_norm_topk_prob_top_k_gt_1();
    printf("  test_route_deepseek_norm_topk_prob_top_k_gt_1: passed\n");
    test_route_deepseek_norm_topk_prob_ignored_when_top_k_1();
    printf("  test_route_deepseek_norm_topk_prob_ignored_when_top_k_1: passed\n");

    test_route_deepseek_group_keeps_topk_group_only();
    printf("  test_route_deepseek_group_keeps_topk_group_only: passed\n");
    test_route_deepseek_group_score_uses_top2_sum();
    printf("  test_route_deepseek_group_score_uses_top2_sum: passed\n");
    test_route_deepseek_k_drop_zero_skips_mask();
    printf("  test_route_deepseek_k_drop_zero_skips_mask: passed\n");
    test_route_deepseek_bias_breaks_group_ties();
    printf("  test_route_deepseek_bias_breaks_group_ties: passed\n");

    test_route_deepseek_batch_seq_matches_host_rows();
    printf("  test_route_deepseek_batch_seq_matches_host_rows: passed\n");
    test_route_deepseek_bf16_logits_match_f32_host();
    printf("  test_route_deepseek_bf16_logits_match_f32_host: passed\n");

    test_route_deepseek_rejects_top_k_gt_E();
    printf("  test_route_deepseek_rejects_top_k_gt_E: passed\n");
    test_route_deepseek_rejects_bad_n_group_divisibility();
    printf("  test_route_deepseek_rejects_bad_n_group_divisibility: passed\n");
    test_route_deepseek_rejects_bad_topk_group();
    printf("  test_route_deepseek_rejects_bad_topk_group: passed\n");
    test_route_deepseek_rejects_tiny_group();
    printf("  test_route_deepseek_rejects_tiny_group: passed\n");
    test_route_deepseek_rejects_null_bias();
    printf("  test_route_deepseek_rejects_null_bias: passed\n");
    test_route_deepseek_rejects_bias_rank_ne_1();
    printf("  test_route_deepseek_rejects_bias_rank_ne_1: passed\n");
    test_route_deepseek_rejects_bias_last_dim_ne_E();
    printf("  test_route_deepseek_rejects_bias_last_dim_ne_E: passed\n");

    test_route_deepseek_matches_mlx_lm_golden_vectors();
    printf("  test_route_deepseek_matches_mlx_lm_golden_vectors: passed\n");

    test_fwd_moe_deepseek_compose_routed_only();
    printf("  test_fwd_moe_deepseek_compose_routed_only: passed\n");
    test_fwd_moe_deepseek_compose_with_shared_always_active();
    printf("  test_fwd_moe_deepseek_compose_with_shared_always_active: passed\n");

    test_route_deepseek_failure_leaves_out_untouched();
    printf("  test_route_deepseek_failure_leaves_out_untouched: passed\n");
    test_route_deepseek_success_replaces_out();
    printf("  test_route_deepseek_success_replaces_out: passed\n");

    mlx_stream_free(gpu);
    printf("all test_fwd_moe_deepseek_gpu tests passed\n");
    return 0;
}
