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

/* GPU suite for #107 MoE block (fwd_moe). Synthetic experts only. */

static mlx_stream gpu;
static model_config_t cfg;

/* ---- helpers ---- */

static void cfg_init_quant(int bits, int gs) {
    memset(&cfg, 0, sizeof(cfg));
    cfg.quant_bits = bits;
    cfg.quant_group_size = gs;
    cfg.hidden_act = HIDDEN_ACT_SILU;
    cfg.hidden_size = 16;
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

/* Softmax over last axis for a single row of length E (double precision). */
static void softmax_row(const float *logits, double *probs, int E) {
    double maxv = logits[0];
    for (int i = 1; i < E; i++)
        if (logits[i] > maxv) maxv = logits[i];
    double sum = 0.0;
    for (int i = 0; i < E; i++) {
        probs[i] = exp((double)logits[i] - maxv);
        sum += probs[i];
    }
    for (int i = 0; i < E; i++) probs[i] /= sum;
}

static void quantize_triplet(weight_triplet_t *tri, mlx_array w_bf16,
                             int gs, int bits) {
    memset(tri, 0, sizeof(*tri));
    mlx_vector_array qr = mlx_vector_array_new();
    assert(MLXB_CHECK(mlx_quantize(
        &qr, w_bf16,
        (mlx_optional_int){.value = gs, .has_value = true},
        (mlx_optional_int){.value = bits, .has_value = true},
        "affine", mlx_array_empty, gpu)));
    tri->weight = mlx_array_new();
    tri->scales = mlx_array_new();
    tri->biases = mlx_array_new();
    assert(MLXB_CHECK(mlx_vector_array_get(&tri->weight, qr, 0)));
    assert(MLXB_CHECK(mlx_vector_array_get(&tri->scales, qr, 1)));
    assert(MLXB_CHECK(mlx_vector_array_get(&tri->biases, qr, 2)));
    tri->quantized = true;
    mlx_vector_array_free(qr);
}

static weight_triplet_t make_bf16_triplet(mlx_array w) {
    weight_triplet_t tri;
    memset(&tri, 0, sizeof(tri));
    /* Take ownership of w into the triplet. */
    tri.weight = w;
    tri.quantized = false;
    return tri;
}

static weight_triplet_t clone_bf16_weight(const float *data, const int *shape,
                                          int ndim) {
    return make_bf16_triplet(make_bf16(data, shape, ndim));
}

/* Build dense linear weight [out, in] with deterministic values. */
static void fill_linear(float *w, int out, int in, float scale, int seed) {
    for (int o = 0; o < out; o++)
        for (int i = 0; i < in; i++)
            w[o * in + i] =
                scale * (float)(((o * 17 + i * 3 + seed) % 13) - 6);
}

/* Expert stack quant [E, out, in] and bf16 pre-transposed [E, in, out]. */
static void fill_expert_eoi(float *w, int E, int out, int in, int seed) {
    for (int e = 0; e < E; e++)
        for (int o = 0; o < out; o++)
            for (int i = 0; i < in; i++)
                w[((e * out) + o) * in + i] =
                    0.05f * (float)(((e * 11 + o * 5 + i + seed) % 13) - 6);
}

static void transpose_eoi_to_eio(const float *eoi, float *eio,
                                 int E, int out, int in) {
    for (int e = 0; e < E; e++)
        for (int o = 0; o < out; o++)
            for (int i = 0; i < in; i++)
                eio[((e * in) + i) * out + o] = eoi[((e * out) + o) * in + i];
}

/* ---- Cycle 2: route_softmax ---- */

static void test_route_softmax_topk_indices_and_scores(void) {
    /* logits [1,1,4] = [0, 10, 1, 9]; top_k=2 -> experts {1,3} */
    float logits_h[] = {0.f, 10.f, 1.f, 9.f};
    int lshape[] = {1, 1, 4};
    mlx_array logits = make_bf16(logits_h, lshape, 3);

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_softmax(&inds, &scores, logits, 2, false, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(inds)));
    assert(MLXB_CHECK(mlx_array_eval(scores)));
    assert(mlx_array_ndim(inds) == 3);
    assert(mlx_array_dim(inds, 0) == 1);
    assert(mlx_array_dim(inds, 1) == 1);
    assert(mlx_array_dim(inds, 2) == 2);

    int32_t ih[2];
    {
        mlx_array i32 = mlx_array_new();
        assert(MLXB_CHECK(mlx_astype(&i32, inds, MLX_INT32, gpu)));
        assert(MLXB_CHECK(mlx_array_eval(i32)));
        const int32_t *d = mlx_array_data_int32(i32);
        assert(d);
        ih[0] = d[0];
        ih[1] = d[1];
        mlx_array_free(i32);
    }
    /* Set equality: {1, 3} */
    assert((ih[0] == 1 && ih[1] == 3) || (ih[0] == 3 && ih[1] == 1));

    double probs[4];
    softmax_row(logits_h, probs, 4);
    float sh[2];
    read_f32(scores, sh, 2);
    double exp0 = probs[ih[0]];
    double exp1 = probs[ih[1]];
    assert(fabs(sh[0] - exp0) < 2e-2);
    assert(fabs(sh[1] - exp1) < 2e-2);

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(logits);
}

static void test_route_softmax_norm_topk_prob(void) {
    float logits_h[] = {0.f, 10.f, 1.f, 9.f};
    int lshape[] = {1, 1, 4};
    mlx_array logits = make_bf16(logits_h, lshape, 3);

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_softmax(&inds, &scores, logits, 2, true, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(scores)));

    float sh[2];
    read_f32(scores, sh, 2);
    float sum = sh[0] + sh[1];
    assert(fabsf(sum - 1.f) < 1e-3f);
    assert(sh[0] > 0.f && sh[1] > 0.f);

    /* Proportional to un-normalized top softmax weights. */
    int32_t ih[2];
    {
        mlx_array i32 = mlx_array_new();
        assert(MLXB_CHECK(mlx_astype(&i32, inds, MLX_INT32, gpu)));
        assert(MLXB_CHECK(mlx_array_eval(i32)));
        const int32_t *d = mlx_array_data_int32(i32);
        ih[0] = d[0];
        ih[1] = d[1];
        mlx_array_free(i32);
    }
    double probs[4];
    softmax_row(logits_h, probs, 4);
    double t0 = probs[ih[0]], t1 = probs[ih[1]];
    double n0 = t0 / (t0 + t1), n1 = t1 / (t0 + t1);
    assert(fabs(sh[0] - n0) < 2e-2);
    assert(fabs(sh[1] - n1) < 2e-2);

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(logits);
}

static void test_route_softmax_batch_seq_shapes(void) {
    /* logits [2,3,4], K=2 -> inds/scores [2,3,2] */
    float logits_h[2 * 3 * 4];
    for (int i = 0; i < 2 * 3 * 4; i++)
        logits_h[i] = 0.1f * (float)((i * 7) % 11);
    int lshape[] = {2, 3, 4};
    mlx_array logits = make_bf16(logits_h, lshape, 3);

    mlx_array inds = mlx_array_new();
    mlx_array scores = mlx_array_new();
    assert(fwd_moe_route_softmax(&inds, &scores, logits, 2, false, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(inds)));
    assert(MLXB_CHECK(mlx_array_eval(scores)));
    assert(mlx_array_ndim(inds) == 3);
    assert(mlx_array_dim(inds, 0) == 2);
    assert(mlx_array_dim(inds, 1) == 3);
    assert(mlx_array_dim(inds, 2) == 2);
    assert(mlx_array_dim(scores, 0) == 2);
    assert(mlx_array_dim(scores, 1) == 3);
    assert(mlx_array_dim(scores, 2) == 2);
    assert(is_finite_f32(scores, gpu));

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(logits);
}

/* ---- Cycle 3: gather_qmm decode ---- */

static void test_gather_qmm_single_token_matches_manual_expert(void) {
    const int E = 4, out = 8, in = 32, gs = 32, bits = 4;
    cfg_init_quant(bits, gs);

    float w_h[E * out * in];
    fill_expert_eoi(w_h, E, out, in, 1);
    int wshape[] = {E, out, in};
    mlx_array w_bf = make_bf16(w_h, wshape, 3);
    weight_triplet_t tri;
    quantize_triplet(&tri, w_bf, gs, bits);
    mlx_array_free(w_bf);

    /* x [1,1,1,1,in] decode expand shape */
    float x_h[in];
    for (int i = 0; i < in; i++) x_h[i] = 0.02f * (float)(i + 1);
    int xshape[] = {1, 1, 1, 1, in};
    mlx_array x = make_bf16(x_h, xshape, 5);

    int e_pick = 2;
    uint32_t idata[] = {(uint32_t)e_pick};
    int ishape[] = {1, 1, 1};
    mlx_array inds = mlx_array_new_data(idata, ishape, 3, MLX_UINT32);

    mlx_array got = mlx_array_new();
    assert(fwd_gather_expert_linear(&got, x, inds, &tri, (mlx_array){.ctx = NULL},
                                    false, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(got)));
    assert(is_finite_f32(got, gpu));

    /* Reference: dequant full stack, take expert e, x @ W_e^T */
    mlx_array dq = mlx_array_new();
    assert(MLXB_CHECK(mlx_dequantize(
        &dq, tri.weight, tri.scales, tri.biases,
        (mlx_optional_int){.value = gs, .has_value = true},
        (mlx_optional_int){.value = bits, .has_value = true},
        "affine", mlx_array_empty,
        (mlx_optional_dtype){.value = MLX_BFLOAT16, .has_value = true},
        gpu)));
    uint32_t e_one[] = {(uint32_t)e_pick};
    mlx_array e_idx = mlx_array_new_data(e_one, (int[]){1}, 1, MLX_UINT32);
    mlx_array ew = mlx_array_new(), ew2 = mlx_array_new(), wt = mlx_array_new();
    assert(MLXB_CHECK(mlx_take_axis(&ew, dq, e_idx, 0, gpu))); /* [1,out,in] */
    assert(MLXB_CHECK(mlx_reshape(&ew2, ew, (int[]){out, in}, 2, gpu)));
    assert(MLXB_CHECK(mlx_transpose(&wt, ew2, gpu))); /* [in,out] */

    mlx_array x2 = make_bf16(x_h, (int[]){1, in}, 2);
    mlx_array ref = mlx_array_new();
    assert(MLXB_CHECK(mlx_matmul(&ref, x2, wt, gpu)));
    assert(MLXB_CHECK(mlx_array_eval(ref)));

    float got_h[8], ref_h[8];
    size_t gsz = mlx_array_size(got);
    assert(gsz == (size_t)out);
    read_f32(got, got_h, (size_t)out);
    read_f32(ref, ref_h, (size_t)out);
    assert(max_abs_diff(got_h, ref_h, (size_t)out) < 0.05f);
    assert(has_nonzero(got_h, (size_t)out, 1e-4f));

    mlx_array_free(ref);
    mlx_array_free(x2);
    mlx_array_free(wt);
    mlx_array_free(ew2);
    mlx_array_free(ew);
    mlx_array_free(e_idx);
    mlx_array_free(dq);
    mlx_array_free(got);
    mlx_array_free(inds);
    mlx_array_free(x);
    weights_triplet_free(&tri);
}

/* ---- Cycle 4: bf16 gather_mm ---- */

static void test_gather_mm_bf16_matches_per_expert_matmul(void) {
    const int E = 4, out = 8, in = 16;
    cfg_init_quant(4, 64); /* unused for bf16 path */

    float eoi[E * out * in];
    fill_expert_eoi(eoi, E, out, in, 3);
    float eio[E * in * out];
    transpose_eoi_to_eio(eoi, eio, E, out, in);
    weight_triplet_t tri =
        clone_bf16_weight(eio, (int[]){E, in, out}, 3);

    float x_h[in];
    for (int i = 0; i < in; i++) x_h[i] = 0.03f * (float)(i + 2);
    mlx_array x = make_bf16(x_h, (int[]){1, 1, 1, 1, in}, 5);

    int e_pick = 1;
    uint32_t idata[] = {(uint32_t)e_pick};
    mlx_array inds = mlx_array_new_data(idata, (int[]){1, 1, 1}, 3, MLX_UINT32);

    mlx_array got = mlx_array_new();
    assert(fwd_gather_expert_linear(&got, x, inds, &tri, (mlx_array){.ctx = NULL},
                                    false, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(got)));

    /* Reference: matmul x[1,in] @ w_e[in,out] */
    mlx_array x2 = make_bf16(x_h, (int[]){1, in}, 2);
    mlx_array we = make_bf16(eio + e_pick * in * out, (int[]){in, out}, 2);
    mlx_array ref = mlx_array_new();
    assert(MLXB_CHECK(mlx_matmul(&ref, x2, we, gpu)));
    assert(MLXB_CHECK(mlx_array_eval(ref)));

    float got_h[out], ref_h[out];
    read_f32(got, got_h, out);
    read_f32(ref, ref_h, out);
    assert(max_abs_diff(got_h, ref_h, out) < 0.05f);
    assert(has_nonzero(got_h, out, 1e-4f));

    mlx_array_free(ref);
    mlx_array_free(we);
    mlx_array_free(x2);
    mlx_array_free(got);
    mlx_array_free(inds);
    mlx_array_free(x);
    weights_triplet_free(&tri);
}

/* ---- Cycle 5: contiguous gotcha ---- */

static void test_gather_qmm_contiguous_slice_matches_independent_quant(void) {
    /* Geometry near real 4-bit/gs64 layout so the strided path fires. */
    const int E = 16, M = 128, IN = 512, gs = 64, bits = 4;
    cfg_init_quant(bits, gs);

    size_t packed_n = (size_t)E * 2 * M * IN;
    float *w_host = (float *)malloc(packed_n * sizeof(float));
    assert(w_host);
    for (size_t i = 0; i < packed_n; i++)
        w_host[i] = 0.07f * (float)((int)(i % 17) - 8);

    mlx_array w_bf = make_bf16(w_host, (int[]){E, 2 * M, IN}, 3);
    weight_triplet_t packed;
    quantize_triplet(&packed, w_bf, gs, bits);
    mlx_array_free(w_bf);

    /* Lazy slice gate half [E, M, ...] on q/scales/biases. */
    int g_start[] = {0, 0, 0};
    int g_stop_q[] = {E, M, mlx_array_dim(packed.weight, 2)};
    int g_stop_s[] = {E, M, mlx_array_dim(packed.scales, 2)};
    int g_stop_b[] = {E, M, mlx_array_dim(packed.biases, 2)};
    int strides[] = {1, 1, 1};

    mlx_array lazy_q = mlx_array_new(), lazy_s = mlx_array_new(),
              lazy_b = mlx_array_new();
    assert(MLXB_CHECK(mlx_slice(&lazy_q, packed.weight, g_start, 3,
                                g_stop_q, 3, strides, 3, gpu)));
    assert(MLXB_CHECK(mlx_slice(&lazy_s, packed.scales, g_start, 3,
                                g_stop_s, 3, strides, 3, gpu)));
    assert(MLXB_CHECK(mlx_slice(&lazy_b, packed.biases, g_start, 3,
                                g_stop_b, 3, strides, 3, gpu)));

    /* Contiguous materialization of the slice (production helper). */
    mlx_array cq = mlx_array_new(), cs = mlx_array_new(), cb = mlx_array_new();
    assert(fwd_array_contiguous(&cq, lazy_q, gpu) == 0);
    assert(fwd_array_contiguous(&cs, lazy_s, gpu) == 0);
    assert(fwd_array_contiguous(&cb, lazy_b, gpu) == 0);

    weight_triplet_t cont_tri;
    memset(&cont_tri, 0, sizeof(cont_tri));
    cont_tri.weight = cq;
    cont_tri.scales = cs;
    cont_tri.biases = cb;
    cont_tri.quantized = true;

    /* Independent reference: quantize gate rows directly. */
    float *g_host = (float *)malloc((size_t)E * M * IN * sizeof(float));
    assert(g_host);
    for (int e = 0; e < E; e++)
        for (int r = 0; r < M; r++)
            for (int c = 0; c < IN; c++)
                g_host[((e * M) + r) * IN + c] =
                    w_host[((e * 2 * M) + r) * IN + c];
    mlx_array g_bf = make_bf16(g_host, (int[]){E, M, IN}, 3);
    weight_triplet_t ref_tri;
    quantize_triplet(&ref_tri, g_bf, gs, bits);
    mlx_array_free(g_bf);

    /* Decode shape: x [1,1,1,1,IN], inds pick expert 3 */
    float *x_host = (float *)malloc((size_t)IN * sizeof(float));
    assert(x_host);
    for (int i = 0; i < IN; i++)
        x_host[i] = 0.11f * (float)(i % 7) - 0.3f;
    mlx_array x_dec = make_bf16(x_host, (int[]){1, 1, 1, 1, IN}, 5);
    uint32_t idata[] = {3};
    mlx_array inds_dec =
        mlx_array_new_data(idata, (int[]){1, 1, 1}, 3, MLX_UINT32);

    mlx_array out_cont = mlx_array_new();
    mlx_array out_ref = mlx_array_new();
    assert(fwd_gather_expert_linear(&out_cont, x_dec, inds_dec, &cont_tri,
                                    (mlx_array){.ctx = NULL}, false, &cfg,
                                    gpu) == 0);
    assert(fwd_gather_expert_linear(&out_ref, x_dec, inds_dec, &ref_tri,
                                    (mlx_array){.ctx = NULL}, false, &cfg,
                                    gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out_cont)));
    assert(MLXB_CHECK(mlx_array_eval(out_ref)));

    float *cont_h = (float *)malloc((size_t)M * sizeof(float));
    float *ref_h = (float *)malloc((size_t)M * sizeof(float));
    assert(cont_h && ref_h);
    read_f32(out_cont, cont_h, M);
    read_f32(out_ref, ref_h, M);
    assert(max_abs_diff(cont_h, ref_h, M) < 0.05f);
    assert(has_nonzero(cont_h, M, 1e-3f));

    /* Sorted prefill shape: N=E, sorted inds [0..E) */
    float *xr = (float *)malloc((size_t)E * IN * sizeof(float));
    assert(xr);
    for (size_t i = 0; i < (size_t)E * IN; i++)
        xr[i] = 0.05f * (float)(i % 11) - 0.2f;
    mlx_array x_pref = make_bf16(xr, (int[]){E, 1, IN}, 3);
    uint32_t *inds_h = (uint32_t *)malloc((size_t)E * sizeof(uint32_t));
    assert(inds_h);
    for (int i = 0; i < E; i++) inds_h[i] = (uint32_t)i;
    mlx_array inds_pref =
        mlx_array_new_data(inds_h, (int[]){E}, 1, MLX_UINT32);

    mlx_array out_c2 = mlx_array_new();
    mlx_array out_r2 = mlx_array_new();
    assert(fwd_gather_expert_linear(&out_c2, x_pref, inds_pref, &cont_tri,
                                    (mlx_array){.ctx = NULL}, true, &cfg,
                                    gpu) == 0);
    assert(fwd_gather_expert_linear(&out_r2, x_pref, inds_pref, &ref_tri,
                                    (mlx_array){.ctx = NULL}, true, &cfg,
                                    gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out_c2)));
    assert(MLXB_CHECK(mlx_array_eval(out_r2)));

    size_t n2 = (size_t)E * M;
    float *c2 = (float *)malloc(n2 * sizeof(float));
    float *r2 = (float *)malloc(n2 * sizeof(float));
    assert(c2 && r2);
    read_f32(out_c2, c2, n2);
    read_f32(out_r2, r2, n2);
    assert(max_abs_diff(c2, r2, n2) < 0.05f);
    assert(has_nonzero(c2, n2, 1e-3f));

    free(r2);
    free(c2);
    mlx_array_free(out_r2);
    mlx_array_free(out_c2);
    mlx_array_free(inds_pref);
    free(inds_h);
    mlx_array_free(x_pref);
    free(xr);
    free(ref_h);
    free(cont_h);
    mlx_array_free(out_ref);
    mlx_array_free(out_cont);
    mlx_array_free(inds_dec);
    mlx_array_free(x_dec);
    free(x_host);
    weights_triplet_free(&ref_tri);
    free(g_host);
    /* cont_tri owns cq/cs/cb */
    weights_triplet_free(&cont_tri);
    mlx_array_free(lazy_b);
    mlx_array_free(lazy_s);
    mlx_array_free(lazy_q);
    weights_triplet_free(&packed);
    free(w_host);
}

/* ---- Cycle 6/7: switch_glu ---- */

static void build_quant_switch(weight_triplet_t *gate, weight_triplet_t *up,
                               weight_triplet_t *down, int E, int H, int MI,
                               int gs, int bits) {
    float *g = (float *)malloc((size_t)E * MI * H * sizeof(float));
    float *u = (float *)malloc((size_t)E * MI * H * sizeof(float));
    float *d = (float *)malloc((size_t)E * H * MI * sizeof(float));
    assert(g && u && d);
    fill_expert_eoi(g, E, MI, H, 10);
    fill_expert_eoi(u, E, MI, H, 20);
    fill_expert_eoi(d, E, H, MI, 30);
    mlx_array gb = make_bf16(g, (int[]){E, MI, H}, 3);
    mlx_array ub = make_bf16(u, (int[]){E, MI, H}, 3);
    mlx_array db = make_bf16(d, (int[]){E, H, MI}, 3);
    quantize_triplet(gate, gb, gs, bits);
    quantize_triplet(up, ub, gs, bits);
    quantize_triplet(down, db, gs, bits);
    mlx_array_free(db);
    mlx_array_free(ub);
    mlx_array_free(gb);
    free(d);
    free(u);
    free(g);
}

static void build_bf16_switch(weight_triplet_t *gate, weight_triplet_t *up,
                              weight_triplet_t *down, int E, int H, int MI) {
    float *g_eoi = (float *)malloc((size_t)E * MI * H * sizeof(float));
    float *u_eoi = (float *)malloc((size_t)E * MI * H * sizeof(float));
    float *d_eoi = (float *)malloc((size_t)E * H * MI * sizeof(float));
    float *g_eio = (float *)malloc((size_t)E * H * MI * sizeof(float));
    float *u_eio = (float *)malloc((size_t)E * H * MI * sizeof(float));
    float *d_eio = (float *)malloc((size_t)E * MI * H * sizeof(float));
    assert(g_eoi && u_eoi && d_eoi && g_eio && u_eio && d_eio);
    fill_expert_eoi(g_eoi, E, MI, H, 11);
    fill_expert_eoi(u_eoi, E, MI, H, 21);
    fill_expert_eoi(d_eoi, E, H, MI, 31);
    transpose_eoi_to_eio(g_eoi, g_eio, E, MI, H);
    transpose_eoi_to_eio(u_eoi, u_eio, E, MI, H);
    transpose_eoi_to_eio(d_eoi, d_eio, E, H, MI);
    *gate = clone_bf16_weight(g_eio, (int[]){E, H, MI}, 3);
    *up = clone_bf16_weight(u_eio, (int[]){E, H, MI}, 3);
    *down = clone_bf16_weight(d_eio, (int[]){E, MI, H}, 3);
    free(d_eio);
    free(u_eio);
    free(g_eio);
    free(d_eoi);
    free(u_eoi);
    free(g_eoi);
}

/* Manual per-token SwitchGLU via gather on decode path (reference). */
static void switch_glu_ref_decode(mlx_array *out, mlx_array x, mlx_array inds,
                                  const weight_triplet_t *gate,
                                  const weight_triplet_t *up,
                                  const weight_triplet_t *down) {
    /* Force decode path by calling gather with expanded x ourselves. */
    int B = mlx_array_dim(x, 0);
    int S = mlx_array_dim(x, 1);
    int H = mlx_array_dim(x, 2);
    int exp_shape[] = {B, S, 1, 1, H};
    mlx_array x_exp = mlx_array_new();
    assert(MLXB_CHECK(mlx_reshape(&x_exp, x, exp_shape, 5, gpu)));

    mlx_array g = mlx_array_new(), u = mlx_array_new(), d = mlx_array_new();
    mlx_array gs = mlx_array_new(), us = mlx_array_new();
    mlx_array act = mlx_array_new(), act_exp = mlx_array_new();
    mlx_array gsig = mlx_array_new(), silu = mlx_array_new();

    assert(fwd_gather_expert_linear(&g, x_exp, inds, gate,
                                    (mlx_array){.ctx = NULL}, false, &cfg,
                                    gpu) == 0);
    assert(fwd_gather_expert_linear(&u, x_exp, inds, up,
                                    (mlx_array){.ctx = NULL}, false, &cfg,
                                    gpu) == 0);
    assert(MLXB_CHECK(mlx_squeeze_axis(&gs, g, -2, gpu)));
    assert(MLXB_CHECK(mlx_squeeze_axis(&us, u, -2, gpu)));
    assert(MLXB_CHECK(mlx_sigmoid(&gsig, gs, gpu)));
    assert(MLXB_CHECK(mlx_multiply(&silu, gs, gsig, gpu)));
    assert(MLXB_CHECK(mlx_multiply(&act, silu, us, gpu)));
    assert(MLXB_CHECK(mlx_expand_dims(&act_exp, act, -2, gpu)));
    assert(fwd_gather_expert_linear(&d, act_exp, inds, down,
                                    (mlx_array){.ctx = NULL}, false, &cfg,
                                    gpu) == 0);
    assert(MLXB_CHECK(mlx_squeeze_axis(out, d, -2, gpu)));

    mlx_array_free(silu);
    mlx_array_free(gsig);
    mlx_array_free(act_exp);
    mlx_array_free(act);
    mlx_array_free(us);
    mlx_array_free(gs);
    mlx_array_free(d);
    mlx_array_free(u);
    mlx_array_free(g);
    mlx_array_free(x_exp);
}

static void test_switch_glu_decode_matches_three_gather_steps(void) {
    const int E = 4, H = 32, MI = 32, K = 2, gs = 32, bits = 4;
    cfg_init_quant(bits, gs);
    cfg.hidden_size = H;

    weight_triplet_t gate, up, down;
    build_quant_switch(&gate, &up, &down, E, H, MI, gs, bits);

    float x_h[H];
    for (int i = 0; i < H; i++) x_h[i] = 0.01f * (float)(i + 1);
    mlx_array x = make_bf16(x_h, (int[]){1, 1, H}, 3);
    uint32_t idata[] = {1, 3};
    mlx_array inds =
        mlx_array_new_data(idata, (int[]){1, 1, K}, 3, MLX_UINT32);

    mlx_array got = mlx_array_new();
    assert(fwd_switch_glu(&got, x, inds, &gate, &up, &down, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(got)));
    assert(mlx_array_ndim(got) == 4);
    assert(mlx_array_dim(got, 0) == 1);
    assert(mlx_array_dim(got, 1) == 1);
    assert(mlx_array_dim(got, 2) == K);
    assert(mlx_array_dim(got, 3) == H);
    assert(is_finite_f32(got, gpu));

    mlx_array ref = mlx_array_new();
    switch_glu_ref_decode(&ref, x, inds, &gate, &up, &down);
    assert(MLXB_CHECK(mlx_array_eval(ref)));

    size_t n = (size_t)K * H;
    float *gh = (float *)malloc(n * sizeof(float));
    float *rh = (float *)malloc(n * sizeof(float));
    read_f32(got, gh, n);
    read_f32(ref, rh, n);
    assert(max_abs_diff(gh, rh, n) < 0.05f);
    assert(has_nonzero(gh, n, 1e-4f));

    free(rh);
    free(gh);
    mlx_array_free(ref);
    mlx_array_free(got);
    mlx_array_free(inds);
    mlx_array_free(x);
    weights_triplet_free(&down);
    weights_triplet_free(&up);
    weights_triplet_free(&gate);
}

static void test_switch_glu_sorted_path_matches_unsorted_reference(void) {
    /* B=1,S=4,K=2,E=8 -> S>1 forces sort (R2). */
    const int E = 8, H = 32, MI = 32, B = 1, S = 4, K = 2, gs = 32, bits = 4;
    cfg_init_quant(bits, gs);
    cfg.hidden_size = H;

    weight_triplet_t gate, up, down;
    build_quant_switch(&gate, &up, &down, E, H, MI, gs, bits);

    float x_h[B * S * H];
    for (int i = 0; i < B * S * H; i++)
        x_h[i] = 0.01f * (float)((i % 9) + 1);
    mlx_array x = make_bf16(x_h, (int[]){B, S, H}, 3);

    uint32_t idata[B * S * K];
    for (int i = 0; i < B * S * K; i++) idata[i] = (uint32_t)(i % E);
    mlx_array inds =
        mlx_array_new_data(idata, (int[]){B, S, K}, 3, MLX_UINT32);

    mlx_array got = mlx_array_new();
    assert(fwd_switch_glu(&got, x, inds, &gate, &up, &down, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(got)));
    assert(mlx_array_dim(got, 0) == B);
    assert(mlx_array_dim(got, 1) == S);
    assert(mlx_array_dim(got, 2) == K);
    assert(mlx_array_dim(got, 3) == H);

    /* Reference: naive per-position decode gather (unsorted). */
    mlx_array ref = mlx_array_new();
    switch_glu_ref_decode(&ref, x, inds, &gate, &up, &down);
    assert(MLXB_CHECK(mlx_array_eval(ref)));

    size_t n = (size_t)B * S * K * H;
    float *gh = (float *)malloc(n * sizeof(float));
    float *rh = (float *)malloc(n * sizeof(float));
    read_f32(got, gh, n);
    read_f32(ref, rh, n);
    assert(max_abs_diff(gh, rh, n) < 0.08f);
    assert(has_nonzero(gh, n, 1e-4f));

    free(rh);
    free(gh);
    mlx_array_free(ref);
    mlx_array_free(got);
    mlx_array_free(inds);
    mlx_array_free(x);
    weights_triplet_free(&down);
    weights_triplet_free(&up);
    weights_triplet_free(&gate);
}

static void test_switch_glu_sort_threshold_s1_small_stays_decode_path(void) {
    /* S=1, K=2, B=1 - decode path; smoke correctness. */
    const int E = 4, H = 32, MI = 32, gs = 32, bits = 4;
    cfg_init_quant(bits, gs);
    cfg.hidden_size = H;

    weight_triplet_t gate, up, down;
    build_quant_switch(&gate, &up, &down, E, H, MI, gs, bits);

    float x_h[H];
    for (int i = 0; i < H; i++) x_h[i] = 0.02f * (float)(i + 1);
    mlx_array x = make_bf16(x_h, (int[]){1, 1, H}, 3);
    uint32_t idata[] = {0, 2};
    mlx_array inds =
        mlx_array_new_data(idata, (int[]){1, 1, 2}, 3, MLX_UINT32);

    mlx_array got = mlx_array_new();
    assert(fwd_switch_glu(&got, x, inds, &gate, &up, &down, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(got)));
    assert(mlx_array_ndim(got) == 4);
    assert(mlx_array_dim(got, 2) == 2);
    assert(mlx_array_dim(got, 3) == H);
    assert(is_finite_f32(got, gpu));

    mlx_array_free(got);
    mlx_array_free(inds);
    mlx_array_free(x);
    weights_triplet_free(&down);
    weights_triplet_free(&up);
    weights_triplet_free(&gate);
}

/* R2 second disjunct: S==1 && total_inds >= 64 must sort. */
static void test_switch_glu_sort_threshold_s1_large_uses_sorted_path(void) {
    const int B = 1, S = 1, K = 64, E = 64, H = 32, MI = 32, gs = 32, bits = 4;
    cfg_init_quant(bits, gs);
    cfg.hidden_size = H;

    weight_triplet_t gate, up, down;
    build_quant_switch(&gate, &up, &down, E, H, MI, gs, bits);

    float x_h[H];
    for (int i = 0; i < H; i++) x_h[i] = 0.01f * (float)(i + 1);
    mlx_array x = make_bf16(x_h, (int[]){B, S, H}, 3);

    uint32_t idata[K];
    for (int i = 0; i < K; i++) idata[i] = (uint32_t)(i % E);
    mlx_array inds =
        mlx_array_new_data(idata, (int[]){B, S, K}, 3, MLX_UINT32);

    mlx_array got = mlx_array_new();
    assert(fwd_switch_glu(&got, x, inds, &gate, &up, &down, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(got)));
    assert(mlx_array_ndim(got) == 4);
    assert(mlx_array_dim(got, 0) == B);
    assert(mlx_array_dim(got, 1) == S);
    assert(mlx_array_dim(got, 2) == K);
    assert(mlx_array_dim(got, 3) == H);

    mlx_array ref = mlx_array_new();
    switch_glu_ref_decode(&ref, x, inds, &gate, &up, &down);
    assert(MLXB_CHECK(mlx_array_eval(ref)));

    size_t n = (size_t)B * S * K * H;
    float *gh = (float *)malloc(n * sizeof(float));
    float *rh = (float *)malloc(n * sizeof(float));
    read_f32(got, gh, n);
    read_f32(ref, rh, n);
    assert(max_abs_diff(gh, rh, n) < 0.08f);
    assert(has_nonzero(gh, n, 1e-4f));

    free(rh);
    free(gh);
    mlx_array_free(ref);
    mlx_array_free(got);
    mlx_array_free(inds);
    mlx_array_free(x);
    weights_triplet_free(&down);
    weights_triplet_free(&up);
    weights_triplet_free(&gate);
}

/* ---- Cycle 8-10: fwd_moe / fwd_moe_combine ---- */

static weight_triplet_t make_router(int E, int H, int seed) {
    float *w = (float *)malloc((size_t)E * H * sizeof(float));
    assert(w);
    fill_linear(w, E, H, 0.05f, seed);
    weight_triplet_t tri = clone_bf16_weight(w, (int[]){E, H}, 2);
    free(w);
    return tri;
}

static weight_triplet_t make_dense_linear(int out, int in, int seed) {
    float *w = (float *)malloc((size_t)out * in * sizeof(float));
    assert(w);
    fill_linear(w, out, in, 0.04f, seed);
    weight_triplet_t tri = clone_bf16_weight(w, (int[]){out, in}, 2);
    free(w);
    return tri;
}

static void test_fwd_moe_combine_matches_manual_weighted_sum(void) {
    const int E = 4, H = 32, MI = 32, S = 2, K = 2, gs = 32, bits = 4;
    cfg_init_quant(bits, gs);
    cfg.hidden_size = H;

    weight_triplet_t gate, up, down;
    build_quant_switch(&gate, &up, &down, E, H, MI, gs, bits);

    float x_h[S * H];
    for (int i = 0; i < S * H; i++) x_h[i] = 0.01f * (float)(i + 1);
    mlx_array x = make_bf16(x_h, (int[]){1, S, H}, 3);

    /* Fixed logits -> known scores path. */
    float logits_h[S * E];
    for (int i = 0; i < S * E; i++)
        logits_h[i] = 0.2f * (float)((i * 3) % 7);
    mlx_array logits = make_bf16(logits_h, (int[]){1, S, E}, 3);

    mlx_array inds = mlx_array_new(), scores = mlx_array_new();
    assert(fwd_moe_route_softmax(&inds, &scores, logits, K, true, gpu) == 0);

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
    assert(mlx_array_dim(out, 1) == S);
    assert(mlx_array_dim(out, 2) == H);

    /* Manual ref: (y_exp * scores[..., None]).sum(-2) */
    mlx_array scores_exp = mlx_array_new(), weighted = mlx_array_new(),
              ref = mlx_array_new();
    assert(MLXB_CHECK(mlx_expand_dims(&scores_exp, scores, -1, gpu)));
    assert(MLXB_CHECK(mlx_multiply(&weighted, y_exp, scores_exp, gpu)));
    assert(MLXB_CHECK(mlx_sum_axis(&ref, weighted, -2, false, gpu)));
    assert(MLXB_CHECK(mlx_array_eval(ref)));

    size_t n = (size_t)S * H;
    float *oh = (float *)malloc(n * sizeof(float));
    float *rh = (float *)malloc(n * sizeof(float));
    read_f32(out, oh, n);
    read_f32(ref, rh, n);
    assert(max_abs_diff(oh, rh, n) < 0.05f);
    assert(has_nonzero(oh, n, 1e-5f));

    free(rh);
    free(oh);
    mlx_array_free(ref);
    mlx_array_free(weighted);
    mlx_array_free(scores_exp);
    mlx_array_free(out);
    mlx_array_free(y_exp);
    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(logits);
    mlx_array_free(x);
    weights_triplet_free(&down);
    weights_triplet_free(&up);
    weights_triplet_free(&gate);
}

static void test_fwd_moe_combine_shared_always_active(void) {
    const int E = 4, H = 32, MI = 32, SI = 24, S = 2, K = 2, gs = 32, bits = 4;
    cfg_init_quant(bits, gs);
    cfg.hidden_size = H;

    weight_triplet_t gate, up, down;
    build_quant_switch(&gate, &up, &down, E, H, MI, gs, bits);

    float x_h[S * H];
    for (int i = 0; i < S * H; i++) x_h[i] = 0.015f * (float)(i + 1);
    mlx_array x = make_bf16(x_h, (int[]){1, S, H}, 3);

    float logits_h[S * E];
    for (int i = 0; i < S * E; i++)
        logits_h[i] = 0.15f * (float)((i * 5) % 9);
    mlx_array logits = make_bf16(logits_h, (int[]){1, S, E}, 3);

    mlx_array inds = mlx_array_new(), scores = mlx_array_new();
    assert(fwd_moe_route_softmax(&inds, &scores, logits, K, true, gpu) == 0);
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

    /* routed-only combine + dense swiglu residual */
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

    size_t n = (size_t)S * H;
    float *oh = (float *)malloc(n * sizeof(float));
    float *rh = (float *)malloc(n * sizeof(float));
    read_f32(out, oh, n);
    read_f32(ref, rh, n);
    assert(max_abs_diff(oh, rh, n) < 0.08f);

    free(rh);
    free(oh);
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
    mlx_array_free(logits);
    mlx_array_free(x);
    weights_triplet_free(&mw.shared_down);
    weights_triplet_free(&mw.shared_up);
    weights_triplet_free(&mw.shared_gate);
    weights_triplet_free(&down);
    weights_triplet_free(&up);
    weights_triplet_free(&gate);
}

static void test_fwd_moe_routed_only_shape_and_manual_parity(void) {
    const int E = 4, H = 32, MI = 32, S = 3, K = 2, gs = 32, bits = 4;
    cfg_init_quant(bits, gs);
    cfg.hidden_size = H;

    fwd_moe_weights_t mw;
    memset(&mw, 0, sizeof(mw));
    mw.router = make_router(E, H, 1);
    build_quant_switch(&mw.switch_gate, &mw.switch_up, &mw.switch_down,
                       E, H, MI, gs, bits);
    mw.has_shared = false;

    fwd_moe_params_t p = {.num_experts = E, .top_k = K, .norm_topk_prob = true};

    float x_h[1 * S * H];
    for (int i = 0; i < S * H; i++) x_h[i] = 0.01f * (float)(i + 1);
    mlx_array x = make_bf16(x_h, (int[]){1, S, H}, 3);

    mlx_array out = mlx_array_new();
    assert(fwd_moe(&out, x, &mw, &p, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out)));
    assert(mlx_array_ndim(out) == 3);
    assert(mlx_array_dim(out, 0) == 1);
    assert(mlx_array_dim(out, 1) == S);
    assert(mlx_array_dim(out, 2) == H);
    assert(is_finite_f32(out, gpu));

    /* Manual: linear -> route -> switch -> weighted sum */
    mlx_array logits = mlx_array_new();
    assert(fwd_linear(&logits, x, &mw.router, &cfg, gpu) == 0);
    mlx_array inds = mlx_array_new(), scores = mlx_array_new();
    assert(fwd_moe_route_softmax(&inds, &scores, logits, K, true, gpu) == 0);
    mlx_array y_exp = mlx_array_new();
    assert(fwd_switch_glu(&y_exp, x, inds, &mw.switch_gate, &mw.switch_up,
                          &mw.switch_down, &cfg, gpu) == 0);
    mlx_array scores_exp = mlx_array_new(), weighted = mlx_array_new(),
              ref = mlx_array_new();
    assert(MLXB_CHECK(mlx_expand_dims(&scores_exp, scores, -1, gpu)));
    assert(MLXB_CHECK(mlx_multiply(&weighted, y_exp, scores_exp, gpu)));
    assert(MLXB_CHECK(mlx_sum_axis(&ref, weighted, -2, false, gpu)));
    assert(MLXB_CHECK(mlx_array_eval(ref)));

    size_t n = (size_t)S * H;
    float *oh = (float *)malloc(n * sizeof(float));
    float *rh = (float *)malloc(n * sizeof(float));
    read_f32(out, oh, n);
    read_f32(ref, rh, n);
    assert(max_abs_diff(oh, rh, n) < 0.05f);
    assert(has_nonzero(oh, n, 1e-5f));

    free(rh);
    free(oh);
    mlx_array_free(ref);
    mlx_array_free(weighted);
    mlx_array_free(scores_exp);
    mlx_array_free(y_exp);
    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(logits);
    mlx_array_free(out);
    mlx_array_free(x);
    weights_triplet_free(&mw.switch_down);
    weights_triplet_free(&mw.switch_up);
    weights_triplet_free(&mw.switch_gate);
    weights_triplet_free(&mw.router);

    /* Also exercise norm_topk_prob=false */
    memset(&mw, 0, sizeof(mw));
    mw.router = make_router(E, H, 2);
    build_quant_switch(&mw.switch_gate, &mw.switch_up, &mw.switch_down,
                       E, H, MI, gs, bits);
    p.norm_topk_prob = false;
    x = make_bf16(x_h, (int[]){1, S, H}, 3);
    out = mlx_array_new();
    assert(fwd_moe(&out, x, &mw, &p, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out)));
    assert(is_finite_f32(out, gpu));
    mlx_array_free(out);
    mlx_array_free(x);
    weights_triplet_free(&mw.switch_down);
    weights_triplet_free(&mw.switch_up);
    weights_triplet_free(&mw.switch_gate);
    weights_triplet_free(&mw.router);
}

static void test_fwd_moe_shared_always_active_adds_dense_swiglu(void) {
    const int E = 4, H = 32, MI = 32, SI = 24, S = 2, K = 2, gs = 32, bits = 4;
    cfg_init_quant(bits, gs);
    cfg.hidden_size = H;

    fwd_moe_weights_t mw;
    memset(&mw, 0, sizeof(mw));
    mw.router = make_router(E, H, 5);
    build_quant_switch(&mw.switch_gate, &mw.switch_up, &mw.switch_down,
                       E, H, MI, gs, bits);
    mw.shared_gate = make_dense_linear(SI, H, 40);
    mw.shared_up = make_dense_linear(SI, H, 41);
    mw.shared_down = make_dense_linear(H, SI, 42);
    mw.has_shared = true;
    mw.has_shared_expert_gate = false;

    fwd_moe_params_t p = {.num_experts = E, .top_k = K, .norm_topk_prob = true};

    float x_h[S * H];
    for (int i = 0; i < S * H; i++) x_h[i] = 0.015f * (float)(i + 1);
    mlx_array x = make_bf16(x_h, (int[]){1, S, H}, 3);

    mlx_array out = mlx_array_new();
    assert(fwd_moe(&out, x, &mw, &p, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out)));

    /* Reference = routed-only + shared swiglu */
    fwd_moe_weights_t mw_r = mw;
    mw_r.has_shared = false;
    mlx_array routed = mlx_array_new();
    assert(fwd_moe(&routed, x, &mw_r, &p, &cfg, gpu) == 0);

    mlx_array shared = mlx_array_new();
    {
        /* dense swiglu via linear pieces */
        mlx_array g = mlx_array_new(), u = mlx_array_new(),
                  gate_s = mlx_array_new(), silu = mlx_array_new(),
                  mid = mlx_array_new();
        assert(fwd_linear(&g, x, &mw.shared_gate, &cfg, gpu) == 0);
        assert(fwd_linear(&u, x, &mw.shared_up, &cfg, gpu) == 0);
        assert(MLXB_CHECK(mlx_sigmoid(&gate_s, g, gpu)));
        assert(MLXB_CHECK(mlx_multiply(&silu, g, gate_s, gpu)));
        assert(MLXB_CHECK(mlx_multiply(&mid, silu, u, gpu)));
        assert(fwd_linear(&shared, mid, &mw.shared_down, &cfg, gpu) == 0);
        mlx_array_free(mid);
        mlx_array_free(silu);
        mlx_array_free(gate_s);
        mlx_array_free(u);
        mlx_array_free(g);
    }
    mlx_array ref = mlx_array_new();
    assert(MLXB_CHECK(mlx_add(&ref, routed, shared, gpu)));
    assert(MLXB_CHECK(mlx_array_eval(ref)));

    size_t n = (size_t)S * H;
    float *oh = (float *)malloc(n * sizeof(float));
    float *rh = (float *)malloc(n * sizeof(float));
    read_f32(out, oh, n);
    read_f32(ref, rh, n);
    assert(max_abs_diff(oh, rh, n) < 0.08f);

    free(rh);
    free(oh);
    mlx_array_free(ref);
    mlx_array_free(shared);
    mlx_array_free(routed);
    mlx_array_free(out);
    mlx_array_free(x);
    weights_triplet_free(&mw.shared_down);
    weights_triplet_free(&mw.shared_up);
    weights_triplet_free(&mw.shared_gate);
    weights_triplet_free(&mw.switch_down);
    weights_triplet_free(&mw.switch_up);
    weights_triplet_free(&mw.switch_gate);
    weights_triplet_free(&mw.router);
}

static void test_fwd_moe_shared_sigmoid_gate(void) {
    const int E = 4, H = 32, MI = 32, SI = 24, S = 2, K = 2, gs = 32, bits = 4;
    cfg_init_quant(bits, gs);
    cfg.hidden_size = H;

    fwd_moe_weights_t mw;
    memset(&mw, 0, sizeof(mw));
    mw.router = make_router(E, H, 7);
    build_quant_switch(&mw.switch_gate, &mw.switch_up, &mw.switch_down,
                       E, H, MI, gs, bits);
    mw.shared_gate = make_dense_linear(SI, H, 50);
    mw.shared_up = make_dense_linear(SI, H, 51);
    mw.shared_down = make_dense_linear(H, SI, 52);
    mw.shared_expert_gate = make_dense_linear(1, H, 53);
    mw.has_shared = true;
    mw.has_shared_expert_gate = true;

    fwd_moe_params_t p = {.num_experts = E, .top_k = K, .norm_topk_prob = true};

    float x_h[S * H];
    for (int i = 0; i < S * H; i++) x_h[i] = 0.012f * (float)(i + 1);
    mlx_array x = make_bf16(x_h, (int[]){1, S, H}, 3);

    mlx_array out = mlx_array_new();
    assert(fwd_moe(&out, x, &mw, &p, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out)));

    /* ref = routed + sigmoid(gate_lin(x)) * shared_swiglu(x) */
    fwd_moe_weights_t mw_r = mw;
    mw_r.has_shared = false;
    mw_r.has_shared_expert_gate = false;
    mlx_array routed = mlx_array_new();
    assert(fwd_moe(&routed, x, &mw_r, &p, &cfg, gpu) == 0);

    mlx_array g = mlx_array_new(), u = mlx_array_new(),
              gate_s = mlx_array_new(), silu = mlx_array_new(),
              mid = mlx_array_new(), shared = mlx_array_new(),
              gl = mlx_array_new(), gsig = mlx_array_new(),
              gated = mlx_array_new(), ref = mlx_array_new();
    assert(fwd_linear(&g, x, &mw.shared_gate, &cfg, gpu) == 0);
    assert(fwd_linear(&u, x, &mw.shared_up, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_sigmoid(&gate_s, g, gpu)));
    assert(MLXB_CHECK(mlx_multiply(&silu, g, gate_s, gpu)));
    assert(MLXB_CHECK(mlx_multiply(&mid, silu, u, gpu)));
    assert(fwd_linear(&shared, mid, &mw.shared_down, &cfg, gpu) == 0);
    assert(fwd_linear(&gl, x, &mw.shared_expert_gate, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_sigmoid(&gsig, gl, gpu)));
    assert(MLXB_CHECK(mlx_multiply(&gated, gsig, shared, gpu)));
    assert(MLXB_CHECK(mlx_add(&ref, routed, gated, gpu)));
    assert(MLXB_CHECK(mlx_array_eval(ref)));

    size_t n = (size_t)S * H;
    float *oh = (float *)malloc(n * sizeof(float));
    float *rh = (float *)malloc(n * sizeof(float));
    read_f32(out, oh, n);
    read_f32(ref, rh, n);
    assert(max_abs_diff(oh, rh, n) < 0.08f);

    free(rh);
    free(oh);
    mlx_array_free(ref);
    mlx_array_free(gated);
    mlx_array_free(gsig);
    mlx_array_free(gl);
    mlx_array_free(shared);
    mlx_array_free(mid);
    mlx_array_free(silu);
    mlx_array_free(gate_s);
    mlx_array_free(u);
    mlx_array_free(g);
    mlx_array_free(routed);
    mlx_array_free(out);
    mlx_array_free(x);
    weights_triplet_free(&mw.shared_expert_gate);
    weights_triplet_free(&mw.shared_down);
    weights_triplet_free(&mw.shared_up);
    weights_triplet_free(&mw.shared_gate);
    weights_triplet_free(&mw.switch_down);
    weights_triplet_free(&mw.switch_up);
    weights_triplet_free(&mw.switch_gate);
    weights_triplet_free(&mw.router);
}

/* ---- Cycle 11: error hygiene ---- */

static void test_fwd_moe_error_hygiene(void) {
    cfg_init_quant(4, 64);
    cfg.hidden_size = 16;

    mlx_array x = make_bf16((float[]){0.1f, 0.2f}, (int[]){1, 1, 2}, 3);
    /* Note: H=2 above is intentional mismatch for some checks; rebuild. */
    mlx_array_free(x);
    float x_h[16];
    for (int i = 0; i < 16; i++) x_h[i] = 0.01f * (float)i;
    x = make_bf16(x_h, (int[]){1, 1, 16}, 3);

    /* On failure *out must be unchanged - pin a live empty array. */
    mlx_array out = mlx_array_new();
    void *sentinel_ctx = out.ctx;

    assert(fwd_moe(NULL, x, NULL, NULL, NULL, gpu) == -1);

    fwd_moe_weights_t mw;
    memset(&mw, 0, sizeof(mw));
    fwd_moe_params_t p = {.num_experts = 4, .top_k = 2, .norm_topk_prob = false};
    assert(fwd_moe(&out, x, NULL, &p, &cfg, gpu) == -1);
    assert(out.ctx == sentinel_ctx);

    assert(fwd_moe(&out, x, &mw, NULL, &cfg, gpu) == -1);
    assert(fwd_moe(&out, x, &mw, &p, NULL, gpu) == -1);

    p.top_k = 0;
    assert(fwd_moe(&out, x, &mw, &p, &cfg, gpu) == -1);
    p.top_k = 8;
    p.num_experts = 4;
    assert(fwd_moe(&out, x, &mw, &p, &cfg, gpu) == -1);
    p.top_k = 2;

    /* Missing switch triplets */
    mw.router = make_router(4, 16, 99);
    assert(fwd_moe(&out, x, &mw, &p, &cfg, gpu) == -1);
    assert(out.ctx == sentinel_ctx);

    /* top_k > E on route */
    mlx_array logits = make_bf16((float[]){1.f, 2.f, 3.f, 4.f},
                                 (int[]){1, 1, 4}, 3);
    mlx_array inds = mlx_array_new(), scores = mlx_array_new();
    assert(fwd_moe_route_softmax(&inds, &scores, logits, 5, false, gpu) == -1);
    assert(fwd_moe_route_softmax(&inds, &scores, logits, 0, false, gpu) == -1);

    mlx_array_free(scores);
    mlx_array_free(inds);
    mlx_array_free(logits);
    weights_triplet_free(&mw.router);
    /* out still holds sentinel - free once */
    mlx_array_free(out);
    mlx_array_free(x);
}

/* M2: num_experts must match router logits last dim. */
static void test_fwd_moe_rejects_router_expert_count_mismatch(void) {
    const int E_router = 4, E_claim = 8, H = 32, MI = 32, K = 2, gs = 32,
              bits = 4;
    cfg_init_quant(bits, gs);
    cfg.hidden_size = H;

    fwd_moe_weights_t mw;
    memset(&mw, 0, sizeof(mw));
    mw.router = make_router(E_router, H, 11);
    build_quant_switch(&mw.switch_gate, &mw.switch_up, &mw.switch_down,
                       E_router, H, MI, gs, bits);
    mw.has_shared = false;

    /* top_k still <= E_router so bare route_softmax would succeed. */
    fwd_moe_params_t p = {
        .num_experts = E_claim, .top_k = K, .norm_topk_prob = false};

    float x_h[H];
    for (int i = 0; i < H; i++) x_h[i] = 0.01f * (float)(i + 1);
    mlx_array x = make_bf16(x_h, (int[]){1, 1, H}, 3);

    mlx_array out = mlx_array_new();
    void *sentinel_ctx = out.ctx;

    assert(fwd_moe(&out, x, &mw, &p, &cfg, gpu) == -1);
    assert(out.ctx == sentinel_ctx);

    mlx_array_free(out);
    mlx_array_free(x);
    weights_triplet_free(&mw.switch_down);
    weights_triplet_free(&mw.switch_up);
    weights_triplet_free(&mw.switch_gate);
    weights_triplet_free(&mw.router);
}

/* L3: gate flag without shared is inconsistent. */
static void test_fwd_moe_rejects_gate_without_shared(void) {
    const int E = 4, H = 32, MI = 32, K = 2, gs = 32, bits = 4;
    cfg_init_quant(bits, gs);
    cfg.hidden_size = H;

    fwd_moe_weights_t mw;
    memset(&mw, 0, sizeof(mw));
    mw.router = make_router(E, H, 12);
    build_quant_switch(&mw.switch_gate, &mw.switch_up, &mw.switch_down,
                       E, H, MI, gs, bits);
    /* Live gate triplet so only the flag inconsistency can reject. */
    mw.shared_expert_gate = make_dense_linear(1, H, 70);
    mw.has_shared = false;
    mw.has_shared_expert_gate = true;

    fwd_moe_params_t p = {.num_experts = E, .top_k = K, .norm_topk_prob = false};

    float x_h[H];
    for (int i = 0; i < H; i++) x_h[i] = 0.01f * (float)(i + 1);
    mlx_array x = make_bf16(x_h, (int[]){1, 1, H}, 3);

    mlx_array out = mlx_array_new();
    void *sentinel_ctx = out.ctx;

    assert(fwd_moe(&out, x, &mw, &p, &cfg, gpu) == -1);
    assert(out.ctx == sentinel_ctx);

    mlx_array_free(out);
    mlx_array_free(x);
    weights_triplet_free(&mw.shared_expert_gate);
    weights_triplet_free(&mw.switch_down);
    weights_triplet_free(&mw.switch_up);
    weights_triplet_free(&mw.switch_gate);
    weights_triplet_free(&mw.router);
}

/* ---- Cycle 12: quant + bf16 integration ---- */

static void test_fwd_moe_all_bf16_experts(void) {
    const int E = 4, H = 32, MI = 32, S = 2, K = 2;
    cfg_init_quant(4, 64);
    cfg.hidden_size = H;

    fwd_moe_weights_t mw;
    memset(&mw, 0, sizeof(mw));
    mw.router = make_router(E, H, 8);
    build_bf16_switch(&mw.switch_gate, &mw.switch_up, &mw.switch_down, E, H, MI);
    mw.has_shared = false;

    fwd_moe_params_t p = {.num_experts = E, .top_k = K, .norm_topk_prob = true};

    float x_h[S * H];
    for (int i = 0; i < S * H; i++) x_h[i] = 0.02f * (float)(i + 1);
    mlx_array x = make_bf16(x_h, (int[]){1, S, H}, 3);

    mlx_array out = mlx_array_new();
    assert(fwd_moe(&out, x, &mw, &p, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out)));
    assert(mlx_array_dim(out, 2) == H);
    assert(is_finite_f32(out, gpu));

    float oh[S * H];
    read_f32(out, oh, (size_t)S * H);
    assert(has_nonzero(oh, (size_t)S * H, 1e-5f));

    mlx_array_free(out);
    mlx_array_free(x);
    weights_triplet_free(&mw.switch_down);
    weights_triplet_free(&mw.switch_up);
    weights_triplet_free(&mw.switch_gate);
    weights_triplet_free(&mw.router);
}

static void test_fwd_moe_quant_experts_bf16_shared(void) {
    const int E = 4, H = 32, MI = 32, SI = 24, S = 2, K = 2, gs = 32, bits = 4;
    cfg_init_quant(bits, gs);
    cfg.hidden_size = H;

    fwd_moe_weights_t mw;
    memset(&mw, 0, sizeof(mw));
    mw.router = make_router(E, H, 9);
    build_quant_switch(&mw.switch_gate, &mw.switch_up, &mw.switch_down,
                       E, H, MI, gs, bits);
    mw.shared_gate = make_dense_linear(SI, H, 60);
    mw.shared_up = make_dense_linear(SI, H, 61);
    mw.shared_down = make_dense_linear(H, SI, 62);
    mw.has_shared = true;

    fwd_moe_params_t p = {.num_experts = E, .top_k = K, .norm_topk_prob = false};

    float x_h[S * H];
    for (int i = 0; i < S * H; i++) x_h[i] = 0.01f * (float)(i + 3);
    mlx_array x = make_bf16(x_h, (int[]){1, S, H}, 3);

    mlx_array out = mlx_array_new();
    assert(fwd_moe(&out, x, &mw, &p, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out)));
    assert(is_finite_f32(out, gpu));

    mlx_array_free(out);
    mlx_array_free(x);
    weights_triplet_free(&mw.shared_down);
    weights_triplet_free(&mw.shared_up);
    weights_triplet_free(&mw.shared_gate);
    weights_triplet_free(&mw.switch_down);
    weights_triplet_free(&mw.switch_up);
    weights_triplet_free(&mw.switch_gate);
    weights_triplet_free(&mw.router);
}

int main(void) {
    gpu = mlxbridge_gpu_stream();
    cfg_init_quant(4, 64);

    test_route_softmax_topk_indices_and_scores();
    printf("  test_route_softmax_topk_indices_and_scores: passed\n");
    test_route_softmax_norm_topk_prob();
    printf("  test_route_softmax_norm_topk_prob: passed\n");
    test_route_softmax_batch_seq_shapes();
    printf("  test_route_softmax_batch_seq_shapes: passed\n");

    test_gather_qmm_single_token_matches_manual_expert();
    printf("  test_gather_qmm_single_token_matches_manual_expert: passed\n");
    test_gather_mm_bf16_matches_per_expert_matmul();
    printf("  test_gather_mm_bf16_matches_per_expert_matmul: passed\n");

    test_gather_qmm_contiguous_slice_matches_independent_quant();
    printf("  test_gather_qmm_contiguous_slice_matches_independent_quant: passed\n");

    test_switch_glu_decode_matches_three_gather_steps();
    printf("  test_switch_glu_decode_matches_three_gather_steps: passed\n");
    test_switch_glu_sorted_path_matches_unsorted_reference();
    printf("  test_switch_glu_sorted_path_matches_unsorted_reference: passed\n");
    test_switch_glu_sort_threshold_s1_small_stays_decode_path();
    printf("  test_switch_glu_sort_threshold_s1_small_stays_decode_path: passed\n");
    test_switch_glu_sort_threshold_s1_large_uses_sorted_path();
    printf("  test_switch_glu_sort_threshold_s1_large_uses_sorted_path: passed\n");

    test_fwd_moe_combine_matches_manual_weighted_sum();
    printf("  test_fwd_moe_combine_matches_manual_weighted_sum: passed\n");
    test_fwd_moe_combine_shared_always_active();
    printf("  test_fwd_moe_combine_shared_always_active: passed\n");
    test_fwd_moe_routed_only_shape_and_manual_parity();
    printf("  test_fwd_moe_routed_only_shape_and_manual_parity: passed\n");
    test_fwd_moe_shared_always_active_adds_dense_swiglu();
    printf("  test_fwd_moe_shared_always_active_adds_dense_swiglu: passed\n");
    test_fwd_moe_shared_sigmoid_gate();
    printf("  test_fwd_moe_shared_sigmoid_gate: passed\n");

    test_fwd_moe_error_hygiene();
    printf("  test_fwd_moe_error_hygiene: passed\n");
    test_fwd_moe_rejects_router_expert_count_mismatch();
    printf("  test_fwd_moe_rejects_router_expert_count_mismatch: passed\n");
    test_fwd_moe_rejects_gate_without_shared();
    printf("  test_fwd_moe_rejects_gate_without_shared: passed\n");

    test_fwd_moe_all_bf16_experts();
    printf("  test_fwd_moe_all_bf16_experts: passed\n");
    test_fwd_moe_quant_experts_bf16_shared();
    printf("  test_fwd_moe_quant_experts_bf16_shared: passed\n");

    mlx_stream_free(gpu);
    printf("all test_fwd_moe_gpu tests passed\n");
    return 0;
}
