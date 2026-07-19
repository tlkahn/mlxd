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

static model_config_t cfg_dense;
static weights_t w_dense;
static model_config_t cfg_quant;
static weights_t w_quant;
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

/* ---- B1.1: fwd_linear ---- */

static void test_fwd_linear(void) {
    /* Dense: q_proj layer 0: x [1,3,64] -> out [1,3,64] bf16 finite */
    float xdata[192];
    for (int i = 0; i < 192; i++) xdata[i] = 0.01f * (float)i;
    int xshape[] = {1, 3, 64};
    mlx_array x_f32 = mlx_array_new_data(xdata, xshape, 3, MLX_FLOAT32);
    mlx_array x = mlx_array_new();
    MLXB_CHECK(mlx_astype(&x, x_f32, MLX_BFLOAT16, gpu));

    char name[256];
    weights_tensor_name(name, sizeof(name), &cfg_dense, 0, "self_attn.q_proj");
    weight_triplet_t tri;
    assert(weights_get_triplet(&tri, &w_dense, name) == 0);
    assert(!tri.quantized);

    mlx_array out = mlx_array_new();
    assert(fwd_linear(&out, x, &tri, &cfg_dense, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out)));
    assert(mlx_array_ndim(out) == 3);
    assert(mlx_array_dim(out, 0) == 1);
    assert(mlx_array_dim(out, 1) == 3);
    assert(mlx_array_dim(out, 2) == 64);
    assert(mlx_array_dtype(out) == MLX_BFLOAT16);
    assert(is_finite_f32(out, gpu));

    weights_triplet_free(&tri);
    mlx_array_free(out);
    mlx_array_free(x);
    mlx_array_free(x_f32);

    /* Quantized: q_proj layer 0, verify matches dequantize + transpose + matmul */
    x_f32 = mlx_array_new_data(xdata, xshape, 3, MLX_FLOAT32);
    x = mlx_array_new();
    MLXB_CHECK(mlx_astype(&x, x_f32, MLX_BFLOAT16, gpu));

    weights_tensor_name(name, sizeof(name), &cfg_quant, 0, "self_attn.q_proj");
    assert(weights_get_triplet(&tri, &w_quant, name) == 0);
    assert(tri.quantized);

    mlx_array qmm_out = mlx_array_new();
    assert(fwd_linear(&qmm_out, x, &tri, &cfg_quant, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(qmm_out)));
    assert(mlx_array_dtype(qmm_out) == MLX_BFLOAT16);

    /* Reference: dequantize + transpose + matmul */
    mlx_array dq = mlx_array_new();
    MLXB_CHECK(mlx_dequantize(&dq, tri.weight, tri.scales, tri.biases,
        (mlx_optional_int){.value = cfg_quant.quant_group_size, .has_value = true},
        (mlx_optional_int){.value = cfg_quant.quant_bits, .has_value = true},
        "affine", (mlx_array){.ctx = NULL}, (mlx_optional_dtype){.has_value = false}, gpu));
    mlx_array dq_t = mlx_array_new();
    MLXB_CHECK(mlx_transpose(&dq_t, dq, gpu));
    mlx_array ref = mlx_array_new();
    MLXB_CHECK(mlx_matmul(&ref, x, dq_t, gpu));

    /* Compare */
    mlx_array qmm_f32 = mlx_array_new();
    mlx_array ref_f32 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&qmm_f32, qmm_out, MLX_FLOAT32, gpu));
    MLXB_CHECK(mlx_astype(&ref_f32, ref, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(qmm_f32)));
    assert(MLXB_CHECK(mlx_array_eval(ref_f32)));

    const float *qd = mlx_array_data_float32(qmm_f32);
    const float *rd = mlx_array_data_float32(ref_f32);
    size_t n = mlx_array_size(qmm_f32);
    float maxdiff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(qd[i] - rd[i]);
        if (diff > maxdiff) maxdiff = diff;
    }
    assert(maxdiff < 1e-2f);

    mlx_array_free(ref_f32);
    mlx_array_free(qmm_f32);
    mlx_array_free(ref);
    mlx_array_free(dq_t);
    mlx_array_free(dq);
    mlx_array_free(qmm_out);
    weights_triplet_free(&tri);
    mlx_array_free(x);
    mlx_array_free(x_f32);
}

/* ---- B1.2: fwd_embed ---- */

static void test_fwd_embed(void) {
    int32_t ids_data[] = {0, 1, 2};
    int ids_shape[] = {1, 3};
    mlx_array ids = mlx_array_new_data(ids_data, ids_shape, 2, MLX_INT32);

    mlx_array out = mlx_array_new();
    assert(fwd_embed(&out, ids, &w_dense, &cfg_dense, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out)));
    assert(mlx_array_ndim(out) == 3);
    assert(mlx_array_dim(out, 0) == 1);
    assert(mlx_array_dim(out, 1) == 3);
    assert(mlx_array_dim(out, 2) == 64);
    assert(mlx_array_dtype(out) == MLX_BFLOAT16);
    assert(is_finite_f32(out, gpu));

    /* Reference: manual take + astype */
    char wname[270];
    weights_tensor_name(wname, sizeof(wname), &cfg_dense, -1, "embed_tokens");
    char full[280];
    snprintf(full, sizeof(full), "%s.weight", wname);
    mlx_array embed_w = mlx_array_new();
    assert(weights_get(&embed_w, &w_dense, full) == 0);

    int flat_shape[] = {3};
    mlx_array flat_ids = mlx_array_new();
    MLXB_CHECK(mlx_reshape(&flat_ids, ids, flat_shape, 1, gpu));
    mlx_array ref_taken = mlx_array_new();
    MLXB_CHECK(mlx_take_axis(&ref_taken, embed_w, flat_ids, 0, gpu));
    mlx_array ref_bf16 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&ref_bf16, ref_taken, MLX_BFLOAT16, gpu));
    int ref_shape[] = {1, 3, 64};
    mlx_array ref = mlx_array_new();
    MLXB_CHECK(mlx_reshape(&ref, ref_bf16, ref_shape, 3, gpu));

    mlx_array out_f32 = mlx_array_new();
    mlx_array ref_f32 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&out_f32, out, MLX_FLOAT32, gpu));
    MLXB_CHECK(mlx_astype(&ref_f32, ref, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(out_f32)));
    assert(MLXB_CHECK(mlx_array_eval(ref_f32)));
    const float *od = mlx_array_data_float32(out_f32);
    const float *rd = mlx_array_data_float32(ref_f32);
    for (size_t i = 0; i < 192; i++)
        assert(fabsf(od[i] - rd[i]) < 1e-3f);

    mlx_array_free(ref_f32);
    mlx_array_free(out_f32);
    mlx_array_free(ref);
    mlx_array_free(ref_bf16);
    mlx_array_free(ref_taken);
    mlx_array_free(flat_ids);
    mlx_array_free(embed_w);
    mlx_array_free(out);
    mlx_array_free(ids);

    /* Quantized embed: must produce bf16 output + match manual dequant reference */
    int32_t ids2_data[] = {0, 1, 2};
    mlx_array ids2 = mlx_array_new_data(ids2_data, ids_shape, 2, MLX_INT32);
    mlx_array out2 = mlx_array_new();
    assert(fwd_embed(&out2, ids2, &w_quant, &cfg_quant, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out2)));
    assert(mlx_array_ndim(out2) == 3);
    assert(mlx_array_dim(out2, 2) == 64);
    assert(mlx_array_dtype(out2) == MLX_BFLOAT16);
    assert(is_finite_f32(out2, gpu));

    /* Reference: manual take on weight/scales/biases rows, dequantize, cast bf16 */
    char qname[256];
    weights_tensor_name(qname, sizeof(qname), &cfg_quant, -1, "embed_tokens");
    weight_triplet_t etri;
    assert(weights_get_triplet(&etri, &w_quant, qname) == 0);
    assert(etri.quantized);

    int flat2_shape[] = {3};
    mlx_array flat2 = mlx_array_new();
    MLXB_CHECK(mlx_reshape(&flat2, ids2, flat2_shape, 1, gpu));

    mlx_array tw = mlx_array_new();
    mlx_array ts = mlx_array_new();
    mlx_array tb = mlx_array_new();
    MLXB_CHECK(mlx_take_axis(&tw, etri.weight, flat2, 0, gpu));
    MLXB_CHECK(mlx_take_axis(&ts, etri.scales, flat2, 0, gpu));
    MLXB_CHECK(mlx_take_axis(&tb, etri.biases, flat2, 0, gpu));

    mlx_array dq2 = mlx_array_new();
    MLXB_CHECK(mlx_dequantize(&dq2, tw, ts, tb,
        (mlx_optional_int){.value = cfg_quant.quant_group_size, .has_value = true},
        (mlx_optional_int){.value = cfg_quant.quant_bits, .has_value = true},
        "affine", (mlx_array){.ctx = NULL}, (mlx_optional_dtype){.has_value = false}, gpu));

    mlx_array dq2_bf16 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&dq2_bf16, dq2, MLX_BFLOAT16, gpu));
    int ref2_shape[] = {1, 3, 64};
    mlx_array ref2 = mlx_array_new();
    MLXB_CHECK(mlx_reshape(&ref2, dq2_bf16, ref2_shape, 3, gpu));

    mlx_array out2_f32 = mlx_array_new();
    mlx_array ref2_f32 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&out2_f32, out2, MLX_FLOAT32, gpu));
    MLXB_CHECK(mlx_astype(&ref2_f32, ref2, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(out2_f32)));
    assert(MLXB_CHECK(mlx_array_eval(ref2_f32)));
    const float *qod = mlx_array_data_float32(out2_f32);
    const float *qrd = mlx_array_data_float32(ref2_f32);
    for (size_t qi = 0; qi < 192; qi++)
        assert(fabsf(qod[qi] - qrd[qi]) < 1e-3f);

    mlx_array_free(ref2_f32);
    mlx_array_free(out2_f32);
    mlx_array_free(ref2);
    mlx_array_free(dq2_bf16);
    mlx_array_free(dq2);
    mlx_array_free(tb);
    mlx_array_free(ts);
    mlx_array_free(tw);
    mlx_array_free(flat2);
    weights_triplet_free(&etri);
    mlx_array_free(out2);
    mlx_array_free(ids2);
}

/* ---- B1.3: fwd_rmsnorm ---- */

static void test_fwd_rmsnorm(void) {
    float xdata[192];
    for (int i = 0; i < 192; i++) xdata[i] = 0.01f * (float)(i + 1);
    int xshape[] = {1, 3, 64};
    mlx_array x_f32 = mlx_array_new_data(xdata, xshape, 3, MLX_FLOAT32);
    mlx_array x = mlx_array_new();
    MLXB_CHECK(mlx_astype(&x, x_f32, MLX_BFLOAT16, gpu));

    char name[256], wname[270];
    weights_tensor_name(name, sizeof(name), &cfg_dense, 0, "input_layernorm");
    snprintf(wname, sizeof(wname), "%s.weight", name);
    mlx_array norm_w = mlx_array_new();
    assert(weights_get(&norm_w, &w_dense, wname) == 0);

    mlx_array out = mlx_array_new();
    assert(fwd_rmsnorm(&out, x, norm_w, cfg_dense.rms_norm_eps, false, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out)));
    assert(mlx_array_ndim(out) == 3);
    assert(mlx_array_dim(out, 0) == 1);
    assert(mlx_array_dim(out, 1) == 3);
    assert(mlx_array_dim(out, 2) == 64);
    assert(is_finite_f32(out, gpu));

    /* Per-head probe: weight [16] over [1,4,3,16] -> shape preserved */
    int head_shape[] = {1, 4, 3, 16};
    mlx_array x_head = mlx_array_new();
    MLXB_CHECK(mlx_reshape(&x_head, x, head_shape, 4, gpu));

    weights_tensor_name(name, sizeof(name), &cfg_dense, 0, "self_attn.q_norm");
    snprintf(wname, sizeof(wname), "%s.weight", name);
    mlx_array qnorm_w = mlx_array_new();
    assert(weights_get(&qnorm_w, &w_dense, wname) == 0);

    mlx_array out_head = mlx_array_new();
    assert(fwd_rmsnorm(&out_head, x_head, qnorm_w, cfg_dense.rms_norm_eps, false, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out_head)));
    assert(mlx_array_ndim(out_head) == 4);
    assert(mlx_array_dim(out_head, 0) == 1);
    assert(mlx_array_dim(out_head, 1) == 4);
    assert(mlx_array_dim(out_head, 2) == 3);
    assert(mlx_array_dim(out_head, 3) == 16);
    assert(is_finite_f32(out_head, gpu));

    mlx_array_free(out_head);
    mlx_array_free(qnorm_w);
    mlx_array_free(x_head);
    mlx_array_free(out);
    mlx_array_free(norm_w);
    mlx_array_free(x);
    mlx_array_free(x_f32);
}

/* ---- B1.4: fwd_attention ---- */

static void test_fwd_attention(void) {
    float xdata[192];
    for (int i = 0; i < 192; i++) xdata[i] = 0.01f * (float)(i + 1);
    int xshape[] = {1, 3, 64};
    mlx_array x_f32 = mlx_array_new_data(xdata, xshape, 3, MLX_FLOAT32);
    mlx_array x = mlx_array_new();
    MLXB_CHECK(mlx_astype(&x, x_f32, MLX_BFLOAT16, gpu));

    kvcache_t kv;
    assert(kvcache_init(&kv, cfg_dense.num_hidden_layers,
                        cfg_dense.num_key_value_heads, cfg_dense.head_dim) == 0);

    mlx_array out = mlx_array_new();
    assert(fwd_attention(&out, x, 0, &w_dense, &cfg_dense, &kv, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out)));
    assert(mlx_array_ndim(out) == 3);
    assert(mlx_array_dim(out, 0) == 1);
    assert(mlx_array_dim(out, 1) == 3);
    assert(mlx_array_dim(out, 2) == 64);
    assert(mlx_array_dtype(out) == MLX_BFLOAT16);
    assert(is_finite_f32(out, gpu));

    /* Causal probe: perturb position 2 input, output at position 0 unchanged */
    kvcache_t kv2;
    assert(kvcache_init(&kv2, cfg_dense.num_hidden_layers,
                        cfg_dense.num_key_value_heads, cfg_dense.head_dim) == 0);
    float xdata2[192];
    memcpy(xdata2, xdata, sizeof(xdata));
    for (int d = 0; d < 64; d++) xdata2[128 + d] += 100.0f;
    mlx_array x2_f32 = mlx_array_new_data(xdata2, xshape, 3, MLX_FLOAT32);
    mlx_array x2 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&x2, x2_f32, MLX_BFLOAT16, gpu));

    mlx_array out2 = mlx_array_new();
    assert(fwd_attention(&out2, x2, 0, &w_dense, &cfg_dense, &kv2, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out2)));

    mlx_array out_f32 = mlx_array_new();
    mlx_array out2_f32 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&out_f32, out, MLX_FLOAT32, gpu));
    MLXB_CHECK(mlx_astype(&out2_f32, out2, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(out_f32)));
    assert(MLXB_CHECK(mlx_array_eval(out2_f32)));
    const float *d1 = mlx_array_data_float32(out_f32);
    const float *d2 = mlx_array_data_float32(out2_f32);
    for (int i = 0; i < 64; i++)
        assert(fabsf(d1[i] - d2[i]) < 1e-3f);

    mlx_array_free(out2_f32);
    mlx_array_free(out_f32);
    mlx_array_free(out2);
    mlx_array_free(x2);
    mlx_array_free(x2_f32);
    kvcache_free(&kv2);
    mlx_array_free(out);
    mlx_array_free(x);
    mlx_array_free(x_f32);
    kvcache_free(&kv);
}

/* ---- B1.5: fwd_swiglu ---- */

static void test_fwd_swiglu(void) {
    float xdata[192];
    for (int i = 0; i < 192; i++) xdata[i] = 0.01f * (float)(i + 1);
    int xshape[] = {1, 3, 64};
    mlx_array x_f32 = mlx_array_new_data(xdata, xshape, 3, MLX_FLOAT32);
    mlx_array x = mlx_array_new();
    MLXB_CHECK(mlx_astype(&x, x_f32, MLX_BFLOAT16, gpu));

    mlx_array out = mlx_array_new();
    assert(fwd_swiglu(&out, x, 0, &w_dense, &cfg_dense, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out)));
    assert(mlx_array_ndim(out) == 3);
    assert(mlx_array_dim(out, 0) == 1);
    assert(mlx_array_dim(out, 1) == 3);
    assert(mlx_array_dim(out, 2) == 64);
    assert(is_finite_f32(out, gpu));

    /* Cross-check vs manual: down(silu(gate(x)) * up(x)) */
    char name[256];
    weight_triplet_t gate_tri, up_tri, down_tri;

    weights_tensor_name(name, sizeof(name), &cfg_dense, 0, "mlp.gate_proj");
    assert(weights_get_triplet(&gate_tri, &w_dense, name) == 0);
    weights_tensor_name(name, sizeof(name), &cfg_dense, 0, "mlp.up_proj");
    assert(weights_get_triplet(&up_tri, &w_dense, name) == 0);
    weights_tensor_name(name, sizeof(name), &cfg_dense, 0, "mlp.down_proj");
    assert(weights_get_triplet(&down_tri, &w_dense, name) == 0);

    mlx_array g = mlx_array_new();
    mlx_array u = mlx_array_new();
    mlx_array gs = mlx_array_new();
    mlx_array silu_g = mlx_array_new();
    mlx_array gated = mlx_array_new();
    mlx_array ref = mlx_array_new();
    fwd_linear(&g, x, &gate_tri, &cfg_dense, gpu);
    fwd_linear(&u, x, &up_tri, &cfg_dense, gpu);
    MLXB_CHECK(mlx_sigmoid(&gs, g, gpu));
    MLXB_CHECK(mlx_multiply(&silu_g, g, gs, gpu));
    MLXB_CHECK(mlx_multiply(&gated, silu_g, u, gpu));
    fwd_linear(&ref, gated, &down_tri, &cfg_dense, gpu);

    mlx_array out_f32 = mlx_array_new();
    mlx_array ref_f32 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&out_f32, out, MLX_FLOAT32, gpu));
    MLXB_CHECK(mlx_astype(&ref_f32, ref, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(out_f32)));
    assert(MLXB_CHECK(mlx_array_eval(ref_f32)));
    const float *od = mlx_array_data_float32(out_f32);
    const float *rd = mlx_array_data_float32(ref_f32);
    size_t n = mlx_array_size(out_f32);
    for (size_t i = 0; i < n; i++)
        assert(fabsf(od[i] - rd[i]) < 1e-2f);

    mlx_array_free(ref_f32);
    mlx_array_free(out_f32);
    mlx_array_free(ref);
    mlx_array_free(gated);
    mlx_array_free(silu_g);
    mlx_array_free(gs);
    mlx_array_free(u);
    mlx_array_free(g);
    weights_triplet_free(&down_tri);
    weights_triplet_free(&up_tri);
    weights_triplet_free(&gate_tri);
    mlx_array_free(out);
    mlx_array_free(x);
    mlx_array_free(x_f32);
}

/* ---- B1.6: decoder layer + lm_head ---- */

static void test_decoder_layer(void) {
    float xdata[192];
    for (int i = 0; i < 192; i++) xdata[i] = 0.01f * (float)(i + 1);
    int xshape[] = {1, 3, 64};
    mlx_array x_f32 = mlx_array_new_data(xdata, xshape, 3, MLX_FLOAT32);
    mlx_array x = mlx_array_new();
    MLXB_CHECK(mlx_astype(&x, x_f32, MLX_BFLOAT16, gpu));

    kvcache_t kv;
    assert(kvcache_init(&kv, cfg_dense.num_hidden_layers,
                        cfg_dense.num_key_value_heads, cfg_dense.head_dim) == 0);

    mlx_array out = mlx_array_new();
    assert(fwd_decoder_layer(&out, x, 0, &w_dense, &cfg_dense, &kv, gpu) == 0);
    assert(MLXB_CHECK(mlx_array_eval(out)));
    assert(mlx_array_ndim(out) == 3);
    assert(mlx_array_dim(out, 0) == 1);
    assert(mlx_array_dim(out, 1) == 3);
    assert(mlx_array_dim(out, 2) == 64);
    assert(is_finite_f32(out, gpu));

    /* Final norm + lm_head -> logits [1,3,256] */
    char name[256], wname[270];
    weights_tensor_name(name, sizeof(name), &cfg_dense, -1, "norm");
    snprintf(wname, sizeof(wname), "%s.weight", name);
    mlx_array norm_w = mlx_array_new();
    assert(weights_get(&norm_w, &w_dense, wname) == 0);

    mlx_array normed = mlx_array_new();
    assert(fwd_rmsnorm(&normed, out, norm_w, cfg_dense.rms_norm_eps, false, gpu) == 0);

    /* lm_head (untied: separate lm_head weight) */
    if (!cfg_dense.tie_word_embeddings) {
        weight_triplet_t lm_tri;
        assert(weights_get_triplet(&lm_tri, &w_dense, "lm_head") == 0);
        mlx_array logits = mlx_array_new();
        assert(fwd_linear(&logits, normed, &lm_tri, &cfg_dense, gpu) == 0);
        assert(MLXB_CHECK(mlx_array_eval(logits)));
        assert(mlx_array_dim(logits, 2) == 256);
        assert(is_finite_f32(logits, gpu));
        mlx_array_free(logits);
        weights_triplet_free(&lm_tri);
    } else {
        /* Tied: transpose embed + matmul */
        weights_tensor_name(name, sizeof(name), &cfg_dense, -1, "embed_tokens");
        snprintf(wname, sizeof(wname), "%s.weight", name);
        mlx_array embed_w = mlx_array_new();
        assert(weights_get(&embed_w, &w_dense, wname) == 0);
        mlx_array embed_t = mlx_array_new();
        MLXB_CHECK(mlx_transpose(&embed_t, embed_w, gpu));
        mlx_array logits = mlx_array_new();
        MLXB_CHECK(mlx_matmul(&logits, normed, embed_t, gpu));
        assert(MLXB_CHECK(mlx_array_eval(logits)));
        assert(mlx_array_dim(logits, 2) == 256);
        assert(is_finite_f32(logits, gpu));
        mlx_array_free(logits);
        mlx_array_free(embed_t);
        mlx_array_free(embed_w);
    }

    mlx_array_free(normed);
    mlx_array_free(norm_w);
    mlx_array_free(out);
    mlx_array_free(x);
    mlx_array_free(x_f32);
    kvcache_free(&kv);
}

/* ---- B1.7: emodel + model_forward ---- */

static void test_emodel(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tiny_qwen3", FIXTURES);

    engine_model_t em;
    char lerr[256] = {0};
    assert(engine_model_load(&em, path, lerr, sizeof(lerr)) == 0);

    int32_t ids_data[] = {1, 2, 3, 4, 5};
    int ids_shape[] = {1, 5};
    mlx_array ids = mlx_array_new_data(ids_data, ids_shape, 2, MLX_INT32);

    kvcache_t kv;
    assert(kvcache_init(&kv, em.cfg.num_hidden_layers,
                        em.cfg.num_key_value_heads, em.cfg.head_dim) == 0);

    /* want_logits=true -> logits [1,256] */
    mlx_array logits = mlx_array_new();
    assert(model_forward(&em, ids, &kv, true, &logits) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits)));
    assert(mlx_array_ndim(logits) == 2);
    assert(mlx_array_dim(logits, 0) == 1);
    assert(mlx_array_dim(logits, 1) == 256);
    assert(is_finite_f32(logits, gpu));

    mlx_array_free(logits);
    kvcache_free(&kv);

    /* want_logits=false -> rc 0, offset advanced */
    kvcache_t kv2;
    assert(kvcache_init(&kv2, em.cfg.num_hidden_layers,
                        em.cfg.num_key_value_heads, em.cfg.head_dim) == 0);
    assert(model_forward(&em, ids, &kv2, false, NULL) == 0);
    assert(kvcache_layer_offset(&kv2, 0) == 5);

    kvcache_free(&kv2);
    mlx_array_free(ids);
    engine_model_free(&em);
}

/* ---- B1.7b: emodel tied lm_head path ---- */

static void test_emodel_tied(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tiny_qwen3", FIXTURES);

    engine_model_t em;
    char lerr[256] = {0};
    assert(engine_model_load(&em, path, lerr, sizeof(lerr)) == 0);

    em.cfg.tie_word_embeddings = true;

    int32_t ids_data[] = {1, 2, 3, 4, 5};
    int ids_shape[] = {1, 5};
    mlx_array ids = mlx_array_new_data(ids_data, ids_shape, 2, MLX_INT32);

    kvcache_t kv;
    assert(kvcache_init(&kv, em.cfg.num_hidden_layers,
                        em.cfg.num_key_value_heads, em.cfg.head_dim) == 0);

    mlx_array logits = mlx_array_new();
    assert(model_forward(&em, ids, &kv, true, &logits) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits)));
    assert(mlx_array_ndim(logits) == 2);
    assert(mlx_array_dim(logits, 0) == 1);
    assert(mlx_array_dim(logits, 1) == em.cfg.vocab_size);
    assert(is_finite_f32(logits, gpu));

    mlx_array_free(logits);
    kvcache_free(&kv);
    mlx_array_free(ids);
    engine_model_free(&em);
}

/* ---- B1.7c: tied+quantized lm_head path ---- */

static void test_emodel_tied_quant(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tiny_qwen3_tied", FIXTURES);

    engine_model_t em;
    char lerr[256] = {0};
    int rc = engine_model_load(&em, path, lerr, sizeof(lerr));
    if (rc != 0) {
        fprintf(stderr, "load failed: %s\n", lerr);
        assert(0);
    }
    assert(em.cfg.tie_word_embeddings);

    int32_t ids_data[] = {1, 2, 3, 4, 5};
    int ids_shape[] = {1, 5};
    mlx_array ids = mlx_array_new_data(ids_data, ids_shape, 2, MLX_INT32);

    kvcache_t kv;
    assert(kvcache_init(&kv, em.cfg.num_hidden_layers,
                        em.cfg.num_key_value_heads, em.cfg.head_dim) == 0);

    mlx_array logits = mlx_array_new();
    assert(model_forward(&em, ids, &kv, true, &logits) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits)));
    assert(mlx_array_ndim(logits) == 2);
    assert(mlx_array_dim(logits, 0) == 1);
    assert(mlx_array_dim(logits, 1) == em.cfg.vocab_size);
    assert(is_finite_f32(logits, gpu));

    mlx_array_free(logits);
    kvcache_free(&kv);
    mlx_array_free(ids);
    engine_model_free(&em);
}

/* ---- B2.5: incremental == full-context anchor ---- */

static void test_incremental_equals_full(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tiny_qwen3", FIXTURES);

    engine_model_t em;
    char lerr[256] = {0};
    assert(engine_model_load(&em, path, lerr, sizeof(lerr)) == 0);

    int32_t prompt[] = {10, 20, 30, 40};

    /* Path A: full context, all 4 tokens at once */
    kvcache_t kv_a;
    assert(kvcache_init(&kv_a, em.cfg.num_hidden_layers,
                        em.cfg.num_key_value_heads, em.cfg.head_dim) == 0);

    int shape_all[] = {1, 4};
    mlx_array ids_all = mlx_array_new_data(prompt, shape_all, 2, MLX_INT32);
    mlx_array logits_a = mlx_array_new();
    assert(model_forward(&em, ids_all, &kv_a, true, &logits_a) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_a)));

    /* Path B: prefill 3, then decode 1 */
    kvcache_t kv_b;
    assert(kvcache_init(&kv_b, em.cfg.num_hidden_layers,
                        em.cfg.num_key_value_heads, em.cfg.head_dim) == 0);

    int shape_pre[] = {1, 3};
    mlx_array ids_pre = mlx_array_new_data(prompt, shape_pre, 2, MLX_INT32);
    assert(model_forward(&em, ids_pre, &kv_b, false, NULL) == 0);

    int shape_dec[] = {1, 1};
    mlx_array ids_dec = mlx_array_new_data(&prompt[3], shape_dec, 2, MLX_INT32);
    mlx_array logits_b = mlx_array_new();
    assert(model_forward(&em, ids_dec, &kv_b, true, &logits_b) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_b)));

    /* Compare: argmax should match */
    mlx_array la_f32 = mlx_array_new();
    mlx_array lb_f32 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&la_f32, logits_a, MLX_FLOAT32, gpu));
    MLXB_CHECK(mlx_astype(&lb_f32, logits_b, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(la_f32)));
    assert(MLXB_CHECK(mlx_array_eval(lb_f32)));

    const float *da = mlx_array_data_float32(la_f32);
    const float *db = mlx_array_data_float32(lb_f32);
    int vocab = em.cfg.vocab_size;

    int argmax_a = 0, argmax_b = 0;
    for (int i = 1; i < vocab; i++) {
        if (da[i] > da[argmax_a]) argmax_a = i;
        if (db[i] > db[argmax_b]) argmax_b = i;
    }
    assert(argmax_a == argmax_b);

    /* Max abs logit diff < 5e-2 (bf16 tolerance) */
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
    engine_model_free(&em);
}

/* ---- quantized model_forward coverage ---- */

static void test_emodel_quant(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tiny_qwen3_sharded", FIXTURES);

    engine_model_t em;
    char lerr[256] = {0};
    int rc = engine_model_load(&em, path, lerr, sizeof(lerr));
    if (rc != 0) {
        fprintf(stderr, "load failed: %s\n", lerr);
        assert(0);
    }

    int32_t ids_data[] = {1, 2, 3, 4, 5};
    int ids_shape[] = {1, 5};
    mlx_array ids = mlx_array_new_data(ids_data, ids_shape, 2, MLX_INT32);

    kvcache_t kv;
    assert(kvcache_init(&kv, em.cfg.num_hidden_layers,
                        em.cfg.num_key_value_heads, em.cfg.head_dim) == 0);

    mlx_array logits = mlx_array_new();
    assert(model_forward(&em, ids, &kv, true, &logits) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits)));
    assert(mlx_array_ndim(logits) == 2);
    assert(mlx_array_dim(logits, 0) == 1);
    assert(mlx_array_dim(logits, 1) == em.cfg.vocab_size);
    assert(is_finite_f32(logits, gpu));

    mlx_array_free(logits);
    kvcache_free(&kv);
    mlx_array_free(ids);
    engine_model_free(&em);
}

static void test_incremental_equals_full_quant(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tiny_qwen3_sharded", FIXTURES);

    engine_model_t em;
    char lerr[256] = {0};
    assert(engine_model_load(&em, path, lerr, sizeof(lerr)) == 0);

    int32_t prompt[] = {10, 20, 30, 40};

    kvcache_t kv_a;
    assert(kvcache_init(&kv_a, em.cfg.num_hidden_layers,
                        em.cfg.num_key_value_heads, em.cfg.head_dim) == 0);

    int shape_all[] = {1, 4};
    mlx_array ids_all = mlx_array_new_data(prompt, shape_all, 2, MLX_INT32);
    mlx_array logits_a = mlx_array_new();
    assert(model_forward(&em, ids_all, &kv_a, true, &logits_a) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_a)));

    kvcache_t kv_b;
    assert(kvcache_init(&kv_b, em.cfg.num_hidden_layers,
                        em.cfg.num_key_value_heads, em.cfg.head_dim) == 0);

    int shape_pre[] = {1, 3};
    mlx_array ids_pre = mlx_array_new_data(prompt, shape_pre, 2, MLX_INT32);
    assert(model_forward(&em, ids_pre, &kv_b, false, NULL) == 0);

    int shape_dec[] = {1, 1};
    mlx_array ids_dec = mlx_array_new_data(&prompt[3], shape_dec, 2, MLX_INT32);
    mlx_array logits_b = mlx_array_new();
    assert(model_forward(&em, ids_dec, &kv_b, true, &logits_b) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_b)));

    mlx_array la_f32 = mlx_array_new();
    mlx_array lb_f32 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&la_f32, logits_a, MLX_FLOAT32, gpu));
    MLXB_CHECK(mlx_astype(&lb_f32, logits_b, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(la_f32)));
    assert(MLXB_CHECK(mlx_array_eval(lb_f32)));

    const float *da = mlx_array_data_float32(la_f32);
    const float *db = mlx_array_data_float32(lb_f32);
    int vocab = em.cfg.vocab_size;

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
    assert(maxdiff < 0.1f);

    mlx_array_free(lb_f32);
    mlx_array_free(la_f32);
    mlx_array_free(logits_b);
    mlx_array_free(ids_dec);
    mlx_array_free(ids_pre);
    kvcache_free(&kv_b);
    mlx_array_free(logits_a);
    mlx_array_free(ids_all);
    kvcache_free(&kv_a);
    engine_model_free(&em);
}

/* ---- B2.6: chunked prefill (S>1 with nonempty cache) ---- */

static void test_chunked_prefill_equals_full(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tiny_qwen3", FIXTURES);

    engine_model_t em;
    char lerr[256] = {0};
    assert(engine_model_load(&em, path, lerr, sizeof(lerr)) == 0);

    int32_t prompt[] = {10, 20, 30, 40};

    /* Path A: full context, 4 tokens at once */
    kvcache_t kv_a;
    assert(kvcache_init(&kv_a, em.cfg.num_hidden_layers,
                        em.cfg.num_key_value_heads, em.cfg.head_dim) == 0);

    int shape_all[] = {1, 4};
    mlx_array ids_all = mlx_array_new_data(prompt, shape_all, 2, MLX_INT32);
    mlx_array logits_a = mlx_array_new();
    assert(model_forward(&em, ids_all, &kv_a, true, &logits_a) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_a)));

    /* Path B: chunk 1 = tokens 0-1 (S=2, empty cache), chunk 2 = tokens 2-3 (S=2, offset 2) */
    kvcache_t kv_b;
    assert(kvcache_init(&kv_b, em.cfg.num_hidden_layers,
                        em.cfg.num_key_value_heads, em.cfg.head_dim) == 0);

    int shape_c[] = {1, 2};
    mlx_array ids_c1 = mlx_array_new_data(prompt, shape_c, 2, MLX_INT32);
    assert(model_forward(&em, ids_c1, &kv_b, false, NULL) == 0);
    assert(kvcache_layer_offset(&kv_b, 0) == 2);

    mlx_array ids_c2 = mlx_array_new_data(&prompt[2], shape_c, 2, MLX_INT32);
    mlx_array logits_b = mlx_array_new();
    assert(model_forward(&em, ids_c2, &kv_b, true, &logits_b) == 0);
    assert(MLXB_CHECK(mlx_array_eval(logits_b)));

    /* Compare: argmax should match */
    mlx_array la_f32 = mlx_array_new();
    mlx_array lb_f32 = mlx_array_new();
    MLXB_CHECK(mlx_astype(&la_f32, logits_a, MLX_FLOAT32, gpu));
    MLXB_CHECK(mlx_astype(&lb_f32, logits_b, MLX_FLOAT32, gpu));
    assert(MLXB_CHECK(mlx_array_eval(la_f32)));
    assert(MLXB_CHECK(mlx_array_eval(lb_f32)));

    const float *da = mlx_array_data_float32(la_f32);
    const float *db = mlx_array_data_float32(lb_f32);
    int vocab = em.cfg.vocab_size;

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
    mlx_array_free(ids_c2);
    mlx_array_free(ids_c1);
    kvcache_free(&kv_b);
    mlx_array_free(logits_a);
    mlx_array_free(ids_all);
    kvcache_free(&kv_a);
    engine_model_free(&em);
}

int main(void) {
    gpu = mlxbridge_gpu_stream();

    /* Load dense fixture */
    char dense_path[512];
    snprintf(dense_path, sizeof(dense_path), "%s/tiny_qwen3", FIXTURES);
    assert(model_config_load(&cfg_dense, dense_path) == 0);
    char err[256];
    assert(weights_load(&w_dense, dense_path, &cfg_dense, err, sizeof(err)) == 0);

    /* Load sharded/quantized fixture */
    char quant_path[512];
    snprintf(quant_path, sizeof(quant_path), "%s/tiny_qwen3_sharded", FIXTURES);
    assert(model_config_load(&cfg_quant, quant_path) == 0);
    assert(weights_load(&w_quant, quant_path, &cfg_quant, err, sizeof(err)) == 0);

    test_fwd_linear();
    printf("  test_fwd_linear: passed\n");

    test_fwd_embed();
    printf("  test_fwd_embed: passed\n");

    test_fwd_rmsnorm();
    printf("  test_fwd_rmsnorm: passed\n");

    test_fwd_attention();
    printf("  test_fwd_attention: passed\n");

    test_fwd_swiglu();
    printf("  test_fwd_swiglu: passed\n");

    test_decoder_layer();
    printf("  test_decoder_layer: passed\n");

    test_emodel();
    printf("  test_emodel: passed\n");

    test_emodel_tied();
    printf("  test_emodel_tied: passed\n");

    test_emodel_tied_quant();
    printf("  test_emodel_tied_quant: passed\n");

    test_incremental_equals_full();
    printf("  test_incremental_equals_full: passed\n");

    test_emodel_quant();
    printf("  test_emodel_quant: passed\n");

    test_incremental_equals_full_quant();
    printf("  test_incremental_equals_full_quant: passed\n");

    test_chunked_prefill_equals_full();
    printf("  test_chunked_prefill_equals_full: passed\n");

    weights_free(&w_quant);
    model_config_free(&cfg_quant);
    weights_free(&w_dense);
    model_config_free(&cfg_dense);

    printf("test_forward_gpu: all passed\n");
    return 0;
}
