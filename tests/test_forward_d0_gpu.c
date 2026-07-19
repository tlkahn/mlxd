/* Stage D0 phase 2 forward-path feature tests (F1-F7c).
 * Synthetic in-memory weights/config; no on-disk fixtures. */

#include "engine/forward.h"
#include "engine/kvcache.h"
#include "mlxbridge/mlxbridge.h"
#include "model/model.h"
#include "model/weights.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static mlx_stream gpu;

enum { HID = 32, NH = 4, NKV = 2, HD = 8 };

/* Deterministic small values in [-0.5, 0.5). */
static float det_val(size_t i) {
    return (float)((i * 37 + 11) % 97) / 97.0f - 0.5f;
}

static mlx_array bf16_from_f32(const float *data, const int *shape, int nd) {
    mlx_array f = mlx_array_new_data(data, shape, nd, MLX_FLOAT32);
    mlx_array b = mlx_array_new();
    assert(MLXB_CHECK(mlx_astype(&b, f, MLX_BFLOAT16, gpu)));
    mlx_array_free(f);
    return b;
}

static mlx_array det_bf16(const int *shape, int nd, size_t seed) {
    size_t n = 1;
    for (int i = 0; i < nd; i++) n *= (size_t)shape[i];
    float *data = malloc(n * sizeof(float));
    assert(data);
    for (size_t i = 0; i < n; i++) data[i] = det_val(i + seed);
    mlx_array b = bf16_from_f32(data, shape, nd);
    free(data);
    return b;
}

/* Insert arr into the weights map; the map holds its own reference. */
static void put_tensor(weights_t *w, const char *name, mlx_array arr) {
    assert(mlx_map_string_to_array_insert(w->params, name, arr) == 0);
    mlx_array_free(arr);
    w->count++;
}

static void weights_begin(weights_t *w) {
    memset(w, 0, sizeof(*w));
    w->params = mlx_map_string_to_array_new();
}

/* Baseline synthetic config: qwen3-flavored, all D0 flags off. */
static model_config_t base_cfg(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.family = MODEL_QWEN3;
    cfg.weight_prefix = "";
    cfg.vocab_size = 16;
    cfg.hidden_size = HID;
    cfg.num_hidden_layers = 1;
    cfg.num_attention_heads = NH;
    cfg.num_key_value_heads = NKV;
    cfg.head_dim = HD;
    cfg.intermediate_size = 48;
    cfg.rms_norm_eps = 1e-6f;
    cfg.query_pre_attn_scalar = HD;
    cfg.rope_theta = 10000.0f;
    cfg.rope_scaling_factor = 1.0f;
    cfg.partial_rotary_factor = 1.0f;
    cfg.hidden_act = HIDDEN_ACT_SILU;
    return cfg;
}

/* One attention layer worth of synthetic weights (layer 0). */
static void put_attn_weights(weights_t *w) {
    int qw[] = {NH * HD, HID};
    int kw[] = {NKV * HD, HID};
    int ow[] = {HID, NH * HD};
    put_tensor(w, "layers.0.self_attn.q_proj.weight", det_bf16(qw, 2, 1));
    put_tensor(w, "layers.0.self_attn.k_proj.weight", det_bf16(kw, 2, 2));
    put_tensor(w, "layers.0.self_attn.v_proj.weight", det_bf16(kw, 2, 3));
    put_tensor(w, "layers.0.self_attn.o_proj.weight", det_bf16(ow, 2, 4));
}

static mlx_array det_input(int S, size_t seed) {
    int xs[] = {1, S, HID};
    return det_bf16(xs, 3, seed);
}

static float max_abs_diff(mlx_array a, mlx_array b) {
    mlx_array af = mlx_array_new();
    mlx_array bf = mlx_array_new();
    assert(MLXB_CHECK(mlx_astype(&af, a, MLX_FLOAT32, gpu)));
    assert(MLXB_CHECK(mlx_astype(&bf, b, MLX_FLOAT32, gpu)));
    assert(MLXB_CHECK(mlx_array_eval(af)));
    assert(MLXB_CHECK(mlx_array_eval(bf)));
    assert(mlx_array_size(af) == mlx_array_size(bf));
    const float *ad = mlx_array_data_float32(af);
    const float *bd = mlx_array_data_float32(bf);
    size_t n = mlx_array_size(af);
    float maxdiff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(ad[i] - bd[i]);
        if (diff > maxdiff) maxdiff = diff;
    }
    mlx_array_free(bf);
    mlx_array_free(af);
    return maxdiff;
}

/* Dense-only manual reference for one attention layer (layer 0), no qk-norm.
 * scale/base/rope_scale are explicit; mask_mode + optional mask array too.
 * with_bias adds layers.0.self_attn.{q,k,v}_proj.bias after each projection. */
static mlx_array manual_attention_ex(mlx_array x, const weights_t *w,
                                     float scale, float base, float rope_scale,
                                     int offset, const char *mask_mode,
                                     mlx_array mask_arr, bool with_bias) {
    int B = mlx_array_dim(x, 0);
    int S = mlx_array_dim(x, 1);

    mlx_array proj[3];
    const char *names[] = {"layers.0.self_attn.q_proj.weight",
                           "layers.0.self_attn.k_proj.weight",
                           "layers.0.self_attn.v_proj.weight"};
    const char *bias_names[] = {"layers.0.self_attn.q_proj.bias",
                                "layers.0.self_attn.k_proj.bias",
                                "layers.0.self_attn.v_proj.bias"};
    for (int i = 0; i < 3; i++) {
        mlx_array wt = mlx_array_new();
        mlx_array wtt = mlx_array_new();
        proj[i] = mlx_array_new();
        assert(weights_get(&wt, w, names[i]) == 0);
        assert(MLXB_CHECK(mlx_transpose(&wtt, wt, gpu)));
        assert(MLXB_CHECK(mlx_matmul(&proj[i], x, wtt, gpu)));
        if (with_bias) {
            mlx_array b = mlx_array_new();
            mlx_array biased = mlx_array_new();
            assert(weights_get(&b, w, bias_names[i]) == 0);
            assert(MLXB_CHECK(mlx_add(&biased, proj[i], b, gpu)));
            mlx_array_free(proj[i]);
            proj[i] = biased;
            mlx_array_free(b);
        }
        mlx_array_free(wtt);
        mlx_array_free(wt);
    }

    int q_shape[] = {B, S, NH, HD};
    int kv_shape[] = {B, S, NKV, HD};
    int perm[] = {0, 2, 1, 3};
    mlx_array qr = mlx_array_new(), kr = mlx_array_new(), vr = mlx_array_new();
    mlx_array qt = mlx_array_new(), kt = mlx_array_new(), vt = mlx_array_new();
    assert(MLXB_CHECK(mlx_reshape(&qr, proj[0], q_shape, 4, gpu)));
    assert(MLXB_CHECK(mlx_reshape(&kr, proj[1], kv_shape, 4, gpu)));
    assert(MLXB_CHECK(mlx_reshape(&vr, proj[2], kv_shape, 4, gpu)));
    assert(MLXB_CHECK(mlx_transpose_axes(&qt, qr, perm, 4, gpu)));
    assert(MLXB_CHECK(mlx_transpose_axes(&kt, kr, perm, 4, gpu)));
    assert(MLXB_CHECK(mlx_transpose_axes(&vt, vr, perm, 4, gpu)));

    mlx_array qrope = mlx_array_new(), krope = mlx_array_new();
    assert(MLXB_CHECK(mlx_fast_rope(&qrope, qt, HD, false,
        (mlx_optional_float){.value = base, .has_value = true},
        rope_scale, offset, (mlx_array){.ctx = NULL}, gpu)));
    assert(MLXB_CHECK(mlx_fast_rope(&krope, kt, HD, false,
        (mlx_optional_float){.value = base, .has_value = true},
        rope_scale, offset, (mlx_array){.ctx = NULL}, gpu)));

    mlx_array attn = mlx_array_new();
    assert(MLXB_CHECK(mlx_fast_scaled_dot_product_attention(
        &attn, qrope, krope, vt, scale, mask_mode,
        mask_arr, (mlx_array){.ctx = NULL}, gpu)));

    mlx_array back = mlx_array_new(), flat = mlx_array_new();
    int out_shape[] = {B, S, NH * HD};
    assert(MLXB_CHECK(mlx_transpose_axes(&back, attn, perm, 4, gpu)));
    assert(MLXB_CHECK(mlx_reshape(&flat, back, out_shape, 3, gpu)));

    mlx_array ow = mlx_array_new(), owt = mlx_array_new();
    mlx_array result = mlx_array_new();
    assert(weights_get(&ow, w, "layers.0.self_attn.o_proj.weight") == 0);
    assert(MLXB_CHECK(mlx_transpose(&owt, ow, gpu)));
    assert(MLXB_CHECK(mlx_matmul(&result, flat, owt, gpu)));

    mlx_array_free(owt);
    mlx_array_free(ow);
    mlx_array_free(flat);
    mlx_array_free(back);
    mlx_array_free(attn);
    mlx_array_free(krope);
    mlx_array_free(qrope);
    mlx_array_free(vt);
    mlx_array_free(kt);
    mlx_array_free(qt);
    mlx_array_free(vr);
    mlx_array_free(kr);
    mlx_array_free(qr);
    for (int i = 0; i < 3; i++) mlx_array_free(proj[i]);
    return result;
}

static mlx_array manual_attention(mlx_array x, const weights_t *w,
                                  float scale, float base, float rope_scale,
                                  int offset, const char *mask_mode,
                                  mlx_array mask_arr) {
    return manual_attention_ex(x, w, scale, base, rope_scale, offset,
                               mask_mode, mask_arr, false);
}

/* ---- F1: attention scale from config ---- */

static void test_f1_attn_scale_from_config(void) {
    weights_t w;
    weights_begin(&w);
    put_attn_weights(&w);

    mlx_array x = det_input(3, 100);

    /* query_pre_attn_scalar = 4*head_dim -> scale halves */
    model_config_t cfg = base_cfg();
    cfg.query_pre_attn_scalar = 4 * HD;

    kvcache_t kv;
    assert(kvcache_init(&kv, 1, NKV, HD) == 0);
    mlx_array out = mlx_array_new();
    assert(fwd_attention(&out, x, 0, &w, &cfg, &kv, gpu) == 0);

    mlx_array ref = manual_attention(x, &w, 1.0f / sqrtf(4.0f * (float)HD),
                                     cfg.rope_theta, 1.0f, 0, "causal",
                                     (mlx_array){.ctx = NULL});
    assert(max_abs_diff(out, ref) < 2e-3f);
    mlx_array_free(ref);
    mlx_array_free(out);
    kvcache_free(&kv);

    /* gemma4 family -> fixed scale 1.0 regardless of query_pre_attn_scalar */
    cfg.family = MODEL_GEMMA4;
    cfg.query_pre_attn_scalar = HD;
    assert(kvcache_init(&kv, 1, NKV, HD) == 0);
    out = mlx_array_new();
    assert(fwd_attention(&out, x, 0, &w, &cfg, &kv, gpu) == 0);
    ref = manual_attention(x, &w, 1.0f, cfg.rope_theta, 1.0f, 0, "causal",
                           (mlx_array){.ctx = NULL});
    assert(max_abs_diff(out, ref) < 2e-3f);
    mlx_array_free(ref);
    mlx_array_free(out);
    kvcache_free(&kv);

    /* baseline: query_pre_attn_scalar == head_dim keeps default behavior */
    cfg = base_cfg();
    assert(kvcache_init(&kv, 1, NKV, HD) == 0);
    out = mlx_array_new();
    assert(fwd_attention(&out, x, 0, &w, &cfg, &kv, gpu) == 0);
    ref = manual_attention(x, &w, 1.0f / sqrtf((float)HD), cfg.rope_theta,
                           1.0f, 0, "causal", (mlx_array){.ctx = NULL});
    assert(max_abs_diff(out, ref) < 2e-3f);
    mlx_array_free(ref);
    mlx_array_free(out);
    kvcache_free(&kv);

    mlx_array_free(x);
    weights_free(&w);
}

/* ---- F2: q/k/v projection bias ---- */

static void test_f2_qkv_bias(void) {
    weights_t w;
    weights_begin(&w);
    put_attn_weights(&w);
    int qb[] = {NH * HD};
    int kb[] = {NKV * HD};
    put_tensor(&w, "layers.0.self_attn.q_proj.bias", det_bf16(qb, 1, 5));
    put_tensor(&w, "layers.0.self_attn.k_proj.bias", det_bf16(kb, 1, 6));
    put_tensor(&w, "layers.0.self_attn.v_proj.bias", det_bf16(kb, 1, 7));

    mlx_array x = det_input(3, 200);
    float scale = 1.0f / sqrtf((float)HD);

    /* attention_bias on: output must follow the bias path */
    model_config_t cfg = base_cfg();
    cfg.attention_bias = true;

    kvcache_t kv;
    assert(kvcache_init(&kv, 1, NKV, HD) == 0);
    mlx_array out = mlx_array_new();
    assert(fwd_attention(&out, x, 0, &w, &cfg, &kv, gpu) == 0);
    mlx_array ref = manual_attention_ex(x, &w, scale, cfg.rope_theta, 1.0f, 0,
                                        "causal", (mlx_array){.ctx = NULL},
                                        true);
    assert(max_abs_diff(out, ref) < 2e-3f);

    /* sanity: bias path differs from the no-bias reference */
    mlx_array ref_nobias = manual_attention(x, &w, scale, cfg.rope_theta, 1.0f,
                                            0, "causal",
                                            (mlx_array){.ctx = NULL});
    assert(max_abs_diff(out, ref_nobias) > 1e-2f);
    mlx_array_free(ref_nobias);
    mlx_array_free(ref);
    mlx_array_free(out);
    kvcache_free(&kv);

    /* flag off: bias tensors present in the map are ignored */
    cfg.attention_bias = false;
    assert(kvcache_init(&kv, 1, NKV, HD) == 0);
    out = mlx_array_new();
    assert(fwd_attention(&out, x, 0, &w, &cfg, &kv, gpu) == 0);
    ref = manual_attention(x, &w, scale, cfg.rope_theta, 1.0f, 0, "causal",
                           (mlx_array){.ctx = NULL});
    assert(max_abs_diff(out, ref) < 2e-3f);
    mlx_array_free(ref);
    mlx_array_free(out);
    kvcache_free(&kv);

    mlx_array_free(x);
    weights_free(&w);
}

/* ---- F3: rmsnorm (1 + weight) offset ---- */

static void test_f3_rmsnorm_offset(void) {
    int xs[] = {1, 3, HID};
    int ws[] = {HID};
    mlx_array x = det_bf16(xs, 3, 300);
    mlx_array wgt = det_bf16(ws, 1, 301);
    float eps = 1e-6f;

    /* offset on: equals fast_rms_norm(x, bf16(1) + w) */
    mlx_array out = mlx_array_new();
    assert(fwd_rmsnorm(&out, x, wgt, eps, true, gpu) == 0);

    mlx_array one_f32 = mlx_array_new_float(1.0f);
    mlx_array one = mlx_array_new();
    mlx_array weff = mlx_array_new();
    mlx_array ref = mlx_array_new();
    assert(MLXB_CHECK(mlx_astype(&one, one_f32, MLX_BFLOAT16, gpu)));
    assert(MLXB_CHECK(mlx_add(&weff, one, wgt, gpu)));
    assert(MLXB_CHECK(mlx_fast_rms_norm(&ref, x, weff, eps, gpu)));
    assert(max_abs_diff(out, ref) < 1e-3f);

    /* differs from the no-offset result */
    mlx_array plain = mlx_array_new();
    assert(MLXB_CHECK(mlx_fast_rms_norm(&plain, x, wgt, eps, gpu)));
    assert(max_abs_diff(out, plain) > 1e-2f);

    /* offset off: unchanged plain fast_rms_norm */
    mlx_array out_plain = mlx_array_new();
    assert(fwd_rmsnorm(&out_plain, x, wgt, eps, false, gpu) == 0);
    assert(max_abs_diff(out_plain, plain) < 1e-6f);

    mlx_array_free(out_plain);
    mlx_array_free(plain);
    mlx_array_free(ref);
    mlx_array_free(weff);
    mlx_array_free(one);
    mlx_array_free(one_f32);
    mlx_array_free(out);
    mlx_array_free(wgt);
    mlx_array_free(x);
}

/* ---- F4: embed scaling by bf16-rounded sqrt(hidden_size) ---- */

/* bf16-rounded sqrt(hidden_size) scalar, matching the reference semantics:
   f32 sqrt then astype bf16. */
static mlx_array embed_scale_bf16(void) {
    mlx_array f = mlx_array_new_float(sqrtf((float)HID));
    mlx_array b = mlx_array_new();
    assert(MLXB_CHECK(mlx_astype(&b, f, MLX_BFLOAT16, gpu)));
    mlx_array_free(f);
    return b;
}

static void test_f4_embed_scaling(void) {
    enum { VOCAB = 16 };
    int32_t ids_data[] = {0, 5, 11};
    int ids_shape[] = {1, 3};
    mlx_array ids = mlx_array_new_data(ids_data, ids_shape, 2, MLX_INT32);

    /* --- dense --- */
    weights_t w;
    weights_begin(&w);
    int es[] = {VOCAB, HID};
    put_tensor(&w, "embed_tokens.weight", det_bf16(es, 2, 400));

    model_config_t cfg = base_cfg();

    /* flag off: unchanged plain gather */
    mlx_array out_plain = mlx_array_new();
    assert(fwd_embed(&out_plain, ids, &w, &cfg, gpu) == 0);

    mlx_array embed_w = mlx_array_new();
    mlx_array flat = mlx_array_new();
    mlx_array taken = mlx_array_new();
    mlx_array ref_plain = mlx_array_new();
    int flat_shape[] = {3};
    int out_shape[] = {1, 3, HID};
    assert(weights_get(&embed_w, &w, "embed_tokens.weight") == 0);
    assert(MLXB_CHECK(mlx_reshape(&flat, ids, flat_shape, 1, gpu)));
    assert(MLXB_CHECK(mlx_take_axis(&taken, embed_w, flat, 0, gpu)));
    assert(MLXB_CHECK(mlx_reshape(&ref_plain, taken, out_shape, 3, gpu)));
    assert(max_abs_diff(out_plain, ref_plain) < 1e-6f);

    /* flag on: multiplied by bf16-rounded sqrt(hidden_size) */
    cfg.scale_embeddings = true;
    mlx_array out_scaled = mlx_array_new();
    assert(fwd_embed(&out_scaled, ids, &w, &cfg, gpu) == 0);

    mlx_array scale = embed_scale_bf16();
    mlx_array ref_scaled = mlx_array_new();
    assert(MLXB_CHECK(mlx_multiply(&ref_scaled, ref_plain, scale, gpu)));
    assert(max_abs_diff(out_scaled, ref_scaled) < 1e-3f);
    assert(max_abs_diff(out_scaled, ref_plain) > 1e-2f);

    mlx_array_free(ref_scaled);
    mlx_array_free(out_scaled);
    mlx_array_free(ref_plain);
    mlx_array_free(taken);
    mlx_array_free(flat);
    mlx_array_free(embed_w);
    mlx_array_free(out_plain);
    weights_free(&w);

    /* --- quantized --- */
    weights_t wq;
    weights_begin(&wq);
    model_config_t qcfg = base_cfg();
    qcfg.quant_bits = 4;
    qcfg.quant_group_size = 32;
    qcfg.scale_embeddings = true;

    mlx_array ew = det_bf16(es, 2, 401);
    mlx_vector_array quant = mlx_vector_array_new();
    assert(MLXB_CHECK(mlx_quantize(&quant, ew,
        (mlx_optional_int){.value = qcfg.quant_group_size, .has_value = true},
        (mlx_optional_int){.value = qcfg.quant_bits, .has_value = true},
        "affine", (mlx_array){.ctx = NULL}, gpu)));
    mlx_array qw = mlx_array_new(), qs = mlx_array_new(), qb = mlx_array_new();
    assert(MLXB_CHECK(mlx_vector_array_get(&qw, quant, 0)));
    assert(MLXB_CHECK(mlx_vector_array_get(&qs, quant, 1)));
    assert(MLXB_CHECK(mlx_vector_array_get(&qb, quant, 2)));
    put_tensor(&wq, "embed_tokens.weight", qw);
    put_tensor(&wq, "embed_tokens.scales", qs);
    put_tensor(&wq, "embed_tokens.biases", qb);
    mlx_vector_array_free(quant);
    mlx_array_free(ew);

    mlx_array out_q = mlx_array_new();
    assert(fwd_embed(&out_q, ids, &wq, &qcfg, gpu) == 0);

    /* reference: dequantized gather, then the same bf16 scalar multiply */
    weight_triplet_t tri;
    assert(weights_get_triplet(&tri, &wq, "embed_tokens") == 0);
    assert(tri.quantized);
    mlx_array dqw = mlx_array_new(), dqs = mlx_array_new(),
              dqb = mlx_array_new(), dq = mlx_array_new();
    mlx_array flatq = mlx_array_new();
    assert(MLXB_CHECK(mlx_reshape(&flatq, ids, flat_shape, 1, gpu)));
    assert(MLXB_CHECK(mlx_take_axis(&dqw, tri.weight, flatq, 0, gpu)));
    assert(MLXB_CHECK(mlx_take_axis(&dqs, tri.scales, flatq, 0, gpu)));
    assert(MLXB_CHECK(mlx_take_axis(&dqb, tri.biases, flatq, 0, gpu)));
    assert(MLXB_CHECK(mlx_dequantize(&dq, dqw, dqs, dqb,
        (mlx_optional_int){.value = qcfg.quant_group_size, .has_value = true},
        (mlx_optional_int){.value = qcfg.quant_bits, .has_value = true},
        "affine", (mlx_array){.ctx = NULL},
        (mlx_optional_dtype){.has_value = false}, gpu)));
    mlx_array dq_shaped = mlx_array_new(), dq_bf16 = mlx_array_new();
    mlx_array ref_q = mlx_array_new();
    assert(MLXB_CHECK(mlx_reshape(&dq_shaped, dq, out_shape, 3, gpu)));
    assert(MLXB_CHECK(mlx_astype(&dq_bf16, dq_shaped, MLX_BFLOAT16, gpu)));
    assert(MLXB_CHECK(mlx_multiply(&ref_q, dq_bf16, scale, gpu)));
    assert(max_abs_diff(out_q, ref_q) < 1e-3f);

    mlx_array_free(ref_q);
    mlx_array_free(dq_bf16);
    mlx_array_free(dq_shaped);
    mlx_array_free(dq);
    mlx_array_free(dqb);
    mlx_array_free(dqs);
    mlx_array_free(dqw);
    mlx_array_free(flatq);
    weights_triplet_free(&tri);
    mlx_array_free(out_q);
    mlx_array_free(scale);
    weights_free(&wq);
    mlx_array_free(ids);
}

/* ---- F5: sandwich decoder wiring (has_pre_ff_norm) ---- */

enum { FF = 48 };

static void put_mlp_weights(weights_t *w) {
    int gu[] = {FF, HID};
    int dn[] = {HID, FF};
    put_tensor(w, "layers.0.mlp.gate_proj.weight", det_bf16(gu, 2, 11));
    put_tensor(w, "layers.0.mlp.up_proj.weight", det_bf16(gu, 2, 12));
    put_tensor(w, "layers.0.mlp.down_proj.weight", det_bf16(dn, 2, 13));
}

/* Norm weights scaled down so det_val's [-0.5,0.5) range stays well-behaved. */
static void put_norm(weights_t *w, const char *name, size_t seed) {
    int ns[] = {HID};
    put_tensor(w, name, det_bf16(ns, 1, seed));
}

static mlx_array get_w(const weights_t *w, const char *name) {
    mlx_array a = mlx_array_new();
    assert(weights_get(&a, w, name) == 0);
    return a;
}

static void test_f5_sandwich_decoder(void) {
    weights_t w;
    weights_begin(&w);
    put_attn_weights(&w);
    put_mlp_weights(&w);
    put_norm(&w, "layers.0.input_layernorm.weight", 21);
    put_norm(&w, "layers.0.post_attention_layernorm.weight", 22);
    put_norm(&w, "layers.0.pre_feedforward_layernorm.weight", 23);
    put_norm(&w, "layers.0.post_feedforward_layernorm.weight", 24);

    mlx_array x = det_input(3, 500);
    model_config_t cfg = base_cfg();
    cfg.has_pre_ff_norm = true;

    kvcache_t kv;
    assert(kvcache_init(&kv, 1, NKV, HD) == 0);
    mlx_array out = mlx_array_new();
    assert(fwd_decoder_layer(&out, x, 0, &w, &cfg, &kv, gpu) == 0);
    kvcache_free(&kv);

    /* manual sandwich composition from tested primitives */
    mlx_array ln_in = get_w(&w, "layers.0.input_layernorm.weight");
    mlx_array ln_pa = get_w(&w, "layers.0.post_attention_layernorm.weight");
    mlx_array ln_pre = get_w(&w, "layers.0.pre_feedforward_layernorm.weight");
    mlx_array ln_post = get_w(&w, "layers.0.post_feedforward_layernorm.weight");

    kvcache_t kv2;
    assert(kvcache_init(&kv2, 1, NKV, HD) == 0);
    mlx_array normed1 = mlx_array_new(), attn_out = mlx_array_new();
    mlx_array attn_normed = mlx_array_new(), h = mlx_array_new();
    mlx_array mlp_in = mlx_array_new(), mlp_out = mlx_array_new();
    mlx_array mlp_normed = mlx_array_new(), ref = mlx_array_new();
    float eps = cfg.rms_norm_eps;
    assert(fwd_rmsnorm(&normed1, x, ln_in, eps, false, gpu) == 0);
    assert(fwd_attention(&attn_out, normed1, 0, &w, &cfg, &kv2, gpu) == 0);
    assert(fwd_rmsnorm(&attn_normed, attn_out, ln_pa, eps, false, gpu) == 0);
    assert(MLXB_CHECK(mlx_add(&h, x, attn_normed, gpu)));
    assert(fwd_rmsnorm(&mlp_in, h, ln_pre, eps, false, gpu) == 0);
    assert(fwd_swiglu(&mlp_out, mlp_in, 0, &w, &cfg, gpu) == 0);
    assert(fwd_rmsnorm(&mlp_normed, mlp_out, ln_post, eps, false, gpu) == 0);
    assert(MLXB_CHECK(mlx_add(&ref, h, mlp_normed, gpu)));
    kvcache_free(&kv2);

    assert(max_abs_diff(out, ref) < 2e-3f);
    mlx_array_free(out);

    /* flag off: default wiring untouched */
    cfg.has_pre_ff_norm = false;
    assert(kvcache_init(&kv, 1, NKV, HD) == 0);
    out = mlx_array_new();
    assert(fwd_decoder_layer(&out, x, 0, &w, &cfg, &kv, gpu) == 0);
    kvcache_free(&kv);

    kvcache_t kv3;
    assert(kvcache_init(&kv3, 1, NKV, HD) == 0);
    mlx_array h1 = mlx_array_new(), normed2 = mlx_array_new();
    mlx_array mlp_std = mlx_array_new(), ref_std = mlx_array_new();
    mlx_array attn_std = mlx_array_new();
    assert(fwd_attention(&attn_std, normed1, 0, &w, &cfg, &kv3, gpu) == 0);
    assert(MLXB_CHECK(mlx_add(&h1, x, attn_std, gpu)));
    assert(fwd_rmsnorm(&normed2, h1, ln_pa, eps, false, gpu) == 0);
    assert(fwd_swiglu(&mlp_std, normed2, 0, &w, &cfg, gpu) == 0);
    assert(MLXB_CHECK(mlx_add(&ref_std, h1, mlp_std, gpu)));
    kvcache_free(&kv3);

    assert(max_abs_diff(out, ref_std) < 2e-3f);
    /* and the two wirings genuinely differ */
    assert(max_abs_diff(ref, ref_std) > 1e-2f);

    mlx_array_free(ref_std);
    mlx_array_free(mlp_std);
    mlx_array_free(normed2);
    mlx_array_free(h1);
    mlx_array_free(attn_std);
    mlx_array_free(out);
    mlx_array_free(ref);
    mlx_array_free(mlp_normed);
    mlx_array_free(mlp_out);
    mlx_array_free(mlp_in);
    mlx_array_free(h);
    mlx_array_free(attn_normed);
    mlx_array_free(attn_out);
    mlx_array_free(normed1);
    mlx_array_free(ln_post);
    mlx_array_free(ln_pre);
    mlx_array_free(ln_pa);
    mlx_array_free(ln_in);
    mlx_array_free(x);
    weights_free(&w);
}

/* ---- F6: per-layer rope (local base freq, linear scaling factor) ---- */

static void test_f6_per_layer_rope(void) {
    weights_t w;
    weights_begin(&w);
    put_attn_weights(&w);

    mlx_array x = det_input(3, 600);
    float scale = 1.0f / sqrtf((float)HD);

    /* (a) local layer uses rope_local_base_freq: layer 0 local under
       pattern 6; rope_theta deliberately bogus to prove it is unused */
    model_config_t cfg = base_cfg();
    cfg.has_sliding_window = true;
    cfg.sliding_window = 1024; /* wider than the sequence: mask is a no-op */
    cfg.sliding_window_pattern = 6;
    cfg.rope_theta = 999999.0f;
    cfg.rope_local_base_freq = 5000.0f;
    assert(!model_layer_is_global(&cfg, 0));

    kvcache_t kv;
    assert(kvcache_init(&kv, 1, NKV, HD) == 0);
    mlx_array out = mlx_array_new();
    assert(fwd_attention(&out, x, 0, &w, &cfg, &kv, gpu) == 0);
    mlx_array ref = manual_attention(x, &w, scale, 5000.0f, 1.0f, 0, "causal",
                                     (mlx_array){.ctx = NULL});
    assert(max_abs_diff(out, ref) < 2e-3f);

    /* sanity: differs from rope over the bogus global base */
    mlx_array ref_bogus = manual_attention(x, &w, scale, 999999.0f, 1.0f, 0,
                                           "causal", (mlx_array){.ctx = NULL});
    assert(max_abs_diff(out, ref_bogus) > 1e-3f);
    mlx_array_free(ref_bogus);
    mlx_array_free(ref);
    mlx_array_free(out);
    kvcache_free(&kv);

    /* (b) global layer with linear rope scaling factor f -> rope scale 1/f */
    cfg = base_cfg();
    cfg.rope_scaling_type = "linear";
    cfg.rope_scaling_factor = 4.0f;
    assert(model_layer_is_global(&cfg, 0));

    assert(kvcache_init(&kv, 1, NKV, HD) == 0);
    out = mlx_array_new();
    assert(fwd_attention(&out, x, 0, &w, &cfg, &kv, gpu) == 0);
    ref = manual_attention(x, &w, scale, cfg.rope_theta, 0.25f, 0, "causal",
                           (mlx_array){.ctx = NULL});
    assert(max_abs_diff(out, ref) < 2e-3f);

    /* decode step keeps absolute positions (offset via kvcache) */
    mlx_array x1 = det_input(1, 601);
    mlx_array out1 = mlx_array_new();
    assert(fwd_attention(&out1, x1, 0, &w, &cfg, &kv, gpu) == 0);
    assert(mlx_array_dim(out1, 1) == 1);
    mlx_array_free(out1);
    mlx_array_free(x1);

    mlx_array_free(ref);
    mlx_array_free(out);
    kvcache_free(&kv);
    mlx_array_free(x);
    weights_free(&w);
}

/* ---- F7a: sliding-window mask helpers ---- */

/* Pull mask values as f32; returns malloc'd buffer of mlx_array_size floats. */
static float *mask_values(mlx_array m, size_t *n) {
    mlx_array f = mlx_array_new();
    assert(MLXB_CHECK(mlx_astype(&f, m, MLX_FLOAT32, gpu)));
    assert(MLXB_CHECK(mlx_array_eval(f)));
    *n = mlx_array_size(f);
    float *buf = malloc(*n * sizeof(float));
    assert(buf);
    memcpy(buf, mlx_array_data_float32(f), *n * sizeof(float));
    mlx_array_free(f);
    return buf;
}

static void test_f7a_sliding_window_masks(void) {
    /* prefill: q_len=3, kv_len=5, window=2
       abs row of query i is kv_len - q_len + i; allowed cols are
       (abs_row - window, abs_row], i.e. the last `window` positions. */
    mlx_array m = mlx_array_new();
    assert(fwd_sliding_window_mask(&m, 3, 5, 2, gpu) == 0);
    assert(mlx_array_ndim(m) == 4);
    assert(mlx_array_dim(m, 0) == 1 && mlx_array_dim(m, 1) == 1);
    assert(mlx_array_dim(m, 2) == 3 && mlx_array_dim(m, 3) == 5);
    assert(mlx_array_dtype(m) == MLX_BFLOAT16);

    size_t n = 0;
    float *v = mask_values(m, &n);
    assert(n == 15);
    const int allowed[3][5] = {
        {0, 1, 1, 0, 0}, /* abs row 2: cols 1,2 */
        {0, 0, 1, 1, 0}, /* abs row 3: cols 2,3 */
        {0, 0, 0, 1, 1}, /* abs row 4: cols 3,4 */
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 5; j++) {
            float val = v[i * 5 + j];
            if (allowed[i][j])
                assert(val == 0.0f);
            else
                assert(isinf(val) && val < 0.0f);
        }
    }
    free(v);
    mlx_array_free(m);

    /* decode: kv_len=5, window=2 -> cols before kv_len-window masked */
    m = mlx_array_new();
    assert(fwd_sliding_window_decode_mask(&m, 5, 2, gpu) == 0);
    assert(mlx_array_ndim(m) == 4);
    assert(mlx_array_dim(m, 0) == 1 && mlx_array_dim(m, 1) == 1);
    assert(mlx_array_dim(m, 2) == 1 && mlx_array_dim(m, 3) == 5);
    assert(mlx_array_dtype(m) == MLX_BFLOAT16);

    v = mask_values(m, &n);
    assert(n == 5);
    for (int j = 0; j < 5; j++) {
        if (j < 3)
            assert(isinf(v[j]) && v[j] < 0.0f);
        else
            assert(v[j] == 0.0f);
    }
    free(v);
    mlx_array_free(m);

    /* decode within window: no position masked */
    m = mlx_array_new();
    assert(fwd_sliding_window_decode_mask(&m, 2, 4, gpu) == 0);
    v = mask_values(m, &n);
    assert(n == 2);
    for (int j = 0; j < 2; j++) assert(v[j] == 0.0f);
    free(v);
    mlx_array_free(m);
}

int main(void) {
    gpu = mlxbridge_gpu_stream();

    test_f1_attn_scale_from_config();
    printf("test_f1_attn_scale_from_config passed\n");

    test_f2_qkv_bias();
    printf("test_f2_qkv_bias passed\n");

    test_f3_rmsnorm_offset();
    printf("test_f3_rmsnorm_offset passed\n");

    test_f4_embed_scaling();
    printf("test_f4_embed_scaling passed\n");

    test_f5_sandwich_decoder();
    printf("test_f5_sandwich_decoder passed\n");

    test_f6_per_layer_rope();
    printf("test_f6_per_layer_rope passed\n");

    test_f7a_sliding_window_masks();
    printf("test_f7a_sliding_window_masks passed\n");

    printf("All forward D0 GPU tests passed\n");
    return 0;
}
