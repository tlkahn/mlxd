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

int main(void) {
    gpu = mlxbridge_gpu_stream();

    test_f1_attn_scale_from_config();
    printf("test_f1_attn_scale_from_config passed\n");

    test_f2_qkv_bias();
    printf("test_f2_qkv_bias passed\n");

    test_f3_rmsnorm_offset();
    printf("test_f3_rmsnorm_offset passed\n");

    printf("All forward D0 GPU tests passed\n");
    return 0;
}
