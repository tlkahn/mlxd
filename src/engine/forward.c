#include "engine/forward.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

int fwd_linear(mlx_array *out, mlx_array x, const weight_triplet_t *tri,
               const model_config_t *cfg, mlx_stream s) {
    if (!out || !tri) return -1;
    int rc = -1;
    mlx_array result = mlx_array_new();

    if (tri->quantized) {
        mlx_array xc = mlx_array_new();
        if (!MLXB_CHECK(mlx_contiguous(&xc, x, false, s))) {
            mlx_array_free(xc);
            goto cleanup;
        }
        if (!MLXB_CHECK(mlx_quantized_matmul(
                &result, xc, tri->weight, tri->scales, tri->biases,
                true,
                (mlx_optional_int){.value = cfg->quant_group_size, .has_value = true},
                (mlx_optional_int){.value = cfg->quant_bits, .has_value = true},
                "affine", s))) {
            mlx_array_free(xc);
            goto cleanup;
        }
        mlx_array_free(xc);
    } else {
        mlx_array wt = mlx_array_new();
        if (!MLXB_CHECK(mlx_transpose(&wt, tri->weight, s))) {
            mlx_array_free(wt);
            goto cleanup;
        }
        if (!MLXB_CHECK(mlx_matmul(&result, x, wt, s))) {
            mlx_array_free(wt);
            goto cleanup;
        }
        mlx_array_free(wt);
    }

    if (mlx_array_dtype(result) != MLX_BFLOAT16) {
        mlx_array casted = mlx_array_new();
        if (!MLXB_CHECK(mlx_astype(&casted, result, MLX_BFLOAT16, s))) {
            mlx_array_free(casted);
            goto cleanup;
        }
        mlx_array_free(result);
        result = casted;
    }

    mlx_array_free(*out);
    *out = result;
    return 0;

cleanup:
    mlx_array_free(result);
    return rc;
}

int fwd_embed(mlx_array *out, mlx_array ids, const weights_t *w,
              const model_config_t *cfg, mlx_stream s) {
    if (!out || !w || !cfg) return -1;
    int rc = -1;

    char name[256];
    if (weights_tensor_name(name, sizeof(name), cfg, -1, "embed_tokens") != 0)
        return -1;

    mlx_array flat = mlx_array_new();
    mlx_array taken = mlx_array_new();
    mlx_array result = mlx_array_new();
    mlx_array dq = mlx_array_new();
    mlx_array dq_weight = mlx_array_new();
    mlx_array dq_scales = mlx_array_new();
    mlx_array dq_biases = mlx_array_new();
    mlx_array embed_w = mlx_array_new();
    mlx_array bf16 = mlx_array_new();

    int ndim = (int)mlx_array_ndim(ids);
    int B = ndim >= 2 ? mlx_array_dim(ids, 0) : 1;
    int S = ndim >= 2 ? mlx_array_dim(ids, 1) : mlx_array_dim(ids, 0);
    int total = B * S;
    int flat_shape[] = {total};

    if (!MLXB_CHECK(mlx_reshape(&flat, ids, flat_shape, 1, s))) goto cleanup;

    weight_triplet_t tri;
    if (weights_get_triplet(&tri, w, name) != 0) goto cleanup;

    if (tri.quantized) {
        if (!MLXB_CHECK(mlx_take_axis(&dq_weight, tri.weight, flat, 0, s)) ||
            !MLXB_CHECK(mlx_take_axis(&dq_scales, tri.scales, flat, 0, s)) ||
            !MLXB_CHECK(mlx_take_axis(&dq_biases, tri.biases, flat, 0, s)) ||
            !MLXB_CHECK(mlx_dequantize(
                &dq, dq_weight, dq_scales, dq_biases,
                (mlx_optional_int){.value = cfg->quant_group_size, .has_value = true},
                (mlx_optional_int){.value = cfg->quant_bits, .has_value = true},
                "affine",
                (mlx_array){.ctx = NULL},
                (mlx_optional_dtype){.has_value = false},
                s))) {
            weights_triplet_free(&tri);
            goto cleanup;
        }

        int out_shape[] = {B, S, cfg->hidden_size};
        mlx_array reshaped = mlx_array_new();
        if (!MLXB_CHECK(mlx_reshape(&reshaped, dq, out_shape, 3, s))) {
            mlx_array_free(reshaped);
            weights_triplet_free(&tri);
            goto cleanup;
        }
        if (!MLXB_CHECK(mlx_astype(&result, reshaped, MLX_BFLOAT16, s))) {
            mlx_array_free(reshaped);
            weights_triplet_free(&tri);
            goto cleanup;
        }
        mlx_array_free(reshaped);
    } else {
        char wname[270];
        snprintf(wname, sizeof(wname), "%s.weight", name);
        if (weights_get(&embed_w, w, wname) != 0) {
            weights_triplet_free(&tri);
            goto cleanup;
        }

        if (!MLXB_CHECK(mlx_take_axis(&taken, embed_w, flat, 0, s))) {
            weights_triplet_free(&tri);
            goto cleanup;
        }

        if (!MLXB_CHECK(mlx_astype(&bf16, taken, MLX_BFLOAT16, s))) {
            weights_triplet_free(&tri);
            goto cleanup;
        }

        int out_shape[] = {B, S, cfg->hidden_size};
        if (!MLXB_CHECK(mlx_reshape(&result, bf16, out_shape, 3, s))) {
            weights_triplet_free(&tri);
            goto cleanup;
        }
    }

    /* gemma-style embed scaling: bf16-rounded sqrt(hidden_size), bf16 x bf16 */
    if (cfg->scale_embeddings) {
        mlx_array scale_f32 = mlx_array_new_float(sqrtf((float)cfg->hidden_size));
        mlx_array scale = mlx_array_new();
        mlx_array scaled = mlx_array_new();
        bool ok = MLXB_CHECK(mlx_astype(&scale, scale_f32, MLX_BFLOAT16, s)) &&
                  MLXB_CHECK(mlx_multiply(&scaled, result, scale, s));
        mlx_array_free(scale);
        mlx_array_free(scale_f32);
        if (!ok) {
            mlx_array_free(scaled);
            weights_triplet_free(&tri);
            goto cleanup;
        }
        mlx_array_free(result);
        result = scaled;
    }

    weights_triplet_free(&tri);
    mlx_array_free(*out);
    *out = result;
    result = mlx_array_new();
    rc = 0;

cleanup:
    mlx_array_free(bf16);
    mlx_array_free(embed_w);
    mlx_array_free(dq_biases);
    mlx_array_free(dq_scales);
    mlx_array_free(dq_weight);
    mlx_array_free(dq);
    mlx_array_free(result);
    mlx_array_free(taken);
    mlx_array_free(flat);
    return rc;
}

int fwd_rmsnorm(mlx_array *out, mlx_array x, mlx_array weight, float eps,
                bool add_unit_offset, mlx_stream s) {
    if (!out) return -1;
    int rc = -1;
    mlx_array result = mlx_array_new();
    mlx_array w_eff = mlx_array_new();

    if (add_unit_offset) {
        mlx_array one_f32 = mlx_array_new_float(1.0f);
        mlx_array one = mlx_array_new();
        if (!MLXB_CHECK(mlx_astype(&one, one_f32, MLX_BFLOAT16, s)) ||
            !MLXB_CHECK(mlx_add(&w_eff, one, weight, s))) {
            mlx_array_free(one);
            mlx_array_free(one_f32);
            goto cleanup;
        }
        mlx_array_free(one);
        mlx_array_free(one_f32);
    } else {
        if (!MLXB_CHECK(mlx_array_set(&w_eff, weight))) goto cleanup;
    }

    if (!MLXB_CHECK(mlx_fast_rms_norm(&result, x, w_eff, eps, s)))
        goto cleanup;
    mlx_array_free(*out);
    *out = result;
    result = mlx_array_new();
    rc = 0;

cleanup:
    mlx_array_free(w_eff);
    mlx_array_free(result);
    return rc;
}

/* Cast an f32 0/-inf mask to bf16 and reshape to [1,1,rows,cols]. */
static int mask_finalize(mlx_array *out, mlx_array vals, int rows, int cols,
                         mlx_stream s) {
    int rc = -1;
    mlx_array bf16 = mlx_array_new();
    mlx_array result = mlx_array_new();
    int shape[] = {1, 1, rows, cols};
    if (!MLXB_CHECK(mlx_astype(&bf16, vals, MLX_BFLOAT16, s))) goto cleanup;
    if (!MLXB_CHECK(mlx_reshape(&result, bf16, shape, 4, s))) goto cleanup;
    mlx_array_free(*out);
    *out = result;
    result = mlx_array_new();
    rc = 0;

cleanup:
    mlx_array_free(result);
    mlx_array_free(bf16);
    return rc;
}

int fwd_sliding_window_mask(mlx_array *out, int q_len, int kv_len, int window,
                            mlx_stream s) {
    if (!out || q_len <= 0 || kv_len <= 0 || window <= 0 || q_len > kv_len) return -1;
    int rc = -1;
    mlx_array rows = mlx_array_new();
    mlx_array cols = mlx_array_new();
    mlx_array rows2 = mlx_array_new();
    mlx_array cols2 = mlx_array_new();
    mlx_array diff = mlx_array_new();
    mlx_array causal = mlx_array_new();
    mlx_array within = mlx_array_new();
    mlx_array allowed = mlx_array_new();
    mlx_array vals = mlx_array_new();
    mlx_array zero_i = mlx_array_new_int(0);
    mlx_array win_i = mlx_array_new_int(window);
    mlx_array zero_f = mlx_array_new_float(0.0f);
    mlx_array neginf_f = mlx_array_new_float(-INFINITY);
    int rshape[] = {q_len, 1};
    int cshape[] = {1, kv_len};

    /* absolute position of query row i is kv_len - q_len + i */
    if (!MLXB_CHECK(mlx_arange(&rows, (double)(kv_len - q_len), (double)kv_len,
                               1.0, MLX_INT32, s))) goto cleanup;
    if (!MLXB_CHECK(mlx_arange(&cols, 0.0, (double)kv_len, 1.0, MLX_INT32, s)))
        goto cleanup;
    if (!MLXB_CHECK(mlx_reshape(&rows2, rows, rshape, 2, s))) goto cleanup;
    if (!MLXB_CHECK(mlx_reshape(&cols2, cols, cshape, 2, s))) goto cleanup;
    if (!MLXB_CHECK(mlx_subtract(&diff, rows2, cols2, s))) goto cleanup;
    /* allowed iff 0 <= row - col < window */
    if (!MLXB_CHECK(mlx_greater_equal(&causal, diff, zero_i, s))) goto cleanup;
    if (!MLXB_CHECK(mlx_less(&within, diff, win_i, s))) goto cleanup;
    if (!MLXB_CHECK(mlx_logical_and(&allowed, causal, within, s))) goto cleanup;
    if (!MLXB_CHECK(mlx_where(&vals, allowed, zero_f, neginf_f, s)))
        goto cleanup;
    rc = mask_finalize(out, vals, q_len, kv_len, s);

cleanup:
    mlx_array_free(neginf_f);
    mlx_array_free(zero_f);
    mlx_array_free(win_i);
    mlx_array_free(zero_i);
    mlx_array_free(vals);
    mlx_array_free(allowed);
    mlx_array_free(within);
    mlx_array_free(causal);
    mlx_array_free(diff);
    mlx_array_free(cols2);
    mlx_array_free(rows2);
    mlx_array_free(cols);
    mlx_array_free(rows);
    return rc;
}

int fwd_sliding_window_decode_mask(mlx_array *out, int kv_len, int window,
                                   mlx_stream s) {
    if (!out || kv_len <= 0 || window <= 0) return -1;
    int rc = -1;
    mlx_array cols = mlx_array_new();
    mlx_array allowed = mlx_array_new();
    mlx_array vals = mlx_array_new();
    mlx_array cutoff_i = mlx_array_new_int(kv_len - window);
    mlx_array zero_f = mlx_array_new_float(0.0f);
    mlx_array neginf_f = mlx_array_new_float(-INFINITY);

    if (!MLXB_CHECK(mlx_arange(&cols, 0.0, (double)kv_len, 1.0, MLX_INT32, s)))
        goto cleanup;
    if (!MLXB_CHECK(mlx_greater_equal(&allowed, cols, cutoff_i, s)))
        goto cleanup;
    if (!MLXB_CHECK(mlx_where(&vals, allowed, zero_f, neginf_f, s)))
        goto cleanup;
    rc = mask_finalize(out, vals, 1, kv_len, s);

cleanup:
    mlx_array_free(neginf_f);
    mlx_array_free(zero_f);
    mlx_array_free(cutoff_i);
    mlx_array_free(vals);
    mlx_array_free(allowed);
    mlx_array_free(cols);
    return rc;
}

int fwd_rope_llama3_freqs(const model_config_t *cfg, float *out, int n) {
    if (!cfg || !out) return -1;
    if (!cfg->rope_scaling_type || strcmp(cfg->rope_scaling_type, "llama3") != 0)
        return -1;
    if (n != cfg->head_dim / 2) return -1;

    double base = (double)cfg->rope_theta;
    double factor = (double)cfg->rope_scaling_factor;
    double low_freq_factor = (double)cfg->rope_low_freq_factor;
    double high_freq_factor = (double)cfg->rope_high_freq_factor;
    double old_ctx = (double)cfg->rope_original_max_position_embeddings;
    int head_dim = cfg->head_dim;

    double low_wl = old_ctx / low_freq_factor;
    double high_wl = old_ctx / high_freq_factor;

    for (int i = 0; i < n; i++) {
        double freq = pow(base, 2.0 * i / (double)head_dim);
        double wavelen = 2.0 * M_PI * freq;

        if (wavelen > low_wl) {
            freq = freq * factor;
        } else if (wavelen > high_wl) {
            double smooth = (old_ctx / wavelen - low_freq_factor) /
                            (high_freq_factor - low_freq_factor);
            freq = freq / ((1.0 - smooth) / factor + smooth);
        }
        out[i] = (float)freq;
    }
    return 0;
}

/* Custom rope frequency seam: when a family supplies a freqs array the rope
   call switches to base has_value=false, scale 1.0. */
static mlx_array fwd_rope_freqs(const model_config_t *cfg, int layer) {
    (void)layer;
    if (cfg->rope_scaling_type &&
        strcmp(cfg->rope_scaling_type, "llama3") == 0) {
        int n = cfg->head_dim / 2;
        float buf[256];
        if (n > 256 || fwd_rope_llama3_freqs(cfg, buf, n) != 0)
            return (mlx_array){.ctx = NULL};
        int shape[] = {n};
        return mlx_array_new_data(buf, shape, 1, MLX_FLOAT32);
    }
    return (mlx_array){.ctx = NULL};
}

/* Per-layer rope: global layers use rope_theta (scaled by a linear factor
   when set); local layers use rope_local_base_freq at scale 1.0. */
static int fwd_rope_apply(mlx_array *res, mlx_array x, int dims,
                          const model_config_t *cfg, int layer, int offset,
                          mlx_stream s) {
    mlx_array freqs = fwd_rope_freqs(cfg, layer);
    if (freqs.ctx) {
        int ok = MLXB_CHECK(mlx_fast_rope(res, x, dims, false,
            (mlx_optional_float){.has_value = false}, 1.0f, offset, freqs, s));
        mlx_array_free(freqs);
        return ok ? 0 : -1;
    }

    bool global = model_layer_is_global(cfg, layer);
    float base = cfg->rope_theta;
    float scale = 1.0f;
    if (global) {
        /* Apply 1/factor for simple (NULL or "linear") scaling types.
           llama3-style uses the freqs seam path above; proportional goes
           through fwd_rope_freqs. */
        bool simple = !cfg->rope_scaling_type ||
                      strcmp(cfg->rope_scaling_type, "linear") == 0;
        if (simple && cfg->rope_scaling_factor > 0.0f)
            scale = 1.0f / cfg->rope_scaling_factor;
    } else if (cfg->rope_local_base_freq > 0.0f) {
        base = cfg->rope_local_base_freq;
    }

    return MLXB_CHECK(mlx_fast_rope(res, x, dims, false,
        (mlx_optional_float){.value = base, .has_value = true},
        scale, offset, (mlx_array){.ctx = NULL}, s)) ? 0 : -1;
}

/* Add {base}.bias to *io in place (broadcast over [B,S,D]). */
static int add_proj_bias(mlx_array *io, const weights_t *w, const char *base,
                         mlx_stream s) {
    char bname[270];
    snprintf(bname, sizeof(bname), "%s.bias", base);
    int rc = -1;
    mlx_array bias = mlx_array_new();
    mlx_array sum = mlx_array_new();
    if (weights_get(&bias, w, bname) != 0) goto cleanup;
    if (!MLXB_CHECK(mlx_add(&sum, *io, bias, s))) goto cleanup;
    mlx_array_free(*io);
    *io = sum;
    sum = mlx_array_new();
    rc = 0;

cleanup:
    mlx_array_free(sum);
    mlx_array_free(bias);
    return rc;
}

int fwd_attention(mlx_array *out, mlx_array x, int layer,
                  const weights_t *w, const model_config_t *cfg,
                  kvcache_t *kv, mlx_stream s) {
    if (!out || !w || !cfg || !kv) return -1;
    int rc = -1;
    char name[256];

    int S = mlx_array_dim(x, 1);
    int n_heads = cfg->num_attention_heads;
    int n_kv = cfg->num_key_value_heads;
    int hd = cfg->head_dim;
    /* gemma4 pins scale to 1.0; query_pre_attn_scalar defaults to head_dim */
    int qpas = cfg->query_pre_attn_scalar > 0 ? cfg->query_pre_attn_scalar : hd;
    float attn_scale =
        cfg->family == MODEL_GEMMA4 ? 1.0f : 1.0f / sqrtf((float)qpas);

    mlx_array q = mlx_array_new();
    mlx_array k = mlx_array_new();
    mlx_array v = mlx_array_new();
    mlx_array q_reshaped = mlx_array_new();
    mlx_array k_reshaped = mlx_array_new();
    mlx_array v_reshaped = mlx_array_new();
    mlx_array q_normed = mlx_array_new();
    mlx_array k_normed = mlx_array_new();
    mlx_array q_transposed = mlx_array_new();
    mlx_array k_transposed = mlx_array_new();
    mlx_array v_transposed = mlx_array_new();
    mlx_array q_roped = mlx_array_new();
    mlx_array k_roped = mlx_array_new();
    mlx_array kview = mlx_array_new();
    mlx_array vview = mlx_array_new();
    mlx_array sdpa_mask = mlx_array_new();
    mlx_array attn_out = mlx_array_new();
    mlx_array attn_back = mlx_array_new();
    mlx_array attn_reshaped = mlx_array_new();
    mlx_array result = mlx_array_new();

    /* Q/K/V projections */
    weight_triplet_t q_tri, k_tri, v_tri, o_tri;
    memset(&q_tri, 0, sizeof(q_tri));
    memset(&k_tri, 0, sizeof(k_tri));
    memset(&v_tri, 0, sizeof(v_tri));
    memset(&o_tri, 0, sizeof(o_tri));

    weights_tensor_name(name, sizeof(name), cfg, layer, "self_attn.q_proj");
    if (weights_get_triplet(&q_tri, w, name) != 0) goto cleanup;
    if (fwd_linear(&q, x, &q_tri, cfg, s) != 0) goto cleanup;
    if (cfg->attention_bias && add_proj_bias(&q, w, name, s) != 0) goto cleanup;

    weights_tensor_name(name, sizeof(name), cfg, layer, "self_attn.k_proj");
    if (weights_get_triplet(&k_tri, w, name) != 0) goto cleanup;
    if (fwd_linear(&k, x, &k_tri, cfg, s) != 0) goto cleanup;
    if (cfg->attention_bias && add_proj_bias(&k, w, name, s) != 0) goto cleanup;

    weights_tensor_name(name, sizeof(name), cfg, layer, "self_attn.v_proj");
    if (weights_get_triplet(&v_tri, w, name) != 0) goto cleanup;
    if (fwd_linear(&v, x, &v_tri, cfg, s) != 0) goto cleanup;
    if (cfg->attention_bias && add_proj_bias(&v, w, name, s) != 0) goto cleanup;

    /* Reshape to [B,S,H,D] */
    int B = mlx_array_dim(x, 0);
    int q_shape[] = {B, S, n_heads, hd};
    int kv_shape[] = {B, S, n_kv, hd};
    if (!MLXB_CHECK(mlx_reshape(&q_reshaped, q, q_shape, 4, s))) goto cleanup;
    if (!MLXB_CHECK(mlx_reshape(&k_reshaped, k, kv_shape, 4, s))) goto cleanup;
    if (!MLXB_CHECK(mlx_reshape(&v_reshaped, v, kv_shape, 4, s))) goto cleanup;

    /* QK norm on [B,S,H,D] layout */
    if (cfg->has_qk_norm) {
        mlx_array q_norm_w = mlx_array_new();
        mlx_array k_norm_w = mlx_array_new();
        weights_tensor_name(name, sizeof(name), cfg, layer, "self_attn.q_norm");
        char wname[270];
        snprintf(wname, sizeof(wname), "%s.weight", name);
        if (weights_get(&q_norm_w, w, wname) != 0) {
            mlx_array_free(k_norm_w);
            mlx_array_free(q_norm_w);
            goto cleanup;
        }
        weights_tensor_name(name, sizeof(name), cfg, layer, "self_attn.k_norm");
        snprintf(wname, sizeof(wname), "%s.weight", name);
        if (weights_get(&k_norm_w, w, wname) != 0) {
            mlx_array_free(k_norm_w);
            mlx_array_free(q_norm_w);
            goto cleanup;
        }
        if (fwd_rmsnorm(&q_normed, q_reshaped, q_norm_w, cfg->rms_norm_eps,
                        cfg->norm_has_offset, s) != 0 ||
            fwd_rmsnorm(&k_normed, k_reshaped, k_norm_w, cfg->rms_norm_eps,
                        cfg->norm_has_offset, s) != 0) {
            mlx_array_free(k_norm_w);
            mlx_array_free(q_norm_w);
            goto cleanup;
        }
        mlx_array_free(k_norm_w);
        mlx_array_free(q_norm_w);
    } else {
        if (!MLXB_CHECK(mlx_array_set(&q_normed, q_reshaped))) goto cleanup;
        if (!MLXB_CHECK(mlx_array_set(&k_normed, k_reshaped))) goto cleanup;
    }

    /* Transpose to [B,H,S,D] */
    int perm[] = {0, 2, 1, 3};
    if (!MLXB_CHECK(mlx_transpose_axes(&q_transposed, q_normed, perm, 4, s))) goto cleanup;
    if (!MLXB_CHECK(mlx_transpose_axes(&k_transposed, k_normed, perm, 4, s))) goto cleanup;
    if (!MLXB_CHECK(mlx_transpose_axes(&v_transposed, v_reshaped, perm, 4, s))) goto cleanup;

    /* RoPE - offset read BEFORE kvcache_update */
    int offset = kvcache_layer_offset(kv, layer);
    assert(offset >= 0);
    if (fwd_rope_apply(&q_roped, q_transposed, hd, cfg, layer, offset, s) != 0)
        goto cleanup;
    if (fwd_rope_apply(&k_roped, k_transposed, hd, cfg, layer, offset, s) != 0)
        goto cleanup;

    /* KV cache update */
    bool local = !model_layer_is_global(cfg, layer) &&
                 cfg->has_sliding_window && cfg->sliding_window > 0;
    int max_kv = local ? cfg->sliding_window : 0;
    if (kvcache_update(kv, layer, k_roped, v_transposed, max_kv, &kview, &vview, s) != 0)
        goto cleanup;

    /* SDPA: local decode relies on the max_kv view trim in kvcache_update to
       enforce the window; nothing re-selects a mask if that trim goes away.
       If a later design stops trimming (e.g. max_kv = 0 for local layers),
       re-wire fwd_sliding_window_decode_mask here or SWA silently becomes
       full attention on decode. */
    const char *mask_mode;
    if (local && S > 1) {
        int kv_len = mlx_array_dim(kview, 2);
        if (fwd_sliding_window_mask(&sdpa_mask, S, kv_len, cfg->sliding_window, s) != 0)
            goto cleanup;
        mask_mode = "array";
    } else if (local) {
        mask_mode = "";
    } else {
        mask_mode = S > 1 ? "causal" : "";
    }
    if (!MLXB_CHECK(mlx_fast_scaled_dot_product_attention(
            &attn_out, q_roped, kview, vview, attn_scale, mask_mode,
            sdpa_mask, (mlx_array){.ctx = NULL}, s)))
        goto cleanup;

    /* Transpose back and reshape */
    int perm_back[] = {0, 2, 1, 3};
    if (!MLXB_CHECK(mlx_transpose_axes(&attn_back, attn_out, perm_back, 4, s)))
        goto cleanup;
    int out_shape[] = {B, S, n_heads * hd};
    if (!MLXB_CHECK(mlx_reshape(&attn_reshaped, attn_back, out_shape, 3, s)))
        goto cleanup;

    /* O projection */
    weights_tensor_name(name, sizeof(name), cfg, layer, "self_attn.o_proj");
    if (weights_get_triplet(&o_tri, w, name) != 0) goto cleanup;
    if (fwd_linear(&result, attn_reshaped, &o_tri, cfg, s) != 0) goto cleanup;

    mlx_array_free(*out);
    *out = result;
    rc = 0;
    result = mlx_array_new();

cleanup:
    weights_triplet_free(&o_tri);
    weights_triplet_free(&v_tri);
    weights_triplet_free(&k_tri);
    weights_triplet_free(&q_tri);
    mlx_array_free(result);
    mlx_array_free(attn_reshaped);
    mlx_array_free(attn_back);
    mlx_array_free(attn_out);
    mlx_array_free(sdpa_mask);
    mlx_array_free(vview);
    mlx_array_free(kview);
    mlx_array_free(k_roped);
    mlx_array_free(q_roped);
    mlx_array_free(v_transposed);
    mlx_array_free(k_transposed);
    mlx_array_free(q_transposed);
    mlx_array_free(k_normed);
    mlx_array_free(q_normed);
    mlx_array_free(v_reshaped);
    mlx_array_free(k_reshaped);
    mlx_array_free(q_reshaped);
    mlx_array_free(v);
    mlx_array_free(k);
    mlx_array_free(q);
    return rc;
}

int fwd_swiglu(mlx_array *out, mlx_array x, int layer,
               const weights_t *w, const model_config_t *cfg, mlx_stream s) {
    if (!out || !w || !cfg) return -1;
    int rc = -1;
    char name[256];

    mlx_array gate = mlx_array_new();
    mlx_array up = mlx_array_new();
    mlx_array gate_sig = mlx_array_new();
    mlx_array silu = mlx_array_new();
    mlx_array gated = mlx_array_new();
    mlx_array down_in = mlx_array_new();
    mlx_array result = mlx_array_new();

    weight_triplet_t gate_tri, up_tri, down_tri;
    memset(&gate_tri, 0, sizeof(gate_tri));
    memset(&up_tri, 0, sizeof(up_tri));
    memset(&down_tri, 0, sizeof(down_tri));

    weights_tensor_name(name, sizeof(name), cfg, layer, "mlp.gate_proj");
    if (weights_get_triplet(&gate_tri, w, name) != 0) goto cleanup;
    if (fwd_linear(&gate, x, &gate_tri, cfg, s) != 0) goto cleanup;

    weights_tensor_name(name, sizeof(name), cfg, layer, "mlp.up_proj");
    if (weights_get_triplet(&up_tri, w, name) != 0) goto cleanup;
    if (fwd_linear(&up, x, &up_tri, cfg, s) != 0) goto cleanup;

    /* silu(gate) = gate * sigmoid(gate) */
    if (!MLXB_CHECK(mlx_sigmoid(&gate_sig, gate, s))) goto cleanup;
    if (!MLXB_CHECK(mlx_multiply(&silu, gate, gate_sig, s))) goto cleanup;

    /* silu(gate) * up */
    if (!MLXB_CHECK(mlx_multiply(&down_in, silu, up, s))) goto cleanup;

    /* down projection */
    weights_tensor_name(name, sizeof(name), cfg, layer, "mlp.down_proj");
    if (weights_get_triplet(&down_tri, w, name) != 0) goto cleanup;
    if (fwd_linear(&result, down_in, &down_tri, cfg, s) != 0) goto cleanup;

    mlx_array_free(*out);
    *out = result;
    rc = 0;
    result = mlx_array_new();

cleanup:
    weights_triplet_free(&down_tri);
    weights_triplet_free(&up_tri);
    weights_triplet_free(&gate_tri);
    mlx_array_free(result);
    mlx_array_free(down_in);
    mlx_array_free(gated);
    mlx_array_free(silu);
    mlx_array_free(gate_sig);
    mlx_array_free(up);
    mlx_array_free(gate);
    return rc;
}

int fwd_decoder_layer(mlx_array *out, mlx_array x, int layer,
                      const weights_t *w, const model_config_t *cfg,
                      kvcache_t *kv, mlx_stream s) {
    if (!out || !w || !cfg || !kv) return -1;
    int rc = -1;
    char name[256], wname[270];

    mlx_array ln1_w = mlx_array_new();
    mlx_array ln2_w = mlx_array_new();
    mlx_array ln_pre_w = mlx_array_new();
    mlx_array ln_post_w = mlx_array_new();
    mlx_array normed1 = mlx_array_new();
    mlx_array normed2 = mlx_array_new();
    mlx_array attn_out = mlx_array_new();
    mlx_array attn_normed = mlx_array_new();
    mlx_array mlp_out = mlx_array_new();
    mlx_array mlp_normed = mlx_array_new();
    mlx_array h1 = mlx_array_new();
    mlx_array result = mlx_array_new();
    float eps = cfg->rms_norm_eps;
    bool offs = cfg->norm_has_offset;

    /* input_layernorm */
    weights_tensor_name(name, sizeof(name), cfg, layer, "input_layernorm");
    snprintf(wname, sizeof(wname), "%s.weight", name);
    if (weights_get(&ln1_w, w, wname) != 0) goto cleanup;
    if (fwd_rmsnorm(&normed1, x, ln1_w, eps, offs, s) != 0) goto cleanup;

    /* attention */
    if (fwd_attention(&attn_out, normed1, layer, w, cfg, kv, s) != 0) goto cleanup;

    weights_tensor_name(name, sizeof(name), cfg, layer, "post_attention_layernorm");
    snprintf(wname, sizeof(wname), "%s.weight", name);
    if (weights_get(&ln2_w, w, wname) != 0) goto cleanup;

    if (cfg->has_pre_ff_norm) {
        /* sandwich: h = x + norm(attn_out); out = h + norm(mlp(norm(h))) */
        weights_tensor_name(name, sizeof(name), cfg, layer,
                            "pre_feedforward_layernorm");
        snprintf(wname, sizeof(wname), "%s.weight", name);
        if (weights_get(&ln_pre_w, w, wname) != 0) goto cleanup;
        weights_tensor_name(name, sizeof(name), cfg, layer,
                            "post_feedforward_layernorm");
        snprintf(wname, sizeof(wname), "%s.weight", name);
        if (weights_get(&ln_post_w, w, wname) != 0) goto cleanup;

        if (fwd_rmsnorm(&attn_normed, attn_out, ln2_w, eps, offs, s) != 0)
            goto cleanup;
        if (!MLXB_CHECK(mlx_add(&h1, x, attn_normed, s))) goto cleanup;
        if (fwd_rmsnorm(&normed2, h1, ln_pre_w, eps, offs, s) != 0)
            goto cleanup;
        if (fwd_swiglu(&mlp_out, normed2, layer, w, cfg, s) != 0) goto cleanup;
        if (fwd_rmsnorm(&mlp_normed, mlp_out, ln_post_w, eps, offs, s) != 0)
            goto cleanup;
        if (!MLXB_CHECK(mlx_add(&result, h1, mlp_normed, s))) goto cleanup;
    } else {
        /* pre-norm: h = x + attn_out; out = h + mlp(norm(h)) */
        if (!MLXB_CHECK(mlx_add(&h1, x, attn_out, s))) goto cleanup;
        if (fwd_rmsnorm(&normed2, h1, ln2_w, eps, offs, s) != 0) goto cleanup;
        if (fwd_swiglu(&mlp_out, normed2, layer, w, cfg, s) != 0) goto cleanup;
        if (!MLXB_CHECK(mlx_add(&result, h1, mlp_out, s))) goto cleanup;
    }

    mlx_array_free(*out);
    *out = result;
    rc = 0;
    result = mlx_array_new();

cleanup:
    mlx_array_free(result);
    mlx_array_free(h1);
    mlx_array_free(mlp_normed);
    mlx_array_free(mlp_out);
    mlx_array_free(attn_normed);
    mlx_array_free(attn_out);
    mlx_array_free(normed2);
    mlx_array_free(normed1);
    mlx_array_free(ln_post_w);
    mlx_array_free(ln_pre_w);
    mlx_array_free(ln2_w);
    mlx_array_free(ln1_w);
    return rc;
}
