#include "engine/forward.h"

#include "core/log.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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
    if (cfg->head_dim < 2 || cfg->head_dim % 2 != 0) return -1;
    if (n != cfg->head_dim / 2) return -1;
    if (cfg->rope_scaling_factor <= 0.0f) return -1;
    if (cfg->rope_original_max_position_embeddings <= 0) return -1;
    if (cfg->rope_low_freq_factor <= 0.0f) return -1;
    if (cfg->rope_high_freq_factor <= cfg->rope_low_freq_factor) return -1;

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

int fwd_rope_proportional_freqs(const model_config_t *cfg, float *out, int n) {
    if (!cfg || !out) return -1;
    if (!cfg->rope_proportional) return -1;
    int ghd = cfg->global_head_dim;
    if (ghd < 2 || ghd % 2 != 0) return -1;
    if (n != ghd / 2) return -1;
    if (cfg->rope_proportional_factor <= 0.0f) return -1;
    float prf = cfg->partial_rotary_factor_global;
    if (prf <= 0.0f || prf > 1.0f) return -1;

    int rotated_dims = (int)((float)ghd * prf);
    int rotated_half = rotated_dims / 2;

    double base = (double)cfg->rope_theta;
    double factor = (double)cfg->rope_proportional_factor;

    for (int i = 0; i < n; i++) {
        if (i < rotated_half) {
            out[i] = (float)(factor * pow(base, 2.0 * i / (double)ghd));
        } else {
            out[i] = INFINITY;
        }
    }
    return 0;
}

int fwd_rope_freqs_build(mlx_array *out, const model_config_t *cfg) {
    if (!out || !cfg) return -1;

    if (cfg->rope_proportional) {
        int n = cfg->global_head_dim / 2;
        float *buf = malloc((size_t)n * sizeof(float));
        if (!buf) return -1;
        if (fwd_rope_proportional_freqs(cfg, buf, n) != 0) {
            free(buf);
            return -1;
        }
        int shape[] = {n};
        mlx_array arr = mlx_array_new_data(buf, shape, 1, MLX_FLOAT32);
        free(buf);
        if (!arr.ctx) return -1;
        if (out->ctx) mlx_array_free(*out);
        *out = arr;
        return 0;
    }

    if (!cfg->rope_scaling_type ||
        strcmp(cfg->rope_scaling_type, "llama3") != 0) {
        *out = (mlx_array){.ctx = NULL};
        return 0;
    }
    int n = cfg->head_dim / 2;
    float *buf = malloc((size_t)n * sizeof(float));
    if (!buf) return -1;
    if (fwd_rope_llama3_freqs(cfg, buf, n) != 0) {
        free(buf);
        return -1;
    }
    int shape[] = {n};
    mlx_array arr = mlx_array_new_data(buf, shape, 1, MLX_FLOAT32);
    free(buf);
    if (!arr.ctx) return -1;
    if (out->ctx) mlx_array_free(*out);
    *out = arr;
    return 0;
}

int fwd_rope_apply(mlx_array *res, mlx_array x, int dims,
                   const model_config_t *cfg, int layer, int offset,
                   mlx_array rope_freqs, mlx_stream s) {
    bool global = model_layer_is_global(cfg, layer);

    if (rope_freqs.ctx && (!cfg->rope_proportional || global)) {
        return MLXB_CHECK(mlx_fast_rope(res, x, dims, false,
            (mlx_optional_float){.has_value = false}, 1.0f, offset,
            rope_freqs, s)) ? 0 : -1;
    }

    if (cfg->rope_scaling_type &&
        strcmp(cfg->rope_scaling_type, "llama3") == 0)
        return -1;

    float base = cfg->rope_theta;
    float scale = 1.0f;
    if (global) {
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
                  kvcache_t *kv, mlx_array rope_freqs, mlx_stream s) {
    if (!out || !w || !cfg || !kv) return -1;
    int rc = -1;
    char name[256];

    int S = mlx_array_dim(x, 1);
    int B = mlx_array_dim(x, 0);
    int n_heads = cfg->num_attention_heads;
    int hd_l = model_layer_head_dim(cfg, layer);
    int n_kv_l = model_layer_kv_heads(cfg, layer);
    int qpas = cfg->query_pre_attn_scalar > 0 ? cfg->query_pre_attn_scalar : hd_l;
    float attn_scale =
        cfg->attn_scale_one ? 1.0f : 1.0f / sqrtf((float)qpas);
    bool global = model_layer_is_global(cfg, layer);
    int kv_src = model_kv_source_layer(cfg, layer);

    mlx_array q = mlx_array_new();
    mlx_array k = mlx_array_new();
    mlx_array v = mlx_array_new();
    mlx_array q_reshaped = mlx_array_new();
    mlx_array k_reshaped = mlx_array_new();
    mlx_array v_reshaped = mlx_array_new();
    mlx_array v_normed = mlx_array_new();
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
    mlx_array vnorm_w = mlx_array_new();

    weight_triplet_t q_tri, k_tri, v_tri, o_tri;
    memset(&q_tri, 0, sizeof(q_tri));
    memset(&k_tri, 0, sizeof(k_tri));
    memset(&v_tri, 0, sizeof(v_tri));
    memset(&o_tri, 0, sizeof(o_tri));

    /* Q projection (always present) */
    weights_tensor_name(name, sizeof(name), cfg, layer, "self_attn.q_proj");
    if (weights_get_triplet(&q_tri, w, name) != 0) goto cleanup;
    if (fwd_linear(&q, x, &q_tri, cfg, s) != 0) goto cleanup;
    if (cfg->attention_bias && add_proj_bias(&q, w, name, s) != 0) goto cleanup;

    int q_shape[] = {B, S, n_heads, hd_l};
    if (!MLXB_CHECK(mlx_reshape(&q_reshaped, q, q_shape, 4, s))) goto cleanup;

    /* Q norm */
    if (cfg->has_qk_norm) {
        mlx_array q_norm_w = mlx_array_new();
        weights_tensor_name(name, sizeof(name), cfg, layer, "self_attn.q_norm");
        char wname[270];
        snprintf(wname, sizeof(wname), "%s.weight", name);
        if (weights_get(&q_norm_w, w, wname) != 0) {
            mlx_array_free(q_norm_w);
            goto cleanup;
        }
        if (fwd_rmsnorm(&q_normed, q_reshaped, q_norm_w, cfg->rms_norm_eps,
                        cfg->norm_has_offset, s) != 0) {
            mlx_array_free(q_norm_w);
            goto cleanup;
        }
        mlx_array_free(q_norm_w);
    } else {
        if (!MLXB_CHECK(mlx_array_set(&q_normed, q_reshaped))) goto cleanup;
    }

    /* Transpose Q to [B,H,S,D] and apply RoPE */
    int perm[] = {0, 2, 1, 3};
    if (!MLXB_CHECK(mlx_transpose_axes(&q_transposed, q_normed, perm, 4, s))) goto cleanup;

    if (model_layer_kv_shared(cfg, layer) && kv_src < 0) {
        log_error("fwd_attention: kv-shared layer %d has no same-type source", layer);
        goto cleanup;
    }

    if (kv_src >= 0) {
        /* KV-shared layer: skip K/V projection, read from source layer's cache.
           Rope offset for Q = source's offset - S (source ran earlier this forward). */
        int src_offset = kvcache_layer_offset(kv, kv_src) - S;
        if (src_offset < 0) src_offset = 0;
        if (fwd_rope_apply(&q_roped, q_transposed, hd_l, cfg, layer,
                           src_offset, rope_freqs, s) != 0)
            goto cleanup;

        bool src_local = !model_layer_is_global(cfg, kv_src) &&
                         cfg->has_sliding_window && cfg->sliding_window > 0;
        int src_max_kv = src_local ? cfg->sliding_window : 0;
        if (kvcache_view(kv, kv_src, src_max_kv, S, &kview, &vview, s) != 0)
            goto cleanup;
    } else {
        /* K/V owning layer */
        int offset = kvcache_layer_offset(kv, layer);
        assert(offset >= 0);
        if (fwd_rope_apply(&q_roped, q_transposed, hd_l, cfg, layer,
                           offset, rope_freqs, s) != 0)
            goto cleanup;

        /* K projection */
        weights_tensor_name(name, sizeof(name), cfg, layer, "self_attn.k_proj");
        if (weights_get_triplet(&k_tri, w, name) != 0) goto cleanup;
        if (fwd_linear(&k, x, &k_tri, cfg, s) != 0) goto cleanup;
        if (cfg->attention_bias && add_proj_bias(&k, w, name, s) != 0) goto cleanup;

        int kv_shape[] = {B, S, n_kv_l, hd_l};
        if (!MLXB_CHECK(mlx_reshape(&k_reshaped, k, kv_shape, 4, s))) goto cleanup;

        /* V projection: k_eq_v on global layers reuses raw K */
        if (cfg->attention_k_eq_v && global) {
            if (!MLXB_CHECK(mlx_array_set(&v_reshaped, k_reshaped))) goto cleanup;
        } else {
            weights_tensor_name(name, sizeof(name), cfg, layer, "self_attn.v_proj");
            if (weights_get_triplet(&v_tri, w, name) != 0) goto cleanup;
            if (fwd_linear(&v, x, &v_tri, cfg, s) != 0) goto cleanup;
            if (cfg->attention_bias && add_proj_bias(&v, w, name, s) != 0) goto cleanup;
            if (!MLXB_CHECK(mlx_reshape(&v_reshaped, v, kv_shape, 4, s))) goto cleanup;
        }

        /* K norm */
        if (cfg->has_qk_norm) {
            mlx_array k_norm_w = mlx_array_new();
            weights_tensor_name(name, sizeof(name), cfg, layer, "self_attn.k_norm");
            char wname[270];
            snprintf(wname, sizeof(wname), "%s.weight", name);
            if (weights_get(&k_norm_w, w, wname) != 0) {
                mlx_array_free(k_norm_w);
                goto cleanup;
            }
            if (fwd_rmsnorm(&k_normed, k_reshaped, k_norm_w, cfg->rms_norm_eps,
                            cfg->norm_has_offset, s) != 0) {
                mlx_array_free(k_norm_w);
                goto cleanup;
            }
            mlx_array_free(k_norm_w);
        } else {
            if (!MLXB_CHECK(mlx_array_set(&k_normed, k_reshaped))) goto cleanup;
        }

        /* V norm: parameter-free RMS norm (weight = ones) */
        if (cfg->has_v_norm) {
            int vnw_shape[] = {hd_l};
            if (!MLXB_CHECK(mlx_ones(&vnorm_w, vnw_shape, 1, MLX_BFLOAT16, s)))
                goto cleanup;
            if (fwd_rmsnorm(&v_normed, v_reshaped, vnorm_w, cfg->rms_norm_eps,
                            false, s) != 0)
                goto cleanup;
        } else {
            if (!MLXB_CHECK(mlx_array_set(&v_normed, v_reshaped))) goto cleanup;
        }

        /* Transpose K, V to [B,H,S,D] */
        if (!MLXB_CHECK(mlx_transpose_axes(&k_transposed, k_normed, perm, 4, s)))
            goto cleanup;
        if (!MLXB_CHECK(mlx_transpose_axes(&v_transposed, v_normed, perm, 4, s)))
            goto cleanup;

        /* RoPE on K */
        if (fwd_rope_apply(&k_roped, k_transposed, hd_l, cfg, layer,
                           offset, rope_freqs, s) != 0)
            goto cleanup;

        /* KV cache update */
        bool local = !global && cfg->has_sliding_window && cfg->sliding_window > 0;
        int max_kv = local ? cfg->sliding_window : 0;
        if (kvcache_update(kv, layer, k_roped, v_transposed, max_kv,
                           &kview, &vview, s) != 0)
            goto cleanup;
    }

    /* SDPA mask: local decode window enforcement is the max_kv view trim in
       kvcache_update/kvcache_view, not a decode mask. If trimming ever goes
       away, re-wire fwd_sliding_window_decode_mask here. */
    bool local_attn = !global && cfg->has_sliding_window && cfg->sliding_window > 0;
    const char *mask_mode;
    if (local_attn && S > 1) {
        int kv_len = mlx_array_dim(kview, 2);
        if (fwd_sliding_window_mask(&sdpa_mask, S, kv_len, cfg->sliding_window, s) != 0)
            goto cleanup;
        mask_mode = "array";
    } else if (local_attn) {
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
    int out_shape[] = {B, S, n_heads * hd_l};
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
    mlx_array_free(vnorm_w);
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
    mlx_array_free(v_normed);
    mlx_array_free(v_reshaped);
    mlx_array_free(k_reshaped);
    mlx_array_free(q_reshaped);
    mlx_array_free(v);
    mlx_array_free(k);
    mlx_array_free(q);
    return rc;
}

int fwd_softcap(mlx_array *out, mlx_array logits, float cap, mlx_stream s) {
    if (!out || cap <= 0.0f) return -1;
    int rc = -1;
    mlx_array cap_bf = mlx_array_new();
    mlx_array scaled = mlx_array_new();
    mlx_array th = mlx_array_new();
    mlx_array result = mlx_array_new();

    mlx_array cap_f32 = mlx_array_new_float(cap);
    if (!MLXB_CHECK(mlx_astype(&cap_bf, cap_f32, MLX_BFLOAT16, s))) {
        mlx_array_free(cap_f32);
        goto cleanup;
    }
    mlx_array_free(cap_f32);

    if (!MLXB_CHECK(mlx_divide(&scaled, logits, cap_bf, s))) goto cleanup;
    if (!MLXB_CHECK(mlx_tanh(&th, scaled, s))) goto cleanup;
    if (!MLXB_CHECK(mlx_multiply(&result, th, cap_bf, s))) goto cleanup;

    mlx_array_free(*out);
    *out = result;
    result = mlx_array_new();
    rc = 0;

cleanup:
    mlx_array_free(result);
    mlx_array_free(th);
    mlx_array_free(scaled);
    mlx_array_free(cap_bf);
    return rc;
}

int fwd_gelu_approx(mlx_array *out, mlx_array x, mlx_stream s) {
    if (!out) return -1;

    mlx_array g3 = mlx_array_new();
    mlx_array coeff = mlx_array_new();
    mlx_array g_inner = mlx_array_new();
    mlx_array scaled = mlx_array_new();
    mlx_array tanh_val = mlx_array_new();
    mlx_array one_plus = mlx_array_new();
    mlx_array half_gate = mlx_array_new();
    mlx_array result = mlx_array_new();

    float sqrt_2_over_pi = 0.7978845608f;
    float coeff_val = 0.044715f;
    mlx_array sqrt2pi_a = mlx_array_new_float(sqrt_2_over_pi);
    mlx_array coeff_a = mlx_array_new_float(coeff_val);
    mlx_array sqrt2pi_bf = mlx_array_new();
    mlx_array coeff_bf = mlx_array_new();
    if (!MLXB_CHECK(mlx_astype(&sqrt2pi_bf, sqrt2pi_a, MLX_BFLOAT16, s)) ||
        !MLXB_CHECK(mlx_astype(&coeff_bf, coeff_a, MLX_BFLOAT16, s)))
    {
        mlx_array_free(sqrt2pi_a); mlx_array_free(coeff_a);
        mlx_array_free(sqrt2pi_bf); mlx_array_free(coeff_bf);
        goto fail;
    }

    mlx_array g2 = mlx_array_new();
    bool ok = MLXB_CHECK(mlx_multiply(&g2, x, x, s)) &&
              MLXB_CHECK(mlx_multiply(&g3, g2, x, s)) &&
              MLXB_CHECK(mlx_multiply(&coeff, coeff_bf, g3, s)) &&
              MLXB_CHECK(mlx_add(&g_inner, x, coeff, s)) &&
              MLXB_CHECK(mlx_multiply(&scaled, sqrt2pi_bf, g_inner, s)) &&
              MLXB_CHECK(mlx_tanh(&tanh_val, scaled, s));
    mlx_array_free(g2);
    mlx_array_free(sqrt2pi_a); mlx_array_free(coeff_a);
    mlx_array_free(sqrt2pi_bf); mlx_array_free(coeff_bf);
    mlx_array_free(g3); mlx_array_free(coeff);
    mlx_array_free(g_inner); mlx_array_free(scaled);
    if (!ok) {
        mlx_array_free(tanh_val); mlx_array_free(one_plus);
        mlx_array_free(half_gate); mlx_array_free(result);
        return -1;
    }

    mlx_array one_f = mlx_array_new_float(1.0f);
    mlx_array half_f = mlx_array_new_float(0.5f);
    mlx_array one_b = mlx_array_new();
    mlx_array half_b = mlx_array_new();
    ok = MLXB_CHECK(mlx_astype(&one_b, one_f, MLX_BFLOAT16, s)) &&
         MLXB_CHECK(mlx_astype(&half_b, half_f, MLX_BFLOAT16, s)) &&
         MLXB_CHECK(mlx_add(&one_plus, one_b, tanh_val, s)) &&
         MLXB_CHECK(mlx_multiply(&half_gate, half_b, x, s)) &&
         MLXB_CHECK(mlx_multiply(&result, half_gate, one_plus, s));
    mlx_array_free(one_f); mlx_array_free(half_f);
    mlx_array_free(one_b); mlx_array_free(half_b);
    mlx_array_free(tanh_val); mlx_array_free(one_plus);
    mlx_array_free(half_gate);
    if (!ok) { mlx_array_free(result); return -1; }

    mlx_array_free(*out);
    *out = result;
    return 0;

fail:
    mlx_array_free(g3); mlx_array_free(coeff);
    mlx_array_free(g_inner); mlx_array_free(scaled);
    mlx_array_free(tanh_val); mlx_array_free(one_plus);
    mlx_array_free(half_gate); mlx_array_free(result);
    return -1;
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

    if (cfg->hidden_act == HIDDEN_ACT_GELU_APPROX) {
        if (fwd_gelu_approx(&silu, gate, s) != 0) goto cleanup;
    } else {
        /* silu(gate) = gate * sigmoid(gate) */
        if (!MLXB_CHECK(mlx_sigmoid(&gate_sig, gate, s))) goto cleanup;
        if (!MLXB_CHECK(mlx_multiply(&silu, gate, gate_sig, s))) goto cleanup;
    }

    /* activation(gate) * up */
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
    mlx_array_free(silu);
    mlx_array_free(gate_sig);
    mlx_array_free(up);
    mlx_array_free(gate);
    return rc;
}

/* Embed-take helper: shared logic for embedding table lookup.
   Takes flat ids from an embed weight, handles quant and non-quant paths.
   Result is bf16 [total, embed_dim]. */
static int embed_take(mlx_array *out, mlx_array flat_ids,
                      const weights_t *w, const char *base_name,
                      const model_config_t *cfg, mlx_stream s) {
    int rc = -1;
    weight_triplet_t tri;
    if (weights_get_triplet(&tri, w, base_name) != 0) return -1;

    mlx_array taken = mlx_array_new();
    mlx_array result = mlx_array_new();

    if (tri.quantized) {
        mlx_array dq_w = mlx_array_new();
        mlx_array dq_s = mlx_array_new();
        mlx_array dq_b = mlx_array_new();
        mlx_array dq = mlx_array_new();
        bool ok = MLXB_CHECK(mlx_take_axis(&dq_w, tri.weight, flat_ids, 0, s)) &&
                  MLXB_CHECK(mlx_take_axis(&dq_s, tri.scales, flat_ids, 0, s)) &&
                  MLXB_CHECK(mlx_take_axis(&dq_b, tri.biases, flat_ids, 0, s)) &&
                  MLXB_CHECK(mlx_dequantize(
                      &dq, dq_w, dq_s, dq_b,
                      (mlx_optional_int){.value = cfg->quant_group_size, .has_value = true},
                      (mlx_optional_int){.value = cfg->quant_bits, .has_value = true},
                      "affine", (mlx_array){.ctx = NULL},
                      (mlx_optional_dtype){.has_value = false}, s)) &&
                  MLXB_CHECK(mlx_astype(&result, dq, MLX_BFLOAT16, s));
        mlx_array_free(dq);
        mlx_array_free(dq_b);
        mlx_array_free(dq_s);
        mlx_array_free(dq_w);
        if (!ok) goto cleanup;
    } else {
        mlx_array bf16 = mlx_array_new();
        bool ok = MLXB_CHECK(mlx_take_axis(&taken, tri.weight, flat_ids, 0, s)) &&
                  MLXB_CHECK(mlx_astype(&bf16, taken, MLX_BFLOAT16, s));
        if (!ok) { mlx_array_free(bf16); goto cleanup; }
        mlx_array_free(result);
        result = bf16;
    }

    mlx_array_free(*out);
    *out = result;
    result = mlx_array_new();
    rc = 0;

cleanup:
    weights_triplet_free(&tri);
    mlx_array_free(result);
    mlx_array_free(taken);
    return rc;
}

int fwd_ple_inputs(mlx_array *out, mlx_array ids, mlx_array h,
                   const weights_t *w, const model_config_t *cfg,
                   mlx_stream s) {
    if (!out || !w || !cfg) return -1;
    int ple_dim = cfg->hidden_size_per_layer_input;
    if (ple_dim <= 0) return -1;
    int rc = -1;

    int ndim = (int)mlx_array_ndim(ids);
    int B = ndim >= 2 ? mlx_array_dim(ids, 0) : 1;
    int S = ndim >= 2 ? mlx_array_dim(ids, 1) : mlx_array_dim(ids, 0);
    int total = B * S;
    int L = cfg->num_hidden_layers;

    char name[256];
    weights_tensor_name(name, sizeof(name), cfg, -1, "embed_tokens_per_layer");

    mlx_array flat = mlx_array_new();
    mlx_array ple_emb = mlx_array_new();
    mlx_array ple_scale = mlx_array_new();
    mlx_array ple_scaled = mlx_array_new();
    mlx_array h_proj = mlx_array_new();
    mlx_array h_scale = mlx_array_new();
    mlx_array h_scaled = mlx_array_new();
    mlx_array h_reshaped = mlx_array_new();
    mlx_array ple_reshaped = mlx_array_new();
    mlx_array sum = mlx_array_new();
    mlx_array normed = mlx_array_new();
    mlx_array combine_scale = mlx_array_new();
    mlx_array result = mlx_array_new();

    int flat_shape[] = {total};
    if (!MLXB_CHECK(mlx_reshape(&flat, ids, flat_shape, 1, s))) goto cleanup;

    /* embed_tokens_per_layer take + scale by sqrt(ple_dim) */
    if (embed_take(&ple_emb, flat, w, name, cfg, s) != 0) goto cleanup;
    {
        int emb_rs[] = {B, S, L * ple_dim};
        mlx_array tmp = mlx_array_new();
        if (!MLXB_CHECK(mlx_reshape(&tmp, ple_emb, emb_rs, 3, s))) {
            mlx_array_free(tmp);
            goto cleanup;
        }
        mlx_array_free(ple_emb);
        ple_emb = tmp;
    }
    {
        mlx_array sf32 = mlx_array_new_float(sqrtf((float)ple_dim));
        if (!MLXB_CHECK(mlx_astype(&ple_scale, sf32, MLX_BFLOAT16, s))) {
            mlx_array_free(sf32);
            goto cleanup;
        }
        mlx_array_free(sf32);
    }
    if (!MLXB_CHECK(mlx_multiply(&ple_scaled, ple_emb, ple_scale, s)))
        goto cleanup;

    /* h through per_layer_model_projection, scale by 1/sqrt(hidden_size) */
    {
        char proj_name[256];
        weights_tensor_name(proj_name, sizeof(proj_name), cfg, -1,
                            "per_layer_model_projection");
        weight_triplet_t proj_tri;
        if (weights_get_triplet(&proj_tri, w, proj_name) != 0) goto cleanup;
        if (fwd_linear(&h_proj, h, &proj_tri, cfg, s) != 0) {
            weights_triplet_free(&proj_tri);
            goto cleanup;
        }
        weights_triplet_free(&proj_tri);
    }
    {
        mlx_array sf32 = mlx_array_new_float(1.0f / sqrtf((float)cfg->hidden_size));
        if (!MLXB_CHECK(mlx_astype(&h_scale, sf32, MLX_BFLOAT16, s))) {
            mlx_array_free(sf32);
            goto cleanup;
        }
        mlx_array_free(sf32);
    }
    if (!MLXB_CHECK(mlx_multiply(&h_scaled, h_proj, h_scale, s))) goto cleanup;

    /* Reshape both to [B, S, L, ple_dim] */
    {
        int rs[] = {B, S, L, ple_dim};
        if (!MLXB_CHECK(mlx_reshape(&h_reshaped, h_scaled, rs, 4, s)))
            goto cleanup;
        if (!MLXB_CHECK(mlx_reshape(&ple_reshaped, ple_scaled, rs, 4, s)))
            goto cleanup;
    }

    /* RMS norm with per_layer_projection_norm on h_reshaped (the projection) */
    {
        char norm_name[256];
        weights_tensor_name(norm_name, sizeof(norm_name), cfg, -1,
                            "per_layer_projection_norm");
        char wn[270];
        snprintf(wn, sizeof(wn), "%s.weight", norm_name);
        mlx_array nw = mlx_array_new();
        if (weights_get(&nw, w, wn) != 0) {
            mlx_array_free(nw);
            goto cleanup;
        }
        if (fwd_rmsnorm(&normed, h_reshaped, nw, cfg->rms_norm_eps,
                        cfg->norm_has_offset, s) != 0) {
            mlx_array_free(nw);
            goto cleanup;
        }
        mlx_array_free(nw);
    }

    /* (normed + scaled_emb) * (1/sqrt(2)) */
    if (!MLXB_CHECK(mlx_add(&sum, normed, ple_reshaped, s))) goto cleanup;
    {
        mlx_array sf32 = mlx_array_new_float(1.0f / sqrtf(2.0f));
        if (!MLXB_CHECK(mlx_astype(&combine_scale, sf32, MLX_BFLOAT16, s))) {
            mlx_array_free(sf32);
            goto cleanup;
        }
        mlx_array_free(sf32);
    }
    if (!MLXB_CHECK(mlx_multiply(&result, sum, combine_scale, s))) goto cleanup;

    mlx_array_free(*out);
    *out = result;
    result = mlx_array_new();
    rc = 0;

cleanup:
    mlx_array_free(result);
    mlx_array_free(combine_scale);
    mlx_array_free(normed);
    mlx_array_free(sum);
    mlx_array_free(ple_reshaped);
    mlx_array_free(h_reshaped);
    mlx_array_free(h_scaled);
    mlx_array_free(h_scale);
    mlx_array_free(h_proj);
    mlx_array_free(ple_scaled);
    mlx_array_free(ple_scale);
    mlx_array_free(ple_emb);
    mlx_array_free(flat);
    return rc;
}

int fwd_ple_apply(mlx_array *io_h, int layer, mlx_array ple_inputs,
                  const weights_t *w, const model_config_t *cfg,
                  mlx_stream s) {
    if (!io_h || !w || !cfg) return -1;
    if (!ple_inputs.ctx) return 0;
    int ple_dim = cfg->hidden_size_per_layer_input;
    if (ple_dim <= 0) return 0;
    int rc = -1;

    int B = mlx_array_dim(ple_inputs, 0);
    int S = mlx_array_dim(ple_inputs, 1);
    char name[256];

    mlx_array sliced = mlx_array_new();
    mlx_array gate_in = mlx_array_new();
    mlx_array gate_act = mlx_array_new();
    mlx_array gated = mlx_array_new();
    mlx_array projected = mlx_array_new();
    mlx_array normed = mlx_array_new();
    mlx_array result = mlx_array_new();

    /* Slice layer from ple_inputs: [B,S,L,ple_dim] -> [B,S,ple_dim] */
    {
        int start[] = {0, 0, layer, 0};
        int stop[] = {B, S, layer + 1, ple_dim};
        int strides[] = {1, 1, 1, 1};
        mlx_array sl = mlx_array_new();
        if (!MLXB_CHECK(mlx_slice(&sl, ple_inputs, start, 4, stop, 4, strides, 4, s))) {
            mlx_array_free(sl);
            goto cleanup;
        }
        int rs[] = {B, S, ple_dim};
        if (!MLXB_CHECK(mlx_reshape(&sliced, sl, rs, 3, s))) {
            mlx_array_free(sl);
            goto cleanup;
        }
        mlx_array_free(sl);
    }

    /* gate = gelu_approx(per_layer_input_gate @ h) */
    {
        weights_tensor_name(name, sizeof(name), cfg, layer,
                            "per_layer_input_gate");
        weight_triplet_t tri;
        if (weights_get_triplet(&tri, w, name) != 0) goto cleanup;
        if (fwd_linear(&gate_in, *io_h, &tri, cfg, s) != 0) {
            weights_triplet_free(&tri);
            goto cleanup;
        }
        weights_triplet_free(&tri);
    }
    if (fwd_gelu_approx(&gate_act, gate_in, s) != 0) goto cleanup;

    /* gated = sliced * gate */
    if (!MLXB_CHECK(mlx_multiply(&gated, sliced, gate_act, s))) goto cleanup;

    /* project via per_layer_projection */
    {
        weights_tensor_name(name, sizeof(name), cfg, layer,
                            "per_layer_projection");
        weight_triplet_t tri;
        if (weights_get_triplet(&tri, w, name) != 0) goto cleanup;
        if (fwd_linear(&projected, gated, &tri, cfg, s) != 0) {
            weights_triplet_free(&tri);
            goto cleanup;
        }
        weights_triplet_free(&tri);
    }

    /* RMS norm with post_per_layer_input_norm */
    {
        weights_tensor_name(name, sizeof(name), cfg, layer,
                            "post_per_layer_input_norm");
        char wn[270];
        snprintf(wn, sizeof(wn), "%s.weight", name);
        mlx_array nw = mlx_array_new();
        if (weights_get(&nw, w, wn) != 0) {
            mlx_array_free(nw);
            goto cleanup;
        }
        if (fwd_rmsnorm(&normed, projected, nw, cfg->rms_norm_eps,
                        cfg->norm_has_offset, s) != 0) {
            mlx_array_free(nw);
            goto cleanup;
        }
        mlx_array_free(nw);
    }

    /* h += normed */
    if (!MLXB_CHECK(mlx_add(&result, *io_h, normed, s))) goto cleanup;
    mlx_array_free(*io_h);
    *io_h = result;
    result = mlx_array_new();
    rc = 0;

cleanup:
    mlx_array_free(result);
    mlx_array_free(normed);
    mlx_array_free(projected);
    mlx_array_free(gated);
    mlx_array_free(gate_act);
    mlx_array_free(gate_in);
    mlx_array_free(sliced);
    return rc;
}

int fwd_decoder_layer(mlx_array *out, mlx_array x, int layer,
                      const weights_t *w, const model_config_t *cfg,
                      kvcache_t *kv, mlx_array rope_freqs,
                      mlx_array ple_inputs, mlx_stream s) {
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
    if (fwd_attention(&attn_out, normed1, layer, w, cfg, kv, rope_freqs, s) != 0) goto cleanup;

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

    /* PLE: apply per-layer embedding after MLP residual */
    if (ple_inputs.ctx) {
        if (fwd_ple_apply(&result, layer, ple_inputs, w, cfg, s) != 0)
            goto cleanup;
    }

    /* layer_scalar: multiply the layer output when present */
    {
        weights_tensor_name(name, sizeof(name), cfg, layer, "layer_scalar");
        mlx_array ls = mlx_array_new();
        if (weights_get(&ls, w, name) == 0) {
            mlx_array scaled = mlx_array_new();
            if (!MLXB_CHECK(mlx_multiply(&scaled, result, ls, s))) {
                mlx_array_free(scaled);
                mlx_array_free(ls);
                goto cleanup;
            }
            mlx_array_free(ls);
            mlx_array_free(result);
            result = scaled;
        } else {
            mlx_array_free(ls);
        }
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
