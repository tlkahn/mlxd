#include "engine/sampler.h"

#include <math.h>

sampling_params_t sampling_resolve(const sampling_params_t *req, unsigned set_mask,
                                   const model_config_t *cfg) {
    sampling_params_t out = SAMPLING_PARAMS_DEFAULT;

    if (set_mask & SAMPLING_SET_TEMPERATURE)
        out.temperature = req->temperature;
    else if (cfg && cfg->has_gen_temperature)
        out.temperature = cfg->gen_temperature;

    if (set_mask & SAMPLING_SET_TOP_P)
        out.top_p = req->top_p;
    else if (cfg && cfg->has_gen_top_p)
        out.top_p = cfg->gen_top_p;

    if (set_mask & SAMPLING_SET_TOP_K)
        out.top_k = req->top_k;
    else if (cfg && cfg->has_gen_top_k)
        out.top_k = cfg->gen_top_k;

    if (set_mask & SAMPLING_SET_MIN_P)
        out.min_p = req->min_p;

    if (set_mask & SAMPLING_SET_SEED)
        out.seed = req->seed;

    return out;
}

/* Below this, sampling degenerates to argmax (matches mlx-serve's cutoff). */
#define SAMPLER_GREEDY_TEMP 0.01f

static int last_dim(mlx_array a) {
    size_t nd = mlx_array_ndim(a);
    return nd == 0 ? 1 : mlx_array_dim(a, (int)nd - 1);
}

int sampler_key_init(sampler_key_t *k, int seed) {
    if (seed < 0) {
        k->key = mlx_array_new();
        k->seeded = false;
        return 0;
    }
    k->key = mlx_array_new();
    if (!MLXB_CHECK(mlx_random_key(&k->key, (uint64_t)seed))) {
        mlx_array_free(k->key);
        k->key = mlx_array_new();
        k->seeded = false;
        return -1;
    }
    k->seeded = true;
    return 0;
}

int sampler_key_next(sampler_key_t *k, mlx_array *subkey_out, mlx_stream s) {
    if (!k->seeded) {
        *subkey_out = mlx_array_new();
        return 0;
    }
    mlx_array new_key = mlx_array_new();
    mlx_array subkey = mlx_array_new();
    if (!MLXB_CHECK(mlx_random_split(&new_key, &subkey, k->key, s))) {
        mlx_array_free(new_key);
        mlx_array_free(subkey);
        return -1;
    }
    mlx_array_free(k->key);
    k->key = new_key;
    *subkey_out = subkey;
    return 0;
}

void sampler_key_free(sampler_key_t *k) {
    mlx_array_free(k->key);
    k->key = mlx_array_new();
    k->seeded = false;
}

/* *out = where(mask, logits, -inf). Frees mask. */
static int mask_neg_inf(mlx_array mask, mlx_array logits, mlx_stream s, mlx_array *out) {
    mlx_array neg_inf = mlx_array_new_float(-INFINITY);
    mlx_array res = mlx_array_new();
    int rc = MLXB_CHECK(mlx_where(&res, mask, logits, neg_inf, s)) ? 0 : -1;
    mlx_array_free(neg_inf);
    mlx_array_free(mask);
    if (rc != 0) {
        mlx_array_free(res);
        return -1;
    }
    *out = res;
    return 0;
}

int sampler_apply_top_k(mlx_array logits, int k, mlx_stream s, mlx_array *out) {
    if (k <= 0 || k >= last_dim(logits)) {
        mlx_array copy = mlx_array_new();
        if (!MLXB_CHECK(mlx_copy(&copy, logits, s))) {
            mlx_array_free(copy);
            return -1;
        }
        *out = copy;
        return 0;
    }

    mlx_array topk = mlx_array_new();
    mlx_array cutoff = mlx_array_new();
    mlx_array mask = mlx_array_new();
    if (!MLXB_CHECK(mlx_topk_axis(&topk, logits, k, -1, s)) ||
        !MLXB_CHECK(mlx_min_axis(&cutoff, topk, -1, true, s)) ||
        !MLXB_CHECK(mlx_greater_equal(&mask, logits, cutoff, s))) {
        mlx_array_free(topk);
        mlx_array_free(cutoff);
        mlx_array_free(mask);
        return -1;
    }
    mlx_array_free(topk);
    mlx_array_free(cutoff);
    return mask_neg_inf(mask, logits, s, out);
}

int sampler_apply_top_p(mlx_array logits, float p, mlx_stream s, mlx_array *out) {
    if (p >= 1.0f) {
        mlx_array copy = mlx_array_new();
        if (!MLXB_CHECK(mlx_copy(&copy, logits, s))) {
            mlx_array_free(copy);
            return -1;
        }
        *out = copy;
        return 0;
    }

    /* Ascending sort, softmax, inclusive cumsum: tokens whose bottom-tail mass
     * stays <= 1-p fall outside the nucleus. Recover the smallest in-nucleus
     * logit as a scalar cutoff so the mask applies in original vocab order. */
    mlx_array sorted = mlx_array_new();
    mlx_array probs = mlx_array_new();
    mlx_array cumsum = mlx_array_new();
    mlx_array in_nucleus = mlx_array_new();
    mlx_array nucleus_logits = mlx_array_new();
    mlx_array cutoff = mlx_array_new();
    mlx_array mask = mlx_array_new();
    mlx_array threshold = mlx_array_new_float(1.0f - p);
    mlx_array pos_inf = mlx_array_new_float(INFINITY);
    int rc = -1;

    if (!MLXB_CHECK(mlx_sort_axis(&sorted, logits, -1, s)) ||
        !MLXB_CHECK(mlx_softmax_axis(&probs, sorted, -1, true, s)) ||
        !MLXB_CHECK(mlx_cumsum(&cumsum, probs, -1, false, true, s)) ||
        !MLXB_CHECK(mlx_greater(&in_nucleus, cumsum, threshold, s)) ||
        !MLXB_CHECK(mlx_where(&nucleus_logits, in_nucleus, sorted, pos_inf, s)) ||
        !MLXB_CHECK(mlx_min_axis(&cutoff, nucleus_logits, -1, true, s)) ||
        !MLXB_CHECK(mlx_greater_equal(&mask, logits, cutoff, s)))
        goto out;
    rc = 0;

out:
    mlx_array_free(sorted);
    mlx_array_free(probs);
    mlx_array_free(cumsum);
    mlx_array_free(in_nucleus);
    mlx_array_free(nucleus_logits);
    mlx_array_free(cutoff);
    mlx_array_free(threshold);
    mlx_array_free(pos_inf);
    if (rc != 0) {
        mlx_array_free(mask);
        return -1;
    }
    return mask_neg_inf(mask, logits, s, out);
}

int sampler_apply_min_p(mlx_array logits, float mp, mlx_stream s, mlx_array *out) {
    if (mp <= 0.0f) {
        mlx_array copy = mlx_array_new();
        if (!MLXB_CHECK(mlx_copy(&copy, logits, s))) {
            mlx_array_free(copy);
            return -1;
        }
        *out = copy;
        return 0;
    }

    /* Threshold in logit space: keep logits >= max(logits) + log(min_p).
     * Exact under softmax without computing softmax. */
    mlx_array max_logit = mlx_array_new();
    mlx_array log_mp = mlx_array_new_float(logf(mp));
    mlx_array threshold = mlx_array_new();
    mlx_array mask = mlx_array_new();
    int rc = -1;

    if (!MLXB_CHECK(mlx_max_axis(&max_logit, logits, -1, true, s)) ||
        !MLXB_CHECK(mlx_add(&threshold, max_logit, log_mp, s)) ||
        !MLXB_CHECK(mlx_greater_equal(&mask, logits, threshold, s)))
        goto out;
    rc = 0;

out:
    mlx_array_free(max_logit);
    mlx_array_free(log_mp);
    mlx_array_free(threshold);
    if (rc != 0) {
        mlx_array_free(mask);
        return -1;
    }
    return mask_neg_inf(mask, logits, s, out);
}

/* Pre-temperature log-probs: log(softmax(logits)) from raw logits.
 * Deliberate divergence from mlx-serve which scales first; matches mlx-lm. */
static int compute_raw_log_probs(mlx_array logits, mlx_stream s, mlx_array *lp_out) {
    mlx_array lse = mlx_array_new();
    mlx_array lp = mlx_array_new();
    if (!MLXB_CHECK(mlx_logsumexp_axis(&lse, logits, -1, true, s)) ||
        !MLXB_CHECK(mlx_subtract(&lp, logits, lse, s))) {
        mlx_array_free(lse);
        mlx_array_free(lp);
        return -1;
    }
    mlx_array_free(lse);
    *lp_out = lp;
    return 0;
}

static int compute_logprob_at(mlx_array log_probs, mlx_array tok, mlx_stream s,
                              mlx_array *logprob_out) {
    mlx_array tok_idx = mlx_array_new();
    mlx_array lp_raw = mlx_array_new();
    mlx_array lp = mlx_array_new();
    if (!MLXB_CHECK(mlx_reshape(&tok_idx, tok, (int[]){1, 1}, 2, s)) ||
        !MLXB_CHECK(mlx_take_along_axis(&lp_raw, log_probs, tok_idx, -1, s)) ||
        !MLXB_CHECK(mlx_astype(&lp, lp_raw, MLX_FLOAT32, s))) {
        mlx_array_free(tok_idx);
        mlx_array_free(lp_raw);
        mlx_array_free(lp);
        return -1;
    }
    mlx_array_free(tok_idx);
    mlx_array_free(lp_raw);
    *logprob_out = lp;
    return 0;
}

int sampler_sample_lazy(mlx_array logits, const sampling_params_t *sp, mlx_array subkey,
                        bool want_logprob, mlx_stream s, mlx_array *tok_out,
                        mlx_array *logprob_out) {
    mlx_array tok = mlx_array_new();

    /* temperature scale: logits / T */
    mlx_array scaled = mlx_array_new();
    if (sp->temperature < SAMPLER_GREEDY_TEMP) {
        /* Greedy short-circuit: argmax, optionally compute logprob from raw logits */
        if (!MLXB_CHECK(mlx_argmax_axis(&tok, logits, -1, false, s))) {
            mlx_array_free(tok);
            mlx_array_free(scaled);
            return -1;
        }
        if (want_logprob && logprob_out) {
            mlx_array log_probs = mlx_array_new();
            if (compute_raw_log_probs(logits, s, &log_probs) != 0) {
                mlx_array_free(log_probs);
                mlx_array_free(tok);
                mlx_array_free(scaled);
                return -1;
            }
            if (compute_logprob_at(log_probs, tok, s, logprob_out) != 0) {
                mlx_array_free(log_probs);
                mlx_array_free(tok);
                mlx_array_free(scaled);
                return -1;
            }
            mlx_array_free(log_probs);
        }
        mlx_array_free(scaled);
        *tok_out = tok;
        return 0;
    }

    if (sp->temperature != 1.0f) {
        mlx_array temp = mlx_array_new_float(sp->temperature);
        int rc = MLXB_CHECK(mlx_divide(&scaled, logits, temp, s)) ? 0 : -1;
        mlx_array_free(temp);
        if (rc != 0)
            goto fail;
    } else {
        if (!MLXB_CHECK(mlx_copy(&scaled, logits, s)))
            goto fail;
    }

    /* Pre-temperature, pre-filter log-probs from raw logits (matches mlx-lm) */
    mlx_array log_probs = mlx_array_new();
    if (want_logprob && logprob_out) {
        if (compute_raw_log_probs(logits, s, &log_probs) != 0) {
            mlx_array_free(log_probs);
            goto fail;
        }
    }

    /* filters: top_k, then top_p, then min_p (spec order) */
    if (sp->top_k > 0) {
        mlx_array filtered = mlx_array_new();
        if (sampler_apply_top_k(scaled, sp->top_k, s, &filtered) != 0) {
            mlx_array_free(filtered);
            mlx_array_free(log_probs);
            goto fail;
        }
        mlx_array_free(scaled);
        scaled = filtered;
    }

    if (sp->top_p < 1.0f) {
        mlx_array filtered = mlx_array_new();
        if (sampler_apply_top_p(scaled, sp->top_p, s, &filtered) != 0) {
            mlx_array_free(filtered);
            mlx_array_free(log_probs);
            goto fail;
        }
        mlx_array_free(scaled);
        scaled = filtered;
    }

    if (sp->min_p > 0.0f) {
        mlx_array filtered = mlx_array_new();
        if (sampler_apply_min_p(scaled, sp->min_p, s, &filtered) != 0) {
            mlx_array_free(filtered);
            mlx_array_free(log_probs);
            goto fail;
        }
        mlx_array_free(scaled);
        scaled = filtered;
    }

    /* categorical softmaxes internally; -inf masks become probability 0 */
    if (!MLXB_CHECK(mlx_random_categorical(&tok, scaled, -1, subkey, s))) {
        mlx_array_free(log_probs);
        goto fail;
    }
    mlx_array_free(scaled);

    if (want_logprob && logprob_out) {
        if (compute_logprob_at(log_probs, tok, s, logprob_out) != 0) {
            mlx_array_free(log_probs);
            mlx_array_free(tok);
            return -1;
        }
    }
    mlx_array_free(log_probs);
    *tok_out = tok;
    return 0;

fail:
    mlx_array_free(scaled);
    mlx_array_free(tok);
    return -1;
}
