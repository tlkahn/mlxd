#include "engine/sampler.h"

#include <math.h>

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

int sampler_sample_lazy(mlx_array logits, const sampling_params_t *sp, mlx_array subkey,
                        bool want_logprob, mlx_stream s, mlx_array *tok_out,
                        mlx_array *logprob_out) {
    (void)want_logprob;
    (void)logprob_out;

    mlx_array tok = mlx_array_new();
    if (sp->temperature < SAMPLER_GREEDY_TEMP) {
        if (!MLXB_CHECK(mlx_argmax_axis(&tok, logits, -1, false, s))) {
            mlx_array_free(tok);
            return -1;
        }
        *tok_out = tok;
        return 0;
    }

    /* temperature scale: logits / T (oracle divides; skip the no-op T == 1) */
    mlx_array scaled = mlx_array_new();
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

    /* filters: top_k, then top_p, then min_p (spec order) */
    if (sp->top_k > 0) {
        mlx_array filtered = mlx_array_new();
        if (sampler_apply_top_k(scaled, sp->top_k, s, &filtered) != 0) {
            mlx_array_free(filtered);
            goto fail;
        }
        mlx_array_free(scaled);
        scaled = filtered;
    }

    if (sp->top_p < 1.0f) {
        mlx_array filtered = mlx_array_new();
        if (sampler_apply_top_p(scaled, sp->top_p, s, &filtered) != 0) {
            mlx_array_free(filtered);
            goto fail;
        }
        mlx_array_free(scaled);
        scaled = filtered;
    }

    if (sp->min_p > 0.0f) {
        mlx_array filtered = mlx_array_new();
        if (sampler_apply_min_p(scaled, sp->min_p, s, &filtered) != 0) {
            mlx_array_free(filtered);
            goto fail;
        }
        mlx_array_free(scaled);
        scaled = filtered;
    }

    /* categorical softmaxes internally; -inf masks become probability 0 */
    if (!MLXB_CHECK(mlx_random_categorical(&tok, scaled, -1, subkey, s)))
        goto fail;
    mlx_array_free(scaled);
    *tok_out = tok;
    return 0;

fail:
    mlx_array_free(scaled);
    mlx_array_free(tok);
    return -1;
}
