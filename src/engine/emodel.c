#include "engine/emodel.h"
#include "engine/forward.h"
#include "core/log.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

int engine_model_load(engine_model_t *em, const char *model_dir) {
    if (!em || !model_dir) return -1;
    memset(em, 0, sizeof(*em));

    if (model_config_load(&em->cfg, model_dir) != 0) {
        log_error("emodel: failed to load config from %s", model_dir);
        return -1;
    }

    char err[256] = {0};
    if (weights_load(&em->w, model_dir, &em->cfg, err, sizeof(err)) != 0) {
        log_error("emodel: failed to load weights: %s", err);
        model_config_free(&em->cfg);
        memset(em, 0, sizeof(*em));
        return -1;
    }

    em->stream = mlxbridge_gpu_stream();
    em->attn_scale = 1.0f / sqrtf((float)em->cfg.head_dim);
    em->stub = false;
    return 0;
}

void engine_model_free(engine_model_t *em) {
    if (!em) return;
    weights_free(&em->w);
    model_config_free(&em->cfg);
    memset(em, 0, sizeof(*em));
}

int model_forward(engine_model_t *em, mlx_array ids, kvcache_t *kv,
                  bool want_logits, mlx_array *logits_out) {
    if (!em || !kv) return -1;
    int rc = -1;
    mlx_stream s = em->stream;
    const model_config_t *cfg = &em->cfg;
    const weights_t *w = &em->w;

    mlx_array h = mlx_array_new();
    mlx_array layer_out = mlx_array_new();
    mlx_array normed = mlx_array_new();
    mlx_array sliced = mlx_array_new();
    mlx_array logits = mlx_array_new();

    /* Embedding */
    if (fwd_embed(&h, ids, w, cfg, s) != 0) goto cleanup;

    /* Decoder layers */
    for (int L = 0; L < cfg->num_hidden_layers; L++) {
        if (fwd_decoder_layer(&layer_out, h, L, w, cfg, kv, s) != 0)
            goto cleanup;
        mlx_array_free(h);
        h = layer_out;
        layer_out = mlx_array_new();
    }

    if (!want_logits) {
        rc = 0;
        goto cleanup;
    }

    /* Final norm */
    char name[256], wname[270];
    weights_tensor_name(name, sizeof(name), cfg, -1, "norm");
    snprintf(wname, sizeof(wname), "%s.weight", name);
    mlx_array norm_w = mlx_array_new();
    if (weights_get(&norm_w, w, wname) != 0) {
        mlx_array_free(norm_w);
        goto cleanup;
    }
    if (fwd_rmsnorm(&normed, h, norm_w, cfg->rms_norm_eps, s) != 0) {
        mlx_array_free(norm_w);
        goto cleanup;
    }
    mlx_array_free(norm_w);

    /* Slice last position: [B, S, hidden] -> [B, 1, hidden] */
    int S = mlx_array_dim(normed, 1);
    int start[] = {0, S - 1, 0};
    int stop[] = {mlx_array_dim(normed, 0), S, cfg->hidden_size};
    int strides[] = {1, 1, 1};
    if (!MLXB_CHECK(mlx_slice(&sliced, normed, start, 3, stop, 3, strides, 3, s)))
        goto cleanup;

    /* lm_head */
    if (cfg->tie_word_embeddings) {
        weights_tensor_name(name, sizeof(name), cfg, -1, "embed_tokens");
        snprintf(wname, sizeof(wname), "%s.weight", name);
        mlx_array embed_w = mlx_array_new();
        if (weights_get(&embed_w, w, wname) != 0) {
            mlx_array_free(embed_w);
            goto cleanup;
        }
        mlx_array embed_t = mlx_array_new();
        if (!MLXB_CHECK(mlx_transpose(&embed_t, embed_w, s))) {
            mlx_array_free(embed_t);
            mlx_array_free(embed_w);
            goto cleanup;
        }
        if (!MLXB_CHECK(mlx_matmul(&logits, sliced, embed_t, s))) {
            mlx_array_free(embed_t);
            mlx_array_free(embed_w);
            goto cleanup;
        }
        mlx_array_free(embed_t);
        mlx_array_free(embed_w);
    } else {
        weight_triplet_t lm_tri;
        if (weights_get_triplet(&lm_tri, w, "lm_head") != 0)
            goto cleanup;
        if (fwd_linear(&logits, sliced, &lm_tri, cfg, false, s) != 0) {
            weights_triplet_free(&lm_tri);
            goto cleanup;
        }
        weights_triplet_free(&lm_tri);
    }

    /* Reshape to [B, vocab] */
    int B = mlx_array_dim(logits, 0);
    int reshape[] = {B, cfg->vocab_size};
    mlx_array final_logits = mlx_array_new();
    if (!MLXB_CHECK(mlx_reshape(&final_logits, logits, reshape, 2, s))) {
        mlx_array_free(final_logits);
        goto cleanup;
    }

    if (logits_out) {
        mlx_array_free(*logits_out);
        *logits_out = final_logits;
    } else {
        mlx_array_free(final_logits);
    }
    rc = 0;

cleanup:
    mlx_array_free(logits);
    mlx_array_free(sliced);
    mlx_array_free(normed);
    mlx_array_free(layer_out);
    mlx_array_free(h);
    return rc;
}
