#include "engine/emodel.h"
#include "engine/forward.h"
#include "core/log.h"

#include <stdio.h>
#include <string.h>


#define REJECT(cond, msg) do { \
    if (cond) { \
        if (err && errlen > 0) snprintf(err, errlen, "%s", msg); \
        return -1; \
    } \
} while (0)

/* Flags that relax reject_dense_common for families that need the feature. */
enum {
    REJECT_ALLOW_PARTIAL_ROPE    = 1 << 0, /* skip partial_rotary_factor != 1 */
    REJECT_ALLOW_ATTN_GATE       = 1 << 1, /* skip attn_output_gate reject */
    REJECT_ALLOW_SLIDING_WINDOW  = 1 << 2, /* skip has_sliding_window reject */
};

static int reject_dense_common(const model_config_t *cfg,
                                char *err, size_t errlen,
                                unsigned flags) {
    REJECT(cfg->attention_bias,
           "attention_bias not supported");
    if (!(flags & REJECT_ALLOW_SLIDING_WINDOW))
        REJECT(cfg->has_sliding_window,
               "sliding window attention not supported");
    REJECT(cfg->num_experts > 0,
           "MoE models not supported");
    REJECT(cfg->has_hybrid_layers,
           "hybrid layer architectures not supported");
    REJECT(cfg->hidden_act != HIDDEN_ACT_SILU,
           "only SiLU activation supported");
    REJECT(cfg->norm_has_offset,
           "norm_has_offset not supported");
    REJECT(cfg->scale_embeddings,
           "scale_embeddings not supported");
    REJECT(cfg->has_pre_ff_norm,
           "pre-feedforward norm not supported");
    if (!(flags & REJECT_ALLOW_PARTIAL_ROPE))
        REJECT(cfg->partial_rotary_factor != 1.0f,
               "partial rotary embedding not supported");
    if (!(flags & REJECT_ALLOW_ATTN_GATE))
        REJECT(cfg->attn_output_gate,
               "attention output gate not supported");
    return 0;
}

int engine_model_check_supported(const model_config_t *cfg,
                                 char *err, size_t errlen) {
    if (!cfg) return -1;

    switch (cfg->family) {
    case MODEL_QWEN3:
        if (reject_dense_common(cfg, err, errlen, 0) != 0) return -1;
        REJECT(cfg->rope_scaling_type != NULL,
               "RoPE scaling not supported");
        return 0;

    case MODEL_LLAMA:
        if (reject_dense_common(cfg, err, errlen, 0) != 0) return -1;
        REJECT(cfg->rope_scaling_type != NULL &&
               strcmp(cfg->rope_scaling_type, "linear") != 0 &&
               strcmp(cfg->rope_scaling_type, "llama3") != 0 &&
               strcmp(cfg->rope_scaling_type, "default") != 0,
               "unsupported RoPE scaling type");
        if (cfg->rope_scaling_type &&
            strcmp(cfg->rope_scaling_type, "llama3") == 0) {
            REJECT(cfg->rope_scaling_factor <= 0.0f,
                   "llama3 rope_scaling_factor must be > 0");
            REJECT(cfg->rope_original_max_position_embeddings <= 0,
                   "llama3 rope_original_max_position_embeddings must be > 0");
            REJECT(cfg->rope_low_freq_factor <= 0.0f,
                   "llama3 rope_low_freq_factor must be > 0");
            REJECT(cfg->rope_high_freq_factor <= cfg->rope_low_freq_factor,
                   "llama3 rope_high_freq_factor must be > rope_low_freq_factor");
        }
        REJECT(cfg->has_qk_norm,
               "qk_norm not supported for llama");
        return 0;

    case MODEL_MISTRAL:
        /* llama-shaped dense + optional uniform sliding window (pattern<=0
           => all layers local via model_layer_is_global). v0.2+ null-window
           configs are allowed and behave llama-identically. */
        if (reject_dense_common(cfg, err, errlen,
                                REJECT_ALLOW_SLIDING_WINDOW) != 0)
            return -1;
        REJECT(cfg->has_qk_norm,
               "qk_norm not supported for mistral");
        REJECT(cfg->rope_scaling_type != NULL,
               "RoPE scaling not supported");
        return 0;

    case MODEL_GEMMA4:
        REJECT(cfg->num_experts > 0,
               "MoE models not supported");
        REJECT(cfg->use_double_wide_mlp,
               "use_double_wide_mlp not supported (E2B-class unverified)");
        REJECT(cfg->hidden_act != HIDDEN_ACT_GELU_APPROX,
               "gemma4 requires gelu_pytorch_tanh activation");
        REJECT(cfg->num_kv_shared_layers >= cfg->num_hidden_layers,
               "num_kv_shared_layers must be < num_hidden_layers");
        REJECT(cfg->rope_proportional &&
               cfg->rope_proportional_factor <= 0.0f,
               "rope_proportional_factor must be > 0");
        REJECT(cfg->partial_rotary_factor_global <= 0.0f ||
               cfg->partial_rotary_factor_global > 1.0f,
               "partial_rotary_factor_global must be in (0, 1]");
        /* Plain partial_rotary_factor is the full-attn dims path (qwen3_5).
           gemma4 uses partial_rotary_factor_global via the freqs path instead. */
        REJECT(cfg->partial_rotary_factor != 1.0f,
               "partial rotary embedding not supported (gemma4 uses rope_parameters.full_attention)");
        /* Reject configs where a KV-shared layer has no same-type source */
        for (int i = 0; i < cfg->num_hidden_layers; i++) {
            if (model_layer_kv_shared(cfg, i) &&
                model_kv_source_layer(cfg, i) < 0) {
                REJECT(true,
                       "KV-shared layer has no same-type source");
            }
        }
        return 0;

    case MODEL_QWEN3_5: {
        /* Pure-dense qwen3_5 only. Common dense rejects apply, but partial rope
           and attn_output_gate are required features here (not rejects).
           Hybrid linear-attention / MoE stay Stage E. */
        if (reject_dense_common(cfg, err, errlen,
                                REJECT_ALLOW_PARTIAL_ROPE |
                                REJECT_ALLOW_ATTN_GATE) != 0)
            return -1;
        REJECT(cfg->linear_num_key_heads > 0 || cfg->linear_num_value_heads > 0,
               "linear attention / hybrid layers not supported");
        if (cfg->has_explicit_layer_types) {
            for (int i = 0; i < cfg->num_hidden_layers; i++) {
                REJECT(!cfg->layer_is_global[i],
                       "linear attention / hybrid layers not supported");
            }
        }
        REJECT(cfg->partial_rotary_factor <= 0.0f ||
               cfg->partial_rotary_factor > 1.0f,
               "partial_rotary_factor must be in (0, 1]");
        /* mlx.fast.rope requires dims even and positive. Mirror mlx-lm's
           int(head_dim * prf) and fail fast on odd/degenerate products. */
        {
            int dims = (int)((float)cfg->head_dim * cfg->partial_rotary_factor);
            REJECT(dims < 2 || (dims & 1),
                   "partial_rotary_factor yields invalid rope dims (must be even, >= 2)");
        }
        /* interval==0 unset, interval==1 pure dense (every layer full-attn).
           interval>1 implies hybrid: is_linear = (layer_idx+1) % interval != 0. */
        REJECT(cfg->full_attention_interval > 1,
               "full_attention_interval > 1 implies hybrid layers; not supported");
        REJECT(cfg->rope_scaling_type != NULL,
               "RoPE scaling not supported");
        /* Gate is mandatory: mlx-lm gates unconditionally; no oracle or real
           checkpoint for gate-off. Explicit false is parsed but unsupported. */
        REJECT(!cfg->attn_output_gate,
               "qwen3_5 requires attn_output_gate");
        return 0;
    }

    default:
        REJECT(true,
               "unsupported model family (only qwen3/llama/gemma4/qwen3_5/mistral dense supported)");
    }
}

#undef REJECT

/* test_forward_gemma3_gpu.c:load_gemma3 mirrors this path minus the support
   gate; keep in sync until gemma3 is whitelisted. */
int engine_model_load(engine_model_t *em, const char *model_dir,
                      char *err, size_t errlen) {
    if (!em || !model_dir) return -1;
    memset(em, 0, sizeof(*em));

    if (model_config_load(&em->cfg, model_dir) != 0) {
        if (err && errlen > 0)
            snprintf(err, errlen, "failed to load config from %s", model_dir);
        return -1;
    }

    if (engine_model_check_supported(&em->cfg, err, errlen) != 0) {
        model_config_free(&em->cfg);
        memset(em, 0, sizeof(*em));
        return -1;
    }

    if (weights_load(&em->w, model_dir, &em->cfg, err, errlen) != 0) {
        model_config_free(&em->cfg);
        memset(em, 0, sizeof(*em));
        return -1;
    }

    em->rope_freqs = (mlx_array){.ctx = NULL};
    if (fwd_rope_freqs_build(&em->rope_freqs, &em->cfg) != 0) {
        if (err && errlen > 0)
            snprintf(err, errlen, "failed to build rope freqs");
        weights_free(&em->w);
        model_config_free(&em->cfg);
        memset(em, 0, sizeof(*em));
        return -1;
    }

    em->stream = mlxbridge_gpu_stream();
    return 0;
}

void engine_model_free(engine_model_t *em) {
    if (!em) return;
    if (em->rope_freqs.ctx) mlx_array_free(em->rope_freqs);
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
    mlx_array ple = (mlx_array){.ctx = NULL};

    /* Embedding */
    if (fwd_embed(&h, ids, w, cfg, s) != 0) goto cleanup;

    /* PLE: compute ple_inputs once after embedding */
    if (cfg->hidden_size_per_layer_input > 0) {
        ple = mlx_array_new();
        if (fwd_ple_inputs(&ple, ids, h, w, cfg, s) != 0)
            goto cleanup;
    }

    /* Decoder layers */
    for (int L = 0; L < cfg->num_hidden_layers; L++) {
        if (fwd_decoder_layer(&layer_out, h, L, w, cfg, kv, em->rope_freqs, ple, s) != 0)
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
    if (fwd_rmsnorm(&normed, h, norm_w, cfg->rms_norm_eps,
                    cfg->norm_has_offset, s) != 0) {
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
    {
        const char *lm_base = cfg->tie_word_embeddings
            ? name  /* reuse name buffer */
            : "lm_head";
        if (cfg->tie_word_embeddings)
            weights_tensor_name(name, sizeof(name), cfg, -1, "embed_tokens");
        weight_triplet_t lm_tri;
        if (weights_get_triplet(&lm_tri, w, lm_base) != 0)
            goto cleanup;
        if (fwd_linear(&logits, sliced, &lm_tri, cfg, s) != 0) {
            weights_triplet_free(&lm_tri);
            goto cleanup;
        }
        weights_triplet_free(&lm_tri);
    }

    /* Logit softcapping */
    if (cfg->final_logit_softcapping > 0) {
        mlx_array capped = mlx_array_new();
        if (fwd_softcap(&capped, logits, cfg->final_logit_softcapping, s) != 0) {
            mlx_array_free(capped);
            goto cleanup;
        }
        mlx_array_free(logits);
        logits = capped;
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
    if (ple.ctx) mlx_array_free(ple);
    return rc;
}
