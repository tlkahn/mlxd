#include "model/model.h"

#include "core/log.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <yyjson/yyjson.h>

static char *dup_str(const char *s) {
    if (!s)
        return NULL;
    size_t n = strlen(s);
    char  *p = malloc(n + 1);
    if (p)
        memcpy(p, s, n + 1);
    return p;
}

static char *path_join(const char *dir, const char *name) {
    size_t dlen = strlen(dir);
    size_t nlen = strlen(name);
    char  *p    = malloc(dlen + 1 + nlen + 1);
    if (!p)
        return NULL;
    memcpy(p, dir, dlen);
    p[dlen] = '/';
    memcpy(p + dlen + 1, name, nlen);
    p[dlen + 1 + nlen] = '\0';
    return p;
}

/* Absent = def; wrong type / n < 1 / overflow = -1. */
static int get_dim_int(yyjson_val *obj, const char *key, int *out, int def) {
    yyjson_val *v = yyjson_obj_get(obj, key);
    if (!v) {
        *out = def;
        return 0;
    }
    if (!yyjson_is_int(v))
        return -1;
    int64_t n = yyjson_get_sint(v);
    if (n < 1 || n > INT_MAX)
        return -1;
    *out = (int)n;
    return 0;
}

/* Absent = def; wrong type / n < 0 / overflow = -1. 0 is legal. */
static int get_int_nonneg(yyjson_val *obj, const char *key, int *out, int def) {
    yyjson_val *v = yyjson_obj_get(obj, key);
    if (!v) {
        *out = def;
        return 0;
    }
    if (!yyjson_is_int(v))
        return -1;
    int64_t n = yyjson_get_sint(v);
    if (n < 0 || n > INT_MAX)
        return -1;
    *out = (int)n;
    return 0;
}

/* Absent = def; accepts int or real via yyjson_is_num; wrong type = -1. */
static int get_f32(yyjson_val *obj, const char *key, float *out, float def) {
    yyjson_val *v = yyjson_obj_get(obj, key);
    if (!v) {
        *out = def;
        return 0;
    }
    if (!yyjson_is_num(v))
        return -1;
    *out = (float)yyjson_get_num(v);
    return 0;
}

/* Absent = def; wrong type = -1. */
static int get_bool(yyjson_val *obj, const char *key, bool *out, bool def) {
    yyjson_val *v = yyjson_obj_get(obj, key);
    if (!v) {
        *out = def;
        return 0;
    }
    if (!yyjson_is_bool(v))
        return -1;
    *out = yyjson_get_bool(v);
    return 0;
}

static void add_eos(model_config_t *cfg, uint32_t id) {
    for (int i = 0; i < cfg->num_eos_tokens; i++) {
        if (cfg->eos_token_ids[i] == id)
            return;
    }
    if (cfg->num_eos_tokens >= MLXD_MAX_EOS_TOKENS) {
        log_warn("eos_token_ids capped at %d, dropping id %u",
                 MLXD_MAX_EOS_TOKENS, (unsigned)id);
        return;
    }
    cfg->eos_token_ids[cfg->num_eos_tokens++] = id;
}

static void ensure_gemma_terminators(model_config_t *cfg) {
    add_eos(cfg, 1);
    add_eos(cfg, 106);
}

static int parse_eos_value(model_config_t *cfg, yyjson_val *v) {
    if (!v)
        return 0;
    if (yyjson_is_int(v)) {
        int64_t n = yyjson_get_sint(v);
        if (n < 0 || n > (int64_t)UINT32_MAX)
            return -1;
        add_eos(cfg, (uint32_t)n);
        return 0;
    }
    if (yyjson_is_arr(v)) {
        size_t idx, max;
        yyjson_val *item;
        yyjson_arr_foreach(v, idx, max, item) {
            if (!yyjson_is_int(item))
                return -1;
            int64_t n = yyjson_get_sint(item);
            if (n < 0 || n > (int64_t)UINT32_MAX)
                return -1;
            add_eos(cfg, (uint32_t)n);
        }
        return 0;
    }
    /* Other types: ignore (mlx-serve parity for non-int/array). */
    return 0;
}

static void apply_hidden_act_string(model_config_t *cfg, yyjson_val *cfg_obj) {
    yyjson_val *v = yyjson_obj_get(cfg_obj, "hidden_act");
    if (!v || !yyjson_is_str(v))
        return;
    const char *s = yyjson_get_str(v);
    if (strcmp(s, "silu") == 0)
        cfg->hidden_act = HIDDEN_ACT_SILU;
    else if (strcmp(s, "gelu_pytorch_tanh") == 0)
        cfg->hidden_act = HIDDEN_ACT_GELU_APPROX;
}

static void set_llama_style_defaults(model_config_t *cfg, yyjson_val *cfg_obj) {
    cfg->weight_prefix        = "model";
    cfg->norm_has_offset      = false;
    cfg->scale_embeddings     = false;
    cfg->has_pre_ff_norm      = false;
    cfg->has_qk_norm          = false;
    /* Preserve parsed rope_scaling_factor (llama3 block); default is already 1.0. */
    cfg->rope_local_base_freq = cfg->rope_theta;
    if (yyjson_obj_get(cfg_obj, "head_dim") == NULL && cfg->num_attention_heads > 0)
        cfg->head_dim = cfg->hidden_size / cfg->num_attention_heads;
    apply_hidden_act_string(cfg, cfg_obj);
    if (yyjson_obj_get(cfg_obj, "query_pre_attn_scalar") == NULL)
        cfg->query_pre_attn_scalar = cfg->head_dim;
}

model_family_t model_family_from_type(const char *model_type) {
    if (!model_type)
        return MODEL_FAMILY_UNKNOWN;
    if (strcmp(model_type, "gemma3") == 0 ||
        strcmp(model_type, "gemma3_text") == 0)
        return MODEL_GEMMA3;
    if (strcmp(model_type, "gemma4") == 0 ||
        strcmp(model_type, "gemma4_text") == 0 ||
        strcmp(model_type, "gemma4_unified") == 0 ||
        strcmp(model_type, "gemma4_unified_text") == 0)
        return MODEL_GEMMA4;
    if (strcmp(model_type, "qwen2") == 0)
        return MODEL_QWEN2;
    if (strcmp(model_type, "qwen3") == 0)
        return MODEL_QWEN3;
    if (strcmp(model_type, "qwen3_5") == 0 ||
        strcmp(model_type, "qwen3_5_text") == 0)
        return MODEL_QWEN3_5;
    if (strcmp(model_type, "qwen3_5_moe") == 0 ||
        strcmp(model_type, "qwen3_5_moe_text") == 0)
        return MODEL_QWEN3_5_MOE;
    if (strcmp(model_type, "llama") == 0)
        return MODEL_LLAMA;
    if (strcmp(model_type, "mistral") == 0)
        return MODEL_MISTRAL;
    if (strncmp(model_type, "lfm2", 4) == 0)
        return MODEL_LFM2;
    if (strcmp(model_type, "nemotron_h") == 0)
        return MODEL_NEMOTRON_H;
    if (strcmp(model_type, "deepseek_v4") == 0)
        return MODEL_DEEPSEEK_V4;
    if (strcmp(model_type, "bert") == 0)
        return MODEL_BERT;
    return MODEL_FAMILY_UNKNOWN;
}

bool model_layer_is_global(const model_config_t *cfg, int layer) {
    if (!cfg)
        return true;
    if (!cfg->has_sliding_window)
        return true;
    if (cfg->has_explicit_layer_types && layer >= 0 && layer < MLXD_MAX_LAYERS)
        return cfg->layer_is_global[layer];
    /* Deliberate deviation from mlx-serve which defaults pattern=6 globally,
       mis-marking mistral layers as global. pattern=0 means all-sliding. */
    if (cfg->sliding_window_pattern <= 0)
        return false;
    return (layer % cfg->sliding_window_pattern) ==
           (cfg->sliding_window_pattern - 1);
}

static void apply_family_defaults(model_config_t *cfg, yyjson_val *cfg_obj,
                                  bool has_text_config,
                                  bool sliding_window_key_present) {
    model_family_t fam = model_family_from_type(cfg->model_type);
    cfg->family        = fam;

    switch (fam) {
    case MODEL_GEMMA3:
        if (yyjson_obj_get(cfg_obj, "num_attention_heads") == NULL)
            cfg->num_attention_heads = 8;
        if (yyjson_obj_get(cfg_obj, "num_key_value_heads") == NULL)
            cfg->num_key_value_heads = 4;
        if (yyjson_obj_get(cfg_obj, "head_dim") == NULL)
            cfg->head_dim = 256;
        cfg->weight_prefix =
            has_text_config ? "language_model.model" : "model";
        cfg->tie_word_embeddings = true;
        cfg->hidden_act          = HIDDEN_ACT_GELU_APPROX;
        cfg->norm_has_offset     = true;
        cfg->scale_embeddings    = true;
        cfg->has_pre_ff_norm     = true;
        cfg->has_qk_norm         = true;
        if (!sliding_window_key_present) {
            cfg->has_sliding_window = true;
            if (cfg->sliding_window == 0)
                cfg->sliding_window = 1024;
        }
        if (yyjson_obj_get(cfg_obj, "sliding_window_pattern") == NULL)
            cfg->sliding_window_pattern = 6;
        if (cfg->rope_scaling_factor == 1.0f) {
            yyjson_val *rs = yyjson_obj_get(cfg_obj, "rope_scaling");
            if (!rs) {
                cfg->rope_scaling_factor = 8.0f;
            } else if (!yyjson_is_obj(rs)) {
                cfg->rope_scaling_factor = 8.0f;
            } else if (yyjson_obj_get(rs, "factor") == NULL) {
                cfg->rope_scaling_factor = 8.0f;
            }
        }
        if (yyjson_obj_get(cfg_obj, "query_pre_attn_scalar") == NULL)
            cfg->query_pre_attn_scalar = cfg->head_dim;
        ensure_gemma_terminators(cfg);
        break;

    case MODEL_GEMMA4:
        cfg->weight_prefix       = "language_model.model";
        cfg->hidden_act          = HIDDEN_ACT_GELU_APPROX;
        cfg->norm_has_offset     = false;
        cfg->scale_embeddings    = true;
        cfg->has_pre_ff_norm     = true;
        cfg->has_qk_norm         = true;
        cfg->has_v_norm          = true;
        cfg->rope_scaling_factor = 1.0f;
        if (yyjson_obj_get(cfg_obj, "query_pre_attn_scalar") == NULL)
            cfg->query_pre_attn_scalar = cfg->head_dim;
        ensure_gemma_terminators(cfg);
        break;

    case MODEL_QWEN3_5:
    case MODEL_QWEN3_5_MOE:
        cfg->weight_prefix =
            has_text_config ? "language_model.model" : "model";
        cfg->norm_has_offset      = false;
        cfg->scale_embeddings     = false;
        cfg->has_pre_ff_norm      = false;
        cfg->has_qk_norm          = true;
        cfg->hidden_act           = HIDDEN_ACT_SILU;
        cfg->has_sliding_window   = false;
        cfg->attn_output_gate     = true;
        cfg->rope_scaling_factor  = 1.0f;
        cfg->rope_local_base_freq = cfg->rope_theta;
        if (yyjson_obj_get(cfg_obj, "query_pre_attn_scalar") == NULL)
            cfg->query_pre_attn_scalar = cfg->head_dim;
        break;

    case MODEL_QWEN3:
        set_llama_style_defaults(cfg, cfg_obj);
        cfg->has_qk_norm = true;
        break;

    case MODEL_QWEN2:
        set_llama_style_defaults(cfg, cfg_obj);
        if (yyjson_obj_get(cfg_obj, "attention_bias") == NULL)
            cfg->attention_bias = true;
        break;

    case MODEL_LLAMA:
    case MODEL_MISTRAL:
        set_llama_style_defaults(cfg, cfg_obj);
        break;

    case MODEL_LFM2:
        cfg->weight_prefix =
            has_text_config ? "language_model.model" : "model";
        cfg->hidden_act           = HIDDEN_ACT_SILU;
        cfg->norm_has_offset      = false;
        cfg->scale_embeddings     = false;
        cfg->has_pre_ff_norm      = false;
        cfg->has_qk_norm          = true;
        cfg->has_sliding_window   = false;
        cfg->has_hybrid_layers    = true;
        cfg->rope_scaling_factor  = 1.0f;
        cfg->rope_local_base_freq = cfg->rope_theta;
        if (cfg->head_dim == 0 && cfg->num_attention_heads > 0)
            cfg->head_dim = cfg->hidden_size / cfg->num_attention_heads;
        cfg->query_pre_attn_scalar = cfg->head_dim;
        cfg->tie_word_embeddings   = true;
        {
            yyjson_val *te = yyjson_obj_get(cfg_obj, "tie_embedding");
            if (te && yyjson_is_bool(te))
                cfg->tie_word_embeddings = yyjson_get_bool(te);
        }
        if (yyjson_obj_get(cfg_obj, "rms_norm_eps") == NULL) {
            yyjson_val *ne = yyjson_obj_get(cfg_obj, "norm_eps");
            if (ne && yyjson_is_num(ne))
                cfg->rms_norm_eps = (float)yyjson_get_num(ne);
        }
        {
            yyjson_val *cd = yyjson_obj_get(cfg_obj, "conv_dim");
            if (cd && yyjson_is_int(cd)) {
                int64_t n = yyjson_get_sint(cd);
                if (n >= 0 && n <= INT_MAX)
                    cfg->lfm_conv_dim = (int)n;
            }
        }
        {
            /* HF LFM2 name for the conv kernel length. */
            yyjson_val *cl = yyjson_obj_get(cfg_obj, "conv_L_cache");
            if (cl && yyjson_is_int(cl)) {
                int64_t n = yyjson_get_sint(cl);
                if (n >= 1 && n <= INT_MAX)
                    cfg->lfm_conv_kernel = (int)n;
            }
        }
        {
            yyjson_val *lt = yyjson_obj_get(cfg_obj, "layer_types");
            if (yyjson_is_arr(lt)) {
                size_t idx, max;
                yyjson_val *item;
                yyjson_arr_foreach(lt, idx, max, item) {
                    if (idx >= (size_t)MLXD_MAX_LAYERS)
                        break;
                    if (yyjson_is_str(item) &&
                        strcmp(yyjson_get_str(item), "conv") == 0)
                        cfg->layer_kinds[idx] = LAYER_KIND_GATED_CONV;
                    else
                        cfg->layer_kinds[idx] = LAYER_KIND_ATTENTION;
                }
            }
        }
        if (cfg->num_eos_tokens == 0) {
            yyjson_val *eos = yyjson_obj_get(cfg_obj, "eos_token_id");
            (void)parse_eos_value(cfg, eos);
        }
        break;

    case MODEL_NEMOTRON_H:
        cfg->weight_prefix        = "backbone";
        cfg->hidden_act           = HIDDEN_ACT_SILU;
        cfg->mamba_mlp_act        = HIDDEN_ACT_RELU_SQ;
        cfg->norm_has_offset      = false;
        cfg->scale_embeddings     = false;
        cfg->has_pre_ff_norm      = false;
        cfg->has_qk_norm          = false;
        cfg->has_sliding_window   = false;
        cfg->has_hybrid_layers    = true;
        cfg->rope_scaling_factor  = 1.0f;
        cfg->rope_local_base_freq = cfg->rope_theta;
        cfg->query_pre_attn_scalar = cfg->head_dim;
        if (yyjson_obj_get(cfg_obj, "rms_norm_eps") == NULL) {
            yyjson_val *lne = yyjson_obj_get(cfg_obj, "layer_norm_epsilon");
            if (lne && yyjson_is_num(lne))
                cfg->rms_norm_eps = (float)yyjson_get_num(lne);
        }
        {
            int tmp;
            if (get_int_nonneg(cfg_obj, "mamba_num_heads", &tmp, 0) == 0)
                cfg->mamba_num_heads = tmp;
            if (get_int_nonneg(cfg_obj, "mamba_head_dim", &tmp, 0) == 0)
                cfg->mamba_head_dim = tmp;
            if (get_int_nonneg(cfg_obj, "n_groups", &tmp, cfg->mamba_n_groups) ==
                0)
                cfg->mamba_n_groups = tmp;
            if (get_int_nonneg(cfg_obj, "ssm_state_size", &tmp,
                               cfg->ssm_state_size) == 0)
                cfg->ssm_state_size = tmp;
            if (get_int_nonneg(cfg_obj, "conv_kernel", &tmp,
                               cfg->mamba_conv_kernel) == 0)
                cfg->mamba_conv_kernel = tmp;
            if (get_int_nonneg(cfg_obj, "expand", &tmp, cfg->mamba_expand) == 0)
                cfg->mamba_expand = tmp;
            if (get_int_nonneg(cfg_obj, "chunk_size", &tmp,
                               cfg->mamba_chunk_size) == 0)
                cfg->mamba_chunk_size = tmp;
        }
        {
            yyjson_val *tsl = yyjson_obj_get(cfg_obj, "time_step_limit");
            if (yyjson_is_arr(tsl) && yyjson_arr_size(tsl) >= 2) {
                yyjson_val *a = yyjson_arr_get(tsl, 0);
                yyjson_val *b = yyjson_arr_get(tsl, 1);
                if (a && yyjson_is_num(a))
                    cfg->time_step_min = (float)yyjson_get_num(a);
                if (b && yyjson_is_num(b))
                    cfg->time_step_max = (float)yyjson_get_num(b);
            }
        }
        {
            yyjson_val *pat = yyjson_obj_get(cfg_obj, "hybrid_override_pattern");
            if (yyjson_is_str(pat)) {
                const char *s = yyjson_get_str(pat);
                for (size_t i = 0; s[i] != '\0' && i < (size_t)MLXD_MAX_LAYERS;
                     i++) {
                    switch (s[i]) {
                    case 'M':
                        cfg->layer_kinds[i] = LAYER_KIND_MAMBA2;
                        break;
                    case '-':
                        cfg->layer_kinds[i] = LAYER_KIND_MLP;
                        break;
                    case '*':
                        cfg->layer_kinds[i] = LAYER_KIND_ATTENTION;
                        break;
                    case 'E':
                        cfg->layer_kinds[i] = LAYER_KIND_MOE;
                        break;
                    default:
                        cfg->layer_kinds[i] = LAYER_KIND_ATTENTION;
                        break;
                    }
                }
            }
        }
        if (cfg->num_eos_tokens == 0) {
            yyjson_val *eos = yyjson_obj_get(cfg_obj, "eos_token_id");
            (void)parse_eos_value(cfg, eos);
        }
        break;

    case MODEL_DEEPSEEK_V4:
        set_llama_style_defaults(cfg, cfg_obj);
        cfg->hidden_act = HIDDEN_ACT_SILU;
        break;

    case MODEL_BERT:
        cfg->is_encoder_only     = true;
        cfg->weight_prefix       = "";
        cfg->hidden_act          = HIDDEN_ACT_GELU_APPROX;
        cfg->tie_word_embeddings = true;
        cfg->has_sliding_window  = false;
        cfg->has_pre_ff_norm     = false;
        cfg->has_qk_norm         = false;
        cfg->scale_embeddings    = false;
        cfg->norm_has_offset     = false;
        if (cfg->num_attention_heads > 0)
            cfg->head_dim = cfg->hidden_size / cfg->num_attention_heads;
        cfg->num_key_value_heads   = cfg->num_attention_heads;
        cfg->query_pre_attn_scalar = cfg->head_dim;
        {
            yyjson_val *lne = yyjson_obj_get(cfg_obj, "layer_norm_eps");
            if (lne) {
                if (!yyjson_is_num(lne)) {
                    /* leave default; strict type checked below via flag */
                } else {
                    cfg->layer_norm_eps = (float)yyjson_get_num(lne);
                }
            }
        }
        {
            yyjson_val *tvs = yyjson_obj_get(cfg_obj, "type_vocab_size");
            if (tvs) {
                if (yyjson_is_int(tvs)) {
                    int64_t n = yyjson_get_sint(tvs);
                    if (n >= 0 && n <= INT_MAX)
                        cfg->type_vocab_size = (int)n;
                }
            }
        }
        break;

    case MODEL_FAMILY_UNKNOWN:
    default:
        set_llama_style_defaults(cfg, cfg_obj);
        break;
    }

    /* MoE-field discrimination: dense qwen3_5 + experts -> moe family. */
    if (cfg->family == MODEL_QWEN3_5 && cfg->num_experts > 0)
        cfg->family = MODEL_QWEN3_5_MOE;
}

static int parse_generation_config(model_config_t *cfg, const char *model_dir) {
    char *path = path_join(model_dir, "generation_config.json");
    if (!path)
        return 0; /* non-fatal */
    yyjson_doc *doc = yyjson_read_file(path, 0, NULL, NULL);
    free(path);
    if (!doc)
        return 0; /* missing / malformed = non-fatal */

    yyjson_val *root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        return 0;
    }

    yyjson_val *v;
    if ((v = yyjson_obj_get(root, "temperature")) && yyjson_is_num(v)) {
        float t = (float)yyjson_get_num(v);
        if (t >= 0.0f && t <= 2.0f) {
            cfg->has_gen_temperature = true;
            cfg->gen_temperature     = t;
        }
    }
    if ((v = yyjson_obj_get(root, "top_p")) && yyjson_is_num(v)) {
        float p = (float)yyjson_get_num(v);
        if (p > 0.0f && p <= 1.0f) {
            cfg->has_gen_top_p = true;
            cfg->gen_top_p     = p;
        }
    }
    if ((v = yyjson_obj_get(root, "top_k")) && yyjson_is_int(v)) {
        int64_t k = yyjson_get_sint(v);
        if (k > 0 && k <= 1000) {
            cfg->has_gen_top_k = true;
            cfg->gen_top_k     = (int)k;
        }
    }

    /* mlxd superset: union eos from generation_config.json */
    (void)parse_eos_value(cfg, yyjson_obj_get(root, "eos_token_id"));

    yyjson_doc_free(doc);
    return 0;
}

int model_config_load(model_config_t *cfg, const char *model_dir) {
    if (!cfg)
        return -1;
    memset(cfg, 0, sizeof(*cfg));
    if (!model_dir)
        return -1;

    /* Non-zero defaults (set before family dispatch). */
    cfg->rms_norm_eps           = 1e-6f;
    cfg->rope_theta             = 1e6f;
    cfg->rope_local_base_freq   = 1e4f;
    cfg->rope_scaling_factor    = 1.0f;
    cfg->rope_low_freq_factor   = 1.0f;
    cfg->rope_high_freq_factor  = 1.0f;
    cfg->rope_proportional_factor = 1.0f;
    cfg->partial_rotary_factor  = 1.0f;
    cfg->partial_rotary_factor_global = 1.0f;
    cfg->sliding_window_pattern = 0;
    cfg->quant_group_size       = 64;
    cfg->quant_mode             = QUANT_MODE_AFFINE;
    cfg->linear_key_head_dim    = 128;
    cfg->linear_value_head_dim  = 128;
    cfg->linear_conv_kernel_dim = 4;
    cfg->lfm_conv_kernel        = 3;
    cfg->mamba_n_groups         = 8;
    cfg->ssm_state_size         = 128;
    cfg->mamba_conv_kernel      = 4;
    cfg->mamba_expand           = 2;
    cfg->mamba_chunk_size       = 256;
    cfg->time_step_max          = INFINITY;
    cfg->mamba_mlp_act          = HIDDEN_ACT_RELU_SQ;
    cfg->layer_norm_eps         = 1e-12f;
    cfg->weight_prefix          = "model";

    char *path = path_join(model_dir, "config.json");
    if (!path)
        return -1;

    yyjson_doc *doc = yyjson_read_file(path, 0, NULL, NULL);
    free(path);
    if (!doc)
        return -1;

    yyjson_val *root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        return -1;
    }

    /* model_type is required */
    yyjson_val *mt     = yyjson_obj_get(root, "model_type");
    const char *mt_str = yyjson_get_str(mt);
    if (!mt_str) {
        yyjson_doc_free(doc);
        return -1;
    }

    cfg->model_type = dup_str(mt_str);
    if (!cfg->model_type) {
        yyjson_doc_free(doc);
        return -1;
    }

    /* architectures: first element of JSON array, optional */
    yyjson_val *archs = yyjson_obj_get(root, "architectures");
    if (yyjson_is_arr(archs)) {
        yyjson_val *first = yyjson_arr_get_first(archs);
        const char *a_str = yyjson_get_str(first);
        if (a_str) {
            cfg->architectures = dup_str(a_str);
            if (!cfg->architectures) {
                model_config_free(cfg);
                yyjson_doc_free(doc);
                return -1;
            }
        }
    }

    /* cfg_obj = text_config if present and object, else root */
    bool        has_text_config = false;
    yyjson_val *tc              = yyjson_obj_get(root, "text_config");
    yyjson_val *cfg_obj         = root;
    if (tc && yyjson_is_obj(tc)) {
        cfg_obj         = tc;
        has_text_config = true;
    }

    /* Core dims from cfg_obj */
    if (get_dim_int(cfg_obj, "vocab_size", &cfg->vocab_size, 0) ||
        get_dim_int(cfg_obj, "hidden_size", &cfg->hidden_size, 0) ||
        get_dim_int(cfg_obj, "num_hidden_layers", &cfg->num_hidden_layers, 0) ||
        get_dim_int(cfg_obj, "num_attention_heads", &cfg->num_attention_heads,
                    0) ||
        get_dim_int(cfg_obj, "num_key_value_heads", &cfg->num_key_value_heads,
                    cfg->num_attention_heads) ||
        get_dim_int(cfg_obj, "max_position_embeddings",
                    &cfg->max_position_embeddings, 0) ||
        get_int_nonneg(cfg_obj, "intermediate_size", &cfg->intermediate_size,
                       0) ||
        get_dim_int(cfg_obj, "head_dim", &cfg->head_dim, 0) ||
        get_f32(cfg_obj, "rms_norm_eps", &cfg->rms_norm_eps,
                cfg->rms_norm_eps) ||
        get_f32(cfg_obj, "rope_theta", &cfg->rope_theta, cfg->rope_theta) ||
        get_dim_int(cfg_obj, "query_pre_attn_scalar",
                    &cfg->query_pre_attn_scalar, 0) ||
        get_int_nonneg(cfg_obj, "num_experts", &cfg->num_experts, 0) ||
        get_int_nonneg(cfg_obj, "num_experts_per_tok",
                       &cfg->num_experts_per_tok, 0) ||
        get_int_nonneg(cfg_obj, "moe_intermediate_size",
                       &cfg->moe_intermediate_size, 0) ||
        get_int_nonneg(cfg_obj, "shared_expert_intermediate_size",
                       &cfg->shared_expert_intermediate_size, 0) ||
        get_int_nonneg(cfg_obj, "linear_num_key_heads",
                       &cfg->linear_num_key_heads, 0) ||
        get_int_nonneg(cfg_obj, "linear_num_value_heads",
                       &cfg->linear_num_value_heads, 0) ||
        get_int_nonneg(cfg_obj, "linear_key_head_dim",
                       &cfg->linear_key_head_dim, cfg->linear_key_head_dim) ||
        get_int_nonneg(cfg_obj, "linear_value_head_dim",
                       &cfg->linear_value_head_dim,
                       cfg->linear_value_head_dim) ||
        get_int_nonneg(cfg_obj, "linear_conv_kernel_dim",
                       &cfg->linear_conv_kernel_dim,
                       cfg->linear_conv_kernel_dim) ||
        get_int_nonneg(cfg_obj, "full_attention_interval",
                       &cfg->full_attention_interval, 0) ||
        get_bool(cfg_obj, "attn_output_gate", &cfg->attn_output_gate, false) ||
        get_dim_int(cfg_obj, "sliding_window_pattern",
                    &cfg->sliding_window_pattern,
                    cfg->sliding_window_pattern) ||
        get_f32(cfg_obj, "rope_local_base_freq", &cfg->rope_local_base_freq,
                cfg->rope_local_base_freq) ||
        get_bool(cfg_obj, "attention_bias", &cfg->attention_bias, false)) {
        model_config_free(cfg);
        yyjson_doc_free(doc);
        return -1;
    }

    /* top_k_experts alias for num_experts_per_tok */
    {
        yyjson_val *tke = yyjson_obj_get(cfg_obj, "top_k_experts");
        if (tke && yyjson_is_int(tke)) {
            int64_t n = yyjson_get_sint(tke);
            if (n < 0 || n > INT_MAX) {
                model_config_free(cfg);
                yyjson_doc_free(doc);
                return -1;
            }
            cfg->num_experts_per_tok = (int)n;
        } else if (tke && !yyjson_is_null(tke)) {
            model_config_free(cfg);
            yyjson_doc_free(doc);
            return -1;
        }
    }

    if (cfg->num_hidden_layers > MLXD_MAX_LAYERS) {
        model_config_free(cfg);
        yyjson_doc_free(doc);
        return -1;
    }

    /* sliding_window: int = enabled, null = disabled, other = -1 */
    bool        sliding_window_key_present = false;
    yyjson_val *sw = yyjson_obj_get(cfg_obj, "sliding_window");
    if (sw) {
        sliding_window_key_present = true;
        if (yyjson_is_null(sw)) {
            cfg->has_sliding_window = false;
        } else if (yyjson_is_int(sw)) {
            int64_t n = yyjson_get_sint(sw);
            if (n < 0 || n > INT_MAX) {
                model_config_free(cfg);
                yyjson_doc_free(doc);
                return -1;
            }
            cfg->sliding_window     = (int)n;
            cfg->has_sliding_window = true;
        } else {
            model_config_free(cfg);
            yyjson_doc_free(doc);
            return -1;
        }
    }

    /* rope_scaling object */
    yyjson_val *rs = yyjson_obj_get(cfg_obj, "rope_scaling");
    if (rs && !yyjson_is_null(rs)) {
        if (!yyjson_is_obj(rs)) {
            model_config_free(cfg);
            yyjson_doc_free(doc);
            return -1;
        }
        yyjson_val *rtype = yyjson_obj_get(rs, "rope_type");
        if (!rtype)
            rtype = yyjson_obj_get(rs, "type");
        if (rtype) {
            if (!yyjson_is_str(rtype)) {
                model_config_free(cfg);
                yyjson_doc_free(doc);
                return -1;
            }
            cfg->rope_scaling_type = dup_str(yyjson_get_str(rtype));
            if (!cfg->rope_scaling_type) {
                model_config_free(cfg);
                yyjson_doc_free(doc);
                return -1;
            }
        }
        if (get_f32(rs, "factor", &cfg->rope_scaling_factor,
                    cfg->rope_scaling_factor) ||
            get_f32(rs, "low_freq_factor", &cfg->rope_low_freq_factor,
                    cfg->rope_low_freq_factor) ||
            get_f32(rs, "high_freq_factor", &cfg->rope_high_freq_factor,
                    cfg->rope_high_freq_factor) ||
            get_int_nonneg(rs, "original_max_position_embeddings",
                           &cfg->rope_original_max_position_embeddings, 0)) {
            model_config_free(cfg);
            yyjson_doc_free(doc);
            return -1;
        }
    }

    /* rope_parameters: gemma4 nested blocks take priority; flat keys (qwen3_5)
       are only used when no nested sub-object is present. */
    yyjson_val *rp = yyjson_obj_get(cfg_obj, "rope_parameters");
    if (rp && yyjson_is_obj(rp)) {
        yyjson_val *fa = yyjson_obj_get(rp, "full_attention");
        yyjson_val *sa = yyjson_obj_get(rp, "sliding_attention");
        bool has_nested = (fa && yyjson_is_obj(fa)) || (sa && yyjson_is_obj(sa));
        if (fa && yyjson_is_obj(fa)) {
            if (get_f32(fa, "rope_theta", &cfg->rope_theta, cfg->rope_theta)) {
                model_config_free(cfg);
                yyjson_doc_free(doc);
                return -1;
            }
            if (get_f32(fa, "partial_rotary_factor",
                        &cfg->partial_rotary_factor_global,
                        cfg->partial_rotary_factor_global)) {
                model_config_free(cfg);
                yyjson_doc_free(doc);
                return -1;
            }
            yyjson_val *rt = yyjson_obj_get(fa, "rope_type");
            if (rt && yyjson_is_str(rt) &&
                strcmp(yyjson_get_str(rt), "proportional") == 0) {
                cfg->rope_proportional = true;
                if (get_f32(fa, "factor", &cfg->rope_proportional_factor,
                            cfg->rope_proportional_factor)) {
                    model_config_free(cfg);
                    yyjson_doc_free(doc);
                    return -1;
                }
            }
        }
        if (sa && yyjson_is_obj(sa)) {
            if (get_f32(sa, "rope_theta", &cfg->rope_local_base_freq,
                        cfg->rope_local_base_freq)) {
                model_config_free(cfg);
                yyjson_doc_free(doc);
                return -1;
            }
        }
        if (!has_nested) {
            if (get_f32(rp, "rope_theta", &cfg->rope_theta, cfg->rope_theta) ||
                get_f32(rp, "partial_rotary_factor",
                        &cfg->partial_rotary_factor,
                        cfg->partial_rotary_factor)) {
                model_config_free(cfg);
                yyjson_doc_free(doc);
                return -1;
            }
        }
    } else if (rp && !yyjson_is_null(rp)) {
        model_config_free(cfg);
        yyjson_doc_free(doc);
        return -1;
    }

    /* layer_types -> layer_is_global (gemma4) */
    yyjson_val *lt = yyjson_obj_get(cfg_obj, "layer_types");
    if (yyjson_is_arr(lt)) {
        cfg->has_explicit_layer_types = true;
        size_t idx, max;
        yyjson_val *item;
        yyjson_arr_foreach(lt, idx, max, item) {
            if (idx >= (size_t)MLXD_MAX_LAYERS)
                break;
            if (yyjson_is_str(item) &&
                strcmp(yyjson_get_str(item), "full_attention") == 0)
                cfg->layer_is_global[idx] = true;
            else
                cfg->layer_is_global[idx] = false;
        }
    }

    /* gemma4 extras from cfg_obj */
    if (get_int_nonneg(cfg_obj, "global_head_dim", &cfg->global_head_dim, 0) ||
        get_int_nonneg(cfg_obj, "num_global_key_value_heads",
                       &cfg->num_global_key_value_heads, 0) ||
        get_int_nonneg(cfg_obj, "num_kv_shared_layers",
                       &cfg->num_kv_shared_layers, 0) ||
        get_bool(cfg_obj, "attention_k_eq_v", &cfg->attention_k_eq_v, false) ||
        get_f32(cfg_obj, "final_logit_softcapping",
                &cfg->final_logit_softcapping, 0.0f) ||
        get_int_nonneg(cfg_obj, "hidden_size_per_layer_input",
                       &cfg->hidden_size_per_layer_input, 0)) {
        model_config_free(cfg);
        yyjson_doc_free(doc);
        return -1;
    }

    /* tie_word_embeddings: root then cfg_obj override */
    if (get_bool(root, "tie_word_embeddings", &cfg->tie_word_embeddings,
                 false)) {
        model_config_free(cfg);
        yyjson_doc_free(doc);
        return -1;
    }
    if (cfg_obj != root) {
        yyjson_val *tw = yyjson_obj_get(cfg_obj, "tie_word_embeddings");
        if (tw) {
            if (!yyjson_is_bool(tw)) {
                model_config_free(cfg);
                yyjson_doc_free(doc);
                return -1;
            }
            cfg->tie_word_embeddings = yyjson_get_bool(tw);
        }
    }

    /* max_position_embeddings root fallback when cfg_obj gave 0 */
    if (cfg->max_position_embeddings == 0 && cfg_obj != root) {
        if (get_dim_int(root, "max_position_embeddings",
                        &cfg->max_position_embeddings, 0)) {
            model_config_free(cfg);
            yyjson_doc_free(doc);
            return -1;
        }
    }

    /* quantization (root-only) */
    yyjson_val *q = yyjson_obj_get(root, "quantization");
    if (q && !yyjson_is_null(q)) {
        if (!yyjson_is_obj(q)) {
            model_config_free(cfg);
            yyjson_doc_free(doc);
            return -1;
        }
        if (get_int_nonneg(q, "bits", &cfg->quant_bits, 0) ||
            get_int_nonneg(q, "group_size", &cfg->quant_group_size,
                           cfg->quant_group_size)) {
            model_config_free(cfg);
            yyjson_doc_free(doc);
            return -1;
        }
        yyjson_val *mode = yyjson_obj_get(q, "mode");
        if (mode) {
            if (!yyjson_is_str(mode)) {
                model_config_free(cfg);
                yyjson_doc_free(doc);
                return -1;
            }
            const char *ms = yyjson_get_str(mode);
            if (strcmp(ms, "affine") != 0) {
                log_error("unsupported quantization mode '%s' "
                          "(only affine is supported)",
                          ms);
                model_config_free(cfg);
                yyjson_doc_free(doc);
                return -1;
            }
            cfg->quant_mode = QUANT_MODE_AFFINE;
        }
    }

    /* eos_token_id (root) */
    if (parse_eos_value(cfg, yyjson_obj_get(root, "eos_token_id"))) {
        model_config_free(cfg);
        yyjson_doc_free(doc);
        return -1;
    }

    /* bert type_vocab_size strict wrong-type check (absent = 0) */
    if (model_family_from_type(cfg->model_type) == MODEL_BERT) {
        yyjson_val *tvs = yyjson_obj_get(cfg_obj, "type_vocab_size");
        if (tvs && !yyjson_is_int(tvs) && !yyjson_is_null(tvs)) {
            model_config_free(cfg);
            yyjson_doc_free(doc);
            return -1;
        }
        yyjson_val *lne = yyjson_obj_get(cfg_obj, "layer_norm_eps");
        if (lne && !yyjson_is_num(lne) && !yyjson_is_null(lne)) {
            model_config_free(cfg);
            yyjson_doc_free(doc);
            return -1;
        }
    }

    apply_family_defaults(cfg, cfg_obj, has_text_config,
                          sliding_window_key_present);

    yyjson_doc_free(doc);

    /* generation_config.json: missing/malformed non-fatal */
    (void)parse_generation_config(cfg, model_dir);

    return 0;
}

void model_config_free(model_config_t *cfg) {
    if (!cfg)
        return;
    free(cfg->model_type);
    cfg->model_type = NULL;
    free(cfg->architectures);
    cfg->architectures = NULL;
    free(cfg->rope_scaling_type);
    cfg->rope_scaling_type = NULL;
}
