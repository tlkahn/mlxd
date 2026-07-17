#ifndef MLXD_MODEL_H
#define MLXD_MODEL_H

#include <stdbool.h>
#include <stdint.h>

#define MLXD_MAX_LAYERS 128
#define MLXD_MAX_EOS_TOKENS 8

typedef enum {
    MODEL_FAMILY_UNKNOWN = 0,
    MODEL_GEMMA3,
    MODEL_GEMMA4,
    MODEL_QWEN2,
    MODEL_QWEN3,
    MODEL_QWEN3_5,
    MODEL_QWEN3_5_MOE,
    MODEL_LLAMA,
    MODEL_MISTRAL,
    MODEL_LFM2,
    MODEL_NEMOTRON_H,
    MODEL_DEEPSEEK_V4,
    MODEL_BERT,
} model_family_t;

typedef enum {
    HIDDEN_ACT_SILU = 0,
    HIDDEN_ACT_GELU_APPROX,
    HIDDEN_ACT_RELU_SQ,
} hidden_act_t;

typedef enum {
    QUANT_MODE_AFFINE = 0,
} quant_mode_t;

typedef enum {
    LAYER_KIND_ATTENTION = 0,
    LAYER_KIND_GATED_CONV,
    LAYER_KIND_MAMBA2,
    LAYER_KIND_MLP,
    LAYER_KIND_MOE,
} layer_kind_t;

typedef struct {
    /* Existing identity / core dims (unchanged order) */
    char *model_type;
    char *architectures;
    int   vocab_size;
    int   hidden_size;
    int   num_hidden_layers;
    int   num_attention_heads;
    int   num_key_value_heads;
    int   max_position_embeddings;

    /* Identity */
    model_family_t family;
    const char    *weight_prefix; /* static string, never freed */

    /* Core dims */
    int   intermediate_size;
    int   head_dim;
    float rms_norm_eps;
    int   query_pre_attn_scalar;

    /* RoPE */
    float  rope_theta;
    float  rope_local_base_freq;
    char  *rope_scaling_type; /* owned; NULL = none */
    float  rope_scaling_factor;
    float  rope_low_freq_factor;
    float  rope_high_freq_factor;
    int    rope_original_max_position_embeddings;
    bool   rope_proportional;
    float  rope_proportional_factor;
    float  partial_rotary_factor;
    float  partial_rotary_factor_global;

    /* Sliding window */
    bool has_sliding_window;
    int  sliding_window;
    /* gemma convention: layers (pattern-1), (2*pattern-1), ... are global;
       0 = no global layers when sliding window is enabled. */
    int  sliding_window_pattern;
    bool has_explicit_layer_types;
    bool layer_is_global[MLXD_MAX_LAYERS];

    /* Attention */
    bool attention_bias;

    /* Quantization */
    int          quant_bits;
    int          quant_group_size;
    quant_mode_t quant_mode;

    /* Behavioral flags */
    bool         tie_word_embeddings;
    hidden_act_t hidden_act;
    bool         norm_has_offset;
    bool         scale_embeddings;
    bool         has_pre_ff_norm;
    bool         has_qk_norm;

    /* MoE */
    int num_experts;
    int num_experts_per_tok;
    int moe_intermediate_size;
    int shared_expert_intermediate_size;

    /* Linear attention (qwen3_5) */
    int  linear_num_key_heads;
    int  linear_num_value_heads;
    int  linear_key_head_dim;
    int  linear_value_head_dim;
    int  linear_conv_kernel_dim;
    int  full_attention_interval;
    bool attn_output_gate;

    /* Hybrid layers */
    bool          has_hybrid_layers;
    layer_kind_t  layer_kinds[MLXD_MAX_LAYERS];

    /* lfm2 */
    int lfm_conv_kernel;
    int lfm_conv_dim;

    /* Mamba2 (nemotron_h) */
    int          mamba_num_heads;
    int          mamba_head_dim;
    int          mamba_n_groups;
    int          ssm_state_size;
    int          mamba_conv_kernel;
    int          mamba_expand;
    float        time_step_min;
    float        time_step_max;
    int          mamba_chunk_size;
    hidden_act_t mamba_mlp_act;

    /* gemma4 extras */
    int   global_head_dim;
    int   num_global_key_value_heads;
    int   num_kv_shared_layers;
    float final_logit_softcapping;
    int   hidden_size_per_layer_input;
    bool  has_v_norm;
    bool  attention_k_eq_v;

    /* bert */
    bool  is_encoder_only;
    float layer_norm_eps;
    int   type_vocab_size;

    /* EOS */
    uint32_t eos_token_ids[MLXD_MAX_EOS_TOKENS];
    int      num_eos_tokens;

    /* generation_config.json sampling defaults */
    bool  has_gen_temperature;
    float gen_temperature;
    bool  has_gen_top_p;
    float gen_top_p;
    bool  has_gen_top_k;
    int   gen_top_k;
} model_config_t;

/* Map model_type string to family enum. NULL / unknown -> MODEL_FAMILY_UNKNOWN. */
model_family_t model_family_from_type(const char *model_type);

/* Global-attention layer predicate: explicit map when present, else gemma
 * sliding-window pattern convention. */
bool model_layer_is_global(const model_config_t *cfg, int layer);

/* Load model config from a directory (reads config.json + generation_config.json).
 * Returns 0 on success, -1 on error. */
int model_config_load(model_config_t *cfg, const char *model_dir);

void model_config_free(model_config_t *cfg);

#endif
