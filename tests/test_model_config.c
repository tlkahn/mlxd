#include "model/model.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MLXD_FIXTURES_DIR
#error "MLXD_FIXTURES_DIR must be defined"
#endif

/* --- Cycle I1: happy path ------------------------------------------------- */

static void test_happy_path(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config");
    assert(rc == 0);

    assert(cfg.model_type != NULL);
    assert(strcmp(cfg.model_type, "llama") == 0);

    assert(cfg.architectures != NULL);
    assert(strcmp(cfg.architectures, "LlamaForCausalLM") == 0);

    assert(cfg.vocab_size == 32000);
    assert(cfg.hidden_size == 4096);
    assert(cfg.num_hidden_layers == 32);
    assert(cfg.num_attention_heads == 32);
    assert(cfg.max_position_embeddings == 2048);

    model_config_free(&cfg);
}

/* --- Cycle I2: kv-heads defaults to num_attention_heads ------------------ */

static void test_kv_heads_default(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config");
    assert(rc == 0);
    assert(cfg.num_key_value_heads == 32);
    model_config_free(&cfg);
}

/* --- Cycle I3: missing config / NULL args -------------------------------- */

static void test_missing_config(void) {
    model_config_t cfg;
    assert(model_config_load(&cfg, MLXD_FIXTURES_DIR "/no_such_dir") == -1);
    assert(model_config_load(NULL, MLXD_FIXTURES_DIR "/model_config") == -1);

    /* NULL model_dir must zero cfg so free-after-failure is safe */
    memset(&cfg, 0xAB, sizeof(cfg));
    assert(model_config_load(&cfg, NULL) == -1);
    assert(cfg.model_type == NULL);
    assert(cfg.architectures == NULL);
    model_config_free(&cfg);
}

/* --- Cycle I4: missing model_type ---------------------------------------- */

static void test_missing_model_type(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_no_type");
    assert(rc == -1);
    /* cfg must be safe to free after a failed load */
    model_config_free(&cfg);
}

/* --- Cycle I6: corrupt JSON ---------------------------------------------- */

static void test_bad_json(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_bad_json");
    assert(rc == -1);
    model_config_free(&cfg);
}

/* --- Cycle I7: root is not an object ------------------------------------- */

static void test_root_not_object(void) {
    model_config_t cfg;
    int rc =
        model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_root_array");
    assert(rc == -1);
    model_config_free(&cfg);
}

/* --- Cycle I8: model_type present but not a string ----------------------- */

static void test_model_type_not_string(void) {
    model_config_t cfg;
    int rc = model_config_load(
        &cfg, MLXD_FIXTURES_DIR "/model_config_type_not_string");
    assert(rc == -1);
    model_config_free(&cfg);
}

/* --- Cycle I9: reject malformed numeric fields --------------------------- */

static void test_malformed_numerics(void) {
    struct {
        const char *fixture;
        const char *label;
    } cases[] = {
        {MLXD_FIXTURES_DIR "/model_config_num_wrong_type", "wrong type"},
        {MLXD_FIXTURES_DIR "/model_config_num_negative", "negative"},
        {MLXD_FIXTURES_DIR "/model_config_num_zero", "zero"},
        {MLXD_FIXTURES_DIR "/model_config_num_overflow", "overflow"},
        {MLXD_FIXTURES_DIR "/model_config_kv_wrong_type", "kv wrong type"},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        model_config_t cfg;
        memset(&cfg, 0xAB, sizeof(cfg));
        int rc = model_config_load(&cfg, cases[i].fixture);
        assert(rc == -1);
        model_config_free(&cfg);
    }
}

/* --- Cycle I5: free semantics -------------------------------------------- */

static void test_free_semantics(void) {
    /* double free after successful load */
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config");
    assert(rc == 0);
    model_config_free(&cfg);
    model_config_free(&cfg);

    /* NULL argument is a no-op */
    model_config_free(NULL);

    /* free after failed load is safe (poison to catch missing zeroing) */
    model_config_t cfg2;
    memset(&cfg2, 0xAB, sizeof(cfg2));
    assert(model_config_load(&cfg2, MLXD_FIXTURES_DIR "/no_such_dir") == -1);
    model_config_free(&cfg2);
}

/* --- A1 Cycle 1: extended core dims + defaults --------------------------- */

static void test_extended_core_dims(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_qwen3");
    assert(rc == 0);
    assert(cfg.hidden_size == 1024);
    assert(cfg.num_attention_heads == 16);
    assert(cfg.num_key_value_heads == 8);
    assert(cfg.head_dim == 128);
    assert(cfg.intermediate_size == 3072);
    assert(cfg.rope_theta == 5000000.0f);
    assert(cfg.rms_norm_eps == 1e-5f);
    assert(cfg.tie_word_embeddings == true);
    assert(cfg.family == MODEL_QWEN3);
    assert(cfg.weight_prefix != NULL && strcmp(cfg.weight_prefix, "model") == 0);
    assert(cfg.has_qk_norm == true);
    model_config_free(&cfg);

    /* llama fixture defaults */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config");
    assert(rc == 0);
    assert(cfg.family == MODEL_LLAMA);
    assert(cfg.head_dim == 128); /* 4096/32 */
    assert(cfg.rope_theta == 1e6f);
    assert(cfg.rms_norm_eps == 1e-6f);
    assert(cfg.attention_bias == false);
    assert(cfg.intermediate_size == 0);
    assert(cfg.weight_prefix != NULL && strcmp(cfg.weight_prefix, "model") == 0);
    model_config_free(&cfg);

    /* wrong-type float rejected */
    memset(&cfg, 0xAB, sizeof(cfg));
    rc = model_config_load(&cfg,
                           MLXD_FIXTURES_DIR "/model_config_float_wrong_type");
    assert(rc == -1);
    model_config_free(&cfg);
}

/* --- A1 Cycle 2: rope scaling block -------------------------------------- */

static void test_rope_scaling(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_llama3");
    assert(rc == 0);
    assert(cfg.rope_theta == 500000.0f);
    assert(cfg.rope_scaling_type != NULL);
    assert(strcmp(cfg.rope_scaling_type, "llama3") == 0);
    assert(cfg.rope_scaling_factor == 8.0f);
    assert(cfg.rope_low_freq_factor == 1.0f);
    assert(cfg.rope_high_freq_factor == 4.0f);
    assert(cfg.rope_original_max_position_embeddings == 8192);
    model_config_free(&cfg);

    /* absent on llama fixture */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config");
    assert(rc == 0);
    assert(cfg.rope_scaling_type == NULL);
    assert(cfg.rope_scaling_factor == 1.0f);
    model_config_free(&cfg);

    /* type key accepted when rope_type absent */
    rc = model_config_load(&cfg,
                           MLXD_FIXTURES_DIR "/model_config_rope_type_alias");
    assert(rc == 0);
    assert(cfg.rope_scaling_type != NULL);
    assert(strcmp(cfg.rope_scaling_type, "linear") == 0);
    assert(cfg.rope_scaling_factor == 2.0f);
    model_config_free(&cfg);

    /* wrong type rejected */
    memset(&cfg, 0xAB, sizeof(cfg));
    rc = model_config_load(
        &cfg, MLXD_FIXTURES_DIR "/model_config_rope_scaling_wrong_type");
    assert(rc == -1);
    model_config_free(&cfg);
}

/* --- A1 Cycle 3: quantization -------------------------------------------- */

static void test_quantization(void) {
    model_config_t cfg;
    int rc =
        model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_quant_affine");
    assert(rc == 0);
    assert(cfg.quant_bits == 4);
    assert(cfg.quant_group_size == 32);
    assert(cfg.quant_mode == QUANT_MODE_AFFINE);
    model_config_free(&cfg);

    /* absent mode key = affine */
    rc = model_config_load(&cfg,
                           MLXD_FIXTURES_DIR "/model_config_quant_no_mode");
    assert(rc == 0);
    assert(cfg.quant_bits == 4);
    assert(cfg.quant_group_size == 64);
    assert(cfg.quant_mode == QUANT_MODE_AFFINE);
    model_config_free(&cfg);

    /* absent quantization = bits 0 + group 64 + AFFINE */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config");
    assert(rc == 0);
    assert(cfg.quant_bits == 0);
    assert(cfg.quant_group_size == 64);
    assert(cfg.quant_mode == QUANT_MODE_AFFINE);
    model_config_free(&cfg);

    const char *bad[] = {
        MLXD_FIXTURES_DIR "/model_config_quant_nvfp4",
        MLXD_FIXTURES_DIR "/model_config_quant_mxfp4",
        MLXD_FIXTURES_DIR "/model_config_quant_mxfp8",
        MLXD_FIXTURES_DIR "/model_config_quant_unknown",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        memset(&cfg, 0xAB, sizeof(cfg));
        rc = model_config_load(&cfg, bad[i]);
        assert(rc == -1);
        model_config_free(&cfg);
    }
}

/* --- A1 Cycle 4: model_family_from_type ---------------------------------- */

static void test_family_from_type(void) {
    assert(model_family_from_type("gemma3") == MODEL_GEMMA3);
    assert(model_family_from_type("gemma3_text") == MODEL_GEMMA3);
    assert(model_family_from_type("gemma4") == MODEL_GEMMA4);
    assert(model_family_from_type("gemma4_unified_text") == MODEL_GEMMA4);
    assert(model_family_from_type("qwen2") == MODEL_QWEN2);
    assert(model_family_from_type("qwen3") == MODEL_QWEN3);
    assert(model_family_from_type("qwen3_5") == MODEL_QWEN3_5);
    assert(model_family_from_type("qwen3_5_text") == MODEL_QWEN3_5);
    assert(model_family_from_type("qwen3_5_moe") == MODEL_QWEN3_5_MOE);
    assert(model_family_from_type("qwen3_5_moe_text") == MODEL_QWEN3_5_MOE);
    assert(model_family_from_type("llama") == MODEL_LLAMA);
    assert(model_family_from_type("mistral") == MODEL_MISTRAL);
    assert(model_family_from_type("lfm2") == MODEL_LFM2);
    assert(model_family_from_type("lfm2_vl") == MODEL_LFM2);
    assert(model_family_from_type("lfm2_moe") == MODEL_LFM2);
    assert(model_family_from_type("lfm2_audio") == MODEL_LFM2);
    assert(model_family_from_type("nemotron_h") == MODEL_NEMOTRON_H);
    assert(model_family_from_type("deepseek_v3") == MODEL_DEEPSEEK_V4);
    assert(model_family_from_type("deepseek_v3_2") == MODEL_DEEPSEEK_V4);
    assert(model_family_from_type("deepseek_v32") == MODEL_DEEPSEEK_V4);
    assert(model_family_from_type("deepseek_v4") == MODEL_DEEPSEEK_V4);
    assert(model_family_from_type("bert") == MODEL_BERT);

    assert(model_family_from_type("qwen3_moe") == MODEL_FAMILY_UNKNOWN);
    assert(model_family_from_type("qwen3_next") == MODEL_FAMILY_UNKNOWN);
    assert(model_family_from_type("diffusion_gemma") == MODEL_FAMILY_UNKNOWN);
    assert(model_family_from_type("gpt2") == MODEL_FAMILY_UNKNOWN);
    assert(model_family_from_type(NULL) == MODEL_FAMILY_UNKNOWN);
}

/* --- A1 Cycle 5: per-family defaults ------------------------------------- */

static int eos_contains(const model_config_t *cfg, uint32_t id) {
    for (int i = 0; i < cfg->num_eos_tokens; i++) {
        if (cfg->eos_token_ids[i] == id)
            return 1;
    }
    return 0;
}

static void test_family_defaults(void) {
    model_config_t cfg;
    int rc;

    /* gemma3_text: silent heads fill, forced tie, gelu, flags, rope 8.0 */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_gemma3_text");
    assert(rc == 0);
    assert(cfg.family == MODEL_GEMMA3);
    assert(cfg.num_attention_heads == 8);
    assert(cfg.num_key_value_heads == 4);
    assert(cfg.head_dim == 256);
    assert(cfg.weight_prefix != NULL && strcmp(cfg.weight_prefix, "model") == 0);
    assert(cfg.tie_word_embeddings == true);
    assert(cfg.hidden_act == HIDDEN_ACT_GELU_APPROX);
    assert(cfg.norm_has_offset == true);
    assert(cfg.scale_embeddings == true);
    assert(cfg.has_pre_ff_norm == true);
    assert(cfg.has_qk_norm == true);
    assert(cfg.rope_scaling_factor == 8.0f);
    assert(eos_contains(&cfg, 1));
    assert(eos_contains(&cfg, 106));
    model_config_free(&cfg);

    /* gemma4 */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_gemma4");
    assert(rc == 0);
    assert(cfg.family == MODEL_GEMMA4);
    assert(cfg.weight_prefix != NULL &&
           strcmp(cfg.weight_prefix, "language_model.model") == 0);
    assert(cfg.has_v_norm == true);
    assert(cfg.norm_has_offset == false);
    assert(cfg.hidden_act == HIDDEN_ACT_GELU_APPROX);
    assert(cfg.attn_scale_one == true);
    assert(eos_contains(&cfg, 1));
    assert(eos_contains(&cfg, 106));
    model_config_free(&cfg);

    /* qwen2 */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_qwen2");
    assert(rc == 0);
    assert(cfg.family == MODEL_QWEN2);
    assert(cfg.attention_bias == true);
    assert(cfg.has_qk_norm == false);
    assert(cfg.head_dim == 896 / 14);
    model_config_free(&cfg);

    /* mistral */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_mistral");
    assert(rc == 0);
    assert(cfg.family == MODEL_MISTRAL);
    assert(cfg.weight_prefix != NULL && strcmp(cfg.weight_prefix, "model") == 0);
    assert(cfg.hidden_act == HIDDEN_ACT_SILU);
    model_config_free(&cfg);

    /* lfm2 */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_lfm2");
    assert(rc == 0);
    assert(cfg.family == MODEL_LFM2);
    assert(cfg.tie_word_embeddings == true);
    assert(cfg.has_qk_norm == true);
    assert(cfg.has_hybrid_layers == true);
    assert(cfg.head_dim == 1024 / 16);
    model_config_free(&cfg);

    /* nemotron_h */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_nemotron_h");
    assert(rc == 0);
    assert(cfg.family == MODEL_NEMOTRON_H);
    assert(cfg.weight_prefix != NULL &&
           strcmp(cfg.weight_prefix, "backbone") == 0);
    assert(cfg.mamba_mlp_act == HIDDEN_ACT_RELU_SQ);
    assert(cfg.has_hybrid_layers == true);
    model_config_free(&cfg);

    /* bert */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_bert");
    assert(rc == 0);
    assert(cfg.family == MODEL_BERT);
    assert(cfg.is_encoder_only == true);
    assert(cfg.weight_prefix != NULL && strcmp(cfg.weight_prefix, "") == 0);
    assert(cfg.num_key_value_heads == cfg.num_attention_heads);
    assert(cfg.layer_norm_eps == 1e-12f);
    assert(cfg.type_vocab_size == 2);
    assert(cfg.head_dim == 768 / 12);
    model_config_free(&cfg);

    /* deepseek_v4 */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_deepseek_v4");
    assert(rc == 0);
    assert(cfg.family == MODEL_DEEPSEEK_V4);
    model_config_free(&cfg);

    /* qwen3 reuse */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_qwen3");
    assert(rc == 0);
    assert(cfg.family == MODEL_QWEN3);
    assert(cfg.has_qk_norm == true);
    assert(cfg.attn_scale_one == false);
    assert(cfg.weight_prefix != NULL && strcmp(cfg.weight_prefix, "model") == 0);
    model_config_free(&cfg);
}

/* --- A1 Cycle 6: qwen3_5 dense vs moe ------------------------------------ */

static void test_qwen3_5_dense_vs_moe(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_qwen3_5");
    assert(rc == 0);
    assert(cfg.family == MODEL_QWEN3_5);
    assert(cfg.attn_output_gate == true);
    assert(cfg.hidden_act == HIDDEN_ACT_SILU);
    assert(cfg.has_qk_norm == true);
    assert(cfg.has_sliding_window == false);
    assert(cfg.weight_prefix != NULL && strcmp(cfg.weight_prefix, "model") == 0);
    model_config_free(&cfg);

    /* qwen3_5 + num_experts upgrades to MOE */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_qwen3_5_moe");
    assert(rc == 0);
    assert(cfg.family == MODEL_QWEN3_5_MOE);
    assert(cfg.num_experts == 64);
    assert(cfg.num_experts_per_tok == 8);
    assert(cfg.attn_output_gate == true);
    model_config_free(&cfg);

    /* explicit model_type qwen3_5_moe */
    rc = model_config_load(
        &cfg, MLXD_FIXTURES_DIR "/model_config_qwen3_5_moe_type");
    assert(rc == 0);
    assert(cfg.family == MODEL_QWEN3_5_MOE);
    model_config_free(&cfg);
}

/* --- A1 Cycle 7: nested text_config -------------------------------------- */

static void test_nested_text_config(void) {
    model_config_t cfg;
    int rc =
        model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_gemma3_mm");
    assert(rc == 0);
    assert(cfg.family == MODEL_GEMMA3);
    assert(cfg.hidden_size == 2560);
    assert(cfg.num_hidden_layers == 34);
    assert(cfg.num_attention_heads == 8);
    assert(cfg.num_key_value_heads == 4);
    assert(cfg.head_dim == 256);
    assert(cfg.vocab_size == 262208);
    assert(cfg.weight_prefix != NULL &&
           strcmp(cfg.weight_prefix, "language_model.model") == 0);
    assert(cfg.tie_word_embeddings == true);
    assert(cfg.max_position_embeddings == 131072);
    assert(cfg.quant_bits == 4);
    assert(cfg.quant_group_size == 64);
    assert(eos_contains(&cfg, 1));
    assert(eos_contains(&cfg, 106));
    model_config_free(&cfg);
}

/* --- A1 Cycle 8: sliding window + layer patterns ------------------------- */

static void test_sliding_window_and_layers(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_mistral");
    assert(rc == 0);
    assert(cfg.has_sliding_window == true);
    assert(cfg.sliding_window == 4096);
    model_config_free(&cfg);

    /* null window = disabled */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_window_null");
    assert(rc == 0);
    assert(cfg.has_sliding_window == false);
    model_config_free(&cfg);

    /* gemma4 layer_types */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_gemma4");
    assert(rc == 0);
    assert(cfg.has_explicit_layer_types == true);
    assert(cfg.layer_is_global[0] == false);
    assert(cfg.layer_is_global[5] == true);
    assert(cfg.layer_is_global[7] == true);
    assert(model_layer_is_global(&cfg, 5) == true);
    assert(model_layer_is_global(&cfg, 0) == false);
    model_config_free(&cfg);

    /* gemma3 pattern convention: pattern 6 => layer 5 global */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_gemma3_text");
    assert(rc == 0);
    assert(cfg.has_sliding_window == true);
    assert(cfg.sliding_window_pattern == 6);
    assert(model_layer_is_global(&cfg, 5) == true);
    assert(model_layer_is_global(&cfg, 4) == false);
    assert(model_layer_is_global(&cfg, 11) == true);
    model_config_free(&cfg);

    /* modulo path: OOB layer >= num_hidden_layers wraps via pattern */
    assert(model_layer_is_global(&cfg, 29) == true);  /* 29 % 6 == 5 */
    assert(model_layer_is_global(&cfg, 30) == false); /* 30 % 6 == 0 */
    assert(model_layer_is_global(&cfg, 125) == true); /* 125 % 6 == 5 */

    /* mistral: sliding_window but no pattern - all layers are sliding, none global */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_mistral");
    assert(rc == 0);
    assert(model_layer_is_global(&cfg, 0) == false);
    assert(model_layer_is_global(&cfg, 4) == false);
    assert(model_layer_is_global(&cfg, 5) == false);
    assert(model_layer_is_global(&cfg, 11) == false);
    model_config_free(&cfg);

    /* gemma3 without explicit sliding_window_pattern key: family default fills 6 */
    rc = model_config_load(&cfg,
                           MLXD_FIXTURES_DIR "/model_config_gemma3_no_pattern");
    assert(rc == 0);
    assert(cfg.sliding_window_pattern == 6);
    assert(model_layer_is_global(&cfg, 5) == true);
    assert(model_layer_is_global(&cfg, 4) == false);
    assert(model_layer_is_global(&cfg, 11) == true);
    model_config_free(&cfg);

    /* too many layers */
    memset(&cfg, 0xAB, sizeof(cfg));
    rc = model_config_load(&cfg,
                           MLXD_FIXTURES_DIR "/model_config_too_many_layers");
    assert(rc == -1);
    model_config_free(&cfg);
}

/* --- A1 Cycle 9: Stage-E dims -------------------------------------------- */

static void test_stage_e_dims(void) {
    model_config_t cfg;
    int rc =
        model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_qwen3_5_moe");
    assert(rc == 0);
    assert(cfg.partial_rotary_factor == 0.25f);
    assert(cfg.rope_theta == 10000000.0f);
    assert(cfg.linear_num_key_heads == 16);
    assert(cfg.linear_num_value_heads == 32);
    assert(cfg.linear_key_head_dim == 128);
    assert(cfg.linear_value_head_dim == 128);
    assert(cfg.linear_conv_kernel_dim == 4);
    assert(cfg.full_attention_interval == 4);
    assert(cfg.moe_intermediate_size == 512);
    assert(cfg.shared_expert_intermediate_size == 512);
    model_config_free(&cfg);

    /* lfm2 hybrid + aliases */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_lfm2");
    assert(rc == 0);
    assert(cfg.layer_kinds[0] == LAYER_KIND_GATED_CONV);
    assert(cfg.layer_kinds[1] == LAYER_KIND_ATTENTION);
    assert(cfg.layer_kinds[2] == LAYER_KIND_GATED_CONV);
    assert(cfg.layer_kinds[3] == LAYER_KIND_ATTENTION);
    assert(cfg.lfm_conv_dim == 1024);
    assert(cfg.lfm_conv_kernel == 5); /* conv_L_cache */
    assert(cfg.rms_norm_eps == 1e-5f);
    assert(cfg.tie_word_embeddings == true);
    model_config_free(&cfg);

    /* nemotron_h mamba + hybrid pattern */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_nemotron_h");
    assert(rc == 0);
    assert(cfg.mamba_num_heads == 128);
    assert(cfg.mamba_head_dim == 64);
    assert(cfg.mamba_n_groups == 8);
    assert(cfg.ssm_state_size == 128);
    assert(cfg.mamba_conv_kernel == 4);
    assert(cfg.mamba_expand == 2);
    assert(cfg.mamba_chunk_size == 256);
    assert(cfg.time_step_min == 0.001f);
    assert(cfg.time_step_max == 100.0f);
    assert(cfg.layer_kinds[0] == LAYER_KIND_MAMBA2);
    assert(cfg.layer_kinds[1] == LAYER_KIND_MLP);
    assert(cfg.layer_kinds[2] == LAYER_KIND_MAMBA2);
    assert(cfg.layer_kinds[3] == LAYER_KIND_ATTENTION);
    assert(cfg.layer_kinds[4] == LAYER_KIND_MOE);
    assert(cfg.rms_norm_eps == 1e-5f); /* layer_norm_epsilon fallback */
    model_config_free(&cfg);

    /* gemma4 dual-head extras + nested rope_parameters */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_gemma4");
    assert(rc == 0);
    assert(cfg.rope_proportional == true);
    assert(cfg.rope_proportional_factor == 8.0f);
    assert(cfg.partial_rotary_factor_global == 0.5f);
    assert(cfg.rope_theta == 1000000.0f);
    assert(cfg.rope_local_base_freq == 10000.0f);
    assert(cfg.global_head_dim == 512);
    assert(cfg.num_global_key_value_heads == 4);
    assert(cfg.num_kv_shared_layers == 2);
    assert(cfg.final_logit_softcapping == 30.0f);
    assert(cfg.hidden_size_per_layer_input == 256);
    assert(cfg.attention_k_eq_v == true);
    assert(cfg.use_double_wide_mlp == false);
    model_config_free(&cfg);
}

/* --- D5 Cycle 2: per-layer geometry helpers ------------------------------ */

static void test_per_layer_geometry(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_gemma4");
    assert(rc == 0);

    /* gemma4 fixture: head_dim=256, global_head_dim=512,
       num_kv_heads=8, num_global_kv_heads=4,
       layers: 0-4 sliding (local), 5,7 global; 8 layers total,
       num_kv_shared_layers=2 => layers 6,7 are shared */

    /* model_layer_head_dim: global layers get global_head_dim */
    assert(model_layer_head_dim(&cfg, 0) == 256);  /* local */
    assert(model_layer_head_dim(&cfg, 5) == 512);  /* global */
    assert(model_layer_head_dim(&cfg, 7) == 512);  /* global */
    assert(model_layer_head_dim(&cfg, 4) == 256);  /* local */

    /* model_layer_kv_heads: global layers get num_global_key_value_heads when attention_k_eq_v */
    assert(model_layer_kv_heads(&cfg, 0) == 8);   /* local */
    assert(model_layer_kv_heads(&cfg, 5) == 4);   /* global */
    assert(model_layer_kv_heads(&cfg, 7) == 4);   /* global */
    assert(model_layer_kv_heads(&cfg, 3) == 8);   /* local */

    /* model_kv_source_layer: non-shared layers return -1 */
    assert(model_kv_source_layer(&cfg, 0) == -1);
    assert(model_kv_source_layer(&cfg, 5) == -1);

    /* shared layers (last num_kv_shared_layers=2): layers 6 and 7.
       layer 6 is local (sliding_attention), source = last non-shared local
       scanning downward: layers 0..5, last local = 4.
       layer 7 is global, source = last non-shared global scanning down = 5. */
    assert(model_kv_source_layer(&cfg, 6) == 4);
    assert(model_kv_source_layer(&cfg, 7) == 5);

    /* k_eq_v gating: with attention_k_eq_v=true (fixture), global layers
       return num_global_key_value_heads. When flipped to false, they must
       fall back to num_key_value_heads (mlx-lm/HF semantics). */
    assert(cfg.attention_k_eq_v == true);
    assert(model_layer_kv_heads(&cfg, 5) == 4);  /* global, k_eq_v=true */
    cfg.attention_k_eq_v = false;
    assert(model_layer_kv_heads(&cfg, 5) == 8);  /* global, k_eq_v=false => base */
    assert(model_layer_kv_heads(&cfg, 0) == 8);  /* local unchanged */
    cfg.attention_k_eq_v = true;

    /* model_layer_kv_shared: shared layers return true */
    assert(model_layer_kv_shared(&cfg, 0) == false);
    assert(model_layer_kv_shared(&cfg, 5) == false);
    assert(model_layer_kv_shared(&cfg, 6) == true);
    assert(model_layer_kv_shared(&cfg, 7) == true);

    /* Fail-closed: mutate layer_is_global so shared layer 6 (local) has
       no matching local layer in the non-shared prefix. All prefix layers
       become global => kv_source_layer for local shared layer 6 returns -1. */
    for (int i = 0; i < 6; i++) cfg.layer_is_global[i] = true;
    assert(model_kv_source_layer(&cfg, 6) == -1);
    /* Restore */
    for (int i = 0; i < 6; i++) cfg.layer_is_global[i] = (i == 5);

    /* qwen3 (no gemma4 extras): all layers return base dims, no sharing */
    model_config_free(&cfg);
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_qwen3");
    assert(rc == 0);
    assert(model_layer_head_dim(&cfg, 0) == 128);
    assert(model_layer_kv_heads(&cfg, 0) == 8);
    assert(model_kv_source_layer(&cfg, 0) == -1);
    assert(model_kv_source_layer(&cfg, 5) == -1);
    model_config_free(&cfg);
}

/* --- LFM2: explicit rms_norm_eps takes precedence over norm_eps alias ---- */

static void test_lfm2_norm_eps_precedence(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg,
                               MLXD_FIXTURES_DIR "/model_config_lfm2_both_eps");
    assert(rc == 0);
    assert(cfg.rms_norm_eps == 1e-6f);
    model_config_free(&cfg);
}

/* --- rope_parameters: nested blocks shadow flat keys --------------------- */

static void test_rope_nested_shadows_flat(void) {
    model_config_t cfg;
    int rc =
        model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_rope_both");
    assert(rc == 0);
    assert(cfg.rope_theta == 1000000.0f);
    assert(cfg.partial_rotary_factor == 1.0f);
    assert(cfg.rope_local_base_freq == 10000.0f);
    assert(cfg.partial_rotary_factor_global == 0.5f);
    model_config_free(&cfg);
}

/* --- PR61 Cycle 1: layer predicate ordering + gemma4 window default ------ */

static void test_layer_predicate_and_gemma4_window(void) {
    model_config_t cfg;
    int rc;

    /* gemma4 without explicit sliding_window key: silent fill */
    rc = model_config_load(&cfg,
                           MLXD_FIXTURES_DIR "/model_config_gemma4_no_window");
    assert(rc == 0);
    assert(cfg.has_sliding_window == true);
    assert(cfg.sliding_window == 1024);
    /* explicit layer_types map still governs per-layer predicate */
    assert(model_layer_is_global(&cfg, 0) == false);
    assert(model_layer_is_global(&cfg, 5) == true);
    assert(model_layer_is_global(&cfg, 7) == true);
    /* OOB: safe default = true */
    assert(model_layer_is_global(&cfg, -1) == true);
    assert(model_layer_is_global(&cfg, MLXD_MAX_LAYERS) == true);
    model_config_free(&cfg);

    /* LFM2: has_sliding_window=false but explicit layer_types present.
       conv layers must report false (explicit map wins). */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_lfm2");
    assert(rc == 0);
    assert(cfg.has_sliding_window == false);
    assert(cfg.has_explicit_layer_types == true);
    assert(model_layer_is_global(&cfg, 0) == false); /* conv */
    assert(model_layer_is_global(&cfg, 1) == true);  /* full_attention */
    assert(model_layer_is_global(&cfg, 2) == false); /* conv */
    assert(model_layer_is_global(&cfg, 3) == true);  /* full_attention */
    model_config_free(&cfg);
}

/* --- PR61 Cycle 3: strict wrong-type enforcement for family fields ------- */

static void test_strict_wrong_type_family_fields(void) {
    const char *bad[] = {
        /* 3a: hidden_act non-string */
        MLXD_FIXTURES_DIR "/model_config_hidden_act_wrong_type",
        /* 3b: nemotron mamba field wrong type */
        MLXD_FIXTURES_DIR "/model_config_nemotron_mamba_wrong_type",
        /* 3b: nemotron time_step_limit wrong type */
        MLXD_FIXTURES_DIR "/model_config_nemotron_tsl_wrong_type",
        /* 3b: nemotron hybrid_override_pattern wrong type */
        MLXD_FIXTURES_DIR "/model_config_nemotron_pattern_wrong_type",
        /* 3c: lfm2 tie_embedding wrong type */
        MLXD_FIXTURES_DIR "/model_config_lfm2_tie_wrong_type",
        /* 3c: lfm2 norm_eps wrong type */
        MLXD_FIXTURES_DIR "/model_config_lfm2_norm_wrong_type",
        /* 3c: lfm2 conv_dim wrong type */
        MLXD_FIXTURES_DIR "/model_config_lfm2_conv_dim_wrong_type",
        /* 3c: lfm2 conv_L_cache wrong type */
        MLXD_FIXTURES_DIR "/model_config_lfm2_conv_L_wrong_type",
        /* 3d: generic layer_types non-array */
        MLXD_FIXTURES_DIR "/model_config_layer_types_wrong_type",
        /* 3d: generic layer_types non-string element */
        MLXD_FIXTURES_DIR "/model_config_layer_types_elem_wrong_type",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        model_config_t cfg;
        memset(&cfg, 0xAB, sizeof(cfg));
        int rc = model_config_load(&cfg, bad[i]);
        assert(rc == -1);
        model_config_free(&cfg);
    }
}

/* --- PR61 Cycle 2: qwen3.5 head_dim fallback ----------------------------- */

static void test_qwen3_5_head_dim_fallback(void) {
    model_config_t cfg;
    int rc = model_config_load(
        &cfg, MLXD_FIXTURES_DIR "/model_config_qwen3_5_no_head_dim");
    assert(rc == 0);
    assert(cfg.family == MODEL_QWEN3_5);
    assert(cfg.head_dim == 2048 / 16); /* hidden_size / num_attention_heads */
    assert(cfg.query_pre_attn_scalar == cfg.head_dim);
    model_config_free(&cfg);
}

/* --- PR61 Cycle 4: gemma3 rope_scaling null branch pinning --------------- */

static void test_gemma3_rope_null(void) {
    model_config_t cfg;
    int rc = model_config_load(
        &cfg, MLXD_FIXTURES_DIR "/model_config_gemma3_rope_null");
    assert(rc == 0);
    assert(cfg.family == MODEL_GEMMA3);
    /* Generic parse skips null; gemma3 fallback fires on !yyjson_is_obj(rs)
       and yields factor 8.0 - this IS the null-handling path. */
    assert(cfg.rope_scaling_factor == 8.0f);
    model_config_free(&cfg);
}

/* --- A1 Cycle 10: EOS from config.json ----------------------------------- */

static void test_eos_tokens(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_qwen3");
    assert(rc == 0);
    /* config.json has 151645; generation_config unions 151643 + 151645 */
    assert(eos_contains(&cfg, 151645));
    assert(eos_contains(&cfg, 151643));
    assert(cfg.num_eos_tokens == 2);
    model_config_free(&cfg);

    /* array with duplicate 151643 - fixture carries the dup for dedup coverage */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_eos_array");
    assert(rc == 0);
    assert(cfg.num_eos_tokens == 2);
    assert(cfg.eos_token_ids[0] == 151643);
    assert(cfg.eos_token_ids[1] == 151645);
    model_config_free(&cfg);

    /* gemma3 scalar eos 106 => additive {1, 106} without dup */
    rc = model_config_load(&cfg,
                           MLXD_FIXTURES_DIR "/model_config_gemma3_eos_scalar");
    assert(rc == 0);
    assert(eos_contains(&cfg, 1));
    assert(eos_contains(&cfg, 106));
    assert(cfg.num_eos_tokens == 2);
    model_config_free(&cfg);

    /* array of 10 capped at 8 */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_eos_cap");
    assert(rc == 0);
    assert(cfg.num_eos_tokens == 8);
    model_config_free(&cfg);
}

/* --- P2a: nested text_config EOS union ----------------------------------- */

static void test_nested_text_config_eos(void) {
    model_config_t cfg;

    /* qwen3_5 VL-style: EOS only inside text_config, no root EOS */
    int rc = model_config_load(
        &cfg, MLXD_FIXTURES_DIR "/model_config_qwen3_5_nested_eos");
    assert(rc == 0);
    assert(cfg.num_eos_tokens == 1);
    assert(eos_contains(&cfg, 151645));
    model_config_free(&cfg);

    /* union + dedupe: root=100, text_config=[100,200] => {100,200} */
    rc = model_config_load(&cfg,
                           MLXD_FIXTURES_DIR "/model_config_nested_eos_union");
    assert(rc == 0);
    assert(cfg.num_eos_tokens == 2);
    assert(cfg.eos_token_ids[0] == 100);
    assert(eos_contains(&cfg, 200));
    model_config_free(&cfg);

    /* lfm2 VL: nested EOS via text_config (safety net for refactor) */
    rc = model_config_load(&cfg,
                           MLXD_FIXTURES_DIR "/model_config_lfm2_vl_eos");
    assert(rc == 0);
    assert(eos_contains(&cfg, 99999));
    model_config_free(&cfg);
}

/* --- PR61-R4 Cycle 1: get_f32 float overflow guard ----------------------- */

static void test_float_overflow(void) {
    model_config_t cfg;

    /* get_f32 path: rope_theta 1e39 overflows float */
    memset(&cfg, 0xAB, sizeof(cfg));
    int rc = model_config_load(&cfg,
                               MLXD_FIXTURES_DIR "/model_config_float_overflow");
    assert(rc == -1);
    model_config_free(&cfg);

    /* direct eps: lfm2 norm_eps 1e39 */
    memset(&cfg, 0xAB, sizeof(cfg));
    rc = model_config_load(
        &cfg, MLXD_FIXTURES_DIR "/model_config_norm_eps_overflow");
    assert(rc == -1);
    model_config_free(&cfg);

    /* direct eps: nemotron layer_norm_epsilon 1e39 */
    memset(&cfg, 0xAB, sizeof(cfg));
    rc = model_config_load(
        &cfg, MLXD_FIXTURES_DIR "/model_config_nemotron_eps_overflow");
    assert(rc == -1);
    model_config_free(&cfg);

    /* direct eps: bert layer_norm_eps 1e39 */
    memset(&cfg, 0xAB, sizeof(cfg));
    rc = model_config_load(
        &cfg, MLXD_FIXTURES_DIR "/model_config_bert_eps_overflow");
    assert(rc == -1);
    model_config_free(&cfg);
}

/* --- A1 Cycle 11: generation_config.json --------------------------------- */

static void test_generation_config(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_qwen3");
    assert(rc == 0);
    assert(cfg.has_gen_temperature == true);
    assert(cfg.gen_temperature == 0.7f);
    assert(cfg.has_gen_top_p == true);
    assert(cfg.gen_top_p == 0.8f);
    assert(cfg.has_gen_top_k == true);
    assert(cfg.gen_top_k == 20);
    /* eos unioned + deduped (config 151645 + gen [151643, 151645]) */
    assert(cfg.num_eos_tokens == 2);
    model_config_free(&cfg);

    /* out-of-range values dropped, rc 0 */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_gen_bad");
    assert(rc == 0);
    assert(cfg.has_gen_temperature == false);
    assert(cfg.has_gen_top_p == false);
    assert(cfg.has_gen_top_k == false);
    model_config_free(&cfg);

    /* absence (llama fixture) = all has_ false */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config");
    assert(rc == 0);
    assert(cfg.has_gen_temperature == false);
    assert(cfg.has_gen_top_p == false);
    assert(cfg.has_gen_top_k == false);
    model_config_free(&cfg);

    /* free semantics covers rope_scaling_type */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_llama3");
    assert(rc == 0);
    assert(cfg.rope_scaling_type != NULL);
    model_config_free(&cfg);
    model_config_free(&cfg); /* double free safe */

    model_config_t cfg2;
    memset(&cfg2, 0xAB, sizeof(cfg2));
    assert(model_config_load(
               &cfg2, MLXD_FIXTURES_DIR "/model_config_rope_scaling_wrong_type") ==
           -1);
    model_config_free(&cfg2);
}

/* --- PR61 review pin: bert head_dim overwrite is oracle-correct ---------- */

static void test_bert_explicit_head_dim_ignored(void) {
    model_config_t cfg;
    int rc =
        model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_bert_head_dim");
    assert(rc == 0);
    assert(cfg.family == MODEL_BERT);
    assert(cfg.head_dim == 768 / 12);
    model_config_free(&cfg);
}

/* --- Phase 3: tiny_gemma3 fixture config --------------------------------- */

static void test_tiny_gemma3_config(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/tiny_gemma3");
    assert(rc == 0);

    assert(cfg.family == MODEL_GEMMA3);
    assert(cfg.vocab_size == 256);
    assert(cfg.hidden_size == 64);
    assert(cfg.num_hidden_layers == 2);
    assert(cfg.num_attention_heads == 4);
    assert(cfg.num_key_value_heads == 2);
    assert(cfg.head_dim == 16);
    assert(cfg.intermediate_size == 128);

    assert(cfg.has_sliding_window == true);
    assert(cfg.sliding_window == 4);
    assert(cfg.sliding_window_pattern == 2);

    assert(cfg.rope_theta == 1000000.0f);
    assert(cfg.rope_local_base_freq == 10000.0f);
    assert(cfg.query_pre_attn_scalar == 64);

    assert(cfg.norm_has_offset == true);
    assert(cfg.scale_embeddings == true);
    assert(cfg.has_pre_ff_norm == true);
    assert(cfg.has_qk_norm == true);
    assert(cfg.tie_word_embeddings == true);

    assert(cfg.rope_scaling_factor == 8.0f);
    assert(cfg.hidden_act == HIDDEN_ACT_GELU_APPROX);
    assert(cfg.attention_bias == false);

    model_config_free(&cfg);
}

static void test_tiny_gemma4_config(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/tiny_gemma4");
    assert(rc == 0);

    assert(cfg.family == MODEL_GEMMA4);
    assert(cfg.vocab_size == 256);
    assert(cfg.hidden_size == 64);
    assert(cfg.num_hidden_layers == 4);
    assert(cfg.num_attention_heads == 4);
    assert(cfg.num_key_value_heads == 2);
    assert(cfg.head_dim == 16);
    assert(cfg.intermediate_size == 128);

    assert(cfg.global_head_dim == 32);
    assert(cfg.num_global_key_value_heads == 1);
    assert(cfg.num_kv_shared_layers == 2);
    assert(cfg.attention_k_eq_v == true);
    assert(cfg.use_double_wide_mlp == false);
    assert(cfg.attn_scale_one == true);

    assert(cfg.has_sliding_window == true);
    assert(cfg.sliding_window == 4);
    assert(cfg.has_explicit_layer_types == true);
    assert(cfg.layer_is_global[0] == false);
    assert(cfg.layer_is_global[1] == true);
    assert(cfg.layer_is_global[2] == false);
    assert(cfg.layer_is_global[3] == true);

    assert(cfg.rope_proportional == true);
    assert(cfg.rope_proportional_factor == 8.0f);
    assert(cfg.partial_rotary_factor_global == 0.5f);
    assert(cfg.rope_theta == 1000000.0f);
    assert(cfg.rope_local_base_freq == 10000.0f);

    assert(cfg.tie_word_embeddings == true);
    assert(cfg.has_v_norm == true);
    assert(cfg.has_qk_norm == true);
    assert(cfg.has_pre_ff_norm == true);
    assert(cfg.hidden_act == HIDDEN_ACT_GELU_APPROX);
    assert(cfg.norm_has_offset == false);
    assert(cfg.scale_embeddings == true);

    /* per-layer geometry */
    assert(model_layer_head_dim(&cfg, 0) == 16);
    assert(model_layer_head_dim(&cfg, 1) == 32);
    assert(model_layer_kv_heads(&cfg, 0) == 2);
    assert(model_layer_kv_heads(&cfg, 1) == 1);
    assert(model_kv_source_layer(&cfg, 0) == -1);
    assert(model_kv_source_layer(&cfg, 1) == -1);
    assert(model_kv_source_layer(&cfg, 2) == 0);
    assert(model_kv_source_layer(&cfg, 3) == 1);

    model_config_free(&cfg);
}

static void test_gemma4_family_defaults(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg,
                               MLXD_FIXTURES_DIR "/model_config_gemma4_minimal");
    assert(rc == 0);
    assert(cfg.family == MODEL_GEMMA4);
    assert(cfg.final_logit_softcapping == 30.0f);
    assert(cfg.hidden_size_per_layer_input == 256);
    assert(cfg.use_double_wide_mlp == false);
    model_config_free(&cfg);

    /* Non-gemma4: defaults stay at 0 */
    rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config");
    assert(rc == 0);
    assert(cfg.family == MODEL_LLAMA);
    assert(cfg.final_logit_softcapping == 0.0f);
    assert(cfg.hidden_size_per_layer_input == 0);
    model_config_free(&cfg);
}

static void test_gemma4_null_defaults(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg,
                               MLXD_FIXTURES_DIR "/model_config_gemma4_null_defaults");
    assert(rc == 0);
    assert(cfg.family == MODEL_GEMMA4);
    assert(cfg.final_logit_softcapping == 30.0f);
    assert(cfg.hidden_size_per_layer_input == 256);
    model_config_free(&cfg);
}

static void test_gemma4_hidden_act_parsed(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg,
                               MLXD_FIXTURES_DIR "/model_config_gemma4_silu");
    assert(rc == 0);
    assert(cfg.family == MODEL_GEMMA4);
    assert(cfg.hidden_act == HIDDEN_ACT_SILU);
    model_config_free(&cfg);
}

int main(void) {
    test_happy_path();
    test_kv_heads_default();
    test_missing_config();
    test_missing_model_type();
    test_bad_json();
    test_root_not_object();
    test_model_type_not_string();
    test_malformed_numerics();
    test_free_semantics();

    test_extended_core_dims();
    test_rope_scaling();
    test_quantization();
    test_family_from_type();
    test_family_defaults();
    test_qwen3_5_dense_vs_moe();
    test_nested_text_config();
    test_sliding_window_and_layers();
    test_stage_e_dims();
    test_per_layer_geometry();
    test_lfm2_norm_eps_precedence();
    test_rope_nested_shadows_flat();
    test_layer_predicate_and_gemma4_window();
    test_strict_wrong_type_family_fields();
    test_qwen3_5_head_dim_fallback();
    test_gemma3_rope_null();
    test_eos_tokens();
    test_nested_text_config_eos();
    test_generation_config();
    test_float_overflow();
    test_bert_explicit_head_dim_ignored();
    test_tiny_gemma3_config();
    test_tiny_gemma4_config();
    test_gemma4_family_defaults();
    test_gemma4_hidden_act_parsed();
    test_gemma4_null_defaults();

    printf("test_model_config: all passed\n");
    return 0;
}
