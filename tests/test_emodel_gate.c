#include "engine/emodel.h"
#include "model/model.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static model_config_t make_supported(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.family = MODEL_QWEN3;
    cfg.hidden_act = HIDDEN_ACT_SILU;
    cfg.partial_rotary_factor = 1.0f;
    return cfg;
}

static void test_supported_passes(void) {
    model_config_t cfg = make_supported();
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) == 0);
    assert(err[0] == '\0');
}

static void test_reject_unsupported_family(void) {
    model_config_t cfg = make_supported();
    cfg.family = MODEL_GEMMA3;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "unsupported model family (only qwen3/llama dense supported)") == 0);
}

static void test_reject_attention_bias(void) {
    model_config_t cfg = make_supported();
    cfg.attention_bias = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "attention_bias not supported") == 0);
}

static void test_reject_sliding_window(void) {
    model_config_t cfg = make_supported();
    cfg.has_sliding_window = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "sliding window attention not supported") == 0);
}

static void test_reject_moe(void) {
    model_config_t cfg = make_supported();
    cfg.num_experts = 8;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "MoE models not supported") == 0);
}

static void test_reject_hybrid_layers(void) {
    model_config_t cfg = make_supported();
    cfg.has_hybrid_layers = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "hybrid layer architectures not supported") == 0);
}

static void test_reject_non_silu(void) {
    model_config_t cfg = make_supported();
    cfg.hidden_act = HIDDEN_ACT_GELU_APPROX;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "only SiLU activation supported") == 0);
}

static void test_reject_norm_offset(void) {
    model_config_t cfg = make_supported();
    cfg.norm_has_offset = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "norm_has_offset not supported") == 0);
}

static void test_reject_scale_embeddings(void) {
    model_config_t cfg = make_supported();
    cfg.scale_embeddings = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "scale_embeddings not supported") == 0);
}

static void test_reject_pre_ff_norm(void) {
    model_config_t cfg = make_supported();
    cfg.has_pre_ff_norm = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "pre-feedforward norm not supported") == 0);
}

static void test_reject_rope_scaling(void) {
    model_config_t cfg = make_supported();
    cfg.rope_scaling_type = "linear";
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "RoPE scaling not supported") == 0);
}

static void test_reject_partial_rotary(void) {
    model_config_t cfg = make_supported();
    cfg.partial_rotary_factor = 0.5f;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "partial rotary embedding not supported") == 0);
}

static void test_reject_attn_output_gate(void) {
    model_config_t cfg = make_supported();
    cfg.attn_output_gate = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "attention output gate not supported") == 0);
}

static void test_reject_all_other_families(void) {
    static const model_family_t others[] = {
        MODEL_FAMILY_UNKNOWN, MODEL_GEMMA3, MODEL_GEMMA4,
        MODEL_QWEN2, MODEL_QWEN3_5, MODEL_QWEN3_5_MOE,
        MODEL_MISTRAL, MODEL_LFM2,
        MODEL_NEMOTRON_H, MODEL_DEEPSEEK_V4, MODEL_BERT,
    };
    for (size_t i = 0; i < sizeof(others) / sizeof(others[0]); i++) {
        model_config_t cfg = make_supported();
        cfg.family = others[i];
        char err[256] = {0};
        assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
        assert(strcmp(err, "unsupported model family (only qwen3/llama dense supported)") == 0);
    }
}

/* --- llama-specific gate tests ------------------------------------------- */

static model_config_t make_llama_supported(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.family = MODEL_LLAMA;
    cfg.hidden_act = HIDDEN_ACT_SILU;
    cfg.partial_rotary_factor = 1.0f;
    return cfg;
}

static void test_llama_plain_passes(void) {
    model_config_t cfg = make_llama_supported();
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) == 0);
}

static void test_llama_rope_llama3_passes(void) {
    model_config_t cfg = make_llama_supported();
    cfg.rope_scaling_type = "llama3";
    cfg.rope_scaling_factor = 8.0f;
    cfg.rope_original_max_position_embeddings = 8192;
    cfg.rope_low_freq_factor = 1.0f;
    cfg.rope_high_freq_factor = 4.0f;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) == 0);
}

static void test_llama_rope_linear_passes(void) {
    model_config_t cfg = make_llama_supported();
    cfg.rope_scaling_type = "linear";
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) == 0);
}

static void test_llama_rope_default_passes(void) {
    model_config_t cfg = make_llama_supported();
    cfg.rope_scaling_type = "default";
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) == 0);
}

static void test_llama_reject_rope_yarn(void) {
    model_config_t cfg = make_llama_supported();
    cfg.rope_scaling_type = "yarn";
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "unsupported RoPE scaling type") == 0);
}

static void test_llama_reject_qk_norm(void) {
    model_config_t cfg = make_llama_supported();
    cfg.has_qk_norm = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "qk_norm not supported for llama") == 0);
}

static void test_llama_reject_attention_bias(void) {
    model_config_t cfg = make_llama_supported();
    cfg.attention_bias = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "attention_bias not supported") == 0);
}

static void test_llama_reject_sliding_window(void) {
    model_config_t cfg = make_llama_supported();
    cfg.has_sliding_window = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "sliding window attention not supported") == 0);
}

static void test_llama_reject_moe(void) {
    model_config_t cfg = make_llama_supported();
    cfg.num_experts = 8;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "MoE models not supported") == 0);
}

static void test_llama_reject_non_silu(void) {
    model_config_t cfg = make_llama_supported();
    cfg.hidden_act = HIDDEN_ACT_GELU_APPROX;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "only SiLU activation supported") == 0);
}

static void test_llama_reject_partial_rotary(void) {
    model_config_t cfg = make_llama_supported();
    cfg.partial_rotary_factor = 0.5f;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "partial rotary embedding not supported") == 0);
}

/* --- llama3 rope parameter validation ----------------------------------- */

static void test_llama_reject_llama3_factor_zero(void) {
    model_config_t cfg = make_llama_supported();
    cfg.rope_scaling_type = "llama3";
    cfg.rope_scaling_factor = 0.0f;
    cfg.rope_original_max_position_embeddings = 8192;
    cfg.rope_low_freq_factor = 1.0f;
    cfg.rope_high_freq_factor = 4.0f;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strstr(err, "rope_scaling_factor") != NULL);
}

static void test_llama_reject_llama3_original_max_zero(void) {
    model_config_t cfg = make_llama_supported();
    cfg.rope_scaling_type = "llama3";
    cfg.rope_scaling_factor = 8.0f;
    cfg.rope_original_max_position_embeddings = 0;
    cfg.rope_low_freq_factor = 1.0f;
    cfg.rope_high_freq_factor = 4.0f;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strstr(err, "original_max_position") != NULL);
}

static void test_llama_reject_llama3_low_freq_zero(void) {
    model_config_t cfg = make_llama_supported();
    cfg.rope_scaling_type = "llama3";
    cfg.rope_scaling_factor = 8.0f;
    cfg.rope_original_max_position_embeddings = 8192;
    cfg.rope_low_freq_factor = 0.0f;
    cfg.rope_high_freq_factor = 4.0f;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strstr(err, "low_freq_factor") != NULL);
}

static void test_llama_reject_llama3_high_leq_low(void) {
    model_config_t cfg = make_llama_supported();
    cfg.rope_scaling_type = "llama3";
    cfg.rope_scaling_factor = 8.0f;
    cfg.rope_original_max_position_embeddings = 8192;
    cfg.rope_low_freq_factor = 4.0f;
    cfg.rope_high_freq_factor = 4.0f;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strstr(err, "high_freq_factor") != NULL);
}

static void test_llama_llama3_valid_still_passes(void) {
    model_config_t cfg = make_llama_supported();
    cfg.rope_scaling_type = "llama3";
    cfg.rope_scaling_factor = 8.0f;
    cfg.rope_original_max_position_embeddings = 8192;
    cfg.rope_low_freq_factor = 1.0f;
    cfg.rope_high_freq_factor = 4.0f;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) == 0);
}

int main(void) {
    test_supported_passes();
    printf("  test_supported_passes: passed\n");

    test_reject_unsupported_family();
    printf("  test_reject_unsupported_family: passed\n");

    test_reject_attention_bias();
    printf("  test_reject_attention_bias: passed\n");

    test_reject_sliding_window();
    printf("  test_reject_sliding_window: passed\n");

    test_reject_moe();
    printf("  test_reject_moe: passed\n");

    test_reject_hybrid_layers();
    printf("  test_reject_hybrid_layers: passed\n");

    test_reject_non_silu();
    printf("  test_reject_non_silu: passed\n");

    test_reject_norm_offset();
    printf("  test_reject_norm_offset: passed\n");

    test_reject_scale_embeddings();
    printf("  test_reject_scale_embeddings: passed\n");

    test_reject_pre_ff_norm();
    printf("  test_reject_pre_ff_norm: passed\n");

    test_reject_rope_scaling();
    printf("  test_reject_rope_scaling: passed\n");

    test_reject_partial_rotary();
    printf("  test_reject_partial_rotary: passed\n");

    test_reject_attn_output_gate();
    printf("  test_reject_attn_output_gate: passed\n");

    test_reject_all_other_families();
    printf("  test_reject_all_other_families: passed\n");

    test_llama_plain_passes();
    printf("  test_llama_plain_passes: passed\n");

    test_llama_rope_llama3_passes();
    printf("  test_llama_rope_llama3_passes: passed\n");

    test_llama_rope_linear_passes();
    printf("  test_llama_rope_linear_passes: passed\n");

    test_llama_rope_default_passes();
    printf("  test_llama_rope_default_passes: passed\n");

    test_llama_reject_rope_yarn();
    printf("  test_llama_reject_rope_yarn: passed\n");

    test_llama_reject_qk_norm();
    printf("  test_llama_reject_qk_norm: passed\n");

    test_llama_reject_attention_bias();
    printf("  test_llama_reject_attention_bias: passed\n");

    test_llama_reject_sliding_window();
    printf("  test_llama_reject_sliding_window: passed\n");

    test_llama_reject_moe();
    printf("  test_llama_reject_moe: passed\n");

    test_llama_reject_non_silu();
    printf("  test_llama_reject_non_silu: passed\n");

    test_llama_reject_partial_rotary();
    printf("  test_llama_reject_partial_rotary: passed\n");

    test_llama_reject_llama3_factor_zero();
    printf("  test_llama_reject_llama3_factor_zero: passed\n");

    test_llama_reject_llama3_original_max_zero();
    printf("  test_llama_reject_llama3_original_max_zero: passed\n");

    test_llama_reject_llama3_low_freq_zero();
    printf("  test_llama_reject_llama3_low_freq_zero: passed\n");

    test_llama_reject_llama3_high_leq_low();
    printf("  test_llama_reject_llama3_high_leq_low: passed\n");

    test_llama_llama3_valid_still_passes();
    printf("  test_llama_llama3_valid_still_passes: passed\n");

    printf("test_emodel_gate: all passed\n");
    return 0;
}
