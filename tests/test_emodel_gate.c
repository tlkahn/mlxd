#include "engine/emodel.h"
#include "model/model.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#ifndef MLXD_FIXTURES_DIR
#error "MLXD_FIXTURES_DIR must be defined"
#endif

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
    assert(strcmp(err,
        "unsupported model family (only qwen3/llama/gemma4/qwen3_5 dense supported)") == 0);
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
        MODEL_FAMILY_UNKNOWN, MODEL_GEMMA3,
        MODEL_QWEN2, MODEL_QWEN3_5_MOE,
        MODEL_MISTRAL, MODEL_LFM2,
        MODEL_NEMOTRON_H, MODEL_DEEPSEEK_V4, MODEL_BERT,
    };
    for (size_t i = 0; i < sizeof(others) / sizeof(others[0]); i++) {
        model_config_t cfg = make_supported();
        cfg.family = others[i];
        char err[256] = {0};
        assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
        assert(strcmp(err,
            "unsupported model family (only qwen3/llama/gemma4/qwen3_5 dense supported)") == 0);
    }
}

/* --- qwen3_5 pure-dense gate tests (Stage D6 / R1) ----------------------- */

static model_config_t make_qwen3_5_supported(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.family = MODEL_QWEN3_5;
    cfg.hidden_act = HIDDEN_ACT_SILU;
    cfg.attn_output_gate = true;
    cfg.partial_rotary_factor = 0.25f;
    cfg.head_dim = 16; /* 16 * 0.25 = 4, even and >= 2 */
    cfg.has_qk_norm = true;
    cfg.num_experts = 0;
    cfg.linear_num_key_heads = 0;
    cfg.linear_num_value_heads = 0;
    cfg.full_attention_interval = 0;
    return cfg;
}

static void test_qwen3_5_pure_dense_passes(void) {
    model_config_t cfg = make_qwen3_5_supported();
    char err[256] = {0};
    assert(cfg.attn_output_gate == true);
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) == 0);
    assert(err[0] == '\0');
}

static void test_qwen3_5_rejects_gate_off(void) {
    model_config_t cfg = make_qwen3_5_supported();
    cfg.attn_output_gate = false;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strstr(err, "qwen3_5 requires attn_output_gate") != NULL);
}

static void test_qwen3_5_full_attention_interval(void) {
    model_config_t cfg = make_qwen3_5_supported();
    char err[256] = {0};

    /* interval > 1 implies hybrid layers (is_linear when (idx+1)%interval != 0) */
    cfg.full_attention_interval = 4;
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strstr(err, "full_attention_interval") != NULL);

    /* interval == 1 is pure dense (every layer full-attn) */
    cfg.full_attention_interval = 1;
    memset(err, 0, sizeof(err));
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) == 0);

    /* interval == 0 accepted (unset / no hybrid schedule) */
    cfg.full_attention_interval = 0;
    memset(err, 0, sizeof(err));
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) == 0);
}

static void test_qwen3_5_rejects_odd_rope_dims(void) {
    model_config_t cfg = make_qwen3_5_supported();
    /* head_dim=6, prf=0.5 -> dims=3 (odd); mx.fast.rope requires even >= 2 */
    cfg.head_dim = 6;
    cfg.partial_rotary_factor = 0.5f;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strstr(err, "partial_rotary_factor yields invalid rope dims") != NULL);

    /* even product stays accepted (16 * 0.25 = 4) */
    cfg.head_dim = 16;
    cfg.partial_rotary_factor = 0.25f;
    memset(err, 0, sizeof(err));
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) == 0);
}

static void test_qwen3_5_moe_family_still_rejected(void) {
    model_config_t cfg = make_qwen3_5_supported();
    cfg.family = MODEL_QWEN3_5_MOE;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err,
        "unsupported model family (only qwen3/llama/gemma4/qwen3_5 dense supported)") == 0);
}

static void test_qwen3_5_rejects_linear_heads(void) {
    model_config_t cfg = make_qwen3_5_supported();
    cfg.linear_num_key_heads = 16;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strstr(err, "linear attention") != NULL);
}

static void test_qwen3_5_rejects_experts(void) {
    model_config_t cfg = make_qwen3_5_supported();
    cfg.num_experts = 8;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strcmp(err, "MoE models not supported") == 0);
}

static void test_qwen3_5_rejects_partial_rotary_out_of_range(void) {
    model_config_t cfg = make_qwen3_5_supported();
    cfg.partial_rotary_factor = 0.0f;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strstr(err, "partial_rotary_factor") != NULL);

    cfg.partial_rotary_factor = 1.5f;
    memset(err, 0, sizeof(err));
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strstr(err, "partial_rotary_factor") != NULL);
}

static void test_qwen3_5_loaded_boundary_fixtures(void) {
    /* pure-dense full fixture -> accept */
    {
        model_config_t cfg;
        int rc = model_config_load(
            &cfg, MLXD_FIXTURES_DIR "/model_config_qwen3_5_dense_full");
        assert(rc == 0);
        assert(cfg.family == MODEL_QWEN3_5);
        char err[256] = {0};
        assert(engine_model_check_supported(&cfg, err, sizeof(err)) == 0);
        model_config_free(&cfg);
    }

    /* moe fixture -> family MOE -> reject */
    {
        model_config_t cfg;
        int rc = model_config_load(
            &cfg, MLXD_FIXTURES_DIR "/model_config_qwen3_5_moe");
        assert(rc == 0);
        assert(cfg.family == MODEL_QWEN3_5_MOE);
        char err[256] = {0};
        assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
        model_config_free(&cfg);
    }

    /* hybrid-dense (linear heads, num_experts==0) stays MODEL_QWEN3_5 under R1
       but gate rejects linear attention */
    {
        model_config_t cfg;
        int rc = model_config_load(
            &cfg, MLXD_FIXTURES_DIR "/model_config_qwen3_5_hybrid_dense");
        assert(rc == 0);
        assert(cfg.family == MODEL_QWEN3_5);
        assert(cfg.num_experts == 0);
        assert(cfg.linear_num_key_heads > 0);
        char err[256] = {0};
        assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
        assert(strstr(err, "linear attention") != NULL);
        model_config_free(&cfg);
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

/* --- gemma4-specific gate tests ----------------------------------------- */

static model_config_t make_gemma4_supported(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.family = MODEL_GEMMA4;
    cfg.hidden_act = HIDDEN_ACT_GELU_APPROX;
    cfg.use_double_wide_mlp = false;
    cfg.norm_has_offset = false;
    cfg.scale_embeddings = true;
    cfg.has_pre_ff_norm = true;
    cfg.has_sliding_window = true;
    cfg.sliding_window = 4;
    cfg.has_qk_norm = true;
    cfg.partial_rotary_factor = 1.0f;
    cfg.partial_rotary_factor_global = 0.5f;
    cfg.rope_proportional = true;
    cfg.rope_proportional_factor = 8.0f;
    cfg.num_hidden_layers = 4;
    cfg.num_kv_shared_layers = 2;
    cfg.has_explicit_layer_types = true;
    cfg.layer_is_global[0] = false;
    cfg.layer_is_global[1] = true;
    cfg.layer_is_global[2] = false;
    cfg.layer_is_global[3] = true;
    return cfg;
}

static void test_gemma4_supported_passes(void) {
    model_config_t cfg = make_gemma4_supported();
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) == 0);
}

static void test_gemma4_reject_all_shared(void) {
    model_config_t cfg = make_gemma4_supported();
    cfg.num_kv_shared_layers = cfg.num_hidden_layers;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strstr(err, "num_kv_shared_layers") != NULL);
}

static void test_gemma4_reject_proportional_factor_zero(void) {
    model_config_t cfg = make_gemma4_supported();
    cfg.rope_proportional_factor = 0.0f;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strstr(err, "rope_proportional_factor") != NULL);
}

static void test_gemma4_reject_partial_rotary_global_zero(void) {
    model_config_t cfg = make_gemma4_supported();
    cfg.partial_rotary_factor_global = 0.0f;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strstr(err, "partial_rotary_factor_global") != NULL);
}

static void test_gemma4_reject_partial_rotary_global_over_one(void) {
    model_config_t cfg = make_gemma4_supported();
    cfg.partial_rotary_factor_global = 1.5f;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strstr(err, "partial_rotary_factor_global") != NULL);
}

static void test_gemma4_reject_unresolved_kv_source(void) {
    /* All shared layers global but no global non-shared layers below boundary */
    model_config_t cfg = make_gemma4_supported();
    cfg.num_hidden_layers = 4;
    cfg.num_kv_shared_layers = 2;
    cfg.has_explicit_layer_types = true;
    /* layers 0,1 non-shared local; layers 2,3 shared global.
       Layer 2 (shared global) needs a global source below boundary=2, but
       layers 0,1 are both local => no source. */
    cfg.layer_is_global[0] = false;
    cfg.layer_is_global[1] = false;
    cfg.layer_is_global[2] = true;
    cfg.layer_is_global[3] = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(strstr(err, "no same-type source") != NULL);
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

    test_gemma4_supported_passes();
    printf("  test_gemma4_supported_passes: passed\n");

    test_gemma4_reject_all_shared();
    printf("  test_gemma4_reject_all_shared: passed\n");

    test_gemma4_reject_proportional_factor_zero();
    printf("  test_gemma4_reject_proportional_factor_zero: passed\n");

    test_gemma4_reject_partial_rotary_global_zero();
    printf("  test_gemma4_reject_partial_rotary_global_zero: passed\n");

    test_gemma4_reject_partial_rotary_global_over_one();
    printf("  test_gemma4_reject_partial_rotary_global_over_one: passed\n");

    test_gemma4_reject_unresolved_kv_source();
    printf("  test_gemma4_reject_unresolved_kv_source: passed\n");

    /* use_double_wide_mlp rejection */
    {
        model_config_t cfg = make_gemma4_supported();
        cfg.use_double_wide_mlp = true;
        char err[256] = {0};
        assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
        assert(strstr(err, "use_double_wide_mlp") != NULL);
    }
    printf("  test_gemma4_reject_use_double_wide_mlp: passed\n");

    /* hidden_act rejection */
    {
        model_config_t cfg = make_gemma4_supported();
        cfg.hidden_act = HIDDEN_ACT_SILU;
        char err[256] = {0};
        assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
        assert(strstr(err, "gelu_pytorch_tanh") != NULL);
    }
    printf("  test_gemma4_reject_hidden_act_silu: passed\n");

    /* E2E: config.json with hidden_act=silu should be rejected by the gate */
    {
        model_config_t cfg;
        int rc = model_config_load(&cfg,
                                   MLXD_FIXTURES_DIR "/model_config_gemma4_silu");
        assert(rc == 0);
        char err[256] = {0};
        assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
        assert(strstr(err, "gelu_pytorch_tanh") != NULL);
        model_config_free(&cfg);
    }
    printf("  test_gemma4_silu_config_rejected: passed\n");

    /* E2E: minimal config (omits use_double_wide_mlp) should pass the gate */
    {
        model_config_t cfg;
        int rc = model_config_load(&cfg,
                                   MLXD_FIXTURES_DIR "/model_config_gemma4_minimal");
        assert(rc == 0);
        char err[256] = {0};
        assert(engine_model_check_supported(&cfg, err, sizeof(err)) == 0);
        model_config_free(&cfg);
    }
    printf("  test_gemma4_minimal_config_passes: passed\n");

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

    test_qwen3_5_pure_dense_passes();
    printf("  test_qwen3_5_pure_dense_passes: passed\n");

    test_qwen3_5_rejects_gate_off();
    printf("  test_qwen3_5_rejects_gate_off: passed\n");

    test_qwen3_5_full_attention_interval();
    printf("  test_qwen3_5_full_attention_interval: passed\n");

    test_qwen3_5_rejects_odd_rope_dims();
    printf("  test_qwen3_5_rejects_odd_rope_dims: passed\n");

    test_qwen3_5_moe_family_still_rejected();
    printf("  test_qwen3_5_moe_family_still_rejected: passed\n");

    test_qwen3_5_rejects_linear_heads();
    printf("  test_qwen3_5_rejects_linear_heads: passed\n");

    test_qwen3_5_rejects_experts();
    printf("  test_qwen3_5_rejects_experts: passed\n");

    test_qwen3_5_rejects_partial_rotary_out_of_range();
    printf("  test_qwen3_5_rejects_partial_rotary_out_of_range: passed\n");

    test_qwen3_5_loaded_boundary_fixtures();
    printf("  test_qwen3_5_loaded_boundary_fixtures: passed\n");

    printf("test_emodel_gate: all passed\n");
    return 0;
}
