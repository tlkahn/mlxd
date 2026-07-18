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
    assert(err[0] != '\0');
}

static void test_reject_attention_bias(void) {
    model_config_t cfg = make_supported();
    cfg.attention_bias = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(err[0] != '\0');
}

static void test_reject_sliding_window(void) {
    model_config_t cfg = make_supported();
    cfg.has_sliding_window = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(err[0] != '\0');
}

static void test_reject_moe(void) {
    model_config_t cfg = make_supported();
    cfg.num_experts = 8;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(err[0] != '\0');
}

static void test_reject_hybrid_layers(void) {
    model_config_t cfg = make_supported();
    cfg.has_hybrid_layers = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(err[0] != '\0');
}

static void test_reject_non_silu(void) {
    model_config_t cfg = make_supported();
    cfg.hidden_act = HIDDEN_ACT_GELU_APPROX;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(err[0] != '\0');
}

static void test_reject_norm_offset(void) {
    model_config_t cfg = make_supported();
    cfg.norm_has_offset = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(err[0] != '\0');
}

static void test_reject_scale_embeddings(void) {
    model_config_t cfg = make_supported();
    cfg.scale_embeddings = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(err[0] != '\0');
}

static void test_reject_pre_ff_norm(void) {
    model_config_t cfg = make_supported();
    cfg.has_pre_ff_norm = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(err[0] != '\0');
}

static void test_reject_rope_scaling(void) {
    model_config_t cfg = make_supported();
    cfg.rope_scaling_type = "linear";
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(err[0] != '\0');
}

static void test_reject_partial_rotary(void) {
    model_config_t cfg = make_supported();
    cfg.partial_rotary_factor = 0.5f;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(err[0] != '\0');
}

static void test_reject_attn_output_gate(void) {
    model_config_t cfg = make_supported();
    cfg.attn_output_gate = true;
    char err[256] = {0};
    assert(engine_model_check_supported(&cfg, err, sizeof(err)) != 0);
    assert(err[0] != '\0');
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

    printf("test_emodel_gate: all passed\n");
    return 0;
}
