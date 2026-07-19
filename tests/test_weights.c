#include "model/weights.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

extern char *mkdtemp(char *);

#define FIXTURES MLXD_FIXTURES_DIR

/* Helper: write a string to a file in a tmpdir */
static void write_file(const char *dir, const char *name, const char *content) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "w");
    assert(f);
    fwrite(content, 1, strlen(content), f);
    fclose(f);
}

static void unlink_file(const char *dir, const char *name) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    unlink(path);
}

/* ---- Cycle 16: shard enumeration ---- */

static void test_shard_index_preferred(void) {
    char **paths = NULL;
    size_t count = 0;
    bool from_index = false;
    int rc = weights_enumerate_shards(FIXTURES "/shard_index", &paths, &count,
                                      &from_index);
    assert(rc == 0);
    assert(from_index == true);
    assert(count == 2);

    const char *base1 = strrchr(paths[0], '/');
    const char *base2 = strrchr(paths[1], '/');
    assert(base1 && base2);
    assert(strcmp(base1 + 1, "model-00001-of-00002.safetensors") == 0);
    assert(strcmp(base2 + 1, "model-00002-of-00002.safetensors") == 0);

    weights_free_shard_paths(paths, count);
}

static void test_shard_single_file(void) {
    char **paths = NULL;
    size_t count = 0;
    bool from_index = false;
    int rc = weights_enumerate_shards(FIXTURES "/shard_single", &paths, &count,
                                      &from_index);
    assert(rc == 0);
    assert(from_index == false);
    assert(count == 1);

    const char *base = strrchr(paths[0], '/');
    assert(base);
    assert(strcmp(base + 1, "model.safetensors") == 0);

    weights_free_shard_paths(paths, count);
}

static void test_shard_glob_sorted(void) {
    char **paths = NULL;
    size_t count = 0;
    bool from_index = false;
    int rc = weights_enumerate_shards(FIXTURES "/shard_glob", &paths, &count,
                                      &from_index);
    assert(rc == 0);
    assert(from_index == false);
    assert(count == 3);

    const char *b0 = strrchr(paths[0], '/');
    const char *b1 = strrchr(paths[1], '/');
    const char *b2 = strrchr(paths[2], '/');
    assert(strcmp(b0 + 1, "a.safetensors") == 0);
    assert(strcmp(b1 + 1, "b.safetensors") == 0);
    assert(strcmp(b2 + 1, "c.safetensors") == 0);

    weights_free_shard_paths(paths, count);
}

static void test_shard_missing_error(void) {
    char **paths = NULL;
    size_t count = 0;
    int rc = weights_enumerate_shards(FIXTURES "/shard_empty", &paths, &count,
                                      NULL);
    assert(rc == -1);
}

/* ---- Cycle 17: tensor naming ---- */

static void test_tensor_name_with_prefix(void) {
    model_config_t cfg = {0};
    cfg.weight_prefix = "model";

    char buf[256];
    assert(weights_tensor_name(buf, sizeof(buf), &cfg, 3,
                               "self_attn.q_proj.weight") == 0);
    assert(strcmp(buf, "model.layers.3.self_attn.q_proj.weight") == 0);
}

static void test_tensor_name_global(void) {
    model_config_t cfg = {0};
    cfg.weight_prefix = "model";

    char buf[256];
    assert(weights_tensor_name(buf, sizeof(buf), &cfg, -1, "norm.weight") == 0);
    assert(strcmp(buf, "model.norm.weight") == 0);
}

static void test_tensor_name_empty_prefix(void) {
    model_config_t cfg = {0};
    cfg.weight_prefix = "";

    char buf[256];
    assert(weights_tensor_name(buf, sizeof(buf), &cfg, 0,
                               "attention.self.query.weight") == 0);
    assert(strcmp(buf, "layers.0.attention.self.query.weight") == 0);

    assert(weights_tensor_name(buf, sizeof(buf), &cfg, -1,
                               "embeddings.word_embeddings.weight") == 0);
    assert(strcmp(buf, "embeddings.word_embeddings.weight") == 0);
}

static void test_tensor_name_long_prefix(void) {
    model_config_t cfg = {0};
    cfg.weight_prefix = "language_model.model";

    char buf[256];
    assert(weights_tensor_name(buf, sizeof(buf), &cfg, 0,
                               "self_attn.q_proj.weight") == 0);
    assert(strcmp(buf,
           "language_model.model.layers.0.self_attn.q_proj.weight") == 0);
}

static void test_tensor_name_buffer_overflow(void) {
    model_config_t cfg = {0};
    cfg.weight_prefix = "model";

    char buf[10];
    assert(weights_tensor_name(buf, sizeof(buf), &cfg, 0,
                               "self_attn.q_proj.weight") == -1);
}

/* ---- Cycle 1: tri-state index handling (B2+L6) ---- */

static void test_malformed_index_not_glob(void) {
    char tmpdir[] = "/tmp/mlxd_c1a_XXXXXX";
    assert(mkdtemp(tmpdir) != NULL);

    write_file(tmpdir, "model.safetensors.index.json", "{invalid json");
    write_file(tmpdir, "stray.safetensors", "dummy");

    char **paths = NULL;
    size_t count = 0;
    bool from_index = false;
    int rc = weights_enumerate_shards(tmpdir, &paths, &count, &from_index);
    assert(rc == -2);
    assert(paths == NULL);
    assert(count == 0);

    unlink_file(tmpdir, "model.safetensors.index.json");
    unlink_file(tmpdir, "stray.safetensors");
    rmdir(tmpdir);
}

static void test_empty_weight_map_not_glob(void) {
    char tmpdir[] = "/tmp/mlxd_c1b_XXXXXX";
    assert(mkdtemp(tmpdir) != NULL);

    write_file(tmpdir, "model.safetensors.index.json",
               "{\"weight_map\": {}}\n");
    write_file(tmpdir, "stray.safetensors", "dummy");

    char **paths = NULL;
    size_t count = 0;
    bool from_index = false;
    int rc = weights_enumerate_shards(tmpdir, &paths, &count, &from_index);
    assert(rc == -2);
    assert(paths == NULL);
    assert(count == 0);

    unlink_file(tmpdir, "model.safetensors.index.json");
    unlink_file(tmpdir, "stray.safetensors");
    rmdir(tmpdir);
}

static void test_enumerate_zeros_outputs_on_failure(void) {
    char **paths = (char **)0xDEADBEEF;
    size_t count = 999;
    int rc = weights_enumerate_shards(FIXTURES "/shard_empty", &paths, &count,
                                      NULL);
    assert(rc == -1);
    assert(paths == NULL);
    assert(count == 0);
}

/* ---- Cycle 2: sanitize weight_map filenames (M5) ---- */

static const char *bad_fnames[] = {
    "../evil.safetensors",
    "/abs/path.safetensors",
    "sub/dir.safetensors",
    "noext",
};

static void test_index_path_escape_rejected(void) {
    for (int i = 0; i < 4; i++) {
        char tmpdir[] = "/tmp/mlxd_c2_XXXXXX";
        assert(mkdtemp(tmpdir) != NULL);

        char idx[1024];
        snprintf(idx, sizeof(idx),
                 "{\"weight_map\": {\"x\": \"%s\"}}\n", bad_fnames[i]);
        write_file(tmpdir, "model.safetensors.index.json", idx);
        write_file(tmpdir, "stray.safetensors", "dummy");

        char **paths = NULL;
        size_t count = 0;
        int rc = weights_enumerate_shards(tmpdir, &paths, &count, NULL);
        assert(rc == -2);
        assert(paths == NULL);
        assert(count == 0);

        unlink_file(tmpdir, "model.safetensors.index.json");
        unlink_file(tmpdir, "stray.safetensors");
        rmdir(tmpdir);
    }
}

/* ---- M6: reject non-regular files ---- */

static void test_glob_skips_directory(void) {
    char tmpdir[] = "/tmp/mlxd_m6a_XXXXXX";
    assert(mkdtemp(tmpdir) != NULL);

    /* Create a directory whose name ends in .safetensors */
    char dirpath[512];
    snprintf(dirpath, sizeof(dirpath), "%s/fake.safetensors", tmpdir);
    assert(mkdir(dirpath, 0755) == 0);

    /* Create one real safetensors file */
    write_file(tmpdir, "a.safetensors", "dummy");

    char **paths = NULL;
    size_t count = 0;
    int rc = weights_enumerate_shards(tmpdir, &paths, &count, NULL);
    assert(rc == 0);
    assert(count == 1);
    const char *base = strrchr(paths[0], '/');
    assert(base);
    assert(strcmp(base + 1, "a.safetensors") == 0);

    weights_free_shard_paths(paths, count);
    unlink_file(tmpdir, "a.safetensors");
    rmdir(dirpath);
    rmdir(tmpdir);
}

static void test_single_is_directory(void) {
    char tmpdir[] = "/tmp/mlxd_m6b_XXXXXX";
    assert(mkdtemp(tmpdir) != NULL);

    /* model.safetensors is a directory, not a file */
    char dirpath[512];
    snprintf(dirpath, sizeof(dirpath), "%s/model.safetensors", tmpdir);
    assert(mkdir(dirpath, 0755) == 0);

    /* A real safetensors file that glob should find */
    write_file(tmpdir, "x.safetensors", "dummy");

    char **paths = NULL;
    size_t count = 0;
    int rc = weights_enumerate_shards(tmpdir, &paths, &count, NULL);
    assert(rc == 0);
    assert(count == 1);
    const char *base = strrchr(paths[0], '/');
    assert(base);
    assert(strcmp(base + 1, "x.safetensors") == 0);

    weights_free_shard_paths(paths, count);
    unlink_file(tmpdir, "x.safetensors");
    rmdir(dirpath);
    rmdir(tmpdir);
}

/* ---- Cycle 1: expected tensor names for qwen3 ---- */

static void test_expected_names_qwen3(void) {
    model_config_t cfg = {0};
    cfg.family = MODEL_QWEN3;
    cfg.weight_prefix = "model";
    cfg.num_hidden_layers = 2;
    cfg.has_qk_norm = true;
    cfg.tie_word_embeddings = false;

    int count = weights_expected_names(&cfg, NULL, 0);
    assert(count == 25);

    weight_expected_t names[25];
    int rc = weights_expected_names(&cfg, names, 25);
    assert(rc == 25);

    /* Verify embed_tokens */
    bool found_embed = false;
    for (int i = 0; i < rc; i++) {
        if (strcmp(names[i].name, "model.embed_tokens") == 0) {
            assert(names[i].kind == WEIGHT_KIND_EMBED);
            found_embed = true;
        }
    }
    assert(found_embed);

    /* Verify a per-layer matmul */
    bool found_q0 = false, found_q1 = false;
    for (int i = 0; i < rc; i++) {
        if (strcmp(names[i].name, "model.layers.0.self_attn.q_proj") == 0) {
            assert(names[i].kind == WEIGHT_KIND_MATMUL);
            found_q0 = true;
        }
        if (strcmp(names[i].name, "model.layers.1.self_attn.q_proj") == 0) {
            assert(names[i].kind == WEIGHT_KIND_MATMUL);
            found_q1 = true;
        }
    }
    assert(found_q0 && found_q1);

    /* Verify norms (with qk_norm) */
    bool found_inorm = false, found_qnorm = false, found_knorm = false;
    for (int i = 0; i < rc; i++) {
        if (strcmp(names[i].name, "model.layers.0.input_layernorm") == 0) {
            assert(names[i].kind == WEIGHT_KIND_NORM);
            found_inorm = true;
        }
        if (strcmp(names[i].name, "model.layers.0.self_attn.q_norm") == 0) {
            assert(names[i].kind == WEIGHT_KIND_NORM);
            found_qnorm = true;
        }
        if (strcmp(names[i].name, "model.layers.0.self_attn.k_norm") == 0) {
            assert(names[i].kind == WEIGHT_KIND_NORM);
            found_knorm = true;
        }
    }
    assert(found_inorm && found_qnorm && found_knorm);

    /* Verify global norm */
    bool found_gnorm = false;
    for (int i = 0; i < rc; i++) {
        if (strcmp(names[i].name, "model.norm") == 0) {
            assert(names[i].kind == WEIGHT_KIND_NORM);
            found_gnorm = true;
        }
    }
    assert(found_gnorm);

    /* Verify lm_head present (not tied) */
    bool found_lmhead = false;
    for (int i = 0; i < rc; i++) {
        if (strcmp(names[i].name, "lm_head") == 0) {
            assert(names[i].kind == WEIGHT_KIND_MATMUL);
            found_lmhead = true;
        }
    }
    assert(found_lmhead);

    /* Capacity too small -> error */
    weight_expected_t small[2];
    assert(weights_expected_names(&cfg, small, 2) == -1);
}

/* ---- Cycle 2: expected name flag variants ---- */

static void test_expected_names_tied(void) {
    model_config_t cfg = {0};
    cfg.family = MODEL_QWEN3;
    cfg.weight_prefix = "model";
    cfg.num_hidden_layers = 2;
    cfg.has_qk_norm = true;
    cfg.tie_word_embeddings = true;

    int count = weights_expected_names(&cfg, NULL, 0);
    assert(count == 24);

    weight_expected_t names[24];
    int rc = weights_expected_names(&cfg, names, 24);
    assert(rc == 24);

    for (int i = 0; i < rc; i++)
        assert(strcmp(names[i].name, "lm_head") != 0);
}

static void test_expected_names_no_qk_norm(void) {
    model_config_t cfg = {0};
    cfg.family = MODEL_QWEN3;
    cfg.weight_prefix = "model";
    cfg.num_hidden_layers = 2;
    cfg.has_qk_norm = false;
    cfg.tie_word_embeddings = false;

    /* 25 - 4 (2 layers * 2 qk norms) = 21 */
    int count = weights_expected_names(&cfg, NULL, 0);
    assert(count == 21);

    weight_expected_t names[21];
    int rc = weights_expected_names(&cfg, names, 21);
    assert(rc == 21);

    for (int i = 0; i < rc; i++) {
        assert(strstr(names[i].name, "q_norm") == NULL);
        assert(strstr(names[i].name, "k_norm") == NULL);
    }
}

static void test_expected_names_llama(void) {
    model_config_t cfg = {0};
    cfg.family = MODEL_LLAMA;
    cfg.weight_prefix = "model";
    cfg.num_hidden_layers = 2;
    cfg.has_qk_norm = false;
    cfg.attention_bias = false;
    cfg.tie_word_embeddings = false;

    /* 1 embed + 2*(7 matmuls + 2 norms) + final norm + lm_head = 21 */
    int count = weights_expected_names(&cfg, NULL, 0);
    assert(count == 21);

    weight_expected_t names[21];
    int rc = weights_expected_names(&cfg, names, 21);
    assert(rc == 21);

    bool found_q0 = false;
    for (int i = 0; i < rc; i++) {
        if (strcmp(names[i].name, "model.layers.0.self_attn.q_proj") == 0) {
            assert(names[i].kind == WEIGHT_KIND_MATMUL);
            found_q0 = true;
        }
        assert(strstr(names[i].name, "q_norm") == NULL);
        assert(strstr(names[i].name, "k_norm") == NULL);
    }
    assert(found_q0);

    bool found_lmhead = false;
    for (int i = 0; i < rc; i++) {
        if (strcmp(names[i].name, "lm_head") == 0) {
            assert(names[i].kind == WEIGHT_KIND_MATMUL);
            found_lmhead = true;
        }
    }
    assert(found_lmhead);

    /* tied: 20 (no lm_head) */
    cfg.tie_word_embeddings = true;
    count = weights_expected_names(&cfg, NULL, 0);
    assert(count == 20);

    weight_expected_t tied_names[20];
    rc = weights_expected_names(&cfg, tied_names, 20);
    assert(rc == 20);
    for (int i = 0; i < rc; i++)
        assert(strcmp(tied_names[i].name, "lm_head") != 0);
}

/* ---- Cycle D0-1: pin exact emission order ---- */

static void test_expected_names_qwen3_exact_order(void) {
    model_config_t cfg = {0};
    cfg.family = MODEL_QWEN3;
    cfg.weight_prefix = "model";
    cfg.num_hidden_layers = 1;
    cfg.has_qk_norm = true;
    cfg.tie_word_embeddings = false;

    int count = weights_expected_names(&cfg, NULL, 0);
    assert(count == 14);

    weight_expected_t names[14];
    int rc = weights_expected_names(&cfg, names, 14);
    assert(rc == 14);

    typedef struct { const char *name; weight_kind_t kind; } expect_t;
    static const expect_t expected[] = {
        {"model.embed_tokens",                          WEIGHT_KIND_EMBED},
        {"model.layers.0.self_attn.q_proj",             WEIGHT_KIND_MATMUL},
        {"model.layers.0.self_attn.k_proj",             WEIGHT_KIND_MATMUL},
        {"model.layers.0.self_attn.v_proj",             WEIGHT_KIND_MATMUL},
        {"model.layers.0.self_attn.o_proj",             WEIGHT_KIND_MATMUL},
        {"model.layers.0.mlp.gate_proj",                WEIGHT_KIND_MATMUL},
        {"model.layers.0.mlp.up_proj",                  WEIGHT_KIND_MATMUL},
        {"model.layers.0.mlp.down_proj",                WEIGHT_KIND_MATMUL},
        {"model.layers.0.input_layernorm",              WEIGHT_KIND_NORM},
        {"model.layers.0.post_attention_layernorm",     WEIGHT_KIND_NORM},
        {"model.layers.0.self_attn.q_norm",             WEIGHT_KIND_NORM},
        {"model.layers.0.self_attn.k_norm",             WEIGHT_KIND_NORM},
        {"model.norm",                                  WEIGHT_KIND_NORM},
        {"lm_head",                                     WEIGHT_KIND_MATMUL},
    };

    for (int i = 0; i < 14; i++) {
        assert(strcmp(names[i].name, expected[i].name) == 0);
        assert(names[i].kind == expected[i].kind);
    }
}

static void test_expected_names_gemma4(void) {
    model_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.family = MODEL_GEMMA4;
    cfg.weight_prefix = "language_model.model";
    cfg.num_hidden_layers = 4;
    cfg.has_qk_norm = true;
    cfg.has_pre_ff_norm = true;
    cfg.tie_word_embeddings = true;
    cfg.attention_k_eq_v = true;
    cfg.hidden_size_per_layer_input = 8;
    cfg.num_kv_shared_layers = 2;
    cfg.has_explicit_layer_types = true;
    cfg.layer_is_global[0] = false;
    cfg.layer_is_global[1] = true;
    cfg.layer_is_global[2] = false;
    cfg.layer_is_global[3] = true;

    int count = weights_expected_names(&cfg, NULL, 0);
    assert(count == 66);

    weight_expected_t names[66];
    int rc = weights_expected_names(&cfg, names, 66);
    assert(rc == 66);

    /* Spot checks */
    assert(strcmp(names[0].name, "language_model.model.embed_tokens") == 0);
    assert(names[0].kind == WEIGHT_KIND_EMBED);

    /* PLE globals should be present */
    bool found_ple_emb = false, found_ple_proj = false, found_ple_norm = false;
    for (int i = 0; i < rc; i++) {
        if (strcmp(names[i].name, "language_model.model.embed_tokens_per_layer") == 0) {
            assert(names[i].kind == WEIGHT_KIND_EMBED);
            found_ple_emb = true;
        }
        if (strcmp(names[i].name, "language_model.model.per_layer_model_projection") == 0) {
            assert(names[i].kind == WEIGHT_KIND_MATMUL);
            found_ple_proj = true;
        }
        if (strcmp(names[i].name, "language_model.model.per_layer_projection_norm") == 0) {
            assert(names[i].kind == WEIGHT_KIND_NORM);
            found_ple_norm = true;
        }
    }
    assert(found_ple_emb && found_ple_proj && found_ple_norm);

    /* Non-shared layer 0 (local): has k_proj, k_norm, v_proj */
    bool found_k0 = false, found_kn0 = false, found_v0 = false;
    for (int i = 0; i < rc; i++) {
        if (strcmp(names[i].name, "language_model.model.layers.0.self_attn.k_proj") == 0)
            found_k0 = true;
        if (strcmp(names[i].name, "language_model.model.layers.0.self_attn.k_norm") == 0)
            found_kn0 = true;
        if (strcmp(names[i].name, "language_model.model.layers.0.self_attn.v_proj") == 0)
            found_v0 = true;
    }
    assert(found_k0 && found_kn0 && found_v0);

    /* Non-shared layer 1 (global, k_eq_v): has k_proj, k_norm but no v_proj */
    bool found_k1 = false, found_kn1 = false, found_v1 = false;
    for (int i = 0; i < rc; i++) {
        if (strcmp(names[i].name, "language_model.model.layers.1.self_attn.k_proj") == 0)
            found_k1 = true;
        if (strcmp(names[i].name, "language_model.model.layers.1.self_attn.k_norm") == 0)
            found_kn1 = true;
        if (strcmp(names[i].name, "language_model.model.layers.1.self_attn.v_proj") == 0)
            found_v1 = true;
    }
    assert(found_k1 && found_kn1 && !found_v1);

    /* Shared layer 2: no k_proj, k_norm, v_proj */
    bool found_k2 = false, found_kn2 = false, found_v2 = false;
    for (int i = 0; i < rc; i++) {
        if (strcmp(names[i].name, "language_model.model.layers.2.self_attn.k_proj") == 0)
            found_k2 = true;
        if (strcmp(names[i].name, "language_model.model.layers.2.self_attn.k_norm") == 0)
            found_kn2 = true;
        if (strcmp(names[i].name, "language_model.model.layers.2.self_attn.v_proj") == 0)
            found_v2 = true;
    }
    assert(!found_k2 && !found_kn2 && !found_v2);

    /* PLE per-layer always present */
    bool found_ple_gate0 = false, found_ple_gate2 = false;
    for (int i = 0; i < rc; i++) {
        if (strcmp(names[i].name, "language_model.model.layers.0.per_layer_input_gate") == 0)
            found_ple_gate0 = true;
        if (strcmp(names[i].name, "language_model.model.layers.2.per_layer_input_gate") == 0)
            found_ple_gate2 = true;
    }
    assert(found_ple_gate0 && found_ple_gate2);

    /* layer_scalar: bare weight per layer */
    int bare_count = 0;
    for (int i = 0; i < rc; i++) {
        if (names[i].kind == WEIGHT_KIND_BARE) {
            assert(strstr(names[i].name, "layer_scalar") != NULL);
            bare_count++;
        }
    }
    assert(bare_count == 4);
    bool found_ls0 = false, found_ls3 = false;
    for (int i = 0; i < rc; i++) {
        if (strcmp(names[i].name, "language_model.model.layers.0.layer_scalar") == 0) {
            assert(names[i].kind == WEIGHT_KIND_BARE);
            found_ls0 = true;
        }
        if (strcmp(names[i].name, "language_model.model.layers.3.layer_scalar") == 0) {
            assert(names[i].kind == WEIGHT_KIND_BARE);
            found_ls3 = true;
        }
    }
    assert(found_ls0 && found_ls3);

    /* Capacity too small -> error */
    weight_expected_t small[2];
    assert(weights_expected_names(&cfg, small, 2) == -1);
}

static void test_expected_names_other_families_zero(void) {
    static const model_family_t others[] = {
        MODEL_FAMILY_UNKNOWN, MODEL_GEMMA3,
        MODEL_QWEN2, MODEL_QWEN3_5, MODEL_QWEN3_5_MOE,
        MODEL_MISTRAL, MODEL_LFM2,
        MODEL_NEMOTRON_H, MODEL_DEEPSEEK_V4, MODEL_BERT,
    };

    for (size_t i = 0; i < sizeof(others) / sizeof(others[0]); i++) {
        model_config_t cfg = {0};
        cfg.family = others[i];
        cfg.weight_prefix = "model";
        cfg.num_hidden_layers = 2;
        assert(weights_expected_names(&cfg, NULL, 0) == 0);
    }
}

/* ---- Cycle CR-A: descriptor-driven emitter seam ---- */

static void test_expected_names_from_desc_synthetic(void) {
    static const char *const syn_matmuls[] = { "attn.wq", "ffn.w1", NULL };
    static const char *const syn_norms[]   = { "ln1", NULL };

    weights_family_desc_t desc = {
        .family         = MODEL_FAMILY_UNKNOWN,
        .layer_matmuls  = syn_matmuls,
        .layer_norms    = syn_norms,
        .layer_qk_norms = NULL,
        .layer_biases   = NULL,
        .extra_tensors  = NULL,
    };

    model_config_t cfg = {0};
    cfg.family = MODEL_FAMILY_UNKNOWN;
    cfg.weight_prefix = "model";
    cfg.num_hidden_layers = 1;
    cfg.tie_word_embeddings = true;

    int count = weights_expected_names_from_desc(&desc, &cfg, NULL, 0);
    assert(count == 5);

    weight_expected_t names[5];
    assert(weights_expected_names_from_desc(&desc, &cfg, names, 5) == 5);

    assert(strcmp(names[0].name, "model.embed_tokens") == 0);
    assert(names[0].kind == WEIGHT_KIND_EMBED);
    assert(strcmp(names[1].name, "model.layers.0.attn.wq") == 0);
    assert(names[1].kind == WEIGHT_KIND_MATMUL);
    assert(strcmp(names[2].name, "model.layers.0.ffn.w1") == 0);
    assert(names[2].kind == WEIGHT_KIND_MATMUL);
    assert(strcmp(names[3].name, "model.layers.0.ln1") == 0);
    assert(names[3].kind == WEIGHT_KIND_NORM);
    assert(strcmp(names[4].name, "model.norm") == 0);
    assert(names[4].kind == WEIGHT_KIND_NORM);
}

/* ---- Cycle CR-B: wire layer_biases ---- */

static void test_expected_names_from_desc_biases(void) {
    static const char *const syn_matmuls[] = { "attn.wq", NULL };
    static const char *const syn_norms[]   = { "ln1", NULL };
    static const char *const syn_biases[]  = { "attn.wq.bias", NULL };

    weights_family_desc_t desc = {
        .family         = MODEL_FAMILY_UNKNOWN,
        .layer_matmuls  = syn_matmuls,
        .layer_norms    = syn_norms,
        .layer_qk_norms = NULL,
        .layer_biases   = syn_biases,
        .extra_tensors  = NULL,
    };

    model_config_t cfg = {0};
    cfg.family = MODEL_FAMILY_UNKNOWN;
    cfg.weight_prefix = "model";
    cfg.num_hidden_layers = 1;
    cfg.tie_word_embeddings = true;

    cfg.attention_bias = false;
    int count_no_bias = weights_expected_names_from_desc(&desc, &cfg, NULL, 0);
    assert(count_no_bias == 4);

    weight_expected_t names_no_bias[4];
    assert(weights_expected_names_from_desc(&desc, &cfg, names_no_bias, 4) == 4);
    for (int i = 0; i < 4; i++)
        assert(names_no_bias[i].kind != WEIGHT_KIND_BIAS);

    cfg.attention_bias = true;
    int count_bias = weights_expected_names_from_desc(&desc, &cfg, NULL, 0);
    assert(count_bias == 5);

    weight_expected_t names_bias[5];
    assert(weights_expected_names_from_desc(&desc, &cfg, names_bias, 5) == 5);

    assert(strcmp(names_bias[0].name, "model.embed_tokens") == 0);
    assert(names_bias[0].kind == WEIGHT_KIND_EMBED);
    assert(strcmp(names_bias[1].name, "model.layers.0.attn.wq") == 0);
    assert(names_bias[1].kind == WEIGHT_KIND_MATMUL);
    assert(strcmp(names_bias[2].name, "model.layers.0.attn.wq.bias") == 0);
    assert(names_bias[2].kind == WEIGHT_KIND_BIAS);
    assert(strcmp(names_bias[3].name, "model.layers.0.ln1") == 0);
    assert(names_bias[3].kind == WEIGHT_KIND_NORM);
    assert(strcmp(names_bias[4].name, "model.norm") == 0);
    assert(names_bias[4].kind == WEIGHT_KIND_NORM);
}

/* ---- Cycle CR-C: wire extra_tensors ---- */

static void test_expected_names_from_desc_extras(void) {
    static const char *const syn_matmuls[] = { "attn.wq", NULL };
    static const char *const syn_norms[]   = { "ln1", NULL };
    static const weight_extra_t syn_extras[] = {
        {"embed_tokens_per_layer", WEIGHT_KIND_EMBED},
        {"v_norm_global",          WEIGHT_KIND_NORM},
        {NULL, 0},
    };

    weights_family_desc_t desc = {
        .family         = MODEL_FAMILY_UNKNOWN,
        .layer_matmuls  = syn_matmuls,
        .layer_norms    = syn_norms,
        .layer_qk_norms = NULL,
        .layer_biases   = NULL,
        .extra_tensors  = syn_extras,
    };

    model_config_t cfg = {0};
    cfg.family = MODEL_FAMILY_UNKNOWN;
    cfg.weight_prefix = "model";
    cfg.num_hidden_layers = 1;
    cfg.tie_word_embeddings = false;

    int count = weights_expected_names_from_desc(&desc, &cfg, NULL, 0);
    assert(count == 7);

    weight_expected_t names[7];
    assert(weights_expected_names_from_desc(&desc, &cfg, names, 7) == 7);

    assert(strcmp(names[0].name, "model.embed_tokens") == 0);
    assert(names[0].kind == WEIGHT_KIND_EMBED);
    assert(strcmp(names[1].name, "model.layers.0.attn.wq") == 0);
    assert(names[1].kind == WEIGHT_KIND_MATMUL);
    assert(strcmp(names[2].name, "model.layers.0.ln1") == 0);
    assert(names[2].kind == WEIGHT_KIND_NORM);
    assert(strcmp(names[3].name, "model.norm") == 0);
    assert(names[3].kind == WEIGHT_KIND_NORM);
    assert(strcmp(names[4].name, "lm_head") == 0);
    assert(names[4].kind == WEIGHT_KIND_MATMUL);
    assert(strcmp(names[5].name, "model.embed_tokens_per_layer") == 0);
    assert(names[5].kind == WEIGHT_KIND_EMBED);
    assert(strcmp(names[6].name, "model.v_norm_global") == 0);
    assert(names[6].kind == WEIGHT_KIND_NORM);
}

/* ---- main ---- */

int main(void) {
    test_shard_index_preferred();
    printf("  test_shard_index_preferred: passed\n");

    test_shard_single_file();
    printf("  test_shard_single_file: passed\n");

    test_shard_glob_sorted();
    printf("  test_shard_glob_sorted: passed\n");

    test_shard_missing_error();
    printf("  test_shard_missing_error: passed\n");

    test_tensor_name_with_prefix();
    printf("  test_tensor_name_with_prefix: passed\n");

    test_tensor_name_global();
    printf("  test_tensor_name_global: passed\n");

    test_tensor_name_empty_prefix();
    printf("  test_tensor_name_empty_prefix: passed\n");

    test_tensor_name_long_prefix();
    printf("  test_tensor_name_long_prefix: passed\n");

    test_tensor_name_buffer_overflow();
    printf("  test_tensor_name_buffer_overflow: passed\n");

    test_malformed_index_not_glob();
    printf("  test_malformed_index_not_glob: passed\n");

    test_empty_weight_map_not_glob();
    printf("  test_empty_weight_map_not_glob: passed\n");

    test_enumerate_zeros_outputs_on_failure();
    printf("  test_enumerate_zeros_outputs_on_failure: passed\n");

    test_index_path_escape_rejected();
    printf("  test_index_path_escape_rejected: passed\n");

    test_glob_skips_directory();
    printf("  test_glob_skips_directory: passed\n");

    test_single_is_directory();
    printf("  test_single_is_directory: passed\n");

    test_expected_names_qwen3();
    printf("  test_expected_names_qwen3: passed\n");

    test_expected_names_tied();
    printf("  test_expected_names_tied: passed\n");

    test_expected_names_no_qk_norm();
    printf("  test_expected_names_no_qk_norm: passed\n");

    test_expected_names_llama();
    printf("  test_expected_names_llama: passed\n");

    test_expected_names_qwen3_exact_order();
    printf("  test_expected_names_qwen3_exact_order: passed\n");

    test_expected_names_gemma4();
    printf("  test_expected_names_gemma4: passed\n");

    test_expected_names_other_families_zero();
    printf("  test_expected_names_other_families_zero: passed\n");

    test_expected_names_from_desc_synthetic();
    printf("  test_expected_names_from_desc_synthetic: passed\n");

    test_expected_names_from_desc_biases();
    printf("  test_expected_names_from_desc_biases: passed\n");

    test_expected_names_from_desc_extras();
    printf("  test_expected_names_from_desc_extras: passed\n");

    printf("test_weights: all passed\n");
    return 0;
}
