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

static void test_expected_names_non_qwen3(void) {
    model_config_t cfg = {0};
    cfg.family = MODEL_LLAMA;
    cfg.weight_prefix = "model";
    cfg.num_hidden_layers = 2;

    assert(weights_expected_names(&cfg, NULL, 0) == 0);
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

    test_expected_names_non_qwen3();
    printf("  test_expected_names_non_qwen3: passed\n");

    printf("test_weights: all passed\n");
    return 0;
}
