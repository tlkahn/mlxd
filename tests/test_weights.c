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

    printf("test_weights: all passed\n");
    return 0;
}
