#include "registry/registry.h"
#include "registry/registry_internal.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

extern char *mkdtemp(char *);

#define FIXTURE_CACHE MLXD_FIXTURES_DIR "/registry_cache"

static void test_resolve_org_model(void) {
    setenv("MLXD_CACHE_DIR", FIXTURE_CACHE, 1);
    char *path = registry_resolve("mlx-community/Qwen3-0.6B-4bit");
    assert(path != NULL);
    assert(strstr(path, "mlx-community--Qwen3-0.6B-4bit") != NULL);
    free(path);
    unsetenv("MLXD_CACHE_DIR");
}

static void test_resolve_not_found(void) {
    setenv("MLXD_CACHE_DIR", FIXTURE_CACHE, 1);
    char *path = registry_resolve("nonexistent/model");
    assert(path == NULL);
    unsetenv("MLXD_CACHE_DIR");
}

static void test_resolve_local_path(void) {
    char *path = registry_resolve(FIXTURE_CACHE "/models/mlx-community--Qwen3-0.6B-4bit");
    assert(path != NULL);
    assert(strstr(path, "mlx-community--Qwen3-0.6B-4bit") != NULL);
    free(path);
}

static void test_resolve_local_path_no_config(void) {
    char *path = registry_resolve(FIXTURE_CACHE "/models/not-a-model");
    assert(path == NULL);
}

static void test_resolve_null(void) {
    assert(registry_resolve(NULL) == NULL);
}

static char tmpdir_buf[256];

static void make_tmpdir(void) {
    snprintf(tmpdir_buf, sizeof(tmpdir_buf), "%s", "/tmp/mlxd_test_resolve_XXXXXX");
    assert(mkdtemp(tmpdir_buf) != NULL);
}

static int mkdirs(const char *path) {
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
                return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
        return -1;
    return 0;
}

static void write_file(const char *path, const char *data) {
    FILE *f = fopen(path, "w");
    assert(f != NULL);
    fputs(data, f);
    fclose(f);
}

static void test_resolve_revision_match(void) {
    make_tmpdir();
    setenv("MLXD_CACHE_DIR", tmpdir_buf, 1);

    char model_dir[512];
    snprintf(model_dir, sizeof(model_dir), "%s/models/org--mymodel", tmpdir_buf);
    assert(mkdirs(model_dir) == 0);

    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);
    write_file(config_path, "{\"model_type\":\"test\"}");

    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s/.mlxd-meta.json", model_dir);
    write_file(meta_path, "{\"revision\":\"v1\"}");

    char *path = registry_resolve("org/mymodel:v1");
    assert(path != NULL);
    free(path);

    unsetenv("MLXD_CACHE_DIR");
}

static void test_resolve_revision_mismatch(void) {
    make_tmpdir();
    setenv("MLXD_CACHE_DIR", tmpdir_buf, 1);

    char model_dir[512];
    snprintf(model_dir, sizeof(model_dir), "%s/models/org--mymodel2", tmpdir_buf);
    assert(mkdirs(model_dir) == 0);

    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);
    write_file(config_path, "{\"model_type\":\"test\"}");

    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s/.mlxd-meta.json", model_dir);
    write_file(meta_path, "{\"revision\":\"v1\"}");

    char *path = registry_resolve("org/mymodel2:v2");
    assert(path == NULL);

    unsetenv("MLXD_CACHE_DIR");
}

static void test_resolve_no_revision_returns_dir(void) {
    make_tmpdir();
    setenv("MLXD_CACHE_DIR", tmpdir_buf, 1);

    char model_dir[512];
    snprintf(model_dir, sizeof(model_dir), "%s/models/org--mymodel3", tmpdir_buf);
    assert(mkdirs(model_dir) == 0);

    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);
    write_file(config_path, "{\"model_type\":\"test\"}");

    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s/.mlxd-meta.json", model_dir);
    write_file(meta_path, "{\"revision\":\"v1\"}");

    char *path = registry_resolve("org/mymodel3");
    assert(path != NULL);
    free(path);

    unsetenv("MLXD_CACHE_DIR");
}

static void test_resolve_tilde_expansion(void) {
    make_tmpdir();

    char model_dir[512];
    snprintf(model_dir, sizeof(model_dir), "%s/models/foo", tmpdir_buf);
    assert(mkdirs(model_dir) == 0);

    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);
    write_file(config_path, "{\"model_type\":\"test\"}");

    const char *h = getenv("HOME");
    char *orig_home = h ? strdup(h) : NULL;
    setenv("HOME", tmpdir_buf, 1);

    char *path = registry_resolve("~/models/foo");
    assert(path != NULL);
    assert(strcmp(path, model_dir) == 0);
    free(path);

    if (orig_home) { setenv("HOME", orig_home, 1); free(orig_home); }
    else unsetenv("HOME");
}

static void test_resolve_revision_missing_meta(void) {
    make_tmpdir();
    setenv("MLXD_CACHE_DIR", tmpdir_buf, 1);

    char model_dir[512];
    snprintf(model_dir, sizeof(model_dir), "%s/models/org--mymodel4", tmpdir_buf);
    assert(mkdirs(model_dir) == 0);

    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);
    write_file(config_path, "{\"model_type\":\"test\"}");

    char *path = registry_resolve("org/mymodel4:v1");
    assert(path == NULL);

    unsetenv("MLXD_CACHE_DIR");
}

int main(void) {
    test_resolve_org_model();
    test_resolve_not_found();
    test_resolve_local_path();
    test_resolve_local_path_no_config();
    test_resolve_null();
    test_resolve_revision_match();
    test_resolve_revision_mismatch();
    test_resolve_no_revision_returns_dir();
    test_resolve_tilde_expansion();
    test_resolve_revision_missing_meta();
    printf("test_registry_resolve: all passed\n");
    return 0;
}
