#include "registry/registry.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE_CACHE MLXD_FIXTURES_DIR "/registry_cache"

static void test_discover_finds_models(void) {
    setenv("MLXD_CACHE_DIR", FIXTURE_CACHE, 1);
    unsetenv("MLXD_HF_HUB_DIR");
    int count = -1;
    registry_model_info_t *models = model_discover(&count);
    assert(count == 2);
    assert(models != NULL);

    int found_qwen = 0, found_other = 0;
    for (int i = 0; i < count; i++) {
        assert(models[i].id != NULL);
        assert(models[i].path != NULL);
        assert(models[i].size_bytes > 0);
        assert(models[i].mtime > 0);
        if (strcmp(models[i].id, "mlx-community/Qwen3-0.6B-4bit") == 0)
            found_qwen = 1;
        if (strcmp(models[i].id, "org/other-model") == 0)
            found_other = 1;
    }
    assert(found_qwen);
    assert(found_other);

    registry_model_list_free(models, count);
    unsetenv("MLXD_CACHE_DIR");
}

static void test_discover_skips_no_config(void) {
    setenv("MLXD_CACHE_DIR", FIXTURE_CACHE, 1);
    unsetenv("MLXD_HF_HUB_DIR");
    int count = -1;
    registry_model_info_t *models = model_discover(&count);
    for (int i = 0; i < count; i++) {
        assert(strcmp(models[i].id, "not-a-model") != 0);
    }
    registry_model_list_free(models, count);
    unsetenv("MLXD_CACHE_DIR");
}

static void test_discover_skips_part_only(void) {
    setenv("MLXD_CACHE_DIR", FIXTURE_CACHE, 1);
    unsetenv("MLXD_HF_HUB_DIR");
    int count = -1;
    registry_model_info_t *models = model_discover(&count);
    for (int i = 0; i < count; i++) {
        assert(strstr(models[i].id, "half") == NULL);
    }
    registry_model_list_free(models, count);
    unsetenv("MLXD_CACHE_DIR");
}

static void test_discover_empty_cache(void) {
    setenv("MLXD_CACHE_DIR", "/nonexistent/path", 1);
    unsetenv("MLXD_HF_HUB_DIR");
    int count = -1;
    registry_model_info_t *models = model_discover(&count);
    assert(count == 0);
    assert(models == NULL);
    unsetenv("MLXD_CACHE_DIR");
}

static void test_discover_hub_cache(void) {
    setenv("MLXD_CACHE_DIR", "/nonexistent", 1);
    setenv("MLXD_HF_HUB_DIR", MLXD_FIXTURES_DIR "/hf_hub_cache/hub", 1);
    int count = -1;
    registry_model_info_t *models = model_discover(&count);
    assert(count == 1);
    assert(models != NULL);
    assert(strcmp(models[0].id, "org/hubmodel") == 0);
    assert(strstr(models[0].path, "snapshots/abc123") != NULL);
    assert(models[0].size_bytes > 0);
    registry_model_list_free(models, count);
    unsetenv("MLXD_CACHE_DIR");
    unsetenv("MLXD_HF_HUB_DIR");
}

static void test_discover_dedup_mlxd_wins(void) {
    setenv("MLXD_CACHE_DIR", FIXTURE_CACHE, 1);
    setenv("MLXD_HF_HUB_DIR", MLXD_FIXTURES_DIR "/hf_hub_cache/hub", 1);
    int count = -1;
    registry_model_info_t *models = model_discover(&count);
    assert(models != NULL);
    int hub_only_found = 0;
    for (int i = 0; i < count; i++) {
        if (strcmp(models[i].id, "org/hubmodel") == 0) {
            hub_only_found = 1;
            assert(strstr(models[i].path, "snapshots/abc123") != NULL);
        }
        if (strcmp(models[i].id, "mlx-community/Qwen3-0.6B-4bit") == 0) {
            assert(strstr(models[i].path, "registry_cache") != NULL);
        }
    }
    assert(hub_only_found);
    registry_model_list_free(models, count);
    unsetenv("MLXD_CACHE_DIR");
    unsetenv("MLXD_HF_HUB_DIR");
}

static void test_free_null_safe(void) {
    registry_model_list_free(NULL, 0);
    registry_model_list_free(NULL, 5);
}

static void test_discover_null_count(void) {
    setenv("MLXD_CACHE_DIR", FIXTURE_CACHE, 1);
    unsetenv("MLXD_HF_HUB_DIR");
    registry_model_info_t *models = model_discover(NULL);
    assert(models == NULL);
    unsetenv("MLXD_CACHE_DIR");
}

int main(void) {
    test_discover_finds_models();
    test_discover_skips_no_config();
    test_discover_skips_part_only();
    test_discover_empty_cache();
    test_discover_hub_cache();
    test_discover_dedup_mlxd_wins();
    test_free_null_safe();
    test_discover_null_count();
    printf("test_registry_discover: all passed\n");
    return 0;
}
