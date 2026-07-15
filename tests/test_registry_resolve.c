#include "registry/registry.h"
#include "registry/registry_internal.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(void) {
    test_resolve_org_model();
    test_resolve_not_found();
    test_resolve_local_path();
    test_resolve_local_path_no_config();
    test_resolve_null();
    printf("test_registry_resolve: all passed\n");
    return 0;
}
