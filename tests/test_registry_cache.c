#include "registry/registry_internal.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_cache_root_env_override(void) {
    setenv("MLXD_CACHE_DIR", "/custom/cache", 1);
    char *root = reg_cache_root();
    assert(root != NULL);
    assert(strcmp(root, "/custom/cache") == 0);
    free(root);
    unsetenv("MLXD_CACHE_DIR");
}

static void test_cache_root_home_fallback(void) {
    unsetenv("MLXD_CACHE_DIR");
    char *home = getenv("HOME");
    assert(home != NULL);
    char *root = reg_cache_root();
    assert(root != NULL);
    char expected[4096];
    snprintf(expected, sizeof(expected), "%s/.cache/mlxd", home);
    assert(strcmp(root, expected) == 0);
    free(root);
}

static void test_model_cache_dir(void) {
    setenv("MLXD_CACHE_DIR", "/cache", 1);
    char *dir = reg_model_cache_dir("myorg", "mymodel");
    assert(dir != NULL);
    assert(strcmp(dir, "/cache/models/myorg--mymodel") == 0);
    free(dir);
    unsetenv("MLXD_CACHE_DIR");
}

static void test_endpoint_default(void) {
    unsetenv("HF_ENDPOINT");
    char *ep = reg_endpoint();
    assert(ep != NULL);
    assert(strcmp(ep, "https://huggingface.co") == 0);
    free(ep);
}

static void test_endpoint_env_override(void) {
    setenv("HF_ENDPOINT", "http://localhost:8080", 1);
    char *ep = reg_endpoint();
    assert(ep != NULL);
    assert(strcmp(ep, "http://localhost:8080") == 0);
    free(ep);
    unsetenv("HF_ENDPOINT");
}

int main(void) {
    test_cache_root_env_override();
    test_cache_root_home_fallback();
    test_model_cache_dir();
    test_endpoint_default();
    test_endpoint_env_override();
    printf("test_registry_cache: all passed\n");
    return 0;
}
