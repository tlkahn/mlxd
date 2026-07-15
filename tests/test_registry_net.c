#include "registry/registry.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

extern char *mkdtemp(char *);

int main(void) {
    if (!getenv("MLXD_NET_TESTS")) {
        printf("test_registry_net: skipped (set MLXD_NET_TESTS=1 to enable)\n");
        return 0;
    }

    char tmpdir[] = "/tmp/mlxd_net_test_XXXXXX";
    assert(mkdtemp(tmpdir) != NULL);
    setenv("MLXD_CACHE_DIR", tmpdir, 1);

    char *dir = registry_pull("mlx-community/Qwen3-0.6B-4bit", NULL);
    assert(dir != NULL);

    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", dir);
    struct stat st;
    assert(stat(config_path, &st) == 0);

    char *resolved = registry_resolve("mlx-community/Qwen3-0.6B-4bit");
    assert(resolved != NULL);
    assert(strcmp(resolved, dir) == 0);
    free(resolved);

    int count = 0;
    registry_model_info_t *models = model_discover(&count);
    assert(count >= 1);
    int found = 0;
    for (int i = 0; i < count; i++) {
        if (strcmp(models[i].id, "mlx-community/Qwen3-0.6B-4bit") == 0)
            found = 1;
    }
    assert(found);
    registry_model_list_free(models, count);

    free(dir);
    unsetenv("MLXD_CACHE_DIR");
    printf("test_registry_net: all passed\n");
    return 0;
}
