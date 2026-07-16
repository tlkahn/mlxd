#ifndef MLXD_REGISTRY_H
#define MLXD_REGISTRY_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *filename;
    size_t      file_index;
    size_t      file_count;
    uint64_t    file_downloaded;
    uint64_t    file_total;
} registry_progress_t;

typedef int (*registry_progress_cb)(const registry_progress_t *p, void *userdata);

typedef struct {
    const char          *revision;
    const char          *token;
    registry_progress_cb progress;
    void                *userdata;
} registry_pull_opts_t;

char *registry_pull(const char *spec, const registry_pull_opts_t *opts);
char *registry_resolve(const char *specifier);

typedef struct {
    char    *id;
    char    *path;
    uint64_t size_bytes;
    int64_t  mtime;
} registry_model_info_t;

registry_model_info_t *registry_discover(int *count);
void registry_model_list_free(registry_model_info_t *models, int count);

#endif
