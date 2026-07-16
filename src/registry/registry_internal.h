#ifndef MLXD_REGISTRY_INTERNAL_H
#define MLXD_REGISTRY_INTERNAL_H

#include "registry/registry.h"

#include <stddef.h>
#include <stdint.h>

#define MLXD_USER_AGENT "mlxd/0.1.0"

typedef struct {
    char *org;
    char *model;
    char *revision;
    char *local_path;
} reg_spec_t;

int   reg_spec_parse(const char *spec, reg_spec_t *out);
void  reg_spec_free(reg_spec_t *s);

char *reg_cache_root(void);
char *reg_model_cache_dir(const char *org, const char *model);
char *reg_endpoint(void);

int   reg_file_wanted(const char *rfilename);
int   reg_rfilename_safe(const char *rfilename);

int   reg_parse_file_plan(const char *json, size_t len, char ***files,
                          int64_t **sizes, size_t *n);
void  reg_file_plan_free(char **files, int64_t *sizes, size_t n);

int   reg_download_file(const char *url, const char *dest, const char *token,
                        const char *filename,
                        registry_progress_cb cb, void *ud, size_t idx, size_t count);

void  reg_curl_init_once(void);
char *reg_meta_read_revision(const char *dir);

char *reg_dup_str(const char *s);
char *reg_path_join(const char *dir, const char *name);
int   reg_dir_has_config_json(const char *dir);

char *reg_hf_hub_dir(void);
char *reg_find_latest_snapshot(const char *model_dir);

#endif
