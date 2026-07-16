#include "registry/registry.h"
#include "registry/registry_internal.h"

#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void scan_dir_stats(const char *dir, uint64_t *size_out, int64_t *mtime_out) {
    *size_out = 0;
    *mtime_out = 0;
    DIR *d = opendir(dir);
    if (!d)
        return;
    struct dirent *ent;
    while ((ent = readdir(d))) {
        if (ent->d_name[0] == '.')
            continue;
        char *full = reg_path_join(dir, ent->d_name);
        if (!full)
            continue;
        struct stat st;
        if (stat(full, &st) == 0 && S_ISREG(st.st_mode)) {
            *size_out += (uint64_t)st.st_size;
            if (st.st_mtime > *mtime_out)
                *mtime_out = st.st_mtime;
        }
        free(full);
    }
    closedir(d);
}

/* HF Hub forbids "--" in org/model names, so the first separator is unambiguous. */
static char *id_from_dirname(const char *dirname) {
    const char *sep = strstr(dirname, "--");
    if (!sep || sep == dirname)
        return NULL;
    size_t org_len = (size_t)(sep - dirname);
    const char *model = sep + 2;
    if (model[0] == '\0')
        return NULL;
    size_t model_len = strlen(model);
    char *id = malloc(org_len + 1 + model_len + 1);
    if (!id)
        return NULL;
    memcpy(id, dirname, org_len);
    id[org_len] = '/';
    memcpy(id + org_len + 1, model, model_len);
    id[org_len + 1 + model_len] = '\0';
    return id;
}

static registry_model_info_t *scan_mlxd_cache(int *count) {
    *count = 0;
    char *root = reg_cache_root();
    if (!root)
        return NULL;
    char *models_dir = reg_path_join(root, "models");
    free(root);
    if (!models_dir)
        return NULL;

    DIR *d = opendir(models_dir);
    if (!d) {
        free(models_dir);
        return NULL;
    }

    int cap = 8;
    registry_model_info_t *arr = calloc((size_t)cap, sizeof(*arr));
    if (!arr) {
        closedir(d);
        free(models_dir);
        return NULL;
    }

    struct dirent *ent;
    while ((ent = readdir(d))) {
        if (ent->d_name[0] == '.')
            continue;
        char *full = reg_path_join(models_dir, ent->d_name);
        if (!full)
            continue;

        if (!reg_dir_has_config_json(full)) {
            free(full);
            continue;
        }

        char *id = id_from_dirname(ent->d_name);
        if (!id) {
            free(full);
            continue;
        }

        if (*count >= cap) {
            cap *= 2;
            registry_model_info_t *tmp = realloc(arr, (size_t)cap * sizeof(*arr));
            if (!tmp) {
                free(id);
                free(full);
                continue;
            }
            arr = tmp;
        }

        uint64_t sz;
        int64_t mt;
        scan_dir_stats(full, &sz, &mt);

        arr[*count].id = id;
        arr[*count].path = full;
        arr[*count].size_bytes = sz;
        arr[*count].mtime = mt;
        (*count)++;
    }

    closedir(d);
    free(models_dir);

    if (*count == 0) {
        free(arr);
        return NULL;
    }
    return arr;
}

char *reg_hf_hub_dir(void) {
    const char *env = getenv("MLXD_HF_HUB_DIR");
    if (env && env[0])
        return reg_dup_str(env);
    const char *hf_home = getenv("HF_HOME");
    if (hf_home && hf_home[0])
        return reg_path_join(hf_home, "hub");
    const char *home = getenv("HOME");
    if (!home)
        return NULL;
    char *cache = reg_path_join(home, ".cache/huggingface/hub");
    return cache;
}

char *reg_find_latest_snapshot(const char *model_dir) {
    char *snap_dir = reg_path_join(model_dir, "snapshots");
    if (!snap_dir)
        return NULL;
    DIR *d = opendir(snap_dir);
    if (!d) {
        free(snap_dir);
        return NULL;
    }
    char *best = NULL;
    struct stat best_st = {0};
    struct dirent *ent;
    while ((ent = readdir(d))) {
        if (ent->d_name[0] == '.')
            continue;
        char *full = reg_path_join(snap_dir, ent->d_name);
        if (!full)
            continue;
        if (!reg_dir_has_config_json(full)) {
            free(full);
            continue;
        }
        struct stat st;
        if (stat(full, &st) == 0) {
            if (!best || st.st_mtime > best_st.st_mtime) {
                free(best);
                best = full;
                best_st = st;
            } else {
                free(full);
            }
        } else {
            free(full);
        }
    }
    closedir(d);
    free(snap_dir);
    return best;
}

static char *hub_id_from_dirname(const char *dirname) {
    const char *prefix = "models--";
    size_t plen = 8;
    if (strncmp(dirname, prefix, plen) != 0)
        return NULL;
    const char *rest = dirname + plen;
    const char *sep = strstr(rest, "--");
    if (!sep || sep == rest)
        return NULL;
    size_t org_len = (size_t)(sep - rest);
    const char *model = sep + 2;
    if (model[0] == '\0')
        return NULL;
    size_t model_len = strlen(model);
    char *id = malloc(org_len + 1 + model_len + 1);
    if (!id)
        return NULL;
    memcpy(id, rest, org_len);
    id[org_len] = '/';
    memcpy(id + org_len + 1, model, model_len);
    id[org_len + 1 + model_len] = '\0';
    return id;
}

static int id_in_list(const registry_model_info_t *arr, int count, const char *id) {
    for (int i = 0; i < count; i++) {
        if (strcmp(arr[i].id, id) == 0)
            return 1;
    }
    return 0;
}

static registry_model_info_t *scan_hub_cache(registry_model_info_t *arr, int *count, int *cap) {
    char *hub = reg_hf_hub_dir();
    if (!hub)
        return arr;
    DIR *d = opendir(hub);
    if (!d) {
        free(hub);
        return arr;
    }

    struct dirent *ent;
    while ((ent = readdir(d))) {
        if (ent->d_name[0] == '.')
            continue;
        char *id = hub_id_from_dirname(ent->d_name);
        if (!id)
            continue;
        if (arr && id_in_list(arr, *count, id)) {
            free(id);
            continue;
        }
        char *model_dir = reg_path_join(hub, ent->d_name);
        if (!model_dir) {
            free(id);
            continue;
        }
        char *snap_path = reg_find_latest_snapshot(model_dir);
        free(model_dir);
        if (!snap_path) {
            free(id);
            continue;
        }

        if (!arr) {
            *cap = 8;
            arr = calloc((size_t)*cap, sizeof(*arr));
            if (!arr) {
                free(id);
                free(snap_path);
                continue;
            }
        }
        if (*count >= *cap) {
            *cap *= 2;
            registry_model_info_t *tmp = realloc(arr, (size_t)*cap * sizeof(*arr));
            if (!tmp) {
                free(id);
                free(snap_path);
                continue;
            }
            arr = tmp;
        }

        uint64_t sz;
        int64_t mt;
        scan_dir_stats(snap_path, &sz, &mt);

        arr[*count].id = id;
        arr[*count].path = snap_path;
        arr[*count].size_bytes = sz;
        arr[*count].mtime = mt;
        (*count)++;
    }
    closedir(d);
    free(hub);
    return arr;
}

registry_model_info_t *registry_discover(int *count) {
    if (!count)
        return NULL;
    int cap = 0;
    registry_model_info_t *arr = scan_mlxd_cache(count);
    if (arr)
        cap = *count < 8 ? 8 : *count;
    arr = scan_hub_cache(arr, count, &cap);
    if (*count == 0 && arr) {
        free(arr);
        return NULL;
    }
    return arr;
}

void registry_model_list_free(registry_model_info_t *models, int count) {
    if (!models)
        return;
    for (int i = 0; i < count; i++) {
        free(models[i].id);
        free(models[i].path);
    }
    free(models);
}
