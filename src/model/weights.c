#include "model/weights.h"
#include "core/log.h"
#include "mlxbridge/mlxbridge.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <yyjson/yyjson.h>

/* ---------- CPU helpers (no mlx calls) ---------- */

static char *path_join(const char *dir, const char *name) {
    size_t dlen = strlen(dir);
    size_t nlen = strlen(name);
    char *p = malloc(dlen + 1 + nlen + 1);
    if (!p) return NULL;
    memcpy(p, dir, dlen);
    p[dlen] = '/';
    memcpy(p + dlen + 1, name, nlen + 1);
    return p;
}

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

static int shards_from_index(const char *model_dir, char ***out_paths,
                             size_t *out_count) {
    char *idx_path = path_join(model_dir, "model.safetensors.index.json");
    if (!idx_path) return -1;

    yyjson_doc *doc = yyjson_read_file(idx_path, 0, NULL, NULL);
    free(idx_path);
    if (!doc) return -1;

    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *wm = yyjson_obj_get(root, "weight_map");
    if (!wm || !yyjson_is_obj(wm)) {
        yyjson_doc_free(doc);
        return -1;
    }

    size_t cap = 16, n = 0;
    char **files = malloc(cap * sizeof(char *));
    if (!files) { yyjson_doc_free(doc); return -1; }

    yyjson_obj_iter it;
    yyjson_obj_iter_init(wm, &it);
    yyjson_val *key, *val;
    while ((key = yyjson_obj_iter_next(&it)) != NULL) {
        val = yyjson_obj_iter_get_val(key);
        const char *fname = yyjson_get_str(val);
        if (!fname) continue;

        bool dup = false;
        for (size_t i = 0; i < n; i++) {
            const char *base = strrchr(files[i], '/');
            const char *cmp = base ? base + 1 : files[i];
            if (strcmp(cmp, fname) == 0) { dup = true; break; }
        }
        if (dup) continue;

        if (n >= cap) {
            cap *= 2;
            char **tmp = realloc(files, cap * sizeof(char *));
            if (!tmp) goto fail;
            files = tmp;
        }
        char *full = path_join(model_dir, fname);
        if (!full) goto fail;
        files[n++] = full;
    }
    yyjson_doc_free(doc);

    if (n == 0) { free(files); return -1; }

    qsort(files, n, sizeof(char *), cmp_str);
    *out_paths = files;
    *out_count = n;
    return 0;

fail:
    for (size_t i = 0; i < n; i++) free(files[i]);
    free(files);
    yyjson_doc_free(doc);
    return -1;
}

static int shards_single(const char *model_dir, char ***out_paths,
                          size_t *out_count) {
    char *p = path_join(model_dir, "model.safetensors");
    if (!p) return -1;

    struct stat st;
    if (stat(p, &st) != 0) { free(p); return -1; }

    char **files = malloc(sizeof(char *));
    if (!files) { free(p); return -1; }
    files[0] = p;
    *out_paths = files;
    *out_count = 1;
    return 0;
}

static int shards_glob(const char *model_dir, char ***out_paths,
                        size_t *out_count) {
    DIR *d = opendir(model_dir);
    if (!d) return -1;

    size_t cap = 16, n = 0;
    char **files = malloc(cap * sizeof(char *));
    if (!files) { closedir(d); return -1; }

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        size_t len = strlen(ent->d_name);
        if (len < 13) continue;
        if (strcmp(ent->d_name + len - 12, ".safetensors") != 0) continue;
        if (n >= cap) {
            cap *= 2;
            char **tmp = realloc(files, cap * sizeof(char *));
            if (!tmp) goto fail;
            files = tmp;
        }
        char *full = path_join(model_dir, ent->d_name);
        if (!full) goto fail;
        files[n++] = full;
    }
    closedir(d);

    if (n == 0) { free(files); return -1; }

    qsort(files, n, sizeof(char *), cmp_str);
    *out_paths = files;
    *out_count = n;
    return 0;

fail:
    for (size_t i = 0; i < n; i++) free(files[i]);
    free(files);
    closedir(d);
    return -1;
}

int weights_enumerate_shards(const char *model_dir, char ***paths,
                             size_t *count, bool *from_index) {
    if (!model_dir || !paths || !count) return -1;

    if (shards_from_index(model_dir, paths, count) == 0) {
        if (from_index) *from_index = true;
        return 0;
    }
    if (shards_single(model_dir, paths, count) == 0) {
        if (from_index) *from_index = false;
        return 0;
    }
    if (shards_glob(model_dir, paths, count) == 0) {
        if (from_index) *from_index = false;
        return 0;
    }
    return -1;
}

void weights_free_shard_paths(char **paths, size_t count) {
    if (!paths) return;
    for (size_t i = 0; i < count; i++)
        free(paths[i]);
    free(paths);
}

int weights_tensor_name(char *buf, size_t n, const model_config_t *cfg,
                        int layer, const char *suffix) {
    if (!buf || !n || !cfg || !suffix) return -1;

    const char *prefix = cfg->weight_prefix ? cfg->weight_prefix : "";
    bool empty_prefix = (prefix[0] == '\0');

    int written;
    if (layer < 0) {
        if (empty_prefix)
            written = snprintf(buf, n, "%s", suffix);
        else
            written = snprintf(buf, n, "%s.%s", prefix, suffix);
    } else {
        if (empty_prefix)
            written = snprintf(buf, n, "layers.%d.%s", layer, suffix);
        else
            written = snprintf(buf, n, "%s.layers.%d.%s", prefix, layer, suffix);
    }

    if (written < 0 || (size_t)written >= n)
        return -1;
    return 0;
}

/* ---------- GPU / engine-thread helpers ---------- */

static int merge_map(mlx_map_string_to_array dst,
                     mlx_map_string_to_array src) {
    mlx_map_string_to_array_iterator it =
        mlx_map_string_to_array_iterator_new(src);
    const char *key = NULL;
    mlx_array val = mlx_array_new();
    while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0 &&
           key != NULL) {
        if (mlx_map_string_to_array_insert(dst, key, val) != 0) {
            mlx_array_free(val);
            mlx_map_string_to_array_iterator_free(it);
            return -1;
        }
        key = NULL;
    }
    mlx_array_free(val);
    mlx_map_string_to_array_iterator_free(it);
    return 0;
}

static size_t compute_total_bytes(mlx_map_string_to_array params) {
    size_t total = 0;
    mlx_map_string_to_array_iterator it =
        mlx_map_string_to_array_iterator_new(params);
    const char *key = NULL;
    mlx_array val = mlx_array_new();
    while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0 &&
           key != NULL) {
        total += mlx_array_size(val) * mlx_array_itemsize(val);
        key = NULL;
    }
    mlx_array_free(val);
    mlx_map_string_to_array_iterator_free(it);
    return total;
}

int weights_load(weights_t *w, const char *model_dir,
                 const model_config_t *cfg, char *err, size_t errlen) {
    if (!w || !model_dir || !cfg) return -1;
    memset(w, 0, sizeof(*w));

    if (cfg->family == MODEL_DEEPSEEK_V4) {
        if (err && errlen > 0)
            snprintf(err, errlen, "deepseek_v4 requires GGUF format (not supported)");
        return -1;
    }

    char **shard_paths = NULL;
    size_t shard_count = 0;
    bool from_index = false;
    if (weights_enumerate_shards(model_dir, &shard_paths, &shard_count,
                                 &from_index) != 0) {
        if (err && errlen > 0)
            snprintf(err, errlen, "no safetensors files found in %s", model_dir);
        return -1;
    }

    mlx_map_string_to_array merged = mlx_map_string_to_array_new();

    for (size_t i = 0; i < shard_count; i++) {
        mlx_map_string_to_array shard_params = mlx_map_string_to_array_new();
        mlx_map_string_to_string shard_meta = mlx_map_string_to_string_new();

        if (mlxbridge_load_safetensors(&shard_params, &shard_meta,
                                       shard_paths[i]) != 0) {
            if (err && errlen > 0)
                snprintf(err, errlen, "failed to load %s", shard_paths[i]);
            mlxbridge_map_free(shard_params, shard_meta);
            mlx_map_string_to_array_free(merged);
            weights_free_shard_paths(shard_paths, shard_count);
            return -1;
        }

        if (merge_map(merged, shard_params) != 0) {
            if (err && errlen > 0)
                snprintf(err, errlen, "failed merging shard %s", shard_paths[i]);
            mlxbridge_map_free(shard_params, shard_meta);
            mlx_map_string_to_array_free(merged);
            weights_free_shard_paths(shard_paths, shard_count);
            return -1;
        }
        mlxbridge_map_free(shard_params, shard_meta);
    }
    weights_free_shard_paths(shard_paths, shard_count);

    w->params = merged;
    w->meta = mlx_map_string_to_string_new();
    w->count = mlxbridge_map_count(merged);
    w->total_bytes = compute_total_bytes(merged);

    log_info("weights: loaded %zu tensors (%zu bytes) from %zu shard(s)",
             w->count, w->total_bytes, shard_count);
    return 0;
}

int weights_get(mlx_array *out, const weights_t *w, const char *name) {
    if (!w || !out || !name) return -1;
    return mlxbridge_map_get(out, w->params, name);
}

int weights_get_triplet(weight_triplet_t *out, const weights_t *w,
                        const char *base) {
    if (!w || !out || !base) return -1;

    char name[256];
    snprintf(name, sizeof(name), "%s.weight", base);
    mlx_array weight = mlx_array_new();
    if (mlxbridge_map_get(&weight, w->params, name) != 0) {
        mlx_array_free(weight);
        return -1;
    }

    snprintf(name, sizeof(name), "%s.scales", base);
    mlx_array scales = mlx_array_new();
    bool has_scales = (mlxbridge_map_get(&scales, w->params, name) == 0);

    snprintf(name, sizeof(name), "%s.biases", base);
    mlx_array biases = mlx_array_new();
    bool has_biases = (mlxbridge_map_get(&biases, w->params, name) == 0);

    out->weight = weight;
    out->scales = scales;
    out->biases = biases;
    out->quantized = has_scales && has_biases;
    return 0;
}

size_t weights_count(const weights_t *w) {
    return w ? w->count : 0;
}

size_t weights_total_bytes(const weights_t *w) {
    return w ? w->total_bytes : 0;
}

void weights_free(weights_t *w) {
    if (!w) return;
    if (w->params.ctx)
        mlx_map_string_to_array_free(w->params);
    if (w->meta.ctx)
        mlx_map_string_to_string_free(w->meta);
    memset(w, 0, sizeof(*w));
}
