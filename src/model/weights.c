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

/* Returns 0 = ok, 1 = index file absent, -2 = present but invalid */
static int shards_from_index(const char *model_dir, char ***out_paths,
                             size_t *out_count) {
    char *idx_path = path_join(model_dir, "model.safetensors.index.json");
    if (!idx_path) return -2;

    struct stat st;
    if (stat(idx_path, &st) != 0) {
        free(idx_path);
        return 1;
    }

    yyjson_doc *doc = yyjson_read_file(idx_path, 0, NULL, NULL);
    free(idx_path);
    if (!doc) return -2;

    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *wm = yyjson_obj_get(root, "weight_map");
    if (!wm || !yyjson_is_obj(wm)) {
        yyjson_doc_free(doc);
        return -2;
    }

    size_t cap = 16, n = 0;
    char **files = malloc(cap * sizeof(char *));
    if (!files) { yyjson_doc_free(doc); return -2; }

    yyjson_obj_iter it;
    yyjson_obj_iter_init(wm, &it);
    yyjson_val *key, *val;
    while ((key = yyjson_obj_iter_next(&it)) != NULL) {
        val = yyjson_obj_iter_get_val(key);
        const char *fname = yyjson_get_str(val);
        if (!fname) continue;

        size_t flen = strlen(fname);
        if (flen < 13 || strchr(fname, '/') != NULL ||
            strcmp(fname, "..") == 0 || strncmp(fname, "..", 2) == 0 ||
            strcmp(fname + flen - 12, ".safetensors") != 0)
            goto fail;

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

    if (n == 0) { free(files); return -2; }

    qsort(files, n, sizeof(char *), cmp_str);
    *out_paths = files;
    *out_count = n;
    return 0;

fail:
    for (size_t i = 0; i < n; i++) free(files[i]);
    free(files);
    yyjson_doc_free(doc);
    return -2;
}

static int shards_single(const char *model_dir, char ***out_paths,
                          size_t *out_count) {
    char *p = path_join(model_dir, "model.safetensors");
    if (!p) return -1;

    struct stat st;
    if (stat(p, &st) != 0 || !S_ISREG(st.st_mode)) { free(p); return -1; }

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

        char *full = path_join(model_dir, ent->d_name);
        if (!full) goto fail;

        struct stat entry_st;
        if (stat(full, &entry_st) != 0 || !S_ISREG(entry_st.st_mode)) {
            free(full);
            continue;
        }

        if (n >= cap) {
            cap *= 2;
            char **tmp = realloc(files, cap * sizeof(char *));
            if (!tmp) { free(full); goto fail; }
            files = tmp;
        }
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

    *paths = NULL;
    *count = 0;
    if (from_index) *from_index = false;

    int idx_rc = shards_from_index(model_dir, paths, count);
    if (idx_rc == 0) {
        if (from_index) *from_index = true;
        return 0;
    }
    if (idx_rc == -2)
        return -2;

    /* idx_rc == 1: index absent, fall through to single-file / glob */
    if (shards_single(model_dir, paths, count) == 0)
        return 0;
    if (shards_glob(model_dir, paths, count) == 0)
        return 0;
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

/* ---------- Expected tensor names (CPU, no mlx) ---------- */

static int emit_expected(weight_expected_t *out, int capacity, int *pos,
                         const char *name, weight_kind_t kind) {
    if (out) {
        if (*pos >= capacity) return -1;
        size_t len = strlen(name);
        if (len >= sizeof(out[0].name)) return -1;
        memcpy(out[*pos].name, name, len + 1);
        out[*pos].kind = kind;
    }
    (*pos)++;
    return 0;
}

static const char *qwen3_layer_matmuls[] = {
    "self_attn.q_proj", "self_attn.k_proj",
    "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj",    "mlp.up_proj",
    "mlp.down_proj",
};
static const size_t qwen3_n_matmuls = 7;

static const char *qwen3_layer_norms[] = {
    "input_layernorm", "post_attention_layernorm",
};
static const size_t qwen3_n_norms = 2;

static const char *qwen3_layer_qk_norms[] = {
    "self_attn.q_norm", "self_attn.k_norm",
};
static const size_t qwen3_n_qk_norms = 2;

int weights_expected_names(const model_config_t *cfg,
                           weight_expected_t *out, int capacity) {
    if (!cfg) return -1;
    if (cfg->family != MODEL_QWEN3) return 0;

    int pos = 0;
    char buf[256];

    /* embed_tokens */
    if (weights_tensor_name(buf, sizeof(buf), cfg, -1, "embed_tokens") == 0) {
        if (emit_expected(out, capacity, &pos, buf, WEIGHT_KIND_EMBED) != 0)
            return -1;
    }

    /* per-layer tensors */
    for (int layer = 0; layer < cfg->num_hidden_layers; layer++) {
        for (size_t j = 0; j < qwen3_n_matmuls; j++) {
            if (weights_tensor_name(buf, sizeof(buf), cfg, layer,
                                    qwen3_layer_matmuls[j]) == 0) {
                if (emit_expected(out, capacity, &pos, buf,
                                  WEIGHT_KIND_MATMUL) != 0)
                    return -1;
            }
        }
        for (size_t j = 0; j < qwen3_n_norms; j++) {
            if (weights_tensor_name(buf, sizeof(buf), cfg, layer,
                                    qwen3_layer_norms[j]) == 0) {
                if (emit_expected(out, capacity, &pos, buf,
                                  WEIGHT_KIND_NORM) != 0)
                    return -1;
            }
        }
        if (cfg->has_qk_norm) {
            for (size_t j = 0; j < qwen3_n_qk_norms; j++) {
                if (weights_tensor_name(buf, sizeof(buf), cfg, layer,
                                        qwen3_layer_qk_norms[j]) == 0) {
                    if (emit_expected(out, capacity, &pos, buf,
                                      WEIGHT_KIND_NORM) != 0)
                        return -1;
                }
            }
        }
    }

    /* global norm */
    if (weights_tensor_name(buf, sizeof(buf), cfg, -1, "norm") == 0) {
        if (emit_expected(out, capacity, &pos, buf, WEIGHT_KIND_NORM) != 0)
            return -1;
    }

    /* lm_head (not prefixed, not present when tied) */
    if (!cfg->tie_word_embeddings) {
        if (emit_expected(out, capacity, &pos, "lm_head",
                          WEIGHT_KIND_MATMUL) != 0)
            return -1;
    }

    return pos;
}

/* ---------- GPU / engine-thread helpers ---------- */

/* Ownership contract for insert/free: see mlxbridge.h. */
static int merge_map(mlx_map_string_to_array dst,
                     mlx_map_string_to_array src,
                     const char **dup_key) {
    mlx_map_string_to_array_iterator it =
        mlx_map_string_to_array_iterator_new(src);
    const char *key = NULL;
    mlx_array val = mlx_array_new();
    mlx_array probe = mlx_array_new();
    while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0 &&
           key != NULL) {
        if (mlx_map_string_to_array_get(&probe, dst, key) == 0) {
            if (dup_key) *dup_key = key;
            mlx_array_free(probe);
            mlx_array_free(val);
            mlx_map_string_to_array_iterator_free(it);
            return -1;
        }
        if (mlx_map_string_to_array_insert(dst, key, val) != 0) {
            mlx_array_free(probe);
            mlx_array_free(val);
            mlx_map_string_to_array_iterator_free(it);
            return -1;
        }
        key = NULL;
    }
    mlx_array_free(probe);
    mlx_array_free(val);
    mlx_map_string_to_array_iterator_free(it);
    return 0;
}

static void map_stats(mlx_map_string_to_array params,
                      size_t *out_count, size_t *out_bytes) {
    size_t count = 0, total = 0;
    mlx_map_string_to_array_iterator it =
        mlx_map_string_to_array_iterator_new(params);
    const char *key = NULL;
    mlx_array val = mlx_array_new();
    while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0 &&
           key != NULL) {
        count++;
        total += mlx_array_size(val) * mlx_array_itemsize(val);
        key = NULL;
    }
    mlx_array_free(val);
    mlx_map_string_to_array_iterator_free(it);
    *out_count = count;
    *out_bytes = total;
}

static int read_index_keys(const char *model_dir, char ***out_keys,
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

    size_t n_keys = yyjson_obj_size(wm);
    char **keys = malloc(n_keys * sizeof(char *));
    if (!keys) { yyjson_doc_free(doc); return -1; }

    yyjson_obj_iter it;
    yyjson_obj_iter_init(wm, &it);
    yyjson_val *key;
    size_t i = 0;
    while ((key = yyjson_obj_iter_next(&it)) != NULL && i < n_keys) {
        const char *kstr = yyjson_get_str(key);
        if (!kstr) continue;
        keys[i] = strdup(kstr);
        if (!keys[i]) {
            for (size_t j = 0; j < i; j++) free(keys[j]);
            free(keys);
            yyjson_doc_free(doc);
            return -1;
        }
        i++;
    }
    yyjson_doc_free(doc);
    *out_keys = keys;
    *out_count = i;
    return 0;
}

static bool map_has_key(mlx_map_string_to_array map, const char *key) {
    mlx_array probe = mlx_array_new();
    bool found = (mlx_map_string_to_array_get(&probe, map, key) == 0);
    mlx_array_free(probe);
    return found;
}

static bool dtype_is_floating(mlx_dtype dt) {
    return dt == MLX_FLOAT16 || dt == MLX_BFLOAT16 || dt == MLX_FLOAT32 ||
           dt == MLX_FLOAT64;
}

static int map_get_dtype(mlx_map_string_to_array map, const char *key,
                         mlx_dtype *out) {
    mlx_array probe = mlx_array_new();
    if (mlx_map_string_to_array_get(&probe, map, key) != 0) {
        mlx_array_free(probe);
        return -1;
    }
    *out = mlx_array_dtype(probe);
    mlx_array_free(probe);
    return 0;
}

static bool map_has_prefix(mlx_map_string_to_array map, const char *prefix) {
    size_t plen = strlen(prefix);
    mlx_map_string_to_array_iterator it =
        mlx_map_string_to_array_iterator_new(map);
    const char *key = NULL;
    mlx_array val = mlx_array_new();
    bool found = false;
    while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0 &&
           key != NULL) {
        if (strncmp(key, prefix, plen) == 0) { found = true; break; }
        key = NULL;
    }
    mlx_array_free(val);
    mlx_map_string_to_array_iterator_free(it);
    return found;
}

static int weights_validate(mlx_map_string_to_array merged,
                            const model_config_t *cfg,
                            char **index_keys, size_t n_index_keys,
                            char *err, size_t errlen) {
    /* Index cross-check: every declared key must exist in merged */
    if (index_keys) {
        for (size_t i = 0; i < n_index_keys; i++) {
            if (!map_has_key(merged, index_keys[i])) {
                if (err && errlen > 0)
                    snprintf(err, errlen,
                             "index declares tensor \"%s\" but it was not found in shards",
                             index_keys[i]);
                return -1;
            }
        }
    }

    /* Layer coverage */
    const char *prefix = cfg->weight_prefix ? cfg->weight_prefix : "";
    bool empty_prefix = (prefix[0] == '\0');

    for (int i = 0; i < cfg->num_hidden_layers; i++) {
        char layer_prefix[256];
        int wr;
        if (empty_prefix)
            wr = snprintf(layer_prefix, sizeof(layer_prefix), "layers.%d.", i);
        else
            wr = snprintf(layer_prefix, sizeof(layer_prefix),
                          "%s.layers.%d.", prefix, i);
        if (wr < 0 || (size_t)wr >= sizeof(layer_prefix)) continue;

        if (!map_has_prefix(merged, layer_prefix)) {
            if (err && errlen > 0)
                snprintf(err, errlen, "missing tensors for layer %d", i);
            return -1;
        }
    }

    if (cfg->num_hidden_layers > 0) {
        char embed_name[256];
        int wr;
        if (empty_prefix)
            wr = snprintf(embed_name, sizeof(embed_name), "embed_tokens.weight");
        else
            wr = snprintf(embed_name, sizeof(embed_name),
                          "%s.embed_tokens.weight", prefix);
        if (wr >= 0 && (size_t)wr < sizeof(embed_name)) {
            if (!map_has_key(merged, embed_name)) {
                if (err && errlen > 0)
                    snprintf(err, errlen, "missing %s", embed_name);
                return -1;
            }
        }
    }

    /* Quant dtype check */
    if (cfg->quant_bits > 0) {
        bool found_quant_triplet = false;
        mlx_map_string_to_array_iterator it =
            mlx_map_string_to_array_iterator_new(merged);
        const char *key = NULL;
        mlx_array val = mlx_array_new();
        while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0 &&
               key != NULL) {
            size_t klen = strlen(key);
            if (klen > 7 && strcmp(key + klen - 7, ".scales") == 0) {
                char base[256];
                size_t blen = klen - 7;
                if (blen >= sizeof(base)) { key = NULL; continue; }
                memcpy(base, key, blen);
                base[blen] = '\0';

                char wname[270];
                snprintf(wname, sizeof(wname), "%s.weight", base);
                mlx_array warr = mlx_array_new();
                if (mlx_map_string_to_array_get(&warr, merged, wname) == 0) {
                    mlx_dtype wdt = mlx_array_dtype(warr);
                    if (wdt != MLX_UINT32) {
                        if (err && errlen > 0)
                            snprintf(err, errlen,
                                     "quantized weight %s has dtype %d, expected uint32",
                                     wname, wdt);
                        mlx_array_free(warr);
                        mlx_array_free(val);
                        mlx_map_string_to_array_iterator_free(it);
                        return -1;
                    }
                    found_quant_triplet = true;
                }
                mlx_array_free(warr);
            }
            key = NULL;
        }
        mlx_array_free(val);
        mlx_map_string_to_array_iterator_free(it);

        if (!found_quant_triplet) {
            if (err && errlen > 0)
                snprintf(err, errlen,
                         "quantized config (bits=%d) but no quantized weights found (expected uint32 dtype with scales/biases)",
                         cfg->quant_bits);
            return -1;
        }
    }

    /* Orphan-triplet sweep (all families): .scales or .biases without
       a matching sibling or base .weight fails load. */
    {
        mlx_map_string_to_array_iterator it =
            mlx_map_string_to_array_iterator_new(merged);
        const char *key = NULL;
        mlx_array val = mlx_array_new();
        while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0 &&
               key != NULL) {
            size_t klen = strlen(key);
            bool is_scales = (klen > 7 &&
                              strcmp(key + klen - 7, ".scales") == 0);
            bool is_biases = (klen > 7 &&
                              strcmp(key + klen - 7, ".biases") == 0);
            if (is_scales || is_biases) {
                char base[256];
                size_t blen = klen - 7;
                if (blen >= sizeof(base)) { key = NULL; continue; }
                memcpy(base, key, blen);
                base[blen] = '\0';

                char wname[270];
                snprintf(wname, sizeof(wname), "%s.weight", base);
                if (!map_has_key(merged, wname)) {
                    if (err && errlen > 0)
                        snprintf(err, errlen,
                                 "orphan %s for \"%s\" (base .weight missing)",
                                 is_scales ? ".scales" : ".biases", base);
                    mlx_array_free(val);
                    mlx_map_string_to_array_iterator_free(it);
                    return -1;
                }

                char sibling[270];
                snprintf(sibling, sizeof(sibling), "%s.%s", base,
                         is_scales ? "biases" : "scales");
                if (!map_has_key(merged, sibling)) {
                    if (err && errlen > 0)
                        snprintf(err, errlen,
                                 "orphan %s for \"%s\" (missing %s)",
                                 is_scales ? ".scales" : ".biases", base,
                                 is_scales ? ".biases" : ".scales");
                    mlx_array_free(val);
                    mlx_map_string_to_array_iterator_free(it);
                    return -1;
                }
            }
            key = NULL;
        }
        mlx_array_free(val);
        mlx_map_string_to_array_iterator_free(it);
    }

    /* Strict expected-set validation (qwen3 only) */
    {
        int n_expected = weights_expected_names(cfg, NULL, 0);
        if (n_expected > 0) {
            weight_expected_t *expected = malloc(
                (size_t)n_expected * sizeof(weight_expected_t));
            if (!expected) {
                if (err && errlen > 0)
                    snprintf(err, errlen, "allocation failed for expected tensors");
                return -1;
            }
            int erc = weights_expected_names(cfg, expected, n_expected);
            if (erc < 0) {
                free(expected);
                if (err && errlen > 0)
                    snprintf(err, errlen,
                             "failed to generate expected tensor list");
                return -1;
            }

            int strict_rc = 0;
            for (int i = 0; i < erc; i++) {
                char wname[270];
                snprintf(wname, sizeof(wname), "%s.weight", expected[i].name);

                if (!map_has_key(merged, wname)) {
                    if (err && errlen > 0)
                        snprintf(err, errlen,
                                 "missing expected tensor \"%s\"", wname);
                    strict_rc = -1;
                    break;
                }

                mlx_dtype wdt;
                if (map_get_dtype(merged, wname, &wdt) != 0) continue;

                if (expected[i].kind == WEIGHT_KIND_NORM) {
                    if (!dtype_is_floating(wdt)) {
                        if (err && errlen > 0)
                            snprintf(err, errlen,
                                     "norm tensor \"%s\" has non-float dtype %d",
                                     wname, wdt);
                        strict_rc = -1;
                        break;
                    }
                } else {
                    /* MATMUL or EMBED: dense (float) or quantized triplet */
                    if (wdt == MLX_UINT32) {
                        if (cfg->quant_bits <= 0) {
                            if (err && errlen > 0)
                                snprintf(err, errlen,
                                         "tensor \"%s\" is uint32 but config has no quantization",
                                         wname);
                            strict_rc = -1;
                            break;
                        }
                        char sname[270], bname[270];
                        snprintf(sname, sizeof(sname), "%s.scales",
                                 expected[i].name);
                        snprintf(bname, sizeof(bname), "%s.biases",
                                 expected[i].name);
                        if (!map_has_key(merged, sname) ||
                            !map_has_key(merged, bname)) {
                            if (err && errlen > 0)
                                snprintf(err, errlen,
                                         "quantized tensor \"%s\" missing scales/biases",
                                         expected[i].name);
                            strict_rc = -1;
                            break;
                        }
                        mlx_dtype sdt, bdt;
                        if (map_get_dtype(merged, sname, &sdt) == 0 &&
                            !dtype_is_floating(sdt)) {
                            if (err && errlen > 0)
                                snprintf(err, errlen,
                                         "scales \"%s\" has non-float dtype %d",
                                         sname, sdt);
                            strict_rc = -1;
                            break;
                        }
                        if (map_get_dtype(merged, bname, &bdt) == 0 &&
                            !dtype_is_floating(bdt)) {
                            if (err && errlen > 0)
                                snprintf(err, errlen,
                                         "biases \"%s\" has non-float dtype %d",
                                         bname, bdt);
                            strict_rc = -1;
                            break;
                        }
                    } else if (!dtype_is_floating(wdt)) {
                        if (err && errlen > 0)
                            snprintf(err, errlen,
                                     "tensor \"%s\" has unexpected dtype %d "
                                     "(expected float or uint32)",
                                     wname, wdt);
                        strict_rc = -1;
                        break;
                    }
                }
            }
            free(expected);
            if (strict_rc != 0)
                return -1;
        }
    }

    return 0;
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
    int enum_rc = weights_enumerate_shards(model_dir, &shard_paths, &shard_count,
                                           &from_index);
    if (enum_rc != 0) {
        if (err && errlen > 0) {
            if (enum_rc == -2)
                snprintf(err, errlen,
                         "invalid model.safetensors.index.json in %s", model_dir);
            else
                snprintf(err, errlen,
                         "no safetensors files found in %s", model_dir);
        }
        return -1;
    }

    mlx_map_string_to_array merged = mlx_map_string_to_array_new();

    for (size_t i = 0; i < shard_count; i++) {
        struct stat shard_st;
        if (stat(shard_paths[i], &shard_st) != 0 ||
            !S_ISREG(shard_st.st_mode)) {
            if (err && errlen > 0)
                snprintf(err, errlen, "shard file not found: %s",
                         shard_paths[i]);
            mlx_map_string_to_array_free(merged);
            weights_free_shard_paths(shard_paths, shard_count);
            return -1;
        }

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

        const char *dup_key = NULL;
        if (merge_map(merged, shard_params, &dup_key) != 0) {
            if (err && errlen > 0) {
                if (dup_key)
                    snprintf(err, errlen, "duplicate tensor \"%s\" in shard %s",
                             dup_key, shard_paths[i]);
                else
                    snprintf(err, errlen, "failed merging shard %s",
                             shard_paths[i]);
            }
            mlxbridge_map_free(shard_params, shard_meta);
            mlx_map_string_to_array_free(merged);
            weights_free_shard_paths(shard_paths, shard_count);
            return -1;
        }
        mlxbridge_map_free(shard_params, shard_meta);
    }
    weights_free_shard_paths(shard_paths, shard_count);

    /* Index cross-check: read index keys if we loaded from index */
    char **index_keys = NULL;
    size_t n_index_keys = 0;
    if (from_index)
        read_index_keys(model_dir, &index_keys, &n_index_keys);

    if (weights_validate(merged, cfg, index_keys, n_index_keys,
                         err, errlen) != 0) {
        for (size_t i = 0; i < n_index_keys; i++) free(index_keys[i]);
        free(index_keys);
        mlx_map_string_to_array_free(merged);
        return -1;
    }
    for (size_t i = 0; i < n_index_keys; i++) free(index_keys[i]);
    free(index_keys);

    w->params = merged;
    map_stats(merged, &w->count, &w->total_bytes);

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
    memset(out, 0, sizeof(*out));

    char name[256];
    int wr = snprintf(name, sizeof(name), "%s.weight", base);
    if (wr < 0 || (size_t)wr >= sizeof(name)) return -1;

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

    if (has_scales != has_biases) {
        mlx_array_free(biases);
        mlx_array_free(scales);
        mlx_array_free(weight);
        return -1;
    }

    out->weight = weight;
    out->scales = scales;
    out->biases = biases;
    out->quantized = has_scales && has_biases;
    return 0;
}

void weights_triplet_free(weight_triplet_t *t) {
    if (!t) return;
    if (t->weight.ctx) mlx_array_free(t->weight);
    if (t->scales.ctx) mlx_array_free(t->scales);
    if (t->biases.ctx) mlx_array_free(t->biases);
    memset(t, 0, sizeof(*t));
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
    memset(w, 0, sizeof(*w));
}
