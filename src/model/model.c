#include "model/model.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <yyjson/yyjson.h>

static char *dup_str(const char *s) {
    if (!s)
        return NULL;
    size_t n = strlen(s);
    char  *p = malloc(n + 1);
    if (p)
        memcpy(p, s, n + 1);
    return p;
}

static char *path_join(const char *dir, const char *name) {
    size_t dlen = strlen(dir);
    size_t nlen = strlen(name);
    char  *p    = malloc(dlen + 1 + nlen + 1);
    if (!p)
        return NULL;
    memcpy(p, dir, dlen);
    p[dlen] = '/';
    memcpy(p + dlen + 1, name, nlen);
    p[dlen + 1 + nlen] = '\0';
    return p;
}

static int get_dim_int(yyjson_val *root, const char *key, int *out, int def) {
    yyjson_val *v = yyjson_obj_get(root, key);
    if (!v) {
        *out = def;
        return 0;
    }
    if (!yyjson_is_int(v))
        return -1;
    int64_t n = yyjson_get_sint(v);
    if (n < 1 || n > INT_MAX)
        return -1;
    *out = (int)n;
    return 0;
}

int model_config_load(model_config_t *cfg, const char *model_dir) {
    if (!cfg)
        return -1;
    memset(cfg, 0, sizeof(*cfg));
    if (!model_dir)
        return -1;

    char *path = path_join(model_dir, "config.json");
    if (!path)
        return -1;

    yyjson_doc *doc = yyjson_read_file(path, 0, NULL, NULL);
    free(path);
    if (!doc)
        return -1;

    yyjson_val *root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        return -1;
    }

    /* model_type is required */
    yyjson_val *mt = yyjson_obj_get(root, "model_type");
    const char *mt_str = yyjson_get_str(mt);
    if (!mt_str) {
        yyjson_doc_free(doc);
        return -1;
    }

    cfg->model_type = dup_str(mt_str);
    if (!cfg->model_type) {
        yyjson_doc_free(doc);
        return -1;
    }

    /* architectures: first element of JSON array, optional */
    yyjson_val *archs = yyjson_obj_get(root, "architectures");
    if (yyjson_is_arr(archs)) {
        yyjson_val *first = yyjson_arr_get_first(archs);
        const char *a_str = yyjson_get_str(first);
        if (a_str) {
            cfg->architectures = dup_str(a_str);
            if (!cfg->architectures) {
                free(cfg->model_type);
                cfg->model_type = NULL;
                yyjson_doc_free(doc);
                return -1;
            }
        }
    }

    if (get_dim_int(root, "vocab_size", &cfg->vocab_size, 0) ||
        get_dim_int(root, "hidden_size", &cfg->hidden_size, 0) ||
        get_dim_int(root, "num_hidden_layers", &cfg->num_hidden_layers, 0) ||
        get_dim_int(root, "num_attention_heads", &cfg->num_attention_heads,
                    0) ||
        get_dim_int(root, "num_key_value_heads", &cfg->num_key_value_heads,
                    cfg->num_attention_heads) ||
        get_dim_int(root, "max_position_embeddings",
                    &cfg->max_position_embeddings, 0)) {
        model_config_free(cfg);
        yyjson_doc_free(doc);
        return -1;
    }

    yyjson_doc_free(doc);
    return 0;
}

void model_config_free(model_config_t *cfg) {
    if (!cfg)
        return;
    free(cfg->model_type);
    cfg->model_type = NULL;
    free(cfg->architectures);
    cfg->architectures = NULL;
}

