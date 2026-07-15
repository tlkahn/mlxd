#include "model/model.h"

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

    /* numeric fields - absent stays 0 */
    yyjson_val *v;
    v = yyjson_obj_get(root, "vocab_size");
    if (yyjson_is_int(v))
        cfg->vocab_size = (int)yyjson_get_int(v);

    v = yyjson_obj_get(root, "hidden_size");
    if (yyjson_is_int(v))
        cfg->hidden_size = (int)yyjson_get_int(v);

    v = yyjson_obj_get(root, "num_hidden_layers");
    if (yyjson_is_int(v))
        cfg->num_hidden_layers = (int)yyjson_get_int(v);

    v = yyjson_obj_get(root, "num_attention_heads");
    if (yyjson_is_int(v))
        cfg->num_attention_heads = (int)yyjson_get_int(v);

    v = yyjson_obj_get(root, "num_key_value_heads");
    if (yyjson_is_int(v))
        cfg->num_key_value_heads = (int)yyjson_get_int(v);
    else
        cfg->num_key_value_heads = cfg->num_attention_heads;

    v = yyjson_obj_get(root, "max_position_embeddings");
    if (yyjson_is_int(v))
        cfg->max_position_embeddings = (int)yyjson_get_int(v);

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

char **model_discover(int *count) {
    *count = 0;
    return NULL;
}
