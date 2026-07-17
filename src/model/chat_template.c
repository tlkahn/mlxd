#include "model/chat_template.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yyjson/yyjson.h"

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

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, (size_t)sz, f);
    buf[n] = '\0';
    fclose(f);
    return buf;
}

static char *try_jinja_file(const char *dir) {
    char *path = path_join(dir, "chat_template.jinja");
    if (!path) return NULL;
    char *content = read_file(path);
    free(path);
    return content;
}

static char *try_tokenizer_config(const char *dir) {
    char *path = path_join(dir, "tokenizer_config.json");
    if (!path) return NULL;
    char *json = read_file(path);
    free(path);
    if (!json) return NULL;

    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    free(json);
    if (!doc) return NULL;

    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *ct = yyjson_obj_get(root, "chat_template");
    char *result = NULL;

    if (yyjson_is_str(ct)) {
        result = strdup(yyjson_get_str(ct));
    } else if (yyjson_is_arr(ct)) {
        yyjson_val *val;
        yyjson_arr_iter iter;
        yyjson_arr_iter_init(ct, &iter);
        const char *first = NULL;
        while ((val = yyjson_arr_iter_next(&iter))) {
            const char *name = yyjson_get_str(yyjson_obj_get(val, "name"));
            const char *tmpl = yyjson_get_str(yyjson_obj_get(val, "template"));
            if (!tmpl) continue;
            if (!first) first = tmpl;
            if (name && strcmp(name, "default") == 0) {
                result = strdup(tmpl);
                break;
            }
        }
        if (!result && first)
            result = strdup(first);
    }

    yyjson_doc_free(doc);
    return result;
}

char *chat_template_load(const char *model_dir) {
    if (!model_dir) return NULL;

    char *tmpl = try_jinja_file(model_dir);
    if (tmpl) return tmpl;

    return try_tokenizer_config(model_dir);
}
