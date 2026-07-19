#include "model/chat_template.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yyjson/yyjson.h>

static char *slurp_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = malloc(sz > 0 ? (size_t)sz : 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    if (out_len) *out_len = (size_t)sz;
    return buf;
}

char *model_chat_template_read(const char *model_dir) {
    if (!model_dir) return NULL;

    size_t pathlen = strlen(model_dir) + sizeof("/tokenizer_config.json");
    char *path = malloc(pathlen);
    if (!path) return NULL;
    snprintf(path, pathlen, "%s/tokenizer_config.json", model_dir);

    size_t len = 0;
    char *buf = slurp_file(path, &len);
    free(path);
    if (buf) {
        yyjson_doc *doc = yyjson_read(buf, len, 0);
        free(buf);
        if (doc) {
            char *result = NULL;
            yyjson_val *root = yyjson_doc_get_root(doc);
            if (yyjson_is_obj(root)) {
                yyjson_val *tmpl = yyjson_obj_get(root, "chat_template");
                if (yyjson_is_str(tmpl))
                    result = strdup(yyjson_get_str(tmpl));
            }
            yyjson_doc_free(doc);
            if (result) return result;
        }
    }

    /* Fallback: chat_template.jinja (raw Jinja source) */
    pathlen = strlen(model_dir) + sizeof("/chat_template.jinja");
    path = malloc(pathlen);
    if (!path) return NULL;
    snprintf(path, pathlen, "%s/chat_template.jinja", model_dir);
    buf = slurp_file(path, &len);
    free(path);
    if (buf) {
        char *result = realloc(buf, len + 1);
        if (!result) { free(buf); return NULL; }
        result[len] = '\0';
        return result;
    }

    /* Fallback: chat_template.json containing a JSON string */
    pathlen = strlen(model_dir) + sizeof("/chat_template.json");
    path = malloc(pathlen);
    if (!path) return NULL;
    snprintf(path, pathlen, "%s/chat_template.json", model_dir);
    buf = slurp_file(path, &len);
    free(path);
    if (!buf) return NULL;
    yyjson_doc *doc = yyjson_read(buf, len, 0);
    free(buf);
    if (!doc) return NULL;
    char *result = NULL;
    yyjson_val *root = yyjson_doc_get_root(doc);
    if (yyjson_is_str(root))
        result = strdup(yyjson_get_str(root));
    yyjson_doc_free(doc);
    return result;
}
