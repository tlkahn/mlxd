#include "cli/io.h"
#include "model/detok.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <yyjson/yyjson.h>

char *cli_run_messages_json(const char *prompt) {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) return NULL;

    yyjson_mut_val *arr = yyjson_mut_arr(doc);
    yyjson_mut_doc_set_root(doc, arr);

    yyjson_mut_val *msg = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_str(doc, msg, "role", "user");
    yyjson_mut_obj_add_strn(doc, msg, "content", prompt, strlen(prompt));
    yyjson_mut_arr_append(arr, msg);

    char *json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    return json;
}

int cli_run_consume(stream_t *s, const tokenizer_t *tok, FILE *out, bool flush_each,
                    const _Atomic int *cancel_flag, finish_reason_t *reason,
                    char *err, size_t errsz) {
    detok_t *d = detok_create(tok);

    for (;;) {
        if (cancel_flag && atomic_load(cancel_flag) > 0)
            stream_cancel(s);

        chunk_t c;
        if (!stream_next(s, &c, 100))
            continue;

        switch (c.tag) {
        case CHUNK_TOKEN: {
            if (d) {
                char *text = NULL;
                size_t text_len = 0;
                if (detok_feed(d, c.token.id, &text, &text_len) == 0 && text_len > 0) {
                    fwrite(text, 1, text_len, out);
                    if (flush_each) fflush(out);
                }
                free(text);
            } else {
                const char *text = tokenizer_decode_token(tok, c.token.id);
                if (text) {
                    fputs(text, out);
                    if (flush_each) fflush(out);
                }
            }
            break;
        }
        case CHUNK_DONE:
            if (d) {
                char *text = NULL;
                size_t text_len = 0;
                if (detok_flush(d, &text, &text_len) == 0 && text_len > 0) {
                    fwrite(text, 1, text_len, out);
                    if (flush_each) fflush(out);
                }
                free(text);
            }
            *reason = c.done;
            detok_free(d);
            return 0;

        case CHUNK_ERROR:
            if (err && errsz > 0)
                snprintf(err, errsz, "%s", c.error);
            free(c.error);
            detok_free(d);
            return -1;
        }
    }
}

void cli_human_size(uint64_t bytes, char *buf, size_t bufsz) {
    if (bytes >= (uint64_t)1024 * 1024 * 1024)
        snprintf(buf, bufsz, "%.1f GB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
    else if (bytes >= 1024 * 1024)
        snprintf(buf, bufsz, "%.1f MB", (double)bytes / (1024.0 * 1024.0));
    else if (bytes >= 1024)
        snprintf(buf, bufsz, "%.1f KB", (double)bytes / 1024.0);
    else
        snprintf(buf, bufsz, "%" PRIu64 " B", bytes);
}

char *cli_list_json(const registry_model_info_t *models, int count) {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) return NULL;

    yyjson_mut_val *arr = yyjson_mut_arr(doc);
    yyjson_mut_doc_set_root(doc, arr);

    for (int i = 0; i < count; i++) {
        yyjson_mut_val *obj = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_str(doc, obj, "id", models[i].id);
        yyjson_mut_obj_add_str(doc, obj, "path", models[i].path);
        yyjson_mut_obj_add_uint(doc, obj, "size_bytes", models[i].size_bytes);
        yyjson_mut_obj_add_sint(doc, obj, "mtime", models[i].mtime);
        yyjson_mut_arr_append(arr, obj);
    }

    char *json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    return json;
}
