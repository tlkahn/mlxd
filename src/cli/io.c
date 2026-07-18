#include "cli/io.h"
#include "model/detok.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <yyjson/yyjson.h>

char *cli_resolve_run_prompt(const char *positional, FILE *stdin_stream) {
    if (positional)
        return strdup(positional);
    if (!stdin_stream)
        return NULL;
    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    for (;;) {
        size_t n = fread(buf + len, 1, cap - len, stdin_stream);
        len += n;
        if (n == 0) break;
        if (len == cap) {
            cap *= 2;
            char *tmp = realloc(buf, cap);
            if (!tmp) { free(buf); return NULL; }
            buf = tmp;
        }
    }
    if (len == 0) { free(buf); return NULL; }
    buf[len] = '\0';
    return buf;
}

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

/* Grace period (seconds) after cancel before we give up waiting for the
   engine's terminal chunk.  stream_cancel does not inject DONE; the engine
   injects DONE/FINISH_CANCELLED on its next stream_push failure.  This
   deadline is defense against that contract breaking. */
#define CANCEL_GRACE_SEC 2

int cli_run_consume(stream_t *s, const tokenizer_t *tok, FILE *out, bool flush_each,
                    const _Atomic int *cancel_flag, finish_reason_t *reason,
                    char *err, size_t errsz, bool token_ids) {
    detok_t *d = token_ids ? NULL : detok_create(tok);
    bool cancel_sent = false;
    struct timespec grace_deadline = {0, 0};

    for (;;) {
        if (cancel_flag && atomic_load(cancel_flag) > 0 && !cancel_sent) {
            stream_cancel(s);
            cancel_sent = true;
            clock_gettime(CLOCK_MONOTONIC, &grace_deadline);
            grace_deadline.tv_sec += CANCEL_GRACE_SEC;
        }

        chunk_t c;
        if (!stream_next(s, &c, 100)) {
            if (cancel_sent) {
                struct timespec now;
                clock_gettime(CLOCK_MONOTONIC, &now);
                if (now.tv_sec > grace_deadline.tv_sec ||
                    (now.tv_sec == grace_deadline.tv_sec &&
                     now.tv_nsec >= grace_deadline.tv_nsec)) {
                    detok_free(d);
                    *reason = FINISH_CANCELLED;
                    return 0;
                }
                usleep(10000);
            }
            continue;
        }

        switch (c.tag) {
        case CHUNK_TOKEN: {
            if (token_ids) {
                fprintf(out, "%" PRId32 "\n", c.token.id);
                if (flush_each) fflush(out);
                break;
            }
            if (d) {
                char *text = NULL;
                size_t text_len = 0;
                if (detok_feed(d, c.token.id, &text, &text_len) != 0) {
                    free(text);
                    detok_free(d);
                    d = NULL;
                } else if (text_len > 0) {
                    fwrite(text, 1, text_len, out);
                    if (flush_each) fflush(out);
                    free(text);
                    break;
                } else {
                    free(text);
                    break;
                }
            }
            const char *raw = tokenizer_decode_token(tok, c.token.id);
            if (raw) {
                fputs(raw, out);
                if (flush_each) fflush(out);
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
