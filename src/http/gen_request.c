#include "http/gen_request.h"
#include "http/conn_io.h"
#include "http/gen.h"
#include "http/response.h"
#include "http/sse.h"
#include "core/openai.h"
#include "engine/engine.h"
#include "model/detok.h"
#include "model/tokenizer.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <uv.h>

#define BACKPRESSURE_HWM (1024 * 1024)

typedef enum {
    GR_RUNNING,
    GR_CLOSING,
} gr_state_t;

typedef struct gen_request {
    stream_t    *stream;
    uv_async_t   async;
    uv_timer_t   timer;
    bool         timer_inited;
    detok_t     *detok;
    gr_state_t   state;
    bool         conn_gone;
    conn_t      *conn;

    bool         chat;
    bool         sse;
    bool         include_usage;
    bool         role_sent;

    char        *id;
    char        *model_id;
    int64_t      created;

    int          prompt_tokens;
    int          completion_tokens;

    char        *accum;
    size_t       accum_len;
    size_t       accum_cap;

    int          pending_closes;
} gen_request_t;

/* --- Forward declarations ------------------------------------------------- */

static void gen_on_async(uv_async_t *handle);
static void on_conn_gone(void *ctx);
static void try_teardown(gen_request_t *gr);
static void begin_close(gen_request_t *gr);
static void on_close_cb(uv_handle_t *handle);
static void on_timer(uv_timer_t *handle);
static void notify_cb(void *ctx);

/* --- Accumulator helpers -------------------------------------------------- */

static int accum_append(gen_request_t *gr, const char *s, size_t len) {
    if (len == 0) return 0;
    size_t need = gr->accum_len + len;
    if (need > gr->accum_cap) {
        size_t new_cap = gr->accum_cap ? gr->accum_cap : 64;
        while (new_cap < need) new_cap *= 2;
        char *tmp = realloc(gr->accum, new_cap + 1);
        if (!tmp) return -1;
        gr->accum = tmp;
        gr->accum_cap = new_cap;
    }
    memcpy(gr->accum + gr->accum_len, s, len);
    gr->accum_len += len;
    gr->accum[gr->accum_len] = '\0';
    return 0;
}

/* --- Start ---------------------------------------------------------------- */

int gen_request_start(const gen_request_start_params_t *p, const char **err) {
    int32_t *ids = NULL;
    int n_ids;

    if (!p->ctx->tokenizer) {
        *err = "model not loaded";
        return 503;
    }

    if (p->chat && !p->ctx->chat_template) {
        *err = "chat template not configured";
        return 400;
    }

    if (p->chat) {
        n_ids = gen_build_chat_prompt(p->ctx->tokenizer, p->ctx->chat_template,
                                      p->messages_json, p->tools_json, &ids, err);
    } else {
        n_ids = gen_build_completion_prompt(p->ctx->tokenizer, p->prompt, &ids, err);
    }
    if (n_ids < 0) return 500;

    stream_t *s = stream_create(64);
    if (!s) {
        free(ids);
        *err = "out of memory";
        return 500;
    }
    stream_retain(s);

    detok_t *dt = detok_create(p->ctx->tokenizer);
    if (!dt) {
        stream_release(s);
        stream_release(s);
        free(ids);
        *err = "detokenizer init failed";
        return 500;
    }

    gen_request_t *gr = calloc(1, sizeof(*gr));
    if (!gr) {
        detok_free(dt);
        stream_release(s);
        stream_release(s);
        free(ids);
        *err = "out of memory";
        return 500;
    }

    gr->stream = s;
    gr->detok = dt;
    gr->state = GR_RUNNING;
    gr->conn = p->conn;
    gr->chat = p->chat;
    gr->sse = p->stream;
    gr->include_usage = p->include_usage;
    gr->prompt_tokens = n_ids;
    gr->created = time(NULL);
    gr->id = gen_make_id(p->chat ? "chatcmpl-" : "cmpl-");
    gr->model_id = strdup(p->model_id ? p->model_id : "unknown");

    uv_async_init(p->ctx->loop, &gr->async, gen_on_async);
    gr->async.data = gr;

    gen_params_t params = p->params;
    if (params.max_tokens <= 0) {
        params.max_tokens = p->chat ? INT_MAX : 16;
    }

    stream_set_notify(s, notify_cb, gr);

    engine_cmd_t *cmd = calloc(1, sizeof(*cmd));
    if (!cmd) {
        detok_free(dt);
        stream_release(s);
        stream_release(s);
        free(gr->id);
        free(gr->model_id);
        free(gr);
        free(ids);
        *err = "out of memory";
        return 500;
    }
    cmd->tag = CMD_GENERATE;
    cmd->generate.params = params;
    cmd->generate.token_ids = ids;
    cmd->generate.token_count = n_ids;
    cmd->generate.stream = s;

    http_conn_set_observer(p->conn, on_conn_gone, gr);
    engine_post(p->ctx->engine, cmd);

    if (gr->sse) {
        size_t hlen;
        char *head = http_build_sse_head(&hlen);
        if (head)
            http_conn_write(p->conn, head, hlen, false);
    }

    return 0;
}

/* --- Notify callback (engine thread -> loop thread) ----------------------- */

static void notify_cb(void *ctx) {
    gen_request_t *gr = ctx;
    uv_async_send(&gr->async);
}

/* --- Async drain (loop thread) -------------------------------------------- */

static void emit_sse_chunk(gen_request_t *gr, const char *text, size_t len) {
    if (gr->conn_gone) return;
    if (gr->chat) {
        if (!gr->role_sent) {
            gr->role_sent = true;
            char *sse = gen_sse_chunk(&(gen_sse_chunk_params_t){
                .id = gr->id, .model = gr->model_id, .created = gr->created,
                .role_first = true,
                .delta_text = (len > 0) ? text : NULL,
            });
            if (sse) {
                http_conn_write(gr->conn, sse, strlen(sse), false);
                return;
            }
        }
        if (len > 0) {
            char *sse = gen_sse_chunk(&(gen_sse_chunk_params_t){
                .id = gr->id, .model = gr->model_id, .created = gr->created,
                .delta_text = text,
            });
            if (sse)
                http_conn_write(gr->conn, sse, strlen(sse), false);
        }
    } else {
        if (len > 0) {
            char *sse = gen_sse_completion_chunk(&(gen_sse_completion_chunk_params_t){
                .id = gr->id, .model = gr->model_id, .created = gr->created,
                .delta_text = text,
            });
            if (sse)
                http_conn_write(gr->conn, sse, strlen(sse), false);
        }
    }
}

static void finish_response(gen_request_t *gr, finish_reason_t reason) {
    char *flush_out = NULL;
    size_t flush_len = 0;
    detok_flush(gr->detok, &flush_out, &flush_len);

    if (!gr->conn_gone) {
        if (gr->sse) {
            if (flush_out && flush_len > 0)
                emit_sse_chunk(gr, flush_out, flush_len);

            if (gr->chat) {
                char *final_sse = gen_sse_chunk(&(gen_sse_chunk_params_t){
                    .id = gr->id, .model = gr->model_id, .created = gr->created,
                    .final = true, .reason = reason,
                });
                if (final_sse)
                    http_conn_write(gr->conn, final_sse, strlen(final_sse), false);

                if (gr->include_usage) {
                    usage_t u = {
                        .prompt_tokens = gr->prompt_tokens,
                        .completion_tokens = gr->completion_tokens,
                        .total_tokens = gr->prompt_tokens + gr->completion_tokens,
                    };
                    char *usage_sse = gen_sse_chunk(&(gen_sse_chunk_params_t){
                        .id = gr->id, .model = gr->model_id, .created = gr->created,
                        .include_usage = true, .usage = &u,
                    });
                    if (usage_sse)
                        http_conn_write(gr->conn, usage_sse, strlen(usage_sse), false);
                }
            } else {
                char *final_sse = gen_sse_completion_chunk(&(gen_sse_completion_chunk_params_t){
                    .id = gr->id, .model = gr->model_id, .created = gr->created,
                    .final = true, .reason = reason,
                });
                if (final_sse)
                    http_conn_write(gr->conn, final_sse, strlen(final_sse), false);
            }

            char *done = sse_done();
            if (done) {
                http_conn_set_observer(gr->conn, NULL, NULL);
                http_conn_write(gr->conn, done, strlen(done), true);
            }
        } else {
            if (flush_out && flush_len > 0)
                accum_append(gr, flush_out, flush_len);

            usage_t u = {
                .prompt_tokens = gr->prompt_tokens,
                .completion_tokens = gr->completion_tokens,
                .total_tokens = gr->prompt_tokens + gr->completion_tokens,
            };
            char *json;
            if (gr->chat) {
                json = gen_build_chat_response(gr->id, gr->model_id, gr->created,
                                               gr->accum, reason, &u);
            } else {
                json = gen_build_completion_response(gr->id, gr->model_id, gr->created,
                                                     gr->accum, reason, &u);
            }
            if (json) {
                size_t wire_len;
                char *wire = http_build_response(200, "application/json", json,
                                                 strlen(json), false, &wire_len);
                free(json);
                if (wire) {
                    http_conn_set_observer(gr->conn, NULL, NULL);
                    http_conn_write(gr->conn, wire, wire_len, true);
                }
            }
        }
    }
    free(flush_out);
    gr->conn = NULL;
    try_teardown(gr);
}

static void finish_error(gen_request_t *gr, char *msg) {
    if (!gr->conn_gone) {
        if (gr->sse) {
            char *sse_msg = sse_format(msg, strlen(msg));
            if (sse_msg)
                http_conn_write(gr->conn, sse_msg, strlen(sse_msg), false);
            char *done = sse_done();
            if (done) {
                http_conn_set_observer(gr->conn, NULL, NULL);
                http_conn_write(gr->conn, done, strlen(done), true);
            }
        } else {
            yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
            if (doc) {
                yyjson_mut_val *root = error_envelope_serialize(
                    msg, "server_error", NULL, doc);
                yyjson_mut_doc_set_root(doc, root);
                char *json = yyjson_mut_write(doc, 0, NULL);
                yyjson_mut_doc_free(doc);
                if (json) {
                    size_t wire_len;
                    char *wire = http_build_response(500, "application/json", json,
                                                     strlen(json), false, &wire_len);
                    free(json);
                    if (wire) {
                        http_conn_set_observer(gr->conn, NULL, NULL);
                        http_conn_write(gr->conn, wire, wire_len, true);
                    }
                }
            }
        }
    }
    free(msg);
    gr->conn = NULL;
    try_teardown(gr);
}

static void gen_on_async(uv_async_t *handle) {
    gen_request_t *gr = handle->data;
    if (gr->state == GR_CLOSING) return;

    chunk_t c;
    while (stream_next(gr->stream, &c, 0)) {
        switch (c.tag) {
        case CHUNK_TOKEN: {
            gr->completion_tokens++;
            char *piece = NULL;
            size_t piece_len = 0;
            detok_feed(gr->detok, c.token.id, &piece, &piece_len);
            if (gr->sse) {
                if (piece && piece_len > 0)
                    emit_sse_chunk(gr, piece, piece_len);
                if (!gr->conn_gone &&
                    http_conn_write_queue_size(gr->conn) > BACKPRESSURE_HWM) {
                    stream_cancel(gr->stream);
                }
            } else {
                if (piece && piece_len > 0)
                    accum_append(gr, piece, piece_len);
            }
            free(piece);
            break;
        }
        case CHUNK_ERROR:
            finish_error(gr, c.error);
            return;
        case CHUNK_DONE:
            finish_response(gr, c.done);
            return;
        }
    }
}

/* --- Connection gone observer (loop thread) ------------------------------- */

static void on_conn_gone(void *ctx) {
    gen_request_t *gr = ctx;
    gr->conn_gone = true;
    gr->conn = NULL;
    stream_cancel(gr->stream);
    try_teardown(gr);
}

/* --- Teardown state machine ----------------------------------------------- */

static void try_teardown(gen_request_t *gr) {
    if (gr->state == GR_CLOSING) return;
    if (atomic_load(&gr->stream->refcount) == 1) {
        begin_close(gr);
        return;
    }
    if (!gr->timer_inited) {
        gr->timer_inited = true;
        uv_timer_init(gr->async.loop, &gr->timer);
        gr->timer.data = gr;
        uv_timer_start(&gr->timer, on_timer, 1, 1);
    }
}

static void on_timer(uv_timer_t *handle) {
    gen_request_t *gr = handle->data;
    if (gr->state == GR_CLOSING) return;
    if (atomic_load(&gr->stream->refcount) == 1) {
        uv_timer_stop(&gr->timer);
        begin_close(gr);
    }
}

static void begin_close(gen_request_t *gr) {
    gr->state = GR_CLOSING;
    stream_release(gr->stream);
    gr->stream = NULL;
    detok_free(gr->detok);
    gr->detok = NULL;
    free(gr->accum);
    gr->accum = NULL;
    free(gr->id);
    gr->id = NULL;
    free(gr->model_id);
    gr->model_id = NULL;

    gr->pending_closes = 1;
    if (gr->timer_inited) gr->pending_closes++;

    uv_close((uv_handle_t *)&gr->async, on_close_cb);
    if (gr->timer_inited)
        uv_close((uv_handle_t *)&gr->timer, on_close_cb);
}

static void on_close_cb(uv_handle_t *handle) {
    gen_request_t *gr = handle->data;
    if (--gr->pending_closes == 0)
        free(gr);
}
