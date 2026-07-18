/* Fault-injection tests for gen_request.c lifecycle and OOM paths.
 * Compiles a private copy of gen_request.c with redirected fallible calls
 * and mock conn_io (server.c is NOT linked). */

#ifndef MLXD_FIXTURES_DIR
#error "MLXD_FIXTURES_DIR must be defined"
#endif

/* Minimal struct conn - must precede conn_io.h typedef */
struct conn { int dummy; };

/* Pre-include every header gen_request.c uses so include guards make
 * the #include lines inside gen_request.c no-ops. This lets the FI
 * macros affect only the C source text, not the declarations. */
#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <uv.h>
#include <yyjson/yyjson.h>

#include "core/openai.h"
#include "core/types.h"
#include "engine/engine.h"
#include "http/conn_io.h"
#include "http/gen.h"
#include "http/gen_request.h"
#include "http/response.h"
#include "http/serve_ctx.h"
#include "http/sse.h"
#include "model/detok.h"
#include "model/tokenizer.h"

/* === Mock conn_io state =================================================== */

typedef struct {
    char *buf;
    size_t len;
    bool close_after;
} mock_write_t;

static mock_write_t mock_writes[128];
static int mock_write_count;
static int mock_observer_set_count;
static int mock_observer_clear_count;
static int mock_close_count;
static size_t mock_write_queue_ret;
static void (*mock_on_gone)(void *);
static void *mock_on_gone_ctx;

static void mock_reset(void) {
    for (int i = 0; i < mock_write_count; i++)
        free(mock_writes[i].buf);
    memset(mock_writes, 0, sizeof(mock_writes));
    mock_write_count = 0;
    mock_observer_set_count = 0;
    mock_observer_clear_count = 0;
    mock_close_count = 0;
    mock_write_queue_ret = 0;
    mock_on_gone = NULL;
    mock_on_gone_ctx = NULL;
}

/* === Mock conn_io implementations ========================================= */

void http_conn_write(conn_t *c, char *wire, size_t len, bool close_after) {
    (void)c;
    if (mock_write_count < 128) {
        mock_writes[mock_write_count].buf = wire;
        mock_writes[mock_write_count].len = len;
        mock_writes[mock_write_count].close_after = close_after;
        mock_write_count++;
    } else {
        free(wire);
    }
}

size_t http_conn_write_queue_size(conn_t *c) {
    (void)c;
    return mock_write_queue_ret;
}

void http_conn_set_observer(conn_t *c, void (*on_gone)(void *), void *ctx) {
    (void)c;
    if (on_gone) {
        mock_observer_set_count++;
        mock_on_gone = on_gone;
        mock_on_gone_ctx = ctx;
    } else {
        mock_observer_clear_count++;
        mock_on_gone = NULL;
        mock_on_gone_ctx = NULL;
    }
}

void http_conn_close(conn_t *c) {
    (void)c;
    mock_close_count++;
}

/* === Fault-injection forward declarations ================================= */

static void *fi_calloc(size_t nmemb, size_t size);
static char *fi_gen_make_id(const char *prefix);
static char *fi_strdup(const char *s);
static char *fi_http_build_sse_head(size_t *out_len);
static char *fi_sse_done(void);
static char *fi_http_build_response(int status, const char *content_type,
                                    const char *body, size_t body_len,
                                    bool keep_alive, size_t *out_len);
static char *fi_gen_sse_chunk(const gen_sse_chunk_params_t *p);
static int   fi_detok_feed(detok_t *d, int32_t id, char **out, size_t *out_len);
static int   fi_detok_flush(detok_t *d, char **out, size_t *out_len);
static bool  fi_stream_next(stream_t *s, chunk_t *out, int timeout_ms);

/* === Macro redirects + private copy of gen_request.c ====================== */

#define calloc              fi_calloc
#define gen_make_id         fi_gen_make_id
#define strdup              fi_strdup
#define http_build_sse_head fi_http_build_sse_head
#define sse_done            fi_sse_done
#define http_build_response fi_http_build_response
#define gen_sse_chunk       fi_gen_sse_chunk
#define detok_feed          fi_detok_feed
#define detok_flush         fi_detok_flush
#define stream_next         fi_stream_next
#define gen_request_start   fi_gen_request_start
#include "http/gen_request.c"
#undef calloc
#undef gen_make_id
#undef strdup
#undef http_build_sse_head
#undef sse_done
#undef http_build_response
#undef gen_sse_chunk
#undef detok_feed
#undef detok_flush
#undef stream_next
#undef gen_request_start

/* === Fault-injection controls ============================================= */

static int  fi_calloc_skip;
static bool fi_fail_calloc;
static bool fi_fail_gen_make_id;
static bool fi_fail_strdup;
static bool fi_fail_sse_head;
static bool fi_fail_sse_done;
static bool fi_fail_http_build_response;
static int  fi_fail_gen_sse_chunk_skip;
static bool fi_fail_gen_sse_chunk;
static bool fi_fail_detok_feed;
static bool fi_fail_detok_flush;
static bool fi_force_context_length;

static void fi_reset(void) {
    fi_calloc_skip = 0;
    fi_fail_calloc = false;
    fi_fail_gen_make_id = false;
    fi_fail_strdup = false;
    fi_fail_sse_head = false;
    fi_fail_sse_done = false;
    fi_fail_http_build_response = false;
    fi_fail_gen_sse_chunk_skip = 0;
    fi_fail_gen_sse_chunk = false;
    fi_fail_detok_feed = false;
    fi_fail_detok_flush = false;
    fi_force_context_length = false;
}

/* === Fault-injection wrapper implementations ============================== */

static void *fi_calloc(size_t nmemb, size_t size) {
    if (fi_fail_calloc) {
        if (fi_calloc_skip > 0) { fi_calloc_skip--; }
        else return NULL;
    }
    void *p = malloc(nmemb * size);
    if (p) memset(p, 0, nmemb * size);
    return p;
}

static char *fi_gen_make_id(const char *prefix) {
    if (fi_fail_gen_make_id) return NULL;
    return gen_make_id(prefix);
}

static char *fi_strdup(const char *s) {
    if (fi_fail_strdup) return NULL;
    return strdup(s);
}

static char *fi_http_build_sse_head(size_t *out_len) {
    if (fi_fail_sse_head) return NULL;
    return http_build_sse_head(out_len);
}

static char *fi_sse_done(void) {
    if (fi_fail_sse_done) return NULL;
    return sse_done();
}

static char *fi_http_build_response(int status, const char *content_type,
                                    const char *body, size_t body_len,
                                    bool keep_alive, size_t *out_len) {
    if (fi_fail_http_build_response) return NULL;
    return http_build_response(status, content_type, body, body_len, keep_alive, out_len);
}

static char *fi_gen_sse_chunk(const gen_sse_chunk_params_t *p) {
    if (fi_fail_gen_sse_chunk) {
        if (fi_fail_gen_sse_chunk_skip > 0) { fi_fail_gen_sse_chunk_skip--; }
        else return NULL;
    }
    return gen_sse_chunk(p);
}

static int fi_detok_feed(detok_t *d, int32_t id, char **out, size_t *out_len) {
    if (fi_fail_detok_feed) {
        *out = NULL;
        *out_len = 0;
        return -1;
    }
    return detok_feed(d, id, out, out_len);
}

static int fi_detok_flush(detok_t *d, char **out, size_t *out_len) {
    if (fi_fail_detok_flush) {
        *out = NULL;
        *out_len = 0;
        return -1;
    }
    return detok_flush(d, out, out_len);
}

static bool fi_stream_next(stream_t *s, chunk_t *out, int timeout_ms) {
    bool ok = stream_next(s, out, timeout_ms);
    if (ok && fi_force_context_length && out->tag == CHUNK_ERROR) {
        out->error_kind = GEN_ERR_CONTEXT_LENGTH;
        free(out->error);
        out->error = strdup(
            "prompt token count 513 exceeds max_position_embeddings 512");
    }
    return ok;
}

/* === Test fixture ========================================================= */

#define TRIVIAL_TMPL "{{ messages[0].content }}"

typedef struct {
    engine_t     eng;
    tokenizer_t *tok;
    uv_loop_t    loop;
    conn_t       dummy_conn;
    serve_ctx_t  sctx;
} test_fixture_t;

static test_fixture_t fixture_up(void) {
    test_fixture_t f = {0};
    engine_init(&f.eng);

    engine_cmd_t *cmd = calloc(1, sizeof(*cmd));
    assert(cmd);
    cmd->tag = CMD_LOAD;
    cmd->load.model_path = strdup(MLXD_STUB_MODEL_PATH);
    engine_post(&f.eng, cmd);
    usleep(10000);

    f.tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(f.tok);

    uv_loop_init(&f.loop);

    f.sctx = (serve_ctx_t){
        .engine = &f.eng,
        .tokenizer = f.tok,
        .chat_template = TRIVIAL_TMPL,
        .model_id = "gpt2",
        .loop = &f.loop,
    };
    return f;
}

static void fixture_down(test_fixture_t *f) {
    engine_destroy(&f->eng);
    int rc = uv_loop_close(&f->loop);
    assert(rc == 0);
    tokenizer_free(f->tok);
}

static int start_chat_sse(test_fixture_t *f, const char *content) {
    gen_request_start_params_t p = {
        .ctx = &f->sctx,
        .conn = &f->dummy_conn,
        .chat = true,
        .stream = true,
        .model_id = "gpt2",
        .params = {.sampling = SAMPLING_PARAMS_DEFAULT},
        .messages_json = content ? content : "[{\"role\":\"user\",\"content\":\"hello\"}]",
    };
    const char *err = NULL;
    return fi_gen_request_start(&p, &err);
}

static int start_chat_nonstream(test_fixture_t *f) {
    gen_request_start_params_t p = {
        .ctx = &f->sctx,
        .conn = &f->dummy_conn,
        .chat = true,
        .stream = false,
        .model_id = "gpt2",
        .params = {.sampling = SAMPLING_PARAMS_DEFAULT},
        .messages_json = "[{\"role\":\"user\",\"content\":\"hello\"}]",
    };
    const char *err = NULL;
    return fi_gen_request_start(&p, &err);
}

/* === T1: cmd calloc OOM -> 500, clean uv_loop_close (#1) ================= */

static void test_cmd_calloc_oom(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    fi_fail_calloc = true;
    fi_calloc_skip = 1;

    gen_request_start_params_t p = {
        .ctx = &f.sctx,
        .conn = &f.dummy_conn,
        .chat = true,
        .stream = true,
        .model_id = "gpt2",
        .params = {.sampling = SAMPLING_PARAMS_DEFAULT},
        .messages_json = "[{\"role\":\"user\",\"content\":\"hi\"}]",
    };
    const char *err = NULL;
    int rc = fi_gen_request_start(&p, &err);
    assert(rc == 500);

    assert(uv_loop_close(&f.loop) == 0);
    engine_destroy(&f.eng);
    tokenizer_free(f.tok);
}

/* === T2: sse_done OOM -> observer cleared, conn closed (#2/#12) =========== */

static void test_sse_done_fail_clears_observer(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    fi_fail_sse_done = true;

    int rc = start_chat_sse(&f, NULL);
    assert(rc == 0);

    uv_run(&f.loop, UV_RUN_DEFAULT);

    assert(mock_observer_clear_count >= 1);
    assert(mock_close_count >= 1);

    fixture_down(&f);
}

/* === T2b: http_build_response OOM -> observer cleared, conn closed (#2) === */

static void test_build_response_fail_clears_observer(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    fi_fail_http_build_response = true;

    int rc = start_chat_nonstream(&f);
    assert(rc == 0);

    uv_run(&f.loop, UV_RUN_DEFAULT);

    assert(mock_observer_clear_count >= 1);
    assert(mock_close_count >= 1);

    fixture_down(&f);
}

/* === T3: gen_make_id NULL -> 500, no uv handles (#4) ====================== */

static void test_gen_make_id_null(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    fi_fail_gen_make_id = true;

    gen_request_start_params_t p = {
        .ctx = &f.sctx,
        .conn = &f.dummy_conn,
        .chat = true,
        .stream = true,
        .model_id = "gpt2",
        .params = {.sampling = SAMPLING_PARAMS_DEFAULT},
        .messages_json = "[{\"role\":\"user\",\"content\":\"hi\"}]",
    };
    const char *err = NULL;
    int rc = fi_gen_request_start(&p, &err);
    assert(rc == 500);
    assert(uv_loop_close(&f.loop) == 0);

    engine_destroy(&f.eng);
    tokenizer_free(f.tok);
}

/* === T3b: strdup NULL -> 500 (#4) ========================================= */

static void test_strdup_null(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    fi_fail_strdup = true;

    gen_request_start_params_t p = {
        .ctx = &f.sctx,
        .conn = &f.dummy_conn,
        .chat = true,
        .stream = true,
        .model_id = "gpt2",
        .params = {.sampling = SAMPLING_PARAMS_DEFAULT},
        .messages_json = "[{\"role\":\"user\",\"content\":\"hi\"}]",
    };
    const char *err = NULL;
    int rc = fi_gen_request_start(&p, &err);
    assert(rc == 500);
    assert(uv_loop_close(&f.loop) == 0);

    engine_destroy(&f.eng);
    tokenizer_free(f.tok);
}

/* === T4: http_build_sse_head OOM -> 500, not deferred (#12) =============== */

static void test_sse_head_fail(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    fi_fail_sse_head = true;

    gen_request_start_params_t p = {
        .ctx = &f.sctx,
        .conn = &f.dummy_conn,
        .chat = true,
        .stream = true,
        .model_id = "gpt2",
        .params = {.sampling = SAMPLING_PARAMS_DEFAULT},
        .messages_json = "[{\"role\":\"user\",\"content\":\"hi\"}]",
    };
    const char *err = NULL;
    int rc = fi_gen_request_start(&p, &err);
    assert(rc == 500);
    assert(uv_loop_close(&f.loop) == 0);

    engine_destroy(&f.eng);
    tokenizer_free(f.tok);
}

/* === T5: gen_sse_chunk fail on first (role) -> role retried (#8) =========== */

static void test_role_retry_on_sse_chunk_fail(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    fi_fail_gen_sse_chunk = true;
    fi_fail_gen_sse_chunk_skip = 0;

    int rc = start_chat_sse(&f, NULL);
    assert(rc == 0);

    fi_fail_gen_sse_chunk = false;

    uv_run(&f.loop, UV_RUN_DEFAULT);

    bool found_role = false;
    for (int i = 0; i < mock_write_count; i++) {
        if (mock_writes[i].buf && strstr(mock_writes[i].buf, "\"role\""))
            found_role = true;
    }
    assert(found_role);

    fixture_down(&f);
}

/* === T6: cmd params.stop cleared (bonus dangling stop) ==================== */

static void test_stop_cleared_in_cmd(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    char *stop_arr[] = {"foo", "bar"};
    gen_request_start_params_t p = {
        .ctx = &f.sctx,
        .conn = &f.dummy_conn,
        .chat = true,
        .stream = false,
        .model_id = "gpt2",
        .params = {
            .sampling = SAMPLING_PARAMS_DEFAULT,
            .stop = stop_arr,
            .stop_count = 2,
        },
        .messages_json = "[{\"role\":\"user\",\"content\":\"hi\"}]",
    };
    const char *err = NULL;
    int rc = fi_gen_request_start(&p, &err);
    assert(rc == 0);

    assert(p.params.stop == stop_arr);
    assert(p.params.stop_count == 2);

    uv_run(&f.loop, UV_RUN_DEFAULT);
    fixture_down(&f);
}

/* === T7: SSE error event is valid JSON with "error" object (#6) =========== */

static void test_sse_error_is_json(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    engine_cmd_t *unload = calloc(1, sizeof(*unload));
    assert(unload);
    unload->tag = CMD_UNLOAD;
    engine_post(&f.eng, unload);
    usleep(10000);

    gen_request_start_params_t p = {
        .ctx = &f.sctx,
        .conn = &f.dummy_conn,
        .chat = true,
        .stream = true,
        .model_id = "gpt2",
        .params = {.sampling = SAMPLING_PARAMS_DEFAULT},
        .messages_json = "[{\"role\":\"user\",\"content\":\"hi\"}]",
    };
    const char *err = NULL;
    int rc = fi_gen_request_start(&p, &err);
    assert(rc == 0);

    uv_run(&f.loop, UV_RUN_DEFAULT);

    bool found_error_event = false;
    for (int i = 0; i < mock_write_count; i++) {
        if (!mock_writes[i].buf) continue;
        char *data_start = strstr(mock_writes[i].buf, "data: {");
        if (!data_start) continue;
        data_start += 6;
        char *nl = strstr(data_start, "\n");
        if (!nl) continue;
        size_t jlen = (size_t)(nl - data_start);
        yyjson_doc *doc = yyjson_read(data_start, jlen, 0);
        if (!doc) continue;
        yyjson_val *root = yyjson_doc_get_root(doc);
        yyjson_val *errobj = yyjson_obj_get(root, "error");
        if (errobj) {
            assert(yyjson_obj_get(errobj, "message"));
            assert(yyjson_obj_get(errobj, "type"));
            found_error_event = true;
        }
        yyjson_doc_free(doc);
    }
    assert(found_error_event);

    fixture_down(&f);
}

/* === T8: detok_feed error -> stream cancelled, error event (#3) =========== */

static void test_detok_feed_error_cancels_stream(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    int rc = start_chat_sse(&f, NULL);
    assert(rc == 0);

    fi_fail_detok_feed = true;

    uv_run(&f.loop, UV_RUN_DEFAULT);

    bool has_terminal = false;
    for (int i = 0; i < mock_write_count; i++) {
        if (mock_writes[i].close_after)
            has_terminal = true;
    }
    assert(has_terminal || mock_close_count > 0);

    fixture_down(&f);
}

/* === T9: detok_flush error -> finish_error (#3) =========================== */

static void test_detok_flush_error(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    fi_fail_detok_flush = true;

    int rc = start_chat_sse(&f, NULL);
    assert(rc == 0);

    uv_run(&f.loop, UV_RUN_DEFAULT);

    assert(mock_observer_clear_count >= 1);
    bool has_terminal = false;
    for (int i = 0; i < mock_write_count; i++) {
        if (mock_writes[i].close_after)
            has_terminal = true;
    }
    assert(has_terminal || mock_close_count > 0);

    fixture_down(&f);
}

/* === T10: backpressure -> suppress_writes, conn closed (#7) =============== */

static void test_backpressure_suppresses_writes(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    mock_write_queue_ret = 2 * 1024 * 1024;

    int rc = start_chat_sse(&f, NULL);
    assert(rc == 0);

    uv_run(&f.loop, UV_RUN_DEFAULT);

    assert(mock_close_count >= 1);

    fixture_down(&f);
}

/* === T11: stream_sole_owner API (#13) ===================================== */

static void test_stream_sole_owner(void) {
    stream_t *s = stream_create(8);
    assert(s);
    assert(stream_sole_owner(s));
    stream_retain(s);
    assert(!stream_sole_owner(s));
    stream_release(s);
    assert(stream_sole_owner(s));
    stream_release(s);
}

/* === T12: gr calloc OOM -> 500, no uv handles ============================= */

static void test_gr_calloc_oom(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    fi_fail_calloc = true;
    fi_calloc_skip = 0;

    gen_request_start_params_t p = {
        .ctx = &f.sctx,
        .conn = &f.dummy_conn,
        .chat = true,
        .stream = true,
        .model_id = "gpt2",
        .params = {.sampling = SAMPLING_PARAMS_DEFAULT},
        .messages_json = "[{\"role\":\"user\",\"content\":\"hi\"}]",
    };
    const char *err = NULL;
    int rc = fi_gen_request_start(&p, &err);
    assert(rc == 500);
    assert(uv_loop_close(&f.loop) == 0);

    engine_destroy(&f.eng);
    tokenizer_free(f.tok);
}

/* === T13: normal SSE sequence completes cleanly =========================== */

static void test_normal_sse_completes(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    int rc = start_chat_sse(&f, NULL);
    assert(rc == 0);

    uv_run(&f.loop, UV_RUN_DEFAULT);

    bool found_done = false;
    for (int i = 0; i < mock_write_count; i++) {
        if (mock_writes[i].buf && strstr(mock_writes[i].buf, "[DONE]"))
            found_done = true;
    }
    assert(found_done);
    assert(mock_observer_clear_count >= 1);

    fixture_down(&f);
}

/* === B3.7: GEN_ERR_CONTEXT_LENGTH -> HTTP 400 + code ===================== */

static void test_context_length_nonstream_400(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    /* Unload so engine emits CHUNK_ERROR; FI rewrites kind to CONTEXT_LENGTH. */
    engine_cmd_t *unload = calloc(1, sizeof(*unload));
    assert(unload);
    unload->tag = CMD_UNLOAD;
    engine_post(&f.eng, unload);
    usleep(10000);

    fi_force_context_length = true;

    int rc = start_chat_nonstream(&f);
    assert(rc == 0);

    uv_run(&f.loop, UV_RUN_DEFAULT);

    bool found = false;
    for (int i = 0; i < mock_write_count; i++) {
        if (!mock_writes[i].buf) continue;
        if (strstr(mock_writes[i].buf, "HTTP/1.1 400") &&
            strstr(mock_writes[i].buf, "context_length_exceeded")) {
            found = true;
            /* also check type */
            assert(strstr(mock_writes[i].buf, "invalid_request_error"));
        }
    }
    assert(found);

    fixture_down(&f);
}

static void test_context_length_sse_code(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    engine_cmd_t *unload = calloc(1, sizeof(*unload));
    assert(unload);
    unload->tag = CMD_UNLOAD;
    engine_post(&f.eng, unload);
    usleep(10000);

    fi_force_context_length = true;

    int rc = start_chat_sse(&f, NULL);
    assert(rc == 0);

    uv_run(&f.loop, UV_RUN_DEFAULT);

    bool found = false;
    for (int i = 0; i < mock_write_count; i++) {
        if (!mock_writes[i].buf) continue;
        if (strstr(mock_writes[i].buf, "context_length_exceeded") &&
            strstr(mock_writes[i].buf, "invalid_request_error")) {
            found = true;
        }
    }
    assert(found);

    fixture_down(&f);
}

/* === T14: normal non-stream completes cleanly ============================= */

static void test_normal_nonstream_completes(void) {
    mock_reset();
    fi_reset();
    test_fixture_t f = fixture_up();

    int rc = start_chat_nonstream(&f);
    assert(rc == 0);

    uv_run(&f.loop, UV_RUN_DEFAULT);

    bool found_response = false;
    for (int i = 0; i < mock_write_count; i++) {
        if (mock_writes[i].buf && strstr(mock_writes[i].buf, "chat.completion"))
            found_response = true;
    }
    assert(found_response);
    assert(mock_observer_clear_count >= 1);

    fixture_down(&f);
}

/* === main ================================================================= */

int main(void) {
    test_cmd_calloc_oom();
    test_sse_done_fail_clears_observer();
    test_build_response_fail_clears_observer();
    test_gen_make_id_null();
    test_strdup_null();
    test_sse_head_fail();
    test_role_retry_on_sse_chunk_fail();
    test_stop_cleared_in_cmd();
    test_sse_error_is_json();
    test_detok_feed_error_cancels_stream();
    test_detok_flush_error();
    test_backpressure_suppresses_writes();
    test_stream_sole_owner();
    test_gr_calloc_oom();
    test_normal_sse_completes();
    test_normal_nonstream_completes();
    test_context_length_nonstream_400();
    test_context_length_sse_code();
    printf("test_http_gen_request: all passed\n");
    return 0;
}
