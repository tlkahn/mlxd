#include "http/server.h"
#include "http/conn.h"
#include "http/conn_io.h"
#include "http/handler.h"
#include "http/response.h"
#include "http/router.h"
#include "http/serve_ctx.h"
#include "core/log.h"
#include "core/openai.h"

#include <stdlib.h>
#include <string.h>
#include <uv.h>
#include <yyjson/yyjson.h>

/* --- Types --------------------------------------------------------------- */

struct http_server {
    uv_loop_t      loop;
    uv_tcp_t       listener;
    uv_async_t     stop_async;
    http_router_t *router;
    serve_ctx_t    serve_ctx;
    int            port;
    size_t         max_body_bytes;
};

typedef struct conn {
    uv_tcp_t           handle;
    http_server_t     *server;
    http_parser_ctx_t *parser;
    char               rbuf[65536];
    bool               closing;
    bool               gen_owned;
    void             (*on_gone)(void *);
    void              *on_gone_ctx;
} conn_t;

typedef struct {
    uv_write_t req;
    char      *buf;
    conn_t    *conn;
    bool       close_after;
} write_req_t;

/* --- Forward declarations ------------------------------------------------ */

static void on_new_connection(uv_stream_t *server, int status);
static void on_read(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf);
static void on_write(uv_write_t *req, int status);
static void on_conn_close(uv_handle_t *handle);
static void on_stop_async(uv_async_t *handle);
static void close_conn(conn_t *c);
static void dispatch(conn_t *c, http_parsed_request_t *pr);

/* --- Error wire builder -------------------------------------------------- */

static char *build_error_wire(int status, const char *type, const char *code,
                              const char *msg, bool keep_alive, size_t *out_len) {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) return NULL;
    yyjson_mut_val *root = error_envelope_serialize(msg, type, code, doc);
    yyjson_mut_doc_set_root(doc, root);
    char *json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    if (!json) return NULL;
    size_t json_len = strlen(json);
    char *wire = http_build_response(status, "application/json", json, json_len,
                                     keep_alive, out_len);
    free(json);
    return wire;
}

/* --- Write helpers ------------------------------------------------------- */

static void queue_write(conn_t *c, char *wire, size_t wire_len, bool close_after) {
    if (c->closing) { free(wire); return; }
    write_req_t *wr = malloc(sizeof(*wr));
    if (!wr) {
        free(wire);
        close_conn(c);
        return;
    }
    wr->buf = wire;
    wr->conn = c;
    wr->close_after = close_after;

    uv_buf_t uvbuf = uv_buf_init(wire, (unsigned int)wire_len);
    int rc = uv_write(&wr->req, (uv_stream_t *)&c->handle, &uvbuf, 1, on_write);
    if (rc < 0) {
        free(wire);
        free(wr);
        close_conn(c);
    }
}

static void write_error(conn_t *c, int status, const char *type, const char *code,
                        const char *msg, bool keep_alive) {
    size_t wire_len = 0;
    char *wire = build_error_wire(status, type, code, msg, keep_alive, &wire_len);
    if (!wire) {
        close_conn(c);
        return;
    }
    queue_write(c, wire, wire_len, !keep_alive);
}

/* --- Connection lifecycle ------------------------------------------------ */

/* Returns the connection's static rbuf. All bytes MUST be consumed
 * synchronously within on_read before returning to the loop. */
static void alloc_cb(uv_handle_t *handle, size_t suggested, uv_buf_t *buf) {
    (void)suggested;
    conn_t *c = (conn_t *)handle;
    buf->base = c->rbuf;
    buf->len = sizeof(c->rbuf);
}

static void close_conn(conn_t *c) {
    if (c->closing) return;
    c->closing = true;
    if (c->on_gone) {
        void (*cb)(void *) = c->on_gone;
        void *ctx = c->on_gone_ctx;
        c->on_gone = NULL;
        c->on_gone_ctx = NULL;
        cb(ctx);
    }
    uv_close((uv_handle_t *)&c->handle, on_conn_close);
}

static void on_conn_close(uv_handle_t *handle) {
    conn_t *c = (conn_t *)handle;
    http_parser_free(c->parser);
    free(c);
}

static void on_new_connection(uv_stream_t *server, int status) {
    if (status < 0) return;

    http_server_t *srv = (http_server_t *)server->data;
    conn_t *c = calloc(1, sizeof(*c));
    if (!c) return;

    c->server = srv;
    uv_tcp_init(&srv->loop, &c->handle);

    if (uv_accept(server, (uv_stream_t *)&c->handle) != 0) {
        uv_close((uv_handle_t *)&c->handle, on_conn_close);
        return;
    }

    c->parser = http_parser_create(srv->max_body_bytes);
    if (!c->parser) {
        uv_close((uv_handle_t *)&c->handle, on_conn_close);
        return;
    }

    uv_read_start((uv_stream_t *)&c->handle, alloc_cb, on_read);
}

/* --- Read + dispatch ----------------------------------------------------- */

static void on_read(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf) {
    (void)buf;
    conn_t *c = (conn_t *)stream;

    if (nread < 0) {
        close_conn(c);
        return;
    }
    if (nread == 0) return;

    if (c->gen_owned) return;

    size_t off = 0;
    while (off < (size_t)nread && !c->closing) {
        http_parsed_request_t pr = {0};
        http_parse_status_t st = http_parser_feed(c->parser, c->rbuf + off,
                                                  (size_t)nread - off, &pr);
        switch (st) {
        case HTTP_PARSE_INCOMPLETE:
            return;
        case HTTP_PARSE_ERROR:
            write_error(c, 400, "invalid_request_error", NULL,
                        "Malformed HTTP request", false);
            return;
        case HTTP_PARSE_TOO_LARGE:
            write_error(c, 413, "payload_too_large", "payload_too_large",
                        "Request body too large", false);
            return;
        case HTTP_PARSE_COMPLETE:
            off += http_parser_consumed(c->parser);
            dispatch(c, &pr);
            if (!pr.keep_alive || c->closing || c->gen_owned) return;
            http_parser_reset(c->parser);
            break;
        }
    }
}

static void dispatch(conn_t *c, http_parsed_request_t *pr) {
    http_server_t *srv = c->server;

    if (strcmp(pr->method, "OPTIONS") == 0) {
        size_t pf_len = 0;
        char *pf = http_build_preflight(pr->keep_alive, &pf_len);
        if (!pf) {
            close_conn(c);
            return;
        }
        queue_write(c, pf, pf_len, !pr->keep_alive);
        return;
    }

    void *handler_ctx = NULL;
    http_handler_fn handler = http_router_match(srv->router, pr->method,
                                                pr->path, &handler_ctx);

    if (handler) {
        http_request_t req = {
            .method   = pr->method,
            .path     = pr->path,
            .ctx      = c,
            .body     = pr->body,
            .body_len = pr->body_len,
        };
        http_response_t resp = {0};
        handler(&req, &resp, handler_ctx);

        if (resp.deferred) {
            c->gen_owned = true;
            return;
        }

        size_t wire_len = 0;
        char *wire = http_build_response(resp.status, resp.content_type,
                                         resp.body, resp.body_len,
                                         pr->keep_alive, &wire_len);
        free(resp.body);
        if (!wire) {
            close_conn(c);
            return;
        }
        queue_write(c, wire, wire_len, !pr->keep_alive);
    } else if (http_router_path_exists(srv->router, pr->path)) {
        write_error(c, 405, "method_not_allowed", "method_not_allowed",
                    "Method not allowed", pr->keep_alive);
        if (!pr->keep_alive) return;
    } else {
        write_error(c, 404, "not_found_error", "not_found",
                    "Not found", pr->keep_alive);
        if (!pr->keep_alive) return;
    }
}

/* --- Write callback ------------------------------------------------------ */

static void on_write(uv_write_t *req, int status) {
    write_req_t *wr = (write_req_t *)req;
    free(wr->buf);
    bool should_close = wr->close_after || status < 0;
    conn_t *c = wr->conn;
    free(wr);
    if (should_close)
        close_conn(c);
}

/* --- Stop (walk + close) ------------------------------------------------- */

static void walk_close_cb(uv_handle_t *handle, void *arg) {
    http_server_t *srv = (http_server_t *)arg;
    if (uv_is_closing(handle)) return;
    if (handle->type == UV_TCP && handle != (uv_handle_t *)&srv->listener)
        close_conn((conn_t *)handle);
    else if (handle->type == UV_ASYNC || handle->type == UV_TIMER)
        /* gen_request is the sole UV_ASYNC/UV_TIMER owner on this loop;
         * handles self-close via begin_close after the stream drains. */
        return;
    else
        uv_close(handle, NULL);
}

static void on_stop_async(uv_async_t *handle) {
    http_server_t *srv = (http_server_t *)handle->data;
    uv_close((uv_handle_t *)&srv->listener, NULL);
    uv_close((uv_handle_t *)&srv->stop_async, NULL);
    uv_walk(&srv->loop, walk_close_cb, srv);
}

/* --- Public API ---------------------------------------------------------- */

http_server_t *http_server_create(const http_server_config_t *config) {
    if (!config) return NULL;

    http_server_t *srv = calloc(1, sizeof(*srv));
    if (!srv) return NULL;

    if (uv_loop_init(&srv->loop) != 0) goto fail_free;

    if (uv_tcp_init(&srv->loop, &srv->listener) != 0) goto fail_loop;
    srv->listener.data = srv;

    const char *host = config->host ? config->host : "127.0.0.1";
    struct sockaddr_in addr;
    if (uv_ip4_addr(host, config->port, &addr) != 0) goto fail_close_listener;

    if (uv_tcp_bind(&srv->listener, (const struct sockaddr *)&addr, 0) != 0)
        goto fail_close_listener;

    if (uv_listen((uv_stream_t *)&srv->listener, 128, on_new_connection) != 0)
        goto fail_close_listener;

    struct sockaddr_in bound;
    int namelen = sizeof(bound);
    if (uv_tcp_getsockname(&srv->listener, (struct sockaddr *)&bound, &namelen) != 0)
        goto fail_close_listener;
    srv->port = ntohs(bound.sin_port);

    if (uv_async_init(&srv->loop, &srv->stop_async, on_stop_async) != 0)
        goto fail_close_listener;
    srv->stop_async.data = srv;

    srv->max_body_bytes = config->max_body_bytes;

    srv->serve_ctx = (serve_ctx_t){
        .engine        = config->engine,
        .tokenizer     = config->tokenizer,
        .chat_template = config->chat_template,
        .model_id      = config->model_id,
        .loop          = &srv->loop,
    };

    srv->router = http_router_create();
    if (!srv->router) goto fail_close_async;

    handler_register_all(srv->router, &srv->serve_ctx);

    return srv;

fail_close_async:
    uv_close((uv_handle_t *)&srv->stop_async, NULL);
fail_close_listener:
    uv_close((uv_handle_t *)&srv->listener, NULL);
    uv_run(&srv->loop, UV_RUN_NOWAIT);
fail_loop:
    uv_loop_close(&srv->loop);
fail_free:
    free(srv);
    return NULL;
}

int http_server_start(http_server_t *srv) {
    if (!srv) return -1;
    return uv_run(&srv->loop, UV_RUN_DEFAULT);
}

void http_server_stop(http_server_t *srv) {
    if (!srv) return;
    uv_async_send(&srv->stop_async);
}

void http_server_destroy(http_server_t *srv) {
    if (!srv) return;
    int rc = uv_loop_close(&srv->loop);
    if (rc == UV_EBUSY) {
        log_warn("http_server_destroy: uv_loop_close returned EBUSY, retrying (a handle may have leaked)");
        uv_run(&srv->loop, UV_RUN_NOWAIT);
        uv_loop_close(&srv->loop);
    }
    http_router_destroy(srv->router);
    free(srv);
}

int http_server_port(const http_server_t *srv) {
    return srv ? srv->port : -1;
}

/* --- conn_io API (used by gen_request) ------------------------------------ */

void http_conn_write(conn_t *c, char *wire, size_t len, bool close_after) {
    queue_write(c, wire, len, close_after);
}

size_t http_conn_write_queue_size(conn_t *c) {
    return uv_stream_get_write_queue_size((uv_stream_t *)&c->handle);
}

void http_conn_set_observer(conn_t *c, void (*on_gone)(void *), void *ctx) {
    c->on_gone = on_gone;
    c->on_gone_ctx = ctx;
}

void http_conn_close(conn_t *c) {
    close_conn(c);
}
