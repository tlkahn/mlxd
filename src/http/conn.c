#include "http/conn.h"

#include <llhttp.h>
#include <stdlib.h>
#include <string.h>

struct http_parser_ctx {
    llhttp_t          parser;
    llhttp_settings_t settings;

    char  *url_buf;
    size_t url_len;
    size_t url_cap;

    char  *body_buf;
    size_t body_len;
    size_t body_cap;
    size_t max_body;

    const char *method_ptr;

    bool complete;
    bool too_large;
    bool keep_alive;

    size_t consumed;
    http_parse_status_t last_status;
};

static int append(char **buf, size_t *len, size_t *cap, const char *data, size_t n) {
    size_t need = *len + n + 1;
    if (need > *cap) {
        size_t new_cap = *cap ? *cap : 64;
        while (new_cap < need) new_cap *= 2;
        char *tmp = realloc(*buf, new_cap);
        if (!tmp) return -1;
        *buf = tmp;
        *cap = new_cap;
    }
    memcpy(*buf + *len, data, n);
    *len += n;
    return 0;
}

#define MLXD_HTTP_MAX_URL 8192

static int on_url(llhttp_t *parser, const char *at, size_t length) {
    http_parser_ctx_t *ctx = parser->data;
    if (ctx->url_len + length > MLXD_HTTP_MAX_URL)
        return -1;
    return append(&ctx->url_buf, &ctx->url_len, &ctx->url_cap, at, length);
}

static int on_body(llhttp_t *parser, const char *at, size_t length) {
    http_parser_ctx_t *ctx = parser->data;
    if (ctx->body_len + length > ctx->max_body) {
        ctx->too_large = true;
        return -1;
    }
    return append(&ctx->body_buf, &ctx->body_len, &ctx->body_cap, at, length);
}

static int on_headers_complete(llhttp_t *parser) {
    http_parser_ctx_t *ctx = parser->data;
    uint64_t cl = parser->content_length;
    if (cl != (uint64_t)-1 && cl > ctx->max_body) {
        ctx->too_large = true;
        return -1;
    }
    return 0;
}

static int on_message_complete(llhttp_t *parser) {
    http_parser_ctx_t *ctx = parser->data;
    ctx->keep_alive = llhttp_should_keep_alive(parser);
    ctx->complete = true;
    llhttp_pause(parser);
    return HPE_PAUSED;
}

http_parser_ctx_t *http_parser_create(size_t max_body_bytes) {
    http_parser_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->max_body = max_body_bytes ? max_body_bytes : (1 << 20);

    llhttp_settings_init(&ctx->settings);
    ctx->settings.on_url = on_url;
    ctx->settings.on_headers_complete = on_headers_complete;
    ctx->settings.on_body = on_body;
    ctx->settings.on_message_complete = on_message_complete;

    llhttp_init(&ctx->parser, HTTP_REQUEST, &ctx->settings);
    ctx->parser.data = ctx;

    return ctx;
}

static void clear_buffers(http_parser_ctx_t *ctx) {
    ctx->url_len = 0;
    ctx->body_len = 0;
    ctx->complete = false;
    ctx->too_large = false;
    ctx->keep_alive = false;
    ctx->method_ptr = NULL;
    ctx->consumed = 0;
}

http_parse_status_t http_parser_feed(http_parser_ctx_t *p, const char *data,
                                     size_t len, http_parsed_request_t *out) {
    if (p->last_status == HTTP_PARSE_COMPLETE ||
        p->last_status == HTTP_PARSE_ERROR ||
        p->last_status == HTTP_PARSE_TOO_LARGE)
        return p->last_status;

    llhttp_errno_t err = llhttp_execute(&p->parser, data, len);

    if (p->too_large) {
        p->consumed = len;
        p->last_status = HTTP_PARSE_TOO_LARGE;
        return p->last_status;
    }

    if (err != HPE_OK && err != HPE_PAUSED && !p->complete) {
        p->consumed = len;
        p->last_status = HTTP_PARSE_ERROR;
        return p->last_status;
    }

    if (p->complete) {
        p->consumed = (err == HPE_PAUSED)
            ? (size_t)(llhttp_get_error_pos(&p->parser) - data)
            : len;

        if (p->url_buf) {
            p->url_buf[p->url_len] = '\0';
            char *q = p->url_buf;
            while (*q && *q != '?' && *q != '#') q++;
            *q = '\0';
        } else {
            p->url_buf = calloc(1, 1);
            if (!p->url_buf) { p->last_status = HTTP_PARSE_ERROR; return p->last_status; }
        }

        if (p->body_buf)
            p->body_buf[p->body_len] = '\0';

        p->method_ptr = llhttp_method_name(
            (llhttp_method_t)llhttp_get_method(&p->parser));

        if (out) {
            out->method = p->method_ptr;
            out->path = p->url_buf;
            out->body = p->body_len ? p->body_buf : NULL;
            out->body_len = p->body_len;
            out->keep_alive = p->keep_alive;
        }

        p->last_status = HTTP_PARSE_COMPLETE;
        return p->last_status;
    }

    p->consumed = len;
    return HTTP_PARSE_INCOMPLETE;
}

size_t http_parser_consumed(const http_parser_ctx_t *p) {
    return p ? p->consumed : 0;
}

void http_parser_reset(http_parser_ctx_t *p) {
    if (!p) return;
    clear_buffers(p);
    p->last_status = HTTP_PARSE_INCOMPLETE;
    llhttp_init(&p->parser, HTTP_REQUEST, &p->settings);
    p->parser.data = p;
}

void http_parser_free(http_parser_ctx_t *p) {
    if (!p) return;
    free(p->url_buf);
    free(p->body_buf);
    free(p);
}
