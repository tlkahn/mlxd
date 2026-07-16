#ifndef MLXD_HTTP_CONN_H
#define MLXD_HTTP_CONN_H

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    HTTP_PARSE_INCOMPLETE,
    HTTP_PARSE_COMPLETE,
    HTTP_PARSE_ERROR,
    HTTP_PARSE_TOO_LARGE,
} http_parse_status_t;

/* Parsed request fields. method/path/body point into ctx-owned buffers,
 * valid until the next http_parser_reset or http_parser_free. Caller
 * never frees these pointers. path is the request-target with query and
 * fragment stripped. body is NULL when the request has no body. */
typedef struct {
    const char *method;
    char       *path;
    char       *body;
    size_t      body_len;
    bool        keep_alive;
} http_parsed_request_t;

typedef struct http_parser_ctx http_parser_ctx_t;

/* Create a parser context. max_body_bytes == 0 defaults to 1 MiB. */
http_parser_ctx_t *http_parser_create(size_t max_body_bytes);

/* Feed raw bytes. Returns COMPLETE when a full request is ready (fills out).
 * Once COMPLETE, ERROR, or TOO_LARGE is returned, all subsequent feeds return
 * the same status until http_parser_reset is called (sticky).
 * On COMPLETE the parser pauses; pipelined bytes beyond the completed message
 * are not consumed. Call http_parser_consumed() to learn how many bytes were
 * processed, then reset and re-feed the remainder. */
http_parse_status_t http_parser_feed(http_parser_ctx_t *p, const char *data,
                                     size_t len, http_parsed_request_t *out);

/* Bytes consumed by the last http_parser_feed call. After COMPLETE this equals
 * the length of the finished message; the caller can re-feed
 * data + consumed .. data + len after a reset. */
size_t http_parser_consumed(const http_parser_ctx_t *p);

/* Reset for the next keep-alive request on the same connection. */
void http_parser_reset(http_parser_ctx_t *p);

void http_parser_free(http_parser_ctx_t *p);

#endif
