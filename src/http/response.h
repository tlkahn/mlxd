#ifndef MLXD_HTTP_RESPONSE_H
#define MLXD_HTTP_RESPONSE_H

#include <stdbool.h>
#include <stddef.h>

/* Build a complete HTTP response (status line + headers + body).
 * Includes Access-Control-Allow-Origin: * for CORS.
 * body may be NULL (body_len must be 0). Returns heap-allocated buffer;
 * *out_len is set to the total byte count. Caller frees. */
char *http_build_response(int status, const char *content_type, const char *body,
                          size_t body_len, bool keep_alive, size_t *out_len);

/* Build an SSE response head (no Content-Length, Connection: close). */
char *http_build_sse_head(size_t *out_len);

const char *http_status_reason(int status);

#endif
