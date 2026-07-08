#ifndef MLXD_HTTP_ROUTER_H
#define MLXD_HTTP_ROUTER_H

#include <stdbool.h>
#include <stddef.h>

/* Route dispatch. Maps URL path + method to a handler. */

typedef struct {
    const char *method;
    const char *path;
    void       *ctx;
    /* request body, headers, etc. filled by server */
    const char *body;
    size_t      body_len;
} http_request_t;

typedef struct {
    int         status;
    const char *content_type;
    char       *body;
    size_t      body_len;
    /* set to true for SSE streaming */
    _Bool       streaming;
} http_response_t;

typedef void (*http_handler_fn)(const http_request_t *req, http_response_t *resp, void *ctx);

typedef struct http_router http_router_t;

http_router_t *http_router_create(void);
void http_router_add(http_router_t *r, const char *method, const char *path, http_handler_fn fn,
                     void *ctx);
http_handler_fn http_router_match(const http_router_t *r, const char *method, const char *path,
                                  void **ctx);
void http_router_destroy(http_router_t *r);

#endif
