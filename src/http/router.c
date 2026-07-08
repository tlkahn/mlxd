#include "http/router.h"

#include <stdlib.h>

/* TODO: implement during http module migration */

struct http_router {
    int dummy;
};

http_router_t *http_router_create(void) { return NULL; }

void http_router_add(http_router_t *r, const char *method, const char *path, http_handler_fn fn,
                     void *ctx) {
    (void)r;
    (void)method;
    (void)path;
    (void)fn;
    (void)ctx;
}

http_handler_fn http_router_match(const http_router_t *r, const char *method, const char *path,
                                  void **ctx) {
    (void)r;
    (void)method;
    (void)path;
    (void)ctx;
    return NULL;
}

void http_router_destroy(http_router_t *r) { free(r); }
