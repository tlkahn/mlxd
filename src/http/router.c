#include "http/router.h"

#include <stdlib.h>
#include <string.h>

struct route {
    const char     *method;
    const char     *path;
    http_handler_fn fn;
    void           *ctx;
};

struct http_router {
    struct route *routes;
    size_t        n;
    size_t        cap;
};

http_router_t *http_router_create(void) {
    http_router_t *r = calloc(1, sizeof(*r));
    return r;
}

void http_router_add(http_router_t *r, const char *method, const char *path, http_handler_fn fn,
                     void *ctx) {
    if (!r) return;
    if (r->n == r->cap) {
        size_t new_cap = r->cap ? r->cap * 2 : 4;
        struct route *tmp = realloc(r->routes, new_cap * sizeof(*tmp));
        if (!tmp) return;
        r->routes = tmp;
        r->cap = new_cap;
    }
    r->routes[r->n++] = (struct route){.method = method, .path = path, .fn = fn, .ctx = ctx};
}

http_handler_fn http_router_match(const http_router_t *r, const char *method, const char *path,
                                  void **ctx) {
    if (!r) return NULL;
    for (size_t i = 0; i < r->n; i++) {
        if (strcmp(r->routes[i].method, method) == 0 && strcmp(r->routes[i].path, path) == 0) {
            if (ctx) *ctx = r->routes[i].ctx;
            return r->routes[i].fn;
        }
    }
    return NULL;
}

bool http_router_path_exists(const http_router_t *r, const char *path) {
    if (!r) return false;
    for (size_t i = 0; i < r->n; i++) {
        if (strcmp(r->routes[i].path, path) == 0)
            return true;
    }
    return false;
}

void http_router_destroy(http_router_t *r) {
    if (!r) return;
    free(r->routes);
    free(r);
}
