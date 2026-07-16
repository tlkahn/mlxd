#include "http/router.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static void h1(const http_request_t *req, http_response_t *resp, void *ctx) {
    (void)req; (void)resp; (void)ctx;
}

static void h2(const http_request_t *req, http_response_t *resp, void *ctx) {
    (void)req; (void)resp; (void)ctx;
}

static void test_match_hit(void) {
    http_router_t *r = http_router_create();
    assert(r != NULL);
    int c1 = 42;
    http_router_add(r, "GET", "/v1/models", h1, &c1);
    void *ctx = NULL;
    http_handler_fn fn = http_router_match(r, "GET", "/v1/models", &ctx);
    assert(fn == h1);
    assert(ctx == &c1);
    http_router_destroy(r);
}

static void test_unknown_path(void) {
    http_router_t *r = http_router_create();
    assert(r != NULL);
    http_router_add(r, "GET", "/v1/models", h1, NULL);
    void *ctx = NULL;
    http_handler_fn fn = http_router_match(r, "GET", "/nope", &ctx);
    assert(fn == NULL);
    assert(!http_router_path_exists(r, "/nope"));
    http_router_destroy(r);
}

static void test_known_path_wrong_method(void) {
    http_router_t *r = http_router_create();
    assert(r != NULL);
    http_router_add(r, "GET", "/v1/models", h1, NULL);
    void *ctx = NULL;
    http_handler_fn fn = http_router_match(r, "POST", "/v1/models", &ctx);
    assert(fn == NULL);
    assert(http_router_path_exists(r, "/v1/models"));
    http_router_destroy(r);
}

static void test_multiple_routes(void) {
    http_router_t *r = http_router_create();
    assert(r != NULL);
    int c1 = 1, c2 = 2;
    http_router_add(r, "GET", "/v1/models", h1, &c1);
    http_router_add(r, "POST", "/v1/chat/completions", h2, &c2);
    void *ctx = NULL;
    http_handler_fn fn = http_router_match(r, "GET", "/v1/models", &ctx);
    assert(fn == h1);
    assert(ctx == &c1);
    ctx = NULL;
    fn = http_router_match(r, "POST", "/v1/chat/completions", &ctx);
    assert(fn == h2);
    assert(ctx == &c2);
    http_router_destroy(r);
}

static void test_add_returns_zero_on_success(void) {
    http_router_t *r = http_router_create();
    assert(r != NULL);
    int rc = http_router_add(r, "GET", "/a", h1, NULL);
    assert(rc == 0);
    http_router_destroy(r);
}

static void test_add_returns_negative_on_null(void) {
    int rc = http_router_add(NULL, "GET", "/a", h1, NULL);
    assert(rc == -1);
}

static void test_destroy_empty(void) {
    http_router_t *r = http_router_create();
    assert(r != NULL);
    http_router_destroy(r);
}

int main(void) {
    test_match_hit();
    test_unknown_path();
    test_known_path_wrong_method();
    test_multiple_routes();
    test_add_returns_zero_on_success();
    test_add_returns_negative_on_null();
    test_destroy_empty();
    printf("test_http_router: all passed\n");
    return 0;
}
