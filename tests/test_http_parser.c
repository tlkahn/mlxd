#include "http/conn.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void test_simple_get(void) {
    http_parser_ctx_t *p = http_parser_create(0);
    assert(p != NULL);
    const char *req = "GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n";
    http_parsed_request_t out = {0};
    http_parse_status_t s = http_parser_feed(p, req, strlen(req), &out);
    assert(s == HTTP_PARSE_COMPLETE);
    assert(strcmp(out.method, "GET") == 0);
    assert(strcmp(out.path, "/v1/models") == 0);
    assert(out.keep_alive == true);
    http_parser_free(p);
}

static void test_post_with_body(void) {
    http_parser_ctx_t *p = http_parser_create(0);
    const char *req =
        "POST /v1/chat HTTP/1.1\r\n"
        "Host: x\r\n"
        "Content-Length: 13\r\n"
        "\r\n"
        "{\"key\":\"val\"}";
    http_parsed_request_t out = {0};
    http_parse_status_t s = http_parser_feed(p, req, strlen(req), &out);
    assert(s == HTTP_PARSE_COMPLETE);
    assert(strcmp(out.method, "POST") == 0);
    assert(out.body_len == 13);
    assert(memcmp(out.body, "{\"key\":\"val\"}", 13) == 0);
    http_parser_free(p);
}

static void test_partial_delivery(void) {
    http_parser_ctx_t *p = http_parser_create(0);
    const char *full = "GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n";
    size_t half = strlen(full) / 2;
    http_parsed_request_t out = {0};
    http_parse_status_t s = http_parser_feed(p, full, half, &out);
    assert(s == HTTP_PARSE_INCOMPLETE);
    s = http_parser_feed(p, full + half, strlen(full) - half, &out);
    assert(s == HTTP_PARSE_COMPLETE);
    assert(strcmp(out.method, "GET") == 0);
    assert(strcmp(out.path, "/v1/models") == 0);
    http_parser_free(p);
}

static void test_body_split(void) {
    http_parser_ctx_t *p = http_parser_create(0);
    const char *head =
        "POST /x HTTP/1.1\r\n"
        "Host: x\r\n"
        "Content-Length: 10\r\n"
        "\r\n"
        "abcde";
    const char *tail = "fghij";
    http_parsed_request_t out = {0};
    http_parse_status_t s = http_parser_feed(p, head, strlen(head), &out);
    assert(s == HTTP_PARSE_INCOMPLETE);
    s = http_parser_feed(p, tail, strlen(tail), &out);
    assert(s == HTTP_PARSE_COMPLETE);
    assert(out.body_len == 10);
    assert(memcmp(out.body, "abcdefghij", 10) == 0);
    http_parser_free(p);
}

static void test_malformed(void) {
    http_parser_ctx_t *p = http_parser_create(0);
    const char *bad = "GARBAGE DATA\r\n\r\n";
    http_parsed_request_t out = {0};
    http_parse_status_t s = http_parser_feed(p, bad, strlen(bad), &out);
    assert(s == HTTP_PARSE_ERROR);
    s = http_parser_feed(p, "GET / HTTP/1.1\r\n\r\n", 18, &out);
    assert(s == HTTP_PARSE_ERROR);
    http_parser_free(p);
}

static void test_too_large(void) {
    http_parser_ctx_t *p = http_parser_create(16);
    const char *req =
        "POST /x HTTP/1.1\r\n"
        "Host: x\r\n"
        "Content-Length: 32\r\n"
        "\r\n"
        "01234567890123456789012345678901";
    http_parsed_request_t out = {0};
    http_parse_status_t s = http_parser_feed(p, req, strlen(req), &out);
    assert(s == HTTP_PARSE_TOO_LARGE);
    s = http_parser_feed(p, "GET / HTTP/1.1\r\n\r\n", 18, &out);
    assert(s == HTTP_PARSE_TOO_LARGE);
    http_parser_free(p);
}

static void test_connection_close(void) {
    http_parser_ctx_t *p = http_parser_create(0);
    const char *req =
        "GET / HTTP/1.1\r\n"
        "Host: x\r\n"
        "Connection: close\r\n"
        "\r\n";
    http_parsed_request_t out = {0};
    http_parse_status_t s = http_parser_feed(p, req, strlen(req), &out);
    assert(s == HTTP_PARSE_COMPLETE);
    assert(out.keep_alive == false);
    http_parser_free(p);
}

static void test_reset_reuse(void) {
    http_parser_ctx_t *p = http_parser_create(0);
    const char *req1 = "GET /a HTTP/1.1\r\nHost: x\r\n\r\n";
    http_parsed_request_t out = {0};
    http_parse_status_t s = http_parser_feed(p, req1, strlen(req1), &out);
    assert(s == HTTP_PARSE_COMPLETE);
    assert(strcmp(out.path, "/a") == 0);

    http_parser_reset(p);

    const char *req2 = "POST /b HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\nhi";
    s = http_parser_feed(p, req2, strlen(req2), &out);
    assert(s == HTTP_PARSE_COMPLETE);
    assert(strcmp(out.path, "/b") == 0);
    assert(strcmp(out.method, "POST") == 0);
    assert(out.body_len == 2);
    assert(memcmp(out.body, "hi", 2) == 0);

    http_parser_free(p);
}

int main(void) {
    test_simple_get();
    test_post_with_body();
    test_partial_delivery();
    test_body_split();
    test_malformed();
    test_too_large();
    test_connection_close();
    test_reset_reuse();
    printf("test_http_parser: all passed\n");
    return 0;
}
