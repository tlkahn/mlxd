#include "http/response.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_200_exact_bytes(void) {
    size_t out_len = 0;
    char *r = http_build_response(200, "application/json", "{}", 2, true, &out_len);
    assert(r != NULL);
    const char *expected =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: 2\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: keep-alive\r\n"
        "\r\n"
        "{}";
    assert(out_len == strlen(expected));
    assert(memcmp(r, expected, out_len) == 0);
    free(r);
}

static void test_status_reasons(void) {
    assert(strcmp(http_status_reason(204), "No Content") == 0);
    assert(strcmp(http_status_reason(400), "Bad Request") == 0);
    assert(strcmp(http_status_reason(404), "Not Found") == 0);
    assert(strcmp(http_status_reason(405), "Method Not Allowed") == 0);
    assert(strcmp(http_status_reason(413), "Payload Too Large") == 0);
    assert(strcmp(http_status_reason(500), "Internal Server Error") == 0);
    assert(strcmp(http_status_reason(501), "Not Implemented") == 0);
    assert(strcmp(http_status_reason(503), "Service Unavailable") == 0);
}

static void test_connection_close(void) {
    size_t out_len = 0;
    char *r = http_build_response(200, "text/plain", "ok", 2, false, &out_len);
    assert(r != NULL);
    assert(strstr(r, "Connection: close\r\n") != NULL);
    assert(strstr(r, "keep-alive") == NULL);
    free(r);
}

static void test_sse_head(void) {
    size_t out_len = 0;
    char *r = http_build_sse_head(&out_len);
    assert(r != NULL);
    assert(strstr(r, "text/event-stream") != NULL);
    assert(strstr(r, "Cache-Control: no-cache") != NULL);
    assert(strstr(r, "Connection: close") != NULL);
    assert(strstr(r, "Content-Length") == NULL);
    assert(out_len == strlen(r));
    free(r);
}

static void test_unknown_status(void) {
    assert(strcmp(http_status_reason(499), "Unknown") == 0);
    size_t out_len = 0;
    char *r = http_build_response(499, "text/plain", "x", 1, true, &out_len);
    assert(r != NULL);
    assert(strstr(r, "499 Unknown") != NULL);
    free(r);
}

static void test_header_injection_rejected(void) {
    size_t out_len = 0;
    char *r = http_build_response(200, "text/plain\r\nX-Injected: yes", "ok", 2,
                                   true, &out_len);
    assert(r == NULL);
}

static void test_null_content_type_rejected(void) {
    size_t out_len = 0;
    char *r = http_build_response(200, NULL, "ok", 2, true, &out_len);
    assert(r == NULL);
}

static void test_null_body(void) {
    size_t out_len = 0;
    char *r = http_build_response(204, "text/plain", NULL, 0, true, &out_len);
    assert(r != NULL);
    assert(strstr(r, "Content-Length: 0\r\n") != NULL);
    free(r);
}

int main(void) {
    test_200_exact_bytes();
    test_status_reasons();
    test_connection_close();
    test_sse_head();
    test_unknown_status();
    test_header_injection_rejected();
    test_null_content_type_rejected();
    test_null_body();
    printf("test_http_response: all passed\n");
    return 0;
}
