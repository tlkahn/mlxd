#include "http/sse.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_sse_format_basic(void) {
    char *s = sse_format("hi", 2);
    assert(s != NULL);
    assert(strcmp(s, "data: hi\n\n") == 0);
    free(s);
}

static void test_sse_format_empty(void) {
    char *s = sse_format("", 0);
    assert(s != NULL);
    assert(strcmp(s, "data: \n\n") == 0);
    free(s);
}

static void test_sse_format_embedded_newline(void) {
    char *s = sse_format("a\nb", 3);
    assert(s != NULL);
    assert(strcmp(s, "data: a\ndata: b\n\n") == 0);
    free(s);
}

static void test_sse_format_multiline(void) {
    char *s = sse_format("x\ny\nz", 5);
    assert(s != NULL);
    assert(strcmp(s, "data: x\ndata: y\ndata: z\n\n") == 0);
    free(s);
}

static void test_sse_format_trailing_newline(void) {
    char *s = sse_format("a\n", 2);
    assert(s != NULL);
    assert(strcmp(s, "data: a\ndata: \n\n") == 0);
    free(s);
}

static void test_sse_done(void) {
    char *s = sse_done();
    assert(s != NULL);
    assert(strcmp(s, "data: [DONE]\n\n") == 0);
    free(s);
}

int main(void) {
    test_sse_format_basic();
    test_sse_format_empty();
    test_sse_format_embedded_newline();
    test_sse_format_multiline();
    test_sse_format_trailing_newline();
    test_sse_done();
    printf("test_sse: all passed\n");
    return 0;
}
