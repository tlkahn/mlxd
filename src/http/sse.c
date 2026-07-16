#include "http/sse.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *sse_format(const char *data, size_t len) {
    if (len == 0) {
        char *buf = malloc(8 + 1);
        if (!buf) return NULL;
        memcpy(buf, "data: \n\n", 8);
        buf[8] = '\0';
        return buf;
    }

    size_t lines = 1;
    for (size_t i = 0; i < len; i++)
        if (data[i] == '\n') lines++;

    size_t total = lines * 6 + len + 1 + 1;
    char *buf = malloc(total);
    if (!buf) return NULL;

    size_t off = 0;
    const char *p = data;
    const char *end = data + len;
    for (;;) {
        const char *nl = (p < end) ? memchr(p, '\n', (size_t)(end - p)) : NULL;
        size_t seg = nl ? (size_t)(nl - p) : (size_t)(end - p);
        memcpy(buf + off, "data: ", 6);
        off += 6;
        if (seg > 0) {
            memcpy(buf + off, p, seg);
            off += seg;
        }
        buf[off++] = '\n';
        if (!nl) break;
        p = nl + 1;
    }
    buf[off++] = '\n';
    buf[off] = '\0';
    return buf;
}

char *sse_done(void) { return strdup("data: [DONE]\n\n"); }
