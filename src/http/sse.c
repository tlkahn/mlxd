#include "http/sse.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* TODO(epic-c): embedded newlines produce invalid SSE framing; split into
 * multiple "data:" lines per the SSE spec. */
char *sse_format(const char *data, size_t len) {
    /* "data: " + data + "\n\n" + NUL */
    size_t total = 6 + len + 2 + 1;
    char *buf = malloc(total);
    if (!buf)
        return NULL;
    snprintf(buf, total, "data: %.*s\n\n", (int)len, data);
    return buf;
}

char *sse_done(void) { return strdup("data: [DONE]\n\n"); }
