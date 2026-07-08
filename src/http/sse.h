#ifndef MLXD_HTTP_SSE_H
#define MLXD_HTTP_SSE_H

#include <stddef.h>

/* Format a Server-Sent Event message.
 * Returns a heap-allocated string "data: <data>\n\n".
 * Caller must free. */
char *sse_format(const char *data, size_t len);

/* Format the SSE terminator "[DONE]". */
char *sse_done(void);

#endif
