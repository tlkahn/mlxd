#ifndef MLXD_HTTP_CONN_IO_H
#define MLXD_HTTP_CONN_IO_H

#include <stdbool.h>
#include <stddef.h>

typedef struct conn conn_t;

/* Takes ownership of `wire` on all paths (caller must not free).
 * close_after=true closes the connection after the write completes. */
void http_conn_write(conn_t *c, char *wire, size_t len, bool close_after);
size_t http_conn_write_queue_size(conn_t *c);
void http_conn_set_observer(conn_t *c, void (*on_gone)(void *), void *ctx);
void http_conn_close(conn_t *c);

#endif
