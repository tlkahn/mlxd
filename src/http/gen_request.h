#ifndef MLXD_HTTP_GEN_REQUEST_H
#define MLXD_HTTP_GEN_REQUEST_H

#include "core/types.h"
#include "http/serve_ctx.h"

#include <stdbool.h>

typedef struct conn conn_t;

typedef struct {
    serve_ctx_t *ctx;
    conn_t      *conn;
    bool         chat;
    bool         stream;
    bool         include_usage;
    const char  *model_id;
    gen_params_t params;
    /* chat mode */
    const char  *messages_json;
    const char  *tools_json;
    const char  *extra_json; /* borrowed; consumed synchronously during prompt build */
    /* completion mode */
    const char  *prompt;
} gen_request_start_params_t;

/* Start an async generation request. Returns 0 on success (response will be
 * written asynchronously), or an HTTP status code on error (*err set to a
 * static literal for respond_json_error). */
int gen_request_start(const gen_request_start_params_t *p, const char **err);

/* Initiate graceful drain: cancel the stream so the engine injects
 * DONE/FINISH_CANCELLED, then finish_response flushes data: [DONE]
 * and closes the connection. Does NOT tear down the gen_request
 * or detach the conn - the normal finish path handles that.
 * ctx is a gen_request_t* (void* because server.c has no visibility). */
void gen_request_drain(void *ctx);

#endif
