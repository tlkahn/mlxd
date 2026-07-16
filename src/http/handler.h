#ifndef MLXD_HTTP_HANDLER_H
#define MLXD_HTTP_HANDLER_H

#include "http/router.h"
#include "http/serve_ctx.h"

void handler_register_all(http_router_t *router, serve_ctx_t *ctx);

#endif
