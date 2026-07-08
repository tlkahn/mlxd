#ifndef MLXD_HTTP_HANDLER_H
#define MLXD_HTTP_HANDLER_H

#include "http/router.h"

/* Register all API handlers on the router. */
void handler_register_all(http_router_t *router, void *engine_ctx);

#endif
