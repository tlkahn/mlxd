#ifndef MLXD_HTTP_SERVE_CTX_H
#define MLXD_HTTP_SERVE_CTX_H

#include "engine/engine.h"

#include <uv.h>

typedef struct tokenizer tokenizer_t;

typedef struct {
    engine_t    *engine;
    tokenizer_t *tokenizer;
    const char  *chat_template;
    const char  *model_id;
    uv_loop_t   *loop;
} serve_ctx_t;

#endif
