#ifndef MLXD_HTTP_SERVER_H
#define MLXD_HTTP_SERVER_H

#include "engine/engine.h"

typedef struct tokenizer tokenizer_t;
typedef struct http_server http_server_t;

typedef struct {
    const char  *host;
    int          port;
    engine_t    *engine;
    tokenizer_t *tokenizer;
    const char  *chat_template;
    const char  *model_id;
    size_t       max_body_bytes;    /* 0 = default (1 MiB) */
    uint64_t     drain_deadline_ms; /* 0 = default (5000) */
} http_server_config_t;

http_server_t *http_server_create(const http_server_config_t *config);
int  http_server_start(http_server_t *srv);
void http_server_stop(http_server_t *srv);
void http_server_destroy(http_server_t *srv);
int  http_server_port(const http_server_t *srv);

#endif
