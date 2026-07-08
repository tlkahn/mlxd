#include "http/server.h"

#include <stdlib.h>

/* TODO: implement during http module migration */

struct http_server {
    http_server_config_t config;
    /* uv_loop_t, uv_tcp_t, etc. */
};

http_server_t *http_server_create(const http_server_config_t *config) {
    (void)config;
    return NULL;
}

int http_server_start(http_server_t *srv) {
    (void)srv;
    return -1;
}

void http_server_stop(http_server_t *srv) { (void)srv; }

void http_server_destroy(http_server_t *srv) { free(srv); }
