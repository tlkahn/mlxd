#include "engine/engine.h"

#include <stdlib.h>

/* TODO: implement during engine module migration */

stream_t *stream_create(int capacity) {
    (void)capacity;
    return NULL;
}

void stream_release(stream_t *s) { (void)s; }

bool stream_push(stream_t *s, chunk_t chunk) {
    (void)s;
    (void)chunk;
    return false;
}

bool stream_next(stream_t *s, chunk_t *out, int timeout_ms) {
    (void)s;
    (void)out;
    (void)timeout_ms;
    return false;
}

void stream_cancel(stream_t *s) { (void)s; }

int engine_init(engine_t *eng) {
    (void)eng;
    return -1;
}

void engine_destroy(engine_t *eng) { (void)eng; }

void engine_post(engine_t *eng, engine_cmd_t *cmd) {
    (void)eng;
    (void)cmd;
}
