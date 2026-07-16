#ifndef MLXD_ENGINE_H
#define MLXD_ENGINE_H

#include "core/types.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>

/* --- Stream (per-request token channel) ----------------------------------- */

typedef struct {
    chunk_t        *buf;
    int             cap;
    int             len;
    int             read_pos;
    pthread_mutex_t mtx;
    pthread_cond_t  cond;
    atomic_bool     cancelled;
    atomic_int      refcount;
    void          (*on_push)(void *);
    void           *on_push_ctx;
} stream_t;

stream_t *stream_create(int capacity);
void      stream_retain(stream_t *s);
void      stream_release(stream_t *s);
bool      stream_push(stream_t *s, chunk_t chunk);
bool      stream_next(stream_t *s, chunk_t *out, int timeout_ms);
/* Mark the stream as cancelled. Does NOT itself inject a DONE marker.
 * Any in-flight generate ends with a DONE/FINISH_CANCELLED terminal
 * injected by the engine (handle_generate abort, cmd_cancel, post-
 * shutdown rejection). Buffered chunks drain normally via stream_next;
 * after the last chunk it returns false. */
void      stream_cancel(stream_t *s);
void      stream_set_notify(stream_t *s, void (*cb)(void *), void *ctx);
bool      stream_sole_owner(const stream_t *s);

/* --- Engine commands ------------------------------------------------------ */

typedef enum {
    CMD_GENERATE,
    CMD_EMBED,
    CMD_LOAD,
    CMD_UNLOAD,
    CMD_RECLAIM,
    CMD_STOP,
} engine_cmd_tag_t;

typedef struct engine_cmd {
    engine_cmd_tag_t tag;
    union {
        struct {
            gen_params_t  params;
            int32_t      *token_ids;
            int           token_count;
            stream_t     *stream;
        } generate;
        struct {
            char *model_path;
        } load;
        struct {
            stream_t *stream; /* ownership: engine calls stream_release */
        } reclaim;
    };
    struct engine_cmd *next;
} engine_cmd_t;

/* --- Engine --------------------------------------------------------------- */

typedef struct {
    pthread_t       thread;
    pthread_mutex_t mailbox_mtx;
    pthread_cond_t  mailbox_cond;
    engine_cmd_t   *mailbox_head;
    engine_cmd_t   *mailbox_tail;
    atomic_bool     shutdown;
    char           *loaded_model;
    stream_t       *inflight;
} engine_t;

int  engine_init(engine_t *eng);

/* Shut down: cancel in-flight work, join engine thread, drain pending.
   Producers must be quiesced before this returns. */
void engine_destroy(engine_t *eng);

/* Post cmd for async processing. Concurrent posts during destroy are
   safe (re-checked under lock). After destroy returns, posts are
   rejected via atomic fast-path only - the mutex is already destroyed. */
void engine_post(engine_t *eng, engine_cmd_t *cmd);

#endif
