#ifndef MLXD_ENGINE_H
#define MLXD_ENGINE_H

#include "core/types.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>

/* Opaque: concrete layout lives in engine/emodel.h (engine thread only). */
typedef struct engine_model engine_model_t;

/* Exact strcmp sentinel for the CPU-only echo path. No I/O, no mlx. */
#define MLXD_STUB_MODEL_PATH "stub"

/* Prefill chunk size (tokens). Cancel/shutdown polled between chunks. */
#define MLXD_PREFILL_CHUNK 512

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

/* --- Load state ----------------------------------------------------------- */

typedef enum {
    LOAD_IDLE = 0,
    LOAD_IN_PROGRESS,
    LOAD_OK,
    LOAD_FAILED,
} load_state_t;

/* --- Engine --------------------------------------------------------------- */

typedef struct {
    pthread_t       thread;
    pthread_mutex_t mailbox_mtx;
    pthread_cond_t  mailbox_cond;
    engine_cmd_t   *mailbox_head;
    engine_cmd_t   *mailbox_tail;
    atomic_bool     shutdown;
    atomic_int      load_state;
    pthread_mutex_t load_mtx;
    char            load_error[256];
    engine_model_t *model;        /* owned by engine thread; shell allocated in init */
    char           *loaded_model; /* retained path string for logging / unload */
    stream_t       *inflight;
} engine_t;

int  engine_init(engine_t *eng);

/* Shut down: cancel in-flight work, join engine thread, drain pending.
   Producers must be quiesced before this returns. */
void engine_destroy(engine_t *eng);

/* Signal shutdown without joining. Sets shutdown flag, cancels in-flight
   stream, and wakes the engine thread. Safe from any thread including
   libuv callbacks. engine_destroy still required for full cleanup. */
void engine_signal_shutdown(engine_t *eng);

/* Load-state observers. Safe from any thread. */
load_state_t engine_load_state(const engine_t *eng);
/* Copies under load_mtx into buf; returns 0, or -1 if buf too small / NULL.
   Non-const: takes load_mtx (synchronized reader). */
int          engine_load_error(engine_t *eng, char *buf, size_t n);

/* Compat shim: true once CMD_LOAD has reached LOAD_OK. */
static inline bool engine_loaded(const engine_t *eng) {
    return engine_load_state(eng) == LOAD_OK;
}

/* Poll until LOAD_OK / LOAD_FAILED / timeout / cancel.
   timeout_ms < 0 means wait forever (cancel still honored).
   Returns 0 on LOAD_OK, -1 on LOAD_FAILED, timeout, or cancel.
   On LOAD_FAILED, engine_load_error is populated. */
int engine_wait_load(engine_t *eng, int timeout_ms);

/* Like engine_wait_load, but also aborts early when cancel(ud) returns true
   (return -1). Used by cmd_run for Ctrl-C during load. cancel may be NULL.
   timeout_ms < 0 means wait forever (cancel still honored). */
int engine_wait_load_until(engine_t *eng, int timeout_ms,
                           bool (*cancel)(void *), void *ud);

/* Post cmd for async processing. Concurrent posts during destroy are
   safe (re-checked under lock). After destroy returns, posts are
   rejected via atomic fast-path only - the mutex is already destroyed. */
void engine_post(engine_t *eng, engine_cmd_t *cmd);

/* Post a CMD_LOAD. Takes ownership of model_path (freed by the engine on
   all paths). Synchronously sets LOAD_IN_PROGRESS and clears load_error
   when accepted so engine_wait_load cannot observe a stale terminal state.
   On shutdown rejection, frees path and returns -1 without changing
   load_state. Returns 0 if enqueued. */
int engine_post_load(engine_t *eng, char *model_path);

#endif
