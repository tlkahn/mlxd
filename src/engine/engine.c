#include "engine/engine.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Stream -------------------------------------------------------------- */

stream_t *stream_create(int capacity) {
    if (capacity <= 0) return NULL;
    stream_t *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->buf = calloc((size_t)capacity, sizeof(chunk_t));
    if (!s->buf) { free(s); return NULL; }
    s->cap = capacity;
    s->len = 0;
    s->read_pos = 0;
    pthread_mutex_init(&s->mtx, NULL);
    pthread_cond_init(&s->cond, NULL);
    atomic_store(&s->cancelled, false);
    atomic_store(&s->refcount, 1);
    s->on_push = NULL;
    s->on_push_ctx = NULL;
    return s;
}

void stream_retain(stream_t *s) {
    atomic_fetch_add(&s->refcount, 1);
}

void stream_release(stream_t *s) {
    if (!s) return;
    if (atomic_fetch_sub(&s->refcount, 1) == 1) {
        for (int i = 0; i < s->len; i++) {
            int idx = (s->read_pos + i) % s->cap;
            if (s->buf[idx].tag == CHUNK_ERROR)
                free(s->buf[idx].error);
        }
        pthread_mutex_destroy(&s->mtx);
        pthread_cond_destroy(&s->cond);
        free(s->buf);
        free(s);
    }
}

bool stream_push(stream_t *s, chunk_t chunk) {
    pthread_mutex_lock(&s->mtx);
    while (s->len == s->cap && !atomic_load(&s->cancelled)) {
        pthread_cond_wait(&s->cond, &s->mtx);
    }
    if (atomic_load(&s->cancelled)) {
        pthread_mutex_unlock(&s->mtx);
        if (chunk.tag == CHUNK_ERROR)
            free(chunk.error);
        return false;
    }
    int write_pos = (s->read_pos + s->len) % s->cap;
    s->buf[write_pos] = chunk;
    s->len++;
    void (*cb)(void *) = s->on_push;
    void *ctx = s->on_push_ctx;
    pthread_cond_broadcast(&s->cond);
    pthread_mutex_unlock(&s->mtx);
    if (cb) cb(ctx);
    return true;
}

bool stream_next(stream_t *s, chunk_t *out, int timeout_ms) {
    pthread_mutex_lock(&s->mtx);

    if (timeout_ms < 0) {
        while (s->len == 0 && !atomic_load(&s->cancelled)) {
            pthread_cond_wait(&s->cond, &s->mtx);
        }
    } else if (timeout_ms > 0) {
        struct timespec deadline;
        clock_gettime(CLOCK_REALTIME, &deadline);
        long total_ns = deadline.tv_nsec + (long)timeout_ms * 1000000L;
        deadline.tv_sec += total_ns / 1000000000L;
        deadline.tv_nsec = total_ns % 1000000000L;
        while (s->len == 0 && !atomic_load(&s->cancelled)) {
            int rc = pthread_cond_timedwait(&s->cond, &s->mtx, &deadline);
            if (rc != 0) break;
        }
    }

    if (s->len == 0) {
        pthread_mutex_unlock(&s->mtx);
        return false;
    }
    *out = s->buf[s->read_pos];
    s->read_pos = (s->read_pos + 1) % s->cap;
    s->len--;
    pthread_cond_broadcast(&s->cond);
    pthread_mutex_unlock(&s->mtx);
    return true;
}

void stream_cancel(stream_t *s) {
    pthread_mutex_lock(&s->mtx);
    atomic_store(&s->cancelled, true);
    pthread_cond_broadcast(&s->cond);
    pthread_mutex_unlock(&s->mtx);
}

void stream_set_notify(stream_t *s, void (*cb)(void *), void *ctx) {
    pthread_mutex_lock(&s->mtx);
    s->on_push = cb;
    s->on_push_ctx = ctx;
    pthread_mutex_unlock(&s->mtx);
}

/* ---- Stream helpers (static) --------------------------------------------- */

static bool stream_push_error(stream_t *s, const char *msg) {
    /* strdup ownership transfers to stream_push (stored in ring buffer,
     * or freed on cancel). Analyzer cannot track through computed index. */
    char *dup = strdup(msg);
    if (!dup) return false;
    __attribute__((suppress))
    return stream_push(s, (chunk_t){.tag = CHUNK_ERROR, .error = dup});
}

static void stream_finish_cancelled(stream_t *s) {
    pthread_mutex_lock(&s->mtx);
    if (s->len < s->cap) {
        int write_pos = (s->read_pos + s->len) % s->cap;
        s->buf[write_pos] = (chunk_t){.tag = CHUNK_DONE, .done = FINISH_CANCELLED};
        s->len++;
    } else {
        int last = (s->read_pos + s->len - 1) % s->cap;
        if (s->buf[last].tag == CHUNK_ERROR)
            free(s->buf[last].error);
        s->buf[last] = (chunk_t){.tag = CHUNK_DONE, .done = FINISH_CANCELLED};
    }
    atomic_store(&s->cancelled, true);
    void (*cb)(void *) = s->on_push;
    void *ctx = s->on_push_ctx;
    pthread_cond_broadcast(&s->cond);
    pthread_mutex_unlock(&s->mtx);
    if (cb) cb(ctx);
}

/* ---- Command helpers ----------------------------------------------------- */

static void cmd_cancel(engine_cmd_t *cmd) {
    switch (cmd->tag) {
    case CMD_GENERATE:
        stream_finish_cancelled(cmd->generate.stream);
        stream_release(cmd->generate.stream);
        free(cmd->generate.token_ids);
        break;
    case CMD_RECLAIM:
        stream_release(cmd->reclaim.stream);
        break;
    case CMD_LOAD:
        free(cmd->load.model_path);
        break;
    default:
        break;
    }
    free(cmd);
}

/* ---- Mailbox ------------------------------------------------------------- */

static engine_cmd_t *mailbox_dequeue(engine_t *eng) {
    pthread_mutex_lock(&eng->mailbox_mtx);
    while (!eng->mailbox_head && !atomic_load(&eng->shutdown)) {
        pthread_cond_wait(&eng->mailbox_cond, &eng->mailbox_mtx);
    }
    engine_cmd_t *cmd = eng->mailbox_head;
    if (cmd) {
        eng->mailbox_head = cmd->next;
        if (!eng->mailbox_head) eng->mailbox_tail = NULL;
        cmd->next = NULL;
    }
    pthread_mutex_unlock(&eng->mailbox_mtx);
    return cmd;
}

/* ---- Engine thread ------------------------------------------------------- */

static void handle_generate(engine_t *eng, engine_cmd_t *cmd) {
    stream_t *s = cmd->generate.stream;

    pthread_mutex_lock(&eng->mailbox_mtx);
    eng->inflight = s;
    pthread_mutex_unlock(&eng->mailbox_mtx);

    if (!eng->loaded_model) {
        bool ok = stream_push_error(s, "model not loaded");
        if (ok)
            ok = stream_push(s, (chunk_t){.tag = CHUNK_DONE, .done = FINISH_STOP});
        if (!ok && atomic_load(&eng->shutdown))
            stream_finish_cancelled(s);
        goto done;
    }

    {
        int count = cmd->generate.token_count;
        int max = cmd->generate.params.max_tokens; /* validated at HTTP boundary */
        int limit = count < max ? count : max;
        bool truncated = limit < count;
        bool aborted = false;

        for (int i = 0; i < limit; i++) {
            if (atomic_load(&eng->shutdown)) {
                stream_finish_cancelled(s);
                aborted = true;
                break;
            }
            chunk_t tok = {.tag = CHUNK_TOKEN, .token = {.id = cmd->generate.token_ids[i], .logprob = 0}};
            if (!stream_push(s, tok)) {
                stream_finish_cancelled(s);
                aborted = true;
                break;
            }
        }

        if (!aborted) {
            finish_reason_t reason = truncated ? FINISH_LENGTH : FINISH_STOP;
            stream_push(s, (chunk_t){.tag = CHUNK_DONE, .done = reason});
        }
    }

done:
    pthread_mutex_lock(&eng->mailbox_mtx);
    eng->inflight = NULL;
    pthread_mutex_unlock(&eng->mailbox_mtx);

    stream_release(s);
    free(cmd->generate.token_ids);
    free(cmd);
}

static void *engine_thread_main(void *arg) {
    engine_t *eng = arg;

    while (!atomic_load(&eng->shutdown)) {
        engine_cmd_t *cmd = mailbox_dequeue(eng);
        if (!cmd) continue;

        switch (cmd->tag) {
        case CMD_STOP:
            free(cmd);
            return NULL;
        case CMD_LOAD:
            free(eng->loaded_model);
            eng->loaded_model = cmd->load.model_path;
            free(cmd);
            break;
        case CMD_UNLOAD:
            free(eng->loaded_model);
            eng->loaded_model = NULL;
            free(cmd);
            break;
        case CMD_GENERATE:
            handle_generate(eng, cmd);
            break;
        case CMD_RECLAIM:
            stream_release(cmd->reclaim.stream);
            free(cmd);
            break;
        case CMD_EMBED:
            free(cmd);
            break;
        }
    }

    return NULL;
}

/* ---- Engine public API --------------------------------------------------- */

int engine_init(engine_t *eng) {
    memset(eng, 0, sizeof(*eng));
    pthread_mutex_init(&eng->mailbox_mtx, NULL);
    pthread_cond_init(&eng->mailbox_cond, NULL);
    eng->mailbox_head = NULL;
    eng->mailbox_tail = NULL;
    atomic_store(&eng->shutdown, false);
    eng->loaded_model = NULL;
    eng->inflight = NULL;

    if (pthread_create(&eng->thread, NULL, engine_thread_main, eng) != 0) {
        pthread_mutex_destroy(&eng->mailbox_mtx);
        pthread_cond_destroy(&eng->mailbox_cond);
        return -1;
    }
    return 0;
}

void engine_destroy(engine_t *eng) {
    atomic_store(&eng->shutdown, true);

    pthread_mutex_lock(&eng->mailbox_mtx);
    if (eng->inflight)
        stream_cancel(eng->inflight);
    pthread_cond_signal(&eng->mailbox_cond);
    pthread_mutex_unlock(&eng->mailbox_mtx);

    pthread_join(eng->thread, NULL);

    pthread_mutex_lock(&eng->mailbox_mtx);
    engine_cmd_t *cmd = eng->mailbox_head;
    eng->mailbox_head = NULL;
    eng->mailbox_tail = NULL;
    pthread_mutex_unlock(&eng->mailbox_mtx);

    while (cmd) {
        engine_cmd_t *next = cmd->next;
        cmd_cancel(cmd);
        cmd = next;
    }

    free(eng->loaded_model);
    eng->loaded_model = NULL;

    pthread_mutex_destroy(&eng->mailbox_mtx);
    pthread_cond_destroy(&eng->mailbox_cond);
}

void engine_post(engine_t *eng, engine_cmd_t *cmd) {
    if (atomic_load(&eng->shutdown)) {
        cmd_cancel(cmd);
        return;
    }
    cmd->next = NULL;
    pthread_mutex_lock(&eng->mailbox_mtx);
    if (atomic_load(&eng->shutdown)) {
        pthread_mutex_unlock(&eng->mailbox_mtx);
        cmd_cancel(cmd);
        return;
    }
    if (eng->mailbox_tail) {
        eng->mailbox_tail->next = cmd;
    } else {
        eng->mailbox_head = cmd;
    }
    eng->mailbox_tail = cmd;
    pthread_cond_signal(&eng->mailbox_cond);
    pthread_mutex_unlock(&eng->mailbox_mtx);
}
