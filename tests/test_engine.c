#include "engine/engine.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Stage 1: stream create / release ------------------------------------ */

static void test_stream_create_basic(void) {
    stream_t *s = stream_create(4);
    assert(s != NULL);
    assert(s->cap == 4);
    assert(s->len == 0);
    assert(s->read_pos == 0);
    assert(atomic_load(&s->refcount) == 1);
    assert(!atomic_load(&s->cancelled));
    assert(s->buf != NULL);
    stream_release(s);
}

static void test_stream_release_null(void) {
    stream_release(NULL);
}

/* ---- Stage 2: stream retain / two-side refcount -------------------------- */

static void test_stream_retain(void) {
    stream_t *s = stream_create(4);
    stream_retain(s);
    assert(atomic_load(&s->refcount) == 2);
    stream_release(s);
    assert(atomic_load(&s->refcount) == 1);
    stream_release(s);
}

/* ---- Stage 3: push/next single-thread FIFO (poll mode) ------------------- */

static void test_push_next_single(void) {
    stream_t *s = stream_create(4);
    chunk_t in = {.tag = CHUNK_TOKEN, .token = {.id = 7, .logprob = -0.5f}};
    assert(stream_push(s, in));
    assert(s->len == 1);

    chunk_t out;
    assert(stream_next(s, &out, 0));
    assert(out.tag == CHUNK_TOKEN);
    assert(out.token.id == 7);
    assert(out.token.logprob == -0.5f);

    assert(!stream_next(s, &out, 0));

    chunk_t done = {.tag = CHUNK_DONE, .done = FINISH_STOP};
    assert(stream_push(s, done));
    chunk_t err = {.tag = CHUNK_ERROR, .error = strdup("fail")};
    assert(stream_push(s, err));

    assert(stream_next(s, &out, 0));
    assert(out.tag == CHUNK_DONE);
    assert(out.done == FINISH_STOP);
    assert(stream_next(s, &out, 0));
    assert(out.tag == CHUNK_ERROR);
    assert(strcmp(out.error, "fail") == 0);
    free(out.error);

    stream_release(s);
}

/* ---- Stage 4: ring wraparound ------------------------------------------- */

static void test_ring_wraparound(void) {
    stream_t *s = stream_create(2);
    chunk_t out;

    for (int i = 0; i < 10; i++) {
        chunk_t in = {.tag = CHUNK_TOKEN, .token = {.id = i, .logprob = 0}};
        assert(stream_push(s, in));
        assert(stream_next(s, &out, 0));
        assert(out.tag == CHUNK_TOKEN);
        assert(out.token.id == i);
    }

    chunk_t a = {.tag = CHUNK_TOKEN, .token = {.id = 100, .logprob = 0}};
    chunk_t b = {.tag = CHUNK_TOKEN, .token = {.id = 101, .logprob = 0}};
    assert(stream_push(s, a));
    assert(stream_push(s, b));
    assert(stream_next(s, &out, 0));
    assert(out.token.id == 100);
    assert(stream_next(s, &out, 0));
    assert(out.token.id == 101);

    stream_release(s);
}

/* ---- Stage 5: blocking next timeout + backpressure push ------------------ */

static long ms_elapsed(struct timespec *start) {
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return (now.tv_sec - start->tv_sec) * 1000 +
           (now.tv_nsec - start->tv_nsec) / 1000000;
}

static void test_next_timeout(void) {
    stream_t *s = stream_create(4);
    chunk_t out;
    struct timespec t0;
    clock_gettime(CLOCK_REALTIME, &t0);
    assert(!stream_next(s, &out, 20));
    long elapsed = ms_elapsed(&t0);
    assert(elapsed >= 15);
    stream_release(s);
}

static void test_next_blocking_immediate(void) {
    stream_t *s = stream_create(4);
    chunk_t in = {.tag = CHUNK_TOKEN, .token = {.id = 42, .logprob = 0}};
    assert(stream_push(s, in));

    chunk_t out;
    assert(stream_next(s, &out, -1));
    assert(out.token.id == 42);
    stream_release(s);
}

struct bp_ctx {
    stream_t *s;
    int       values[3];
};

static void *backpressure_producer(void *arg) {
    struct bp_ctx *ctx = arg;
    for (int i = 0; i < 3; i++) {
        chunk_t c = {.tag = CHUNK_TOKEN, .token = {.id = i, .logprob = 0}};
        stream_push(ctx->s, c);
        ctx->values[i] = i;
    }
    chunk_t done = {.tag = CHUNK_DONE, .done = FINISH_STOP};
    stream_push(ctx->s, done);
    return NULL;
}

static void test_backpressure_push(void) {
    stream_t *s = stream_create(1);
    struct bp_ctx ctx = {.s = s};
    pthread_t tid;
    pthread_create(&tid, NULL, backpressure_producer, &ctx);

    chunk_t out;
    for (int i = 0; i < 3; i++) {
        assert(stream_next(s, &out, -1));
        assert(out.tag == CHUNK_TOKEN);
        assert(out.token.id == i);
    }
    assert(stream_next(s, &out, -1));
    assert(out.tag == CHUNK_DONE);

    pthread_join(tid, NULL);
    stream_release(s);
}

/* ---- Stage 6: cancel semantics ------------------------------------------ */

static void test_cancel_drains_then_false(void) {
    stream_t *s = stream_create(4);
    chunk_t in = {.tag = CHUNK_TOKEN, .token = {.id = 99, .logprob = 0}};
    assert(stream_push(s, in));
    stream_cancel(s);
    assert(!stream_push(s, in));

    chunk_t out;
    assert(stream_next(s, &out, 0));
    assert(out.token.id == 99);
    assert(!stream_next(s, &out, 0));
    stream_release(s);
}

static void *cancel_after_delay(void *arg) {
    stream_t *s = arg;
    struct timespec ts = {.tv_sec = 0, .tv_nsec = 30 * 1000000};
    nanosleep(&ts, NULL);
    stream_cancel(s);
    return NULL;
}

static void test_cancel_unblocks_next(void) {
    stream_t *s = stream_create(4);
    pthread_t tid;
    pthread_create(&tid, NULL, cancel_after_delay, s);

    chunk_t out;
    assert(!stream_next(s, &out, -1));
    pthread_join(tid, NULL);
    stream_release(s);
}

static void *cancel_unblock_push_thread(void *arg) {
    stream_t *s = arg;
    chunk_t c = {.tag = CHUNK_TOKEN, .token = {.id = 1, .logprob = 0}};
    stream_push(s, c);
    chunk_t c2 = {.tag = CHUNK_TOKEN, .token = {.id = 2, .logprob = 0}};
    bool ok = stream_push(s, c2);
    (void)ok;
    return NULL;
}

static void test_cancel_unblocks_push(void) {
    stream_t *s = stream_create(1);
    pthread_t tid;
    pthread_create(&tid, NULL, cancel_unblock_push_thread, s);

    struct timespec ts = {.tv_sec = 0, .tv_nsec = 30 * 1000000};
    nanosleep(&ts, NULL);
    stream_cancel(s);
    pthread_join(tid, NULL);

    chunk_t out;
    while (stream_next(s, &out, 0)) {}
    stream_release(s);
}

/* ---- Stage 7: cross-thread SPSC stress ---------------------------------- */

#define STRESS_COUNT 10000

struct stress_ctx {
    stream_t *s;
};

static void *stress_producer(void *arg) {
    struct stress_ctx *ctx = arg;
    for (int i = 0; i < STRESS_COUNT; i++) {
        chunk_t c = {.tag = CHUNK_TOKEN, .token = {.id = i, .logprob = 0}};
        stream_push(ctx->s, c);
    }
    chunk_t done = {.tag = CHUNK_DONE, .done = FINISH_STOP};
    stream_push(ctx->s, done);
    return NULL;
}

static void test_spsc_stress(void) {
    stream_t *s = stream_create(8);
    struct stress_ctx ctx = {.s = s};
    pthread_t tid;
    pthread_create(&tid, NULL, stress_producer, &ctx);

    chunk_t out;
    int expected = 0;
    while (stream_next(s, &out, -1)) {
        if (out.tag == CHUNK_TOKEN) {
            assert(out.token.id == expected);
            expected++;
        } else if (out.tag == CHUNK_DONE) {
            break;
        }
    }
    assert(expected == STRESS_COUNT);
    pthread_join(tid, NULL);
    stream_release(s);
}

/* ---- Stage 8: engine thread start/stop ---------------------------------- */

static void test_engine_start_stop(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);
    assert(!atomic_load(&eng.shutdown));
    assert(eng.mailbox_head == NULL);
    assert(eng.mailbox_tail == NULL);
    assert(eng.loaded_model == NULL);
    engine_destroy(&eng);
}

/* ---- Stage 9: generate with no model ------------------------------------ */

static void test_generate_no_model(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    stream_t *s = stream_create(8);
    stream_retain(s);
    assert(atomic_load(&s->refcount) == 2);

    int32_t ids[] = {1, 2, 3};
    engine_cmd_t *cmd = calloc(1, sizeof(*cmd));
    cmd->tag = CMD_GENERATE;
    cmd->generate.token_ids = malloc(sizeof(ids));
    memcpy(cmd->generate.token_ids, ids, sizeof(ids));
    cmd->generate.token_count = 3;
    cmd->generate.params.max_tokens = 4;
    cmd->generate.stream = s;
    engine_post(&eng, cmd);

    chunk_t out;
    assert(stream_next(s, &out, -1));
    assert(out.tag == CHUNK_ERROR);
    assert(strcmp(out.error, "model not loaded") == 0);
    free(out.error);

    assert(stream_next(s, &out, -1));
    assert(out.tag == CHUNK_DONE);
    assert(out.done == FINISH_STOP);

    assert(!stream_next(s, &out, 100));
    assert(atomic_load(&s->refcount) == 1);
    stream_release(s);
    engine_destroy(&eng);
}

/* ---- Stage 10: CMD_LOAD then generate echoes tokens --------------------- */

static void test_load_then_generate(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    engine_cmd_t *load = calloc(1, sizeof(*load));
    load->tag = CMD_LOAD;
    load->load.model_path = strdup("/fake");
    engine_post(&eng, load);

    stream_t *s = stream_create(16);
    stream_retain(s);

    int32_t ids[] = {5, 6, 7};
    engine_cmd_t *gen = calloc(1, sizeof(*gen));
    gen->tag = CMD_GENERATE;
    gen->generate.token_ids = malloc(sizeof(ids));
    memcpy(gen->generate.token_ids, ids, sizeof(ids));
    gen->generate.token_count = 3;
    gen->generate.params.max_tokens = 10;
    gen->generate.stream = s;
    engine_post(&eng, gen);

    chunk_t out;
    for (int i = 0; i < 3; i++) {
        assert(stream_next(s, &out, -1));
        assert(out.tag == CHUNK_TOKEN);
        assert(out.token.id == ids[i]);
    }
    assert(stream_next(s, &out, -1));
    assert(out.tag == CHUNK_DONE);
    assert(out.done == FINISH_STOP);

    stream_release(s);
    engine_destroy(&eng);
}

/* ---- Stage 10b: generate with max_tokens truncation --------------------- */

static void test_generate_truncation(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    engine_cmd_t *load = calloc(1, sizeof(*load));
    load->tag = CMD_LOAD;
    load->load.model_path = strdup("/fake");
    engine_post(&eng, load);

    stream_t *s = stream_create(16);
    stream_retain(s);

    int32_t ids[] = {10, 11, 12, 13, 14};
    engine_cmd_t *gen = calloc(1, sizeof(*gen));
    gen->tag = CMD_GENERATE;
    gen->generate.token_ids = malloc(sizeof(ids));
    memcpy(gen->generate.token_ids, ids, sizeof(ids));
    gen->generate.token_count = 5;
    gen->generate.params.max_tokens = 3;
    gen->generate.stream = s;
    engine_post(&eng, gen);

    chunk_t out;
    for (int i = 0; i < 3; i++) {
        assert(stream_next(s, &out, -1));
        assert(out.tag == CHUNK_TOKEN);
        assert(out.token.id == ids[i]);
    }
    assert(stream_next(s, &out, -1));
    assert(out.tag == CHUNK_DONE);
    assert(out.done == FINISH_LENGTH);

    stream_release(s);
    engine_destroy(&eng);
}

/* ---- Stage 11: cancel mid-generation ------------------------------------ */

static void test_cancel_mid_generate(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    engine_cmd_t *load = calloc(1, sizeof(*load));
    load->tag = CMD_LOAD;
    load->load.model_path = strdup("/fake");
    engine_post(&eng, load);

    stream_t *s = stream_create(1);
    stream_retain(s);

    int32_t *ids = malloc(1000 * sizeof(int32_t));
    for (int i = 0; i < 1000; i++) ids[i] = i;
    engine_cmd_t *gen = calloc(1, sizeof(*gen));
    gen->tag = CMD_GENERATE;
    gen->generate.token_ids = ids;
    gen->generate.token_count = 1000;
    gen->generate.params.max_tokens = 1000;
    gen->generate.stream = s;
    engine_post(&eng, gen);

    chunk_t out;
    assert(stream_next(s, &out, -1));
    assert(out.tag == CHUNK_TOKEN);
    assert(out.token.id == 0);

    stream_cancel(s);

    int count = 1;
    while (stream_next(s, &out, 0)) {
        if (out.tag == CHUNK_TOKEN) count++;
        if (out.tag == CHUNK_ERROR) free(out.error);
    }
    assert(count < 1000);

    for (int w = 0; w < 200 && atomic_load(&s->refcount) > 1; w++) {
        struct timespec ws = {.tv_sec = 0, .tv_nsec = 1000000};
        nanosleep(&ws, NULL);
    }
    assert(atomic_load(&s->refcount) == 1);
    stream_release(s);
    engine_destroy(&eng);
}

/* ---- Stage 12: CMD_RECLAIM releases engine ref -------------------------- */

static void test_reclaim(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    stream_t *s = stream_create(4);
    stream_retain(s);
    assert(atomic_load(&s->refcount) == 2);

    engine_cmd_t *cmd = calloc(1, sizeof(*cmd));
    cmd->tag = CMD_RECLAIM;
    cmd->reclaim.stream = s;
    engine_post(&eng, cmd);

    struct timespec ts = {.tv_sec = 0, .tv_nsec = 50 * 1000000};
    nanosleep(&ts, NULL);

    assert(atomic_load(&s->refcount) == 1);
    stream_release(s);
    engine_destroy(&eng);
}

/* ---- Stage 13: notify hook fires after push ----------------------------- */

static void notify_counter(void *ctx) {
    atomic_int *counter = ctx;
    atomic_fetch_add(counter, 1);
}

static void test_notify_hook(void) {
    stream_t *s = stream_create(8);
    atomic_int counter = 0;
    stream_set_notify(s, notify_counter, &counter);

    chunk_t c = {.tag = CHUNK_TOKEN, .token = {.id = 1, .logprob = 0}};
    stream_push(s, c);
    stream_push(s, c);
    stream_push(s, c);
    assert(atomic_load(&counter) == 3);

    stream_cancel(s);
    assert(!stream_push(s, c));
    assert(atomic_load(&counter) == 3);

    chunk_t out;
    while (stream_next(s, &out, 0)) {}
    stream_release(s);
}

/* ---- Stage 14: shutdown cancels in-flight ------------------------------- */

static void test_shutdown_cancels_inflight(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    engine_cmd_t *load = calloc(1, sizeof(*load));
    load->tag = CMD_LOAD;
    load->load.model_path = strdup("/fake");
    engine_post(&eng, load);

    stream_t *s = stream_create(4);
    stream_retain(s);

    int32_t *ids = malloc(100000 * sizeof(int32_t));
    for (int i = 0; i < 100000; i++) ids[i] = i;
    engine_cmd_t *gen = calloc(1, sizeof(*gen));
    gen->tag = CMD_GENERATE;
    gen->generate.token_ids = ids;
    gen->generate.token_count = 100000;
    gen->generate.params.max_tokens = 100000;
    gen->generate.stream = s;
    engine_post(&eng, gen);

    struct timespec ts = {.tv_sec = 0, .tv_nsec = 5 * 1000000};
    nanosleep(&ts, NULL);

    engine_destroy(&eng);

    chunk_t out;
    bool saw_done = false;
    while (stream_next(s, &out, 0)) {
        if (out.tag == CHUNK_DONE) {
            assert(out.done == FINISH_CANCELLED);
            saw_done = true;
        }
        if (out.tag == CHUNK_ERROR) free(out.error);
    }
    assert(saw_done);
    assert(atomic_load(&s->refcount) == 1);
    stream_release(s);
}

/* ---- Stage 15: multi-producer MPSC stress ------------------------------- */

#define MPSC_THREADS    4
#define MPSC_ITERS    250

struct mpsc_ctx {
    engine_t *eng;
    int       thread_id;
    atomic_int *completed;
};

static void *mpsc_producer(void *arg) {
    struct mpsc_ctx *ctx = arg;
    for (int i = 0; i < MPSC_ITERS; i++) {
        stream_t *s = stream_create(8);
        stream_retain(s);

        int32_t *ids = malloc(sizeof(int32_t));
        ids[0] = ctx->thread_id * MPSC_ITERS + i;
        engine_cmd_t *cmd = calloc(1, sizeof(*cmd));
        cmd->tag = CMD_GENERATE;
        cmd->generate.token_ids = ids;
        cmd->generate.token_count = 1;
        cmd->generate.params.max_tokens = 1;
        cmd->generate.stream = s;
        engine_post(ctx->eng, cmd);

        chunk_t out;
        while (stream_next(s, &out, -1)) {
            if (out.tag == CHUNK_ERROR) free(out.error);
            if (out.tag == CHUNK_DONE) break;
        }
        stream_release(s);
        atomic_fetch_add(ctx->completed, 1);
    }
    return NULL;
}

static void test_mpsc_stress(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    engine_cmd_t *load = calloc(1, sizeof(*load));
    load->tag = CMD_LOAD;
    load->load.model_path = strdup("/fake");
    engine_post(&eng, load);

    atomic_int completed = 0;
    struct mpsc_ctx ctxs[MPSC_THREADS];
    pthread_t tids[MPSC_THREADS];
    for (int i = 0; i < MPSC_THREADS; i++) {
        ctxs[i] = (struct mpsc_ctx){.eng = &eng, .thread_id = i, .completed = &completed};
        pthread_create(&tids[i], NULL, mpsc_producer, &ctxs[i]);
    }
    for (int i = 0; i < MPSC_THREADS; i++) {
        pthread_join(tids[i], NULL);
    }
    assert(atomic_load(&completed) == MPSC_THREADS * MPSC_ITERS);
    engine_destroy(&eng);
}

/* ---- Stage 16: shutdown ordering with pending work ---------------------- */

static void test_shutdown_ordering(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    engine_cmd_t *load = calloc(1, sizeof(*load));
    load->tag = CMD_LOAD;
    load->load.model_path = strdup("/fake");
    engine_post(&eng, load);

    stream_t *streams[10];
    for (int i = 0; i < 10; i++) {
        streams[i] = stream_create(16);
        stream_retain(streams[i]);

        int32_t *ids = malloc(100 * sizeof(int32_t));
        for (int j = 0; j < 100; j++) ids[j] = j;
        engine_cmd_t *gen = calloc(1, sizeof(*gen));
        gen->tag = CMD_GENERATE;
        gen->generate.token_ids = ids;
        gen->generate.token_count = 100;
        gen->generate.params.max_tokens = 100;
        gen->generate.stream = streams[i];
        engine_post(&eng, gen);
    }

    struct timespec ts = {.tv_sec = 0, .tv_nsec = 2 * 1000000};
    nanosleep(&ts, NULL);

    engine_destroy(&eng);

    for (int i = 0; i < 10; i++) {
        chunk_t out;
        bool saw_done = false;
        while (stream_next(streams[i], &out, 0)) {
            if (out.tag == CHUNK_DONE) saw_done = true;
            if (out.tag == CHUNK_ERROR) free(out.error);
        }
        assert(saw_done);
        assert(atomic_load(&streams[i]->refcount) == 1);
        stream_release(streams[i]);
    }
}

/* ---- Stage 14b: post after shutdown rejects ----------------------------- */

static void test_post_after_shutdown(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);
    engine_destroy(&eng);

    stream_t *s = stream_create(4);
    stream_retain(s);

    int32_t *ids = malloc(sizeof(int32_t));
    ids[0] = 42;
    engine_cmd_t *cmd = calloc(1, sizeof(*cmd));
    cmd->tag = CMD_GENERATE;
    cmd->generate.token_ids = ids;
    cmd->generate.token_count = 1;
    cmd->generate.params.max_tokens = 1;
    cmd->generate.stream = s;
    engine_post(&eng, cmd);

    chunk_t out;
    bool saw_done = false;
    while (stream_next(s, &out, 0)) {
        if (out.tag == CHUNK_DONE) {
            assert(out.done == FINISH_CANCELLED);
            saw_done = true;
        }
        if (out.tag == CHUNK_ERROR) free(out.error);
    }
    assert(saw_done);
    assert(atomic_load(&s->refcount) == 1);
    stream_release(s);
}

/* ---- main --------------------------------------------------------------- */

int main(void) {
    test_stream_create_basic();
    test_stream_release_null();
    test_stream_retain();
    test_push_next_single();
    test_ring_wraparound();
    test_next_timeout();
    test_next_blocking_immediate();
    test_backpressure_push();
    test_cancel_drains_then_false();
    test_cancel_unblocks_next();
    test_cancel_unblocks_push();
    test_spsc_stress();
    test_engine_start_stop();
    test_generate_no_model();
    test_load_then_generate();
    test_generate_truncation();
    test_cancel_mid_generate();
    test_reclaim();
    test_notify_hook();
    test_shutdown_cancels_inflight();
    test_mpsc_stress();
    test_shutdown_ordering();
    test_post_after_shutdown();
    printf("test_engine: all passed\n");
    return 0;
}
