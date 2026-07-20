#include "engine/engine.h"
#include "engine/emodel.h"
#include "engine/sampler.h"
#include "mlxbridge/mlxbridge.h"

#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef MLXD_FIXTURES_DIR
#error "MLXD_FIXTURES_DIR must be defined"
#endif

#define FIXTURES MLXD_FIXTURES_DIR

/* Pinned greedy sequence for prompt {1,2}, max_tokens=8 on tiny_qwen3.
   Guards silent but still-deterministic decode/prefill regressions. */
static const int32_t kOracle[] = {76, 112, 63, 76, 195, 184, 121, 189};

static load_state_t poll_load_terminal(engine_t *eng, int timeout_ms) {
    for (int i = 0; i < timeout_ms; i++) {
        load_state_t st = engine_load_state(eng);
        if (st == LOAD_OK || st == LOAD_FAILED)
            return st;
        usleep(1000);
    }
    return engine_load_state(eng);
}

static void post_load(engine_t *eng, const char *path) {
    char *p = strdup(path);
    assert(p);
    assert(engine_post_load(eng, p) == 0);
}

/* ---- B3.3: real CMD_LOAD of tiny_qwen3 ---------------------------------- */

static void test_load_tiny_qwen3_ok(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    post_load(&eng, FIXTURES "/tiny_qwen3");
    load_state_t st = poll_load_terminal(&eng, 30000);
    assert(st == LOAD_OK);
    assert(engine_loaded(&eng));

    char err[256];
    assert(engine_load_error(&eng, err, sizeof(err)) == 0);
    assert(err[0] == '\0');

    engine_destroy(&eng);
}

/* ---- B3.6: oversized prompt --------------------------------------------- */

static void test_oversized_prompt(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    stream_t *s = stream_create(16);
    stream_retain(s);

    /* tiny_qwen3 max_position_embeddings is 2048 (#46 multi-chunk grain). */
    int n = 2049;
    int32_t *ids = calloc((size_t)n, sizeof(int32_t));
    assert(ids);

    engine_cmd_t *gen = calloc(1, sizeof(*gen));
    assert(gen);
    gen->tag = CMD_GENERATE;
    gen->generate.token_ids = ids;
    gen->generate.token_count = n;
    gen->generate.params.max_tokens = 8;
    gen->generate.stream = s;
    engine_post(&eng, gen);

    chunk_t out;
    assert(stream_next(s, &out, 5000));
    assert(out.tag == CHUNK_ERROR);
    assert(out.error_kind == GEN_ERR_CONTEXT_LENGTH);
    assert(strstr(out.error, "2049") != NULL);
    assert(strstr(out.error, "2048") != NULL);
    free(out.error);

    assert(stream_next(s, &out, 5000));
    assert(out.tag == CHUNK_DONE);

    stream_release(s);
    engine_destroy(&eng);
}

/* ---- B3.8: greedy argmax ------------------------------------------------ */

static void test_greedy_argmax(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    float data[8] = {0.1f, -1.0f, 0.5f, 0.2f, -0.3f, 2.5f, 1.0f, 0.0f};
    int shape[] = {1, 8};
    mlx_array logits = mlx_array_new_data(data, shape, 2, MLX_FLOAT32);
    mlx_array nokey = mlx_array_new();

    sampling_params_t sp = SAMPLING_PARAMS_DEFAULT;
    sp.temperature = 0.0f;

    mlx_array tok = mlx_array_new();
    assert(sampler_sample_lazy(logits, &sp, nokey, false, s, &tok, NULL) == 0);
    int32_t id = -1;
    assert(mlxbridge_item_int32(&id, tok) == 0);
    assert(id == 5);

    mlx_array_free(tok);
    mlx_array_free(nokey);
    mlx_array_free(logits);
}

/* ---- Stream helpers ----------------------------------------------------- */

static int collect_tokens_lp(stream_t *s, int32_t *out_ids, float *out_lps,
                             int max_ids, finish_reason_t *reason_out) {
    int n = 0;
    chunk_t c;
    *reason_out = FINISH_STOP;
    while (stream_next(s, &c, 30000)) {
        if (c.tag == CHUNK_TOKEN) {
            if (n < max_ids) {
                out_ids[n] = c.token.id;
                if (out_lps) out_lps[n] = c.token.logprob;
            }
            n++;
        } else if (c.tag == CHUNK_DONE) {
            *reason_out = c.done;
            break;
        } else if (c.tag == CHUNK_ERROR) {
            fprintf(stderr, "unexpected error: %s\n", c.error);
            free(c.error);
            return -1;
        }
    }
    return n;
}

static int collect_tokens(stream_t *s, int32_t *out_ids, int max_ids,
                          finish_reason_t *reason_out) {
    return collect_tokens_lp(s, out_ids, NULL, max_ids, reason_out);
}

/* ---- B3.9/B3.10: deterministic generate --------------------------------- */

static void test_generate_deterministic(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    int32_t prompt[] = {1, 2};
    int max_new = 8;

    int32_t seq_a[16];
    int32_t seq_b[16];
    finish_reason_t reason_a, reason_b;

    /* run 1 */
    {
        stream_t *s = stream_create(32);
        stream_retain(s);
        engine_cmd_t *gen = calloc(1, sizeof(*gen));
        assert(gen);
        gen->tag = CMD_GENERATE;
        gen->generate.token_ids = malloc(sizeof(prompt));
        memcpy(gen->generate.token_ids, prompt, sizeof(prompt));
        gen->generate.token_count = 2;
        gen->generate.params.max_tokens = max_new;
        gen->generate.params.sampling.temperature = 0.0f;
        gen->generate.params.sampling_set = SAMPLING_SET_TEMPERATURE;
        gen->generate.stream = s;
        engine_post(&eng, gen);

        int n = collect_tokens(s, seq_a, 16, &reason_a);
        assert(n == max_new);
        assert(reason_a == FINISH_LENGTH);
        stream_release(s);
    }

    /* run 2 - must match */
    {
        stream_t *s = stream_create(32);
        stream_retain(s);
        engine_cmd_t *gen = calloc(1, sizeof(*gen));
        assert(gen);
        gen->tag = CMD_GENERATE;
        gen->generate.token_ids = malloc(sizeof(prompt));
        memcpy(gen->generate.token_ids, prompt, sizeof(prompt));
        gen->generate.token_count = 2;
        gen->generate.params.max_tokens = max_new;
        gen->generate.params.sampling.temperature = 0.0f;
        gen->generate.params.sampling_set = SAMPLING_SET_TEMPERATURE;
        gen->generate.stream = s;
        engine_post(&eng, gen);

        int n = collect_tokens(s, seq_b, 16, &reason_b);
        assert(n == max_new);
        assert(reason_b == FINISH_LENGTH);
        stream_release(s);
    }

    for (int i = 0; i < max_new; i++)
        assert(seq_a[i] == seq_b[i]);

    assert(max_new == (int)(sizeof kOracle / sizeof kOracle[0]));
    for (int i = 0; i < max_new; i++)
        assert(seq_a[i] == kOracle[i]);

    printf("  deterministic seq:");
    for (int i = 0; i < max_new; i++)
        printf(" %d", seq_a[i]);
    printf("\n");

    engine_destroy(&eng);
}

/* ---- B3.9: eos_token_ids -> FINISH_STOP --------------------------------- */

static void test_generate_eos_finish_stop(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);
    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    /* Install a known future token id as eos without touching the fixture. */
    const int stop_at = 2; /* third token: 63 */
    eng.model->cfg.eos_token_ids[0] = (uint32_t)kOracle[stop_at];
    eng.model->cfg.num_eos_tokens = 1;

    stream_t *s = stream_create(32);
    stream_retain(s);
    int32_t prompt[] = {1, 2};
    engine_cmd_t *gen = calloc(1, sizeof(*gen));
    assert(gen);
    gen->tag = CMD_GENERATE;
    gen->generate.token_ids = malloc(sizeof(prompt));
    memcpy(gen->generate.token_ids, prompt, sizeof(prompt));
    gen->generate.token_count = 2;
    gen->generate.params.max_tokens = 8; /* > stop_at+1 so length is not the reason */
    gen->generate.params.sampling.temperature = 0.0f;
    gen->generate.params.sampling_set = SAMPLING_SET_TEMPERATURE;
    gen->generate.stream = s;
    engine_post(&eng, gen);

    int32_t got[16];
    finish_reason_t reason = FINISH_LENGTH;
    int n = collect_tokens(s, got, 16, &reason);

    assert(n == stop_at);
    for (int i = 0; i < n; i++)
        assert(got[i] == kOracle[i]);
    assert(reason == FINISH_STOP);

    stream_release(s);
    engine_destroy(&eng);
}

/* ---- B3.11: cancel mid-generate ---------------------------------------- */

static void test_cancel_mid_generate_real(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    stream_t *s = stream_create(4);
    stream_retain(s);

    int32_t prompt[] = {1, 2};
    engine_cmd_t *gen = calloc(1, sizeof(*gen));
    assert(gen);
    gen->tag = CMD_GENERATE;
    gen->generate.token_ids = malloc(sizeof(prompt));
    memcpy(gen->generate.token_ids, prompt, sizeof(prompt));
    gen->generate.token_count = 2;
    gen->generate.params.max_tokens = 1000;
    gen->generate.stream = s;
    engine_post(&eng, gen);

    chunk_t out;
    assert(stream_next(s, &out, 30000));
    assert(out.tag == CHUNK_TOKEN);

    stream_cancel(s);

    /* Wait for engine to finish (injects DONE/CANCELLED, then releases). */
    for (int w = 0; w < 5000 && atomic_load(&s->refcount) > 1; w++)
        usleep(1000);
    assert(atomic_load(&s->refcount) == 1);

    bool saw_cancelled = false;
    while (stream_next(s, &out, 0)) {
        if (out.tag == CHUNK_DONE) {
            assert(out.done == FINISH_CANCELLED);
            saw_cancelled = true;
        }
        if (out.tag == CHUNK_ERROR)
            free(out.error);
    }
    assert(saw_cancelled);

    stream_release(s);
    engine_destroy(&eng);
}

/* ---- B3.11: shutdown mid-generate -------------------------------------- */

static void test_shutdown_mid_generate_real(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    stream_t *s = stream_create(4);
    stream_retain(s);

    int32_t prompt[] = {1, 2};
    engine_cmd_t *gen = calloc(1, sizeof(*gen));
    assert(gen);
    gen->tag = CMD_GENERATE;
    gen->generate.token_ids = malloc(sizeof(prompt));
    memcpy(gen->generate.token_ids, prompt, sizeof(prompt));
    gen->generate.token_count = 2;
    gen->generate.params.max_tokens = 1000;
    gen->generate.stream = s;
    engine_post(&eng, gen);

    /* Wait for first token, then shutdown */
    chunk_t out;
    bool got_token = stream_next(s, &out, 30000);
    if (got_token && out.tag == CHUNK_ERROR) {
        free(out.error);
        assert(!"unexpected error before shutdown");
    }

    engine_signal_shutdown(&eng);
    engine_destroy(&eng);

    bool saw_done = false;
    if (got_token && out.tag == CHUNK_DONE) {
        assert(out.done == FINISH_CANCELLED);
        saw_done = true;
    }
    while (stream_next(s, &out, 0)) {
        if (out.tag == CHUNK_DONE) {
            assert(out.done == FINISH_CANCELLED);
            saw_done = true;
        }
        if (out.tag == CHUNK_ERROR)
            free(out.error);
    }
    assert(saw_done);
    assert(atomic_load(&s->refcount) == 1);
    stream_release(s);
}

/* ---- C2.11: engine logprob ---- */

static void test_generate_logprobs(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    int32_t prompt[] = {1, 2};
    int max_new = 8;

    /* logprobs=true, temperature=0 (greedy): every chunk has logprob < 0 and finite */
    {
        stream_t *s = stream_create(32);
        stream_retain(s);
        engine_cmd_t *gen = calloc(1, sizeof(*gen));
        assert(gen);
        gen->tag = CMD_GENERATE;
        gen->generate.token_ids = malloc(sizeof(prompt));
        memcpy(gen->generate.token_ids, prompt, sizeof(prompt));
        gen->generate.token_count = 2;
        gen->generate.params.max_tokens = max_new;
        gen->generate.params.logprobs = true;
        gen->generate.params.sampling = SAMPLING_PARAMS_DEFAULT;
        gen->generate.params.sampling.temperature = 0.0f;
        gen->generate.params.sampling_set = SAMPLING_SET_TEMPERATURE;
        gen->generate.stream = s;
        engine_post(&eng, gen);

        int32_t ids[16];
        float lps[16];
        finish_reason_t reason;
        int n = collect_tokens_lp(s, ids, lps, 16, &reason);
        assert(n == max_new);
        assert(reason == FINISH_LENGTH);
        for (int i = 0; i < n; i++) {
            assert(isfinite(lps[i]));
            assert(lps[i] < 0.0f);
        }
        stream_release(s);
    }

    /* logprobs=false: all logprobs are 0 */
    {
        stream_t *s = stream_create(32);
        stream_retain(s);
        engine_cmd_t *gen = calloc(1, sizeof(*gen));
        assert(gen);
        gen->tag = CMD_GENERATE;
        gen->generate.token_ids = malloc(sizeof(prompt));
        memcpy(gen->generate.token_ids, prompt, sizeof(prompt));
        gen->generate.token_count = 2;
        gen->generate.params.max_tokens = max_new;
        gen->generate.params.logprobs = false;
        gen->generate.params.sampling = SAMPLING_PARAMS_DEFAULT;
        gen->generate.params.sampling.temperature = 0.0f;
        gen->generate.params.sampling_set = SAMPLING_SET_TEMPERATURE;
        gen->generate.stream = s;
        engine_post(&eng, gen);

        int32_t ids[16];
        float lps[16];
        finish_reason_t reason;
        int n = collect_tokens_lp(s, ids, lps, 16, &reason);
        assert(n == max_new);
        for (int i = 0; i < n; i++)
            assert(lps[i] == 0.0f);
        stream_release(s);
    }

    engine_destroy(&eng);
}

/* ---- C1.8: seeded sampling via engine ---- */

static void test_generate_seeded_sampling(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    int32_t prompt[] = {1, 2};
    int max_new = 8;

    /* Two runs with temperature=5, seed=7 must be identical */
    int32_t seq_a[16], seq_b[16];
    finish_reason_t reason_a, reason_b;

    for (int run = 0; run < 2; run++) {
        int32_t *seq = (run == 0) ? seq_a : seq_b;
        finish_reason_t *reason = (run == 0) ? &reason_a : &reason_b;

        stream_t *s = stream_create(32);
        stream_retain(s);
        engine_cmd_t *gen = calloc(1, sizeof(*gen));
        assert(gen);
        gen->tag = CMD_GENERATE;
        gen->generate.token_ids = malloc(sizeof(prompt));
        memcpy(gen->generate.token_ids, prompt, sizeof(prompt));
        gen->generate.token_count = 2;
        gen->generate.params.max_tokens = max_new;
        gen->generate.params.sampling.temperature = 5.0f;
        gen->generate.params.sampling.seed = 7;
        gen->generate.params.sampling.top_p = 1.0f;
        gen->generate.params.sampling.top_k = -1;
        gen->generate.params.sampling.min_p = 0.0f;
        gen->generate.params.sampling_set = SAMPLING_SET_TEMPERATURE | SAMPLING_SET_SEED |
                                            SAMPLING_SET_TOP_P | SAMPLING_SET_TOP_K |
                                            SAMPLING_SET_MIN_P;
        gen->generate.stream = s;
        engine_post(&eng, gen);

        int n = collect_tokens(s, seq, 16, reason);
        assert(n == max_new);
        assert(*reason == FINISH_LENGTH);
        stream_release(s);
    }

    /* Seeded runs are identical */
    for (int i = 0; i < max_new; i++)
        assert(seq_a[i] == seq_b[i]);

    /* Must differ from greedy oracle (temperature 5 is far from greedy) */
    bool differs = false;
    for (int i = 0; i < max_new; i++) {
        if (seq_a[i] != kOracle[i]) { differs = true; break; }
    }
    assert(differs);

    engine_destroy(&eng);
}

/* ---- C2.14: gen-config defaults reach the GPU ---- */

static void test_generate_genconfig_defaults(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    /* Inject gen_temperature=0 into the model config (post-load injection) */
    eng.model->cfg.has_gen_temperature = true;
    eng.model->cfg.gen_temperature = 0.0f;

    /* Request does NOT set temperature (sampling_set=0) -> resolve picks gen_config's 0 -> greedy */
    int32_t prompt[] = {1, 2};
    int max_new = 8;
    {
        stream_t *s = stream_create(32);
        stream_retain(s);
        engine_cmd_t *gen = calloc(1, sizeof(*gen));
        assert(gen);
        gen->tag = CMD_GENERATE;
        gen->generate.token_ids = malloc(sizeof(prompt));
        memcpy(gen->generate.token_ids, prompt, sizeof(prompt));
        gen->generate.token_count = 2;
        gen->generate.params.max_tokens = max_new;
        gen->generate.stream = s;
        engine_post(&eng, gen);

        int32_t ids[16];
        finish_reason_t reason;
        int n = collect_tokens(s, ids, 16, &reason);
        assert(n == max_new);
        for (int i = 0; i < max_new; i++)
            assert(ids[i] == kOracle[i]);
        stream_release(s);
    }

    /* Request sets temperature=5 + seed (mask on) -> overrides gen_config -> not greedy */
    {
        stream_t *s = stream_create(32);
        stream_retain(s);
        engine_cmd_t *gen = calloc(1, sizeof(*gen));
        assert(gen);
        gen->tag = CMD_GENERATE;
        gen->generate.token_ids = malloc(sizeof(prompt));
        memcpy(gen->generate.token_ids, prompt, sizeof(prompt));
        gen->generate.token_count = 2;
        gen->generate.params.max_tokens = max_new;
        gen->generate.params.sampling.temperature = 5.0f;
        gen->generate.params.sampling.seed = 99;
        gen->generate.params.sampling_set = SAMPLING_SET_TEMPERATURE | SAMPLING_SET_SEED;
        gen->generate.stream = s;
        engine_post(&eng, gen);

        int32_t ids[16];
        finish_reason_t reason;
        int n = collect_tokens(s, ids, 16, &reason);
        assert(n == max_new);
        bool differs = false;
        for (int i = 0; i < max_new; i++) {
            if (ids[i] != kOracle[i]) { differs = true; break; }
        }
        assert(differs);
        stream_release(s);
    }

    engine_destroy(&eng);
}

/* ---- C2.15: multi-eos termination ---- */

static void test_generate_multi_eos(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);

    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    int32_t prompt[] = {1, 2};
    int max_new = 8;

    /* Inject two eos ids: kOracle[2] (63) and kOracle[5] (184) */
    eng.model->cfg.eos_token_ids[0] = (uint32_t)kOracle[2];
    eng.model->cfg.eos_token_ids[1] = (uint32_t)kOracle[5];
    eng.model->cfg.num_eos_tokens = 2;

    /* Should stop at first hit (kOracle[2] = 63, position 2) */
    {
        stream_t *s = stream_create(32);
        stream_retain(s);
        engine_cmd_t *gen = calloc(1, sizeof(*gen));
        assert(gen);
        gen->tag = CMD_GENERATE;
        gen->generate.token_ids = malloc(sizeof(prompt));
        memcpy(gen->generate.token_ids, prompt, sizeof(prompt));
        gen->generate.token_count = 2;
        gen->generate.params.max_tokens = max_new;
        gen->generate.params.sampling.temperature = 0.0f;
        gen->generate.params.sampling_set = SAMPLING_SET_TEMPERATURE;
        gen->generate.stream = s;
        engine_post(&eng, gen);

        int32_t ids[16];
        finish_reason_t reason;
        int n = collect_tokens(s, ids, 16, &reason);
        assert(n == 2);
        assert(reason == FINISH_STOP);
        for (int i = 0; i < n; i++)
            assert(ids[i] == kOracle[i]);
        stream_release(s);
    }

    /* Inject only the later id (kOracle[5] = 184): should stop at position 5 */
    eng.model->cfg.eos_token_ids[0] = (uint32_t)kOracle[5];
    eng.model->cfg.eos_token_ids[1] = 0;
    eng.model->cfg.num_eos_tokens = 1;
    {
        stream_t *s = stream_create(32);
        stream_retain(s);
        engine_cmd_t *gen = calloc(1, sizeof(*gen));
        assert(gen);
        gen->tag = CMD_GENERATE;
        gen->generate.token_ids = malloc(sizeof(prompt));
        memcpy(gen->generate.token_ids, prompt, sizeof(prompt));
        gen->generate.token_count = 2;
        gen->generate.params.max_tokens = max_new;
        gen->generate.params.sampling.temperature = 0.0f;
        gen->generate.params.sampling_set = SAMPLING_SET_TEMPERATURE;
        gen->generate.stream = s;
        engine_post(&eng, gen);

        int32_t ids[16];
        finish_reason_t reason;
        int n = collect_tokens(s, ids, 16, &reason);
        assert(n == 5);
        assert(reason == FINISH_STOP);
        for (int i = 0; i < n; i++)
            assert(ids[i] == kOracle[i]);
        stream_release(s);
    }

    engine_destroy(&eng);
}

/* ---- #46: bound shutdown drain (hooks + deadlines) ---------------------- */

/* Drain bound for tiny_qwen3 fixture (ms). HTTP default is 5000; engine-side
   tests use a stricter bound so a stuck loop cannot hide behind slack. */
#define MLXD_TEST_DRAIN_BOUND_MS 2000

typedef struct {
    pthread_mutex_t mtx;
    pthread_cond_t  cond;
    int             entered; /* times the target barrier site was hit */
    int             release; /* non-zero => hook may return */
    int             trip_at; /* fire barrier when entered reaches this */
    /* optional counters / last-seen values */
    int             prefill_calls;
    int             last_pos;
    int             last_chunk_len;
    int             decode_calls;
    int             seed_calls;
    /* for multi-chunk position pin */
    int             prefill_pos[8];
    int             prefill_len[8];
} cancel_barrier_t;

static void barrier_init(cancel_barrier_t *b) {
    memset(b, 0, sizeof(*b));
    pthread_mutex_init(&b->mtx, NULL);
    pthread_cond_init(&b->cond, NULL);
    b->trip_at = 1;
}

static void barrier_destroy(cancel_barrier_t *b) {
    pthread_mutex_destroy(&b->mtx);
    pthread_cond_destroy(&b->cond);
}

/* Block in hook until test thread signals release (after trip_at hits). */
static void barrier_maybe_block(cancel_barrier_t *b) {
    pthread_mutex_lock(&b->mtx);
    b->entered++;
    if (b->entered == b->trip_at) {
        pthread_cond_broadcast(&b->cond);
        while (!b->release)
            pthread_cond_wait(&b->cond, &b->mtx);
    }
    pthread_mutex_unlock(&b->mtx);
}

static void barrier_wait_entered(cancel_barrier_t *b) {
    pthread_mutex_lock(&b->mtx);
    while (b->entered < b->trip_at)
        pthread_cond_wait(&b->cond, &b->mtx);
    pthread_mutex_unlock(&b->mtx);
}

static void barrier_release(cancel_barrier_t *b) {
    pthread_mutex_lock(&b->mtx);
    b->release = 1;
    pthread_cond_broadcast(&b->cond);
    pthread_mutex_unlock(&b->mtx);
}

static int64_t mono_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/* Wait until stream_sole_owner or timeout_ms. Returns elapsed ms from T0. */
static int64_t wait_sole_owner_ms(stream_t *s, int64_t t0, int timeout_ms) {
    for (;;) {
        if (stream_sole_owner(s))
            return mono_ms() - t0;
        if (mono_ms() - t0 > timeout_ms)
            return mono_ms() - t0;
        usleep(200);
    }
}

static bool drain_saw_cancelled(stream_t *s) {
    bool saw = false;
    chunk_t out;
    while (stream_next(s, &out, 0)) {
        if (out.tag == CHUNK_DONE && out.done == FINISH_CANCELLED)
            saw = true;
        if (out.tag == CHUNK_ERROR)
            free(out.error);
    }
    return saw;
}

static int count_tokens_drained(stream_t *s, finish_reason_t *reason_out) {
    int n = 0;
    chunk_t out;
    *reason_out = FINISH_STOP;
    while (stream_next(s, &out, 0)) {
        if (out.tag == CHUNK_TOKEN)
            n++;
        else if (out.tag == CHUNK_DONE)
            *reason_out = out.done;
        else if (out.tag == CHUNK_ERROR)
            free(out.error);
    }
    return n;
}

static void hook_prefill_count(void *ud, int pos, int chunk_len) {
    cancel_barrier_t *b = ud;
    pthread_mutex_lock(&b->mtx);
    if (b->prefill_calls < 8) {
        b->prefill_pos[b->prefill_calls] = pos;
        b->prefill_len[b->prefill_calls] = chunk_len;
    }
    b->last_pos = pos;
    b->last_chunk_len = chunk_len;
    b->prefill_calls++;
    pthread_mutex_unlock(&b->mtx);
    barrier_maybe_block(b);
}

static void hook_prefill_count_only(void *ud, int pos, int chunk_len) {
    cancel_barrier_t *b = ud;
    pthread_mutex_lock(&b->mtx);
    if (b->prefill_calls < 8) {
        b->prefill_pos[b->prefill_calls] = pos;
        b->prefill_len[b->prefill_calls] = chunk_len;
    }
    b->last_pos = pos;
    b->last_chunk_len = chunk_len;
    b->prefill_calls++;
    pthread_mutex_unlock(&b->mtx);
}

static void hook_decode_barrier(void *ud, int step) {
    cancel_barrier_t *b = ud;
    pthread_mutex_lock(&b->mtx);
    b->decode_calls++;
    pthread_mutex_unlock(&b->mtx);
    (void)step;
    barrier_maybe_block(b);
}

static void hook_decode_count_only(void *ud, int step) {
    cancel_barrier_t *b = ud;
    (void)step;
    pthread_mutex_lock(&b->mtx);
    b->decode_calls++;
    pthread_mutex_unlock(&b->mtx);
}

static void hook_seed_barrier(void *ud) {
    cancel_barrier_t *b = ud;
    pthread_mutex_lock(&b->mtx);
    b->seed_calls++;
    pthread_mutex_unlock(&b->mtx);
    barrier_maybe_block(b);
}

static stream_t *post_generate_prompt(engine_t *eng, const int32_t *ids, int n,
                                      int max_tokens) {
    stream_t *s = stream_create(32);
    stream_retain(s);
    engine_cmd_t *gen = calloc(1, sizeof(*gen));
    assert(gen);
    gen->tag = CMD_GENERATE;
    gen->generate.token_ids = malloc((size_t)n * sizeof(int32_t));
    assert(gen->generate.token_ids);
    memcpy(gen->generate.token_ids, ids, (size_t)n * sizeof(int32_t));
    gen->generate.token_count = n;
    gen->generate.params.max_tokens = max_tokens;
    gen->generate.params.sampling.temperature = 0.0f;
    gen->generate.params.sampling_set = SAMPLING_SET_TEMPERATURE;
    gen->generate.stream = s;
    engine_post(eng, gen);
    return s;
}

static int32_t *make_prompt(int n, int32_t fill) {
    int32_t *ids = malloc((size_t)n * sizeof(int32_t));
    assert(ids);
    for (int i = 0; i < n; i++)
        ids[i] = fill + (i & 7);
    return ids;
}

/* Cycle 2: short prompt fires prefill hook once. */
static void test_prefill_hook_fires_once_short_prompt(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);
    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    cancel_barrier_t b;
    barrier_init(&b);
    b.trip_at = 0; /* never block */
    engine_gen_hooks_t hooks = {
        .on_prefill_chunk = hook_prefill_count_only,
        .ud = &b,
    };
    engine_set_gen_hooks(&eng, &hooks);

    int32_t prompt[] = {1, 2, 3, 4}; /* prefill len 3 */
    stream_t *s = post_generate_prompt(&eng, prompt, 4, 1);

    finish_reason_t reason;
    int32_t got[8];
    int n = collect_tokens(s, got, 8, &reason);
    assert(n == 1);
    assert(reason == FINISH_LENGTH);

    assert(b.prefill_calls == 1);
    assert(b.last_pos == 0);
    assert(b.last_chunk_len == 3);

    stream_release(s);
    barrier_destroy(&b);
    engine_destroy(&eng);
}

/* Cycle 3: decode hook fires once per emitted step. */
static void test_decode_hook_fires_per_step(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);
    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    cancel_barrier_t b;
    barrier_init(&b);
    b.trip_at = 0;
    engine_gen_hooks_t hooks = {
        .on_decode_step = hook_decode_count_only,
        .ud = &b,
    };
    engine_set_gen_hooks(&eng, &hooks);

    int32_t prompt[] = {1, 2};
    int max_new = 4;
    stream_t *s = post_generate_prompt(&eng, prompt, 2, max_new);

    finish_reason_t reason;
    int32_t got[8];
    int n = collect_tokens(s, got, 8, &reason);
    assert(n == max_new);
    assert(reason == FINISH_LENGTH);
    assert(b.decode_calls == max_new);

    stream_release(s);
    barrier_destroy(&b);
    engine_destroy(&eng);
}

/* Cycle 4: cancel at decode step 0 within deadline. */
static void test_cancel_mid_decode_within_deadline(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);
    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    cancel_barrier_t b;
    barrier_init(&b);
    b.trip_at = 1; /* first decode step */
    engine_gen_hooks_t hooks = {
        .on_decode_step = hook_decode_barrier,
        .ud = &b,
    };
    engine_set_gen_hooks(&eng, &hooks);

    int32_t prompt[] = {1, 2};
    stream_t *s = post_generate_prompt(&eng, prompt, 2, 1000);

    barrier_wait_entered(&b);
    int64_t t0 = mono_ms();
    stream_cancel(s);
    barrier_release(&b);

    int64_t dt = wait_sole_owner_ms(s, t0, MLXD_TEST_DRAIN_BOUND_MS + 500);
    assert(dt < MLXD_TEST_DRAIN_BOUND_MS);
    assert(stream_sole_owner(s));
    assert(drain_saw_cancelled(s));

    stream_release(s);
    barrier_destroy(&b);
    engine_destroy(&eng);
}

/* Cycle 5: shutdown at decode step 0 within deadline. */
static void test_shutdown_mid_decode_within_deadline(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);
    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    cancel_barrier_t b;
    barrier_init(&b);
    b.trip_at = 1;
    engine_gen_hooks_t hooks = {
        .on_decode_step = hook_decode_barrier,
        .ud = &b,
    };
    engine_set_gen_hooks(&eng, &hooks);

    int32_t prompt[] = {1, 2};
    stream_t *s = post_generate_prompt(&eng, prompt, 2, 1000);

    barrier_wait_entered(&b);
    int64_t t0 = mono_ms();
    engine_signal_shutdown(&eng);
    barrier_release(&b);

    int64_t dt = wait_sole_owner_ms(s, t0, MLXD_TEST_DRAIN_BOUND_MS + 500);
    assert(dt < MLXD_TEST_DRAIN_BOUND_MS);
    assert(stream_sole_owner(s));
    assert(drain_saw_cancelled(s));

    stream_release(s);
    barrier_destroy(&b);
    engine_destroy(&eng);
}

/* Cycle 6: cancel mid-prefill within deadline; zero tokens emitted. */
static void test_cancel_mid_prefill_within_deadline(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);
    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    cancel_barrier_t b;
    barrier_init(&b);
    b.trip_at = 1; /* first prefill chunk */
    engine_gen_hooks_t hooks = {
        .on_prefill_chunk = hook_prefill_count,
        .ud = &b,
    };
    engine_set_gen_hooks(&eng, &hooks);

    int n_prompt = 64;
    int32_t *prompt = make_prompt(n_prompt, 1);
    stream_t *s = post_generate_prompt(&eng, prompt, n_prompt, 8);
    free(prompt);

    barrier_wait_entered(&b);
    int64_t t0 = mono_ms();
    stream_cancel(s);
    barrier_release(&b);

    int64_t dt = wait_sole_owner_ms(s, t0, MLXD_TEST_DRAIN_BOUND_MS + 500);
    assert(dt < MLXD_TEST_DRAIN_BOUND_MS);
    assert(stream_sole_owner(s));

    finish_reason_t reason = FINISH_STOP;
    int tokens = count_tokens_drained(s, &reason);
    assert(tokens == 0);
    assert(reason == FINISH_CANCELLED);

    stream_release(s);
    barrier_destroy(&b);
    engine_destroy(&eng);
}

/* Cycle 7: shutdown mid-prefill within deadline. */
static void test_shutdown_mid_prefill_within_deadline(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);
    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    cancel_barrier_t b;
    barrier_init(&b);
    b.trip_at = 1;
    engine_gen_hooks_t hooks = {
        .on_prefill_chunk = hook_prefill_count,
        .ud = &b,
    };
    engine_set_gen_hooks(&eng, &hooks);

    int n_prompt = 64;
    int32_t *prompt = make_prompt(n_prompt, 1);
    stream_t *s = post_generate_prompt(&eng, prompt, n_prompt, 8);
    free(prompt);

    barrier_wait_entered(&b);
    int64_t t0 = mono_ms();
    engine_signal_shutdown(&eng);
    barrier_release(&b);

    int64_t dt = wait_sole_owner_ms(s, t0, MLXD_TEST_DRAIN_BOUND_MS + 500);
    assert(dt < MLXD_TEST_DRAIN_BOUND_MS);
    assert(stream_sole_owner(s));

    finish_reason_t reason = FINISH_STOP;
    int tokens = count_tokens_drained(s, &reason);
    assert(tokens == 0);
    assert(reason == FINISH_CANCELLED);

    stream_release(s);
    barrier_destroy(&b);
    engine_destroy(&eng);
}

/* Cycle 8: cancel between prefill and seed forward. */
static void test_cancel_between_prefill_and_seed(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);
    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    cancel_barrier_t b;
    barrier_init(&b);
    b.trip_at = 1;
    engine_gen_hooks_t hooks = {
        .on_before_seed = hook_seed_barrier,
        .ud = &b,
    };
    engine_set_gen_hooks(&eng, &hooks);

    int n_prompt = 16;
    int32_t *prompt = make_prompt(n_prompt, 1);
    stream_t *s = post_generate_prompt(&eng, prompt, n_prompt, 8);
    free(prompt);

    barrier_wait_entered(&b);
    int64_t t0 = mono_ms();
    stream_cancel(s);
    barrier_release(&b);

    int64_t dt = wait_sole_owner_ms(s, t0, MLXD_TEST_DRAIN_BOUND_MS + 500);
    assert(dt < MLXD_TEST_DRAIN_BOUND_MS);
    assert(stream_sole_owner(s));

    finish_reason_t reason = FINISH_STOP;
    int tokens = count_tokens_drained(s, &reason);
    assert(tokens == 0);
    assert(reason == FINISH_CANCELLED);
    assert(b.seed_calls == 1);

    stream_release(s);
    barrier_destroy(&b);
    engine_destroy(&eng);
}

/* Cycle 9: multi-chunk prefill grain (max_pos=2048, chunk=512). */
static void test_prefill_multi_chunk_hook_positions(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);
    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    cancel_barrier_t b;
    barrier_init(&b);
    b.trip_at = 0; /* never block */
    engine_gen_hooks_t hooks = {
        .on_prefill_chunk = hook_prefill_count_only,
        .ud = &b,
    };
    engine_set_gen_hooks(&eng, &hooks);

    /* prefill len = CHUNK + 32; prompt = prefill + 1 seed token */
    int n_prompt = MLXD_PREFILL_CHUNK + 32 + 1;
    int32_t *prompt = make_prompt(n_prompt, 1);
    stream_t *s = post_generate_prompt(&eng, prompt, n_prompt, 1);
    free(prompt);

    finish_reason_t reason;
    int32_t got[4];
    int n = collect_tokens(s, got, 4, &reason);
    assert(n == 1);
    assert(reason == FINISH_LENGTH);

    assert(b.prefill_calls == 2);
    assert(b.prefill_pos[0] == 0);
    assert(b.prefill_len[0] == MLXD_PREFILL_CHUNK);
    assert(b.prefill_pos[1] == MLXD_PREFILL_CHUNK);
    assert(b.prefill_len[1] == 32);

    stream_release(s);
    barrier_destroy(&b);
    engine_destroy(&eng);
}

/* Cycle 9: cancel on second prefill chunk. */
static void test_cancel_after_first_prefill_chunk(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);
    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    cancel_barrier_t b;
    barrier_init(&b);
    b.trip_at = 2; /* second prefill chunk */
    engine_gen_hooks_t hooks = {
        .on_prefill_chunk = hook_prefill_count,
        .ud = &b,
    };
    engine_set_gen_hooks(&eng, &hooks);

    int n_prompt = MLXD_PREFILL_CHUNK + 32 + 1;
    int32_t *prompt = make_prompt(n_prompt, 1);
    stream_t *s = post_generate_prompt(&eng, prompt, n_prompt, 8);
    free(prompt);

    barrier_wait_entered(&b);
    assert(b.prefill_pos[0] == 0);
    assert(b.prefill_pos[1] == MLXD_PREFILL_CHUNK);

    int64_t t0 = mono_ms();
    stream_cancel(s);
    barrier_release(&b);

    int64_t dt = wait_sole_owner_ms(s, t0, MLXD_TEST_DRAIN_BOUND_MS + 500);
    assert(dt < MLXD_TEST_DRAIN_BOUND_MS);
    assert(stream_sole_owner(s));

    finish_reason_t reason = FINISH_STOP;
    int tokens = count_tokens_drained(s, &reason);
    assert(tokens == 0);
    assert(reason == FINISH_CANCELLED);

    stream_release(s);
    barrier_destroy(&b);
    engine_destroy(&eng);
}

/* Cycle 10 helper: release barrier shortly after destroy starts joining. */
typedef struct {
    cancel_barrier_t *b;
    int delay_ms;
} release_after_arg_t;

static void *release_after_fn(void *arg) {
    release_after_arg_t *a = arg;
    usleep((useconds_t)a->delay_ms * 1000);
    barrier_release(a->b);
    return NULL;
}

/* Cycle 10: engine_destroy joins within deadline while blocked in prefill. */
static void test_destroy_joins_within_deadline_mid_prefill(void) {
    engine_t eng;
    assert(engine_init(&eng) == 0);
    post_load(&eng, FIXTURES "/tiny_qwen3");
    assert(poll_load_terminal(&eng, 30000) == LOAD_OK);

    cancel_barrier_t b;
    barrier_init(&b);
    b.trip_at = 1;
    engine_gen_hooks_t hooks = {
        .on_prefill_chunk = hook_prefill_count,
        .ud = &b,
    };
    engine_set_gen_hooks(&eng, &hooks);

    int n_prompt = 64;
    int32_t *prompt = make_prompt(n_prompt, 1);
    stream_t *s = post_generate_prompt(&eng, prompt, n_prompt, 8);
    free(prompt);

    barrier_wait_entered(&b);

    /* Helper releases the hook barrier so join cannot deadlock. */
    release_after_arg_t ra = {.b = &b, .delay_ms = 20};
    pthread_t thr;
    assert(pthread_create(&thr, NULL, release_after_fn, &ra) == 0);

    int64_t t0 = mono_ms();
    engine_destroy(&eng);
    int64_t dt = mono_ms() - t0;
    assert(dt < MLXD_TEST_DRAIN_BOUND_MS);

    pthread_join(thr, NULL);

    assert(stream_sole_owner(s));
    finish_reason_t reason = FINISH_STOP;
    int tokens = count_tokens_drained(s, &reason);
    assert(tokens == 0);
    assert(reason == FINISH_CANCELLED);

    stream_release(s);
    barrier_destroy(&b);
}

int main(void) {
    test_load_tiny_qwen3_ok();
    printf("  test_load_tiny_qwen3_ok: passed\n");

    test_oversized_prompt();
    printf("  test_oversized_prompt: passed\n");

    test_greedy_argmax();
    printf("  test_greedy_argmax: passed\n");

    test_generate_deterministic();
    printf("  test_generate_deterministic: passed\n");

    test_generate_eos_finish_stop();
    printf("  test_generate_eos_finish_stop: passed\n");

    test_cancel_mid_generate_real();
    printf("  test_cancel_mid_generate_real: passed\n");

    test_shutdown_mid_generate_real();
    printf("  test_shutdown_mid_generate_real: passed\n");

    test_generate_seeded_sampling();
    printf("  test_generate_seeded_sampling: passed\n");

    test_generate_logprobs();
    printf("  test_generate_logprobs: passed\n");

    test_generate_genconfig_defaults();
    printf("  test_generate_genconfig_defaults: passed\n");

    test_generate_multi_eos();
    printf("  test_generate_multi_eos: passed\n");

    test_prefill_hook_fires_once_short_prompt();
    printf("  test_prefill_hook_fires_once_short_prompt: passed\n");

    test_decode_hook_fires_per_step();
    printf("  test_decode_hook_fires_per_step: passed\n");

    test_cancel_mid_decode_within_deadline();
    printf("  test_cancel_mid_decode_within_deadline: passed\n");

    test_shutdown_mid_decode_within_deadline();
    printf("  test_shutdown_mid_decode_within_deadline: passed\n");

    test_cancel_mid_prefill_within_deadline();
    printf("  test_cancel_mid_prefill_within_deadline: passed\n");

    test_shutdown_mid_prefill_within_deadline();
    printf("  test_shutdown_mid_prefill_within_deadline: passed\n");

    test_cancel_between_prefill_and_seed();
    printf("  test_cancel_between_prefill_and_seed: passed\n");

    test_prefill_multi_chunk_hook_positions();
    printf("  test_prefill_multi_chunk_hook_positions: passed\n");

    test_cancel_after_first_prefill_chunk();
    printf("  test_cancel_after_first_prefill_chunk: passed\n");

    test_destroy_joins_within_deadline_mid_prefill();
    printf("  test_destroy_joins_within_deadline_mid_prefill: passed\n");

    printf("test_engine_gpu: all passed\n");
    return 0;
}
