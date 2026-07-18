#include "engine/engine.h"
#include "engine/engine_internal.h"
#include "engine/emodel.h"
#include "mlxbridge/mlxbridge.h"

#include <assert.h>
#include <math.h>
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

    int n = 513;
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
    assert(strstr(out.error, "513") != NULL);
    assert(strstr(out.error, "512") != NULL);
    free(out.error);

    assert(stream_next(s, &out, 5000));
    assert(out.tag == CHUNK_DONE);

    stream_release(s);
    engine_destroy(&eng);
}

/* ---- B3.8: greedy argmax ------------------------------------------------ */

static void test_greedy_argmax(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    /* logits [1, 8] with unique max at index 5 */
    float data[8] = {0.1f, -1.0f, 0.5f, 0.2f, -0.3f, 2.5f, 1.0f, 0.0f};
    int shape[] = {1, 8};
    mlx_array logits_f32 = mlx_array_new_data(data, shape, 2, MLX_FLOAT32);
    mlx_array logits = mlx_array_new();
    assert(MLXB_CHECK(mlx_astype(&logits, logits_f32, MLX_BFLOAT16, s)));

    int32_t id = -1;
    assert(greedy_next_id(logits, s, &id) == 0);
    assert(id == 5);

    mlx_array_free(logits);
    mlx_array_free(logits_f32);
}

/* ---- B3.9/B3.10: deterministic generate --------------------------------- */

static int collect_tokens(stream_t *s, int32_t *out_ids, int max_ids,
                          finish_reason_t *reason_out) {
    int n = 0;
    chunk_t c;
    *reason_out = FINISH_STOP;
    while (stream_next(s, &c, 30000)) {
        if (c.tag == CHUNK_TOKEN) {
            if (n < max_ids)
                out_ids[n] = c.token.id;
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

static int collect_tokens_lp(stream_t *s, int32_t *out_ids, float *out_lps,
                             int max_ids, finish_reason_t *reason_out) {
    int n = 0;
    chunk_t c;
    *reason_out = FINISH_STOP;
    while (stream_next(s, &c, 30000)) {
        if (c.tag == CHUNK_TOKEN) {
            if (n < max_ids) {
                out_ids[n] = c.token.id;
                out_lps[n] = c.token.logprob;
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

    printf("test_engine_gpu: all passed\n");
    return 0;
}
