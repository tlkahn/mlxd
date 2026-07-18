#include "engine/engine.h"
#include "mlxbridge/mlxbridge.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef MLXD_FIXTURES_DIR
#error "MLXD_FIXTURES_DIR must be defined"
#endif

#define FIXTURES MLXD_FIXTURES_DIR

/* greedy_next_id is implemented in engine.c (non-static for GPU tests). */
int greedy_next_id(mlx_array logits, mlx_stream s, int32_t *id_out);

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
    engine_cmd_t *load = calloc(1, sizeof(*load));
    assert(load);
    load->tag = CMD_LOAD;
    load->load.model_path = strdup(path);
    assert(load->load.model_path);
    engine_post(eng, load);
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

    printf("  deterministic seq:");
    for (int i = 0; i < max_new; i++)
        printf(" %d", seq_a[i]);
    printf("\n");

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

int main(void) {
    test_load_tiny_qwen3_ok();
    printf("  test_load_tiny_qwen3_ok: passed\n");

    test_oversized_prompt();
    printf("  test_oversized_prompt: passed\n");

    test_greedy_argmax();
    printf("  test_greedy_argmax: passed\n");

    test_generate_deterministic();
    printf("  test_generate_deterministic: passed\n");

    test_cancel_mid_generate_real();
    printf("  test_cancel_mid_generate_real: passed\n");

    test_shutdown_mid_generate_real();
    printf("  test_shutdown_mid_generate_real: passed\n");

    printf("test_engine_gpu: all passed\n");
    return 0;
}
