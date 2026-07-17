#include "cli/io.h"
#include "core/types.h"
#include "engine/engine.h"
#include "model/detok.h"
#include "model/tokenizer.h"
#include "registry/registry.h"

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <yyjson/yyjson.h>

#ifndef MLXD_FIXTURES_DIR
#error "MLXD_FIXTURES_DIR must be defined"
#endif

/* --- Cycle 8: cli_run_messages_json --------------------------------------- */

static void test_messages_json_plain(void) {
    char *json = cli_run_messages_json("Hello world");
    assert(json != NULL);

    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(yyjson_is_arr(root));
    assert(yyjson_arr_size(root) == 1);

    yyjson_val *msg = yyjson_arr_get_first(root);
    assert(yyjson_is_obj(msg));
    assert(strcmp(yyjson_get_str(yyjson_obj_get(msg, "role")), "user") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(msg, "content")), "Hello world") == 0);

    yyjson_doc_free(doc);
    free(json);
}

static void test_messages_json_special_chars(void) {
    const char *prompt = "He said \"hello\"\nnew line\ttab \xC3\xA9";
    char *json = cli_run_messages_json(prompt);
    assert(json != NULL);

    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *msg = yyjson_arr_get_first(root);
    const char *content = yyjson_get_str(yyjson_obj_get(msg, "content"));
    assert(strcmp(content, prompt) == 0);

    yyjson_doc_free(doc);
    free(json);
}

/* --- Cycle 9: cli_human_size ---------------------------------------------- */

static void test_human_size_bytes(void) {
    char buf[32];
    cli_human_size(512, buf, sizeof(buf));
    assert(strcmp(buf, "512 B") == 0);
}

static void test_human_size_kb(void) {
    char buf[32];
    cli_human_size(1024, buf, sizeof(buf));
    assert(strstr(buf, "KB") != NULL);
}

static void test_human_size_mb(void) {
    char buf[32];
    cli_human_size(1024ULL * 1024, buf, sizeof(buf));
    assert(strstr(buf, "MB") != NULL);
}

static void test_human_size_gb(void) {
    char buf[32];
    cli_human_size(1024ULL * 1024 * 1024, buf, sizeof(buf));
    assert(strstr(buf, "GB") != NULL);
}

static void test_human_size_zero(void) {
    char buf[32];
    cli_human_size(0, buf, sizeof(buf));
    assert(strcmp(buf, "0 B") == 0);
}

/* --- Cycle 10: cli_list_json ---------------------------------------------- */

static void test_list_json_empty(void) {
    char *json = cli_list_json(NULL, 0);
    assert(json != NULL);
    assert(strcmp(json, "[]") == 0);
    free(json);
}

static void test_list_json_entries(void) {
    registry_model_info_t models[2] = {
        {.id = "model-a", .path = "/path/a", .size_bytes = 4096, .mtime = 1700000000},
        {.id = "model-b", .path = "/path/b", .size_bytes = 2048, .mtime = 1700001000},
    };
    char *json = cli_list_json(models, 2);
    assert(json != NULL);

    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(yyjson_is_arr(root));
    assert(yyjson_arr_size(root) == 2);

    yyjson_val *first = yyjson_arr_get_first(root);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(first, "id")), "model-a") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(first, "path")), "/path/a") == 0);
    assert(yyjson_get_uint(yyjson_obj_get(first, "size_bytes")) == 4096);
    assert(yyjson_get_sint(yyjson_obj_get(first, "mtime")) == 1700000000);

    yyjson_doc_free(doc);
    free(json);
}

/* --- Cycle 11: cli_run_consume happy path --------------------------------- */

static void test_run_consume_happy(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    const char *text = "hello world";
    int32_t *ids = NULL;
    int n = tokenizer_encode_alloc(tok, text, strlen(text), false, &ids);
    assert(n > 0);

    stream_t *s = stream_create(n + 1);
    assert(s != NULL);

    for (int i = 0; i < n; i++) {
        chunk_t c = {.tag = CHUNK_TOKEN, .token = {.id = ids[i], .logprob = 0.0f}};
        assert(stream_push(s, c));
    }
    chunk_t done = {.tag = CHUNK_DONE, .done = FINISH_STOP};
    assert(stream_push(s, done));

    char *membuf = NULL;
    size_t memlen = 0;
    FILE *out = open_memstream(&membuf, &memlen);
    assert(out != NULL);

    _Atomic int cancel = 0;
    finish_reason_t reason = FINISH_LENGTH;
    char err[256] = {0};
    int rc = cli_run_consume(s, tok, out, false, &cancel, &reason, err, sizeof(err));
    fclose(out);

    assert(rc == 0);
    assert(reason == FINISH_STOP);
    assert(memlen > 0);

    char *full = tokenizer_decode(tok, ids, n);
    assert(full != NULL);
    assert(strcmp(membuf, full) == 0);

    free(full);
    free(membuf);
    free(ids);
    stream_release(s);
    tokenizer_free(tok);
}

static void test_run_consume_flush_each(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    const char *text = "hello";
    int32_t *ids = NULL;
    int n = tokenizer_encode_alloc(tok, text, strlen(text), false, &ids);
    assert(n > 0);

    stream_t *s = stream_create(n + 1);
    for (int i = 0; i < n; i++) {
        chunk_t c = {.tag = CHUNK_TOKEN, .token = {.id = ids[i], .logprob = 0.0f}};
        stream_push(s, c);
    }
    chunk_t done = {.tag = CHUNK_DONE, .done = FINISH_STOP};
    stream_push(s, done);

    char *membuf = NULL;
    size_t memlen = 0;
    FILE *out = open_memstream(&membuf, &memlen);

    _Atomic int cancel = 0;
    finish_reason_t reason = FINISH_LENGTH;
    char err[256] = {0};
    int rc = cli_run_consume(s, tok, out, true, &cancel, &reason, err, sizeof(err));
    fclose(out);

    assert(rc == 0);
    assert(reason == FINISH_STOP);
    assert(memlen > 0);

    free(membuf);
    free(ids);
    stream_release(s);
    tokenizer_free(tok);
}

/* --- Cycle 12: cli_run_consume terminal cases ----------------------------- */

static void test_run_consume_error(void) {
    stream_t *s = stream_create(4);
    chunk_t err_chunk = {.tag = CHUNK_ERROR, .error = strdup("model exploded")};
    stream_push(s, err_chunk);

    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    char *membuf = NULL;
    size_t memlen = 0;
    FILE *out = open_memstream(&membuf, &memlen);

    _Atomic int cancel = 0;
    finish_reason_t reason = FINISH_STOP;
    char err[256] = {0};
    int rc = cli_run_consume(s, tok, out, false, &cancel, &reason, err, sizeof(err));
    fclose(out);

    assert(rc == -1);
    assert(strstr(err, "model exploded") != NULL);

    free(membuf);
    stream_release(s);
    tokenizer_free(tok);
}

static void test_run_consume_cancelled(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    stream_t *s = stream_create(4);
    chunk_t done = {.tag = CHUNK_DONE, .done = FINISH_CANCELLED};
    stream_push(s, done);

    char *membuf = NULL;
    size_t memlen = 0;
    FILE *out = open_memstream(&membuf, &memlen);

    _Atomic int cancel = 0;
    finish_reason_t reason = FINISH_STOP;
    char err[256] = {0};
    int rc = cli_run_consume(s, tok, out, false, &cancel, &reason, err, sizeof(err));
    fclose(out);

    assert(rc == 0);
    assert(reason == FINISH_CANCELLED);

    free(membuf);
    stream_release(s);
    tokenizer_free(tok);
}

static void test_run_consume_cancel_flag(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    stream_t *s = stream_create(4);
    chunk_t done = {.tag = CHUNK_DONE, .done = FINISH_CANCELLED};
    stream_push(s, done);

    char *membuf = NULL;
    size_t memlen = 0;
    FILE *out = open_memstream(&membuf, &memlen);

    _Atomic int cancel = 1;
    finish_reason_t reason = FINISH_STOP;
    char err[256] = {0};
    int rc = cli_run_consume(s, tok, out, false, &cancel, &reason, err, sizeof(err));
    fclose(out);

    assert(rc == 0);
    assert(reason == FINISH_CANCELLED);

    free(membuf);
    stream_release(s);
    tokenizer_free(tok);
}

/* --- Review fix: delayed producer exercises stream_next timeout path ----- */

typedef struct {
    stream_t *stream;
    int32_t *ids;
    int n_ids;
    int delay_ms;
} delayed_producer_ctx_t;

static void *delayed_producer(void *arg) {
    delayed_producer_ctx_t *ctx = arg;
    usleep((useconds_t)ctx->delay_ms * 1000);
    for (int i = 0; i < ctx->n_ids; i++) {
        chunk_t c = {.tag = CHUNK_TOKEN, .token = {.id = ctx->ids[i], .logprob = 0.0f}};
        stream_push(ctx->stream, c);
    }
    chunk_t done = {.tag = CHUNK_DONE, .done = FINISH_STOP};
    stream_push(ctx->stream, done);
    return NULL;
}

static void test_run_consume_delayed_producer(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    const char *text = "hi";
    int32_t *ids = NULL;
    int n = tokenizer_encode_alloc(tok, text, strlen(text), false, &ids);
    assert(n > 0);

    stream_t *s = stream_create(n + 1);
    assert(s != NULL);

    delayed_producer_ctx_t ctx = {
        .stream = s, .ids = ids, .n_ids = n, .delay_ms = 150
    };
    pthread_t thr;
    int prc = pthread_create(&thr, NULL, delayed_producer, &ctx);
    assert(prc == 0);

    char *membuf = NULL;
    size_t memlen = 0;
    FILE *out = open_memstream(&membuf, &memlen);
    assert(out != NULL);

    _Atomic int cancel = 0;
    finish_reason_t reason = FINISH_LENGTH;
    char err[256] = {0};
    int rc = cli_run_consume(s, tok, out, false, &cancel, &reason, err, sizeof(err));
    fclose(out);

    assert(rc == 0);
    assert(reason == FINISH_STOP);
    assert(memlen > 0);

    pthread_join(thr, NULL);

    free(membuf);
    free(ids);
    stream_release(s);
    tokenizer_free(tok);
}

/* --- cli_resolve_run_prompt ----------------------------------------------- */

static void test_resolve_run_prompt_positional_ignores_stdin(void) {
    char *result = cli_resolve_run_prompt("hello world", NULL);
    assert(result != NULL);
    assert(strcmp(result, "hello world") == 0);
    free(result);
}

static void test_resolve_run_prompt_empty_stdin_returns_null(void) {
    FILE *f = fopen("/dev/null", "r");
    assert(f != NULL);
    char *result = cli_resolve_run_prompt(NULL, f);
    assert(result == NULL);
    fclose(f);
}

static void test_resolve_run_prompt_reads_stdin_when_no_positional(void) {
    const char *input = "prompt from stdin";
    FILE *f = fmemopen((void *)input, strlen(input), "r");
    assert(f != NULL);
    char *result = cli_resolve_run_prompt(NULL, f);
    assert(result != NULL);
    assert(strcmp(result, "prompt from stdin") == 0);
    free(result);
    fclose(f);
}

static void test_resolve_run_prompt_null_stdin_returns_null(void) {
    char *result = cli_resolve_run_prompt(NULL, NULL);
    assert(result == NULL);
}

int main(void) {
    /* cli_resolve_run_prompt */
    test_resolve_run_prompt_positional_ignores_stdin();
    test_resolve_run_prompt_empty_stdin_returns_null();
    test_resolve_run_prompt_reads_stdin_when_no_positional();
    test_resolve_run_prompt_null_stdin_returns_null();

    /* cycle 8 */
    test_messages_json_plain();
    test_messages_json_special_chars();

    /* cycle 9 */
    test_human_size_bytes();
    test_human_size_kb();
    test_human_size_mb();
    test_human_size_gb();
    test_human_size_zero();

    /* cycle 10 */
    test_list_json_empty();
    test_list_json_entries();

    /* cycle 11 */
    test_run_consume_happy();
    test_run_consume_flush_each();

    /* cycle 12 */
    test_run_consume_error();
    test_run_consume_cancelled();
    test_run_consume_cancel_flag();

    /* review fix: delayed producer (exercises stream_next timeout path) */
    test_run_consume_delayed_producer();

    printf("test_cli_io: all passed\n");
    return 0;
}
