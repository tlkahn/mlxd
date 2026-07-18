#include "http/gen.h"
#include "core/openai.h"
#include "core/types.h"
#include "http/sse.h"
#include "model/tokenizer.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yyjson/yyjson.h>

#ifndef MLXD_FIXTURES_DIR
#error "MLXD_FIXTURES_DIR must be defined"
#endif

static const char *TRIVIAL_TMPL = "{{ messages[0].content }}";

#define THINK_COND_TMPL \
    "{{ messages[0].content }}" \
    "{% if enable_thinking is defined and enable_thinking is false %}" \
    "<think></think>" \
    "{% endif %}"

static void test_chat_prompt_matches_direct(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    const char *msgs = "[{\"role\":\"user\",\"content\":\"hello world\"}]";
    int32_t *ids = NULL;
    const char *err = NULL;
    int n = gen_build_chat_prompt(tok, TRIVIAL_TMPL, msgs, NULL, NULL, &ids, &err);
    assert(n > 0);
    assert(err == NULL);

    int32_t *direct_ids = NULL;
    int direct_n = tokenizer_encode_alloc(tok, "hello world", 11, false, &direct_ids);
    assert(direct_n == n);
    assert(memcmp(ids, direct_ids, (size_t)n * sizeof(int32_t)) == 0);

    free(ids);
    free(direct_ids);
    tokenizer_free(tok);
}

static void test_completion_prompt(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    const char *prompt = "Once upon a time";
    int32_t *ids = NULL;
    const char *err = NULL;
    int n = gen_build_completion_prompt(tok, prompt, &ids, &err);
    assert(n > 0);

    int32_t *direct_ids = NULL;
    int direct_n = tokenizer_encode_alloc(tok, prompt, strlen(prompt), false, &direct_ids);
    assert(direct_n == n);
    assert(memcmp(ids, direct_ids, (size_t)n * sizeof(int32_t)) == 0);

    free(ids);
    free(direct_ids);
    tokenizer_free(tok);
}

static void test_bad_template(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    const char *msgs = "[{\"role\":\"user\",\"content\":\"hi\"}]";
    int32_t *ids = NULL;
    const char *err = NULL;
    int n = gen_build_chat_prompt(tok, "{{ ", msgs, NULL, NULL, &ids, &err);
    assert(n == -1);
    assert(err != NULL);

    tokenizer_free(tok);
}

static yyjson_doc *parse_sse_payload(const char *sse) {
    assert(strncmp(sse, "data: ", 6) == 0);
    const char *json_start = sse + 6;
    const char *end = strstr(json_start, "\n\n");
    assert(end != NULL);
    size_t json_len = (size_t)(end - json_start);
    yyjson_doc *doc = yyjson_read(json_start, json_len, 0);
    assert(doc != NULL);
    return doc;
}

static void test_sse_chunk_role_first(void) {
    usage_t u = {.prompt_tokens = 5, .completion_tokens = 0, .total_tokens = 5};
    char *sse = gen_sse_chunk(&(gen_sse_chunk_params_t){
        .id = "id-1", .model = "gpt2", .created = 1234,
        .role_first = true, .usage = &u,
    });
    assert(sse != NULL);
    yyjson_doc *doc = parse_sse_payload(sse);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *choices = yyjson_obj_get(root, "choices");
    assert(yyjson_arr_size(choices) == 1);
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    yyjson_val *delta = yyjson_obj_get(c0, "delta");
    assert(strcmp(yyjson_get_str(yyjson_obj_get(delta, "role")), "assistant") == 0);
    yyjson_doc_free(doc);
    free(sse);
}

static void test_sse_chunk_content(void) {
    char *sse = gen_sse_chunk(&(gen_sse_chunk_params_t){
        .id = "id-1", .model = "gpt2", .created = 1234,
        .delta_text = "hello",
    });
    assert(sse != NULL);
    yyjson_doc *doc = parse_sse_payload(sse);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *choices = yyjson_obj_get(root, "choices");
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    yyjson_val *delta = yyjson_obj_get(c0, "delta");
    assert(strcmp(yyjson_get_str(yyjson_obj_get(delta, "content")), "hello") == 0);
    yyjson_doc_free(doc);
    free(sse);
}

static void test_sse_chunk_final(void) {
    char *sse = gen_sse_chunk(&(gen_sse_chunk_params_t){
        .id = "id-1", .model = "gpt2", .created = 1234,
        .final = true, .reason = FINISH_STOP,
    });
    assert(sse != NULL);
    yyjson_doc *doc = parse_sse_payload(sse);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *choices = yyjson_obj_get(root, "choices");
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    const char *fr = yyjson_get_str(yyjson_obj_get(c0, "finish_reason"));
    assert(strcmp(fr, finish_reason_wire_str(FINISH_STOP)) == 0);
    yyjson_doc_free(doc);
    free(sse);
}

static void test_sse_chunk_role_and_content_first(void) {
    char *sse = gen_sse_chunk(&(gen_sse_chunk_params_t){
        .id = "id-1", .model = "gpt2", .created = 1234,
        .role_first = true, .delta_text = "hello",
    });
    assert(sse != NULL);
    yyjson_doc *doc = parse_sse_payload(sse);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *choices = yyjson_obj_get(root, "choices");
    assert(yyjson_arr_size(choices) == 1);
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    yyjson_val *delta = yyjson_obj_get(c0, "delta");
    assert(strcmp(yyjson_get_str(yyjson_obj_get(delta, "role")), "assistant") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(delta, "content")), "hello") == 0);
    yyjson_doc_free(doc);
    free(sse);
}

static void test_sse_chunk_usage(void) {
    usage_t u = {.prompt_tokens = 10, .completion_tokens = 20, .total_tokens = 30};
    char *sse = gen_sse_chunk(&(gen_sse_chunk_params_t){
        .id = "id-1", .model = "gpt2", .created = 1234,
        .include_usage = true, .usage = &u,
    });
    assert(sse != NULL);
    yyjson_doc *doc = parse_sse_payload(sse);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *choices = yyjson_obj_get(root, "choices");
    assert(choices != NULL);
    assert(yyjson_arr_size(choices) == 0);
    yyjson_val *usage = yyjson_obj_get(root, "usage");
    assert(usage != NULL);
    assert(yyjson_get_sint(yyjson_obj_get(usage, "prompt_tokens")) == 10);
    assert(yyjson_get_sint(yyjson_obj_get(usage, "completion_tokens")) == 20);
    assert(yyjson_get_sint(yyjson_obj_get(usage, "total_tokens")) == 30);
    yyjson_doc_free(doc);
    free(sse);
}

static void test_chat_response_roundtrip(void) {
    usage_t u = {.prompt_tokens = 5, .completion_tokens = 3, .total_tokens = 8};
    char *json = gen_build_chat_response("chatcmpl-1", "gpt2", 1234, "hi there",
                                         FINISH_STOP, &u, NULL, 0);
    assert(json != NULL);
    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(root, "id")), "chatcmpl-1") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(root, "model")), "gpt2") == 0);
    yyjson_val *choices = yyjson_obj_get(root, "choices");
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    yyjson_val *msg = yyjson_obj_get(c0, "message");
    assert(strcmp(yyjson_get_str(yyjson_obj_get(msg, "content")), "hi there") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(c0, "finish_reason")),
                 finish_reason_wire_str(FINISH_STOP)) == 0);
    yyjson_val *usage = yyjson_obj_get(root, "usage");
    assert(yyjson_get_sint(yyjson_obj_get(usage, "total_tokens")) == 8);
    yyjson_doc_free(doc);
    free(json);
}

static void test_chat_response_null_usage(void) {
    char *json = gen_build_chat_response("chatcmpl-1", "gpt2", 1234, "hi",
                                          FINISH_STOP, NULL, NULL, 0);
    assert(json != NULL);
    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *usage = yyjson_obj_get(root, "usage");
    assert(usage != NULL);
    assert(yyjson_get_sint(yyjson_obj_get(usage, "prompt_tokens")) == 0);
    assert(yyjson_get_sint(yyjson_obj_get(usage, "completion_tokens")) == 0);
    assert(yyjson_get_sint(yyjson_obj_get(usage, "total_tokens")) == 0);
    yyjson_doc_free(doc);
    free(json);
}

static void test_completion_response_null_usage(void) {
    char *json = gen_build_completion_response("cmpl-1", "gpt2", 1234, "x",
                                                FINISH_STOP, NULL);
    assert(json != NULL);
    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *usage = yyjson_obj_get(root, "usage");
    assert(usage != NULL);
    assert(yyjson_get_sint(yyjson_obj_get(usage, "total_tokens")) == 0);
    yyjson_doc_free(doc);
    free(json);
}

static void test_sse_chunk_null_usage_omits(void) {
    char *sse = gen_sse_chunk(&(gen_sse_chunk_params_t){
        .id = "id-1", .model = "gpt2", .created = 1234,
        .include_usage = true,
    });
    assert(sse != NULL);
    yyjson_doc *doc = parse_sse_payload(sse);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *usage = yyjson_obj_get(root, "usage");
    assert(usage == NULL);
    yyjson_doc_free(doc);
    free(sse);
}

static void test_chat_prompt_bad_template_nulls_ids(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    const char *msgs = "[{\"role\":\"user\",\"content\":\"hi\"}]";
    int32_t *ids = (int32_t *)(uintptr_t)0xDEAD;
    const char *err = NULL;
    int n = gen_build_chat_prompt(tok, "{{ ", msgs, NULL, NULL, &ids, &err);
    assert(n == -1);
    assert(err != NULL);
    assert(ids == NULL);

    tokenizer_free(tok);
}

static void test_chat_response_null_content(void) {
    usage_t u = {.prompt_tokens = 5, .completion_tokens = 0, .total_tokens = 5};
    char *json = gen_build_chat_response("chatcmpl-1", "gpt2", 1234, NULL,
                                          FINISH_STOP, &u, NULL, 0);
    assert(json != NULL);
    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *choices = yyjson_obj_get(root, "choices");
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    yyjson_val *msg = yyjson_obj_get(c0, "message");
    yyjson_val *content = yyjson_obj_get(msg, "content");
    assert(content != NULL);
    assert(yyjson_is_null(content));
    yyjson_doc_free(doc);
    free(json);
}

static void test_chat_prompt_special_tokens(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    const char *tmpl = "<|endoftext|>{{ messages[0].content }}";
    const char *msgs = "[{\"role\":\"user\",\"content\":\"hi\"}]";
    int32_t *ids = NULL;
    const char *err = NULL;
    int n = gen_build_chat_prompt(tok, tmpl, msgs, NULL, NULL, &ids, &err);
    assert(n > 0);
    assert(err == NULL);
    assert(ids[0] == 50256);

    free(ids);
    tokenizer_free(tok);
}

static void test_chat_prompt_user_content_specials(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    const char *msgs = "[{\"role\":\"user\",\"content\":\"<|endoftext|>\"}]";
    int32_t *ids = NULL;
    const char *err = NULL;
    int n = gen_build_chat_prompt(tok, TRIVIAL_TMPL, msgs, NULL, NULL, &ids, &err);
    assert(n > 0);
    assert(err == NULL);
    assert(ids[0] == 50256);

    free(ids);
    tokenizer_free(tok);
}

static void test_make_id_format(void) {
    char *id1 = gen_make_id("chatcmpl-");
    assert(id1 != NULL);
    assert(strncmp(id1, "chatcmpl-", 9) == 0);
    assert(strlen(id1) == 9 + 24);
    for (int i = 9; i < 33; i++) {
        char c = id1[i];
        assert((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    }

    char *id2 = gen_make_id("cmpl-");
    assert(id2 != NULL);
    assert(strncmp(id2, "cmpl-", 5) == 0);
    assert(strlen(id2) == 5 + 24);

    assert(strcmp(id1, id2) != 0);

    free(id1);
    free(id2);
}

static void test_sse_completion_chunk_delta(void) {
    char *sse = gen_sse_completion_chunk(&(gen_sse_completion_chunk_params_t){
        .id = "cmpl-1", .model = "gpt2", .created = 1234,
        .delta_text = "hello",
    });
    assert(sse != NULL);
    yyjson_doc *doc = parse_sse_payload(sse);
    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(root, "object")), "text_completion") == 0);
    yyjson_val *choices = yyjson_obj_get(root, "choices");
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(c0, "text")), "hello") == 0);
    assert(yyjson_get_sint(yyjson_obj_get(c0, "index")) == 0);
    assert(yyjson_is_null(yyjson_obj_get(c0, "logprobs")));
    assert(yyjson_is_null(yyjson_obj_get(c0, "finish_reason")));
    yyjson_doc_free(doc);
    free(sse);
}

static void test_sse_completion_chunk_final(void) {
    char *sse = gen_sse_completion_chunk(&(gen_sse_completion_chunk_params_t){
        .id = "cmpl-1", .model = "gpt2", .created = 1234,
        .final = true, .reason = FINISH_LENGTH,
    });
    assert(sse != NULL);
    yyjson_doc *doc = parse_sse_payload(sse);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *choices = yyjson_obj_get(root, "choices");
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    const char *fr = yyjson_get_str(yyjson_obj_get(c0, "finish_reason"));
    assert(strcmp(fr, "length") == 0);
    yyjson_doc_free(doc);
    free(sse);
}

static void test_completion_response_roundtrip(void) {
    usage_t u = {.prompt_tokens = 4, .completion_tokens = 6, .total_tokens = 10};
    char *json = gen_build_completion_response("cmpl-1", "gpt2", 1234, "once upon",
                                               FINISH_LENGTH, &u);
    assert(json != NULL);
    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(root, "id")), "cmpl-1") == 0);
    yyjson_val *choices = yyjson_obj_get(root, "choices");
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(c0, "text")), "once upon") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(c0, "finish_reason")),
                 finish_reason_wire_str(FINISH_LENGTH)) == 0);
    yyjson_doc_free(doc);
    free(json);
}

static void test_sse_chunk_with_logprob(void) {
    token_logprob_t entry = {
        .token = "hi", .logprob = -0.25f,
        .top_logprobs = NULL, .top_logprob_count = 0,
    };
    char *sse = gen_sse_chunk(&(gen_sse_chunk_params_t){
        .id = "id-lp", .model = "gpt2", .created = 1234,
        .delta_text = "hi",
        .logprob = &entry,
    });
    assert(sse != NULL);
    yyjson_doc *doc = parse_sse_payload(sse);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *c0 = yyjson_arr_get(yyjson_obj_get(root, "choices"), 0);
    yyjson_val *lp = yyjson_obj_get(c0, "logprobs");
    assert(lp != NULL && !yyjson_is_null(lp));
    yyjson_val *cont = yyjson_obj_get(lp, "content");
    assert(yyjson_arr_size(cont) == 1);
    yyjson_val *e0 = yyjson_arr_get(cont, 0);
    assert(fabs(yyjson_get_real(yyjson_obj_get(e0, "logprob")) - (-0.25)) < 1e-6);
    yyjson_doc_free(doc);
    free(sse);
}

static void test_sse_chunk_no_logprob_null(void) {
    char *sse = gen_sse_chunk(&(gen_sse_chunk_params_t){
        .id = "id-nl", .model = "gpt2", .created = 1234,
        .delta_text = "hi",
    });
    assert(sse != NULL);
    yyjson_doc *doc = parse_sse_payload(sse);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *c0 = yyjson_arr_get(yyjson_obj_get(root, "choices"), 0);
    yyjson_val *lp = yyjson_obj_get(c0, "logprobs");
    assert(lp != NULL && yyjson_is_null(lp));
    yyjson_doc_free(doc);
    free(sse);
}

static void test_chat_response_with_logprobs(void) {
    token_logprob_t entries[1] = {{
        .token = "ok", .logprob = -0.1f,
        .top_logprobs = NULL, .top_logprob_count = 0,
    }};
    usage_t u = {.prompt_tokens = 1, .completion_tokens = 1, .total_tokens = 2};
    char *json = gen_build_chat_response("chatcmpl-1", "gpt2", 1234, "ok",
                                          FINISH_STOP, &u, entries, 1);
    assert(json != NULL);
    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *c0 = yyjson_arr_get(yyjson_obj_get(root, "choices"), 0);
    yyjson_val *lp = yyjson_obj_get(c0, "logprobs");
    assert(lp != NULL && !yyjson_is_null(lp));
    yyjson_val *cont = yyjson_obj_get(lp, "content");
    assert(yyjson_arr_size(cont) == 1);
    assert(fabs(yyjson_get_real(yyjson_obj_get(yyjson_arr_get(cont, 0), "logprob")) - (-0.1)) < 1e-6);
    yyjson_doc_free(doc);
    free(json);
}

static void test_chat_response_no_logprobs_null(void) {
    usage_t u = {.prompt_tokens = 1, .completion_tokens = 1, .total_tokens = 2};
    char *json = gen_build_chat_response("chatcmpl-1", "gpt2", 1234, "ok",
                                          FINISH_STOP, &u, NULL, 0);
    assert(json != NULL);
    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *c0 = yyjson_arr_get(yyjson_obj_get(root, "choices"), 0);
    yyjson_val *lp = yyjson_obj_get(c0, "logprobs");
    assert(lp != NULL && yyjson_is_null(lp));
    yyjson_doc_free(doc);
    free(json);
}

static void test_sse_error_format(void) {
    char *sse = gen_sse_error("something went wrong");
    assert(sse != NULL);
    yyjson_doc *doc = parse_sse_payload(sse);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *err = yyjson_obj_get(root, "error");
    assert(err != NULL);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(err, "message")), "something went wrong") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(err, "type")), "server_error") == 0);
    yyjson_doc_free(doc);
    free(sse);
}

static void test_chat_prompt_extra_json(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    const char *msgs = "[{\"role\":\"user\",\"content\":\"hello world\"}]";

    int32_t *ids = NULL;
    const char *err = NULL;
    int n = gen_build_chat_prompt(tok, THINK_COND_TMPL, msgs, NULL,
                                  "{\"enable_thinking\":false}", &ids, &err);
    assert(n > 0);
    assert(err == NULL);

    int32_t *expected = NULL;
    int expected_n = tokenizer_encode_alloc(tok, "hello world<think></think>",
                                             strlen("hello world<think></think>"),
                                             false, &expected);
    assert(expected_n == n);
    assert(memcmp(ids, expected, (size_t)n * sizeof(int32_t)) == 0);
    free(ids);
    free(expected);

    ids = NULL;
    err = NULL;
    n = gen_build_chat_prompt(tok, THINK_COND_TMPL, msgs, NULL, NULL, &ids, &err);
    assert(n > 0);
    assert(err == NULL);

    expected = NULL;
    expected_n = tokenizer_encode_alloc(tok, "hello world",
                                         strlen("hello world"),
                                         false, &expected);
    assert(expected_n == n);
    assert(memcmp(ids, expected, (size_t)n * sizeof(int32_t)) == 0);
    free(ids);
    free(expected);

    tokenizer_free(tok);
}

int main(void) {
    test_chat_prompt_matches_direct();
    test_completion_prompt();
    test_bad_template();
    test_sse_chunk_role_first();
    test_sse_chunk_content();
    test_sse_chunk_final();
    test_sse_chunk_role_and_content_first();
    test_sse_chunk_usage();
    test_chat_response_roundtrip();
    test_chat_response_null_usage();
    test_completion_response_null_usage();
    test_sse_chunk_null_usage_omits();
    test_chat_response_null_content();
    test_chat_prompt_bad_template_nulls_ids();
    test_chat_prompt_special_tokens();
    test_chat_prompt_user_content_specials();
    test_make_id_format();
    test_sse_completion_chunk_delta();
    test_sse_completion_chunk_final();
    test_completion_response_roundtrip();
    test_sse_chunk_with_logprob();
    test_sse_chunk_no_logprob_null();
    test_chat_response_with_logprobs();
    test_chat_response_no_logprobs_null();
    test_sse_error_format();
    test_chat_prompt_extra_json();
    printf("test_http_gen: all passed\n");
    return 0;
}
