#include "http/gen.h"
#include "core/openai.h"
#include "core/types.h"
#include "http/sse.h"
#include "model/tokenizer.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yyjson/yyjson.h>

#ifndef MLXD_FIXTURES_DIR
#error "MLXD_FIXTURES_DIR must be defined"
#endif

static const char *TRIVIAL_TMPL = "{{ messages[0].content }}";

static void test_chat_prompt_matches_direct(void) {
    tokenizer_t *tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(tok != NULL);

    const char *msgs = "[{\"role\":\"user\",\"content\":\"hello world\"}]";
    int32_t *ids = NULL;
    const char *err = NULL;
    int n = gen_build_chat_prompt(tok, TRIVIAL_TMPL, msgs, NULL, &ids, &err);
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
    int n = gen_build_chat_prompt(tok, "{{ ", msgs, NULL, &ids, &err);
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
    char *sse = gen_sse_chunk("id-1", "gpt2", 1234, true, NULL, false,
                              FINISH_STOP, false, &u);
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
    usage_t u = {0};
    char *sse = gen_sse_chunk("id-1", "gpt2", 1234, false, "hello", false,
                              FINISH_STOP, false, &u);
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
    usage_t u = {0};
    char *sse = gen_sse_chunk("id-1", "gpt2", 1234, false, NULL, true,
                              FINISH_STOP, false, &u);
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

static void test_sse_chunk_usage(void) {
    usage_t u = {.prompt_tokens = 10, .completion_tokens = 20, .total_tokens = 30};
    char *sse = gen_sse_chunk("id-1", "gpt2", 1234, false, NULL, false,
                              FINISH_STOP, true, &u);
    assert(sse != NULL);
    yyjson_doc *doc = parse_sse_payload(sse);
    yyjson_val *root = yyjson_doc_get_root(doc);
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
                                         FINISH_STOP, &u);
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

int main(void) {
    test_chat_prompt_matches_direct();
    test_completion_prompt();
    test_bad_template();
    test_sse_chunk_role_first();
    test_sse_chunk_content();
    test_sse_chunk_final();
    test_sse_chunk_usage();
    test_chat_response_roundtrip();
    test_completion_response_roundtrip();
    printf("test_http_gen: all passed\n");
    return 0;
}
