#include "core/openai.h"
#include "core/types.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yyjson/yyjson.h>

/* Fixtures under tests/fixtures/ are authored from the official OpenAI API
 * reference shapes and are byte-identical to the mlxd-zig sibling. Path is
 * injected by the Makefile via -DMLXD_FIXTURES_DIR. */

#ifndef MLXD_FIXTURES_DIR
#error "MLXD_FIXTURES_DIR must be defined"
#endif

/* Read a fixture file whole. Caller owns the returned NUL-terminated buffer. */
static char *read_fixture(const char *name) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", MLXD_FIXTURES_DIR, name);
    FILE *f = fopen(path, "rb");
    assert(f != NULL);
    assert(fseek(f, 0, SEEK_END) == 0);
    long len = ftell(f);
    assert(len >= 0);
    assert(fseek(f, 0, SEEK_SET) == 0);
    char *buf = malloc((size_t)len + 1);
    assert(buf != NULL);
    assert(fread(buf, 1, (size_t)len, f) == (size_t)len);
    buf[len] = '\0';
    fclose(f);
    return buf;
}

/* Read a fixture and parse it into a yyjson_doc. Caller owns the doc and the
 * buffer; free the buffer only after done reading the doc's strings (doc keeps
 * its own copy on YYJSON_READ_NOFLAG). Returns doc; frees buffer internally. */
static yyjson_doc *parse_fixture(const char *name) {
    char *buf = read_fixture(name);
    yyjson_doc *doc = yyjson_read(buf, strlen(buf), 0);
    free(buf);
    assert(doc != NULL);
    return doc;
}

/* --- Cycle 0: helpers read and parse the error envelope fixture ----------- */

static void test_helper_reads_and_parses_error_envelope(void) {
    yyjson_doc *doc = parse_fixture("error_envelope.json");
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *error = yyjson_obj_get(root, "error");
    assert(error != NULL);
    const char *msg = yyjson_get_str(yyjson_obj_get(error, "message"));
    assert(msg != NULL);
    assert(strcmp(msg, "The model 'no-such-model' does not exist") == 0);
    yyjson_doc_free(doc);
}

/* Serialize a mut_val as the root of a fresh doc, write it, reparse it, and
 * hand back the reparsed doc plus the JSON text. Caller frees both. */
static yyjson_doc *roundtrip(yyjson_mut_doc *mdoc, yyjson_mut_val *root, char **json_out) {
    assert(root != NULL);
    yyjson_mut_doc_set_root(mdoc, root);
    char *json = yyjson_mut_write(mdoc, 0, NULL);
    assert(json != NULL);
    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    assert(doc != NULL);
    if (json_out)
        *json_out = json;
    else
        free(json);
    return doc;
}

/* --- Cycle 1: error envelope serialize ------------------------------------ */

static void test_error_envelope_serialize(void) {
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root =
        error_envelope_serialize("bad thing", "invalid_request_error", "model_not_found", mdoc);
    yyjson_doc *doc = roundtrip(mdoc, root, NULL);

    yyjson_val *error = yyjson_obj_get(yyjson_doc_get_root(doc), "error");
    assert(error != NULL);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(error, "message")), "bad thing") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(error, "type")), "invalid_request_error") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(error, "code")), "model_not_found") == 0);
    /* Signature carries no param field; it must not be emitted. */
    assert(yyjson_obj_get(error, "param") == NULL);

    yyjson_doc_free(doc);
    yyjson_mut_doc_free(mdoc);
}

/* --- Cycle 2: model list serialize ---------------------------------------- */

static void test_model_list_serialize(void) {
    model_info_t models[2] = {
        {.id = "qwen3-4b", .created = 1686935002, .owned_by = "mlxd"},
        {.id = "gemma4-12b", .created = 1705953180, .owned_by = "mlxd"},
    };
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    yyjson_doc *doc = roundtrip(mdoc, model_list_serialize(models, 2, mdoc), NULL);

    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(root, "object")), "list") == 0);
    yyjson_val *data = yyjson_obj_get(root, "data");
    assert(yyjson_arr_size(data) == 2);
    yyjson_val *m0 = yyjson_arr_get(data, 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(m0, "id")), "qwen3-4b") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(m0, "object")), "model") == 0);
    assert(yyjson_get_sint(yyjson_obj_get(m0, "created")) == 1686935002);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(m0, "owned_by")), "mlxd") == 0);
    yyjson_val *m1 = yyjson_arr_get(data, 1);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(m1, "id")), "gemma4-12b") == 0);

    yyjson_doc_free(doc);
    yyjson_mut_doc_free(mdoc);
}

/* --- Cycle 3: completion request parse (string prompt) -------------------- */

static void test_completion_request_parse_string(void) {
    yyjson_doc *doc = parse_fixture("completion_req.json");
    completion_request_t req;
    const char *err = NULL;
    assert(completion_request_parse(&req, yyjson_doc_get_root(doc), &err) == 0);
    assert(strcmp(req.model, "qwen3-4b") == 0);
    assert(strcmp(req.prompt, "Say this is a test") == 0);
    assert(req.params.max_tokens == 7);
    assert(req.params.sampling.temperature == 0.0f);
    assert(req.params.stop_count == 1);
    assert(strcmp(req.params.stop[0], "\n") == 0);
    completion_request_free(&req);
    yyjson_doc_free(doc);
}

/* --- Cycle 4: completion response round-trip ------------------------------ */

static void test_completion_response_roundtrip(void) {
    completion_response_t resp = {
        .id = "cmpl-abc123",
        .model = "qwen3-4b",
        .created = 1728933352,
        .finish_reason = FINISH_LENGTH,
        .text = "\n\nThis is indeed a test",
        .usage = {.prompt_tokens = 5, .completion_tokens = 7, .total_tokens = 12},
    };
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    yyjson_doc *doc = roundtrip(mdoc, completion_response_serialize(&resp, mdoc), NULL);

    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(root, "id")), "cmpl-abc123") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(root, "object")), "text_completion") == 0);
    yyjson_val *choices = yyjson_obj_get(root, "choices");
    assert(yyjson_arr_size(choices) == 1);
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(c0, "text")), "\n\nThis is indeed a test") == 0);
    assert(yyjson_get_sint(yyjson_obj_get(c0, "index")) == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(c0, "finish_reason")), "length") == 0);
    yyjson_val *usage = yyjson_obj_get(root, "usage");
    assert(yyjson_get_sint(yyjson_obj_get(usage, "total_tokens")) == 12);

    yyjson_doc_free(doc);
    yyjson_mut_doc_free(mdoc);
}

/* --- Cycle 5: chat request parse (string content) ------------------------- */

static void test_chat_request_parse_string(void) {
    yyjson_doc *doc = parse_fixture("chat_req_string.json");
    chat_completion_request_t req;
    const char *err = NULL;
    assert(chat_completion_request_parse(&req, yyjson_doc_get_root(doc), &err) == 0);
    assert(strcmp(req.model, "qwen3-4b") == 0);
    assert(req.message_count == 2);
    assert(req.messages[0].role == ROLE_SYSTEM);
    assert(req.messages[0].content.kind == CONTENT_STRING);
    assert(strcmp(req.messages[0].content.string, "You are a helpful assistant.") == 0);
    assert(req.messages[1].role == ROLE_USER);
    assert(req.messages[1].content.kind == CONTENT_STRING);
    assert(strcmp(req.messages[1].content.string, "Hello!") == 0);
    assert(req.params.sampling.temperature == 0.7f);
    assert(req.params.max_tokens == 256);
    assert(req.params.sampling.seed == -1);
    assert(req.params.stream == true);
    assert(req.include_usage == true);
    assert(req.tool_count == 0);
    chat_completion_request_free(&req);
    yyjson_doc_free(doc);
}

/* --- Cycle 6: chat request parse (content parts, tools, tool_choice obj) --- */

static void test_chat_request_parse_parts_tools(void) {
    yyjson_doc *doc = parse_fixture("chat_req_parts_tools.json");
    chat_completion_request_t req;
    const char *err = NULL;
    assert(chat_completion_request_parse(&req, yyjson_doc_get_root(doc), &err) == 0);
    assert(req.message_count == 1);
    message_content_t *c = &req.messages[0].content;
    assert(c->kind == CONTENT_PARTS);
    assert(c->part_count == 2);
    assert(strcmp(c->parts[0].type, "text") == 0);
    assert(strcmp(c->parts[0].text, "What is in this image?") == 0);
    assert(strcmp(c->parts[1].type, "image_url") == 0);
    assert(c->parts[1].has_image_url);
    assert(strcmp(c->parts[1].image_url.url, "https://example.com/cat.png") == 0);

    assert(req.tool_count == 1);
    assert(strcmp(req.tools[0].type, "function") == 0);
    assert(strcmp(req.tools[0].function.name, "get_weather") == 0);
    assert(req.tools[0].function.parameters_json != NULL);
    /* parameters_json is the raw JSON object text produced by yyjson_val_write. */
    yyjson_doc *pdoc = yyjson_read(req.tools[0].function.parameters_json,
                                   strlen(req.tools[0].function.parameters_json), 0);
    assert(pdoc != NULL);
    assert(yyjson_is_obj(yyjson_doc_get_root(pdoc)));
    yyjson_doc_free(pdoc);

    assert(req.tool_choice.kind == TOOL_CHOICE_FUNCTION);
    assert(strcmp(req.tool_choice.function_name, "get_weather") == 0);
    chat_completion_request_free(&req);
    yyjson_doc_free(doc);
}

/* --- Cycle 7: chat request parse (tool_choice string, stop array) --------- */

static void test_chat_request_parse_toolchoice_string(void) {
    yyjson_doc *doc = parse_fixture("chat_req_toolchoice_string.json");
    chat_completion_request_t req;
    const char *err = NULL;
    assert(chat_completion_request_parse(&req, yyjson_doc_get_root(doc), &err) == 0);
    assert(req.tool_choice.kind == TOOL_CHOICE_REQUIRED);
    assert(req.tool_choice.function_name == NULL);
    assert(req.params.stop_count == 2);
    assert(strcmp(req.params.stop[0], "\n\n") == 0);
    assert(strcmp(req.params.stop[1], "END") == 0);
    chat_completion_request_free(&req);
    yyjson_doc_free(doc);
}

/* --- Cycle 8: chat response round-trip (basic) ---------------------------- */

static void test_chat_response_roundtrip_basic(void) {
    chat_completion_response_t resp = {
        .id = "chatcmpl-basic",
        .model = "qwen3-4b",
        .created = 1728933352,
        .finish_reason = FINISH_STOP,
        .content = "Hi there",
        .usage = {.prompt_tokens = 9, .completion_tokens = 2, .total_tokens = 11},
    };
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    yyjson_doc *doc = roundtrip(mdoc, chat_completion_response_serialize(&resp, mdoc), NULL);

    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(root, "object")), "chat.completion") == 0);
    yyjson_val *choices = yyjson_obj_get(root, "choices");
    assert(yyjson_arr_size(choices) == 1);
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    assert(yyjson_get_sint(yyjson_obj_get(c0, "index")) == 0);
    yyjson_val *msg = yyjson_obj_get(c0, "message");
    assert(strcmp(yyjson_get_str(yyjson_obj_get(msg, "role")), "assistant") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(msg, "content")), "Hi there") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(c0, "finish_reason")), "stop") == 0);
    yyjson_val *usage = yyjson_obj_get(root, "usage");
    assert(yyjson_get_sint(yyjson_obj_get(usage, "prompt_tokens")) == 9);
    assert(yyjson_get_sint(yyjson_obj_get(usage, "completion_tokens")) == 2);
    assert(yyjson_get_sint(yyjson_obj_get(usage, "total_tokens")) == 11);

    yyjson_doc_free(doc);
    yyjson_mut_doc_free(mdoc);
}

/* --- Cycle 9: chat response round-trip (tool calls) ----------------------- */

static void test_chat_response_roundtrip_tool_calls(void) {
    tool_call_t calls[1] = {{
        .id = "call_abc123",
        .function_name = "get_weather",
        .arguments = "{\"location\": \"San Francisco, CA\", \"unit\": \"celsius\"}",
    }};
    chat_completion_response_t resp = {
        .id = "chatcmpl-abc123",
        .model = "qwen3-4b",
        .created = 1728933352,
        .finish_reason = FINISH_TOOL_CALLS,
        .content = NULL,
        .tool_calls = calls,
        .tool_call_count = 1,
        .usage = {.prompt_tokens = 82, .completion_tokens = 17, .total_tokens = 99},
    };
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    yyjson_doc *doc = roundtrip(mdoc, chat_completion_response_serialize(&resp, mdoc), NULL);

    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *c0 = yyjson_arr_get(yyjson_obj_get(root, "choices"), 0);
    yyjson_val *msg = yyjson_obj_get(c0, "message");
    yyjson_val *content = yyjson_obj_get(msg, "content");
    assert(content == NULL || yyjson_is_null(content));
    yyjson_val *tcs = yyjson_obj_get(msg, "tool_calls");
    assert(yyjson_arr_size(tcs) == 1);
    yyjson_val *tc0 = yyjson_arr_get(tcs, 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(tc0, "id")), "call_abc123") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(tc0, "type")), "function") == 0);
    yyjson_val *fn = yyjson_obj_get(tc0, "function");
    assert(strcmp(yyjson_get_str(yyjson_obj_get(fn, "name")), "get_weather") == 0);
    /* arguments survives double-encoding byte-identically through the round-trip. */
    assert(strcmp(yyjson_get_str(yyjson_obj_get(fn, "arguments")), calls[0].arguments) == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(c0, "finish_reason")), "tool_calls") == 0);
    yyjson_val *usage = yyjson_obj_get(root, "usage");
    assert(yyjson_get_sint(yyjson_obj_get(usage, "total_tokens")) == 99);

    yyjson_doc_free(doc);
    yyjson_mut_doc_free(mdoc);
}

/* --- Cycle 10: chat response round-trip (logprobs) ------------------------ */

static void test_chat_response_roundtrip_logprobs(void) {
    top_logprob_t tops0[2] = {
        {.token = "Hi", .logprob = -0.31725305f},
        {.token = "Hello", .logprob = -1.3190403f},
    };
    token_logprob_t content[2] = {
        {.token = "Hi", .logprob = -0.31725305f, .top_logprobs = tops0, .top_logprob_count = 2},
        {.token = " there", .logprob = -0.02380986f, .top_logprobs = NULL, .top_logprob_count = 0},
    };
    chat_completion_response_t resp = {
        .id = "chatcmpl-logprobs1",
        .model = "qwen3-4b",
        .created = 1728933352,
        .finish_reason = FINISH_STOP,
        .content = "Hi there",
        .logprobs_content = content,
        .logprobs_count = 2,
        .usage = {.prompt_tokens = 9, .completion_tokens = 2, .total_tokens = 11},
    };
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    yyjson_doc *doc = roundtrip(mdoc, chat_completion_response_serialize(&resp, mdoc), NULL);

    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *c0 = yyjson_arr_get(yyjson_obj_get(root, "choices"), 0);
    yyjson_val *lp = yyjson_obj_get(c0, "logprobs");
    assert(lp != NULL);
    yyjson_val *cont = yyjson_obj_get(lp, "content");
    assert(yyjson_arr_size(cont) == 2);

    yyjson_val *t0 = yyjson_arr_get(cont, 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(t0, "token")), "Hi") == 0);
    assert(fabs(yyjson_get_real(yyjson_obj_get(t0, "logprob")) - (-0.31725305)) < 1e-6);
    yyjson_val *tp0 = yyjson_obj_get(t0, "top_logprobs");
    assert(yyjson_arr_size(tp0) == 2);
    yyjson_val *tp0_1 = yyjson_arr_get(tp0, 1);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(tp0_1, "token")), "Hello") == 0);
    assert(fabs(yyjson_get_real(yyjson_obj_get(tp0_1, "logprob")) - (-1.3190403)) < 1e-6);

    yyjson_val *t1 = yyjson_arr_get(cont, 1);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(t1, "token")), " there") == 0);
    assert(fabs(yyjson_get_real(yyjson_obj_get(t1, "logprob")) - (-0.02380986)) < 1e-6);
    assert(yyjson_arr_size(yyjson_obj_get(t1, "top_logprobs")) == 0);

    yyjson_doc_free(doc);
    yyjson_mut_doc_free(mdoc);
}

/* --- Cycle 11: chunk serialize (streaming delta) -------------------------- */

static void test_chunk_serialize_stream(void) {
    chat_completion_chunk_t chunk = {
        .id = "chatcmpl-abc123",
        .model = "qwen3-4b",
        .created = 1728933352,
        .has_choice = true,
        .has_role = true,
        .role = ROLE_ASSISTANT,
        .delta_content = "Hello",
    };
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    yyjson_doc *doc = roundtrip(mdoc, chat_completion_chunk_serialize(&chunk, mdoc), NULL);

    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(root, "object")), "chat.completion.chunk") == 0);
    yyjson_val *choices = yyjson_obj_get(root, "choices");
    assert(yyjson_arr_size(choices) == 1);
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    yyjson_val *delta = yyjson_obj_get(c0, "delta");
    assert(strcmp(yyjson_get_str(yyjson_obj_get(delta, "role")), "assistant") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(delta, "content")), "Hello") == 0);
    /* finish_reason present as JSON null (not the enum string). */
    yyjson_val *fr = yyjson_obj_get(c0, "finish_reason");
    assert(fr != NULL && yyjson_is_null(fr));

    yyjson_doc_free(doc);
    yyjson_mut_doc_free(mdoc);
}

/* --- Cycle 12: chunk serialize (final include_usage chunk) ---------------- */

static void test_chunk_serialize_final_usage(void) {
    chat_completion_chunk_t chunk = {
        .id = "chatcmpl-abc123",
        .model = "qwen3-4b",
        .created = 1728933352,
        .has_choice = false,
        .has_usage = true,
        .usage = {.prompt_tokens = 9, .completion_tokens = 12, .total_tokens = 21},
    };
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    yyjson_doc *doc = roundtrip(mdoc, chat_completion_chunk_serialize(&chunk, mdoc), NULL);

    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(yyjson_arr_size(yyjson_obj_get(root, "choices")) == 0);
    yyjson_val *usage = yyjson_obj_get(root, "usage");
    assert(yyjson_get_sint(yyjson_obj_get(usage, "prompt_tokens")) == 9);
    assert(yyjson_get_sint(yyjson_obj_get(usage, "completion_tokens")) == 12);
    assert(yyjson_get_sint(yyjson_obj_get(usage, "total_tokens")) == 21);

    yyjson_doc_free(doc);
    yyjson_mut_doc_free(mdoc);
}

/* --- Cycle 13: chunk serialize (streamed tool-call fragments) ------------- */

static void test_chunk_serialize_tool_call_delta(void) {
    delta_tool_call_t frags[2] = {
        {.index = 0,
         .id = "call_abc123",
         .type = "function",
         .function_name = "get_weather",
         .arguments = ""}, /* present-and-empty */
        {.index = 0, .arguments = "{\"location\":"}, /* id/type/name NULL -> omitted */
    };
    chat_completion_chunk_t chunk = {
        .id = "chatcmpl-abc123",
        .model = "qwen3-4b",
        .created = 1728933352,
        .has_choice = true,
        .tool_calls = frags,
        .tool_call_count = 2,
    };
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    yyjson_doc *doc = roundtrip(mdoc, chat_completion_chunk_serialize(&chunk, mdoc), NULL);

    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *delta = yyjson_obj_get(yyjson_arr_get(yyjson_obj_get(root, "choices"), 0), "delta");
    yyjson_val *tcs = yyjson_obj_get(delta, "tool_calls");
    assert(yyjson_arr_size(tcs) == 2);

    yyjson_val *f0 = yyjson_arr_get(tcs, 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(f0, "id")), "call_abc123") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(f0, "type")), "function") == 0);
    yyjson_val *fn0 = yyjson_obj_get(f0, "function");
    assert(strcmp(yyjson_get_str(yyjson_obj_get(fn0, "name")), "get_weather") == 0);
    yyjson_val *args0 = yyjson_obj_get(fn0, "arguments");
    assert(args0 != NULL && yyjson_is_str(args0) && strcmp(yyjson_get_str(args0), "") == 0);

    /* Continuation fragment: id/type/name keys omitted entirely, arguments present. */
    yyjson_val *f1 = yyjson_arr_get(tcs, 1);
    assert(yyjson_obj_get(f1, "id") == NULL);
    assert(yyjson_obj_get(f1, "type") == NULL);
    yyjson_val *fn1 = yyjson_obj_get(f1, "function");
    assert(yyjson_obj_get(fn1, "name") == NULL);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(fn1, "arguments")), "{\"location\":") == 0);

    yyjson_doc_free(doc);
    yyjson_mut_doc_free(mdoc);
}

/* --- Cycle 14: embedding request parse ------------------------------------ */

static void test_embedding_request_parse(void) {
    yyjson_doc *doc = parse_fixture("embedding_req_float.json");
    embedding_request_t req;
    const char *err = NULL;
    assert(embedding_request_parse(&req, yyjson_doc_get_root(doc), &err) == 0);
    assert(strcmp(req.model, "bert-base") == 0);
    assert(strcmp(req.input, "The food was delicious.") == 0);
    assert(strcmp(req.encoding_format, "float") == 0);
    /* dimensions and user are deferred: present in the fixture, ignored here. */
    embedding_request_free(&req);
    yyjson_doc_free(doc);
}

/* --- Cycle 15: embedding response round-trip (float) ---------------------- */

/* Shared across cycles 15/16: the reference embedding vector. */
static const float kEmbedding[3] = {0.0023064255f, -0.009327292f, -0.0028842222f};

static void test_embedding_response_roundtrip_float(void) {
    float vals[3] = {kEmbedding[0], kEmbedding[1], kEmbedding[2]};
    embedding_data_t data[1] = {{.index = 0, .values = vals, .value_count = 3}};
    embedding_response_t resp = {
        .model = "bert-base",
        .data = data,
        .data_count = 1,
        .usage = {.prompt_tokens = 8, .completion_tokens = 0, .total_tokens = 8},
        .encoding_format = "float",
    };
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    yyjson_doc *doc = roundtrip(mdoc, embedding_response_serialize(&resp, mdoc), NULL);

    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(root, "object")), "list") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(root, "model")), "bert-base") == 0);
    yyjson_val *d0 = yyjson_arr_get(yyjson_obj_get(root, "data"), 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(d0, "object")), "embedding") == 0);
    assert(yyjson_get_sint(yyjson_obj_get(d0, "index")) == 0);
    yyjson_val *emb = yyjson_obj_get(d0, "embedding");
    assert(yyjson_arr_size(emb) == 3);
    assert(fabs(yyjson_get_real(yyjson_arr_get(emb, 0)) - 0.0023064255) < 1e-6);
    assert(fabs(yyjson_get_real(yyjson_arr_get(emb, 1)) - (-0.009327292)) < 1e-6);
    assert(fabs(yyjson_get_real(yyjson_arr_get(emb, 2)) - (-0.0028842222)) < 1e-6);
    yyjson_val *usage = yyjson_obj_get(root, "usage");
    assert(yyjson_get_sint(yyjson_obj_get(usage, "prompt_tokens")) == 8);
    assert(yyjson_get_sint(yyjson_obj_get(usage, "total_tokens")) == 8);

    yyjson_doc_free(doc);
    yyjson_mut_doc_free(mdoc);
}

/* --- Cycle 16: embedding response serialize (base64) ---------------------- */

static void test_embedding_response_serialize_base64(void) {
    float vals[3] = {kEmbedding[0], kEmbedding[1], kEmbedding[2]};
    embedding_data_t data[1] = {{.index = 0, .values = vals, .value_count = 3}};
    embedding_response_t resp = {
        .model = "bert-base",
        .data = data,
        .data_count = 1,
        .usage = {.prompt_tokens = 8, .completion_tokens = 0, .total_tokens = 8},
        .encoding_format = "base64",
    };
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    yyjson_doc *doc = roundtrip(mdoc, embedding_response_serialize(&resp, mdoc), NULL);

    yyjson_val *d0 = yyjson_arr_get(yyjson_obj_get(yyjson_doc_get_root(doc), "data"), 0);
    yyjson_val *emb = yyjson_obj_get(d0, "embedding");
    assert(yyjson_is_str(emb));
    /* Verified little-endian f32 base64 of the vector. NOT the fixture's
     * embedding_resp_base64.json string "bYIzPBRJGLwj0Dy7", which decodes to a
     * different vector (the Zig reference treats it as an opaque string and
     * never ties it to the floats). See the plan Context section. */
    assert(strcmp(yyjson_get_str(emb), "ZicXO4DRGLw4BT27") == 0);

    yyjson_doc_free(doc);
    yyjson_mut_doc_free(mdoc);
}

/* --- Cycle 17: finish_reason never leaks "cancelled" onto the wire -------- */

static void test_finish_reason_never_leaks_cancelled(void) {
    /* (a) The wire mapping is total and collapses the internal cancelled tag. */
    assert(strcmp(finish_reason_wire_str(FINISH_STOP), "stop") == 0);
    assert(strcmp(finish_reason_wire_str(FINISH_LENGTH), "length") == 0);
    assert(strcmp(finish_reason_wire_str(FINISH_TOOL_CALLS), "tool_calls") == 0);
    assert(strcmp(finish_reason_wire_str(FINISH_CONTENT_FILTER), "content_filter") == 0);
    assert(strcmp(finish_reason_wire_str(FINISH_CANCELLED), "stop") == 0);

    /* (b) A serialized chat response carrying FINISH_CANCELLED shows "stop". */
    chat_completion_response_t resp = {
        .id = "x", .model = "m", .created = 1, .finish_reason = FINISH_CANCELLED, .content = "hi"};
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    yyjson_doc *doc = roundtrip(mdoc, chat_completion_response_serialize(&resp, mdoc), NULL);
    yyjson_val *c0 = yyjson_arr_get(yyjson_obj_get(yyjson_doc_get_root(doc), "choices"), 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(c0, "finish_reason")), "stop") == 0);
    yyjson_doc_free(doc);
    yyjson_mut_doc_free(mdoc);

    /* ...and the completion response too. */
    completion_response_t cr = {
        .id = "x", .model = "m", .created = 1, .finish_reason = FINISH_CANCELLED, .text = "t"};
    yyjson_mut_doc *m2 = yyjson_mut_doc_new(NULL);
    yyjson_doc *d2 = roundtrip(m2, completion_response_serialize(&cr, m2), NULL);
    yyjson_val *cc0 = yyjson_arr_get(yyjson_obj_get(yyjson_doc_get_root(d2), "choices"), 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(cc0, "finish_reason")), "stop") == 0);
    yyjson_doc_free(d2);
    yyjson_mut_doc_free(m2);
}

/* --- Cycle 18: escaping guard (control/quote/backslash/UTF-8 round-trip) --- */

static void test_escaping_control_quote_backslash_roundtrip(void) {
    /* Corpus: control bytes 0x01..0x1F, quote, backslash, then 2/3/4-byte UTF-8
     * (é, 中, 😀). 0x00 is omitted: char* is NUL-terminated, a deliberate
     * deviation from Zig's byte slices. All bytes here are valid UTF-8. */
    unsigned char corpus[64];
    size_t k = 0;
    for (int b = 0x01; b <= 0x1F; b++)
        corpus[k++] = (unsigned char)b;
    corpus[k++] = '"';
    corpus[k++] = '\\';
    corpus[k++] = 0xC3;
    corpus[k++] = 0xA9; /* é */
    corpus[k++] = 0xE4;
    corpus[k++] = 0xB8;
    corpus[k++] = 0xAD; /* 中 */
    corpus[k++] = 0xF0;
    corpus[k++] = 0x9F;
    corpus[k++] = 0x98;
    corpus[k++] = 0x80; /* 😀 */
    corpus[k] = '\0';
    char *corpus_s = (char *)corpus;

    /* Inner JSON document {"text": corpus} feeds the double-encode path when it
     * is carried as a tool-call arguments string. */
    yyjson_mut_doc *innerdoc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *iobj = yyjson_mut_obj(innerdoc);
    yyjson_mut_obj_add_strcpy(innerdoc, iobj, "text", corpus_s);
    yyjson_mut_doc_set_root(innerdoc, iobj);
    char *inner = yyjson_mut_write(innerdoc, 0, NULL);
    assert(inner != NULL);

    tool_call_t calls[1] = {{.id = "call_1", .function_name = "f", .arguments = inner}};
    chat_completion_response_t resp = {
        .id = "x",
        .model = "m",
        .created = 1,
        .finish_reason = FINISH_STOP,
        .content = corpus_s, /* single-encode path */
        .tool_calls = calls,
        .tool_call_count = 1,
    };
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    char *json = NULL;
    yyjson_doc *doc = roundtrip(mdoc, chat_completion_response_serialize(&resp, mdoc), &json);

    /* (a) No raw control byte < 0x20 survives into the serialized output. */
    for (size_t i = 0; json[i]; i++)
        assert((unsigned char)json[i] >= 0x20);
    /* (b) Escapes are present: short escapes, \u00xx, and the two specials. */
    assert(strstr(json, "\\n") != NULL);
    assert(strstr(json, "\\t") != NULL);
    assert(strstr(json, "\\u0001") != NULL);
    assert(strstr(json, "\\\"") != NULL);
    assert(strstr(json, "\\\\") != NULL);
    /* Double-escaping evidence: the inner  appears backslash-doubled. */
    assert(strstr(json, "\\\\u0001") != NULL);

    /* (c) Both paths recover byte-identically after reparse. */
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *msg = yyjson_obj_get(yyjson_arr_get(yyjson_obj_get(root, "choices"), 0), "message");
    assert(strcmp(yyjson_get_str(yyjson_obj_get(msg, "content")), corpus_s) == 0);
    yyjson_val *tc0 = yyjson_arr_get(yyjson_obj_get(msg, "tool_calls"), 0);
    const char *args = yyjson_get_str(yyjson_obj_get(yyjson_obj_get(tc0, "function"), "arguments"));
    assert(strcmp(args, inner) == 0);
    yyjson_doc *idoc = yyjson_read(args, strlen(args), 0);
    assert(idoc != NULL);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(yyjson_doc_get_root(idoc), "text")), corpus_s) == 0);
    yyjson_doc_free(idoc);

    free(json);
    yyjson_doc_free(doc);
    yyjson_mut_doc_free(mdoc);
    free(inner);
    yyjson_mut_doc_free(innerdoc);
}

/* --- Cycle 19: invalid-UTF-8 write contract (characterization) ------------ */

/* Pins the upstream contract every layer above core relies on: yyjson's
 * default writer REFUSES invalid UTF-8 (returns NULL), and only the explicit
 * YYJSON_WRITE_ALLOW_INVALID_UNICODE opt-in emits it (substituting U+FFFD).
 * Core never sets that flag, so detokenizer/http layers must guarantee valid
 * UTF-8 before serializing. The "ab\xFF" "cd" literal is split so the hex
 * escape is exactly one byte (0xFF) followed by "cd". */
static void test_invalid_utf8_write_behavior(void) {
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *s = yyjson_mut_strncpy(mdoc, "ab\xFF" "cd", 5);
    assert(s != NULL);
    yyjson_mut_doc_set_root(mdoc, s);

    char *def = yyjson_mut_write(mdoc, 0, NULL);
    assert(def == NULL); /* default writer rejects invalid UTF-8 */

    char *allowed = yyjson_mut_write(mdoc, YYJSON_WRITE_ALLOW_INVALID_UNICODE, NULL);
    assert(allowed != NULL); /* opt-in flag emits it (U+FFFD substitution) */
    free(allowed);

    yyjson_mut_doc_free(mdoc);
}

int main(void) {
    test_helper_reads_and_parses_error_envelope();
    test_error_envelope_serialize();
    test_model_list_serialize();
    test_completion_request_parse_string();
    test_completion_response_roundtrip();
    test_chat_request_parse_string();
    test_chat_request_parse_parts_tools();
    test_chat_request_parse_toolchoice_string();
    test_chat_response_roundtrip_basic();
    test_chat_response_roundtrip_tool_calls();
    test_chat_response_roundtrip_logprobs();
    test_chunk_serialize_stream();
    test_chunk_serialize_final_usage();
    test_chunk_serialize_tool_call_delta();
    test_embedding_request_parse();
    test_embedding_response_roundtrip_float();
    test_embedding_response_serialize_base64();
    test_finish_reason_never_leaks_cancelled();
    test_escaping_control_quote_backslash_roundtrip();
    test_invalid_utf8_write_behavior();
    printf("test_openai: all passed\n");
    return 0;
}
