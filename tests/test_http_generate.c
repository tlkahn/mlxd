#include "http/server.h"
#include "model/tokenizer.h"
#include "http_client.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <yyjson/yyjson.h>

#ifndef MLXD_FIXTURES_DIR
#error "MLXD_FIXTURES_DIR must be defined"
#endif

#define TRIVIAL_TMPL "{{ messages[0].content }}"

/* --- Fixture --------------------------------------------------------------- */

typedef struct {
    engine_t       eng;
    tokenizer_t   *tok;
    http_server_t *srv;
    pthread_t      th;
    int            port;
} gen_fixture_t;

static void *server_thread(void *arg) {
    http_server_start((http_server_t *)arg);
    return NULL;
}

static gen_fixture_t fixture_up(bool load_model, bool set_tokenizer,
                                const char *chat_template) {
    gen_fixture_t f = {0};
    engine_init(&f.eng);

    if (load_model) {
        engine_cmd_t *cmd = calloc(1, sizeof(*cmd));
        assert(cmd != NULL);
        cmd->tag = CMD_LOAD;
        cmd->load.model_path = strdup("gpt2");
        engine_post(&f.eng, cmd);
        usleep(10000);
    }

    f.tok = set_tokenizer
        ? tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json")
        : NULL;

    http_server_config_t cfg = {
        .port = 0,
        .engine = &f.eng,
        .tokenizer = f.tok,
        .chat_template = chat_template,
        .model_id = "gpt2",
    };
    f.srv = http_server_create(&cfg);
    assert(f.srv != NULL);
    f.port = http_server_port(f.srv);
    assert(f.port > 0);
    int rc = pthread_create(&f.th, NULL, server_thread, f.srv);
    assert(rc == 0);
    for (int i = 0; i < 500; i++) {
        int fd = http_client_connect("127.0.0.1", f.port);
        if (fd >= 0) { close(fd); break; }
        usleep(1000);
        assert(i < 499);
    }
    return f;
}

static void fixture_down(gen_fixture_t *f) {
    http_server_stop(f->srv);
    pthread_join(f->th, NULL);
    http_server_destroy(f->srv);
    engine_destroy(&f->eng);
    if (f->tok) tokenizer_free(f->tok);
}

/* --- Helpers --------------------------------------------------------------- */

static http_client_response_t post_json(int port, const char *path,
                                        const char *body) {
    char raw[4096];
    snprintf(raw, sizeof(raw),
        "POST %s HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n%s", path, strlen(body), body);

    http_client_response_t resp;
    int rc = http_client_request("127.0.0.1", port, raw, &resp);
    assert(rc == 0);
    return resp;
}

/* --- 9a: chat echo roundtrip --------------------------------------------- */

static void test_chat_echo_roundtrip(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    http_client_response_t resp = post_json(f.port, "/v1/chat/completions",
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"hello world\"}]}");
    assert(resp.status == 200);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);

    assert(strcmp(yyjson_get_str(yyjson_obj_get(root, "object")), "chat.completion") == 0);

    const char *id = yyjson_get_str(yyjson_obj_get(root, "id"));
    assert(id != NULL);
    assert(strncmp(id, "chatcmpl-", 9) == 0);

    yyjson_val *choices = yyjson_obj_get(root, "choices");
    assert(yyjson_arr_size(choices) == 1);
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    yyjson_val *msg = yyjson_obj_get(c0, "message");
    const char *content = yyjson_get_str(yyjson_obj_get(msg, "content"));
    assert(content != NULL);
    assert(strcmp(content, "hello world") == 0);

    yyjson_val *usage = yyjson_obj_get(root, "usage");
    int pt = (int)yyjson_get_sint(yyjson_obj_get(usage, "prompt_tokens"));
    int ct = (int)yyjson_get_sint(yyjson_obj_get(usage, "completion_tokens"));
    int tt = (int)yyjson_get_sint(yyjson_obj_get(usage, "total_tokens"));
    assert(pt > 0);
    assert(ct == pt);
    assert(tt == pt + ct);

    yyjson_doc_free(doc);
    http_client_response_free(&resp);
    fixture_down(&f);
}

/* --- 9b: completion echo roundtrip --------------------------------------- */

static void test_completion_echo_roundtrip(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    http_client_response_t resp = post_json(f.port, "/v1/completions",
        "{\"model\":\"gpt2\",\"prompt\":\"Once upon a time\"}");
    assert(resp.status == 200);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);

    assert(strcmp(yyjson_get_str(yyjson_obj_get(root, "object")), "text_completion") == 0);

    const char *id = yyjson_get_str(yyjson_obj_get(root, "id"));
    assert(id != NULL);
    assert(strncmp(id, "cmpl-", 5) == 0);

    yyjson_val *choices = yyjson_obj_get(root, "choices");
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    const char *text = yyjson_get_str(yyjson_obj_get(c0, "text"));
    assert(text != NULL);
    assert(strcmp(text, "Once upon a time") == 0);

    yyjson_doc_free(doc);
    http_client_response_free(&resp);
    fixture_down(&f);
}

/* --- 9c: max_tokens defaults --------------------------------------------- */

static void test_max_tokens_defaults(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    /* Completions with a >16-token prompt and no max_tokens: default 16 truncates */
    const char *long_prompt = "one two three four five six seven eight nine ten "
                              "eleven twelve thirteen fourteen fifteen sixteen "
                              "seventeen eighteen nineteen twenty";
    char body[512];
    snprintf(body, sizeof(body),
        "{\"model\":\"gpt2\",\"prompt\":\"%s\"}", long_prompt);

    http_client_response_t resp1 = post_json(f.port, "/v1/completions", body);
    assert(resp1.status == 200);

    yyjson_doc *doc1 = yyjson_read(resp1.body, resp1.body_len, 0);
    yyjson_val *root1 = yyjson_doc_get_root(doc1);
    yyjson_val *choices1 = yyjson_obj_get(root1, "choices");
    yyjson_val *c0_1 = yyjson_arr_get(choices1, 0);
    const char *fr1 = yyjson_get_str(yyjson_obj_get(c0_1, "finish_reason"));
    assert(strcmp(fr1, "length") == 0);

    yyjson_val *usage1 = yyjson_obj_get(root1, "usage");
    int ct1 = (int)yyjson_get_sint(yyjson_obj_get(usage1, "completion_tokens"));
    assert(ct1 == 16);

    yyjson_doc_free(doc1);
    http_client_response_free(&resp1);

    /* Chat with the same long content: default INT_MAX -> full echo, finish_reason=stop */
    char chat_body[512];
    snprintf(chat_body, sizeof(chat_body),
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}]}", long_prompt);

    http_client_response_t resp2 = post_json(f.port, "/v1/chat/completions", chat_body);
    assert(resp2.status == 200);

    yyjson_doc *doc2 = yyjson_read(resp2.body, resp2.body_len, 0);
    yyjson_val *root2 = yyjson_doc_get_root(doc2);
    yyjson_val *choices2 = yyjson_obj_get(root2, "choices");
    yyjson_val *c0_2 = yyjson_arr_get(choices2, 0);
    const char *fr2 = yyjson_get_str(yyjson_obj_get(c0_2, "finish_reason"));
    assert(strcmp(fr2, "stop") == 0);

    yyjson_val *usage2 = yyjson_obj_get(root2, "usage");
    int ct2 = (int)yyjson_get_sint(yyjson_obj_get(usage2, "completion_tokens"));
    int pt2 = (int)yyjson_get_sint(yyjson_obj_get(usage2, "prompt_tokens"));
    assert(ct2 == pt2);

    yyjson_doc_free(doc2);
    http_client_response_free(&resp2);

    fixture_down(&f);
}

/* --- 9d: no model -> 500 ------------------------------------------------- */

static void test_no_model_500(void) {
    gen_fixture_t f = fixture_up(false, true, TRIVIAL_TMPL);

    http_client_response_t resp = post_json(f.port, "/v1/chat/completions",
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    assert(resp.status == 500);

    char conn_hdr[64];
    bool has = http_client_header(&resp, "Connection", conn_hdr, sizeof(conn_hdr));
    assert(has);
    assert(strcmp(conn_hdr, "close") == 0);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *err = yyjson_obj_get(root, "error");
    assert(err != NULL);

    yyjson_doc_free(doc);
    http_client_response_free(&resp);
    fixture_down(&f);
}

/* --- 9e: missing messages -> 400 ----------------------------------------- */

static void test_missing_messages_400(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    http_client_response_t resp = post_json(f.port, "/v1/chat/completions",
        "{\"model\":\"gpt2\"}");
    assert(resp.status == 400);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *err = yyjson_obj_get(root, "error");
    assert(err != NULL);

    yyjson_doc_free(doc);
    http_client_response_free(&resp);
    fixture_down(&f);
}

/* --- 9f: null tokenizer -> 503 ------------------------------------------- */

static void test_null_tokenizer_503(void) {
    gen_fixture_t f = fixture_up(true, false, TRIVIAL_TMPL);

    http_client_response_t resp = post_json(f.port, "/v1/chat/completions",
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    assert(resp.status == 503);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *err = yyjson_obj_get(root, "error");
    assert(err != NULL);

    yyjson_doc_free(doc);
    http_client_response_free(&resp);
    fixture_down(&f);
}

/* --- 9g: null template -> chat 400, completions ok ----------------------- */

static void test_null_template_400_chat(void) {
    gen_fixture_t f = fixture_up(true, true, NULL);

    http_client_response_t resp = post_json(f.port, "/v1/chat/completions",
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    assert(resp.status == 400);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_doc_free(doc);
    http_client_response_free(&resp);
    fixture_down(&f);
}

static void test_null_template_completion_ok(void) {
    gen_fixture_t f = fixture_up(true, true, NULL);

    http_client_response_t resp = post_json(f.port, "/v1/completions",
        "{\"model\":\"gpt2\",\"prompt\":\"hello\"}");
    assert(resp.status == 200);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_doc_free(doc);
    http_client_response_free(&resp);
    fixture_down(&f);
}

/* --- SSE helpers --------------------------------------------------------- */

static int sse_connect_and_post(int port, const char *path, const char *body,
                                char *hdrbuf, size_t hdrcap) {
    int fd = http_client_connect("127.0.0.1", port);
    assert(fd >= 0);

    char raw[4096];
    snprintf(raw, sizeof(raw),
        "POST %s HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "\r\n%s", path, strlen(body), body);
    assert(http_client_send_all(fd, raw, strlen(raw)) == 0);

    int rc = http_client_recv_headers(fd, hdrbuf, hdrcap);
    assert(rc > 0);
    return fd;
}

static yyjson_doc *parse_sse_json(const char *event) {
    assert(strncmp(event, "data: ", 6) == 0);
    const char *json_start = event + 6;
    const char *end = strstr(json_start, "\n\n");
    assert(end != NULL);
    size_t json_len = (size_t)(end - json_start);
    return yyjson_read(json_start, json_len, 0);
}

/* --- 10a: chat SSE sequence ---------------------------------------------- */

static void test_chat_sse_sequence(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    char hdrbuf[4096];
    int fd = sse_connect_and_post(f.port, "/v1/chat/completions",
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"hello world\"}],\"stream\":true}",
        hdrbuf, sizeof(hdrbuf));

    assert(strstr(hdrbuf, "200 OK") != NULL);
    assert(strstr(hdrbuf, "text/event-stream") != NULL);
    assert(strstr(hdrbuf, "Connection: close") != NULL);

    char evbuf[4096];
    char concat[4096] = {0};
    size_t concat_len = 0;
    bool got_role = false;
    bool got_finish = false;
    bool got_done = false;

    while (!got_done) {
        int n = http_client_recv_sse_event(fd, evbuf, sizeof(evbuf));
        assert(n > 0);

        if (strncmp(evbuf, "data: [DONE]", 12) == 0) {
            got_done = true;
            break;
        }

        yyjson_doc *doc = parse_sse_json(evbuf);
        assert(doc != NULL);
        yyjson_val *root = yyjson_doc_get_root(doc);
        yyjson_val *choices = yyjson_obj_get(root, "choices");

        if (yyjson_arr_size(choices) > 0) {
            yyjson_val *c0 = yyjson_arr_get(choices, 0);
            yyjson_val *delta = yyjson_obj_get(c0, "delta");
            const char *role = yyjson_get_str(yyjson_obj_get(delta, "role"));
            if (role && strcmp(role, "assistant") == 0)
                got_role = true;
            const char *content = yyjson_get_str(yyjson_obj_get(delta, "content"));
            if (content) {
                size_t clen = strlen(content);
                assert(concat_len + clen < sizeof(concat));
                memcpy(concat + concat_len, content, clen);
                concat_len += clen;
                concat[concat_len] = '\0';
            }
            yyjson_val *fr = yyjson_obj_get(c0, "finish_reason");
            if (fr && !yyjson_is_null(fr))
                got_finish = true;
        }
        yyjson_doc_free(doc);
    }

    assert(got_role);
    assert(got_finish);
    assert(got_done);
    assert(strcmp(concat, "hello world") == 0);

    close(fd);
    fixture_down(&f);
}

/* --- 10b: chat SSE include_usage ----------------------------------------- */

static void test_chat_sse_include_usage(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    char hdrbuf[4096];
    int fd = sse_connect_and_post(f.port, "/v1/chat/completions",
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"hello world\"}],"
        "\"stream\":true,\"stream_options\":{\"include_usage\":true}}",
        hdrbuf, sizeof(hdrbuf));
    assert(strstr(hdrbuf, "200 OK") != NULL);

    char evbuf[4096];
    bool got_usage = false;
    int usage_pt = 0, usage_ct = 0, usage_tt = 0;
    bool got_done = false;

    while (!got_done) {
        int n = http_client_recv_sse_event(fd, evbuf, sizeof(evbuf));
        assert(n > 0);
        if (strncmp(evbuf, "data: [DONE]", 12) == 0) {
            got_done = true;
            break;
        }
        yyjson_doc *doc = parse_sse_json(evbuf);
        assert(doc != NULL);
        yyjson_val *root = yyjson_doc_get_root(doc);
        yyjson_val *usage = yyjson_obj_get(root, "usage");
        if (usage && yyjson_is_obj(usage)) {
            got_usage = true;
            usage_pt = (int)yyjson_get_sint(yyjson_obj_get(usage, "prompt_tokens"));
            usage_ct = (int)yyjson_get_sint(yyjson_obj_get(usage, "completion_tokens"));
            usage_tt = (int)yyjson_get_sint(yyjson_obj_get(usage, "total_tokens"));
            yyjson_val *choices = yyjson_obj_get(root, "choices");
            assert(yyjson_arr_size(choices) == 0);
        }
        yyjson_doc_free(doc);
    }

    assert(got_usage);
    assert(usage_pt > 0);
    assert(usage_ct > 0);
    assert(usage_tt == usage_pt + usage_ct);
    assert(got_done);

    close(fd);
    fixture_down(&f);
}

/* --- 10c: completion SSE sequence ---------------------------------------- */

static void test_completion_sse_sequence(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    char hdrbuf[4096];
    int fd = sse_connect_and_post(f.port, "/v1/completions",
        "{\"model\":\"gpt2\",\"prompt\":\"hello world\",\"stream\":true,\"max_tokens\":100}",
        hdrbuf, sizeof(hdrbuf));
    assert(strstr(hdrbuf, "200 OK") != NULL);

    char evbuf[4096];
    char concat[4096] = {0};
    size_t concat_len = 0;
    bool got_finish = false;
    bool got_done = false;

    while (!got_done) {
        int n = http_client_recv_sse_event(fd, evbuf, sizeof(evbuf));
        assert(n > 0);
        if (strncmp(evbuf, "data: [DONE]", 12) == 0) {
            got_done = true;
            break;
        }
        yyjson_doc *doc = parse_sse_json(evbuf);
        assert(doc != NULL);
        yyjson_val *root = yyjson_doc_get_root(doc);
        assert(strcmp(yyjson_get_str(yyjson_obj_get(root, "object")), "text_completion") == 0);
        yyjson_val *choices = yyjson_obj_get(root, "choices");
        yyjson_val *c0 = yyjson_arr_get(choices, 0);
        const char *text = yyjson_get_str(yyjson_obj_get(c0, "text"));
        if (text && strlen(text) > 0) {
            size_t tlen = strlen(text);
            assert(concat_len + tlen < sizeof(concat));
            memcpy(concat + concat_len, text, tlen);
            concat_len += tlen;
            concat[concat_len] = '\0';
        }
        yyjson_val *fr = yyjson_obj_get(c0, "finish_reason");
        if (fr && !yyjson_is_null(fr))
            got_finish = true;
        yyjson_doc_free(doc);
    }

    assert(got_finish);
    assert(got_done);
    assert(strcmp(concat, "hello world") == 0);

    close(fd);
    fixture_down(&f);
}

/* --- 13a: client disconnect cancels generation --------------------------- */

static void test_client_disconnect_cancels(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    /* Use a long prompt so the engine's echo loop parks on stream_push
     * (ring cap 64 fills up). The client disconnecting mid-stream should
     * cancel the stream, unblock the engine, and allow clean teardown. */
    char long_content[4096];
    memset(long_content, 0, sizeof(long_content));
    int off = 0;
    for (int i = 0; i < 200 && off < 3800; i++)
        off += snprintf(long_content + off, sizeof(long_content) - (size_t)off,
                        "word%d ", i);

    char body[8192];
    snprintf(body, sizeof(body),
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],\"stream\":true}",
        long_content);

    char hdrbuf[4096];
    int fd = sse_connect_and_post(f.port, "/v1/chat/completions", body,
                                  hdrbuf, sizeof(hdrbuf));
    assert(strstr(hdrbuf, "200 OK") != NULL);

    /* Read a few events to confirm streaming started */
    char evbuf[4096];
    int n = http_client_recv_sse_event(fd, evbuf, sizeof(evbuf));
    assert(n > 0);

    /* Abruptly close the client fd mid-stream */
    close(fd);

    /* Small delay for the server to notice the disconnect */
    usleep(50000);

    /* The test is that fixture_down completes without hang or crash.
     * Under tsan, this validates there are no data races in the
     * teardown path. */
    fixture_down(&f);
}

/* --- 13b: server stop during stream -------------------------------------- */

static void test_server_stop_during_stream(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    char long_content[4096];
    memset(long_content, 0, sizeof(long_content));
    int off = 0;
    for (int i = 0; i < 200 && off < 3800; i++)
        off += snprintf(long_content + off, sizeof(long_content) - (size_t)off,
                        "word%d ", i);

    char body[8192];
    snprintf(body, sizeof(body),
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],\"stream\":true}",
        long_content);

    char hdrbuf[4096];
    int fd = sse_connect_and_post(f.port, "/v1/chat/completions", body,
                                  hdrbuf, sizeof(hdrbuf));
    assert(strstr(hdrbuf, "200 OK") != NULL);

    /* Read a few events */
    char evbuf[4096];
    http_client_recv_sse_event(fd, evbuf, sizeof(evbuf));

    /* Stop the server while the client is still connected and streaming.
     * This tests that walk_close_cb correctly handles gen-owned handles
     * and doesn't cause EBUSY on uv_loop_close. */
    fixture_down(&f);

    /* Server side is already gone; close() only releases the local fd. */
    close(fd);
}

/* --- 11a: negative temperature -> 400 (#10) ------------------------------ */

static void test_bad_temperature_400(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    http_client_response_t resp = post_json(f.port, "/v1/chat/completions",
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"temperature\":-1}");
    assert(resp.status == 400);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *err = yyjson_obj_get(root, "error");
    assert(err);
    const char *type = yyjson_get_str(yyjson_obj_get(err, "type"));
    assert(type && strcmp(type, "invalid_request_error") == 0);

    yyjson_doc_free(doc);
    http_client_response_free(&resp);
    fixture_down(&f);
}

/* --- 11b: bad top_p on completions -> 400 (#10) -------------------------- */

static void test_bad_top_p_completions_400(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    http_client_response_t resp = post_json(f.port, "/v1/completions",
        "{\"model\":\"gpt2\",\"prompt\":\"hello\",\"top_p\":2.0}");
    assert(resp.status == 400);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *err = yyjson_obj_get(root, "error");
    assert(err);

    yyjson_doc_free(doc);
    http_client_response_free(&resp);
    fixture_down(&f);
}

/* --- 11c: n/logprobs/stop silently ignored (#10 policy pin) -------------- */

static void test_ignored_params_succeed(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    http_client_response_t resp = post_json(f.port, "/v1/chat/completions",
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
        "\"n\":3,\"logprobs\":true,\"stop\":[\"x\"]}");
    assert(resp.status == 200);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc);
    yyjson_doc_free(doc);
    http_client_response_free(&resp);
    fixture_down(&f);
}

/* --- 11d: error type maps from status (#9) ------------------------------- */

static void test_no_model_error_type(void) {
    gen_fixture_t f = fixture_up(false, true, TRIVIAL_TMPL);

    http_client_response_t resp = post_json(f.port, "/v1/chat/completions",
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    assert(resp.status == 500);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *err = yyjson_obj_get(root, "error");
    assert(err);
    const char *type = yyjson_get_str(yyjson_obj_get(err, "type"));
    assert(type && strcmp(type, "server_error") == 0);

    yyjson_doc_free(doc);
    http_client_response_free(&resp);
    fixture_down(&f);
}

/* --- 13c: disconnect on no-model path ------------------------------------ */

static void test_disconnect_no_model(void) {
    gen_fixture_t f = fixture_up(false, true, TRIVIAL_TMPL);

    /* On the no-model path, the engine pushes CHUNK_ERROR then CHUNK_DONE.
     * If the consumer disconnects before draining, the ERROR push may fail
     * (consumer cancelled), and NO terminal chunk is injected. The teardown
     * must still complete via the refcount gate. */
    char hdrbuf[4096];
    int fd = sse_connect_and_post(f.port, "/v1/chat/completions",
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true}",
        hdrbuf, sizeof(hdrbuf));

    /* Immediately close without reading any events */
    close(fd);

    usleep(50000);
    fixture_down(&f);
}

/* --- 14a: pipelined gen requests produce single response ----------------- */

static void test_pipelined_gen_requests_single_response(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    int fd = http_client_connect("127.0.0.1", f.port);
    assert(fd >= 0);

    struct timeval tv = {.tv_sec = 5, .tv_usec = 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    const char *body = "{\"model\":\"gpt2\",\"messages\":"
                       "[{\"role\":\"user\",\"content\":\"hello\"}]}";
    size_t blen = strlen(body);
    char raw[8192];
    snprintf(raw, sizeof(raw),
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "\r\n"
        "%s"
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "\r\n"
        "%s",
        blen, body, blen, body);

    assert(http_client_send_all(fd, raw, strlen(raw)) == 0);

    char resp[65536];
    size_t total = 0;
    while (total < sizeof(resp) - 1) {
        ssize_t n = read(fd, resp + total, sizeof(resp) - 1 - total);
        if (n <= 0) break;
        total += (size_t)n;
    }
    resp[total] = '\0';
    close(fd);

    int count = 0;
    const char *p = resp;
    while ((p = strstr(p, "HTTP/1.1")) != NULL) {
        count++;
        p += 8;
    }
    assert(count == 1);
    assert(strstr(resp, "HTTP/1.1 200") != NULL);
    assert(strstr(resp, "Connection: close") != NULL);

    fixture_down(&f);
}

/* --- 14b: chat empty generation -> content:"" not null ------------------- */

static void test_chat_empty_generation_content_empty_string(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    http_client_response_t resp = post_json(f.port, "/v1/chat/completions",
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"\"}]}");
    assert(resp.status == 200);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);

    yyjson_val *choices = yyjson_obj_get(root, "choices");
    assert(yyjson_arr_size(choices) == 1);
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    yyjson_val *msg = yyjson_obj_get(c0, "message");
    yyjson_val *content_val = yyjson_obj_get(msg, "content");
    assert(content_val != NULL);
    assert(yyjson_is_str(content_val));
    assert(strcmp(yyjson_get_str(content_val), "") == 0);

    yyjson_doc_free(doc);
    http_client_response_free(&resp);
    fixture_down(&f);
}

/* --- 14c: completion empty generation -> text:"" not missing ------------- */

static void test_completion_empty_generation_text_field(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    http_client_response_t resp = post_json(f.port, "/v1/completions",
        "{\"model\":\"gpt2\",\"prompt\":\"\"}");
    assert(resp.status == 200);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);

    yyjson_val *choices = yyjson_obj_get(root, "choices");
    assert(yyjson_arr_size(choices) == 1);
    yyjson_val *c0 = yyjson_arr_get(choices, 0);
    yyjson_val *text_val = yyjson_obj_get(c0, "text");
    assert(text_val != NULL);
    assert(yyjson_is_str(text_val));
    assert(strcmp(yyjson_get_str(text_val), "") == 0);

    yyjson_doc_free(doc);
    http_client_response_free(&resp);
    fixture_down(&f);
}

/* --- 14d: SSE chat zero-token -> role chunk emitted ---------------------- */

static void test_chat_sse_zero_token_role_chunk(void) {
    gen_fixture_t f = fixture_up(true, true, TRIVIAL_TMPL);

    char hdrbuf[4096];
    int fd = sse_connect_and_post(f.port, "/v1/chat/completions",
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"\"}],"
        "\"stream\":true}",
        hdrbuf, sizeof(hdrbuf));

    assert(strstr(hdrbuf, "200 OK") != NULL);

    char evbuf[4096];
    bool got_role = false;
    bool got_finish = false;
    bool got_done = false;

    while (!got_done) {
        int n = http_client_recv_sse_event(fd, evbuf, sizeof(evbuf));
        assert(n > 0);

        if (strncmp(evbuf, "data: [DONE]", 12) == 0) {
            got_done = true;
            break;
        }

        yyjson_doc *doc = parse_sse_json(evbuf);
        assert(doc != NULL);
        yyjson_val *root = yyjson_doc_get_root(doc);
        yyjson_val *choices = yyjson_obj_get(root, "choices");
        if (yyjson_arr_size(choices) > 0) {
            yyjson_val *c0 = yyjson_arr_get(choices, 0);
            yyjson_val *delta = yyjson_obj_get(c0, "delta");
            const char *role = yyjson_get_str(yyjson_obj_get(delta, "role"));
            if (role && strcmp(role, "assistant") == 0)
                got_role = true;
            yyjson_val *fr = yyjson_obj_get(c0, "finish_reason");
            if (fr && !yyjson_is_null(fr))
                got_finish = true;
        }
        yyjson_doc_free(doc);
    }

    assert(got_role);
    assert(got_finish);
    assert(got_done);

    close(fd);
    fixture_down(&f);
}

/* --- main ---------------------------------------------------------------- */

int main(void) {
    setenv("MLXD_CACHE_DIR", MLXD_FIXTURES_DIR "/registry_cache", 1);
    unsetenv("MLXD_HF_HUB_DIR");

    test_chat_echo_roundtrip();
    test_completion_echo_roundtrip();
    test_max_tokens_defaults();
    test_no_model_500();
    test_missing_messages_400();
    test_null_tokenizer_503();
    test_null_template_400_chat();
    test_null_template_completion_ok();
    test_chat_sse_sequence();
    test_chat_sse_include_usage();
    test_completion_sse_sequence();
    test_bad_temperature_400();
    test_bad_top_p_completions_400();
    test_ignored_params_succeed();
    test_no_model_error_type();
    test_client_disconnect_cancels();
    test_server_stop_during_stream();
    test_disconnect_no_model();
    test_pipelined_gen_requests_single_response();
    test_chat_empty_generation_content_empty_string();
    test_completion_empty_generation_text_field();
    test_chat_sse_zero_token_role_chunk();
    printf("test_http_generate: all passed\n");

    unsetenv("MLXD_CACHE_DIR");
    return 0;
}
