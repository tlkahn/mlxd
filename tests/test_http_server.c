#include "http/server.h"
#include "model/tokenizer.h"
#include "http_client.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <yyjson/yyjson.h>

/* --- Test harness --------------------------------------------------------- */

typedef struct {
    http_server_t *srv;
    pthread_t      th;
    int            port;
} srv_fixture_t;

static void *server_thread(void *arg) {
    http_server_start((http_server_t *)arg);
    return NULL;
}

static srv_fixture_t fixture_up(http_server_config_t cfg) {
    srv_fixture_t f = {0};
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

static void fixture_down(srv_fixture_t *f) {
    http_server_stop(f->srv);
    pthread_join(f->th, NULL);
    http_server_destroy(f->srv);
    f->srv = NULL;
}

/* --- Stage 6 tests -------------------------------------------------------- */

static void test_bind_reports_port(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng};
    srv_fixture_t f = fixture_up(cfg);
    assert(f.port > 0);
    fixture_down(&f);
    engine_destroy(&eng);
}

static void test_unknown_route_404(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng};
    srv_fixture_t f = fixture_up(cfg);

    http_client_response_t resp;
    int rc = http_client_request("127.0.0.1", f.port,
        "GET /nope HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        &resp);
    assert(rc == 0);
    assert(resp.status == 404);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *err = yyjson_obj_get(root, "error");
    assert(err != NULL);
    const char *type = yyjson_get_str(yyjson_obj_get(err, "type"));
    assert(type != NULL);
    assert(strcmp(type, "not_found_error") == 0);
    const char *msg = yyjson_get_str(yyjson_obj_get(err, "message"));
    assert(msg != NULL);
    assert(strlen(msg) > 0);
    yyjson_doc_free(doc);

    char cors[256];
    bool has_cors = http_client_header(&resp, "Access-Control-Allow-Origin", cors, sizeof(cors));
    assert(has_cors);
    assert(strcmp(cors, "*") == 0);

    http_client_response_free(&resp);
    fixture_down(&f);
    engine_destroy(&eng);
}

static void test_start_stop_clean(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng};
    srv_fixture_t f = fixture_up(cfg);
    fixture_down(&f);
    engine_destroy(&eng);
}

/* --- Stage 7 tests -------------------------------------------------------- */

static void test_models_list(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng};
    srv_fixture_t f = fixture_up(cfg);

    http_client_response_t resp;
    int rc = http_client_request("127.0.0.1", f.port,
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        &resp);
    assert(rc == 0);
    assert(resp.status == 200);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(root, "object")), "list") == 0);

    yyjson_val *data = yyjson_obj_get(root, "data");
    assert(yyjson_is_arr(data));

    int found_qwen = 0, found_other = 0;
    size_t idx, max;
    yyjson_val *item;
    yyjson_arr_foreach(data, idx, max, item) {
        assert(strcmp(yyjson_get_str(yyjson_obj_get(item, "object")), "model") == 0);
        assert(strcmp(yyjson_get_str(yyjson_obj_get(item, "owned_by")), "mlxd") == 0);
        const char *id = yyjson_get_str(yyjson_obj_get(item, "id"));
        if (strcmp(id, "mlx-community/Qwen3-0.6B-4bit") == 0) found_qwen = 1;
        if (strcmp(id, "org/other-model") == 0) found_other = 1;
    }
    assert(found_qwen);
    assert(found_other);

    yyjson_doc_free(doc);
    http_client_response_free(&resp);
    fixture_down(&f);
    engine_destroy(&eng);
}

/* --- Stage 8 tests -------------------------------------------------------- */

static void test_options_preflight(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng};
    srv_fixture_t f = fixture_up(cfg);

    http_client_response_t resp;
    int rc = http_client_request("127.0.0.1", f.port,
        "OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        &resp);
    assert(rc == 0);
    assert(resp.status == 204);

    char val[256];
    assert(http_client_header(&resp, "Access-Control-Allow-Origin", val, sizeof(val)));
    assert(strcmp(val, "*") == 0);

    assert(http_client_header(&resp, "Access-Control-Allow-Methods", val, sizeof(val)));
    assert(strstr(val, "GET") != NULL);
    assert(strstr(val, "POST") != NULL);
    assert(strstr(val, "OPTIONS") != NULL);

    assert(http_client_header(&resp, "Access-Control-Allow-Headers", val, sizeof(val)));
    assert(strstr(val, "Content-Type") != NULL);
    assert(strstr(val, "Authorization") != NULL);

    assert(resp.body_len == 0);

    http_client_response_free(&resp);
    fixture_down(&f);
    engine_destroy(&eng);
}

/* --- Stage 14 tests ------------------------------------------------------- */

static void test_embeddings_501(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng};
    srv_fixture_t f = fixture_up(cfg);

    const char *body = "{\"model\":\"m\",\"input\":\"hi\"}";
    char raw[512];
    snprintf(raw, sizeof(raw),
        "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n%s", strlen(body), body);

    http_client_response_t resp;
    int rc = http_client_request("127.0.0.1", f.port, raw, &resp);
    assert(rc == 0);
    assert(resp.status == 501);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *err = yyjson_obj_get(root, "error");
    assert(strcmp(yyjson_get_str(yyjson_obj_get(err, "type")),
                 "not_implemented_error") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(err, "code")),
                 "not_implemented") == 0);
    yyjson_doc_free(doc);

    http_client_response_free(&resp);
    fixture_down(&f);
    engine_destroy(&eng);
}

static void test_embeddings_invalid_400(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng};
    srv_fixture_t f = fixture_up(cfg);

    const char *body = "{\"model\":\"m\"}";
    char raw[512];
    snprintf(raw, sizeof(raw),
        "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n%s", strlen(body), body);

    http_client_response_t resp;
    int rc = http_client_request("127.0.0.1", f.port, raw, &resp);
    assert(rc == 0);
    assert(resp.status == 400);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *err = yyjson_obj_get(root, "error");
    assert(strcmp(yyjson_get_str(yyjson_obj_get(err, "type")),
                 "invalid_request_error") == 0);
    yyjson_doc_free(doc);

    http_client_response_free(&resp);
    fixture_down(&f);
    engine_destroy(&eng);
}

/* --- Stage 11 tests ------------------------------------------------------- */

static void test_malformed_json_400(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng};
    srv_fixture_t f = fixture_up(cfg);

    const char *body = "{bad";
    char raw[512];
    snprintf(raw, sizeof(raw),
        "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n%s", strlen(body), body);

    http_client_response_t resp;
    int rc = http_client_request("127.0.0.1", f.port, raw, &resp);
    assert(rc == 0);
    assert(resp.status == 400);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *err = yyjson_obj_get(root, "error");
    assert(err != NULL);
    yyjson_doc_free(doc);

    http_client_response_free(&resp);
    fixture_down(&f);
    engine_destroy(&eng);
}

static void test_body_too_large_413(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng, .max_body_bytes = 64};
    srv_fixture_t f = fixture_up(cfg);

    char body[128];
    memset(body, 'A', sizeof(body) - 1);
    body[sizeof(body) - 1] = '\0';

    char raw[512];
    snprintf(raw, sizeof(raw),
        "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n%s", strlen(body), body);

    http_client_response_t resp;
    int rc = http_client_request("127.0.0.1", f.port, raw, &resp);
    assert(rc == 0);
    assert(resp.status == 413);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *err = yyjson_obj_get(root, "error");
    assert(strcmp(yyjson_get_str(yyjson_obj_get(err, "type")),
                 "payload_too_large") == 0);
    yyjson_doc_free(doc);

    assert(!resp.keep_alive);

    http_client_response_free(&resp);
    fixture_down(&f);
    engine_destroy(&eng);
}

static void test_wrong_method_405(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng};
    srv_fixture_t f = fixture_up(cfg);

    http_client_response_t resp;
    int rc = http_client_request("127.0.0.1", f.port,
        "PUT /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        &resp);
    assert(rc == 0);
    assert(resp.status == 405);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *err = yyjson_obj_get(root, "error");
    assert(strcmp(yyjson_get_str(yyjson_obj_get(err, "type")),
                 "method_not_allowed") == 0);
    yyjson_doc_free(doc);

    http_client_response_free(&resp);
    fixture_down(&f);
    engine_destroy(&eng);
}

static void test_malformed_http_400(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng};
    srv_fixture_t f = fixture_up(cfg);

    http_client_response_t resp;
    int rc = http_client_request("127.0.0.1", f.port,
        "GARBAGE\r\n\r\n", &resp);
    assert(rc == 0);
    assert(resp.status == 400);

    yyjson_doc *doc = yyjson_read(resp.body, resp.body_len, 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *err = yyjson_obj_get(root, "error");
    assert(err != NULL);
    yyjson_doc_free(doc);

    http_client_response_free(&resp);
    fixture_down(&f);
    engine_destroy(&eng);
}

/* --- Stage 12 tests ------------------------------------------------------- */

static void test_keep_alive_two_requests(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng};
    srv_fixture_t f = fixture_up(cfg);

    int fd = http_client_connect("127.0.0.1", f.port);
    assert(fd >= 0);

    const char *req1 =
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";
    assert(http_client_send_all(fd, req1, strlen(req1)) == 0);

    http_client_response_t resp1;
    assert(http_client_recv_response(fd, &resp1) == 0);
    assert(resp1.status == 200);
    assert(resp1.keep_alive);

    const char *req2 =
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    assert(http_client_send_all(fd, req2, strlen(req2)) == 0);

    http_client_response_t resp2;
    assert(http_client_recv_response(fd, &resp2) == 0);
    assert(resp2.status == 200);

    http_client_response_free(&resp1);
    http_client_response_free(&resp2);
    close(fd);
    fixture_down(&f);
    engine_destroy(&eng);
}

static void test_connection_close(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng};
    srv_fixture_t f = fixture_up(cfg);

    int fd = http_client_connect("127.0.0.1", f.port);
    assert(fd >= 0);

    const char *req =
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    assert(http_client_send_all(fd, req, strlen(req)) == 0);

    http_client_response_t resp;
    assert(http_client_recv_response(fd, &resp) == 0);
    assert(resp.status == 200);
    assert(!resp.keep_alive);

    char val[256];
    assert(http_client_header(&resp, "Connection", val, sizeof(val)));
    assert(strcmp(val, "close") == 0);

    char buf[1];
    ssize_t n = read(fd, buf, 1);
    assert(n <= 0);

    http_client_response_free(&resp);
    close(fd);
    fixture_down(&f);
    engine_destroy(&eng);
}

static void test_options_preflight_close(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng};
    srv_fixture_t f = fixture_up(cfg);

    int fd = http_client_connect("127.0.0.1", f.port);
    assert(fd >= 0);

    const char *req =
        "OPTIONS /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Connection: close\r\n"
        "\r\n";
    assert(http_client_send_all(fd, req, strlen(req)) == 0);

    http_client_response_t resp;
    assert(http_client_recv_response(fd, &resp) == 0);
    assert(resp.status == 204);

    char val[256];
    assert(http_client_header(&resp, "Connection", val, sizeof(val)));
    assert(strcmp(val, "close") == 0);

    char buf[1];
    assert(read(fd, buf, 1) <= 0);

    http_client_response_free(&resp);
    close(fd);
    fixture_down(&f);
    engine_destroy(&eng);
}

static void test_stop_with_open_connections(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng};
    srv_fixture_t f = fixture_up(cfg);

    int fd1 = http_client_connect("127.0.0.1", f.port);
    assert(fd1 >= 0);
    int fd2 = http_client_connect("127.0.0.1", f.port);
    assert(fd2 >= 0);

    const char *req =
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";
    assert(http_client_send_all(fd1, req, strlen(req)) == 0);
    assert(http_client_send_all(fd2, req, strlen(req)) == 0);

    http_client_response_t resp1, resp2;
    assert(http_client_recv_response(fd1, &resp1) == 0);
    assert(resp1.status == 200);
    assert(http_client_recv_response(fd2, &resp2) == 0);
    assert(resp2.status == 200);

    fixture_down(&f);

    char buf[1];
    assert(read(fd1, buf, 1) <= 0);
    assert(read(fd2, buf, 1) <= 0);

    http_client_response_free(&resp1);
    http_client_response_free(&resp2);
    close(fd1);
    close(fd2);
    engine_destroy(&eng);
}

/* --- C3: EBUSY warning on destroy ----------------------------------------- */

static void test_destroy_ebusy_warns(void) {
    engine_t eng;
    engine_init(&eng);
    http_server_config_t cfg = {.port = 0, .engine = &eng};
    http_server_t *srv = http_server_create(&cfg);
    assert(srv != NULL);

    char tmppath[] = "/tmp/mlxd_test_ebusy_XXXXXX";
    int tmpfd = mkstemp(tmppath);
    assert(tmpfd >= 0);

    int saved_stderr = dup(STDERR_FILENO);
    assert(saved_stderr >= 0);
    fflush(stderr);
    dup2(tmpfd, STDERR_FILENO);
    close(tmpfd);

    http_server_destroy(srv);

    fflush(stderr);
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);

    FILE *f = fopen(tmppath, "r");
    assert(f != NULL);
    char buf[1024] = {0};
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    unlink(tmppath);

    assert(n > 0);
    assert(strstr(buf, "EBUSY") != NULL);

    engine_destroy(&eng);
}

/* --- Gen-capable fixture (Stage 15 drain tests) --------------------------- */

#define TRIVIAL_TMPL "{{ messages[0].content }}"

typedef struct {
    engine_t       eng;
    tokenizer_t   *tok;
    http_server_t *srv;
    pthread_t      th;
    int            port;
} gen_srv_fixture_t;

static gen_srv_fixture_t gen_fixture_up(uint64_t drain_deadline_ms) {
    gen_srv_fixture_t f = {0};
    engine_init(&f.eng);

    engine_cmd_t *cmd = calloc(1, sizeof(*cmd));
    assert(cmd != NULL);
    cmd->tag = CMD_LOAD;
    cmd->load.model_path = strdup("gpt2");
    engine_post(&f.eng, cmd);
    usleep(10000);

    f.tok = tokenizer_load(MLXD_FIXTURES_DIR "/gpt2/tokenizer.json");
    assert(f.tok != NULL);

    http_server_config_t cfg = {
        .port = 0,
        .engine = &f.eng,
        .tokenizer = f.tok,
        .chat_template = TRIVIAL_TMPL,
        .model_id = "gpt2",
        .drain_deadline_ms = drain_deadline_ms,
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

static void gen_fixture_down(gen_srv_fixture_t *f) {
    http_server_stop(f->srv);
    pthread_join(f->th, NULL);
    http_server_destroy(f->srv);
    engine_destroy(&f->eng);
    tokenizer_free(f->tok);
}

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

static uint64_t now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 + (uint64_t)tv.tv_usec / 1000;
}

/* --- Stage 15 Cycle A: stop mid-stream delivers data: [DONE] -------------- */

static void test_stop_mid_stream_delivers_done(void) {
    gen_srv_fixture_t f = gen_fixture_up(5000);

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

    char evbuf[4096];
    int n = http_client_recv_sse_event(fd, evbuf, sizeof(evbuf));
    assert(n > 0);

    struct timeval tv = {.tv_sec = 3, .tv_usec = 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    http_server_stop(f.srv);

    bool got_done = false;
    while (!got_done) {
        n = http_client_recv_sse_event(fd, evbuf, sizeof(evbuf));
        if (n <= 0) break;
        if (strncmp(evbuf, "data: [DONE]", 12) == 0)
            got_done = true;
    }
    assert(got_done);

    pthread_join(f.th, NULL);
    http_server_destroy(f.srv);
    engine_destroy(&f.eng);
    tokenizer_free(f.tok);
    close(fd);
}

/* --- Stage 15 Cycle B: bounded drain deadline ----------------------------- */

static void test_stop_drain_deadline_bounded(void) {
    gen_srv_fixture_t f = gen_fixture_up(150);

    char *long_content = malloc(16384);
    assert(long_content);
    int off = 0;
    for (int i = 0; i < 2000 && off < 15000; i++)
        off += snprintf(long_content + off, 16384 - (size_t)off, "word%d ", i);
    long_content[off] = '\0';

    size_t bodycap = (size_t)off + 256;
    char *body = malloc(bodycap);
    assert(body);
    snprintf(body, bodycap,
        "{\"model\":\"gpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],\"stream\":true}",
        long_content);
    free(long_content);

    int fd = http_client_connect("127.0.0.1", f.port);
    assert(fd >= 0);

    int rcvbuf = 1;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    size_t blen = strlen(body);
    size_t rawcap = blen + 512;
    char *raw = malloc(rawcap);
    assert(raw);
    int rawlen = snprintf(raw, rawcap,
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "\r\n%s", blen, body);
    assert(http_client_send_all(fd, raw, (size_t)rawlen) == 0);
    free(raw);
    free(body);

    char hdrbuf[4096];
    int hrc = http_client_recv_headers(fd, hdrbuf, sizeof(hdrbuf));
    assert(hrc > 0);
    assert(strstr(hdrbuf, "200 OK") != NULL);

    usleep(500000);

    uint64_t t0 = now_ms();
    http_server_stop(f.srv);
    pthread_join(f.th, NULL);
    uint64_t elapsed = now_ms() - t0;

    assert(elapsed < 1500);

    http_server_destroy(f.srv);
    engine_destroy(&f.eng);
    tokenizer_free(f.tok);
    close(fd);
}

/* --- Stage 15 Cycle C: idle stop is prompt -------------------------------- */

static void test_stop_idle_is_prompt(void) {
    gen_srv_fixture_t f = gen_fixture_up(5000);

    int fd1 = http_client_connect("127.0.0.1", f.port);
    assert(fd1 >= 0);
    int fd2 = http_client_connect("127.0.0.1", f.port);
    assert(fd2 >= 0);

    const char *req = "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";
    assert(http_client_send_all(fd1, req, strlen(req)) == 0);
    assert(http_client_send_all(fd2, req, strlen(req)) == 0);

    http_client_response_t resp1, resp2;
    assert(http_client_recv_response(fd1, &resp1) == 0);
    assert(resp1.status == 200);
    assert(http_client_recv_response(fd2, &resp2) == 0);
    assert(resp2.status == 200);

    uint64_t t0 = now_ms();
    gen_fixture_down(&f);
    uint64_t elapsed = now_ms() - t0;

    assert(elapsed < 1000);

    char buf[1];
    assert(read(fd1, buf, 1) <= 0);
    assert(read(fd2, buf, 1) <= 0);

    http_client_response_free(&resp1);
    http_client_response_free(&resp2);
    close(fd1);
    close(fd2);
}

/* --- main ----------------------------------------------------------------- */

int main(void) {
    setenv("MLXD_CACHE_DIR", MLXD_FIXTURES_DIR "/registry_cache", 1);
    unsetenv("MLXD_HF_HUB_DIR");

    test_bind_reports_port();
    test_unknown_route_404();
    test_start_stop_clean();
    test_models_list();
    test_options_preflight();
    test_embeddings_501();
    test_embeddings_invalid_400();
    test_malformed_json_400();
    test_body_too_large_413();
    test_wrong_method_405();
    test_malformed_http_400();
    test_keep_alive_two_requests();
    test_connection_close();
    test_options_preflight_close();
    test_stop_with_open_connections();
    test_destroy_ebusy_warns();
    test_stop_mid_stream_delivers_done();
    test_stop_drain_deadline_bounded();
    test_stop_idle_is_prompt();
    printf("test_http_server: all passed\n");

    unsetenv("MLXD_CACHE_DIR");
    return 0;
}
