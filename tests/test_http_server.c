#include "http/server.h"
#include "http_client.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
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
    usleep(20000);
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
    printf("test_http_server: all passed\n");

    unsetenv("MLXD_CACHE_DIR");
    return 0;
}
