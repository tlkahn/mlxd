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

/* --- main ----------------------------------------------------------------- */

int main(void) {
    test_bind_reports_port();
    test_unknown_route_404();
    test_start_stop_clean();
    printf("test_http_server: all passed\n");
    return 0;
}
