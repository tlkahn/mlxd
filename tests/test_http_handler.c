/* Fault-injection tests for handler.c OOM paths. Compiles a private copy of
 * handler.c with calloc/yyjson_mut_doc_new/yyjson_mut_write redirected through
 * injectable wrappers, following the pattern from test_tok_map.c. */

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yyjson/yyjson.h>

/* Forward-declare fault-injection wrappers before #include-ing handler.c. */
static void *fi_calloc(size_t nmemb, size_t size);
static yyjson_mut_doc *fi_yyjson_mut_doc_new(const yyjson_alc *alc);
static char *fi_yyjson_mut_write(const yyjson_mut_doc *doc,
                                 yyjson_write_flag flg, size_t *len);

#define calloc             fi_calloc
#define yyjson_mut_doc_new fi_yyjson_mut_doc_new
#define yyjson_mut_write   fi_yyjson_mut_write
#define handler_register_all fi_handler_register_all
#include "http/handler.c"
#undef calloc
#undef yyjson_mut_doc_new
#undef yyjson_mut_write

/* --- Fault-injection controls --------------------------------------------- */

static bool fi_fail_calloc;
static bool fi_fail_doc_new;
static bool fi_fail_write;

static void *fi_calloc(size_t nmemb, size_t size) {
    if (fi_fail_calloc) return NULL;
    void *p = malloc(nmemb * size);
    if (p) memset(p, 0, nmemb * size);
    return p;
}

static yyjson_mut_doc *fi_yyjson_mut_doc_new(const yyjson_alc *alc) {
    if (fi_fail_doc_new) return NULL;
    return yyjson_mut_doc_new(alc);
}

static char *fi_yyjson_mut_write(const yyjson_mut_doc *doc,
                                 yyjson_write_flag flg, size_t *len) {
    if (fi_fail_write) return NULL;
    return yyjson_mut_write(doc, flg, len);
}

/* --- Helpers -------------------------------------------------------------- */

static void reset_faults(void) {
    fi_fail_calloc  = false;
    fi_fail_doc_new = false;
    fi_fail_write   = false;
}

/* --- respond_json_error tests --------------------------------------------- */

static void test_respond_json_error_doc_new_oom(void) {
    reset_faults();
    fi_fail_doc_new = true;
    http_response_t resp = {0};
    respond_json_error(&resp, 500, "server_error", NULL, "boom");
    assert(resp.status == 500);
    assert(resp.content_type != NULL);
}

static void test_respond_json_error_write_oom(void) {
    reset_faults();
    fi_fail_write = true;
    http_response_t resp = {0};
    respond_json_error(&resp, 500, "server_error", NULL, "boom");
    assert(resp.status == 500);
    assert(resp.content_type != NULL);
}

static void test_respond_json_error_success(void) {
    reset_faults();
    http_response_t resp = {0};
    respond_json_error(&resp, 400, "invalid_request_error", NULL, "bad input");
    assert(resp.status == 400);
    assert(resp.content_type != NULL);
    assert(strcmp(resp.content_type, "application/json") == 0);
    assert(resp.body != NULL);
    assert(resp.body_len > 0);
    free(resp.body);
}

/* --- handle_models tests -------------------------------------------------- */

static void test_handle_models_calloc_oom(void) {
    reset_faults();
    fi_fail_calloc = true;
    http_request_t req = {.method = "GET", .path = "/v1/models"};
    http_response_t resp = {0};
    handle_models(&req, &resp, NULL);
    assert(resp.status == 500);
    assert(resp.content_type != NULL);
}

static void test_handle_models_doc_new_oom(void) {
    reset_faults();
    fi_fail_doc_new = true;
    http_request_t req = {.method = "GET", .path = "/v1/models"};
    http_response_t resp = {0};
    handle_models(&req, &resp, NULL);
    assert(resp.status == 500);
    assert(resp.content_type != NULL);
}

static void test_handle_models_success(void) {
    reset_faults();
    http_request_t req = {.method = "GET", .path = "/v1/models"};
    http_response_t resp = {0};
    handle_models(&req, &resp, NULL);
    assert(resp.status == 200);
    assert(resp.content_type != NULL);
    assert(strcmp(resp.content_type, "application/json") == 0);
    assert(resp.body != NULL);
    assert(resp.body_len > 0);
    free(resp.body);
}

/* --- main ----------------------------------------------------------------- */

int main(void) {
    setenv("MLXD_CACHE_DIR", MLXD_FIXTURES_DIR "/registry_cache", 1);
    unsetenv("MLXD_HF_HUB_DIR");

    test_respond_json_error_doc_new_oom();
    test_respond_json_error_write_oom();
    test_respond_json_error_success();
    test_handle_models_calloc_oom();
    test_handle_models_doc_new_oom();
    test_handle_models_success();
    printf("test_http_handler: all passed\n");

    unsetenv("MLXD_CACHE_DIR");
    return 0;
}
