#include "mock_hub.h"

#include <assert.h>
#include <curl/curl.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    char  buf[4096];
    size_t len;
} membuf_t;

static size_t write_cb(void *data, size_t size, size_t nmemb, void *userdata) {
    membuf_t *m = (membuf_t *)userdata;
    size_t total = size * nmemb;
    if (m->len + total < sizeof(m->buf)) {
        memcpy(m->buf + m->len, data, total);
        m->len += total;
        m->buf[m->len] = '\0';
    }
    return total;
}

static void test_basic_get(void) {
    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/hello", 200, "world", 5, 0);
    assert(mock_hub_start(&hub) == 0);

    char url[128];
    mock_hub_base_url(&hub, url, sizeof(url));
    strcat(url, "/hello");

    CURL *c = curl_easy_init();
    membuf_t mem = {0};
    curl_easy_setopt(c, CURLOPT_URL, url);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &mem);
    CURLcode res = curl_easy_perform(c);
    assert(res == CURLE_OK);

    long code;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
    assert(code == 200);
    assert(mem.len == 5);
    assert(memcmp(mem.buf, "world", 5) == 0);

    curl_easy_cleanup(c);

    mock_recorded_t rec;
    mock_hub_get_recorded(&hub, &rec);
    assert(strcmp(rec.path, "/hello") == 0);

    mock_hub_stop(&hub);
}

static void test_auth_header_recorded(void) {
    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/auth", 200, "ok", 2, 0);
    assert(mock_hub_start(&hub) == 0);

    char url[128];
    mock_hub_base_url(&hub, url, sizeof(url));
    strcat(url, "/auth");

    CURL *c = curl_easy_init();
    membuf_t mem = {0};
    struct curl_slist *hdrs = curl_slist_append(NULL, "Authorization: Bearer secret123");
    curl_easy_setopt(c, CURLOPT_URL, url);
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &mem);
    curl_easy_perform(c);
    curl_slist_free_all(hdrs);
    curl_easy_cleanup(c);

    mock_recorded_t rec;
    mock_hub_get_recorded(&hub, &rec);
    assert(strcmp(rec.auth, "Bearer secret123") == 0);

    mock_hub_stop(&hub);
}

static void test_404_no_route(void) {
    mock_hub_t hub;
    mock_hub_init(&hub);
    assert(mock_hub_start(&hub) == 0);

    char url[128];
    mock_hub_base_url(&hub, url, sizeof(url));
    strcat(url, "/noroute");

    CURL *c = curl_easy_init();
    membuf_t mem = {0};
    curl_easy_setopt(c, CURLOPT_URL, url);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &mem);
    curl_easy_perform(c);

    long code;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
    assert(code == 404);
    curl_easy_cleanup(c);

    mock_hub_stop(&hub);
}

int main(void) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    test_basic_get();
    test_auth_header_recorded();
    test_404_no_route();
    curl_global_cleanup();
    printf("test_registry_mockhub: all passed\n");
    return 0;
}
