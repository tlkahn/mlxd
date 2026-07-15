#include "mock_hub.h"
#include "registry/registry_internal.h"

#include <assert.h>
#include <curl/curl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

extern char *mkdtemp(char *);

static char tmpdir_buf[256];

static void make_tmpdir(void) {
    snprintf(tmpdir_buf, sizeof(tmpdir_buf), "%s", "/tmp/mlxd_test_dl_XXXXXX");
    assert(mkdtemp(tmpdir_buf) != NULL);
}

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, (size_t)sz, f);
    buf[sz] = '\0';
    fclose(f);
    if (len) *len = (size_t)sz;
    return buf;
}

static int progress_called;
static uint64_t last_downloaded;

static int test_progress_cb(const registry_progress_t *p, void *ud) {
    (void)ud;
    progress_called++;
    last_downloaded = p->file_downloaded;
    return 0;
}

static void test_happy_path(void) {
    const char *body = "hello world download content";
    size_t body_len = strlen(body);

    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/file", 200, body, body_len, 0);
    assert(mock_hub_start(&hub) == 0);

    make_tmpdir();
    char url[256];
    mock_hub_base_url(&hub, url, sizeof(url));
    strcat(url, "/file");

    char dest[512];
    snprintf(dest, sizeof(dest), "%s/testfile.bin", tmpdir_buf);

    progress_called = 0;
    last_downloaded = 0;
    int rc = reg_download_file(url, dest, NULL, test_progress_cb, NULL, 0, 1);
    assert(rc == 0);

    size_t flen;
    char *content = read_file(dest, &flen);
    assert(content != NULL);
    assert(flen == body_len);
    assert(memcmp(content, body, body_len) == 0);
    free(content);

    struct stat st;
    char part[512];
    snprintf(part, sizeof(part), "%s.part", dest);
    assert(stat(part, &st) != 0);

    mock_hub_stop(&hub);
}

static void test_auth_header(void) {
    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/auth", 200, "ok", 2, 0);
    assert(mock_hub_start(&hub) == 0);

    make_tmpdir();
    char url[256];
    mock_hub_base_url(&hub, url, sizeof(url));
    strcat(url, "/auth");

    char dest[512];
    snprintf(dest, sizeof(dest), "%s/auth.bin", tmpdir_buf);

    int rc = reg_download_file(url, dest, "secret", NULL, NULL, 0, 1);
    assert(rc == 0);

    mock_recorded_t rec;
    mock_hub_get_recorded(&hub, &rec);
    assert(strcmp(rec.auth, "Bearer secret") == 0);

    mock_hub_stop(&hub);
}

static void test_no_token_no_auth(void) {
    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/noauth", 200, "ok", 2, 0);
    assert(mock_hub_start(&hub) == 0);

    make_tmpdir();
    char url[256];
    mock_hub_base_url(&hub, url, sizeof(url));
    strcat(url, "/noauth");

    char dest[512];
    snprintf(dest, sizeof(dest), "%s/noauth.bin", tmpdir_buf);

    unsetenv("HF_TOKEN");
    int rc = reg_download_file(url, dest, NULL, NULL, NULL, 0, 1);
    assert(rc == 0);

    mock_recorded_t rec;
    mock_hub_get_recorded(&hub, &rec);
    assert(rec.auth[0] == '\0');

    mock_hub_stop(&hub);
}

static void test_404_error(void) {
    mock_hub_t hub;
    mock_hub_init(&hub);
    assert(mock_hub_start(&hub) == 0);

    make_tmpdir();
    char url[256];
    mock_hub_base_url(&hub, url, sizeof(url));
    strcat(url, "/missing");

    char dest[512];
    snprintf(dest, sizeof(dest), "%s/missing.bin", tmpdir_buf);

    int rc = reg_download_file(url, dest, NULL, NULL, NULL, 0, 1);
    assert(rc == -1);

    struct stat st;
    char part[512];
    snprintf(part, sizeof(part), "%s.part", dest);
    assert(stat(part, &st) != 0);

    mock_hub_stop(&hub);
}

static void test_401_error(void) {
    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/gated", 401, "", 0, 0);
    assert(mock_hub_start(&hub) == 0);

    make_tmpdir();
    char url[256];
    mock_hub_base_url(&hub, url, sizeof(url));
    strcat(url, "/gated");

    char dest[512];
    snprintf(dest, sizeof(dest), "%s/gated.bin", tmpdir_buf);

    int rc = reg_download_file(url, dest, NULL, NULL, NULL, 0, 1);
    assert(rc == -1);

    mock_hub_stop(&hub);
}

static void write_file(const char *path, const char *data, size_t len) {
    FILE *f = fopen(path, "wb");
    assert(f != NULL);
    fwrite(data, 1, len, f);
    fclose(f);
}

static void test_resume_with_range(void) {
    const char *full = "ABCDEFGHIJ";
    size_t full_len = 10;

    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/resumable", 200, full, full_len, 1);
    assert(mock_hub_start(&hub) == 0);

    make_tmpdir();
    char dest[512];
    snprintf(dest, sizeof(dest), "%s/resumed.bin", tmpdir_buf);
    char part[512];
    snprintf(part, sizeof(part), "%s.part", dest);

    write_file(part, "ABCDE", 5);

    char url[256];
    mock_hub_base_url(&hub, url, sizeof(url));
    strcat(url, "/resumable");

    int rc = reg_download_file(url, dest, NULL, NULL, NULL, 0, 1);
    assert(rc == 0);

    mock_recorded_t rec;
    mock_hub_get_recorded(&hub, &rec);
    assert(strstr(rec.range, "bytes=5-") != NULL);

    size_t flen;
    char *content = read_file(dest, &flen);
    assert(content != NULL);
    assert(flen == 10);
    assert(memcmp(content, "ABCDEFGHIJ", 10) == 0);
    free(content);

    mock_hub_stop(&hub);
}

static void test_resume_416_complete(void) {
    const char *full = "COMPLETE";
    size_t full_len = 8;

    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/done", 200, full, full_len, 1);
    assert(mock_hub_start(&hub) == 0);

    make_tmpdir();
    char dest[512];
    snprintf(dest, sizeof(dest), "%s/done.bin", tmpdir_buf);
    char part[512];
    snprintf(part, sizeof(part), "%s.part", dest);

    write_file(part, "COMPLETE", 8);

    char url[256];
    mock_hub_base_url(&hub, url, sizeof(url));
    strcat(url, "/done");

    int rc = reg_download_file(url, dest, NULL, NULL, NULL, 0, 1);
    assert(rc == 0);

    size_t flen;
    char *content = read_file(dest, &flen);
    assert(content != NULL);
    assert(flen == 8);
    assert(memcmp(content, "COMPLETE", 8) == 0);
    free(content);

    mock_hub_stop(&hub);
}

int main(void) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    test_happy_path();
    test_auth_header();
    test_no_token_no_auth();
    test_404_error();
    test_401_error();
    test_resume_with_range();
    test_resume_416_complete();
    curl_global_cleanup();
    printf("test_registry_download: all passed\n");
    return 0;
}
