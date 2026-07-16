#include "mock_hub.h"
#include "registry/registry_internal.h"

#include <assert.h>
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
static const char *seen_filename;

static int test_progress_cb(const registry_progress_t *p, void *ud) {
    (void)ud;
    progress_called++;
    last_downloaded = p->file_downloaded;
    seen_filename = p->filename;
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
    seen_filename = NULL;
    int rc = reg_download_file(url, dest, NULL, "testfile.bin",
                               test_progress_cb, NULL, 0, 1);
    assert(rc == 0);

    assert(seen_filename != NULL);
    assert(strcmp(seen_filename, "testfile.bin") == 0);

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

    mock_recorded_t rec;
    mock_hub_get_recorded(&hub, &rec);
    assert(strncmp(rec.ua, "mlxd/", 5) == 0);

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

    int rc = reg_download_file(url, dest, "secret", "auth.bin", NULL, NULL, 0, 1);
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
    int rc = reg_download_file(url, dest, NULL, "noauth.bin", NULL, NULL, 0, 1);
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

    int rc = reg_download_file(url, dest, NULL, "missing.bin", NULL, NULL, 0, 1);
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

    int rc = reg_download_file(url, dest, NULL, "gated.bin", NULL, NULL, 0, 1);
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

    int rc = reg_download_file(url, dest, NULL, "resumed.bin", NULL, NULL, 0, 1);
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

static uint64_t max_file_downloaded;
static uint64_t max_file_total;

static int resume_progress_cb(const registry_progress_t *p, void *ud) {
    (void)ud;
    if (p->file_downloaded > max_file_downloaded)
        max_file_downloaded = p->file_downloaded;
    if (p->file_total > max_file_total)
        max_file_total = p->file_total;
    return 0;
}

static void test_resume_progress_aware(void) {
    const char *full = "ABCDEFGHIJ";
    size_t full_len = 10;

    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/rp", 200, full, full_len, 1);
    assert(mock_hub_start(&hub) == 0);

    make_tmpdir();
    char dest[512];
    snprintf(dest, sizeof(dest), "%s/rp.bin", tmpdir_buf);
    char part[512];
    snprintf(part, sizeof(part), "%s.part", dest);
    write_file(part, "ABCDE", 5);

    char url[256];
    mock_hub_base_url(&hub, url, sizeof(url));
    strcat(url, "/rp");

    max_file_downloaded = 0;
    max_file_total = 0;
    int rc = reg_download_file(url, dest, NULL, "rp.bin",
                               resume_progress_cb, NULL, 0, 1);
    assert(rc == 0);
    assert(max_file_total == 10);
    assert(max_file_downloaded == 10);

    size_t flen;
    char *content = read_file(dest, &flen);
    assert(content != NULL);
    assert(flen == 10);
    assert(memcmp(content, "ABCDEFGHIJ", 10) == 0);
    free(content);

    mock_hub_stop(&hub);
}

static void test_transport_error_preserves_part(void) {
    make_tmpdir();
    char dest[512];
    snprintf(dest, sizeof(dest), "%s/terr.bin", tmpdir_buf);
    char part[512];
    snprintf(part, sizeof(part), "%s.part", dest);
    write_file(part, "partial", 7);

    int rc = reg_download_file("http://127.0.0.1:1/x", dest, NULL, "terr.bin",
                               NULL, NULL, 0, 1);
    assert(rc == -1);

    struct stat st;
    assert(stat(part, &st) == 0);
    assert(st.st_size == 7);
}

static void test_200_on_resume(void) {
    const char *full = "ABCDEFGHIJ";
    size_t full_len = 10;

    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/noresume", 200, full, full_len, 0);
    assert(mock_hub_start(&hub) == 0);

    make_tmpdir();
    char dest[512];
    snprintf(dest, sizeof(dest), "%s/nr.bin", tmpdir_buf);
    char part[512];
    snprintf(part, sizeof(part), "%s.part", dest);
    write_file(part, "ABCDE", 5);

    char url[256];
    mock_hub_base_url(&hub, url, sizeof(url));
    strcat(url, "/noresume");

    max_file_downloaded = 0;
    max_file_total = 0;
    int rc = reg_download_file(url, dest, NULL, "nr.bin",
                               resume_progress_cb, NULL, 0, 1);
    assert(rc == 0);

    size_t flen;
    char *content = read_file(dest, &flen);
    assert(content != NULL);
    assert(flen == 10);
    assert(memcmp(content, "ABCDEFGHIJ", 10) == 0);
    free(content);

    assert(max_file_total == 0 || max_file_downloaded <= max_file_total);

    mock_hub_stop(&hub);
}

static void test_416_oversized_part_retries(void) {
    const char *full = "COMPLETE";
    size_t full_len = 8;

    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/sized", 200, full, full_len, 1);
    assert(mock_hub_start(&hub) == 0);

    make_tmpdir();
    char dest[512];
    snprintf(dest, sizeof(dest), "%s/sized.bin", tmpdir_buf);
    char part[512];
    snprintf(part, sizeof(part), "%s.part", dest);
    write_file(part, "OVERSIZE_BAD", 12);

    char url[256];
    mock_hub_base_url(&hub, url, sizeof(url));
    strcat(url, "/sized");

    int rc = reg_download_file(url, dest, NULL, "sized.bin", NULL, NULL, 0, 1);
    assert(rc == 0);

    size_t flen;
    char *content = read_file(dest, &flen);
    assert(content != NULL);
    assert(flen == 8);
    assert(memcmp(content, "COMPLETE", 8) == 0);
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

    int rc = reg_download_file(url, dest, NULL, "done.bin", NULL, NULL, 0, 1);
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
    test_happy_path();
    test_auth_header();
    test_no_token_no_auth();
    test_404_error();
    test_401_error();
    test_resume_progress_aware();
    test_transport_error_preserves_part();
    test_200_on_resume();
    test_416_oversized_part_retries();
    test_resume_with_range();
    test_resume_416_complete();
    printf("test_registry_download: all passed\n");
    return 0;
}
