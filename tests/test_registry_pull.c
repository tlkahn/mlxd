#include "mock_hub.h"
#include "registry/registry.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <yyjson/yyjson.h>

extern char *mkdtemp(char *);

static char tmpdir_buf[256];

static void make_tmpdir(void) {
    snprintf(tmpdir_buf, sizeof(tmpdir_buf), "%s", "/tmp/mlxd_test_pull_XXXXXX");
    assert(mkdtemp(tmpdir_buf) != NULL);
}

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static char *read_file_str(const char *path) {
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
    return buf;
}

static const char *api_json =
    "{\"sha\":\"commit_abc123\",\"siblings\":["
    "{\"rfilename\":\"config.json\"},"
    "{\"rfilename\":\"model.safetensors\"},"
    "{\"rfilename\":\"README.md\"}"
    "]}";

static int progress_count;
static size_t last_file_count;
static size_t seen_file_indices[16];
static char seen_filenames[16][256];

static int pull_progress_cb(const registry_progress_t *p, void *ud) {
    (void)ud;
    if (progress_count < 16) {
        seen_file_indices[progress_count] = p->file_index;
        if (p->filename)
            snprintf(seen_filenames[progress_count], 256, "%s", p->filename);
        else
            seen_filenames[progress_count][0] = '\0';
    }
    last_file_count = p->file_count;
    progress_count++;
    return 0;
}

static void test_pull_happy(void) {
    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/api/models/testorg/testmodel", 200,
                 api_json, strlen(api_json), 0);
    mock_hub_add(&hub, "GET", "/testorg/testmodel/resolve/main/config.json", 200,
                 "{\"model_type\":\"test\"}", 20, 0);
    mock_hub_add(&hub, "GET", "/testorg/testmodel/resolve/main/model.safetensors", 200,
                 "safetensor_data", 15, 0);
    assert(mock_hub_start(&hub) == 0);

    make_tmpdir();
    char base_url[128];
    mock_hub_base_url(&hub, base_url, sizeof(base_url));
    setenv("MLXD_CACHE_DIR", tmpdir_buf, 1);
    setenv("HF_ENDPOINT", base_url, 1);

    progress_count = 0;
    last_file_count = 0;
    memset(seen_filenames, 0, sizeof(seen_filenames));
    registry_pull_opts_t opts = {0};
    (void)seen_file_indices;
    opts.progress = pull_progress_cb;
    char *dir = registry_pull("testorg/testmodel", &opts);
    assert(dir != NULL);

    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", dir);
    assert(file_exists(config_path));

    char model_path[512];
    snprintf(model_path, sizeof(model_path), "%s/model.safetensors", dir);
    assert(file_exists(model_path));

    char readme_path[512];
    snprintf(readme_path, sizeof(readme_path), "%s/README.md", dir);
    assert(!file_exists(readme_path));

    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s/.mlxd-meta.json", dir);
    assert(file_exists(meta_path));

    char *meta_str = read_file_str(meta_path);
    assert(meta_str != NULL);
    yyjson_doc *meta_doc = yyjson_read(meta_str, strlen(meta_str), 0);
    assert(meta_doc != NULL);
    yyjson_val *meta_root = yyjson_doc_get_root(meta_doc);
    const char *rev = yyjson_get_str(yyjson_obj_get(meta_root, "revision"));
    assert(rev != NULL && strcmp(rev, "main") == 0);
    const char *sha = yyjson_get_str(yyjson_obj_get(meta_root, "commit"));
    assert(sha != NULL && strcmp(sha, "commit_abc123") == 0);
    yyjson_doc_free(meta_doc);
    free(meta_str);

    assert(last_file_count == 2);

    assert(progress_count > 0);
    for (int i = 0; i < progress_count && i < 16; i++) {
        assert(seen_filenames[i][0] != '\0');
        assert(strcmp(seen_filenames[i], "config.json") == 0 ||
               strcmp(seen_filenames[i], "model.safetensors") == 0);
    }

    mock_recorded_t rec;
    mock_hub_get_recorded(&hub, &rec);
    assert(strncmp(rec.ua, "mlxd/", 5) == 0);

    free(dir);
    unsetenv("MLXD_CACHE_DIR");
    unsetenv("HF_ENDPOINT");
    mock_hub_stop(&hub);
}

static int abort_progress_cb(const registry_progress_t *p, void *ud) {
    (void)ud;
    (void)p;
    return 1;
}

static void test_pull_abort(void) {
    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/api/models/org/abortmodel", 200,
                 api_json, strlen(api_json), 0);
    mock_hub_add(&hub, "GET", "/org/abortmodel/resolve/main/config.json", 200,
                 "{\"t\":1}", 6, 0);
    mock_hub_add(&hub, "GET", "/org/abortmodel/resolve/main/model.safetensors", 200,
                 "safetensor_data", 15, 0);
    assert(mock_hub_start(&hub) == 0);

    make_tmpdir();
    char base_url[128];
    mock_hub_base_url(&hub, base_url, sizeof(base_url));
    setenv("MLXD_CACHE_DIR", tmpdir_buf, 1);
    setenv("HF_ENDPOINT", base_url, 1);

    registry_pull_opts_t opts = {0};
    opts.progress = abort_progress_cb;
    char *dir = registry_pull("org/abortmodel", &opts);
    assert(dir == NULL);

    unsetenv("MLXD_CACHE_DIR");
    unsetenv("HF_ENDPOINT");
    mock_hub_stop(&hub);
}

static void test_pull_404(void) {
    mock_hub_t hub;
    mock_hub_init(&hub);
    assert(mock_hub_start(&hub) == 0);

    make_tmpdir();
    char base_url[128];
    mock_hub_base_url(&hub, base_url, sizeof(base_url));
    setenv("MLXD_CACHE_DIR", tmpdir_buf, 1);
    setenv("HF_ENDPOINT", base_url, 1);

    char *dir = registry_pull("org/notfound", NULL);
    assert(dir == NULL);

    unsetenv("MLXD_CACHE_DIR");
    unsetenv("HF_ENDPOINT");
    mock_hub_stop(&hub);
}

static void test_pull_repull_skips_existing(void) {
    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/api/models/org/repull", 200,
                 api_json, strlen(api_json), 0);
    mock_hub_add(&hub, "GET", "/org/repull/resolve/main/config.json", 200,
                 "{\"model_type\":\"test\"}", 20, 0);
    mock_hub_add(&hub, "GET", "/org/repull/resolve/main/model.safetensors", 200,
                 "safetensor_data", 15, 0);
    assert(mock_hub_start(&hub) == 0);

    make_tmpdir();
    char base_url[128];
    mock_hub_base_url(&hub, base_url, sizeof(base_url));
    setenv("MLXD_CACHE_DIR", tmpdir_buf, 1);
    setenv("HF_ENDPOINT", base_url, 1);

    char *dir1 = registry_pull("org/repull", NULL);
    assert(dir1 != NULL);

    char *dir2 = registry_pull("org/repull", NULL);
    assert(dir2 != NULL);
    assert(strcmp(dir1, dir2) == 0);

    free(dir1);
    free(dir2);
    unsetenv("MLXD_CACHE_DIR");
    unsetenv("HF_ENDPOINT");
    mock_hub_stop(&hub);
}

static void test_pull_revision_override(void) {
    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/api/models/org/revmodel/revision/v2", 200,
                 api_json, strlen(api_json), 0);
    mock_hub_add(&hub, "GET", "/org/revmodel/resolve/v2/config.json", 200,
                 "{\"model_type\":\"test\"}", 20, 0);
    mock_hub_add(&hub, "GET", "/org/revmodel/resolve/v2/model.safetensors", 200,
                 "safetensor_data", 15, 0);
    assert(mock_hub_start(&hub) == 0);

    make_tmpdir();
    char base_url[128];
    mock_hub_base_url(&hub, base_url, sizeof(base_url));
    setenv("MLXD_CACHE_DIR", tmpdir_buf, 1);
    setenv("HF_ENDPOINT", base_url, 1);

    registry_pull_opts_t opts = {0};
    opts.revision = "v2";
    char *dir = registry_pull("org/revmodel", &opts);
    assert(dir != NULL);

    mock_recorded_t rec;
    mock_hub_get_recorded(&hub, &rec);
    assert(strstr(rec.path, "/v2/") != NULL);

    free(dir);
    unsetenv("MLXD_CACHE_DIR");
    unsetenv("HF_ENDPOINT");
    mock_hub_stop(&hub);
}

static void test_pull_revision_change_redownloads(void) {
    make_tmpdir();
    setenv("MLXD_CACHE_DIR", tmpdir_buf, 1);

    static const char *api_v1 =
        "{\"sha\":\"sha_v1\",\"siblings\":["
        "{\"rfilename\":\"config.json\"},"
        "{\"rfilename\":\"model.safetensors\"}"
        "]}";
    static const char *api_v2 =
        "{\"sha\":\"sha_v2\",\"siblings\":["
        "{\"rfilename\":\"config.json\"},"
        "{\"rfilename\":\"model.safetensors\"}"
        "]}";

    mock_hub_t hub;
    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/api/models/org/revchange/revision/main", 200,
                 api_v1, strlen(api_v1), 0);
    mock_hub_add(&hub, "GET", "/org/revchange/resolve/main/config.json", 200,
                 "{\"v\":1}", strlen("{\"v\":1}"), 0);
    mock_hub_add(&hub, "GET", "/org/revchange/resolve/main/model.safetensors", 200,
                 "data_v1", strlen("data_v1"), 0);
    assert(mock_hub_start(&hub) == 0);

    char base_url[128];
    mock_hub_base_url(&hub, base_url, sizeof(base_url));
    setenv("HF_ENDPOINT", base_url, 1);

    char *dir1 = registry_pull("org/revchange", NULL);
    assert(dir1 != NULL);

    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", dir1);
    char *c1 = read_file_str(config_path);
    assert(c1 != NULL);
    assert(strcmp(c1, "{\"v\":1}") == 0);
    free(c1);

    mock_hub_stop(&hub);

    mock_hub_init(&hub);
    mock_hub_add(&hub, "GET", "/api/models/org/revchange/revision/v2", 200,
                 api_v2, strlen(api_v2), 0);
    mock_hub_add(&hub, "GET", "/org/revchange/resolve/v2/config.json", 200,
                 "{\"v\":2}", strlen("{\"v\":2}"), 0);
    mock_hub_add(&hub, "GET", "/org/revchange/resolve/v2/model.safetensors", 200,
                 "data_v2", strlen("data_v2"), 0);
    assert(mock_hub_start(&hub) == 0);

    mock_hub_base_url(&hub, base_url, sizeof(base_url));
    setenv("HF_ENDPOINT", base_url, 1);

    char *dir2 = registry_pull("org/revchange:v2", NULL);
    assert(dir2 != NULL);

    char *c2 = read_file_str(config_path);
    assert(c2 != NULL);
    assert(strcmp(c2, "{\"v\":2}") == 0);
    free(c2);

    free(dir1);
    free(dir2);
    unsetenv("MLXD_CACHE_DIR");
    unsetenv("HF_ENDPOINT");
    mock_hub_stop(&hub);
}

int main(void) {
    test_pull_happy();
    test_pull_abort();
    test_pull_404();
    test_pull_repull_skips_existing();
    test_pull_revision_override();
    test_pull_revision_change_redownloads();
    printf("test_registry_pull: all passed\n");
    return 0;
}
