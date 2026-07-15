#include "registry/registry_internal.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    fread(buf, 1, (size_t)sz, f);
    buf[sz] = '\0';
    fclose(f);
    *len = (size_t)sz;
    return buf;
}

static void test_parse_file_plan(void) {
    size_t len;
    char *json = read_file(MLXD_FIXTURES_DIR "/hf_api_model.json", &len);
    assert(json != NULL);

    char **files = NULL;
    size_t n = 0;
    int rc = reg_parse_file_plan(json, len, &files, &n);
    assert(rc == 0);
    assert(n == 8);

    const char *expected[] = {
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "generation_config.json", "special_tokens_map.json",
        "chat_template.jinja", "model.safetensors",
        "model.safetensors.index.json"
    };

    for (size_t i = 0; i < 8; i++) {
        int found = 0;
        for (size_t j = 0; j < n; j++) {
            if (strcmp(files[j], expected[i]) == 0) {
                found = 1;
                break;
            }
        }
        assert(found);
    }

    reg_file_plan_free(files, n);
    free(json);
}

static void test_file_wanted_truth_table(void) {
    assert(reg_file_wanted("config.json") == 1);
    assert(reg_file_wanted("tokenizer.json") == 1);
    assert(reg_file_wanted("tokenizer_config.json") == 1);
    assert(reg_file_wanted("generation_config.json") == 1);
    assert(reg_file_wanted("special_tokens_map.json") == 1);
    assert(reg_file_wanted("chat_template.jinja") == 1);
    assert(reg_file_wanted("model.safetensors") == 1);
    assert(reg_file_wanted("model-00001-of-00002.safetensors") == 1);
    assert(reg_file_wanted("model.safetensors.index.json") == 1);
    assert(reg_file_wanted("README.md") == 0);
    assert(reg_file_wanted(".gitattributes") == 0);
    assert(reg_file_wanted("onnx/model.onnx") == 0);
    assert(reg_file_wanted("pytorch_model.bin") == 0);
}

static void test_rfilename_safe(void) {
    assert(reg_rfilename_safe("config.json") == 1);
    assert(reg_rfilename_safe("model.safetensors") == 1);
    assert(reg_rfilename_safe("../evil") == 0);
    assert(reg_rfilename_safe("/etc/passwd") == 0);
    assert(reg_rfilename_safe("sub\\dir\\file.bin") == 0);
    assert(reg_rfilename_safe("foo/../bar") == 0);
    assert(reg_rfilename_safe("") == 0);
}

static void test_empty_siblings(void) {
    const char *json = "{\"siblings\":[]}";
    char **files = NULL;
    size_t n = 99;
    int rc = reg_parse_file_plan(json, strlen(json), &files, &n);
    assert(rc == 0);
    assert(n == 0);
    assert(files == NULL);
}

static void test_malformed_json(void) {
    const char *json = "{broken";
    char **files = NULL;
    size_t n = 0;
    int rc = reg_parse_file_plan(json, strlen(json), &files, &n);
    assert(rc == -1);
}

static void test_missing_siblings(void) {
    const char *json = "{\"sha\":\"abc\"}";
    char **files = NULL;
    size_t n = 0;
    int rc = reg_parse_file_plan(json, strlen(json), &files, &n);
    assert(rc == -1);
}

int main(void) {
    test_parse_file_plan();
    test_file_wanted_truth_table();
    test_rfilename_safe();
    test_empty_siblings();
    test_malformed_json();
    test_missing_siblings();
    printf("test_registry_plan: all passed\n");
    return 0;
}
