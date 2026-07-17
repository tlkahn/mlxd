#include "model/chat_template.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_jinja_file(void) {
    char *tmpl = chat_template_load(MLXD_FIXTURES_DIR "/chat_tmpl_jinja");
    assert(tmpl != NULL);
    assert(strstr(tmpl, "messages") != NULL);
    free(tmpl);
}

static void test_config_string(void) {
    char *tmpl = chat_template_load(MLXD_FIXTURES_DIR "/chat_tmpl_config");
    assert(tmpl != NULL);
    assert(strcmp(tmpl, "Hello from config") == 0);
    free(tmpl);
}

static void test_config_array_default(void) {
    char *tmpl = chat_template_load(MLXD_FIXTURES_DIR "/chat_tmpl_config_array");
    assert(tmpl != NULL);
    assert(strcmp(tmpl, "default tmpl") == 0);
    free(tmpl);
}

static void test_config_array_first(void) {
    char *tmpl = chat_template_load(MLXD_FIXTURES_DIR "/chat_tmpl_config_array_nofirst");
    assert(tmpl != NULL);
    assert(strcmp(tmpl, "tool only") == 0);
    free(tmpl);
}

static void test_absent(void) {
    char *tmpl = chat_template_load(MLXD_FIXTURES_DIR "/chat_tmpl_none");
    assert(tmpl == NULL);
}

static void test_null_dir(void) {
    char *tmpl = chat_template_load(NULL);
    assert(tmpl == NULL);
}

static void test_corrupt_json(void) {
    char *tmpl = chat_template_load(MLXD_FIXTURES_DIR "/chat_tmpl_corrupt");
    assert(tmpl == NULL);
}

int main(void) {
    test_jinja_file();
    test_config_string();
    test_config_array_default();
    test_config_array_first();
    test_absent();
    test_null_dir();
    test_corrupt_json();
    printf("test_chat_template: all tests passed\n");
    return 0;
}
