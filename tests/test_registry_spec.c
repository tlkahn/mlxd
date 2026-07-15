#include "registry/registry_internal.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void test_org_model(void) {
    reg_spec_t s = {0};
    int rc = reg_spec_parse("mlx-community/Qwen3-0.6B-4bit", &s);
    assert(rc == 0);
    assert(strcmp(s.org, "mlx-community") == 0);
    assert(strcmp(s.model, "Qwen3-0.6B-4bit") == 0);
    assert(s.revision == NULL);
    assert(s.local_path == NULL);
    reg_spec_free(&s);
}

static void test_org_model_revision(void) {
    reg_spec_t s = {0};
    int rc = reg_spec_parse("mlx-community/Qwen3-0.6B-4bit:main", &s);
    assert(rc == 0);
    assert(strcmp(s.org, "mlx-community") == 0);
    assert(strcmp(s.model, "Qwen3-0.6B-4bit") == 0);
    assert(strcmp(s.revision, "main") == 0);
    assert(s.local_path == NULL);
    reg_spec_free(&s);
}

static void test_bare_model_rejected(void) {
    reg_spec_t s = {0};
    assert(reg_spec_parse("model", &s) == -1);
}

static void test_local_path_dot(void) {
    reg_spec_t s = {0};
    int rc = reg_spec_parse("./my-model", &s);
    assert(rc == 0);
    assert(s.org == NULL);
    assert(s.model == NULL);
    assert(s.revision == NULL);
    assert(strcmp(s.local_path, "./my-model") == 0);
    reg_spec_free(&s);
}

static void test_local_path_slash(void) {
    reg_spec_t s = {0};
    int rc = reg_spec_parse("/tmp/my-model", &s);
    assert(rc == 0);
    assert(strcmp(s.local_path, "/tmp/my-model") == 0);
    reg_spec_free(&s);
}

static void test_local_path_tilde(void) {
    reg_spec_t s = {0};
    int rc = reg_spec_parse("~/models/foo", &s);
    assert(rc == 0);
    assert(strcmp(s.local_path, "~/models/foo") == 0);
    reg_spec_free(&s);
}

static void test_too_many_slashes(void) {
    reg_spec_t s = {0};
    assert(reg_spec_parse("a/b/c", &s) == -1);
}

static void test_empty_string(void) {
    reg_spec_t s = {0};
    assert(reg_spec_parse("", &s) == -1);
}

static void test_null_input(void) {
    reg_spec_t s = {0};
    assert(reg_spec_parse(NULL, &s) == -1);
}

static void test_free_null_safe(void) {
    reg_spec_free(NULL);
}

static void test_slash_model_is_local_path(void) {
    reg_spec_t s = {0};
    int rc = reg_spec_parse("/model", &s);
    assert(rc == 0);
    assert(strcmp(s.local_path, "/model") == 0);
    reg_spec_free(&s);
}

static void test_empty_model(void) {
    reg_spec_t s = {0};
    assert(reg_spec_parse("org/", &s) == -1);
}

static void test_empty_revision(void) {
    reg_spec_t s = {0};
    assert(reg_spec_parse("org/model:", &s) == -1);
}

int main(void) {
    test_org_model();
    test_org_model_revision();
    test_bare_model_rejected();
    test_local_path_dot();
    test_local_path_slash();
    test_local_path_tilde();
    test_too_many_slashes();
    test_empty_string();
    test_null_input();
    test_free_null_safe();
    test_slash_model_is_local_path();
    test_empty_model();
    test_empty_revision();
    printf("test_registry_spec: all passed\n");
    return 0;
}
