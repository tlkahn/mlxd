#include "registry/registry_internal.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
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

static char *save_home(void) {
    const char *h = getenv("HOME");
    return h ? strdup(h) : NULL;
}

static void restore_home(char *saved) {
    if (saved) { setenv("HOME", saved, 1); free(saved); }
    else unsetenv("HOME");
}

static void test_local_path_tilde(void) {
    char *orig = save_home();
    setenv("HOME", "/test/home", 1);
    reg_spec_t s = {0};
    int rc = reg_spec_parse("~/models/foo", &s);
    assert(rc == 0);
    assert(s.org == NULL);
    assert(s.model == NULL);
    assert(strcmp(s.local_path, "/test/home/models/foo") == 0);
    reg_spec_free(&s);
    restore_home(orig);
}

static void test_tilde_bare(void) {
    char *orig = save_home();
    setenv("HOME", "/test/home", 1);
    reg_spec_t s = {0};
    int rc = reg_spec_parse("~", &s);
    assert(rc == 0);
    assert(strcmp(s.local_path, "/test/home") == 0);
    reg_spec_free(&s);
    restore_home(orig);
}

static void test_tilde_user_rejected(void) {
    reg_spec_t s = {0};
    assert(reg_spec_parse("~user/x", &s) == -1);
}

static void test_tilde_no_home(void) {
    char *orig = save_home();
    unsetenv("HOME");
    reg_spec_t s = {0};
    assert(reg_spec_parse("~/foo", &s) == -1);
    restore_home(orig);
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

static void test_charset_guard(void) {
    reg_spec_t s = {0};
    assert(reg_spec_parse("org/model with space", &s) == -1);
    assert(reg_spec_parse("org/model#fragment", &s) == -1);
    assert(reg_spec_parse("org/model%20encoded", &s) == -1);
    assert(reg_spec_parse("org name/model", &s) == -1);
    assert(reg_spec_parse("org/model:rev with space", &s) == -1);
    assert(reg_spec_parse("org/model:rev#x", &s) == -1);

    int rc = reg_spec_parse("org.name/model-v1.2:rev_3", &s);
    assert(rc == 0);
    assert(strcmp(s.org, "org.name") == 0);
    assert(strcmp(s.model, "model-v1.2") == 0);
    assert(strcmp(s.revision, "rev_3") == 0);
    reg_spec_free(&s);
}

int main(void) {
    test_org_model();
    test_org_model_revision();
    test_bare_model_rejected();
    test_local_path_dot();
    test_local_path_slash();
    test_local_path_tilde();
    test_tilde_bare();
    test_tilde_user_rejected();
    test_tilde_no_home();
    test_too_many_slashes();
    test_empty_string();
    test_null_input();
    test_free_null_safe();
    test_slash_model_is_local_path();
    test_empty_model();
    test_empty_revision();
    test_charset_guard();
    printf("test_registry_spec: all passed\n");
    return 0;
}
