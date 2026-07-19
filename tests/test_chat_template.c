#include "model/chat_template.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

extern char *mkdtemp(char *);

static void write_file(const char *dir, const char *name, const char *content) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "w");
    assert(f);
    fwrite(content, 1, strlen(content), f);
    fclose(f);
}

static void unlink_file(const char *dir, const char *name) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    unlink(path);
}

static void test_tokenizer_config_wins(void) {
    char tmpdir[] = "/tmp/mlxd_ct_a_XXXXXX";
    assert(mkdtemp(tmpdir) != NULL);

    write_file(tmpdir, "tokenizer_config.json",
               "{\"chat_template\": \"from_config\"}");
    write_file(tmpdir, "chat_template.jinja", "from_jinja");

    char *result = model_chat_template_read(tmpdir);
    assert(result != NULL);
    assert(strcmp(result, "from_config") == 0);
    free(result);

    unlink_file(tmpdir, "tokenizer_config.json");
    unlink_file(tmpdir, "chat_template.jinja");
    rmdir(tmpdir);
}

static void test_jinja_fallback(void) {
    char tmpdir[] = "/tmp/mlxd_ct_b_XXXXXX";
    assert(mkdtemp(tmpdir) != NULL);

    write_file(tmpdir, "tokenizer_config.json", "{\"other_key\": 42}");
    write_file(tmpdir, "chat_template.jinja", "jinja_content_here");

    char *result = model_chat_template_read(tmpdir);
    assert(result != NULL);
    assert(strcmp(result, "jinja_content_here") == 0);
    free(result);

    unlink_file(tmpdir, "tokenizer_config.json");
    unlink_file(tmpdir, "chat_template.jinja");
    rmdir(tmpdir);
}

static void test_json_string_fallback(void) {
    char tmpdir[] = "/tmp/mlxd_ct_c_XXXXXX";
    assert(mkdtemp(tmpdir) != NULL);

    write_file(tmpdir, "chat_template.json", "\"json_string_template\"");

    char *result = model_chat_template_read(tmpdir);
    assert(result != NULL);
    assert(strcmp(result, "json_string_template") == 0);
    free(result);

    unlink_file(tmpdir, "chat_template.json");
    rmdir(tmpdir);
}

static void test_none_returns_null(void) {
    char tmpdir[] = "/tmp/mlxd_ct_d_XXXXXX";
    assert(mkdtemp(tmpdir) != NULL);

    char *result = model_chat_template_read(tmpdir);
    assert(result == NULL);

    rmdir(tmpdir);
}

static void test_null_dir_returns_null(void) {
    char *result = model_chat_template_read(NULL);
    assert(result == NULL);
}

static void test_jinja_only_no_tokenizer_config(void) {
    char tmpdir[] = "/tmp/mlxd_ct_e_XXXXXX";
    assert(mkdtemp(tmpdir) != NULL);

    write_file(tmpdir, "chat_template.jinja", "{% for m in messages %}{{ m.content }}{% endfor %}");

    char *result = model_chat_template_read(tmpdir);
    assert(result != NULL);
    assert(strstr(result, "messages") != NULL);
    free(result);

    unlink_file(tmpdir, "chat_template.jinja");
    rmdir(tmpdir);
}

static void test_tokenizer_config_non_string_template(void) {
    char tmpdir[] = "/tmp/mlxd_ct_f_XXXXXX";
    assert(mkdtemp(tmpdir) != NULL);

    write_file(tmpdir, "tokenizer_config.json",
               "{\"chat_template\": 42}");
    write_file(tmpdir, "chat_template.jinja", "jinja_wins");

    char *result = model_chat_template_read(tmpdir);
    assert(result != NULL);
    assert(strcmp(result, "jinja_wins") == 0);
    free(result);

    unlink_file(tmpdir, "tokenizer_config.json");
    unlink_file(tmpdir, "chat_template.jinja");
    rmdir(tmpdir);
}

int main(void) {
    test_tokenizer_config_wins();
    printf("  test_tokenizer_config_wins: passed\n");

    test_jinja_fallback();
    printf("  test_jinja_fallback: passed\n");

    test_json_string_fallback();
    printf("  test_json_string_fallback: passed\n");

    test_none_returns_null();
    printf("  test_none_returns_null: passed\n");

    test_null_dir_returns_null();
    printf("  test_null_dir_returns_null: passed\n");

    test_jinja_only_no_tokenizer_config();
    printf("  test_jinja_only_no_tokenizer_config: passed\n");

    test_tokenizer_config_non_string_template();
    printf("  test_tokenizer_config_non_string_template: passed\n");

    printf("test_chat_template: all passed\n");
    return 0;
}
