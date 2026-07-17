#include "cli/cli.h"
#include "cli/cmds.h"

#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "yyjson/yyjson.h"

/* --- Phase B: usage/version strings --------------------------------------- */

static void test_usage_contains_subcommands(void) {
    const char *u = cli_usage_str();
    assert(strstr(u, "serve") != NULL);
    assert(strstr(u, "run") != NULL);
    assert(strstr(u, "pull") != NULL);
    assert(strstr(u, "list") != NULL);
}

static void test_usage_documents_options(void) {
    const char *u = cli_usage_str();
    assert(strstr(u, "--port") != NULL);
    assert(strstr(u, "--json") != NULL);
    assert(strstr(u, "--max-tokens") != NULL);
    assert(strstr(u, "MODEL") != NULL);
    assert(strstr(u, "PROMPT") != NULL);
}

static void test_version_string(void) {
    assert(strlen(MLXD_VERSION) > 0);
    assert(strstr(MLXD_VERSION, ".") != NULL);
}

/* --- Phase B: cli_list_json ----------------------------------------------- */

static void test_list_json_two_models(void) {
    registry_model_info_t models[2] = {
        {.id = "org/model-a", .path = "/cache/a", .size_bytes = 1024, .mtime = 1700000000},
        {.id = "org/model-b", .path = "/cache/b", .size_bytes = 2048, .mtime = 1700000001},
    };
    char *json = cli_list_json(models, 2);
    assert(json != NULL);

    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(yyjson_is_arr(root));
    assert(yyjson_arr_size(root) == 2);

    yyjson_val *first = yyjson_arr_get_first(root);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(first, "id")), "org/model-a") == 0);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(first, "path")), "/cache/a") == 0);
    assert(yyjson_get_uint(yyjson_obj_get(first, "size_bytes")) == 1024);
    assert(yyjson_get_sint(yyjson_obj_get(first, "mtime")) == 1700000000);

    yyjson_val *second = yyjson_arr_get(root, 1);
    assert(strcmp(yyjson_get_str(yyjson_obj_get(second, "id")), "org/model-b") == 0);

    yyjson_doc_free(doc);
    free(json);
}

static void test_list_json_empty(void) {
    char *json = cli_list_json(NULL, 0);
    assert(json != NULL);

    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(yyjson_is_arr(root));
    assert(yyjson_arr_size(root) == 0);

    yyjson_doc_free(doc);
    free(json);
}

/* --- Phase C: cli_cmd_list ------------------------------------------------- */

#define FIXTURE_CACHE MLXD_FIXTURES_DIR "/registry_cache"

static void test_cmd_list_table(void) {
    setenv("MLXD_CACHE_DIR", FIXTURE_CACHE, 1);
    setenv("MLXD_HF_HUB_DIR", "/nonexistent", 1);

    char buf[8192] = {0};
    char errbuf[4096] = {0};
    FILE *out = fmemopen(buf, sizeof(buf), "w");
    FILE *err_f = fmemopen(errbuf, sizeof(errbuf), "w");
    cli_args_t args = {0};
    args.cmd = CLI_LIST;
    args.list.json = false;

    int rc = cli_cmd_list(&args, out, err_f);
    fclose(out);
    fclose(err_f);

    assert(rc == 0);
    assert(strstr(buf, "MODEL") != NULL);
    assert(strstr(buf, "mlx-community/Qwen3-0.6B-4bit") != NULL);

    unsetenv("MLXD_CACHE_DIR");
    unsetenv("MLXD_HF_HUB_DIR");
}

static void test_cmd_list_json(void) {
    setenv("MLXD_CACHE_DIR", FIXTURE_CACHE, 1);
    setenv("MLXD_HF_HUB_DIR", "/nonexistent", 1);

    char buf[8192] = {0};
    FILE *out = fmemopen(buf, sizeof(buf), "w");
    FILE *err_f = fmemopen(NULL, 1024, "w");
    cli_args_t args = {0};
    args.cmd = CLI_LIST;
    args.list.json = true;

    int rc = cli_cmd_list(&args, out, err_f);
    fclose(out);
    fclose(err_f);

    assert(rc == 0);
    yyjson_doc *doc = yyjson_read(buf, strlen(buf), 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(yyjson_is_arr(root));
    assert(yyjson_arr_size(root) >= 1);

    bool found = false;
    yyjson_val *val;
    yyjson_arr_iter iter;
    yyjson_arr_iter_init(root, &iter);
    while ((val = yyjson_arr_iter_next(&iter))) {
        yyjson_val *id_val = yyjson_obj_get(val, "id");
        if (id_val && strcmp(yyjson_get_str(id_val), "mlx-community/Qwen3-0.6B-4bit") == 0)
            found = true;
    }
    assert(found);
    yyjson_doc_free(doc);

    unsetenv("MLXD_CACHE_DIR");
    unsetenv("MLXD_HF_HUB_DIR");
}

static void test_cmd_list_empty(void) {
    setenv("MLXD_CACHE_DIR", "/nonexistent/path", 1);
    setenv("MLXD_HF_HUB_DIR", "/nonexistent", 1);

    char buf[4096] = {0};
    char errbuf[4096] = {0};
    FILE *out = fmemopen(buf, sizeof(buf), "w");
    FILE *err_f = fmemopen(errbuf, sizeof(errbuf), "w");
    cli_args_t args = {0};
    args.cmd = CLI_LIST;
    args.list.json = false;

    int rc = cli_cmd_list(&args, out, err_f);
    fclose(out);
    fclose(err_f);

    assert(rc == 0);
    assert(strstr(errbuf, "no models found") != NULL);

    unsetenv("MLXD_CACHE_DIR");
    unsetenv("MLXD_HF_HUB_DIR");
}

static void test_cmd_list_empty_json(void) {
    setenv("MLXD_CACHE_DIR", "/nonexistent/path", 1);
    setenv("MLXD_HF_HUB_DIR", "/nonexistent", 1);

    char buf[4096] = {0};
    FILE *out = fmemopen(buf, sizeof(buf), "w");
    FILE *err_f = fmemopen(NULL, 1024, "w");
    cli_args_t args = {0};
    args.cmd = CLI_LIST;
    args.list.json = true;

    int rc = cli_cmd_list(&args, out, err_f);
    fclose(out);
    fclose(err_f);

    assert(rc == 0);
    yyjson_doc *doc = yyjson_read(buf, strlen(buf), 0);
    assert(doc != NULL);
    yyjson_val *root = yyjson_doc_get_root(doc);
    assert(yyjson_is_arr(root));
    assert(yyjson_arr_size(root) == 0);
    yyjson_doc_free(doc);

    unsetenv("MLXD_CACHE_DIR");
    unsetenv("MLXD_HF_HUB_DIR");
}

/* --- Phase D: cli_cmd_run ------------------------------------------------- */

#define RUN_MODEL MLXD_FIXTURES_DIR "/run_model"

static void test_cmd_run_oneshot(void) {
    char buf[8192] = {0};
    char errbuf[4096] = {0};
    FILE *out = fmemopen(buf, sizeof(buf), "w");
    FILE *err_f = fmemopen(errbuf, sizeof(errbuf), "w");
    FILE *in_f = fmemopen(NULL, 1, "r");

    cli_args_t args = {0};
    args.cmd = CLI_RUN;
    args.run.model = RUN_MODEL;
    args.run.prompt = "hello world";
    args.run.max_tokens = 256;
    args.run.temperature = 1.0f;
    args.run.stream = false;

    int rc = cli_cmd_run(&args, in_f, out, err_f);
    fclose(out);
    fclose(err_f);
    fclose(in_f);

    assert(rc == 0);
    assert(strstr(buf, "hello world") != NULL);
}

static void test_cmd_run_streaming(void) {
    char buf[8192] = {0};
    char errbuf[4096] = {0};
    FILE *out = fmemopen(buf, sizeof(buf), "w");
    FILE *err_f = fmemopen(errbuf, sizeof(errbuf), "w");
    FILE *in_f = fmemopen(NULL, 1, "r");

    cli_args_t args = {0};
    args.cmd = CLI_RUN;
    args.run.model = RUN_MODEL;
    args.run.prompt = "hello world";
    args.run.max_tokens = 256;
    args.run.temperature = 1.0f;
    args.run.stream = true;

    int rc = cli_cmd_run(&args, in_f, out, err_f);
    fclose(out);
    fclose(err_f);
    fclose(in_f);

    assert(rc == 0);
    assert(strstr(buf, "hello world") != NULL);
}

static void test_cmd_run_stdin(void) {
    char buf[8192] = {0};
    char errbuf[4096] = {0};
    FILE *out = fmemopen(buf, sizeof(buf), "w");
    FILE *err_f = fmemopen(errbuf, sizeof(errbuf), "w");

    char input[] = "hi there\n";
    FILE *in_f = fmemopen(input, strlen(input), "r");

    cli_args_t args = {0};
    args.cmd = CLI_RUN;
    args.run.model = RUN_MODEL;
    args.run.prompt = NULL;
    args.run.max_tokens = 256;
    args.run.temperature = 1.0f;
    args.run.stream = false;

    int rc = cli_cmd_run(&args, in_f, out, err_f);
    fclose(out);
    fclose(err_f);
    fclose(in_f);

    assert(rc == 0);
    assert(strstr(buf, "hi there") != NULL);
}

static void test_cmd_run_bad_model(void) {
    setenv("MLXD_CACHE_DIR", "/nonexistent", 1);
    setenv("MLXD_HF_HUB_DIR", "/nonexistent", 1);

    char errbuf[4096] = {0};
    FILE *out = fmemopen(NULL, 1024, "w");
    FILE *err_f = fmemopen(errbuf, sizeof(errbuf), "w");
    FILE *in_f = fmemopen(NULL, 1, "r");

    cli_args_t args = {0};
    args.cmd = CLI_RUN;
    args.run.model = "no/such-model";
    args.run.prompt = "hello";
    args.run.max_tokens = 256;
    args.run.temperature = 1.0f;

    int rc = cli_cmd_run(&args, in_f, out, err_f);
    fclose(out);
    fclose(err_f);
    fclose(in_f);

    assert(rc != 0);

    unsetenv("MLXD_CACHE_DIR");
    unsetenv("MLXD_HF_HUB_DIR");
}

/* --- Phase F: cli_sigint_decide ------------------------------------------- */

static void test_sigint_decide(void) {
    assert(cli_sigint_decide(1) == SIGINT_STOP);
    assert(cli_sigint_decide(2) == SIGINT_FORCE_EXIT);
    assert(cli_sigint_decide(3) == SIGINT_FORCE_EXIT);
}

/* --- Phase F: cli_main dispatch ------------------------------------------- */

static void test_main_help(void) {
    char buf[4096] = {0};
    FILE *out = fmemopen(buf, sizeof(buf), "w");
    FILE *err_f = fmemopen(NULL, 1024, "w");
    char *argv[] = {"mlxd", "--help"};
    int rc = cli_main(2, argv, stdin, out, err_f);
    fclose(out);
    fclose(err_f);
    assert(rc == 0);
    assert(strstr(buf, "serve") != NULL);
}

static void test_main_version(void) {
    char buf[4096] = {0};
    FILE *out = fmemopen(buf, sizeof(buf), "w");
    FILE *err_f = fmemopen(NULL, 1024, "w");
    char *argv[] = {"mlxd", "--version"};
    int rc = cli_main(2, argv, stdin, out, err_f);
    fclose(out);
    fclose(err_f);
    assert(rc == 0);
    assert(strstr(buf, MLXD_VERSION) != NULL);
}

static void test_main_unknown_command(void) {
    char errbuf[4096] = {0};
    FILE *out = fmemopen(NULL, 1024, "w");
    FILE *err_f = fmemopen(errbuf, sizeof(errbuf), "w");
    char *argv[] = {"mlxd", "bogus"};
    int rc = cli_main(2, argv, stdin, out, err_f);
    fclose(out);
    fclose(err_f);
    assert(rc == 1);
    assert(strstr(errbuf, "bogus") != NULL);
    assert(strstr(errbuf, "usage") != NULL || strstr(errbuf, "serve") != NULL);
}

static void test_main_serve_help(void) {
    char buf[4096] = {0};
    FILE *out = fmemopen(buf, sizeof(buf), "w");
    FILE *err_f = fmemopen(NULL, 1024, "w");
    char *argv[] = {"mlxd", "serve", "--help"};
    int rc = cli_main(3, argv, stdin, out, err_f);
    fclose(out);
    fclose(err_f);
    assert(rc == 0);
    assert(strstr(buf, "serve") != NULL);
}

static void test_main_no_args(void) {
    char errbuf[4096] = {0};
    FILE *out = fmemopen(NULL, 1024, "w");
    FILE *err_f = fmemopen(errbuf, sizeof(errbuf), "w");
    char *argv[] = {"mlxd"};
    int rc = cli_main(1, argv, stdin, out, err_f);
    fclose(out);
    fclose(err_f);
    assert(rc == 1);
}

/* --- Coverage: cli_cmd_serve bad model ----------------------------------- */

static void test_cmd_serve_bad_model(void) {
    setenv("MLXD_CACHE_DIR", "/nonexistent", 1);
    setenv("MLXD_HF_HUB_DIR", "/nonexistent", 1);

    char errbuf[4096] = {0};
    FILE *out = fmemopen(NULL, 1024, "w");
    FILE *err_f = fmemopen(errbuf, sizeof(errbuf), "w");

    cli_args_t args = {0};
    args.cmd = CLI_SERVE;
    args.serve.host = "127.0.0.1";
    args.serve.port = 0;
    args.serve.model = "no/such-model";

    int rc = cli_cmd_serve(&args, out, err_f);
    fclose(out);
    fclose(err_f);

    assert(rc == 1);
    assert(strstr(errbuf, "cannot resolve") != NULL);

    unsetenv("MLXD_CACHE_DIR");
    unsetenv("MLXD_HF_HUB_DIR");
}

/* --- Coverage: SIGINT integration ---------------------------------------- */

static void test_serve_sigint_graceful(void) {
    int pipefd[2];
    assert(pipe(pipefd) == 0);

    pid_t child = fork();
    assert(child >= 0);

    if (child == 0) {
        close(pipefd[0]);
        FILE *out = fdopen(pipefd[1], "w");
        FILE *err_f = fopen("/dev/null", "w");

        cli_args_t args = {0};
        args.cmd = CLI_SERVE;
        args.serve.host = "127.0.0.1";
        args.serve.port = 0;

        int rc = cli_cmd_serve(&args, out, err_f);
        fclose(out);
        fclose(err_f);
        _exit(rc);
    }

    close(pipefd[1]);
    FILE *rd = fdopen(pipefd[0], "r");
    char line[256];
    assert(fgets(line, sizeof(line), rd) != NULL);
    assert(strstr(line, "serving on") != NULL);

    kill(child, SIGINT);

    int status;
    waitpid(child, &status, 0);
    fclose(rd);

    assert(WIFEXITED(status));
    assert(WEXITSTATUS(status) == 0);
}

static void double_sigint(int sig) {
    (void)sig;
    raise(SIGINT);
    raise(SIGINT);
}

static void test_serve_sigint_force_exit(void) {
    int pipefd[2];
    assert(pipe(pipefd) == 0);

    pid_t child = fork();
    assert(child >= 0);

    if (child == 0) {
        close(pipefd[0]);

        struct sigaction usr = {0};
        usr.sa_handler = double_sigint;
        sigemptyset(&usr.sa_mask);
        sigaction(SIGUSR1, &usr, NULL);

        FILE *out = fdopen(pipefd[1], "w");
        FILE *err_f = fopen("/dev/null", "w");

        cli_args_t args = {0};
        args.cmd = CLI_SERVE;
        args.serve.host = "127.0.0.1";
        args.serve.port = 0;

        int rc = cli_cmd_serve(&args, out, err_f);
        fclose(out);
        fclose(err_f);
        _exit(rc);
    }

    close(pipefd[1]);
    FILE *rd = fdopen(pipefd[0], "r");
    char line[256];
    assert(fgets(line, sizeof(line), rd) != NULL);
    assert(strstr(line, "serving on") != NULL);

    kill(child, SIGUSR1);

    int status;
    waitpid(child, &status, 0);
    fclose(rd);

    assert(WIFEXITED(status) && WEXITSTATUS(status) == 130);
}

int main(void) {
    test_usage_contains_subcommands();
    test_usage_documents_options();
    test_version_string();
    test_list_json_two_models();
    test_list_json_empty();
    test_cmd_list_table();
    test_cmd_list_json();
    test_cmd_list_empty();
    test_cmd_list_empty_json();
    test_cmd_run_oneshot();
    test_cmd_run_streaming();
    test_cmd_run_stdin();
    test_cmd_run_bad_model();
    test_sigint_decide();
    test_main_help();
    test_main_version();
    test_main_unknown_command();
    test_cmd_serve_bad_model();
    test_serve_sigint_graceful();
    test_serve_sigint_force_exit();
    test_main_serve_help();
    test_main_no_args();
    printf("test_cli_cmds: all tests passed\n");
    return 0;
}
