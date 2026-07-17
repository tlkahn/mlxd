#include "cli/cli.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

/* --- helpers --------------------------------------------------------------- */

static cli_args_t parse(int argc, const char **argv) {
    cli_args_t args;
    cli_parse(argc, (char **)argv, &args);
    return args;
}

/* --- help / version -------------------------------------------------------- */

static void test_help_long(void) {
    const char *argv[] = {"mlxd", "--help"};
    cli_args_t a = parse(2, argv);
    assert(a.cmd == CLI_HELP);
}

static void test_help_short(void) {
    const char *argv[] = {"mlxd", "-h"};
    cli_args_t a = parse(2, argv);
    assert(a.cmd == CLI_HELP);
}

static void test_version(void) {
    const char *argv[] = {"mlxd", "--version"};
    cli_args_t a = parse(2, argv);
    assert(a.cmd == CLI_VERSION);
}

/* --- errors ---------------------------------------------------------------- */

static void test_no_args(void) {
    const char *argv[] = {"mlxd"};
    cli_args_t a = parse(1, argv);
    assert(a.cmd == CLI_ERROR);
    assert(strlen(a.err) > 0);
}

static void test_unknown_command(void) {
    const char *argv[] = {"mlxd", "frobnicate"};
    cli_args_t a = parse(2, argv);
    assert(a.cmd == CLI_ERROR);
    assert(strstr(a.err, "frobnicate") != NULL);
}

/* --- serve ------------------------------------------------------------------ */

static void test_serve_defaults(void) {
    const char *argv[] = {"mlxd", "serve"};
    cli_args_t a = parse(2, argv);
    assert(a.cmd == CLI_SERVE);
    assert(a.serve.model == NULL);
    assert(strcmp(a.serve.host, "127.0.0.1") == 0);
    assert(a.serve.port == 8080);
}

static void test_serve_flags(void) {
    const char *argv[] = {"mlxd", "serve", "--model", "/m/dir",
                          "--host", "0.0.0.0", "--port", "9000"};
    cli_args_t a = parse(8, argv);
    assert(a.cmd == CLI_SERVE);
    assert(strcmp(a.serve.model, "/m/dir") == 0);
    assert(strcmp(a.serve.host, "0.0.0.0") == 0);
    assert(a.serve.port == 9000);
}

static void test_serve_missing_value(void) {
    const char *argv[] = {"mlxd", "serve", "--port"};
    cli_args_t a = parse(3, argv);
    assert(a.cmd == CLI_ERROR);
    assert(strstr(a.err, "--port") != NULL);
}

static void test_serve_bad_port(void) {
    const char *argv[] = {"mlxd", "serve", "--port", "junk"};
    cli_args_t a = parse(4, argv);
    assert(a.cmd == CLI_ERROR);

    const char *argv2[] = {"mlxd", "serve", "--port", "70000"};
    a = parse(4, argv2);
    assert(a.cmd == CLI_ERROR);

    const char *argv3[] = {"mlxd", "serve", "--port", "-1"};
    a = parse(4, argv3);
    assert(a.cmd == CLI_ERROR);
}

static void test_serve_threads_rejected(void) {
    const char *argv[] = {"mlxd", "serve", "--threads", "4"};
    cli_args_t a = parse(4, argv);
    assert(a.cmd == CLI_ERROR);
    assert(strstr(a.err, "--threads") != NULL);
}

static void test_serve_unknown_flag(void) {
    const char *argv[] = {"mlxd", "serve", "--bogus"};
    cli_args_t a = parse(3, argv);
    assert(a.cmd == CLI_ERROR);
    assert(strstr(a.err, "--bogus") != NULL);
}

static void test_serve_help(void) {
    const char *argv[] = {"mlxd", "serve", "--help"};
    cli_args_t a = parse(3, argv);
    assert(a.cmd == CLI_HELP);
}

static void test_run_help(void) {
    const char *argv[] = {"mlxd", "run", "-h"};
    cli_args_t a = parse(3, argv);
    assert(a.cmd == CLI_HELP);
}

static void test_pull_help(void) {
    const char *argv[] = {"mlxd", "pull", "--help"};
    cli_args_t a = parse(3, argv);
    assert(a.cmd == CLI_HELP);
}

static void test_list_help(void) {
    const char *argv[] = {"mlxd", "list", "--help"};
    cli_args_t a = parse(3, argv);
    assert(a.cmd == CLI_HELP);
}

/* --- run -------------------------------------------------------------------- */

static void test_run_defaults(void) {
    const char *argv[] = {"mlxd", "run", "my-model"};
    cli_args_t a = parse(3, argv);
    assert(a.cmd == CLI_RUN);
    assert(strcmp(a.run.model, "my-model") == 0);
    assert(a.run.prompt == NULL);
    assert(a.run.max_tokens == 256);
    assert(a.run.temperature == 1.0f);
    assert(!a.run.stream);
}

static void test_run_full(void) {
    const char *argv[] = {"mlxd", "run", "my-model", "hello there",
                          "--max-tokens", "32", "--temperature", "0.5",
                          "--stream"};
    cli_args_t a = parse(9, argv);
    assert(a.cmd == CLI_RUN);
    assert(strcmp(a.run.model, "my-model") == 0);
    assert(strcmp(a.run.prompt, "hello there") == 0);
    assert(a.run.max_tokens == 32);
    assert(a.run.temperature == 0.5f);
    assert(a.run.stream);
}

static void test_run_missing_model(void) {
    const char *argv[] = {"mlxd", "run"};
    cli_args_t a = parse(2, argv);
    assert(a.cmd == CLI_ERROR);
}

static void test_run_extra_positional(void) {
    const char *argv[] = {"mlxd", "run", "m", "p", "surplus"};
    cli_args_t a = parse(5, argv);
    assert(a.cmd == CLI_ERROR);
}

static void test_run_bad_max_tokens(void) {
    const char *argv[] = {"mlxd", "run", "m", "--max-tokens", "0"};
    cli_args_t a = parse(5, argv);
    assert(a.cmd == CLI_ERROR);

    const char *argv2[] = {"mlxd", "run", "m", "--max-tokens", "abc"};
    a = parse(5, argv2);
    assert(a.cmd == CLI_ERROR);
}

static void test_run_bad_temperature(void) {
    const char *argv[] = {"mlxd", "run", "m", "--temperature", "hot"};
    cli_args_t a = parse(5, argv);
    assert(a.cmd == CLI_ERROR);

    const char *argv2[] = {"mlxd", "run", "m", "--temperature", "-0.5"};
    a = parse(5, argv2);
    assert(a.cmd == CLI_ERROR);
}

/* --- pull ------------------------------------------------------------------- */

static void test_pull(void) {
    const char *argv[] = {"mlxd", "pull", "org/model"};
    cli_args_t a = parse(3, argv);
    assert(a.cmd == CLI_PULL);
    assert(strcmp(a.pull.spec, "org/model") == 0);
}

static void test_pull_missing_spec(void) {
    const char *argv[] = {"mlxd", "pull"};
    cli_args_t a = parse(2, argv);
    assert(a.cmd == CLI_ERROR);
}

static void test_pull_extra_arg(void) {
    const char *argv[] = {"mlxd", "pull", "a", "b"};
    cli_args_t a = parse(4, argv);
    assert(a.cmd == CLI_ERROR);
}

/* --- list ------------------------------------------------------------------- */

static void test_list_defaults(void) {
    const char *argv[] = {"mlxd", "list"};
    cli_args_t a = parse(2, argv);
    assert(a.cmd == CLI_LIST);
    assert(!a.list.json);
}

static void test_list_json(void) {
    const char *argv[] = {"mlxd", "list", "--json"};
    cli_args_t a = parse(3, argv);
    assert(a.cmd == CLI_LIST);
    assert(a.list.json);
}

static void test_list_unknown_flag(void) {
    const char *argv[] = {"mlxd", "list", "--frob"};
    cli_args_t a = parse(3, argv);
    assert(a.cmd == CLI_ERROR);
}

int main(void) {
    test_help_long();
    test_help_short();
    test_version();
    test_no_args();
    test_unknown_command();
    test_serve_defaults();
    test_serve_flags();
    test_serve_missing_value();
    test_serve_bad_port();
    test_serve_threads_rejected();
    test_serve_unknown_flag();
    test_serve_help();
    test_run_help();
    test_pull_help();
    test_list_help();
    test_run_defaults();
    test_run_full();
    test_run_missing_model();
    test_run_extra_positional();
    test_run_bad_max_tokens();
    test_run_bad_temperature();
    test_pull();
    test_pull_missing_spec();
    test_pull_extra_arg();
    test_list_defaults();
    test_list_json();
    test_list_unknown_flag();
    printf("test_cli: all tests passed\n");
    return 0;
}
