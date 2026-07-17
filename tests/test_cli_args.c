#include "cli/args.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

/* --- Cycle 1: cli_command ------------------------------------------------- */

static void test_command_serve(void) {
    char *argv[] = {"mlxd", "serve"};
    assert(cli_command(2, argv) == CLI_CMD_SERVE);
}

static void test_command_run(void) {
    char *argv[] = {"mlxd", "run"};
    assert(cli_command(2, argv) == CLI_CMD_RUN);
}

static void test_command_pull(void) {
    char *argv[] = {"mlxd", "pull"};
    assert(cli_command(2, argv) == CLI_CMD_PULL);
}

static void test_command_list(void) {
    char *argv[] = {"mlxd", "list"};
    assert(cli_command(2, argv) == CLI_CMD_LIST);
}

static void test_command_help_long(void) {
    char *argv[] = {"mlxd", "--help"};
    assert(cli_command(2, argv) == CLI_CMD_HELP);
}

static void test_command_help_short(void) {
    char *argv[] = {"mlxd", "-h"};
    assert(cli_command(2, argv) == CLI_CMD_HELP);
}

static void test_command_version(void) {
    char *argv[] = {"mlxd", "--version"};
    assert(cli_command(2, argv) == CLI_CMD_VERSION);
}

static void test_command_unknown(void) {
    char *argv[] = {"mlxd", "bogus"};
    assert(cli_command(2, argv) == CLI_CMD_UNKNOWN);
}

static void test_command_none(void) {
    char *argv[] = {"mlxd"};
    assert(cli_command(1, argv) == CLI_CMD_NONE);
}

/* --- Cycle 2: cli_parse_serve happy path ---------------------------------- */

static void test_serve_defaults(void) {
    char *argv[] = {"mlxd", "serve", "--model", "my-model"};
    cli_serve_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_serve(4, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(strcmp(opts.model, "my-model") == 0);
    assert(strcmp(opts.host, "127.0.0.1") == 0);
    assert(opts.port == 8080);
}

static void test_serve_port(void) {
    char *argv[] = {"mlxd", "serve", "--model", "m", "--port", "9090"};
    cli_serve_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_serve(6, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(opts.port == 9090);
}

/* --- Cycle 3: cli_parse_serve errors -------------------------------------- */

static void test_serve_missing_model(void) {
    char *argv[] = {"mlxd", "serve"};
    cli_serve_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_serve(2, argv, &opts, err, sizeof(err));
    assert(rc == -1);
    assert(strlen(err) > 0);
}

static void test_serve_port_non_numeric(void) {
    char *argv[] = {"mlxd", "serve", "--model", "m", "--port", "abc"};
    cli_serve_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_serve(6, argv, &opts, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "port") != NULL || strstr(err, "invalid") != NULL);
}

static void test_serve_port_zero(void) {
    char *argv[] = {"mlxd", "serve", "--model", "m", "--port", "0"};
    cli_serve_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_serve(6, argv, &opts, err, sizeof(err));
    assert(rc == -1);
}

static void test_serve_port_too_high(void) {
    char *argv[] = {"mlxd", "serve", "--model", "m", "--port", "70000"};
    cli_serve_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_serve(6, argv, &opts, err, sizeof(err));
    assert(rc == -1);
}

static void test_serve_flag_missing_value(void) {
    char *argv[] = {"mlxd", "serve", "--model"};
    cli_serve_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_serve(3, argv, &opts, err, sizeof(err));
    assert(rc == -1);
}

static void test_serve_unknown_flag(void) {
    char *argv[] = {"mlxd", "serve", "--model", "m", "--threads", "4"};
    cli_serve_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_serve(6, argv, &opts, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "unknown") != NULL || strstr(err, "--threads") != NULL);
}

/* --- Cycle 4: cli_parse_serve --host -------------------------------------- */

static void test_serve_host(void) {
    char *argv[] = {"mlxd", "serve", "--model", "m", "--host", "0.0.0.0"};
    cli_serve_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_serve(6, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(strcmp(opts.host, "0.0.0.0") == 0);
}

static void test_serve_host_missing_value(void) {
    char *argv[] = {"mlxd", "serve", "--model", "m", "--host"};
    cli_serve_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_serve(5, argv, &opts, err, sizeof(err));
    assert(rc == -1);
}

/* --- Cycle 5: cli_parse_run positionals ----------------------------------- */

static void test_run_model_and_prompt(void) {
    char *argv[] = {"mlxd", "run", "my-model", "Hello world"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(4, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(strcmp(opts.model, "my-model") == 0);
    assert(strcmp(opts.prompt, "Hello world") == 0);
}

static void test_run_model_only(void) {
    char *argv[] = {"mlxd", "run", "my-model"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(3, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(strcmp(opts.model, "my-model") == 0);
    assert(opts.prompt == NULL);
}

static void test_run_missing_model(void) {
    char *argv[] = {"mlxd", "run"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(2, argv, &opts, err, sizeof(err));
    assert(rc == -1);
    assert(strlen(err) > 0);
}

/* --- Cycle 6: cli_parse_run flags ----------------------------------------- */

static void test_run_max_tokens(void) {
    char *argv[] = {"mlxd", "run", "m", "hi", "--max-tokens", "128"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(6, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(opts.max_tokens == 128);
}

static void test_run_max_tokens_invalid(void) {
    char *argv[] = {"mlxd", "run", "m", "hi", "--max-tokens", "abc"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(6, argv, &opts, err, sizeof(err));
    assert(rc == -1);
}

static void test_run_temperature(void) {
    char *argv[] = {"mlxd", "run", "m", "hi", "--temperature", "0.7"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(6, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(opts.temperature > 0.69f && opts.temperature < 0.71f);
    assert(opts.temperature_set);
}

static void test_run_stream(void) {
    char *argv[] = {"mlxd", "run", "m", "hi", "--stream"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(5, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(opts.stream);
}

static void test_run_unknown_flag(void) {
    char *argv[] = {"mlxd", "run", "m", "--bogus"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(4, argv, &opts, err, sizeof(err));
    assert(rc == -1);
}

static void test_run_defaults(void) {
    char *argv[] = {"mlxd", "run", "m", "hello"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(4, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(opts.max_tokens == 0);
    assert(!opts.temperature_set);
    assert(!opts.stream);
}

static void test_run_flags_before_prompt(void) {
    char *argv[] = {"mlxd", "run", "m", "--stream", "--max-tokens", "64", "hello"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(7, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(opts.stream);
    assert(opts.max_tokens == 64);
    assert(strcmp(opts.prompt, "hello") == 0);
}

/* --- Cycle 7: cli_parse_pull and cli_parse_list --------------------------- */

static void test_pull_spec(void) {
    char *argv[] = {"mlxd", "pull", "mlx-community/gemma-2b"};
    cli_pull_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_pull(3, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(strcmp(opts.spec, "mlx-community/gemma-2b") == 0);
}

static void test_pull_missing_spec(void) {
    char *argv[] = {"mlxd", "pull"};
    cli_pull_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_pull(2, argv, &opts, err, sizeof(err));
    assert(rc == -1);
    assert(strlen(err) > 0);
}

static void test_list_default(void) {
    char *argv[] = {"mlxd", "list"};
    cli_list_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_list(2, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(!opts.json);
}

static void test_list_json(void) {
    char *argv[] = {"mlxd", "list", "--json"};
    cli_list_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_list(3, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(opts.json);
}

static void test_list_unknown_flag(void) {
    char *argv[] = {"mlxd", "list", "--verbose"};
    cli_list_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_list(3, argv, &opts, err, sizeof(err));
    assert(rc == -1);
}

int main(void) {
    /* cycle 1: cli_command */
    test_command_serve();
    test_command_run();
    test_command_pull();
    test_command_list();
    test_command_help_long();
    test_command_help_short();
    test_command_version();
    test_command_unknown();
    test_command_none();

    /* cycle 2: serve happy path */
    test_serve_defaults();
    test_serve_port();

    /* cycle 3: serve errors */
    test_serve_missing_model();
    test_serve_port_non_numeric();
    test_serve_port_zero();
    test_serve_port_too_high();
    test_serve_flag_missing_value();
    test_serve_unknown_flag();

    /* cycle 4: serve --host */
    test_serve_host();
    test_serve_host_missing_value();

    /* cycle 5: run positionals */
    test_run_model_and_prompt();
    test_run_model_only();
    test_run_missing_model();

    /* cycle 6: run flags */
    test_run_max_tokens();
    test_run_max_tokens_invalid();
    test_run_temperature();
    test_run_stream();
    test_run_unknown_flag();
    test_run_defaults();
    test_run_flags_before_prompt();

    /* cycle 7: pull + list */
    test_pull_spec();
    test_pull_missing_spec();
    test_list_default();
    test_list_json();
    test_list_unknown_flag();

    printf("test_cli_args: all passed\n");
    return 0;
}
