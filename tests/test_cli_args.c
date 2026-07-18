#include "cli/args.h"
#include "core/types.h"

#include <assert.h>
#include <math.h>
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
    assert(strstr(err, "invalid port") != NULL);
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
    assert(!opts.top_p_set);
    assert(!opts.top_k_set);
    assert(!opts.min_p_set);
    assert(!opts.seed_set);
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

/* --- B4 cycle 2: --raw flag ----------------------------------------------- */

static void test_run_raw_flag(void) {
    char *argv[] = {"mlxd", "run", "m", "p", "--raw"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(5, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(opts.raw);
}

static void test_run_raw_default_false(void) {
    char *argv[] = {"mlxd", "run", "m", "hello"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(4, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(!opts.raw);
}

/* --- B4 cycle 3: --token-ids flag ----------------------------------------- */

static void test_run_token_ids_flag(void) {
    char *argv[] = {"mlxd", "run", "m", "p", "--token-ids"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(5, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(opts.token_ids);
}

static void test_run_token_ids_default_false(void) {
    char *argv[] = {"mlxd", "run", "m", "hello"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(4, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(!opts.token_ids);
}

static void test_run_raw_and_token_ids_together(void) {
    char *argv[] = {"mlxd", "run", "m", "p", "--raw", "--token-ids"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(6, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(opts.raw);
    assert(opts.token_ids);
}

/* --- Review fix: extra positionals ---------------------------------------- */

static void test_pull_extra_positional(void) {
    char *argv[] = {"mlxd", "pull", "spec1", "spec2"};
    cli_pull_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_pull(4, argv, &opts, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "unexpected") != NULL);
}

static void test_run_extra_positional(void) {
    char *argv[] = {"mlxd", "run", "model", "prompt", "extra"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(5, argv, &opts, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "unexpected") != NULL);
}

/* --- Review fix: negative temperature ------------------------------------- */

static void test_run_temperature_negative(void) {
    char *argv[] = {"mlxd", "run", "m", "--temperature", "-1"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(5, argv, &opts, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "temperature must be >= 0") != NULL);
}

static void test_run_temperature_zero(void) {
    char *argv[] = {"mlxd", "run", "m", "--temperature", "0"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(5, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(opts.temperature_set);
    assert(opts.temperature == 0.0f);
}

static void test_run_temperature_nan_rejected(void) {
    char *argv[] = {"mlxd", "run", "m", "--temperature", "nan"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(5, argv, &opts, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "temperature") != NULL);
}

static void test_run_temperature_inf_rejected(void) {
    char *argv[] = {"mlxd", "run", "m", "--temperature", "inf"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(5, argv, &opts, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "temperature") != NULL);
}

static void test_run_max_tokens_overflow_rejected(void) {
    char *argv[] = {"mlxd", "run", "m", "--max-tokens", "99999999999999"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(5, argv, &opts, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "max-tokens") != NULL);
}

/* --- Review fix: -- end-of-options marker --------------------------------- */

static void test_run_dashdash_prompt(void) {
    char *argv[] = {"mlxd", "run", "model", "--", "-not-a-flag"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(5, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(strcmp(opts.model, "model") == 0);
    assert(strcmp(opts.prompt, "-not-a-flag") == 0);
}

static void test_run_dashdash_model(void) {
    char *argv[] = {"mlxd", "run", "--", "-weird/model-path"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(4, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(strcmp(opts.model, "-weird/model-path") == 0);
}

static void test_run_dashdash_known_flag_as_positional(void) {
    char *argv[] = {"mlxd", "run", "model", "--", "--stream"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(5, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(strcmp(opts.prompt, "--stream") == 0);
    assert(!opts.stream);
}

static void test_pull_dashdash_spec(void) {
    char *argv[] = {"mlxd", "pull", "--", "-spec"};
    cli_pull_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_pull(4, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(strcmp(opts.spec, "-spec") == 0);
}

static void test_run_unknown_flag_without_dashdash(void) {
    char *argv[] = {"mlxd", "run", "m", "-x"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(4, argv, &opts, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "unknown") != NULL);
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

/* --- C2 cycle 17: sampling flags ------------------------------------------ */

static void test_run_sampling_flags(void) {
    char *argv[] = {"mlxd", "run", "m", "hi",
                    "--top-p", "0.9", "--top-k", "40", "--min-p", "0.05", "--seed", "42"};
    cli_run_opts_t opts = {0};
    char err[256] = {0};
    int rc = cli_parse_run(12, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(opts.top_p_set);
    assert(opts.top_p > 0.89f && opts.top_p < 0.91f);
    assert(opts.top_k_set);
    assert(opts.top_k == 40);
    assert(opts.min_p_set);
    assert(opts.min_p > 0.04f && opts.min_p < 0.06f);
    assert(opts.seed_set);
    assert(opts.seed == 42);
}

static void test_run_sampling_bad_values(void) {
    char err[256];

    /* top-p out of range */
    char *argv1[] = {"mlxd", "run", "m", "--top-p", "1.5"};
    cli_run_opts_t opts = {0};
    memset(err, 0, sizeof(err));
    assert(cli_parse_run(5, argv1, &opts, err, sizeof(err)) == -1);
    assert(strstr(err, "top-p") != NULL);

    /* top-k < -1 */
    char *argv2[] = {"mlxd", "run", "m", "--top-k", "-2"};
    memset(&opts, 0, sizeof(opts));
    memset(err, 0, sizeof(err));
    assert(cli_parse_run(5, argv2, &opts, err, sizeof(err)) == -1);
    assert(strstr(err, "top-k") != NULL);

    /* min-p negative */
    char *argv3[] = {"mlxd", "run", "m", "--min-p", "-0.1"};
    memset(&opts, 0, sizeof(opts));
    memset(err, 0, sizeof(err));
    assert(cli_parse_run(5, argv3, &opts, err, sizeof(err)) == -1);
    assert(strstr(err, "min-p") != NULL);

    /* seed non-numeric */
    char *argv4[] = {"mlxd", "run", "m", "--seed", "abc"};
    memset(&opts, 0, sizeof(opts));
    memset(err, 0, sizeof(err));
    assert(cli_parse_run(5, argv4, &opts, err, sizeof(err)) == -1);
    assert(strstr(err, "seed") != NULL);

    /* seed negative */
    char *argv5[] = {"mlxd", "run", "m", "--seed", "-5"};
    memset(&opts, 0, sizeof(opts));
    memset(err, 0, sizeof(err));
    assert(cli_parse_run(5, argv5, &opts, err, sizeof(err)) == -1);
    assert(strstr(err, "seed") != NULL);
}

/* --- C3 cycle 1: cli_parse_run must fully initialize all fields ---------- */

static void test_run_parse_initializes_poisoned_struct(void) {
    char *argv[] = {"mlxd", "run", "m", "hello"};
    cli_run_opts_t opts;
    memset(&opts, 0xA5, sizeof(opts));
    char err[256] = {0};
    int rc = cli_parse_run(4, argv, &opts, err, sizeof(err));
    assert(rc == 0);
    assert(!opts.top_p_set);
    assert(!opts.top_k_set);
    assert(!opts.min_p_set);
    assert(!opts.seed_set);
    assert(!opts.temperature_set);
    assert(!opts.stream);
    assert(!opts.raw);
    assert(!opts.token_ids);
    assert(opts.max_tokens == 0);
    assert(opts.temperature == 0.0f);
    assert(opts.top_p == 0.0f);
    assert(opts.min_p == 0.0f);
    assert(opts.top_k == 0);
    assert(opts.seed == 0);
}

/* --- Cycle 7 (review): run_opts_apply_sampling greedy default ----------- */

static void test_run_opts_apply_sampling_defaults(void) {
    cli_run_opts_t opts = {0};
    gen_params_t params = {.sampling = SAMPLING_PARAMS_DEFAULT};
    run_opts_apply_sampling(&opts, &params);
    assert(params.sampling.temperature == 0.0f);
    assert(params.sampling_set & SAMPLING_SET_TEMPERATURE);
    assert(!(params.sampling_set & SAMPLING_SET_TOP_P));
    assert(!(params.sampling_set & SAMPLING_SET_TOP_K));
    assert(!(params.sampling_set & SAMPLING_SET_MIN_P));
    assert(!(params.sampling_set & SAMPLING_SET_SEED));
}

static void test_run_opts_apply_sampling_explicit_temp(void) {
    cli_run_opts_t opts = {0};
    opts.temperature = 0.8f;
    opts.temperature_set = true;
    opts.top_k = 40;
    opts.top_k_set = true;
    gen_params_t params = {.sampling = SAMPLING_PARAMS_DEFAULT};
    run_opts_apply_sampling(&opts, &params);
    assert(fabsf(params.sampling.temperature - 0.8f) < 1e-6f);
    assert(params.sampling_set & SAMPLING_SET_TEMPERATURE);
    assert(params.sampling.top_k == 40);
    assert(params.sampling_set & SAMPLING_SET_TOP_K);
    assert(!(params.sampling_set & SAMPLING_SET_TOP_P));
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

    /* B4 cycle 2: --raw flag */
    test_run_raw_flag();
    test_run_raw_default_false();

    /* B4 cycle 3: --token-ids flag */
    test_run_token_ids_flag();
    test_run_token_ids_default_false();
    test_run_raw_and_token_ids_together();

    /* review fix: extra positionals */
    test_pull_extra_positional();
    test_run_extra_positional();

    /* review fix: negative temperature */
    test_run_temperature_negative();
    test_run_temperature_zero();
    test_run_temperature_nan_rejected();
    test_run_temperature_inf_rejected();
    test_run_max_tokens_overflow_rejected();

    /* review fix: -- end-of-options marker */
    test_run_dashdash_prompt();
    test_run_dashdash_model();
    test_run_dashdash_known_flag_as_positional();
    test_pull_dashdash_spec();
    test_run_unknown_flag_without_dashdash();

    /* cycle 7: pull + list */
    test_pull_spec();
    test_pull_missing_spec();
    test_list_default();
    test_list_json();
    test_list_unknown_flag();

    /* C2 cycle 17: sampling flags */
    test_run_sampling_flags();
    test_run_sampling_bad_values();

    /* C3 cycle 1: poisoned struct initialization */
    test_run_parse_initializes_poisoned_struct();

    /* Cycle 7 (review): run_opts_apply_sampling */
    test_run_opts_apply_sampling_defaults();
    test_run_opts_apply_sampling_explicit_temp();

    printf("test_cli_args: all passed\n");
    return 0;
}
