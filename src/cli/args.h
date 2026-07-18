#ifndef CLI_ARGS_H
#define CLI_ARGS_H

#include "core/types.h"

#include <stdbool.h>
#include <stddef.h>

#define MLXD_VERSION "0.1.0"

typedef enum {
    CLI_CMD_SERVE,
    CLI_CMD_RUN,
    CLI_CMD_PULL,
    CLI_CMD_LIST,
    CLI_CMD_HELP,
    CLI_CMD_VERSION,
    CLI_CMD_UNKNOWN,
    CLI_CMD_NONE,
} cli_cmd_t;

cli_cmd_t cli_command(int argc, char **argv);

typedef struct {
    const char *model;
    const char *host;
    int port;
} cli_serve_opts_t;

typedef struct {
    const char *model;
    const char *prompt;
    int max_tokens;
    float temperature;
    bool temperature_set;
    float top_p;
    bool top_p_set;
    int top_k;
    bool top_k_set;
    float min_p;
    bool min_p_set;
    int seed;
    bool seed_set;
    bool stream;
    bool raw;
    bool token_ids;
    bool no_think;
} cli_run_opts_t;

typedef struct {
    const char *spec;
} cli_pull_opts_t;

typedef struct {
    bool json;
} cli_list_opts_t;

int cli_parse_serve(int argc, char **argv, cli_serve_opts_t *out, char *err, size_t errsz);
int cli_parse_run(int argc, char **argv, cli_run_opts_t *out, char *err, size_t errsz);
int cli_parse_pull(int argc, char **argv, cli_pull_opts_t *out, char *err, size_t errsz);
int cli_parse_list(int argc, char **argv, cli_list_opts_t *out, char *err, size_t errsz);

void run_opts_apply_sampling(const cli_run_opts_t *opts, gen_params_t *params);

const char *cli_run_extra_json(const cli_run_opts_t *opts);

#endif
