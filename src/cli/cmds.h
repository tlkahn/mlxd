#ifndef MLXD_CLI_CMDS_H
#define MLXD_CLI_CMDS_H

#include "cli/cli.h"
#include "registry/registry.h"

#include <stdio.h>

typedef enum {
    SIGINT_STOP,
    SIGINT_FORCE_EXIT,
} sigint_action_t;

sigint_action_t cli_sigint_decide(int count);

char *cli_list_json(const registry_model_info_t *models, int count);

int cli_cmd_list(const cli_args_t *args, FILE *out, FILE *err);
int cli_cmd_pull(const cli_args_t *args, FILE *out, FILE *err);
int cli_cmd_run(const cli_args_t *args, FILE *in, FILE *out, FILE *err);
int cli_cmd_serve(const cli_args_t *args, FILE *out, FILE *err);

int cli_main(int argc, char **argv, FILE *in, FILE *out, FILE *err);

#endif
