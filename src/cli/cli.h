#ifndef MLXD_CLI_H
#define MLXD_CLI_H

#include <stdbool.h>

#define MLXD_VERSION "0.1.0"

typedef enum {
    CLI_ERROR,
    CLI_HELP,
    CLI_VERSION,
    CLI_SERVE,
    CLI_RUN,
    CLI_PULL,
    CLI_LIST,
} cli_cmd_t;

typedef struct {
    cli_cmd_t cmd;

    struct {
        const char *model;   /* NULL = start without a model */
        const char *host;    /* default 127.0.0.1 */
        int         port;    /* default 8080; 0 = ephemeral */
        int         threads; /* 0 = default */
    } serve;

    struct {
        const char *model;       /* required */
        const char *prompt;      /* NULL = read from stdin */
        int         max_tokens;  /* default 256 */
        float       temperature; /* default 1.0 */
        bool        stream;      /* default false (one-shot) */
    } run;

    struct {
        const char *spec; /* required: org/model */
    } pull;

    struct {
        bool json; /* --json output */
    } list;

    char err[256]; /* set when cmd == CLI_ERROR */
} cli_args_t;

/* Parse argv into *out. Never fails hard: unknown commands, bad flags and
 * missing values yield CLI_ERROR with a message in out->err.
 * Returns out->cmd. */
cli_cmd_t cli_parse(int argc, char **argv, cli_args_t *out);

const char *cli_usage_str(void);

#endif
