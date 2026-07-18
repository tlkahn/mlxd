#include "cli/args.h"
#include "core/types.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

cli_cmd_t cli_command(int argc, char **argv) {
    if (argc < 2) return CLI_CMD_NONE;
    const char *cmd = argv[1];
    if (strcmp(cmd, "serve") == 0)    return CLI_CMD_SERVE;
    if (strcmp(cmd, "run") == 0)      return CLI_CMD_RUN;
    if (strcmp(cmd, "pull") == 0)     return CLI_CMD_PULL;
    if (strcmp(cmd, "list") == 0)     return CLI_CMD_LIST;
    if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) return CLI_CMD_HELP;
    if (strcmp(cmd, "--version") == 0) return CLI_CMD_VERSION;
    return CLI_CMD_UNKNOWN;
}

int cli_parse_serve(int argc, char **argv, cli_serve_opts_t *out, char *err, size_t errsz) {
    out->model = NULL;
    out->host = "127.0.0.1";
    out->port = 8080;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0) {
            if (i + 1 >= argc) {
                snprintf(err, errsz, "--model requires a value");
                return -1;
            }
            out->model = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0) {
            if (i + 1 >= argc) {
                snprintf(err, errsz, "--port requires a value");
                return -1;
            }
            char *end;
            long v = strtol(argv[++i], &end, 10);
            if (*end != '\0' || v < 1 || v > 65535) {
                snprintf(err, errsz, "invalid port '%s'", argv[i]);
                return -1;
            }
            out->port = (int)v;
        } else if (strcmp(argv[i], "--host") == 0) {
            if (i + 1 >= argc) {
                snprintf(err, errsz, "--host requires a value");
                return -1;
            }
            out->host = argv[++i];
        } else {
            snprintf(err, errsz, "unknown option '%s'", argv[i]);
            return -1;
        }
    }

    if (!out->model) {
        snprintf(err, errsz, "--model is required");
        return -1;
    }
    return 0;
}

int cli_parse_run(int argc, char **argv, cli_run_opts_t *out, char *err, size_t errsz) {
    out->model = NULL;
    out->prompt = NULL;
    out->max_tokens = 0;
    out->temperature = 0.0f;
    out->temperature_set = false;
    out->stream = false;
    out->raw = false;
    out->token_ids = false;

    int positional = 0;
    bool opts_done = false;
    for (int i = 2; i < argc; i++) {
        if (!opts_done && strcmp(argv[i], "--") == 0) {
            opts_done = true;
            continue;
        }
        if (!opts_done && strcmp(argv[i], "--max-tokens") == 0) {
            if (i + 1 >= argc) {
                snprintf(err, errsz, "--max-tokens requires a value");
                return -1;
            }
            char *end;
            errno = 0;
            long v = strtol(argv[++i], &end, 10);
            if (*end != '\0' || v < 0 || v > INT_MAX || errno == ERANGE) {
                snprintf(err, errsz, "invalid max-tokens '%s'", argv[i]);
                return -1;
            }
            out->max_tokens = (int)v;
        } else if (!opts_done && strcmp(argv[i], "--temperature") == 0) {
            if (i + 1 >= argc) {
                snprintf(err, errsz, "--temperature requires a value");
                return -1;
            }
            char *end;
            float v = strtof(argv[++i], &end);
            if (*end != '\0' || !isfinite(v)) {
                snprintf(err, errsz, "invalid temperature '%s'", argv[i]);
                return -1;
            }
            if (v < 0.0f) {
                snprintf(err, errsz, "temperature must be >= 0");
                return -1;
            }
            out->temperature = v;
            out->temperature_set = true;
        } else if (!opts_done && strcmp(argv[i], "--top-p") == 0) {
            if (i + 1 >= argc) {
                snprintf(err, errsz, "--top-p requires a value");
                return -1;
            }
            char *end;
            float v = strtof(argv[++i], &end);
            if (*end != '\0' || !isfinite(v)) {
                snprintf(err, errsz, "invalid top-p '%s'", argv[i]);
                return -1;
            }
            if (v < 0.0f || v > 1.0f) {
                snprintf(err, errsz, "top-p must be in [0, 1]");
                return -1;
            }
            out->top_p = v;
            out->top_p_set = true;
        } else if (!opts_done && strcmp(argv[i], "--top-k") == 0) {
            if (i + 1 >= argc) {
                snprintf(err, errsz, "--top-k requires a value");
                return -1;
            }
            char *end;
            errno = 0;
            long v = strtol(argv[++i], &end, 10);
            if (*end != '\0' || errno == ERANGE) {
                snprintf(err, errsz, "invalid top-k '%s'", argv[i]);
                return -1;
            }
            if (v < -1 || v > INT_MAX) {
                snprintf(err, errsz, "top-k must be >= -1");
                return -1;
            }
            out->top_k = (int)v;
            out->top_k_set = true;
        } else if (!opts_done && strcmp(argv[i], "--min-p") == 0) {
            if (i + 1 >= argc) {
                snprintf(err, errsz, "--min-p requires a value");
                return -1;
            }
            char *end;
            float v = strtof(argv[++i], &end);
            if (*end != '\0' || !isfinite(v)) {
                snprintf(err, errsz, "invalid min-p '%s'", argv[i]);
                return -1;
            }
            if (v < 0.0f || v > 1.0f) {
                snprintf(err, errsz, "min-p must be in [0, 1]");
                return -1;
            }
            out->min_p = v;
            out->min_p_set = true;
        } else if (!opts_done && strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                snprintf(err, errsz, "--seed requires a value");
                return -1;
            }
            char *end;
            errno = 0;
            long v = strtol(argv[++i], &end, 10);
            if (*end != '\0' || errno == ERANGE) {
                snprintf(err, errsz, "invalid seed '%s'", argv[i]);
                return -1;
            }
            if (v < 0 || v > INT_MAX) {
                snprintf(err, errsz, "seed must be >= 0");
                return -1;
            }
            out->seed = (int)v;
            out->seed_set = true;
        } else if (!opts_done && strcmp(argv[i], "--stream") == 0) {
            out->stream = true;
        } else if (!opts_done && strcmp(argv[i], "--raw") == 0) {
            out->raw = true;
        } else if (!opts_done && strcmp(argv[i], "--token-ids") == 0) {
            out->token_ids = true;
        } else if (!opts_done && argv[i][0] == '-') {
            snprintf(err, errsz, "unknown option '%s'", argv[i]);
            return -1;
        } else {
            if (positional == 0)
                out->model = argv[i];
            else if (positional == 1)
                out->prompt = argv[i];
            else {
                snprintf(err, errsz, "unexpected argument '%s'", argv[i]);
                return -1;
            }
            positional++;
        }
    }

    if (!out->model) {
        snprintf(err, errsz, "MODEL is required");
        return -1;
    }
    return 0;
}

int cli_parse_pull(int argc, char **argv, cli_pull_opts_t *out, char *err, size_t errsz) {
    out->spec = NULL;

    if (argc < 3) {
        snprintf(err, errsz, "model spec is required");
        return -1;
    }

    bool opts_done = false;
    for (int i = 2; i < argc; i++) {
        if (!opts_done && strcmp(argv[i], "--") == 0) {
            opts_done = true;
            continue;
        }
        if (!opts_done && argv[i][0] == '-') {
            snprintf(err, errsz, "unknown option '%s'", argv[i]);
            return -1;
        }
        if (!out->spec) {
            out->spec = argv[i];
        } else {
            snprintf(err, errsz, "unexpected argument '%s'", argv[i]);
            return -1;
        }
    }
    return 0;
}

int cli_parse_list(int argc, char **argv, cli_list_opts_t *out, char *err, size_t errsz) {
    out->json = false;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) {
            out->json = true;
        } else {
            snprintf(err, errsz, "unknown option '%s'", argv[i]);
            return -1;
        }
    }
    return 0;
}

void run_opts_apply_sampling(const cli_run_opts_t *opts, gen_params_t *params) {
    if (opts->temperature_set) {
        params->sampling.temperature = opts->temperature;
        params->sampling_set |= SAMPLING_SET_TEMPERATURE;
    } else {
        params->sampling.temperature = 0.0f;
        params->sampling_set |= SAMPLING_SET_TEMPERATURE;
    }
    if (opts->top_p_set) {
        params->sampling.top_p = opts->top_p;
        params->sampling_set |= SAMPLING_SET_TOP_P;
    }
    if (opts->top_k_set) {
        params->sampling.top_k = opts->top_k;
        params->sampling_set |= SAMPLING_SET_TOP_K;
    }
    if (opts->min_p_set) {
        params->sampling.min_p = opts->min_p;
        params->sampling_set |= SAMPLING_SET_MIN_P;
    }
    if (opts->seed_set) {
        params->sampling.seed = opts->seed;
        params->sampling_set |= SAMPLING_SET_SEED;
    }
}
