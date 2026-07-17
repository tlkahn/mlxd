#include "cli/cli.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static cli_cmd_t cli_errorf(cli_args_t *out, const char *fmt, const char *arg) {
    out->cmd = CLI_ERROR;
    snprintf(out->err, sizeof(out->err), fmt, arg);
    return out->cmd;
}

static cli_cmd_t cli_error(cli_args_t *out, const char *msg) {
    return cli_errorf(out, "%s", msg);
}

/* Parse a whole-string integer in [min, max]. Returns 0 on success. */
static int parse_int(const char *s, long min, long max, int *out) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < min || v > max)
        return -1;
    *out = (int)v;
    return 0;
}

/* Fetch the value of flag argv[*i], advancing *i. Returns NULL if absent. */
static const char *flag_value(int argc, char **argv, int *i) {
    if (*i + 1 >= argc)
        return NULL;
    return argv[++*i];
}

static cli_cmd_t parse_serve(int argc, char **argv, cli_args_t *out) {
    out->cmd = CLI_SERVE;
    out->serve.host = "127.0.0.1";
    out->serve.port = 8080;

    for (int i = 2; i < argc; i++) {
        const char *flag = argv[i];
        if (strcmp(flag, "--help") == 0 || strcmp(flag, "-h") == 0) {
            out->cmd = CLI_HELP;
            return out->cmd;
        } else if (strcmp(flag, "--model") == 0) {
            if (!(out->serve.model = flag_value(argc, argv, &i)))
                return cli_errorf(out, "%s requires a value", flag);
        } else if (strcmp(flag, "--host") == 0) {
            if (!(out->serve.host = flag_value(argc, argv, &i)))
                return cli_errorf(out, "%s requires a value", flag);
        } else if (strcmp(flag, "--port") == 0) {
            const char *v = flag_value(argc, argv, &i);
            if (!v)
                return cli_errorf(out, "%s requires a value", flag);
            if (parse_int(v, 0, 65535, &out->serve.port) != 0)
                return cli_errorf(out, "invalid port '%s'", v);
        } else {
            return cli_errorf(out, "serve: unknown option '%s'", flag);
        }
    }
    return out->cmd;
}

/* Parse a whole-string non-negative float. Returns 0 on success. */
static int parse_float(const char *s, float *out) {
    char *end = NULL;
    errno = 0;
    float v = strtof(s, &end);
    if (errno != 0 || end == s || *end != '\0' || v < 0.0f)
        return -1;
    *out = v;
    return 0;
}

static cli_cmd_t parse_run(int argc, char **argv, cli_args_t *out) {
    out->cmd = CLI_RUN;
    out->run.max_tokens = 256;
    out->run.temperature = 1.0f;

    int npos = 0;
    for (int i = 2; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            out->cmd = CLI_HELP;
            return out->cmd;
        } else if (strcmp(arg, "--max-tokens") == 0) {
            const char *v = flag_value(argc, argv, &i);
            if (!v)
                return cli_errorf(out, "%s requires a value", arg);
            if (parse_int(v, 1, 1 << 30, &out->run.max_tokens) != 0)
                return cli_errorf(out, "invalid max-tokens '%s'", v);
        } else if (strcmp(arg, "--temperature") == 0) {
            const char *v = flag_value(argc, argv, &i);
            if (!v)
                return cli_errorf(out, "%s requires a value", arg);
            if (parse_float(v, &out->run.temperature) != 0)
                return cli_errorf(out, "invalid temperature '%s'", v);
        } else if (strcmp(arg, "--stream") == 0) {
            out->run.stream = true;
        } else if (arg[0] == '-') {
            return cli_errorf(out, "run: unknown option '%s'", arg);
        } else if (npos == 0) {
            out->run.model = arg;
            npos++;
        } else if (npos == 1) {
            out->run.prompt = arg;
            npos++;
        } else {
            return cli_errorf(out, "run: unexpected argument '%s'", arg);
        }
    }

    if (!out->run.model)
        return cli_error(out, "run: missing MODEL argument");
    return out->cmd;
}

static cli_cmd_t parse_pull(int argc, char **argv, cli_args_t *out) {
    out->cmd = CLI_PULL;

    if (argc >= 3 && (strcmp(argv[2], "--help") == 0 || strcmp(argv[2], "-h") == 0)) {
        out->cmd = CLI_HELP;
        return out->cmd;
    }
    if (argc < 3)
        return cli_error(out, "pull: missing MODEL_SPEC argument");
    if (argc > 3)
        return cli_errorf(out, "pull: unexpected argument '%s'", argv[3]);

    out->pull.spec = argv[2];
    return out->cmd;
}

static cli_cmd_t parse_list(int argc, char **argv, cli_args_t *out) {
    out->cmd = CLI_LIST;

    for (int i = 2; i < argc; i++) {
        const char *flag = argv[i];
        if (strcmp(flag, "--help") == 0 || strcmp(flag, "-h") == 0) {
            out->cmd = CLI_HELP;
            return out->cmd;
        } else if (strcmp(flag, "--json") == 0) {
            out->list.json = true;
        } else {
            return cli_errorf(out, "list: unknown option '%s'", flag);
        }
    }
    return out->cmd;
}

static const char usage_text[] =
    "usage: mlxd <command> [options]\n"
    "\n"
    "commands:\n"
    "  serve [--model MODEL] [--host HOST] [--port PORT]\n"
    "        start OpenAI-compatible HTTP server\n"
    "\n"
    "  run [--max-tokens N] [--temperature T] [--stream] MODEL [PROMPT]\n"
    "        one-shot or interactive text generation\n"
    "\n"
    "  pull MODEL_SPEC\n"
    "        download model from HuggingFace (e.g. org/model)\n"
    "\n"
    "  list [--json]\n"
    "        list locally available models\n";

const char *cli_usage_str(void) {
    return usage_text;
}

cli_cmd_t cli_parse(int argc, char **argv, cli_args_t *out) {
    memset(out, 0, sizeof(*out));

    if (argc < 2)
        return cli_error(out, "missing command");

    const char *cmd = argv[1];

    if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        out->cmd = CLI_HELP;
        return out->cmd;
    }
    if (strcmp(cmd, "--version") == 0) {
        out->cmd = CLI_VERSION;
        return out->cmd;
    }

    if (strcmp(cmd, "serve") == 0)
        return parse_serve(argc, argv, out);
    if (strcmp(cmd, "run") == 0)
        return parse_run(argc, argv, out);
    if (strcmp(cmd, "pull") == 0)
        return parse_pull(argc, argv, out);
    if (strcmp(cmd, "list") == 0)
        return parse_list(argc, argv, out);

    return cli_errorf(out, "unknown command '%s'", cmd);
}
