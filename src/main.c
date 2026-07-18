#include "cli/args.h"
#include "cli/io.h"
#include "engine/engine.h"
#include "http/gen.h"
#include "http/server.h"
#include "model/tokenizer.h"
#include "registry/registry.h"

#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <yyjson/yyjson.h>

static void usage(void) {
    fprintf(stderr,
            "usage: mlxd <command> [options]\n"
            "\n"
            "commands:\n"
            "  serve   start OpenAI-compatible HTTP server\n"
            "  run     one-shot text generation\n"
            "  pull    download model from HuggingFace\n"
            "  list    list locally available models\n"
            "\n"
            "serve options:\n"
            "  --model <dir-or-spec>   model directory or HuggingFace spec (required)\n"
            "  --host <addr>           bind address (default: 127.0.0.1)\n"
            "  --port <N>              listen port (default: 8080)\n"
            "\n"
            "run options:\n"
            "  MODEL                   model directory or HuggingFace spec (required)\n"
            "  [PROMPT]                prompt text (reads stdin if omitted)\n"
            "  --max-tokens <N>        maximum tokens to generate (0 or unset: unlimited)\n"
            "  --temperature <F>       sampling temperature\n"
            "  --top-p <F>            nucleus sampling threshold [0,1]\n"
            "  --top-k <N>            top-k filter (-1 to disable)\n"
            "  --min-p <F>            minimum probability filter [0,1]\n"
            "  --seed <N>             random seed for reproducibility\n"
            "  --stream                flush each token to stdout as generated\n"
            "  --raw                   force raw completion even when a chat template is present\n"
            "  --token-ids             print generated token IDs instead of text\n"
            "  --                      end of options; remaining args are positional\n"
            "\n"
            "pull options:\n"
            "  <model-spec>            HuggingFace model spec (required)\n"
            "\n"
            "list options:\n"
            "  --json                  output as JSON array\n"
            "\n"
            "global flags:\n"
            "  --help, -h              show this help\n"
            "  --version               show version\n");
}

/* --- Signal handling for serve -------------------------------------------- */

static _Atomic int g_sigint_count;
static http_server_t *g_srv;

static void serve_sigint_handler(int sig) {
    (void)sig;
    int prev = atomic_fetch_add(&g_sigint_count, 1);
    if (prev == 0)
        http_server_stop(g_srv);
    else
        _exit(130);
}

/* --- Signal handling for run ---------------------------------------------- */

static _Atomic int g_run_cancel;

static void run_sigint_handler(int sig) {
    (void)sig;
    int prev = atomic_fetch_add(&g_run_cancel, 1);
    if (prev >= 1)
        _exit(130);
}

/* --- Helpers -------------------------------------------------------------- */

static char *slurp_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = malloc(sz > 0 ? (size_t)sz : 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    if (out_len) *out_len = (size_t)sz;
    return buf;
}

static char *read_chat_template(const char *model_dir) {
    size_t pathlen = strlen(model_dir) + sizeof("/tokenizer_config.json");
    char *path = malloc(pathlen);
    if (!path) return NULL;
    snprintf(path, pathlen, "%s/tokenizer_config.json", model_dir);

    size_t len = 0;
    char *buf = slurp_file(path, &len);
    free(path);
    if (!buf) return NULL;

    yyjson_doc *doc = yyjson_read(buf, len, 0);
    free(buf);
    if (!doc) return NULL;

    char *result = NULL;
    yyjson_val *root = yyjson_doc_get_root(doc);
    if (yyjson_is_obj(root)) {
        yyjson_val *tmpl = yyjson_obj_get(root, "chat_template");
        if (yyjson_is_str(tmpl))
            result = strdup(yyjson_get_str(tmpl));
    }
    yyjson_doc_free(doc);
    return result;
}

static bool file_exists(const char *dir, const char *name) {
    size_t pathlen = strlen(dir) + 1 + strlen(name) + 1;
    char *path = malloc(pathlen);
    if (!path) return false;
    snprintf(path, pathlen, "%s/%s", dir, name);
    struct stat st;
    bool exists = stat(path, &st) == 0;
    free(path);
    return exists;
}

static const char *resolve_model_dir(const char *model_arg, char **resolved_out) {
    *resolved_out = NULL;
    if (file_exists(model_arg, "tokenizer.json"))
        return model_arg;
    *resolved_out = registry_resolve(model_arg);
    return *resolved_out;
}

static bool run_cancel_pred(void *ud) {
    (void)ud;
    return atomic_load(&g_run_cancel) > 0;
}

/* Print load failure distinguishing timeout vs LOAD_FAILED. */
static void print_load_failure(const char *prefix, engine_t *eng) {
    char err[256] = {0};
    if (engine_load_state(eng) == LOAD_FAILED &&
        engine_load_error(eng, err, sizeof(err)) == 0 && err[0] != '\0') {
        fprintf(stderr, "%s: load failed: %s\n", prefix, err);
    } else {
        fprintf(stderr, "%s: timed out waiting for model load\n", prefix);
    }
}

/* --- Subcommands ---------------------------------------------------------- */

static int pull_progress(const registry_progress_t *p, void *ud) {
    (void)ud;
    if (p->file_total > 0) {
        unsigned pct = (unsigned)(p->file_downloaded * 100 / p->file_total);
        fprintf(stderr, "\r  %s  %zu/%zu  %u%%",
                p->filename ? p->filename : "?",
                p->file_index + 1, p->file_count, pct);
    } else {
        fprintf(stderr, "\r  %s  %zu/%zu  %" PRIu64 " bytes",
                p->filename ? p->filename : "?",
                p->file_index + 1, p->file_count,
                p->file_downloaded);
    }
    return 0;
}

static int cmd_pull(int argc, char **argv) {
    cli_pull_opts_t opts;
    char err[256] = {0};
    if (cli_parse_pull(argc, argv, &opts, err, sizeof(err)) != 0) {
        fprintf(stderr, "mlxd pull: %s\n", err);
        return 1;
    }
    registry_pull_opts_t pull_opts = {0};
    pull_opts.progress = pull_progress;
    char *dir = registry_pull(opts.spec, &pull_opts);
    if (!dir) {
        fprintf(stderr, "\nfailed to pull model\n");
        return 1;
    }
    fprintf(stderr, "\n");
    printf("%s\n", dir);
    free(dir);
    return 0;
}

static int cmd_list(int argc, char **argv) {
    cli_list_opts_t opts;
    char err[256] = {0};
    if (cli_parse_list(argc, argv, &opts, err, sizeof(err)) != 0) {
        fprintf(stderr, "mlxd list: %s\n", err);
        return 1;
    }

    int count = 0;
    registry_model_info_t *models = registry_discover(&count);
    if (!models || count == 0) {
        if (opts.json)
            printf("[]\n");
        else
            printf("no models found\n");
        return 0;
    }

    if (opts.json) {
        char *json = cli_list_json(models, count);
        if (json) {
            printf("%s\n", json);
            free(json);
        } else {
            fprintf(stderr, "mlxd list: out of memory\n");
            registry_model_list_free(models, count);
            return 1;
        }
    } else {
        printf("%-40s  %10s  %s\n", "MODEL", "SIZE", "MODIFIED");
        printf("%-40s  %10s  %s\n", "-----", "----", "--------");

        for (int i = 0; i < count; i++) {
            char size_buf[32];
            cli_human_size(models[i].size_bytes, size_buf, sizeof(size_buf));

            char time_buf[64] = "unknown";
            if (models[i].mtime > 0) {
                time_t t = (time_t)models[i].mtime;
                struct tm *tm = localtime(&t);
                if (tm)
                    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M", tm);
            }

            printf("%-40s  %10s  %s\n", models[i].id, size_buf, time_buf);
        }
    }

    registry_model_list_free(models, count);
    return 0;
}

static int cmd_serve(int argc, char **argv) {
    cli_serve_opts_t opts;
    char err[256] = {0};
    if (cli_parse_serve(argc, argv, &opts, err, sizeof(err)) != 0) {
        fprintf(stderr, "mlxd serve: %s\n", err);
        fprintf(stderr, "usage: mlxd serve --model <dir-or-spec> [--host ADDR] [--port N]\n");
        return 1;
    }

    char *resolved_dir = NULL;
    const char *model_dir = resolve_model_dir(opts.model, &resolved_dir);
    if (!model_dir) {
        fprintf(stderr, "mlxd serve: cannot find model '%s'\n", opts.model);
        return 1;
    }

    tokenizer_t *tok = tokenizer_load_dir(model_dir);
    if (!tok) {
        fprintf(stderr, "mlxd serve: failed to load tokenizer from '%s'\n", model_dir);
        free(resolved_dir);
        return 1;
    }

    char *chat_template = read_chat_template(model_dir);
    if (!chat_template)
        fprintf(stderr, "mlxd serve: no chat_template found; chat endpoints will return 400\n");

    engine_t eng;
    if (engine_init(&eng) != 0) {
        fprintf(stderr, "mlxd serve: failed to start engine thread\n");
        tokenizer_free(tok);
        free(chat_template);
        free(resolved_dir);
        return 1;
    }

    char *load_path = strdup(model_dir);
    if (!load_path) {
        fprintf(stderr, "mlxd serve: out of memory\n");
        engine_destroy(&eng);
        tokenizer_free(tok);
        free(chat_template);
        free(resolved_dir);
        return 1;
    }
    if (engine_post_load(&eng, load_path) != 0) {
        fprintf(stderr, "mlxd serve: failed to post model load\n");
        engine_destroy(&eng);
        tokenizer_free(tok);
        free(chat_template);
        free(resolved_dir);
        return 1;
    }

    /* Unbounded wait: large HF checkpoints can exceed 30s on first Metal compile. */
    if (engine_wait_load(&eng, -1) != 0) {
        print_load_failure("mlxd serve", &eng);
        engine_destroy(&eng);
        tokenizer_free(tok);
        free(chat_template);
        free(resolved_dir);
        return 1;
    }

    http_server_config_t cfg = {
        .host = opts.host,
        .port = opts.port,
        .engine = &eng,
        .tokenizer = tok,
        .chat_template = chat_template,
        .model_id = model_dir,
        .drain_deadline_ms = 0,
    };
    http_server_t *srv = http_server_create(&cfg);
    if (!srv) {
        fprintf(stderr, "mlxd serve: failed to create server on %s:%d\n",
                opts.host, opts.port);
        engine_destroy(&eng);
        tokenizer_free(tok);
        free(chat_template);
        free(resolved_dir);
        return 1;
    }

    fprintf(stderr, "mlxd serve: listening on %s:%d\n",
            opts.host, http_server_port(srv));

    g_srv = srv;
    atomic_store(&g_sigint_count, 0);
    struct sigaction sa = {.sa_handler = serve_sigint_handler};
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);

    http_server_start(srv);

    g_srv = NULL;
    signal(SIGINT, SIG_DFL);

    engine_destroy(&eng);
    http_server_destroy(srv);
    tokenizer_free(tok);
    free(chat_template);
    free(resolved_dir);
    return 0;
}

static int cmd_run(int argc, char **argv) {
    cli_run_opts_t opts;
    char err[256] = {0};
    if (cli_parse_run(argc, argv, &opts, err, sizeof(err)) != 0) {
        fprintf(stderr, "mlxd run: %s\n", err);
        fprintf(stderr, "usage: mlxd run MODEL [PROMPT] [--max-tokens N] [--temperature F] [--stream] [--raw] [--token-ids] [--]\n");
        return 1;
    }

    int rc = 1;
    char *resolved_dir = NULL;
    tokenizer_t *tok = NULL;
    char *chat_template = NULL;
    char *prompt_buf = NULL;
    int32_t *ids = NULL;
    bool eng_inited = false;
    engine_t eng;
    stream_t *s = NULL;
    finish_reason_t reason = FINISH_STOP;

    const char *model_dir = resolve_model_dir(opts.model, &resolved_dir);
    if (!model_dir) {
        fprintf(stderr, "mlxd run: cannot find model '%s'\n", opts.model);
        goto cleanup;
    }

    tok = tokenizer_load_dir(model_dir);
    if (!tok) {
        fprintf(stderr, "mlxd run: failed to load tokenizer from '%s'\n", model_dir);
        goto cleanup;
    }

    if (!opts.raw)
        chat_template = read_chat_template(model_dir);

    prompt_buf = cli_resolve_run_prompt(opts.prompt, stdin);
    if (!prompt_buf) {
        fprintf(stderr, "mlxd run: no prompt provided and stdin is empty\n");
        goto cleanup;
    }

    int n_ids = -1;
    const char *build_err = NULL;

    if (chat_template) {
        char *messages_json = cli_run_messages_json(prompt_buf);
        if (!messages_json) {
            fprintf(stderr, "mlxd run: failed to build messages JSON\n");
            goto cleanup;
        }
        n_ids = gen_build_chat_prompt(tok, chat_template, messages_json, NULL,
                                      &ids, &build_err);
        free(messages_json);
    } else {
        n_ids = gen_build_completion_prompt(tok, prompt_buf, &ids, &build_err);
    }

    free(prompt_buf);
    prompt_buf = NULL;

    if (n_ids < 0) {
        fprintf(stderr, "mlxd run: failed to tokenize prompt: %s\n",
                build_err ? build_err : "unknown error");
        goto cleanup;
    }

    atomic_store(&g_run_cancel, 0);
    struct sigaction sa = {.sa_handler = run_sigint_handler};
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);

    if (engine_init(&eng) != 0) {
        fprintf(stderr, "mlxd run: failed to start engine thread\n");
        goto cleanup;
    }
    eng_inited = true;

    char *load_path = strdup(model_dir);
    if (!load_path) {
        fprintf(stderr, "mlxd run: out of memory\n");
        goto cleanup;
    }
    if (engine_post_load(&eng, load_path) != 0) {
        fprintf(stderr, "mlxd run: failed to post model load\n");
        goto cleanup;
    }

    /* Unbounded wait; Ctrl-C still aborts via run_cancel_pred. */
    if (engine_wait_load_until(&eng, -1, run_cancel_pred, NULL) != 0) {
        if (atomic_load(&g_run_cancel) > 0) {
            reason = FINISH_CANCELLED;
            rc = 0;
            goto cleanup;
        }
        print_load_failure("mlxd run", &eng);
        goto cleanup;
    }

    s = stream_create(256);
    if (!s) {
        fprintf(stderr, "mlxd run: out of memory\n");
        goto cleanup;
    }

    gen_params_t params = {
        .sampling = SAMPLING_PARAMS_DEFAULT,
        .max_tokens = opts.max_tokens <= 0 ? INT_MAX : opts.max_tokens,
        .n = 1,
        .stream = true,
    };
    if (opts.temperature_set) {
        params.sampling.temperature = opts.temperature;
        params.sampling_set |= SAMPLING_SET_TEMPERATURE;
    }
    if (opts.top_p_set) {
        params.sampling.top_p = opts.top_p;
        params.sampling_set |= SAMPLING_SET_TOP_P;
    }
    if (opts.top_k_set) {
        params.sampling.top_k = opts.top_k;
        params.sampling_set |= SAMPLING_SET_TOP_K;
    }
    if (opts.min_p_set) {
        params.sampling.min_p = opts.min_p;
        params.sampling_set |= SAMPLING_SET_MIN_P;
    }
    if (opts.seed_set) {
        params.sampling.seed = opts.seed;
        params.sampling_set |= SAMPLING_SET_SEED;
    }

    engine_cmd_t *gen_cmd = calloc(1, sizeof(*gen_cmd));
    if (!gen_cmd) {
        fprintf(stderr, "mlxd run: out of memory\n");
        goto cleanup;
    }
    gen_cmd->tag = CMD_GENERATE;
    gen_cmd->generate.params = params;
    gen_cmd->generate.params.stop = NULL;
    gen_cmd->generate.params.stop_count = 0;
    gen_cmd->generate.token_ids = ids;
    gen_cmd->generate.token_count = n_ids;
    ids = NULL;
    stream_retain(s);
    gen_cmd->generate.stream = s;

    engine_post(&eng, gen_cmd);

    rc = cli_run_consume(s, tok, stdout, opts.stream, &g_run_cancel,
                         &reason, err, sizeof(err), opts.token_ids);

    if (rc != 0)
        fprintf(stderr, "\nmlxd run: %s\n", err);
    else if (!opts.token_ids)
        printf("\n");

cleanup:
    signal(SIGINT, SIG_DFL);
    free(prompt_buf);
    free(ids);
    if (s) stream_release(s);
    if (eng_inited) engine_destroy(&eng);
    tokenizer_free(tok);
    free(chat_template);
    free(resolved_dir);

    if (rc != 0) return 1;
    if (reason == FINISH_CANCELLED) return 130;
    return 0;
}

int main(int argc, char **argv) {
    cli_cmd_t cmd = cli_command(argc, argv);

    switch (cmd) {
    case CLI_CMD_SERVE:   return cmd_serve(argc, argv);
    case CLI_CMD_RUN:     return cmd_run(argc, argv);
    case CLI_CMD_PULL:    return cmd_pull(argc, argv);
    case CLI_CMD_LIST:    return cmd_list(argc, argv);
    case CLI_CMD_HELP:    usage(); return 0;
    case CLI_CMD_VERSION:
        printf("mlxd %s\n", MLXD_VERSION);
        return 0;
    case CLI_CMD_UNKNOWN:
        fprintf(stderr, "mlxd: unknown command '%s'\n", argv[1]);
        usage();
        return 1;
    case CLI_CMD_NONE:
        usage();
        return 1;
    }

    return 1;
}
