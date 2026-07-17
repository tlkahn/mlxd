#include "cli/cmds.h"
#include "cli/cli.h"
#include "engine/engine.h"
#include "model/detok.h"
#include "model/tokenizer.h"
#include "registry/registry.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "yyjson/yyjson.h"

/* --- Pure helpers --------------------------------------------------------- */

sigint_action_t cli_sigint_decide(int count) {
    return (count <= 1) ? SIGINT_STOP : SIGINT_FORCE_EXIT;
}

char *cli_list_json(const registry_model_info_t *models, int count) {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *arr = yyjson_mut_arr(doc);
    yyjson_mut_doc_set_root(doc, arr);

    for (int i = 0; i < count; i++) {
        yyjson_mut_val *obj = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_str(doc, obj, "id", models[i].id);
        yyjson_mut_obj_add_str(doc, obj, "path", models[i].path);
        yyjson_mut_obj_add_uint(doc, obj, "size_bytes", models[i].size_bytes);
        yyjson_mut_obj_add_sint(doc, obj, "mtime", models[i].mtime);
        yyjson_mut_arr_append(arr, obj);
    }

    char *json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    return json;
}

/* --- human_size / pull_progress (moved from main.c) ----------------------- */

static void human_size(uint64_t bytes, char *buf, size_t bufsz) {
    if (bytes >= (uint64_t)1024 * 1024 * 1024)
        snprintf(buf, bufsz, "%.1f GB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
    else if (bytes >= 1024 * 1024)
        snprintf(buf, bufsz, "%.1f MB", (double)bytes / (1024.0 * 1024.0));
    else if (bytes >= 1024)
        snprintf(buf, bufsz, "%.1f KB", (double)bytes / 1024.0);
    else
        snprintf(buf, bufsz, "%" PRIu64 " B", bytes);
}

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

/* --- cli_cmd_list --------------------------------------------------------- */

int cli_cmd_list(const cli_args_t *args, FILE *out, FILE *err) {
    int count = 0;
    registry_model_info_t *models = registry_discover(&count);

    if (args->list.json) {
        char *json = cli_list_json(models, count);
        fprintf(out, "%s\n", json);
        free(json);
        registry_model_list_free(models, count);
        return 0;
    }

    if (!models || count == 0) {
        fprintf(err, "no models found\n");
        registry_model_list_free(models, count);
        return 0;
    }

    fprintf(out, "%-40s  %10s  %s\n", "MODEL", "SIZE", "MODIFIED");
    fprintf(out, "%-40s  %10s  %s\n", "-----", "----", "--------");

    for (int i = 0; i < count; i++) {
        char size_buf[32];
        human_size(models[i].size_bytes, size_buf, sizeof(size_buf));

        char time_buf[64] = "unknown";
        if (models[i].mtime > 0) {
            time_t t = (time_t)models[i].mtime;
            struct tm *tm = localtime(&t);
            if (tm)
                strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M", tm);
        }

        fprintf(out, "%-40s  %10s  %s\n", models[i].id, size_buf, time_buf);
    }

    registry_model_list_free(models, count);
    return 0;
}

/* --- cli_cmd_pull --------------------------------------------------------- */

int cli_cmd_pull(const cli_args_t *args, FILE *out, FILE *err) {
    registry_pull_opts_t opts = {0};
    opts.progress = pull_progress;
    char *dir = registry_pull(args->pull.spec, &opts);
    if (!dir) {
        fprintf(err, "\nfailed to pull model\n");
        return 1;
    }
    fprintf(err, "\n");
    fprintf(out, "%s\n", dir);
    free(dir);
    return 0;
}

/* --- stdin slurp ---------------------------------------------------------- */

static char *slurp_stdin(FILE *in) {
    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;

    for (;;) {
        size_t n = fread(buf + len, 1, cap - len, in);
        len += n;
        if (n == 0) break;
        if (len == cap) {
            cap *= 2;
            char *tmp = realloc(buf, cap);
            if (!tmp) { free(buf); return NULL; }
            buf = tmp;
        }
    }

    while (len > 0 && buf[len - 1] == '\n')
        len--;
    buf[len] = '\0';
    return buf;
}

/* --- cli_cmd_run ---------------------------------------------------------- */

int cli_cmd_run(const cli_args_t *args, FILE *in, FILE *out, FILE *err) {
    int rc = 1;
    const char *model_spec = args->run.model;

    char *model_dir = registry_resolve(model_spec);
    if (!model_dir) {
        fprintf(err, "run: cannot resolve model '%s'\n", model_spec);
        return 1;
    }

    tokenizer_t *tok = tokenizer_load_dir(model_dir);
    if (!tok) {
        fprintf(err, "run: failed to load tokenizer from '%s'\n", model_dir);
        free(model_dir);
        return 1;
    }

    engine_t eng;
    if (engine_init(&eng) != 0) {
        fprintf(err, "run: failed to init engine\n");
        tokenizer_free(tok);
        free(model_dir);
        return 1;
    }

    engine_cmd_t *load_cmd = calloc(1, sizeof(*load_cmd));
    load_cmd->tag = CMD_LOAD;
    load_cmd->load.model_path = strdup(model_dir);
    engine_post(&eng, load_cmd);

    char *prompt = NULL;
    if (args->run.prompt) {
        prompt = strdup(args->run.prompt);
    } else {
        prompt = slurp_stdin(in);
    }
    if (!prompt || prompt[0] == '\0') {
        fprintf(err, "run: empty prompt\n");
        free(prompt);
        engine_destroy(&eng);
        tokenizer_free(tok);
        free(model_dir);
        return 1;
    }

    int32_t *ids = NULL;
    int id_count = tokenizer_encode_alloc(tok, prompt, strlen(prompt), false, &ids);
    if (id_count < 0) {
        fprintf(err, "run: tokenizer encode failed\n");
        free(prompt);
        engine_destroy(&eng);
        tokenizer_free(tok);
        free(model_dir);
        return 1;
    }

    stream_t *s = stream_create(id_count + 16);
    stream_retain(s);

    engine_cmd_t *gen_cmd = calloc(1, sizeof(*gen_cmd));
    gen_cmd->tag = CMD_GENERATE;
    gen_cmd->generate.params.max_tokens = args->run.max_tokens;
    gen_cmd->generate.params.sampling.temperature = args->run.temperature;
    gen_cmd->generate.token_ids = ids;
    gen_cmd->generate.token_count = id_count;
    gen_cmd->generate.stream = s;
    engine_post(&eng, gen_cmd);

    if (args->run.stream) {
        detok_t *dt = detok_create(tok);
        chunk_t c;
        while (stream_next(s, &c, -1)) {
            if (c.tag == CHUNK_TOKEN) {
                if (dt) {
                    char *text = NULL;
                    size_t text_len = 0;
                    detok_feed(dt, c.token.id, &text, &text_len);
                    if (text_len > 0)
                        fwrite(text, 1, text_len, out);
                    free(text);
                }
            } else if (c.tag == CHUNK_DONE) {
                if (dt) {
                    char *text = NULL;
                    size_t text_len = 0;
                    detok_flush(dt, &text, &text_len);
                    if (text_len > 0)
                        fwrite(text, 1, text_len, out);
                    free(text);
                }
                rc = 0;
                break;
            } else if (c.tag == CHUNK_ERROR) {
                fprintf(err, "run: %s\n", c.error);
                free(c.error);
                break;
            }
        }
        fprintf(out, "\n");
        detok_free(dt);
    } else {
        int32_t *out_ids = NULL;
        int out_cap = 0, out_len = 0;
        chunk_t c;
        while (stream_next(s, &c, -1)) {
            if (c.tag == CHUNK_TOKEN) {
                if (out_len == out_cap) {
                    out_cap = out_cap ? out_cap * 2 : 64;
                    out_ids = realloc(out_ids, sizeof(int32_t) * (size_t)out_cap);
                }
                out_ids[out_len++] = c.token.id;
            } else if (c.tag == CHUNK_DONE) {
                rc = 0;
                break;
            } else if (c.tag == CHUNK_ERROR) {
                fprintf(err, "run: %s\n", c.error);
                free(c.error);
                break;
            }
        }
        if (rc == 0 && out_len > 0) {
            char *text = tokenizer_decode(tok, out_ids, out_len);
            if (text) {
                fprintf(out, "%s\n", text);
                free(text);
            }
        }
        free(out_ids);
    }

    stream_release(s);
    engine_destroy(&eng);
    tokenizer_free(tok);
    free(model_dir);
    free(prompt);
    return rc;
}

/* --- cli_cmd_serve -------------------------------------------------------- */

int cli_cmd_serve(const cli_args_t *args, FILE *out, FILE *err) {
    (void)args;
    (void)out;
    fprintf(err, "serve: not yet implemented\n");
    return 1;
}

/* --- cli_main ------------------------------------------------------------- */

int cli_main(int argc, char **argv, FILE *in, FILE *out, FILE *err) {
    cli_args_t args;
    cli_parse(argc, argv, &args);

    switch (args.cmd) {
    case CLI_HELP:
        fprintf(out, "%s", cli_usage_str());
        return 0;
    case CLI_VERSION:
        fprintf(out, "mlxd %s\n", MLXD_VERSION);
        return 0;
    case CLI_ERROR:
        fprintf(err, "mlxd: %s\n%s", args.err, cli_usage_str());
        return 1;
    case CLI_SERVE:
        return cli_cmd_serve(&args, out, err);
    case CLI_RUN:
        return cli_cmd_run(&args, in, out, err);
    case CLI_PULL:
        return cli_cmd_pull(&args, out, err);
    case CLI_LIST:
        return cli_cmd_list(&args, out, err);
    }
    return 1;
}
