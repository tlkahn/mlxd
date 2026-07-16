#include "core/log.h"
#include "engine/engine.h"
#include "http/server.h"
#include "model/tokenizer.h"
#include "registry/registry.h"

#include <errno.h>
#include <inttypes.h>
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
            "  run     one-shot or interactive text generation\n"
            "  pull    download model from HuggingFace\n"
            "  list    list locally available models\n");
}

/* --- Signal handling for serve -------------------------------------------- */

static _Atomic int g_sigint_count;
static http_server_t *g_srv;

static void sigint_handler(int sig) {
    (void)sig;
    int prev = atomic_fetch_add(&g_sigint_count, 1);
    if (prev == 0)
        http_server_stop(g_srv);
    else
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

static int cmd_pull(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: mlxd pull <model-spec>\n");
        return 1;
    }
    registry_pull_opts_t opts = {0};
    opts.progress = pull_progress;
    char *dir = registry_pull(argv[2], &opts);
    if (!dir) {
        fprintf(stderr, "\nfailed to pull model\n");
        return 1;
    }
    fprintf(stderr, "\n");
    printf("%s\n", dir);
    free(dir);
    return 0;
}

static int cmd_list(void) {
    int count = 0;
    registry_model_info_t *models = registry_discover(&count);
    if (!models || count == 0) {
        printf("no models found\n");
        return 0;
    }

    printf("%-40s  %10s  %s\n", "MODEL", "SIZE", "MODIFIED");
    printf("%-40s  %10s  %s\n", "-----", "----", "--------");

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

        printf("%-40s  %10s  %s\n", models[i].id, size_buf, time_buf);
    }

    registry_model_list_free(models, count);
    return 0;
}

static int cmd_serve(int argc, char **argv) {
    int port = 8080;
    const char *model_arg = NULL;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            char *end;
            long v = strtol(argv[++i], &end, 10);
            if (*end != '\0' || v < 1 || v > 65535) {
                fprintf(stderr, "mlxd serve: invalid port '%s'\n", argv[i]);
                return 1;
            }
            port = (int)v;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_arg = argv[++i];
        } else {
            fprintf(stderr, "mlxd serve: unknown option '%s'\n", argv[i]);
            fprintf(stderr, "usage: mlxd serve --model <dir-or-spec> [--port N]\n");
            return 1;
        }
    }

    if (!model_arg) {
        fprintf(stderr, "mlxd serve: --model is required\n");
        fprintf(stderr, "usage: mlxd serve --model <dir-or-spec> [--port N]\n");
        return 1;
    }

    const char *model_dir = model_arg;
    char *resolved_dir = NULL;
    if (!file_exists(model_arg, "tokenizer.json")) {
        resolved_dir = registry_resolve(model_arg);
        if (!resolved_dir) {
            fprintf(stderr, "mlxd serve: cannot find model '%s'\n", model_arg);
            return 1;
        }
        model_dir = resolved_dir;
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
    engine_init(&eng);

    engine_cmd_t *cmd = calloc(1, sizeof(*cmd));
    if (!cmd) {
        fprintf(stderr, "mlxd serve: out of memory\n");
        tokenizer_free(tok);
        free(chat_template);
        free(resolved_dir);
        engine_destroy(&eng);
        return 1;
    }
    cmd->tag = CMD_LOAD;
    cmd->load.model_path = strdup(model_dir);
    engine_post(&eng, cmd);

    http_server_config_t cfg = {
        .host = "127.0.0.1",
        .port = port,
        .engine = &eng,
        .tokenizer = tok,
        .chat_template = chat_template,
        .model_id = model_dir,
        .drain_deadline_ms = 0,
    };
    http_server_t *srv = http_server_create(&cfg);
    if (!srv) {
        fprintf(stderr, "mlxd serve: failed to create server on port %d\n", port);
        engine_destroy(&eng);
        tokenizer_free(tok);
        free(chat_template);
        free(resolved_dir);
        return 1;
    }

    fprintf(stderr, "mlxd serve: listening on 127.0.0.1:%d\n",
            http_server_port(srv));

    atomic_store(&g_sigint_count, 0);
    g_srv = srv;
    struct sigaction sa = {.sa_handler = sigint_handler};
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);

    http_server_start(srv);

    engine_destroy(&eng);
    http_server_destroy(srv);
    tokenizer_free(tok);
    free(chat_template);
    free(resolved_dir);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    const char *cmd = argv[1];

    if (strcmp(cmd, "serve") == 0) {
        return cmd_serve(argc, argv);
    }
    if (strcmp(cmd, "run") == 0) {
        fprintf(stderr, "mlxd run: not yet implemented\n");
        return 1;
    }
    if (strcmp(cmd, "pull") == 0) {
        return cmd_pull(argc, argv);
    }
    if (strcmp(cmd, "list") == 0) {
        return cmd_list();
    }

    if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        usage();
        return 0;
    }

    fprintf(stderr, "mlxd: unknown command '%s'\n", cmd);
    usage();
    return 1;
}
