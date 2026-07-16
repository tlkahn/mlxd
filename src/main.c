#include "core/log.h"
#include "registry/registry.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

int main(int argc, char **argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    const char *cmd = argv[1];

    if (strcmp(cmd, "serve") == 0) {
        fprintf(stderr, "mlxd serve: not yet implemented\n");
        return 1;
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
