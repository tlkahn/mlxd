#include "cli/cmds.h"
#include "cli/cli.h"
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

/* --- cli_cmd_run ---------------------------------------------------------- */

int cli_cmd_run(const cli_args_t *args, FILE *in, FILE *out, FILE *err) {
    const char *model_spec = args->run.model;

    char *model_dir = registry_resolve(model_spec);
    if (!model_dir) {
        fprintf(err, "run: cannot resolve model '%s'\n", model_spec);
        return 1;
    }

    /* Defer full implementation to Phase D */
    (void)in;
    (void)out;
    (void)args;
    fprintf(err, "run: not yet implemented\n");
    free(model_dir);
    return 1;
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
