#include "core/log.h"

#include <stdio.h>
#include <string.h>

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
        fprintf(stderr, "mlxd pull: not yet implemented\n");
        return 1;
    }
    if (strcmp(cmd, "list") == 0) {
        fprintf(stderr, "mlxd list: not yet implemented\n");
        return 1;
    }

    if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        usage();
        return 0;
    }

    fprintf(stderr, "mlxd: unknown command '%s'\n", cmd);
    usage();
    return 1;
}
