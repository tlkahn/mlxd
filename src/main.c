#include "cli/cmds.h"

#include <stdio.h>

int main(int argc, char **argv) {
    return cli_main(argc, argv, stdin, stdout, stderr);
}
