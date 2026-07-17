#ifndef CLI_IO_H
#define CLI_IO_H

#include "core/types.h"
#include "engine/engine.h"
#include "model/tokenizer.h"
#include "registry/registry.h"

#include <stdbool.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

char *cli_resolve_run_prompt(const char *positional, FILE *stdin_stream);

char *cli_run_messages_json(const char *prompt);

int cli_run_consume(stream_t *s, const tokenizer_t *tok, FILE *out, bool flush_each,
                    const _Atomic int *cancel_flag, finish_reason_t *reason,
                    char *err, size_t errsz);

void cli_human_size(uint64_t bytes, char *buf, size_t bufsz);

char *cli_list_json(const registry_model_info_t *models, int count);

#endif
