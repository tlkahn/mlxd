#ifndef MLXD_HTTP_GEN_H
#define MLXD_HTTP_GEN_H

#include "core/types.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct tokenizer tokenizer_t;

/* Build a token-id prompt from a chat template + messages JSON.
 * Returns id count (>= 0) on success, -1 on error (*err set to a static
 * literal - caller does NOT free *err). *out_ids is malloc'd; caller frees. */
int gen_build_chat_prompt(const tokenizer_t *tok, const char *chat_template,
                          const char *messages_json, const char *tools_json,
                          int32_t **out_ids, const char **err);

/* Build a token-id prompt from a raw completion string.
 * Same return / ownership contract as gen_build_chat_prompt. */
int gen_build_completion_prompt(const tokenizer_t *tok, const char *prompt,
                                int32_t **out_ids, const char **err);

/* Build an SSE-wrapped chat.completion.chunk JSON.
 * role_first: emit delta.role = "assistant".
 * delta_text: emit delta.content (NULL omits the key).
 * final: emit finish_reason.
 * include_usage: emit usage object.
 * Returns heap-allocated SSE string ("data: ...\n\n"). Caller frees. */
char *gen_sse_chunk(const char *id, const char *model, int64_t created,
                    bool role_first, const char *delta_text, bool final,
                    finish_reason_t reason, bool include_usage, const usage_t *usage);

/* Build a complete chat completion response JSON. u may be NULL (zeroed usage).
 * Caller frees. */
char *gen_build_chat_response(const char *id, const char *model, int64_t created,
                              const char *content, finish_reason_t reason,
                              const usage_t *u);

/* Build a complete text completion response JSON. u may be NULL (zeroed usage).
 * Caller frees. */
char *gen_build_completion_response(const char *id, const char *model, int64_t created,
                                    const char *text, finish_reason_t reason,
                                    const usage_t *u);

#endif
