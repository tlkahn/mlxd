#ifndef MLXD_HTTP_GEN_H
#define MLXD_HTTP_GEN_H

#include "core/openai.h"
#include "core/types.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct tokenizer tokenizer_t;

/* Build a token-id prompt from a chat template + messages JSON.
 * The rendered template is encoded with parse_special=true so that
 * template-owned control tokens (e.g. <|endoftext|>, <|im_start|>) are
 * atomized to their canonical ids (HF apply_chat_template parity).
 * Caveat: special-token literals inside user message content are also
 * atomized; this matches HF/vLLM behavior and is pinned by test.
 * Returns id count (>= 0) on success, -1 on error (*err set to a static
 * literal - caller does NOT free *err). *out_ids is malloc'd; caller frees. */
int gen_build_chat_prompt(const tokenizer_t *tok, const char *chat_template,
                          const char *messages_json, const char *tools_json,
                          int32_t **out_ids, const char **err);

/* Build a token-id prompt from a raw completion string.
 * Same return / ownership contract as gen_build_chat_prompt. */
int gen_build_completion_prompt(const tokenizer_t *tok, const char *prompt,
                                int32_t **out_ids, const char **err);

typedef struct {
    const char *id;
    const char *model;
    int64_t created;
    bool role_first;
    const char *delta_text;
    bool final;
    finish_reason_t reason;
    const token_logprob_t *logprob;
    bool include_usage;
    const usage_t *usage;
} gen_sse_chunk_params_t;

/* Build an SSE-wrapped chat.completion.chunk JSON.
 * Fields default to zero/false/NULL via compound-literal designated init.
 * Returns heap-allocated SSE string ("data: ...\n\n"). Caller frees. */
char *gen_sse_chunk(const gen_sse_chunk_params_t *p);

/* Build a complete chat completion response JSON. u may be NULL (zeroed usage).
 * Caller frees. */
char *gen_build_chat_response(const char *id, const char *model, int64_t created,
                              const char *content, finish_reason_t reason,
                              const usage_t *u,
                              const token_logprob_t *logprobs, int logprobs_count);

/* Build a complete text completion response JSON. u may be NULL (zeroed usage).
 * Caller frees. */
char *gen_build_completion_response(const char *id, const char *model, int64_t created,
                                    const char *text, finish_reason_t reason,
                                    const usage_t *u);

/* Build an SSE-wrapped OpenAI error event JSON.
 * type defaults to "server_error" when NULL; code is omitted when NULL.
 * Returns heap-allocated SSE string. Caller frees. */
char *gen_sse_error(const char *msg);
char *gen_sse_error_ex(const char *msg, const char *type, const char *code);

/* Generate a unique request ID: "<prefix><24 lowercase hex chars>".
 * prefix is typically "chatcmpl-" or "cmpl-". Caller frees. */
char *gen_make_id(const char *prefix);

typedef struct {
    const char *id;
    const char *model;
    int64_t created;
    const char *delta_text;
    bool final;
    finish_reason_t reason;
} gen_sse_completion_chunk_params_t;

/* Build an SSE-wrapped text_completion chunk JSON (legacy completions streaming).
 * Returns heap-allocated SSE string. Caller frees. */
char *gen_sse_completion_chunk(const gen_sse_completion_chunk_params_t *p);

#endif
