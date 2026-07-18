#ifndef MLXD_CORE_OPENAI_H
#define MLXD_CORE_OPENAI_H

#include "core/types.h"
#include <yyjson/yyjson.h>

/* All OpenAI wire DTOs.
 *
 * Parse functions take a yyjson_val (object root) and populate the struct.
 * They return 0 on success, -1 on error (and set *err).
 *
 * Serialize functions return a yyjson_mut_val* added to the given mut_doc.
 *
 * Structs own their strings (caller must free via the corresponding _free). */

/* --- Chat completion request ---------------------------------------------- */

typedef struct {
    char           *model;
    message_t      *messages;
    int             message_count;
    gen_params_t    params;
    tool_t         *tools;
    int             tool_count;
    tool_choice_t   tool_choice;
    bool            include_usage; /* stream_options.include_usage */
} chat_completion_request_t;

int  chat_completion_request_parse(chat_completion_request_t *req, yyjson_val *root, const char **err);
void chat_completion_request_free(chat_completion_request_t *req);

/* --- Chat completion response --------------------------------------------- */

/* One candidate in a sampled token's top_logprobs list. */
typedef struct {
    char  *token;
    float  logprob;
} top_logprob_t;

/* Logprob detail for one sampled token. */
typedef struct {
    char          *token;
    float          logprob;
    top_logprob_t *top_logprobs;
    int            top_logprob_count;
} token_logprob_t;

typedef struct {
    char            *id;
    char            *model;
    int64_t          created;
    finish_reason_t  finish_reason;
    char            *content;
    tool_call_t     *tool_calls;
    int              tool_call_count;
    token_logprob_t *logprobs_content; /* NULL when logprobs were not requested */
    int              logprobs_count;
    usage_t          usage;
} chat_completion_response_t;

yyjson_mut_val *chat_completion_response_serialize(const chat_completion_response_t *resp,
                                                   yyjson_mut_doc *doc);
void chat_completion_response_free(chat_completion_response_t *resp);

/* --- Chat completion chunk (streaming SSE payload) -----------------------
 *
 * Flattened to the engine-emitting shape (n=1). For every optional string, a
 * NULL pointer omits the key entirely while "" emits a present-but-empty
 * string; this mirrors the Zig reference's emit_null_optional_fields=false and
 * lets streamed tool-call fragments carry a header once and deltas thereafter. */

typedef struct {
    int   index;
    char *id;            /* NULL -> omit key */
    char *type;          /* NULL -> omit key */
    char *function_name; /* NULL -> omit key */
    char *arguments;     /* NULL -> omit key; "" -> present-and-empty */
} delta_tool_call_t;

typedef struct {
    char              *id;
    char              *model;
    int64_t            created;
    bool               has_choice; /* false -> empty choices[] (final usage chunk) */
    bool               has_role;
    role_t             role;
    char              *delta_content; /* NULL -> omit key */
    delta_tool_call_t *tool_calls;
    int                tool_call_count;
    bool               has_finish_reason;
    finish_reason_t    finish_reason;
    token_logprob_t   *logprobs;
    int                logprob_count;
    bool               has_usage;
    usage_t            usage;
} chat_completion_chunk_t;

yyjson_mut_val *chat_completion_chunk_serialize(const chat_completion_chunk_t *chunk,
                                                yyjson_mut_doc *doc);
void chat_completion_chunk_free(chat_completion_chunk_t *chunk);

/* --- Completion request ---------------------------------------------------
 *
 * `prompt` is string-only: the OpenAI string-array and token-ID-array forms are
 * deferred (v2); parsing an array prompt returns -1 with a clear error. */

typedef struct {
    char         *model;
    char         *prompt;
    gen_params_t  params;
} completion_request_t;

int  completion_request_parse(completion_request_t *req, yyjson_val *root, const char **err);
void completion_request_free(completion_request_t *req);

/* --- Completion response -------------------------------------------------- */

typedef struct {
    char            *id;
    char            *model;
    int64_t          created;
    finish_reason_t  finish_reason;
    char            *text;
    usage_t          usage;
} completion_response_t;

yyjson_mut_val *completion_response_serialize(const completion_response_t *resp,
                                              yyjson_mut_doc *doc);
void completion_response_free(completion_response_t *resp);

/* --- Embedding request ----------------------------------------------------
 *
 * `input` is string-only (array forms deferred to v2). `encoding_format`
 * defaults to "float" when absent. The wire `dimensions` and `user` fields are
 * accepted but ignored here; honoring them is deferred to the engine layer. */

typedef struct {
    char  *model;
    char  *input;
    char  *encoding_format;
} embedding_request_t;

int  embedding_request_parse(embedding_request_t *req, yyjson_val *root, const char **err);
void embedding_request_free(embedding_request_t *req);

/* --- Embedding response --------------------------------------------------- */

typedef struct {
    int    index;
    float *values;
    int    value_count;
} embedding_data_t;

typedef struct {
    char             *model;
    embedding_data_t *data;
    int               data_count;
    usage_t           usage;
    char             *encoding_format; /* "float" (default) | "base64" */
} embedding_response_t;

yyjson_mut_val *embedding_response_serialize(const embedding_response_t *resp, yyjson_mut_doc *doc);
void embedding_response_free(embedding_response_t *resp);

/* --- Model list ----------------------------------------------------------- */

typedef struct {
    char    *id;
    int64_t  created;
    char    *owned_by;
} model_info_t;

yyjson_mut_val *model_list_serialize(const model_info_t *models, int count, yyjson_mut_doc *doc);

/* --- Error envelope ------------------------------------------------------- */

yyjson_mut_val *error_envelope_serialize(const char *message, const char *type, const char *code,
                                         yyjson_mut_doc *doc);

#endif
