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
} chat_completion_request_t;

int  chat_completion_request_parse(chat_completion_request_t *req, yyjson_val *root, const char **err);
void chat_completion_request_free(chat_completion_request_t *req);

/* --- Chat completion response --------------------------------------------- */

typedef struct {
    char            *id;
    char            *model;
    int64_t          created;
    finish_reason_t  finish_reason;
    char            *content;
    tool_call_t     *tool_calls;
    int              tool_call_count;
    usage_t          usage;
} chat_completion_response_t;

yyjson_mut_val *chat_completion_response_serialize(const chat_completion_response_t *resp,
                                                   yyjson_mut_doc *doc);
void chat_completion_response_free(chat_completion_response_t *resp);

/* --- Completion request --------------------------------------------------- */

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

/* --- Embedding request ---------------------------------------------------- */

typedef struct {
    char  *model;
    char  *input;
    char  *encoding_format;
} embedding_request_t;

int  embedding_request_parse(embedding_request_t *req, yyjson_val *root, const char **err);
void embedding_request_free(embedding_request_t *req);

/* --- Model list ----------------------------------------------------------- */

typedef struct {
    char *id;
    char *owned_by;
} model_info_t;

yyjson_mut_val *model_list_serialize(const model_info_t *models, int count, yyjson_mut_doc *doc);

/* --- Error envelope ------------------------------------------------------- */

yyjson_mut_val *error_envelope_serialize(const char *message, const char *type, const char *code,
                                         yyjson_mut_doc *doc);

#endif
