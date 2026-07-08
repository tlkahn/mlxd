#include "core/openai.h"

#include <stdlib.h>
#include <string.h>

/* TODO: implement parse/serialize functions module-by-module */

int chat_completion_request_parse(chat_completion_request_t *req, yyjson_val *root,
                                  const char **err) {
    (void)req;
    (void)root;
    *err = "not yet implemented";
    return -1;
}

void chat_completion_request_free(chat_completion_request_t *req) {
    if (!req)
        return;
    free(req->model);
    for (int i = 0; i < req->message_count; i++)
        message_free(&req->messages[i]);
    free(req->messages);
}

yyjson_mut_val *chat_completion_response_serialize(const chat_completion_response_t *resp,
                                                   yyjson_mut_doc *doc) {
    (void)resp;
    (void)doc;
    return NULL;
}

void chat_completion_response_free(chat_completion_response_t *resp) {
    if (!resp)
        return;
    free(resp->id);
    free(resp->model);
    free(resp->content);
    for (int i = 0; i < resp->tool_call_count; i++)
        tool_call_free(&resp->tool_calls[i]);
    free(resp->tool_calls);
}

int completion_request_parse(completion_request_t *req, yyjson_val *root, const char **err) {
    (void)req;
    (void)root;
    *err = "not yet implemented";
    return -1;
}

void completion_request_free(completion_request_t *req) {
    if (!req)
        return;
    free(req->model);
    free(req->prompt);
}

yyjson_mut_val *completion_response_serialize(const completion_response_t *resp,
                                              yyjson_mut_doc *doc) {
    (void)resp;
    (void)doc;
    return NULL;
}

void completion_response_free(completion_response_t *resp) {
    if (!resp)
        return;
    free(resp->id);
    free(resp->model);
    free(resp->text);
}

int embedding_request_parse(embedding_request_t *req, yyjson_val *root, const char **err) {
    (void)req;
    (void)root;
    *err = "not yet implemented";
    return -1;
}

void embedding_request_free(embedding_request_t *req) {
    if (!req)
        return;
    free(req->model);
    free(req->input);
    free(req->encoding_format);
}

yyjson_mut_val *model_list_serialize(const model_info_t *models, int count, yyjson_mut_doc *doc) {
    (void)models;
    (void)count;
    (void)doc;
    return NULL;
}

yyjson_mut_val *error_envelope_serialize(const char *message, const char *type, const char *code,
                                         yyjson_mut_doc *doc) {
    (void)message;
    (void)type;
    (void)code;
    (void)doc;
    return NULL;
}
