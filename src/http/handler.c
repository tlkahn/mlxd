#include "http/handler.h"
#include "http/gen_request.h"
#include "core/openai.h"
#include "registry/registry.h"

#include <stdlib.h>
#include <string.h>
#include <yyjson/yyjson.h>

/* --- Shared helpers ------------------------------------------------------ */

static void respond_json_error(http_response_t *resp, int status,
                               const char *type, const char *code,
                               const char *msg) {
    resp->content_type = "application/json";
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) { resp->status = 500; return; }
    yyjson_mut_val *root = error_envelope_serialize(msg, type, code, doc);
    yyjson_mut_doc_set_root(doc, root);
    resp->body = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    if (!resp->body) { resp->status = 500; return; }
    resp->body_len = strlen(resp->body);
    resp->status = status;
}

/* --- GET /v1/models ------------------------------------------------------ */

static void handle_models(const http_request_t *req, http_response_t *resp,
                          void *ctx) {
    (void)req;
    (void)ctx;
    resp->content_type = "application/json";

    int count = 0;
    registry_model_info_t *rl = registry_discover(&count);

    model_info_t *models = NULL;
    if (count > 0) {
        models = calloc((size_t)count, sizeof(*models));
        if (!models) {
            registry_model_list_free(rl, count);
            resp->status = 500;
            return;
        }
        for (int i = 0; i < count; i++) {
            models[i].id = rl[i].id;
            models[i].created = rl[i].mtime;
            models[i].owned_by = "mlxd";
        }
    }

    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) {
        free(models);
        registry_model_list_free(rl, count);
        resp->status = 500;
        return;
    }

    yyjson_mut_val *root = model_list_serialize(models, count, doc);
    yyjson_mut_doc_set_root(doc, root);
    resp->body = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    free(models);
    registry_model_list_free(rl, count);

    if (!resp->body) { resp->status = 500; return; }
    resp->body_len = strlen(resp->body);
    resp->status = 200;
}

/* --- POST /v1/embeddings (501 stub) -------------------------------------- */

static void handle_embeddings(const http_request_t *req, http_response_t *resp,
                              void *ctx) {
    (void)ctx;

    yyjson_doc *doc = yyjson_read(req->body, req->body_len, 0);
    if (!doc) {
        respond_json_error(resp, 400, "invalid_request_error", NULL,
                           "Invalid JSON in request body");
        return;
    }

    yyjson_val *root = yyjson_doc_get_root(doc);
    embedding_request_t ereq = {0};
    const char *err = NULL;
    int rc = embedding_request_parse(&ereq, root, &err);
    if (rc < 0) {
        respond_json_error(resp, 400, "invalid_request_error", NULL,
                           err ? err : "Invalid embedding request");
        yyjson_doc_free(doc);
        return;
    }

    embedding_request_free(&ereq);
    yyjson_doc_free(doc);

    respond_json_error(resp, 501, "not_implemented_error", "not_implemented",
                       "Embeddings are not yet implemented");
}

static const char *error_type_for_status(int status) {
    switch (status) {
    case 400: return "invalid_request_error";
    case 503: return "service_unavailable";
    default:  return "server_error";
    }
}

/* --- POST /v1/chat/completions ------------------------------------------- */

static void handle_chat_completions(const http_request_t *req,
                                    http_response_t *resp, void *ctx) {
    serve_ctx_t *sctx = ctx;

    if (!sctx->tokenizer) {
        respond_json_error(resp, 503, "service_unavailable", "model_not_loaded",
                           "No model loaded");
        return;
    }
    if (!sctx->chat_template) {
        respond_json_error(resp, 400, "invalid_request_error", NULL,
                           "Chat template not configured");
        return;
    }

    yyjson_doc *doc = yyjson_read(req->body, req->body_len, 0);
    if (!doc) {
        respond_json_error(resp, 400, "invalid_request_error", NULL,
                           "Invalid JSON in request body");
        return;
    }

    yyjson_val *root = yyjson_doc_get_root(doc);
    chat_completion_request_t creq = {0};
    const char *parse_err = NULL;
    if (chat_completion_request_parse(&creq, root, &parse_err) < 0) {
        respond_json_error(resp, 400, "invalid_request_error", NULL,
                           parse_err ? parse_err : "Invalid chat completion request");
        yyjson_doc_free(doc);
        return;
    }

    if (!sampling_params_validate(&creq.params.sampling, &parse_err)) {
        respond_json_error(resp, 400, "invalid_request_error", NULL, parse_err);
        chat_completion_request_free(&creq);
        yyjson_doc_free(doc);
        return;
    }

    /* Non-OpenAI extensions are parsed here by design; core/openai stays spec-pure. */
    const char *extra_json = NULL;
    yyjson_val *et = yyjson_obj_get(root, "enable_thinking");
    if (et) {
        if (yyjson_is_bool(et)) {
            extra_json = yyjson_get_bool(et)
                ? "{\"enable_thinking\":true}"
                : "{\"enable_thinking\":false}";
        } else {
            respond_json_error(resp, 400, "invalid_request_error", NULL,
                               "enable_thinking must be a boolean");
            chat_completion_request_free(&creq);
            yyjson_doc_free(doc);
            return;
        }
    }

    /* Serialize from parsed creq so content-part arrays are flattened to
     * strings before Jinja render (issue #126). Raw yyjson_val_write would
     * pass arrays through and break string-content HF templates. */
    yyjson_mut_doc *msg_doc = yyjson_mut_doc_new(NULL);
    if (!msg_doc) {
        respond_json_error(resp, 500, "server_error", NULL, "out of memory");
        chat_completion_request_free(&creq);
        yyjson_doc_free(doc);
        return;
    }
    yyjson_mut_val *msg_arr =
        messages_template_serialize(creq.messages, creq.message_count, msg_doc);
    if (!msg_arr) {
        respond_json_error(resp, 500, "server_error", NULL, "out of memory");
        yyjson_mut_doc_free(msg_doc);
        chat_completion_request_free(&creq);
        yyjson_doc_free(doc);
        return;
    }
    yyjson_mut_doc_set_root(msg_doc, msg_arr);
    char *messages_json = yyjson_mut_write(msg_doc, 0, NULL);
    yyjson_mut_doc_free(msg_doc);
    if (!messages_json) {
        respond_json_error(resp, 500, "server_error", NULL, "out of memory");
        chat_completion_request_free(&creq);
        yyjson_doc_free(doc);
        return;
    }

    yyjson_val *tools_val = yyjson_obj_get(root, "tools");
    char *tools_json = tools_val ? yyjson_val_write(tools_val, 0, NULL) : NULL;

    gen_request_start_params_t p = {
        .ctx = sctx,
        .conn = req->ctx,
        .chat = true,
        .stream = creq.params.stream,
        .include_usage = creq.include_usage,
        .model_id = sctx->model_id,
        .params = creq.params,
        .messages_json = messages_json,
        .tools_json = tools_json,
        .extra_json = extra_json,
    };

    const char *err = NULL;
    int rc = gen_request_start(&p, &err);
    if (rc != 0) {
        respond_json_error(resp, rc, error_type_for_status(rc), NULL,
                           err ? err : "generation failed");
        free(messages_json);
        free(tools_json);
        chat_completion_request_free(&creq);
        yyjson_doc_free(doc);
        return;
    }

    resp->deferred = true;
    free(messages_json);
    free(tools_json);
    chat_completion_request_free(&creq);
    yyjson_doc_free(doc);
}

/* --- POST /v1/completions ------------------------------------------------ */

static void handle_completions(const http_request_t *req,
                               http_response_t *resp, void *ctx) {
    serve_ctx_t *sctx = ctx;

    if (!sctx->tokenizer) {
        respond_json_error(resp, 503, "service_unavailable", "model_not_loaded",
                           "No model loaded");
        return;
    }

    yyjson_doc *doc = yyjson_read(req->body, req->body_len, 0);
    if (!doc) {
        respond_json_error(resp, 400, "invalid_request_error", NULL,
                           "Invalid JSON in request body");
        return;
    }

    yyjson_val *root = yyjson_doc_get_root(doc);
    completion_request_t creq = {0};
    const char *parse_err = NULL;
    if (completion_request_parse(&creq, root, &parse_err) < 0) {
        respond_json_error(resp, 400, "invalid_request_error", NULL,
                           parse_err ? parse_err : "Invalid completion request");
        yyjson_doc_free(doc);
        return;
    }

    if (!sampling_params_validate(&creq.params.sampling, &parse_err)) {
        respond_json_error(resp, 400, "invalid_request_error", NULL, parse_err);
        completion_request_free(&creq);
        yyjson_doc_free(doc);
        return;
    }

    gen_request_start_params_t p = {
        .ctx = sctx,
        .conn = req->ctx,
        .chat = false,
        .stream = creq.params.stream,
        .model_id = sctx->model_id,
        .params = creq.params,
        .prompt = creq.prompt,
    };

    const char *err = NULL;
    int rc = gen_request_start(&p, &err);
    if (rc != 0) {
        respond_json_error(resp, rc, error_type_for_status(rc), NULL,
                           err ? err : "generation failed");
        completion_request_free(&creq);
        yyjson_doc_free(doc);
        return;
    }

    resp->deferred = true;
    completion_request_free(&creq);
    yyjson_doc_free(doc);
}

/* --- Registration -------------------------------------------------------- */

void handler_register_all(http_router_t *router, serve_ctx_t *ctx) {
    http_router_add(router, "GET", "/v1/models", handle_models, ctx);
    http_router_add(router, "POST", "/v1/embeddings", handle_embeddings, ctx);
    http_router_add(router, "POST", "/v1/chat/completions", handle_chat_completions, ctx);
    http_router_add(router, "POST", "/v1/completions", handle_completions, ctx);
}
