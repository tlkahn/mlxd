#include "http/handler.h"
#include "core/openai.h"
#include "registry/registry.h"

#include <stdlib.h>
#include <string.h>
#include <yyjson/yyjson.h>

/* --- Shared helpers ------------------------------------------------------ */

static void respond_json_error(http_response_t *resp, int status,
                               const char *type, const char *code,
                               const char *msg) {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) { resp->status = 500; return; }
    yyjson_mut_val *root = error_envelope_serialize(msg, type, code, doc);
    yyjson_mut_doc_set_root(doc, root);
    resp->body = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    if (!resp->body) { resp->status = 500; return; }
    resp->body_len = strlen(resp->body);
    resp->status = status;
    resp->content_type = "application/json";
}

/* --- GET /v1/models ------------------------------------------------------ */

static void handle_models(const http_request_t *req, http_response_t *resp,
                          void *ctx) {
    (void)req;
    (void)ctx;

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
    resp->content_type = "application/json";
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

/* --- Registration -------------------------------------------------------- */

void handler_register_all(http_router_t *router, serve_ctx_t *ctx) {
    http_router_add(router, "GET", "/v1/models", handle_models, ctx);
    http_router_add(router, "POST", "/v1/embeddings", handle_embeddings, ctx);
}
