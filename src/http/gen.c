#include "http/gen.h"
#include "chat/chat.h"
#include "core/openai.h"
#include "http/sse.h"
#include "model/tokenizer.h"

#include <stdlib.h>
#include <string.h>

#include <yyjson/yyjson.h>

static char *serialize_mut_root(yyjson_mut_doc *mdoc, yyjson_mut_val *root) {
    yyjson_mut_doc_set_root(mdoc, root);
    char *json = yyjson_mut_write(mdoc, 0, NULL);
    yyjson_mut_doc_free(mdoc);
    return json;
}

int gen_build_chat_prompt(const tokenizer_t *tok, const char *chat_template,
                          const char *messages_json, const char *tools_json,
                          int32_t **out_ids, const char **err) {
    chat_render_params_t p = {
        .tmpl = chat_template,
        .messages_json = messages_json,
        .tools_json = tools_json,
        .add_generation_prompt = true,
    };
    *out_ids = NULL;
    chat_diagnostics_t diag = {0};
    char *rendered = chat_render(&p, &diag);
    if (!rendered) {
        *err = "template render failed";
        chat_diagnostics_free(&diag);
        return -1;
    }
    chat_diagnostics_free(&diag);

    int n = tokenizer_encode_alloc(tok, rendered, strlen(rendered), false, out_ids);
    free(rendered);
    if (n < 0) {
        *err = "tokenizer encode failed";
        return -1;
    }
    return n;
}

int gen_build_completion_prompt(const tokenizer_t *tok, const char *prompt,
                                int32_t **out_ids, const char **err) {
    int n = tokenizer_encode_alloc(tok, prompt, strlen(prompt), false, out_ids);
    if (n < 0) {
        *err = "tokenizer encode failed";
        return -1;
    }
    return n;
}

char *gen_sse_chunk(const gen_sse_chunk_params_t *p) {
    bool has_choice = p->role_first || p->delta_text || p->final;

    chat_completion_chunk_t chunk = {
        .id = (char *)p->id,
        .model = (char *)p->model,
        .created = p->created,
        .has_choice = has_choice,
        .has_role = p->role_first,
        .role = ROLE_ASSISTANT,
        .delta_content = (char *)p->delta_text,
        .has_finish_reason = p->final,
        .finish_reason = p->reason,
        .has_usage = p->include_usage && p->usage,
    };
    if (chunk.has_usage)
        chunk.usage = *p->usage;

    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    if (!mdoc) return NULL;

    yyjson_mut_val *root = chat_completion_chunk_serialize(&chunk, mdoc);
    char *json = serialize_mut_root(mdoc, root);
    if (!json) return NULL;

    char *sse = sse_format(json, strlen(json));
    free(json);
    return sse;
}

char *gen_build_chat_response(const char *id, const char *model, int64_t created,
                              const char *content, finish_reason_t reason,
                              const usage_t *u) {
    usage_t zero = {0};
    chat_completion_response_t resp = {
        .id = (char *)id,
        .model = (char *)model,
        .created = created,
        .finish_reason = reason,
        .content = (char *)content,
        .usage = u ? *u : zero,
    };

    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    if (!mdoc) return NULL;

    yyjson_mut_val *root = chat_completion_response_serialize(&resp, mdoc);
    return serialize_mut_root(mdoc, root);
}

char *gen_build_completion_response(const char *id, const char *model, int64_t created,
                                    const char *text, finish_reason_t reason,
                                    const usage_t *u) {
    usage_t zero = {0};
    completion_response_t resp = {
        .id = (char *)id,
        .model = (char *)model,
        .created = created,
        .finish_reason = reason,
        .text = (char *)text,
        .usage = u ? *u : zero,
    };

    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    if (!mdoc) return NULL;

    yyjson_mut_val *root = completion_response_serialize(&resp, mdoc);
    return serialize_mut_root(mdoc, root);
}
