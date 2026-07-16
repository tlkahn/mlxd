#include "http/gen.h"
#include "chat/chat.h"
#include "core/openai.h"
#include "http/sse.h"
#include "model/tokenizer.h"

#include <stdint.h>
#include <stdio.h>
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

    int n = tokenizer_encode_alloc(tok, rendered, strlen(rendered), true, out_ids);
    free(rendered);
    if (n < 0) {
        *err = "tokenizer encode failed";
        return -2;
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

char *gen_sse_error(const char *msg) {
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    if (!mdoc) return NULL;
    yyjson_mut_val *root = error_envelope_serialize(msg, "server_error", NULL, mdoc);
    char *json = serialize_mut_root(mdoc, root);
    if (!json) return NULL;
    char *sse = sse_format(json, strlen(json));
    free(json);
    return sse;
}

/* macOS/BSD libc; also glibc >= 2.36. Link-time dependency only. */
extern void arc4random_buf(void *buf, size_t nbytes);

char *gen_make_id(const char *prefix) {
    size_t plen = prefix ? strlen(prefix) : 0;
    char *id = malloc(plen + 24 + 1);
    if (!id) return NULL;
    if (plen > 0) memcpy(id, prefix, plen);
    uint32_t r[3];
    arc4random_buf(r, sizeof(r));
    snprintf(id + plen, 25, "%08x%08x%08x", r[0], r[1], r[2]);
    return id;
}

char *gen_sse_completion_chunk(const gen_sse_completion_chunk_params_t *p) {
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    if (!mdoc) return NULL;

    yyjson_mut_val *root = yyjson_mut_obj(mdoc);
    yyjson_mut_obj_add_strcpy(mdoc, root, "id", p->id);
    yyjson_mut_obj_add_strcpy(mdoc, root, "object", "text_completion");
    yyjson_mut_obj_add_int(mdoc, root, "created", p->created);
    yyjson_mut_obj_add_strcpy(mdoc, root, "model", p->model);

    yyjson_mut_val *choices = yyjson_mut_arr(mdoc);
    yyjson_mut_obj_add_val(mdoc, root, "choices", choices);

    yyjson_mut_val *c0 = yyjson_mut_obj(mdoc);
    yyjson_mut_obj_add_strcpy(mdoc, c0, "text", p->delta_text ? p->delta_text : "");
    yyjson_mut_obj_add_int(mdoc, c0, "index", 0);
    yyjson_mut_obj_add_null(mdoc, c0, "logprobs");
    if (p->final)
        yyjson_mut_obj_add_strcpy(mdoc, c0, "finish_reason",
                                  finish_reason_wire_str(p->reason));
    else
        yyjson_mut_obj_add_null(mdoc, c0, "finish_reason");
    yyjson_mut_arr_add_val(choices, c0);

    char *json = serialize_mut_root(mdoc, root);
    if (!json) return NULL;

    char *sse = sse_format(json, strlen(json));
    free(json);
    return sse;
}
