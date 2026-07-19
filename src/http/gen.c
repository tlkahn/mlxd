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

/* Merge caller extra_json with bos_token/eos_token from the tokenizer.
   Caller keys win on conflict so enable_thinking etc. stay authoritative.
   Returns 0 on success with *out set to a heap string (possibly "{}");
   returns -1 on OOM; returns -2 when extra_json is non-empty but invalid
   (malformed JSON or non-object root). Caller frees *out on success. */
static int merge_chat_extra_json(const tokenizer_t *tok,
                                 const char *extra_json,
                                 char **out) {
    *out = NULL;
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    if (!mdoc) return -1;
    yyjson_mut_val *obj = yyjson_mut_obj(mdoc);
    if (!obj) {
        yyjson_mut_doc_free(mdoc);
        return -1;
    }
    yyjson_mut_doc_set_root(mdoc, obj);

    /* Seed with tokenizer specials when available. */
    int32_t bos = tokenizer_bos_id(tok);
    int32_t eos = tokenizer_eos_id(tok);
    if (bos >= 0) {
        const char *s = tokenizer_decode_token(tok, bos);
        if (s) yyjson_mut_obj_add_strcpy(mdoc, obj, "bos_token", s);
    }
    if (eos >= 0) {
        const char *s = tokenizer_decode_token(tok, eos);
        if (s) yyjson_mut_obj_add_strcpy(mdoc, obj, "eos_token", s);
    }

    /* Overlay caller extras (enable_thinking, etc.). */
    if (extra_json && extra_json[0]) {
        yyjson_doc *edoc = yyjson_read(extra_json, strlen(extra_json), 0);
        if (!edoc) {
            yyjson_mut_doc_free(mdoc);
            return -2;
        }
        yyjson_val *eroot = yyjson_doc_get_root(edoc);
        if (!yyjson_is_obj(eroot)) {
            yyjson_doc_free(edoc);
            yyjson_mut_doc_free(mdoc);
            return -2;
        }
        yyjson_val *key, *val;
        yyjson_obj_iter iter = yyjson_obj_iter_with(eroot);
        while ((key = yyjson_obj_iter_next(&iter))) {
            val = yyjson_obj_iter_get_val(key);
            const char *k = yyjson_get_str(key);
            yyjson_mut_val *mv = yyjson_val_mut_copy(mdoc, val);
            yyjson_mut_val *mk = k ? yyjson_mut_strcpy(mdoc, k) : NULL;
            if (mk && mv)
                yyjson_mut_obj_put(obj, mk, mv);
        }
        yyjson_doc_free(edoc);
    }

    *out = yyjson_mut_write(mdoc, 0, NULL);
    yyjson_mut_doc_free(mdoc);
    return *out ? 0 : -1;
}

int gen_build_chat_prompt(const tokenizer_t *tok, const char *chat_template,
                          const char *messages_json, const char *tools_json,
                          const char *extra_json,
                          int32_t **out_ids, const char **err) {
    char *merged = NULL;
    int mrc = merge_chat_extra_json(tok, extra_json, &merged);
    if (mrc == -2) {
        *err = "invalid extra_json";
        return -1;
    }
    if (mrc != 0 || !merged) {
        *err = "out of memory";
        return -1;
    }

    chat_render_params_t p = {
        .tmpl = chat_template,
        .messages_json = messages_json,
        .tools_json = tools_json,
        .extra_json = merged,
        .add_generation_prompt = true,
    };
    *out_ids = NULL;
    chat_diagnostics_t diag = {0};
    char *rendered = chat_render(&p, &diag);
    free(merged);
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
        .logprobs = (token_logprob_t *)p->logprob,
        .logprob_count = p->logprob ? 1 : 0,
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
                              const usage_t *u,
                              const token_logprob_t *logprobs, int logprobs_count) {
    usage_t zero = {0};
    chat_completion_response_t resp = {
        .id = (char *)id,
        .model = (char *)model,
        .created = created,
        .finish_reason = reason,
        .content = (char *)content,
        .logprobs_content = (token_logprob_t *)logprobs,
        .logprobs_count = logprobs_count,
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

char *gen_sse_error_ex(const char *msg, const char *type, const char *code) {
    yyjson_mut_doc *mdoc = yyjson_mut_doc_new(NULL);
    if (!mdoc) return NULL;
    yyjson_mut_val *root = error_envelope_serialize(
        msg, type ? type : "server_error", code, mdoc);
    char *json = serialize_mut_root(mdoc, root);
    if (!json) return NULL;
    char *sse = sse_format(json, strlen(json));
    free(json);
    return sse;
}

char *gen_sse_error(const char *msg) {
    return gen_sse_error_ex(msg, "server_error", NULL);
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
