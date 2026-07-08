#include "core/openai.h"

#include <stdlib.h>
#include <string.h>

/* --- Shared parse helpers -------------------------------------------------
 *
 * Parse side mirrors the Zig reference's alloc_always: every string is duped
 * out of the yyjson doc so the caller may free the doc immediately after parse
 * and the DTO still owns valid memory. */

static char *dup_str(const char *s) {
    if (!s)
        return NULL;
    size_t n = strlen(s);
    char *p = malloc(n + 1);
    if (p)
        memcpy(p, s, n + 1);
    return p;
}

/* strdup a JSON string value, or NULL if v is not a string. */
static char *json_str_dup(yyjson_val *v) {
    const char *s = yyjson_get_str(v);
    return s ? dup_str(s) : NULL;
}

/* Parse an OpenAI "stop" value: a string becomes a 1-element array; an array of
 * strings becomes n elements; absent -> NULL/0. Returns 0, or -1 on OOM. */
static int parse_stop(yyjson_val *v, char ***out, int *count, const char **err) {
    *out = NULL;
    *count = 0;
    if (!v)
        return 0;
    if (yyjson_is_str(v)) {
        char **arr = malloc(sizeof(char *));
        if (!arr) {
            *err = "out of memory";
            return -1;
        }
        arr[0] = dup_str(yyjson_get_str(v));
        if (!arr[0]) {
            free(arr);
            *err = "out of memory";
            return -1;
        }
        *out = arr;
        *count = 1;
        return 0;
    }
    if (yyjson_is_arr(v)) {
        size_t n = yyjson_arr_size(v);
        if (n == 0)
            return 0;
        char **arr = calloc(n, sizeof(char *));
        if (!arr) {
            *err = "out of memory";
            return -1;
        }
        size_t idx, max;
        yyjson_val *item;
        yyjson_arr_foreach(v, idx, max, item) {
            arr[idx] = json_str_dup(item);
            if (!arr[idx]) {
                for (size_t i = 0; i < idx; i++)
                    free(arr[i]);
                free(arr);
                *err = "stop values must be strings";
                return -1;
            }
        }
        *out = arr;
        *count = (int)n;
    }
    return 0;
}

static void free_str_array(char **arr, int count) {
    if (!arr)
        return;
    for (int i = 0; i < count; i++)
        free(arr[i]);
    free(arr);
}

/* --- Shared serialize helpers ---------------------------------------------
 *
 * Serialize side uses copying yyjson APIs (strcpy/strncpy/rawcpy) so the DTO's
 * lifetime is independent of the mut_doc, and the copy path is where yyjson
 * performs all JSON string escaping. No manual escape helpers anywhere. */

static void add_usage(yyjson_mut_doc *doc, yyjson_mut_val *parent, const usage_t *u) {
    yyjson_mut_val *usage = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_int(doc, usage, "prompt_tokens", u->prompt_tokens);
    yyjson_mut_obj_add_int(doc, usage, "completion_tokens", u->completion_tokens);
    yyjson_mut_obj_add_int(doc, usage, "total_tokens", u->total_tokens);
    yyjson_mut_obj_add_val(doc, parent, "usage", usage);
}

static int role_from_str(const char *s, role_t *out) {
    if (!s)
        return -1;
    if (strcmp(s, "system") == 0)
        *out = ROLE_SYSTEM;
    else if (strcmp(s, "user") == 0)
        *out = ROLE_USER;
    else if (strcmp(s, "assistant") == 0)
        *out = ROLE_ASSISTANT;
    else if (strcmp(s, "tool") == 0)
        *out = ROLE_TOOL;
    else
        return -1;
    return 0;
}

/* Parse a message "content" value. Absent or null -> CONTENT_NONE; a string ->
 * CONTENT_STRING; an array of parts -> CONTENT_PARTS (cycle 6). Returns 0, or
 * -1 on a malformed value. */
static int parse_content(yyjson_val *v, message_content_t *out) {
    out->kind = CONTENT_NONE;
    out->string = NULL;
    out->parts = NULL;
    out->part_count = 0;
    if (!v || yyjson_is_null(v))
        return 0;
    if (yyjson_is_str(v)) {
        out->kind = CONTENT_STRING;
        out->string = json_str_dup(v);
        return 0;
    }
    if (yyjson_is_arr(v)) {
        out->kind = CONTENT_PARTS;
        size_t n = yyjson_arr_size(v);
        if (n == 0)
            return 0;
        out->parts = calloc(n, sizeof(content_part_t));
        if (!out->parts)
            return -1;
        out->part_count = (int)n;
        size_t idx, max;
        yyjson_val *item;
        yyjson_arr_foreach(v, idx, max, item) {
            content_part_t *p = &out->parts[idx];
            p->type = json_str_dup(yyjson_obj_get(item, "type"));
            p->text = json_str_dup(yyjson_obj_get(item, "text"));
            yyjson_val *iu = yyjson_obj_get(item, "image_url");
            if (yyjson_is_obj(iu)) {
                p->has_image_url = true;
                p->image_url.url = json_str_dup(yyjson_obj_get(iu, "url"));
            }
        }
        return 0;
    }
    return -1;
}

/* Serialize a yyjson subtree to compact JSON text. yyjson allocates the buffer
 * with the default (malloc) allocator, so the caller frees it with free().
 * Used to carry tool `parameters` as opaque raw JSON. */
static char *json_val_dup(yyjson_val *v) {
    if (!v)
        return NULL;
    return yyjson_val_write(v, 0, NULL);
}

static int parse_tools(chat_completion_request_t *req, yyjson_val *tools, const char **err) {
    if (!tools)
        return 0;
    if (!yyjson_is_arr(tools)) {
        *err = "tools must be an array";
        return -1;
    }
    size_t n = yyjson_arr_size(tools);
    if (n == 0)
        return 0;
    req->tools = calloc(n, sizeof(tool_t));
    if (!req->tools) {
        *err = "out of memory";
        return -1;
    }
    req->tool_count = (int)n;
    size_t idx, max;
    yyjson_val *t;
    yyjson_arr_foreach(tools, idx, max, t) {
        tool_t *tool = &req->tools[idx];
        tool->type = json_str_dup(yyjson_obj_get(t, "type"));
        yyjson_val *fn = yyjson_obj_get(t, "function");
        if (yyjson_is_obj(fn)) {
            tool->function.name = json_str_dup(yyjson_obj_get(fn, "name"));
            tool->function.description = json_str_dup(yyjson_obj_get(fn, "description"));
            tool->function.parameters_json = json_val_dup(yyjson_obj_get(fn, "parameters"));
        }
    }
    return 0;
}

/* Parse tool_choice. Object form -> TOOL_CHOICE_FUNCTION with the named
 * function; string form (auto/none/required) lands in cycle 7. */
static int parse_tool_choice(tool_choice_t *tc, yyjson_val *v, const char **err) {
    tc->kind = TOOL_CHOICE_UNSET;
    tc->function_name = NULL;
    if (!v)
        return 0;
    if (yyjson_is_str(v)) {
        const char *s = yyjson_get_str(v);
        if (strcmp(s, "auto") == 0)
            tc->kind = TOOL_CHOICE_AUTO;
        else if (strcmp(s, "none") == 0)
            tc->kind = TOOL_CHOICE_NONE;
        else if (strcmp(s, "required") == 0)
            tc->kind = TOOL_CHOICE_REQUIRED;
        else {
            *err = "invalid tool_choice";
            return -1;
        }
        return 0;
    }
    if (yyjson_is_obj(v)) {
        yyjson_val *fn = yyjson_obj_get(v, "function");
        if (!yyjson_is_obj(fn)) {
            *err = "tool_choice object missing function";
            return -1;
        }
        tc->function_name = json_str_dup(yyjson_obj_get(fn, "name"));
        if (!tc->function_name) {
            *err = "tool_choice function missing name";
            return -1;
        }
        tc->kind = TOOL_CHOICE_FUNCTION;
    }
    return 0;
}

static int parse_messages(chat_completion_request_t *req, yyjson_val *msgs, const char **err) {
    if (!yyjson_is_arr(msgs)) {
        *err = "missing required field: messages";
        return -1;
    }
    size_t n = yyjson_arr_size(msgs);
    req->messages = calloc(n ? n : 1, sizeof(message_t));
    if (!req->messages) {
        *err = "out of memory";
        return -1;
    }
    req->message_count = (int)n;
    size_t idx, max;
    yyjson_val *m;
    yyjson_arr_foreach(msgs, idx, max, m) {
        message_t *msg = &req->messages[idx];
        if (role_from_str(yyjson_get_str(yyjson_obj_get(m, "role")), &msg->role) != 0) {
            *err = "invalid or missing message role";
            return -1;
        }
        if (parse_content(yyjson_obj_get(m, "content"), &msg->content) != 0) {
            *err = "invalid message content";
            return -1;
        }
        msg->name = json_str_dup(yyjson_obj_get(m, "name"));
        msg->tool_call_id = json_str_dup(yyjson_obj_get(m, "tool_call_id"));
        yyjson_val *tcs = yyjson_obj_get(m, "tool_calls");
        if (yyjson_is_arr(tcs)) {
            size_t tc_n = yyjson_arr_size(tcs);
            if (tc_n > 0) {
                msg->tool_calls = calloc(tc_n, sizeof(tool_call_t));
                if (!msg->tool_calls) {
                    *err = "out of memory";
                    return -1;
                }
                msg->tool_call_count = (int)tc_n;
                size_t tc_idx, tc_max;
                yyjson_val *tc;
                yyjson_arr_foreach(tcs, tc_idx, tc_max, tc) {
                    tool_call_t *call = &msg->tool_calls[tc_idx];
                    call->id = json_str_dup(yyjson_obj_get(tc, "id"));
                    yyjson_val *fn = yyjson_obj_get(tc, "function");
                    if (yyjson_is_obj(fn)) {
                        call->function_name = json_str_dup(yyjson_obj_get(fn, "name"));
                        call->arguments = json_str_dup(yyjson_obj_get(fn, "arguments"));
                    }
                }
            }
        }
    }
    return 0;
}

/* Overlay OpenAI sampling knobs (temperature, top_p, seed) present on `root`
 * onto an already-defaulted gen_params. top_k/min_p are not on the wire. */
static void parse_sampling(gen_params_t *params, yyjson_val *root) {
    yyjson_val *v;
    if ((v = yyjson_obj_get(root, "temperature")) && yyjson_is_num(v))
        params->sampling.temperature = (float)yyjson_get_num(v);
    if ((v = yyjson_obj_get(root, "top_p")) && yyjson_is_num(v))
        params->sampling.top_p = (float)yyjson_get_num(v);
    if ((v = yyjson_obj_get(root, "seed")) && yyjson_is_int(v))
        params->sampling.seed = (int)yyjson_get_sint(v);
}

int chat_completion_request_parse(chat_completion_request_t *req, yyjson_val *root,
                                  const char **err) {
    memset(req, 0, sizeof(*req));
    if (!yyjson_is_obj(root)) {
        *err = "request body must be a JSON object";
        return -1;
    }
    req->params.sampling = SAMPLING_PARAMS_DEFAULT;
    req->params.n = 1;
    req->tool_choice.kind = TOOL_CHOICE_UNSET;

    yyjson_val *model = yyjson_obj_get(root, "model");
    if (!yyjson_is_str(model)) {
        *err = "missing required field: model";
        return -1;
    }
    req->model = json_str_dup(model);

    if (parse_messages(req, yyjson_obj_get(root, "messages"), err) != 0)
        goto fail;

    parse_sampling(&req->params, root);
    /* max_completion_tokens wins over the legacy max_tokens when both present. */
    yyjson_val *mct = yyjson_obj_get(root, "max_completion_tokens");
    yyjson_val *mt = yyjson_obj_get(root, "max_tokens");
    if (yyjson_is_int(mct))
        req->params.max_tokens = (int)yyjson_get_sint(mct);
    else if (yyjson_is_int(mt))
        req->params.max_tokens = (int)yyjson_get_sint(mt);

    yyjson_val *stream = yyjson_obj_get(root, "stream");
    if (yyjson_is_bool(stream))
        req->params.stream = yyjson_get_bool(stream);
    yyjson_val *lp = yyjson_obj_get(root, "logprobs");
    if (yyjson_is_bool(lp))
        req->params.logprobs = yyjson_get_bool(lp);
    yyjson_val *tlp = yyjson_obj_get(root, "top_logprobs");
    if (yyjson_is_int(tlp))
        req->params.top_logprobs = (int)yyjson_get_sint(tlp);

    yyjson_val *so = yyjson_obj_get(root, "stream_options");
    if (yyjson_is_obj(so)) {
        yyjson_val *iu = yyjson_obj_get(so, "include_usage");
        if (yyjson_is_bool(iu))
            req->include_usage = yyjson_get_bool(iu);
    }

    if (parse_stop(yyjson_obj_get(root, "stop"), &req->params.stop, &req->params.stop_count, err) != 0)
        goto fail;
    if (parse_tools(req, yyjson_obj_get(root, "tools"), err) != 0)
        goto fail;
    if (parse_tool_choice(&req->tool_choice, yyjson_obj_get(root, "tool_choice"), err) != 0)
        goto fail;
    return 0;

fail:
    chat_completion_request_free(req);
    return -1;
}

void chat_completion_request_free(chat_completion_request_t *req) {
    if (!req)
        return;
    free(req->model);
    for (int i = 0; i < req->message_count; i++)
        message_free(&req->messages[i]);
    free(req->messages);
    for (int i = 0; i < req->tool_count; i++)
        tool_free(&req->tools[i]);
    free(req->tools);
    tool_choice_free(&req->tool_choice);
    free_str_array(req->params.stop, req->params.stop_count);
}

yyjson_mut_val *chat_completion_response_serialize(const chat_completion_response_t *resp,
                                                   yyjson_mut_doc *doc) {
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_strcpy(doc, root, "id", resp->id);
    yyjson_mut_obj_add_strcpy(doc, root, "object", "chat.completion");
    yyjson_mut_obj_add_int(doc, root, "created", resp->created);
    yyjson_mut_obj_add_strcpy(doc, root, "model", resp->model);

    yyjson_mut_val *choices = yyjson_mut_arr(doc);
    yyjson_mut_obj_add_val(doc, root, "choices", choices);
    yyjson_mut_val *c0 = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_int(doc, c0, "index", 0);

    yyjson_mut_val *msg = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_strcpy(doc, msg, "role", "assistant");
    if (resp->content)
        yyjson_mut_obj_add_strcpy(doc, msg, "content", resp->content);
    else
        yyjson_mut_obj_add_null(doc, msg, "content");
    if (resp->tool_call_count > 0) {
        yyjson_mut_val *tcs = yyjson_mut_arr(doc);
        for (int i = 0; i < resp->tool_call_count; i++) {
            const tool_call_t *tc = &resp->tool_calls[i];
            yyjson_mut_val *o = yyjson_mut_obj(doc);
            yyjson_mut_obj_add_strcpy(doc, o, "id", tc->id);
            yyjson_mut_obj_add_strcpy(doc, o, "type", "function");
            yyjson_mut_val *fn = yyjson_mut_obj(doc);
            yyjson_mut_obj_add_strcpy(doc, fn, "name", tc->function_name);
            /* arguments is opaque JSON text; strcpy escapes it, so the wire
             * carries the doubly-encoded string OpenAI clients expect. */
            yyjson_mut_obj_add_strcpy(doc, fn, "arguments", tc->arguments ? tc->arguments : "");
            yyjson_mut_obj_add_val(doc, o, "function", fn);
            yyjson_mut_arr_add_val(tcs, o);
        }
        yyjson_mut_obj_add_val(doc, msg, "tool_calls", tcs);
    }
    yyjson_mut_obj_add_val(doc, c0, "message", msg);

    if (resp->logprobs_content) {
        yyjson_mut_val *lp = yyjson_mut_obj(doc);
        yyjson_mut_val *cont = yyjson_mut_arr(doc);
        for (int i = 0; i < resp->logprobs_count; i++) {
            const token_logprob_t *tl = &resp->logprobs_content[i];
            yyjson_mut_val *o = yyjson_mut_obj(doc);
            yyjson_mut_obj_add_strcpy(doc, o, "token", tl->token);
            yyjson_mut_obj_add_real(doc, o, "logprob", tl->logprob);
            yyjson_mut_val *tops = yyjson_mut_arr(doc);
            for (int j = 0; j < tl->top_logprob_count; j++) {
                yyjson_mut_val *to = yyjson_mut_obj(doc);
                yyjson_mut_obj_add_strcpy(doc, to, "token", tl->top_logprobs[j].token);
                yyjson_mut_obj_add_real(doc, to, "logprob", tl->top_logprobs[j].logprob);
                yyjson_mut_arr_add_val(tops, to);
            }
            yyjson_mut_obj_add_val(doc, o, "top_logprobs", tops);
            yyjson_mut_arr_add_val(cont, o);
        }
        yyjson_mut_obj_add_val(doc, lp, "content", cont);
        yyjson_mut_obj_add_val(doc, c0, "logprobs", lp);
    }

    /* Wire mapping only: the internal "cancelled" tag must never reach clients. */
    yyjson_mut_obj_add_strcpy(doc, c0, "finish_reason", finish_reason_wire_str(resp->finish_reason));
    yyjson_mut_arr_add_val(choices, c0);

    add_usage(doc, root, &resp->usage);
    return root;
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
    for (int i = 0; i < resp->logprobs_count; i++) {
        token_logprob_t *tl = &resp->logprobs_content[i];
        free(tl->token);
        for (int j = 0; j < tl->top_logprob_count; j++)
            free(tl->top_logprobs[j].token);
        free(tl->top_logprobs);
    }
    free(resp->logprobs_content);
}

yyjson_mut_val *chat_completion_chunk_serialize(const chat_completion_chunk_t *chunk,
                                                yyjson_mut_doc *doc) {
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_strcpy(doc, root, "id", chunk->id);
    yyjson_mut_obj_add_strcpy(doc, root, "object", "chat.completion.chunk");
    yyjson_mut_obj_add_int(doc, root, "created", chunk->created);
    yyjson_mut_obj_add_strcpy(doc, root, "model", chunk->model);

    yyjson_mut_val *choices = yyjson_mut_arr(doc);
    yyjson_mut_obj_add_val(doc, root, "choices", choices);
    if (chunk->has_choice) {
        yyjson_mut_val *c0 = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_int(doc, c0, "index", 0);

        yyjson_mut_val *delta = yyjson_mut_obj(doc);
        if (chunk->has_role)
            yyjson_mut_obj_add_strcpy(doc, delta, "role", role_str(chunk->role));
        if (chunk->delta_content)
            yyjson_mut_obj_add_strcpy(doc, delta, "content", chunk->delta_content);
        if (chunk->tool_call_count > 0) {
            yyjson_mut_val *tcs = yyjson_mut_arr(doc);
            for (int i = 0; i < chunk->tool_call_count; i++) {
                const delta_tool_call_t *tc = &chunk->tool_calls[i];
                yyjson_mut_val *o = yyjson_mut_obj(doc);
                yyjson_mut_obj_add_int(doc, o, "index", tc->index);
                if (tc->id)
                    yyjson_mut_obj_add_strcpy(doc, o, "id", tc->id);
                if (tc->type)
                    yyjson_mut_obj_add_strcpy(doc, o, "type", tc->type);
                if (tc->function_name || tc->arguments) {
                    yyjson_mut_val *fn = yyjson_mut_obj(doc);
                    if (tc->function_name)
                        yyjson_mut_obj_add_strcpy(doc, fn, "name", tc->function_name);
                    if (tc->arguments)
                        yyjson_mut_obj_add_strcpy(doc, fn, "arguments", tc->arguments);
                    yyjson_mut_obj_add_val(doc, o, "function", fn);
                }
                yyjson_mut_arr_add_val(tcs, o);
            }
            yyjson_mut_obj_add_val(doc, delta, "tool_calls", tcs);
        }
        yyjson_mut_obj_add_val(doc, c0, "delta", delta);

        if (chunk->has_finish_reason)
            yyjson_mut_obj_add_strcpy(doc, c0, "finish_reason",
                                      finish_reason_wire_str(chunk->finish_reason));
        else
            yyjson_mut_obj_add_null(doc, c0, "finish_reason");
        yyjson_mut_arr_add_val(choices, c0);
    }

    if (chunk->has_usage)
        add_usage(doc, root, &chunk->usage);
    return root;
}

void chat_completion_chunk_free(chat_completion_chunk_t *chunk) {
    if (!chunk)
        return;
    free(chunk->id);
    free(chunk->model);
    free(chunk->delta_content);
    for (int i = 0; i < chunk->tool_call_count; i++) {
        delta_tool_call_t *tc = &chunk->tool_calls[i];
        free(tc->id);
        free(tc->type);
        free(tc->function_name);
        free(tc->arguments);
    }
    free(chunk->tool_calls);
}

int completion_request_parse(completion_request_t *req, yyjson_val *root, const char **err) {
    memset(req, 0, sizeof(*req));
    if (!yyjson_is_obj(root)) {
        *err = "request body must be a JSON object";
        return -1;
    }
    req->params.sampling = SAMPLING_PARAMS_DEFAULT;
    req->params.n = 1;

    yyjson_val *model = yyjson_obj_get(root, "model");
    if (!yyjson_is_str(model)) {
        *err = "missing required field: model";
        return -1;
    }
    req->model = json_str_dup(model);

    /* prompt is string-only here; the OpenAI string-array and token-ID-array
     * forms are deferred (v2), so an array prompt is a clear error. */
    yyjson_val *prompt = yyjson_obj_get(root, "prompt");
    if (yyjson_is_arr(prompt)) {
        *err = "completion prompt array form not supported";
        goto fail;
    }
    if (!yyjson_is_str(prompt)) {
        *err = "missing required field: prompt";
        goto fail;
    }
    req->prompt = json_str_dup(prompt);

    parse_sampling(&req->params, root);
    yyjson_val *mt = yyjson_obj_get(root, "max_tokens");
    if (yyjson_is_int(mt))
        req->params.max_tokens = (int)yyjson_get_sint(mt);
    yyjson_val *stream = yyjson_obj_get(root, "stream");
    if (yyjson_is_bool(stream))
        req->params.stream = yyjson_get_bool(stream);
    if (parse_stop(yyjson_obj_get(root, "stop"), &req->params.stop, &req->params.stop_count, err) != 0)
        goto fail;
    return 0;

fail:
    completion_request_free(req);
    return -1;
}

void completion_request_free(completion_request_t *req) {
    if (!req)
        return;
    free(req->model);
    free(req->prompt);
    free_str_array(req->params.stop, req->params.stop_count);
}

yyjson_mut_val *completion_response_serialize(const completion_response_t *resp,
                                              yyjson_mut_doc *doc) {
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_strcpy(doc, root, "id", resp->id);
    yyjson_mut_obj_add_strcpy(doc, root, "object", "text_completion");
    yyjson_mut_obj_add_int(doc, root, "created", resp->created);
    yyjson_mut_obj_add_strcpy(doc, root, "model", resp->model);

    yyjson_mut_val *choices = yyjson_mut_arr(doc);
    yyjson_mut_obj_add_val(doc, root, "choices", choices);
    yyjson_mut_val *c0 = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_strcpy(doc, c0, "text", resp->text);
    yyjson_mut_obj_add_int(doc, c0, "index", 0);
    /* Wire mapping only: the internal "cancelled" tag must never reach clients. */
    yyjson_mut_obj_add_strcpy(doc, c0, "finish_reason", finish_reason_wire_str(resp->finish_reason));
    yyjson_mut_arr_add_val(choices, c0);

    add_usage(doc, root, &resp->usage);
    return root;
}

void completion_response_free(completion_response_t *resp) {
    if (!resp)
        return;
    free(resp->id);
    free(resp->model);
    free(resp->text);
}

int embedding_request_parse(embedding_request_t *req, yyjson_val *root, const char **err) {
    memset(req, 0, sizeof(*req));
    if (!yyjson_is_obj(root)) {
        *err = "request body must be a JSON object";
        return -1;
    }

    yyjson_val *model = yyjson_obj_get(root, "model");
    if (!yyjson_is_str(model)) {
        *err = "missing required field: model";
        return -1;
    }
    req->model = json_str_dup(model);

    /* input is string-only here; the string-array and token-ID-array forms are
     * deferred (v2). */
    yyjson_val *input = yyjson_obj_get(root, "input");
    if (yyjson_is_arr(input)) {
        *err = "embedding input array form not supported";
        goto fail;
    }
    if (!yyjson_is_str(input)) {
        *err = "missing required field: input";
        goto fail;
    }
    req->input = json_str_dup(input);
    if (!req->input) {
        *err = "out of memory";
        goto fail;
    }

    /* encoding_format defaults to "float" when absent. dimensions and user are
     * accepted on the wire but ignored here (deferred to the engine layer). */
    yyjson_val *ef = yyjson_obj_get(root, "encoding_format");
    req->encoding_format = yyjson_is_str(ef) ? json_str_dup(ef) : dup_str("float");
    if (!req->encoding_format) {
        *err = "out of memory";
        goto fail;
    }
    return 0;

fail:
    embedding_request_free(req);
    return -1;
}

void embedding_request_free(embedding_request_t *req) {
    if (!req)
        return;
    free(req->model);
    free(req->input);
    free(req->encoding_format);
}

/* Base64-encode raw bytes into `out` (which must hold ((n+2)/3)*4 + 1 bytes).
 * This is a binary data encoding, not JSON string escaping, so hand-rolling it
 * does not breach the no-manual-escaping invariant. */
static void base64_encode(const uint8_t *in, size_t n, char *out) {
    static const char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t o = 0, i = 0;
    for (; i + 3 <= n; i += 3) {
        uint32_t v = ((uint32_t)in[i] << 16) | ((uint32_t)in[i + 1] << 8) | in[i + 2];
        out[o++] = alphabet[(v >> 18) & 0x3F];
        out[o++] = alphabet[(v >> 12) & 0x3F];
        out[o++] = alphabet[(v >> 6) & 0x3F];
        out[o++] = alphabet[v & 0x3F];
    }
    size_t rem = n - i;
    if (rem == 1) {
        uint32_t v = (uint32_t)in[i] << 16;
        out[o++] = alphabet[(v >> 18) & 0x3F];
        out[o++] = alphabet[(v >> 12) & 0x3F];
        out[o++] = '=';
        out[o++] = '=';
    } else if (rem == 2) {
        uint32_t v = ((uint32_t)in[i] << 16) | ((uint32_t)in[i + 1] << 8);
        out[o++] = alphabet[(v >> 18) & 0x3F];
        out[o++] = alphabet[(v >> 12) & 0x3F];
        out[o++] = alphabet[(v >> 6) & 0x3F];
        out[o++] = '=';
    }
    out[o] = '\0';
}

/* Add an embedding vector as a base64 string of little-endian f32 bytes. The
 * LE byte order is written explicitly (via each float's bit pattern) so it is
 * correct regardless of host endianness. */
static void add_embedding_base64(yyjson_mut_doc *doc, yyjson_mut_val *obj,
                                 const embedding_data_t *d) {
    size_t nbytes = (size_t)d->value_count * 4;
    uint8_t *bytes = malloc(nbytes ? nbytes : 1);
    for (int j = 0; j < d->value_count; j++) {
        uint32_t bits;
        memcpy(&bits, &d->values[j], 4);
        bytes[j * 4 + 0] = (uint8_t)(bits & 0xFF);
        bytes[j * 4 + 1] = (uint8_t)((bits >> 8) & 0xFF);
        bytes[j * 4 + 2] = (uint8_t)((bits >> 16) & 0xFF);
        bytes[j * 4 + 3] = (uint8_t)((bits >> 24) & 0xFF);
    }
    char *b64 = malloc(((nbytes + 2) / 3) * 4 + 1);
    base64_encode(bytes, nbytes, b64);
    yyjson_mut_obj_add_strcpy(doc, obj, "embedding", b64);
    free(bytes);
    free(b64);
}

yyjson_mut_val *embedding_response_serialize(const embedding_response_t *resp,
                                             yyjson_mut_doc *doc) {
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_strcpy(doc, root, "object", "list");

    bool b64 = resp->encoding_format && strcmp(resp->encoding_format, "base64") == 0;
    yyjson_mut_val *data = yyjson_mut_arr(doc);
    yyjson_mut_obj_add_val(doc, root, "data", data);
    for (int i = 0; i < resp->data_count; i++) {
        const embedding_data_t *d = &resp->data[i];
        yyjson_mut_val *o = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_strcpy(doc, o, "object", "embedding");
        yyjson_mut_obj_add_int(doc, o, "index", d->index);
        if (b64) {
            add_embedding_base64(doc, o, d);
        } else {
            yyjson_mut_val *emb = yyjson_mut_arr(doc);
            for (int j = 0; j < d->value_count; j++)
                yyjson_mut_arr_add_real(doc, emb, d->values[j]);
            yyjson_mut_obj_add_val(doc, o, "embedding", emb);
        }
        yyjson_mut_arr_add_val(data, o);
    }

    yyjson_mut_obj_add_strcpy(doc, root, "model", resp->model);
    add_usage(doc, root, &resp->usage);
    return root;
}

void embedding_response_free(embedding_response_t *resp) {
    if (!resp)
        return;
    free(resp->model);
    for (int i = 0; i < resp->data_count; i++)
        free(resp->data[i].values);
    free(resp->data);
    free(resp->encoding_format);
}

yyjson_mut_val *model_list_serialize(const model_info_t *models, int count, yyjson_mut_doc *doc) {
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_strcpy(doc, root, "object", "list");
    yyjson_mut_val *data = yyjson_mut_arr(doc);
    yyjson_mut_obj_add_val(doc, root, "data", data);
    for (int i = 0; i < count; i++) {
        yyjson_mut_val *m = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_strcpy(doc, m, "id", models[i].id);
        yyjson_mut_obj_add_strcpy(doc, m, "object", "model");
        yyjson_mut_obj_add_int(doc, m, "created", models[i].created);
        yyjson_mut_obj_add_strcpy(doc, m, "owned_by", models[i].owned_by);
        yyjson_mut_arr_add_val(data, m);
    }
    return root;
}

yyjson_mut_val *error_envelope_serialize(const char *message, const char *type, const char *code,
                                         yyjson_mut_doc *doc) {
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_val *error = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "error", error);
    if (message)
        yyjson_mut_obj_add_strcpy(doc, error, "message", message);
    if (type)
        yyjson_mut_obj_add_strcpy(doc, error, "type", type);
    if (code)
        yyjson_mut_obj_add_strcpy(doc, error, "code", code);
    return root;
}
