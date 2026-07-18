#include "core/types.h"

#include <stdlib.h>
#include <string.h>

/* --- Finish reason -------------------------------------------------------- */

const char *finish_reason_str(finish_reason_t r) {
    switch (r) {
    case FINISH_STOP:           return "stop";
    case FINISH_LENGTH:         return "length";
    case FINISH_TOOL_CALLS:     return "tool_calls";
    case FINISH_CONTENT_FILTER: return "content_filter";
    case FINISH_CANCELLED:      return "cancelled";
    }
    return "unknown";
}

const char *finish_reason_wire_str(finish_reason_t r) {
    if (r == FINISH_CANCELLED)
        return "stop";
    return finish_reason_str(r);
}

/* --- Sampling validation -------------------------------------------------- */

bool sampling_params_validate(const sampling_params_t *p, const char **err) {
    if (p->temperature < 0.0f) {
        *err = "temperature must be >= 0";
        return false;
    }
    if (p->top_p < 0.0f || p->top_p > 1.0f) {
        *err = "top_p must be in [0, 1]";
        return false;
    }
    if (p->min_p < 0.0f || p->min_p > 1.0f) {
        *err = "min_p must be in [0, 1]";
        return false;
    }
    if (p->top_k < -1) {
        *err = "top_k must be >= -1";
        return false;
    }
    return true;
}

/* --- Role ----------------------------------------------------------------- */

const char *role_str(role_t r) {
    switch (r) {
    case ROLE_SYSTEM:    return "system";
    case ROLE_USER:      return "user";
    case ROLE_ASSISTANT: return "assistant";
    case ROLE_TOOL:      return "tool";
    }
    return "unknown";
}

/* --- Cleanup -------------------------------------------------------------- */

void tool_call_free(tool_call_t *tc) {
    if (!tc)
        return;
    free(tc->id);
    free(tc->function_name);
    free(tc->arguments);
}

void content_part_free(content_part_t *part) {
    if (!part)
        return;
    free(part->type);
    free(part->text);
    free(part->image_url.url);
}

void message_content_free(message_content_t *content) {
    if (!content)
        return;
    free(content->string);
    for (int i = 0; i < content->part_count; i++)
        content_part_free(&content->parts[i]);
    free(content->parts);
}

void message_free(message_t *msg) {
    if (!msg)
        return;
    message_content_free(&msg->content);
    free(msg->name);
    free(msg->tool_call_id);
    for (int i = 0; i < msg->tool_call_count; i++)
        tool_call_free(&msg->tool_calls[i]);
    free(msg->tool_calls);
}

void tool_free(tool_t *tool) {
    if (!tool)
        return;
    free(tool->type);
    free(tool->function.name);
    free(tool->function.description);
    free(tool->function.parameters_json);
}

void tool_choice_free(tool_choice_t *choice) {
    if (!choice)
        return;
    free(choice->function_name);
}
