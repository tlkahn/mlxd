#include "chat/chat.h"

#include <jinja_cpp/jinja_wrapper.h>
#include <stdlib.h>
#include <string.h>

char *chat_render(const chat_render_params_t *params, chat_diagnostics_t *diag) {
    if (!params || !params->tmpl || !params->messages_json) {
        if (diag)
            diag->error = strdup("missing required render parameters");
        return NULL;
    }

    if (strlen(params->tmpl) == 0) {
        if (diag)
            diag->error = strdup("empty template");
        return NULL;
    }

    char *result = jinja_render_chat(params->tmpl, params->messages_json,
                                     params->tools_json, params->extra_json,
                                     params->add_generation_prompt ? 1 : 0);
    if (!result) {
        const char *err = jinja_last_error();
        if (diag)
            diag->error = strdup(err ? err : "jinja render failed");
        return NULL;
    }

    return result;
}

void chat_diagnostics_free(chat_diagnostics_t *diag) {
    if (!diag)
        return;
    free(diag->error);
    diag->error = NULL;
}
