#include "chat/chat.h"

#include <jinja_cpp/jinja_wrapper.h>
#include <stdlib.h>
#include <string.h>

static bool has_nul_byte(const char *s, size_t len) {
    return memchr(s, '\0', len) != NULL;
}

char *chat_render(const chat_render_params_t *params, chat_diagnostics_t *diag) {
    if (!params || !params->tmpl || !params->messages_json) {
        if (diag)
            diag->error = strdup("missing required render parameters");
        return NULL;
    }

    size_t eff_tmpl_len = params->tmpl_len ? params->tmpl_len : strlen(params->tmpl);
    size_t eff_msgs_len = params->messages_json_len ? params->messages_json_len : strlen(params->messages_json);
    size_t eff_tools_len = params->tools_json ? (params->tools_json_len ? params->tools_json_len : strlen(params->tools_json)) : 0;
    size_t eff_extra_len = params->extra_json ? (params->extra_json_len ? params->extra_json_len : strlen(params->extra_json)) : 0;

    const struct { const char *s; size_t len; } nul_fields[] = {
        { params->tmpl, eff_tmpl_len },
        { params->messages_json, eff_msgs_len },
        { params->tools_json, eff_tools_len },
        { params->extra_json, eff_extra_len },
    };
    for (size_t i = 0; i < sizeof(nul_fields) / sizeof(nul_fields[0]); i++) {
        if (nul_fields[i].s && nul_fields[i].len > 0 && has_nul_byte(nul_fields[i].s, nul_fields[i].len)) {
            // strdup may return NULL on OOM; callers should treat NULL return as the primary error signal
            if (diag)
                diag->error = strdup("input contains interior NUL byte");
            return NULL;
        }
    }

    if (eff_tmpl_len == 0) {
        if (diag)
            diag->error = strdup("empty template");
        return NULL;
    }

    // TODO(F5): pass lengths to jinja_render_chat to prevent NUL truncation at the wrapper boundary
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
