#ifndef MLXD_CHAT_H
#define MLXD_CHAT_H

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    const char *tmpl;
    const char *messages_json;
    const char *tools_json;
    const char *extra_json;
    bool        add_generation_prompt;
    size_t      tmpl_len;            // 0 = strlen-measured; must exclude terminal NUL
    size_t      messages_json_len;
    size_t      tools_json_len;
    size_t      extra_json_len;
} chat_render_params_t;

typedef struct {
    char *error;
} chat_diagnostics_t;

/* Render a chat template via the vendored jinja engine.
 * Returns a heap-allocated string on success, NULL on error (check diag->error).
 * Caller must free the returned string. */
char *chat_render(const chat_render_params_t *params, chat_diagnostics_t *diag);

void chat_diagnostics_free(chat_diagnostics_t *diag);

#endif
