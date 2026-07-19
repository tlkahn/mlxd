#ifndef MLXD_CHAT_TEMPLATE_H
#define MLXD_CHAT_TEMPLATE_H

/* Load chat template from a model directory.
   Priority: tokenizer_config.json "chat_template" string,
   then chat_template.jinja, then chat_template.json (JSON string root).
   Returns heap-allocated string or NULL. Caller frees. */
char *model_chat_template_read(const char *model_dir);

#endif
