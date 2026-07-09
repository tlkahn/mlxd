#include "chat/chat.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *CHATML =
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}";

static const char *MESSAGES =
    "[{\"role\":\"user\",\"content\":\"hello\"}]";

/* --- NULL / missing params ------------------------------------------------ */

static void test_null_params(void) {
    chat_diagnostics_t diag = {0};
    char *r = chat_render(NULL, &diag);
    assert(r == NULL);
    assert(diag.error != NULL);
    chat_diagnostics_free(&diag);
}

static void test_null_tmpl(void) {
    chat_render_params_t p = {.tmpl = NULL, .messages_json = MESSAGES};
    chat_diagnostics_t diag = {0};
    char *r = chat_render(&p, &diag);
    assert(r == NULL);
    assert(diag.error != NULL);
    chat_diagnostics_free(&diag);
}

static void test_null_messages(void) {
    chat_render_params_t p = {.tmpl = CHATML, .messages_json = NULL};
    chat_diagnostics_t diag = {0};
    char *r = chat_render(&p, &diag);
    assert(r == NULL);
    assert(diag.error != NULL);
    chat_diagnostics_free(&diag);
}

static void test_empty_template(void) {
    chat_render_params_t p = {.tmpl = "", .messages_json = MESSAGES};
    chat_diagnostics_t diag = {0};
    char *r = chat_render(&p, &diag);
    assert(r == NULL);
    assert(diag.error != NULL);
    chat_diagnostics_free(&diag);
}

/* --- NULL diag must not crash -------------------------------------------- */

static void test_null_diag_on_error(void) {
    char *r = chat_render(NULL, NULL);
    assert(r == NULL);
}

static void test_null_diag_empty_template(void) {
    chat_render_params_t p = {.tmpl = "", .messages_json = MESSAGES};
    char *r = chat_render(&p, NULL);
    assert(r == NULL);
}

/* --- Happy path ---------------------------------------------------------- */

static void test_render_chatml(void) {
    chat_render_params_t p = {
        .tmpl = CHATML,
        .messages_json = MESSAGES,
        .tools_json = NULL,
        .extra_json = NULL,
        .add_generation_prompt = true,
    };
    chat_diagnostics_t diag = {0};
    char *r = chat_render(&p, &diag);
    assert(r != NULL);
    assert(diag.error == NULL);
    assert(strstr(r, "<|im_start|>user") != NULL);
    assert(strstr(r, "hello") != NULL);
    assert(strstr(r, "<|im_start|>assistant") != NULL);
    free(r);
    chat_diagnostics_free(&diag);
}

static void test_render_no_generation_prompt(void) {
    chat_render_params_t p = {
        .tmpl = CHATML,
        .messages_json = MESSAGES,
        .add_generation_prompt = false,
    };
    chat_diagnostics_t diag = {0};
    char *r = chat_render(&p, &diag);
    assert(r != NULL);
    assert(diag.error == NULL);
    assert(strstr(r, "<|im_start|>user") != NULL);
    assert(strstr(r, "<|im_start|>assistant") == NULL);
    free(r);
    chat_diagnostics_free(&diag);
}

/* --- diagnostics_free edge cases ----------------------------------------- */

static void test_diagnostics_free_null(void) {
    chat_diagnostics_free(NULL);
}

static void test_diagnostics_free_idempotent(void) {
    chat_diagnostics_t diag = {0};
    chat_render(NULL, &diag);
    assert(diag.error != NULL);
    chat_diagnostics_free(&diag);
    assert(diag.error == NULL);
    chat_diagnostics_free(&diag);
}

/* --- tools_json and extra_json pass-through ------------------------------ */

static void test_render_tools_json(void) {
    chat_render_params_t p = {
        .tmpl = "{{ tools | length }}",
        .messages_json = "[]",
        .tools_json = "[{\"type\":\"function\"}]",
    };
    chat_diagnostics_t diag = {0};
    char *r = chat_render(&p, &diag);
    assert(r != NULL);
    assert(diag.error == NULL);
    assert(strcmp(r, "1") == 0);
    free(r);
    chat_diagnostics_free(&diag);
}

static void test_render_extra_json(void) {
    chat_render_params_t p = {
        .tmpl = "{{ bos_token }}hello{{ eos_token }}",
        .messages_json = "[]",
        .extra_json = "{\"bos_token\":\"<s>\",\"eos_token\":\"</s>\"}",
    };
    chat_diagnostics_t diag = {0};
    char *r = chat_render(&p, &diag);
    assert(r != NULL);
    assert(diag.error == NULL);
    assert(strcmp(r, "<s>hello</s>") == 0);
    free(r);
    chat_diagnostics_free(&diag);
}

/* --- NUL-byte rejection -------------------------------------------------- */

static void test_nul_at_position_zero(void) {
    const char tmpl[] = "\0real_template";
    chat_render_params_t p = {
        .tmpl = tmpl,
        .messages_json = "[]",
        .tmpl_len = sizeof(tmpl) - 1,
    };
    chat_diagnostics_t diag = {0};
    char *r = chat_render(&p, &diag);
    assert(r == NULL);
    assert(diag.error != NULL);
    assert(strstr(diag.error, "NUL") != NULL);
    chat_diagnostics_free(&diag);
}

static void test_nul_in_template(void) {
    const char tmpl[] = "hi\0there";
    chat_render_params_t p = {
        .tmpl = tmpl,
        .messages_json = "[]",
        .tmpl_len = sizeof(tmpl) - 1,
    };
    chat_diagnostics_t diag = {0};
    char *r = chat_render(&p, &diag);
    assert(r == NULL);
    assert(diag.error != NULL);
    assert(strstr(diag.error, "NUL") != NULL);
    chat_diagnostics_free(&diag);
}

static void test_nul_in_messages(void) {
    const char msgs[] = "[]\0x";
    chat_render_params_t p = {
        .tmpl = "ok",
        .messages_json = msgs,
        .messages_json_len = sizeof(msgs) - 1,
    };
    chat_diagnostics_t diag = {0};
    char *r = chat_render(&p, &diag);
    assert(r == NULL);
    assert(diag.error != NULL);
    assert(strstr(diag.error, "NUL") != NULL);
    chat_diagnostics_free(&diag);
}

static void test_nul_in_tools(void) {
    const char tools[] = "[]\0x";
    chat_render_params_t p = {
        .tmpl = "ok",
        .messages_json = "[]",
        .tools_json = tools,
        .tools_json_len = sizeof(tools) - 1,
    };
    chat_diagnostics_t diag = {0};
    char *r = chat_render(&p, &diag);
    assert(r == NULL);
    assert(diag.error != NULL);
    assert(strstr(diag.error, "NUL") != NULL);
    chat_diagnostics_free(&diag);
}

static void test_nul_in_extra(void) {
    const char extra[] = "{}\0x";
    chat_render_params_t p = {
        .tmpl = "ok",
        .messages_json = "[]",
        .extra_json = extra,
        .extra_json_len = sizeof(extra) - 1,
    };
    chat_diagnostics_t diag = {0};
    char *r = chat_render(&p, &diag);
    assert(r == NULL);
    assert(diag.error != NULL);
    assert(strstr(diag.error, "NUL") != NULL);
    chat_diagnostics_free(&diag);
}

/* --- Clean string with explicit strlen len (no false positive) ----------- */

static void test_sizeof_minus_one_is_clean(void) {
    chat_render_params_t p = {
        .tmpl = CHATML,
        .messages_json = MESSAGES,
        .tmpl_len = strlen(CHATML),
        .messages_json_len = strlen(MESSAGES),
    };
    chat_diagnostics_t diag = {0};
    char *r = chat_render(&p, &diag);
    assert(r != NULL);
    assert(diag.error == NULL);
    free(r);
    chat_diagnostics_free(&diag);
}

/* --- NUL check with default (zero) _len fields --------------------------- */

static void test_nul_check_default_len_smoke(void) {
    chat_render_params_t p = {
        .tmpl = CHATML,
        .messages_json = MESSAGES,
    };
    chat_diagnostics_t diag = {0};
    char *r = chat_render(&p, &diag);
    assert(r != NULL);
    assert(diag.error == NULL);
    free(r);
    chat_diagnostics_free(&diag);
}

/* --- Invalid template (jinja error path) --------------------------------- */

static void test_render_invalid_template(void) {
    chat_render_params_t p = {
        .tmpl = "{% if %}",
        .messages_json = MESSAGES,
    };
    chat_diagnostics_t diag = {0};
    char *r = chat_render(&p, &diag);
    assert(r == NULL);
    assert(diag.error != NULL);
    chat_diagnostics_free(&diag);
}

int main(void) {
    test_null_params();
    test_null_tmpl();
    test_null_messages();
    test_empty_template();
    test_null_diag_on_error();
    test_null_diag_empty_template();
    test_render_chatml();
    test_render_no_generation_prompt();
    test_render_tools_json();
    test_render_extra_json();
    test_nul_at_position_zero();
    test_nul_in_template();
    test_nul_in_messages();
    test_nul_in_tools();
    test_nul_in_extra();
    test_sizeof_minus_one_is_clean();
    test_nul_check_default_len_smoke();
    test_diagnostics_free_null();
    test_diagnostics_free_idempotent();
    test_render_invalid_template();
    printf("test_chat: all passed\n");
    return 0;
}
