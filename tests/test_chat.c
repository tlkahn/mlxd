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
    test_diagnostics_free_null();
    test_diagnostics_free_idempotent();
    test_render_invalid_template();
    printf("test_chat: all passed\n");
    return 0;
}
