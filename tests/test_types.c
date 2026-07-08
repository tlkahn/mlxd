#include "core/types.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void test_finish_reason_str(void) {
    assert(strcmp(finish_reason_str(FINISH_STOP), "stop") == 0);
    assert(strcmp(finish_reason_str(FINISH_LENGTH), "length") == 0);
    assert(strcmp(finish_reason_str(FINISH_TOOL_CALLS), "tool_calls") == 0);
    assert(strcmp(finish_reason_str(FINISH_CONTENT_FILTER), "content_filter") == 0);
    assert(strcmp(finish_reason_str(FINISH_CANCELLED), "cancelled") == 0);
}

static void test_finish_reason_wire_maps_cancelled(void) {
    assert(strcmp(finish_reason_wire_str(FINISH_CANCELLED), "stop") == 0);
    assert(strcmp(finish_reason_wire_str(FINISH_STOP), "stop") == 0);
    assert(strcmp(finish_reason_wire_str(FINISH_LENGTH), "length") == 0);
}

static void test_sampling_params_defaults(void) {
    sampling_params_t p = SAMPLING_PARAMS_DEFAULT;
    const char *err = NULL;
    assert(sampling_params_validate(&p, &err));
}

static void test_sampling_params_invalid_temperature(void) {
    sampling_params_t p = SAMPLING_PARAMS_DEFAULT;
    p.temperature = -1.0f;
    const char *err = NULL;
    assert(!sampling_params_validate(&p, &err));
    assert(err != NULL);
}

static void test_sampling_params_invalid_top_p(void) {
    sampling_params_t p = SAMPLING_PARAMS_DEFAULT;
    p.top_p = 1.5f;
    const char *err = NULL;
    assert(!sampling_params_validate(&p, &err));
}

static void test_role_str(void) {
    assert(strcmp(role_str(ROLE_SYSTEM), "system") == 0);
    assert(strcmp(role_str(ROLE_USER), "user") == 0);
    assert(strcmp(role_str(ROLE_ASSISTANT), "assistant") == 0);
    assert(strcmp(role_str(ROLE_TOOL), "tool") == 0);
}

static void test_message_free_null(void) {
    message_free(NULL);
}

static void test_tool_call_free_null(void) {
    tool_call_free(NULL);
}

int main(void) {
    test_finish_reason_str();
    test_finish_reason_wire_maps_cancelled();
    test_sampling_params_defaults();
    test_sampling_params_invalid_temperature();
    test_sampling_params_invalid_top_p();
    test_role_str();
    test_message_free_null();
    test_tool_call_free_null();
    printf("test_types: all passed\n");
    return 0;
}
