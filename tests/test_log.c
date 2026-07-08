#include "core/log.h"

#include <assert.h>
#include <stdio.h>

static void test_default_level(void) {
    assert(log_get_level() == LOG_INFO);
}

static void test_set_level(void) {
    log_set_level(LOG_DEBUG);
    assert(log_get_level() == LOG_DEBUG);
    log_set_level(LOG_ERROR);
    assert(log_get_level() == LOG_ERROR);
    log_set_level(LOG_INFO);
}

static void test_log_emits(void) {
    log_set_level(LOG_DEBUG);
    log_error("test error %d", 42);
    log_warn("test warn %s", "hello");
    log_info("test info");
    log_debug("test debug");
    log_set_level(LOG_INFO);
}

int main(void) {
    test_default_level();
    test_set_level();
    test_log_emits();
    printf("test_log: all passed\n");
    return 0;
}
