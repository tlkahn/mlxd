#include "core/sysinfo.h"

#include <assert.h>
#include <stdio.h>

static void test_rss_nonzero(void) {
    size_t rss = sysinfo_rss();
    assert(rss > 0);
}

static void test_available_memory(void) {
    uint64_t avail = sysinfo_available_memory();
    assert(avail > 0);
}

static void test_memory_usage_pct(void) {
    double pct = sysinfo_memory_usage_pct();
    assert(pct > 0.0 && pct <= 100.0);
}

int main(void) {
    test_rss_nonzero();
    test_available_memory();
    test_memory_usage_pct();
    printf("test_sysinfo: all passed\n");
    return 0;
}
