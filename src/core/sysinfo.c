#include "core/sysinfo.h"

#include <mach/mach.h>
#include <mach/mach_host.h>

size_t sysinfo_rss(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return 0;
    return (size_t)info.resident_size;
}

uint64_t sysinfo_available_memory(void) {
    vm_statistics64_data_t stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&stats, &count) !=
        KERN_SUCCESS)
        return 0;
    vm_size_t page_size;
    host_page_size(mach_host_self(), &page_size);
    return (uint64_t)(stats.free_count + stats.inactive_count) * page_size;
}

double sysinfo_memory_usage_pct(void) {
    vm_statistics64_data_t stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&stats, &count) !=
        KERN_SUCCESS)
        return 0.0;
    uint64_t used = (uint64_t)(stats.active_count + stats.wire_count);
    uint64_t total =
        (uint64_t)(stats.active_count + stats.inactive_count + stats.free_count + stats.wire_count);
    if (total == 0)
        return 0.0;
    return (double)used / (double)total * 100.0;
}
