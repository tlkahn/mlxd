#ifndef MLXD_CORE_SYSINFO_H
#define MLXD_CORE_SYSINFO_H

#include <stddef.h>
#include <stdint.h>

size_t   sysinfo_rss(void);
uint64_t sysinfo_available_memory(void);
double   sysinfo_memory_usage_pct(void);

#endif
