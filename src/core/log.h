#ifndef MLXD_CORE_LOG_H
#define MLXD_CORE_LOG_H

typedef enum {
    LOG_ERROR,
    LOG_WARN,
    LOG_INFO,
    LOG_DEBUG,
} log_level_t;

void log_set_level(log_level_t level);
log_level_t log_get_level(void);

void log_error(const char *fmt, ...);
void log_warn(const char *fmt, ...);
void log_info(const char *fmt, ...);
void log_debug(const char *fmt, ...);

#endif
