#include "core/log.h"

#include <stdarg.h>
#include <stdio.h>
#include <time.h>

static log_level_t g_level = LOG_INFO;

void log_set_level(log_level_t level) { g_level = level; }

log_level_t log_get_level(void) { return g_level; }

static const char *level_str(log_level_t level) {
    switch (level) {
    case LOG_ERROR: return "ERR";
    case LOG_WARN:  return "WRN";
    case LOG_INFO:  return "INF";
    case LOG_DEBUG: return "DBG";
    }
    return "???";
}

static void log_emit(log_level_t level, const char *fmt, va_list ap) {
    if (level > g_level)
        return;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    localtime_r(&ts.tv_sec, &tm);

    fprintf(stderr, "%02d:%02d:%02d.%03ld [%s] ",
            tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000, level_str(level));
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
}

void log_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    log_emit(LOG_ERROR, fmt, ap);
    va_end(ap);
}

void log_warn(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    log_emit(LOG_WARN, fmt, ap);
    va_end(ap);
}

void log_info(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    log_emit(LOG_INFO, fmt, ap);
    va_end(ap);
}

void log_debug(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    log_emit(LOG_DEBUG, fmt, ap);
    va_end(ap);
}
