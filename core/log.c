// core/log.c
#include "log.h"
#include <stdarg.h>
#include <stdio.h>

static const char *level_name(FeLogLevel lvl) {
    switch (lvl) {
        case FE_LOG_DEBUG: return "DEBUG";
        case FE_LOG_INFO:  return "INFO";
        case FE_LOG_WARN:  return "WARN";
        case FE_LOG_ERROR: return "ERROR";
        default:           return "?";
    }
}

void fe_log(FeLogLevel lvl, const char *file, int line, const char *fmt, ...) {
    fprintf(stderr, "[ferrite %s %s:%d] ", level_name(lvl), file, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fputc('\n', stderr);
}
