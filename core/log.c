// core/log.c
#include "log.h"
#include "platform.h"
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
    char linebuf[256];
    int off = snprintf(linebuf, sizeof(linebuf),
                       "[ferrite %s %s:%d] ", level_name(lvl), file, line);

    va_list args;
    va_start(args, fmt);
    off += vsnprintf(linebuf + (size_t)off, sizeof(linebuf) - (size_t)off,
                     fmt, args);
    va_end(args);

    if ((size_t)off >= sizeof(linebuf)) off = (int)sizeof(linebuf) - 2;
    linebuf[off] = '\n';
    linebuf[off + 1] = '\0';
    fe_platform_log_write(linebuf);
}
