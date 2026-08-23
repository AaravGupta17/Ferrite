// core/log.h
#ifndef FERRITE_LOG_H
#define FERRITE_LOG_H

#include <stddef.h>

/*
 * Logging — a minimal stderr logger for load/init-time diagnostics.
 *
 * Policy: kernels never log (no allocation, no I/O in hot paths).
 * The importer and engine log at module boundaries only — model
 * load, runtime init, and fatal dispatch errors.
 *
 * Levels are filtered at compile time by FERRITE_LOG_LEVEL: messages
 * below the minimum compile to nothing. Default minimum is WARN so
 * release builds stay silent unless something is wrong.
 */

typedef enum {
    FE_LOG_DEBUG = 0,
    FE_LOG_INFO  = 1,
    FE_LOG_WARN  = 2,
    FE_LOG_ERROR = 3,
} FeLogLevel;

#ifndef FERRITE_LOG_LEVEL
#define FERRITE_LOG_LEVEL FE_LOG_WARN
#endif

/* Core sink. Prefer the fe_log_* wrappers below, which capture file/line. */
void fe_log(FeLogLevel lvl, const char *file, int line, const char *fmt, ...);

#if FERRITE_LOG_LEVEL <= FE_LOG_DEBUG
#define fe_log_debug(...) fe_log(FE_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#else
#define fe_log_debug(...) ((void)0)
#endif

#if FERRITE_LOG_LEVEL <= FE_LOG_INFO
#define fe_log_info(...) fe_log(FE_LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#else
#define fe_log_info(...) ((void)0)
#endif

#if FERRITE_LOG_LEVEL <= FE_LOG_WARN
#define fe_log_warn(...) fe_log(FE_LOG_WARN, __FILE__, __LINE__, __VA_ARGS__)
#else
#define fe_log_warn(...) ((void)0)
#endif

#if FERRITE_LOG_LEVEL <= FE_LOG_ERROR
#define fe_log_error(...) fe_log(FE_LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#else
#define fe_log_error(...) ((void)0)
#endif

#endif // FERRITE_LOG_H
