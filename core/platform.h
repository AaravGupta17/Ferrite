// core/platform.h — platform seam (Section 2.6).
//
// Every OS/compiler-specific bit in Ferrite funnels through this header:
// monotonic timing for the profiler, the log sink, and the thread-pool
// primitives for the parallel subsystem. Host (desktop) implementations live
// in platform.c; device ports (Pi, ESP32) provide their own platform.c and
// the FERRITE_NO_* definitions to compile the rest of Ferrite unchanged.
#ifndef FERRITE_PLATFORM_H
#define FERRITE_PLATFORM_H

#include <stdint.h>

/*
 * Monotonic clock returned in nanoseconds. Must never go backwards across the
 * same process, matching the POSIX CLOCK_MONOTONIC contract the profiler
 * relies on. Device ports substitute their own source (ESP32: esp_timer).
 */
uint64_t fe_platform_now_ns(void);

/*
 * Log sink. Default host build prints to stderr; device ports either forward
 * to a UART/console or compile it out. All load/init-time diagnostics in
 * core/log.h route through this (kernels still never log).
 */
void fe_platform_log_write(const char *line);

/* Thread pool primitives — null when built with multi-threading for
 * build-time probing or declared FERRITE_NO_PARALLEL. Each returns a
 * boolean "supported" flag so callers can fall back to in-thread work. */
int  fe_platform_threads_supported(void);
int  fe_platform_thread_create(void **handle, void *(*fn)(void *), void *arg);
void fe_platform_thread_join(void *handle);

/* Number of hardware threads, for the default pool size. Never 0. */
int fe_platform_cpu_count(void);

#endif /* FERRITE_PLATFORM_H */