/* core/platform.c — host (desktop) implementation of the platform seam.
 *
 * POSIX CLOCK_MONOTONIC timing (MinGW winpthreads provides clock_gettime),
 * stderr log sink, and pthread-based thread primitives. Device ports replace
 * this file with their own implementations; nothing else in the tree touches
 * OS APIs directly.
 */

#include "platform.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <pthread.h>

uint64_t fe_platform_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void fe_platform_log_write(const char *line) {
    fputs(line, stderr);
}

int fe_platform_threads_supported(void) {
    return 1;
}

int fe_platform_thread_create(void **handle, void *(*fn)(void *), void *arg) {
    pthread_t *id = (pthread_t *)malloc(sizeof(pthread_t));
    if (!id) return 0;
    if (pthread_create(id, NULL, fn, arg) != 0) {
        free(id);
        return 0;
    }
    *handle = id;
    return 1;
}

void fe_platform_thread_join(void *handle) {
    pthread_t *id = (pthread_t *)handle;
    pthread_join(*id, NULL);
    free(id);
}

int fe_platform_cpu_count(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}