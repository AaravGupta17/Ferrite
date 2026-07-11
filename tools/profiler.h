// tools/profiler.h
#ifndef FERRITE_PROFILER_H
#define FERRITE_PROFILER_H

#include "types.h"
#include <stdint.h>
#include <stddef.h>

#define FE_PROF_MAX_OPS 64

/*
 * Per-operator timing record.
 * Uses CLOCK_MONOTONIC for wall-clock accuracy.
 */
typedef struct {
    char     name[64];
    uint64_t total_ns;    /* total nanoseconds across all calls */
    int      call_count;
} FeProfRecord;

/*
 * FeProfiler — accumulates timing across one or more inference runs.
 * Reset between runs with fe_profiler_reset().
 */
typedef struct {
    FeProfRecord records[FE_PROF_MAX_OPS];
    int          n_records;
    int          enabled;
} FeProfiler;

/* Initialise and enable the profiler. */
void fe_profiler_init(FeProfiler *p);

/* Reset all counters — call between inference runs. */
void fe_profiler_reset(FeProfiler *p);

/* Record time for a named operator. */
void fe_profiler_record(FeProfiler *p, const char *op_name, uint64_t ns);

/* Get current time in nanoseconds. */
uint64_t fe_profiler_now_ns(void);

/* Print a formatted report sorted by total time. */
void fe_profiler_print(const FeProfiler *p);

#endif // FERRITE_PROFILER_H