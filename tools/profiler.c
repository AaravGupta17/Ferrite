// tools/profiler.c
#include "profiler.h"
#include <string.h>
#include <stdio.h>
#include <time.h>

void fe_profiler_init(FeProfiler *p) {
    memset(p, 0, sizeof(FeProfiler));
    p->enabled = 1;
}

void fe_profiler_reset(FeProfiler *p) {
    for (int i = 0; i < p->n_records; i++) {
        p->records[i].total_ns   = 0;
        p->records[i].call_count = 0;
    }
}

uint64_t fe_profiler_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void fe_profiler_record(FeProfiler *p, const char *op_name, uint64_t ns) {
    if (!p || !p->enabled) return;

    /* Find existing record */
    for (int i = 0; i < p->n_records; i++) {
        if (strncmp(p->records[i].name, op_name, 63) == 0) {
            p->records[i].total_ns   += ns;
            p->records[i].call_count += 1;
            return;
        }
    }

    /* New record */
    if (p->n_records >= FE_PROF_MAX_OPS) return;
    FeProfRecord *r = &p->records[p->n_records++];
    strncpy(r->name, op_name, 63);
    r->name[63]    = '\0';
    r->total_ns    = ns;
    r->call_count  = 1;
}

/* Sort records by total_ns descending — simple insertion sort */
static void sort_records(FeProfRecord *records, int n) {
    for (int i = 1; i < n; i++) {
        FeProfRecord key = records[i];
        int j = i - 1;
        while (j >= 0 && records[j].total_ns < key.total_ns) {
            records[j + 1] = records[j];
            j--;
        }
        records[j + 1] = key;
    }
}

void fe_profiler_print(const FeProfiler *p) {
    if (!p || p->n_records == 0) {
        printf("Profiler: no data\n");
        return;
    }

    /* Copy and sort */
    FeProfRecord sorted[FE_PROF_MAX_OPS];
    memcpy(sorted, p->records, p->n_records * sizeof(FeProfRecord));
    sort_records(sorted, p->n_records);

    /* Compute total */
    uint64_t total_ns = 0;
    for (int i = 0; i < p->n_records; i++)
        total_ns += sorted[i].total_ns;

    printf("\n%-20s %6s %12s %10s %8s\n",
           "Operator", "Calls", "Total(ms)", "Avg(ms)", "% Time");
    printf("%-20s %6s %12s %10s %8s\n",
           "--------", "-----", "---------", "-------", "------");

    for (int i = 0; i < p->n_records; i++) {
        FeProfRecord *r = &sorted[i];
        double total_ms = (double)r->total_ns / 1e6;
        double avg_ms   = total_ms / r->call_count;
        double pct      = total_ns > 0
                        ? 100.0 * r->total_ns / total_ns
                        : 0.0;
        printf("%-20s %6d %12.3f %10.3f %7.1f%%\n",
               r->name, r->call_count, total_ms, avg_ms, pct);
    }

    printf("%-20s %6s %12.3f\n",
           "TOTAL", "", (double)total_ns / 1e6);
}