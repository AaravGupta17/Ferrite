// parallel/threadpool.h
#ifndef FERRITE_THREADPOOL_H
#define FERRITE_THREADPOOL_H

#include "types.h"

/* Stage 11 parallel runtime: a fixed-worker thread pool with a FIFO task
 * queue. Workers pull (fn, ctx, index) items; the queue and the outstanding
 * counter are protected by one mutex + one condition variable pair, so
 * submit/wait are trivially correct and the pool is single-file and
 * zero-dependency apart from pthreads (winpthread on Windows).
 *
 * Jobs partition work by integer index; there is no inter-task data sharing
 * the pool knows about — callers own their synchronization (the parallel
 * GEMM in parallel/parallel_gemm.c splits disjoint output rows, so none is
 * needed). fe_cpu_count() reports usable hardware threads for the default. */

typedef void (*FeTaskFn)(void *ctx, int index);

typedef struct FeTaskJob {
    FeTaskFn   fn;
    void      *ctx;
    int        index;
    struct FeTaskJob *next;   /* intrusive FIFO */
} FeTaskJob;

typedef struct FeThreadPool {
    void        *threads;     /* FeThread* work area, opaque */
    int          nthreads;
    FeTaskJob   *head, *tail;
    int          n_jobs;
    int          outstanding;
    void        *sync;        /* pthread mutex/cond work area, opaque */
    int          shutdown;
} FeThreadPool;

int fe_cpu_count(void);

FeStatus fe_threadpool_init(FeThreadPool *tp, int nthreads);
FeStatus fe_threadpool_submit(FeThreadPool *tp,
                              FeTaskFn fn, void *ctx, int index);
FeStatus fe_threadpool_wait(FeThreadPool *tp);
void     fe_threadpool_destroy(FeThreadPool *tp);

#endif // FERRITE_THREADPOOL_H