// parallel/threadpool.c
/*
 * Malloc-backed worker threads with a mutex+cond FIFO queue. Fixed worker
 * count chosen at init (or defaulted to hardware threads). submit() enqueues
 * one job, wait() blocks until every enqueued job has finished. destroy()
 * flags shutdown and joins all workers.
 *
 * Thread creation/joining goes through the platform seam (core/platform.h) so
 * device ports supply their own primitives; the pool's mutex+cond are direct
 * pthread objects, which every supported host provides.
 */
#include "threadpool.h"
#include "platform.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

int fe_cpu_count(void) {
    return fe_platform_cpu_count();
}

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t  work;      /* signaled: a job arrived */
    pthread_cond_t  idle;      /* signaled: outstanding == 0 */
} FePoolSync;

typedef struct {
    void        *handle;     /* opaque thread handle from FePlatThreadCreate */
    FeThreadPool *pool;
} FeThread;

static void *fe_pool_worker(void *arg) {
    FeThread *self = (FeThread *)arg;
    FeThreadPool *tp = self->pool;
    FePoolSync *s = (FePoolSync *)tp->sync;

    for (;;) {
        pthread_mutex_lock(&s->lock);
        while (tp->n_jobs == 0 && !tp->shutdown) {
            pthread_cond_wait(&s->work, &s->lock);
        }
        if (tp->n_jobs == 0 && tp->shutdown) {
            pthread_mutex_unlock(&s->lock);
            break;
        }
        FeTaskJob *job = tp->head;
        tp->head = job->next;
        if (!tp->head) tp->tail = NULL;
        tp->n_jobs--;
        pthread_mutex_unlock(&s->lock);

        job->fn(job->ctx, job->index);

        pthread_mutex_lock(&s->lock);
        tp->outstanding--;
        if (tp->outstanding == 0) pthread_cond_broadcast(&s->idle);
        pthread_mutex_unlock(&s->lock);
        free(job);
    }
    return NULL;
}

FeStatus fe_threadpool_init(FeThreadPool *tp, int nthreads) {
    if (!tp) return FE_ERR_NULL;
    memset(tp, 0, sizeof(*tp));
    if (nthreads <= 0) nthreads = fe_cpu_count();
    if (nthreads < 1) nthreads = 1;
    tp->nthreads = nthreads;

    tp->sync = calloc(1, sizeof(FePoolSync));
    if (!tp->sync) return FE_ERR_NOMEM;
    FePoolSync *s = (FePoolSync *)tp->sync;
    pthread_mutex_init(&s->lock, NULL);
    pthread_cond_init(&s->work, NULL);
    pthread_cond_init(&s->idle, NULL);

    tp->threads = calloc((size_t)nthreads, sizeof(FeThread));
    if (!tp->threads) {
        fe_threadpool_destroy(tp);
        return FE_ERR_NOMEM;
    }
    FeThread *ts = (FeThread *)tp->threads;
    for (int i = 0; i < nthreads; i++) {
        ts[i].pool = tp;
        if (!fe_platform_thread_create(&ts[i].handle, fe_pool_worker,
                                       &ts[i])) {
            tp->nthreads = i;   /* only the created workers will be joined */
            fe_threadpool_destroy(tp);
            return FE_ERR_NOMEM;
        }
    }
    return FE_OK;
}

FeStatus fe_threadpool_submit(FeThreadPool *tp,
                              FeTaskFn fn, void *ctx, int index) {
    if (!tp || !fn) return FE_ERR_NULL;

    FeTaskJob *job = calloc(1, sizeof(FeTaskJob));
    if (!job) return FE_ERR_NOMEM;
    job->fn = fn; job->ctx = ctx; job->index = index;

    FePoolSync *s = (FePoolSync *)tp->sync;
    pthread_mutex_lock(&s->lock);
    if (tp->tail) tp->tail->next = job;
    else          tp->head = job;
    tp->tail = job;
    tp->n_jobs++;
    tp->outstanding++;
    pthread_cond_signal(&s->work);
    pthread_mutex_unlock(&s->lock);
    return FE_OK;
}

FeStatus fe_threadpool_wait(FeThreadPool *tp) {
    if (!tp) return FE_ERR_NULL;
    FePoolSync *s = (FePoolSync *)tp->sync;
    pthread_mutex_lock(&s->lock);
    while (tp->outstanding > 0) pthread_cond_wait(&s->idle, &s->lock);
    pthread_mutex_unlock(&s->lock);
    return FE_OK;
}

void fe_threadpool_destroy(FeThreadPool *tp) {
    if (!tp) return;
    if (tp->sync) {
        FePoolSync *s = (FePoolSync *)tp->sync;
        pthread_mutex_lock(&s->lock);
        tp->shutdown = 1;
        pthread_cond_broadcast(&s->work);
        pthread_mutex_unlock(&s->lock);

        if (tp->threads) {
            FeThread *ts = (FeThread *)tp->threads;
            for (int i = 0; i < tp->nthreads; i++)
                fe_platform_thread_join(ts[i].handle);
            free(tp->threads);
        }
        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->work);
        pthread_cond_destroy(&s->idle);
        free(tp->sync);
        tp->sync = NULL;
    }
    /* Free any jobs never consumed (destroy while work outstanding). */
    FeTaskJob *job = tp->head;
    while (job) { FeTaskJob *n = job->next; free(job); job = n; }
    tp->head = tp->tail = NULL;
    tp->threads = NULL;
}