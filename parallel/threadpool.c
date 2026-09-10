// parallel/threadpool.c
/*
 * Malloc-backed worker threads with a mutex+cond FIFO queue. Fixed worker
 * count chosen at init (or defaulted to hardware threads). submit() enqueues
 * one job, wait() blocks until every enqueued job has finished. destroy()
 * flags shutdown and joins all workers.
 */
#include "threadpool.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#if defined(_POSIX_VERSION) && _POSIX_VERSION >= 200112L
#endif
#endif

int fe_cpu_count(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t  work;      /* signaled: a job arrived */
    pthread_cond_t  idle;      /* signaled: outstanding == 0 */
} FePoolSync;

typedef struct {
    pthread_t   id;
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
        if (pthread_create(&ts[i].id, NULL, fe_pool_worker, &ts[i]) != 0) {
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
            for (int i = 0; i < tp->nthreads; i++) pthread_join(ts[i].id, NULL);
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