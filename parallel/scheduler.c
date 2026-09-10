// parallel/scheduler.c
/*
 * Ready-set scheduler over FeGraph + topo_order.
 *
 * Data:
 *   need[t]   — producers still outstanding for tensor t (init: count of
 *                nodes producing t; weights/inputs start at 0).
 *   met[n]    — inputs of node n already produced (monotonic).
 *
 * A node becomes ready exactly when met[n] reaches n_inputs — a single
 * transition — so it is enqueued once and only once. Two execution modes:
 *
 *   sequential (pool == NULL): the in-memory ready queue is owned by the
 *       calling thread; a finished node pushes its newly-ready children onto
 *       it and the main loop drains it. exec_order[] records causal order.
 *
 *   parallel (pool != NULL): a finished worker submits each newly-ready
 *       child straight to the pool via fe_threadpool_submit, which bumps the
 *       outstanding counter *before* the parent's own counter decrements
 *       (submit happens inside scheduler_execute, the decrement in the worker
 *       loop afterwards). The pool's join barrier therefore absorbs the whole
 *       cascade: one fe_threadpool_wait() covers it without a shared queue.
 */
#include "scheduler.h"
#include "threadpool.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    const FeGraph *g;
    FeSchedNodeFn  fn;
    void          *ctx;
    FeThreadPool  *pool;      /* NULL for sequential mode */
    int           *need;      /* outstanding producers per tensor */
    int           *met;       /* met inputs per node */
    int           *enqueued;
    int           *ready;     /* sequential-mode queue */
    int            qhead, qtail;
    FeSchedResult *result;
} Scheduler;

/* Execute one node, then release every consumer whose last producer this
 * node just finished. Newly-ready nodes go to the pool (parallel) or the
 * in-memory queue (sequential). */
static void scheduler_execute(void *ctxv, int node) {
    Scheduler *s = (Scheduler *)ctxv;
    const FeNode *n = &s->g->nodes[node];

    s->fn(s->ctx, node);
    s->result->exec_order[s->result->n_executed++] = node;

    for (int o = 0; o < n->n_outputs; o++) {
        int t = n->outputs[o];
        s->need[t]--;
        if (s->need[t] > 0) continue;
        for (int c = 0; c < s->g->n_nodes; c++) {
            const FeNode *cn = &s->g->nodes[c];
            int consumes = 0;
            for (int i = 0; i < cn->n_inputs; i++)
                if (cn->inputs[i] == t) { consumes = 1; break; }
            if (!consumes) continue;
            int met = ++s->met[c];
            if (met == cn->n_inputs && !s->enqueued[c]) {
                s->enqueued[c] = 1;
                if (s->pool) {
                    fe_threadpool_submit(s->pool, scheduler_execute, s, c);
                } else {
                    s->ready[s->qtail++] = c;
                }
            }
        }
    }
}

FeStatus fe_scheduler_run(const FeGraph *g,
                          FeSchedNodeFn fn, void *ctx,
                          FeThreadPool *pool,
                          FeSchedResult *result) {
    if (!g || !fn || !result) return FE_ERR_NULL;
    if (g->n_nodes <= 0) return FE_ERR_SHAPE;

    Scheduler sc;
    memset(&sc, 0, sizeof(sc));
    sc.g = g; sc.fn = fn; sc.ctx = ctx; sc.pool = (pool && pool->nthreads > 1) ? pool : NULL;
    sc.result = result;
    result->n_executed = 0;

    sc.need = calloc((size_t)g->n_tensors, sizeof(int));
    sc.met  = calloc((size_t)g->n_nodes, sizeof(int));
    sc.enqueued = calloc((size_t)g->n_nodes, sizeof(int));
    sc.ready = calloc((size_t)g->n_nodes, sizeof(int));
    if (!sc.need || !sc.met || !sc.enqueued || !sc.ready) {
        free(sc.need); free(sc.met); free(sc.enqueued); free(sc.ready);
        return FE_ERR_NOMEM;
    }

    for (int n = 0; n < g->n_nodes; n++) {
        for (int o = 0; o < g->nodes[n].n_outputs; o++)
            sc.need[g->nodes[n].outputs[o]]++;
        for (int i = 0; i < g->nodes[n].n_inputs; i++)
            if (sc.need[g->nodes[n].inputs[i]] == 0) sc.met[n]++;
    }

    for (int n = 0; n < g->n_nodes; n++) {
        if (sc.met[n] == g->nodes[n].n_inputs && !sc.enqueued[n]) {
            sc.enqueued[n] = 1;
            if (sc.pool) {
                FeStatus s = fe_threadpool_submit(sc.pool, scheduler_execute, &sc, n);
                if (s != FE_OK) {
                    free(sc.need); free(sc.met); free(sc.enqueued); free(sc.ready);
                    return s;
                }
            } else {
                sc.ready[sc.qtail++] = n;
            }
        }
    }

    if (sc.pool) {
        FeStatus s = fe_threadpool_wait(sc.pool);
        if (s != FE_OK) {
            free(sc.need); free(sc.met); free(sc.enqueued); free(sc.ready);
            return s;
        }
        if (sc.result->n_executed != g->n_nodes) {
            free(sc.need); free(sc.met); free(sc.enqueued); free(sc.ready);
            return FE_ERR_SHAPE;
        }
    } else {
        while (sc.qhead < sc.qtail) {
            int node = sc.ready[sc.qhead++];
            scheduler_execute(&sc, node);
        }
        if (sc.result->n_executed != g->n_nodes) {
            free(sc.need); free(sc.met); free(sc.enqueued); free(sc.ready);
            return FE_ERR_SHAPE;
        }
    }

    free(sc.need); free(sc.met); free(sc.enqueued); free(sc.ready);
    return FE_OK;
}