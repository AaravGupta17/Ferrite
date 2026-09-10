// parallel/scheduler.h
#ifndef FERRITE_SCHEDULER_H
#define FERRITE_SCHEDULER_H

#include "types.h"
#include "../graph/graph.h"

typedef struct FeThreadPool FeThreadPool;

/* Graph-parallel task scheduler. Runs the given graph's topologically sorted
 * nodes against a ready set: a node becomes executable the moment every input
 * tensor has been produced (weights and caller-seeded tensors are "already
 * produced"). Ready nodes execute via the worker pool when one is supplied —
 * a node's fn may discover child nodes and resubmit them, which the pool's
 * outstanding counter turns into an automatic join barrier.
 *
 * correctness contract: exec_order[] records every node exactly once, in an
 * order where every node appears after all of its producers. */
typedef void (*FeSchedNodeFn)(void *ctx, int node_index);

typedef struct {
    int *exec_order;   /* caller-allocated, capacity = n_nodes; filled */
    int  n_executed;
} FeSchedResult;

FeStatus fe_scheduler_run(const FeGraph *g,
                          FeSchedNodeFn fn, void *ctx,
                          FeThreadPool *pool,
                          FeSchedResult *result);

#endif // FERRITE_SCHEDULER_H