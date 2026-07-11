// planner/memory_planner.c
#include "memory_planner.h"
#include "tensor.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "types.h"
#include "graph.h"
#include "allocator.h"
#include <stddef.h>
/* Alignment for every tensor buffer — 64 bytes for SIMD */
#define PLANNER_ALIGN 64

static size_t align_up(size_t x, size_t align) {
    return (x + align - 1) & ~(align - 1);
}

/*
 * Step 1: Compute tensor lifetimes.
 *
 * Walk the graph in topological order.
 * For each node at step s:
 *   - Each output tensor: first_use = s
 *   - Each input tensor:  last_use  = s  (update if already set)
 *
 * Weights and the input tensor are excluded from planning —
 * they have fixed storage.
 */
static int compute_lifetimes(const FeGraph *g, FeTensorLifetime *lifetimes,
                               int *n_lifetimes) {
    /* Initialize lifetime table */
    int n = 0;
    int tensor_to_lt[FE_MAX_ALLOCS];
    memset(tensor_to_lt, -1, sizeof(tensor_to_lt));

    for (int step = 0; step < g->n_nodes; step++) {
        const FeNode *node = &g->nodes[g->topo_order[step]];

        /* Outputs: record first_use = this step */
        for (int o = 0; o < node->n_outputs; o++) {
            int tidx = node->outputs[o];
            if (g->tensors[tidx].is_weight) continue;

            if (tensor_to_lt[tidx] == -1) {
                tensor_to_lt[tidx] = n;
                lifetimes[n].tensor_idx = tidx;
                lifetimes[n].first_use  = step;
                lifetimes[n].last_use   = step;

                /* Compute size */
                const FeTensorEntry *e = &g->tensors[tidx];
                size_t nbytes = fe_dtype_size(e->dtype);
                for (int d = 0; d < e->ndim; d++) nbytes *= (size_t)e->shape[d];
                lifetimes[n].size_bytes = (int)nbytes;
                n++;
            }
        }

        /* Inputs: update last_use = this step */
        for (int i = 0; i < node->n_inputs; i++) {
            int tidx = node->inputs[i];
            if (g->tensors[tidx].is_weight) continue;
            int lt = tensor_to_lt[tidx];
            if (lt != -1) lifetimes[lt].last_use = step;
        }
    }

    *n_lifetimes = n;
    return n;
}

/*
 * Step 2: Greedy buffer assignment.
 *
 * Process tensors in order of first_use.
 * For each tensor, find the smallest free buffer that fits.
 * If none found, allocate a new buffer at the current high-water mark.
 *
 * A buffer is "free" at step s if its owning tensor's last_use < s.
 *
 * This is a greedy interval graph coloring — not optimal but
 * produces good results in practice and runs in O(n^2).
 */
FeStatus fe_plan_memory(const FeGraph *g, FePlan *plan) {
    if (!g || !plan) return FE_ERR_NULL;
    if (!g->topo_valid) return FE_ERR_SHAPE;

    memset(plan, 0, sizeof(FePlan));

    /* Compute lifetimes */
    FeTensorLifetime *lts = plan->lifetimes;
    compute_lifetimes(g, lts, &plan->n_lifetimes);

    /*
     * Greedy assignment.
     *
     * free_pool: list of (offset, size, last_freed_at_step) entries
     * representing buffer regions that have been released.
     */
    typedef struct {
        size_t offset;
        size_t size;
        int    freed_at;   /* last_use of the tensor that owned this */
    } FreeSlot;

    FreeSlot pool[FE_MAX_ALLOCS];
    int      n_pool = 0;
    size_t   high_water = 0;

    /* Initialize all offsets to invalid */
    memset(plan->offsets, 0xFF, sizeof(plan->offsets));

    for (int i = 0; i < plan->n_lifetimes; i++) {
        FeTensorLifetime *lt = &lts[i];
        size_t need = align_up((size_t)lt->size_bytes, PLANNER_ALIGN);

        /* Search free pool for a slot that:
         *   1. Was freed before this tensor's first_use
         *   2. Is large enough to hold this tensor */
        int best = -1;
        size_t best_size = SIZE_MAX;

        for (int p = 0; p < n_pool; p++) {
            if (pool[p].freed_at < lt->first_use &&
                pool[p].size >= need &&
                pool[p].size < best_size) {
                best = p;
                best_size = pool[p].size;
            }
        }

        size_t assigned_offset;
        if (best != -1) {
            /* Reuse existing slot */
            assigned_offset = pool[best].offset;
            /* Remove from pool by swapping with last */
            pool[best] = pool[--n_pool];
        } else {
            /* Allocate new region at high-water mark */
            assigned_offset = high_water;
            high_water += need;
        }

        plan->offsets[lt->tensor_idx] = assigned_offset;

        /* Return this slot to the pool after last_use */
        pool[n_pool].offset   = assigned_offset;
        pool[n_pool].size     = need;
        pool[n_pool].freed_at = lt->last_use;
        n_pool++;
    }

    plan->total_activation_bytes = high_water;
    return FE_OK;
}
FeStatus fe_plan_apply(FeGraph *g, FePlan *plan,
                        void *activation_buf, size_t buf_size,
                        FeArena *metadata_arena) {
    if (!g || !plan || !activation_buf || !metadata_arena) return FE_ERR_NULL;
    if (buf_size < plan->total_activation_bytes) return FE_ERR_NOMEM;

    unsigned char *base = (unsigned char *)activation_buf;

    for (int i = 0; i < plan->n_lifetimes; i++) {
        int tidx = plan->lifetimes[i].tensor_idx;
        FeTensorEntry *e = &g->tensors[tidx];

        /* Build a non-owning tensor pointing into the activation buffer */
        if (!e->tensor) {
            e->tensor = fe_arena_alloc(metadata_arena, sizeof(FeTensor), _Alignof(FeTensor));
            if (!e->tensor) return FE_ERR_NOMEM;
        }

        e->tensor->data      = base + plan->offsets[tidx];
        e->tensor->dtype     = e->dtype;
        e->tensor->ndim      = e->ndim;
        e->tensor->nbytes    = (size_t)plan->lifetimes[i].size_bytes;
        e->tensor->owns_data = false;
        memcpy(e->tensor->shape, e->shape, e->ndim * sizeof(int));

        /* Compute strides */
        e->tensor->strides[e->ndim - 1] = 1;
        for (int d = e->ndim - 2; d >= 0; d--)
            e->tensor->strides[d] = e->tensor->strides[d+1] * e->shape[d+1];
    }

    return FE_OK;
}

void fe_plan_print(const FeGraph *g, const FePlan *plan) {
    printf("Memory Plan: %zu bytes total activation memory\n",
           plan->total_activation_bytes);
    printf("%-20s %6s %8s %8s %8s\n",
           "Tensor", "Bytes", "Offset", "First", "Last");
    printf("%-20s %6s %8s %8s %8s\n",
           "------", "-----", "------", "-----", "----");

    for (int i = 0; i < plan->n_lifetimes; i++) {
        const FeTensorLifetime *lt = &plan->lifetimes[i];
        printf("%-20s %6d %8zu %8d %8d\n",
               g->tensors[lt->tensor_idx].name,
               lt->size_bytes,
               plan->offsets[lt->tensor_idx],
               lt->first_use,
               lt->last_use);
    }

    /* Compute how much memory buffer reuse saved */
    size_t naive_total = 0;
    for (int i = 0; i < plan->n_lifetimes; i++)
        naive_total += align_up((size_t)plan->lifetimes[i].size_bytes,
                                PLANNER_ALIGN);

    printf("\nNaive (no reuse): %zu bytes\n", naive_total);
    printf("Planned (reuse):  %zu bytes\n", plan->total_activation_bytes);
    if (naive_total > 0)
        printf("Savings:          %.1f%%\n",
               100.0 * (1.0 - (double)plan->total_activation_bytes
                                     / naive_total));
}