// planner/memory_planner.h
#ifndef FERRITE_MEMORY_PLANNER_H
#define FERRITE_MEMORY_PLANNER_H

#include "types.h"
#include "graph.h"
#include "allocator.h"
#include <stddef.h>

#define FE_MAX_ALLOCS 1024

/*
 * Lifetime of a single tensor in the graph.
 * Expressed as indices into the topological order.
 */
typedef struct {
    int tensor_idx;   /* index into g->tensors[]         */
    int first_use;    /* topo step when first produced   */
    int last_use;     /* topo step when last consumed    */
    int size_bytes;   /* bytes required                  */
} FeTensorLifetime;

/*
 * A buffer allocation: a contiguous region of the activation buffer
 * assigned to one or more tensors whose lifetimes don't overlap.
 */
typedef struct {
    size_t offset;      /* byte offset from activation buffer base */
    size_t size_bytes;  /* size of this region                     */
    int    tensor_idx;  /* which tensor currently owns this region  */
} FeBufferAlloc;

/*
 * FePlan — the result of memory planning.
 *
 * Contains:
 *   - per-tensor buffer offsets (index into activation buffer)
 *   - total activation buffer size required
 *   - lifetime analysis results for inspection/debugging
 */
typedef struct {
    size_t          total_activation_bytes;  /* minimum buffer size needed */
    size_t          offsets[FE_MAX_ALLOCS];  /* offset per tensor index    */
    FeTensorLifetime lifetimes[FE_MAX_ALLOCS];
    int             n_lifetimes;
} FePlan;

/*
 * Analyse tensor lifetimes and produce a static memory plan.
 *
 * After this call:
 *   plan->offsets[i] = byte offset in activation buffer for tensor i
 *   plan->total_activation_bytes = minimum buffer size required
 *
 * Weights are excluded — they have their own arena.
 * Input tensor is excluded — it's provided by the caller.
 */
FeStatus fe_plan_memory(const FeGraph *g, FePlan *plan);

/*
 * Apply the plan to the graph: assign tensor data pointers
 * from the provided activation buffer using pre-computed offsets.
 *
 * Must be called once before fe_runtime_run when using the planner.
 * The activation_buf must be at least plan->total_activation_bytes bytes.
 */
FeStatus fe_plan_apply(FeGraph *g, FePlan *plan,
                        void *activation_buf, size_t buf_size,
                        FeArena *metadata_arena);

/* Print the memory plan for debugging. */
void fe_plan_print(const FeGraph *g, const FePlan *plan);

#endif // FERRITE_MEMORY_PLANNER_H