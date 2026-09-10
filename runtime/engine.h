#ifndef FERRITE_ENGINE_H
#define FERRITE_ENGINE_H

#include "types.h"
#include "tensor.h"
#include "graph.h"
#include "allocator.h"
#include "profiler.h"
#include "exec_plan.h"

typedef struct FeRuntime FeRuntime;
struct FeRuntime {
    FeGraph    *graph;
    FeArena     weight_arena;
    FeArena     activation_arena;
    FeProfiler *profiler;
    FeExecPlan  exec;          /* cached kernel resolution + memory plan */
};

FeStatus fe_runtime_init(FeRuntime *rt, FeGraph *graph,
                          void *weight_buf,     size_t weight_size,
                          void *activation_buf, size_t activation_size);

FeStatus fe_runtime_alloc_weights(FeRuntime *rt);

/* Runs inference via the static execution plan (Stage 13): the first run
 * (or a run with a new input shape) re-plans memory and re-resolves kernels;
 * steady-state runs execute the cached flat plan. */
FeStatus fe_runtime_run(FeRuntime *rt,
                         const FeTensor *input,
                         FeTensor *output);

/* Runs `n` inferences of the current graph on `inputs`/`outputs` (one
 * sample per entry; all samples share the caller's graph input shape).
 * Each call is fe_runtime_run, so after the first sample the static plan is
 * cached and samples 2..n skip all re-analysis (plan rebuild happens only on
 * a shape change). Returns the first error encountered. */
FeStatus fe_runtime_run_batch(FeRuntime *rt,
                              FeTensor *const *inputs,
                              FeTensor *const *outputs,
                              int n);

/*
 * Calibrate activation ranges: run `n` samples of the caller's provided
 * shape through the runtime and record, for every non-weight float tensor,
 * the maximum |x| observed across the sample set. `ranges` must point to
 * `graph->n_tensors` floats and is fully written on FE_OK.
 *
 * This is the data-collection half of the quantization calibration pipeline:
 * the observed per-tensor ranges are what a static-activation-scale build
 * consumes (the engine today quantizes activations dynamically per run).
 */
FeStatus fe_runtime_calibrate(FeRuntime *rt,
                              FeTensor *const *inputs, int n,
                              FeTensor *output, float *ranges);

void fe_runtime_print_trace(const FeRuntime *rt);

#endif
