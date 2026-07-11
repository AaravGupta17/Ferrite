#ifndef FERRITE_ENGINE_H
#define FERRITE_ENGINE_H

#include "types.h"
#include "tensor.h"
#include "graph.h"
#include "allocator.h"
#include "profiler.h"

typedef struct {
    FeGraph    *graph;
    FeArena     weight_arena;
    FeArena     activation_arena;
    FeProfiler *profiler;
} FeRuntime;

FeStatus fe_runtime_init(FeRuntime *rt, FeGraph *graph,
                          void *weight_buf,     size_t weight_size,
                          void *activation_buf, size_t activation_size);

FeStatus fe_runtime_alloc_weights(FeRuntime *rt);

FeStatus fe_runtime_run(FeRuntime *rt,
                         const FeTensor *input,
                         FeTensor *output);

void fe_runtime_print_trace(const FeRuntime *rt);

#endif
