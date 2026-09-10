// runtime/model_ser.h — FEMD model artifact save/load (Section 2.2).
//
// ferrite-compile serializes a fully-planned model to one self-contained
// artifact (magic "FEMD"): per-tensor registry entries (shape, dtype,
// is_weight, per-channel scales, static act_scale) with each weight's bytes
// inline, nodes in topological order (op + tensor indices + attrs), and the
// resolved memory plan (arena total + per-tensor offsets). The loader
// (femd_load) reconstructs a FeGraph + FePlan and resolves every node to a
// kernel through fe_kernels_get — no protobuf, no shape inference, no
// optimizer reach the device.
//
// The reader is deliberately strict: little-endian fixed-width fields,
// magic + version check, every count bounds-checked against config.h
// ceilings, and a trailing-byte check (nothing may follow the payload).
// This is a no-sandbox device input, so it is at least as defensive as the
// ONNX parser.
#ifndef FERRITE_MODEL_SER_H
#define FERRITE_MODEL_SER_H

#include "types.h"
#include "graph.h"
#include "allocator.h"
#include "memory_planner.h"
#include "exec_plan.h"
#include <stdio.h>

#define FERRITE_MODEL_MAGIC     "FEMD"
#define FERRITE_MODEL_VERSION   1u

/* A femd_load'd model: self-contained graph metadata, cached plan, and the
 * pre-resolved flat step array (kernel fns, not op codes). */
typedef struct {
    FeGraph    graph;
    FePlan     plan;
    int        n_steps;       /* == graph.n_nodes */
    FeExecStep steps[FE_MAX_NODES];
    int        in_tidx;       /* graph input tensor index  */
    int        out_tidx;      /* graph output tensor index */
    int        in_ndim;       /* the input shape the artifact was planned
                               * for; a different shape is rejected loudly */
    int        in_shape[FERRITE_MAX_DIMS];
} FeModel;

/*
 * Serialize a fully-planned model to `f`. The graph must already be
 * topologically sorted and memory-planned (fe_onnx_load + fe_optimize +
 * fe_plan_memory), with weight data bound to the weight tensors
 * (fe_runtime_alloc_weights); `in_tidx`/`out_tidx` are the graph
 * input/output tensor indices. Weights are serialized inline per tensor.
 */
FeStatus fe_model_save(FILE *f, const FeGraph *g, const FePlan *plan,
                       int in_tidx, int out_tidx);

/*
 * Parse a FEMD artifact. Weight bytes are arena-allocated from
 * `weight_arena`; activation data is not read (the device supplies its own
 * buffer at run time and fe_plan_apply assigns offsets). Every node is
 * resolved to a kernel wrapper; an op without a kernel fails loudly.
 */
FeStatus fe_model_load(FILE *f, FeArena *weight_arena, FeModel *m);

/*
 * Execute one inference from the in-memory model. Mirrors the runtime's
 * static-plan loop but performs zero shape analysis: an input whose shape
 * differs from the artifact's planned input shape is rejected (the artifact
 * is fixed-shape by construction). `weight_arena`/`activation_arena` are
 * the caller's arenas (weights on device live in RAM or flash-mapped
 * memory; activations must be >= plan.total_activation_bytes plus the
 * FeTensor metadata overhead fe_plan_apply needs).
 */
FeStatus fe_model_run(FeModel *m, FeArena *weight_arena,
                      FeArena *activation_arena,
                      const FeTensor *input, FeTensor *output);

#endif /* FERRITE_MODEL_SER_H */