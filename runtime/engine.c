#include "engine.h"
#include "ops.h"
#include "profiler.h"
#include <string.h>
#include <stdio.h>
#include <assert.h>

static FeStatus dispatch_node(FeRuntime *rt, const FeNode *node) {
    FeGraph *g = rt->graph;

    #define IN(i)  (g->tensors[node->inputs [i]].tensor)
    #define OUT(i) (g->tensors[node->outputs[i]].tensor)

    uint64_t t0 = rt->profiler ? fe_profiler_now_ns() : 0;

    FeStatus s = FE_OK;
    switch (node->op) {
        case FE_OP_INPUT:
        case FE_OP_OUTPUT:
            s = FE_OK;
            break;
        case FE_OP_MATMUL:
            s = fe_matmul(IN(0), IN(1), OUT(0));
            break;
        case FE_OP_LINEAR:
            s = fe_linear(IN(0), IN(1), IN(2), OUT(0));
            break;
        case FE_OP_RELU:
            s = fe_relu(IN(0), OUT(0));
            break;
        case FE_OP_SOFTMAX:
            s = fe_softmax(IN(0), OUT(0));
            break;
        case FE_OP_ADD:
            s = fe_bias_add(IN(0), IN(1), OUT(0));
            break;
        case FE_OP_FLATTEN: {
            FeTensor *src = IN(0);
            FeTensor *dst = OUT(0);
            int total = fe_tensor_numel(src);
            int flat_shape[] = {src->shape[0], total / src->shape[0]};
            FeTensor *view = fe_tensor_reshape(src, 2, flat_shape);
            if (!view) { s = FE_ERR_SHAPE; break; }
            fe_tensor_copy(dst, view);
            fe_tensor_free(view);
            break;
        }
        default:
            fprintf(stderr, "Engine: unimplemented op %d\n", node->op);
            s = FE_ERR_SHAPE;
    }

    if (rt->profiler && t0) {
        uint64_t elapsed = fe_profiler_now_ns() - t0;
        fe_profiler_record(rt->profiler, node->name, elapsed);
    }

    #undef IN
    #undef OUT
    return s;
}

FeStatus fe_runtime_init(FeRuntime *rt, FeGraph *graph,
                          void *weight_buf,     size_t weight_size,
                          void *activation_buf, size_t activation_size) {
    if (!rt || !graph || !weight_buf || !activation_buf) return FE_ERR_NULL;

    rt->graph    = graph;
    rt->profiler = NULL;

    FeStatus s;
    s = fe_arena_init(&rt->weight_arena,     weight_buf,     weight_size);
    if (s != FE_OK) return s;
    s = fe_arena_init(&rt->activation_arena, activation_buf, activation_size);
    if (s != FE_OK) return s;

    if (!graph->topo_valid) {
        s = fe_graph_topo_sort(graph);
        if (s != FE_OK) return s;
    }
    return FE_OK;
}

FeStatus fe_runtime_alloc_weights(FeRuntime *rt) {
    FeGraph *g = rt->graph;
    for (int i = 0; i < g->n_tensors; i++) {
        FeTensorEntry *e = &g->tensors[i];
        if (!e->is_weight) continue;
        e->tensor = fe_arena_alloc_tensor(&rt->weight_arena,
                                           e->dtype, e->ndim, e->shape);
        if (!e->tensor) return FE_ERR_NOMEM;
    }
    return FE_OK;
}

static FeStatus alloc_activations(FeRuntime *rt) {
    FeGraph *g = rt->graph;
    for (int i = 0; i < g->n_tensors; i++) {
        FeTensorEntry *e = &g->tensors[i];
        if (e->is_weight)    continue;
        if (e->tensor != NULL) continue;
        e->tensor = fe_arena_alloc_tensor(&rt->activation_arena,
                                           e->dtype, e->ndim, e->shape);
        if (!e->tensor) return FE_ERR_NOMEM;
    }
    return FE_OK;
}

FeStatus fe_runtime_run(FeRuntime *rt,
                         const FeTensor *input,
                         FeTensor *output) {
    if (!rt || !input || !output) return FE_ERR_NULL;

    FeGraph *g = rt->graph;
    fe_arena_reset(&rt->activation_arena);

    for (int i = 0; i < g->n_tensors; i++) {
        FeTensorEntry *e = &g->tensors[i];
        if (!e->is_weight) e->tensor = NULL;
    }

    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i].op == FE_OP_INPUT) {
            int out_idx = g->nodes[i].outputs[0];
            g->tensors[out_idx].tensor = (FeTensor *)input;
            break;
        }
    }

    FeStatus s = alloc_activations(rt);
    if (s != FE_OK) return s;

    for (int i = 0; i < g->n_nodes; i++) {
        int node_idx = g->topo_order[i];
        s = dispatch_node(rt, &g->nodes[node_idx]);
        if (s != FE_OK) {
            fprintf(stderr, "Engine: node '%s' failed with %d\n",
                    g->nodes[node_idx].name, s);
            return s;
        }
    }

    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i].op == FE_OP_OUTPUT) {
            int in_idx = g->nodes[i].inputs[0];
            return fe_tensor_copy(output, g->tensors[in_idx].tensor);
        }
    }

    FeNode *last = &g->nodes[g->topo_order[g->n_nodes - 1]];
    if (last->n_outputs > 0) {
        int out_idx = last->outputs[0];
        return fe_tensor_copy(output, g->tensors[out_idx].tensor);
    }

    return FE_OK;
}

void fe_runtime_print_trace(const FeRuntime *rt) {
    FeGraph *g = rt->graph;
    printf("Execution trace (%d nodes):\n", g->n_nodes);
    for (int i = 0; i < g->n_nodes; i++) {
        int idx = g->topo_order[i];
        printf("  step %d: %s\n", i, g->nodes[idx].name);
    }
}