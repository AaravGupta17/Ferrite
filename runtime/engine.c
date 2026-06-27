// runtime/engine.c
#include "engine.h"
#include "ops.h"
#include <string.h>
#include <stdio.h>
#include <assert.h>

FeStatus fe_runtime_init(FeRuntime *rt, FeGraph *graph,
                          void *weight_buf,     size_t weight_size,
                          void *activation_buf, size_t activation_size) {
    if (!rt || !graph || !weight_buf || !activation_buf) return FE_ERR_NULL;

    rt->graph = graph;
    FeStatus s;
    s = fe_arena_init(&rt->weight_arena,     weight_buf,     weight_size);
    if (s != FE_OK) return s;
    s = fe_arena_init(&rt->activation_arena, activation_buf, activation_size);
    if (s != FE_OK) return s;

    /* Graph must have a valid topological order */
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

/*
 * Allocate all activation tensors from the activation arena.
 * Called at the start of each inference run after arena reset.
 */
static FeStatus alloc_activations(FeRuntime *rt) {
    FeGraph *g = rt->graph;

    for (int i = 0; i < g->n_tensors; i++) {
        FeTensorEntry *e = &g->tensors[i];
        if (e->is_weight) continue;   /* weights already allocated */
        if (e->tensor != NULL) continue; /* input tensor set by caller */

        e->tensor = fe_arena_alloc_tensor(&rt->activation_arena,
                                           e->dtype, e->ndim, e->shape);
        if (!e->tensor) return FE_ERR_NOMEM;
    }
    return FE_OK;
}

/*
 * Dispatch a single node — call the correct kernel with the right tensors.
 */
static FeStatus dispatch_node(FeRuntime *rt, const FeNode *node) {
    FeGraph  *g  = rt->graph;

    /* Convenience macro: get tensor pointer for input/output index */
    #define IN(i)  (g->tensors[node->inputs [i]].tensor)
    #define OUT(i) (g->tensors[node->outputs[i]].tensor)

    switch (node->op) {
        case FE_OP_INPUT:
        case FE_OP_OUTPUT:
            return FE_OK;   /* handled externally */

        case FE_OP_MATMUL:
            return fe_matmul(IN(0), IN(1), OUT(0));

        case FE_OP_LINEAR:
            return fe_linear(IN(0), IN(1), IN(2), OUT(0));

        case FE_OP_RELU:
            return fe_relu(IN(0), OUT(0));

        case FE_OP_SOFTMAX:
            return fe_softmax(IN(0), OUT(0));

        case FE_OP_ADD:
            return fe_bias_add(IN(0), IN(1), OUT(0));

        case FE_OP_FLATTEN: {
            /* Reshape to [batch, -1] — no data copy if contiguous */
            FeTensor *src = IN(0);
            FeTensor *dst = OUT(0);
            int total = fe_tensor_numel(src);
            int flat_shape[] = {src->shape[0], total / src->shape[0]};
            FeTensor *view = fe_tensor_reshape(src, 2, flat_shape);
            if (!view) return FE_ERR_SHAPE;
            fe_tensor_copy(dst, view);
            fe_tensor_free(view);
            return FE_OK;
        }

        default:
            fprintf(stderr, "Engine: unimplemented op %d\n", node->op);
            return FE_ERR_SHAPE;
    }

    #undef IN
    #undef OUT
}

FeStatus fe_runtime_run(FeRuntime *rt,
                         const FeTensor *input,
                         FeTensor *output) {
    if (!rt || !input || !output) return FE_ERR_NULL;

    FeGraph *g = rt->graph;

    /* Step 1: reset activation arena — free all previous activations */
    fe_arena_reset(&rt->activation_arena);

    /* Step 2: find and wire input tensor */
    for (int i = 0; i < g->n_tensors; i++) {
        FeTensorEntry *e = &g->tensors[i];
        if (!e->is_weight) e->tensor = NULL;  /* clear stale pointers */
    }

    /* Find the graph input node and point its tensor at caller's input */
    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i].op == FE_OP_INPUT) {
            int out_idx = g->nodes[i].outputs[0];
            g->tensors[out_idx].tensor = (FeTensor *)input; /* read-only use */
            break;
        }
    }

    /* Step 3: allocate all activation tensors */
    FeStatus s = alloc_activations(rt);
    if (s != FE_OK) return s;

    /* Step 4: execute in topological order */
    for (int i = 0; i < g->n_nodes; i++) {
        int node_idx = g->topo_order[i];
        s = dispatch_node(rt, &g->nodes[node_idx]);
        if (s != FE_OK) {
            fprintf(stderr, "Engine: node '%s' failed with %d\n",
                    g->nodes[node_idx].name, s);
            return s;
        }
    }

    /* Step 5: copy result to caller's output tensor */
    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i].op == FE_OP_OUTPUT) {
            int in_idx = g->nodes[i].inputs[0];
            return fe_tensor_copy(output, g->tensors[in_idx].tensor);
        }
    }

    /* No explicit output node — copy last tensor produced */
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