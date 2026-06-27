// graph/graph.c
#include "graph.h"
#include <string.h>
#include <stdio.h>
#include <assert.h>

void fe_graph_init(FeGraph *g) {
    memset(g, 0, sizeof(FeGraph));
}

int fe_graph_add_tensor(FeGraph *g, const char *name,
                         FeDtype dtype, int ndim, const int *shape,
                         int is_weight) {
    if (g->n_tensors >= FE_MAX_TENSORS) return -1;

    FeTensorEntry *e = &g->tensors[g->n_tensors];
    strncpy(e->name, name, FE_NAME_LEN - 1);
    e->dtype     = dtype;
    e->ndim      = ndim;
    e->is_weight = is_weight;
    e->tensor    = NULL;
    memcpy(e->shape, shape, ndim * sizeof(int));

    return g->n_tensors++;
}

int fe_graph_add_node(FeGraph *g, const char *name, FeOpType op,
                       const int *inputs,  int n_inputs,
                       const int *outputs, int n_outputs) {
    if (g->n_nodes >= FE_MAX_NODES) return -1;
    assert(n_inputs  <= FE_MAX_NODE_INPUTS);
    assert(n_outputs <= FE_MAX_NODE_OUTPUTS);

    FeNode *node = &g->nodes[g->n_nodes];
    strncpy(node->name, name, FE_NAME_LEN - 1);
    node->op        = op;
    node->n_inputs  = n_inputs;
    node->n_outputs = n_outputs;
    memcpy(node->inputs,  inputs,  n_inputs  * sizeof(int));
    memcpy(node->outputs, outputs, n_outputs * sizeof(int));

    g->topo_valid = 0;  /* invalidate any previous sort */
    return g->n_nodes++;
}

/*
 * Kahn's algorithm for topological sort.
 *
 * How it works:
 * 1. Count in-degree of every node (how many inputs are produced
 *    by other nodes, not by weights/inputs).
 * 2. Start with all nodes whose in-degree is 0 (no dependencies).
 * 3. Process each zero-in-degree node: add to order, decrement
 *    in-degree of nodes that consume its outputs.
 * 4. If all nodes processed: valid DAG. If not: cycle detected.
 *
 * This is O(N + E) where N = nodes, E = edges (tensor connections).
 */
FeStatus fe_graph_topo_sort(FeGraph *g) {
    int in_degree[FE_MAX_NODES] = {0};

    /*
     * Build a map: tensor_index -> which node produces it.
     * producer[t] = node index that outputs tensor t, or -1.
     */
    int producer[FE_MAX_TENSORS];
    memset(producer, -1, sizeof(producer));

    for (int i = 0; i < g->n_nodes; i++) {
        FeNode *n = &g->nodes[i];
        for (int o = 0; o < n->n_outputs; o++) {
            producer[n->outputs[o]] = i;
        }
    }

    /* Compute in-degree: count inputs that come from another node */
    for (int i = 0; i < g->n_nodes; i++) {
        FeNode *n = &g->nodes[i];
        for (int inp = 0; inp < n->n_inputs; inp++) {
            int tensor_idx = n->inputs[inp];
            if (producer[tensor_idx] != -1) {
                in_degree[i]++;
            }
        }
    }

    /* Queue of nodes with in-degree 0 */
    int queue[FE_MAX_NODES];
    int q_head = 0, q_tail = 0;
    for (int i = 0; i < g->n_nodes; i++) {
        if (in_degree[i] == 0) queue[q_tail++] = i;
    }

    int order_idx = 0;
    while (q_head < q_tail) {
        int node_idx = queue[q_head++];
        g->topo_order[order_idx++] = node_idx;

        /* For each output tensor of this node, find consuming nodes */
        FeNode *n = &g->nodes[node_idx];
        for (int o = 0; o < n->n_outputs; o++) {
            int out_tensor = n->outputs[o];
            for (int j = 0; j < g->n_nodes; j++) {
                FeNode *consumer = &g->nodes[j];
                for (int inp = 0; inp < consumer->n_inputs; inp++) {
                    if (consumer->inputs[inp] == out_tensor) {
                        if (--in_degree[j] == 0) {
                            queue[q_tail++] = j;
                        }
                    }
                }
            }
        }
    }

    if (order_idx != g->n_nodes) return FE_ERR_SHAPE; /* cycle */

    g->topo_valid = 1;
    return FE_OK;
}

static const char *op_name(FeOpType op) {
    switch (op) {
        case FE_OP_INPUT:     return "Input";
        case FE_OP_OUTPUT:    return "Output";
        case FE_OP_MATMUL:    return "MatMul";
        case FE_OP_LINEAR:    return "Linear";
        case FE_OP_RELU:      return "ReLU";
        case FE_OP_SOFTMAX:   return "Softmax";
        case FE_OP_CONV1D:    return "Conv1D";
        case FE_OP_BATCHNORM: return "BatchNorm";
        case FE_OP_ADD:       return "Add";
        case FE_OP_FLATTEN:   return "Flatten";
        default:              return "Unknown";
    }
}

void fe_graph_print(const FeGraph *g) {
    printf("FeGraph: %d nodes, %d tensors\n", g->n_nodes, g->n_tensors);
    printf("Execution order:\n");
    for (int i = 0; i < g->n_nodes; i++) {
        int idx = g->topo_valid ? g->topo_order[i] : i;
        const FeNode *n = &g->nodes[idx];
        printf("  [%d] %s (%s)  inputs:[", i, n->name, op_name(n->op));
        for (int j = 0; j < n->n_inputs;  j++) printf("%d%s", n->inputs[j],  j<n->n_inputs-1  ? "," : "");
        printf("] outputs:[");
        for (int j = 0; j < n->n_outputs; j++) printf("%d%s", n->outputs[j], j<n->n_outputs-1 ? "," : "");
        printf("]\n");
    }
}