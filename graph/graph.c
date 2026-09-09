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
    e->scales    = NULL;
    e->n_scales  = 0;
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
    if (n_inputs  > 0 && inputs)  memcpy(node->inputs,  inputs,  n_inputs  * sizeof(int));
if (n_outputs > 0 && outputs) memcpy(node->outputs, outputs, n_outputs * sizeof(int));

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
        case FE_OP_SUB:       return "Sub";
        case FE_OP_MUL:       return "Mul";
        case FE_OP_DIV:       return "Div";
        case FE_OP_NEG:       return "Neg";
        case FE_OP_EXP:       return "Exp";
        case FE_OP_LOG:       return "Log";
        case FE_OP_POW:       return "Pow";
        case FE_OP_SIGMOID:   return "Sigmoid";
        case FE_OP_TANH:      return "Tanh";
        case FE_OP_GELU:      return "Gelu";
        case FE_OP_LEAKY_RELU:return "LeakyRelu";
        case FE_OP_ELU:       return "Elu";
        case FE_OP_SWISH:     return "Swish";
        case FE_OP_GEMM:      return "Gemm";
        case FE_OP_TRANSPOSE: return "Transpose";
        case FE_OP_CONV2D:    return "Conv2D";
        case FE_OP_MAXPOOL:   return "MaxPool";
        case FE_OP_AVGPOOL:   return "AvgPool";
        case FE_OP_LAYERNORM: return "LayerNorm";
        case FE_OP_GROUPNORM: return "GroupNorm";
        case FE_OP_ATTENTION: return "Attention";
        case FE_OP_MULTIHEAD_ATTN: return "MHA";
        case FE_OP_EMBEDDING: return "Embedding";
        case FE_OP_POSITIONAL_ENCOD: return "PosEnc";
        default:              return "Unknown";
    }
}

FeStatus fe_graph_validate(const FeGraph *g) {
    if (!g) return FE_ERR_NULL;

    /* --- Tensor registry: every entry must be well-formed --- */
    for (int t = 0; t < g->n_tensors; t++) {
        const FeTensorEntry *e = &g->tensors[t];

        if (e->dtype != DTYPE_FLOAT32 && e->dtype != DTYPE_INT8 &&
            e->dtype != DTYPE_INT32   && e->dtype != DTYPE_FLOAT64) {
            fprintf(stderr, "fe_graph_validate: tensor %d (%s) has unrecognized dtype %d\n",
                    t, e->name, e->dtype);
            return FE_ERR_DTYPE;
        }
        if (e->ndim < 0 || e->ndim > FERRITE_MAX_DIMS) {
            fprintf(stderr, "fe_graph_validate: tensor %d (%s) has invalid ndim %d\n",
                    t, e->name, e->ndim);
            return FE_ERR_SHAPE;
        }
        for (int d = 0; d < e->ndim; d++) {
            if (e->shape[d] < 0) {
                fprintf(stderr, "fe_graph_validate: tensor %d (%s) has negative extent shape[%d]=%d\n",
                        t, e->name, d, e->shape[d]);
                return FE_ERR_SHAPE;
            }
        }
    }

    /* --- Node edges: every input/output index must resolve into the registry --- */
    for (int i = 0; i < g->n_nodes; i++) {
        const FeNode *n = &g->nodes[i];
        for (int k = 0; k < n->n_inputs; k++) {
            if (n->inputs[k] < 0 || n->inputs[k] >= g->n_tensors) {
                fprintf(stderr, "fe_graph_validate: node %d (%s) input %d is out of range (%d)\n",
                        i, n->name, k, n->inputs[k]);
                return FE_ERR_BOUNDS;
            }
        }
        for (int k = 0; k < n->n_outputs; k++) {
            if (n->outputs[k] < 0 || n->outputs[k] >= g->n_tensors) {
                fprintf(stderr, "fe_graph_validate: node %d (%s) output %d is out of range (%d)\n",
                        i, n->name, k, n->outputs[k]);
                return FE_ERR_BOUNDS;
            }
        }
    }

    /* --- Producer uniqueness: a tensor may be written by at most one node --- */
    int producer_count[FE_MAX_TENSORS] = {0};
    for (int i = 0; i < g->n_nodes; i++) {
        const FeNode *n = &g->nodes[i];
        for (int k = 0; k < n->n_outputs; k++) {
            producer_count[n->outputs[k]]++;
        }
    }
    for (int t = 0; t < g->n_tensors; t++) {
        if (producer_count[t] > 1) {
            fprintf(stderr, "fe_graph_validate: tensor %d (%s) has %d producers (must be at most 1)\n",
                    t, g->tensors[t].name, producer_count[t]);
            return FE_ERR_SHAPE;
        }
    }

    return FE_OK;
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