// graph/graph.h
#ifndef FERRITE_GRAPH_H
#define FERRITE_GRAPH_H

#include "types.h"
#include "tensor.h"
#include <stddef.h>

#define FE_MAX_NODE_INPUTS  8
#define FE_MAX_NODE_OUTPUTS 4
#define FE_MAX_NODES        512
#define FE_MAX_TENSORS      1024
#define FE_NAME_LEN         64

/*
 * Operator types supported by the graph IR.
 * Every ONNX op we care about maps to one of these.
 */
typedef enum {
    FE_OP_INPUT      = 0,
    FE_OP_OUTPUT     = 1,
    FE_OP_MATMUL     = 2,
    FE_OP_LINEAR     = 3,
    FE_OP_RELU       = 4,
    FE_OP_SOFTMAX    = 5,
    FE_OP_CONV1D     = 6,
    FE_OP_BATCHNORM  = 7,
    FE_OP_ADD        = 8,
    FE_OP_FLATTEN    = 9,

    /* Stage 3 — basic math */
    FE_OP_SUB        = 10,
    FE_OP_MUL        = 11,
    FE_OP_DIV        = 12,
    FE_OP_NEG        = 13,
    FE_OP_EXP        = 14,
    FE_OP_LOG        = 15,
    FE_OP_POW        = 16,

    /* Stage 3 — activations */
    FE_OP_SIGMOID    = 17,
    FE_OP_TANH       = 18,
    FE_OP_GELU       = 19,
    FE_OP_LEAKY_RELU = 20,
    FE_OP_ELU        = 21,
    FE_OP_SWISH      = 22,

    /* Stage 3 — linear algebra */
    FE_OP_GEMM       = 23,
    FE_OP_TRANSPOSE  = 24,

    /* Stage 3 — CNN */
    FE_OP_CONV2D     = 25,
    FE_OP_MAXPOOL    = 26,
    FE_OP_AVGPOOL    = 27,
    FE_OP_LAYERNORM  = 28,
    FE_OP_GROUPNORM  = 29,

    /* Stage 3 — sequence */
    FE_OP_ATTENTION         = 30,
    FE_OP_MULTIHEAD_ATTN    = 31,
    FE_OP_EMBEDDING         = 32,
    FE_OP_POSITIONAL_ENCOD  = 33,
} FeOpType;

/*
 * A tensor entry in the graph's tensor registry.
 * At graph-build time, only shape and dtype are known.
 * The actual data pointer is assigned by the memory planner.
 */
typedef struct {
    char     name[FE_NAME_LEN];
    int      shape[FERRITE_MAX_DIMS];
    int      ndim;
    FeDtype  dtype;
    FeTensor *tensor;   /* NULL until memory planner runs */
    int      is_weight; /* 1 if this tensor holds model weights */
} FeTensorEntry;

/*
 * A node in the computation graph.
 * Inputs and outputs are indices into the graph's tensor registry.
 */
typedef struct {
    char     name[FE_NAME_LEN];
    FeOpType op;
    int      inputs [FE_MAX_NODE_INPUTS];   /* tensor indices */
    int      outputs[FE_MAX_NODE_OUTPUTS];  /* tensor indices */
    int      n_inputs;
    int      n_outputs;

    /* Op-specific attributes */
    union {
        struct { int axis; }                 softmax;
        struct { int stride; int pad; }      conv1d;
        struct { float negative_slope; }     leaky_relu;
        struct { float alpha; }              elu;
        struct { int stride_h, stride_w, pad_h, pad_w; } conv2d;
        struct { int kh, kw, sh, sw; }       pool;
        struct { float eps; }                layernorm;
        struct { int groups; float eps; }    groupnorm;
        struct { float eps; }                batchnorm;
        struct { int transA, transB; float alpha, beta; } gemm;
        struct { int num_heads; }            multihead;
    } attrs;
} FeNode;

/*
 * The computation graph.
 * Owns all nodes and tensor metadata.
 * Does not own tensor data buffers — those belong to arenas.
 */
typedef struct {
    FeNode        nodes  [FE_MAX_NODES];
    FeTensorEntry tensors[FE_MAX_TENSORS];
    int           n_nodes;
    int           n_tensors;

    /* Topological order — populated by fe_graph_topo_sort */
    int           topo_order[FE_MAX_NODES];
    int           topo_valid;
} FeGraph;

/* --- Lifecycle --- */
void     fe_graph_init(FeGraph *g);

/* --- Building the graph --- */

/*
 * Register a tensor with the graph. Returns its index, or -1 on failure.
 * Shape may be {0} if unknown at build time (inferred later).
 */
int fe_graph_add_tensor(FeGraph *g, const char *name,
                         FeDtype dtype, int ndim, const int *shape,
                         int is_weight);

/*
 * Add a node to the graph. Returns its index, or -1 on failure.
 * inputs/outputs are tensor indices from fe_graph_add_tensor.
 */
int fe_graph_add_node(FeGraph *g, const char *name, FeOpType op,
                       const int *inputs,  int n_inputs,
                       const int *outputs, int n_outputs);

/* --- Analysis --- */

/*
 * Compute topological order via Kahn's algorithm.
 * Result stored in g->topo_order. Must be called before execution.
 * Returns FE_OK, or FE_ERR_SHAPE if graph has a cycle.
 */
FeStatus fe_graph_topo_sort(FeGraph *g);

/* Print graph structure for debugging. */
void     fe_graph_print(const FeGraph *g);

#endif // FERRITE_GRAPH_H