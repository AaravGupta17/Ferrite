/* runtime/exec_plan.c — static execution plan (Stage 13). */
#include "engine.h"
#include "exec_plan.h"
#include "ops.h"
#include "quant.h"
#include "profiler.h"
#include "memory_planner.h"
#include "shape_infer.h"
#include "log.h"
#include <string.h>
#include <stdio.h>

/*
 * Kernels are resolved to function pointers at plan-build time via a table
 * indexed by FeOpType; the run loop is a flat array walk with no switch.
 * Only the profiler timing depends on the node name, and only mgmt (plan,
 * input binding) reads the graph during a run.
 */

#define EXEC_BEGIN() uint64_t t0 = rt->profiler ? fe_profiler_now_ns() : 0
#define EXEC_END()                                              \
    do {                                                        \
        if (rt->profiler && t0)                                 \
            fe_profiler_record(rt->profiler, node->name,        \
                               fe_profiler_now_ns() - t0);      \
    } while (0)

#define IN(i)  (g->tensors[node->inputs [i]].tensor)
#define OUT(i) (g->tensors[node->outputs[i]].tensor)

static FeStatus ex_noop(FeRuntime *rt, FeNode *node) {
    (void)rt; (void)node;
    return FE_OK;
}

static FeStatus ex_matmul(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    /* A per-channel-quantized INT8 weight (fe_quantize_model) routes to the
     * engine-path INT8 kernel; everything else stays float. */
    FeTensorEntry *we  = &g->tensors[node->inputs[1]];
    FeTensor       *C   = OUT(0);
    FeStatus s;
    if (we->tensor && we->tensor->dtype == DTYPE_INT8) {
        if (!we->scales) {
            fe_log_error("Exec plan: INT8 weight '%s' has no recorded scales",
                         we->name);
            EXEC_END();
            return FE_ERR_SHAPE;
        }
        s = fe_matmul_int8_dyn(IN(0), we->tensor, we->scales, C);
    } else {
        s = fe_matmul(IN(0), IN(1), C);
    }
    EXEC_END();
    return s;
}

static FeStatus ex_linear(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeTensorEntry *we  = &g->tensors[node->inputs[1]];
    FeTensor       *C   = OUT(0);
    FeStatus s;
    if (we->tensor && we->tensor->dtype == DTYPE_INT8) {
        if (!we->scales) {
            fe_log_error("Exec plan: INT8 weight '%s' has no recorded scales",
                         we->name);
            EXEC_END();
            return FE_ERR_SHAPE;
        }
        s = fe_linear_int8(IN(0), we->tensor, we->scales, IN(2), C);
    } else {
        s = fe_linear(IN(0), IN(1), IN(2), C);
    }
    EXEC_END();
    return s;
}

static FeStatus ex_relu(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_relu(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_softmax(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_softmax(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_bias_add(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_bias_add(IN(0), IN(1), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_conv1d(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    int stride = node->attrs.conv1d.stride > 0 ? node->attrs.conv1d.stride : 1;
    FeStatus s = fe_conv1d(IN(0), IN(1), node->n_inputs > 2 ? IN(2) : NULL,
                           OUT(0), stride, node->attrs.conv1d.pad);
    EXEC_END();
    return s;
}

static FeStatus ex_flatten(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeTensor *src = IN(0);
    int total = fe_tensor_numel(src);
    int flat_shape[] = { (src->ndim > 0 && src->shape[0]) ? src->shape[0] : 1,
                         total / ((src->ndim > 0 && src->shape[0]) ? src->shape[0] : 1) };
    FeTensor *view = fe_tensor_reshape(src, 2, flat_shape);
    FeStatus s = view ? fe_tensor_copy(OUT(0), view) : FE_ERR_SHAPE;
    if (view) fe_tensor_free(view);
    EXEC_END();
    return s;
}

static FeStatus ex_sub(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_sub(IN(0), IN(1), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_mul(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_mul(IN(0), IN(1), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_div(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_div(IN(0), IN(1), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_neg(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_neg(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_exp(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_exp(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_log(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_ln(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_pow(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_pow(IN(0), IN(1), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_sigmoid(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_sigmoid(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_tanh(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_tanh(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_gelu(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_gelu(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_leaky_relu(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_leaky_relu(IN(0), OUT(0),
                               node->attrs.leaky_relu.negative_slope);
    EXEC_END();
    return s;
}

static FeStatus ex_elu(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_elu(IN(0), OUT(0), node->attrs.elu.alpha);
    EXEC_END();
    return s;
}

static FeStatus ex_swish(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_swish(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_gemm(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_gemm(IN(0), node->attrs.gemm.transA,
                         IN(1), node->attrs.gemm.transB,
                         OUT(0), node->attrs.gemm.alpha,
                         node->attrs.gemm.beta);
    EXEC_END();
    return s;
}

static FeStatus ex_transpose(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_transpose(IN(0), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_conv2d(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_conv2d(IN(0), IN(1), node->n_inputs > 2 ? IN(2) : NULL,
                           OUT(0),
                           node->attrs.conv2d.stride_h, node->attrs.conv2d.stride_w,
                           node->attrs.conv2d.pad_h,    node->attrs.conv2d.pad_w);
    EXEC_END();
    return s;
}

static FeStatus ex_maxpool(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_maxpool(IN(0), OUT(0), node->attrs.pool.kh,
                            node->attrs.pool.kw, node->attrs.pool.sh,
                            node->attrs.pool.sw);
    EXEC_END();
    return s;
}

static FeStatus ex_avgpool(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_avgpool(IN(0), OUT(0), node->attrs.pool.kh,
                            node->attrs.pool.kw, node->attrs.pool.sh,
                            node->attrs.pool.sw);
    EXEC_END();
    return s;
}

static FeStatus ex_batchnorm(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_batchnorm(IN(0), IN(1), IN(2), IN(3), IN(4), OUT(0),
                              node->attrs.batchnorm.eps);
    EXEC_END();
    return s;
}

static FeStatus ex_layernorm(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_layernorm(IN(0), IN(1), IN(2), OUT(0),
                              node->attrs.layernorm.eps);
    EXEC_END();
    return s;
}

static FeStatus ex_groupnorm(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_groupnorm(IN(0), IN(1), IN(2), OUT(0),
                              node->attrs.groupnorm.groups,
                              node->attrs.groupnorm.eps);
    EXEC_END();
    return s;
}

static FeStatus ex_attention(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_attention(IN(0), IN(1), IN(2), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_multihead(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_multihead_attention(IN(0), IN(1), IN(2), IN(3), IN(4),
                                        OUT(0), node->attrs.multihead.num_heads);
    EXEC_END();
    return s;
}

static FeStatus ex_embedding(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_embedding(IN(0), IN(1), OUT(0));
    EXEC_END();
    return s;
}

static FeStatus ex_posenc(FeRuntime *rt, FeNode *node) {
    FeGraph *g = rt->graph;
    EXEC_BEGIN();
    FeStatus s = fe_positional_encoding(IN(0), OUT(0));
    EXEC_END();
    return s;
}

#undef IN
#undef OUT

/* Resolver: FeOpType -> kernel wrapper. Indexed by the enum value so the
 * plan build is a table load, not a scan. Unlisted ops map to NULL. */
static FeExecFn const k_fns[] = {
    [FE_OP_INPUT]              = ex_noop,
    [FE_OP_OUTPUT]             = ex_noop,
    [FE_OP_MATMUL]             = ex_matmul,
    [FE_OP_LINEAR]             = ex_linear,
    [FE_OP_RELU]               = ex_relu,
    [FE_OP_SOFTMAX]            = ex_softmax,
    [FE_OP_CONV1D]             = ex_conv1d,
    [FE_OP_BATCHNORM]          = ex_batchnorm,
    [FE_OP_ADD]                = ex_bias_add,
    [FE_OP_FLATTEN]            = ex_flatten,
    [FE_OP_SUB]                = ex_sub,
    [FE_OP_MUL]                = ex_mul,
    [FE_OP_DIV]                = ex_div,
    [FE_OP_NEG]                = ex_neg,
    [FE_OP_EXP]                = ex_exp,
    [FE_OP_LOG]                = ex_log,
    [FE_OP_POW]                = ex_pow,
    [FE_OP_SIGMOID]            = ex_sigmoid,
    [FE_OP_TANH]               = ex_tanh,
    [FE_OP_GELU]               = ex_gelu,
    [FE_OP_LEAKY_RELU]         = ex_leaky_relu,
    [FE_OP_ELU]                = ex_elu,
    [FE_OP_SWISH]              = ex_swish,
    [FE_OP_GEMM]               = ex_gemm,
    [FE_OP_TRANSPOSE]          = ex_transpose,
    [FE_OP_CONV2D]             = ex_conv2d,
    [FE_OP_MAXPOOL]            = ex_maxpool,
    [FE_OP_AVGPOOL]            = ex_avgpool,
    [FE_OP_LAYERNORM]          = ex_layernorm,
    [FE_OP_GROUPNORM]          = ex_groupnorm,
    [FE_OP_ATTENTION]          = ex_attention,
    [FE_OP_MULTIHEAD_ATTN]     = ex_multihead,
    [FE_OP_EMBEDDING]          = ex_embedding,
    [FE_OP_POSITIONAL_ENCOD]   = ex_posenc,
};

/*
 * Locate the graph's input tensor. Hand-built graphs model the input with
 * an explicit FE_OP_INPUT node; ONNX-loaded graphs have no such node, so
 * the input is the first non-weight tensor that some node consumes but no
 * node produces. Returns -1 if there is none.
 */
static int find_graph_input(const FeGraph *g) {
    for (int i = 0; i < g->n_nodes; i++)
        if (g->nodes[i].op == FE_OP_INPUT)
            return g->nodes[i].outputs[0];

    for (int i = 0; i < g->n_tensors; i++) {
        const FeTensorEntry *e = &g->tensors[i];
        if (e->is_weight) continue;
        int consumed = 0, produced = 0;
        for (int n = 0; n < g->n_nodes && !consumed; n++)
            for (int k = 0; k < g->nodes[n].n_inputs; k++)
                if (g->nodes[n].inputs[k] == i) { consumed = 1; break; }
        for (int n = 0; n < g->n_nodes && !produced; n++)
            for (int k = 0; k < g->nodes[n].n_outputs; k++)
                if (g->nodes[n].outputs[k] == i) { produced = 1; break; }
        if (consumed && !produced) return i;
    }
    return -1;
}

/* Clear every produced (non-weight) tensor shape so shape inference
 * recomputes them from scratch; the graph input's own shape is saved first
 * and restored unchanged so a static (ONNX value_info) input keeps it. */
static void reset_produced_shapes(FeGraph *g, int in_tidx,
                                  int *save_ndim,
                                  int save_shape[FERRITE_MAX_DIMS]) {
    FeTensorEntry *ie = &g->tensors[in_tidx];
    *save_ndim = ie->ndim;
    for (int i = 0; i < ie->ndim; i++) save_shape[i] = ie->shape[i];

    for (int i = 0; i < g->n_tensors; i++) {
        FeTensorEntry *e = &g->tensors[i];
        if (e->is_weight) continue;
        e->ndim = 0;
        memset(e->shape, 0, sizeof(e->shape));
    }

    ie->ndim = *save_ndim;
    for (int i = 0; i < ie->ndim; i++) ie->shape[i] = save_shape[i];
}

/*
 * Infer output shapes for dynamic tensors. A dynamic graph input ({0,...})
 * is seeded from the caller's tensor shape; once identified as dynamic it is
 * re-seeded on every plan build (force_dynamic), because the first seed
 * overwrites the zero marker. Static inputs (ONNX value_info) are untouched.
 * Returns 1 when the graph input is dynamic.
 */
static int infer_shapes(FeRuntime *rt, const FeTensor *input,
                        int force_dynamic) {
    FeGraph *g = rt->graph;

    int in_tidx = find_graph_input(g);
    if (in_tidx >= 0) {
        FeTensorEntry *e = &g->tensors[in_tidx];
        int dynamic = !e->is_weight && e->ndim > 0 &&
                      (force_dynamic || e->shape[0] == 0);
        if (dynamic) {
            e->ndim = input->ndim;
            memcpy(e->shape, input->shape, input->ndim * sizeof(int));
        }
        fe_infer_shapes(g);
        return dynamic;
    }
    fe_infer_shapes(g);
    return 0;
}

/* Copy the final output into the caller's buffer. Order: OUTPUT-node input,
 * else the last planned step's output (the engine's documented behavior). */
static FeStatus copy_output(FeRuntime *rt, FeTensor *output) {
    FeGraph *g = rt->graph;
    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i].op == FE_OP_OUTPUT && g->nodes[i].n_inputs > 0) {
            int in_idx = g->nodes[i].inputs[0];
            if (!g->tensors[in_idx].tensor) return FE_ERR_SHAPE;
            return fe_tensor_copy(output, g->tensors[in_idx].tensor);
        }
    }
    FeNode *last = &g->nodes[g->topo_order[g->n_nodes - 1]];
    if (last->n_outputs > 0)
        return fe_tensor_copy(output, g->tensors[last->outputs[0]].tensor);
    return FE_ERR_SHAPE;
}

FeStatus fe_exec_build(FeExecPlan *ep, FeRuntime *rt) {
    if (!ep || !rt) return FE_ERR_NULL;
    FeGraph *g = rt->graph;
    if (!g->topo_valid) {
        FeStatus s = fe_graph_topo_sort(g);
        if (s != FE_OK) return s;
    }
    ep->n_steps = g->n_nodes;
    for (int i = 0; i < g->n_nodes; i++) {
        int idx = g->topo_order[i];
        FeExecFn fn = k_fns[g->nodes[idx].op];
        if (!fn) {
            fprintf(stderr, "Engine: unimplemented op %d\n", g->nodes[idx].op);
            return FE_ERR_SHAPE;
        }
        ep->steps[i].fn = fn;
        ep->steps[i].node_index = idx;
    }
    ep->steps_valid = 1;
    return FE_OK;
}

FeStatus fe_exec_run(FeExecPlan *ep, FeRuntime *rt,
                     const FeTensor *input, FeTensor *output) {
    if (!ep || !rt || !input || !output) return FE_ERR_NULL;
    FeGraph *g = rt->graph;
    FeStatus s;

    if (!ep->steps_valid) {
        s = fe_exec_build(ep, rt);
        if (s != FE_OK) return s;
        ep->plan_valid = 0;
    }

    /* Shape change detection: the one piece of per-run analysis. */
    int shape_changed = !ep->plan_valid ||
        ep->bound_input_ndim != input->ndim ||
        memcmp(ep->bound_input_shape, input->shape,
               input->ndim * sizeof(int)) != 0;

    if (shape_changed) {
        /* Shape inference only fills unknown shapes — it never shrinks an
         * already-known one. A dynamic-batch rebuild must forget every
         * produced shape and recompute from the input, so the graph input's
         * own (authoritative) shape is saved and restored first. */
        int in_tidx = find_graph_input(g);
        if (in_tidx < 0) return FE_ERR_SHAPE;
        int save_ndim;
        int save_shape[FERRITE_MAX_DIMS];
        reset_produced_shapes(g, in_tidx, &save_ndim, save_shape);

        int dynamic = infer_shapes(rt, input, ep->input_dynamic);
        ep->input_dynamic |= dynamic;
        s = fe_plan_memory(g, &ep->plan);
        if (s != FE_OK) return s;
        ep->n_plan_builds++;
        ep->bound_input_ndim = input->ndim;
        memcpy(ep->bound_input_shape, input->shape,
               input->ndim * sizeof(int));
        ep->plan_valid = 1;
    }

    fe_arena_reset(&rt->activation_arena);
    for (int i = 0; i < g->n_tensors; i++) {
        FeTensorEntry *e = &g->tensors[i];
        if (!e->is_weight) e->tensor = NULL;
    }

    s = fe_plan_apply(g, &ep->plan,
                      rt->activation_arena.base,
                      rt->activation_arena.size,
                      &rt->activation_arena);
    if (s != FE_OK) return s;

    int in_tidx = find_graph_input(g);
    if (in_tidx < 0) return FE_ERR_SHAPE;
    g->tensors[in_tidx].tensor = (FeTensor *)input;

    /* Allocate the output tensor when it has no producer (planner skips
     * producer-less tensors); hand-built graphs use an OUTPUT node. */
    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i].op != FE_OP_OUTPUT || g->nodes[i].n_inputs == 0)
            continue;
        int in_idx = g->nodes[i].inputs[0];
        if (g->tensors[in_idx].tensor) continue;
        g->tensors[in_idx].tensor = fe_arena_alloc_tensor(
            &rt->activation_arena, g->tensors[in_idx].dtype,
            g->tensors[in_idx].ndim, g->tensors[in_idx].shape);
        if (!g->tensors[in_idx].tensor) return FE_ERR_NOMEM;
        break;
    }

    /* The flat execution loop: no dispatch switch, no lifetime analysis. */
    for (int i = 0; i < ep->n_steps; i++) {
        s = ep->steps[i].fn(rt, &g->nodes[ep->steps[i].node_index]);
        if (s != FE_OK) {
            fprintf(stderr, "Engine: node '%s' failed with %d\n",
                    g->nodes[ep->steps[i].node_index].name, s);
            return s;
        }
    }

    return copy_output(rt, output);
}