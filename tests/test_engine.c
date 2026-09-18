// tests/test_engine.c
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "../core/tensor.h"
#include "../core/allocator.h"
#include "../graph/graph.h"
#include "../planner/memory_planner.h"
#include "../runtime/engine.h"
#include "../importer/onnx.h"

#define EPSILON 1e-4f
#define WEIGHT_BUF_SIZE  (1024 * 1024)   /* 1MB for weights      */
#define ACT_BUF_SIZE     (1024 * 1024)   /* 1MB for activations  */

static unsigned char weight_buf[WEIGHT_BUF_SIZE] __attribute__((aligned(64)));
static unsigned char act_buf   [ACT_BUF_SIZE]    __attribute__((aligned(64)));

/*
 * Build and run a minimal 2-layer network:
 *
 *   input [1,4] -> Linear(4->8) -> ReLU -> Linear(8->3) -> Softmax -> output [1,3]
 *
 * We set weights to identity-like values so we can verify the output
 * analytically rather than just checking it doesn't crash.
 */
static void test_two_layer_mlp(void) {
    FeGraph g;
    fe_graph_init(&g);

    /* Tensor shapes */
    int s_in[]  = {1, 4};
    int s_w0[]  = {4, 8};
    int s_b0[]  = {8};
    int s_h0[]  = {1, 8};
    int s_w1[]  = {8, 3};
    int s_b1[]  = {3};
    int s_out[] = {1, 3};

    /* Register tensors */
    int t_in  = fe_graph_add_tensor(&g, "input",  DTYPE_FLOAT32, 2, s_in,  0);
    int t_w0  = fe_graph_add_tensor(&g, "w0",     DTYPE_FLOAT32, 2, s_w0,  1);
    int t_b0  = fe_graph_add_tensor(&g, "b0",     DTYPE_FLOAT32, 1, s_b0,  1);
    int t_h0  = fe_graph_add_tensor(&g, "h0",     DTYPE_FLOAT32, 2, s_h0,  0);
    int t_r0  = fe_graph_add_tensor(&g, "relu0",  DTYPE_FLOAT32, 2, s_h0,  0);
    int t_w1  = fe_graph_add_tensor(&g, "w1",     DTYPE_FLOAT32, 2, s_w1,  1);
    int t_b1  = fe_graph_add_tensor(&g, "b1",     DTYPE_FLOAT32, 1, s_b1,  1);
    int t_h1  = fe_graph_add_tensor(&g, "h1",     DTYPE_FLOAT32, 2, s_out, 0);
    int t_out = fe_graph_add_tensor(&g, "output", DTYPE_FLOAT32, 2, s_out, 0);

    /* Register nodes */
    int lin0_in[]  = {t_in, t_w0, t_b0}; int lin0_out[] = {t_h0};
    int relu_in[]  = {t_h0};              int relu_out[] = {t_r0};
    int lin1_in[]  = {t_r0, t_w1, t_b1}; int lin1_out[] = {t_h1};
    int soft_in[]  = {t_h1};              int soft_out[] = {t_out};

    fe_graph_add_node(&g, "input",   FE_OP_INPUT,   NULL,      0, &t_in, 1);
    fe_graph_add_node(&g, "linear0", FE_OP_LINEAR,  lin0_in,   3, lin0_out, 1);
    fe_graph_add_node(&g, "relu0",   FE_OP_RELU,    relu_in,   1, relu_out, 1);
    fe_graph_add_node(&g, "linear1", FE_OP_LINEAR,  lin1_in,   3, lin1_out, 1);
    fe_graph_add_node(&g, "softmax", FE_OP_SOFTMAX, soft_in,   1, soft_out, 1);
    fe_graph_add_node(&g, "output",  FE_OP_OUTPUT,  &t_out,    1, NULL,     0);

    /* Initialise runtime */
    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g,
                            weight_buf, WEIGHT_BUF_SIZE,
                            act_buf,    ACT_BUF_SIZE) == FE_OK);

    /* Allocate weight tensors */
    assert(fe_runtime_alloc_weights(&rt) == FE_OK);

    /* Fill weights: w0 = 0.1, b0 = 0, w1 = 0.1, b1 = 0 */
    float *w0 = (float *)g.tensors[t_w0].tensor->data;
    float *b0 = (float *)g.tensors[t_b0].tensor->data;
    float *w1 = (float *)g.tensors[t_w1].tensor->data;
    float *b1 = (float *)g.tensors[t_b1].tensor->data;

    for (int i = 0; i < 4*8; i++) w0[i] = 0.1f;
    for (int i = 0; i < 8;   i++) b0[i] = 0.0f;
    for (int i = 0; i < 8*3; i++) w1[i] = 0.1f;
    for (int i = 0; i < 3;   i++) b1[i] = 0.0f;

    /* Build input: [1, 2, 3, 4] */
    FeTensor *input = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
    float in_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(input->data, in_data, sizeof(in_data));

    /* Allocate output */
    FeTensor *output = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_out);

    /* Run inference */
    assert(fe_runtime_run(&rt, input, output) == FE_OK);

    /* Softmax output must sum to 1.0 */
    float *out = (float *)output->data;
    float sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        assert(out[i] > 0.0f);
        sum += out[i];
    }
    assert(fabsf(sum - 1.0f) < EPSILON);

    /*
     * With uniform weights, all 3 output logits are equal,
     * so softmax should give exactly 1/3 each.
     */
    for (int i = 0; i < 3; i++) {
        assert(fabsf(out[i] - (1.0f/3.0f)) < EPSILON);
    }

    fe_runtime_print_trace(&rt);
    fe_tensor_free(input);
    fe_tensor_free(output);
    printf("PASS test_two_layer_mlp\n");
}

/*
 * Stage 3 end-to-end graph: exercise the new math/activation/gemm ops
 * through the engine in one fused pipeline.
 *
 *   input [1,3] --EXP--> [1,3] --SIGMOID--> t1 --LN(beta=bias)--> t2
 *              --GEMM(W=[3,2])--> [1,2] --SOFTMAX--> output [1,2]
 *
 * We set layernorm gamma=1/beta=0 so it is the identity, W is the identity
 * projection, and softmax of identical logits yields 1/2 each.
 */
static void test_stage3_graph(void) {
    FeGraph g;
    fe_graph_init(&g);

    int s_in[]  = {1, 3};
    int s_12[]  = {1, 2};
    int s_13[]  = {1, 3};
    int s_w[]   = {3, 2};
    int s_g[]   = {3};   /* layernorm gamma */
    int s_b[]   = {3};   /* layernorm beta  */

    int t_in  = fe_graph_add_tensor(&g, "input", DTYPE_FLOAT32, 2, s_in,  0);
    int t_e   = fe_graph_add_tensor(&g, "exp",   DTYPE_FLOAT32, 2, s_13,  0);
    int t_s   = fe_graph_add_tensor(&g, "sig",   DTYPE_FLOAT32, 2, s_13,  0);
    int t_ln  = fe_graph_add_tensor(&g, "ln",    DTYPE_FLOAT32, 2, s_13,  0);
    int t_w   = fe_graph_add_tensor(&g, "w",     DTYPE_FLOAT32, 2, s_w,   1);
    int t_lg  = fe_graph_add_tensor(&g, "lg",    DTYPE_FLOAT32, 1, s_g,   1);
    int t_lb  = fe_graph_add_tensor(&g, "lb",    DTYPE_FLOAT32, 1, s_b,   1);
    int t_geo = fe_graph_add_tensor(&g, "geo",   DTYPE_FLOAT32, 2, s_12,  0);
    int t_out = fe_graph_add_tensor(&g, "output",DTYPE_FLOAT32, 2, s_12,  0);

    int e_in[] = {t_in};  int e_out[] = {t_e};
    int s_in2[] = {t_e};  int s_out2[] = {t_s};
    int ln_in[] = {t_s, t_lg, t_lb};  int ln_out[] = {t_ln};
    int g_in[] = {t_ln, t_w};  int g_out[] = {t_geo};
    int sm_in[] = {t_geo}; int sm_out[] = {t_out};

    fe_graph_add_node(&g, "input", FE_OP_INPUT,    NULL,    0, &t_in, 1);
    fe_graph_add_node(&g, "exp",   FE_OP_EXP,      e_in,    1, e_out, 1);
    fe_graph_add_node(&g, "sig",   FE_OP_SIGMOID,  s_in2,   1, s_out2, 1);
    int n_ln = fe_graph_add_node(&g, "ln",    FE_OP_LAYERNORM, ln_in,  3, ln_out, 1);
    g.nodes[n_ln].attrs.layernorm.eps = 1e-5f;
    int n_geo = fe_graph_add_node(&g, "gemm",  FE_OP_GEMM,     g_in,    2, g_out, 1);
    g.nodes[n_geo].attrs.gemm.alpha = 1.0f;
    g.nodes[n_geo].attrs.gemm.beta  = 0.0f;
    g.nodes[n_geo].attrs.gemm.transA = 0;
    g.nodes[n_geo].attrs.gemm.transB = 0;
    fe_graph_add_node(&g, "soft",  FE_OP_SOFTMAX,  sm_in,   1, sm_out, 1);
    fe_graph_add_node(&g, "output",FE_OP_OUTPUT,   &t_out,  1, NULL,   0);

    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g,
                           weight_buf, WEIGHT_BUF_SIZE,
                           act_buf,    ACT_BUF_SIZE) == FE_OK);
    assert(fe_runtime_alloc_weights(&rt) == FE_OK);

    float *w   = (float *)g.tensors[t_w].tensor->data;
    float *lg  = (float *)g.tensors[t_lg].tensor->data;
    float *lb  = (float *)g.tensors[t_lb].tensor->data;
    for (int i = 0; i < 3; i++) lg[i] = 1.0f;
    for (int i = 0; i < 3; i++) lb[i] = 0.0f;
    /* W: [1, 0; 1, 0; 1, 0] -> both outputs get the same logit = sum of the 3 */
    for (int r = 0; r < 3; r++) { w[r * 2 + 0] = 1.0f; w[r * 2 + 1] = 1.0f; }

    FeTensor *input = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
    float in_data[] = {1.0f, 2.0f, 3.0f};
    memcpy(input->data, in_data, sizeof(in_data));
    FeTensor *output = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_12);

    assert(fe_runtime_run(&rt, input, output) == FE_OK);

    float *out = (float *)output->data;
    float sum = out[0] + out[1];
    assert(fabsf(out[0] - out[1]) < EPSILON);
    assert(fabsf(sum - 1.0f) < EPSILON);

    fe_runtime_print_trace(&rt);
    fe_tensor_free(input);
    fe_tensor_free(output);
    printf("PASS test_stage3_graph\n");
}

static void test_activation_footprint_equals_plan(void) {
    /* Build a small MLP, run the planner on it, then assert the runtime's
     * activation arena usage never exceeds the planner's total. */
    FeGraph g;
    fe_graph_init(&g);

    int s_in[]  = {1, 4};
    int s_w0[]  = {4, 8};
    int s_b0[]  = {8};
    int s_h0[]  = {1, 8};
    int s_w1[]  = {8, 3};
    int s_b1[]  = {3};
    int s_out[] = {1, 3};

    int t_in  = fe_graph_add_tensor(&g, "input", DTYPE_FLOAT32, 2, s_in, 0);
    int t_w0  = fe_graph_add_tensor(&g, "w0",    DTYPE_FLOAT32, 2, s_w0, 1);
    int t_b0  = fe_graph_add_tensor(&g, "b0",    DTYPE_FLOAT32, 1, s_b0, 1);
    int t_h0  = fe_graph_add_tensor(&g, "h0",    DTYPE_FLOAT32, 2, s_h0, 0);
    int t_r0  = fe_graph_add_tensor(&g, "relu0", DTYPE_FLOAT32, 2, s_h0, 0);
    int t_w1  = fe_graph_add_tensor(&g, "w1",    DTYPE_FLOAT32, 2, s_w1, 1);
    int t_b1  = fe_graph_add_tensor(&g, "b1",    DTYPE_FLOAT32, 1, s_b1, 1);
    int t_h1  = fe_graph_add_tensor(&g, "h1",    DTYPE_FLOAT32, 2, s_out, 0);

    int lin0_in[] = {t_in, t_w0, t_b0}; int lin0_out[] = {t_h0};
    int relu_in[] = {t_h0};             int relu_out[] = {t_r0};
    int lin1_in[] = {t_r0, t_w1, t_b1}; int lin1_out[] = {t_h1};

    fe_graph_add_node(&g, "input",   FE_OP_INPUT,  NULL,    0, &t_in, 1);
    fe_graph_add_node(&g, "linear0", FE_OP_LINEAR, lin0_in, 3, lin0_out, 1);
    fe_graph_add_node(&g, "relu0",   FE_OP_RELU,   relu_in, 1, relu_out, 1);
    fe_graph_add_node(&g, "linear1", FE_OP_LINEAR, lin1_in, 3, lin1_out, 1);

    assert(fe_graph_topo_sort(&g) == FE_OK);
    FePlan plan;
    assert(fe_plan_memory(&g, &plan) == FE_OK);

    /* Compute naive (no-reuse) footprint, aligned to 64 bytes as the
     * planner does, so the comparison is fair. */
    size_t naive = 0;
    for (int i = 0; i < plan.n_lifetimes; i++) {
        size_t need = (size_t)plan.lifetimes[i].size_bytes;
        naive += (need + 63) & ~(size_t)63;
    }
    /* Reuse across disjoint lifetimes must save memory. */
    assert(plan.total_activation_bytes < naive);
    assert(plan.total_activation_bytes > 0);

    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g,
                           weight_buf, WEIGHT_BUF_SIZE,
                           act_buf,    ACT_BUF_SIZE) == FE_OK);
    assert(fe_runtime_alloc_weights(&rt) == FE_OK);

    float *w0 = (float *)g.tensors[t_w0].tensor->data;
    float *b0 = (float *)g.tensors[t_b0].tensor->data;
    float *w1 = (float *)g.tensors[t_w1].tensor->data;
    float *b1 = (float *)g.tensors[t_b1].tensor->data;
    for (int i = 0; i < 4*8; i++) w0[i] = 0.1f;
    for (int i = 0; i < 8;   i++) b0[i] = 0.0f;
    for (int i = 0; i < 8*3; i++) w1[i] = 0.1f;
    for (int i = 0; i < 3;   i++) b1[i] = 0.0f;

    FeTensor *input = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
    float in_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(input->data, in_data, sizeof(in_data));
    FeTensor *output = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_out);

    assert(fe_runtime_run(&rt, input, output) == FE_OK);

    /* All three output logits are equal: with input [1,2,3,4],
     * h0 = 1.0 per unit, relu keeps 1.0, h1 = 8 * 1.0 * 0.1 = 0.8. */
    float *out = (float *)output->data;
    assert(fabsf(out[0] - 0.8f) < EPSILON);
    assert(fabsf(out[1] - 0.8f) < EPSILON);
    assert(fabsf(out[2] - 0.8f) < EPSILON);

    /* The activation arena peak is the planner's reserved data region plus
     * per-tensor metadata structs appended after it — data alone must not
     * exceed the plan total, and the whole thing stays within ~a few KB of
     * the planned data bytes. */
    assert(fe_arena_peak(&rt.activation_arena) >= plan.total_activation_bytes);
    assert(fe_arena_peak(&rt.activation_arena) <=
           plan.total_activation_bytes + 4 * 1024);

    fe_tensor_free(input);
    fe_tensor_free(output);
    printf("PASS test_activation_footprint_equals_plan (plan=%zu bytes, "
           "naive=%zu)\n", plan.total_activation_bytes, naive);
}

/*
 * End-to-end: load a real ONNX model through the importer and run it
 * through the engine. ONNX graphs have no FE_OP_INPUT/FE_OP_OUTPUT nodes,
 * which used to leave the input tensor unbound (NULL data) — the first
 * consumer crashed. tiny_mlp is MatMul(fc1.weight_T)+bias -> ReLU ->
 * MatMul(fc2.weight_T)+bias -> Softmax.
 */
static void test_onnx_load_and_run(void) {
    FeGraph g;
    FeArena weight_arena;
    unsigned char wbuf[200 * 1024] __attribute__((aligned(64)));
    fe_arena_init(&weight_arena, wbuf, sizeof(wbuf));

    assert(fe_onnx_load(&g, &weight_arena, "tests/tiny_mlp.onnx") == FE_OK);
    assert(g.topo_valid);

    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g,
                           wbuf, sizeof(wbuf),
                           act_buf, ACT_BUF_SIZE) == FE_OK);

    int in_shape[]  = {1, 4};
    int out_shape[] = {1, 3};

    FeTensor *input = fe_tensor_alloc(DTYPE_FLOAT32, 2, in_shape);
    float in_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(input->data, in_data, sizeof(in_data));
    FeTensor *output = fe_tensor_alloc(DTYPE_FLOAT32, 2, out_shape);

    assert(fe_runtime_run(&rt, input, output) == FE_OK);

    /* Softmax output must normalize to 1 across the 3 classes. */
    float *out = (float *)output->data;
    float sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        assert(out[i] > 0.0f && out[i] < 1.0f);
        sum += out[i];
    }
    assert(fabsf(sum - 1.0f) < EPSILON);

    /* Deterministic: a second run reproduces the first exactly. */
    FeTensor *output2 = fe_tensor_alloc(DTYPE_FLOAT32, 2, out_shape);
    assert(fe_runtime_run(&rt, input, output2) == FE_OK);
    assert(memcmp(output->data, output2->data, 3 * sizeof(float)) == 0);

    fe_tensor_free(input);
    fe_tensor_free(output);
    fe_tensor_free(output2);
    printf("PASS test_onnx_load_and_run\n");
}

/*
 * Stage 13: static execution plan. A same-shaped second run must reuse the
 * cached plan (no re-analysis) and reproduce the first run bit-for-bit; a
 * shape change triggers exactly one rebuild, then caching resumes.
 */
static void test_exec_plan_static(void) {
    FeGraph g;
    fe_graph_init(&g);

    /* Dynamic batch dim on the input so we can exercise the rebuild path. */
    int s_in[]  = {0, 4};
    int s_w[]   = {4, 2};
    int s_b[]   = {2};
    int s_h[]   = {0, 2};

    int t_in  = fe_graph_add_tensor(&g, "input", DTYPE_FLOAT32, 2, s_in, 0);
    int t_w   = fe_graph_add_tensor(&g, "w",     DTYPE_FLOAT32, 2, s_w,  1);
    int t_b   = fe_graph_add_tensor(&g, "b",     DTYPE_FLOAT32, 1, s_b,  1);
    int t_h   = fe_graph_add_tensor(&g, "h",     DTYPE_FLOAT32, 2, s_h,  0);

    int lin_in[] = {t_in, t_w, t_b}; int lin_out[] = {t_h};

    fe_graph_add_node(&g, "input", FE_OP_INPUT,  NULL,   0, &t_in, 1);
    fe_graph_add_node(&g, "linear", FE_OP_LINEAR, lin_in, 3, lin_out, 1);
    fe_graph_add_node(&g, "output", FE_OP_OUTPUT, &t_h,  1, NULL,    0);

    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g,
                           weight_buf, WEIGHT_BUF_SIZE,
                           act_buf,    ACT_BUF_SIZE) == FE_OK);
    assert(fe_runtime_alloc_weights(&rt) == FE_OK);

    float *w = (float *)g.tensors[t_w].tensor->data;
    float *b = (float *)g.tensors[t_b].tensor->data;
    for (int i = 0; i < 4*2; i++) w[i] = 0.5f;
    for (int i = 0; i < 2;   i++) b[i] = 0.1f;

    /* Batch-1 run: each output col = 4 * 0.5 + 0.1 = 2.1. */
    int in1_shape[]  = {1, 4};
    int out1_shape[] = {1, 2};
    FeTensor *in1  = fe_tensor_alloc(DTYPE_FLOAT32, 2, in1_shape);
    FeTensor *out1 = fe_tensor_alloc(DTYPE_FLOAT32, 2, out1_shape);
    FeTensor *out2 = fe_tensor_alloc(DTYPE_FLOAT32, 2, out1_shape);
    float in1_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
    memcpy(in1->data, in1_data, sizeof(in1_data));

    assert(fe_runtime_run(&rt, in1, out1) == FE_OK);
    float *o = (float *)out1->data;
    assert(fabsf(o[0] - 2.1f) < EPSILON);
    assert(fabsf(o[1] - 2.1f) < EPSILON);
    assert(rt.exec.n_plan_builds == 1);

    /* Same shape again: outputs identical, no re-analysis. */
    assert(fe_runtime_run(&rt, in1, out2) == FE_OK);
    assert(memcmp(out1->data, out2->data, 2 * sizeof(float)) == 0);
    assert(rt.exec.n_plan_builds == 1);

    /* Batch-3 shape change: exactly one rebuild, then correct output. */
    int in3_shape[]  = {3, 4};
    int out3_shape[] = {3, 2};
    FeTensor *in3  = fe_tensor_alloc(DTYPE_FLOAT32, 2, in3_shape);
    FeTensor *out3 = fe_tensor_alloc(DTYPE_FLOAT32, 2, out3_shape);
    float in3_data[3 * 4];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 4; c++)
            in3_data[r * 4 + c] = (float)(c + 1);   /* row sum = 10 */
    memcpy(in3->data, in3_data, sizeof(in3_data));

    assert(fe_runtime_run(&rt, in3, out3) == FE_OK);
    assert(rt.exec.n_plan_builds == 2);
    float *o3 = (float *)out3->data;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 2; c++)
            assert(fabsf(o3[r * 2 + c] - (10.0f * 0.5f + 0.1f)) < EPSILON);

    /* And caching resumes: a fourth call must not re-plan. */
    FeTensor *out4 = fe_tensor_alloc(DTYPE_FLOAT32, 2, out3_shape);
    assert(fe_runtime_run(&rt, in3, out4) == FE_OK);
    assert(memcmp(out3->data, out4->data, 6 * sizeof(float)) == 0);
    assert(rt.exec.n_plan_builds == 2);

    fe_tensor_free(in1);  fe_tensor_free(out1); fe_tensor_free(out2);
    fe_tensor_free(in3);  fe_tensor_free(out3); fe_tensor_free(out4);
    printf("PASS test_exec_plan_static (n_plan_builds=%llu)\n",
           (unsigned long long)rt.exec.n_plan_builds);
}

/* A K-mismatched MatMul must make fe_runtime_run propagate the shape-infer
 * failure (FE_ERR_SHAPE) at plan-build time. The old bug swallowed the error
 * and let the planner run on a produced tensor sitting at `ndim == 0`, which
 * wrote strides[-1] in fe_plan_apply. */
static void test_shape_infer_error_propagates(void) {
    FeGraph g;
    fe_graph_init(&g);

    int s_in[]  = {1, 4};
    int s_bad[] = {5, 3};   /* K mismatch: 4 != 5 */
    int t_in = fe_graph_add_tensor(&g, "input", DTYPE_FLOAT32, 2, s_in, 0);
    int t_w  = fe_graph_add_tensor(&g, "w",     DTYPE_FLOAT32, 2, s_bad, 1);
    int t_out= fe_graph_add_tensor(&g, "out",   DTYPE_FLOAT32, 2, s_bad, 0);

    int mm_in[] = {t_in, t_w}; int mm_out[] = {t_out};
    fe_graph_add_node(&g, "input", FE_OP_INPUT,  NULL, 0, &t_in, 1);
    fe_graph_add_node(&g, "mm",    FE_OP_MATMUL, mm_in, 2, mm_out, 1);
    fe_graph_add_node(&g, "output", FE_OP_OUTPUT, &t_out, 1, NULL, 0);

    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g,
                           weight_buf, WEIGHT_BUF_SIZE,
                           act_buf,    ACT_BUF_SIZE) == FE_OK);
    assert(fe_runtime_alloc_weights(&rt) == FE_OK);

    FeTensor *input = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
    FeTensor *output = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_bad);
    float in_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(input->data, in_data, sizeof(in_data));

    assert(fe_runtime_run(&rt, input, output) == FE_ERR_SHAPE);

    fe_tensor_free(input);
    fe_tensor_free(output);
    printf("PASS test_shape_infer_error_propagates\n");
}

/* A single-node conv1d graph through the engine. Hand-computed:
 *   x [1,1,4] = {1,2,3,4}, w [1,1,2] = {0.5, 1.0}, b = {0.1}, stride 1, pad 0
 *   y[l] = x[l]*0.5 + x[l+1]*1.0 + 0.1  →  {2.6, 4.1, 5.6} */
static void test_conv1d_graph(void) {
    FeGraph g;
    fe_graph_init(&g);

    int s_in[]  = {1, 1, 4};
    int s_w[]   = {1, 1, 2};
    int s_b[]   = {1};
    int s_out[] = {1, 1, 3};

    int t_in  = fe_graph_add_tensor(&g, "input", DTYPE_FLOAT32, 3, s_in,  0);
    int t_w   = fe_graph_add_tensor(&g, "w",     DTYPE_FLOAT32, 3, s_w,   1);
    int t_b   = fe_graph_add_tensor(&g, "b",     DTYPE_FLOAT32, 1, s_b,   1);
    int t_out = fe_graph_add_tensor(&g, "out",   DTYPE_FLOAT32, 3, s_out, 0);

    int cv_in[]  = {t_in, t_w, t_b};
    int cv_out[] = {t_out};
    fe_graph_add_node(&g, "input", FE_OP_INPUT,  NULL,   0, &t_in, 1);
    int n = fe_graph_add_node(&g, "conv1d", FE_OP_CONV1D, cv_in, 3, cv_out, 1);
    g.nodes[n].attrs.conv1d.stride = 1;
    g.nodes[n].attrs.conv1d.pad     = 0;
    fe_graph_add_node(&g, "output", FE_OP_OUTPUT, &t_out,  1, NULL, 0);

    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g,
                           weight_buf, WEIGHT_BUF_SIZE,
                           act_buf,    ACT_BUF_SIZE) == FE_OK);
    assert(fe_runtime_alloc_weights(&rt) == FE_OK);

    float *w = (float *)g.tensors[t_w].tensor->data;
    float *b = (float *)g.tensors[t_b].tensor->data;
    float wd[] = {0.5f, 1.0f};
    memcpy(w, wd, sizeof(wd));
    b[0] = 0.1f;

    FeTensor *in  = fe_tensor_alloc(DTYPE_FLOAT32, 3, s_in);
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 3, s_out);
    float xd[] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(in->data, xd, sizeof(xd));

    assert(fe_runtime_run(&rt, in, out) == FE_OK);
    float *o = (float *)out->data;
    assert(fabsf(o[0] - 2.6f) < EPSILON);
    assert(fabsf(o[1] - 4.1f) < EPSILON);
    assert(fabsf(o[2] - 5.6f) < EPSILON);

    fe_tensor_free(in);
    fe_tensor_free(out);
    printf("PASS test_conv1d_graph\n");
}

/* A single-node BatchNorm graph. Hand-computed (inference mode):
 *   gamma {1.5, 0.5}, beta {0.1, -0.1}, mean {1, 2}, var {4, 1}, eps 1e-4
 *   x {3, 2} →
 *     c0: (3-1)/sqrt(4) * 1.5 + 0.1 = 1.6
 *     c1: (2-2)/sqrt(1) * 0.5 - 0.1 = -0.1    */
static void test_batchnorm_graph(void) {
    FeGraph g;
    fe_graph_init(&g);

    int s_in[]  = {1, 2};
    int s_c[]   = {2};

    int t_in   = fe_graph_add_tensor(&g, "input", DTYPE_FLOAT32, 2, s_in, 0);
    int t_gamma = fe_graph_add_tensor(&g, "gamma", DTYPE_FLOAT32, 1, s_c, 1);
    int t_beta  = fe_graph_add_tensor(&g, "beta",  DTYPE_FLOAT32, 1, s_c, 1);
    int t_mean  = fe_graph_add_tensor(&g, "mean",  DTYPE_FLOAT32, 1, s_c, 1);
    int t_var   = fe_graph_add_tensor(&g, "var",   DTYPE_FLOAT32, 1, s_c, 1);
    int t_out   = fe_graph_add_tensor(&g, "out",   DTYPE_FLOAT32, 2, s_in, 0);

    int bn_in[]  = {t_in, t_gamma, t_beta, t_mean, t_var};
    fe_graph_add_node(&g, "input",   FE_OP_INPUT,    NULL,   0, &t_in, 1);
    int n = fe_graph_add_node(&g, "bn", FE_OP_BATCHNORM, bn_in, 5, &t_out, 1);
    g.nodes[n].attrs.batchnorm.eps = 1e-4f;
    fe_graph_add_node(&g, "output", FE_OP_OUTPUT, &t_out, 1, NULL, 0);

    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g,
                           weight_buf, WEIGHT_BUF_SIZE,
                           act_buf,    ACT_BUF_SIZE) == FE_OK);
    assert(fe_runtime_alloc_weights(&rt) == FE_OK);

    float *gamma = (float *)g.tensors[t_gamma].tensor->data;
    float *beta  = (float *)g.tensors[t_beta].tensor->data;
    float *mean  = (float *)g.tensors[t_mean].tensor->data;
    float *var   = (float *)g.tensors[t_var].tensor->data;
    float gd[] = {1.5f, 0.5f}, bd[] = {0.1f, -0.1f};
    float md[] = {1.0f, 2.0f}, vd[] = {4.0f, 1.0f};
    memcpy(gamma, gd, sizeof(gd));
    memcpy(beta,  bd, sizeof(bd));
    memcpy(mean,  md, sizeof(md));
    memcpy(var,   vd, sizeof(vd));

    FeTensor *in  = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
    float xd[] = {3.0f, 2.0f};
    memcpy(in->data, xd, sizeof(xd));

    assert(fe_runtime_run(&rt, in, out) == FE_OK);
    float *o = (float *)out->data;
    assert(fabsf(o[0] - 1.6f) < EPSILON);
    assert(fabsf(o[1] + 0.1f) < EPSILON);

    fe_tensor_free(in);
    fe_tensor_free(out);
    printf("PASS test_batchnorm_graph\n");
}

/* fe_runtime_run_batch: N same-shaped Linear inferences. The static plan is
 * shared, so exactly one plan build happens across the whole batch. */
static void test_runtime_run_batch(void) {
    FeGraph g;
    fe_graph_init(&g);

    int s_in[]  = {1, 4};
    int s_w[]   = {4, 2};
    int s_b[]   = {2};
    int s_h[]   = {1, 2};

    int t_in = fe_graph_add_tensor(&g, "input", DTYPE_FLOAT32, 2, s_in, 0);
    int t_w  = fe_graph_add_tensor(&g, "w",     DTYPE_FLOAT32, 2, s_w,  1);
    int t_b  = fe_graph_add_tensor(&g, "b",     DTYPE_FLOAT32, 1, s_b,  1);
    int t_h  = fe_graph_add_tensor(&g, "h",     DTYPE_FLOAT32, 2, s_h,  0);

    int lin_in[] = {t_in, t_w, t_b};
    fe_graph_add_node(&g, "input",  FE_OP_INPUT,  NULL,   0, &t_in, 1);
    fe_graph_add_node(&g, "linear", FE_OP_LINEAR, lin_in, 3, &t_h,  1);
    fe_graph_add_node(&g, "output", FE_OP_OUTPUT, &t_h,   1, NULL,  0);

    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g,
                           weight_buf, WEIGHT_BUF_SIZE,
                           act_buf,    ACT_BUF_SIZE) == FE_OK);
    assert(fe_runtime_alloc_weights(&rt) == FE_OK);

    float *w = (float *)g.tensors[t_w].tensor->data;
    float *b = (float *)g.tensors[t_b].tensor->data;
    for (int i = 0; i < 4*2; i++) w[i] = 0.5f;
    for (int i = 0; i < 2;   i++) b[i] = 0.1f;

    #define BN 3
    FeTensor *in[BN], *out[BN];
    float rows[BN][4] = {{1, 1, 1, 1}, {2, 2, 2, 2}, {1, 2, 3, 4}};
    float expect[BN]  = {2.1f, 4.1f, 5.1f};   /* row_sum*0.5 + 0.1 */
    for (int i = 0; i < BN; i++) {
        in[i]  = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
        out[i] = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_h);
        memcpy(in[i]->data, rows[i], sizeof(rows[i]));
    }

    assert(fe_runtime_run_batch(&rt, in, out, BN) == FE_OK);
    for (int i = 0; i < BN; i++) {
        float *o = (float *)out[i]->data;
        assert(fabsf(o[0] - expect[i]) < EPSILON);
        assert(fabsf(o[1] - expect[i]) < EPSILON);
    }
    assert(rt.exec.n_plan_builds == 1);   /* one plan, shared by the batch */

    for (int i = 0; i < BN; i++) {
        fe_tensor_free(in[i]);
        fe_tensor_free(out[i]);
    }
    printf("PASS test_runtime_run_batch (n=%d, n_plan_builds=1)\n", BN);
}

/* Budget stress: a 256-node MatMul chain sharing one weight tensor drives the
 * graph, topo-sort, validate, planner, and static plan near their capacity
 * (260 nodes / 258 tensors, under the 512/1024 caps). N=1 exercises the
 * AVX2 remainder fallback. Weight is the identity copy, so every output
 * equals the input exactly — a cheap, exact oracle for 256 layers. */
static void test_chain_stress(void) {
    FeGraph g;
    fe_graph_init(&g);

    int s1[] = {1, 1};
    int t_in = fe_graph_add_tensor(&g, "in", DTYPE_FLOAT32, 2, s1, 0);
    int t_w  = fe_graph_add_tensor(&g, "w",  DTYPE_FLOAT32, 2, s1, 1);

    int prev = t_in;
    const int L = 256;
    for (int i = 0; i < L; i++) {
        char name[32];
        snprintf(name, sizeof(name), "t%d", i);
        int t_out = fe_graph_add_tensor(&g, name, DTYPE_FLOAT32, 2, s1, 0);
        int in[]  = {prev, t_w};
        fe_graph_add_node(&g, name, FE_OP_MATMUL, in, 2, &t_out, 1);
        prev = t_out;
    }
    fe_graph_add_node(&g, "output", FE_OP_OUTPUT, &prev, 1, NULL, 0);

    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g,
                           weight_buf, WEIGHT_BUF_SIZE,
                           act_buf,    ACT_BUF_SIZE) == FE_OK);
    assert(fe_runtime_alloc_weights(&rt) == FE_OK);
    ((float *)g.tensors[t_w].tensor->data)[0] = 1.0f;

    FeTensor *in  = fe_tensor_alloc(DTYPE_FLOAT32, 2, s1);
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 2, s1);
    ((float *)in->data)[0] = 3.5f;

    assert(fe_runtime_run(&rt, in, out) == FE_OK);
    assert(fabsf(((float *)out->data)[0] - 3.5f) < 1e-6f);

    fe_tensor_free(in); fe_tensor_free(out);
    printf("PASS test_chain_stress (%d nodes, %d tensors)\n",
           g.n_nodes, g.n_tensors);
}

/* 12 Linear+ReLU pairs with K=N=19 (N%8!=0, exercising the scalar remainder
 * path end-to-end). w diagonal is 0.5, b=0: after 12 pairs every output
 * equals x * 0.5^12 (exact in binary). */
static void test_linear_chain_remainder(void) {
    FeGraph g;
    fe_graph_init(&g);

    int s_in[] = {1, 19}, s_w[] = {19, 19}, s_b[] = {19};
    int t_in = fe_graph_add_tensor(&g, "in", DTYPE_FLOAT32, 2, s_in, 0);

    int prev = t_in;
    const int PAIRS = 12;
    for (int i = 0; i < PAIRS; i++) {
        char wname[16], bname[16], oname[16];
        snprintf(wname, sizeof(wname), "w%d", i);
        snprintf(bname, sizeof(bname), "b%d", i);
        snprintf(oname, sizeof(oname), "o%d", i);
        int t_w = fe_graph_add_tensor(&g, wname, DTYPE_FLOAT32, 2, s_w, 1);
        int t_b = fe_graph_add_tensor(&g, bname, DTYPE_FLOAT32, 1, s_b, 1);
        int t_o = fe_graph_add_tensor(&g, oname, DTYPE_FLOAT32, 2, s_in, 0);
        char rn[24];
        snprintf(rn, sizeof(rn), "r%d", i);
        int t_r = fe_graph_add_tensor(&g, rn, DTYPE_FLOAT32, 2, s_in, 0);
        int lin[] = {prev, t_w, t_b};
        fe_graph_add_node(&g, wname, FE_OP_LINEAR, lin, 3, &t_o, 1);
        fe_graph_add_node(&g, rn, FE_OP_RELU, &t_o, 1, &t_r, 1);
        prev = t_r;
    }
    fe_graph_add_node(&g, "output", FE_OP_OUTPUT, &prev, 1, NULL, 0);

    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g,
                           weight_buf, WEIGHT_BUF_SIZE,
                           act_buf,    ACT_BUF_SIZE) == FE_OK);
    assert(fe_runtime_alloc_weights(&rt) == FE_OK);

    for (int i = 0; i < PAIRS; i++) {
        char wname[16], bname[16];
        snprintf(wname, sizeof(wname), "w%d", i);
        snprintf(bname, sizeof(bname), "b%d", i);
        int tw = -1, tb = -1;
        for (int t = 0; t < g.n_tensors; t++) {
            if (strcmp(g.tensors[t].name, wname) == 0) tw = t;
            if (strcmp(g.tensors[t].name, bname) == 0) tb = t;
        }
        float *w = (float *)g.tensors[tw].tensor->data;
        memset(w, 0, sizeof(float) * 19 * 19);
        for (int k = 0; k < 19; k++) w[k * 19 + k] = 0.5f;
        memset(g.tensors[tb].tensor->data, 0, sizeof(float) * 19);
    }

    FeTensor *in  = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
    for (int i = 0; i < 19; i++) ((float *)in->data)[i] = 1.0f;

    assert(fe_runtime_run(&rt, in, out) == FE_OK);
    float oracle = powf(0.5f, (float)PAIRS);   /* 0.5^12, exact */
    for (int j = 0; j < 19; j++)
        assert(fabsf(((float *)out->data)[j] - oracle) < 1e-7f);

    fe_tensor_free(in); fe_tensor_free(out);
    printf("PASS test_linear_chain_remainder (N=19, %d pairs)\n", PAIRS);
}

int main(void) {
    test_two_layer_mlp();
    test_stage3_graph();
    test_activation_footprint_equals_plan();
    test_onnx_load_and_run();
    test_exec_plan_static();
    test_shape_infer_error_propagates();
    test_conv1d_graph();
    test_batchnorm_graph();
    test_runtime_run_batch();
    test_chain_stress();
    test_linear_chain_remainder();
    printf("\nAll tests passed.\n");
    return 0;
}