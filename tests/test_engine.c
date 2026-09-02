// tests/test_engine.c
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "../core/tensor.h"
#include "../core/allocator.h"
#include "../graph/graph.h"
#include "../runtime/engine.h"

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

int main(void) {
    test_two_layer_mlp();
    test_stage3_graph();
    printf("\nAll tests passed.\n");
    return 0;
}