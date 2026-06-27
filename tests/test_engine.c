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

int main(void) {
    test_two_layer_mlp();
    printf("\nAll tests passed.\n");
    return 0;
}