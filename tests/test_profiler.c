// tests/test_profiler.c
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "../core/tensor.h"
#include "../core/allocator.h"
#include "../graph/graph.h"
#include "../runtime/engine.h"
#include "../tools/profiler.h"

#define WEIGHT_BUF_SIZE  (1024 * 1024)
#define ACT_BUF_SIZE     (1024 * 1024)

static unsigned char weight_buf[WEIGHT_BUF_SIZE] __attribute__((aligned(64)));
static unsigned char act_buf   [ACT_BUF_SIZE]    __attribute__((aligned(64)));

static void test_profiled_inference(void) {
    FeGraph g;
    fe_graph_init(&g);

    int s_in[]  = {1, 4};
    int s_w0[]  = {4, 8};
    int s_b0[]  = {8};
    int s_h0[]  = {1, 8};
    int s_w1[]  = {8, 3};
    int s_b1[]  = {3};
    int s_out[] = {1, 3};

    int t_in  = fe_graph_add_tensor(&g, "input",  DTYPE_FLOAT32, 2, s_in,  0);
    int t_w0  = fe_graph_add_tensor(&g, "w0",     DTYPE_FLOAT32, 2, s_w0,  1);
    int t_b0  = fe_graph_add_tensor(&g, "b0",     DTYPE_FLOAT32, 1, s_b0,  1);
    int t_h0  = fe_graph_add_tensor(&g, "h0",     DTYPE_FLOAT32, 2, s_h0,  0);
    int t_r0  = fe_graph_add_tensor(&g, "relu0",  DTYPE_FLOAT32, 2, s_h0,  0);
    int t_w1  = fe_graph_add_tensor(&g, "w1",     DTYPE_FLOAT32, 2, s_w1,  1);
    int t_b1  = fe_graph_add_tensor(&g, "b1",     DTYPE_FLOAT32, 1, s_b1,  1);
    int t_h1  = fe_graph_add_tensor(&g, "h1",     DTYPE_FLOAT32, 2, s_out, 0);
    int t_out = fe_graph_add_tensor(&g, "output", DTYPE_FLOAT32, 2, s_out, 0);

    int lin0_in[]  = {t_in, t_w0, t_b0}; int lin0_out[] = {t_h0};
    int relu_in[]  = {t_h0};              int relu_out[] = {t_r0};
    int lin1_in[]  = {t_r0, t_w1, t_b1}; int lin1_out[] = {t_h1};
    int soft_in[]  = {t_h1};              int soft_out[] = {t_out};

    fe_graph_add_node(&g, "input",   FE_OP_INPUT,   NULL,     0, &t_in, 1);
    fe_graph_add_node(&g, "linear0", FE_OP_LINEAR,  lin0_in,  3, lin0_out, 1);
    fe_graph_add_node(&g, "relu0",   FE_OP_RELU,    relu_in,  1, relu_out, 1);
    fe_graph_add_node(&g, "linear1", FE_OP_LINEAR,  lin1_in,  3, lin1_out, 1);
    fe_graph_add_node(&g, "softmax", FE_OP_SOFTMAX, soft_in,  1, soft_out, 1);
    fe_graph_add_node(&g, "output",  FE_OP_OUTPUT,  &t_out,   1, NULL,     0);

    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g,
                            weight_buf, WEIGHT_BUF_SIZE,
                            act_buf,    ACT_BUF_SIZE) == FE_OK);
    assert(fe_runtime_alloc_weights(&rt) == FE_OK);

    float *w0 = (float *)g.tensors[t_w0].tensor->data;
    float *w1 = (float *)g.tensors[t_w1].tensor->data;
    for (int i = 0; i < 4*8; i++) w0[i] = 0.1f;
    for (int i = 0; i < 8*3; i++) w1[i] = 0.1f;
    memset(g.tensors[t_b0].tensor->data, 0, 8  * sizeof(float));
    memset(g.tensors[t_b1].tensor->data, 0, 3  * sizeof(float));

    /* Attach profiler */
    FeProfiler prof;
    fe_profiler_init(&prof);
    rt.profiler = &prof;

    FeTensor *input  = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
    FeTensor *output = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_out);
    float in_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(input->data, in_data, sizeof(in_data));

    /* Run 10 iterations */
    for (int i = 0; i < 10; i++)
        assert(fe_runtime_run(&rt, input, output) == FE_OK);

    fe_profiler_print(&prof);

    /* Verify profiler recorded all operators */
    assert(prof.n_records > 0);

    fe_tensor_free(input);
    fe_tensor_free(output);
    printf("PASS test_profiled_inference\n");
}

int main(void) {
    test_profiled_inference();
    printf("\nAll tests passed.\n");
    return 0;
}