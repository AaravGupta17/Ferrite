// tests/test_graph.c
#include <stdio.h>
#include <assert.h>
#include "../graph/graph.h"

/*
 * Build and sort this graph:
 *
 *   input(0) --> linear(1) --> relu(2) --> softmax(3) --> output
 *
 * Tensors:
 *   t0: input  [1,4]
 *   t1: weight [4,8]
 *   t2: bias   [8]
 *   t3: linear output [1,8]
 *   t4: relu output   [1,8]
 *   t5: softmax output [1,8]
 */
static void test_build_and_sort(void) {
    FeGraph g;
    fe_graph_init(&g);

    int s_input[]  = {1, 4};
    int s_weight[] = {4, 8};
    int s_bias[]   = {8};
    int s_out[]    = {1, 8};

    int t0 = fe_graph_add_tensor(&g, "input",   DTYPE_FLOAT32, 2, s_input,  0);
    int t1 = fe_graph_add_tensor(&g, "weight",  DTYPE_FLOAT32, 2, s_weight, 1);
    int t2 = fe_graph_add_tensor(&g, "bias",    DTYPE_FLOAT32, 1, s_bias,   1);
    int t3 = fe_graph_add_tensor(&g, "linear",  DTYPE_FLOAT32, 2, s_out,    0);
    int t4 = fe_graph_add_tensor(&g, "relu",    DTYPE_FLOAT32, 2, s_out,    0);
    int t5 = fe_graph_add_tensor(&g, "softmax", DTYPE_FLOAT32, 2, s_out,    0);

    int linear_in[]  = {t0, t1, t2};
    int linear_out[] = {t3};
    int relu_in[]    = {t3};
    int relu_out[]   = {t4};
    int soft_in[]    = {t4};
    int soft_out[]   = {t5};

    int n0 = fe_graph_add_node(&g, "linear0",  FE_OP_LINEAR,
                                linear_in,  3, linear_out, 1);
    int n1 = fe_graph_add_node(&g, "relu0",    FE_OP_RELU,
                                relu_in,    1, relu_out,   1);
    int n2 = fe_graph_add_node(&g, "softmax0", FE_OP_SOFTMAX,
                                soft_in,    1, soft_out,   1);

    assert(n0 == 0 && n1 == 1 && n2 == 2);
    assert(fe_graph_topo_sort(&g) == FE_OK);

    /* For a linear chain, topo order must be 0->1->2 */
    assert(g.topo_order[0] == 0);
    assert(g.topo_order[1] == 1);
    assert(g.topo_order[2] == 2);

    fe_graph_print(&g);
    printf("PASS test_build_and_sort\n");
}

static void test_topo_diamond(void) {
    /*
     * Diamond graph — two parallel paths that merge:
     *
     *       n0
     *      /  \
     *    n1    n2
     *      \  /
     *       n3
     *
     * Valid topo orders: [n0, n1, n2, n3] or [n0, n2, n1, n3]
     * n3 must always be last.
     */
    FeGraph g;
    fe_graph_init(&g);

    int s[] = {1, 4};
    int t0 = fe_graph_add_tensor(&g, "t0", DTYPE_FLOAT32, 2, s, 0);
    int t1 = fe_graph_add_tensor(&g, "t1", DTYPE_FLOAT32, 2, s, 0);
    int t2 = fe_graph_add_tensor(&g, "t2", DTYPE_FLOAT32, 2, s, 0);
    int t3 = fe_graph_add_tensor(&g, "t3", DTYPE_FLOAT32, 2, s, 0);
    int t4 = fe_graph_add_tensor(&g, "t4", DTYPE_FLOAT32, 2, s, 0);

    int in0[]  = {t0}; int out0[] = {t1, t2};
    int in1[]  = {t1}; int out1[] = {t3};
    int in2[]  = {t2}; int out2[] = {t4};
    int in3[]  = {t3, t4}; int out3[] = {};

    fe_graph_add_node(&g, "n0", FE_OP_RELU, in0, 1, out0, 2);
    fe_graph_add_node(&g, "n1", FE_OP_RELU, in1, 1, out1, 1);
    fe_graph_add_node(&g, "n2", FE_OP_RELU, in2, 1, out2, 1);
    fe_graph_add_node(&g, "n3", FE_OP_ADD,  in3, 2, out3, 0);

    assert(fe_graph_topo_sort(&g) == FE_OK);

    /* n0 must be first, n3 must be last */
    assert(g.topo_order[0] == 0);
    assert(g.topo_order[3] == 3);

    printf("PASS test_topo_diamond\n");
}

static void test_validate_accepts_all_supported_dtypes(void) {
    /* Phase B: validation accepts every dtype the engine can execute —
     * INT16 quantized weights, FP16/BF16 storage-only weights. */
    FeGraph g;
    fe_graph_init(&g);
    int s[] = {1, 4};
    int t0 = fe_graph_add_tensor(&g, "w_i16",  DTYPE_INT16,    2, s, 1);
    int t1 = fe_graph_add_tensor(&g, "w_f16",  DTYPE_FLOAT16,  2, s, 1);
    int t2 = fe_graph_add_tensor(&g, "w_bf16", DTYPE_BFLOAT16, 2, s, 1);
    int in0[] = {};
    int out0[] = {t0, t1, t2};
    fe_graph_add_node(&g, "n0", FE_OP_INPUT, in0, 0, out0, 3);
    assert(fe_graph_validate(&g) == FE_OK);
    printf("PASS test_validate_accepts_all_supported_dtypes\n");
}

static void test_validate_passes_on_valid_graph(void) {
    FeGraph g;
    fe_graph_init(&g);
    int s[] = {1, 4};
    int t0 = fe_graph_add_tensor(&g, "t0", DTYPE_FLOAT32, 2, s, 0);
    int t1 = fe_graph_add_tensor(&g, "t1", DTYPE_FLOAT32, 2, s, 0);
    int in0[] = {t0}; int out0[] = {t1};
    fe_graph_add_node(&g, "n0", FE_OP_RELU, in0, 1, out0, 1);
    assert(fe_graph_validate(&g) == FE_OK);
    printf("PASS test_validate_passes_on_valid_graph\n");
}

static void test_validate_catches_out_of_range_index(void) {
    FeGraph g;
    fe_graph_init(&g);
    int s[] = {1, 4};
    int t0 = fe_graph_add_tensor(&g, "t0", DTYPE_FLOAT32, 2, s, 0);
    int bad_out[] = {t0 + 99}; /* doesn't exist */
    int in0[] = {t0};
    fe_graph_add_node(&g, "n0", FE_OP_RELU, in0, 1, bad_out, 1);
    assert(fe_graph_validate(&g) == FE_ERR_BOUNDS);
    printf("PASS test_validate_catches_out_of_range_index\n");
}

static void test_validate_catches_double_producer(void) {
    FeGraph g;
    fe_graph_init(&g);
    int s[] = {1, 4};
    int t0 = fe_graph_add_tensor(&g, "t0", DTYPE_FLOAT32, 2, s, 0);
    int t1 = fe_graph_add_tensor(&g, "t1", DTYPE_FLOAT32, 2, s, 0);
    int t2 = fe_graph_add_tensor(&g, "t2", DTYPE_FLOAT32, 2, s, 0);
    int in0[] = {t0}; int out0[] = {t2};
    int in1[] = {t1}; int out1[] = {t2}; /* same output as n0 — conflict */
    fe_graph_add_node(&g, "n0", FE_OP_RELU, in0, 1, out0, 1);
    fe_graph_add_node(&g, "n1", FE_OP_RELU, in1, 1, out1, 1);
    assert(fe_graph_validate(&g) == FE_ERR_SHAPE);
    printf("PASS test_validate_catches_double_producer\n");
}

static void test_topo_catches_cycle(void) {
    /* A -> B -> A is a cycle; topo sort must fail and order fewer than n. */
    FeGraph g;
    fe_graph_init(&g);
    int s[] = {1, 4};
    int t0 = fe_graph_add_tensor(&g, "t0", DTYPE_FLOAT32, 2, s, 0);
    int t1 = fe_graph_add_tensor(&g, "t1", DTYPE_FLOAT32, 2, s, 0);

    /* n0: consumes t0, produces t1;  n1: consumes t1, produces t0 */
    int inA[]  = {t0}; int outA[] = {t1};
    int inB[]  = {t1}; int outB[] = {t0};
    fe_graph_add_node(&g, "n0", FE_OP_RELU, inA, 1, outA, 1);
    fe_graph_add_node(&g, "n1", FE_OP_RELU, inB, 1, outB, 1);

    /* Kahn's algorithm must detect the cycle and report an error. */
    assert(fe_graph_topo_sort(&g) != FE_OK);
    printf("PASS test_topo_catches_cycle\n");
}

int main(void) {
    test_build_and_sort();
    test_topo_diamond();
    test_validate_accepts_all_supported_dtypes();
    test_validate_passes_on_valid_graph();
    test_validate_catches_out_of_range_index();
    test_validate_catches_double_producer();
    test_topo_catches_cycle();
    printf("\nAll tests passed.\n");
    return 0;
}