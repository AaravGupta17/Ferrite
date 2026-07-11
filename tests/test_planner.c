// tests/test_planner.c
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "../core/tensor.h"
#include "../graph/graph.h"
#include "../planner/memory_planner.h"
#include "../core/allocator.h"


/*
 * Test graph: linear chain
 *
 *   input[1,4] -> MatMul -> Add -> Relu -> Softmax -> output[1,3]
 *
 * Tensors h0, h1, h2 are intermediate activations.
 * h0 and h2 lifetimes don't overlap — planner should reuse h0's buffer.
 */
static void test_linear_chain_reuse(void) {
    FeGraph g;
    fe_graph_init(&g);

    int s4[]  = {1, 4};
    int s48[] = {4, 8};
    int s8[]  = {8};
    int s18[] = {1, 8};
    int s83[] = {8, 3};
    int s3[]  = {3};
    int s13[] = {1, 3};

    int t_in  = fe_graph_add_tensor(&g, "input",  DTYPE_FLOAT32, 2, s4,  0);
    int t_w0  = fe_graph_add_tensor(&g, "w0",     DTYPE_FLOAT32, 2, s48, 1);
    int t_b0  = fe_graph_add_tensor(&g, "b0",     DTYPE_FLOAT32, 1, s8,  1);
    int t_h0  = fe_graph_add_tensor(&g, "h0",     DTYPE_FLOAT32, 2, s18, 0);
    int t_h1  = fe_graph_add_tensor(&g, "h1",     DTYPE_FLOAT32, 2, s18, 0);
    int t_w1  = fe_graph_add_tensor(&g, "w1",     DTYPE_FLOAT32, 2, s83, 1);
    int t_b1  = fe_graph_add_tensor(&g, "b1",     DTYPE_FLOAT32, 1, s3,  1);
    int t_h2  = fe_graph_add_tensor(&g, "h2",     DTYPE_FLOAT32, 2, s13, 0);
    int t_out = fe_graph_add_tensor(&g, "output", DTYPE_FLOAT32, 2, s13, 0);

    int lin0_in[]  = {t_in, t_w0, t_b0}; int lin0_out[] = {t_h0};
    int relu_in[]  = {t_h0};              int relu_out[] = {t_h1};
    int lin1_in[]  = {t_h1, t_w1, t_b1}; int lin1_out[] = {t_h2};
    int soft_in[]  = {t_h2};              int soft_out[] = {t_out};

    fe_graph_add_node(&g, "input",   FE_OP_INPUT,   NULL,     0, &t_in, 1);
    fe_graph_add_node(&g, "linear0", FE_OP_LINEAR,  lin0_in,  3, lin0_out, 1);
    fe_graph_add_node(&g, "relu0",   FE_OP_RELU,    relu_in,  1, relu_out, 1);
    fe_graph_add_node(&g, "linear1", FE_OP_LINEAR,  lin1_in,  3, lin1_out, 1);
    fe_graph_add_node(&g, "softmax", FE_OP_SOFTMAX, soft_in,  1, soft_out, 1);
    fe_graph_add_node(&g, "output",  FE_OP_OUTPUT,  &t_out,   1, NULL,     0);

    assert(fe_graph_topo_sort(&g) == FE_OK);

    FePlan plan;
    assert(fe_plan_memory(&g, &plan) == FE_OK);

    fe_plan_print(&g, &plan);

    /*
     * h0 is [1,8] = 32 bytes. last_use = step 1 (relu consumes it).
     * h2 is [1,3] = 12 bytes. first_use = step 3.
     * h2 fits in h0's slot — planner must reuse it.
     *
     * Verify: planned total < naive total (reuse happened).
     */
    size_t naive = 0;
    for (int i = 0; i < plan.n_lifetimes; i++) {
        size_t s = (size_t)plan.lifetimes[i].size_bytes;
        naive += (s + 63) & ~(size_t)63;
    }
    assert(plan.total_activation_bytes < naive);
    printf("PASS test_linear_chain_reuse (saved %zu bytes)\n",
           naive - plan.total_activation_bytes);
}
static void test_plan_apply(void) {
    FeGraph g;
    fe_graph_init(&g);

    int s4[]  = {1, 4};
    int s18[] = {1, 8};
    int s48[] = {4, 8};
    int s8[]  = {8};

    int t_in = fe_graph_add_tensor(&g, "input",  DTYPE_FLOAT32, 2, s4,  0);
    int t_w  = fe_graph_add_tensor(&g, "weight", DTYPE_FLOAT32, 2, s48, 1);
    int t_b  = fe_graph_add_tensor(&g, "bias",   DTYPE_FLOAT32, 1, s8,  1);
    int t_h  = fe_graph_add_tensor(&g, "h",      DTYPE_FLOAT32, 2, s18, 0);

    int lin_in[]  = {t_in, t_w, t_b};
    int lin_out[] = {t_h};
    fe_graph_add_node(&g, "input",  FE_OP_INPUT,  NULL,    0, &t_in, 1);
    fe_graph_add_node(&g, "linear", FE_OP_LINEAR, lin_in,  3, lin_out, 1);
    fe_graph_add_node(&g, "output", FE_OP_OUTPUT, &t_h,    1, NULL,    0);

    fe_graph_topo_sort(&g);

    FePlan plan;
    fe_plan_memory(&g, &plan);

    /* Arena for tensor metadata — no heap allocation */
    static unsigned char meta_buf[4096] __attribute__((aligned(64)));
    FeArena meta_arena;
    fe_arena_init(&meta_arena, meta_buf, sizeof(meta_buf));

    /* Activation buffer — static, no malloc */
    static unsigned char act_buf[4096] __attribute__((aligned(64)));

    assert(fe_plan_apply(&g, &plan, act_buf, sizeof(act_buf), &meta_arena)
           == FE_OK);

    char *buf_start = (char *)act_buf;
    char *buf_end   = buf_start + plan.total_activation_bytes + 64;
    char *tptr      = (char *)g.tensors[t_h].tensor->data;
    assert(g.tensors[t_h].tensor != NULL);
    assert(tptr >= buf_start);
    assert(tptr <  buf_end);

    printf("PASS test_plan_apply\n");
}
int main(void) {
    test_linear_chain_reuse();
    test_plan_apply();
    printf("\nAll tests passed.\n");
    return 0;
}