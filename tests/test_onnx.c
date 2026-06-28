#include <stdio.h>
#include <assert.h>
#include "../core/tensor.h"
#include "../core/allocator.h"
#include "../graph/graph.h"
#include "../importer/onnx.h"

#define WEIGHT_BUF_SIZE  (1024 * 1024)
#define ACT_BUF_SIZE     (1024 * 1024)

static unsigned char weight_buf[WEIGHT_BUF_SIZE] __attribute__((aligned(64)));

static void test_load_tiny_mlp(void) {
    FeGraph g;
    FeArena weight_arena;
    fe_arena_init(&weight_arena, weight_buf, WEIGHT_BUF_SIZE);

    FeStatus s = fe_onnx_load(&g, &weight_arena, "tests/tiny_mlp.onnx");
    assert(s == FE_OK);

    /* Must have loaded some nodes and tensors */
    assert(g.n_nodes   > 0);
    assert(g.n_tensors > 0);
    assert(g.topo_valid);

    /* Weight arena must have been used */
    assert(fe_arena_used(&weight_arena) > 0);

    printf("PASS test_load_tiny_mlp (%d nodes, %d tensors)\n",
           g.n_nodes, g.n_tensors);
}

int main(void) {
    test_load_tiny_mlp();
    printf("\nAll tests passed.\n");
    return 0;
}