// tests/test_compiler.c — Stage 14: IR lowering, dead-elim, kernel selection,
// and scheduling. The engine is the oracle: a compiled program must produce
// bit-comparable output to fe_runtime_run on the same graph.
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "../core/tensor.h"
#include "../graph/graph.h"
#include "../compiler/ir.h"
#include "../runtime/engine.h"

#define WEIGHT_BUF_SIZE (1024 * 1024)
#define ACT_BUF_SIZE    (1024 * 1024)
static unsigned char weight_buf[WEIGHT_BUF_SIZE] __attribute__((aligned(64)));
static unsigned char act_buf[ACT_BUF_SIZE] __attribute__((aligned(64)));

static const float EPS = 1e-5f;

/* The two-layer MLP from test_engine: input [1,4] -> Linear -> ReLU ->
 * Linear -> Softmax -> [1,3]. Weights uniform 0.1 → softmax 1/3 each. */
static void build_mlp(FeGraph *g) {
    fe_graph_init(g);
    int s_in[]  = {1, 4};
    int s_w0[]  = {4, 8};
    int s_b0[]  = {8};
    int s_h0[]  = {1, 8};
    int s_w1[]  = {8, 3};
    int s_b1[]  = {3};
    int s_out[] = {1, 3};

    int t_in  = fe_graph_add_tensor(g, "input",  DTYPE_FLOAT32, 2, s_in,  0);
    int t_w0  = fe_graph_add_tensor(g, "w0",     DTYPE_FLOAT32, 2, s_w0,  1);
    int t_b0  = fe_graph_add_tensor(g, "b0",     DTYPE_FLOAT32, 1, s_b0,  1);
    int t_h0  = fe_graph_add_tensor(g, "h0",     DTYPE_FLOAT32, 2, s_h0,  0);
    int t_r0  = fe_graph_add_tensor(g, "relu0",  DTYPE_FLOAT32, 2, s_h0,  0);
    int t_w1  = fe_graph_add_tensor(g, "w1",     DTYPE_FLOAT32, 2, s_w1,  1);
    int t_b1  = fe_graph_add_tensor(g, "b1",     DTYPE_FLOAT32, 1, s_b1,  1);
    int t_h1  = fe_graph_add_tensor(g, "h1",     DTYPE_FLOAT32, 2, s_out, 0);
    int t_out = fe_graph_add_tensor(g, "output", DTYPE_FLOAT32, 2, s_out, 0);

    int lin0_in[] = {t_in, t_w0, t_b0};
    int relu_in[] = {t_h0};
    int lin1_in[] = {t_r0, t_w1, t_b1};
    int soft_in[] = {t_h1};

    fe_graph_add_node(g, "input",   FE_OP_INPUT,   NULL,      0, &t_in, 1);
    fe_graph_add_node(g, "linear0", FE_OP_LINEAR,  lin0_in,   3, &t_h0, 1);
    fe_graph_add_node(g, "relu0",   FE_OP_RELU,    relu_in,   1, &t_r0, 1);
    fe_graph_add_node(g, "linear1", FE_OP_LINEAR,  lin1_in,   3, &t_h1, 1);
    fe_graph_add_node(g, "softmax", FE_OP_SOFTMAX, soft_in,   1, &t_out, 1);
    fe_graph_add_node(g, "output",  FE_OP_OUTPUT,  &t_out,    1, NULL,  0);

    assert(fe_graph_topo_sort(g) == FE_OK);
    assert(fe_graph_validate(g)   == FE_OK);
}

static void seed_mlp_weights(FeGraph *g) {
    float *w0 = (float *)g->tensors[1].tensor->data;
    float *w1 = (float *)g->tensors[5].tensor->data;
    float *b0 = (float *)g->tensors[2].tensor->data;
    float *b1 = (float *)g->tensors[6].tensor->data;
    for (int i = 0; i < 32; i++) w0[i] = 0.1f;
    for (int i = 0; i < 24; i++) w1[i] = 0.1f;
    for (int i = 0; i < 8;  i++) b0[i] = 0.0f;
    for (int i = 0; i < 3;  i++) b1[i] = 0.0f;
}

static void test_compiler_lower_and_codegen(void) {
    FeGraph g;
    build_mlp(&g);

    FeIrProgram p;
    assert(fe_ir_lower(&g, &p) == FE_OK);
    assert(p.n == g.n_nodes);            /* one instruction per node */

    assert(fe_ir_codegen() == FE_OK);    /* every op resolves to a kernel */
    printf("PASS test_compiler_lower_and_codegen (%d instrs)\n", p.n);
}

static void test_compiler_matches_engine(void) {
    FeGraph g;
    build_mlp(&g);

    /* --- engine oracle --- */
    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g,
                           weight_buf, WEIGHT_BUF_SIZE,
                           act_buf,    ACT_BUF_SIZE) == FE_OK);
    assert(fe_runtime_alloc_weights(&rt) == FE_OK);
    seed_mlp_weights(&g);

    int s_in[] = {1, 4}, s_out[] = {1, 3};
    FeTensor *in = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
    float in_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(in->data, in_data, sizeof(in_data));
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_out);
    assert(fe_runtime_run(&rt, in, out) == FE_OK);

    /* --- compiler --- */
    FeIrProgram p;
    assert(fe_ir_lower(&g, &p) == FE_OK);
    assert(fe_ir_codegen() == FE_OK);

    FeTensor *live[FE_MAX_TENSORS];
    int      allocd[FE_MAX_TENSORS] = {0};
    FeIrCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.g = &g;
    ctx.tensors = live;
    ctx.input = in;

    FeTensor *out_ir = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_out);
    ctx.output = out_ir;

    for (int t = 0; t < g.n_tensors; t++) {
        FeTensorEntry *e = &g.tensors[t];
        if (e->is_weight && e->tensor) {
            live[t] = e->tensor;          /* reuse engine-backed weights */
        } else {
            live[t] = fe_tensor_alloc(e->dtype, e->ndim, e->shape);
            allocd[t] = 1;
        }
    }
    assert(fe_ir_run(&p, &ctx) == FE_OK);

    float *a = (float *)out->data, *b = (float *)out_ir->data;
    for (int i = 0; i < 3; i++) {
        assert(fabsf(a[i] - b[i]) < EPS);
        assert(fabsf(b[i] - (1.0f / 3.0f)) < EPS);
    }

    for (int t = 0; t < g.n_tensors; t++)
        if (allocd[t] && live[t] != in) fe_tensor_free(live[t]);
    fe_tensor_free(in); fe_tensor_free(out); fe_tensor_free(out_ir);
    printf("PASS test_compiler_matches_engine (IR == engine output)\n");
}

/* Dead instruction: x1 is produced but consumed by nothing and is not the
 * output — lowering then dead-elim must remove its instruction. */
static void test_ir_dead_elim(void) {
    FeGraph g;
    fe_graph_init(&g);
    int t_in  = fe_graph_add_tensor(&g, "in", DTYPE_FLOAT32, 1, (int[]){1}, 0);
    int t_a   = fe_graph_add_tensor(&g, "a",  DTYPE_FLOAT32, 1, (int[]){1}, 0);
    int t_b   = fe_graph_add_tensor(&g, "b",  DTYPE_FLOAT32, 1, (int[]){1}, 0);
    int t_c   = fe_graph_add_tensor(&g, "c",  DTYPE_FLOAT32, 1, (int[]){1}, 0);

    fe_graph_add_node(&g, "input", FE_OP_INPUT,  NULL,   0, &t_in, 1);
    fe_graph_add_node(&g, "a",     FE_OP_RELU,   &t_in,  1, &t_a,  1);  /* used */
    fe_graph_add_node(&g, "b",     FE_OP_RELU,   &t_in,  1, &t_b,  1);  /* dead */
    fe_graph_add_node(&g, "c",     FE_OP_RELU,   &t_a,   1, &t_c,  1);  /* used */
    fe_graph_add_node(&g, "output",FE_OP_OUTPUT, &t_c,   1, NULL,   0);

    FeIrProgram p;
    assert(fe_ir_lower(&g, &p) == FE_OK);
    int before = p.n;
    fe_ir_dead_elim(&p);
    assert(p.n == before - 1);   /* b-node removed, a/c retained */

    for (int i = 0; i < p.n; i++)
        assert(p.instrs[i].op != IR_NONE);   /* compacted, no holes */
    printf("PASS test_ir_dead_elim (%d -> %d instrs)\n", before, p.n);
}

/* Scheduler: producers-before-consumers with adjacency preference. Build the
 * diamond DAG, lower, schedule, verify every instruction follows its
 * producers (weights count as pre-produced). */
static void test_ir_schedule(void) {
    FeGraph g;
    fe_graph_init(&g);
    int t_in = fe_graph_add_tensor(&g, "in", DTYPE_FLOAT32, 1, (int[]){1}, 0);
    int t_b  = fe_graph_add_tensor(&g, "b",  DTYPE_FLOAT32, 1, (int[]){1}, 0);
    int t_c  = fe_graph_add_tensor(&g, "c",  DTYPE_FLOAT32, 1, (int[]){1}, 0);
    int t_d  = fe_graph_add_tensor(&g, "d",  DTYPE_FLOAT32, 1, (int[]){1}, 0);

    fe_graph_add_node(&g, "input", FE_OP_INPUT,  NULL,   0, &t_in, 1);
    fe_graph_add_node(&g, "bn",    FE_OP_RELU,   &t_in,  1, &t_b,  1);
    fe_graph_add_node(&g, "cn",    FE_OP_RELU,   &t_in,  1, &t_c,  1);
    int dins[] = {t_b, t_c};
    fe_graph_add_node(&g, "dn",    FE_OP_ADD,    dins,   2, &t_d,  1);
    fe_graph_add_node(&g, "output",FE_OP_OUTPUT, &t_d,   1, NULL,   0);

    FeIrProgram p;
    assert(fe_ir_lower(&g, &p) == FE_OK);
    int order[FE_MAX_NODES];
    assert(fe_ir_schedule(&p, order) == FE_OK);

    /* Instruction i: all producers of its srcs must appear earlier. */
    for (int pos = 0; pos < p.n; pos++) {
        int cur = order[pos];
        for (int k = 0; k < p.instrs[cur].n_srcs; k++) {
            int t = p.instrs[cur].srcs[k];
            for (int pi = 0; pi < p.n; pi++) {
                if (p.instrs[pi].dst != t) continue;
                int q;
                for (q = 0; q <= pos; q++)
                    if (order[q] == pi) break;
                assert(q <= pos);   /* producer scheduled no later than us */
            }
        }
    }
    printf("PASS test_ir_schedule (diamond, order:");
    for (int i = 0; i < p.n; i++) printf(" %d", order[i]);
    printf(")\n");
}

int main(void) {
    test_compiler_lower_and_codegen();
    test_compiler_matches_engine();
    test_ir_dead_elim();
    test_ir_schedule();
    printf("\nAll tests passed.\n");
    return 0;
}