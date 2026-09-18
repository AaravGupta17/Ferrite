/* tests/test_opt.c — optimization pass tests (Stage 12). */
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "../core/tensor.h"
#include "../core/allocator.h"
#include "../graph/graph.h"
#include "../optim/shape_infer.h"
#include "../optim/optim.h"
#include "../runtime/engine.h"

static void expect_shape(const FeGraph *g, int idx, int ndim,
                         const int *shape) {
    const FeTensorEntry *e = &g->tensors[idx];
    assert(e->ndim == ndim);
    for (int d = 0; d < ndim; d++) assert(e->shape[d] == shape[d]);
}

/* input[1,3,6] -> conv1d k=2 -> relu -> flatten -> matmul[8,8] -> softmax */
static void test_conv1d_chain(void) {
    FeGraph g;
    fe_graph_init(&g);
    int s_in[] = {1,3,6}, s_w[]  = {4,3,2}, s_mw[] = {20,8};

    int t_in = fe_graph_add_tensor(&g, "in", DTYPE_FLOAT32, 3, s_in,  0);
    int t_w  = fe_graph_add_tensor(&g, "w", DTYPE_FLOAT32, 3, s_w,   1);
    int t_h  = fe_graph_add_tensor(&g, "h", DTYPE_FLOAT32, 0, NULL,  0);
    int t_f  = fe_graph_add_tensor(&g, "f", DTYPE_FLOAT32, 0, NULL,  0);
    int t_mw = fe_graph_add_tensor(&g, "mw", DTYPE_FLOAT32, 2, s_mw, 1);
    int t_y  = fe_graph_add_tensor(&g, "y", DTYPE_FLOAT32, 0, NULL,  0);
    int t_z  = fe_graph_add_tensor(&g, "z", DTYPE_FLOAT32, 0, NULL,  0);
    int t_s  = fe_graph_add_tensor(&g, "s", DTYPE_FLOAT32, 0, NULL,  0);

    int a1_in[] = {t_in, t_w};      int a1_out[] = {t_h};
    int a2_in[] = {t_h};            int a2_out[] = {t_f};
    int a3_in[] = {t_f};            int a3_out[] = {t_y};
    int a4_in[] = {t_y, t_mw};      int a4_out[] = {t_z};
    int a5_in[] = {t_z};            int a5_out[] = {t_s};

    fe_graph_add_node(&g, "input",  FE_OP_INPUT,  NULL, 0, &t_in, 1);
    int c1 = fe_graph_add_node(&g, "conv1", FE_OP_CONV1D,   a1_in, 2, a1_out, 1);
    int c2 = fe_graph_add_node(&g, "relu",  FE_OP_RELU,     a2_in, 1, a2_out, 1);
    int c3 = fe_graph_add_node(&g, "flat",  FE_OP_FLATTEN,  a3_in, 1, a3_out, 1);
    int c4 = fe_graph_add_node(&g, "mm",    FE_OP_MATMUL,   a4_in, 2, a4_out, 1);
    int c5 = fe_graph_add_node(&g, "sm",    FE_OP_SOFTMAX,  a5_in, 1, a5_out, 1);

    g.nodes[c1].attrs.conv1d.stride = 1;
    g.nodes[c1].attrs.conv1d.pad = 0;

    assert(fe_infer_shapes(&g) == FE_OK);

    int e_h[] = {1, 4, 5};    /* (6-2)/1+1 */
    int e_f[] = {1, 4, 5};    /* relu keeps conv shape */
    int e_y[] = {1, 20};      /* flatten: [1, 20] */
    int e_z[] = {1, 8};
    int e_s[] = {1, 8};
    expect_shape(&g, t_h, 3, e_h);
    expect_shape(&g, t_f, 3, e_f);
    expect_shape(&g, t_y, 2, e_y);
    expect_shape(&g, t_z, 2, e_z);
    expect_shape(&g, t_s, 2, e_s);

    /* Run twice: already-known outputs must be left untouched. */
    assert(fe_infer_shapes(&g) == FE_OK);
    expect_shape(&g, t_h, 3, e_h);
    (void)c2; (void)c3; (void)c4; (void)c5;
    printf("PASS test_conv1d_chain\n");
}

static void test_conv2d_pool(void) {
    FeGraph g;
    fe_graph_init(&g);
    int s_in[] = {2, 3, 16, 16};                      /* [N,C,H,W] */
    int s_w[]  = {8, 3, 3, 3};
    int s_c  = fe_graph_add_tensor(&g, "c", DTYPE_FLOAT32, 0, NULL, 0);
    int t_in = fe_graph_add_tensor(&g, "in", DTYPE_FLOAT32, 4, s_in, 0);
    int t_w  = fe_graph_add_tensor(&g, "w", DTYPE_FLOAT32, 4, s_w,  1);
    int t_p  = fe_graph_add_tensor(&g, "p", DTYPE_FLOAT32, 0, NULL, 0);

    int a1_in[] = {t_in, t_w}; int a1_out[] = {s_c};
    int a2_in[] = {s_c};       int a2_out[] = {t_p};
    int c1 = fe_graph_add_node(&g, "conv2", FE_OP_CONV2D,  a1_in, 2, a1_out, 1);
    int c2 = fe_graph_add_node(&g, "pool",  FE_OP_MAXPOOL, a2_in, 1, a2_out, 1);

    g.nodes[c1].attrs.conv2d.stride_h = 1;
    g.nodes[c1].attrs.conv2d.stride_w = 1;
    g.nodes[c1].attrs.conv2d.pad_h = 1;
    g.nodes[c1].attrs.conv2d.pad_w = 1;
    g.nodes[c2].attrs.pool.kh = 2;
    g.nodes[c2].attrs.pool.kw = 2;
    g.nodes[c2].attrs.pool.sh = 2;
    g.nodes[c2].attrs.pool.sw = 2;

    assert(fe_infer_shapes(&g) == FE_OK);
    int e_c[] = {2, 8, 16, 16};   /* (16-3+2)/1+1 */
    int e_p[] = {2, 8, 8, 8};     /* (16-2)/2+1 */
    expect_shape(&g, s_c, 4, e_c);
    expect_shape(&g, t_p, 4, e_p);
    printf("PASS test_conv2d_pool\n");
}

static void test_broadcast_gemm(void) {
    FeGraph g;
    fe_graph_init(&g);
    /* add: [8,3] + [8,3]; pow: [4,4] ^ scalar-shaped [1]; gemm: W^T path */
    int s_a[] = {8,3}, s_b[] = {8,3};
    int s_p0[] = {4,4}, s_p1[] = {1};
    int s_gA[] = {2,4}, s_gB[] = {2,5};

    int t_a  = fe_graph_add_tensor(&g, "a",  DTYPE_FLOAT32, 2, s_a,   0);
    int t_b  = fe_graph_add_tensor(&g, "b",  DTYPE_FLOAT32, 2, s_b,   0);
    int t_sum = fe_graph_add_tensor(&g, "sum", DTYPE_FLOAT32, 0, NULL, 0);
    int t_p0 = fe_graph_add_tensor(&g, "p0", DTYPE_FLOAT32, 2, s_p0,  0);
    int t_p1 = fe_graph_add_tensor(&g, "p1", DTYPE_FLOAT32, 1, s_p1,  0);
    int t_pw = fe_graph_add_tensor(&g, "pw", DTYPE_FLOAT32, 0, NULL,  0);
    int t_gA = fe_graph_add_tensor(&g, "gA", DTYPE_FLOAT32, 2, s_gA,  0);
    int t_gB = fe_graph_add_tensor(&g, "gB", DTYPE_FLOAT32, 2, s_gB,  0);
    int t_g  = fe_graph_add_tensor(&g, "g",  DTYPE_FLOAT32, 0, NULL,  0);

    int q1_in[] = {t_a, t_b};   int q1_out[] = {t_sum};
    int q2_in[] = {t_p0, t_p1}; int q2_out[] = {t_pw};
    int q3_in[] = {t_gA, t_gB}; int q3_out[] = {t_g};
    int e1 = fe_graph_add_node(&g, "add",  FE_OP_ADD,  q1_in, 2, q1_out, 1);
    int e2 = fe_graph_add_node(&g, "pow",  FE_OP_POW,  q2_in, 2, q2_out, 1);
    int e3 = fe_graph_add_node(&g, "gemm", FE_OP_GEMM, q3_in, 2, q3_out, 1);

    /* W^T: A=[2,4] transposed -> op(A)=[4,2] (M=4,K=2); B=[2,5] -> [K=2,N=5]; out [4,5] */
    g.nodes[e3].attrs.gemm.transA = 1;
    g.nodes[e3].attrs.gemm.transB = 0;
    g.nodes[e3].attrs.gemm.alpha = 1.0f;
    g.nodes[e3].attrs.gemm.beta = 0.0f;

    assert(fe_infer_shapes(&g) == FE_OK);
    int e_sum[] = {8, 3};
    expect_shape(&g, t_sum, 2, e_sum);
    int e_pw[]  = {4, 4};   /* scalar broadcasts to full shape */
    expect_shape(&g, t_pw, 2, e_pw);
    int e_g[]   = {4, 5};
    expect_shape(&g, t_g, 2, e_g);
    (void)e1; (void)e2;
    printf("PASS test_broadcast_gemm\n");
}

static void test_embedding_mha(void) {
    FeGraph g;
    fe_graph_init(&g);
    int s_idx[]   = {2, 10};                       /* [batch, seq] */
    int s_tab[]   = {128, 32};                     /* [vocab, d_model] */
    int s_x[]     = {2, 10, 32};                   /* [batch, seq, d_model] */

    int t_idx = fe_graph_add_tensor(&g, "idx", DTYPE_INT32, 2, s_idx,  0);
    int t_tab = fe_graph_add_tensor(&g, "tab", DTYPE_FLOAT32, 2, s_tab, 1);
    int t_emb = fe_graph_add_tensor(&g, "emb", DTYPE_FLOAT32, 0, NULL,  0);
    int t_x   = fe_graph_add_tensor(&g, "x",   DTYPE_FLOAT32, 3, s_x,   0);
    int t_mha = fe_graph_add_tensor(&g, "mha", DTYPE_FLOAT32, 0, NULL,  0);

    int r1_in[] = {t_idx, t_tab}; int r1_out[] = {t_emb};
    int r2_in[] = {t_x};         int r2_out[] = {t_mha};
    int f1 = fe_graph_add_node(&g, "emb", FE_OP_EMBEDDING,       r1_in, 2, r1_out, 1);
    int f2 = fe_graph_add_node(&g, "mha", FE_OP_MULTIHEAD_ATTN,  r2_in, 1, r2_out, 1);

    assert(fe_infer_shapes(&g) == FE_OK);
    int e_emb[] = {2, 10, 32};
    int e_mha[] = {2, 10, 32};
    expect_shape(&g, t_emb, 3, e_emb);
    expect_shape(&g, t_mha, 3, e_mha);
    (void)f1; (void)f2;
    printf("PASS test_embedding_mha\n");
}

static void test_matmul_k_mismatch(void) {
    FeGraph g;
    fe_graph_init(&g);
    int s_a[] = {2, 4}, s_b[] = {5, 3};
    int t_a = fe_graph_add_tensor(&g, "a", DTYPE_FLOAT32, 2, s_a, 0);
    int t_b = fe_graph_add_tensor(&g, "b", DTYPE_FLOAT32, 2, s_b, 0);
    int t_o = fe_graph_add_tensor(&g, "o", DTYPE_FLOAT32, 0, NULL, 0);
    int q_in[] = {t_a, t_b}; int q_out[] = {t_o};
    fe_graph_add_node(&g, "mm", FE_OP_MATMUL, q_in, 2, q_out, 1);
    assert(fe_infer_shapes(&g) == FE_ERR_SHAPE);
    printf("PASS test_matmul_k_mismatch\n");
}

/* ------------------------------------------------------------------ */
/* Stage 12 pass tests                                                */

static FeTensor *make_tensor_vec(FeDtype dt, int ndim, const int *shape,
                                 const float *vals, int n) {
    FeTensor *t = fe_tensor_alloc(dt, ndim, shape);
    assert(t);
    memcpy(t->data, vals, n * sizeof(float));
    return t;
}

/* Convolution of a fully constant subgraph: Transpose(w) with w a weight. */
static void test_fold_transpose_weight(void) {
    FeGraph g;
    fe_graph_init(&g);
    static unsigned char wbuf[4096] __attribute__((aligned(64)));
    FeArena arena;
    fe_arena_init(&arena, wbuf, sizeof(wbuf));

    int s_w[] = {2, 2};
    int t_w   = fe_graph_add_tensor(&g, "w",   DTYPE_FLOAT32, 2, s_w, 0);
    int t_out = fe_graph_add_tensor(&g, "out", DTYPE_FLOAT32, 2, s_w, 0);
    float wdata[] = {1.f, 2.f, 3.f, 4.f};
    g.tensors[t_w].is_weight = 1;
    g.tensors[t_w].tensor = make_tensor_vec(DTYPE_FLOAT32, 2, s_w, wdata, 4);

    int q_in[] = {t_w}; int q_out[] = {t_out};
    int n_t = fe_graph_add_node(&g, "T", FE_OP_TRANSPOSE, q_in, 1, q_out, 1);

    assert(fe_pass_fold_constants(&g, &arena) == FE_OK);
    assert(g.nodes[n_t].op == FE_OP_INPUT);
    assert(g.tensors[t_out].is_weight == 1);
    assert(g.tensors[t_out].tensor && g.tensors[t_out].tensor->data);
    float *out = (float *)g.tensors[t_out].tensor->data;
    assert(out[0] == 1.f && out[1] == 3.f && out[2] == 2.f && out[3] == 4.f);
    fe_tensor_free(g.tensors[t_w].tensor);
    printf("PASS test_fold_transpose_weight\n");
}

/* Transpose(Transpose(x)) == x, and Flatten of a 2D tensor is identity. */
static void test_simplify_identities(void) {
    FeGraph g;
    fe_graph_init(&g);
    int s_x[] = {3, 4};
    int t_x = fe_graph_add_tensor(&g, "x",  DTYPE_FLOAT32, 2, s_x, 0);
    int t_m = fe_graph_add_tensor(&g, "mid",DTYPE_FLOAT32, 2, s_x, 0);
    int t_o = fe_graph_add_tensor(&g, "o2", DTYPE_FLOAT32, 2, s_x, 0);
    int t_f = fe_graph_add_tensor(&g, "f",  DTYPE_FLOAT32, 2, s_x, 0);
    int t_y = fe_graph_add_tensor(&g, "y",  DTYPE_FLOAT32, 2, s_x, 0);

    int a_in[] = {t_x}; int a_out[] = {t_m};
    int b_in[] = {t_m}; int b_out[] = {t_o};
    int c_in[] = {t_o}; int c_out[] = {t_f};
    int d_in[] = {t_f}; int d_out[] = {t_y};
    /* flatten of a 2D input is also an identity */
    fe_graph_add_node(&g, "T1", FE_OP_TRANSPOSE, a_in, 1, a_out, 1);
    int n_t2 = fe_graph_add_node(&g, "T2", FE_OP_TRANSPOSE, b_in, 1, b_out, 1);
    int n_f  = fe_graph_add_node(&g, "flat", FE_OP_FLATTEN, c_in, 1, c_out, 1);
    int n_r  = fe_graph_add_node(&g, "relu", FE_OP_RELU,    d_in, 1, d_out, 1);

    assert(fe_pass_simplify(&g) == FE_OK);
    assert(g.nodes[n_t2].op == FE_OP_INPUT);
    assert(g.nodes[n_f].op  == FE_OP_INPUT);
    /* the consumer chain reads x directly after both identities */
    assert(g.nodes[n_r].inputs[0] == t_x);

    /* Both neutered feeds and T1's unused output are reclaimed. */
    assert(fe_pass_dead_elim(&g) == FE_OK);
    assert(g.n_nodes == 1);       /* only the relu survives */
    assert(g.n_tensors == 2);     /* x and y */
    assert(fe_graph_topo_sort(&g) == FE_OK);
    assert(fe_graph_validate(&g) == FE_OK);
    printf("PASS test_simplify_identities\n");
}

/* Two identical RELU(x) nodes collapse into one. */
static void test_cse_dedup(void) {
    FeGraph g;
    fe_graph_init(&g);
    int s[] = {1, 2};
    int t_x = fe_graph_add_tensor(&g, "x",  DTYPE_FLOAT32, 2, s, 0);
    int t_r1 = fe_graph_add_tensor(&g, "r1", DTYPE_FLOAT32, 2, s, 0);
    int t_r2 = fe_graph_add_tensor(&g, "r2", DTYPE_FLOAT32, 2, s, 0);
    int t_s  = fe_graph_add_tensor(&g, "s",  DTYPE_FLOAT32, 2, s, 0);

    int a_in[] = {t_x}; int a_out[] = {t_r1};
    int b_in[] = {t_x}; int b_out[] = {t_r2};
    int c_in[] = {t_r1, t_r2}; int c_out[] = {t_s};
    fe_graph_add_node(&g, "in",    FE_OP_INPUT, NULL, 0, &t_x, 1);
    fe_graph_add_node(&g, "relu1", FE_OP_RELU,  a_in, 1, a_out, 1);
    int n_r2 = fe_graph_add_node(&g, "relu2", FE_OP_RELU, b_in, 1, b_out, 1);
    int n_add = fe_graph_add_node(&g, "add",   FE_OP_ADD,  c_in, 2, c_out, 1);

    assert(fe_pass_cse(&g) == FE_OK);
    assert(g.nodes[n_r2].op == FE_OP_INPUT);
    assert(g.nodes[n_add].inputs[0] == t_r1);
    assert(g.nodes[n_add].inputs[1] == t_r1);
    assert(fe_graph_topo_sort(&g) == FE_OK);
    printf("PASS test_cse_dedup\n");
}

/* Unused compute chains and unconsumed weights are reclaimed. */
static void test_dead_elim(void) {
    FeGraph g;
    fe_graph_init(&g);
    int s[] = {1, 2};
    int t_x = fe_graph_add_tensor(&g, "x",   DTYPE_FLOAT32, 2, s, 0);
    int t_h = fe_graph_add_tensor(&g, "h",   DTYPE_FLOAT32, 2, s, 0);
    int t_w = fe_graph_add_tensor(&g, "w",   DTYPE_FLOAT32, 2, s, 1);
    int t_d = fe_graph_add_tensor(&g, "d",   DTYPE_FLOAT32, 2, s, 0);
    int t_y = fe_graph_add_tensor(&g, "y",   DTYPE_FLOAT32, 2, s, 0);

    int a_in[] = {t_x, t_x}; int a_out[] = {t_h};
    int b_in[] = {t_x, t_w}; int b_out[] = {t_d};
    int c_in[] = {t_h}; int c_out[] = {t_y};
    fe_graph_add_node(&g, "add",  FE_OP_ADD, a_in, 2, a_out, 1);
    fe_graph_add_node(&g, "sub",  FE_OP_SUB, b_in, 2, b_out, 1);
    fe_graph_add_node(&g, "relu", FE_OP_RELU, c_in, 1, c_out, 1);

    assert(fe_pass_dead_elim(&g) == FE_OK);
    assert(g.n_nodes == 2);      /* add, relu — the sub chain is dead */
    assert(g.n_tensors == 3);    /* x, h, y — w and d reclaimed */
    assert(fe_graph_topo_sort(&g) == FE_OK);
    assert(fe_graph_validate(&g) == FE_OK);
    printf("PASS test_dead_elim\n");
}

/* The graph output itself is an identity Transpose∘Transpose chain with no
 * consumer of the outer output. simplify must NOT neuter the outer node: a
 * graph output has no consumers to relink, so the node has to stay alive to
 * keep producing it. The old bug rewrote it into an INPUT feed anyway,
 * leaving the output with no producer (uninitialized data at run time). */
static void test_terminal_identity_transpose_survives(void) {
    FeGraph g;
    fe_graph_init(&g);
    int s_x[] = {3, 4};
    int t_x = fe_graph_add_tensor(&g, "x",  DTYPE_FLOAT32, 2, s_x, 0);
    int t_m = fe_graph_add_tensor(&g, "mid",DTYPE_FLOAT32, 2, s_x, 0);
    int t_o = fe_graph_add_tensor(&g, "out",DTYPE_FLOAT32, 2, s_x, 0);

    int a_in[] = {t_x}; int a_out[] = {t_m};
    int b_in[] = {t_m}; int b_out[] = {t_o};
    fe_graph_add_node(&g, "T1", FE_OP_TRANSPOSE, a_in, 1, a_out, 1);
    int n_t2 = fe_graph_add_node(&g, "T2", FE_OP_TRANSPOSE, b_in, 1, b_out, 1);

    assert(fe_pass_simplify(&g) == FE_OK);

    /* The outer node must still be a producing TRANSPOSE, not a neutered
     * INPUT feed, and its output must still be written by it. */
    assert(g.nodes[n_t2].op == FE_OP_TRANSPOSE);
    assert(g.nodes[n_t2].n_inputs == 1);

    int produced_by_output = 0;
    for (int i = 0; i < g.n_nodes; i++)
        for (int o = 0; o < g.nodes[i].n_outputs; o++)
            if (g.nodes[i].outputs[o] == t_o) produced_by_output++;
    assert(produced_by_output >= 1);

    printf("PASS test_terminal_identity_transpose_survives\n");
}

/* Zero-extent constant feeding FLATTEN: the fold must fail loudly
 * (FE_ERR_SHAPE), not integer-divide-by-zero when computing the second
 * reshape dimension. */
static void test_fold_flatten_zero_extent(void) {
    FeGraph g;
    fe_graph_init(&g);
    static unsigned char wbuf[4096] __attribute__((aligned(64)));
    FeArena arena;
    fe_arena_init(&arena, wbuf, sizeof(wbuf));

    int s0[] = {0, 4};
    int t_w   = fe_graph_add_tensor(&g, "w",   DTYPE_FLOAT32, 2, s0, 0);
    int t_out = fe_graph_add_tensor(&g, "out", DTYPE_FLOAT32, 2, s0, 0);
    static float dummy;   /* zero-extent tensor: data is unused, never NULL */
    g.tensors[t_w].is_weight = 1;
    g.tensors[t_w].tensor = fe_tensor_from_data(&dummy, DTYPE_FLOAT32, 2, s0);
    assert(g.tensors[t_w].tensor);

    int q_in[] = {t_w}; int q_out[] = {t_out};
    fe_graph_add_node(&g, "flat", FE_OP_FLATTEN, q_in, 1, q_out, 1);

    /* shape[0] == 0 → the old code divided by zero; the guard turns it into
     * a loud FE_ERR_SHAPE instead. */
    assert(fe_pass_fold_constants(&g, &arena) == FE_ERR_SHAPE);
    fe_tensor_free(g.tensors[t_w].tensor);   /* from_data mallocs the struct */
    printf("PASS test_fold_flatten_zero_extent\n");
}

/* ------------------------------------------------------------------ */
/* End-to-end: Conv+BN before vs after fe_optimize                     */

static void build_conv_bn(FeGraph *g) {
    fe_graph_init(g);
    int s_in[] = {1, 2, 5};
    int s_w[]  = {3, 2, 3};
    int s_c[]  = {3};
    int s_h[]  = {1, 3, 3};

    int t_in = fe_graph_add_tensor(g, "in",  DTYPE_FLOAT32, 3, s_in, 0);
    int t_w  = fe_graph_add_tensor(g, "w",   DTYPE_FLOAT32, 3, s_w,  1);
    int t_c  = fe_graph_add_tensor(g, "c",   DTYPE_FLOAT32, 3, s_h,  0);
    int t_g  = fe_graph_add_tensor(g, "g",   DTYPE_FLOAT32, 1, s_c,  1);
    int t_b  = fe_graph_add_tensor(g, "b",   DTYPE_FLOAT32, 1, s_c,  1);
    int t_m  = fe_graph_add_tensor(g, "m",   DTYPE_FLOAT32, 1, s_c,  1);
    int t_v  = fe_graph_add_tensor(g, "v",   DTYPE_FLOAT32, 1, s_c,  1);
    int t_n  = fe_graph_add_tensor(g, "n",   DTYPE_FLOAT32, 3, s_h,  0);
    int t_y  = fe_graph_add_tensor(g, "y",   DTYPE_FLOAT32, 3, s_h,  0);

    int cv_in[] = {t_in, t_w};  int cv_out[] = {t_c};
    int bn_in[] = {t_c, t_g, t_b, t_m, t_v}; int bn_out[] = {t_n};
    int rl_in[] = {t_n};        int rl_out[] = {t_y};

    fe_graph_add_node(g, "input",  FE_OP_INPUT,     NULL, 0, &t_in, 1);
    int n_cv = fe_graph_add_node(g, "conv",  FE_OP_CONV1D,    cv_in, 2, cv_out, 1);
    g->nodes[n_cv].attrs.conv1d.stride = 1;
    g->nodes[n_cv].attrs.conv1d.pad = 0;
    int n_bn = fe_graph_add_node(g, "bn",    FE_OP_BATCHNORM, bn_in, 5, bn_out, 1);
    g->nodes[n_bn].attrs.batchnorm.eps = 1e-5f;
    fe_graph_add_node(g, "relu",  FE_OP_RELU,     rl_in, 1, rl_out, 1);
    fe_graph_add_node(g, "output", FE_OP_OUTPUT,   &t_y,  1, NULL,  0);
}

static int find_tensor(const FeGraph *g, const char *name) {
    for (int i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return i;
    return -1;
}

/* Fill the (pre-fusion) BN coefficient and conv weight tensors by name.
 * Post-fusion graphs have these entries compacted away — the lookup then
 * simply fails and the loop is a no-op, exactly as intended. */
static void fill_conv_bn_weights(FeGraph *g) {
    int t_w = find_tensor(g, "w");
    int t_g = find_tensor(g, "g");
    int t_b = find_tensor(g, "b");
    int t_m = find_tensor(g, "m");
    int t_v = find_tensor(g, "v");
    if (t_w >= 0) {
        float w[] = { 0.5f,-0.2f, 0.9f,  0.1f, 0.3f,-0.4f,
                      0.2f, 0.7f,-0.1f,  0.8f,-0.3f, 0.6f,
                      0.4f, 0.1f, 0.2f, -0.5f, 0.9f, 0.7f };
        assert(g->tensors[t_w].tensor && g->tensors[t_w].tensor->data);
        memcpy(g->tensors[t_w].tensor->data, w, sizeof(w));
    }
    if (t_g >= 0 && t_b >= 0 && t_m >= 0 && t_v >= 0) {
        float g_[] = {1.2f, 0.8f, 1.5f};
        float b_[] = {0.1f,-0.2f, 0.3f};
        float m_[] = { 0.5f, 1.0f,-0.5f};
        float v_[] = { 1.0f, 2.0f, 0.5f};
        memcpy(g->tensors[t_g].tensor->data, g_, sizeof(g_));
        memcpy(g->tensors[t_b].tensor->data, b_, sizeof(b_));
        memcpy(g->tensors[t_m].tensor->data, m_, sizeof(m_));
        memcpy(g->tensors[t_v].tensor->data, v_, sizeof(v_));
    }
}

static void run_graph(const FeGraph *src, const float *in, int in_n,
                      float *out, int out_n) {
    FeGraph g = *src;
    static unsigned char wbuf[128 * 1024] __attribute__((aligned(64)));
    static unsigned char abuf[128 * 1024] __attribute__((aligned(64)));
    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g, wbuf, sizeof(wbuf), abuf, sizeof(abuf))
           == FE_OK);
    assert(fe_runtime_alloc_weights(&rt) == FE_OK);
    fill_conv_bn_weights(&g);

    int s_in[] = {1, 2, 5}, s_out[] = {1, 3, 3};
    FeTensor *input = fe_tensor_alloc(DTYPE_FLOAT32, 3, s_in);
    memcpy(input->data, in, in_n * sizeof(float));
    FeTensor *output = fe_tensor_alloc(DTYPE_FLOAT32, 3, s_out);
    assert(fe_runtime_run(&rt, input, output) == FE_OK);
    memcpy(out, output->data, out_n * sizeof(float));

    fe_tensor_free(input);
    fe_tensor_free(output);
}

static void test_conv_bn_equivalence(void) {
    FeGraph g1, g2;
    build_conv_bn(&g1);
    g2 = g1;

    static unsigned char wa2[128 * 1024] __attribute__((aligned(64)));
    FeArena arena2;
    fe_arena_init(&arena2, wa2, sizeof(wa2));

    /* Materialize g2's weights so fusion can read them, then optimize. */
    for (int i = 0; i < g2.n_tensors; i++)
        if (g2.tensors[i].is_weight)
            g2.tensors[i].tensor =
                fe_arena_alloc_tensor(&arena2, g2.tensors[i].dtype,
                                      g2.tensors[i].ndim, g2.tensors[i].shape);
    fill_conv_bn_weights(&g2);
    assert(fe_optimize(&g2, &arena2) == FE_OK);

    /* Fusion must have fired: the BN is gone entirely (relinked into the
     * conv and reclaimed by dead-elim), the conv now carries fused weights
     * + bias, and the graph has exactly two compute ops left. */
    int compute = 0, bn_nodes = 0;
    for (int i = 0; i < g2.n_nodes; i++) {
        if (g2.nodes[i].op == FE_OP_BATCHNORM) bn_nodes++;
        else if (g2.nodes[i].op != FE_OP_INPUT && g2.nodes[i].op != FE_OP_OUTPUT)
            compute++;
    }
    assert(bn_nodes == 0);
    assert(compute == 2);   /* just conv + relu now */
    for (int i = 0; i < g2.n_nodes; i++)
        if (g2.nodes[i].op == FE_OP_CONV1D)
            assert(g2.nodes[i].n_inputs == 3);

    float in[10];
    for (int i = 0; i < 10; i++) in[i] = 0.3f * (i + 1) + (i % 3) * 0.1f;

    float out1[9], out2[9];
    run_graph(&g1, in, 10, out1, 9);
    run_graph(&g2, in, 10, out2, 9);
    for (int i = 0; i < 9; i++)
        assert(fabsf(out1[i] - out2[i]) < 1e-4f);
    printf("PASS test_conv_bn_equivalence\n");
}

int main(void) {
    test_conv1d_chain();
    test_conv2d_pool();
    test_broadcast_gemm();
    test_embedding_mha();
    test_matmul_k_mismatch();
    test_fold_transpose_weight();
    test_simplify_identities();
    test_terminal_identity_transpose_survives();
    test_fold_flatten_zero_extent();
    test_cse_dedup();
    test_dead_elim();
    test_conv_bn_equivalence();
    printf("\nAll tests passed.\n");
    return 0;
}