// tests/test_parallel.c — Stage 11: thread pool, parallel GEMM, DAG scheduler.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>
#include "../core/tensor.h"
#include "../ops/ops.h"
#include "../parallel/threadpool.h"
#include "../parallel/parallel_gemm.h"
#include "../parallel/scheduler.h"

/* ---- pool: N disjoint slots, one worker each; sum must equal N ---- */

typedef struct {
    int *slots;
    int  n;
} SlotArgs;

static void mark_slot(void *ctx, int index) {
    SlotArgs *s = (SlotArgs *)ctx;
    s->slots[index] = index + 1;
}

static void test_pool_partition(void) {
    FeThreadPool tp;
    assert(fe_threadpool_init(&tp, 4) == FE_OK);

    const int N = 1000;
    int *slots = calloc(N, sizeof(int));
    SlotArgs a = { slots, N };
    for (int i = 0; i < N; i++)
        assert(fe_threadpool_submit(&tp, mark_slot, &a, i) == FE_OK);
    assert(fe_threadpool_wait(&tp) == FE_OK);

    int sum = 0;
    for (int i = 0; i < N; i++) sum += slots[i];
    assert(sum == N * (N + 1) / 2);   /* every slot touched by one worker */

    fe_threadpool_destroy(&tp);
    free(slots);
    printf("PASS test_pool_partition (4 threads, %d tasks)\n", N);
}

/* ---- parallel GEMM must match the sequential oracle ---- */

static float next_rand(unsigned *s) {
    *s = *s * 1664525u + 1013904223u;
    return (float)((*s >> 8) & 0xFFFF) / 256.0f - 128.0f;
}

static void test_parallel_gemm_matches_sequential(void) {
    int M = 37, K = 17, N = 23;
    int sA[] = {M, K}, sB[] = {K, N}, sC[] = {M, N};
    FeTensor *A = fe_tensor_alloc(DTYPE_FLOAT32, 2, sA);
    FeTensor *B = fe_tensor_alloc(DTYPE_FLOAT32, 2, sB);
    FeTensor *C1 = fe_tensor_alloc(DTYPE_FLOAT32, 2, sC);
    FeTensor *C2 = fe_tensor_alloc(DTYPE_FLOAT32, 2, sC);
    FeTensor *C3 = fe_tensor_alloc(DTYPE_FLOAT32, 2, sC);

    unsigned seed = 12345u;
    for (int i = 0; i < M * K; i++) ((float *)A->data)[i] = next_rand(&seed) * 0.01f;
    for (int i = 0; i < K * N; i++) ((float *)B->data)[i] = next_rand(&seed) * 0.01f;

    assert(fe_matmul(A, B, C1) == FE_OK);           /* reference oracle */

    FeThreadPool tp;
    assert(fe_threadpool_init(&tp, 4) == FE_OK);
    assert(fe_matmul_parallel(A, B, C2, &tp) == FE_OK);
    fe_threadpool_destroy(&tp);

    assert(fe_matmul_parallel(A, B, C3, NULL) == FE_OK);   /* sequential path */

    float max_p = 0.0f, max_s = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float r = ((float *)C1->data)[i];
        max_p = fmaxf(max_p, fabsf(((float *)C2->data)[i] - r));
        max_s = fmaxf(max_s, fabsf(((float *)C3->data)[i] - r));
    }
    assert(max_s < 1e-4f);   /* parallel(NULL) == oracle */
    assert(max_p < 1e-4f);   /* pool == oracle */

    fe_tensor_free(A); fe_tensor_free(B); fe_tensor_free(C1);
    fe_tensor_free(C2); fe_tensor_free(C3);
    printf("PASS test_parallel_gemm_matches_sequential (M=37 K=17 N=23, max err %.1e)\n",
           max_p);
}

/* ---- scheduler: causal order, both sequential and pool-backed ---- */

typedef struct {
    int *order;
    int  n;
} Rec;

static pthread_mutex_t rec_lock = PTHREAD_MUTEX_INITIALIZER;

static void record_node(void *ctx, int node) {
    Rec *r = (Rec *)ctx;
    pthread_mutex_lock(&rec_lock);
    r->order[r->n++] = node;
    pthread_mutex_unlock(&rec_lock);
}

/* Build the tiny MLP graph node-by-node without the engine; nodes are
 * executed in dependency order only — their kernels are not the point here. */
static void build_diamond(FeGraph *g) {
    fe_graph_init(g);
    int t_in  = fe_graph_add_tensor(g, "in",  DTYPE_FLOAT32, 2, (int[]){1, 1}, 0);
    int t_b   = fe_graph_add_tensor(g, "b",   DTYPE_FLOAT32, 2, (int[]){1, 1}, 0);
    int t_c   = fe_graph_add_tensor(g, "c",   DTYPE_FLOAT32, 2, (int[]){1, 1}, 0);
    int t_d   = fe_graph_add_tensor(g, "d",   DTYPE_FLOAT32, 2, (int[]){1, 1}, 0);

    fe_graph_add_node(g, "input",  FE_OP_INPUT, NULL, 0, &t_in, 1);
    fe_graph_add_node(g, "bn",     FE_OP_RELU, &t_in, 1, &t_b,  1);
    fe_graph_add_node(g, "cn",     FE_OP_RELU, &t_in, 1, &t_c,  1);
    int dins[] = {t_b, t_c};
    fe_graph_add_node(g, "dn",     FE_OP_ADD, dins, 2, &t_d,  1);
    fe_graph_add_node(g, "output", FE_OP_OUTPUT, &t_d, 1, NULL, 0);
    assert(fe_graph_topo_sort(g) == FE_OK);
}

static void check_causal_order(const FeGraph *g, int *order, int n) {
    /* producer[t] = node whose output list contains t (tensorial tables
     * are small — linear scan is fine for a test). */
    int *producer = calloc((size_t)g->n_tensors, sizeof(int));
    for (int t = 0; t < g->n_tensors; t++) producer[t] = -1;
    for (int n = 0; n < g->n_nodes; n++)
        for (int o = 0; o < g->nodes[n].n_outputs; o++)
            producer[g->nodes[n].outputs[o]] = n;

    for (int pos = 0; pos < n; pos++) {
        int node = order[pos];
        for (int i = 0; i < g->nodes[node].n_inputs; i++) {
            int prod = producer[g->nodes[node].inputs[i]];
            if (prod == -1) continue;   /* weight/input: no producer */
            int q;
            for (q = 0; q < pos; q++)
                if (order[q] == prod) break;
            assert(q < pos);
        }
    }
    free(producer);
}

static void test_scheduler_causal_order(void) {
    FeGraph g;
    build_diamond(&g);

    int *order = calloc((size_t)g.n_nodes, sizeof(int));

    Rec r = { order, 0 };
    assert(fe_scheduler_run(&g, record_node, &r, NULL, NULL) == FE_ERR_NULL);
    /* fe_scheduler_run needs a result struct; NULL results are rejected. */
    assert(r.n == 0);

    FeSchedResult res = { order, 0 };
    assert(fe_scheduler_run(&g, record_node, &r, NULL, &res) == FE_OK);
    assert(res.n_executed == g.n_nodes);
    assert(r.n == g.n_nodes);
    check_causal_order(&g, r.order, r.n);

    FeThreadPool tp;
    assert(fe_threadpool_init(&tp, 4) == FE_OK);
    r.n = 0;
    assert(fe_scheduler_run(&g, record_node, &r, &tp, &res) == FE_OK);
    assert(res.n_executed == g.n_nodes);
    check_causal_order(&g, r.order, r.n);
    fe_threadpool_destroy(&tp);

    free(order);
    printf("PASS test_scheduler_causal_order (diamond, seq + pool)\n");
}

int main(void) {
    test_pool_partition();
    test_parallel_gemm_matches_sequential();
    test_scheduler_causal_order();
    printf("\nAll tests passed.\n");
    return 0;
}