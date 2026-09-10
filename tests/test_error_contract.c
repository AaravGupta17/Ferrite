/*
 * tests/test_error_contract.c — public-API error contract audit (1.4).
 *
 * The API contract expressed in every header:
 *   - constructors return NULL on failure and never touch caller outputs
 *   - fe_* returning FeStatus validate pointers/dtypes/shapes BEFORE
 *     touching outputs, so a failed call leaves outputs unchanged
 *   - no failed call frees or reallocates caller memory
 *
 * This test deliberately triggers each documented error path and asserts
 * that behaviour, with outputs sentinel-filled first and verified untouched
 * afterwards. It is a sweep over the stable public surface (tensor, arena,
 * ops, graph, planner, ser, runtime, FEMD) — the training wheels for the
 * "FE_OK ⇒ all outputs written" invariant, not a substitute for it.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "../core/tensor.h"
#include "../core/allocator.h"
#include "../core/tensor_ser.h"
#include "../graph/graph.h"
#include "../planner/memory_planner.h"
#include "../runtime/engine.h"
#include "../runtime/model_ser.h"
#include "../ops/ops.h"
#include "../importer/onnx.h"

#define SENTINEL 0xDEADBEEFu

static void fill_sentinel(FeTensor *t) {
    uint32_t *p = (uint32_t *)t->data;
    size_t n = t->nbytes / sizeof(uint32_t);
    for (size_t i = 0; i < n; i++) p[i] = SENTINEL;
}

static void assert_sentinel_intact(const FeTensor *t) {
    const uint32_t *p = (const uint32_t *)t->data;
    size_t n = t->nbytes / sizeof(uint32_t);
    for (size_t i = 0; i < n; i++)
        assert(p[i] == SENTINEL);
}

/* --- tensor --- */

static void test_tensor_contract(void) {
    int s4[] = {4}, s6[] = {6};
    FeTensor *src = fe_tensor_alloc(DTYPE_FLOAT32, 1, s4);
    FeTensor *dst = fe_tensor_alloc(DTYPE_FLOAT32, 1, s6);
    FeTensor *dst_i32 = fe_tensor_alloc(DTYPE_INT32, 1, s4);
    assert(src && dst && dst_i32);
    for (int i = 0; i < 4; i++) ((float *)src->data)[i] = (float)i;

    /* Shape mismatch: FE_ERR_SHAPE, dst untouched. */
    fill_sentinel(dst);
    assert(fe_tensor_copy(dst, src) == FE_ERR_SHAPE);
    assert_sentinel_intact(dst);

    /* Dtype mismatch: FE_ERR_DTYPE, dst untouched. */
    fill_sentinel(dst_i32);
    assert(fe_tensor_copy(dst_i32, src) == FE_ERR_DTYPE);
    assert_sentinel_intact(dst_i32);

    /* View constructors: reshape/non-contiguous and element-count mismatch
     * return NULL; transpose never copies and views never own. */
    int s22[] = {2, 2};
    FeTensor *sq = fe_tensor_alloc(DTYPE_FLOAT32, 2, s22);
    FeTensor *tp = fe_tensor_transpose(sq, 0, 1);
    assert(tp && tp->owns_data == false);
    assert(fe_tensor_reshape(tp, 1, s4) == NULL);   /* non-contiguous view */
    assert(fe_tensor_reshape(sq, 1, s6) == NULL);   /* numel mismatch */
    fe_tensor_free(tp);
    fe_tensor_free(sq);

    fe_tensor_free(src);
    fe_tensor_free(dst);
    fe_tensor_free(dst_i32);
    printf("PASS test_tensor_contract\n");
}

/* --- arena --- */

static void test_arena_contract(void) {
    FeArena a;
    assert(fe_arena_init(&a, NULL, 64) == FE_ERR_NULL);
    assert(fe_arena_init(NULL, &a, 64) == FE_ERR_NULL);
    assert(fe_arena_init(&a, &a, 0) == FE_ERR_NULL);

    unsigned char buf[64];
    assert(fe_arena_init(&a, buf, 64) == FE_OK);

    /* Exhaustion returns NULL, arena offset unchanged. */
    size_t before = fe_arena_used(&a);
    assert(fe_arena_alloc(&a, 4096, 64) == NULL);
    assert(fe_arena_used(&a) == before);

    /* Tensor allocation exhaustion returns NULL. */
    assert(fe_arena_alloc_tensor(&a, DTYPE_FLOAT32, 2, (int[]){8, 8}) == NULL);
    printf("PASS test_arena_contract\n");
}

/* --- ops kernels --- */

static void test_ops_contract(void) {
    int s23[] = {2, 3}, s44[] = {4, 4}, s25[] = {2, 5}, s35[] = {3, 5}, s3[] = {3};
    FeTensor *A = fe_tensor_alloc(DTYPE_FLOAT32, 2, s23);
    FeTensor *B = fe_tensor_alloc(DTYPE_FLOAT32, 2, s44);
    FeTensor *B35 = fe_tensor_alloc(DTYPE_FLOAT32, 2, s35);
    FeTensor *C = fe_tensor_alloc(DTYPE_FLOAT32, 2, s25);
    FeTensor *A_i8 = fe_tensor_alloc(DTYPE_INT8, 2, s23);
    FeTensor *u = fe_tensor_alloc(DTYPE_FLOAT32, 1, s3);
    FeTensor *in = fe_tensor_alloc(DTYPE_FLOAT32, 1, s3);
    FeTensor *soft = fe_tensor_alloc(DTYPE_FLOAT32, 2, s25);
    assert(A && B && B35 && C && A_i8 && u && in && soft);

    /* fe_matmul: NULL args. */
    assert(fe_matmul(NULL, B, C) == FE_ERR_NULL);
    assert(fe_matmul(A, NULL, C) == FE_ERR_NULL);
    assert(fe_matmul(A, B, NULL) == FE_ERR_NULL);

    /* Shape mismatch: FE_ERR_SHAPE, C untouched. */
    fill_sentinel(C);
    assert(fe_matmul(A, B, C) == FE_ERR_SHAPE);
    assert_sentinel_intact(C);

    /* Dtype mismatch: FE_ERR_DTYPE, C untouched. */
    fill_sentinel(C);
    assert(fe_matmul(A_i8, B35, C) == FE_ERR_DTYPE);
    assert_sentinel_intact(C);

    /* fe_relu: dtype mismatch leaves out untouched. */
    assert(fe_relu(A_i8, u) == FE_ERR_DTYPE);
    fill_sentinel(u);
    assert(fe_relu(A_i8, u) == FE_ERR_DTYPE);
    assert_sentinel_intact(u);

    /* fe_relu shape mismatch (numel differs) leaves out untouched. */
    fill_sentinel(C);
    assert(fe_relu(in, C) == FE_ERR_SHAPE);
    assert_sentinel_intact(C);

    /* fe_softmax: wrong output size. */
    fill_sentinel(C);
    assert(fe_softmax(A, C) == FE_ERR_SHAPE);
    assert_sentinel_intact(C);

    /* fe_softmax NULL. */
    assert(fe_softmax(NULL, soft) == FE_ERR_NULL);

    fe_tensor_free(A); fe_tensor_free(B); fe_tensor_free(B35); fe_tensor_free(C);
    fe_tensor_free(A_i8); fe_tensor_free(u); fe_tensor_free(in);
    fe_tensor_free(soft);
    printf("PASS test_ops_contract\n");
}

/* --- graph --- */

static void test_graph_contract(void) {
    FeGraph g;
    memset(&g, 0, sizeof(g));

    /* Validate: node referencing a tensor outside the registry. */
    int t_in = fe_graph_add_tensor(&g, "in", DTYPE_FLOAT32, 1, (int[]){4}, 0);
    int t_out = fe_graph_add_tensor(&g, "out", DTYPE_FLOAT32, 1, (int[]){4}, 0);
    assert(t_in == 0 && t_out == 1);
    int bad_ref[] = {7};
    fe_graph_add_node(&g, "bad", FE_OP_RELU, &t_in, 1, bad_ref, 1);
    assert(fe_graph_validate(&g) == FE_ERR_BOUNDS);

    /* Cycle: A produces t0 consumed by B, B produces t1 consumed by A. */
    FeGraph cyc;
    memset(&cyc, 0, sizeof(cyc));
    int c0 = fe_graph_add_tensor(&cyc, "t0", DTYPE_FLOAT32, 1, (int[]){4}, 0);
    int c1 = fe_graph_add_tensor(&cyc, "t1", DTYPE_FLOAT32, 1, (int[]){4}, 0);
    fe_graph_add_node(&cyc, "a", FE_OP_RELU, &c1, 1, &c0, 1);
    fe_graph_add_node(&cyc, "b", FE_OP_RELU, &c0, 1, &c1, 1);
    assert(fe_graph_topo_sort(&cyc) == FE_ERR_SHAPE);

    /* Planner refuses an un-sorted graph. */
    assert(fe_plan_memory(&g, &(FePlan){0}) == FE_ERR_SHAPE);
    printf("PASS test_graph_contract\n");
}

/* --- planner --- */

static void test_planner_contract(void) {
    /* Valid MLP graph (matmul + add), sorted, then plan-apply against an
     * undersized buffer. */
    FeGraph g;
    memset(&g, 0, sizeof(g));
    int t_in  = fe_graph_add_tensor(&g, "in", DTYPE_FLOAT32, 2, (int[]){1, 4}, 0);
    int t_w   = fe_graph_add_tensor(&g, "w",  DTYPE_FLOAT32, 2, (int[]){4, 3}, 1);
    int t_b   = fe_graph_add_tensor(&g, "b",  DTYPE_FLOAT32, 1, (int[]){3}, 1);
    int t_m   = fe_graph_add_tensor(&g, "m",  DTYPE_FLOAT32, 2, (int[]){1, 3}, 0);
    int t_y   = fe_graph_add_tensor(&g, "y",  DTYPE_FLOAT32, 2, (int[]){1, 3}, 0);
    int lin_in[2] = {t_in, t_w};
    int relu_in[2] = {t_m, t_b};
    fe_graph_add_node(&g, "lin", FE_OP_LINEAR, lin_in, 2, &t_m, 1);
    fe_graph_add_node(&g, "bias", FE_OP_ADD, relu_in, 2, &t_y, 1);
    fe_graph_add_node(&g, "out", FE_OP_OUTPUT, &t_y, 1, NULL, 0);
    assert(fe_graph_topo_sort(&g) == FE_OK);

    FePlan plan;
    assert(fe_plan_memory(&g, &plan) == FE_OK);

    /* Undersized data region must be rejected, not silently overflow. */
    unsigned char tiny[8];
    FeArena meta;
    fe_arena_init(&meta, tiny, sizeof(tiny));
    assert(fe_plan_apply(&g, &plan, tiny, sizeof(tiny), &meta) == FE_ERR_NOMEM);
    printf("PASS test_planner_contract (%zu B activations)\n",
           plan.total_activation_bytes);
}

/* --- serialization --- */

static void test_ser_contract(void) {
    assert(fe_tensor_save(NULL, "tests/error_contract.ftn") == FE_ERR_NULL);

    int s4[] = {4};
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, 1, s4);
    assert(t);
    assert(fe_tensor_save(t, "tests/error_contract.ftn") == FE_OK);

    FeTensor *out = NULL;
    assert(fe_tensor_load("tests/no_such_file_xyz.ftn", &out) != FE_OK);
    assert(out == NULL);   /* untouched on failure */

    /* Corrupt: wrong magic. */
    FILE *f = fopen("tests/error_contract_bad.ftn", "wb");
    fwrite("XYZZ", 1, 4, f);
    fclose(f);
    assert(fe_tensor_load("tests/error_contract_bad.ftn", &out) != FE_OK);
    assert(out == NULL);

    fe_tensor_free(t);
    remove("tests/error_contract.ftn");
    remove("tests/error_contract_bad.ftn");
    printf("PASS test_ser_contract\n");
}

/* --- runtime + FEMD --- */

static void test_runtime_contract(void) {
    assert(fe_runtime_run(NULL, NULL, NULL) == FE_ERR_NULL);

    /* FEMD save/load/run NULL args. */
    assert(fe_model_save(NULL, NULL, NULL, 0, 0) == FE_ERR_NULL);
    assert(fe_model_load(NULL, NULL, NULL) == FE_ERR_NULL);
    assert(fe_model_run(NULL, NULL, NULL, NULL, NULL) == FE_ERR_NULL);
    printf("PASS test_runtime_contract\n");
}

static void test_onnx_contract(void) {
    /* Load failure must be loud, not a silent partial graph. */
    FeGraph g;
    FeArena wa;
    unsigned char buf[4096];
    fe_arena_init(&wa, buf, sizeof(buf));
    assert(fe_onnx_load(&g, &wa, "tests/does_not_exist.onnx") != FE_OK);
    printf("PASS test_onnx_contract\n");
}

int main(void) {
    test_tensor_contract();
    test_arena_contract();
    test_ops_contract();
    test_graph_contract();
    test_planner_contract();
    test_ser_contract();
    test_runtime_contract();
    test_onnx_contract();
    printf("ALL ERROR-CONTRACT TESTS PASSED\n");
    return 0;
}