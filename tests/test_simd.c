// tests/test_simd.c
//
// Kernel equivalence: fe_matmul_avx2 must agree with the naive fe_matmul_scalar
// reference within float32 accumulation tolerance, at sizes that exercise both
// the fully-vectorized path (N % 8 == 0) and the scalar-remainder path
// (N % 8 != 0). This is the regression test that was missing from Stage 7 —
// previously the only AVX2-vs-naive check lived in tools/bench_avx2.c, which
// is excluded from the default build and never runs under ctest.
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../core/tensor.h"
#include "../ops/ops.h"
#include "../simd/matmul_avx2.h"

/* Relative + absolute tolerance: AVX2 accumulates in a different order than
 * the scalar triple loop, so bit-exact equality is not the right bar —
 * bounded floating-point drift is. */
static int approx_equal(float a, float b) {
    float diff = fabsf(a - b);
    float scale = fmaxf(fabsf(a), fabsf(b));
    return diff <= 1e-3f + 1e-4f * scale;
}

/* Run naive vs AVX2 matmul on an MxKxN problem and assert every output
 * element agrees. M/K/N are deliberately varied across calls to cover both
 * the vectorized path and the N % 8 != 0 remainder path (see
 * simd/matmul_avx2.h: "N must be a multiple of 8 for the vectorized path;
 * falls back to scalar for remainder columns"). */
static void check_matmul_equivalence(int M, int K, int N, uint32_t seed) {
    int shape_a[] = {M, K};
    int shape_b[] = {K, N};
    int shape_c[] = {M, N};

    FeTensor *A  = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape_a);
    FeTensor *B  = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape_b);
    FeTensor *C1 = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape_c);
    FeTensor *C2 = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape_c);
    assert(A && B && C1 && C2);

    assert(fe_rand_uniform(A, -1.0f, 1.0f, seed)     == FE_OK);
    assert(fe_rand_uniform(B, -1.0f, 1.0f, seed + 1) == FE_OK);

    assert(fe_matmul_scalar(A, B, C1) == FE_OK);
    assert(fe_matmul_avx2(A, B, C2)   == FE_OK);

    const float *c1 = (const float *)C1->data;
    const float *c2 = (const float *)C2->data;
    float max_err = 0.0f;
    int n = M * N;
    for (int i = 0; i < n; i++) {
        float err = fabsf(c1[i] - c2[i]);
        if (err > max_err) max_err = err;
        assert(approx_equal(c1[i], c2[i]));
    }

    printf("PASS matmul_equivalence M=%d K=%d N=%d (max err %.2e)\n",
           M, K, N, max_err);

    fe_tensor_free(A);
    fe_tensor_free(B);
    fe_tensor_free(C1);
    fe_tensor_free(C2);
}

static void test_vectorized_sizes(void) {
    /* N % 8 == 0 — takes the fully-vectorized AVX2 path. */
    check_matmul_equivalence(8,   8,   8,   1001);
    check_matmul_equivalence(32,  16,  64,  1002);
    check_matmul_equivalence(64,  64,  256, 1003);
}

static void test_remainder_sizes(void) {
    /* N % 8 != 0 — exercises the scalar-remainder tail. */
    check_matmul_equivalence(5,   7,   13,  2001);
    check_matmul_equivalence(17,  33,  99,  2002);
    check_matmul_equivalence(1,   4,   3,   2003);   /* 1x1 output edge case */
}

static void test_non_square(void) {
    /* Tall and wide shapes, mixed vectorized/remainder N. */
    check_matmul_equivalence(128, 8,   8,   3001);
    check_matmul_equivalence(2,   512, 130, 3002);
}

int main(void) {
    if (!fe_cpu_has_avx2()) {
        printf("AVX2 not available on this CPU — skipping (not a failure)\n");
        return 0;
    }

    test_vectorized_sizes();
    test_remainder_sizes();
    test_non_square();

    printf("\nAll tests passed.\n");
    return 0;
}