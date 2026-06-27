// ops/matmul.c
#include "ops.h"
#include <string.h>
#include <assert.h>

/*
 * Naive matmul: C = A @ B
 *
 * This is the reference implementation — correct but not optimized.
 * It exists to validate correctness before we add SIMD kernels.
 * The SIMD implementation in simd/ will replace the inner loop.
 *
 * Memory access pattern:
 *   A[i][k] — row-major access, stride 1 in inner loop. Cache-friendly.
 *   B[k][j] — column access, stride N in inner loop. Cache-unfriendly.
 *
 * This is the fundamental matmul performance problem.
 * We measure this, then fix it with tiling in the SIMD module.
 */
FeStatus fe_matmul(const FeTensor *A, const FeTensor *B, FeTensor *C) {
    if (!A || !B || !C) return FE_ERR_NULL;

    /* Shape validation */
    if (A->ndim != 2 || B->ndim != 2 || C->ndim != 2) return FE_ERR_SHAPE;

    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];

    if (B->shape[0] != K)   return FE_ERR_SHAPE;
    if (C->shape[0] != M)   return FE_ERR_SHAPE;
    if (C->shape[1] != N)   return FE_ERR_SHAPE;

    if (A->dtype != DTYPE_FLOAT32 ||
        B->dtype != DTYPE_FLOAT32 ||
        C->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    const float *a = (const float *)A->data;
    const float *b = (const float *)B->data;
    float       *c = (float *)C->data;

    /* Zero output — caller may pass uninitialized buffer */
    memset(c, 0, M * N * sizeof(float));

    /*
     * Standard triple loop.
     * Loop order i-k-j (not i-j-k) improves B access slightly
     * by keeping the k-stride on B in the innermost position,
     * but column access on B is still the bottleneck.
     */
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = a[i * K + k];
            for (int j = 0; j < N; j++) {
                c[i * N + j] += a_ik * b[k * N + j];
            }
        }
    }

    return FE_OK;
}

FeStatus fe_linear(const FeTensor *A, const FeTensor *W,
                   const FeTensor *b, FeTensor *C) {
    /* C = A @ W */
    FeStatus s = fe_matmul(A, W, C);
    if (s != FE_OK) return s;

    /* C += b (broadcast) */
    return fe_bias_add(C, b, C);
}