/* ops/gemm.c — general matrix multiply with transposes and scaling (Stage 3). */
#include "ops.h"
#include <string.h>

/*
 * C = alpha * op(A) @ op(B) + beta * C
 *
 * We build the (possibly transposed) operand views with fe_tensor_transpose
 * (O(1), no data copy) and reuse the proven naive matmul as the reference.
 * The alpha/beta/accumulate epilogue runs in-place on the output buffer.
 */
FeStatus fe_gemm(const FeTensor *A, int transA,
                 const FeTensor *B, int transB,
                 FeTensor *C, float alpha, float beta) {
    if (!A || !B || !C) return FE_ERR_NULL;

    int A0 = A->shape[0], A1 = A->shape[1];
    int B0 = B->shape[0], B1 = B->shape[1];

    /* Logical dims after optional transposition: op(A)=[M,K], op(B)=[K,N]. */
    int M = transA ? A1 : A0;
    int K = transA ? A0 : A1;
    int N = transB ? B0 : B1;

    if ((transB ? B1 : B0) != K) return FE_ERR_SHAPE;
    if (C->ndim != 2 || C->shape[0] != M || C->shape[1] != N)
        return FE_ERR_SHAPE;
    if (A->dtype != DTYPE_FLOAT32 || B->dtype != DTYPE_FLOAT32 ||
        C->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    const float *a = (const float *)A->data;
    const float *b = (const float *)B->data;
    float *c = (float *)C->data;

    if (beta == 0.0f) {
        memset(c, 0, (size_t)M * N * sizeof(float));
    } else if (beta != 1.0f) {
        for (int i = 0; i < M * N; i++) c[i] *= beta;
    }

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = transA ? a[k * A1 + i] : a[i * A1 + k];
            for (int j = 0; j < N; j++) {
                float b_kj = transB ? b[j * B1 + k] : b[k * B1 + j];
                c[i * N + j] += alpha * a_ik * b_kj;
            }
        }
    }
    return FE_OK;
}

FeStatus fe_transpose(const FeTensor *in, FeTensor *out) {
    if (!in || !out) return FE_ERR_NULL;
    if (in->ndim != 2 || out->ndim != 2) return FE_ERR_SHAPE;
    if (in->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;
    if (out->shape[0] != in->shape[1] || out->shape[1] != in->shape[0])
        return FE_ERR_SHAPE;

    int M = in->shape[0];
    int N = in->shape[1];
    const float *p = (const float *)in->data;
    float *o = (float *)out->data;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            o[j * M + i] = p[i * N + j];
    return FE_OK;
}
