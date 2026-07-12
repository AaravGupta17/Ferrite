#include "quant.h"
#include "graph.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

FeStatus fe_quantize(const FeTensor *in, FeTensor *out,
                      FeQuantParams *params) {
    if (!in || !out || !params) return FE_ERR_NULL;
    if (in->dtype  != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (out->dtype != DTYPE_INT8)    return FE_ERR_DTYPE;

    int n = fe_tensor_numel(in);
    if (fe_tensor_numel(out) != n) return FE_ERR_SHAPE;

    const float *src = (const float *)in->data;
    int8_t      *dst = (int8_t *)out->data;

    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > max_abs) max_abs = a;
    }

    if (max_abs < 1e-8f) max_abs = 1e-8f;

    params->scale      = max_abs / 127.0f;
    params->zero_point = 0;

    float inv_scale = 1.0f / params->scale;
    for (int i = 0; i < n; i++) {
        float q = roundf(src[i] * inv_scale);
        if (q >  127.0f) q =  127.0f;
        if (q < -127.0f) q = -127.0f;
        dst[i] = (int8_t)q;
    }

    return FE_OK;
}

FeStatus fe_dequantize(const FeTensor *in, const FeQuantParams *params,
                        FeTensor *out) {
    if (!in || !out || !params) return FE_ERR_NULL;
    if (in->dtype  != DTYPE_INT8)    return FE_ERR_DTYPE;
    if (out->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    int n = fe_tensor_numel(in);
    if (fe_tensor_numel(out) != n) return FE_ERR_SHAPE;

    const int8_t *src = (const int8_t *)in->data;
    float        *dst = (float *)out->data;

    for (int i = 0; i < n; i++)
        dst[i] = (float)src[i] * params->scale;

    return FE_OK;
}

FeStatus fe_matmul_int8(const FeTensor *A, const FeTensor *B, FeTensor *C) {
    if (!A || !B || !C) return FE_ERR_NULL;
    if (A->ndim != 2 || B->ndim != 2 || C->ndim != 2) return FE_ERR_SHAPE;

    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];

    if (B->shape[0] != K) return FE_ERR_SHAPE;
    if (C->shape[0] != M) return FE_ERR_SHAPE;
    if (C->shape[1] != N) return FE_ERR_SHAPE;

    if (A->dtype != DTYPE_FLOAT32 ||
        B->dtype != DTYPE_FLOAT32 ||
        C->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    FeTensor *Aq = fe_tensor_alloc(DTYPE_INT8, 2, A->shape);
    FeTensor *Bq = fe_tensor_alloc(DTYPE_INT8, 2, B->shape);
    if (!Aq || !Bq) {
        fe_tensor_free(Aq);
        fe_tensor_free(Bq);
        return FE_ERR_NOMEM;
    }

    FeQuantParams pA, pB;
    fe_quantize(A, Aq, &pA);
    fe_quantize(B, Bq, &pB);

    const int8_t *a = (const int8_t *)Aq->data;
    const int8_t *b = (const int8_t *)Bq->data;
    float        *c = (float *)C->data;

    float out_scale = pA.scale * pB.scale;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)a[i * K + k] * (int32_t)b[k * N + j];
            }
            c[i * N + j] = (float)acc * out_scale;
        }
    }

    fe_tensor_free(Aq);
    fe_tensor_free(Bq);
    return FE_OK;
}