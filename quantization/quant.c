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

FeStatus fe_quantize_per_channel(const FeTensor *in, FeTensor *out,
                                  float *scales) {
    if (!in || !out || !scales) return FE_ERR_NULL;
    if (in->dtype  != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (out->dtype != DTYPE_INT8)    return FE_ERR_DTYPE;
    if (in->ndim != 2 || out->ndim != 2) return FE_ERR_SHAPE;
    if (in->shape[0] != out->shape[0] || in->shape[1] != out->shape[1])
        return FE_ERR_SHAPE;

    int K = in->shape[0];
    int N = in->shape[1];
    const float *src = (const float *)in->data;
    int8_t      *dst = (int8_t *)out->data;

    for (int j = 0; j < N; j++) scales[j] = 0.0f;
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) {
            float a = fabsf(src[k * N + j]);
            if (a > scales[j]) scales[j] = a;
        }
    for (int j = 0; j < N; j++) {
        if (scales[j] < 1e-8f) scales[j] = 1e-8f;
        scales[j] /= 127.0f;
    }

    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) {
            float q = roundf(src[k * N + j] / scales[j]);
            if (q >  127.0f) q =  127.0f;
            if (q < -127.0f) q = -127.0f;
            dst[k * N + j] = (int8_t)q;
        }

    return FE_OK;
}

FeStatus fe_dequantize_per_channel(const FeTensor *in, const float *scales,
                                    FeTensor *out) {
    if (!in || !out || !scales) return FE_ERR_NULL;
    if (in->dtype  != DTYPE_INT8)    return FE_ERR_DTYPE;
    if (out->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (in->ndim != 2 || out->ndim != 2) return FE_ERR_SHAPE;
    if (in->shape[0] != out->shape[0] || in->shape[1] != out->shape[1])
        return FE_ERR_SHAPE;

    int K = in->shape[0];
    int N = in->shape[1];
    const int8_t *src = (const int8_t *)in->data;
    float        *dst = (float *)out->data;

    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            dst[k * N + j] = (float)src[k * N + j] * scales[j];

    return FE_OK;
}

FeStatus fe_matmul_int8_per_channel(const FeTensor *A, const FeTensor *B,
                                     FeTensor *C) {
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
    float *b_scales = (float *)malloc(N * sizeof(float));
    if (!Aq || !Bq || !b_scales) {
        fe_tensor_free(Aq);
        fe_tensor_free(Bq);
        free(b_scales);
        return FE_ERR_NOMEM;
    }

    FeQuantParams pA;
    fe_quantize(A, Aq, &pA);
    fe_quantize_per_channel(B, Bq, b_scales);

    const int8_t *a = (const int8_t *)Aq->data;
    const int8_t *b = (const int8_t *)Bq->data;
    float        *c = (float *)C->data;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)a[i * K + k] * (int32_t)b[k * N + j];
            }
            c[i * N + j] = (float)acc * pA.scale * b_scales[j];
        }
    }

    fe_tensor_free(Aq);
    fe_tensor_free(Bq);
    free(b_scales);
    return FE_OK;
}

/* Shared core of the engine-path INT8 matmul/linear. Quantizes the float
 * activation on the fly (two passes, no scratch buffer), multiplies int8
 * products into a float accumulator, then applies scale_A and the per-column
 * weight scale. When b != NULL it is added afterward (float bias). */
static FeStatus matmul_int8_dyn_core(const FeTensor *A, const FeTensor *Wq,
                                     const float *w_scales,
                                     FeTensor *C, const float *b) {
    if (!A || !Wq || !C || !w_scales) return FE_ERR_NULL;
    if (A->ndim != 2 || Wq->ndim != 2 || C->ndim != 2) return FE_ERR_SHAPE;

    int M = A->shape[0];
    int K = A->shape[1];
    int N = Wq->shape[1];

    if (Wq->shape[0] != K)          return FE_ERR_SHAPE;
    if (C->shape[0] != M)           return FE_ERR_SHAPE;
    if (C->shape[1] != N)           return FE_ERR_SHAPE;
    if (A->dtype  != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (Wq->dtype != DTYPE_INT8)    return FE_ERR_DTYPE;
    if (C->dtype  != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (b != NULL && A->ndim != 2)  return FE_ERR_SHAPE;

    const float  *a = (const float  *)A->data;
    const int8_t *w = (const int8_t *)Wq->data;
    float        *c = (float *)C->data;

    /* Pass 1: activation scale from max |A| (dynamic quantization). */
    float max_abs = 0.0f;
    for (int i = 0; i < M * K; i++) {
        float v = fabsf(a[i]);
        if (v > max_abs) max_abs = v;
    }
    if (max_abs < 1e-8f) max_abs = 1e-8f;
    float scale_a     = max_abs / 127.0f;
    float inv_scale_a = 1.0f / scale_a;

    /* Pass 2: per-row accumulate, quantizing A on the fly. */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) c[i * N + j] = 0.0f;
        for (int k = 0; k < K; k++) {
            float q = roundf(a[i * K + k] * inv_scale_a);
            if (q >  127.0f) q =  127.0f;
            if (q < -127.0f) q = -127.0f;
            int8_t qa = (int8_t)q;
            for (int j = 0; j < N; j++)
                c[i * N + j] += (float)(qa * (int)w[k * N + j]);
        }
        for (int j = 0; j < N; j++) {
            c[i * N + j] = c[i * N + j] * scale_a * w_scales[j];
            if (b) c[i * N + j] += b[j];
        }
    }

    return FE_OK;
}

FeStatus fe_matmul_int8_dyn(const FeTensor *A, const FeTensor *Wq,
                             const float *w_scales, FeTensor *C) {
    return matmul_int8_dyn_core(A, Wq, w_scales, C, NULL);
}

FeStatus fe_linear_int8(const FeTensor *A, const FeTensor *Wq,
                         const float *w_scales, const FeTensor *b,
                         FeTensor *C) {
    if (!b) return FE_ERR_NULL;
    if (b->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (b->ndim != 1 || b->shape[0] != Wq->shape[1]) return FE_ERR_SHAPE;
    return matmul_int8_dyn_core(A, Wq, w_scales, C, (const float *)b->data);
}

/* Symmetric per-tensor int16 quantization.
 * scale = max(|x|) / 32767;  q = clamp(round(x / scale), -32768, 32767).
 * Input/out must be DTYPE_FLOAT32 and DTYPE_INT16 with matching shape. */
FeStatus fe_quantize_int16(const FeTensor *in, FeTensor *out,
                           float *params) {
    if (!in || !out || !params) return FE_ERR_NULL;
    if (in->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (out->dtype != DTYPE_INT16)  return FE_ERR_DTYPE;
    if (fe_tensor_numel(in) != fe_tensor_numel(out)) return FE_ERR_SHAPE;

    int n = fe_tensor_numel(in);
    const float *src = (const float *)in->data;
    int16_t *dst = (int16_t *)out->data;

    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-8f) max_abs = 1e-8f;

    params[0] = max_abs / 32767.0f;           /* scale stored as first entry */
    float inv_scale = 1.0f / params[0];
    for (int i = 0; i < n; i++) {
        float q = roundf(src[i] * inv_scale);
        if (q >  32767.0f) q =  32767.0f;
        if (q < -32768.0f) q = -32768.0f;
        dst[i] = (int16_t)q;
    }
    return FE_OK;
}

/* Shared core of the engine-path INT16 dynamic-quant matmul/linear.
 * Symmetric per-tensor activation quantization with int16 output accumulator.
 * weight tensor Wq is expected to be per-channel already int16.
 * w_scales[j] = per-output-channel weight scale.
 * C accumulates in float32, then multiplied by scale_A and w_scales[j].
 * If b != NULL, float bias is added after scale. */
static FeStatus matmul_int16_dyn_core(const FeTensor *A, const FeTensor *Wq,
                                      const float *w_scales,
                                      FeTensor *C, const float *b) {
    if (!A || !Wq || !C || !w_scales) return FE_ERR_NULL;
    if (A->ndim != 2 || Wq->ndim != 2 || C->ndim != 2) return FE_ERR_SHAPE;

    int M = A->shape[0];
    int K = A->shape[1];
    int N = Wq->shape[1];

    if (Wq->shape[0] != K)          return FE_ERR_SHAPE;
    if (C->shape[0] != M)           return FE_ERR_SHAPE;
    if (C->shape[1] != N)           return FE_ERR_SHAPE;
    if (A->dtype  != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (Wq->dtype != DTYPE_INT16)   return FE_ERR_DTYPE;
    if (C->dtype  != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    const float  *a = (const float  *)A->data;
    const int16_t* w = (const int16_t*)Wq->data;
    float        *c = (float *)C->data;

    /* Pass 1: activation scale from max |A| (dynamic quantization). */
    float max_abs = 0.0f;
    for (int i = 0; i < M * K; i++) {
        float v = fabsf(a[i]);
        if (v > max_abs) max_abs = v;
    }
    if (max_abs < 1e-8f) max_abs = 1e-8f;
    float scale_a     = max_abs / 127.0f;     /* NOTE: shared reference scale 127 */
    float inv_scale_a = 1.0f / scale_a;

    /* Pass 2: per-row accumulate, quantizing A on the fly. */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) c[i * N + j] = 0.0f;
        for (int k = 0; k < K; k++) {
            float q = roundf(a[i * K + k] * inv_scale_a);
            if (q >  127.0f) q =  127.0f;
            if (q < -127.0f) q = -127.0f;
            int16_t qa = (int16_t)q;
            for (int j = 0; j < N; j++)
                c[i * N + j] += (float)(qa * (int)w[k * N + j]);
        }
        for (int j = 0; j < N; j++) {
            c[i * N + j] = c[i * N + j] * scale_a * w_scales[j];
            if (b) c[i * N + j] += b[j];
        }
    }

    return FE_OK;
}

FeStatus fe_matmul_int16_dyn(const FeTensor *A, const FeTensor *Wq,
                             const float *w_scales, FeTensor *C) {
    return matmul_int16_dyn_core(A, Wq, w_scales, C, NULL);
}

FeStatus fe_linear_int16(const FeTensor *A, const FeTensor *Wq,
                          const float *w_scales, const FeTensor *b,
                          FeTensor *C) {
    if (!b) return FE_ERR_NULL;
    if (b->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (b->ndim != 1 || b->shape[0] != Wq->shape[1]) return FE_ERR_SHAPE;
    return matmul_int16_dyn_core(A, Wq, w_scales, C, (const float *)b->data);
}

/* True when a weight tensor enters a MATMUL or LINEAR node as inputs[1]. */
static int is_linear_weight_input(const FeGraph *g, int tidx) {
    for (int n = 0; n < g->n_nodes; n++)
        if ((g->nodes[n].op == FE_OP_MATMUL || g->nodes[n].op == FE_OP_LINEAR) &&
            g->nodes[n].n_inputs >= 2 && g->nodes[n].inputs[1] == tidx)
            return 1;
    return 0;
}

FeStatus fe_quantize_model(FeGraph *g, FeArena *arena) {
    if (!g || !arena) return FE_ERR_NULL;

    /* The bump pointer may not sit past the existing weight bytes (e.g. the
     * caller re-inited the arena over a buffer that already holds the loaded
     * weights). Append past the end of any weight that lives in this arena
     * so scales never collide with weight metadata or data. */
    size_t used = 0;
    for (int t = 0; t < g->n_tensors; t++) {
        FeTensorEntry *te = &g->tensors[t];
        if (!te->tensor || !te->tensor->data) continue;
        unsigned char *d = (unsigned char *)te->tensor->data;
        if (d < arena->base)                    continue;
        if (d >= arena->base + arena->size)     continue;
        size_t end = (size_t)(d - arena->base) + te->tensor->nbytes;
        if (end > used) used = end;
    }
    if (used > arena->offset) arena->offset = used;

    for (int i = 0; i < g->n_tensors; i++) {
        FeTensorEntry *e = &g->tensors[i];
        if (!e->is_weight || !e->tensor)        continue;
        if (e->tensor->dtype != DTYPE_FLOAT32)  continue;
        if (e->ndim != 2)                       continue;
        if (!is_linear_weight_input(g, i))      continue;

        int K = e->shape[0];
        int N = e->shape[1];

        float *scales = (float *)fe_arena_alloc(arena,
                                                (size_t)N * sizeof(float), 64);
        if (!scales) return FE_ERR_NOMEM;

        FeTensor *q = fe_tensor_alloc(DTYPE_INT8, 2, e->shape);
        if (!q) return FE_ERR_NOMEM;
        fe_quantize_per_channel(e->tensor, q, scales);

        /* Repack the weight into its own existing buffer (in place — the
         * arena was sized for the float data, so there is room). */
        memcpy(e->tensor->data, q->data, (size_t)K * N * sizeof(int8_t));
        e->tensor->dtype  = DTYPE_INT8;
        e->tensor->nbytes = (size_t)K * N;
        e->dtype  = DTYPE_INT8;
        e->scales = scales;
        e->n_scales = N;

        fe_tensor_free(q);
    }
    return FE_OK;
}

FeStatus fe_quantize_weights(FeGraph *g,
                              FeQuantParams *params, int n_params) {
    if (!g || !params) return FE_ERR_NULL;
    int idx = 0;
    for (int i = 0; i < g->n_tensors; i++) {
        FeTensorEntry *e = &g->tensors[i];
        if (!e->is_weight) continue;
        if (idx >= n_params) return FE_ERR_NOMEM;
        if (!e->tensor || e->tensor->dtype != DTYPE_FLOAT32) continue;

        int n = fe_tensor_numel(e->tensor);
        FeTensor *q = fe_tensor_alloc(DTYPE_INT8, e->ndim, e->shape);
        if (!q) return FE_ERR_NOMEM;
        fe_quantize(e->tensor, q, &params[idx++]);

        /* Repack in place: the INT8 data is 1/4 the size, so it fits in the
         * weight's own buffer whether that buffer is malloc- or arena-backed.
         * Never fe_tensor_free the source — arena-backed weights die with
         * the arena. */
        memcpy(e->tensor->data, q->data, (size_t)n);
        e->tensor->dtype  = DTYPE_INT8;
        e->tensor->nbytes = (size_t)n;
        e->dtype = DTYPE_INT8;
        fe_tensor_free(q);
    }
    return FE_OK;
}