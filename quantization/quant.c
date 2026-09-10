#include "quant.h"
#include "graph.h"
#include "fp16.h"
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

/* Per-channel INT16 quantization for the INT16 engine path. Exclusive of the
 * INT8 version in that the full 16-bit grid is used: scale = max/32767 and q
 * clamps to [-32768, 32767]. */
FeStatus fe_quantize_per_channel_16(const FeTensor *in, FeTensor *out,
                                    float *scales) {
    if (!in || !out || !scales) return FE_ERR_NULL;
    if (in->dtype  != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (out->dtype != DTYPE_INT16)   return FE_ERR_DTYPE;
    if (in->ndim != 2 || out->ndim != 2) return FE_ERR_SHAPE;
    if (in->shape[0] != out->shape[0] || in->shape[1] != out->shape[1])
        return FE_ERR_SHAPE;

    int K = in->shape[0];
    int N = in->shape[1];
    const float  *src = (const float  *)in->data;
    int16_t      *dst = (int16_t *)out->data;

    for (int j = 0; j < N; j++) scales[j] = 0.0f;
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) {
            float a = fabsf(src[k * N + j]);
            if (a > scales[j]) scales[j] = a;
        }
    for (int j = 0; j < N; j++) {
        if (scales[j] < 1e-8f) scales[j] = 1e-8f;
        scales[j] /= 32767.0f;
    }

    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) {
            float q = roundf(src[k * N + j] / scales[j]);
            if (q >  32767.0f) q =  32767.0f;
            if (q < -32767.0f) q = -32767.0f;
            dst[k * N + j] = (int16_t)q;
        }

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

/* Shared core of the engine-path INT8 matmul/linear. `act_scale` is the
 * fixed static activation scale when > 0 (one-pass quantization, no max
 * scan); when <= 0 it is derived dynamically from max|A| per call (two-pass).
 * Quantizes the float activation on the fly (no scratch buffer), multiplies
 * int8 products into a float accumulator, then applies scale_A and the
 * per-column weight scale. When b != NULL it is added afterward (float bias). */
static FeStatus matmul_int8_core(const FeTensor *A, const FeTensor *Wq,
                                 const float *w_scales,
                                 FeTensor *C, const float *b,
                                 float act_scale) {
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

    /* Activation scale: fixed (static) when given, else derivable (dynamic). */
    float scale_a;
    if (act_scale > 0.0f) {
        scale_a = act_scale;
    } else {
        float max_abs = 0.0f;
        for (int i = 0; i < M * K; i++) {
            float v = fabsf(a[i]);
            if (v > max_abs) max_abs = v;
        }
        if (max_abs < 1e-8f) max_abs = 1e-8f;
        scale_a = max_abs / 127.0f;
    }
    float inv_scale_a = 1.0f / scale_a;

    /* Pass over A: per-row accumulate, quantizing A on the fly. */
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
    return matmul_int8_core(A, Wq, w_scales, C, NULL, 0.0f);
}

FeStatus fe_linear_int8(const FeTensor *A, const FeTensor *Wq,
                         const float *w_scales, const FeTensor *b,
                         FeTensor *C) {
    if (!b) return FE_ERR_NULL;
    if (b->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (b->ndim != 1 || b->shape[0] != Wq->shape[1]) return FE_ERR_SHAPE;
    return matmul_int8_core(A, Wq, w_scales, C, (const float *)b->data, 0.0f);
}

FeStatus fe_matmul_int8_static(const FeTensor *A, const FeTensor *Wq,
                               const float *w_scales, float act_scale,
                               FeTensor *C) {
    if (act_scale <= 0.0f) return FE_ERR_SHAPE;
    return matmul_int8_core(A, Wq, w_scales, C, NULL, act_scale);
}

FeStatus fe_linear_int8_static(const FeTensor *A, const FeTensor *Wq,
                               const float *w_scales, const FeTensor *b,
                               float act_scale, FeTensor *C) {
    if (act_scale <= 0.0f) return FE_ERR_SHAPE;
    if (!b) return FE_ERR_NULL;
    if (b->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (b->ndim != 1 || b->shape[0] != Wq->shape[1]) return FE_ERR_SHAPE;
    return matmul_int8_core(A, Wq, w_scales, C, (const float *)b->data,
                            act_scale);
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
 * `act_scale` is the fixed static scale when > 0 (one pass); when <= 0 it is
 * derived from max|A| per call (two-pass). Unlike the INT8 path, qA uses the
 * full 16-bit grid: scale = max/32767, q clamped to [-32768, 32767] (the
 * documented Stage 15 quirk — the activation scale reusing max/127 — is
 * fixed here).
 * weight tensor Wq is expected to be per-channel already int16.
 * w_scales[j] = per-output-channel weight scale.
 * C accumulates in float32, then multiplied by scale_A and w_scales[j].
 * If b != NULL, float bias is added after scale. */
static FeStatus matmul_int16_core(const FeTensor *A, const FeTensor *Wq,
                                  const float *w_scales,
                                  FeTensor *C, const float *b,
                                  float act_scale) {
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

    float scale_a;
    if (act_scale > 0.0f) {
        scale_a = act_scale;
    } else {
        float max_abs = 0.0f;
        for (int i = 0; i < M * K; i++) {
            float v = fabsf(a[i]);
            if (v > max_abs) max_abs = v;
        }
        if (max_abs < 1e-8f) max_abs = 1e-8f;
        scale_a = max_abs / 32767.0f;
    }
    float inv_scale_a = 1.0f / scale_a;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) c[i * N + j] = 0.0f;
        for (int k = 0; k < K; k++) {
            float q = roundf(a[i * K + k] * inv_scale_a);
            if (q >  32767.0f) q =  32767.0f;
            if (q < -32767.0f) q = -32767.0f;
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
    return matmul_int16_core(A, Wq, w_scales, C, NULL, 0.0f);
}

FeStatus fe_linear_int16(const FeTensor *A, const FeTensor *Wq,
                          const float *w_scales, const FeTensor *b,
                          FeTensor *C) {
    if (!b) return FE_ERR_NULL;
    if (b->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (b->ndim != 1 || b->shape[0] != Wq->shape[1]) return FE_ERR_SHAPE;
    return matmul_int16_core(A, Wq, w_scales, C, (const float *)b->data, 0.0f);
}

FeStatus fe_matmul_int16_static(const FeTensor *A, const FeTensor *Wq,
                                const float *w_scales, float act_scale,
                                FeTensor *C) {
    if (act_scale <= 0.0f) return FE_ERR_SHAPE;
    return matmul_int16_core(A, Wq, w_scales, C, NULL, act_scale);
}

FeStatus fe_linear_int16_static(const FeTensor *A, const FeTensor *Wq,
                                const float *w_scales, const FeTensor *b,
                                float act_scale, FeTensor *C) {
    if (act_scale <= 0.0f) return FE_ERR_SHAPE;
    if (!b) return FE_ERR_NULL;
    if (b->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (b->ndim != 1 || b->shape[0] != Wq->shape[1]) return FE_ERR_SHAPE;
    return matmul_int16_core(A, Wq, w_scales, C, (const float *)b->data,
                             act_scale);
}

/* True when a weight tensor enters a MATMUL or LINEAR node as inputs[1]. */
static int is_linear_weight_input(const FeGraph *g, int tidx) {
    for (int n = 0; n < g->n_nodes; n++)
        if ((g->nodes[n].op == FE_OP_MATMUL || g->nodes[n].op == FE_OP_LINEAR) &&
            g->nodes[n].n_inputs >= 2 && g->nodes[n].inputs[1] == tidx)
            return 1;
    return 0;
}

/* Before repacking weights, the bump pointer may not sit past the existing
 * weight bytes (e.g. the caller re-inited the arena over a buffer that
 * already holds the loaded weights). Append past the end of any weight that
 * lives in this arena so scales never collide with weight metadata or data.
 * Returns TRUE on success. */
static int bump_arena_to_weights(const FeGraph *g, FeArena *arena) {
    if (!g || !arena) return 0;
    size_t used = 0;
    for (int t = 0; t < g->n_tensors; t++) {
        const FeTensorEntry *te = &g->tensors[t];
        if (!te->tensor || !te->tensor->data) continue;
        unsigned char *d = (unsigned char *)te->tensor->data;
        if (d < arena->base)                    continue;
        if (d >= arena->base + arena->size)     continue;
        size_t end = (size_t)(d - arena->base) + te->tensor->nbytes;
        if (end > used) used = end;
    }
    if (used > arena->offset) arena->offset = used;
    return 1;
}

/* Shared repack of 2-D MatMul/Linear weights into a per-channel quantized
 * storage dtype. `q_dtype` is DTYPE_INT8 or DTYPE_INT16 (the scales work out
 * to max/127 or max/32767 via fe_quantize_per_channel[-16]). Mirrors the old
 * fe_quantize_model path exactly, so on entry the caller already bumped the
 * arena offset past live weight bytes. */
static FeStatus repack_linear_weights_core(FeGraph *g, FeArena *arena,
                                           FeDtype q_dtype) {
    size_t elem =
        q_dtype == DTYPE_INT16 ? sizeof(int16_t) : sizeof(int8_t);
    int n_bytes_in = q_dtype == DTYPE_INT16 ? 2 : 1;

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

        FeTensor *q = fe_tensor_alloc(q_dtype, 2, e->shape);
        if (!q) return FE_ERR_NOMEM;

        if (q_dtype == DTYPE_INT16)
            fe_quantize_per_channel_16(e->tensor, q, scales);
        else
            fe_quantize_per_channel(e->tensor, q, scales);

        /* Repack the weight into its own existing buffer (in place — the
         * arena was sized for the float data, so there is room). */
        memcpy(e->tensor->data, q->data,
               (size_t)K * N * elem);
        e->tensor->dtype  = q_dtype;
        e->tensor->nbytes = (size_t)K * N * (size_t)n_bytes_in;
        e->dtype  = q_dtype;
        e->scales = scales;
        e->n_scales = N;

        fe_tensor_free(q);
    }
    return FE_OK;
}

FeStatus fe_quantize_model(FeGraph *g, FeArena *arena) {
    if (!g || !arena) return FE_ERR_NULL;
    if (!bump_arena_to_weights(g, arena)) return FE_ERR_NOMEM;
    return repack_linear_weights_core(g, arena, DTYPE_INT8);
}

FeStatus fe_quantize_model_int16(FeGraph *g, FeArena *arena) {
    if (!g || !arena) return FE_ERR_NULL;
    if (!bump_arena_to_weights(g, arena)) return FE_ERR_NOMEM;
    return repack_linear_weights_core(g, arena, DTYPE_INT16);
}

/* Apply per-tensor static activation scales (range/127; 0 stays dynamic),
 * then repack the weights to INT8 exactly as fe_quantize_model does. */
FeStatus fe_quantize_model_static(FeGraph *g, FeArena *arena,
                                  const float *ranges) {
    if (!g || !arena || !ranges) return FE_ERR_NULL;

    for (int i = 0; i < g->n_tensors; i++) {
        FeTensorEntry *e = &g->tensors[i];
        if (e->is_weight)        continue;
        if (!e->tensor)          continue;
        if (e->tensor->dtype != DTYPE_FLOAT32) continue;
        if (ranges[i] > 0.0f)
            e->act_scale = ranges[i] / 127.0f;
    }

    if (!bump_arena_to_weights(g, arena)) return FE_ERR_NOMEM;
    return repack_linear_weights_core(g, arena, DTYPE_INT8);
}

/* Repack 2-D MatMul/Linear weights to FLOAT16 (bf16 == 0) or BFLOAT16
 * storage in place. Mirrors the quantize repack pattern: same arena bump,
 * same linear-weight scanning, in-place shrink to half bytes-per-element.
 * No scales recorded — compute upconverts on read via fe_ensure_f32_weight. */
FeStatus fe_repack_model_half(FeGraph *g, FeArena *arena, int bf16) {
    if (!g || !arena) return FE_ERR_NULL;
    if (!bump_arena_to_weights(g, arena)) return FE_ERR_NOMEM;

    FeDtype half_dtype = bf16 ? DTYPE_BFLOAT16 : DTYPE_FLOAT16;

    for (int i = 0; i < g->n_tensors; i++) {
        FeTensorEntry *e = &g->tensors[i];
        if (!e->is_weight || !e->tensor)        continue;
        if (e->tensor->dtype != DTYPE_FLOAT32)  continue;
        if (e->ndim != 2)                       continue;
        if (!is_linear_weight_input(g, i))      continue;

        int K = e->shape[0];
        int N = e->shape[1];
        int n = K * N;

        const float *src = (const float *)e->tensor->data;
        uint16_t    *dst = (uint16_t *)e->tensor->data;

        /* flips of the float buffer converted to halves (in place safe). */
        if (bf16)
            fe_f32_to_bf16_buf(src, dst, n);
        else
            fe_f32_to_fp16_buf(src, dst, n);

        e->tensor->dtype  = half_dtype;
        e->tensor->nbytes = (size_t)n * sizeof(uint16_t);
        e->dtype = half_dtype;
    }
    return FE_OK;
}

/* Upconvert a FLOAT16/BFLOAT16 weight into an FP32 shadow owned by the
 * weight arena. Each half storage cell expands to 4 bytes, so the shadow
 * buffer must not alias the (2 bytes/element) half data: allocate fresh.
 * The exec plan calls this once at build time; the shadow is reused for all
 * subsequent runs of that plan until the arena resets. */
FeStatus fe_ensure_f32_weight(FeGraph *g, FeArena *arena, int tidx) {
    if (!g || !arena) return FE_ERR_NULL;
    if (tidx < 0 || tidx >= g->n_tensors) return FE_ERR_BOUNDS;

    FeTensorEntry *e = &g->tensors[tidx];
    if (!e->tensor || !e->tensor->data) return FE_ERR_NULL;
    if (e->shadow) {
        /* Shadow present: it must still be float and the input still half. */
        if (e->shadow->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
        return FE_OK;
    }
    if (e->tensor->dtype != DTYPE_FLOAT16 &&
        e->tensor->dtype != DTYPE_BFLOAT16)
        return FE_OK;   /* nothing to upconvert */

    int n = fe_tensor_numel(e->tensor);
    FeTensor *f32 = fe_arena_alloc_tensor(arena, DTYPE_FLOAT32,
                                          e->ndim, e->shape);
    if (!f32) return FE_ERR_NOMEM;

    if (e->tensor->dtype == DTYPE_FLOAT16)
        fe_fp16_to_f32_buf((const uint16_t *)e->tensor->data,
                           (float *)f32->data, n);
    else
        fe_bf16_to_f32_buf((const uint16_t *)e->tensor->data,
                           (float *)f32->data, n);

    e->shadow = f32;
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