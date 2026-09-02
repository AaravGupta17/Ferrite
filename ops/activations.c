// ops/activations.c
#include "ops.h"
#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>

FeStatus fe_relu(const FeTensor *in, FeTensor *out) {
    if (!in || !out) return FE_ERR_NULL;
    if (in->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    int n = fe_tensor_numel(in);
    if (fe_tensor_numel(out) != n) return FE_ERR_SHAPE;

    const float *src = (const float *)in->data;
    float       *dst = (float *)out->data;

    for (int i = 0; i < n; i++) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }
    return FE_OK;
}

/*
 * Numerically stable softmax.
 *
 * Naive softmax: exp(x[i]) / sum(exp(x))
 * Problem: exp(large number) overflows float.
 *
 * Stable version: subtract max before exp.
 * exp(x[i] - max) / sum(exp(x - max))
 * This is mathematically identical but never overflows.
 *
 * Applied along the last axis only (standard for classification).
 */
FeStatus fe_softmax(const FeTensor *in, FeTensor *out) {
    if (!in || !out) return FE_ERR_NULL;
    if (in->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;
    if (fe_tensor_numel(in) != fe_tensor_numel(out)) return FE_ERR_SHAPE;

    int rows = 1;
    for (int i = 0; i < in->ndim - 1; i++) rows *= in->shape[i];
    int cols = in->shape[in->ndim - 1];

    const float *src = (const float *)in->data;
    float       *dst = (float *)out->data;

    for (int r = 0; r < rows; r++) {
        const float *row_in  = src + r * cols;
        float       *row_out = dst + r * cols;

        /* Step 1: find max for numerical stability */
        float max_val = -FLT_MAX;
        for (int c = 0; c < cols; c++) {
            if (row_in[c] > max_val) max_val = row_in[c];
        }

        /* Step 2: compute shifted exp and accumulate sum */
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row_out[c] = expf(row_in[c] - max_val);
            sum += row_out[c];
        }

        /* Step 3: normalize */
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row_out[c] *= inv_sum;
        }
    }
    return FE_OK;
}

FeStatus fe_bias_add(const FeTensor *in, const FeTensor *bias, FeTensor *out) {
    if (!in || !bias || !out) return FE_ERR_NULL;
    if (in->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    int cols = in->shape[in->ndim - 1];
    if (bias->shape[0] != cols) return FE_ERR_SHAPE;

    int rows = fe_tensor_numel(in) / cols;

    const float *src = (const float *)in->data;
    const float *b   = (const float *)bias->data;
    float       *dst = (float *)out->data;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            dst[r * cols + c] = src[r * cols + c] + b[c];
        }
    }
    return FE_OK;
}

/* Shared shape/dtype validation for elementwise activations. */
static FeStatus validate_act(const FeTensor *in, const FeTensor *out) {
    if (!in || !out) return FE_ERR_NULL;
    if (in->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;
    if (fe_tensor_numel(in) != fe_tensor_numel(out)) return FE_ERR_SHAPE;
    return FE_OK;
}

FeStatus fe_sigmoid(const FeTensor *in, FeTensor *out) {
    FeStatus s = validate_act(in, out);
    if (s != FE_OK) return s;
    int n = fe_tensor_numel(in);
    const float *p = (const float *)in->data;
    float *o = (float *)out->data;
    for (int i = 0; i < n; i++) o[i] = 1.0f / (1.0f + expf(-p[i]));
    return FE_OK;
}

FeStatus fe_tanh(const FeTensor *in, FeTensor *out) {
    FeStatus s = validate_act(in, out);
    if (s != FE_OK) return s;
    int n = fe_tensor_numel(in);
    const float *p = (const float *)in->data;
    float *o = (float *)out->data;
    for (int i = 0; i < n; i++) o[i] = tanhf(p[i]);
    return FE_OK;
}

/* GELU via the tanh approximation (documented choice; no erf dependency). */
#define GELU_COEF 0.7978845608028654f  /* sqrt(2/pi) */

FeStatus fe_gelu(const FeTensor *in, FeTensor *out) {
    FeStatus s = validate_act(in, out);
    if (s != FE_OK) return s;
    int n = fe_tensor_numel(in);
    const float *p = (const float *)in->data;
    float *o = (float *)out->data;
    for (int i = 0; i < n; i++) {
        float x = p[i];
        o[i] = 0.5f * x * (1.0f + tanhf(GELU_COEF * (x + 0.044715f * x * x * x)));
    }
    return FE_OK;
}

FeStatus fe_leaky_relu(const FeTensor *in, FeTensor *out, float negative_slope) {
    FeStatus s = validate_act(in, out);
    if (s != FE_OK) return s;
    int n = fe_tensor_numel(in);
    const float *p = (const float *)in->data;
    float *o = (float *)out->data;
    for (int i = 0; i < n; i++)
        o[i] = p[i] > 0.0f ? p[i] : negative_slope * p[i];
    return FE_OK;
}

FeStatus fe_elu(const FeTensor *in, FeTensor *out, float alpha) {
    FeStatus s = validate_act(in, out);
    if (s != FE_OK) return s;
    int n = fe_tensor_numel(in);
    const float *p = (const float *)in->data;
    float *o = (float *)out->data;
    for (int i = 0; i < n; i++)
        o[i] = p[i] > 0.0f ? p[i] : alpha * (expf(p[i]) - 1.0f);
    return FE_OK;
}

/* Swish (SiLU): x * sigmoid(x). */
FeStatus fe_swish(const FeTensor *in, FeTensor *out) {
    FeStatus s = validate_act(in, out);
    if (s != FE_OK) return s;
    int n = fe_tensor_numel(in);
    const float *p = (const float *)in->data;
    float *o = (float *)out->data;
    for (int i = 0; i < n; i++)
        o[i] = p[i] / (1.0f + expf(-p[i]));
    return FE_OK;
}