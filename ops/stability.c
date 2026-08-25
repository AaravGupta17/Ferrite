/* ops/stability.c (new file) */
#include "ops.h"
#include <math.h>
#include <float.h>

/* Numerically stable log-sum-exp along a 1D tensor:
 * log(sum(exp(x[i]))) computed as max + log(sum(exp(x[i] - max))) */
FeStatus fe_logsumexp(const FeTensor *in, float *out) {
    if (!in || !out) return FE_ERR_NULL;
    if (in->ndim != 1) return FE_ERR_SHAPE;
    if (in->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    int n = in->shape[0];
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        int idx[] = {i};
        float v = fe_tensor_get_f32(in, idx);
        if (v > max_val) max_val = v;
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        int idx[] = {i};
        sum += expf(fe_tensor_get_f32(in, idx) - max_val);
    }

    *out = max_val + logf(sum);
    return FE_OK;
}

/* Stable softmax over a 1D tensor — same max-subtraction trick,
 * exposed standalone so anything (not just fe_softmax) can reuse it. */
FeStatus fe_stable_exp_normalize(const FeTensor *in, FeTensor *out) {
    if (!in || !out) return FE_ERR_NULL;
    if (in->ndim != 1 || out->ndim != 1) return FE_ERR_SHAPE;
    if (in->shape[0] != out->shape[0]) return FE_ERR_SHAPE;
    if (in->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;

    int n = in->shape[0];
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        int idx[] = {i};
        float v = fe_tensor_get_f32(in, idx);
        if (v > max_val) max_val = v;
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        int idx[] = {i};
        float e = expf(fe_tensor_get_f32(in, idx) - max_val);
        fe_tensor_set_f32(out, idx, e);
        sum += e;
    }

    for (int i = 0; i < n; i++) {
        int idx[] = {i};
        fe_tensor_set_f32(out, idx, fe_tensor_get_f32(out, idx) / sum);
    }

    return FE_OK;
}