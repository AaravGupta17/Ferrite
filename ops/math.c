/* ops/math.c — basic elementwise transcendental ops (Stage 3, Basic family). */
#include "ops.h"
#include <math.h>

static FeStatus validate_same_shape(const FeTensor *a, const FeTensor *out) {
    if (!a || !out) return FE_ERR_NULL;
    if (a->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;
    if (a->ndim != out->ndim) return FE_ERR_SHAPE;
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != out->shape[i]) return FE_ERR_SHAPE;
    return FE_OK;
}

FeStatus fe_exp(const FeTensor *a, FeTensor *out) {
    FeStatus s = validate_same_shape(a, out);
    if (s != FE_OK) return s;
    int n = fe_tensor_numel(a);
    const float *p = (const float *)a->data;
    float *o = (float *)out->data;
    for (int i = 0; i < n; i++) o[i] = expf(p[i]);
    return FE_OK;
}

FeStatus fe_ln(const FeTensor *a, FeTensor *out) {
    FeStatus s = validate_same_shape(a, out);
    if (s != FE_OK) return s;
    int n = fe_tensor_numel(a);
    const float *p = (const float *)a->data;
    float *o = (float *)out->data;
    for (int i = 0; i < n; i++) o[i] = logf(p[i]);
    return FE_OK;
}

FeStatus fe_pow(const FeTensor *a, const FeTensor *b, FeTensor *out) {
    if (!a || !b || !out) return FE_ERR_NULL;
    if (a->ndim != b->ndim || a->ndim != out->ndim) return FE_ERR_SHAPE;
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i] || a->shape[i] != out->shape[i])
            return FE_ERR_SHAPE;
    if (a->dtype != DTYPE_FLOAT32 || b->dtype != DTYPE_FLOAT32 ||
        out->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    int n = fe_tensor_numel(a);
    const float *pa = (const float *)a->data;
    const float *pb = (const float *)b->data;
    float *po = (float *)out->data;
    for (int i = 0; i < n; i++) po[i] = powf(pa[i], pb[i]);
    return FE_OK;
}
