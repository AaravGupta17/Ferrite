/* ops/elementwise.c (new file) */
#include "ops.h"
#include <string.h>

typedef enum { OP_ADD, OP_SUB, OP_MUL, OP_DIV } ElemOp;

static FeStatus elementwise_binary(const FeTensor *a, const FeTensor *b,
                                    FeTensor *out, ElemOp op) {
    if (!a || !b || !out) return FE_ERR_NULL;
    if (a->ndim != b->ndim || a->ndim != out->ndim) return FE_ERR_SHAPE;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i] || a->shape[i] != out->shape[i])
            return FE_ERR_SHAPE;
    }
    if (a->dtype != DTYPE_FLOAT32 || b->dtype != DTYPE_FLOAT32 ||
        out->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    int n = fe_tensor_numel(a);
    const float *pa = (const float *)a->data;
    const float *pb = (const float *)b->data;
    float       *po = (float *)out->data;

    switch (op) {
        case OP_ADD: for (int i = 0; i < n; i++) po[i] = pa[i] + pb[i]; break;
        case OP_SUB: for (int i = 0; i < n; i++) po[i] = pa[i] - pb[i]; break;
        case OP_MUL: for (int i = 0; i < n; i++) po[i] = pa[i] * pb[i]; break;
        case OP_DIV: for (int i = 0; i < n; i++) po[i] = pa[i] / pb[i]; break;
    }
    return FE_OK;
}

FeStatus fe_add(const FeTensor *a, const FeTensor *b, FeTensor *out) {
    return elementwise_binary(a, b, out, OP_ADD);
}
FeStatus fe_sub(const FeTensor *a, const FeTensor *b, FeTensor *out) {
    return elementwise_binary(a, b, out, OP_SUB);
}
FeStatus fe_mul(const FeTensor *a, const FeTensor *b, FeTensor *out) {
    return elementwise_binary(a, b, out, OP_MUL);
}
FeStatus fe_div(const FeTensor *a, const FeTensor *b, FeTensor *out) {
    return elementwise_binary(a, b, out, OP_DIV);
}

/* in ops/elementwise.c, alongside the tensor-tensor versions */

typedef enum { SOP_ADD, SOP_SUB, SOP_MUL, SOP_DIV } ScalarOp;

static FeStatus scalar_binary(const FeTensor *a, float scalar,
                               FeTensor *out, ScalarOp op) {
    if (!a || !out) return FE_ERR_NULL;
    if (a->ndim != out->ndim) return FE_ERR_SHAPE;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != out->shape[i]) return FE_ERR_SHAPE;
    }
    if (a->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;

    int n = fe_tensor_numel(a);
    const float *pa = (const float *)a->data;
    float       *po = (float *)out->data;

    switch (op) {
        case SOP_ADD: for (int i = 0; i < n; i++) po[i] = pa[i] + scalar; break;
        case SOP_SUB: for (int i = 0; i < n; i++) po[i] = pa[i] - scalar; break;
        case SOP_MUL: for (int i = 0; i < n; i++) po[i] = pa[i] * scalar; break;
        case SOP_DIV: for (int i = 0; i < n; i++) po[i] = pa[i] / scalar; break;
    }
    return FE_OK;
}

FeStatus fe_add_scalar(const FeTensor *a, float s, FeTensor *out) {
    return scalar_binary(a, s, out, SOP_ADD);
}
FeStatus fe_sub_scalar(const FeTensor *a, float s, FeTensor *out) {
    return scalar_binary(a, s, out, SOP_SUB);
}
FeStatus fe_mul_scalar(const FeTensor *a, float s, FeTensor *out) {
    return scalar_binary(a, s, out, SOP_MUL);
}
FeStatus fe_div_scalar(const FeTensor *a, float s, FeTensor *out) {
    return scalar_binary(a, s, out, SOP_DIV);
}

/* Also worth adding while we're here — negation is just mul by -1,
 * but a dedicated function reads clearer at call sites */
FeStatus fe_neg(const FeTensor *a, FeTensor *out) {
    return scalar_binary(a, -1.0f, out, SOP_MUL);
}