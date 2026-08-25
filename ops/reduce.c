/* ops/reduce.c (new file) */
#include "ops.h"
#include <float.h>

typedef enum { RED_SUM, RED_MAX, RED_MIN } ReduceOp;

static FeStatus reduce_axis(const FeTensor *in, int axis, FeTensor *out, ReduceOp op) {
    if (!in || !out) return FE_ERR_NULL;
    if (axis < 0 || axis >= in->ndim) return FE_ERR_SHAPE;
    if (in->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;

    /* out must match in's shape with `axis` removed */
    int out_dim = 0;
    for (int i = 0; i < in->ndim; i++) {
        if (i == axis) continue;
        if (out->shape[out_dim] != in->shape[i]) return FE_ERR_SHAPE;
        out_dim++;
    }
    if (out->ndim != out_dim) return FE_ERR_SHAPE;

    int reduce_len = in->shape[axis];
    int out_numel  = fe_tensor_numel(out);

    /* Walk every output position; for each, walk `axis` and combine */
    int in_idx[FERRITE_MAX_DIMS]  = {0};
    int out_idx[FERRITE_MAX_DIMS] = {0};

    for (int o = 0; o < out_numel; o++) {
        /* Map out_idx -> in_idx, with `axis` set to 0 for the first read */
        int d = 0;
        for (int i = 0; i < in->ndim; i++) {
            if (i == axis) { in_idx[i] = 0; continue; }
            in_idx[i] = out_idx[d++];
        }

        float acc = (op == RED_SUM) ? 0.0f
                  : (op == RED_MAX) ? -FLT_MAX
                  : FLT_MAX;

        for (int k = 0; k < reduce_len; k++) {
            in_idx[axis] = k;
            float v = fe_tensor_get_f32(in, in_idx);
            if (op == RED_SUM) acc += v;
            else if (op == RED_MAX) acc = v > acc ? v : acc;
            else acc = v < acc ? v : acc;
        }

        fe_tensor_set_f32(out, out_idx, acc);

        /* Increment out_idx (odometer-style, same pattern as tensor.c) */
        for (int i = out->ndim - 1; i >= 0; i--) {
            if (++out_idx[i] < out->shape[i]) break;
            out_idx[i] = 0;
        }
    }
    return FE_OK;
}

FeStatus fe_sum(const FeTensor *in, int axis, FeTensor *out) {
    return reduce_axis(in, axis, out, RED_SUM);
}
FeStatus fe_max(const FeTensor *in, int axis, FeTensor *out) {
    return reduce_axis(in, axis, out, RED_MAX);
}
FeStatus fe_min(const FeTensor *in, int axis, FeTensor *out) {
    return reduce_axis(in, axis, out, RED_MIN);
}

/* add to ops/reduce.c, or a new tiny ops/dot.c — reduce.c makes sense since it reuses reduction logic conceptually */
FeStatus fe_dot(const FeTensor *a, const FeTensor *b, float *out) {
    if (!a || !b || !out) return FE_ERR_NULL;
    if (a->ndim != 1 || b->ndim != 1) return FE_ERR_SHAPE;
    if (a->shape[0] != b->shape[0]) return FE_ERR_SHAPE;
    if (a->dtype != DTYPE_FLOAT32 || b->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;

    int n = a->shape[0];
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        int idx[] = {i};
        acc += fe_tensor_get_f32(a, idx) * fe_tensor_get_f32(b, idx);
    }
    *out = acc;
    return FE_OK;
}//
// Created by Aarav Gupta on 26-08-2026.
//
