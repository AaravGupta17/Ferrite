// core/tensor.c
#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* Compute row-major strides from shape. strides[i] = product(shape[i+1..ndim-1]) */
static void compute_strides(int ndim, const int *shape, int *strides) {
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

FeTensor *fe_tensor_alloc(FeDtype dtype, int ndim, const int *shape) {
    assert(ndim > 0 && ndim <= FERRITE_MAX_DIMS);

    FeTensor *t = calloc(1, sizeof(FeTensor));
    if (!t) return NULL;

    t->dtype = dtype;
    t->ndim  = ndim;
    memcpy(t->shape, shape, ndim * sizeof(int));
    compute_strides(ndim, shape, t->strides);

    t->nbytes = fe_dtype_size(dtype);
    for (int i = 0; i < ndim; i++) {
        t->nbytes *= (size_t)shape[i];
    }

    t->data = malloc(t->nbytes);
    if (!t->data) { free(t); return NULL; }

    t->owns_data = true;
    return t;
}

FeTensor *fe_tensor_from_data(void *data, FeDtype dtype,
                               int ndim, const int *shape) {
    assert(ndim > 0 && ndim <= FERRITE_MAX_DIMS);

    FeTensor *t = calloc(1, sizeof(FeTensor));
    if (!t) return NULL;

    t->data  = data;
    t->dtype = dtype;
    t->ndim  = ndim;
    memcpy(t->shape, shape, ndim * sizeof(int));
    compute_strides(ndim, shape, t->strides);

    t->nbytes = fe_dtype_size(dtype);
    for (int i = 0; i < ndim; i++) {
        t->nbytes *= (size_t)shape[i];
    }

    t->owns_data = false;
    return t;
}

void fe_tensor_free(FeTensor *t) {
    if (!t) return;
    if (t->owns_data && t->data) free(t->data);
    free(t);
}

FeTensor *fe_tensor_transpose(const FeTensor *t, int ax0, int ax1) {
    assert(ax0 >= 0 && ax0 < t->ndim);
    assert(ax1 >= 0 && ax1 < t->ndim);

    /* Allocate metadata only — no data copy */
    FeTensor *view = calloc(1, sizeof(FeTensor));
    if (!view) return NULL;

    *view = *t;                     /* copy all fields including data pointer */
    view->owns_data = false;        /* view never owns */

    /* Swap shape and strides for the two axes */
    int tmp_shape   = view->shape  [ax0]; view->shape  [ax0] = view->shape  [ax1]; view->shape  [ax1] = tmp_shape;
    int tmp_stride  = view->strides[ax0]; view->strides[ax0] = view->strides[ax1]; view->strides[ax1] = tmp_stride;

    return view;
}

bool fe_tensor_is_contiguous(const FeTensor *t) {
    int expected = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->strides[i] != expected) return false;
        expected *= t->shape[i];
    }
    return true;
}

int fe_tensor_numel(const FeTensor *t) {
    int n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

FeTensor *fe_tensor_contiguous(const FeTensor *t) {
    if (fe_tensor_is_contiguous(t)) {
        /* Already contiguous: return a new owning copy */
        FeTensor *copy = fe_tensor_alloc(t->dtype, t->ndim, t->shape);
        if (!copy) return NULL;
        memcpy(copy->data, t->data, t->nbytes);
        return copy;
    }

    /* Non-contiguous: must copy element by element to repack */
    FeTensor *out = fe_tensor_alloc(t->dtype, t->ndim, t->shape);
    if (!out) return NULL;

    int numel = fe_tensor_numel(t);
    int idx[FERRITE_MAX_DIMS] = {0};

    for (int i = 0; i < numel; i++) {
        /* Compute source offset from multi-dimensional index */
        int src_off = 0;
        for (int d = 0; d < t->ndim; d++) src_off += idx[d] * t->strides[d];

        /* Destination is always contiguous */
        size_t elem = fe_dtype_size(t->dtype);
        memcpy((char *)out->data + i * elem,
               (char *)t->data  + src_off * elem, elem);

        /* Increment multi-dimensional index */
        for (int d = t->ndim - 1; d >= 0; d--) {
            if (++idx[d] < t->shape[d]) break;
            idx[d] = 0;
        }
    }
    return out;
}

FeTensor *fe_tensor_reshape(const FeTensor *t, int ndim, const int *shape) {
    if (!fe_tensor_is_contiguous(t)) return NULL;  /* must be contiguous */

    /* Verify element count is preserved */
    int new_numel = 1;
    for (int i = 0; i < ndim; i++) new_numel *= shape[i];
    if (new_numel != fe_tensor_numel(t)) return NULL;

    FeTensor *view = calloc(1, sizeof(FeTensor));
    if (!view) return NULL;

    view->data      = t->data;
    view->dtype     = t->dtype;
    view->nbytes    = t->nbytes;
    view->ndim      = ndim;
    view->owns_data = false;
    memcpy(view->shape, shape, ndim * sizeof(int));
    compute_strides(ndim, shape, view->strides);
    return view;
}

float fe_tensor_get_f32(const FeTensor *t, const int *idx) {
    assert(t->dtype == DTYPE_FLOAT32);
    int off = 0;
    for (int d = 0; d < t->ndim; d++) off += idx[d] * t->strides[d];
    return ((float *)t->data)[off];
}

void fe_tensor_set_f32(FeTensor *t, const int *idx, float val) {
    assert(t->dtype == DTYPE_FLOAT32);
    int off = 0;
    for (int d = 0; d < t->ndim; d++) off += idx[d] * t->strides[d];
    ((float *)t->data)[off] = val;
}

FeStatus fe_tensor_copy(FeTensor *dst, const FeTensor *src) {
    if (dst->dtype != src->dtype)         return FE_ERR_DTYPE;
    if (fe_tensor_numel(dst) != fe_tensor_numel(src)) return FE_ERR_SHAPE;

    if (fe_tensor_is_contiguous(dst) && fe_tensor_is_contiguous(src)) {
        memcpy(dst->data, src->data, src->nbytes);
        return FE_OK;
    }
    /* Fall back to element-wise copy for non-contiguous tensors */
    int numel = fe_tensor_numel(src);
    int idx[FERRITE_MAX_DIMS] = {0};
    size_t elem = fe_dtype_size(src->dtype);

    for (int i = 0; i < numel; i++) {
        int src_off = 0, dst_off = 0;
        for (int d = 0; d < src->ndim; d++) src_off += idx[d] * src->strides[d];
        for (int d = 0; d < dst->ndim; d++) dst_off += idx[d] * dst->strides[d];
        memcpy((char *)dst->data + dst_off * elem,
               (char *)src->data + src_off * elem, elem);
        for (int d = src->ndim - 1; d >= 0; d--) {
            if (++idx[d] < src->shape[d]) break;
            idx[d] = 0;
        }
    }
    return FE_OK;
}

void fe_tensor_print(const FeTensor *t) {
    printf("Tensor[dtype=%d, ndim=%d, shape=(", t->dtype, t->ndim);
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->shape[i], i < t->ndim - 1 ? ", " : "");
    }
    printf("), strides=(");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->strides[i], i < t->ndim - 1 ? ", " : "");
    }
    printf("), contiguous=%s, nbytes=%zu]\n",
           fe_tensor_is_contiguous(t) ? "true" : "false", t->nbytes);
}