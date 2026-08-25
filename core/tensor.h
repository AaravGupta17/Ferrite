// core/tensor.h
#ifndef FERRITE_TENSOR_H
#define FERRITE_TENSOR_H

#include "types.h"
#include <stdbool.h>
#include <math.h>

/*
 * FeTensor — the universal data structure in Ferrite.
 *
 * A tensor is a strided view into a flat memory buffer.
 * The same buffer can be viewed as different shapes (reshape),
 * in different orders (transpose), or as a subset (slice) —
 * all without copying data.
 *
 * Ownership: a tensor with owns_data=true is responsible for
 * freeing its buffer. Views (owns_data=false) never free.
 */
typedef struct {
    void    *data;
    FeDtype  dtype;
    int      ndim;
    int      shape  [FERRITE_MAX_DIMS];
    int      strides[FERRITE_MAX_DIMS];  /* in elements, not bytes */
    size_t   nbytes;
    bool     owns_data;
} FeTensor;

/* --- Lifecycle --- */

/*
 * Allocate a new tensor with a freshly malloc'd buffer.
 * strides are computed row-major (C order).
 * Returns NULL on allocation failure.
 */
FeTensor *fe_tensor_alloc(FeDtype dtype, int ndim, const int *shape);

/*
 * Create a non-owning view of an existing buffer.
 * Caller retains ownership of data.
 */
FeTensor *fe_tensor_from_data(void *data, FeDtype dtype,
                               int ndim, const int *shape);

FeTensor *fe_tensor_slice(const FeTensor *t, int axis, int start, int len);
FeTensor *fe_tensor_broadcast_to(const FeTensor *t, int ndim, const int *target_shape);
bool fe_tensor_allclose(const FeTensor *a, const FeTensor *b, float tol);
/*
 * Free tensor metadata. If owns_data, also frees the buffer.
 */
void fe_tensor_free(FeTensor *t);

/* --- Shape operations (never copy data) --- */

/*
 * Return a new view with axes ax0 and ax1 swapped.
 * No data movement. O(1).
 */
FeTensor *fe_tensor_transpose(const FeTensor *t, int ax0, int ax1);

/*
 * Attempt to reshape. Succeeds only if tensor is contiguous.
 * Returns a new view on success, NULL on failure.
 */
FeTensor *fe_tensor_reshape(const FeTensor *t, int ndim, const int *shape);

/* --- Memory layout queries --- */

/* True if strides match row-major C order with no gaps. */
bool fe_tensor_is_contiguous(const FeTensor *t);

/*
 * Return a contiguous copy. If already contiguous, returns a
 * new owning tensor with copied data. Caller must free.
 */
FeTensor *fe_tensor_contiguous(const FeTensor *t);

/* Total number of elements. */
int fe_tensor_numel(const FeTensor *t);

/* --- Element access (for testing/debugging only, not hot path) --- */

float   fe_tensor_get_f32(const FeTensor *t, const int *idx);
void    fe_tensor_set_f32(FeTensor *t, const int *idx, float val);

/* --- Utilities --- */
void    fe_tensor_print(const FeTensor *t);  /* debug printing */
FeStatus fe_tensor_copy(FeTensor *dst, const FeTensor *src);

#endif // FERRITE_TENSOR_H