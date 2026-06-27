// core/allocator.c
#include "allocator.h"
#include <string.h>
#include <assert.h>
#include <stdio.h>

FeStatus fe_arena_init(FeArena *a, void *buffer, size_t size) {
    if (!a || !buffer || size == 0) return FE_ERR_NULL;
    a->base   = (unsigned char *)buffer;
    a->size   = size;
    a->offset = 0;
    a->peak   = 0;
    return FE_OK;
}

void *fe_arena_alloc(FeArena *a, size_t size, size_t align) {
    assert(a != NULL);
    assert((align & (align - 1)) == 0);

    /* Align the actual pointer address, not just the offset */
    uintptr_t current    = (uintptr_t)(a->base + a->offset);
    uintptr_t aligned    = (current + align - 1) & ~(uintptr_t)(align - 1);
    size_t    new_offset = aligned - (uintptr_t)a->base + size;

    if (new_offset > a->size) return NULL;

    a->offset = new_offset;
    if (a->offset > a->peak) a->peak = a->offset;

    return (void *)aligned;
}

void fe_arena_reset(FeArena *a) {
    assert(a != NULL);
    a->offset = 0;
    /* peak is intentionally preserved across resets for profiling */
}

FeTensor *fe_arena_alloc_tensor(FeArena *a, FeDtype dtype,
                                 int ndim, const int *shape) {
    assert(ndim > 0 && ndim <= FERRITE_MAX_DIMS);

    /* Allocate the metadata struct from the arena */
    FeTensor *t = fe_arena_alloc(a, sizeof(FeTensor), _Alignof(FeTensor));
    if (!t) return NULL;

    /* Compute total element count and buffer size */
    size_t nbytes = fe_dtype_size(dtype);
    for (int i = 0; i < ndim; i++) nbytes *= (size_t)shape[i];

    /* Allocate the data buffer from the same arena */
    void *data = fe_arena_alloc(a, nbytes, 64);  /* 64-byte align for SIMD */
    if (!data) return NULL;

    /* Initialise the tensor struct in-place */
    t->data      = data;
    t->dtype     = dtype;
    t->ndim      = ndim;
    t->nbytes    = nbytes;
    t->owns_data = false;  /* arena owns everything */

    memcpy(t->shape, shape, ndim * sizeof(int));

    /* Compute row-major strides */
    t->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * shape[i + 1];
    }

    return t;
}

size_t fe_arena_used(const FeArena *a) { return a->offset; }
size_t fe_arena_peak(const FeArena *a) { return a->peak;   }