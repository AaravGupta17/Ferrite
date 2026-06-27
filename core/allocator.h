// core/allocator.h
#ifndef FERRITE_ALLOCATOR_H
#define FERRITE_ALLOCATOR_H

#include "types.h"
#include "tensor.h"
#include <stddef.h>

/*
 * FeArena — a bump-pointer memory arena.
 *
 * All allocations are O(1). Deallocation is all-or-nothing:
 * call fe_arena_reset() to free everything at once.
 *
 * Two intended uses:
 *   - Weight arena:     allocated at model load, never reset
 *   - Activation arena: reset after every inference call
 */
typedef struct {
    unsigned char *base;    /* start of the buffer          */
    size_t         size;    /* total capacity in bytes       */
    size_t         offset;  /* current bump pointer          */
    size_t         peak;    /* high-water mark (for profiling) */
} FeArena;

/*
 * Initialise an arena backed by a caller-supplied buffer.
 * The arena does NOT own the buffer — caller manages its lifetime.
 */
FeStatus fe_arena_init(FeArena *a, void *buffer, size_t size);

/*
 * Allocate `size` bytes aligned to `align` bytes.
 * align must be a power of two.
 * Returns NULL if the arena is exhausted.
 */
void *fe_arena_alloc(FeArena *a, size_t size, size_t align);

/*
 * Reset the bump pointer to zero.
 * All previously allocated memory is considered free.
 * Does NOT zero the buffer — callers must not rely on zeroed memory.
 */
void fe_arena_reset(FeArena *a);

/*
 * Allocate a tensor whose buffer comes from the arena.
 * The FeTensor metadata struct itself is also arena-allocated.
 * The tensor must NOT be passed to fe_tensor_free() —
 * its memory is owned by the arena.
 */
FeTensor *fe_arena_alloc_tensor(FeArena *a, FeDtype dtype,
                                 int ndim, const int *shape);

/* Diagnostic: bytes used and peak usage. */
size_t fe_arena_used(const FeArena *a);
size_t fe_arena_peak(const FeArena *a);

#endif // FERRITE_ALLOCATOR_H