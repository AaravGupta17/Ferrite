// core/types.h
#ifndef FERRITE_TYPES_H
#define FERRITE_TYPES_H

#include <stddef.h>
#include <stdint.h>

#define FERRITE_MAX_DIMS 8

typedef enum {
    DTYPE_FLOAT32 = 0,
    DTYPE_INT8    = 1,
    DTYPE_INT32   = 2,
} FeDtype;

/*
 * FeStatus — the universal return type for fallible Ferrite calls.
 *
 * Contract: every public function that can fail returns FeStatus and
 * nothing else; FE_OK means full success with all outputs written.
 * Constructors that return pointers signal failure with NULL instead
 * (allocation happens before validation is possible). Callees never
 * free caller memory on error, and never partially mutate an output.
 */
typedef enum {
    FE_OK        = 0,
    FE_ERR_NULL  = 1,   /* required pointer argument was NULL          */
    FE_ERR_SHAPE = 2,   /* shape/rank mismatch or unsupported shape    */
    FE_ERR_DTYPE = 3,   /* dtype mismatch, unknown, or not supported   */
    FE_ERR_NOMEM = 4,   /* allocation failed                           */
    FE_ERR_BOUNDS= 5,   /* index out of range                          */
    FE_ERR_IO    = 6,   /* file open/read/write/format failure         */
} FeStatus;

static inline size_t fe_dtype_size(FeDtype dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return 4;
        case DTYPE_INT8:    return 1;
        case DTYPE_INT32:   return 4;
        default:            return 0;
    }
}

#endif // FERRITE_TYPES_H