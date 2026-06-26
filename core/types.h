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

typedef enum {
    FE_OK            = 0,
    FE_ERR_NULL      = 1,
    FE_ERR_SHAPE     = 2,
    FE_ERR_DTYPE     = 3,
    FE_ERR_NOMEM     = 4,
    FE_ERR_BOUNDS    = 5,
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