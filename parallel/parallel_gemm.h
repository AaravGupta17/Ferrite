// parallel/parallel_gemm.h
#ifndef FERRITE_PARALLEL_GEMM_H
#define FERRITE_PARALLEL_GEMM_H

#include "types.h"
#include "../core/tensor.h"
#include <stddef.h>

typedef struct FeThreadPool FeThreadPool;   /* forward decl */

/* Parallel general matmul: C = A @ B with the output rows split across the
 * worker pool. Every worker writes a disjoint row range of C, so no
 * synchronization is needed beyond the pool's join barrier. Each worker uses
 * the scalar reference loop on an A-row slice (fe_tensor_slice — O(1) view),
 * so parallelism is validated against fe_matmul, not re-derived.
 *
 * Passing pool == NULL (or nthreads == 1) runs the sequential reference path
 * — the same code the parallelism is validated against. Shapes/dtypes are
 * validated identically to fe_matmul. */
FeStatus fe_matmul_parallel(const FeTensor *A, const FeTensor *B,
                            FeTensor *C, FeThreadPool *pool);

#endif // FERRITE_PARALLEL_GEMM_H