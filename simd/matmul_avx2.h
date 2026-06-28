// simd/matmul_avx2.h
#ifndef FERRITE_MATMUL_AVX2_H
#define FERRITE_MATMUL_AVX2_H

#include "types.h"
#include "tensor.h"

/*
 * AVX2 tiled matrix multiplication: C = A @ B
 *
 * Uses 8-wide FMA to compute 8 output elements per instruction.
 * Tiles the K dimension to improve cache reuse.
 *
 * Requires: A [M,K], B [K,N], C [M,N] — all contiguous float32.
 * N must be a multiple of 8 for the vectorized path.
 * Falls back to scalar for remainder columns.
 */
FeStatus fe_matmul_avx2(const FeTensor *A, const FeTensor *B, FeTensor *C);

/* Runtime check: returns 1 if AVX2 is available. */
int fe_cpu_has_avx2(void);

#endif // FERRITE_MATMUL_AVX2_H