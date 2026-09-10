// simd/backend.c — SIMD backend selection (Section 2.5).
//
// Chooses the active FeSimdOps table once and caches it:
//   - if AVX2 is compiled in AND the CPU reports it, use fe_matmul_avx2 +
//     the AVX2 elementwise kernels;
//   - otherwise the scalar table, whose matmul slot is NULL so the ops
//     layer resolves it to fe_matmul_scalar.
// Compile-out (FERRITE_NO_AVX2) is handled by the preprocessor so a scalar-
// only build never references AVX2 instructions or symbols.

#include "backend.h"
#include "matmul_avx2.h"
#include "elementwise_avx2.h"

#ifndef FERRITE_NO_AVX2
static const FeSimdOps k_avx2_ops = {
    .matmul = fe_matmul_avx2,
    .relu   = fe_relu_avx2,
    .add    = fe_add_avx2,
    .mul    = fe_mul_avx2,
};
#endif

FeSimdBackend fe_simd_backend(void) {
#ifndef FERRITE_NO_AVX2
    static int cached = -1;
    if (cached < 0)
        cached = fe_cpu_has_avx2() ? (int)FE_SIMD_AVX2 : (int)FE_SIMD_SCALAR;
    return (FeSimdBackend)cached;
#else
    return FE_SIMD_SCALAR;
#endif
}

const FeSimdOps *fe_simd_ops(void) {
#ifndef FERRITE_NO_AVX2
    static const FeSimdOps *table = NULL;
    if (!table) {
        if (fe_simd_backend() == FE_SIMD_AVX2)
            table = &k_avx2_ops;
        else
            table = fe_simd_scalar_ops();
    }
    return table;
#else
    return fe_simd_scalar_ops();
#endif
}