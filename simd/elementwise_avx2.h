// simd/elementwise_avx2.h
#ifndef FERRITE_ELEMENTWISE_AVX2_H
#define FERRITE_ELEMENTWISE_AVX2_H

/*
 * AVX2 elementwise kernels: 8-wide float32 with a scalar tail, so they
 * work for any element count (not just multiples of 8).
 *
 * These are buffer-level helpers — callers already validated dtypes,
 * shapes, and pointer ranges. fe_cpu_has_avx2() is checked at the
 * dispatch site (see matmul_avx2.h).
 *
 * Sigmoid is deliberately NOT vectorized here: an accurate AVX2 exp()
 * needs AVX-512ER or a fast-exp approximation, and Ferrite does not
 * ship approximations that change numerics. Scalar expf remains the
 * sigmoid path.
 */
void fe_relu_avx2(const float *in, float *out, int n);
void fe_add_avx2 (const float *a, const float *b, float *out, int n);
void fe_mul_avx2 (const float *a, const float *b, float *out, int n);

#endif // FERRITE_ELEMENTWISE_AVX2_H