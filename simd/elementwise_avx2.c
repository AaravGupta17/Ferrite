#include "elementwise_avx2.h"
#include <immintrin.h>

#define VEC_W 8

/* Vectors the common 8-element main loop, then finishes the scalar tail
 * for n % 8 != 0. Exact arithmetic — no approximations, no FMA reordering
 * that would change results. */
void fe_relu_avx2(const float *in, float *out, int n) {
    int i = 0;
    for (; i + VEC_W <= n; i += VEC_W) {
        __m256 x = _mm256_loadu_ps(in + i);
        __m256 zero = _mm256_setzero_ps();
        __m256 r = _mm256_max_ps(x, zero);
        _mm256_storeu_ps(out + i, r);
    }
    for (; i < n; i++) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

void fe_add_avx2(const float *a, const float *b, float *out, int n) {
    int i = 0;
    for (; i + VEC_W <= n; i += VEC_W) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
}

void fe_mul_avx2(const float *a, const float *b, float *out, int n) {
    int i = 0;
    for (; i + VEC_W <= n; i += VEC_W) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] * b[i];
}