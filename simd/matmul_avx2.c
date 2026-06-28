#include "matmul_avx2.h"
#include <string.h>
#include <immintrin.h>
#include <cpuid.h>

#define NR 8

int fe_cpu_has_avx2(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx)) return 0;
    if (eax < 7) return 0;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx >> 5) & 1;
}

/*
 * Compute mc x NR tile of C.
 * A_tile: [mc x kc] packed row-major
 * B_tile: [kc x NR] packed row-major
 * C_col:  pointer to C[row_start, nc] with row stride N
 */
static void kernel_mc_nr(
        const float *A_tile, int mc, int kc,
        const float *B_tile,
        float *C_col, int N)
{
    __m256 acc[64];  /* max MC rows */

    for (int i = 0; i < mc; i++) {
        acc[i] = _mm256_loadu_ps(C_col + i * N);
    }

    for (int k = 0; k < kc; k++) {
        __m256 b_row = _mm256_loadu_ps(B_tile + k * NR);
        for (int i = 0; i < mc; i++) {
            __m256 a_ik = _mm256_set1_ps(A_tile[i * kc + k]);
            acc[i] = _mm256_fmadd_ps(a_ik, b_row, acc[i]);
        }
    }

    for (int i = 0; i < mc; i++) {
        _mm256_storeu_ps(C_col + i * N, acc[i]);
    }
}

FeStatus fe_matmul_avx2(const FeTensor *A, const FeTensor *B, FeTensor *C) {
    if (!A || !B || !C)          return FE_ERR_NULL;
    if (A->ndim != 2 || B->ndim != 2 || C->ndim != 2) return FE_ERR_SHAPE;

    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];

    if (B->shape[0] != K) return FE_ERR_SHAPE;
    if (C->shape[0] != M) return FE_ERR_SHAPE;
    if (C->shape[1] != N) return FE_ERR_SHAPE;

    if (A->dtype != DTYPE_FLOAT32 ||
        B->dtype != DTYPE_FLOAT32 ||
        C->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    const float *a = (const float *)A->data;
    const float *b = (const float *)B->data;
    float       *c = (float *)C->data;

    memset(c, 0, (size_t)M * N * sizeof(float));

    /* Tile sizes — tunable */
    int MC = 64;
    int KC = 256;

    /* Packing buffers — sized for max tile */
    float A_tile[64 * 256];
    float B_tile[256 * NR];

    for (int mc = 0; mc < M; mc += MC) {
        int actual_mc = mc + MC < M ? MC : M - mc;

        for (int kc = 0; kc < K; kc += KC) {
            int actual_kc = kc + KC < K ? KC : K - kc;

            /* Pack A tile [actual_mc x actual_kc] */
            for (int i = 0; i < actual_mc; i++)
                for (int k = 0; k < actual_kc; k++)
                    A_tile[i * actual_kc + k] = a[(mc + i) * K + (kc + k)];

            /* Vectorized columns */
            for (int nc = 0; nc + NR <= N; nc += NR) {

                /* Pack B tile [actual_kc x NR] */
                for (int k = 0; k < actual_kc; k++)
                    for (int j = 0; j < NR; j++)
                        B_tile[k * NR + j] = b[(kc + k) * N + (nc + j)];

                /* C pointer: row mc, col nc */
                kernel_mc_nr(A_tile, actual_mc, actual_kc,
                             B_tile,
                             c + mc * N + nc, N);
            }

            /* Scalar remainder */
            int nc_rem = (N / NR) * NR;
            for (int i = 0; i < actual_mc; i++)
                for (int j = nc_rem; j < N; j++) {
                    float sum = c[(mc + i) * N + j];
                    for (int k = 0; k < actual_kc; k++)
                        sum += A_tile[i * actual_kc + k] * b[(kc + k) * N + j];
                    c[(mc + i) * N + j] = sum;
                }
        }
    }

    return FE_OK;
}