#include <stdio.h>
#include <math.h>
#include <time.h>
#include "../core/tensor.h"
#include "../ops/ops.h"
#include "../simd/matmul_avx2.h"

#define N    256
#define RUNS 20

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double bench(FeStatus (*fn)(const FeTensor*, const FeTensor*, FeTensor*),
                    FeTensor *A, FeTensor *B, FeTensor *C) {
    fn(A, B, C);  /* warmup */
    double start = now_ms();
    for (int i = 0; i < RUNS; i++) fn(A, B, C);
    return (now_ms() - start) / RUNS;
}

static void verify(FeTensor *A, FeTensor *B) {
    int shape[] = {N, N};
    FeTensor *C1 = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *C2 = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);

    fe_matmul(A, B, C1);
    fe_matmul_avx2(A, B, C2);

    float *c1 = (float *)C1->data;
    float *c2 = (float *)C2->data;
    float max_err = 0.0f;
    for (int i = 0; i < N*N; i++) {
        float err = fabsf(c1[i] - c2[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error vs naive: %.2e %s\n",
       max_err, max_err < 1.0f ? "(PASS)" : "(FAIL)");

    fe_tensor_free(C1);
    fe_tensor_free(C2);
}

int main(void) {
    if (!fe_cpu_has_avx2()) {
        printf("AVX2 not available on this CPU\n");
        return 1;
    }
    printf("AVX2 available\n");

    int shape[] = {N, N};
    FeTensor *A = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *B = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *C = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);

    float *a = (float *)A->data;
    float *b = (float *)B->data;
    for (int i = 0; i < N*N; i++) { a[i] = (float)i * 0.001f; }
    for (int i = 0; i < N*N; i++) { b[i] = (float)i * 0.001f; }

    verify(A, B);

    double t_naive = bench(fe_matmul,      A, B, C);
    double t_avx2  = bench(fe_matmul_avx2, A, B, C);

    double gflops_naive = (2.0 * N*N*N) / (t_naive * 1e6);
    double gflops_avx2  = (2.0 * N*N*N) / (t_avx2  * 1e6);

    printf("Naive: %.2f ms  %.2f GFLOPS\n", t_naive, gflops_naive);
    printf("AVX2:  %.2f ms  %.2f GFLOPS\n", t_avx2,  gflops_avx2);
    printf("Speedup: %.1fx\n", t_naive / t_avx2);

    fe_tensor_free(A);
    fe_tensor_free(B);
    fe_tensor_free(C);
    return 0;
}