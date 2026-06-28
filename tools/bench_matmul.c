#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../core/tensor.h"
#include "../ops/ops.h"

#define N 256

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    int shape[] = {N, N};
    FeTensor *A = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *B = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *C = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);

    float *a = (float *)A->data;
    float *b = (float *)B->data;
    for (int i = 0; i < N*N; i++) { a[i] = (float)i * 0.001f; }
    for (int i = 0; i < N*N; i++) { b[i] = (float)i * 0.001f; }

    /* Warmup */
    fe_matmul(A, B, C);

    int RUNS = 20;
    double start = now_ms();
    for (int i = 0; i < RUNS; i++) fe_matmul(A, B, C);
    double elapsed = (now_ms() - start) / RUNS;

    double gflops = (2.0 * N * N * N) / (elapsed * 1e6);
    printf("Naive matmul %dx%d: %.2f ms/run, %.2f GFLOPS\n",
           N, N, elapsed, gflops);

    fe_tensor_free(A);
    fe_tensor_free(B);
    fe_tensor_free(C);
    return 0;
}
