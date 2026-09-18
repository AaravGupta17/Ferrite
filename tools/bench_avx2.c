#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../core/tensor.h"
#include "../ops/ops.h"
#include "../simd/matmul_avx2.h"
#include "../simd/elementwise_avx2.h"

#define N    256
#define RUNS 20
#define EW    (1 << 20)   /* 1M floats per elementwise bench */

static int json_mode = 0;

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

static double t_mm_naive, t_mm_avx2, t_ew_naive, t_ew_avx2, t_gemm_base, t_gemm_transb;
static double t_ew_c_naive, t_ew_c_avx2;   /* elementwise checksums from both paths */

static void bench_matmul(void) {
    int shape[] = {N, N};
    FeTensor *A = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *B = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *C = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    float *a = (float *)A->data;
    float *b = (float *)B->data;
    for (int i = 0; i < N*N; i++) { a[i] = (float)i * 0.001f; }
    for (int i = 0; i < N*N; i++) { b[i] = (float)i * 0.001f; }

    double t_naive = bench(fe_matmul_scalar, A, B, C);
    double t_avx2  = bench(fe_matmul_avx2,   A, B, C);

    t_mm_naive = t_naive;
    t_mm_avx2  = t_avx2;

    double flops = 2.0 * N*N*N;
    if (json_mode) return;
    printf("MatMul [%dx%dx%d]: naive %.2f ms (%.2f GFLOPS), AVX2 %.2f ms (%.2f GFLOPS), %.1fx\n",
           N, N, N, t_naive, flops / (t_naive * 1e6), t_avx2,
           flops / (t_avx2 * 1e6), t_naive / t_avx2);
    fe_tensor_free(A); fe_tensor_free(B); fe_tensor_free(C);
}

/* Elementwise: the AVX2 kernel (8-wide + scalar tail) runs on the same buffer
 * pair as a plain scalar loop — same FLOPs, same memory traffic. The timed
 * scalar loop's stores are dead (nothing reads o before free), so a -O3
 * compiler can legally remove it; the checksum pass below forces the stores
 * to stay materialized and doubles as a both-paths-agree sanity check. */
static void bench_elementwise(void) {
    float *a = malloc(sizeof(float) * EW);
    float *b = malloc(sizeof(float) * EW);
    float *o = malloc(sizeof(float) * EW);
    for (int i = 0; i < EW; i++) { a[i] = (float)i; b[i] = 1.0f; }

    fe_add_avx2(a, b, o, EW);   /* warmup */
    double start = now_ms();
    for (int r = 0; r < RUNS; r++) fe_add_avx2(a, b, o, EW);
    double t_avx2 = (now_ms() - start) / RUNS;

    /* The naive reference: a plain scalar loop. Its stores are volatile so
     * the optimizer cannot collapse the RUNS passes (each one writes the same
     * o[] before anything reads it) or vectorize the loop — otherwise this
     * "scalar" baseline silently becomes another AVX2 run. The checksum pass
     * after timing then forces the stores to stay materialized. */
    volatile float *ov = o;
    for (int i = 0; i < EW; i++) ov[i] = a[i] + b[i];   /* warmup */
    start = now_ms();
    for (int r = 0; r < RUNS; r++)
        for (int i = 0; i < EW; i++) ov[i] = a[i] + b[i];
    double t_naive = (now_ms() - start) / RUNS;

    /* Force the naive loop's stores to live (else they are dead), and check
     * the two kernels agree: same data, sequential sum must match exactly. */
    for (int i = 0; i < EW; i++) t_ew_c_naive += o[i];
    fe_add_avx2(a, b, o, EW);
    for (int i = 0; i < EW; i++) t_ew_c_avx2 += o[i];

    t_ew_naive = t_naive;
    t_ew_avx2  = t_avx2;

    double gflops = (double)EW / (t_avx2 * 1e6);
    const char *agree = t_ew_c_naive == t_ew_c_avx2 ? "match" : "MISMATCH";
    if (json_mode) return;
    printf("Elementwise add [%d]: naive %.2f ms, AVX2 %.2f ms (%.2f GFLOP/s), %.1fx (checksums %.9g vs %.9g: %s)\n",
           EW, t_naive, t_avx2, gflops, t_naive / t_avx2,
           t_ew_c_naive, t_ew_c_avx2, agree);
    free(a); free(b); free(o);
}

/* GEMM: the base case (no transpose, alpha=1, beta=0) routes to AVX2; the
 * same logical multiply with transB=1 goes down the scalar path. Both do the
 * same FLOPs, so the ratio is the AVX2-vs-scalar GEMM speedup. */
static void bench_gemm_transposes(void) {
    int sA[] = {N, N}, sB[] = {N, N}, sC[] = {N, N};
    FeTensor *A = fe_tensor_alloc(DTYPE_FLOAT32, 2, sA);
    FeTensor *B = fe_tensor_alloc(DTYPE_FLOAT32, 2, sB);
    FeTensor *C = fe_tensor_alloc(DTYPE_FLOAT32, 2, sC);
    for (int i = 0; i < N*N; i++) {
        ((float *)A->data)[i] = (float)i * 0.001f;
        ((float *)B->data)[i] = (float)i * 0.001f;
    }

    fe_gemm(A, 0, B, 0, C, 1.0f, 0.0f);
    double start = now_ms();
    for (int i = 0; i < RUNS; i++) fe_gemm(A, 0, B, 0, C, 1.0f, 0.0f);
    double t_avx2 = (now_ms() - start) / RUNS;

    fe_gemm(A, 0, B, 1, C, 1.0f, 0.0f);
    start = now_ms();
    for (int i = 0; i < RUNS; i++) fe_gemm(A, 0, B, 1, C, 1.0f, 0.0f);
    double t_scalar = (now_ms() - start) / RUNS;

    t_gemm_base  = t_avx2;
    t_gemm_transb = t_scalar;

    double flops = 2.0 * N*N*N;
    if (json_mode) return;
    printf("GEMM [%dx%dx%d]: base(AVX2) %.2f ms (%.2f GFLOPS), transB(scalar) %.2f ms, %.1fx\n",
           N, N, N, t_avx2, flops / (t_avx2 * 1e6), t_scalar,
           t_scalar / t_avx2);
    fe_tensor_free(A); fe_tensor_free(B); fe_tensor_free(C);
}

int main(int argc, char **argv) {
    if (argc > 1 && strncmp(argv[1], "--json", 6) == 0) json_mode = 1;
    if (!fe_cpu_has_avx2()) {
        printf("AVX2 not available on this CPU\n");
        return 1;
    }
    if (!json_mode) printf("AVX2 available\n\n");
    bench_matmul();
    bench_elementwise();
    bench_gemm_transposes();

    if (json_mode) {
        printf("{\n");
        printf("  \"bench\": \"bench_avx2 (model-independent microbench)\",\n");
        printf("  \"matmul_naive_ms\": %.3f,\n", t_mm_naive);
        printf("  \"matmul_avx2_ms\": %.3f,\n", t_mm_avx2);
        printf("  \"matmul_speedup\": %.2f,\n", t_mm_naive / t_mm_avx2);
        printf("  \"elem_naive_ms\": %.3f,\n", t_ew_naive);
        printf("  \"elem_avx2_ms\": %.3f,\n", t_ew_avx2);
        printf("  \"elem_speedup\": %.2f,\n", t_ew_naive / t_ew_avx2);
        printf("  \"elem_naive_checksum\": %.6g,\n", t_ew_c_naive);
        printf("  \"elem_avx2_checksum\": %.6g,\n", t_ew_c_avx2);
        printf("  \"gemm_base_ms\": %.3f,\n", t_gemm_base);
        printf("  \"gemm_transb_ms\": %.3f,\n", t_gemm_transb);
        printf("  \"gemm_speedup\": %.2f\n", t_gemm_transb / t_gemm_base);
        printf("}\n");
    }
    return 0;
}