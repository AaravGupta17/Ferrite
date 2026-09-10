// parallel/parallel_gemm.c
/*
 * Parallel matmul: C = A @ B with output rows split across the worker pool.
 * Each worker writes a disjoint row range of C — the classic embarrassingly
 * parallel GEMM split — so no per-element synchronization is needed, only the
 * pool's join barrier at fe_threadpool_wait.
 *
 * The per-row compute is the same scalar reference loop fe_matmul_scalar
 * uses, indexed covariantly (i-k-j), so a parallel result is validated
 * against fe_matmul as an oracle: bit-identical for integer-exact inputs and
 * within rounding for general floats.
 */
#include "parallel_gemm.h"
#include "threadpool.h"
#include "../core/tensor.h"
#include <string.h>

typedef struct {
    const FeTensor *A, *B;
    FeTensor       *C;
    int             M, K, N, nchunks;
} GemmArgs;

/* Worker: compute the row range assigned to chunk index `cidx`; the range is
 * derived from (M, nchunks) alone, so the shared args are read-only and no
 * submit-side mutation races a running worker. */
static void gemm_chunk(void *ctx, int cidx) {
    GemmArgs *g = (GemmArgs *)ctx;
    int base = g->M / g->nchunks;
    int rem  = g->M % g->nchunks;
    int row0 = cidx * base + (cidx < rem ? cidx : rem);
    int row1 = row0 + base + (cidx < rem ? 1 : 0);

    const float *a = (const float *)g->A->data;
    const float *b = (const float *)g->B->data;
    float *c = (float *)g->C->data;

    for (int i = row0; i < row1; i++) {
        for (int k = 0; k < g->K; k++) {
            float a_ik = a[i * g->K + k];
            for (int j = 0; j < g->N; j++) {
                c[i * g->N + j] += a_ik * b[k * g->N + j];
            }
        }
    }
}

FeStatus fe_matmul_parallel(const FeTensor *A, const FeTensor *B,
                            FeTensor *C, FeThreadPool *pool) {
    if (!A || !B || !C) return FE_ERR_NULL;
    if (A->ndim != 2 || B->ndim != 2 || C->ndim != 2) return FE_ERR_SHAPE;
    if (A->dtype != DTYPE_FLOAT32 ||
        B->dtype != DTYPE_FLOAT32 ||
        C->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];
    if (B->shape[0] != K) return FE_ERR_SHAPE;
    if (C->shape[0] != M || C->shape[1] != N) return FE_ERR_SHAPE;

    memset(C->data, 0, (size_t)M * N * sizeof(float));

    GemmArgs g = { A, B, C, M, K, N, M };
    if (g.M > 0 && pool && pool->nthreads > 1) {
        g.nchunks = pool->nthreads < M ? pool->nthreads : M;

        for (int c = 0; c < g.nchunks; c++) {
            FeStatus s = fe_threadpool_submit(pool, gemm_chunk, &g, c);
            if (s != FE_OK) return s;
        }
        return fe_threadpool_wait(pool);
    }

    /* Sequential path (pool == NULL, 1 thread, or empty). */
    g.nchunks = 1;
    gemm_chunk(&g, 0);
    return FE_OK;
}