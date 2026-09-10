/*
 * tools/bench_model.c — model-relevant benchmark (Stage 8).
 *
 * Loads the AcousticLeakNet demo model, then reports:
 *   - end-to-end inference latency (mean over N runs, warmup first)
 *   - per-op profiler breakdown
 *   - memory: weight bytes, planned vs naive activation footprint, peak
 *   - naive-vs-AVX2 matmul speedup for every MatMul shape in the model
 *
 * The naive-vs-AVX2 table is derived from the loaded graph, not hardcoded,
 * so it stays correct if the model changes.
 */
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "../core/tensor.h"
#include "../core/allocator.h"
#include "../graph/graph.h"
#include "../planner/memory_planner.h"
#include "../runtime/engine.h"
#include "../importer/onnx.h"
#include "../ops/ops.h"
#include "../simd/matmul_avx2.h"
#include "../tools/profiler.h"
#include "../tools/bench.h"

#define WEIGHT_BUF_SIZE (64 * 1024 * 1024)
#define ACT_BUF_SIZE    (32 * 1024 * 1024)

static unsigned char weight_buf[WEIGHT_BUF_SIZE] __attribute__((aligned(64)));
static unsigned char act_buf   [ACT_BUF_SIZE]    __attribute__((aligned(64)));

#define RUNS 20

typedef struct {
    FeRuntime *rt;
    FeTensor  *in;
    FeTensor  *out;
    FeStatus   s;
} run_args;

static void run_inference(void *p) {
    run_args *a = (run_args *)p;
    a->s = fe_runtime_run(a->rt, a->in, a->out);
}

typedef struct {
    FeTensor *A, *B, *C;
} mm_args;

static void mm_scalar(void *p) { mm_args *a = (mm_args *)p; fe_matmul_scalar(a->A, a->B, a->C); }
static void mm_avx2  (void *p) { mm_args *a = (mm_args *)p; fe_matmul_avx2  (a->A, a->B, a->C); }

int main(void) {
    fe_bench_header("Ferrite Benchmark — AcousticLeakNet");

    /* ---- Load model ---- */
    FeGraph g;
    FeArena weight_arena;
    fe_arena_init(&weight_arena, weight_buf, WEIGHT_BUF_SIZE);

    double t_load = fe_bench_ms();
    if (fe_onnx_load(&g, &weight_arena, "tests/acousticleaknet.onnx") != FE_OK) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    t_load = fe_bench_ms() - t_load;

    /* ---- End-to-end latency ---- */
    FeRuntime rt;
    if (fe_runtime_init(&rt, &g,
                        weight_buf, WEIGHT_BUF_SIZE,
                        act_buf,    ACT_BUF_SIZE) != FE_OK) {
        fprintf(stderr, "runtime init failed\n");
        return 1;
    }

    int in_shape[]  = {1, 1, 1024};
    int out_shape[] = {1, 3};
    FeTensor *input  = fe_tensor_alloc(DTYPE_FLOAT32, 3, in_shape);
    FeTensor *output = fe_tensor_alloc(DTYPE_FLOAT32, 2, out_shape);
    float *sig = (float *)input->data;
    for (int i = 0; i < 1024; i++)
        sig[i] = sinf(2.0f * 3.14159f * 440.0f * i / 8000.0f);

    FeProfiler prof;
    fe_profiler_init(&prof);
    rt.profiler = &prof;

    run_args ra = { &rt, input, output, FE_OK };
    if (fe_runtime_run(&rt, input, output) != FE_OK) {
        fprintf(stderr, "inference failed\n");
        return 1;
    }
    fe_profiler_reset(&prof);

    double t_infer = fe_bench_mean(RUNS, run_inference, &ra);
    double flops = /* approximate useful flops for the whole graph */
        2.0 * (16*7*1024 + 32*80*1024 + 64*96*1024 + 65536*128 + 128*3);

    printf("Load:        %.2f ms\n", t_load);
    printf("Inference:   %.3f ms/run (mean of %d)\n", t_infer, RUNS);
    printf("Throughput:  %.1f inf/s\n", 1000.0 / t_infer);
    printf("Useful FLOPs: %.2f MFLOP/run, %.1f GFLOPS dispatch\n\n",
           flops / 1e6, fe_bench_gflops(t_infer, flops));

    fe_profiler_print(&prof);

    /* ---- Memory ---- */
    FePlan plan;
    if (fe_plan_memory(&g, &plan) == FE_OK) {
        size_t naive = 0;
        for (int i = 0; i < plan.n_lifetimes; i++)
            naive += ((size_t)plan.lifetimes[i].size_bytes + 63) & ~(size_t)63;
        printf("\nMemory:\n");
        printf("  Weights:   %.2f MB\n",
               (double)fe_arena_used(&weight_arena) / (1024 * 1024));
        printf("  Planned:   %.2f KB activations\n",
               (double)plan.total_activation_bytes / 1024.0);
        printf("  Naive:     %.2f KB activations (%.1f%% savings)\n",
               naive / 1024.0,
               100.0 * (1.0 - (double)plan.total_activation_bytes / naive));
        printf("  Peak used: %.2f KB (incl. metadata)\n",
               (double)fe_arena_peak(&rt.activation_arena) / 1024.0);
    }

    /* ---- Naive-vs-AVX2 table, one row per distinct MatMul shape ---- */
    printf("\nMatMul naive vs AVX2 (model shapes):\n");
    printf("  %-22s %8s %8s %8s %7s\n", "MxKxN", "naive ms", "avx2 ms", "speedup", "GFLOPS");
    int seen_shapes[8][3], n_seen = 0;
    for (int n = 0; n < g.n_nodes; n++) {
        FeNode *node = &g.nodes[n];
        if (node->op != FE_OP_MATMUL || node->n_inputs < 2) continue;
        int M = g.tensors[node->inputs[0]].shape[0];
        int K = g.tensors[node->inputs[0]].shape[1];
        int N = g.tensors[node->inputs[1]].shape[1];
        if (M <= 0 || K <= 0 || N <= 0) continue;

        int dup = 0;
        for (int i = 0; i < n_seen; i++)
            if (seen_shapes[i][0] == M && seen_shapes[i][1] == K && seen_shapes[i][2] == N) dup = 1;
        if (dup) continue;
        seen_shapes[n_seen][0] = M; seen_shapes[n_seen][1] = K; seen_shapes[n_seen][2] = N;
        n_seen++;

        int sA[] = {M, K}, sB[] = {K, N}, sC[] = {M, N};
        FeTensor *A  = fe_tensor_alloc(DTYPE_FLOAT32, 2, sA);
        FeTensor *B  = fe_tensor_alloc(DTYPE_FLOAT32, 2, sB);
        FeTensor *C1 = fe_tensor_alloc(DTYPE_FLOAT32, 2, sC);
        FeTensor *C2 = fe_tensor_alloc(DTYPE_FLOAT32, 2, sC);
        assert(A && B && C1 && C2);
        fe_rand_uniform(A, -1.0f, 1.0f, 4242u + (unsigned)n);
        fe_rand_uniform(B, -1.0f, 1.0f, 7777u + (unsigned)n);

        mm_args nv = { A, B, C1 };
        mm_args av = { A, B, C2 };
        double t_n = fe_bench_mean(RUNS, mm_scalar, &nv);
        double t_a = fe_bench_mean(RUNS, mm_avx2,   &av);
        printf("  [%3d x%5d x%5d] %8.3f %8.3f %6.1fx %7.1f\n",
               M, K, N, t_n, t_a, t_n / t_a,
               fe_bench_gflops(t_a, 2.0 * M * K * N));

        fe_tensor_free(A); fe_tensor_free(B); fe_tensor_free(C1); fe_tensor_free(C2);
    }

    fe_tensor_free(input);
    fe_tensor_free(output);
    return 0;
}