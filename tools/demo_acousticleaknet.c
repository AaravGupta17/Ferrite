/*
 * Ferrite Demo: AcousticLeakNet Inference
 *
 * Loads a 1D CNN trained for acoustic pipe leak detection,
 * runs inference on a synthetic input signal, and prints:
 *   - Per-operator latency breakdown
 *   - Class probabilities (no leak / small leak / large leak)
 *   - Memory usage statistics
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../core/tensor.h"
#include "../core/allocator.h"
#include "../graph/graph.h"
#include "../runtime/engine.h"
#include "../importer/onnx.h"
#include "../tools/profiler.h"

/* 32MB weight buffer — fits the large linear layer */
#define WEIGHT_BUF_SIZE   (64 * 1024 * 1024)
/* 32MB activation buffer — conv outputs are large */
#define ACT_BUF_SIZE      (32 * 1024 * 1024)

static unsigned char weight_buf[WEIGHT_BUF_SIZE] __attribute__((aligned(64)));
static unsigned char act_buf   [ACT_BUF_SIZE]    __attribute__((aligned(64)));

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "tests/acousticleaknet.onnx";

    printf("===========================================\n");
    printf("  Ferrite — AcousticLeakNet Demo\n");
    printf("===========================================\n\n");

    /* Step 1: Load ONNX model */
    printf("[1/4] Loading model: %s\n", model_path);
    double t0 = now_ms();

    FeGraph g;
    FeArena weight_arena;
    fe_arena_init(&weight_arena, weight_buf, WEIGHT_BUF_SIZE);

    FeStatus s = fe_onnx_load(&g, &weight_arena, model_path);
    if (s != FE_OK) {
        fprintf(stderr, "Failed to load model: %d\n", s);
        return 1;
    }

    double load_ms = now_ms() - t0;
    printf("    Loaded in %.1f ms\n", load_ms);
    printf("    Nodes:   %d\n", g.n_nodes);
    printf("    Tensors: %d\n", g.n_tensors);
    printf("    Weights: %.1f MB\n",
           (double)fe_arena_used(&weight_arena) / (1024 * 1024));

    /* Step 2: Initialise runtime */
    printf("\n[2/4] Initialising runtime\n");

    FeRuntime rt;
    s = fe_runtime_init(&rt, &g,
                         weight_buf, WEIGHT_BUF_SIZE,
                         act_buf,    ACT_BUF_SIZE);
    if (s != FE_OK) {
        fprintf(stderr, "Runtime init failed: %d\n", s);
        return 1;
    }

    /* Weights already loaded by ONNX importer into weight_arena */
    printf("    Activation buffer: %.1f MB\n",
           (double)ACT_BUF_SIZE / (1024 * 1024));

    /* Step 3: Build input — synthetic acoustic signal */
    printf("\n[3/4] Running inference\n");

    int in_shape[]  = {1, 1, 1024};
    int out_shape[] = {1, 3};
    FeTensor *input  = fe_tensor_alloc(DTYPE_FLOAT32, 3, in_shape);
    FeTensor *output = fe_tensor_alloc(DTYPE_FLOAT32, 2, out_shape);

    /* Synthetic signal: sine wave + noise (simulates acoustic sensor) */
    float *sig = (float *)input->data;
    for (int i = 0; i < 1024; i++) {
        sig[i] = sinf(2.0f * 3.14159f * 440.0f * i / 8000.0f)
               + 0.1f * sinf(2.0f * 3.14159f * 1200.0f * i / 8000.0f);
    }

    /* Attach profiler */
    FeProfiler prof;
    fe_profiler_init(&prof);
    rt.profiler = &prof;

    /* Warmup run */
    s = fe_runtime_run(&rt, input, output);
    if (s != FE_OK) {
        fprintf(stderr, "Inference failed: %d\n", s);
        return 1;
    }
    fe_profiler_reset(&prof);

    /* Timed runs */
    int N_RUNS = 5;
    double t_start = now_ms();
    for (int i = 0; i < N_RUNS; i++) {
        s = fe_runtime_run(&rt, input, output);
        if (s != FE_OK) {
            fprintf(stderr, "Inference failed on run %d: %d\n", i, s);
            return 1;
        }
    }
    double avg_ms = (now_ms() - t_start) / N_RUNS;

    /* Step 4: Results */
    printf("\n[4/4] Results\n");

    float *probs = (float *)output->data;
    const char *classes[] = {"No Leak", "Small Leak", "Large Leak"};

    printf("\n    Leak Classification:\n");
    for (int i = 0; i < 3; i++) {
        int bar_len = (int)(probs[i] * 30);
        printf("    %-12s [", classes[i]);
        for (int b = 0; b < 30; b++) printf(b < bar_len ? "#" : " ");
        printf("] %.1f%%\n", probs[i] * 100.0f);
    }

    /* Verify softmax sums to 1 */
    float sum = probs[0] + probs[1] + probs[2];
    printf("\n    Probabilities sum: %.4f %s\n", sum,
           fabsf(sum - 1.0f) < 1e-3f ? "(valid)" : "(ERROR)");

    printf("\n    Inference time:  %.1f ms/run (avg over %d runs)\n",
           avg_ms, N_RUNS);

    /* Profiler breakdown */
    fe_profiler_print(&prof);

    /* Memory summary */
    printf("\nMemory Summary:\n");
    printf("    Weights:     %.2f MB\n",
           (double)fe_arena_used(&weight_arena) / (1024*1024));
    printf("    Activations: %.2f MB (peak)\n",
           (double)fe_arena_peak(&rt.activation_arena) / (1024*1024));

    fe_tensor_free(input);
    fe_tensor_free(output);

    printf("\n===========================================\n");
    printf("  Ferrite demo complete.\n");
    printf("===========================================\n");
    return 0;
}