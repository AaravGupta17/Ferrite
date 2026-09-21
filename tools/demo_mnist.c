/*
 * Ferrite Demo: MNIST handwritten-digit recognition.
 *
 * Loads a 784-128-10 MLP trained in PyTorch (tools/train_mnist.py), runs the
 * ten embedded test digits through the engine, and prints each one as ASCII
 * art next to Ferrite's prediction, plus latency and memory.
 *
 * The digits are compiled in (tools/mnist_samples.h), so the demo needs no
 * data files -- only the .onnx model.
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
#include "mnist_samples.h"

#define WEIGHT_BUF_SIZE (4 * 1024 * 1024)
#define ACT_BUF_SIZE    (1 * 1024 * 1024)

static unsigned char weight_buf[WEIGHT_BUF_SIZE] __attribute__((aligned(64)));
static unsigned char act_buf   [ACT_BUF_SIZE]    __attribute__((aligned(64)));

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* The model emits logits; softmax here is presentation only. */
static void softmax(const float *in, float *out, int n) {
    float max = in[0];
    for (int i = 1; i < n; i++) if (in[i] > max) max = in[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { out[i] = expf(in[i] - max); sum += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

static void print_digit(const float *px) {
    static const char *ramp = " .:-=+*#%@";
    for (int r = 0; r < 28; r++) {
        printf("    ");
        for (int c = 0; c < 28; c++) {
            float v = px[r * 28 + c];
            int level = (int)(v * 9.0f + 0.5f);
            if (level < 0) level = 0;
            if (level > 9) level = 9;
            putchar(ramp[level]);
            putchar(ramp[level]);          /* two chars wide: squarer pixels */
        }
        putchar('\n');
    }
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "tests/mnist.onnx";

    printf("=============================================\n");
    printf("  Ferrite - MNIST digit recognition\n");
    printf("=============================================\n\n");

    FeGraph g;
    FeArena weight_arena;
    fe_arena_init(&weight_arena, weight_buf, WEIGHT_BUF_SIZE);

    double t0 = now_ms();
    FeStatus s = fe_onnx_load(&g, &weight_arena, model_path);
    if (s != FE_OK) {
        fprintf(stderr,
                "Failed to load %s (status %d).\n"
                "Train it first:  python tools/train_mnist.py\n",
                model_path, s);
        return 1;
    }
    double load_ms = now_ms() - t0;
    size_t weight_bytes = fe_arena_used(&weight_arena);

    FeRuntime rt;
    s = fe_runtime_init(&rt, &g, weight_buf, WEIGHT_BUF_SIZE,
                        act_buf, ACT_BUF_SIZE);
    if (s != FE_OK) { fprintf(stderr, "Runtime init failed: %d\n", s); return 1; }

    int in_shape[]  = {1, 784};
    int out_shape[] = {1, 10};
    FeTensor *input  = fe_tensor_alloc(DTYPE_FLOAT32, 2, in_shape);
    FeTensor *output = fe_tensor_alloc(DTYPE_FLOAT32, 2, out_shape);
    if (!input || !output) { fprintf(stderr, "alloc failed\n"); return 1; }

    int correct = 0;
    double total_ms = 0.0;

    for (int i = 0; i < MNIST_N_SAMPLES; i++) {
        memcpy(input->data, mnist_images[i], MNIST_PIXELS * sizeof(float));

        double t1 = now_ms();
        s = fe_runtime_run(&rt, input, output);
        double dt = now_ms() - t1;
        if (s != FE_OK) { fprintf(stderr, "Inference failed: %d\n", s); return 1; }
        total_ms += dt;

        float probs[10];
        softmax((const float *)output->data, probs, 10);

        int best = 0;
        for (int k = 1; k < 10; k++) if (probs[k] > probs[best]) best = k;
        if (best == mnist_labels[i]) correct++;

        print_digit(mnist_images[i]);
        printf("\n    Ferrite says: %d   (%.1f%% confident)   truth: %d  %s\n",
               best, probs[best] * 100.0f, mnist_labels[i],
               best == mnist_labels[i] ? "OK" : "MISS");
        printf("    %.3f ms\n\n", dt);
    }

    printf("---------------------------------------------\n");
    printf("  %d/%d correct on the embedded samples\n", correct, MNIST_N_SAMPLES);
    printf("  %.3f ms/inference (mean of %d)\n",
           total_ms / MNIST_N_SAMPLES, MNIST_N_SAMPLES);
    printf("\n  model load    %.1f ms\n", load_ms);
    printf("  weights       %.0f KB\n", (double)weight_bytes / 1024.0);
    printf("  activations   %.1f KB peak\n",
           (double)fe_arena_peak(&rt.activation_arena) / 1024.0);
    printf("  dependencies  none\n");
    printf("=============================================\n");

    fe_tensor_free(input);
    fe_tensor_free(output);
    return correct == MNIST_N_SAMPLES ? 0 : 1;
}
