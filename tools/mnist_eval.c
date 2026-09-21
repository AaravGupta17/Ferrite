/*
 * tools/mnist_eval.c - measure Ferrite's own MNIST accuracy.
 *
 * Runs the full 10,000-image test set through fe_runtime_run and reports how
 * many Ferrite itself got right. The README's accuracy claim has to be this
 * number, not PyTorch's -- otherwise a reader has no way to check that the C
 * runtime reproduces the trained model.
 *
 * Inputs are the raw dumps tools/train_mnist.py writes beside the model:
 *   images: 10000 * 784 float32, row-major, already scaled to [0,1]
 *   labels: 10000 uint8
 *
 * usage: mnist_eval <model.onnx> <images.bin> <labels.bin>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../core/tensor.h"
#include "../core/allocator.h"
#include "../graph/graph.h"
#include "../runtime/engine.h"
#include "../importer/onnx.h"

#define WEIGHT_BUF_SIZE (4 * 1024 * 1024)
#define ACT_BUF_SIZE    (1 * 1024 * 1024)
#define PIXELS          784
#define CLASSES         10

static unsigned char weight_buf[WEIGHT_BUF_SIZE] __attribute__((aligned(64)));
static unsigned char act_buf   [ACT_BUF_SIZE]    __attribute__((aligned(64)));

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Read a whole file into a malloc'd buffer. Returns element count via *n. */
static void *slurp(const char *path, size_t *nbytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0) { fclose(f); return NULL; }
    void *buf = malloc((size_t)len);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    *nbytes = (size_t)len;
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <model.onnx> <images.bin> <labels.bin>\n"
                "  (both .bin files are written by tools/train_mnist.py)\n",
                argv[0]);
        return 2;
    }

    size_t img_bytes = 0, lbl_bytes = 0;
    float *images = (float *)slurp(argv[2], &img_bytes);
    unsigned char *labels = (unsigned char *)slurp(argv[3], &lbl_bytes);
    if (!images || !labels) {
        fprintf(stderr, "mnist_eval: cannot read the test dumps\n");
        return 1;
    }

    size_t n_images = img_bytes / (PIXELS * sizeof(float));
    if (n_images == 0 || lbl_bytes < n_images) {
        fprintf(stderr, "mnist_eval: test dumps disagree "
                "(%zu images, %zu labels)\n", n_images, lbl_bytes);
        return 1;
    }

    FeGraph g;
    FeArena weight_arena;
    fe_arena_init(&weight_arena, weight_buf, WEIGHT_BUF_SIZE);
    if (fe_onnx_load(&g, &weight_arena, argv[1]) != FE_OK) {
        fprintf(stderr, "mnist_eval: failed to load %s\n", argv[1]);
        return 1;
    }

    FeRuntime rt;
    if (fe_runtime_init(&rt, &g, weight_buf, WEIGHT_BUF_SIZE,
                        act_buf, ACT_BUF_SIZE) != FE_OK) {
        fprintf(stderr, "mnist_eval: runtime init failed\n");
        return 1;
    }

    int in_shape[]  = {1, PIXELS};
    int out_shape[] = {1, CLASSES};
    FeTensor *input  = fe_tensor_alloc(DTYPE_FLOAT32, 2, in_shape);
    FeTensor *output = fe_tensor_alloc(DTYPE_FLOAT32, 2, out_shape);
    if (!input || !output) { fprintf(stderr, "alloc failed\n"); return 1; }

    size_t correct = 0;
    double t0 = now_ms();

    for (size_t i = 0; i < n_images; i++) {
        memcpy(input->data, images + i * PIXELS, PIXELS * sizeof(float));
        if (fe_runtime_run(&rt, input, output) != FE_OK) {
            fprintf(stderr, "mnist_eval: inference failed at image %zu\n", i);
            return 1;
        }
        const float *logits = (const float *)output->data;
        int best = 0;
        for (int k = 1; k < CLASSES; k++)
            if (logits[k] > logits[best]) best = k;
        if (best == (int)labels[i]) correct++;
    }

    double elapsed = now_ms() - t0;

    printf("Ferrite MNIST accuracy: %.2f%% (%zu/%zu)\n",
           100.0 * (double)correct / (double)n_images, correct, n_images);
    printf("  %.3f ms/image over the full test set\n", elapsed / (double)n_images);
    printf("  %.0f images/s single-threaded\n", n_images / (elapsed / 1000.0));
    printf("  activations %.1f KB peak\n",
           (double)fe_arena_peak(&rt.activation_arena) / 1024.0);

    fe_tensor_free(input);
    fe_tensor_free(output);
    free(images);
    free(labels);
    return 0;
}
