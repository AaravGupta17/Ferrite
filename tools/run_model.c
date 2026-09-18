/*
 * tools/run_model.c — host-side golden runner (Phase D).
 *
 * Loads an ONNX model through the importer, feeds one raw float32 input
 * from a file, runs the engine once, and writes the output tensor to a
 * file. Used by tools/golden_compare.py to diff Ferrite against ONNX
 * Runtime on identical inputs.
 *
 * Binary tensor file format (magic "FRT1", little-endian):
 *   char[4]  magic = {'F','R','T','1'}
 *   int32    version = 1
 *   int32    ndim
 *   int32    dims[ndim]
 *   float32  data[prod(dims)]
 *
 * Usage:
 *   run_model <model.onnx> <input.bin> <output.bin> <out_dims.cfg>
 *
 * where out_dims.cfg is one line of space-separated extent integers (the
 * graph output shape; needed before run because fe_runtime_run writes into
 * a caller-allocated output tensor).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../core/tensor.h"
#include "../core/allocator.h"
#include "../graph/graph.h"
#include "../planner/memory_planner.h"
#include "../runtime/engine.h"
#include "../importer/onnx.h"

#define WEIGHT_BUF_SIZE (1u << 20)
#define ACT_BUF_SIZE    (1u << 20)

static unsigned char weight_buf[WEIGHT_BUF_SIZE] __attribute__((aligned(64)));
static unsigned char act_buf   [ACT_BUF_SIZE]    __attribute__((aligned(64)));

static int write_bin(const char *path, const FeTensor *t) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int32_t version = 1, ndim = (int32_t)t->ndim;
    fwrite("FRT1", 1, 4, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&ndim, sizeof(ndim), 1, f);
    for (int i = 0; i < t->ndim; i++) {
        int32_t d = (int32_t)t->shape[i];
        fwrite(&d, sizeof(d), 1, f);
    }
    int n = fe_tensor_numel(t);
    fwrite(t->data, sizeof(float), (size_t)n, f);
    fclose(f);
    return 0;
}

/* Reads a "FRT1" .bin file into a freshly allocated FeTensor (caller frees). */
static FeTensor *read_bin(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[4];
    int32_t version = 0, ndim = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "FRT1", 4) != 0 ||
        fread(&version, sizeof(version), 1, f) != 1 || version != 1 ||
        fread(&ndim, sizeof(ndim), 1, f) != 1 ||
        ndim < 0 || ndim > FERRITE_MAX_DIMS) {
        fclose(f);
        return NULL;
    }
    int dims[FERRITE_MAX_DIMS];
    for (int i = 0; i < ndim; i++) {
        int32_t d;
        if (fread(&d, sizeof(d), 1, f) != 1 || d < 0) { fclose(f); return NULL; }
        dims[i] = (int)d;
    }
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, ndim, dims);
    if (!t) { fclose(f); return NULL; }
    size_t nbytes = (size_t)fe_tensor_numel(t) * sizeof(float);
    if (fread(t->data, 1, nbytes, f) != nbytes) {
        fe_tensor_free(t);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return t;
}

static int read_out_dims(const char *path, int *dims, int *ndim) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int n = 0, d;
    while (n < FERRITE_MAX_DIMS && fscanf(f, "%d", &d) == 1) dims[n++] = d;
    fclose(f);
    *ndim = n;
    return (n > 0) ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: run_model <model.onnx> <input.bin> "
                        "<output.bin> <out_dims.cfg>\n");
        return 2;
    }

    FeTensor *input = read_bin(argv[2]);
    if (!input) {
        fprintf(stderr, "run_model: cannot read input %s\n", argv[2]);
        return 1;
    }

    int out_dims[FERRITE_MAX_DIMS], out_ndim;
    if (read_out_dims(argv[4], out_dims, &out_ndim) != 0) {
        fprintf(stderr, "run_model: cannot read output dims %s\n", argv[4]);
        fe_tensor_free(input);
        return 1;
    }

    FeGraph g;
    FeArena weight_arena;
    fe_arena_init(&weight_arena, weight_buf, WEIGHT_BUF_SIZE);

    if (fe_onnx_load(&g, &weight_arena, argv[1]) != FE_OK) {
        fprintf(stderr, "run_model: failed to load %s\n", argv[1]);
        fe_tensor_free(input);
        return 1;
    }

    FeRuntime rt;
    if (fe_runtime_init(&rt, &g,
                        weight_buf, WEIGHT_BUF_SIZE,
                        act_buf,    ACT_BUF_SIZE) != FE_OK) {
        fprintf(stderr, "run_model: runtime init failed\n");
        fe_tensor_free(input);
        return 1;
    }

    FeTensor *output = fe_tensor_alloc(DTYPE_FLOAT32, out_ndim, out_dims);
    if (!output) {
        fe_tensor_free(input);
        return 1;
    }

    FeStatus s = fe_runtime_run(&rt, input, output);
    if (s != FE_OK) {
        fprintf(stderr, "run_model: inference failed with status %d\n", s);
        fe_tensor_free(input);
        fe_tensor_free(output);
        return 1;
    }

    int rc = write_bin(argv[3], output);

    fe_tensor_free(input);
    fe_tensor_free(output);
    return rc == 0 ? 0 : 1;
}