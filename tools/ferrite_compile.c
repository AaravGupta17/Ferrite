/*
 * tools/ferrite_compile.c — ONNX model -> FEMD artifact (Section 2.2).
 *
 *   ferrite_compile <model.onnx> <model.femd> [--in-shape WxHxC] [--emit-header]
 *                             [--target esp32] [--max-nodes N] [--max-tensors N]
 *
 * Loads the ONNX model, runs the optimizer pass pipeline (fe_onnx_load does
 * this), fixes or seeds the input shape, computes the memory plan, binds
 * weight data, and serializes the result with fe_model_save. Devices then
 * load the artifact with fe_model_load and run it with fe_model_run — no
 * ONNX parser, no shape inference, no optimizer on the target.
 *
 * --in-shape is only needed when the model's input is dynamic (a zero extent
 * in the ONNX value_info); the artifact is fixed-shape by construction.
 *
 * --emit-header writes a C header with the artifact embedded as a byte array
 * (for flash-mapped deployment).
 *
 * --target esp32 (Section 4.4) enforces what the ESP32 build can run,
 * failing loudly instead of shipping an FEMD the device will reject later:
 * the model must be FLOAT64-free (device kernels are FP32/FP16/BF16/INT8/
 * INT16), and must fit the device ceilings. The ceilings default to the
 * idf/ferrite/config/config.h values and can be overridden with
 * --max-nodes/--max-tensors.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../core/tensor.h"
#include "../core/allocator.h"
#include "../graph/graph.h"
#include "../planner/memory_planner.h"
#include "../importer/onnx.h"
#include "../optim/shape_infer.h"
#include "../runtime/engine.h"
#include "../runtime/model_ser.h"

#define WEIGHT_BUF_SIZE (64u * 1024u * 1024u)
#define UNDEF_TIDX (-1)

static int parse_shape(const char *s, int *ndim_out,
                       int shape[FERRITE_MAX_DIMS]) {
    int ndim = 0;
    const char *p = s;
    while (*p) {
        if (ndim >= FERRITE_MAX_DIMS) return -1;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || v <= 0 || v > 0x7FFFFFFF) return -1;
        shape[ndim++] = (int)v;
        p = end;
        if (*p == 'x' || *p == 'X') p++;
        else if (*p) return -1;
    }
    *ndim_out = ndim;
    return ndim > 0 ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <model.onnx> <model.femd> "
                "[--in-shape WxHxC] [--emit-header] [--target esp32]\n",
                argv[0]);
        return 2;
    }
    const char *onnx_path = argv[1];
    const char *out_path  = argv[2];
    const char *in_shape  = NULL;
    int emit_header = 0;
    int target_esp32 = 0;
    int max_nodes   = 48;    /* ESP32 ceilings (Section 4.4) */
    int max_tensors = 96;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--in-shape") && i + 1 < argc)
            in_shape = argv[++i];
        else if (!strcmp(argv[i], "--emit-header"))
            emit_header = 1;
        else if (!strcmp(argv[i], "--target") && i + 1 < argc) {
            if (!strcmp(argv[++i], "esp32")) target_esp32 = 1;
            else {
                fprintf(stderr, "unknown --target: %s\n", argv[i]);
                return 2;
            }
        } else if (!strcmp(argv[i], "--max-nodes") && i + 1 < argc)
            max_nodes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-tensors") && i + 1 < argc)
            max_tensors = atoi(argv[++i]);
        else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        }
    }

    static unsigned char weight_buf[WEIGHT_BUF_SIZE]
        __attribute__((aligned(64)));

    FeGraph g;
    FeArena weight_arena;
    fe_arena_init(&weight_arena, weight_buf, WEIGHT_BUF_SIZE);

    if (fe_onnx_load(&g, &weight_arena, onnx_path) != FE_OK) {
        fprintf(stderr, "ferrite-compile: failed to load %s\n", onnx_path);
        return 1;
    }

    int in_tidx  = fe_graph_find_input(&g);
    int out_tidx = fe_graph_find_output(&g);
    if (in_tidx < 0) {
        fprintf(stderr, "ferrite-compile: no graph input found\n");
        return 1;
    }
    if (out_tidx < 0) {
        fprintf(stderr, "ferrite-compile: no graph output found\n");
        return 1;
    }

    if (target_esp32) {
        /* Device-target artifact gate (Section 4.4): reject what the ESP32
         * build cannot run at load time, instead of shipping an FEMD that
         * fails on the part. */
        if (g.n_nodes > max_nodes) {
            fprintf(stderr, "ferrite-compile: %d nodes exceed ESP32 ceiling "
                    "%d (raise --max-nodes or split the model)\n",
                    g.n_nodes, max_nodes);
            return 1;
        }
        if (g.n_tensors > max_tensors) {
            fprintf(stderr, "ferrite-compile: %d tensors exceed ESP32 ceiling "
                    "%d (--max-tensors)\n", g.n_tensors, max_tensors);
            return 1;
        }
        for (int i = 0; i < g.n_tensors; i++) {
            FeTensorEntry *e = &g.tensors[i];
            if (e->dtype == DTYPE_FLOAT64) {
                fprintf(stderr, "ferrite-compile: tensor '%s' is FLOAT64; "
                        "ESP32 kernels are FP32/FP16/BF16/INT8/INT16 — "
                        "quantize the model and try again\n", e->name);
                return 1;
            }
        }
        printf("esp32 target check passed (%d nodes, %d tensors, "
               "FLOAT64-free)\n", g.n_nodes, g.n_tensors);
    }
    printf("input  %d %s\n", in_tidx, g.tensors[in_tidx].name);
    printf("output %d %s\n", out_tidx, g.tensors[out_tidx].name);

    FeTensorEntry *ie = &g.tensors[in_tidx];
    int have_shape = ie->ndim > 0 && ie->shape[0] > 0;
    if (!have_shape && !in_shape) {
        fprintf(stderr,
                "ferrite-compile: model input is dynamic; pass "
                "--in-shape WxHxC to fix it\n");
        return 1;
    }
    if (in_shape) {
        int ndim;
        int shape[FERRITE_MAX_DIMS];
        if (parse_shape(in_shape, &ndim, shape) != 0) {
            fprintf(stderr, "ferrite-compile: bad --in-shape '%s'\n",
                    in_shape);
            return 1;
        }
        ie->ndim = ndim;
        for (int d = 0; d < ndim; d++) ie->shape[d] = shape[d];
        fe_infer_shapes(&g);   /* propagate the fixed input shape */
    }

    FePlan plan;
    if (fe_plan_memory(&g, &plan) != FE_OK) {
        fprintf(stderr, "ferrite-compile: memory planning failed\n");
        return 1;
    }

    /* Bind weight bytes to the weight tensors (placeholder check: the
     * importer already backs weights; skip any that already have data). */
    for (int i = 0; i < g.n_tensors; i++) {
        FeTensorEntry *e = &g.tensors[i];
        if (e->is_weight && !e->tensor) {
            e->tensor = fe_arena_alloc_tensor(&weight_arena, e->dtype,
                                              e->ndim, e->shape);
            if (!e->tensor) {
                fprintf(stderr, "ferrite-compile: weight allocate failed\n");
                return 1;
            }
        }
    }

    FILE *f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "ferrite-compile: cannot open %s for writing\n",
                out_path);
        return 1;
    }
    FeStatus s = fe_model_save(f, &g, &plan, in_tidx, out_tidx);
    fclose(f);
    if (s != FE_OK) {
        fprintf(stderr, "ferrite-compile: save failed (%d)\n", s);
        return 1;
    }
    printf("wrote %s (input shape ", out_path);
    for (int d = 0; d < ie->ndim; d++)
        printf("%d%c", ie->shape[d], d + 1 < ie->ndim ? 'x' : '\n');

    if (emit_header) {
        char header_path[512];
        snprintf(header_path, sizeof(header_path), "%s.h", out_path);
        FILE *hf = fopen(out_path, "rb");
        FILE *oh = fopen(header_path, "w");
        if (!hf || !oh) {
            fprintf(stderr, "ferrite-compile: header output failed\n");
            return 1;
        }
        long len;
        fseek(hf, 0, SEEK_END);
        len = ftell(hf);
        rewind(hf);
        if (len <= 0) return 1;
        fprintf(oh, "/* generated by ferrite-compile --emit-header */\n");
        fprintf(oh, "static const unsigned char model_data[] = {\n");
        unsigned char buf[16];
        long total = 0;
        size_t r;
        while ((r = fread(buf, 1, sizeof(buf), hf)) > 0) {
            fprintf(oh, "  ");
            for (size_t i = 0; i < r; i++)
                fprintf(oh, "0x%02x,", buf[i]);
            fprintf(oh, " /* %ld */\n", total);
            total += (long)r;
        }
        fprintf(oh, "};\n/* %ld bytes */\n", total);
        fclose(hf);
        fclose(oh);
        printf("wrote header %s (%ld bytes)\n", header_path, total);
    }

    return 0;
}