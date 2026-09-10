/*
 * tests/test_artifact.c — FEMD artifact round-trip (Section 2.2).
 *
 * Loads tests/tiny_mlp.onnx, computes the memory plan, saves a FEMD artifact,
 * reloads it into a fresh arena pair, and runs it with fe_model_run — the
 * device-only path. Verifies:
 *   - artifact plan matches the direct fe_plan_memory result on the live graph
 *   - fe_model_run output matches fe_runtime_run (same kernels, same plan)
 *   - the loader rejects: bad magic, bad version, truncated payload, and
 *     trailing garbage (strict EOF check)
 *   - fe_model_run rejects an input whose shape differs from the artifact's
 *     planned input shape
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "../core/tensor.h"
#include "../core/allocator.h"
#include "../graph/graph.h"
#include "../planner/memory_planner.h"
#include "../runtime/engine.h"
#include "../runtime/model_ser.h"
#include "../importer/onnx.h"

#define BUF_SIZE (1024 * 1024)

static unsigned char wbuf[BUF_SIZE] __attribute__((aligned(64)));
static unsigned char abuf[BUF_SIZE] __attribute__((aligned(64)));
static unsigned char wbuf2[BUF_SIZE] __attribute__((aligned(64)));
static unsigned char abuf2[BUF_SIZE] __attribute__((aligned(64)));

static int tensors_close(const FeTensor *a, const FeTensor *b) {
    if (!a || !b) return 0;
    if (a->dtype != b->dtype || a->ndim != b->ndim) return 0;
    int n = 1;
    for (int d = 0; d < a->ndim; d++) {
        if (a->shape[d] != b->shape[d]) return 0;
        n *= a->shape[d];
    }
    const float *fa = (const float *)a->data;
    const float *fb = (const float *)b->data;
    for (int i = 0; i < n; i++)
        if (fabsf(fa[i] - fb[i]) > 1e-4f * (fabsf(fa[i]) + 1.0f))
            return 0;
    return 1;
}

static size_t read_all(const char *path, unsigned char **out) {
    FILE *f = fopen(path, "rb");
    assert(f);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    unsigned char *buf = (unsigned char *)malloc((size_t)n);
    assert(buf);
    assert(fread(buf, 1, (size_t)n, f) == (size_t)n);
    fclose(f);
    *out = buf;
    return (size_t)n;
}

static void write_bytes(const char *path, const unsigned char *buf, size_t n) {
    FILE *f = fopen(path, "wb");
    assert(f);
    assert(fwrite(buf, 1, n, f) == n);
    fclose(f);
}

static void test_round_trip(void) {
    /* Load + plan the source model. */
    FeGraph g;
    FeArena src_wa;
    fe_arena_init(&src_wa, wbuf, BUF_SIZE);
    FeStatus s = fe_onnx_load(&g, &src_wa, "tests/tiny_mlp.onnx");
    assert(s == FE_OK);

    int in_tidx  = fe_graph_find_input(&g);
    int out_tidx = fe_graph_find_output(&g);
    assert(in_tidx >= 0 && out_tidx >= 0);

    FePlan plan;
    assert(fe_plan_memory(&g, &plan) == FE_OK);
    assert(plan.total_activation_bytes > 0);

    /* Reference output through the full engine. */
    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g, wbuf, BUF_SIZE, abuf, BUF_SIZE) == FE_OK);

    const FeTensorEntry *ie = &g.tensors[in_tidx];
    const FeTensorEntry *oe = &g.tensors[out_tidx];
    assert(ie->ndim > 0 && ie->shape[0] > 0);   /* tiny_mlp is static */
    assert(oe->ndim > 0 && oe->shape[0] > 0);

    FeTensor *input = fe_tensor_alloc(ie->dtype, ie->ndim, ie->shape);
    FeTensor *ref   = fe_tensor_alloc(oe->dtype, oe->ndim, oe->shape);
    FeTensor *out   = fe_tensor_alloc(oe->dtype, oe->ndim, oe->shape);
    assert(input && ref && out);

    float *id = (float *)input->data;
    int in_n = 1;
    for (int d = 0; d < ie->ndim; d++) in_n *= ie->shape[d];
    for (int i = 0; i < in_n; i++)
        id[i] = sinf(0.123f * (float)i) * 2.0f + 0.5f;

    assert(fe_runtime_run(&rt, input, ref) == FE_OK);

    /* Save the artifact. */
    FILE *f = fopen("tests/test_artifact.femd", "wb");
    assert(f);
    assert(fe_model_save(f, &g, &plan, in_tidx, out_tidx) == FE_OK);
    fclose(f);

    /* Reload into a fresh arena pair and run through the device path. */
    FeArena wa2, aa2;
    fe_arena_init(&wa2, wbuf2, BUF_SIZE);
    fe_arena_init(&aa2, abuf2, BUF_SIZE);
    FeModel m;
    memset(&m, 0, sizeof(m));
    f = fopen("tests/test_artifact.femd", "rb");
    assert(f);
    assert(fe_model_load(f, &wa2, &m) == FE_OK);
    fclose(f);

    assert(m.n_steps == g.n_nodes);
    assert(m.in_tidx == in_tidx && m.out_tidx == out_tidx);
    assert(m.plan.total_activation_bytes == plan.total_activation_bytes);
    for (int i = 0; i < g.n_tensors; i++)
        assert(m.plan.offsets[i] == plan.offsets[i]);

    assert(fe_model_run(&m, &wa2, &aa2, input, out) == FE_OK);
    assert(tensors_close(ref, out));

    /* Wrong-shape input on the fixed-shape artifact must fail loudly. */
    int bad_shape[2] = { 1, ie->shape[ie->ndim - 1] + 1 };
    FeTensor *bad = fe_tensor_alloc(ie->dtype, 2, bad_shape);
    FeArena aa3;
    fe_arena_init(&aa3, abuf2, BUF_SIZE);
    assert(fe_model_run(&m, &wa2, &aa3, bad, out) == FE_ERR_SHAPE);

    fe_tensor_free(input);
    fe_tensor_free(ref);
    fe_tensor_free(out);
    fe_tensor_free(bad);
    printf("PASS test_round_trip (%d nodes, %d tensors, %zu B activations)\n",
           g.n_nodes, g.n_tensors, plan.total_activation_bytes);
}

static void test_corruption(void) {
    unsigned char *bytes;
    size_t n = read_all("tests/test_artifact.femd", &bytes);

    /* (1) Bad magic: first four bytes wrong. */
    unsigned char *bad_magic = (unsigned char *)malloc(n);
    memcpy(bad_magic, bytes, n);
    bad_magic[0] = 'X';
    write_bytes("tests/test_artifact_bad_magic.femd", bad_magic, n);

    /* (2) Bad version. */
    unsigned char *bad_ver = (unsigned char *)malloc(n);
    memcpy(bad_ver, bytes, n);
    bad_ver[4] = 0x09;   /* version 9 != 1 */
    write_bytes("tests/test_artifact_bad_ver.femd", bad_ver, n);

    /* (3) Truncated: drop the final plan bytes -> the loader hits EOF. */
    unsigned char *trunc = (unsigned char *)malloc(n - 1);
    memcpy(trunc, bytes, n - 1);
    write_bytes("tests/test_artifact_trunc.femd", trunc, n - 1);

    /* (4) Trailing garbage: strict EOF check must reject it. */
    unsigned char *trail = (unsigned char *)malloc(n + 4);
    memcpy(trail, bytes, n);
    trail[n] = trail[n + 1] = trail[n + 2] = trail[n + 3] = 0xAA;
    write_bytes("tests/test_artifact_trail.femd", trail, n + 4);

    FeArena wa;
    fe_arena_init(&wa, wbuf2, BUF_SIZE);
    FeModel m;
    FILE *f;

    f = fopen("tests/test_artifact_bad_magic.femd", "rb");
    assert(fe_model_load(f, &wa, &m) != FE_OK); fclose(f);
    f = fopen("tests/test_artifact_bad_ver.femd", "rb");
    assert(fe_model_load(f, &wa, &m) != FE_OK); fclose(f);
    f = fopen("tests/test_artifact_trunc.femd", "rb");
    assert(fe_model_load(f, &wa, &m) != FE_OK); fclose(f);
    f = fopen("tests/test_artifact_trail.femd", "rb");
    assert(fe_model_load(f, &wa, &m) != FE_OK); fclose(f);

    free(bytes);
    free(bad_magic);
    free(bad_ver);
    free(trunc);
    free(trail);
    printf("PASS test_corruption (magic, version, truncation, trailing)\n");
}

int main(void) {
    test_round_trip();
    test_corruption();
    printf("ALL ARTIFACT TESTS PASSED\n");
    return 0;
}