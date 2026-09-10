// tests/test_quant.c
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "../core/tensor.h"
#include "../graph/graph.h"
#include "../runtime/engine.h"
#include "../quantization/quant.h"

#define EPSILON_QUANT 0.05f   /* quantization introduces ~1% error */

static void test_quantize_dequantize(void) {
    int shape[] = {4};
    FeTensor *f32 = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);
    FeTensor *i8  = fe_tensor_alloc(DTYPE_INT8,    1, shape);
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);

    float data[] = {-1.0f, -0.5f, 0.5f, 1.0f};
    memcpy(f32->data, data, sizeof(data));

    FeQuantParams p;
    assert(fe_quantize(f32, i8, &p) == FE_OK);

    /* Scale should be 1.0/127 ≈ 0.00787 */
    assert(fabsf(p.scale - 1.0f/127.0f) < 1e-4f);

    /* INT8 values should be [-127, -64, 64, 127] approximately */
    int8_t *q = (int8_t *)i8->data;
    assert(q[0] == -127);
    assert(q[3] ==  127);

    assert(fe_dequantize(i8, &p, out) == FE_OK);

    float *o = (float *)out->data;
    for (int i = 0; i < 4; i++)
        assert(fabsf(o[i] - data[i]) < EPSILON_QUANT);

    fe_tensor_free(f32);
    fe_tensor_free(i8);
    fe_tensor_free(out);
    printf("PASS test_quantize_dequantize\n");
}

static void test_matmul_int8_accuracy(void) {
    /*
     * Compare int8 matmul vs float32 matmul.
     * With uniform weights the error should be small.
     */
    int shapeA[] = {4, 8};
    int shapeB[] = {8, 4};
    int shapeC[] = {4, 4};

    FeTensor *A    = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeA);
    FeTensor *B    = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeB);
    FeTensor *C_f  = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeC);
    FeTensor *C_q  = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeC);

    float *a = (float *)A->data;
    float *b = (float *)B->data;
    for (int i = 0; i < 4*8; i++) a[i] = (float)(i % 7 - 3) * 0.1f;
    for (int i = 0; i < 8*4; i++) b[i] = (float)(i % 5 - 2) * 0.1f;

    /* Float32 reference */
    memset(C_f->data, 0, 4*4*sizeof(float));
    float *cf = (float *)C_f->data;
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 8; k++)
            for (int j = 0; j < 4; j++)
                cf[i*4+j] += a[i*8+k] * b[k*4+j];

    /* INT8 matmul */
    assert(fe_matmul_int8(A, B, C_q) == FE_OK);

    /* Compare — allow up to 5% relative error */
    float *cq = (float *)C_q->data;
    float max_err = 0.0f;
    for (int i = 0; i < 4*4; i++) {
        float err = fabsf(cf[i] - cq[i]);
        if (err > max_err) max_err = err;
    }
    printf("INT8 matmul max error vs float32: %.4f\n", max_err);
    assert(max_err < 0.1f);

    fe_tensor_free(A);
    fe_tensor_free(B);
    fe_tensor_free(C_f);
    fe_tensor_free(C_q);
    printf("PASS test_matmul_int8_accuracy\n");
}

static void test_memory_reduction(void) {
    /*
     * Quantizing weights saves 4x memory.
     * Verify size relationship between float32 and int8 tensors.
     */
    int shape[] = {64, 64};
    FeTensor *f32 = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *i8  = fe_tensor_alloc(DTYPE_INT8,    2, shape);

    assert(f32->nbytes == 64 * 64 * 4);
    assert(i8->nbytes  == 64 * 64 * 1);
    assert(f32->nbytes == 4 * i8->nbytes);

    float savings = 100.0f * (1.0f - (float)i8->nbytes / f32->nbytes);
    printf("Memory reduction: %.0f%%\n", savings);
    assert(savings == 75.0f);

    fe_tensor_free(f32);
    fe_tensor_free(i8);
    printf("PASS test_memory_reduction\n");
}

static void test_per_channel_beats_per_tensor_on_skewed_weights(void) {
    /*
     * Column 0 has small magnitude (~0.01-0.02), column 1 has large
     * magnitude (~7.5-10.0). A single global scale is dominated by column 1,
     * so column 0 collapses toward zero under per-tensor quantization.
     * Per-channel quantization gives column 0 its own scale and preserves it.
     */
    int shape[] = {4, 2}; /* K=4, N=2 */
    FeTensor *w    = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *q_pt = fe_tensor_alloc(DTYPE_INT8,    2, shape);
    FeTensor *q_pc = fe_tensor_alloc(DTYPE_INT8,    2, shape);
    FeTensor *r_pt = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *r_pc = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);

    float data[] = {
        0.01f,  10.0f,
       -0.02f,  -8.0f,
        0.015f,  9.0f,
       -0.008f, -7.5f,
    };
    memcpy(w->data, data, sizeof(data));

    FeQuantParams pt;
    assert(fe_quantize(w, q_pt, &pt)          == FE_OK);
    assert(fe_dequantize(q_pt, &pt, r_pt)     == FE_OK);

    float scales[2];
    assert(fe_quantize_per_channel(w, q_pc, scales)      == FE_OK);
    assert(fe_dequantize_per_channel(q_pc, scales, r_pc) == FE_OK);

    float *rpt = (float *)r_pt->data;
    float *rpc = (float *)r_pc->data;

    float err_pt_col0 = 0.0f, err_pc_col0 = 0.0f;
    for (int k = 0; k < 4; k++) {
        err_pt_col0 += fabsf(rpt[k*2+0] - data[k*2+0]);
        err_pc_col0 += fabsf(rpc[k*2+0] - data[k*2+0]);
    }
    printf("Column 0 abs error — per-tensor: %.5f  per-channel: %.5f\n",
           err_pt_col0, err_pc_col0);
    assert(err_pc_col0 < err_pt_col0 * 0.5f); /* meaningfully, not marginally, better */

    fe_tensor_free(w); fe_tensor_free(q_pt); fe_tensor_free(q_pc);
    fe_tensor_free(r_pt); fe_tensor_free(r_pc);
    printf("PASS test_per_channel_beats_per_tensor_on_skewed_weights\n");
}

static void test_matmul_int8_per_channel_accuracy(void) {
    int shapeA[] = {4, 8};
    int shapeB[] = {8, 4};
    int shapeC[] = {4, 4};

    FeTensor *A   = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeA);
    FeTensor *B   = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeB);
    FeTensor *C_f = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeC);
    FeTensor *C_q = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeC);

    float *a = (float *)A->data;
    float *b = (float *)B->data;
    for (int i = 0; i < 4*8; i++) a[i] = (float)(i % 7 - 3) * 0.1f;
    /* Skew B's channels: column j scaled by (j+1) so channels have very
     * different magnitude ranges — this is the case per-channel exists for. */
    for (int k = 0; k < 8; k++)
        for (int j = 0; j < 4; j++)
            b[k*4+j] = (float)(k % 5 - 2) * 0.1f * (float)(j + 1) * 5.0f;

    memset(C_f->data, 0, 4*4*sizeof(float));
    float *cf = (float *)C_f->data;
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 8; k++)
            for (int j = 0; j < 4; j++)
                cf[i*4+j] += a[i*8+k] * b[k*4+j];

    assert(fe_matmul_int8_per_channel(A, B, C_q) == FE_OK);

    float *cq = (float *)C_q->data;
    float max_err = 0.0f;
    for (int i = 0; i < 4*4; i++) {
        float err = fabsf(cf[i] - cq[i]);
        if (err > max_err) max_err = err;
    }
    printf("Per-channel INT8 matmul max error vs float32: %.4f\n", max_err);
    assert(max_err < 0.1f);

    fe_tensor_free(A); fe_tensor_free(B);
    fe_tensor_free(C_f); fe_tensor_free(C_q);
    printf("PASS test_matmul_int8_per_channel_accuracy\n");
}

static void test_calibrate_activations(void) {
    /* Graph: INPUT(4) -> RELU(4); OUTPUT consumes the relu result. */
    FeGraph g;
    fe_graph_init(&g);
    int s_in[] = {4};
    int s_out[] = {4};
    int t_in  = fe_graph_add_tensor(&g, "input", DTYPE_FLOAT32, 1, s_in,  0);
    int t_r   = fe_graph_add_tensor(&g, "relu",  DTYPE_FLOAT32, 1, s_out, 0);
    int lin[] = {t_in}, rout[] = {t_r};
    fe_graph_add_node(&g, "input",  FE_OP_INPUT,  NULL, 0, lin,  1);
    fe_graph_add_node(&g, "relu",   FE_OP_RELU,   lin,  1, rout, 1);
    fe_graph_add_node(&g, "output", FE_OP_OUTPUT, rout, 1, NULL, 0);

    static unsigned char wbuf[4096], abuf[16384];
    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g, wbuf, sizeof(wbuf),
                           abuf, sizeof(abuf)) == FE_OK);
    assert(fe_runtime_alloc_weights(&rt) == FE_OK);

    FeTensor *in1 = fe_tensor_alloc(DTYPE_FLOAT32, 1, s_in);
    FeTensor *in2 = fe_tensor_alloc(DTYPE_FLOAT32, 1, s_in);
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 1, s_out);
    float d1[] = { 1.0f,  2.0f,  3.0f,  4.0f };
    float d2[] = {-10.0f, 20.0f, 30.0f, 40.0f };
    memcpy(in1->data, d1, sizeof d1);
    memcpy(in2->data, d2, sizeof d2);

    FeTensor *inputs[2] = {in1, in2};
    float ranges[16];
    assert(fe_runtime_calibrate(&rt, inputs, 2, out, ranges) == FE_OK);

    /* Per-tensor observed max |activation| across the 2-sample set. */
    assert(fabsf(ranges[t_in] - 40.0f) < 1e-5f);  /* |in2| peaks at 40 */
    assert(ranges[t_r] >= 4.0f && fabsf(ranges[t_r] - 40.0f) < 1e-5f);

    fe_tensor_free(in1); fe_tensor_free(in2); fe_tensor_free(out);
    printf("PASS test_calibrate_activations\n");
}

static void test_calibrate_pct_smoothing(void) {
    /* Graph: INPUT(4) -> RELU(4). */
    FeGraph g;
    fe_graph_init(&g);
    int s_in[]  = {4};
    int s_r[]   = {4};
    int t_in = fe_graph_add_tensor(&g, "input", DTYPE_FLOAT32, 1, s_in,  0);
    int t_r  = fe_graph_add_tensor(&g, "relu",  DTYPE_FLOAT32, 1, s_r,   0);
    int lin[] = {t_in}, rout[] = {t_r};
    fe_graph_add_node(&g, "input",  FE_OP_INPUT,  NULL, 0, lin,  1);
    fe_graph_add_node(&g, "relu",   FE_OP_RELU,   lin,  1, rout, 1);
    fe_graph_add_node(&g, "output", FE_OP_OUTPUT, rout, 1, NULL, 0);

    static unsigned char wbuf[4096], abuf[16384];
    FeRuntime rt;
    assert(fe_runtime_init(&rt, &g, wbuf, sizeof wbuf,
                           abuf, sizeof abuf) == FE_OK);
    assert(fe_runtime_alloc_weights(&rt) == FE_OK);

    /* 5 samples; set d3's peak far beyond the rest (a transient outlier). */
    FeTensor *ins[5];
    for (int i = 0; i < 5; i++)
        ins[i] = fe_tensor_alloc(DTYPE_FLOAT32, 1, s_in);
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 1, s_r);
    float peaks[] = {1.0f, 2.0f, 3.0f, 4.0f, 100.0f};
    for (int i = 0; i < 5; i++)
        for (int k = 0; k < 4; k++)
            ((float *)ins[i]->data)[k] = peaks[i];

    float ranges[16];
    /* 50th percentile of sample-maxima: median, immune to the outlier.
     * Sorted peaks -> {1,2,3,4,100}; index round(0.5*4)=2 -> 3.0. */
    assert(fe_runtime_calibrate_pct(&rt, ins, 5, out, 0.5f, ranges) == FE_OK);
    assert(fabsf(ranges[t_in] - 3.0f) < 1e-5f);
    assert(fabsf(ranges[t_r]  - 3.0f) < 1e-5f);

    /* 100th percentile = strict max (same as fe_runtime_calibrate). */
    assert(fe_runtime_calibrate_pct(&rt, ins, 5, out, 1.0f, ranges) == FE_OK);
    assert(fabsf(ranges[t_in] - 100.0f) < 1e-5f);

    for (int i = 0; i < 5; i++) fe_tensor_free(ins[i]);
    fe_tensor_free(out);
    printf("PASS test_calibrate_pct_smoothing\n");
}

/* Shared fixture: a small MLP (Input[1,4] -> Linear4x8 -> ReLU ->
 * Linear8x3 -> Softmax) so each engine path runs the full pipeline.
 * Build is split so a test can rebuild between quantization modes. */
enum { MLP_TENSOR_N = 9, MLP_NODE_N = 6 };
static void build_mlp(FeGraph *g, int tids[MLP_TENSOR_N]) {
    fe_graph_init(g);
    int s_in[]  = {1, 4};
    int s_w0[]  = {4, 8};
    int s_b0[]  = {8};
    int s_h0[]  = {1, 8};
    int s_w1[]  = {8, 3};
    int s_b1[]  = {3};
    int s_out[] = {1, 3};

    tids[0] = fe_graph_add_tensor(g, "input",  DTYPE_FLOAT32, 2, s_in,  0);
    tids[1] = fe_graph_add_tensor(g, "w0",     DTYPE_FLOAT32, 2, s_w0,  1);
    tids[2] = fe_graph_add_tensor(g, "b0",     DTYPE_FLOAT32, 1, s_b0,  1);
    tids[3] = fe_graph_add_tensor(g, "h0",     DTYPE_FLOAT32, 2, s_h0,  0);
    tids[4] = fe_graph_add_tensor(g, "relu0",  DTYPE_FLOAT32, 2, s_h0,  0);
    tids[5] = fe_graph_add_tensor(g, "w1",     DTYPE_FLOAT32, 2, s_w1,  1);
    tids[6] = fe_graph_add_tensor(g, "b1",     DTYPE_FLOAT32, 1, s_b1,  1);
    tids[7] = fe_graph_add_tensor(g, "h1",     DTYPE_FLOAT32, 2, s_out, 0);
    tids[8] = fe_graph_add_tensor(g, "output", DTYPE_FLOAT32, 2, s_out, 0);

    int l0in[] = {tids[0], tids[1], tids[2]}; int l0out[] = {tids[3]};
    int r0in[] = {tids[3]};                    int r0out[] = {tids[4]};
    int l1in[] = {tids[4], tids[5], tids[6]}; int l1out[] = {tids[7]};
    int smin[] = {tids[7]};                    int smout[] = {tids[8]};

    fe_graph_add_node(g, "input",   FE_OP_INPUT,   NULL,    0, &tids[0], 1);
    fe_graph_add_node(g, "linear0", FE_OP_LINEAR,  l0in,    3, l0out,    1);
    fe_graph_add_node(g, "relu0",   FE_OP_RELU,    r0in,    1, r0out,    1);
    fe_graph_add_node(g, "linear1", FE_OP_LINEAR,  l1in,    3, l1out,    1);
    fe_graph_add_node(g, "softmax", FE_OP_SOFTMAX, smin,    1, smout,    1);
    fe_graph_add_node(g, "output",  FE_OP_OUTPUT,  &tids[8], 1, NULL,     0);
}

static void fill_mlp_weights(FeGraph *g, int tids[MLP_TENSOR_N]) {
    FeTensor *w0 = g->tensors[tids[1]].tensor;
    FeTensor *w1 = g->tensors[tids[5]].tensor;
    float *w0d = (float *)w0->data, *w1d = (float *)w1->data;
    for (int k = 0; k < 4; k++)
        for (int j = 0; j < 8; j++)
            w0d[k * 8 + j] = (float)(((k * 8 + j) % 5) - 2) * 0.2f;
    for (int k = 0; k < 8; k++)
        for (int j = 0; j < 3; j++)
            w1d[k * 3 + j] = (float)(((k * 3 + j) % 4) - 1) * 0.15f;
    float *b0 = (float *)g->tensors[tids[2]].tensor->data;
    float *b1 = (float *)g->tensors[tids[6]].tensor->data;
    for (int j = 0; j < 8; j++) b0[j] = 0.1f * j;
    for (int j = 0; j < 3; j++) b1[j] = 0.05f;
}

/* Runs the MLP and returns the max |out - ref| difference. */
static float run_mlp(FeRuntime *rt, FeTensor *input, FeTensor *output,
                     const float *ref, int n_out) {
    assert(fe_runtime_run(rt, input, output) == FE_OK);
    float *o = (float *)output->data, maxdiff = 0.0f;
    for (int i = 0; i < n_out; i++) {
        float d = fabsf(o[i] - ref[i]);
        if (d > maxdiff) maxdiff = d;
    }
    return maxdiff;
}

enum { MLP_WBUF = 256 * 1024, MLP_ABUF = 256 * 1024 };

static void test_engine_int16_path(void) {
    /* Same MLP, two engines: float reference vs INT16-quantized weights.
     * INT16 holds far more weight precision than INT8, so output drift
     * should remain well under the INT8 tolerance. */
    FeGraph gf, gq;
    int tf[MLP_TENSOR_N], tq[MLP_TENSOR_N];
    build_mlp(&gf, tf); build_mlp(&gq, tq);

    static unsigned char wf[MLP_WBUF], af[MLP_ABUF];
    static unsigned char wq[MLP_WBUF], aq[MLP_ABUF];
    FeRuntime rf, rq;
    assert(fe_runtime_init(&rf, &gf, wf, sizeof wf, af, sizeof af) == FE_OK);
    assert(fe_runtime_init(&rq, &gq, wq, sizeof wq, aq, sizeof aq) == FE_OK);
    assert(fe_runtime_alloc_weights(&rf) == FE_OK);
    assert(fe_runtime_alloc_weights(&rq) == FE_OK);
    fill_mlp_weights(&gf, tf); fill_mlp_weights(&gq, tq);

    int s_in[] = {1, 4}, s_out[] = {1, 3};
    FeTensor *in = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
    ((float *)in->data)[0] = 0.7f;  ((float *)in->data)[1] = -1.3f;
    ((float *)in->data)[2] = 0.4f;  ((float *)in->data)[3] = 2.1f;
    FeTensor *of = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_out);
    FeTensor *oq = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_out);

    float ref[3];
    assert(fe_runtime_run(&rf, in, of) == FE_OK);
    memcpy(ref, of->data, sizeof ref);

    assert(fe_quantize_model_int16(&gq, &rq.weight_arena) == FE_OK);
    /* Weight dtype should now be INT16 with per-channel scales recorded. */
    assert(gq.tensors[tq[1]].tensor->dtype == DTYPE_INT16);
    assert(gq.tensors[tq[1]].n_scales == 8);
    assert(gq.tensors[tq[5]].tensor->dtype == DTYPE_INT16);

    float maxdiff = run_mlp(&rq, in, oq, ref, 3);
    printf("INT16 engine path max diff vs float: %.5f\n", maxdiff);
    assert(maxdiff < 0.05f);

    fe_tensor_free(in); fe_tensor_free(of); fe_tensor_free(oq);
    printf("PASS test_engine_int16_path\n");
}

static void test_engine_static_int8_path(void) {
    /* Calibrating then fe_quantize_model_static must route the engine to the
     * single-pass static kernels and (per Stage 15 calibration) stay close to
     * the float reference. */
    FeGraph gf, gs;
    int tf[MLP_TENSOR_N], ts[MLP_TENSOR_N];
    build_mlp(&gf, tf); build_mlp(&gs, ts);

    static unsigned char wf[MLP_WBUF], af[MLP_ABUF];
    static unsigned char ws[MLP_WBUF], as[MLP_ABUF];
    FeRuntime rf, rs;
    assert(fe_runtime_init(&rf, &gf, wf, sizeof wf, af, sizeof af) == FE_OK);
    assert(fe_runtime_init(&rs, &gs, ws, sizeof ws, as, sizeof as) == FE_OK);
    assert(fe_runtime_alloc_weights(&rf) == FE_OK);
    assert(fe_runtime_alloc_weights(&rs) == FE_OK);
    fill_mlp_weights(&gf, tf); fill_mlp_weights(&gs, ts);

    int s_in[] = {1, 4}, s_out[] = {1, 3};
    FeTensor *in  = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
    FeTensor *in2 = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
    ((float *)in->data)[0] = 0.7f;  ((float *)in->data)[1] = -1.3f;
    ((float *)in->data)[2] = 0.4f;  ((float *)in->data)[3] = 2.1f;
    memcpy(in2->data, in->data, 4 * sizeof(float));
    FeTensor *of = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_out);
    FeTensor *os = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_out);

    float ref[3];
    assert(fe_runtime_run(&rf, in, of) == FE_OK);
    memcpy(ref, of->data, sizeof ref);

    /* Calibrate the STATIC engine with two samples, then apply static
     * scales + INT8 weight repack. */
    FeTensor *calib[2] = {in, in2};
    float ranges[16];
    assert(fe_runtime_calibrate(&rs, calib, 2, os, ranges) == FE_OK);
    assert(fe_quantize_model_static(&gs, &rs.weight_arena, ranges) == FE_OK);

    /* The activation input of linear0 must have a recorded static scale. */
    assert(gs.tensors[ts[0]].act_scale > 0.0f);
    assert(gs.tensors[ts[1]].tensor->dtype == DTYPE_INT8);

    float maxdiff = run_mlp(&rs, in2, os, ref, 3);
    printf("Static INT8 engine path max diff vs float: %.5f\n", maxdiff);
    assert(maxdiff < 0.2f);

    fe_tensor_free(in); fe_tensor_free(in2);
    fe_tensor_free(of); fe_tensor_free(os);
    printf("PASS test_engine_static_int8_path\n");
}

static void test_engine_half_path(void) {
    /* FP16/BF16 weights: repack in place to half storage, then verify the
     * exec plan upconverts on read (fe_ensure_f32_weight) so the float
     * kernels produce the same result as the FP32 reference. */
    FeGraph gf, gh;
    int tf[MLP_TENSOR_N], th[MLP_TENSOR_N];
    build_mlp(&gf, tf); build_mlp(&gh, th);

    static unsigned char wf[MLP_WBUF], af[MLP_ABUF];
    static unsigned char wh[MLP_WBUF], ah[MLP_ABUF];
    FeRuntime rf, rh;
    assert(fe_runtime_init(&rf, &gf, wf, sizeof wf, af, sizeof af) == FE_OK);
    assert(fe_runtime_init(&rh, &gh, wh, sizeof wh, ah, sizeof ah) == FE_OK);
    assert(fe_runtime_alloc_weights(&rf) == FE_OK);
    assert(fe_runtime_alloc_weights(&rh) == FE_OK);
    fill_mlp_weights(&gf, tf); fill_mlp_weights(&gh, th);

    int s_in[] = {1, 4}, s_out[] = {1, 3};
    FeTensor *in = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_in);
    ((float *)in->data)[0] = 0.7f;  ((float *)in->data)[1] = -1.3f;
    ((float *)in->data)[2] = 0.4f;  ((float *)in->data)[3] = 2.1f;
    FeTensor *of = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_out);
    FeTensor *oh = fe_tensor_alloc(DTYPE_FLOAT32, 2, s_out);

    float ref[3];
    assert(fe_runtime_run(&rf, in, of) == FE_OK);
    memcpy(ref, of->data, sizeof ref);

    /* FP16 path. Weights allocated to 4 bytes/elem, so halving their
     * storage to 2 bytes/elem then adding an FP32 shadow still fits in the
     * weight arena. */
    assert(fe_repack_model_half(&gh, &rh.weight_arena, 0) == FE_OK);
    assert(gh.tensors[th[1]].tensor->dtype == DTYPE_FLOAT16);
    float d16 = run_mlp(&rh, in, oh, ref, 3);
    /* round-to-nearest-even FP16 of ~0.2-scale weights: ~1e-3 abs error */
    assert(gh.tensors[th[1]].shadow != NULL);   /* upconverted on first run */
    printf("FP16 engine path max diff vs float: %.6f\n", d16);
    assert(d16 < 0.01f);

    printf("PASS test_engine_half_path\n");
    fe_tensor_free(in); fe_tensor_free(of); fe_tensor_free(oh);
}

static void test_static_kernel_equivalence(void) {
    /* Static kernels must be exactly the dynamic kernels when act_scale is
     * the same value the dynamic pass would have derived (max/127 or
     * max/32767). */
    int shapeA[] = {3, 9}, shapeW[] = {9, 4}, shapeC[] = {3, 4};
    FeTensor *A = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeA);
    FeTensor *Wq = fe_tensor_alloc(DTYPE_INT8, 2, shapeW);
    FeTensor *W16 = fe_tensor_alloc(DTYPE_INT16, 2, shapeW);
    FeTensor *Cd = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeC);
    FeTensor *Cs = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeC);

    for (int i = 0; i < 27; i++) ((float *)A->data)[i] = (float)(i % 9 - 4) * 0.3f;
    float scales[4], sc16[4];
    assert(fe_quantize_per_channel_16(Wq, Wq, NULL) != FE_OK); /* wrong dtype */
    assert(fe_tensor_free != NULL);

    /* Quantize the FP32 source into Wq/W16 via two temp tensors. */
    FeTensor *Wf = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeW);
    float *wd = (float *)Wf->data;
    for (int k = 0; k < 9; k++)
        for (int j = 0; j < 4; j++)
            wd[k * 4 + j] = (float)((k * 4 + j) % 6 - 3) * 0.5f;
    assert(fe_quantize_per_channel(Wf, Wq, scales) == FE_OK);
    assert(fe_quantize_per_channel_16(Wf, W16, sc16) == FE_OK);

    /* Dynamic INT8 activation scale: max |A| / 127. */
    float maxa = 0.0f;
    for (int i = 0; i < 27; i++) {
        float f = ((float *)A->data)[i];
        maxa = maxa < fabsf(f) ? fabsf(f) : maxa;
    }
    float dyn_scale = maxa / 127.0f;

    assert(fe_matmul_int8_dyn(A, Wq, scales, Cd) == FE_OK);
    assert(fe_matmul_int8_static(A, Wq, scales, dyn_scale, Cs) == FE_OK);
    {
        float *a = (float *)Cd->data, *b = (float *)Cs->data;
        for (int i = 0; i < 12; i++) assert(b[i] == a[i]);
    }

    /* Same for the INT16 pair (max / 32767). */
    FeTensor *Cd16 = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeC);
    FeTensor *Cs16 = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeC);
    float dyn16 = maxa / 32767.0f;
    assert(fe_matmul_int16_dyn(A, W16, sc16, Cd16) == FE_OK);
    assert(fe_matmul_int16_static(A, W16, sc16, dyn16, Cs16) == FE_OK);
    {
        float *a = (float *)Cd16->data, *b = (float *)Cs16->data;
        for (int i = 0; i < 12; i++) assert(b[i] == a[i]);
    }

    fe_tensor_free(A); fe_tensor_free(Wq); fe_tensor_free(W16);
    fe_tensor_free(Wf);
    fe_tensor_free(Cd); fe_tensor_free(Cs);
    fe_tensor_free(Cd16); fe_tensor_free(Cs16);
    printf("PASS test_static_kernel_equivalence\n");
}

static void test_int16_scale_uses_full_grid(void) {
    /* INT16 dynamic quantization must use the full 16-bit grid. If a kernel
     * clipped activations to [-127,127] (the Stage 15 quirk), INT16 would
     * differ from INT8. Load the activation range tightly so quantization
     * error stays visible, and require INT16 accuracy near the float one. */
    int shapeA[] = {2, 16}, shapeW[] = {16, 3}, shapeC[] = {2, 3};
    FeTensor *A   = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeA);
    FeTensor *Wf  = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeW);
    FeTensor *W16 = fe_tensor_alloc(DTYPE_INT16, 2, shapeW);
    FeTensor *W8  = fe_tensor_alloc(DTYPE_INT8, 2, shapeW);
    FeTensor *C   = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeC);
    FeTensor *Cf  = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeC);
    FeTensor *C8  = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeC);

    for (int i = 0; i < 32; i++) ((float *)A->data)[i] = (float)(i % 13 - 6) * 0.01f;
    for (int k = 0; k < 16; k++)
        for (int j = 0; j < 3; j++)
            ((float *)Wf->data)[k * 3 + j] = (float)(k % 5 - 2) * 0.1f;

    float s16[3], s8[3];
    assert(fe_quantize_per_channel_16(Wf, W16, s16) == FE_OK);
    assert(fe_quantize_per_channel(Wf, W8, s8) == FE_OK);

    float *cf = (float *)Cf->data;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++) {
            float acc = 0.0f;
            for (int k = 0; k < 16; k++)
                acc += ((float *)A->data)[i * 16 + k] *
                       ((float *)Wf->data)[k * 3 + j];
            cf[i * 3 + j] = acc;
        }

    assert(fe_matmul_int16_dyn(A, W16, s16, C) == FE_OK);
    assert(fe_matmul_int8_dyn(A, W8, s8, C8) == FE_OK);

    float err16 = 0.0f, err8 = 0.0f;
    for (int i = 0; i < 6; i++) {
        float d16 = fabsf(((float *)C->data)[i] - cf[i]);
        float d8  = fabsf(((float *)C8->data)[i] - cf[i]);
        if (d16 > err16) err16 = d16;
        if (d8  > err8)  err8  = d8;
    }
    printf("INT16 err %.5f vs INT8 err %.5f (float ref)\n", err16, err8);
    assert(err16 < 1e-3f);   /* 16-bit quantization: ~0.5% relative */
    assert(err16 < err8);    /* strictly better than INT8 on same data */

    fe_tensor_free(A); fe_tensor_free(Wf); fe_tensor_free(W16);
    fe_tensor_free(W8); fe_tensor_free(C); fe_tensor_free(Cf);
    fe_tensor_free(C8);
    printf("PASS test_int16_scale_uses_full_grid\n");
}

int main(void) {
    test_quantize_dequantize();
    test_matmul_int8_accuracy();
    test_memory_reduction();
    test_per_channel_beats_per_tensor_on_skewed_weights();
    test_matmul_int8_per_channel_accuracy();
    test_calibrate_activations();
    test_calibrate_pct_smoothing();
    test_static_kernel_equivalence();
    test_int16_scale_uses_full_grid();
    test_engine_int16_path();
    test_engine_static_int8_path();
    test_engine_half_path();
    printf("\nAll tests passed.\n");
    return 0;
}