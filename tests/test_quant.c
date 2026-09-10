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

int main(void) {
    test_quantize_dequantize();
    test_matmul_int8_accuracy();
    test_memory_reduction();
    test_per_channel_beats_per_tensor_on_skewed_weights();
    test_matmul_int8_per_channel_accuracy();
    test_calibrate_activations();
    printf("\nAll tests passed.\n");
    return 0;
}