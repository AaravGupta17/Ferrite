// tests/test_quant.c
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "../core/tensor.h"
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

int main(void) {
    test_quantize_dequantize();
    test_matmul_int8_accuracy();
    test_memory_reduction();
    printf("\nAll tests passed.\n");
    return 0;
}