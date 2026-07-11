// tests/test_conv1d.c
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "../core/tensor.h"
#include "../ops/ops.h"

#define EPSILON 1e-4f

static int nearly_equal(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

/*
 * Test 1: single filter, no padding, stride 1.
 *
 * input:  [1, 1, 5] = [1, 2, 3, 4, 5]
 * filter: [1, 1, 3] = [1, 0, -1]  (edge detector)
 * output: [1, 1, 3]
 *
 * Expected:
 *   p=0: 1*1 + 2*0 + 3*(-1) = -2
 *   p=1: 2*1 + 3*0 + 4*(-1) = -2
 *   p=2: 3*1 + 4*0 + 5*(-1) = -2
 */
static void test_conv1d_edge_detector(void) {
    int s_in[]  = {1, 1, 5};
    int s_w[]   = {1, 1, 3};
    int s_out[] = {1, 1, 3};

    FeTensor *input  = fe_tensor_alloc(DTYPE_FLOAT32, 3, s_in);
    FeTensor *weight = fe_tensor_alloc(DTYPE_FLOAT32, 3, s_w);
    FeTensor *out    = fe_tensor_alloc(DTYPE_FLOAT32, 3, s_out);

    float in_data[] = {1, 2, 3, 4, 5};
    float w_data[]  = {1, 0, -1};
    memcpy(input->data,  in_data, sizeof(in_data));
    memcpy(weight->data, w_data,  sizeof(w_data));

    assert(fe_conv1d(input, weight, NULL, out, 1, 0) == FE_OK);

    float *o = (float *)out->data;
    assert(nearly_equal(o[0], -2.0f));
    assert(nearly_equal(o[1], -2.0f));
    assert(nearly_equal(o[2], -2.0f));

    fe_tensor_free(input);
    fe_tensor_free(weight);
    fe_tensor_free(out);
    printf("PASS test_conv1d_edge_detector\n");
}

/*
 * Test 2: multiple channels and filters.
 *
 * input:  [1, 2, 4] — 2 channels, length 4
 * filter: [3, 2, 2] — 3 output channels, kernel size 2
 * output: [1, 3, 3]
 *
 * All weights = 1, all inputs = 1.
 * Each output = sum of C_in * K = 2 * 2 = 4 ones = 4.0
 */
static void test_conv1d_multichannel(void) {
    int s_in[]  = {1, 2, 4};
    int s_w[]   = {3, 2, 2};
    int s_out[] = {1, 3, 3};

    FeTensor *input  = fe_tensor_alloc(DTYPE_FLOAT32, 3, s_in);
    FeTensor *weight = fe_tensor_alloc(DTYPE_FLOAT32, 3, s_w);
    FeTensor *out    = fe_tensor_alloc(DTYPE_FLOAT32, 3, s_out);

    float *id = (float *)input->data;
    float *wd = (float *)weight->data;
    for (int i = 0; i < 2*4; i++) id[i] = 1.0f;
    for (int i = 0; i < 3*2*2; i++) wd[i] = 1.0f;

    assert(fe_conv1d(input, weight, NULL, out, 1, 0) == FE_OK);

    float *o = (float *)out->data;
    for (int i = 0; i < 3*3; i++)
        assert(nearly_equal(o[i], 4.0f));

    fe_tensor_free(input);
    fe_tensor_free(weight);
    fe_tensor_free(out);
    printf("PASS test_conv1d_multichannel\n");
}

/*
 * Test 3: padding correctness.
 *
 * input:  [1, 1, 3] = [1, 2, 3]
 * filter: [1, 1, 3] = [1, 1, 1]  (sum filter)
 * pad=1, stride=1 → L_out = (3 - 3 + 2) / 1 + 1 = 3
 *
 * Expected (with zero padding):
 *   p=0: 0*1 + 1*1 + 2*1 = 3
 *   p=1: 1*1 + 2*1 + 3*1 = 6
 *   p=2: 2*1 + 3*1 + 0*1 = 5
 */
static void test_conv1d_padding(void) {
    int s_in[]  = {1, 1, 3};
    int s_w[]   = {1, 1, 3};
    int s_out[] = {1, 1, 3};

    FeTensor *input  = fe_tensor_alloc(DTYPE_FLOAT32, 3, s_in);
    FeTensor *weight = fe_tensor_alloc(DTYPE_FLOAT32, 3, s_w);
    FeTensor *out    = fe_tensor_alloc(DTYPE_FLOAT32, 3, s_out);

    float in_data[] = {1, 2, 3};
    float w_data[]  = {1, 1, 1};
    memcpy(input->data,  in_data, sizeof(in_data));
    memcpy(weight->data, w_data,  sizeof(w_data));

    assert(fe_conv1d(input, weight, NULL, out, 1, 1) == FE_OK);

    float *o = (float *)out->data;
    assert(nearly_equal(o[0], 3.0f));
    assert(nearly_equal(o[1], 6.0f));
    assert(nearly_equal(o[2], 5.0f));

    fe_tensor_free(input);
    fe_tensor_free(weight);
    fe_tensor_free(out);
    printf("PASS test_conv1d_padding\n");
}

int main(void) {
    test_conv1d_edge_detector();
    test_conv1d_multichannel();
    test_conv1d_padding();
    printf("\nAll tests passed.\n");
    return 0;
}