// tests/test_ops.c
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "../core/tensor.h"
#include "../ops/ops.h"

#define EPSILON 1e-5f

static int nearly_equal(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

static void test_matmul_identity(void) {
    /* A @ I = A */
    int shapeA[] = {2, 3};
    int shapeI[] = {3, 3};
    int shapeC[] = {2, 3};

    FeTensor *A = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeA);
    FeTensor *I = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeI);
    FeTensor *C = fe_tensor_alloc(DTYPE_FLOAT32, 2, shapeC);

    float a_data[] = {1,2,3, 4,5,6};
    float i_data[] = {1,0,0, 0,1,0, 0,0,1};
    memcpy(A->data, a_data, sizeof(a_data));
    memcpy(I->data, i_data, sizeof(i_data));

    assert(fe_matmul(A, I, C) == FE_OK);

    int idx0[] = {0,0}; assert(nearly_equal(fe_tensor_get_f32(C, idx0), 1.0f));
    int idx1[] = {0,1}; assert(nearly_equal(fe_tensor_get_f32(C, idx1), 2.0f));
    int idx2[] = {1,2}; assert(nearly_equal(fe_tensor_get_f32(C, idx2), 6.0f));

    fe_tensor_free(A); fe_tensor_free(I); fe_tensor_free(C);
    printf("PASS test_matmul_identity\n");
}

static void test_matmul_known(void) {
    /* [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]] */
    int shape22[] = {2, 2};
    FeTensor *A = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape22);
    FeTensor *B = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape22);
    FeTensor *C = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape22);

    float a[] = {1,2,3,4};
    float b[] = {5,6,7,8};
    memcpy(A->data, a, sizeof(a));
    memcpy(B->data, b, sizeof(b));

    assert(fe_matmul(A, B, C) == FE_OK);

    int idx00[] = {0,0}; assert(nearly_equal(fe_tensor_get_f32(C, idx00), 19.0f));
    int idx01[] = {0,1}; assert(nearly_equal(fe_tensor_get_f32(C, idx01), 22.0f));
    int idx10[] = {1,0}; assert(nearly_equal(fe_tensor_get_f32(C, idx10), 43.0f));
    int idx11[] = {1,1}; assert(nearly_equal(fe_tensor_get_f32(C, idx11), 50.0f));

    fe_tensor_free(A); fe_tensor_free(B); fe_tensor_free(C);
    printf("PASS test_matmul_known\n");
}

static void test_relu(void) {
    int shape[] = {6};
    FeTensor *in  = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);

    float data[] = {-2.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f};
    memcpy(in->data, data, sizeof(data));

    assert(fe_relu(in, out) == FE_OK);

    float *o = (float *)out->data;
    assert(nearly_equal(o[0], 0.0f));
    assert(nearly_equal(o[1], 0.0f));
    assert(nearly_equal(o[2], 0.0f));
    assert(nearly_equal(o[3], 0.5f));
    assert(nearly_equal(o[4], 1.0f));
    assert(nearly_equal(o[5], 3.0f));

    fe_tensor_free(in); fe_tensor_free(out);
    printf("PASS test_relu\n");
}

static void test_softmax_sums_to_one(void) {
    int shape[] = {1, 4};
    FeTensor *in  = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(in->data, data, sizeof(data));

    assert(fe_softmax(in, out) == FE_OK);

    float sum = 0.0f;
    float *o = (float *)out->data;
    for (int i = 0; i < 4; i++) {
        assert(o[i] > 0.0f);   /* all positive */
        sum += o[i];
    }
    assert(nearly_equal(sum, 1.0f));   /* sums to 1 */

    fe_tensor_free(in); fe_tensor_free(out);
    printf("PASS test_softmax_sums_to_one\n");
}

static void test_softmax_numerical_stability(void) {
    /* Large values that would overflow naive softmax */
    int shape[] = {1, 3};
    FeTensor *in  = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);

    float data[] = {1000.0f, 1001.0f, 1002.0f};
    memcpy(in->data, data, sizeof(data));

    assert(fe_softmax(in, out) == FE_OK);

    float sum = 0.0f;
    float *o = (float *)out->data;
    for (int i = 0; i < 3; i++) sum += o[i];
    assert(nearly_equal(sum, 1.0f));   /* must not produce NaN/Inf */

    fe_tensor_free(in); fe_tensor_free(out);
    printf("PASS test_softmax_numerical_stability\n");
}

int main(void) {
    test_matmul_identity();
    test_matmul_known();
    test_relu();
    test_softmax_sums_to_one();
    test_softmax_numerical_stability();
    printf("\nAll tests passed.\n");
    return 0;
}