// tests/test_ops.c
#include <stdio.h>
#include <string.h>
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

static void test_matmul_1x1(void) {
    /* [[3]] @ [[4]] = [[12]] */
    int shape[] = {1, 1};
    FeTensor *A = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *B = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *C = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    ((float *)A->data)[0] = 3.0f;
    ((float *)B->data)[0] = 4.0f;

    assert(fe_matmul(A, B, C) == FE_OK);
    assert(nearly_equal(((float *)C->data)[0], 12.0f));

    fe_tensor_free(A); fe_tensor_free(B); fe_tensor_free(C);
    printf("PASS test_matmul_1x1\n");
}

static void test_matmul_shape_mismatch(void) {
    int sA[] = {2, 3};
    int sB[] = {4, 5};   /* B.rows != A.cols -> must fail loudly */
    int sC[] = {2, 5};
    FeTensor *A = fe_tensor_alloc(DTYPE_FLOAT32, 2, sA);
    FeTensor *B = fe_tensor_alloc(DTYPE_FLOAT32, 2, sB);
    FeTensor *C = fe_tensor_alloc(DTYPE_FLOAT32, 2, sC);
    assert(fe_matmul(A, B, C) == FE_ERR_SHAPE);
    fe_tensor_free(A); fe_tensor_free(B); fe_tensor_free(C);
    printf("PASS test_matmul_shape_mismatch\n");
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

static void test_elementwise(void) {
    int shape[] = {4};
    FeTensor *a = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);
    FeTensor *b = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);

    for (int i = 0; i < 4; i++) {
        ((float *)a->data)[i] = (float)(i + 1);   /* 1,2,3,4 */
        ((float *)b->data)[i] = 2.0f;
    }

    assert(fe_add(a, b, out) == FE_OK);
    assert(((float *)out->data)[0] == 3.0f && ((float *)out->data)[3] == 6.0f);

    assert(fe_mul(a, b, out) == FE_OK);
    assert(((float *)out->data)[0] == 2.0f && ((float *)out->data)[3] == 8.0f);

    assert(fe_div(a, b, out) == FE_OK);
    assert(((float *)out->data)[0] == 0.5f && ((float *)out->data)[3] == 2.0f);

    fe_tensor_free(a); fe_tensor_free(b); fe_tensor_free(out);
    printf("PASS test_elementwise\n");
}

static void test_scalar_ops(void) {
    int shape[] = {3};
    FeTensor *a = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);

    ((float *)a->data)[0] = 1.0f;
    ((float *)a->data)[1] = 2.0f;
    ((float *)a->data)[2] = 3.0f;

    assert(fe_mul_scalar(a, 10.0f, out) == FE_OK);
    assert(((float *)out->data)[0] == 10.0f && ((float *)out->data)[2] == 30.0f);

    assert(fe_neg(a, out) == FE_OK);
    assert(((float *)out->data)[0] == -1.0f && ((float *)out->data)[2] == -3.0f);

    fe_tensor_free(a); fe_tensor_free(out);
    printf("PASS test_scalar_ops\n");
}

static void test_reduce(void) {
    int shape[] = {2, 3};
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    /* [[1,2,3],[4,5,6]] */
    float vals[] = {1,2,3,4,5,6};
    for (int i = 0; i < 6; i++) ((float *)t->data)[i] = vals[i];

    int out_shape[] = {2};
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 1, out_shape);

    assert(fe_sum(t, 1, out) == FE_OK);     /* sum along axis 1 -> [6, 15] */
    assert(((float *)out->data)[0] == 6.0f);
    assert(((float *)out->data)[1] == 15.0f);

    assert(fe_max(t, 1, out) == FE_OK);
    assert(((float *)out->data)[0] == 3.0f);
    assert(((float *)out->data)[1] == 6.0f);

    fe_tensor_free(t); fe_tensor_free(out);
    printf("PASS test_reduce\n");
}

static void test_reduce_middle_axis(void) {
    /* Reduce a 3D tensor over its middle axis: shape [2,3,4] over axis 1. */
    int sIn[] = {2, 3, 4};
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, 3, sIn);
    for (int i = 0; i < 24; i++) ((float *)t->data)[i] = (float)i;

    /* Sum over axis 1. Out shape [2,4].
     * out[b][c] = sum_k t[b][k][c] */
    int sOut[] = {2, 4};
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 2, sOut);
    assert(fe_sum(t, 1, out) == FE_OK);

    for (int b = 0; b < 2; b++)
        for (int c = 0; c < 4; c++) {
            float expect = 0.0f;
            for (int k = 0; k < 3; k++)
                expect += (float)(b * 12 + k * 4 + c);
            assert(nearly_equal(((float *)out->data)[b * 4 + c], expect));
        }

    /* Max over axis 1 — out[b][c] = max over k. Values ascend with d,
     * so within (b,c) the max is at k=2. */
    assert(fe_max(t, 1, out) == FE_OK);
    for (int b = 0; b < 2; b++)
        for (int c = 0; c < 4; c++)
            assert(nearly_equal(((float *)out->data)[b * 4 + c],
                                (float)(b * 12 + 2 * 4 + c)));

    /* Invalid axis rejected. */
    assert(fe_sum(t, 3, out) == FE_ERR_SHAPE);

    fe_tensor_free(t); fe_tensor_free(out);
    printf("PASS test_reduce_middle_axis\n");
}

static void test_dot(void) {
    int shape[] = {3};
    FeTensor *a = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);
    FeTensor *b = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);

    float av[] = {1, 2, 3};
    float bv[] = {4, 5, 6};
    for (int i = 0; i < 3; i++) {
        ((float *)a->data)[i] = av[i];
        ((float *)b->data)[i] = bv[i];
    }

    float result;
    assert(fe_dot(a, b, &result) == FE_OK);
    assert(result == 32.0f);   /* 1*4 + 2*5 + 3*6 = 32 */

    fe_tensor_free(a); fe_tensor_free(b);
    printf("PASS test_dot\n");
}

static void test_stability(void) {
    int shape[] = {3};
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);
    ((float *)t->data)[0] = 1000.0f;
    ((float *)t->data)[1] = 1001.0f;
    ((float *)t->data)[2] = 1002.0f;

    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);
    assert(fe_stable_exp_normalize(t, out) == FE_OK);   /* must not overflow/NaN */

    float sum = ((float *)out->data)[0] + ((float *)out->data)[1] + ((float *)out->data)[2];
    assert(fabsf(sum - 1.0f) < 0.0001f);   /* valid probability distribution */

    float lse;
    assert(fe_logsumexp(t, &lse) == FE_OK);
    assert(lse > 1000.0f && lse < 1003.0f);   /* sane range, no overflow */

    fe_tensor_free(t); fe_tensor_free(out);
    printf("PASS test_stability\n");
}

static void test_rand_uniform(void) {
    int shape[] = {256};
    FeTensor *a = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);
    FeTensor *b = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);

    /* Same seed reproduces byte-identical output across runs. */
    assert(fe_rand_uniform(a, 0.0f, 1.0f, 42u) == FE_OK);
    assert(fe_rand_uniform(b, 0.0f, 1.0f, 42u) == FE_OK);
    for (int i = 0; i < 256; i++)
        assert(((float *)a->data)[i] == ((float *)b->data)[i]);

    /* Different seed diverges. */
    assert(fe_rand_uniform(b, 0.0f, 1.0f, 43u) == FE_OK);
    int same = 1;
    for (int i = 0; i < 256; i++)
        if (((float *)a->data)[i] != ((float *)b->data)[i]) { same = 0; break; }
    assert(!same);

    /* Values stay within [low, high). */
    assert(fe_rand_uniform(a, -5.0f, 5.0f, 7u) == FE_OK);
    for (int i = 0; i < 256; i++) {
        float v = ((float *)a->data)[i];
        assert(v >= -5.0f && v < 5.0f);
    }

    /* Bad args rejected. */
    assert(fe_rand_uniform(a, 1.0f, 1.0f, 0u) == FE_ERR_SHAPE);

    fe_tensor_free(a); fe_tensor_free(b);
    printf("PASS test_rand_uniform\n");
}

static void test_rand_normal(void) {
    int shape[] = {512};
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);

    assert(fe_rand_normal(t, 100.0f, 2.0f, 99u) == FE_OK);

    /* Deterministic: same seed, same stream. */
    float first = ((float *)t->data)[0];
    assert(fe_rand_normal(t, 100.0f, 2.0f, 99u) == FE_OK);
    assert(((float *)t->data)[0] == first);

    /* Measure mean/std over the sample — should be ~close to params. */
    double mean = 0.0;
    for (int i = 0; i < 512; i++) mean += ((float *)t->data)[i];
    mean /= 512.0;
    double var = 0.0;
    for (int i = 0; i < 512; i++) {
        double d = ((float *)t->data)[i] - mean;
        var += d * d;
    }
    var /= 512.0;
    assert(fabs(mean - 100.0) < 1.0);
    assert(fabs(var - 4.0) < 1.0);

    fe_tensor_free(t);
    printf("PASS test_rand_normal\n");
}

/* Zero-size axes: elementwise over an empty tensor is a no-op, and a
 * zero-K matmul is well-defined (fills zeros, touches no data). */
static void test_empty_tensors(void) {
    int s0[] = {0};
    FeTensor *a = fe_tensor_alloc(DTYPE_FLOAT32, 1, s0);
    FeTensor *b = fe_tensor_alloc(DTYPE_FLOAT32, 1, s0);
    FeTensor *o = fe_tensor_alloc(DTYPE_FLOAT32, 1, s0);
    assert(a && b && o);
    assert(fe_tensor_numel(a) == 0 && fe_tensor_numel(o) == 0);
    assert(fe_add(a, b, o) == FE_OK);

    int sA[] = {2, 0}, sB[] = {0, 3}, sC[] = {2, 3};
    FeTensor *A = fe_tensor_alloc(DTYPE_FLOAT32, 2, sA);
    FeTensor *B = fe_tensor_alloc(DTYPE_FLOAT32, 2, sB);
    FeTensor *C = fe_tensor_alloc(DTYPE_FLOAT32, 2, sC);
    assert(fe_matmul(A, B, C) == FE_OK);
    for (int i = 0; i < 6; i++) assert(((float *)C->data)[i] == 0.0f);

    fe_tensor_free(a); fe_tensor_free(b); fe_tensor_free(o);
    fe_tensor_free(A); fe_tensor_free(B); fe_tensor_free(C);
    printf("PASS test_empty_tensors\n");
}

int main(void) {
    test_matmul_identity();
    test_matmul_known();
    test_matmul_1x1();
    test_matmul_shape_mismatch();
    test_empty_tensors();
    test_relu();
    test_softmax_sums_to_one();
    test_softmax_numerical_stability();
    test_elementwise();
    test_scalar_ops();
    test_reduce();
    test_reduce_middle_axis();
    test_dot();
    test_stability();
    test_rand_uniform();
    test_rand_normal();
    printf("\nAll tests passed.\n");
    return 0;
}

