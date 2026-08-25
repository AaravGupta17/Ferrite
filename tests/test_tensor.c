#include <stdio.h>
#include <assert.h>
#include "../core/tensor.h"

static void test_alloc_strides(void) {
    int shape[] = {3, 4};
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    assert(t != NULL);
    assert(t->strides[0] == 4);
    assert(t->strides[1] == 1);
    assert(t->owns_data == true);
    fe_tensor_free(t);
    printf("PASS test_alloc_strides\n");
}

static void test_transpose_no_copy(void) {
    int shape[] = {3, 4};
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *tr = fe_tensor_transpose(t, 0, 1);

    assert(tr->data == t->data);       /* same buffer, no copy */
    assert(tr->shape[0]   == 4);
    assert(tr->shape[1]   == 3);
    assert(tr->strides[0] == 1);
    assert(tr->strides[1] == 4);
    assert(tr->owns_data  == false);

    fe_tensor_free(tr);
    fe_tensor_free(t);
    printf("PASS test_transpose_no_copy\n");
}

static void test_reshape(void) {
    int shape[] = {12};
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);

    int new_shape[] = {3, 4};
    FeTensor *r = fe_tensor_reshape(t, 2, new_shape);
    assert(r != NULL);
    assert(r->shape[0] == 3);
    assert(r->shape[1] == 4);
    assert(r->data == t->data);   /* view, no copy */

    fe_tensor_free(r);
    fe_tensor_free(t);
    printf("PASS test_reshape\n");
}


static void test_slice(void) {
    int shape[] = {4, 3};
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    for (int i = 0; i < 12; i++) ((float *)t->data)[i] = (float)i;

    FeTensor *s = fe_tensor_slice(t, 0, 1, 2);   /* rows 1..2 of a 4x3 tensor */
    assert(s->shape[0] == 2 && s->shape[1] == 3);

    int idx[] = {0, 0};
    assert(fe_tensor_get_f32(s, idx) == 3.0f);   /* row 1, col 0 of original = element 3 */
    idx[0] = 1; idx[1] = 2;
    assert(fe_tensor_get_f32(s, idx) == 8.0f);   /* row 2, col 2 of original = element 8 */

    fe_tensor_free(s);
    fe_tensor_free(t);
    printf("test_slice passed\n");
}

static void test_broadcast(void) {
    int shape[] = {1, 3};                 /* a "row" of 3 values */
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    ((float *)t->data)[0] = 10.0f;
    ((float *)t->data)[1] = 20.0f;
    ((float *)t->data)[2] = 30.0f;

    int target[] = {4, 3};
    FeTensor *b = fe_tensor_broadcast_to(t, 2, target);
    assert(b != NULL);
    assert(b->shape[0] == 4 && b->shape[1] == 3);
    assert(b->strides[0] == 0);   /* broadcast axis */
    assert(b->strides[1] == t->strides[1]);

    /* Every "row" should read back the same 10/20/30 */
    for (int row = 0; row < 4; row++) {
        int idx[] = {row, 1};
        assert(fe_tensor_get_f32(b, idx) == 20.0f);
    }

    fe_tensor_free(b);
    fe_tensor_free(t);
    printf("PASS test_broadcast\n");
}

static void test_reshape_noncontiguous_fails(void) {
    int shape[] = {3, 4};
    FeTensor *t  = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *tr = fe_tensor_transpose(t, 0, 1);  /* non-contiguous */

    int new_shape[] = {12};
    FeTensor *r = fe_tensor_reshape(tr, 1, new_shape);
    assert(r == NULL);   /* must fail */

    fe_tensor_free(tr);
    fe_tensor_free(t);
    printf("PASS test_reshape_noncontiguous_fails\n");
}

static void test_get_set_after_transpose(void) {
    int shape[] = {3, 4};
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);

    /* Fill with row*10 + col */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            int idx[] = {i, j};
            fe_tensor_set_f32(t, idx, (float)(i * 10 + j));
        }
    }

    FeTensor *tr = fe_tensor_transpose(t, 0, 1);  /* 4x3 view */

    /* tr[2][1] should equal t[1][2] = 1*10+2 = 12 */
    int idx[] = {2, 1};
    float val = fe_tensor_get_f32(tr, idx);
    assert(val == 12.0f);

    fe_tensor_free(tr);
    fe_tensor_free(t);
    printf("PASS test_get_set_after_transpose\n");
}

static void test_allclose(void) {
    int shape[] = {2, 2};
    FeTensor *a = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *b = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);

    for (int i = 0; i < 4; i++) {
        ((float *)a->data)[i] = (float)i;
        ((float *)b->data)[i] = (float)i + 0.0001f;   /* tiny diff */
    }

    assert(fe_tensor_allclose(a, b, 0.001f) == true);
    assert(fe_tensor_allclose(a, b, 0.00001f) == false);

    int shape2[] = {2, 3};
    FeTensor *c = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape2);
    assert(fe_tensor_allclose(a, c, 0.1f) == false);   /* shape mismatch */

    fe_tensor_free(a);
    fe_tensor_free(b);
    fe_tensor_free(c);
    printf("PASS test_allclose\n");
}

int main(void) {
    test_alloc_strides();
    test_transpose_no_copy();
    test_reshape();
    test_reshape_noncontiguous_fails();
    test_get_set_after_transpose();
    test_slice();
    test_broadcast();
    test_allclose();
    printf("\nAll tests passed.\n");
    return 0;
}