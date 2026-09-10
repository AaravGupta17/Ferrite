#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "../core/tensor.h"
#include "../core/fp16.h"

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

static void test_slice_noncontiguous(void) {
    /* Slicing a transposed (non-contiguous) tensor must shift the data
     * pointer by `start * strides[axis]` and keep the parent's strides —
     * the view still reads through the transposed layout. */
    int shape[] = {2, 3};
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    for (int i = 0; i < 6; i++) ((float *)t->data)[i] = (float)i;
    /* A[0] = 0 1 2 ; A[1] = 3 4 5, strides {3,1} */
    FeTensor *tr = fe_tensor_transpose(t, 0, 1);
    /* T shape {3,2}, strides {1,3}: T = [0 3; 1 4; 2 5] */

    /* Outer-axis slice: T[1:2, :] = [1 4] — non-contiguous in memory
     * (elements at byte offsets 4 and 16). */
    FeTensor *s1 = fe_tensor_slice(tr, 0, 1, 1);
    assert(s1->shape[0] == 1 && s1->shape[1] == 2);
    assert(s1->strides[0] == 1 && s1->strides[1] == 3);
    assert((char *)s1->data == (char *)t->data + 4);   /* start * stride0 * 4 */
    int idx[] = {0, 0};
    assert(fe_tensor_get_f32(s1, idx) == 1.0f);
    idx[1] = 1;
    assert(fe_tensor_get_f32(s1, idx) == 4.0f);
    fe_tensor_free(s1);

    /* Inner-axis slice: T[:, 1:2] = [3; 4; 5] (single column of the view). */
    FeTensor *s2 = fe_tensor_slice(tr, 1, 1, 1);
    assert(s2->shape[0] == 3 && s2->shape[1] == 1);
    assert(s2->strides[0] == 1 && s2->strides[1] == 3);
    assert((char *)s2->data == (char *)t->data + 1 * 3 * 4);   /* start*stride1*4 */
    int i2[] = {2, 0};
    assert(fe_tensor_get_f32(s2, i2) == 5.0f);
    fe_tensor_free(s2);

    fe_tensor_free(tr);
    fe_tensor_free(t);
    printf("PASS test_slice_noncontiguous\n");
}

static void test_int_dtype_sizing(void) {
    /* Strides are element counts for every dtype; nbytes must scale with
     * fe_dtype_size. INT8/INT32/FLOAT64 are exercised here explicitly. */
    int shape[] = {3, 5};

    FeTensor *i8 = fe_tensor_alloc(DTYPE_INT8, 2, shape);
    FeTensor *i32 = fe_tensor_alloc(DTYPE_INT32, 2, shape);
    FeTensor *f64 = fe_tensor_alloc(DTYPE_FLOAT64, 2, shape);
    assert(i8 && i32 && f64);
    assert(fe_dtype_size(DTYPE_INT8) == 1 && fe_dtype_size(DTYPE_INT32) == 4 &&
           fe_dtype_size(DTYPE_FLOAT64) == 8);
    assert(i8->nbytes == 15);    /* 15 * 1 */
    assert(i32->nbytes == 60);   /* 15 * 4 */
    assert(f64->nbytes == 120);  /* 15 * 8 */
    assert(i32->strides[0] == 5 && i32->strides[1] == 1);
    assert(fe_tensor_numel(i32) == 15);

    /* fe_tensor_copy is dtype-preserving and must round-trip INT32. */
    for (int i = 0; i < 15; i++) ((int32_t *)i32->data)[i] = i * 1000;
    FeTensor *dst = fe_tensor_alloc(DTYPE_INT32, 2, shape);
    assert(fe_tensor_copy(dst, i32) == FE_OK);
    assert(memcmp(dst->data, i32->data, 60) == 0);
    assert(fe_tensor_copy(dst, f64) == FE_ERR_DTYPE);   /* dtype mismatch */

    fe_tensor_free(i8); fe_tensor_free(i32); fe_tensor_free(f64);
    fe_tensor_free(dst);
    printf("PASS test_int_dtype_sizing\n");
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

static void test_broadcast_view_equals_copy(void) {
    /* A stride==0 broadcast view must behave identically to its
     * materialized copy. This is the invariant kernels rely on. */
    int shape[] = {1, 3};
    FeTensor *t  = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    ((float *)t->data)[0] = 1.0f;
    ((float *)t->data)[1] = 2.0f;
    ((float *)t->data)[2] = 3.0f;

    int target[] = {4, 3};
    FeTensor *b = fe_tensor_broadcast_to(t, 2, target);
    assert(b->strides[0] == 0);            /* view, not a copy */
    assert(b->owns_data == false);
    assert(b->data == t->data);

    /* Materialize: fe_tensor_contiguous copies to a dense buffer. */
    FeTensor *materialized = fe_tensor_contiguous(b);
    assert(materialized != NULL);
    assert(materialized->strides[0] == 3);  /* dense, not broadcast */
    assert(materialized->strides[1] == 1);
    assert(materialized->owns_data == true);

    /* The dense copy must reproduce the broadcast pattern:
     * each row is [1,2,3]. */
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 3; c++)
            assert(((float *)materialized->data)[r * 3 + c] == (float)(c + 1));

    fe_tensor_free(b);
    fe_tensor_free(materialized);
    fe_tensor_free(t);
    printf("PASS test_broadcast_view_equals_copy\n");
}

static void test_numel_and_contiguous(void) {
    int shape[] = {2, 3, 4};
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT32, 3, shape);
    assert(fe_tensor_numel(t) == 24);
    assert(fe_tensor_is_contiguous(t) == true);

    /* A transpose makes it non-contiguous. */
    FeTensor *tr = fe_tensor_transpose(t, 0, 2);
    assert(fe_tensor_is_contiguous(tr) == false);
    assert(fe_tensor_numel(tr) == 24);   /* numel unchanged by view */

    fe_tensor_free(tr);
    fe_tensor_free(t);
    printf("PASS test_numel_and_contiguous\n");
}

static void test_copy_ownership(void) {
    /* fe_tensor_copy writes into a caller-allocated output. */
    int shape[] = {2, 3};
    FeTensor *src = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    FeTensor *dst = fe_tensor_alloc(DTYPE_FLOAT32, 2, shape);
    for (int i = 0; i < 6; i++) ((float *)src->data)[i] = (float)(i + 1);

    assert(fe_tensor_copy(dst, src) == FE_OK);
    for (int i = 0; i < 6; i++)
        assert(((float *)dst->data)[i] == (float)(i + 1));

    /* Element-count mismatch must fail, not silently truncate. */
    int bad_shape[] = {3, 3};   /* 9 elements vs src's 6 */
    FeTensor *bad = fe_tensor_alloc(DTYPE_FLOAT32, 2, bad_shape);
    assert(fe_tensor_copy(bad, src) == FE_ERR_SHAPE);

    fe_tensor_free(src); fe_tensor_free(dst); fe_tensor_free(bad);
    printf("PASS test_copy_ownership\n");
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

static void test_float64(void) {
    /* The dtype enum must cover float64 with the correct element size,
     * and the tensor machinery (alloc, contiguous repack, copy) must
     * treat it as an opaque 8-byte element just like any other dtype. */
    assert(fe_dtype_size(DTYPE_FLOAT64) == 8);

    int shape[] = {2, 3};
    FeTensor *t = fe_tensor_alloc(DTYPE_FLOAT64, 2, shape);
    assert(t != NULL);
    assert(t->dtype == DTYPE_FLOAT64);
    assert(t->nbytes == 2 * 3 * 8);

    for (int i = 0; i < 6; i++)
        ((double *)t->data)[i] = 0.5 * i;

    /* Copy preserves bytes across dtypes. */
    FeTensor *u = fe_tensor_alloc(DTYPE_FLOAT64, 2, shape);
    assert(fe_tensor_copy(u, t) == FE_OK);
    for (int i = 0; i < 6; i++)
        assert(((double *)u->data)[i] == 0.5 * i);

    fe_tensor_free(u);
    fe_tensor_free(t);
    printf("PASS test_float64\n");
}

/*
 * Stage 15: FP16 storage support. DTYPE_FLOAT16 is a storage format —
 * compute happens in FP32 after upconvert. Check known bit patterns, the
 * dtype size, and a bulk round-trip against a host-side oracle sweep.
 */
static void test_fp16_roundtrip(void) {
    assert(fe_dtype_size(DTYPE_FLOAT16) == 2);

    /* Exact representable values: bit pattern + round-trip. */
    struct { float v; uint16_t bits; } cases[] = {
        {  1.0f, 0x3C00 }, { -2.0f, 0xC000 }, {  0.5f, 0x3800 },
        {  0.0f, 0x0000 }, { -0.0f, 0x8000 }, { 65504.0f, 0x7BFF },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        uint16_t h = fe_f32_to_fp16(cases[i].v);
        assert(h == cases[i].bits);
        float back = fe_fp16_to_f32(h);
        assert(back == cases[i].v);
    }

    /* Inf saturates to half Inf; NaN keeps the NaN class. */
    assert(fe_f32_to_fp16((float)INFINITY) == 0x7C00);
    assert(fe_f32_to_fp16(-(float)INFINITY) == 0xFC00);
    assert((fe_f32_to_fp16((float)NAN) & 0x7C00) == 0x7C00);

    /* Sweep values through round-trip; FP16 precision <= 2^-10 relative. */
    float vals[] = {0.1f, 0.333f, 3.14159f, 100.0f, -42.5f, 1.0e-4f, 6.1e-5f};
    for (size_t i = 0; i < sizeof(vals) / sizeof(vals[0]); i++) {
        float back = fe_fp16_to_f32(fe_f32_to_fp16(vals[i]));
        float rel = back != 0.0f ? fabsf(back - vals[i]) / fabsf(vals[i]) : 0.0f;
        assert(rel < 0.000978f);           /* 2^-10 + slack */
    }

    /* Bulk conversion with matched position (in place tolerant). */
    float src[4] = {1.0f, 2.0f, -3.5f, 0.25f};
    uint16_t hb[4];
    float dst[4];
    fe_f32_to_fp16_buf(src, hb, 4);
    fe_fp16_to_f32_buf(hb, dst, 4);
    for (int i = 0; i < 4; i++) assert(dst[i] == src[i]);
    printf("PASS test_fp16_roundtrip\n");
}

static void test_bf16_roundtrip(void) {
    assert(fe_dtype_size(DTYPE_BFLOAT16) == 2);

    /* BF16 keeps the FP32 exponent, 8 mantissa bits. Exact values. */
    struct { float v; uint16_t bits; } bcases[] = {
        {  1.0f, 0x3F80 }, { -2.0f, 0xC000 }, {  0.5f, 0x3F00 },
        {  0.0f, 0x0000 }, { -0.0f, 0x8000 },
    };
    for (size_t i = 0; i < sizeof(bcases) / sizeof(bcases[0]); i++) {
        uint16_t b = fe_f32_to_bf16(bcases[i].v);
        assert(b == bcases[i].bits);
        assert(fe_bf16_to_f32(b) == bcases[i].v);
    }

    /* 3.14159f = 0x40490FDB -> bf16 0x4049 (RNE on low 16 bits: 0x0FDB). */
    assert(fe_f32_to_bf16(3.14159f) == 0x4049);
    assert(fe_f32_to_bf16((float)INFINITY) == 0x7F80);

    /* 8 mantissa bits -> relative error <= 2^-8. */
    float bvals[] = {0.1f, 0.333f, 100.0f, -42.5f, 1.0e-4f, 65536.0f};
    for (size_t i = 0; i < sizeof(bvals) / sizeof(bvals[0]); i++) {
        float back = fe_bf16_to_f32(fe_f32_to_bf16(bvals[i]));
        float rel = back != 0.0f ? fabsf(back - bvals[i]) / fabsf(bvals[i]) : 0.0f;
        assert(rel < 0.0039f);              /* 2^-8 + slack */
    }

    float src[4] = {1.0f, 2.0f, -3.5f, 0.25f};
    uint16_t bb[4];
    float dst[4];
    fe_f32_to_bf16_buf(src, bb, 4);
    fe_bf16_to_f32_buf(bb, dst, 4);
    for (int i = 0; i < 4; i++) assert(dst[i] == src[i]);
    printf("PASS test_bf16_roundtrip\n");
}

int main(void) {
    test_alloc_strides();
    test_transpose_no_copy();
    test_reshape();
    test_reshape_noncontiguous_fails();
    test_get_set_after_transpose();
    test_slice();
    test_slice_noncontiguous();
    test_int_dtype_sizing();
    test_broadcast();
    test_broadcast_view_equals_copy();
    test_numel_and_contiguous();
    test_copy_ownership();
    test_allclose();
    test_float64();
    test_fp16_roundtrip();
    test_bf16_roundtrip();
    printf("\nAll tests passed.\n");
    return 0;
}