// tests/test_ops3.c — Stage 3 operator library coverage.
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "../core/tensor.h"
#include "../ops/ops.h"

#define EPS 1e-4f

static int nearly(float a, float b) { return fabsf(a - b) < EPS; }

/* ---------------- Basic math ---------------- */

static void test_exp_log_pow(void) {
    int shape[] = {3};
    FeTensor *in  = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);
    ((float *)in->data)[0] = 0.0f;
    ((float *)in->data)[1] = 1.0f;
    ((float *)in->data)[2] = 2.0f;

    assert(fe_exp(in, out) == FE_OK);
    assert(nearly(((float *)out->data)[0], 1.0f));
    assert(nearly(((float *)out->data)[1], expf(1.0f)));
    assert(nearly(((float *)out->data)[2], expf(2.0f)));

    /* log over positive values */
    ((float *)in->data)[0] = 1.0f;
    ((float *)in->data)[1] = expf(1.0f);
    ((float *)in->data)[2] = expf(2.0f);
    assert(fe_ln(in, out) == FE_OK);
    assert(nearly(((float *)out->data)[1], 1.0f));
    assert(nearly(((float *)out->data)[2], 2.0f));

    FeTensor *b = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);
    ((float *)b->data)[0] = 2.0f;
    ((float *)b->data)[1] = 2.0f;
    ((float *)b->data)[2] = 3.0f;
    ((float *)in->data)[0] = 2.0f;
    ((float *)in->data)[1] = 3.0f;
    ((float *)in->data)[2] = 2.0f;
    assert(fe_pow(in, b, out) == FE_OK);
    assert(nearly(((float *)out->data)[0], 4.0f));   /* 2^2 */
    assert(nearly(((float *)out->data)[1], 9.0f));   /* 3^2 */
    assert(nearly(((float *)out->data)[2], 8.0f));   /* 2^3 */

    fe_tensor_free(in); fe_tensor_free(b); fe_tensor_free(out);
    printf("PASS test_exp_log_pow\n");
}

/* ---------------- Activations ---------------- */

static void test_activations(void) {
    int shape[] = {3};
    FeTensor *in  = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 1, shape);
    ((float *)in->data)[0] = -2.0f;
    ((float *)in->data)[1] = 0.0f;
    ((float *)in->data)[2] = 2.0f;

    assert(fe_sigmoid(in, out) == FE_OK);
    assert(nearly(((float *)out->data)[0], 1.0f / (1.0f + expf(2.0f))));
    assert(nearly(((float *)out->data)[1], 0.5f));

    assert(fe_tanh(in, out) == FE_OK);
    assert(nearly(((float *)out->data)[1], 0.0f));
    assert(nearly(((float *)out->data)[2], tanhf(2.0f)));

    assert(fe_leaky_relu(in, out, 0.1f) == FE_OK);
    assert(nearly(((float *)out->data)[0], -0.2f));
    assert(nearly(((float *)out->data)[2], 2.0f));

    assert(fe_elu(in, out, 1.0f) == FE_OK);
    assert(nearly(((float *)out->data)[0], expf(-2.0f) - 1.0f));
    assert(nearly(((float *)out->data)[1], 0.0f));
    assert(nearly(((float *)out->data)[2], 2.0f));

    assert(fe_swish(in, out) == FE_OK);
    assert(nearly(((float *)out->data)[2], 2.0f / (1.0f + expf(-2.0f))));

    assert(fe_gelu(in, out) == FE_OK);
    assert(nearly(((float *)out->data)[1], 0.0f));   /* gelu(0) = 0 */

    fe_tensor_free(in); fe_tensor_free(out);
    printf("PASS test_activations\n");
}

/* ---------------- Linear algebra ---------------- */

static void test_gemm_transpose(void) {
    int sA[] = {2, 3}, sB[] = {3, 2}, sC[] = {2, 2};
    FeTensor *A = fe_tensor_alloc(DTYPE_FLOAT32, 2, sA);
    FeTensor *B = fe_tensor_alloc(DTYPE_FLOAT32, 2, sB);
    FeTensor *C = fe_tensor_alloc(DTYPE_FLOAT32, 2, sC);

    /* A = [[1,2,3],[4,5,6]], B^T = B */
    float a[] = {1,2,3,4,5,6};
    float b[] = {7,8, 9,10, 11,12};   /* 3x2 */
    memcpy(A->data, a, sizeof(a));
    memcpy(B->data, b, sizeof(b));

    assert(fe_gemm(A, 0, B, 0, C, 1.0f, 0.0f) == FE_OK);
    /* C = A@B: [1*7+2*9+3*11, 1*8+2*10+3*12; 4*7+5*9+6*11, 4*8+5*10+6*12] */
    assert(nearly(((float *)C->data)[0], 58.0f));
    assert(nearly(((float *)C->data)[1], 64.0f));
    assert(nearly(((float *)C->data)[2], 139.0f));
    assert(nearly(((float *)C->data)[3], 154.0f));

    /* transA: A is 2x3, transposed to 3x2; result 3x3 not C; use a square endpoint
     * where transA makes sense: pass A^T (3x2) @ B(2x? ) — instead test identity
     * alpha/beta path below. */

    /* alpha/beta: C = 2 * A @ B + 0.5 * C0 */
    FeTensor *C0 = fe_tensor_alloc(DTYPE_FLOAT32, 2, sC);
    for (int i = 0; i < 4; i++) ((float *)C0->data)[i] = 1.0f;
    assert(fe_gemm(A, 0, B, 0, C0, 2.0f, 0.5f) == FE_OK);
    assert(nearly(((float *)C0->data)[0], 2.0f * 58.0f + 0.5f));
    assert(nearly(((float *)C0->data)[3], 2.0f * 154.0f + 0.5f));

    /* transpose kernel: in 2x3, out 3x2, out[i][j] = in[j][i] */
    int sT[] = {3, 2};
    FeTensor *T = fe_tensor_alloc(DTYPE_FLOAT32, 2, sT);
    assert(fe_transpose(A, T) == FE_OK);
    assert(nearly(((float *)T->data)[0], 1.0f));  /* in[0][0] */
    assert(nearly(((float *)T->data)[1], 4.0f));  /* in[1][0] */
    assert(nearly(((float *)T->data)[4], 3.0f));  /* in[0][2] */

    fe_tensor_free(A); fe_tensor_free(B); fe_tensor_free(C);
    fe_tensor_free(C0); fe_tensor_free(T);
    printf("PASS test_gemm_transpose\n");
}

/* ---------------- CNN: pooling ---------------- */

static void test_pool(void) {
    int sIn[] = {1, 1, 4, 4};
    FeTensor *in = fe_tensor_alloc(DTYPE_FLOAT32, 4, sIn);
    /* [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]] */
    for (int i = 0; i < 16; i++) ((float *)in->data)[i] = (float)(i + 1);

    int sMax[] = {1, 1, 2, 2};
    FeTensor *mx = fe_tensor_alloc(DTYPE_FLOAT32, 4, sMax);
    assert(fe_maxpool(in, mx, 2, 2, 2, 2) == FE_OK);
    /* windows: [1,2;5,6]->6, [3,4;7,8]->8, [9,10;13,14]->14, [11,12;15,16]->16 */
    assert(nearly(((float *)mx->data)[0], 6.0f));
    assert(nearly(((float *)mx->data)[1], 8.0f));
    assert(nearly(((float *)mx->data)[2], 14.0f));
    assert(nearly(((float *)mx->data)[3], 16.0f));

    FeTensor *av = fe_tensor_alloc(DTYPE_FLOAT32, 4, sMax);
    assert(fe_avgpool(in, av, 2, 2, 2, 2) == FE_OK);
    assert(nearly(((float *)av->data)[0], (1+2+5+6)/4.0f));
    assert(nearly(((float *)av->data)[3], (11+12+15+16)/4.0f));

    fe_tensor_free(in); fe_tensor_free(mx); fe_tensor_free(av);
    printf("PASS test_pool\n");
}

/* ---------------- CNN: norms ---------------- */

static void test_batchnorm(void) {
    int sIn[] = {1, 2, 2, 2};
    FeTensor *in = fe_tensor_alloc(DTYPE_FLOAT32, 4, sIn);
    /* channel 0: [1,2,3,4], channel 1: [10,12,14,16] */
    float d[] = {1,2,3,4, 10,12,14,16};
    memcpy(in->data, d, sizeof(d));

    int sC[] = {2};
    FeTensor *g = fe_tensor_alloc(DTYPE_FLOAT32, 1, sC);
    FeTensor *b = fe_tensor_alloc(DTYPE_FLOAT32, 1, sC);
    FeTensor *m = fe_tensor_alloc(DTYPE_FLOAT32, 1, sC);
    FeTensor *v = fe_tensor_alloc(DTYPE_FLOAT32, 1, sC);
    ((float *)g->data)[0] = 1.0f; ((float *)g->data)[1] = 1.0f;
    ((float *)b->data)[0] = 0.0f; ((float *)b->data)[1] = 0.0f;
    ((float *)m->data)[0] = 2.5f; ((float *)m->data)[1] = 13.0f;
    ((float *)v->data)[0] = 1.25f; ((float *)v->data)[1] = 5.0f;
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 4, sIn);

    assert(fe_batchnorm(in, g, b, m, v, out, 1e-5f) == FE_OK);
    float *o = (float *)out->data;
    /* ch0: (1-2.5)/sqrt(1.25) = -1.5/1.118 = -1.3416 */
    assert(nearly(o[0], (1.0f - 2.5f) / sqrtf(1.25f)));
    assert(nearly(o[3], (4.0f - 2.5f) / sqrtf(1.25f)));

    fe_tensor_free(in); fe_tensor_free(g); fe_tensor_free(b);
    fe_tensor_free(m); fe_tensor_free(v); fe_tensor_free(out);
    printf("PASS test_batchnorm\n");
}

static void test_layernorm(void) {
    int sIn[] = {2, 3};   /* mean of each row subtracted/normalized */
    FeTensor *in = fe_tensor_alloc(DTYPE_FLOAT32, 2, sIn);
    /* row0 = [1,2,3] mean 2 var 2/3; row1 = [4,5,6] mean 5 */
    float d[] = {1,2,3,4,5,6};
    memcpy(in->data, d, sizeof(d));

    int sG[] = {3};
    FeTensor *g = fe_tensor_alloc(DTYPE_FLOAT32, 1, sG);
    FeTensor *b = fe_tensor_alloc(DTYPE_FLOAT32, 1, sG);
    for (int i = 0; i < 3; i++) {
        ((float *)g->data)[i] = 1.0f;
        ((float *)b->data)[i] = 0.0f;
    }
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 2, sIn);

    assert(fe_layernorm(in, g, b, out, 1e-5f) == FE_OK);
    float *o = (float *)out->data;
    /* known: (1-2)/sqrt(2/3) = -1/0.8165 = -1.2247, (3-2)=1 -> 1.2247 */
    float inv0 = 1.0f / sqrtf(2.0f / 3.0f);
    assert(nearly(o[0], -inv0));
    assert(nearly(o[2], inv0));

    fe_tensor_free(in); fe_tensor_free(g); fe_tensor_free(b); fe_tensor_free(out);
    printf("PASS test_layernorm\n");
}

static void test_groupnorm(void) {
    int sIn[] = {1, 4, 1, 2};   /* 4 channels, group into 2 */
    FeTensor *in = fe_tensor_alloc(DTYPE_FLOAT32, 4, sIn);
    /* ch values: [1,2],[3,4],[5,6],[7,8] */
    float d[] = {1,2,3,4,5,6,7,8};
    memcpy(in->data, d, sizeof(d));

    int sC[] = {4};
    FeTensor *g = fe_tensor_alloc(DTYPE_FLOAT32, 1, sC);
    FeTensor *b = fe_tensor_alloc(DTYPE_FLOAT32, 1, sC);
    for (int i = 0; i < 4; i++) { ((float *)g->data)[i] = 1.0f; ((float *)b->data)[i] = 0.0f; }
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 4, sIn);

    /* group 0 = ch0,ch1 = [1,2,3,4] mean 2.5 var 1.25; group 1 = [5,6,7,8] mean 6.5 */
    assert(fe_groupnorm(in, g, b, out, 2, 1e-5f) == FE_OK);
    float *o = (float *)out->data;
    assert(nearly(o[0], (1.0f - 2.5f) / sqrtf(1.25f)));
    assert(nearly(o[4], (5.0f - 6.5f) / sqrtf(1.25f)));

    fe_tensor_free(in); fe_tensor_free(g); fe_tensor_free(b); fe_tensor_free(out);
    printf("PASS test_groupnorm\n");
}

/* ---------------- CNN: conv2d ---------------- */

static void test_conv2d(void) {
    int sIn[] = {1, 1, 3, 3};
    FeTensor *in = fe_tensor_alloc(DTYPE_FLOAT32, 4, sIn);
    /* 3x3 identity-ish image */
    float img[] = {1,0,0, 0,1,0, 0,0,1};
    memcpy(in->data, img, sizeof(img));

    int sW[] = {1, 1, 2, 2};
    FeTensor *w = fe_tensor_alloc(DTYPE_FLOAT32, 4, sW);
    float k[] = {1,0, 0,0};   /* picks the top-left of each 2x2 window */
    memcpy(w->data, k, sizeof(k));

    /* OH = (3-2)/1+1 = 2, OW = 2 */
    int sOut[] = {1, 1, 2, 2};
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 4, sOut);

    assert(fe_conv2d(in, w, NULL, out, 1, 1, 0, 0) == FE_OK);
    float *o = (float *)out->data;
    /* window top-left of each output: out[0][0] uses in[0][0]=1,
       out[0][1] uses in[0][1]=0, out[1][0] uses in[1][0]=0, out[1][1] uses in[1][1]=1 */
    assert(nearly(o[0], 1.0f));
    assert(nearly(o[1], 0.0f));
    assert(nearly(o[2], 0.0f));
    assert(nearly(o[3], 1.0f));

    fe_tensor_free(in); fe_tensor_free(w); fe_tensor_free(out);
    printf("PASS test_conv2d\n");
}

/* ---------------- Sequence ---------------- */

static void test_embedding_posenc(void) {
    /* indices [2,3], vocab 5, dim 4 */
    int sIdx[] = {2, 3};
    int32_t idx[] = {0, 1, 2, 3, 4, 0};
    FeTensor *inds = fe_tensor_alloc(DTYPE_INT32, 2, sIdx);
    memcpy(inds->data, idx, sizeof(idx));

    int sT[] = {5, 4};
    FeTensor *tab = fe_tensor_alloc(DTYPE_FLOAT32, 2, sT);
    for (int r = 0; r < 5; r++)
        for (int c = 0; c < 4; c++)
            ((float *)tab->data)[r * 4 + c] = (float)(r * 10 + c);

    int sOut[] = {2, 3, 4};
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 3, sOut);
    assert(fe_embedding(inds, tab, out) == FE_OK);
    float *o = (float *)out->data;
    assert(nearly(o[0], 0.0f));            /* idx 0 -> row0 col0 = 0 */
    assert(nearly(o[4], 10.0f));           /* idx 1 -> row1 col0 = 10 */
    assert(nearly(o[23], 3.0f));           /* last idx 0 -> row0 col3 = 3 */

    fe_tensor_free(inds); fe_tensor_free(tab); fe_tensor_free(out);
    printf("PASS test_embedding_posenc\n");
}

static void test_attention(void) {
    int s[] = {2, 2};   /* seq=2, dk=2, single batch */
    FeTensor *q = fe_tensor_alloc(DTYPE_FLOAT32, 2, s);
    FeTensor *k = fe_tensor_alloc(DTYPE_FLOAT32, 2, s);
    FeTensor *v = fe_tensor_alloc(DTYPE_FLOAT32, 2, s);
    FeTensor *o = fe_tensor_alloc(DTYPE_FLOAT32, 2, s);

    /* Identity attention: Q=K=V=[[1,0],[0,1]] -> softmax focuses each on itself */
    float m[] = {1,0, 0,1};
    memcpy(q->data, m, sizeof(m));
    memcpy(k->data, m, sizeof(m));
    memcpy(v->data, m, sizeof(m));

    assert(fe_attention(q, k, v, o) == FE_OK);
    /* Q=K=V=identity, dk=2. scores row0 = [1,0]/sqrt2 = [0.7071, 0];
       row1 = [0, 0.7071]. softmax rows give:
       row0: w0=e^0.7071/(e^0.7071+1), w1=1/(e^0.7071+1)
       out[0] = w0*[1,0]+w1*[0,1] = [w0, w1]; out[1][1] = w0. */
    float *o_ = (float *)o->data;
    float e = expf(1.0f / sqrtf(2.0f));
    float w0 = e / (e + 1.0f);
    float w1 = 1.0f / (e + 1.0f);
    assert(nearly(o_[0], w0));
    assert(nearly(o_[1], w1));
    assert(nearly(o_[3], w0));

    fe_tensor_free(q); fe_tensor_free(k); fe_tensor_free(v); fe_tensor_free(o);
    printf("PASS test_attention\n");
}

static void test_posenc(void) {
    int sIn[] = {1, 4, 6};   /* batch 1, seq 4, dim 6 */
    FeTensor *in = fe_tensor_alloc(DTYPE_FLOAT32, 3, sIn);
    memset(in->data, 0, (size_t)in->nbytes);
    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 3, sIn);

    assert(fe_positional_encoding(in, out) == FE_OK);
    float *o = (float *)out->data;
    /* A full table must be identical for every batch row. */
    for (int t = 0; t < 4; t++) {
        /* dimension 0: sin(t/10000^0) = sin(t) */
        assert(nearly(o[t * 6 + 0], sinf((float)t)));
        /* dimension 1: cos(t) */
        assert(nearly(o[t * 6 + 1], cosf((float)t)));
    }

    fe_tensor_free(in); fe_tensor_free(out);
    printf("PASS test_posenc\n");
}

static void test_mha(void) {
    int sB[] = {1, 2, 4};   /* batch 1, seq 2, d_model 4 */
    FeTensor *x = fe_tensor_alloc(DTYPE_FLOAT32, 3, sB);
    memset(x->data, 0, (size_t)x->nbytes);
    for (int i = 0; i < 8; i++) ((float *)x->data)[i] = (float)(i % 2);  /* pattern */

    int sW[] = {4, 4};
    FeTensor *wq = fe_tensor_alloc(DTYPE_FLOAT32, 2, sW);
    FeTensor *wk = fe_tensor_alloc(DTYPE_FLOAT32, 2, sW);
    FeTensor *wv = fe_tensor_alloc(DTYPE_FLOAT32, 2, sW);
    FeTensor *wo = fe_tensor_alloc(DTYPE_FLOAT32, 2, sW);
    /* identity projections */
    for (int i = 0; i < 16; i++)
        ((float *)wq->data)[i] = ((float *)wk->data)[i] =
        ((float *)wv->data)[i] = ((float *)wo->data)[i] =
            (i / 4 == i % 4) ? 1.0f : 0.0f;

    FeTensor *out = fe_tensor_alloc(DTYPE_FLOAT32, 3, sB);
    /* Just validate sanity: no error, all finite, shape preserved. */
    assert(fe_multihead_attention(x, wq, wk, wv, wo, out, 2) == FE_OK);
    float *o = (float *)out->data;
    for (size_t i = 0; i < out->nbytes / 4; i++)
        assert(isfinite(o[i]));

    fe_tensor_free(x); fe_tensor_free(wq); fe_tensor_free(wk);
    fe_tensor_free(wv); fe_tensor_free(wo); fe_tensor_free(out);
    printf("PASS test_mha\n");
}

int main(void) {
    test_exp_log_pow();
    test_activations();
    test_gemm_transpose();
    test_pool();
    test_batchnorm();
    test_layernorm();
    test_groupnorm();
    test_conv2d();
    test_embedding_posenc();
    test_attention();
    test_posenc();
    test_mha();
    printf("\nAll Stage 3 op tests passed.\n");
    return 0;
}