/* ops/sequence.c — transformer building blocks (Stage 3, Sequence family).
 *
 * These build on the simpler Stage 2/3 primitives (matmul, softmax) but
 * inline the math where splitting into heads would otherwise force several
 * temporary tensors. As with fe_conv1d, attention needs scratch (the
 * scores matrix); we allocate it up front, pre-check for failure, and free
 * before returning — no allocation sneaks into the element loops.
 */
#include "ops.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Helper: matmul C[M,N] = A[M,K] @ B[K,N] into a caller's raw pointer. */
static void raw_matmul(const float *a, const float *b, float *c,
                       int M, int K, int N) {
    memset(c, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float a_ik = a[(size_t)i * K + k];
            for (int j = 0; j < N; j++)
                c[(size_t)i * N + j] += a_ik * b[(size_t)k * N + j];
        }
}

/* Single-head scaled dot-product attention for one block of rows:
 * out[seq, dk] = softmax((Q @ K^T)/sqrt(dk)) @ V. `scores` is scratch of
 * seq*seq floats (caller-owned). */
static void scaled_dot_attention(const float *q, const float *k,
                                 const float *v, float *out,
                                 float *scores, int seq, int dk) {
    float scale = 1.0f / (float)sqrt((double)dk);
    /* scores[i][j] = dot(Q[i], K[j]) = sum_c Q[i][c] * K[j][c] */
    for (int i = 0; i < seq; i++) {
        float *row = scores + (size_t)i * seq;
        for (int j = 0; j < seq; j++) {
            float s = 0.0f;
            for (int c = 0; c < dk; c++)
                s += q[(size_t)i * dk + c] * k[(size_t)j * dk + c];
            row[j] = s * scale;
        }
    }

    for (int r = 0; r < seq; r++) {
        float *row = scores + (size_t)r * seq;
        float mx = -INFINITY;
        for (int c = 0; c < seq; c++) if (row[c] > mx) mx = row[c];
        float sum = 0.0f;
        for (int c = 0; c < seq; c++) { row[c] = expf(row[c] - mx); sum += row[c]; }
        float inv = 1.0f / sum;
        for (int c = 0; c < seq; c++) row[c] *= inv;
    }
    raw_matmul(scores, v, out, seq, seq, dk);
}

/* Supports [seq, dk] (2D) and [batch, seq, dk] (3D). */
FeStatus fe_attention(const FeTensor *Q, const FeTensor *K,
                      const FeTensor *V, FeTensor *out) {
    if (!Q || !K || !V || !out) return FE_ERR_NULL;
    int ndim = Q->ndim;
    if (ndim != 2 && ndim != 3) return FE_ERR_SHAPE;
    if (K->ndim != ndim || V->ndim != ndim) return FE_ERR_SHAPE;
    for (int d = 0; d < ndim; d++)
        if (Q->shape[d] != K->shape[d] || Q->shape[d] != V->shape[d] ||
            Q->shape[d] != out->shape[d]) return FE_ERR_SHAPE;
    if (Q->dtype != DTYPE_FLOAT32 || K->dtype != DTYPE_FLOAT32 ||
        V->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;

    int dk = Q->shape[ndim - 1];
    int seq = Q->shape[ndim - 2];
    int batch = 1;
    for (int d = 0; d < ndim - 2; d++) batch *= Q->shape[d];
    if (seq < 1 || dk < 1) return FE_ERR_SHAPE;

    float *scores = (float *)malloc((size_t)seq * seq * sizeof(float));
    if (!scores) return FE_ERR_NOMEM;

    const float *q = (const float *)Q->data;
    const float *k = (const float *)K->data;
    const float *v = (const float *)V->data;
    float *o = (float *)out->data;

    for (int b = 0; b < batch; b++) {
        scaled_dot_attention(q + (size_t)b * seq * dk,
                             k + (size_t)b * seq * dk,
                             v + (size_t)b * seq * dk,
                             o + (size_t)b * seq * dk,
                             scores, seq, dk);
    }
    free(scores);
    return FE_OK;
}

/* Multi-head attention over [batch, seq, d_model]. */
FeStatus fe_multihead_attention(const FeTensor *x,
                                const FeTensor *Wq, const FeTensor *Wk,
                                const FeTensor *Wv, const FeTensor *Wo,
                                FeTensor *out, int num_heads) {
    if (!x || !Wq || !Wk || !Wv || !Wo || !out) return FE_ERR_NULL;
    if (x->ndim != 3) return FE_ERR_SHAPE;
    int batch = x->shape[0], seq = x->shape[1], D = x->shape[2];
    if (num_heads < 1 || D % num_heads != 0) return FE_ERR_SHAPE;
    int hd = D / num_heads;

    if (Wq->ndim != 2 || Wq->shape[0] != D || Wq->shape[1] != D ||
        Wk->ndim != 2 || Wk->shape[0] != D || Wk->shape[1] != D ||
        Wv->ndim != 2 || Wv->shape[0] != D || Wv->shape[1] != D ||
        Wo->ndim != 2 || Wo->shape[0] != D || Wo->shape[1] != D)
        return FE_ERR_SHAPE;
    if (out->ndim != 3 || out->shape[0] != batch ||
        out->shape[1] != seq || out->shape[2] != D) return FE_ERR_SHAPE;
    if (x->dtype != DTYPE_FLOAT32 || Wq->dtype != DTYPE_FLOAT32 ||
        Wk->dtype != DTYPE_FLOAT32 || Wv->dtype != DTYPE_FLOAT32 ||
        Wo->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;
    if (batch < 1 || seq < 1 || D < 1) return FE_ERR_SHAPE;

    size_t Bd = (size_t)batch * seq * D;
    size_t chunk = (size_t)seq * hd;

    /* Scratch: projected Q/K/V at full width, per-(b,h) attention output,
     * per-head gathered buffers, and a scores matrix (max hd x hd). */
    float *Q = (float *)malloc(Bd * sizeof(float));
    float *Kp = (float *)malloc(Bd * sizeof(float));
    float *V = (float *)malloc(Bd * sizeof(float));
    float *att = (float *)malloc(Bd * sizeof(float));
    float *scores = (float *)malloc((size_t)hd * hd * sizeof(float));
    float *qb = (float *)malloc(chunk * sizeof(float));
    float *kb = (float *)malloc(chunk * sizeof(float));
    float *vb = (float *)malloc(chunk * sizeof(float));
    float *ob = (float *)malloc(chunk * sizeof(float));
    if (!Q || !Kp || !V || !att || !scores || !qb || !kb || !vb || !ob) {
        free(Q); free(Kp); free(V); free(att); free(scores);
        free(qb); free(kb); free(vb); free(ob);
        return FE_ERR_NOMEM;
    }

    const float *xx = (const float *)x->data;
    const float *wq = (const float *)Wq->data;
    const float *wk = (const float *)Wk->data;
    const float *wv = (const float *)Wv->data;
    const float *wo = (const float *)Wo->data;

    raw_matmul(xx, wq, Q, batch * seq, D, D);
    raw_matmul(xx, wk, Kp, batch * seq, D, D);
    raw_matmul(xx, wv, V, batch * seq, D, D);

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < num_heads; h++) {
            /* Gather this head's interleaved columns into contiguous rows. */
            for (int t = 0; t < seq; t++)
                for (int c = 0; c < hd; c++) {
                    qb[(size_t)t * hd + c] = Q[((size_t)b * seq + t) * D + h * hd + c];
                    kb[(size_t)t * hd + c] = Kp[((size_t)b * seq + t) * D + h * hd + c];
                    vb[(size_t)t * hd + c] = V[((size_t)b * seq + t) * D + h * hd + c];
                }

            scaled_dot_attention(qb, kb, vb, ob, scores, seq, hd);

            /* Store this head's result as a contiguous (b,h) chunk. */
            memcpy(att + (size_t)(b * num_heads + h) * chunk, ob, chunk * sizeof(float));
        }
    }

    /* Scatter the per-head chunks back into full D-width rows. */
    float *o = (float *)out->data;
    for (int b = 0; b < batch; b++)
        for (int h = 0; h < num_heads; h++) {
            const float *chunkp = att + (size_t)(b * num_heads + h) * chunk;
            for (int t = 0; t < seq; t++)
                for (int c = 0; c < hd; c++)
                    o[((size_t)b * seq + t) * D + h * hd + c] = chunkp[(size_t)t * hd + c];
        }

    raw_matmul(o, wo, o, batch * seq, D, D);

    free(Q); free(Kp); free(V); free(att); free(scores);
    free(qb); free(kb); free(vb); free(ob);
    return FE_OK;
}

/* Embedding lookup: out[batch, seq, D] = table[indices[batch, seq], :]. */
FeStatus fe_embedding(const FeTensor *indices, const FeTensor *table,
                      FeTensor *out) {
    if (!indices || !table || !out) return FE_ERR_NULL;
    if (indices->ndim != 2 || table->ndim != 2) return FE_ERR_SHAPE;
    int batch = indices->shape[0], seq = indices->shape[1];
    int vocab = table->shape[0], D = table->shape[1];
    if (indices->dtype != DTYPE_INT32) return FE_ERR_DTYPE;
    if (table->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;
    if (out->ndim != 3 || out->shape[0] != batch ||
        out->shape[1] != seq || out->shape[2] != D) return FE_ERR_SHAPE;
    if (batch < 1 || seq < 1 || D < 1) return FE_ERR_SHAPE;

    const int32_t *idx = (const int32_t *)indices->data;
    const float *tab = (const float *)table->data;
    float *o = (float *)out->data;

    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < seq; t++) {
            int r = idx[(size_t)b * seq + t];
            if (r < 0 || r >= vocab) return FE_ERR_BOUNDS;
            memcpy(o + ((size_t)b * seq + t) * D,
                   tab + (size_t)r * D, (size_t)D * sizeof(float));
        }
    }
    return FE_OK;
}

/* Sinusoidal positional encoding: out[b, t, d] = PE(t, d). */
FeStatus fe_positional_encoding(const FeTensor *in, FeTensor *out) {
    if (!in || !out) return FE_ERR_NULL;
    if (in->ndim != 3) return FE_ERR_SHAPE;
    int batch = in->shape[0], seq = in->shape[1], D = in->shape[2];
    if (out->ndim != 3 || out->shape[0] != batch ||
        out->shape[1] != seq || out->shape[2] != D) return FE_ERR_SHAPE;
    if (in->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;
    if (batch < 1 || seq < 1 || D < 1) return FE_ERR_SHAPE;

    float *o = (float *)out->data;
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < seq; t++) {
            float *row = o + ((size_t)b * seq + t) * D;
            for (int d = 0; d < D; d++) {
                float freq = 1.0f / (float)pow(10000.0, 2.0 * (d / 2) / D);
                row[d] = (d % 2 == 0) ? sinf(t * freq) : cosf(t * freq);
            }
        }
    }
    return FE_OK;
}
