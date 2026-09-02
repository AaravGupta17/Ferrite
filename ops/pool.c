/* ops/pool.c — max and average pooling over 2D spatial maps (Stage 3). */
#include "ops.h"
#include <float.h>

/* Shared pooling driver; `is_max` selects the reduction. */
static FeStatus pool2d(const FeTensor *in, FeTensor *out,
                       int kh, int kw, int sh, int sw, int is_max) {
    if (!in || !out) return FE_ERR_NULL;
    if (in->ndim != 4) return FE_ERR_SHAPE;
    if (in->dtype != DTYPE_FLOAT32 || out->dtype != DTYPE_FLOAT32)
        return FE_ERR_DTYPE;
    if (kh < 1 || kw < 1 || sh < 1 || sw < 1) return FE_ERR_SHAPE;

    int N = in->shape[0], C = in->shape[1];
    int H = in->shape[2], W = in->shape[3];
    int OH = (H - kh) / sh + 1;
    int OW = (W - kw) / sw + 1;

    if (OH < 1 || OW < 1) return FE_ERR_SHAPE;
    if (out->ndim != 4 || out->shape[0] != N || out->shape[1] != C ||
        out->shape[2] != OH || out->shape[3] != OW) return FE_ERR_SHAPE;

    const float *p = (const float *)in->data;
    float *o = (float *)out->data;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float *plane = p + ((size_t)n * C + c) * H * W;
            float *oplane = o + ((size_t)n * C + c) * OH * OW;
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float acc = is_max ? -FLT_MAX : 0.0f;
                    for (int i = 0; i < kh; i++) {
                        for (int j = 0; j < kw; j++) {
                            float v = plane[(oh * sh + i) * W + (ow * sw + j)];
                            if (is_max) acc = v > acc ? v : acc;
                            else        acc += v;
                        }
                    }
                    if (!is_max) acc /= (float)(kh * kw);
                    oplane[oh * OW + ow] = acc;
                }
            }
        }
    }
    return FE_OK;
}

FeStatus fe_maxpool(const FeTensor *in, FeTensor *out,
                    int kh, int kw, int sh, int sw) {
    return pool2d(in, out, kh, kw, sh, sw, 1);
}

FeStatus fe_avgpool(const FeTensor *in, FeTensor *out,
                    int kh, int kw, int sh, int sw) {
    return pool2d(in, out, kh, kw, sh, sw, 0);
}
