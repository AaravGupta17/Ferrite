/* ops/conv2d.c — 2D convolution via im2col -> GEMM (Stage 3, CNN family).
 *
 * The one exception to the "no allocation inside kernels" rule, mirroring
 * fe_conv1d: the im2col scratch buffer is malloc'd here and freed before
 * returning. Everything else follows the kernel contract.
 */
#include "ops.h"
#include <stdlib.h>
#include <string.h>

/* Build the im2col matrix for one batch element:
 *   col: [C_in*KH*KW, OH*OW], col[i, j] = input[n, c, oh*sh+i-pad_h, ow*sw+j-pad_w]
 * out-of-bounds reads (padding) yield 0. */
static void im2col2d(const FeTensor *input, int n,
                     float *col, int C_in, int H, int W,
                     int KH, int KW, int OH, int OW,
                     int sh, int sw, int ph, int pw) {
    const float *src = (const float *)input->data + (size_t)n * C_in * H * W;
    size_t rows = (size_t)C_in * KH * KW;
    memset(col, 0, rows * (size_t)OH * OW * sizeof(float));

    size_t r = 0;
    for (int c = 0; c < C_in; c++) {
        for (int i = 0; i < KH; i++) {
            for (int j = 0; j < KW; j++) {
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        int ih = oh * sh + i - ph;
                        int iw = ow * sw + j - pw;
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                        col[r * OH * OW + oh * OW + ow] =
                            src[((size_t)c * H + ih) * W + iw];
                    }
                }
                r++;
            }
        }
    }
}

FeStatus fe_conv2d(const FeTensor *input, const FeTensor *weight,
                   const FeTensor *bias, FeTensor *out,
                   int sh, int sw, int ph, int pw) {
    if (!input || !weight || !out) return FE_ERR_NULL;
    if (input->ndim  != 4) return FE_ERR_SHAPE;
    if (weight->ndim != 4) return FE_ERR_SHAPE;
    if (out->ndim    != 4) return FE_ERR_SHAPE;

    int N = input->shape[0];
    int C_in = input->shape[1];
    int H = input->shape[2], W = input->shape[3];

    int C_out = weight->shape[0];
    int KH = weight->shape[2], KW = weight->shape[3];
    if (weight->shape[1] != C_in) return FE_ERR_SHAPE;
    if (sh < 1 || sw < 1 || ph < 0 || pw < 0) return FE_ERR_SHAPE;

    int OH = (H - KH + 2 * ph) / sh + 1;
    int OW = (W - KW + 2 * pw) / sw + 1;
    if (OH < 1 || OW < 1) return FE_ERR_SHAPE;
    if (out->shape[0] != N || out->shape[1] != C_out ||
        out->shape[2] != OH || out->shape[3] != OW) return FE_ERR_SHAPE;
    if (input->dtype != DTYPE_FLOAT32 || weight->dtype != DTYPE_FLOAT32 ||
        out->dtype != DTYPE_FLOAT32) return FE_ERR_DTYPE;

    int K = C_in * KH * KW;              /* rows of the im2col matrix */

    float *col = malloc((size_t)K * OH * OW * sizeof(float));
    if (!col) return FE_ERR_NOMEM;

    FeTensor colT;
    colT.data = col; colT.dtype = DTYPE_FLOAT32; colT.ndim = 2;
    colT.shape[0] = K; colT.shape[1] = OH * OW;
    colT.strides[0] = OH * OW; colT.strides[1] = 1;
    colT.nbytes = (size_t)K * OH * OW * sizeof(float);
    colT.owns_data = false;

    /* Weight as [C_out, K] — same memory, reshaped view. */
    int wshape[] = {C_out, K};
    FeTensor *Wview = fe_tensor_reshape(weight, 2, wshape);
    if (!Wview) { free(col); return FE_ERR_SHAPE; }

    FeStatus s = FE_OK;
    for (int n = 0; n < N && s == FE_OK; n++) {
        im2col2d(input, n, col, C_in, H, W, KH, KW, OH, OW, sh, sw, ph, pw);

        FeTensor out_slice;
        out_slice.data = (float *)out->data + (size_t)n * C_out * OH * OW;
        out_slice.dtype = DTYPE_FLOAT32; out_slice.ndim = 2;
        out_slice.shape[0] = C_out; out_slice.shape[1] = OH * OW;
        out_slice.strides[0] = OH * OW; out_slice.strides[1] = 1;
        out_slice.nbytes = (size_t)C_out * OH * OW * sizeof(float);
        out_slice.owns_data = false;

        s = fe_matmul(Wview, &colT, &out_slice);
        if (s != FE_OK) break;

        if (bias) {
            const float *b = (const float *)bias->data;
            float *o = (float *)out_slice.data;
            for (int oc = 0; oc < C_out; oc++)
                for (int px = 0; px < OH * OW; px++)
                    o[(size_t)oc * OH * OW + px] += b[oc];
        }
    }

    fe_tensor_free(Wview);
    free(col);
    return s;
}
