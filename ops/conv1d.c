// ops/conv1d.c
#include "ops.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
/*
 * im2col for one batch element.
 *
 * Reshapes input patches into columns so that convolution
 * becomes a single matrix multiplication.
 *
 * For each output position p in [0, L_out):
 *   col[:, p] = input[batch_idx, :, p*stride : p*stride+K]  (flattened)
 *
 * col shape: [C_in * K, L_out]
 * Zero-padding is applied implicitly — out-of-bounds reads return 0.
 */
FeStatus fe_im2col(const FeTensor *input, FeTensor *col,
                   int K, int stride, int pad,
                   int batch_idx) {
    if (!input || !col) return FE_ERR_NULL;

    int C_in  = input->shape[1];
    int L     = input->shape[2];
    int L_out = col->shape[1];

    const float *src = (const float *)input->data
                     + batch_idx * C_in * L;
    float *dst = (float *)col->data;

    /* Zero the output — handles padding implicitly */
    memset(dst, 0, (size_t)C_in * K * L_out * sizeof(float));

    for (int ic = 0; ic < C_in; ic++) {
        for (int k = 0; k < K; k++) {
            for (int p = 0; p < L_out; p++) {
                int src_pos = p * stride + k - pad;
                if (src_pos < 0 || src_pos >= L) continue;  /* padding */
                dst[(ic * K + k) * L_out + p] =
                    src[ic * L + src_pos];
            }
        }
    }
    return FE_OK;
}

/*
 * Conv1d via im2col + matmul.
 *
 * For each batch element:
 *   1. im2col(input[n]) → col [C_in*K, L_out]
 *   2. weight reshaped  → W   [C_out, C_in*K]
 *   3. out[n] = W @ col        [C_out, L_out]
 *   4. add bias if provided
 *
 * Weight tensor is already [C_out, C_in*K] when viewed correctly —
 * we create a reshape view rather than copying.
 */
FeStatus fe_conv1d(const FeTensor *input, const FeTensor *weight,
                   const FeTensor *bias,  FeTensor *out,
                   int stride, int pad) {
    fprintf(stderr, "conv1d: in=[%d,%d,%d] w=[%d,%d,%d] out=[%d,%d,%d] stride=%d pad=%d\n",
            input->shape[0], input->shape[1], input->shape[2],
            weight->shape[0], weight->shape[1], weight->shape[2],
            out->shape[0], out->shape[1], out->shape[2],
            stride, pad);
    if (!input || !weight || !out) return FE_ERR_NULL;
    if (input->ndim  != 3) return FE_ERR_SHAPE;
    if (weight->ndim != 3) return FE_ERR_SHAPE;
    if (out->ndim    != 3) return FE_ERR_SHAPE;

    int batch = input->shape[0];
    int C_in  = input->shape[1];
    int L     = input->shape[2];
    int C_out = weight->shape[0];
    int K     = weight->shape[2];

    if (weight->shape[1] != C_in) return FE_ERR_SHAPE;

    int L_out = (L - K + 2 * pad) / stride + 1;
    if (L_out <= 0)              return FE_ERR_SHAPE;
    if (out->shape[0] != batch)  return FE_ERR_SHAPE;
    if (out->shape[1] != C_out)  return FE_ERR_SHAPE;
    if (out->shape[2] != L_out)  return FE_ERR_SHAPE;

    /* Allocate im2col buffer: [C_in*K, L_out] */
    int col_shape[] = {C_in * K, L_out};
    float *col_data = malloc((size_t)C_in * K * L_out * sizeof(float));
    if (!col_data) return FE_ERR_NOMEM;

    FeTensor col;
    col.data      = col_data;
    col.dtype     = DTYPE_FLOAT32;
    col.ndim      = 2;
    col.shape[0]  = C_in * K;
    col.shape[1]  = L_out;
    col.strides[0]= L_out;
    col.strides[1]= 1;
    col.nbytes    = (size_t)C_in * K * L_out * sizeof(float);
    col.owns_data = false;

    /* Reshape weight to [C_out, C_in*K] — view, no copy */
    int w_shape[] = {C_out, C_in * K};
    FeTensor *W = fe_tensor_reshape(weight, 2, w_shape);
    if (!W) { free(col_data); return FE_ERR_SHAPE; }

    FeStatus s = FE_OK;

    for (int n = 0; n < batch; n++) {
        /* Step 1: im2col for this batch element */
        s = fe_im2col(input, &col, K, stride, pad, n);
        if (s != FE_OK) break;

        /* Step 2: out_slice = W @ col  →  [C_out, L_out] */
        int out_shape[] = {C_out, L_out};
        FeTensor out_slice;
        out_slice.data      = (float *)out->data + n * C_out * L_out;
        out_slice.dtype     = DTYPE_FLOAT32;
        out_slice.ndim      = 2;
        out_slice.shape[0]  = C_out;
        out_slice.shape[1]  = L_out;
        out_slice.strides[0]= L_out;
        out_slice.strides[1]= 1;
        out_slice.nbytes    = (size_t)C_out * L_out * sizeof(float);
        out_slice.owns_data = false;

        s = fe_matmul(W, &col, &out_slice);
        if (s != FE_OK) break;

        /* Step 3: add bias if provided */
        if (bias) {
            float *o = (float *)out_slice.data;
            const float *b = (const float *)bias->data;
            for (int oc = 0; oc < C_out; oc++)
                for (int p = 0; p < L_out; p++)
                    o[oc * L_out + p] += b[oc];
        }
    }

    fe_tensor_free(W);
    free(col_data);
    return s;
}