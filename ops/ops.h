// ops/ops.h
#ifndef FERRITE_OPS_H
#define FERRITE_OPS_H

#include "tensor.h"
#include "types.h"

/*
 * Operator kernel interface.
 *
 * All kernels follow the same contract:
 *   - Inputs are read-only
 *   - Output tensor is caller-allocated with correct shape
 *   - Returns FE_OK on success, error code otherwise
 *   - No memory allocation inside kernels
 */

/*
 * Matrix multiplication: C = A @ B
 *
 * A: [M, K]
 * B: [K, N]
 * C: [M, N]  (caller allocates)
 */
FeStatus fe_matmul(const FeTensor *A, const FeTensor *B, FeTensor *C);
FeStatus fe_matmul_scalar(const FeTensor *A, const FeTensor *B, FeTensor *C);
/*
 * Matrix multiplication with bias: C = A @ W + b
 *
 * A: [M, K]
 * W: [K, N]
 * b: [N]     (broadcast across M)
 * C: [M, N]  (caller allocates)
 */
FeStatus fe_linear(const FeTensor *A, const FeTensor *W,
                   const FeTensor *b, FeTensor *C);

/*
 * ReLU activation: out[i] = max(0, in[i])
 * Applied elementwise. out must have same shape as in.
 * Can be applied in-place (out == in is valid).
 */
FeStatus fe_relu(const FeTensor *in, FeTensor *out);

/*
 * Softmax along last axis.
 * Numerically stable: subtracts max before exp.
 * out must have same shape as in.
 */
FeStatus fe_softmax(const FeTensor *in, FeTensor *out);

/*
 * Bias add: out = in + bias
 * bias shape must match last dimension of in.
 * Supports in-place (out == in).
 */
FeStatus fe_bias_add(const FeTensor *in, const FeTensor *bias, FeTensor *out);

#endif // FERRITE_OPS_H

/*
 * 1D Convolution: out = conv1d(input, weight, bias)
 *
 * input:  [batch, C_in, L]
 * weight: [C_out, C_in, K]
 * bias:   [C_out] or NULL
 * out:    [batch, C_out, L_out]
 *
 * stride: step size (default 1)
 * pad:    zero-padding on each side
 */
FeStatus fe_conv1d(const FeTensor *input, const FeTensor *weight,
                   const FeTensor *bias,  FeTensor *out,
                   int stride, int pad);

/*
 * im2col: reshape input patches into a matrix for matmul-based conv.
 * col: [C_in * K, L_out] — caller allocates
 */
FeStatus fe_im2col(const FeTensor *input, FeTensor *col,
                   int K, int stride, int pad,
                   int batch_idx);

FeStatus fe_add(const FeTensor *a, const FeTensor *b, FeTensor *out);
FeStatus fe_sub(const FeTensor *a, const FeTensor *b, FeTensor *out);
FeStatus fe_mul(const FeTensor *a, const FeTensor *b, FeTensor *out);
FeStatus fe_div(const FeTensor *a, const FeTensor *b, FeTensor *out);

FeStatus fe_add_scalar(const FeTensor *a, float s, FeTensor *out);
FeStatus fe_sub_scalar(const FeTensor *a, float s, FeTensor *out);
FeStatus fe_mul_scalar(const FeTensor *a, float s, FeTensor *out);
FeStatus fe_div_scalar(const FeTensor *a, float s, FeTensor *out);
FeStatus fe_neg(const FeTensor *a, FeTensor *out);

FeStatus fe_sum(const FeTensor *in, int axis, FeTensor *out);
FeStatus fe_max(const FeTensor *in, int axis, FeTensor *out);
FeStatus fe_min(const FeTensor *in, int axis, FeTensor *out);
/* Returns a scalar via *out, not a tensor — see comment above the definition. */
FeStatus fe_dot(const FeTensor *a, const FeTensor *b, float *out);

FeStatus fe_logsumexp(const FeTensor *in, float *out);
FeStatus fe_stable_exp_normalize(const FeTensor *in, FeTensor *out);