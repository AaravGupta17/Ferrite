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