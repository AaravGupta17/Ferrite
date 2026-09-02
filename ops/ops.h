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
 * GEMM with optional transpositions and alpha/beta scaling:
 *   C = alpha * op(A) @ op(B) + beta * C
 * where op(X) is X if transX==0 else transpose(X).
 * A: [M, K], B: [K, N] (logical dims before transposition),
 * C: [M, N] caller-allocated.
 */
FeStatus fe_gemm(const FeTensor *A, int transA,
                 const FeTensor *B, int transB,
                 FeTensor *C, float alpha, float beta);

/*
 * Transpose a 2D tensor (copy, since it changes memory layout).
 * out has shape [in.shape[1], in.shape[0]], caller-allocated.
 */
FeStatus fe_transpose(const FeTensor *in, FeTensor *out);
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
FeStatus fe_sigmoid(const FeTensor *in, FeTensor *out);
FeStatus fe_tanh(const FeTensor *in, FeTensor *out);
FeStatus fe_gelu(const FeTensor *in, FeTensor *out);
FeStatus fe_leaky_relu(const FeTensor *in, FeTensor *out, float negative_slope);
FeStatus fe_elu(const FeTensor *in, FeTensor *out, float alpha);
FeStatus fe_swish(const FeTensor *in, FeTensor *out);

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
 * 2D Convolution: out = conv2d(input, weight, bias)
 *
 * input:  [N, C_in, H, W]
 * weight: [C_out, C_in, KH, KW]
 * bias:   [C_out] or NULL
 * out:    [N, C_out, H_out, W_out]
 *
 * Implemented via im2col -> GEMM (same discipline as conv1d).
 */
FeStatus fe_conv2d(const FeTensor *input, const FeTensor *weight,
                   const FeTensor *bias, FeTensor *out,
                   int stride_h, int stride_w, int pad_h, int pad_w);

/*
 * Max pooling over a 2D spatial map. Input [N, C, H, W] (or [C, H, W]),
 * output [N, C, OH, OW] with OH = (H - KH)/stride + 1, same for W.
 */
FeStatus fe_maxpool(const FeTensor *in, FeTensor *out,
                    int kh, int kw, int stride_h, int stride_w);

/* Average pooling — same layout and formula as fe_maxpool. */
FeStatus fe_avgpool(const FeTensor *in, FeTensor *out,
                    int kh, int kw, int stride_h, int stride_w);

/*
 * BatchNorm over the channel axis of [N, C, ...] (inference mode):
 *   out = (x - mean[c]) / sqrt(var[c] + eps) * gamma[c] + beta[c]
 * gamma/beta/mean/var each have shape [C].
 */
FeStatus fe_batchnorm(const FeTensor *in, const FeTensor *gamma,
                      const FeTensor *beta, const FeTensor *mean,
                      const FeTensor *var, FeTensor *out, float eps);

/*
 * LayerNorm over the trailing `normalized_dims` axes:
 *   out = (x - mean) / sqrt(var + eps) * gamma + beta
 * gamma/beta have shape equal to the trailing axes being normalized.
 */
FeStatus fe_layernorm(const FeTensor *in, const FeTensor *gamma,
                      const FeTensor *beta, FeTensor *out, float eps);

/*
 * GroupNorm: normalize over channels within each group, then scale/shift.
 * Layout [N, C, H, W]; groups must divide C. gamma/beta have shape [C].
 */
FeStatus fe_groupnorm(const FeTensor *in, const FeTensor *gamma,
                      const FeTensor *beta, FeTensor *out,
                      int groups, float eps);

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
FeStatus fe_pow(const FeTensor *a, const FeTensor *b, FeTensor *out);

FeStatus fe_add_scalar(const FeTensor *a, float s, FeTensor *out);
FeStatus fe_sub_scalar(const FeTensor *a, float s, FeTensor *out);
FeStatus fe_mul_scalar(const FeTensor *a, float s, FeTensor *out);
FeStatus fe_div_scalar(const FeTensor *a, float s, FeTensor *out);
FeStatus fe_neg(const FeTensor *a, FeTensor *out);
FeStatus fe_exp(const FeTensor *a, FeTensor *out);
FeStatus fe_ln(const FeTensor *a, FeTensor *out);

FeStatus fe_sum(const FeTensor *in, int axis, FeTensor *out);
FeStatus fe_max(const FeTensor *in, int axis, FeTensor *out);
FeStatus fe_min(const FeTensor *in, int axis, FeTensor *out);
/* Returns a scalar via *out, not a tensor — see comment above the definition. */
FeStatus fe_dot(const FeTensor *a, const FeTensor *b, float *out);

FeStatus fe_logsumexp(const FeTensor *in, float *out);
FeStatus fe_stable_exp_normalize(const FeTensor *in, FeTensor *out);

/*
 * Deterministic, seedable random tensor generation.
 *
 * Both fill `out` in place (out is caller-allocated, float32) using a
 * fixed PRNG, so a given `seed` always produces byte-identical output —
 * stress tests and reproducible benchmarks depend on this. No system
 * RNG is consulted; the sequence is fully determined by the seed.
 */
FeStatus fe_rand_uniform(FeTensor *out, float low, float high, uint32_t seed);
FeStatus fe_rand_normal(FeTensor *out, float mean, float stddev, uint32_t seed);

/* ---- Sequence ops (Stage 3) ---- */

/*
 * Scaled dot-product attention:
 *   out = softmax(Q @ K^T / sqrt(d_k)) @ V
 * Q, K, V: [seq, d_k] (or [batch, seq, d_k] with same formula per batch).
 * out: same shape as Q. No masking passed in (identity mask).
 */
FeStatus fe_attention(const FeTensor *Q, const FeTensor *K,
                      const FeTensor *V, FeTensor *out);

/*
 * Multi-head attention over [batch, seq, d_model].
 * head_dim = d_model / num_heads. Wq,Wk,Wv,Wout each [d_model, d_model].
 * Returns the projected, concatenated, re-projected result, same shape
 * as the batched input (i.e. [batch, seq, d_model]).
 */
FeStatus fe_multihead_attention(const FeTensor *x,
                                const FeTensor *Wq, const FeTensor *Wk,
                                const FeTensor *Wv, const FeTensor *Wo,
                                FeTensor *out, int num_heads);

/*
 * Embedding lookup: out[i, j] = table[indices[i, j], j]
 * indices: [batch, seq] of int32 row indices into an [vocab, d_model] table.
 * out: [batch, seq, d_model].
 */
FeStatus fe_embedding(const FeTensor *indices, const FeTensor *table,
                      FeTensor *out);

/*
 * Sinusoidal positional encoding (no learned weights).
 * out[i, j] = base[offset[i], j] for a full [max_len, d_model] base table,
 * or, when offset is NULL, a fresh table for sequence positions 0..seq-1.
 */
FeStatus fe_positional_encoding(const FeTensor *in, FeTensor *out);
