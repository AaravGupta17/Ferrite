// quantization/quant.h
#ifndef FERRITE_QUANT_H
#define FERRITE_QUANT_H

#include "types.h"
#include "tensor.h"
#include "types.h"
#include "tensor.h"
#include "graph.h"

/*
 * Per-tensor quantization parameters.
 * scale: float32 value each int8 unit represents
 * zero_point: int8 value that represents 0.0f (symmetric: always 0)
 */
typedef struct {
    float scale;
    int   zero_point;
} FeQuantParams;

/*
 * Quantize a float32 tensor to int8.
 *
 * Uses symmetric per-tensor quantization:
 *   scale = max(|x|) / 127
 *   q = clamp(round(x / scale), -127, 127)
 *
 * out must be DTYPE_INT8 with same shape as in.
 * params is filled with the scale factor for dequantization.
 */
FeStatus fe_quantize(const FeTensor *in, FeTensor *out,
                      FeQuantParams *params);

/*
 * Dequantize an int8 tensor back to float32.
 *
 *   x = q * scale
 *
 * out must be DTYPE_FLOAT32 with same shape as in.
 */
FeStatus fe_dequantize(const FeTensor *in, const FeQuantParams *params,
                        FeTensor *out);

/*
 * Quantized matrix multiplication.
 *
 * A: [M, K] float32  — quantized internally
 * B: [K, N] float32  — quantized internally (weights, pre-quantized in practice)
 * C: [M, N] float32  — dequantized output
 *
 * Internal flow:
 *   1. Quantize A and B to int8
 *   2. Multiply: accumulate in int32
 *   3. Dequantize result to float32
 */
FeStatus fe_matmul_int8(const FeTensor *A, const FeTensor *B, FeTensor *C);

/*
 * Quantize a model's weight tensors in-place.
 * Converts all weight tensors in the graph from float32 to int8.
 * Stores scale factors for dequantization.
 */

/*
 * Per-channel quantization: one scale per output channel (column) of a
 * [K, N] weight matrix, instead of a single scale for the whole tensor.
 *
 * A single global scale is dominated by whichever channel has the largest
 * magnitude, crushing precision on every other channel. Per-channel scaling
 * is what real quantized runtimes do for weights — this is the gap noted
 * in temps/roadmap.md ("per-tensor INT8 is a toy version").
 *
 * For channel j: scale[j] = max_k(|in[k][j]|) / 127
 *                q[k][j]  = clamp(round(in[k][j] / scale[j]), -127, 127)
 *
 * in/out must both be 2D with the same [K, N] shape. `scales` must point
 * to caller-allocated storage for in->shape[1] floats.
 */
FeStatus fe_quantize_per_channel(const FeTensor *in, FeTensor *out,
                                  float *scales);

/*
 * Dequantize a per-channel-quantized [K, N] int8 tensor back to float32:
 *   x[k][j] = q[k][j] * scales[j]
 */
FeStatus fe_dequantize_per_channel(const FeTensor *in, const float *scales,
                                    FeTensor *out);

/*
 * Quantized matmul with per-channel weight quantization.
 *
 * A: [M, K] float32 — quantized per-tensor (one activation scale)
 * B: [K, N] float32 — quantized per-channel (one scale per column/N)
 * C: [M, N] float32 — dequantized: C[i][j] = acc[i][j] * scale_A * scale_B[j]
 */
FeStatus fe_matmul_int8_per_channel(const FeTensor *A, const FeTensor *B,
                                     FeTensor *C);

FeStatus fe_quantize_weights(struct FeGraph *g,
                              FeQuantParams *params, int n_params);

#endif // FERRITE_QUANT_H