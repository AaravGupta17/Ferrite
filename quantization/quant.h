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
FeStatus fe_quantize_weights(struct FeGraph *g,
                              FeQuantParams *params, int n_params);

#endif // FERRITE_QUANT_H