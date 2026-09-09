// quantization/quant.h
#ifndef FERRITE_QUANT_H
#define FERRITE_QUANT_H

#include "types.h"
#include "tensor.h"
#include "graph.h"
#include "allocator.h"

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

/*
 * Quantized matmul with per-channel int16 weight quantization.
 *
 * A: [M, K] float32 — activation scale per inference
 * B: [K, N] int16     — weights already quantized per-channel
 * w_scales: N floats   — per-output-channel weight scales
 * C: [M, N] float32   — C[i][j] = (Σ qA[i][k]·Wq[k][j])·scale_A·w_scales[j]
 *
 * The activation is dynamically quantized per call: max|A| → scale_A → int16
 * products accumulate in int32, then dequantized with scale_A * w_scales[j].
 */
FeStatus fe_matmul_int16_dyn(const FeTensor *A, const FeTensor *Wq,
                             const float *w_scales, FeTensor *C);

/*
 * Quantized linear with per-channel int16 weight quantization and unquantized
 * bias add: C[i][j] += b[j]. Same contract as fe_matmul_int16_dyn plus bias.
 */
FeStatus fe_linear_int16(const FeTensor *A, const FeTensor *Wq,
                          const float *w_scales, const FeTensor *b,
                          FeTensor *C);

/*
 * Engine-path quantized matmul with PRE-quantized weights and dynamic
 * activation quantization — no allocation, safe for the hot path.
 *
 * A: [M, K] float32       — activations, quantized per call (dynamic):
 *                            scale = max|A|/127, then int8 products
 * Wq: [K, N] int8         — weights already quantized per-channel by
 *                            fe_quantize_model (n_scales == N)
 * w_scales: N floats      — per-output-channel weight scales
 * C:  [M, N] float32      — C[i][j] = (Σ qA[i][k]·Wq[k][j])·scale_A·w_scales[j]
 *
 * Two passes over A (max, then product loop); quantized activations are
 * never materialized, so no scratch buffer is needed.
 */
FeStatus fe_matmul_int8_dyn(const FeTensor *A, const FeTensor *Wq,
                             const float *w_scales, FeTensor *C);

/*
 * Engine-path quantized linear layer: fe_matmul_int8_dyn plus an
 * unquantized float bias add: C[i][j] += b[j]. Same pre-quantized-weight
 * contract as fe_matmul_int8_dyn; b is [N] float32 and never quantized.
 */
FeStatus fe_linear_int8(const FeTensor *A, const FeTensor *Wq,
                         const float *w_scales, const FeTensor *b,
                         FeTensor *C);

/*
 * Quantize a graph in place for engine use (Stage 15).
 *
 * Every 2D float weight consumed as inputs[1] of a MATMUL or LINEAR node is
 * per-channel quantized to INT8: its data buffer is repacked in place (no
 * reallocation — the buffer already exists) and a per-output-channel scale
 * array is arena-allocated and recorded on the tensor entry (e->scales /
 * e->n_scales). 1D biases and every other weight stay float32.
 *
 * Activations are left float32; they are quantized dynamically per inference
 * by the engine path (fe_matmul_int8_dyn / fe_linear_int8), which routes
 * MATMUL/LINEAR nodes with INT8 weights to those kernels.
 *
 * Must be called after weights are allocated (post fe_runtime_init +
 * fe_runtime_alloc_weights); the scale arrays live in the weight arena for
 * the graph's lifetime.
 */
FeStatus fe_quantize_model(FeGraph *g, FeArena *weight_arena);

/*
 * Per-tensor quantize a graph's weight tensors in place.
 *
 * Each float32 weight is converted to INT8 reusing its own buffer (the data
 * is packed down into the first numel bytes; nbytes is updated), so the
 * function works whether the weight was malloc- or arena-backed. Store scale
 * factors in params (one per converted weight, in registry order).
 *
 * NOTE: does not record scales on the registry and does not qualify for the
 * engine path — use fe_quantize_model for engine-integrated quantization.
 */
FeStatus fe_quantize_weights(FeGraph *g,
                               FeQuantParams *params, int n_params);

/*
 * Quantize a float32 tensor to int16 using symmetric per-tensor quantization.
 * scale = max(|x|) / 32767;  q = clamp(round(x / scale), -32768, 32767).
 * out must be DTYPE_INT16 with same shape as in. Stores scale in params[0].
 */
FeStatus fe_quantize_int16(const FeTensor *in, FeTensor *out,
                           float *params);

#endif // FERRITE_QUANT_H